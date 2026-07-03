"""Post-hoc failure attribution for the 6-game x 5-seed low-data GEPA sweep.

For every game/seed it:
  1. loads the GEPA-learned perception P and belief B (best_* artifacts),
  2. reconstructs the EXACT clean test split used during the run (same rng walk),
  3. runs the full pipeline  P(X_t),P(X_t+1),B -> F -> Ahat  on the 20 test items,
  4. for each FAILURE attributes the root cause to exactly one of:
       PERCEPTION   - P's two outputs don't faithfully/completely encode the real
                      raw change (incl. P that dropped the change or errored).
       BELIEF       - P is faithful but B is missing/incorrect for the rule F needed.
       F_REASONING  - P faithful AND B has the rule, but F still decoded the wrong action.
       NO_SIGNAL    - the transition produced no observable change in the frames, so the
                      true action is not identifiable from the observation (no component's fault).

PERCEPTION (P errored / dropped the change) and NO_SIGNAL (no raw change) are decided
deterministically from the raw grids; the BELIEF vs F_REASONING split is decided by an
LLM judge given the raw change, P's outputs, B, the choices and F's own reasoning.

Writes a human-readable report to prototypes/perc_invdyn/seeds5_analysis.md
and the raw per-failure records to prototypes/perc_invdyn/seeds5_analysis.json.
"""
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gepa_optimize import (  # noqa: E402
    bake_choices,
    balanced_split,
    load_transitions,
    make_config,
    predict_action,
    run_perceive,
)
from validate_beliefs import _llm_call  # noqa: E402

TSV = Path("/tmp/seeds5_envs.tsv")
SEEDS = (1, 2, 3, 4, 5)
TRAIN_N, TEST_N, K = 5, 20, 5
TASK_MODEL = "google/gemini-2.5-flash"
CLIENT = "openrouter"
OUT_DIR = Path("/tmp/seeds5")

JUDGE_TMPL = """You are auditing a single failure of an inverse-dynamics agent.

The agent must look at two consecutive game frames and decide which ACTION caused the
transition. The pipeline has three parts:
  - PERCEPTION P: code that converts each raw frame into a compact text description. The
    agent F only ever sees P's outputs, never the raw frame.
  - BELIEF B: world-knowledge text describing the game's dynamics / action->effect rules.
  - F: the reasoner that, given P(frame_t), P(frame_t+1) and B, picks the action.

(No external ground-truth change is available -- judge using ONLY the agent's own
pipeline outputs below, exactly the information the agent itself had.)

What PERCEPTION gave the agent:
  P(frame_t)   = {z_t}
  P(frame_t+1) = {z_t1}

The BELIEF text B the agent had:
{beliefs}

The action choices were: {choices}
The TRUE action was: {truth}
The agent PREDICTED: {pred}
The agent's own reasoning was:
{reasoning}

Attribute the root cause of this wrong prediction to EXACTLY ONE of:
  PERCEPTION  - P's two outputs do not contain enough information to tell the actions apart
                (e.g. identical for both states, or missing the distinction the action would
                produce), so F could not have recovered the right action from what P gave it.
  BELIEF      - P's outputs DO differ informatively, but B is missing the rule, or states an
                incorrect rule, that F needed to map that difference to the action.
  F_REASONING - P's outputs differ informatively AND B contains the rule needed, but F still
                reasoned to the wrong action.

Respond with ONLY a JSON object: {{"cause": "PERCEPTION|BELIEF|F_REASONING", "why": "<one sentence>"}}
"""


def load_games():
    games = []
    for line in TSV.read_text().splitlines():
        if not line.strip():
            continue
        env, acts, img, rd = line.split("\t")
        games.append({
            "env": env,
            "actions": set(filter(None, acts.split(","))),
            "image": img == "1",
            "run_dirs": [Path(p) for p in rd.split(",") if p.strip()],
        })
    return games


def rebuild_test(game, seed):
    """Mirror gepa_optimize.main's rng walk exactly to recover the same test split+choices."""
    rng = random.Random(seed)
    transitions = load_transitions(game["run_dirs"], game["actions"])
    for t in transitions:
        t.action = t.action.split()[0]
    rng.shuffle(transitions)
    action_pool = sorted({t.action for t in transitions})
    rest, test_tr = balanced_split(transitions, TEST_N, 10**9, rng)
    _, train_tr = balanced_split(rest, TRAIN_N, 10**9, rng)   # tie mode
    bake_choices(train_tr, action_pool, K, rng)               # consume rng for train (val==train)
    test = bake_choices(test_tr, action_pool, K, rng)
    return test


def _safe_perceive(code, raw):
    try:
        out = run_perceive(code, raw)[0]
    except Exception as e:  # noqa: BLE001
        return None, f"<P_RAISED: {e}>"
    if not out or not str(out).strip():
        return out, "<P_EMPTY>"
    return out, None


async def judge(cfg, rec):
    prompt = JUDGE_TMPL.format(
        z_t=rec["z_t"][:1200], z_t1=rec["z_t1"][:1200],
        beliefs=(rec["beliefs"] or "(empty)")[:2500],
        choices=rec["choices"], truth=rec["truth"], pred=rec["pred"],
        reasoning=(rec["reasoning"] or "(none)")[:1500],
    )
    txt = await asyncio.to_thread(_llm_call, cfg, prompt)
    txt = txt or ""
    cause = "F_REASONING"
    why = ""
    try:
        s = txt[txt.find("{"): txt.rfind("}") + 1]
        obj = json.loads(s)
        cause = obj.get("cause", cause).upper()
        why = obj.get("why", "")
    except Exception:  # noqa: BLE001
        for c in ("PERCEPTION", "BELIEF", "F_REASONING"):
            if c in txt.upper():
                cause = c
                break
    if cause not in ("PERCEPTION", "BELIEF", "F_REASONING"):
        cause = "F_REASONING"
    return cause, why


async def analyze_one(cfg, game, seed, sem):
    env = game["env"]
    rundir = OUT_DIR / env / f"seed{seed}"
    p_file = rundir / f"best_perception_gepa_seed{seed}.py"
    b_file = rundir / f"best_beliefs_gepa_seed{seed}.txt"
    if not p_file.exists():
        return {"env": env, "seed": seed, "status": "MISSING", "acc": None, "fails": []}
    P = p_file.read_text()
    B = b_file.read_text() if b_file.exists() else ""
    test = rebuild_test(game, seed)

    async def run_item(item):
        tr, choices = item["tr"], item["choices"]
        z_t, err_t = _safe_perceive(P, tr.x_t)
        z_t1, err_t1 = _safe_perceive(P, tr.x_t1)
        zt = "" if z_t is None else str(z_t)
        zt1 = "" if z_t1 is None else str(z_t1)
        pred, reasoning, _ = await predict_action(cfg, zt, zt1, B, choices, sem)
        return tr, zt, zt1, (err_t or err_t1), pred, reasoning

    rows = await asyncio.gather(*(run_item(it) for it in test))
    correct = sum(1 for r in rows if r[4] == r[0].action)
    acc = correct / len(rows)

    fails = []
    for tr, zt, zt1, perr, pred, reasoning in rows:
        if pred == tr.action:
            continue
        rec = {
            "env": env, "seed": seed, "truth": tr.action, "pred": pred,
            "z_t": zt, "z_t1": zt1, "beliefs": B,
            "choices": [tr.action, pred], "reasoning": reasoning,
        }
        # deterministic buckets first (internal observables only)
        if perr:
            rec["cause"], rec["why"] = "PERCEPTION", f"P failed to run/produce output {perr}"
        elif zt.strip() == zt1.strip():
            rec["cause"], rec["why"] = "PERCEPTION", "P outputs identical for both states (no discriminative signal)"
        else:
            rec["cause"], rec["why"] = await judge(cfg, rec)
        fails.append(rec)
    return {"env": env, "seed": seed, "status": "OK", "acc": acc, "fails": fails}


async def main():
    cfg = make_config(TASK_MODEL, CLIENT)
    sem = asyncio.Semaphore(8)
    games = load_games()
    results = []
    for game in games:
        for seed in SEEDS:
            r = await analyze_one(cfg, game, seed, sem)
            tag = f"{r['env']} seed{seed}"
            if r["status"] != "OK":
                print(f"{tag:18} {r['status']}")
            else:
                cc = Counter(f["cause"] for f in r["fails"])
                print(f"{tag:18} acc={r['acc']:.2f}  fails={len(r['fails'])}  {dict(cc)}")
            results.append(r)

    # ---- aggregate ----
    per_env = defaultdict(lambda: {"accs": [], "causes": Counter()})
    overall = Counter()
    for r in results:
        if r["status"] != "OK":
            continue
        per_env[r["env"]]["accs"].append(r["acc"])
        for f in r["fails"]:
            per_env[r["env"]]["causes"][f["cause"]] += 1
            overall[f["cause"]] += 1

    lines = ["# Low-data GEPA sweep (gemini-2.5-flash) — 6 games x 5 seeds: failure attribution\n",
             "Setting: 5 train (==val) / 20 clean test, max_metric_calls=120, --start empty, GEPA only.",
             "AutumnBench games text-mode; ARC games image-mode. F=gemini-2.5-flash (matched).\n",
             "Cause buckets: PERCEPTION (learned P incorrect/incomplete), BELIEF (learned B "
             "incorrect/incomplete), F_REASONING (P+B fine, decoder erred), NO_SIGNAL (transition "
             "has no observable effect — action unidentifiable, not a component fault).\n",
             "## Per-game (mean test acc over 5 seeds, chance=0.20) and failure causes\n",
             "| game | mode | mean acc | seeds | PERCEPTION | BELIEF | F_REASONING | NO_SIGNAL |",
             "|------|------|---------:|-------|-----------:|-------:|------------:|----------:|"]
    gmap = {g["env"]: g for g in games}
    for env, d in per_env.items():
        accs = d["accs"]
        c = d["causes"]
        mode = "image" if gmap[env]["image"] else "text"
        seedstr = ", ".join(f"{a:.2f}" for a in accs)
        lines.append(f"| {env} | {mode} | {sum(accs)/len(accs):.2f} | {seedstr} | "
                     f"{c['PERCEPTION']} | {c['BELIEF']} | {c['F_REASONING']} | {c['NO_SIGNAL']} |")
    tot = sum(overall.values())
    lines += ["",
              "## Overall failure attribution (all games, all seeds)\n",
              f"Total failures: {tot}\n",
              "| cause | count | share |", "|-------|------:|------:|"]
    for cause in ("PERCEPTION", "BELIEF", "F_REASONING", "NO_SIGNAL"):
        n = overall[cause]
        lines.append(f"| {cause} | {n} | {(n/tot*100 if tot else 0):.0f}% |")

    # per-failure appendix
    lines += ["", "## Per-failure detail\n"]
    for r in results:
        if r["status"] != "OK" or not r["fails"]:
            continue
        lines.append(f"### {r['env']} seed{r['seed']} (acc={r['acc']:.2f})")
        for f in r["fails"]:
            lines.append(f"- **{f['cause']}** true=`{f['truth']}` pred=`{f['pred']}` — {f['why']}")
        lines.append("")

    (HERE / "seeds5_analysis.md").write_text("\n".join(lines))
    (HERE / "seeds5_analysis.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nOverall:", dict(overall))
    print(f"wrote {HERE/'seeds5_analysis.md'} and seeds5_analysis.json")


if __name__ == "__main__":
    asyncio.run(main())
