"""Controlled comparison: ALTERNATING (P,B,P,B...) vs JOINT (rewrite P+B together
each round) greedy optimization, same 15 DQ8GC transitions, 5 seeds each.

Both modes share: empty start, same forward_eval (inverse dynamics) + compute_g1
gradient, same accept rule (held-out >= best), same rounds. The ONLY difference is
whether each backward pass updates one component (alternate) or both in one
coordinated LLM call (joint).

Metrics per run: held-out acc, final train acc, and an objective completeness check
of the final perception on all 15 ground-truth states (green-recall, gray-recall,
green-collapse flag).

  uv run prototypes/perc_invdyn/compare_alt_joint.py
"""
import asyncio
import csv
import json
import random
import re
from pathlib import Path

from validate import (  # noqa: E402
    _extract_code,
    _parse_tag,
    _llm_call,
    load_transitions,
    make_config,
    perception_runs,
    run_perceive,
)
from validate_beliefs import (  # noqa: E402
    balanced_split,
    compute_g1,
    forward_eval,
    update_beliefs,
    update_perception,
)

DATA = Path("/tmp/dq8gc_first15")
ACTIONS = {"left", "right", "up", "down", "noop", "click"}
ROUNDS = 6
K = 5
SEEDS = [1, 2, 3, 4, 5]
MODEL = "google/gemini-2.5-flash"

JOINT_TMPL = """You are JOINTLY improving a perception module AND a world-knowledge block for a grid environment, so they work as a coherent pair.

PERCEPTION: a Python function `perceive(observation_history: list[str]) -> str` that EXTRACTS decision-relevant state features as text (never raise, never empty, <2000 chars).
WORLD KNOWLEDGE: concise GENERAL facts (coordinate conventions, what entities are, what each action does), shared across ALL states -- only general truths, never state-specific facts or action frequencies.

A predictor uses (perception features of two consecutive states + world knowledge) to identify which action was taken between them, WITHOUT seeing the raw grid. Below are FAILED cases with diagnosed deficiencies.

=== CURRENT PERCEPTION MODULE ===
```python
{code}
```
=== CURRENT WORLD KNOWLEDGE ===
{beliefs}
=== END ===

Diagnosed PERCEPTION deficiencies:
{g1_p}

Diagnosed WORLD_KNOWLEDGE deficiencies:
{g1_b}

Labeled failure cases (perception features of the two states + the TRUE action):
{labeled_cases}

Raw observation examples (what is extractable; do NOT echo the raw grid):
{raw_examples}

Rewrite BOTH components TOGETHER. Decide jointly which information belongs in perception (extracted features) vs world knowledge (general rules) so the pair explains every labeled case. Respond with BOTH, perception first:
```python
<full perceive module>
```
<world_knowledge>
- ...
</world_knowledge>"""


async def joint_update(cfg, code, beliefs, g1_p, g1_b, failures, train, sem):
    labeled = (failures or [])[:8]
    cases = []
    for i, r in enumerate(labeled, 1):
        cases.append(
            f"--- case {i} ---\nSTATE 1 features: {r.z_t or '(empty)'}\n"
            f"STATE 2 features: {r.z_t1 or '(empty)'}\nTRUE action: {r.tr.action}"
        )
    raw_src = [r.tr for r in failures] or train
    ex = []
    for i, tr in enumerate(raw_src[:2], 1):
        ex.append(f"--- example {i} (action: {tr.action}) ---\nX_t: {tr.x_t[:2200]}\nX_t+1: {tr.x_t1[:2200]}")
    prompt = JOINT_TMPL.format(
        code=code or "# (empty)", beliefs=beliefs.strip() or "(empty)",
        g1_p=g1_p or "(none)", g1_b=g1_b or "(none)",
        labeled_cases="\n\n".join(cases) or "(none)", raw_examples="\n\n".join(ex),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    newcode = _extract_code(text)
    newbel = _parse_tag(text, "world_knowledge") or beliefs
    return newcode, newbel, cost


async def run_loop(cfg, mode, train, holdout, action_pool, seed, sem, log):
    rng = random.Random(seed * 7 + 1)
    code, beliefs = "", ""
    cost = 0.0
    best_acc, _, c = await forward_eval(cfg, code, beliefs, holdout, action_pool, K, sem, rng); cost += c
    best_code, best_bel = code, beliefs
    for rnd in range(1, ROUNDS + 1):
        tr_acc, recs, c = await forward_eval(cfg, code, beliefs, train, action_pool, K, sem, rng); cost += c
        failures = [r for r in recs if not r.correct]
        g1_p, g1_b, c = await compute_g1(cfg, failures, beliefs, sem); cost += c
        if mode == "alternate":
            active = "P" if rnd % 2 == 1 else "B"
            if active == "P":
                cand, c = await update_perception(cfg, code, g1_p, beliefs, [r.tr for r in failures] or train, sem); cost += c
                ok, _ = perception_runs(cand, [t.x_t for t in train[:4]])
                if not ok:
                    continue
                new_acc, _, c = await forward_eval(cfg, cand, beliefs, holdout, action_pool, K, sem, rng); cost += c
                if new_acc >= best_acc:
                    best_acc, best_code, code = new_acc, cand, cand
            else:
                cand, c = await update_beliefs(cfg, beliefs, g1_b, failures or recs, sem); cost += c
                new_acc, _, c = await forward_eval(cfg, code, cand, holdout, action_pool, K, sem, rng); cost += c
                if new_acc >= best_acc:
                    best_acc, best_bel, beliefs = new_acc, cand, cand
        else:  # joint
            ncode, nbel, c = await joint_update(cfg, code, beliefs, g1_p, g1_b, failures, train, sem); cost += c
            ok, _ = perception_runs(ncode, [t.x_t for t in train[:4]])
            ncode = ncode if ok else code
            new_acc, _, c = await forward_eval(cfg, ncode, nbel, holdout, action_pool, K, sem, rng); cost += c
            if new_acc >= best_acc:
                best_acc, best_code, best_bel, code, beliefs = new_acc, ncode, nbel, ncode, nbel
    # final train acc with best
    tr_acc, _, c = await forward_eval(cfg, best_code, best_bel, train, action_pool, K, sem, rng); cost += c
    log(f"[{mode} seed{seed}] DONE  held-out={best_acc:.2f}  train={tr_acc:.2f}  cost=${cost:.4f}")
    return {"mode": mode, "seed": seed, "holdout_acc": best_acc, "train_acc": tr_acc,
            "cost": cost, "perception": best_code, "beliefs": best_bel}


# ---- objective completeness eval on ground truth ----
def truth(obs):
    i = obs.find("[[")
    depth = 0; end = -1
    for j in range(i, len(obs)):
        if obs[j] == "[":
            depth += 1
        elif obs[j] == "]":
            depth -= 1
            if depth == 0:
                end = j; break
    g = json.loads(obs[i:end + 1])
    dg = [(r, c) for r, row in enumerate(g) for c, cell in enumerate(row) if cell == "darkgreen"]
    gy = [(r, c) for r, row in enumerate(g) for c, cell in enumerate(row) if cell == "gray"]
    return dg, gy


def mentions(out, rc):
    r, c = rc
    pats = [f"({r},{c})", f"({r}, {c})", f"{r},{c}", f"{r}, {c}",
            f"row {r}, column {c}", f"row {r}, col {c}", f"row {r} column {c}"]
    lo = out.lower()
    return any(p.lower() in lo for p in pats)


def completeness(code, rows):
    gr_tot = gr_hit = gy_tot = gy_hit = 0
    max_green_state = max(rows, key=lambda o: len(truth(o)[0]))
    dg_m, _ = truth(max_green_state)
    out_m = run_perceive(code, max_green_state)[0]
    greens_found_on_max = sum(mentions(out_m, rc) for rc in dg_m)
    for obs in rows:
        dg, gy = truth(obs)
        out = run_perceive(code, obs)[0]
        for rc in dg:
            gr_tot += 1; gr_hit += mentions(out, rc)
        for rc in gy:
            gy_tot += 1; gy_hit += mentions(out, rc)
    return {
        "green_recall": gr_hit / max(1, gr_tot),
        "gray_recall": gy_hit / max(1, gy_tot),
        "max_green_truth": len(dg_m),
        "max_green_found": greens_found_on_max,
        "collapsed": greens_found_on_max < len(dg_m),
    }


async def main():
    cfg = make_config(MODEL, "openrouter")
    transitions = load_transitions([DATA], ACTIONS)
    for t in transitions:
        t.action = t.action.split()[0]
    action_pool = sorted({t.action for t in transitions})
    rows = [r["Observation"] for r in csv.DictReader((DATA / "episode_0/trajectory.csv").open())]
    print(f"transitions={len(transitions)} action_pool={action_pool}")

    sem = asyncio.Semaphore(48)
    lock = asyncio.Lock()

    def log(m):
        print(m, flush=True)

    import sys
    modes = tuple(m for m in ("alternate", "joint") if (len(sys.argv) < 2 or m in sys.argv))
    tasks = []
    for seed in SEEDS:
        for mode in modes:
            rng = random.Random(seed)  # identical split per seed across modes
            train, holdout = balanced_split(transitions, 3, 12, rng)
            tasks.append(run_loop(cfg, mode, train, holdout, action_pool, seed, sem, log))
    results = await asyncio.gather(*tasks)

    for r in results:
        r["completeness"] = completeness(r["perception"], rows)

    outdir = Path("logs/alt_vs_joint")
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump([{k: v for k, v in r.items() if k != "perception"} for r in results],
              (outdir / "results.json").open("w"), indent=2, default=str)
    for r in results:
        tag = f"{r['mode']}_seed{r['seed']}"
        (outdir / f"P_{tag}.py").write_text(r["perception"] or "")
        (outdir / f"B_{tag}.txt").write_text(r["beliefs"] or "")

    print("\n================= RESULTS =================")
    hdr = f"{'mode':<10} {'seed':>4} {'hold':>5} {'train':>6} {'grnRec':>7} {'gryRec':>7} {'maxG':>5} {'found':>6} {'collapse':>8}"
    print(hdr)
    for mode in ("alternate", "joint"):
        for r in sorted([x for x in results if x["mode"] == mode], key=lambda x: x["seed"]):
            c = r["completeness"]
            print(f"{mode:<10} {r['seed']:>4} {r['holdout_acc']:>5.2f} {r['train_acc']:>6.2f} "
                  f"{c['green_recall']:>7.2f} {c['gray_recall']:>7.2f} {c['max_green_truth']:>5} "
                  f"{c['max_green_found']:>6} {str(c['collapsed']):>8}")
    print("\n----------------- AGGREGATE -----------------")
    for mode in ("alternate", "joint"):
        rs = [x for x in results if x["mode"] == mode]
        if not rs:
            continue
        n = len(rs)
        print(f"{mode:<10}  hold={sum(x['holdout_acc'] for x in rs)/n:.2f}  "
              f"train={sum(x['train_acc'] for x in rs)/n:.2f}  "
              f"green_recall={sum(x['completeness']['green_recall'] for x in rs)/n:.2f}  "
              f"gray_recall={sum(x['completeness']['gray_recall'] for x in rs)/n:.2f}  "
              f"collapsed={sum(x['completeness']['collapsed'] for x in rs)}/{n}  "
              f"cost=${sum(x['cost'] for x in rs):.4f} (mean ${sum(x['cost'] for x in rs)/n:.4f})")
    print(f"\nartifacts -> {outdir}")


if __name__ == "__main__":
    asyncio.run(main())
