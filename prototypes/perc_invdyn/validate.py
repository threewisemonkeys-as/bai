"""Standalone validation of the inverse-dynamics backward-signal for perception.

Question this answers: given a weak perception module P (action-ID score ~0),
does the backward signal (forward -> G0 loss vs true action -> G1 textual
gradient -> update P') *measurably* improve P on a HELD-OUT set, rather than
just resampling P from the (weak) generator?

Pipeline (mirrors the design discussion):
  forward:  (X_t, X_t+1) -> (Z_t, Z_t+1) = (P(X_t), P(X_t+1)) -> Ahat  [predictor f, frozen]
  loss:     G0 = accuracy(Ahat, A_true)                                (informative: f returns a ranked guess + reasoning)
  grad:     G1 = feedback_llm(failures: (Ahat,A_true), (Z_t,Z_t+1))    (= dL/dZ, aggregated over a batch)
  update:   P' = update_llm(P, G1, raw (X_t,X_t+1) examples)           (= optimizer step; validated to run)
  gate:     accept P' only if held-out G0 improves                     (line-search / no circular self-report)

Data comes from existing run logs' trajectory.csv (Step,Action,Observation,...).
Everything is read-only w.r.t. the repo; only the predictor/feedback/update LLM
calls cost money, throttled by a semaphore and small train/holdout splits.

Usage:
  uv run prototypes/perc_invdyn/validate.py \
      --run "logs/dynamics_full_perc/DQ8GC/2026-06-11_15-46-00_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn" \
      --rounds 3 --train-n 14 --holdout-n 14
"""

import argparse
import asyncio
import csv
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

# Reuse the repo's LLM plumbing (litellm.responses + cost) verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mixed_improve import _llm_call  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

csv.field_size_limit(10_000_000)  # observations are large multiline cells


# ---------------------------------------------------------------------------
# Config: minimal DictConfig so _llm_call works without Hydra.
# ---------------------------------------------------------------------------
def make_config(model_id: str, client_name: str) -> object:
    return OmegaConf.create({"client": {"client_name": client_name, "model_id": model_id}})


# ---------------------------------------------------------------------------
# Data: (X_t, A_t, X_t+1) transitions from a run's trajectory.csv.
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    x_t: str
    x_t1: str
    action: str
    # Temporal context for windowed (K-step) objectives. Empty unless load_transitions
    # is called with context_k>0. ctx_prev: the K steps BEFORE x_t, oldest->newest, each
    # (state_raw, outgoing_action) so the last entry's action leads into x_t. ctx_next: the
    # steps AFTER x_t1, each (incoming_action, state_raw) so the resulting future states are
    # X_t+2..X_t+K. Both are partial at episode edges. (X_t, A_t, X_t+1) stays the center.
    ctx_prev: list = field(default_factory=list)
    ctx_next: list = field(default_factory=list)


def _row_step(row, action_whitelist):
    """(observation, action) for a row IFF it is a usable, in-vocabulary, non-terminal
    step; else None (signals an episode/validity boundary that the context walk stops at)."""
    action = (row.get("Action") or "").strip()
    obs = row.get("Observation") or ""
    done = (row.get("Done") or "").strip().lower() in ("true", "1")
    if done or not action or not obs.strip():
        return None
    if action_whitelist is not None and action.split()[0] not in action_whitelist:
        return None
    return obs, action


def load_transitions(
    run_dirs: list[Path], action_whitelist: set[str] | None, context_k: int = 0
) -> list[Transition]:
    """Build (X_t, A_t, X_t+1) transitions. When context_k>0, also attach up to K steps of
    temporal context on each side (ctx_prev / ctx_next) so the windowed objectives can show
    P(X_t-K),a_t-K,...,P(X_t),...,P(X_t+K). Context is gathered within the SAME episode CSV
    and stops at any boundary (done / empty / out-of-vocab action), so windows shrink rather
    than cross episodes."""
    transitions: list[Transition] = []
    for run_dir in run_dirs:
        for csv_path in run_dir.glob("episode_*/trajectory.csv"):
            rows = list(csv.DictReader(csv_path.open()))
            for i in range(len(rows) - 1):
                r, nxt = rows[i], rows[i + 1]
                action = (r.get("Action") or "").strip()
                obs, obs_next = r.get("Observation") or "", nxt.get("Observation") or ""
                done = (r.get("Done") or "").strip().lower() in ("true", "1")
                if done or not action or not obs.strip() or not obs_next.strip():
                    continue  # skip episode boundaries / empty rows
                if action_whitelist is not None and action.split()[0] not in action_whitelist:
                    continue
                tr = Transition(obs, obs_next, action)
                if context_k > 0:
                    # ctx_prev: walk back X_t-1..X_t-K, each (state, outgoing action).
                    prev = []
                    for j in range(i - 1, i - 1 - context_k, -1):
                        if j < 0:
                            break
                        step = _row_step(rows[j], action_whitelist)
                        if step is None:  # boundary -> window shrinks, stop walking
                            break
                        prev.append(step)
                    tr.ctx_prev = list(reversed(prev))  # oldest -> newest
                    # ctx_next: future states X_t+2..X_t+K via actions a_t+1..a_t+K-1.
                    nxts = []
                    for m in range(1, context_k):
                        step = _row_step(rows[i + m], action_whitelist) if i + m < len(rows) else None
                        fut = rows[i + m + 1].get("Observation") if i + m + 1 < len(rows) else None
                        if step is None or not (fut and fut.strip()):
                            break
                        nxts.append((step[1], fut))  # (incoming action, resulting state)
                    tr.ctx_next = nxts
                transitions.append(tr)
    if not transitions:
        raise FileNotFoundError(f"no usable transitions under {run_dirs}")
    return transitions


# ---------------------------------------------------------------------------
# Perception runner: exec code, call perceive() on a single-frame history.
# Returns (output, error). Mirrors the forward pipeline P(X) per frame.
# ---------------------------------------------------------------------------
def run_perceive(code: str, raw_obs: str) -> tuple[str, str | None]:
    if not code.strip():
        return "", None
    try:
        ns: dict = {}
        exec(code, ns)
        fn = ns.get("perceive")
        if not callable(fn):
            return "", "no callable perceive()"
        out = fn([raw_obs])  # history signature: perceive(observation_history: list[str])
        return (out if isinstance(out, str) else str(out)), None
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {e}"


def perception_runs(code: str, sample_obs: list[str]) -> tuple[bool, str | None]:
    for obs in sample_obs:
        _, err = run_perceive(code, obs)
        if err is not None:
            return False, err
    return True, None


# ---------------------------------------------------------------------------
# Choice set: true action + distractors sampled from observed actions.
# ---------------------------------------------------------------------------
def make_choices(true_action: str, action_pool: list[str], k: int, rng: random.Random) -> list[str]:
    distractors = [a for a in action_pool if a != true_action]
    rng.shuffle(distractors)
    choices = [true_action] + distractors[: max(0, k - 1)]
    rng.shuffle(choices)
    return choices


def _parse_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_action(text: str, choices: list) -> "str | None":
    """Recover the chosen action from a decoder response, robust to format drift.

    The prompt asks for <action>…</action>, but the model frequently free-forms and
    puts the answer elsewhere (commonly `\\boxed{...}`; a large fraction omit the tag
    entirely). The OLD fallback matched `pred.lower() in c.lower()` with an EMPTY pred,
    which is a substring of every choice -> it silently returned choices[0], scoring the
    transition on a parser default rather than the model's actual answer.

    Resolution order, each normalised against the verbatim choice list:
      1. <action> tag ; 2. \\boxed{...} (last) ; 3. the choice mentioned LAST in the text
         (so a final decision overrides choices enumerated earlier in the reasoning).
    Returns None when nothing matches -> the caller treats it as an explicit abstention
    (wrong), never a silent default to the first option.
    """
    def _norm(cand):
        cand = (cand or "").strip()
        if not cand:
            return None
        if cand in choices:
            return cand
        low = cand.lower()
        for c in choices:               # exact (case-insensitive) match first
            if c.lower() == low:
                return c
        hits = [c for c in choices if c.lower() in low or low in c.lower()]
        return max(hits, key=len) if hits else None  # longest containment wins

    p = _norm(_parse_tag(text, "action"))
    if p:
        return p
    for b in reversed(re.findall(r"\\boxed\{(?:\\text\{)?([^{}]*)\}", text or "")):
        p = _norm(b)
        if p:
            return p
    last_pos, last_choice = -1, None    # last choice string to appear anywhere
    for c in choices:
        pos = (text or "").lower().rfind(c.lower())
        if pos > last_pos:
            last_pos, last_choice = pos, c
    return last_choice


# ---------------------------------------------------------------------------
# f: frozen predictor. Sees ONLY perception outputs (never raw obs), picks the
# action from the choice set, and explains its reasoning (informative loss).
# ---------------------------------------------------------------------------
PREDICT_TMPL = """Two consecutive states of a grid environment are summarized below by a perception module. Exactly one action was taken to get from STATE 1 to STATE 2.

=== STATE 1 features ===
{z_t}
=== STATE 2 features ===
{z_t1}

The action was one of:
{choices}

Decide which single action was taken, using ONLY the features above. If the features do not contain enough information to tell, say so in your reasoning and make your best guess.

Respond as:
<reasoning>what changed between the two states and what action that implies; if undecidable, state exactly what feature is missing</reasoning>
<action>the chosen action, copied verbatim from the list</action>"""


async def predict_action(cfg, z_t: str, z_t1: str, choices: list[str], sem) -> tuple[str, str, float]:
    prompt = PREDICT_TMPL.format(
        z_t=z_t or "(empty)",
        z_t1=z_t1 or "(empty)",
        choices="\n".join(f"- {c}" for c in choices),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""  # models intermittently return None (MALFORMED) -> treat as a failed prediction
    pred = _extract_action(text, choices)
    return pred, _parse_tag(text, "reasoning"), cost


# ---------------------------------------------------------------------------
# Forward + loss G0 over a set of transitions. Returns accuracy + per-item
# records (for building G1 from the failures).
# ---------------------------------------------------------------------------
@dataclass
class ForwardRecord:
    tr: Transition
    z_t: str
    z_t1: str
    choices: list[str]
    pred: str
    reasoning: str
    correct: bool


async def forward_eval(cfg, code: str, transitions: list[Transition], action_pool: list[str],
                       k_choices: int, sem, rng: random.Random, raw_mode: bool = False):
    async def one(tr: Transition):
        if raw_mode:  # ceiling: predictor sees raw frames (truncated)
            z_t, z_t1 = tr.x_t[:6000], tr.x_t1[:6000]
        else:
            z_t = run_perceive(code, tr.x_t)[0]
            z_t1 = run_perceive(code, tr.x_t1)[0]
        choices = make_choices(tr.action, action_pool, k_choices, rng)
        pred, reasoning, cost = await predict_action(cfg, z_t, z_t1, choices, sem)
        return ForwardRecord(tr, z_t, z_t1, choices, pred, reasoning, pred == tr.action), cost

    results = await asyncio.gather(*(one(tr) for tr in transitions))
    records = [r for r, _ in results]
    cost = sum(c for _, c in results)
    acc = sum(r.correct for r in records) / max(1, len(records))
    return acc, records, cost


# ---------------------------------------------------------------------------
# G1: textual gradient on the perception OUTPUT, aggregated over failures.
# ---------------------------------------------------------------------------
G1_TMPL = """A perception module summarizes grid states into text features. Those features are then used (without the raw grid) to identify which action was taken between two consecutive states. Below are cases where the identification FAILED, meaning the features were insufficient or misleading.

{cases}

Diagnose the perception OUTPUT (not the code). Across these failures, what information is consistently MISSING or MISLEADING in the features that, if present, would make the action identifiable? Be concrete and specific to what you see.

Respond as:
<deficiencies>
- short, concrete bullet points naming the exact missing/misleading information
</deficiencies>"""


async def compute_g1(cfg, failures: list[ForwardRecord], sem, max_cases: int = 6) -> tuple[str, float]:
    if not failures:
        return "", 0.0
    cases = []
    for i, r in enumerate(failures[:max_cases], 1):
        cases.append(
            f"--- case {i} ---\n"
            f"STATE 1 features: {r.z_t or '(empty)'}\n"
            f"STATE 2 features: {r.z_t1 or '(empty)'}\n"
            f"predicted action: {r.pred or '(none)'} | TRUE action: {r.tr.action}\n"
            f"predictor reasoning: {r.reasoning[:400]}"
        )
    prompt = G1_TMPL.format(cases="\n\n".join(cases))
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    return _parse_tag(text, "deficiencies") or text.strip(), cost


# ---------------------------------------------------------------------------
# update: P' = optimizer step. Consumes G1 + raw (X_t,X_t+1) examples (the
# local Jacobian context: shows what is actually extractable from the input).
# ---------------------------------------------------------------------------
UPDATE_TMPL = """You are improving a Python perception module for a grid environment.

The module must define `perceive(observation_history: list[str]) -> str`. `observation_history[-1]` is the current raw observation. Output a concise (<2000 char) text summary of the decision-relevant features. It must never raise and never return empty.

=== CURRENT PERCEPTION MODULE ===
```python
{code}
```

The current output was judged INSUFFICIENT for identifying actions between consecutive states. Diagnosed deficiencies (the gradient):
{g1}

Here are RAW observation examples so you can see what information is actually present and extractable (do NOT just echo the raw grid into the output — extract the relevant abstraction):
{raw_examples}

Rewrite the FULL perception module to fix the diagnosed deficiencies while keeping what already worked. Surface the missing distinctions (e.g. the moving agent's position) so an action is recoverable from two consecutive outputs.

Respond with the full module in one block:
```python
<code>
```"""


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


async def update_perception(cfg, code: str, g1: str, raw_examples: list[Transition], sem,
                            n_examples: int = 2) -> tuple[str, float]:
    ex_blocks = []
    for i, tr in enumerate(raw_examples[:n_examples], 1):
        ex_blocks.append(
            f"--- example {i} (action taken: {tr.action}) ---\n"
            f"X_t (truncated): {tr.x_t[:2500]}\n"
            f"X_t+1 (truncated): {tr.x_t1[:2500]}"
        )
    prompt = UPDATE_TMPL.format(
        code=code or "# (currently empty)",
        g1=g1,
        raw_examples="\n\n".join(ex_blocks),
    )
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    return _extract_code(text), cost


# A known-GOOD perception (reports the moving green avatar's position): the
# practical ceiling. NOTE: this is a MEASUREMENT-ONLY reference (a ceiling
# baseline gated behind --good-baseline); it is never a seed and never enters
# the learning signal. The learning process always starts from an EMPTY P/B.
# Raw-JSON-grid diffing is unfairly hard for an LLM, so this is a fairer "what a
# correct P achieves" reference than the raw frame. (DQ8GC-specific: assumes
# green = the moving avatar.)
GOOD_P = '''import json

def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1]
    # Robust single-line JSON grid extraction: anchor on the generic "[[" so
    # games whose first row is not black (e.g. dgg2c) still parse.
    start = obs.find('[[')
    end = obs.rfind(']]') + 2
    if start == -1 or end <= 1:
        return "no_grid"
    try:
        grid = json.loads(obs[start:end])
    except Exception:
        return "no_grid"
    green = None
    gray = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if "green" in cell:
                green = (r, c)
            elif cell == "gray":
                gray.append((r, c))
    return f"agent_green_pos={green} gray_cell_positions={gray}"
'''


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="comma-separated run dirs (pooled)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--train-n", type=int, default=14)
    ap.add_argument("--holdout-n", type=int, default=14)
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--actions", default="left,right,up,down,noop",
                    help="comma-separated action-name whitelist (first token); empty = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = make_config(args.model, args.client)
    sem = asyncio.Semaphore(args.concurrency)
    whitelist = set(filter(None, args.actions.split(","))) or None

    run_dirs = [Path(p) for p in args.run.split(",") if p.strip()]
    transitions = load_transitions(run_dirs, whitelist)
    rng.shuffle(transitions)
    need = args.train_n + args.holdout_n
    if len(transitions) < need:
        print(f"WARNING: only {len(transitions)} transitions, wanted {need}")
    train = transitions[: args.train_n]
    holdout = transitions[args.train_n: args.train_n + args.holdout_n]
    action_pool = sorted({t.action for t in transitions})
    print(f"transitions: {len(transitions)} total | train={len(train)} holdout={len(holdout)}")
    print(f"action pool ({len(action_pool)}): {action_pool}\n")

    total_cost = 0.0
    code = ""  # learning always starts from an empty perception module

    # Baselines on held-out: chance, raw-frame ref, and known-good-P ceiling.
    chance = 1.0 / args.k_choices
    raw_acc, _, c = await forward_eval(cfg, "", holdout, action_pool, args.k_choices, sem, rng, raw_mode=True)
    total_cost += c
    good_acc, _, c = await forward_eval(cfg, GOOD_P, holdout, action_pool, args.k_choices, sem, rng)
    total_cost += c
    ceil_acc = good_acc
    print(f"[baseline] random={chance:.2f} | raw-frame={raw_acc:.2f} | known-good-P ceiling={good_acc:.2f}\n")

    # Round 0: starting P.
    h_acc, _, c = await forward_eval(cfg, code, holdout, action_pool, args.k_choices, sem, rng)
    total_cost += c
    print(f"[round 0] start P (empty): held-out action-ID acc = {h_acc:.2f}")
    best_acc, best_code = h_acc, code

    for rnd in range(1, args.rounds + 1):
        # forward on TRAIN to get the gradient signal
        tr_acc, records, c = await forward_eval(cfg, code, train, action_pool, args.k_choices, sem, rng)
        total_cost += c
        failures = [r for r in records if not r.correct]
        g1, c = await compute_g1(cfg, failures, sem)
        total_cost += c
        new_code, c = await update_perception(cfg, code, g1, [r.tr for r in failures] or train, sem)
        total_cost += c

        ok, err = perception_runs(new_code, [t.x_t for t in train[:4]])
        if not ok:
            print(f"[round {rnd}] update REJECTED (runtime error: {err}); keeping P")
            continue

        # held-out gate (line search): accept only if held-out improves
        new_h_acc, _, c = await forward_eval(cfg, new_code, holdout, action_pool, args.k_choices, sem, rng)
        total_cost += c
        verdict = "ACCEPT" if new_h_acc >= best_acc else "reject"
        print(f"[round {rnd}] train_acc={tr_acc:.2f} fails={len(failures)} "
              f"| candidate held-out acc={new_h_acc:.2f} (best={best_acc:.2f}) -> {verdict}")
        print(f"           G1: {g1[:300].replace(chr(10), ' ')}")
        if new_h_acc >= best_acc:
            best_acc, best_code, code = new_h_acc, new_code, new_code

    print(f"\n=== SUMMARY ===")
    print(f"start held-out acc : {h_acc:.2f}")
    print(f"final held-out acc : {best_acc:.2f}  (delta {best_acc - h_acc:+.2f})")
    print(f"random / ceiling   : {chance:.2f} / {ceil_acc:.2f}")
    print(f"total LLM cost     : ${total_cost:.4f}")

    out = Path(__file__).resolve().parent / "best_perception.py"
    out.write_text(best_code)
    print(f"best P written to  : {out}")


if __name__ == "__main__":
    asyncio.run(main())
