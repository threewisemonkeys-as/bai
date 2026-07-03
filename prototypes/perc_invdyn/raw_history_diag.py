"""Raw-history diagnostic (CEILING, no P training).

Question: on F5W3N, does giving the frozen decoder F one PRIOR raw frame X_{t-1}
(so it can read the deterministic-but-time-clocked background motion) let it
recover the action better than the raw pair (X_t, X_t+1) alone?

This is the cheap upper bound for "would a sequence help": everything is RAW
(bypasses P on purpose), so it answers "is the phase info even sufficient?",
independent of whether any learnable P could preserve it. If +history doesn't
lift the pair here, a through-P sequence won't either.

Paired design: both conditions score the SAME transitions with the SAME choice
sets and seed; the only difference is the presence of STATE 0. Averaged over seeds.

  uv run python prototypes/perc_invdyn/raw_history_diag.py --game f5w3n --seeds 1,2,3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from validate import _llm_call, _parse_tag, make_choices, make_config  # noqa: E402

GAMES = {  # game -> action whitelist (matches clean_sweep.py)
    "f5w3n": {"left", "right", "up", "noop"},
    "dq8gc": {"left", "right", "up", "down", "noop", "click"},
}

PAIR_TMPL = """Two consecutive states of a grid environment are summarized below (raw grids). Exactly one action was taken to get from STATE 1 to STATE 2.

=== STATE 1 ===
{x1}
=== STATE 2 ===
{x2}

The action was one of:
{choices}

Decide which single action was taken. If undecidable, make your best guess.

Respond as:
<reasoning>what changed and what action that implies</reasoning>
<action>the chosen action, copied verbatim from the list</action>"""

NEUTRAL_HIST_TMPL = """Three consecutive states of a grid environment are shown below (raw grids), in time order. Exactly one action was taken to get from STATE 1 to STATE 2. (STATE 0 is the frame just before STATE 1, provided as extra context.)

=== STATE 0 (context) ===
{x0}
=== STATE 1 ===
{x1}
=== STATE 2 ===
{x2}

The action taken from STATE 1 -> STATE 2 was one of:
{choices}

Decide which single action was taken from STATE 1 to STATE 2. If undecidable, make your best guess.

Respond as:
<reasoning>what changed from STATE 1 to STATE 2 and what action that implies</reasoning>
<action>the chosen action, copied verbatim from the list</action>"""

HIST_TMPL = """Three consecutive states of a grid environment are shown below (raw grids).


=== STATE 0 (context only) ===
{x0}
=== STATE 1 ===
{x1}
=== STATE 2 ===
{x2}

The action taken from STATE 1 -> STATE 2 was one of:
{choices}

Respond as:
<reasoning></reasoning>
<action>the chosen action, copied verbatim from the list</action>"""


def load_with_history(game: str):
    """Return list of (x_prev, x_t, x_t1, action) keeping true sequence order."""
    wl = GAMES[game]
    csv_path = HERE / "clean_data" / game / "episode_0" / "trajectory.csv"
    rows = list(csv.DictReader(csv_path.open()))
    items = []
    for i in range(1, len(rows) - 1):  # need a predecessor frame at i-1
        r, nxt, prev = rows[i], rows[i + 1], rows[i - 1]
        action = (r.get("Action") or "").strip()
        obs, obs_next, obs_prev = (
            (r.get("Observation") or ""),
            (nxt.get("Observation") or ""),
            (prev.get("Observation") or ""),
        )
        done = (r.get("Done") or "").strip().lower() in ("true", "1")
        if (
            done
            or not action
            or not obs.strip()
            or not obs_next.strip()
            or not obs_prev.strip()
        ):
            continue
        if action.split()[0] not in wl:
            continue
        items.append((obs_prev, obs, obs_next, action))
    return items


async def predict(cfg, tmpl, kw, choices, sem):
    prompt = tmpl.format(choices="\n".join(f"- {c}" for c in choices), **kw)
    async with sem:
        text, cost = await _llm_call(cfg, prompt)
    text = text or ""
    pred = _parse_tag(text, "action")
    if pred not in choices:
        for c in choices:
            if c.lower() in pred.lower() or (pred and pred.lower() in c.lower()):
                pred = c
                break
    return pred, cost


async def run_seed(cfg, items, pool, k, seed, sem):
    rng = random.Random(seed)
    # one fixed choice set per item per seed -> identical across both conditions (paired)
    choice_sets = [make_choices(a, pool, k, rng) for (_, _, _, a) in items]
    TRUNC = 6000
    pair_tasks, neut_tasks, hist_tasks = [], [], []
    for (x0, x1, x2, a), ch in zip(items, choice_sets):
        pair_tasks.append(
            predict(cfg, PAIR_TMPL, {"x1": x1[:TRUNC], "x2": x2[:TRUNC]}, ch, sem)
        )
        neut_tasks.append(
            predict(
                cfg,
                NEUTRAL_HIST_TMPL,
                {"x0": x0[:TRUNC], "x1": x1[:TRUNC], "x2": x2[:TRUNC]},
                ch,
                sem,
            )
        )
        hist_tasks.append(
            predict(
                cfg,
                HIST_TMPL,
                {"x0": x0[:TRUNC], "x1": x1[:TRUNC], "x2": x2[:TRUNC]},
                ch,
                sem,
            )
        )
    pair = await asyncio.gather(*pair_tasks)
    neut = await asyncio.gather(*neut_tasks)
    hist = await asyncio.gather(*hist_tasks)
    truth = [a for (_, _, _, a) in items]
    acc = lambda res: sum(p == t for (p, _), t in zip(res, truth)) / len(items)
    cost = sum(c for r in (pair, neut, hist) for _, c in r)
    return acc(pair), acc(neut), acc(hist), cost


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="f5w3n")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--task-model", default="google/gemini-2.5-flash")
    ap.add_argument("--client", default="openrouter")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    cfg = make_config(args.task_model, args.client)
    items = load_with_history(args.game)
    pool = sorted({a for (_, _, _, a) in items})
    k = min(args.k_choices, len(pool))
    chance = 1.0 / len(pool) if len(pool) < args.k_choices else 1.0 / args.k_choices
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    sem = asyncio.Semaphore(args.concurrency)

    print(f"game={args.game}  usable transitions (with predecessor) = {len(items)}")
    print(f"action pool = {pool}  k={k}  chance={chance:.2f}\n")
    print(f"{'seed':>4} {'pair':>6} {'neutH':>6} {'guidH':>6} {'$':>7}")
    pa, na, ha = [], [], []
    for s in seeds:
        p, n, h, c = await run_seed(cfg, items, pool, k, s, sem)
        pa.append(p)
        na.append(n)
        ha.append(h)
        print(f"{s:>4} {p:>6.2f} {n:>6.2f} {h:>6.2f} {c:>7.3f}")
    mean = lambda xs: sum(xs) / len(xs)
    mp, mn, mh = mean(pa), mean(na), mean(ha)
    print(f"\n{'MEAN':>4} {mp:>6.2f} {mn:>6.2f} {mh:>6.2f}")
    print(
        f"\npair={mp:.2f}  neutral-history={mn:.2f} (Δ{mn - mp:+.2f})  "
        f"guided-history={mh:.2f} (Δ{mh - mp:+.2f})  n={len(items)}x{len(seeds)}/cond"
    )
    print(
        f"verdict: best history variant ({max(mn, mh):.2f}) "
        f"{'HELPS' if max(mn, mh) - mp > 0.05 else 'does NOT beat'} the pair ({mp:.2f})"
    )


if __name__ == "__main__":
    asyncio.run(main())
