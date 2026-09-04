"""The offline training data, rendered as an in-context block for the ICL planning arm.

The `icl` arm is the *raw* planner handed the SAME offline interaction data the NLWM
world model was fit on, pasted into its prompt. It is the like-for-like control for
"does the learned model do anything the data alone does not?": both arms see the same
transitions, the same planner, the same problems and budgets; only the representation
differs (60 raw transitions vs. the learned perception module + beliefs).

Matching the learner exactly
----------------------------
`rexpure_optimize.build_data` loads `<pool>/train_d*` with 9-frame context backfilled
from `<pool>/drives/train_d*`, strips the observation metadata header, then takes
`--train-n` rows via `balanced_split`. Every game's pool holds exactly 60 transitions
and every launch passed `--train-n 60`, so the split is the identity: the learner's
train batch IS the whole pool. This module therefore renders the whole pool, and
`assert_matches_launch` re-checks that equality against the recorded `launch.json`
rather than trusting it -- if a future pool is larger than `--train-n`, the block would
silently become a superset of what the learner saw and the comparison would be unfair.

Only the ORDER differs (the learner shuffles under its seed); the set is identical.

    uv run python offline_learning/icl_context.py --game bt3gb --stats
    uv run python offline_learning/icl_context.py --game diffusion --render diff | head -40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from invdyn_core import (  # noqa: E402
    backfill_context_from_source, load_transitions, strip_transitions_obs_metadata,
)

DEFAULT_DATA_ROOT = REPO / "offline_learning/human_data"
DEFAULT_POOL = "informative_curated"
POOL_CONTEXT_K = 9          # the context depth the learner backfilled with
RENDERS = ("full", "diff")

_HEADER = """=== OFFLINE INTERACTION DATA ({n} recorded transitions from this environment) ===
Below is the complete offline dataset recorded from THIS environment by a human player.
Each entry shows a state, the action taken in it, and the state that followed. Use them
to infer how the environment behaves -- what each action does, and what changes on its
own. Entries are independent samples drawn from several play sessions: consecutive
entries are NOT consecutive in time, so do not read the list as one trajectory.{extra}
"""

_FOOTER = "=== END OFFLINE INTERACTION DATA ===\n"

_DIFF_NOTE = ("\nNEXT STATE is written as the cells that changed, in the form "
              "`(row, col): before -> after`; every unlisted cell is unchanged.")


def load_pool_transitions(game: str, *, pool: str = DEFAULT_POOL,
                          data_root: Path | None = None, context_k: int = POOL_CONTEXT_K):
    """The learner's training pool for `game`, preprocessed exactly as it was for training."""
    root = Path(data_root or DEFAULT_DATA_ROOT) / game / pool
    if not root.is_dir():
        raise FileNotFoundError(f"no training pool at {root}")
    dirs = sorted(root.glob("train_d*"))
    if not dirs:
        raise FileNotFoundError(f"no train_d* splits under {root}")
    transitions = []
    for td in dirs:
        src = root / "drives" / td.name
        tt = load_transitions([td], None, context_k=context_k)
        if src.is_dir():
            backfill_context_from_source(tt, [src], None, context_k=context_k)
        transitions.extend(tt)
    # the learner ran without --keep-obs-metadata: the Task:/Step: header is a side
    # channel and is stripped there, so it must be stripped here too
    strip_transitions_obs_metadata(transitions)
    return transitions


def _grid(text: str):
    return json.loads(text)


def _diff_lines(a: str, b: str) -> list[str]:
    """The cells that differ, as `(row, col): before -> after`."""
    try:
        ga, gb = _grid(a), _grid(b)
    except (json.JSONDecodeError, TypeError):
        return None
    if len(ga) != len(gb) or any(len(ra) != len(rb) for ra, rb in zip(ga, gb)):
        return None
    return [f"  ({r}, {c}): {x} -> {y}"
            for r, (ra, rb) in enumerate(zip(ga, gb))
            for c, (x, y) in enumerate(zip(ra, rb)) if x != y]


def _render_next(x_t: str, x_t1: str, render: str) -> str:
    if render == "full":
        return f"NEXT STATE:\n{x_t1}"
    changed = _diff_lines(x_t, x_t1)
    if changed is None:                     # ragged or unparsable: fall back, never lie
        return f"NEXT STATE:\n{x_t1}"
    if not changed:
        return "NEXT STATE: no cell changed."
    return "NEXT STATE: identical to STATE except\n" + "\n".join(changed)


def render_block(transitions, *, render: str = "full", context_k: int = 0) -> str:
    """Render the transitions as the prompt block. `context_k` prepends each item's
    preceding frames (the learner's perception saw them); 0 shows bare (s, a, s')."""
    if render not in RENDERS:
        raise ValueError(f"unknown render {render!r}; choose from {', '.join(RENDERS)}")
    extra = _DIFF_NOTE if render == "diff" else ""
    if context_k:
        extra += (f"\nEach entry is preceded by up to {context_k} earlier frame(s) from the "
                  "same session, so time-dependent behaviour is visible.")
    parts = [_HEADER.format(n=len(transitions), extra=extra)]
    for i, t in enumerate(transitions, 1):
        parts.append(f"--- transition {i}/{len(transitions)} ---")
        if context_k:
            ctx = t.ctx_prev[-context_k:] if t.ctx_prev else []
            for j, (state, action) in enumerate(ctx):
                lag = len(ctx) - j
                parts.append(f"EARLIER STATE (t-{lag}):\n{state}")
                parts.append(f"  action taken: {action}")
        parts.append(f"STATE:\n{t.x_t}")
        parts.append(f"ACTION: {t.action}")
        parts.append(_render_next(t.x_t, t.x_t1, render))
        parts.append("")
    parts.append(_FOOTER)
    return "\n".join(parts)


def assert_matches_launch(game: str, transitions, artifact_root: Path) -> dict:
    """Refuse to build a block that is not what the learner trained on.

    Reads the recorded `--train-n` from the rexpure run's launch.json. If the pool is
    larger than the batch the learner actually scored, the block would hand the ICL arm
    MORE data than the world model ever saw and the comparison would be unfair, so this
    raises instead of quietly proceeding. A missing launch.json is reported, not fatal:
    the caller decides (the evaluator warns and continues, so a pool without a paired
    learner run is still usable for a standalone probe)."""
    lj = Path(artifact_root) / "rexpure" / f"{game}_s1" / "launch.json"
    if not lj.is_file():
        return {"checked": False, "reason": f"no launch record at {lj}"}
    cmd = json.loads(lj.read_text()).get("cmd", [])
    train_n = None
    for flag, value in zip(cmd, cmd[1:]):
        if flag == "--train-n":
            train_n = int(value)
    if train_n is None or train_n < 0:
        return {"checked": False, "reason": "launch record has no --train-n"}
    if len(transitions) > train_n:
        raise RuntimeError(
            f"{game}: training pool has {len(transitions)} transitions but the world "
            f"model was fit on --train-n {train_n}; the ICL block would be a superset "
            f"of the learner's data. Subsample to the learner's batch before comparing.")
    return {"checked": True, "train_n": train_n, "pool_n": len(transitions)}


def build_icl_block(game: str, *, pool: str = DEFAULT_POOL, data_root: Path | None = None,
                    render: str = "full", context_k: int = 0,
                    artifact_root: Path | None = None) -> tuple[str, dict]:
    """`(block_text, meta)` for `game`. `artifact_root` enables the fairness check."""
    transitions = load_pool_transitions(game, pool=pool, data_root=data_root)
    check = ({"checked": False, "reason": "no artifact root given"} if artifact_root is None
             else assert_matches_launch(game, transitions, Path(artifact_root)))
    block = render_block(transitions, render=render, context_k=context_k)
    return block, {"game": game, "pool": pool, "render": render, "icl_context_k": context_k,
                   "n_transitions": len(transitions), "chars": len(block),
                   "est_tokens": len(block) // 4, "match_check": check}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, help="a game name, or 'all'")
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--artifact-root", default="logs/2026-08-24/human_curated")
    ap.add_argument("--render", choices=RENDERS, default="full")
    ap.add_argument("--icl-context-k", type=int, default=0)
    ap.add_argument("--stats", action="store_true", help="print sizes only, not the block")
    a = ap.parse_args()

    root = Path(a.data_root)
    games = ([p.name for p in sorted(root.iterdir()) if (p / a.pool).is_dir()]
             if a.game == "all" else [a.game])
    metas = []
    for g in games:
        block, meta = build_icl_block(
            g, pool=a.pool, data_root=root, render=a.render,
            context_k=a.icl_context_k, artifact_root=Path(a.artifact_root))
        metas.append(meta)
        if not a.stats and len(games) == 1:
            print(block)
    if a.stats or len(games) > 1:
        print(f"{'game':14s} {'n':>4s} {'chars':>9s} {'~tokens':>8s}  check")
        for m in metas:
            c = m["match_check"]
            tag = (f"pool {c['pool_n']} == train-n {c['train_n']}" if c.get("checked")
                   else f"UNCHECKED ({c.get('reason')})")
            print(f"{m['game']:14s} {m['n_transitions']:4d} {m['chars']:9d} "
                  f"{m['est_tokens']:8d}  {tag}")


if __name__ == "__main__":
    main()
