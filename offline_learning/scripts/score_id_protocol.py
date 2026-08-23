"""Score any learner against a frozen ID protocol, with one rule for everybody.

Every arm implements the same one-line interface -- given an item, return a subset of
that item's five choices -- and every arm is graded by `id_protocol.score_set`
(`1/|pred|` when the truth is in it, else 0). What differs between arms is only how
the set is produced:

  lmwm    render the window through the learned perception module, prompt the task LLM
          with the learned beliefs, parse its <actions> block
  wc      forward-simulate each choice with the synthesized program and keep the ones
          that reproduce X_t+1 (elimination, no LLM)
  raw     the same prompt as lmwm but on unprocessed frames, no perception, no beliefs
  blind   return all five choices (the uninformed floor: scores 1/k on every item)
  oracle  return the engine-verified s_true (the attainable ceiling)

Results are reported both raw and normalised by that protocol's ceiling, so pools with
different amounts of genuine action-aliasing become comparable.

    uv run python offline_learning/scripts/score_id_protocol.py \
        --protocol logs/.../protocols/bt3gb_human.json \
        --arm lmwm --artifact-dir logs/aug10_human_origin/rexpure/bt3gb_s1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import offline_learning.program_runtime as prt  # noqa: E402
from offline_learning.id_protocol import score_set  # noqa: E402
from offline_learning.invdyn_core import (  # noqa: E402
    build_window,
    make_config,
    predict_action_set_from_window,
    run_async,
)
from offline_learning.validate import (  # noqa: E402
    backfill_context_from_source,
    load_transitions,
)
from offline_learning.worldcoder_optimize import choice_consistent  # noqa: E402


def load_transitions_for(proto: dict) -> list:
    """Re-derive the protocol's transitions with the identical loader, then verify the
    fingerprint so a changed dataset can never be scored as if it were the pinned one."""
    import hashlib
    wl = set(proto["actions"])
    td, sd = proto["test_dirs"], proto["source_dirs"]
    trs = []
    if len(sd) == len(td):
        for t, s in zip(td, sd):
            tt = load_transitions([Path(t)], wl, context_k=proto["context_k"])
            if Path(s).resolve() != Path(t).resolve():
                try:
                    backfill_context_from_source(tt, [Path(s)], wl, proto["context_k"])
                except ValueError:
                    if proto["pool"] != "coverage":
                        raise  # must match id_protocol.build's tolerant coverage path
            trs.extend(tt)
    else:
        trs = load_transitions([Path(p) for p in td], wl, proto["context_k"])
    fp = hashlib.sha1(json.dumps(
        [[t.x_t, t.action, t.x_t1] for t in trs], sort_keys=True).encode()).hexdigest()[:16]
    if fp != proto["fingerprint"]:
        raise SystemExit(f"fingerprint mismatch: data changed under protocol "
                         f"{proto['game']}/{proto['pool']} ({fp} != {proto['fingerprint']})")
    return trs


# ------------------------------------------------------------------------- arms
def arm_blind(proto, trs, art, cfg):
    return [list(it["choices"]) for it in proto["items"]]


def arm_oracle(proto, trs, art, cfg):
    return [it.get("s_true") for it in proto["items"]]


def _lmwm(proto, trs, art, cfg, raw: bool):
    code = "" if raw else (art / "best_perception_rexpure_seed1.py").read_text()
    beliefs = "" if raw else (art / "best_beliefs_rexpure_seed1.txt").read_text()

    async def go():
        sem = asyncio.Semaphore(8)

        async def one(it, tr):
            win, _ = build_window(code, tr)
            pred, _txt, _c, _p = await predict_action_set_from_window(
                cfg, win, beliefs, it["choices"], sem)
            return pred
        return await asyncio.gather(*[one(it, tr)
                                      for it, tr in zip(proto["items"], trs)])
    return run_async(go())


def arm_lmwm(proto, trs, art, cfg):
    return _lmwm(proto, trs, art, cfg, raw=False)


def arm_raw(proto, trs, art, cfg):
    return _lmwm(proto, trs, art, cfg, raw=True)


def arm_wc(proto, trs, art, cfg):
    code = (art / "best_transition_wc_seed1.py").read_text()
    items = prt.prepare_transitions(trs, proto["context_k"])
    # choice_consistent memoises on (it.idx, choice) and PreparedTransition.idx
    # defaults to -1 for every item, so without unique ids the memo returns item 0's
    # verdict for the whole pool. worldcoder_optimize stamps these before scoring too.
    for i, it in enumerate(items):
        it.idx = 30_000_000 + i
    rt = prt.ProgramRuntime(code, timeout_s=1.0)
    memo: dict = {}
    try:
        return [[c for c in it["choices"] if choice_consistent(rt, item, c, memo)]
                for it, item in zip(proto["items"], items)]
    finally:
        rt.close()


ARMS = {"lmwm": arm_lmwm, "wc": arm_wc, "raw": arm_raw,
        "blind": arm_blind, "oracle": arm_oracle}
NEEDS_LLM = {"lmwm", "raw"}
NEEDS_ARTIFACT = {"lmwm", "wc"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--artifact-dir")
    ap.add_argument("--label", help="name for this arm in the output (default: --arm)")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--provider-order", default="groq")
    ap.add_argument("--out", help="write the per-item result JSON here")
    args = ap.parse_args()

    proto = json.loads(Path(args.protocol).read_text())
    if args.arm in NEEDS_ARTIFACT and not args.artifact_dir:
        ap.error(f"--arm {args.arm} needs --artifact-dir")
    trs = load_transitions_for(proto)
    cfg = (make_config(args.model, "openrouter", provider_order=args.provider_order,
                       reasoning_json='{"effort": "low"}')
           if args.arm in NEEDS_LLM else None)

    preds = ARMS[args.arm](proto, trs, Path(args.artifact_dir or "."), cfg)
    rows = []
    for it, pred in zip(proto["items"], preds):
        rows.append({"i": it["i"], "truth": it["truth"], "pred": pred,
                     "s_true": it.get("s_true"),
                     "mechanic": it.get("mechanic"), "synthetic": it.get("synthetic"),
                     "score": score_set(it["truth"], pred),
                     "max": (score_set(it["truth"], it["s_true"])
                             if it.get("s_true") else None)})
    n = len(rows)
    raw = sum(r["score"] for r in rows) / n
    vr = [r for r in rows if r["max"] is not None]
    ceil = sum(r["max"] for r in vr) / len(vr) if vr else None
    norm = (sum(r["score"] for r in vr) / len(vr) / ceil) if ceil else None
    strict = sum(1.0 for r in rows if r["pred"] == [r["truth"]]) / n
    out = {"game": proto["game"], "pool": proto["pool"],
           "arm": args.label or args.arm, "n": n,
           "raw": round(raw, 4),
           "ceiling": round(ceil, 4) if ceil is not None else None,
           "normalised": round(norm, 4) if norm is not None else None,
           "strict": round(strict, 4),
           "mean_set_size": round(sum(len(r["pred"] or []) for r in rows) / n, 3),
           "empty_sets": sum(1 for r in rows if not r["pred"]),
           "protocol_fingerprint": proto["fingerprint"], "rows": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{proto['game']:6s} {proto['pool']:10s} {out['arm']:8s} "
          f"raw={out['raw']:.3f} ceil={out['ceiling']} norm={out['normalised']} "
          f"strict={out['strict']:.3f} |S|={out['mean_set_size']:.2f} "
          f"empty={out['empty_sets']}", flush=True)


if __name__ == "__main__":
    main()
