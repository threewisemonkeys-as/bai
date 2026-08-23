"""Per-MECHANIC inverse-dynamics scores over the coverage exam.

Runs each arm (raw / wc / lmwm, + blind/oracle) on a game's coverage protocol and breaks
the score down by the core-mechanic bucket every item carries, so the single ID number
becomes a diagnostic profile: which dynamics did the learner actually capture. Reuses the
exact arms and one scoring rule from score_id_protocol.py.

Caveat (see mechanics.py): this is an INVERSE exam. Action-triggered buckets are directly
discriminable; purely passive buckets (the fall/slide splits, contagion, patrol, diffuse)
all carry action `noop`, so their score measures "recognised a passive step, not an
input" rather than telling the sub-outcomes apart.

    uv run python offline_learning/scripts/score_coverage.py --game bt3gb \
        --protocol logs/2026-08-11/human_unified/coverage_protocols/bt3gb_coverage.json
    uv run python offline_learning/scripts/score_coverage.py --all \
        --protocol-dir logs/2026-08-11/human_unified/coverage_protocols --normalise
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling script import

import score_id_protocol as sip  # noqa: E402
from offline_learning.id_protocol import score_set  # noqa: E402
from offline_learning.invdyn_core import make_config  # noqa: E402
from offline_learning.mechanics import MECHANICS  # noqa: E402

# arm -> (unified-run learner subdir) for the artifact each arm reads
ART_SUBDIR = {"lmwm": "rexpure", "wc": "worldcoder"}
UNIFIED = ROOT / "logs/2026-08-11/human_unified"


def score_game(game: str, proto_path: Path, arms: list[str], unified: Path,
               model: str, provider: str) -> tuple[list[dict], dict]:
    proto = json.loads(proto_path.read_text())
    trs = sip.load_transitions_for(proto)
    cfg = make_config(model, "openrouter", provider_order=provider,
                      reasoning_json='{"effort": "low"}')
    preds = {}
    for arm in arms:
        art = (unified / ART_SUBDIR[arm] / f"{game}_s1"
               if arm in sip.NEEDS_ARTIFACT else Path("."))
        preds[arm] = sip.ARMS[arm](proto, trs, art, cfg if arm in sip.NEEDS_LLM else None)

    rows = []
    for n, it in enumerate(proto["items"]):
        row = {"mechanic": it.get("mechanic"), "synthetic": it.get("synthetic"),
               "verified": it.get("s_true") is not None,
               "max": (score_set(it["truth"], it["s_true"]) if it.get("s_true") else None)}
        for arm in arms:
            row[arm] = score_set(it["truth"], preds[arm][n])
        rows.append(row)
    return rows, proto


def _agg(rows: list[dict], arms: list[str], keys: list[str], normalise: bool) -> dict:
    sel = [r for r in rows if r["mechanic"] in keys] if keys else rows
    if not sel:
        return {}
    out = {"n": len(sel), "synth": sum(1 for r in sel if r["synthetic"])}
    vr = [r for r in sel if r["max"]]
    ceil = sum(r["max"] for r in vr) / len(vr) if vr else None
    out["ceiling"] = ceil
    for arm in arms:
        raw = sum(r[arm] for r in sel) / len(sel)
        if normalise and ceil:
            vv = sum(r[arm] for r in vr) / len(vr)
            out[arm] = vv / ceil if ceil else None
        else:
            out[arm] = raw
    return out


def print_table(game: str, human: str, rows: list[dict], arms: list[str],
                normalise: bool) -> None:
    tag = "normalised" if normalise else "raw"
    print(f"\n===== {game} / {human}   per-mechanic ID ({tag}, chance {1/5:.2f}) =====")
    head = f"  {'mechanic':<22} {'src':<6} {'n':>3} {'ceil':>5} " + \
           " ".join(f"{a:>6}" for a in arms)
    print(head)
    for m in MECHANICS[game]:
        mid = m["id"]
        a = _agg(rows, arms, [mid], normalise)
        if not a:
            print(f"  {mid:<22} {'-':<6} {'0':>3}   (no items)")
            continue
        src = "synth" if a["synth"] == a["n"] else ("mix" if a["synth"] else "human")
        ce = f"{a['ceiling']:.2f}" if a["ceiling"] is not None else "  - "
        vals = " ".join(f"{a[arm]:6.3f}" if a[arm] is not None else "   -  " for arm in arms)
        print(f"  {mid:<22} {src:<6} {a['n']:>3} {ce:>5} {vals}")
    ov = _agg(rows, arms, [], normalise)
    ce = f"{ov['ceiling']:.2f}" if ov["ceiling"] is not None else "  - "
    vals = " ".join(f"{ov[arm]:6.3f}" if ov[arm] is not None else "   -  " for arm in arms)
    print(f"  {'OVERALL':<22} {'':<6} {ov['n']:>3} {ce:>5} {vals}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--protocol", help="single protocol json (with --game)")
    ap.add_argument("--protocol-dir", default=str(UNIFIED / "coverage_protocols"))
    ap.add_argument("--arms", default="raw,wc,lmwm")
    ap.add_argument("--unified-root", default=str(UNIFIED))
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--provider-order", default="groq")
    ap.add_argument("--normalise", action="store_true",
                    help="divide each arm by the per-mechanic engine ceiling")
    ap.add_argument("--out", help="write the per-item rows + aggregates here")
    args = ap.parse_args()

    from offline_learning.human_replay import GAMES
    games = sorted(GAMES) if args.all else [args.game]
    arms = args.arms.split(",")
    allrows = {}
    for g in games:
        pp = Path(args.protocol) if args.protocol else Path(args.protocol_dir) / f"{g}_coverage.json"
        rows, proto = score_game(g, pp, arms, Path(args.unified_root),
                                  args.model, args.provider_order)
        allrows[g] = {"human": proto.get("program"), "rows": rows}
        print_table(g, proto["game"], rows, arms, args.normalise)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(allrows, indent=1) + "\n")


if __name__ == "__main__":
    main()
