"""Entry points for the manual planning-problem pipeline.

    # 1. play (records sessions) and 2. curate (windows -> problems, live audit)
    uv run python -m offline_learning.manual_plan.cli serve

    # 3. re-audit everything on disk from a cold engine, ignoring stored verdicts
    uv run python -m offline_learning.manual_plan.cli audit --all
    uv run python -m offline_learning.manual_plan.cli audit --game ice --verbose

    # 4. ship the passing set in the shape the eval harness consumes
    uv run python -m offline_learning.manual_plan.cli export --all --out logs/manual_plan/problems.json

    uv run python -m offline_learning.manual_plan.cli report --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline_learning.manual_plan import audit as A
from offline_learning.manual_plan import problems as P
from offline_learning.manual_plan import session as S


def _games(args) -> list[str]:
    if args.all:
        return S.ORDER
    if not args.game:
        raise SystemExit("pass --game GAME (or --all)")
    return [S.canon(g) for g in args.game]


def cmd_serve(args) -> None:
    import uvicorn
    print(f"play    http://{args.host}:{args.port}/static/play.html")
    print(f"curate  http://{args.host}:{args.port}/curate")
    uvicorn.run("offline_learning.manual_plan.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="warning")


def cmd_audit(args) -> None:
    total = bad = 0
    for g in _games(args):
        ps = P.load(g)
        for p in ps:
            r = A.audit(p, n_random=args.n_random)
            p["audit"] = r
            total += 1
            bad += (not r["ok"])
            fails = [s["id"] for s in r["screens"].values() if not s["ok"]]
            print(f"{p['id']:<14} h={p['h']:<3} {'PASS' if r['ok'] else 'FAIL ' + ','.join(fails)}"
                  f"   {p.get('note','')}")
            if args.verbose:
                for s in r["screens"].values():
                    print(f"    {'ok ' if s['ok'] else 'BAD'} {s['id']:<18} {s['detail']}")
                for w in r["warnings"]:
                    print(f"    !   {w}")
        if ps:
            P.save(g, ps)
    print(f"\n{total - bad}/{total} passing")


def cmd_export(args) -> None:
    out = Path(args.out)
    r = P.export(_games(args), out, only_passing=not args.include_failing)
    print(json.dumps(r, indent=1))


def cmd_report(args) -> None:
    rows = []
    for g in _games(args):
        ps = P.load(g)
        ok = [p for p in ps if (p.get("audit") or {}).get("ok")]
        hs = sorted(p["h"] for p in ok)
        rows.append((g, len(S.list_sessions(g)), len(ps), len(ok),
                     f"{hs[0]}-{hs[-1]}" if hs else "-",
                     sum(len(p["gt_actions"]) - p["gt_actions"].count("noop") for p in ok)))
    w = f"{'game':<8}{'sessions':>9}{'problems':>10}{'passing':>9}{'h range':>10}{'non-noop acts':>15}"
    print(w)
    print("-" * len(w))
    for r in rows:
        print(f"{r[0]:<8}{r[1]:>9}{r[2]:>10}{r[3]:>9}{r[4]:>10}{r[5]:>15}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the play + curate web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8764)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    for name, fn, helptext in [("audit", cmd_audit, "re-run every screen from a cold engine"),
                               ("export", cmd_export, "write the passing set for the eval harness"),
                               ("report", cmd_report, "per-game authoring progress")]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--game", action="append")
        p.add_argument("--all", action="store_true")
        p.set_defaults(fn=fn)
        if name == "audit":
            p.add_argument("--n-random", type=int, default=12)
            p.add_argument("--verbose", action="store_true")
        if name == "export":
            p.add_argument("--out", default="logs/manual_plan/problems.json")
            p.add_argument("--include-failing", action="store_true")

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
