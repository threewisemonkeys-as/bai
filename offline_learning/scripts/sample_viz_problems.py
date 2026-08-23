"""Pick a diagnostic subset of coverage planning problems to re-run enriched for the viz.

Uses the COMPLETED main-run outcomes (coverage_plan_eval.json) to categorise every problem
-- world-model win (lmwm hit, raw miss), all-fail, arms-disagree, all-succeed -- then selects
a spread per game across those categories AND across buckets/horizons, so the viz showcases
the interesting cases (not just easy act wins). Writes a problems json in the same shape as
coverage_plan_problems.json so eval_coverage_plan.py --problems <this> re-runs exactly them.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = json.load(open(REPO / "logs/coverage_plan_eval.json"))
FULL = json.load(open(REPO / "logs/coverage_plan_problems.json"))

PER_GAME = int(sys.argv[1]) if len(sys.argv) > 1 else 12
KEY = lambda r: (r["game"], r["seed"], r["t"], r["bucket"], r["mechanic"], r["h"])
by_key = {KEY(p): p for p in FULL["problems"]}

def cat(r):
    s = {a: r[a]["success"] for a in ("raw", "lmwm", "wc")}
    if s["lmwm"] and not s["raw"]:
        return "wmwin"          # world model beats raw-grid
    if not any(s.values()):
        return "allfail"        # nobody solves it
    if len(set(s.values())) > 1:
        return "split"          # arms disagree
    return "allok"

# desired mix per game (falls back to whatever exists)
QUOTA = [("wmwin", 5), ("split", 3), ("allfail", 2), ("allok", 2)]

picked = []
for res in EVAL["results"]:
    game = res["game"]
    pools = defaultdict(list)
    for r in res["rows"]:
        pools[cat(r)].append(r)
    # spread each pool over buckets+horizons before taking
    def spread(rows):
        rows = sorted(rows, key=lambda r: (r["bucket"], r["h"], r["mechanic"], r["seed"]))
        seen, out, rest = set(), [], []
        for r in rows:
            k = (r["bucket"], r["h"])
            (out if k not in seen else rest).append(r)
            seen.add(k)
        return out + rest
    chosen, used = [], set()
    for c, q in QUOTA:
        for r in spread(pools.get(c, []))[:q]:
            if KEY(r) not in used:
                chosen.append(r); used.add(KEY(r))
    # top up to PER_GAME with anything, ensuring >=1 wait and >=1 maintain if they exist
    for want_bucket in ("wait", "maintain"):
        if not any(r["bucket"] == want_bucket for r in chosen):
            extra = [r for row in res["rows"] if (r := row)["bucket"] == want_bucket
                     and KEY(r) not in used]
            if extra:
                chosen.append(extra[0]); used.add(KEY(extra[0]))
    for r in res["rows"]:
        if len(chosen) >= PER_GAME:
            break
        if KEY(r) not in used:
            chosen.append(r); used.add(KEY(r))
    for r in chosen[:PER_GAME]:
        picked.append(by_key[KEY(r)])

out = {**{k: FULL[k] for k in FULL if k != "problems"}, "n": len(picked), "problems": picked}
dst = REPO / "logs/coverage_plan_sample.json"
dst.write_text(json.dumps(out, indent=1))
comp = defaultdict(lambda: defaultdict(int))
for p in picked:
    comp[p["game"]][p["bucket"]] += 1
print(f"sampled {len(picked)} problems -> {dst}")
for g, c in comp.items():
    print(f"  {g}: {dict(c)}")
