#!/usr/bin/env python3
"""Turn bench_planner_models results into the speed/quality table + a wall-clock projection.

The projection answers the only question that matters here: if the curated ONLINE eval
(300 LLM rollouts, cap 50) were re-run with this arm, how long would it take?  It is a
projection, not a measurement -- the measured baseline it is calibrated against is the
606-minute production run, and the finalist gets re-measured end to end.

    uv run python offline_learning/scripts/report_planner_bench.py
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "logs/2026-08-19/planner_bench"

# --- calibration from the completed production run (logs/2026-08-18/curated/eval/online.md)
# All measured from logs/2026-08-18/curated/eval/online.json, not assumed.
PROD_ROLLOUTS = 300          # 2 LLM arms x 30 problems x 5 attempts
PROD_CALLS = 9009            # rounds (8993) + corrective re-asks (16)
PROD_WALL_MIN = 606.0
PROD_CONCURRENCY = 48
PROD_PASS1 = 0.507
CAP = 50                     # actions per rollout
ROUNDS_ON_SUCCESS = 13.8     # a win ends the rollout early...
ROUNDS_ON_FAILURE = 46.6     # ...a loss burns nearly the whole budget


def rounds_for(pass_rate: float) -> float:
    """Expected rounds per rollout at a given success rate. Failures are the expensive
    case -- they burn nearly the whole budget -- so a better planner is also a faster
    one, and the two effects have to be counted together."""
    return pass_rate * ROUNDS_ON_SUCCESS + (1 - pass_rate) * ROUNDS_ON_FAILURE


def paired_section(res: dict, baseline: str) -> list[str]:
    """Every arm ran the identical job list, so arms can be compared cell by cell instead
    of as two independent proportions -- which is the difference between "these look
    similar" and "the difference is inside the noise"."""
    if baseline not in res:
        return []
    random.seed(0)
    cell = {}
    for k, v in res.items():
        g = defaultdict(list)
        for r in v["rows"]:
            g[(r["key"], r["arm_kind"])].append(0 if "error" in r else int(r["success"]))
        cell[k] = {j: sum(x) / len(x) for j, x in g.items()}
    keys = sorted(set.intersection(*[set(d) for d in cell.values()]))

    L = ["", "## Is any of this a real quality difference?", "",
         f"Paired against `{baseline}` over the {len(keys)} identical (problem x arm) "
         f"cells every arm ran, with a paired bootstrap CI and a two-sided sign test. "
         f"An interval spanning 0 means the arms are not distinguishable on this subset.",
         "", "| arm | pass@1 | vs baseline | 95% CI | sign p |", "|---|--:|--:|---|--:|"]
    for k in sorted(cell, key=lambda k: -statistics.mean(cell[k][j] for j in keys)):
        a_ = [cell[k][j] for j in keys]
        b_ = [cell[baseline][j] for j in keys]
        diff = [x - y for x, y in zip(a_, b_)]
        boots = sorted(statistics.mean(random.choices(diff, k=len(diff))) for _ in range(8000))
        w = sum(1 for d in diff if d > 0)
        l = sum(1 for d in diff if d < 0)
        n = w + l
        p = min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0
        L.append(f"| `{k}` | {statistics.mean(a_):.2f} | {statistics.mean(diff):+.2f} | "
                 f"[{boots[200]:+.2f}, {boots[7800]:+.2f}] | {p:.3f} |")
    return L


def probe_section(path: Path, title: str, blurb: str) -> list[str]:
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    ok = {k: v for k, v in d.items() if v["summary"]["n_ok"]}
    bad = {k: v for k, v in d.items() if not v["summary"]["n_ok"]}
    L = ["", f"## {title}", "", blurb, "",
         "| model @ provider | p50 s | tok/s | out tok | thinking tok | $/call |",
         "|---|--:|--:|--:|--:|--:|"]
    for k, v in sorted(ok.items(), key=lambda kv: kv[1]["summary"]["p50_s"]):
        s = v["summary"]
        L.append(f"| `{k}` | {s['p50_s']:.1f} | {s['tok_s']:.0f} | {s['out_tok']:.0f} | "
                 f"{s['rsn_tok']:.0f} | {s['cost_call']:.5f} |")
    if bad:
        L += ["", "Rejected the call outright: "
              + ", ".join(f"`{k}` ({v['rows'][0].get('error', '?')[:60]}...)"
                          for k, v in bad.items())]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=[str(OUT / "arms_results.json")])
    ap.add_argument("--baseline", default="ds-prod")
    ap.add_argument("--out", default=str(OUT / "REPORT.md"))
    ap.add_argument("--providers", default=str(OUT / "providers.json"))
    ap.add_argument("--candidates", default=str(OUT / "candidates.json"))
    a = ap.parse_args()

    res: dict = {}
    for f in a.results:
        p = Path(f)
        if p.exists():
            res |= json.loads(p.read_text())

    dead = {k: v for k, v in res.items() if not v["summary"]["n_ok"]}
    res = {k: v for k, v in res.items() if v["summary"]["n_ok"]}
    base = res.get(a.baseline, {}).get("summary")
    # What per-call latency would explain the production wall if the fan-out were fully
    # busy?  This is the number every arm below is projected against.
    prod_implied = PROD_WALL_MIN * 60 * PROD_CONCURRENCY / PROD_CALLS

    L = ["# Planner model/provider/effort bench", "",
         f"Subset of the curated planning problems, real prompts, plans executed in the "
         f"Autumn engine.  {len(res)} arms measured"
         + (f", {len(dead)} unmeasurable (see the end)." if dead else "."), "",
         f"The online wall projection is RELATIVE: the `{a.baseline}` arm is pinned to "
         f"the {PROD_WALL_MIN:.0f} min that configuration actually took ({PROD_CALLS} "
         f"calls over {PROD_ROLLOUTS} rollouts at concurrency {PROD_CONCURRENCY}), and "
         f"every other arm is scaled by its measured mean latency and by how many calls "
         f"its pass@1 implies -- a win ends a rollout after ~{ROUNDS_ON_SUCCESS:.0f} "
         f"rounds, a loss burns ~{ROUNDS_ON_FAILURE:.0f}. Ratios travel even though "
         f"absolute latency does not: `{a.baseline}` measured "
         + (f"{base['mean_s']:.0f} s mean here against the {prod_implied:.0f} s its "
            f"production wall implies, because its dead provider pin re-rolls the "
            f"routing lottery on every call. The finalist is re-measured end to end "
            f"rather than trusted." if base else ""), "",
         "| arm | p50 s | p90 s | mean s | out tok | think tok | pass@1 | raw | lmwm | "
         "bad plans | $/call | proj. online wall | proj. $ |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]

    rows = sorted(res.items(), key=lambda kv: kv[1]["summary"]["p50_s"])
    base_calls = (rounds_for(base["pass1"]) * PROD_ROLLOUTS * (1 + base["invalid"])
                  if base else 1.0)
    for label, r in rows:
        s = r["summary"]
        rounds = rounds_for(s["pass1"]) * PROD_ROLLOUTS
        calls = rounds * (1 + s["invalid"])          # one corrective re-ask per bad plan
        wall_min = (PROD_WALL_MIN * (calls * s["mean_s"])
                    / (base_calls * base["mean_s"])) if base else 0.0
        L.append(
            f"| `{label}` | {s['p50_s']:.1f} | {s['p90_s']:.1f} | {s['mean_s']:.1f} | "
            f"{s['out_tok']:.0f} | "
            f"{s['rsn_tok']:.0f} | {s['pass1']:.2f} | {s['pass1_raw']:.2f} | "
            f"{s['pass1_lmwm']:.2f} | {s['invalid']:.2f} | {s['cost_call']:.5f} | "
            f"{wall_min:.0f} min | ${calls * s['cost_call']:.0f} |")

    L += ["", "## Errors and routing", "",
          "| arm | failed calls | providers actually used |", "|---|--:|---|"]
    for label, r in rows:
        s = r["summary"]
        L.append(f"| `{label}` | {s['n_err']}/{s['n']} | {', '.join(s['providers'])} |")

    L += paired_section(res, a.baseline)
    L += probe_section(Path(a.providers), "Provider probe: which endpoint is fastest?",
                       "Two real planning prompts per endpoint, sequential within an "
                       "endpoint so the number is latency rather than queueing. Catalog "
                       "throughput stats were only used to pick who to probe.")
    L += probe_section(Path(a.candidates), "Explored candidates",
                       "Same probe, run over a shortlist drawn from a full-catalog scan "
                       "of every endpoint above 200 tok/s plus known-strong models. "
                       "Screened on latency only; survivors were queued for scored arms.")

    if dead:
        L += ["", "## Not measured", "",
              "Every call failed, so these arms have no result at all -- do not read a "
              "blank row as a bad score.", ""]
        L += [f"- `{k}`: {v['rows'][0].get('error', '?')[:160]}" for k, v in dead.items()]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
