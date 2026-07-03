"""Run the GEPA inverse/forward-dynamics learner on the clean manual trajectories,
train=val (tied), with a held-out test for an unbiased inverse-dynamics accuracy.

Each (game, seed) calls gepa_optimize.py once and we parse the summary table.
Results -> logs/<out-name>/results.json.

Default config reproduces the antimemo reference run
(dgg2c_seed1_dkfix_gepa_dsv4flash_analyze_ctxk1_antimemo) EXCEPT context_k=3:
  task+reflection model = deepseek/deepseek-v4-flash, max_metric_calls=2000,
  fd-scorer=exact (composite 0.5*ID + 0.5*FD[exact]), --analyze-mistakes,
  context_k=3, belief-update-period=4 (gepa default), tie-train-val, GEPA-only.

  # full antimemo-config sweep (GEPA-only, seed 1, ctxk3) over the wired games:
  uv run python prototypes/perc_invdyn/clean_sweep.py --seeds 1 --parallel 7

  # head-to-head vs legacy greedy instead (old behaviour):
  uv run python prototypes/perc_invdyn/clean_sweep.py --compare --max-metric-calls 120
"""
from __future__ import annotations
import argparse, asyncio, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "clean_data"

# game -> (action whitelist, keep_action_params, gepa_metric_calls, legacy_rounds)
# keep_action_params=True keeps full action strings so the click LOCATION is part of
# the prediction target (gepa default); False collapses 'click 3 5' -> 'click'.
# 64x64 ARC grids get a larger GEPA budget so reflection has room to DISCOVER the
# right object color (the small autumn grids converge in ~120 calls).
#
# For autumn games where movement actions discriminate transitions, click is collapsed
# (keep=False). For click-only autumn games the click LOCATION is the only signal, so
# keep=True (otherwise every transition is just 'click' and the task is degenerate).
GAMES = {
    "dq8gc": ("left,right,up,down,noop,click", False, 120, 6),
    "f5w3n": ("left,right,up,noop", False, 120, 6),
    "ls20":  ("ACTION1,ACTION2,ACTION3,ACTION4", False, 220, 6),
    "ft09":  ("ACTION6", True, 220, 6),  # predict click LOCATION
    # --- additional ARC-AGI-3 games (whitelists = reset available_actions;
    #     keep=True whenever ACTION6 click is present so click LOCATION is the target) ---
    "vc33":  ("ACTION6", True, 220, 6),
    "tn36":  ("ACTION6", True, 220, 6),
    "lp85":  ("ACTION6", True, 220, 6),
    "sp80":  ("ACTION1,ACTION2,ACTION3,ACTION4,ACTION5,ACTION6", True, 220, 6),
    "dc22":  ("ACTION1,ACTION2,ACTION3,ACTION4,ACTION6", True, 220, 6),
    "cn04":  ("ACTION1,ACTION2,ACTION3,ACTION4,ACTION5,ACTION6", True, 220, 6),
    "m0r0":  ("ACTION1,ACTION2,ACTION3,ACTION4,ACTION5,ACTION6", True, 220, 6),
    "tr87":  ("ACTION1,ACTION2,ACTION3,ACTION4", False, 220, 6),
    "wa30":  ("ACTION1,ACTION2,ACTION3,ACTION4,ACTION5", False, 220, 6),
    # --- original autumn games ---
    "dgg2c": ("left,right,up,down,noop,click", True, 120, 6),  # matches antimemo ref (kept click loc)
    "bt3gb": ("left,right,up,down,noop,click", False, 120, 6),
    "ice":   ("left,right,up,down,noop,click", False, 120, 6),
    "n2ntd": ("left,right,up,down,noop,click", False, 120, 6),
    # --- movement(+click) autumn games: collapse click, movement carries the signal ---
    "7www9": ("left,right,up,down,noop", False, 120, 6),
    "7xf97": ("left,right,up,down,noop,click", False, 120, 6),
    "e3v6m": ("left,right,up,down,noop,click", False, 120, 6),
    "eahcw": ("left,right,up,down,noop,click", False, 120, 6),
    "glacier": ("left,right,up,down,noop,click", False, 120, 6),
    "orchrd": ("left,right,up,down,noop,click", False, 120, 6),
    "qfsvc": ("left,right,up,down,noop,click", False, 120, 6),
    "qqm74": ("left,right,up,down,noop,click", False, 120, 6),
    # --- click-only autumn games: arrows are no-ops, click LOCATION is the signal ---
    "27vwc": ("noop,click", True, 120, 6),
    "83wkq": ("noop,click", True, 120, 6),
    "ada85": ("noop,click", True, 120, 6),
    "aw9wd": ("noop,click", True, 120, 6),
    "nrdf6": ("noop,click", True, 120, 6),
    "ntq4y": ("noop,click", True, 120, 6),
    "s2kt7": ("noop,click", True, 120, 6),
    "va6fq": ("noop,click", True, 120, 6),
    "vqjh6": ("noop,click", True, 120, 6),
}


def parse_summary(text: str) -> dict:
    out = {}
    for key, pat in [
        ("raw", r"raw-frame ref\s+([0-9.]+)"),
        ("gepa_acc", r"GEPA \(pareto.*?\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"),
        ("legacy_acc", r"legacy greedy P/B loop\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"),
        ("chance", r"random=([0-9.]+)"),
        ("start_acc", r"start-P=([0-9.]+)"),
    ]:
        m = re.search(pat, text)
        if not m:
            continue
        if key in ("gepa_acc", "legacy_acc"):
            out[key] = float(m.group(1))
            out[key.replace("acc", "cost")] = float(m.group(2))
            out[key.replace("acc", "time")] = float(m.group(3))
        else:
            out[key] = float(m.group(1))
    return out


async def run_one(game, seed, args, sem, outroot):
    wl, keep, _mmc, lrounds = GAMES[game]
    outd = outroot / f"{game}_seed{seed}"
    refl = args.reflection_model or args.task_model
    data_root = HERE / args.data_root
    if args.cross_traj:  # distinct crafted train/test trajectories (clean_data2/<game>/{train,test})
        run_args = ["--run", str(data_root / game / "train"),
                    "--test-run", str(data_root / game / "test")]
    else:  # pooled: single trajectory, in-pool balanced test carve (clean_data/<game>)
        run_args = ["--run", str(data_root / game)]
    cmd = [
        sys.executable, str(HERE / "gepa_optimize.py"),
        *run_args,
        "--train-n", str(args.train_n), "--val-n", str(args.train_n),
        "--test-n", str(args.test_n), "--tie-train-val",
        "--actions", wl,
        "--fd-scorer", args.fd_scorer,
        "--max-metric-calls", str(args.max_metric_calls),
        "--task-model", args.task_model, "--reflection-model", refl,
        "--context-k", str(args.context_k),
        "--concurrency", str(args.concurrency),
        "--seed", str(seed), "--out-dir", str(outd),
    ]
    if args.analyze:
        cmd.append("--analyze-mistakes")
    if args.warm_start_perception:  # seed the loop from a parser scaffold (gepa --start-perception)
        cmd += ["--start-perception", str(Path(args.warm_start_perception).resolve())]
    if args.compare:  # opt-in head-to-head vs legacy greedy
        cmd += ["--compare", "--legacy-rounds", str(lrounds)]
    if not keep:  # keep=True -> predict click LOCATION (gepa default); False -> collapse to verb
        cmd.append("--collapse-action-params")
    if args.image_mode:  # images to P & B proposers
        cmd.append("--image-mode")
    if args.f_image:  # images to inverse/forward objective scoring calls
        cmd.append("--f-image")
    async with sem:
        print(f"[start] {game} seed{seed}", flush=True)
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(HERE), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
        raw, _ = await proc.communicate()
    text = raw.decode(errors="replace")
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "stdout.txt").write_text(text)
    res = parse_summary(text)
    res.update({"game": game, "seed": seed, "rc": proc.returncode})
    print(f"[done ] {game} seed{seed} rc={proc.returncode} "
          f"gepa={res.get('gepa_acc')} legacy={res.get('legacy_acc')}", flush=True)
    return res


# default sweep: all autumn games with clean_data EXCEPT orchrd, glacier, and the
# dq8gc_clickmove variant (per request). ARC games ls20/ft09 are excluded too.
DEFAULT_GAMES = ("dq8gc,f5w3n,dgg2c,bt3gb,ice,n2ntd,"
                 "27vwc,7www9,7xf97,83wkq,ada85,aw9wd,e3v6m,eahcw,"
                 "nrdf6,ntq4y,qfsvc,qqm74,s2kt7,va6fq,vqjh6")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--games", default=DEFAULT_GAMES)
    ap.add_argument("--train-n", type=int, default=20)
    ap.add_argument("--test-n", type=int, default=10)
    # antimemo-reference GEPA config (context_k bumped 1 -> 3):
    ap.add_argument("--task-model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--reflection-model", default=None, help="defaults to --task-model")
    ap.add_argument("--max-metric-calls", type=int, default=2000)
    ap.add_argument("--context-k", type=int, default=9)
    ap.add_argument("--data-root", default="clean_data2",
                    help="dir under prototypes/perc_invdyn holding the trajectories")
    ap.add_argument("--cross-traj", action=argparse.BooleanOptionalAction, default=True,
                    help="use distinct train/test trajectories (<data-root>/<game>/{train,test}) "
                         "via gepa --test-run; --no-cross-traj reverts to the pooled single-"
                         "trajectory in-pool split")
    ap.add_argument("--fd-scorer", default="exact",
                    choices=["none", "textdiff", "judge", "exact"])
    ap.add_argument("--analyze", action=argparse.BooleanOptionalAction, default=True,
                    help="--analyze-mistakes on the gepa run (default on, matches antimemo)")
    ap.add_argument("--warm-start-perception", default="",
                    help="path to a seed perception module fed to gepa --start-perception "
                         "(warm-start). Use prototypes/perc_invdyn/autumn_seed_perception.py for "
                         "the general Autumn grid-parser scaffold (avoids the parse-collapse local "
                         "optimum). Empty = start from the empty perception (default).")
    ap.add_argument("--image-mode", action="store_true",
                    help="render each state as an image for the P & B proposers (requires a "
                         "vision task/reflection model, e.g. google/gemini-2.5-flash)")
    ap.add_argument("--f-image", action="store_true",
                    help="ALSO show images to the inverse/forward objective SCORING calls "
                         "(independent of --image-mode; large vision-call volume)")
    ap.add_argument("--compare", action="store_true",
                    help="also run legacy greedy P/B loop head-to-head (default GEPA-only)")
    ap.add_argument("--legacy-rounds", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=16, help="per-run F-eval concurrency")
    ap.add_argument("--parallel", type=int, default=7, help="concurrent (game,seed) runs")
    ap.add_argument("--out-name", default="clean_sweep_gepa_padded_ctxk9",
                    help="subdir under logs/ for artifacts")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    games = [g for g in args.games.split(",") if g.strip()]
    outroot = HERE.parent.parent / "logs" / args.out_name
    outroot.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] {len(games)} games x {len(seeds)} seed(s) | model={args.task_model} "
          f"ctxk={args.context_k} mmc={args.max_metric_calls} fd={args.fd_scorer} "
          f"analyze={args.analyze} compare={args.compare} | data={args.data_root} "
          f"cross_traj={args.cross_traj} -> {outroot}", flush=True)

    sem = asyncio.Semaphore(args.parallel)
    tasks = [run_one(g, s, args, sem, outroot) for g in games for s in seeds]
    results = await asyncio.gather(*tasks)
    json.dump(results, (outroot / "results.json").open("w"), indent=2)

    # aggregate
    print("\n================ AGGREGATE (mean over seeds) ================")
    print(f"{'game':<8} {'n':>2} {'chance':>7} {'raw':>6} {'GEPA':>6} {'legacy':>7} "
          f"{'GEPA$':>7} {'leg$':>7}")
    for g in games:
        rs = [r for r in results if r["game"] == g and r.get("gepa_acc") is not None]
        if not rs:
            print(f"{g:<8} (no parsed results)")
            continue
        n = len(rs)
        mean = lambda k: sum(r.get(k, 0) for r in rs) / n
        print(f"{g:<8} {n:>2} {mean('chance'):>7.2f} {mean('raw'):>6.2f} "
              f"{mean('gepa_acc'):>6.2f} {mean('legacy_acc'):>7.2f} "
              f"{mean('gepa_cost'):>7.3f} {mean('legacy_cost'):>7.3f}")
    print(f"\nartifacts -> {outroot}")


if __name__ == "__main__":
    asyncio.run(main())
