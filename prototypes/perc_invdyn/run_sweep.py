"""Durable launcher for the low-data GEPA sweep (6 games x N seeds, ARC + AutumnBench).

Replaces the old ad-hoc shell loop that wrote to /tmp/seeds5 (wiped between sessions).
Everything for later analysis now lands under a single durable, timestamped sweep dir
in the repo:

  logs/perc_invdyn/sweep_<timestamp>/
    envs.tsv                         <- manifest the analysis scripts read (game/mode/dirs)
    <game>/seed<N>/                  <- per-run --out-dir passed to gepa_optimize.py:
        best_perception_gepa_seed<N>.py / best_beliefs_gepa_seed<N>.txt
        test_trace_{raw,gepa,legacy}_seed<N>.json   (per-item P out + F reasoning + truth)
        gepa_run_seed<N>/            <- full GEPA run_dir: all candidates + scores + lineage
        run.log                      <- teed stdout/stderr of the gepa_optimize invocation

Usage:
  uv run prototypes/perc_invdyn/run_sweep.py                  # all games, seeds 1-5
  uv run prototypes/perc_invdyn/run_sweep.py --games ice,ft09 --seeds 1,2
  uv run prototypes/perc_invdyn/run_sweep.py --jobs 4         # run up to 4 (game,seed) in parallel
  uv run prototypes/perc_invdyn/run_sweep.py --dry-run        # print commands only

Then analyse by pointing analyze_seeds.py / dump_seeds5.py at the printed sweep dir
(set OUT_DIR / TSV to it), no /tmp involved.
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# game -> (image_mode, actions, keep_params, [source transition run dirs])
# Source dirs (relative to logs/) are the stepwise_eb_learn runs from
# exploration_runs_catalog.md (recommended trajectory per game). keep_params=True for
# coordinate-click games (vc33/tn36/ft09): the full 'ACTION6 x= y=' string is the
# prediction target (predict the click LOCATION). Short trajectories are padded with a
# second catalog trajectory so each game has >=50 transitions (10 train + 20 val + 20 test).
GAMES = {
    # --- AutumnBench (text grids) ---
    "ice": (False, "left,right,up,down,noop,click", False, [
        "logs/seed_autumn/ice/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "DQ8GC": (False, "left,right,up,down,noop,click", False, [
        "logs/seed_autumn/DQ8GC/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "7WWW9": (False, "left,right,up,down,noop,click", False, [
        "logs/seed_autumn/7WWW9/2026-06-05_14-07-07_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "ADA85": (False, "left,right,up,down,noop,click", False, [  # padded (recommended=39 trans)
        "logs/dynamics_full/ADA85/2026-06-08_12-49-01_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn",
        "logs/dynamics_full_perc_fix/ADA85/2026-06-11_18-24-12_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    # --- ARC-AGI-3 (image-mode, integer grids) ---
    "ls20": (True, "ACTION1,ACTION2,ACTION3,ACTION4,ACTION5", False, [
        "logs/matrix_v6/arc5_noperc_50steps/eb_learn__arc_agi__gemini-2p5-flash__ls20/2026-06-04_11-49-26_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "sp80": (True, "ACTION1,ACTION2,ACTION3,ACTION4,ACTION5", False, [  # padded (recommended=43 trans)
        "logs/matrix_v5/v5_50steps/eb_learn__arc_agi__gemini-2p5-flash__sp80/2026-06-03_16-24-57_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn",
        "logs/arc_parallel/20260528-184557/eb_learn__arc_agi__gemini-2p5-flash__sp80/2026-05-28_18-46-01_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    # coordinate-click games: keep_params=True -> predict click location (full ACTION6 x= y= string)
    "vc33": (True, "ACTION6", True, [  # padded
        "logs/matrix_v6/arc5_noperc_50steps/eb_learn__arc_agi__gemini-2p5-flash__vc33/2026-06-04_11-49-26_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn",
        "logs/matrix_v5/v5_50steps/eb_learn__arc_agi__gemini-2p5-flash__vc33/2026-06-03_16-24-58_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "tn36": (True, "ACTION6", True, [
        "logs/matrix_v5/v5_50steps/eb_learn__arc_agi__gemini-2p5-flash__tn36/2026-06-03_16-24-58_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
    "ft09": (True, "ACTION6", True, [  # padded; ft09 has no clean ACTION6 run (catalog: weak spot)
        "logs/matrix_v6/arc5_noperc_50steps/eb_learn__arc_agi__gemini-2p5-flash__ft09/2026-06-04_11-49-26_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn",
        "logs/matrix_v5/v5_50steps/eb_learn__arc_agi__gemini-2p5-flash__ft09/2026-06-03_16-24-58_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"]),
}

# low-data regime that the sweep used (matches the prior runs)
DEFAULTS = dict(task_model="google/gemini-2.5-flash", client="openrouter",
                train_n=5, test_n=20, val_n=20, k_choices=5, max_metric_calls=120)


def build_cmd(game, seed, out_dir, args):
    image, actions, keep_params, run_dirs = GAMES[game]
    runs = ",".join(str(REPO / d) for d in run_dirs)
    cmd = [sys.executable, str(HERE / "gepa_optimize.py"),
           "--run", runs, "--actions", actions,
           "--seed", str(seed), "--out-dir", str(out_dir),
           "--task-model", args.task_model, "--client", args.client,
           "--train-n", str(args.train_n), "--test-n", str(args.test_n),
           "--val-n", str(args.val_n), "--k-choices", str(args.k_choices),
           "--max-metric-calls", str(args.max_metric_calls)]
    if not args.no_tie:                 # default: low-data tied train==val
        cmd.append("--tie-train-val")
    if image:
        cmd.append("--image-mode")
    if keep_params:                     # coordinate-click games: predict click location
        cmd.append("--keep-action-params")
    if args.compare:
        cmd.append("--compare")
    return cmd


def run_one(game, seed, sweep_dir, args):
    out_dir = sweep_dir / game / f"seed{seed}"
    cmd = build_cmd(game, seed, out_dir, args)
    tag = f"{game} seed{seed}"
    if args.dry_run:
        print(f"[dry-run] {tag}\n  {' '.join(cmd)}")
        return tag, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"
    print(f"[start] {tag} -> {out_dir}  (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=str(REPO), stdout=f,
                            stderr=subprocess.STDOUT).returncode
    status = "ok" if rc == 0 else f"FAILED rc={rc}"
    print(f"[done ] {tag}  {status}  [{time.time()-t0:.0f}s]", flush=True)
    return tag, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(GAMES),
                    help="comma-separated subset of: " + ",".join(GAMES))
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--jobs", type=int, default=1, help="parallel (game,seed) processes")
    ap.add_argument("--sweep-dir", default=None,
                    help="durable output root (default: logs/perc_invdyn/sweep_<timestamp>)")
    ap.add_argument("--compare", action="store_true", help="also run the legacy greedy loop")
    ap.add_argument("--no-tie", action="store_true",
                    help="decouple train/val: GEPA selects on a held-out val-n set (not in-sample train)")
    ap.add_argument("--dry-run", action="store_true")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    args = ap.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    bad = [g for g in games if g not in GAMES]
    if bad:
        ap.error(f"unknown games: {bad}")

    sweep_dir = (Path(args.sweep_dir) if args.sweep_dir
                 else REPO / "logs" / "perc_invdyn" / f"sweep_{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"[sweep] {len(games)} games x {len(seeds)} seeds -> {sweep_dir}")
    if not args.dry_run:
        sweep_dir.mkdir(parents=True, exist_ok=True)
        # manifest the analysis scripts read: game \t mode \t out-subdir \t source-run-dirs
        tsv = sweep_dir / "envs.tsv"
        with open(tsv, "w") as f:
            f.write("game\tmode\tout_subdir\trun_dirs\n")
            for g in games:
                image, _actions, _keep, run_dirs = GAMES[g]
                f.write(f"{g}\t{'image' if image else 'text'}\t{g}\t"
                        f"{','.join(str(REPO / d) for d in run_dirs)}\n")
        print(f"[sweep] manifest: {tsv}\n")

    jobs = [(g, s) for g in games for s in seeds]
    if args.jobs > 1 and not args.dry_run:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(lambda gs: run_one(gs[0], gs[1], sweep_dir, args), jobs))
    else:
        results = [run_one(g, s, sweep_dir, args) for g, s in jobs]

    if not args.dry_run:
        failed = [tag for tag, rc in results if rc != 0]
        print(f"\n[sweep] complete: {len(results)-len(failed)}/{len(results)} ok"
              + (f" | FAILED: {failed}" if failed else ""))
        print(f"[sweep] all artifacts under {sweep_dir}")


if __name__ == "__main__":
    main()
