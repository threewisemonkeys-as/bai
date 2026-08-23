#!/usr/bin/env bash
# GEPA with --image-mode + gemini-2.5-flash (F + proposer) on two ARC games.
# image-mode shows the proposer RENDERED PNGs of the grids (vs the 1500-char text
# prefix); it forces context_k=0 (single-step). Same other params as the in-flight
# deepseek batch: start empty, fd exact 0.5 +reflect, analyze-mistakes, 20/20/10
# tied, 2000 metric calls, seed 1.
set -u
cd /home/ays57/bai
GEPA=offline_learning/gepa_optimize.py
DATA=offline_learning/clean_data
OUT=logs/clean_sweep_fwd

run_one() {
  local game="$1" actions="$2"
  local dir="$OUT/${game}_seed1_imgmode_gemini_analyze"
  mkdir -p "$dir"
  uv run python "$GEPA" --run "$DATA/$game" --actions "$actions" \
    --image-mode \
    --train-n 20 --val-n 20 --test-n 10 --tie-train-val --start empty \
    --fd-scorer exact --fd-weight 0.5 --fd-reflect --analyze-mistakes \
    --accept-ties --max-metric-calls 2000 \
    --task-model google/gemini-2.5-flash --seed 1 --concurrency 16 \
    --out-dir "$dir" > "$dir/run.log" 2>&1
  echo "DONE $game exit=$? -> $dir"
}

run_one ls20 "ACTION1,ACTION2,ACTION3,ACTION4" &
run_one sp80 "ACTION1,ACTION2,ACTION3,ACTION4,ACTION5" &
wait
echo "BOTH IMAGE-MODE GEMINI RUNS COMPLETE"
