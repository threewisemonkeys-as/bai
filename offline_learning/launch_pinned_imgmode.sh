#!/usr/bin/env bash
# GEPA --image-mode + gemini on the finalized sp80/ls20 clean data, with the KEY
# transitions (submits / walls / landmarks) pinned into the scored train set.
set -u; cd /home/ays57/bai
GEPA=offline_learning/gepa_optimize.py; DATA=offline_learning/clean_data; OUT=logs/clean_sweep_fwd
run_one(){ local game="$1" actions="$2" pin="$3"; local dir="$OUT/${game}_seed1_imgmode_gemini_pinned"; mkdir -p "$dir"
  uv run python "$GEPA" --run "$DATA/$game" --actions "$actions" --image-mode \
    --train-n 20 --val-n 20 --test-n 10 --tie-train-val --start empty \
    --fd-scorer exact --fd-weight 0.5 --fd-reflect --analyze-mistakes \
    --pin-train-idx "$pin" --accept-ties --max-metric-calls 2000 \
    --task-model google/gemini-2.5-flash --seed 1 --concurrency 16 \
    --out-dir "$dir" > "$dir/run.log" 2>&1; echo "DONE $game exit=$? -> $dir"; }
run_one sp80 "ACTION1,ACTION2,ACTION3,ACTION4,ACTION5" "11,17" &
run_one ls20 "ACTION1,ACTION2,ACTION3,ACTION4" "6,16,19,20,32" &
wait; echo "BOTH PINNED IMAGE-MODE RUNS COMPLETE"
