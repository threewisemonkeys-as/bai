#!/usr/bin/env bash
# Resume the pinned image-mode+gemini runs from their saved best (warm-start),
# with an ADDITIONAL 2000 metric-call budget. Same seed/split/pins.
set -u; cd /home/ays57/bai
GEPA=prototypes/perc_invdyn/gepa_optimize.py; DATA=prototypes/perc_invdyn/clean_data; OUT=logs/clean_sweep_fwd
run_one(){ local game="$1" actions="$2" pin="$3"; local prev="$OUT/${game}_seed1_imgmode_gemini_pinned"; local dir="$OUT/${game}_seed1_imgmode_gemini_pinned_resume"; mkdir -p "$dir"
  uv run python "$GEPA" --run "$DATA/$game" --actions "$actions" --image-mode \
    --train-n 20 --val-n 20 --test-n 10 --tie-train-val \
    --start-perception "$prev/best_perception_gepa_seed1.py" \
    --start-beliefs "$prev/best_beliefs_gepa_seed1.txt" \
    --fd-scorer exact --fd-weight 0.5 --fd-reflect --analyze-mistakes \
    --pin-train-idx "$pin" --accept-ties --max-metric-calls 2000 \
    --task-model google/gemini-2.5-flash --seed 1 --concurrency 16 \
    --out-dir "$dir" > "$dir/run.log" 2>&1; echo "DONE $game exit=$? -> $dir"; }
run_one sp80 "ACTION1,ACTION2,ACTION3,ACTION4,ACTION5" "11,17" &
run_one ls20 "ACTION1,ACTION2,ACTION3,ACTION4" "6,16,19,20,32" &
wait; echo "BOTH RESUME RUNS COMPLETE"
