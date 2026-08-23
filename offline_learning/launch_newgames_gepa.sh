#!/usr/bin/env bash
# Launch GEPA (deepseek-v4-flash, ctx-k=3, --analyze-mistakes) on the 6 new
# clean-data games, with the SAME parameters as the reference run
# dq8gc_seed1_dkfix_swap_gepa_dsv4flash (minus swap-click-train, which is
# dq8gc-specific). 4 autumn (n2ntd/ice/bt3gb/dgg2c) + 2 arc (sp80/vc33).
set -u
cd /home/ays57/bai
GEPA=offline_learning/gepa_optimize.py
DATA=offline_learning/clean_data
OUT=logs/clean_sweep_fwd

run_one() {
  local game="$1" actions="$2"
  local dir="$OUT/${game}_seed1_dkfix_gepa_dsv4flash_analyze"
  mkdir -p "$dir"
  uv run python "$GEPA" --run "$DATA/$game" --actions "$actions" \
    --train-n 20 --val-n 20 --test-n 10 --tie-train-val --start empty \
    --fd-scorer exact --fd-weight 0.5 --fd-reflect --analyze-mistakes \
    --context-k 3 --accept-ties --max-metric-calls 2000 \
    --task-model deepseek/deepseek-v4-flash --seed 1 --concurrency 16 \
    --out-dir "$dir" > "$dir/run.log" 2>&1
  echo "DONE $game exit=$? -> $dir"
}

run_one n2ntd "left,right,up,down,noop,click" &
run_one ice   "left,right,up,down,noop,click" &
run_one bt3gb "left,right,up,down,noop,click" &
run_one dgg2c "left,right,up,down,noop,click" &
run_one sp80  "ACTION1,ACTION2,ACTION3,ACTION4,ACTION5" &
run_one vc33  "ACTION6" &
wait
echo "ALL 6 NEW-GAME GEPA RUNS COMPLETE"
