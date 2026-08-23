#!/usr/bin/env bash
# Curated planning evals (OFFLINE then ONLINE) for the no-perception ablation.
#
# Same problem set, same planner model and same protocol as the reference run's evals in
# logs/2026-08-18/curated/eval/ -- only --artifact-root changes, so the `lmwm` arm reads the
# ablation's identity-P + learned-beliefs artifacts instead of the learned-P ones.
#
# The `raw` and `wc` arms do not read those artifacts (see
# scripts/seed_curated_online_ckpt.py), so for the ONLINE eval they are seeded from the
# reference run's checkpoint rather than resampled: one shared control, half the spend.
# The OFFLINE eval is cheap and re-runs all three arms.
#
#   bash offline_learning/launch/run_aug19_noperc_curated_eval.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT=logs/2026-08-19/noperc_ablation
OUT=$ROOT/curated_eval
REF=logs/2026-08-18/curated
PY=.venv/bin/python
mkdir -p "$OUT"

for g in bt3gb dq8gc n2ntd s2kt7 83wkq; do
  test -f "$ROOT/rexpure/${g}_s1/best_beliefs_rexpure_seed1.txt" \
    || { echo "missing ablation artifact for $g -- learning not finished"; exit 1; }
done

echo "=== OFFLINE (open-loop) ==="
$PY -u offline_learning/scripts/eval_curated_plan.py \
    --problems "$REF/problems.json" \
    --artifact-root "$ROOT" \
    --out "$OUT/offline" \
    --concurrency 16 2>&1 | tee "$OUT/eval_offline.log"

# The reference run's ONLINE eval may still be in flight. Wait it out before starting ours:
# (1) its checkpoint is what we seed the shared raw/wc arms from, and a partial one means
# paying to resample the rest; (2) both jobs drive the same provider at concurrency 64, so
# running them together earns 429s on both and slows the one that is already hours in.
while pgrep -f "eval_curated_online.py .*logs/2026-08-18/curated" >/dev/null; do
  echo "[gate] reference ONLINE eval still running ($(wc -l < "$REF/eval/online.ckpt.jsonl")/330 rollouts); waiting 5 min"
  sleep 300
done

echo "=== seed shared arms into the ONLINE checkpoint ==="
$PY offline_learning/scripts/seed_curated_online_ckpt.py \
    --from "$REF/eval/online.ckpt.jsonl" --to "$OUT/online.ckpt.jsonl" \
    2>&1 | tee "$OUT/seed_online_ckpt.log"

echo "=== ONLINE (receding horizon) ==="
timeout 43200 $PY -u offline_learning/scripts/eval_curated_online.py \
    --problems "$REF/problems.json" \
    --artifact-root "$ROOT" \
    --offline "$OUT/offline.json" \
    --out "$OUT/online" \
    --concurrency 64 2>&1 | tee "$OUT/eval_online.log"
