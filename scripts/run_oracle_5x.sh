#!/usr/bin/env bash
# Launch 5 parallel oracle runs (20 steps each).
# Learner + scorer: openrouter/google/gemini-2.5-flash.
# Oracle: openrouter/openai/gpt-5.5-pro (STRONG_MODEL).
set -u
cd "$(dirname "$0")/.."

TASKS=(ice 83WKQ S2KT7 DQ8GC EAHCW)
LOG_DIR=/tmp/oracle_5x
mkdir -p "$LOG_DIR"

for task in "${TASKS[@]}"; do
  uv run python -u -m explore.stepwise_eb_learn_oracle "$task" \
      --selector llm \
      --num-steps 20 \
      --candidates-per-step 6 \
      --learner-model openrouter/google/gemini-2.5-flash \
      --scorer-model openrouter/google/gemini-2.5-flash \
      > "$LOG_DIR/$task.log" 2>&1 &
  echo "launched $task  pid=$!"
done

wait
echo "all 5 runs complete"
