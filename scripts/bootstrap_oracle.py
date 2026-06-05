"""Bootstrap oracle artifacts for AutumnBench tasks.

For each task, produces:
    data/oracle/<task>/summary.md
    data/oracle/<task>/ground_truth_beliefs.md   (LLM draft -- review/edit before use)

Usage:
    uv run scripts/bootstrap_oracle.py ice
    uv run scripts/bootstrap_oracle.py 27VWC bbq --force
    uv run scripts/bootstrap_oracle.py --all
    uv run scripts/bootstrap_oracle.py ice --model openai/gpt-5.5-pro
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autumn_oracle import (  # noqa: E402
    PROGRAMS_DIR,
    STRONG_MODEL,
    derive_ground_truth_beliefs,
    summarize_program,
    task_oracle_dir,
    task_program_path,
)


def _list_tasks() -> list[str]:
    return sorted(
        p.stem
        for p in PROGRAMS_DIR.glob("*.sexp")
        if not p.stem.endswith("_change_detection_wrong_program")
    )


async def _bootstrap_one(task: str, *, model: str, force: bool) -> None:
    if not task_program_path(task).exists():
        print(f"[skip] {task}: no .sexp program at {task_program_path(task)}")
        return
    out_dir = task_oracle_dir(task)
    print(f"[{task}] summarizing program with {model}...")
    _, summary_cost = await summarize_program(task, model=model, force=force)
    print(f"[{task}]   summary -> {out_dir / 'summary.md'}  (cost ${summary_cost:.4f})")
    print(f"[{task}] drafting ground-truth beliefs...")
    _, gt_cost = await derive_ground_truth_beliefs(task, model=model, force=force)
    print(f"[{task}]   gt beliefs -> {out_dir / 'ground_truth_beliefs.md'}  (cost ${gt_cost:.4f})")
    print(f"[{task}] EDIT {out_dir / 'ground_truth_beliefs.md'} before running the oracle loop.")


async def _main_async(args: argparse.Namespace) -> None:
    tasks = _list_tasks() if args.all else args.tasks
    if not tasks:
        print("No tasks specified. Pass TASK names or --all.", file=sys.stderr)
        sys.exit(2)
    for task in tasks:
        await _bootstrap_one(task, model=args.model, force=args.force)


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap oracle artifacts for AutumnBench tasks.")
    p.add_argument("tasks", nargs="*", help="Task names (e.g., 'ice', '27VWC').")
    p.add_argument("--all", action="store_true", help="Bootstrap every task with a .sexp program.")
    p.add_argument("--model", default=STRONG_MODEL, help=f"LiteLLM model id (default: {STRONG_MODEL}).")
    p.add_argument("--force", action="store_true", help="Re-generate even if outputs already exist.")
    args = p.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
