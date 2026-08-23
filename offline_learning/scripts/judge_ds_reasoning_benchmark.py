"""Rerun only the blind judge stage for a completed reasoning benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

import benchmark_ds_reasoning_effort_v2 as base
import benchmark_ds_reasoning_effort_v3 as transport


async def main(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    config = json.loads((out / "config.json").read_text())
    config["judge_max_tokens"] = args.judge_max_tokens
    config["judge_concurrency"] = args.judge_concurrency
    config["judge_provider"] = args.judge_provider
    run_args = argparse.Namespace(**config)

    source = Path(run_args.source)
    cases = base.sample(base.jsonl(source / "reflection_calls.jsonl"), "reflection",
                        run_args.per_component)
    cases += base.sample(base.jsonl(source / "analysis_calls.jsonl"), "analysis",
                         run_args.per_component)
    records = base.jsonl(out / "records.jsonl")

    base.streamed_call = transport.streamed_call
    judgments = await base.judge(cases, records, run_args)
    judgment_path = out / f"judgments_{args.judge_max_tokens}.jsonl"
    with judgment_path.open("w") as handle:
        for result in judgments:
            handle.write(json.dumps(result) + "\n")
    summary = base.summarize(records, judgments)
    summary_path = out / f"summary_judged_{args.judge_max_tokens}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["judge_firsts"], indent=2))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--judge-provider", default="cerebras")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--judge-concurrency", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    asyncio.run(main(arguments()))
