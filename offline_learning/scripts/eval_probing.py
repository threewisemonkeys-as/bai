#!/usr/bin/env python3
"""Prepare reproducible probe prompts; add --execute to send bounded model requests."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing_common import read_json  # noqa: E402
from probing_eval import OpenRouterClient, execute, prepare, score_jobs  # noqa: E402
from probing_formats import DEFAULT_FORMAT, FORMATS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoints", default="first_working,p25,p50,p75,final")
    p.add_argument("--arms", default="", help="comma-separated learned arms; baselines remain shared")
    p.add_argument("--forward-modes", default="learned_raw,learned_native,raw,lossless,copy")
    p.add_argument("--reconstruction-modes", default="learned,raw,lossless,constant,hash,shuffled,lossless_inverse,train_mode")
    p.add_argument("--examples", type=int, default=None, help="nested prefix of frozen training examples")
    p.add_argument("--grid-format", default=None,
                   help=f"raw-grid wire format, one of {sorted(FORMATS)} (default: {DEFAULT_FORMAT}); "
                        "'frozen' keeps the original JSON-only wording and strict parser")
    p.add_argument("--model", default=None, help="defaults to the reference run's evaluator model")
    p.add_argument("--provider", default=None, help="defaults to saved evaluator routing; empty disables pinning")
    p.add_argument("--reasoning-json", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--max-prompt-chars", type=int, default=131072)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--execute", action="store_true", help="send model requests (default: prepare and score cached/baseline results)")
    p.add_argument("--max-calls", type=int, default=1000, help="maximum uncached logical requests; each may use --attempts API attempts")
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retry-errors", action="store_true")
    a = p.parse_args()
    manifest = read_json(a.manifest)
    reference = [g["runs"][0]["evaluator"] for g in manifest["games"].values()]
    if any(r != reference[0] for r in reference[1:]) and a.model is None:
        p.error("reference evaluator settings differ; specify the intended model/routing explicitly")
    base = reference[0]
    if base["client"] != "openrouter":
        p.error("the initial probe adapter supports the reference OpenRouter client")
    split = lambda value: sorted(set(filter(None, value.split(","))))
    if a.grid_format is not None and a.grid_format != "frozen" and a.grid_format not in FORMATS:
        p.error(f"unknown --grid-format {a.grid_format!r}; known: {sorted(FORMATS)} or 'frozen'")
    config = {"checkpoints": split(a.checkpoints), "arms": split(a.arms),
        # An absent key means the frozen JSON-only wording, so saved runs keep reproducing
        # their published numbers; prepare() fills in DEFAULT_FORMAT when the flag is absent.
        **({} if a.grid_format is None else {"grid_format":
                                            None if a.grid_format == "frozen" else a.grid_format}),
        "forward_modes": split(a.forward_modes), "reconstruction_modes": split(a.reconstruction_modes),
        "examples": a.examples, "max_prompt_chars": a.max_prompt_chars,
        "model": {"id": a.model or base["model"], "provider": base["provider"] if a.provider is None else a.provider,
                  "reasoning": json.loads(a.reasoning_json) if a.reasoning_json is not None else base["reasoning"],
                  "temperature": a.temperature, "max_output_tokens": a.max_output_tokens, "timeout_s": a.timeout}}
    summary = prepare(manifest, a.out, config)
    print(json.dumps(summary, indent=2))
    if a.execute:
        if any(g["audit"]["replay_status"] != "verified" for g in manifest["games"].values()):
            p.error("model evaluation requires simulator-verified drives; rebuild with --replay")

        async def run():
            client = OpenRouterClient(config["model"])
            try:
                return await execute(a.out, client, concurrency=a.concurrency, attempts=a.attempts,
                                     max_calls=a.max_calls, retry_errors=a.retry_errors)
            finally:
                await client.close()
        print(json.dumps(asyncio.run(run()), indent=2))
    print("Coverage:", json.dumps(score_jobs(a.out)))
    print(f"Prompts: {a.out / 'jobs.jsonl'}")


if __name__ == "__main__":
    main()
