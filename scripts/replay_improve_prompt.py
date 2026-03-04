#!/usr/bin/env python3
"""Extract an improve prompt from an improve.log and replay it against an LLM multiple times."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import litellm

from llm_utils import build_llm_input, extract_llm_response_text


TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - INFO - (.*)$")


def parse_sections(log_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_marker: str | None = None
    current_body: list[str] = []

    for line in log_text.splitlines():
        match = TIMESTAMP_RE.match(line)
        if match:
            if current_marker is not None:
                sections.append((current_marker, "\n".join(current_body).strip()))
            current_marker = match.group(1).strip()
            current_body = []
        else:
            current_body.append(line)

    if current_marker is not None:
        sections.append((current_marker, "\n".join(current_body).strip()))

    return sections


def extract_prompt(
    log_path: Path,
    marker_substring: str = "Baseline improve step prompt",
    attempt: int | None = 1,
) -> str:
    text = log_path.read_text()
    sections = parse_sections(text)

    candidates: list[tuple[str, str]] = []
    marker_substring_l = marker_substring.lower()

    for marker, body in sections:
        if marker_substring_l in marker.lower():
            if attempt is None:
                candidates.append((marker, body))
                continue

            m = re.search(r"attempt\s+(\d+)\s*/\s*(\d+)", marker, re.IGNORECASE)
            if m and int(m.group(1)) == attempt:
                return body

            # If marker has no attempt annotation, keep as fallback.
            if not m:
                candidates.append((marker, body))

    if attempt is None and candidates:
        return candidates[-1][1]
    if candidates:
        return candidates[-1][1]

    raise ValueError(
        f"No section found with marker containing '{marker_substring}' in {log_path}"
    )


def call_model_multiple_times(
    prompt: str,
    model: str,
    num_calls: int,
    delay_seconds: float,
    temperature: float | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
) -> list[str]:
    outputs: list[str] = []

    for i in range(1, num_calls + 1):
        print(f"\n=== Call {i}/{num_calls} | model={model} ===")
        kwargs = {
            "model": model,
            "input": build_llm_input(prompt),
            "num_retries": 5,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if base_url:
            kwargs["base_url"] = base_url

        response = litellm.responses(
            **kwargs,
        )
        text = extract_llm_response_text(response)
        outputs.append(text)
        print(text)

        if i < num_calls and delay_seconds > 0:
            time.sleep(delay_seconds)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Path to improve.log")
    parser.add_argument(
        "--marker",
        default="Baseline improve step prompt",
        help="Marker substring to find prompt section",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="Attempt number to extract from marker (set -1 to ignore)",
    )
    parser.add_argument(
        "--client-name",
        default="google",
        help="Client/provider prefix (e.g., openrouter, google, gemini, openai, vllm)",
    )
    parser.add_argument(
        "--model-id",
        default="gemini-2.5-flash",
        help="Model id without provider prefix",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional base URL for LiteLLM provider routing",
    )
    parser.add_argument("--n", type=int, default=3, help="Number of model calls")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between calls in seconds")
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional max output tokens")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory to save extracted prompt and outputs",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    attempt = None if args.attempt < 0 else args.attempt
    prompt = extract_prompt(log_path, marker_substring=args.marker, attempt=attempt)
    model = f"{args.client_name}/{args.model_id}"

    print("=== Extracted Prompt Preview (first 500 chars) ===")
    print(prompt[:500])
    print(f"\nPrompt length: {len(prompt)} chars")

    outputs = call_model_multiple_times(
        prompt=prompt,
        model=model,
        num_calls=args.n,
        delay_seconds=args.delay,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url or None,
    )

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "extracted_prompt.txt").write_text(prompt)
        for i, out in enumerate(outputs, start=1):
            (out_dir / f"response_{i}.txt").write_text(out)

        meta = {
            "log": str(log_path),
            "marker": args.marker,
            "attempt": attempt,
            "model": model,
            "base_url": args.base_url or None,
            "num_calls": args.n,
            "delay_seconds": args.delay,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt_length_chars": len(prompt),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\nSaved prompt and responses to: {out_dir}")


if __name__ == "__main__":
    main()
