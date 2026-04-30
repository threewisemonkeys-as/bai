"""Smoke test for LiteLLM via OpenRouter using Gemini 2.5 Flash.

Usage:
    uv run scripts/test_openrouter_gemini.py
    uv run scripts/test_openrouter_gemini.py --prompt "Say hello in one short sentence."
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
import litellm


MODEL = "openrouter/google/gemini-2.5-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def main(prompt: str) -> int:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY in the environment or .env", file=sys.stderr)
        return 1

    try:
        response = litellm.responses(
            model=MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=64,
            num_retries=1,
        )
        text = response.output[-1].content[0].text
    except Exception as e:
        print(f"OpenRouter smoke test failed: {e}", file=sys.stderr)
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: connection ok",
        help="Short prompt used to test the connection.",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.prompt))
