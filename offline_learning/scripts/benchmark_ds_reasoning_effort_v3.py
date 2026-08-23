"""V3 transport for the DeepSeek reasoning-effort benchmark.

Uses the OpenAI-compatible clients directly so current DeepSeek V4 parameters
and OpenRouter's router-metadata extension are not filtered by the repository's
older LiteLLM provider schema. Sampling, validation, judging, and summaries are
reused from benchmark_ds_reasoning_effort_v2.
"""

from __future__ import annotations

import asyncio
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

import benchmark_ds_reasoning_effort_v2 as base


def dump(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {"repr": repr(value)}


async def streamed_call(
    backend: str,
    prompt: str,
    effort: str | None,
    args,
    *,
    judge: bool = False,
) -> dict:
    if judge:
        backend = "openrouter"
        model = "openai/gpt-oss-120b"
        provider = args.judge_provider
        max_tokens = args.judge_max_tokens
    elif backend == "openrouter":
        model = "deepseek/deepseek-v4-flash"
        provider = args.openrouter_provider
        max_tokens = args.max_tokens
    else:
        model = "deepseek-v4-flash"
        provider = None
        max_tokens = args.max_tokens

    if backend == "openrouter":
        client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=args.timeout,
        )
        extra_body = {
            "provider": {
                "order": [provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            "usage": {"include": True},
        }
        if effort is not None:
            extra_body["reasoning"] = {"effort": effort}
        extra_headers = {"X-OpenRouter-Metadata": "enabled"}
    else:
        client = AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=args.timeout,
        )
        extra_body = {"thinking": {"type": "enabled"}}
        if effort is not None:
            extra_body["reasoning_effort"] = effort
        extra_headers = None

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_body": extra_body,
    }
    if extra_headers:
        kwargs["extra_headers"] = extra_headers

    start = time.perf_counter()
    stream = await client.chat.completions.create(**kwargs)
    content = []
    reasoning_chars = 0
    first_any = first_reasoning = first_content = None
    final_usage = {}
    router_metadata = {}
    response_metadata = {}
    chunks = 0
    async for chunk in stream:
        now = time.perf_counter()
        chunks += 1
        data = dump(chunk)
        if first_any is None:
            first_any = now - start
        maybe_router = data.get("openrouter_metadata") or data.get("metadata")
        if maybe_router:
            router_metadata = maybe_router
        if not response_metadata:
            response_metadata = {
                "id": data.get("id"),
                "model": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"),
            }
        if data.get("usage"):
            final_usage = base.usage(data["usage"])
        choices = data.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        answer = delta.get("content") or ""
        if reasoning:
            reasoning_chars += len(reasoning)
            if first_reasoning is None:
                first_reasoning = now - start
        if answer:
            content.append(answer)
            if first_content is None:
                first_content = now - start
    elapsed = time.perf_counter() - start
    await client.close()
    generation_start = first_reasoning if first_reasoning is not None else first_content
    generation_s = elapsed - generation_start if generation_start is not None else None
    output_tokens = final_usage.get("output_tokens") or 0
    reasoning_tokens = final_usage.get("reasoning_tokens") or 0
    generated_tokens = output_tokens + reasoning_tokens
    return {
        "text": "".join(content),
        "elapsed_s": elapsed,
        "usage": final_usage,
        "router_metadata": router_metadata,
        "response_metadata": response_metadata,
        "performance": {
            "chunks": chunks,
            "ttft_any_s": first_any,
            "ttft_reasoning_s": first_reasoning,
            "ttft_content_s": first_content,
            "reasoning_chars": reasoning_chars,
            "generation_s": generation_s,
            "observed_output_tokens_per_s": (
                generated_tokens / generation_s
                if generated_tokens and generation_s and generation_s > 0 else None
            ),
        },
    }


if __name__ == "__main__":
    load_dotenv(base.ROOT / ".env")
    base.streamed_call = streamed_call
    asyncio.run(base.main(base.arguments()))
