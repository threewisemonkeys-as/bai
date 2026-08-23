"""Compare DeepSeek low/high reasoning on real reflection and analysis prompts.

Runs the same paired prompt set through OpenRouter (one pinned provider) and the
first-party DeepSeek API. Streaming is enabled to measure TTFT and generation
throughput. Both provider metadata and token/cache usage are persisted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import statistics
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sample(rows: list[dict], kind: str, n: int) -> list[dict]:
    out = []
    for component in ("perception", "world_knowledge"):
        group = sorted(
            (r for r in rows if r.get("component") == component),
            key=lambda r: len(r.get("prompt", "")),
        )
        indices = [round(i * (len(group) - 1) / (n - 1)) for i in range(n)]
        for q, index in enumerate(indices):
            row = dict(group[index])
            source = row.get("call") if kind == "reflection" else row.get("iteration")
            row.update(case_id=f"{kind}_{component}_{source}_q{q}", kind=kind)
            out.append(row)
    return out


def obj(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    try:
        return dict(value)
    except Exception:
        return {"repr": repr(value)}


def usage(usage_obj) -> dict:
    raw = obj(usage_obj)
    details = raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}
    input_details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    return {
        "input_tokens": raw.get("prompt_tokens", raw.get("input_tokens")),
        "output_tokens": raw.get("completion_tokens", raw.get("output_tokens")),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "cache_hit_tokens": raw.get(
            "prompt_cache_hit_tokens", input_details.get("cached_tokens")
        ),
        "cache_miss_tokens": raw.get("prompt_cache_miss_tokens"),
        "cost": raw.get("cost"),
        "raw": raw,
    }


def fenced(text: str) -> str | None:
    match = re.search(r"```(?:python|world)?\s*\n?(.*?)```", text, re.S | re.I)
    return match.group(1).strip() if match else None


def quality(case: dict, text: str) -> dict:
    result = {"nonempty": bool(text.strip())}
    if case["kind"] == "reflection":
        block = fenced(text)
        result["has_fence"] = block is not None
        if case["component"] == "perception":
            result["has_perceive"] = bool(block and re.search(r"\bdef\s+perceive\s*\(", block))
            try:
                compile(block or "", "<candidate>", "exec")
                result["compiles"] = bool(block)
            except Exception as exc:
                result["compiles"] = False
                result["compile_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["substantive"] = bool(block and len(block.split()) >= 25)
    else:
        tail = case["prompt"][-3500:]
        expected = list(dict.fromkeys(re.findall(
            r"<(common_root_causes|m\d+_(?:inv|fd))>", tail
        )))
        matches = {
            tag: re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S) for tag in expected
        }
        result["expected_tags"] = expected
        result["all_tags"] = bool(expected) and all(matches.values())
        counts = {
            tag: len(match.group(1).split())
            for tag, match in matches.items()
            if match and tag != "common_root_causes"
        }
        result["tag_word_counts"] = counts
        result["under_word_limits"] = bool(counts) and all(v <= 80 for v in counts.values())
    return result


async def streamed_call(
    backend: str, prompt: str, effort: str | None, args: argparse.Namespace,
    *, judge: bool = False,
) -> dict:
    if judge:
        model = args.judge_model
        backend = "openrouter"
        provider = args.judge_provider
        max_tokens = args.judge_max_tokens
    else:
        model = (
            "openrouter/deepseek/deepseek-v4-flash"
            if backend == "openrouter" else "deepseek/deepseek-v4-flash"
        )
        provider = args.openrouter_provider
        max_tokens = args.max_tokens
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "timeout": args.timeout,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if backend == "openrouter":
        kwargs.update(
            provider={
                "order": [provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            extra_headers={"X-OpenRouter-Metadata": "enabled"},
            extra_body={"usage": {"include": True}},
        )
        if effort is not None:
            kwargs["extra_body"]["reasoning"] = {"effort": effort}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if effort is not None:
            kwargs["reasoning_effort"] = effort

    start = time.perf_counter()
    stream = await litellm.acompletion(**kwargs)
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
        data = obj(chunk)
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
                "hidden_params": obj(getattr(chunk, "_hidden_params", None)),
            }
        if data.get("usage"):
            final_usage = usage(data["usage"])
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
    generation_start = first_reasoning if first_reasoning is not None else first_content
    generation_s = elapsed - generation_start if generation_start is not None else None
    output_tokens = final_usage.get("output_tokens")
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
                output_tokens / generation_s
                if isinstance(output_tokens, (int, float)) and generation_s and generation_s > 0
                else None
            ),
        },
    }


async def generate(cases: list[dict], args: argparse.Namespace, out: Path) -> list[dict]:
    jobs = [(backend, effort, case) for backend in ("openrouter", "direct")
            for effort in ("low", "high") for case in cases]
    random.Random(args.seed).shuffle(jobs)
    semaphores = {backend: asyncio.Semaphore(args.concurrency) for backend in ("openrouter", "direct")}
    records = []

    async def one(backend: str, effort: str, case: dict):
        async with semaphores[backend]:
            try:
                result = await streamed_call(backend, case["prompt"], effort, args)
                error = None
            except Exception as exc:
                result = {"text": "", "elapsed_s": 0, "usage": {}, "router_metadata": {},
                          "response_metadata": {}, "performance": {}}
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "case_id": case["case_id"], "kind": case["kind"],
                "component": case["component"], "backend": backend, "effort": effort,
                "provider_requested": args.openrouter_provider if backend == "openrouter" else "deepseek-direct",
                "prompt_chars": len(case["prompt"]), "response_chars": len(result["text"]),
                "automatic_quality": quality(case, result["text"]), "error": error, **result,
            }
            records.append(record)
            path = out / "responses" / f"{case['case_id']}__{backend}__{effort}.md"
            path.write_text(result["text"] or f"ERROR: {error}\n")
            print(
                f"done {case['case_id']} {backend}/{effort}: {result['elapsed_s']:.1f}s, "
                f"{len(result['text'])} chars, r={result['usage'].get('reasoning_tokens')}, "
                f"cache={result['usage'].get('cache_hit_tokens')}, error={error or '-'}",
                flush=True,
            )

    await asyncio.gather(*(one(*job) for job in jobs))
    return sorted(records, key=lambda r: (r["case_id"], r["backend"], r["effort"]))


def judge_prompt(case: dict, candidates: list[dict]) -> str:
    bodies = []
    for label, candidate in zip("ABCD", candidates):
        bodies.append(f"CANDIDATE {label}:\n<<<{label}\n{candidate['text']}\n{label}")
    return f"""Blindly evaluate four responses to the same optimizer prompt. Rank substantive
usefulness, grounding in supplied evidence, exact contract compliance, generalizability,
preservation of correct behavior, and concision. Do not reward verbosity.

ORIGINAL PROMPT:\n<<<PROMPT\n{case['prompt']}\nPROMPT

{chr(10).join(bodies)}

Return JSON only:
{{"ranking":["A","B","C","D"],"scores":{{"A":1,"B":1,"C":1,"D":1}},
"reason":"under 150 words"}}
Scores are integers 1-5; ties may appear adjacent but list every label once.
"""


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"parse_error": "no JSON", "raw": text}
    try:
        return json.loads(match.group(0))
    except Exception as exc:
        return {"parse_error": f"{type(exc).__name__}: {exc}", "raw": text}


async def judge(cases: list[dict], records: list[dict], args: argparse.Namespace) -> list[dict]:
    lookup = {(r["case_id"], r["backend"], r["effort"]): r for r in records}
    sem = asyncio.Semaphore(args.judge_concurrency)
    results = []

    async def one(case: dict):
        candidates = [lookup[(case["case_id"], backend, effort)]
                      for backend, effort in (("openrouter", "low"), ("openrouter", "high"),
                                              ("direct", "low"), ("direct", "high"))]
        if any(c["error"] for c in candidates):
            results.append({"case_id": case["case_id"], "skipped": "generation error"})
            return
        rng = random.Random(args.seed + int(hashlib.sha256(case["case_id"].encode()).hexdigest(), 16))
        rng.shuffle(candidates)
        labels = {label: {"backend": c["backend"], "effort": c["effort"]}
                  for label, c in zip("ABCD", candidates)}
        async with sem:
            try:
                response = await streamed_call(
                    "openrouter", judge_prompt(case, candidates), None, args, judge=True
                )
                parsed = parse_json(response["text"])
                results.append({"case_id": case["case_id"], "labels": labels,
                                "judge": parsed, "metadata": response})
                top = parsed.get("ranking", ["?"])[0]
                print(f"judged {case['case_id']}: {labels.get(top, top)}", flush=True)
            except Exception as exc:
                results.append({"case_id": case["case_id"],
                                "error": f"{type(exc).__name__}: {exc}"})

    await asyncio.gather(*(one(case) for case in cases))
    return sorted(results, key=lambda r: r["case_id"])


def med(xs):
    values = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(values) if values else None


def summarize(records: list[dict], judgments: list[dict]) -> dict:
    summary = {"arms": {}, "judge_firsts": {}}
    for backend in ("openrouter", "direct"):
        for effort in ("low", "high"):
            arm = [r for r in records if r["backend"] == backend and r["effort"] == effort
                   and not r["error"]]
            key = f"{backend}_{effort}"
            summary["arms"][key] = {
                "n": len(arm),
                "errors": sum(r["backend"] == backend and r["effort"] == effort and bool(r["error"])
                              for r in records),
                "median_latency_s": med([r["elapsed_s"] for r in arm]),
                "median_ttft_s": med([
                    r["performance"].get("ttft_reasoning_s")
                    if r["performance"].get("ttft_reasoning_s") is not None
                    else r["performance"].get("ttft_content_s") for r in arm
                ]),
                "median_output_tokens": med([r["usage"].get("output_tokens") for r in arm]),
                "median_reasoning_tokens": med([r["usage"].get("reasoning_tokens") for r in arm]),
                "median_cache_hit_tokens": med([r["usage"].get("cache_hit_tokens") for r in arm]),
                "median_observed_output_tps": med([
                    r["performance"].get("observed_output_tokens_per_s") for r in arm
                ]),
                "valid_outputs": sum(
                    r["automatic_quality"].get("has_fence", True)
                    and r["automatic_quality"].get("compiles", True)
                    and r["automatic_quality"].get("all_tags", True)
                    and r["automatic_quality"].get("under_word_limits", True)
                    for r in arm
                ),
            }
    for result in judgments:
        ranking = result.get("judge", {}).get("ranking") or []
        if ranking:
            winner = result["labels"].get(ranking[0])
            if winner:
                key = f"{winner['backend']}_{winner['effort']}"
                summary["judge_firsts"][key] = summary["judge_firsts"].get(key, 0) + 1
    return summary


async def main(args: argparse.Namespace):
    source = Path(args.source)
    cases = sample(jsonl(source / "reflection_calls.jsonl"), "reflection", args.per_component)
    cases += sample(jsonl(source / "analysis_calls.jsonl"), "analysis", args.per_component)
    out = Path(args.out_dir)
    (out / "responses").mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    for case in cases:
        (out / f"prompt_{case['case_id']}.txt").write_text(case["prompt"])
    records = await generate(cases, args, out)
    with (out / "records.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    judgments = await judge(cases, records, args)
    with (out / "judgments.jsonl").open("w") as f:
        for result in judgments:
            f.write(json.dumps(result) + "\n")
    summary = summarize(records, judgments)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(
        ROOT / "logs/aug3_dsflash/bt3gb_strat30_seed1/gepa_run_seed1"
    ))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--openrouter-provider", default="fireworks")
    parser.add_argument("--per-component", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--judge-model", default="openrouter/openai/gpt-oss-120b")
    parser.add_argument("--judge-provider", default="cerebras")
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    asyncio.run(main(arguments()))
