"""Benchmark deepseek-v4-flash across OpenRouter vs DeepSeek native API.

Measures, per request (streaming):
  - TTFT  : time to first token (latency)
  - total : overall request completion time
  - tok/s : completion_tokens / generation_time (first-token -> done)
Endpoints are interleaved to reduce time-of-day bias. Requests are sequential.
"""
import os, time, statistics, argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ENDPOINTS = {
    "openrouter": dict(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model="deepseek/deepseek-v4-flash",
    ),
    "deepseek": dict(
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        model="deepseek-v4-flash",
    ),
}

PROMPTS = [
    "Explain how a hash map works, including collision handling. Be thorough.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Describe the process of photosynthesis step by step.",
    "What are the trade-offs between TCP and UDP? Give concrete examples.",
    "Summarize the causes of the French Revolution.",
]


def one_request(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
        stream_options={"include_usage": True},
    )
    usage = None
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    comp_tokens = usage.completion_tokens if usage else None
    gen_time = total - ttft if ttft else total
    tok_s = comp_tokens / gen_time if comp_tokens and gen_time > 0 else None
    return dict(ttft=ttft, total=total, comp_tokens=comp_tokens, tok_s=tok_s)


def summarize(name, rows):
    ok = [r for r in rows if r.get("ttft") is not None]
    if not ok:
        print(f"\n{name}: NO SUCCESSFUL REQUESTS")
        return
    def stat(key):
        vals = [r[key] for r in ok if r[key] is not None]
        return statistics.mean(vals), statistics.median(vals), min(vals), max(vals)
    print(f"\n=== {name}  (n={len(ok)}/{len(rows)} ok) ===")
    for label, key, unit in [
        ("TTFT (latency)", "ttft", "s"),
        ("Total time    ", "total", "s"),
        ("Throughput    ", "tok_s", "tok/s"),
        ("Compl. tokens ", "comp_tokens", ""),
    ]:
        m, md, lo, hi = stat(key)
        print(f"  {label}: mean={m:7.2f}{unit:>6}  median={md:7.2f}  min={lo:7.2f}  max={hi:7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8, help="requests per endpoint")
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    clients = {n: OpenAI(base_url=c["base_url"], api_key=c["api_key"]) for n, c in ENDPOINTS.items()}
    results = {n: [] for n in ENDPOINTS}

    for i in range(args.runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        for name, cfg in ENDPOINTS.items():
            try:
                r = one_request(clients[name], cfg["model"], prompt, args.max_tokens)
            except Exception as e:
                r = dict(ttft=None, total=None, comp_tokens=None, tok_s=None, err=str(e))
                print(f"[{name}] run {i} ERROR: {e}")
            results[name].append(r)
            tag = "ok" if r.get("ttft") else "ERR"
            if r.get("ttft"):
                print(f"[{name:10s}] run {i}: ttft={r['ttft']:.2f}s total={r['total']:.2f}s "
                      f"tok={r['comp_tokens']} {r['tok_s']:.1f} tok/s")

    for name in ENDPOINTS:
        summarize(name, results[name])


if __name__ == "__main__":
    main()
