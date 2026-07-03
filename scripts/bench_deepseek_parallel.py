"""Parallel load test: many small requests for deepseek-v4-flash.

For each endpoint, fire N small requests through a thread pool at a fixed
concurrency level. Measures aggregate throughput (req/s, completed-tokens/s),
per-request latency distribution, and error/rate-limit rate.
Endpoints are tested back-to-back (not simultaneously) to avoid sharing
the local network/CPU between them.
"""
import os, time, argparse, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ENDPOINTS = {
    "openrouter": dict(base_url="https://openrouter.ai/api/v1",
                       api_key=lambda: os.environ["OPENROUTER_API_KEY"],
                       model="deepseek/deepseek-v4-flash"),
    "deepseek": dict(base_url="https://api.deepseek.com",
                     api_key=lambda: os.environ["DEEPSEEK_API_KEY"],
                     model="deepseek-v4-flash"),
}

PROMPTS = [
    "Classify the sentiment (positive/negative/neutral): 'The food was cold.'",
    "What is 17 * 23? Answer with just the number.",
    "Name the capital of Australia in one word.",
    "Is a tomato a fruit or vegetable? One word.",
    "Translate 'good morning' to French.",
    "What comes next: 2, 4, 8, 16, ? Just the number.",
]


def one(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0,
        )
        dt = time.perf_counter() - t0
        tok = r.usage.completion_tokens if r.usage else 0
        return dict(ok=True, lat=dt, tok=tok)
    except Exception as e:
        dt = time.perf_counter() - t0
        name = type(e).__name__
        is_rate = "429" in str(e) or "rate" in str(e).lower()
        return dict(ok=False, lat=dt, tok=0, err=name, rate=is_rate)


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def run_endpoint(name, cfg, n, conc, max_tokens):
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"]())
    rows = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(one, client, cfg["model"], PROMPTS[i % len(PROMPTS)], max_tokens)
                for i in range(n)]
        for f in as_completed(futs):
            rows.append(f.result())
    wall = time.perf_counter() - t0

    ok = [r for r in rows if r["ok"]]
    errs = [r for r in rows if not r["ok"]]
    rate_errs = [r for r in errs if r.get("rate")]
    lats = [r["lat"] for r in ok]
    toks = sum(r["tok"] for r in ok)

    print(f"\n=== {name}  (n={n}, concurrency={conc}) ===")
    print(f"  wall time            : {wall:.2f} s")
    print(f"  succeeded / failed   : {len(ok)} / {len(errs)}  "
          f"(rate-limited: {len(rate_errs)})")
    print(f"  throughput           : {len(ok)/wall:6.2f} req/s   {toks/wall:7.1f} tok/s")
    if lats:
        print(f"  per-req latency (s)  : mean={statistics.mean(lats):.2f}  "
              f"p50={pct(lats,50):.2f}  p95={pct(lats,95):.2f}  max={max(lats):.2f}")
    if errs:
        from collections import Counter
        print(f"  error types          : {dict(Counter(r['err'] for r in errs))}")
    return dict(name=name, wall=wall, ok=len(ok), err=len(errs),
                rate=len(rate_errs), reqps=len(ok)/wall, tokps=toks/wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100, help="total requests per endpoint")
    ap.add_argument("-c", "--concurrency", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=30)
    args = ap.parse_args()

    summ = []
    for name, cfg in ENDPOINTS.items():
        summ.append(run_endpoint(name, cfg, args.n, args.concurrency, args.max_tokens))

    print("\n--- summary ---")
    for s in summ:
        print(f"  {s['name']:10s}: {s['reqps']:6.2f} req/s  {s['tokps']:7.1f} tok/s  "
              f"wall={s['wall']:.1f}s  ok={s['ok']} err={s['err']} rate-limited={s['rate']}")


if __name__ == "__main__":
    main()
