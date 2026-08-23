# Eval-call (task-model F) cost/speed sweep — gpt-oss-120b challengers, the hidden-thinking trap, and reasoning-effort overrides

**Date:** 2026-08-06 → 2026-08-08
**Question:** the mixed config decodes with `openai/gpt-oss-120b` pinned `cerebras,groq,sambanova` and reflects with
`deepseek/deepseek-v4-flash`. Is there a cheaper and/or faster OpenRouter model+provider for the *eval-call* role
(candidate scoring: set-ID decode + cFD on the stratified split, test50 ID eval)?
**Benchmark harness:** replay the aug4_mixed bt3gb saved cmd eval-only — warm-start the SHIPPED beliefs+perception,
budget 1 (`--max-metric-calls 1`, later `--max-nodes 1` after the rexpure CLI rename), so every arm makes the identical
210 LLM requests (seed eval on train30 with ID + cFD-hard, test50 set-ID, raw/start-P baselines) at concurrency 48,
30s hedge. Cost = real OpenRouter-billed `usage.cost`. Quality gate = test50 set-ID.
**Runs:** `logs/aug6_evalmodel_bench/`, `logs/aug7_qwen_nothink_bench/`, `logs/aug7_altmodels_bench/`; probes in
`logs/aug7_qwen_speed_probe/`, `logs/aug7_evalmodel_probe/`.

## TL;DR

1. **Nothing on OpenRouter beats gpt-oss-120b@cerebras on wall time.** Full-catalog scan (~400 models → 52
   paper-competitive → 17 live-probed → 6 benched): Cerebras is uniquely fast and hosts only 3 models
   (gpt-oss-120b $0.35/$0.75, gemma-4-31b $0.99/$1.49, glm-4.7 $2.25/$2.75) — no cheaper Cerebras option exists.
2. **`reasoning: {"effort": "low"}` is a free upgrade to the current prod config**: reasoning tokens 511→302/call,
   bench 24s / $0.085 / 0.78–0.64 with zero hedges — faster AND cheaper AND quality-tied vs default effort.
   gpt-oss rejects effort `none` ("Reasoning is mandatory for this endpoint", 400); `low` is the floor.
3. **qwen3.7-flash's 14× wall was hidden thinking**: ~2.2k thinking tokens per eval call, billed as output,
   invisible in litellm's usage (visible text was 0.27 chars per billed token). `reasoning {"effort": "none"}`
   kills it: 865s→322s, cost halved again, quality kept (0.83/0.72). Alibaba ignores `effort: low` and the native
   `enable_thinking` passthrough; only none/enabled-false work.
4. **litellm drops the `reasoning` param on every normal path** (responses→completion bridge whitelists request
   fields; `OpenrouterConfig.map_openai_params` clobbers user `extra_body`; `acompletion` rejects
   `reasoning_effort` for openrouter). A top-level `reasoning` kwarg on `acompletion` DOES reach the body — so we
   patch the bridge transform to forward it. **Wired into prod 2026-08-07**: `_patch_litellm_reasoning_passthrough()`
   at import in `explore/mixed_improve.py` + per-config `reasoning_json` (make_config → `_llm_call`) + rexpure flags
   `--task-reasoning-json` / `--reflection-reasoning-json`. Smoke-tested: with override 3.32 chars/tok (all visible),
   without 0.29 (thinking on), no-override behavior unchanged.
5. **Client-side parallelism cannot close a wall gap**: qwen at concurrency 48/96/128 = 322/350/346s and hedges
   *rose* (47→66/61). Eval phases fan out over ≤30–50 items (already batch-capped) and phase wall = tail latency;
   same-provider hedges mostly lose (2–7 wins/run). Only faster serving or phase pipelining would help.
6. **New cost/quality champion for batch work: ling-3.0-flash + thinking off** ($0.021/$0.063 @ Novita):
   $0.0086/run (−90% vs anchor), **0.83/0.74 = best quality of any arm**, 211s wall. Displaces qwen3.7-flash on
   every axis (qwen: $0.0171 / 0.83/0.72 / 322s).

## Final standings (aug7_altmodels_bench, one CLI, identical workload)

| arm | wall | F cost | test ID / strict | note |
|---|---|---|---|---|
| **gpt-oss-120b@cerebras +effort low** (anchor) | **24s** | $0.0852 | 0.78 / 0.64 | wall champion, 0 hedges/retries |
| gpt-oss-20b@groq +effort low | 48s | $0.0198 | 0.76 / 0.64 | best cost-for-wall trade; effort low held 20b's quality too |
| gpt-5.6-luna +effort none | 50s | $0.0242 | 0.77 / 0.66 | unpinned, OpenAI infra absorbed 48-conc cleanly, 400k ctx, ~83-tok answers |
| llama-4-scout@groq | 58s | $0.0263 | 0.73 / 0.66 | hit-rate 0.84 + 59 retries (Groq throttling) — out |
| llama-3.1-8b@groq | 54s | $0.0101 | 0.38 / 0.34 | quality collapse; the 8B floor is below this task |
| ling-3.0-flash@novita +nothink | 211s | **$0.0086** | **0.83 / 0.74** | cheapest AND best quality; 8.8× wall |

Earlier reference points (aug7_qwen_nothink_bench, pre-rename CLI): qwen3.7-flash default 865s/$0.0327/0.83–0.74;
+effort none 322s/$0.0171/0.83–0.72; gpt-oss-120b default effort 101s/$0.0905/0.79–0.64 (15 retries — unlucky arm;
the cleaner aug6 gepa-CLI baseline was 51s).

Probe-stage eliminations on single-call latency (real windowed set-ID prompt, `probe_evalmodel_candidates.py`):
gemini-2.5-flash-lite 10.3s; gpt-5-nano 14.4s (**ignores effort none** — 1.9k reasoning tokens billed anyway);
deepseek-v4-flash+nothink 6.8s (so no single-model mixed config); seed-1.6-flash 7.9s; mistral-small-3.2 14.2s;
gemma-3-27b 14.1s; gemma-4-26b-a4b 4.8s but $0.00144/call; qwen3-30b-a3b 30.2s; ling-2.6-flash 18.8s;
step-3.5-flash / nex-n2-mini multi-minute. In the Cerebras class (≤2s): gpt-oss-20b+low 1.2s, llama-4-scout 1.6s,
llama-3.1-8b 1.7s, gpt-5.6-luna 1.8s (anchor 0.9s).

## Recommendations

- **Wall-priority (default): keep the anchor, add effort low** — it beats the pre-aug7 prod config on all axes.
  Flag: `--task-reasoning-json '{"effort": "low"}'`.
- **Cost-priority with wall still mattering:** gpt-oss-20b@groq+low (−77%, tied quality, 2× wall; same family =
  lowest behavioral risk) or gpt-5.6-luna+none (−72%, +strict, no pin to manage).
- **Batch / multi-seed sweeps (wall irrelevant):** ling-3.0-flash+nothink (−90%, +5pts acc over anchor).
- **Traps to remember:** never judge a model by sticker price — hidden thinking tokens made nemotron-3-nano
  ($0.05/$0.20) *more expensive* than gpt-oss-120b and 24× slower (aug6); gpt-5-nano has the same trap and its
  effort-none is ignored. Groq throttles some models under sustained concurrency (scout's 59 retries, 20b's 3×
  slowdown at default effort on aug6).

## Caveats

Single game (bt3gb), single seed, n=50; identical-config reruns showed a ±0.04 accuracy noise band, so the
ling/qwen quality edge over the anchor (0.83 vs 0.78) needs 1–2 more games before it drives decisions. All wall
numbers are eval-only replays; full-run wall adds reflection (deepseek) and multiplies the eval-call share by the
node budget. Saved pre-refactor launch.json cmds need `sanitize_rexpure_cmd` + the
`/prototypes/perc_invdyn/` → `/offline_learning/` path remap + `--max-metric-calls` → `--max-nodes`.
