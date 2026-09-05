# PRO-LONG as the agentic planning baseline

Plan for running [PRO-LONG](https://github.com/alexisfox7/PRO-LONG) as the **Agentic LLM**
baseline of `experimental_plan.md` on the 86-problem curated Autumn planning set.

The question the arm answers: **given the same training data and the same model, how does a
coding agent that reads the data itself compare to REX-learned beliefs+perception?**

Comparison is against `raw`, `icl` and `lmwm`. `cc_autumn` is a different protocol (live
exploration, flat cap 50, 78 rows, Opus 5) and is not what this arm is measured against.

Everything under a "Findings" heading was measured — against a clone of upstream at `9d2f2d4`
(2026-08-19), against this repo, and against the CLIs installed on this host.

---

## 0. The arm

One variable changes across the four columns: **how the same offline pool becomes a plan.**

| | model | training signal | problems | budget | scorer |
|---|---|---|---|---|---|
| `raw` | deepseek-v4-flash | none | 86 | per-problem (2–60) | online evaluator |
| `icl` | deepseek-v4-flash | the 60 train transitions, pasted into the prompt | 86 | per-problem | online evaluator |
| `lmwm` (NLWM) | deepseek-v4-flash | the same 60, distilled by REX into `(beliefs.txt, perception.py)` | 86 | per-problem | online evaluator |
| **`agent` (this)** | **deepseek-v4-flash** | **the same 60, on disk, with a shell to compute over them** | **86** | **per-problem** | **online evaluator** |

Same model in every cell, same data, same problems, same action budget, same scorer. What
differs is only what happens to the pool between the disk and the plan: nothing (`icl`),
compressed by REX search (`lmwm`), or read by an agent that can run code over it (`agent`).

`icl` landed after this plan's first draft (`ec43f73`) and it is the arm's tightest control.
`icl` and `agent` see the *same bytes* — `icl_context.py` renders the learner's pool into a
prompt block, and the exporter of §5 Phase 2 writes that same pool to disk — so `agent − icl`
isolates *computing over the data* from *reading it*, with the data held fixed. Measured pool
size: 60 transitions, 23k–159k tokens per game (median ~71k; `bt3gb` 71k, `logic_gates` 159k).
It fits a 1M window, which is the point: the agent's advantage cannot be "the data would not
fit" — it has to be that it does something with the data.

**The one axis that is deliberately not matched is inference compute.** An agentic harness
spends many LLM calls per action; `raw`/`lmwm` spend one (plus one corrective re-ask). That
*is* the treatment, not a confound — but it must be priced, so every arm reports tokens and
cost per problem (§6).

---

## 1. Findings: PRO-LONG, and why our fork is a dead end

**F1. Our fork and upstream share a root commit and have fully diverged.** `RGB-Agent` and
`PRO-LONG` both begin at `cf113b9` (2026-03-07). We branched at `c0a685c` (2026-03-11) and
added 10 commits; upstream added ~60 and restructured — the harness moved to
`research/arc-agi-3/prolong_agent/`, and the repo root became a TypeScript npm package
(`prolong init`, durable memory for coding CLIs — the *same method*, delivered differently; see
F2b). `node` is not on this host but `bun` is (`~/.bun/bin/bun`), and the sandbox images install
their CLIs with npm inside the image, so the package is reachable if wanted. No shared path
survives between the fork and either half. A rebase would be a rewrite.

**F2. Upstream deleted the backend our fork depends on — which unblocks us.** Our fork runs
`OpenCodeAgent`/`TextToolAgent`. Upstream has neither; it ships `codex_agent.py` and
`claude_code_agent.py`. Memory `rgb-agent-rootless-setup` records the blocker that stopped the
RGB-Agent runs: opencode 1.16.2 returned empty completions for every OpenRouter model. That
blocker is gone with the backend.

**F2a. There is no opencode *backend* in the current PRO-LONG — but opencode is a
first-class client of the other half (re-checked 2026-09-05).** Upstream
HEAD is still `9d2f2d4` (2026-08-19) — nothing has landed since F1–F10 were measured, so
"latest" is the tree described here. But the distinction is between the repo's *two* PRO-LONGs,
not between opencode and the rest: in the npm package opencode is a **first-class client** —
`CLIENT_NAMES = ["codex", "claude-code", "opencode", "pi"]` in `src/types.ts`, a `--client`
value in `src/cli.ts`, paths in `src/uninstall.ts`, a case in `tests/lifecycle.test.ts`, a row
in the README's client table, and two implementation files (`src/clients/opencode.ts`, the
adapter, and `templates/clients/opencode.ts`, the plugin it writes). What it is *not* is an
agent backend: nothing in that half plays a game (F2b). In the research harness — the half with
the runner, the queue and the `actions.json` contract — the backend was removed deliberately:
`44f53b9 "Refactor agent framework: dual backend (codex + claude-code)"` deletes
`rgb_agent/agent/opencode_agent.py` and `docker/opencode-sandbox/` in one commit.

Porting it back is mechanically easy and substantively a bad trade. `BaseAgent` is a clean seam:
prompt assembly, log sync and `actions.json` parsing are all backend-agnostic, and a backend
only supplies `analyze() -> {hint, plan, actions, meta, cost}`, so an `OpenCodeAgent` drops in.
The price is what that class contains. Ours is 593 lines against codex's 431, and the extra ~160
are pure workaround: a persistent `opencode serve` container, the session id read out of
`opencode.db` by a glob+sqlite one-liner, and `opencode export` *polled* because `run --attach`
exits on premature idle. Upstream also deleted the only parser that speaks opencode's nd-JSON
event stream, so an `opencode_events.py` comes back with it, unmaintained. And the blocker was
never fixed, only routed around: opencode is not installed on this host (it lives inside the
3-month-old `rgb-agent/opencode-sandbox` image), and empty-response reports against OpenRouter
models are still open upstream, notwithstanding OpenRouter's own opencode cookbook. Against
that, codex 0.151 + deepseek-v4-flash is *measured* end-to-end (F6–F8) with a parser that
matches the installed CLI unchanged. ⇒ **codex is the backend.** If opencode is wanted anyway,
the gate is a rebuild of the sandbox image and a re-run of the empty-completion probe *before*
any porting — not after.

**F2b. The npm package is PRO-LONG's method, and two pieces of it are worth taking.** Calling it
"unrelated" was wrong. `prolong init` installs a client hook that appends every session event
and tool call to an append-only `.prolong/log.jsonl` (`templates/runtime.mjs`, 63 lines,
exporting `record(client, event)`), plus a `templates/prolong/SKILL.md` that tells the agent to
retrieve from it with `rg`/`jq`/short scripts and — in as many words — "do not load the entire
log into context unless it is demonstrably small and necessary". That is the same claim as the
research harness's `logs.txt` + "parse it programmatically", productised for a CLI a human
drives. For opencode the hook is a 41-line plugin on `session.*`, `message.updated` and
`tool.execute.before/after`.

What it does not contain is the part we actually need. It records the agent's *own* events; there
is no writer for environment observations, no `actions.json`, no action queue, no `BaseEnv`, no
runner. Nothing steps a world. The missing piece for an opencode arm was never memory — it is
the loop — and the package has none of it, so it does not change F2a's conclusion.

Two things to take anyway:

* **One log instead of two.** `record()` is an ordinary exported function appending JSONL. The
  Autumn runner could append its own `action` / `resulting grid` entries to the *same*
  `.prolong/log.jsonl` the agent's tool calls land in, which is a better design than the
  harness's split between `logs.txt` (environment) and the CLI's own conversation: one file, one
  retrieval discipline, and the agent's past reasoning is greppable alongside what the world did.
* **`SKILL.md` is upstream's own wording** for the retrieval discipline that
  `research/autumn/prompts.py` would otherwise invent, including the untrusted-history and
  don't-inline-the-log rules. Reuse it rather than paraphrasing it.

**F3. Our Autumn driver bypasses the harness entirely.** `RGB-Agent/scripts/run_autumn.py`
uses neither `GameRunner`, `BaseEnv`, `ActionQueue` nor the `actions.json` contract — it is a
private loop with an `[ACTIONS]`-in-response protocol, driving AutumnBench dynamics tasks
(mfp/cd/planning), never curated planning problems. Nothing to carry forward.

**F4. The ARC coupling upstream is small and localised.**

| module | lines | ARC dependency |
|---|--:|---|
| `agent/base.py`, `codex_agent.py`, `claude_code_agent.py`, `codex_events.py`, `claude_events.py`, `action_queue.py`, `utils/`, `metrics/` | **2010** | **none** |
| `agent/prompts.py`, `agent/game_state.py`, `agent/swarm.py`, `environment/runner.py`, `environment/arcagi3.py`, `environment/config.py` | 1212 | `arc_agi`, `arcengine` |

`game_state.py` touches `arcengine.GameAction` on one line, `runner.py` touches
`arcengine.GameState` on one. `BaseEnv` (`reset(task)->obs`, `step(a)->(obs,reward,done)`) is
the seam an Autumn env drops into.

**F5. The method ports cleanly.** PRO-LONG's claim: append every observation/action/outcome to
one `logs.txt`, keep a ~30-line system prompt, and let the agent retrieve *programmatically*
(grep/python) instead of dumping history into context. Agent writes `/workspace/actions.json`;
runner drains it. `--log-window` gives three published conditions. None of that is ARC-specific.

---

## 2. Findings: deepseek-v4-flash **does** drive a coding CLI (measured)

This was the load-bearing unknown; it is resolved. Codex CLI 0.151.0 is on this host.

**F6. Codex 0.151 dropped `wire_api = "chat"`** — provider configs must now use
`wire_api = "responses"`. Both candidate endpoints serve the Responses API for
deepseek-v4-flash (HTTP 200, verified by direct `curl`):
`https://openrouter.ai/api/v1/responses` and `https://api.deepseek.com/responses`.

**F7. Codex + deepseek-v4-flash runs a real tool loop.** End-to-end test (read a file with the
shell, write an output file, report): succeeded against **both** endpoints.

```bash
codex exec --json --skip-git-repo-check -C <workspace> \
  -c 'model_providers.openrouter={name="OpenRouter",base_url="https://openrouter.ai/api/v1",env_key="OPENROUTER_API_KEY",wire_api="responses"}' \
  -c 'model_provider="openrouter"' -c 'model="deepseek/deepseek-v4-flash"'
```

Because two different base_urls both work, **any** Responses-compatible endpoint works — which
is what makes the local proxy in §4 viable rather than speculative.

**F8. Upstream's event parser matches the installed CLI unchanged.** `codex exec --json` emits
`thread.started` / `turn.started` / `item.started` / `item.completed` / `turn.completed`
(with a `usage` block) — exactly the schema `codex_events.py` parses. `_build_codex_args`
already uses `-c` overrides, `--ignore-user-config`, and `exec resume` for session continuity,
all of which 0.151 still supports.

**F9. It is cheap.** deepseek-v4-flash on OpenRouter: **$0.087/M prompt, $0.174/M completion,
$0.017/M cached**. The measured trivial session billed 186,798 input (106,496 cached) + 264
output ≈ **$0.009 per agent call**. Codex's own system prompt and tool schema dominate that,
and it caches. See §7 for the run estimate.

**F10. The egress allowlist is a one-line change.** `docker/openai-proxy` is a squid config
with `acl allowed_host dstdomain .openai.com .chatgpt.com`; point it at `.openrouter.ai`
(or the local proxy) and the sandbox lockdown is preserved rather than opted out of.

**F11. It sustains the real workload, not just the trivial one (measured 2026-09-05).** F7's
end-to-end test was "read a file, write a file". Re-run against the arm's actual workload — the
`bt3gb` pool exported as §5 Phase 2 will export it (60 transitions with 9-frame context, 1.9 MB
over 60 JSON files + `index.csv`), an `AGENTS.md` in the arm's shape, and the task "work out what
each action does, cite transition ids in `findings.md`, then write `actions.json`":

| | round 1 (`exec`) | round 2 (`exec resume`) |
|---|--:|--:|
| commands run | 20 | 7 |
| agent messages | 18 | 8 |
| input tokens (cumulative over the turn's calls) | 734,551 | 1,195,456 |
| cached | 654,848 (89%) | 1,090,560 (91%) |
| output | 16,071 | 23,031 |
| reasoning | **0** | **0** |
| cost | $0.021 | $0.032 |
| wall | 3.5 min | ~3 min |

It does not read files into context: it writes Python that loads the JSONs, diffs `state`
against `next_state`, and extracts connected components. Round 1's `findings.md` is a broadly
correct mechanics model of the world, with supporting transition ids per claim — cloud band that
`left`/`right` move, `down` spawns rain at row 1 under the band, `noop` makes rain fall one row,
`click` toggles the 2×2 corner. Round 2 resumed the *same* session (it opened with "let me
investigate why the right actions stopped working" rather than re-deriving), updated
`findings.md`, and emitted a schema-valid plan. Both rounds produced `actions.json` in the arm's
format. **The loop, the workspace persistence and `exec resume` all work.**

Two operational notes from the same run: `codex exec resume` does **not** accept `-C` (set the
cwd instead), and per-turn input grew 734k → 1.20M between rounds, which is §8's context-growth
risk with a number on it. Note the *per-call* peak within a single round is only ~50–60k — the
window ceiling (F14) does not bite on round 1, it bites once a problem runs several rounds.

**F12. `model_reasoning_effort` unset means reasoning OFF, not "provider default" — and §4 said
to leave it unset.** Everything in F11 was produced with **zero reasoning tokens**. That is not
the model: the same model on the same key produces reasoning through *both* API surfaces —
OpenRouter chat/completions with the planner's own provider pin and no reasoning field returns
145 reasoning tokens, and `/responses` with no reasoning field returns 200. Codex is what
suppresses it. Controlled A/B, same prompt, same config otherwise:

| `-c model_reasoning_effort` | output | reasoning |
|---|--:|--:|
| unset | 48 | **0** |
| `"medium"` | 168 | **123** |

The planner arms run with provider-default thinking ON, and memory `planner-speed-bench-aug19`
records that reasoning tokens — not model identity — are what buy planning quality. So leaving
the effort unset, as §4 originally instructed, would have run a **non-thinking** agent against
thinking baselines and mislabelled the result as a matched comparison. **Decided: `model_reasoning_effort = "medium"`, and it buys a better world model (measured).**
The F11 probe was re-run at medium, same corpus, same task, fresh session:

| | effort unset | **medium** |
|---|--:|--:|
| model calls | 32 | **25** |
| input / cached | 734,551 / 89% | 714,510 / 93% |
| output | 16,071 | 57,398 |
| reasoning | 0 | **46,694 (81% of output)** |
| cost | $0.021 | **$0.026** (+23%) |
| wall | 3.5 min | **13 min** (3.7×) |
| peak per-call context | 62,852 | 48,228 |

Fewer calls, denser ones. The `findings.md` it produces is not marginally better, it is a
different quality of model. Unset found "a gray band that moves, `down` spawns, `noop` falls,
`click` toggles a gold block" — and wrongly folded the falling into `left`/`right`. Medium found
the **latent variable**: the 2×2 gold area is a *battery*; the 3-cell cursor overlapping column 0
drains or recharges it; `down` spawns at the cursor's **middle** column; and the spawned
particle is **blue if the battery is charged, lightblue if not**. It separated `noop`'s falling
from the cursor moves, cited transition ids per claim, and flagged its own residual uncertainty
("the column range varies with context (seems related to the cursor's projection?)") rather than
asserting through it.

A hidden state that determines an observable is precisely what memory `n2ntd-mixed-run-forensics`
records the REX learners failing to induce. That the agent arm finds one at medium effort, on
the same 60 transitions, is the arm's whole hypothesis showing up in a pilot.

`low` and `high` were not swept — medium is the choice of record, not a measured optimum, and
the run must log it.

Silver lining: the F11 quality above is the *floor*, produced with no thinking at all.

**F13. Codex has no metadata for this model.** Every session opens with an error item:
`Model metadata for 'deepseek/deepseek-v4-flash' not found. Defaulting to fallback metadata;
this can degrade performance and cause issues.` Codex is therefore guessing the context window
and the auto-compact threshold, and a wrong (small) guess means a long single-phase session
compacts mid-problem and throws away its own analysis. F14 measures how wrong the guess is and
what actually fixes it.

**F14. `-c model_context_window` is silently ignored; the fix is `model_catalog_json`
(measured).** F13's prescription was wrong — passing `-c model_context_window=1000000` changes
nothing: the session still reports **258,400**, which is codex's fallback (`gpt-5.6-sol`'s
272,000 × the 95% `effective_context_window_percent`, exactly). The real window for
deepseek-v4-flash is 1,048,576, so codex is working with **a quarter** of it, and a long
single-phase session would auto-compact and discard its own analysis long before it needed to.

The mechanism that works is `-c model_catalog_json="<path>"` — a *file path*, schema-validated
at config load, so iterating on it costs no API calls. Do not hand-roll the entry (it has ~35
required fields including a 17,730-character `base_instructions`, which is codex's own harness
prompt and must not be dropped). Instead **clone a real one**:

```python
d   = json.load(open("default_catalog.json"))          # codex debug models > default_catalog.json
m   = [x for x in d["models"] if x["slug"] == "gpt-5.6-sol"][0]
m  |= {"slug": "deepseek/deepseek-v4-flash", "display_name": "DeepSeek v4 Flash",
       "context_window": 1_000_000, "max_context_window": 1_000_000,
       "max_output_tokens": 393_216, "auto_compact_token_limit": 900_000}
json.dump({"models": [m]}, open("ds_catalog.json", "w"))
```

Verified: the session then reports `model_context_window = 950000` (1M × 95%), a 3.7× lift, and
the "Model metadata not found" error disappears. The `1_000_000` is not the headline 1,048,576 —
it is the **minimum across the pinned providers** (alibaba/fp8 serves 1,000,000; parasail and
novita serve 1,048,576), so the arm cannot assume more than the smallest host in its own pin.

**F15. Only the request *body* pins an OpenRouter provider — so the proxy is mandatory, not a
convenience (measured).** §4 assumed codex cannot express OpenRouter's `provider` field. It
cannot, and neither can anything else codex *can* express. Four routes tested by pinning to
`venice` and then asking OpenRouter's `/generation` endpoint who actually served the call:

| route | codex can express it? | HTTP | actually served by |
|---|---|--:|---|
| body `provider.only` | **no** | 200 | **Venice** ✅ |
| model slug `…:venice` | yes (`model=`) | 200 | DeepInfra ❌ |
| query `?provider=venice` | yes (`query_params`) | 200 | SiliconFlow ❌ |
| header `X-OpenRouter-Provider` | yes (`http_headers`) | 200 | Phala ❌ |

The three codex *can* express are silently ignored — 200 OK, random provider. Unpinned, four
consecutive calls landed on Baidu, DeepInfra, SiliconFlow and Phala; the catalogue includes an
**fp4** host (`atlas-cloud`) alongside the fp8 ones, so "same model" without the pin is not the
same hardware or the same quantisation as the planner arms. Codex's provider config exposes only
`base_url`, `env_key`, `wire_api`, `http_headers`, `env_http_headers`, `query_params` — nothing
reaches the JSON body. ⇒ **the injecting proxy of §4 is load-bearing**, and since it is the only
thing touching the body, it is also the right place to normalise every other parity knob
(`provider.only`, `usage.include`, and the sampling params — the planner sends no `temperature`,
`top_p`, `seed` or `max_tokens`, so neither should the arm).

---

## 3. Findings: the evaluation contract, and what leaks out of the training pool

### The contract (from `eval_curated_plan.py`, `eval_curated_online.py`, `planning_v2.py`)

* **Problems**: `logs/2026-08-29/planning_v2/problems.json`, schema v2.2, 86 rows, 15 games.
* **A state is a replay address**, not a snapshot: `(program, seed, prefix)`. Every rollout
  branches through the prefix and hard-checks the replayed start before any paid call.
* **Live engine**: `eval_coverage_online.Branch` — a persistent `AutumnBenchEnvWrapper`
  branched at the prefix, with `step(action)` / `grid()` / `terminated`.
* **Goal presentation is a property of the run.** Every row carries both an exact frame and a
  registered Python checker + sentence; `select_goal_presentation(rows, "nl", "any")` configures
  it and raises on anything else. Under `nl` the goal frame must be **dropped, not merely
  unrendered**.
* **Budgets**: `apply_action_caps(rows, "per-problem")` — 2× reference up to 10 actions, 1.5×
  above. Measured over the 86 rows: **min 2, median 17, max 60, total 1725**. Scaled caps
  require floors measured at the same budget or the run refuses to start.
* **Scoring**: any-step, closed-loop. `make_goal_test(row)` is the live stop test;
  `execute_and_score(row, actions)` is the authoritative re-score of a recorded action list.
  Quiescence-requiring checkers are **waived** online — that is the rule `raw` and `lmwm` were
  scored under, so it is the rule here, with no second rule and no re-score fork.
* **Filters**: `--max-floor 0.95`.

### The pool has two layouts, and only one of them is the training signal

`offline_learning/human_data/<game>/informative_curated/` holds both, uniformly across the 15
games:

* `drives/{train,test}_d0..3/episode_0/trajectory.csv` — 8 **full human sessions**, 159–241
  steps each (`Step,Action,Reasoning,Observation,…`; grids are JSON 2-D arrays of colour-name
  strings). 1.9–12 MB per game, 72 MB total.
* `{train,test}_d0..3/episode_*/trajectory.csv` — **2-row transition pairs** carved out of those
  drives: 60 train, 50 test per game. These are the scored items.

`rexpure_optimize.build_data` reads the **60 train pairs**, backfills 9 frames of context from
`drives/train_d*` *only*, and strips the observation header. That set — not the directory — is
"the training data" this arm has to match, and it is what `icl_context.load_pool_transitions`
already returns for the `icl` arm. Pointing the agent at `informative_curated/` would hand it
the 50 held-out pairs and the four test drives as well: a fifth leak, and the easiest one to
walk into because it looks like generosity rather than contamination.

⇒ The exporter must not walk the directory. It calls the `icl` loader (§5 Phase 2), which by
construction reads only `train_d*`, backfills only from `drives/train_d*`, strips the header,
and never opens `MANIFEST.json` — so leaks 1, 2, 3 and 5 below are structurally absent rather
than scrubbed after the fact.

### Five leaks, verified by reading the pool files

1. `MANIFEST.json` → `"human_game": "ice"` — the world's real name, every game.
2. `MANIFEST.json` → `selection_note` is a **hand-written English dynamics summary**:
   *"night spawns+solid stacking, bulk click-flips with drops on screen, liquid rain+slide over
   long noop falls…"*. That is part of the answer key.
3. Drive metadata → `"task_id": "ice_defect_detection"` — the name again.
4. Every CSV `Observation` cell carries a `Task:/Step:/Phase:/Available actions now:` header.
   Memory `obs-metadata-stripped` records this is **stripped at load** for the NLWM learners
   since 2026-08-04 (`Step` was a cFD side channel). Parity demands the agent see the stripped
   form — otherwise it is not "equivalent data", it is *more* data.
5. `test_d*` and `drives/test_d*` are the learner's held-out targets. They are not part of the
   training signal and must not be exported.

Plus ambient repo hazards: `clean_data3/<game>/dynamics.txt` is ground-truth dynamics,
`planning_nl_goals.py` is the checker source, `problems.json` holds every reference plan. The
workspace must not sit inside this repository and must not be able to reach it.

⇒ Never point the agent at the live pool. **Export a sanitised corpus** (§5, Phase 2).

---

## 4. Setup

### Model plumbing — where the deepseek calls go

The planner arms run `deepseek/deepseek-v4-flash` through OpenRouter pinned to
`--provider-only parasail/fp8,novita/fp8,alibaba/fp8` (`launch_planning_v2_online.py`), with
**no** `--reasoning-json` override — i.e. provider-default thinking, ON.

Two consequences for the agent arm:

* **Set `model_reasoning_effort` explicitly — do not ship upstream's `"none"`, and do not
  leave it unset either.** An earlier draft said unset would inherit the provider's default
  thinking. F12 measures that it does not: unset yields *zero* reasoning tokens, while the
  planner arms get provider-default thinking ON. Pick an effort, calibrate it against the
  planner's reasoning-token volume, and record which level the run used. (Memory
  `planner-speed-bench-aug19`: reasoning tokens, not model identity, buy planning quality — so
  this is not a detail.) Supply model metadata through
  `-c model_catalog_json="<path>"` — a cloned catalog entry, since `-c model_context_window` is
  ignored (F14).
* **Route through a tiny local proxy** rather than pointing codex straight at OpenRouter.
  Codex builds its own request body and cannot express OpenRouter's `provider` field, so a
  ~40-line passthrough that injects `{"provider": {"only": [...]}}` is what buys *routing*
  parity with the planner arms (same fp8 hosts, not just the same model name). It also gives
  per-call token/cost accounting in the same units the other arms log, which §6 needs.
  `scripts/claude_cli_proxy.py` is the precedent. Codex accepts any Responses-compatible
  `base_url` (F7), so the proxy is just another one.

### The agent's settings, in one place

Every knob the arm has, with its value and where the value came from. "measured" means this
plan's Findings established it; "chosen" means it is a judgement that must be recorded in the
run manifest so the arm is reproducible.

| knob | value | basis |
|---|---|---|
| `model` | `deepseek/deepseek-v4-flash` | matched to `raw`/`icl`/`lmwm` |
| provider pin | `parasail/fp8,novita/fp8,alibaba/fp8`, injected into the body by the proxy | **measured** — nothing codex can express pins a provider (F15) |
| `model_reasoning_effort` | **`"medium"`** | **decided**; unset means zero reasoning (F12) |
| model metadata | `-c model_catalog_json=<cloned entry>` | **measured** — `-c model_context_window` is ignored (F14) |
| `context_window` | 1,000,000 → 950,000 effective | **measured** — min across the pinned providers, not the 1,048,576 headline (F14) |
| `max_output_tokens` | 393,216 | **measured** — min across the pin |
| `auto_compact_token_limit` | 900,000 | **chosen** — headroom under the effective window; compaction mid-problem is a data-loss event, so it should never fire |
| sampling params | none sent | **measured** — the planner sends no `temperature`/`top_p`/`seed`/`max_tokens`; the proxy strips any codex adds |
| `--log-window` | `None` (full log) | PRO-LONG's default condition; `25` / `-1` are the published ablations |
| batch length | 1..remaining budget | §4 |
| `agent_retries` | 5 | upstream default; covers malformed `actions.json` |
| `--study-rounds` | 5 | **chosen** — §4; bounds "not yet" rounds |
| per-call timeout | ≥ 15 min | **chosen** — a medium-effort turn ran >15 min on the pilot workload; too tight a timeout would kill working sessions |
| `log_post_board` | `True` | §4 |
| codex version | pinned in the sandbox image | 0.151.0 is what every finding here was measured against |

Two of these are traps rather than settings: `-c model_context_window` looks like it works and
does not, and an unset reasoning effort looks like "provider default" and is not. Both would
have failed silently, producing a run that looked matched and was not.

### One phase per problem — the PRO-LONG loop, unmodified

**86 sessions, one per problem, no learn/plan split.** This is what upstream already does: a
PRO-LONG run is one continuous agent session against one game, and the only structure in it is
the runner's turn loop. An earlier draft of this plan split it in two — learn once per game,
then plan per problem — to mirror how NLWM is built. That imported NLWM's shape into an arm that
does not need it, and it cost machinery: a Phase-A artifact per game, a copy of it into each
problem's workspace, and a rule that a session must never write into another's memory. Dropping
the split deletes all three.

**The session.** Workspace at t=0:

* `drives/` — the 60 exported transitions for that game (§5 Phase 2), the only training signal;
* `logs.txt` — one section, `[INITIAL BOARD STATE]`: the problem's start grid, replayed from
  `(program, seed, prefix)` and hard-checked before any paid call;
* `AGENTS.md` — the system prompt: action alphabet, goal sentence, action cap, log format.

No goal frame (the run is `nl`), no reference plan, no game name.

**The turn.** The agent does whatever it likes in the workspace — grep the transitions, write
and run Python over them, build a model, save notes — and ends the turn by writing
`actions.json`. That write **is** the special action: the only channel through which the session
touches the world. The runner drains the list one action at a time through `Branch`, appends
each action and the grid it produced to `logs.txt`, runs the goal test after each, then resumes
the *same* codex session with the updated log. Ends on goal, `terminated`, or cap.

**Two budgets, and only one of them is the arm's.** Actions drained through `Branch` count
against the per-problem cap (2–60). Tool calls inside the workspace cost tokens and wall clock
and count against nothing. That asymmetry *is* the treatment; §6 prices it.

**Letting the agent think without paying an action.** Upstream's `_call_agent` returns false on
an empty action list, so `_wait_for_plan` treats it as a malformed round: nudge, retry, and
after `agent_retries` an exponential backoff sleep. Under a single phase that is wrong in a way
it was not under two, because turn 1 is now where the corpus gets read, and an agent may
legitimately want two or three turns of study before committing its first action — at `cap = 2`,
the smallest budget in the set, a forced action is most of the arm's budget spent to buy a turn.
So distinguish the two cases in the Autumn runner (~15 lines, no upstream change):

| `actions.json` | meaning | effect |
|---|---|---|
| missing or malformed | failure | existing retry path, unchanged |
| `{"actions": []}` | "not yet" | **study round**: logged, action counter untouched, session resumed with the remaining study/action budget in the prompt |

Bounded by `--study-rounds` (default 5) and recorded per problem. The channel stays single: the
agent's only way to act is `actions.json`, and an empty list means it chose not to.

**What is deliberately not built.** A live `step()` tool inside the container — an MCP server,
or a shell command that pokes the host engine — is the other reading of "a special action that
progresses the game". Reject it: it takes the loop away from the runner, ends `logs.txt` as the
single history, and needs a broker process holding the engine and reachable from an `--internal`
network. `actions.json` already has the same expressive power — a list of length 1 is "one
action, then show me" — at zero new infrastructure. Less orchestration is the whole point.

**On batch length.** PRO-LONG lets the agent choose 1–N actions per plan. Keep that, capped at
the remaining budget. It is not an unfairness in either direction: the online arms replan after
every action, and an agent that wants that can emit lists of length 1; an agent that commits to
ten blind is choosing to give up feedback. Set `log_post_board=True` so every intermediate grid
lands in the log regardless.

**What the single phase changes about the comparison.** Two things, both to be reported rather
than smoothed over:

1. *No amortisation.* NLWM pays for learning once per game and reuses the artifact across that
   game's problems; this agent re-reads the corpus in all 86 sessions. A cost asymmetry, not a
   score one — §7 re-prices it.
2. *The agent reads the data knowing the goal.* NLWM's beliefs are goal-agnostic by
   construction; a single-phase session retrieves against the sentence in front of it. That is
   a real advantage, and it is the honest shape of "a coding agent handed the data". If the
   goal-agnostic version is wanted too, it is exactly the old two-phase design — worth keeping
   as an ablation, not as the arm.

One session per problem — not one per game battery — for the reason
`cc_autumn/autumn-code/rig/curated.py` documents: played as a battery, what the agent learns on
problem 1 it still knows on problem 8, which `raw`, `icl` and `lmwm` never do.

### Conditions

| condition | flag | what it isolates |
|---|---|---|
| `agent` | default | the arm |
| `agent-nodata` | `drives/` absent | **run this second.** The agentic analogue of `raw`: separates "learned from our pool" from "deepseek with a shell is just better at grids" |
| `agent-2phase` | learn once per game, reuse | the old two-phase design, kept as the goal-agnostic ablation (§4) |
| `lw25` / `no-log` | `--log-window 25` / `-1` | PRO-LONG's own published memory ablation, free if the budget allows |

---

## 5. Build plan

### Phase 0 — re-point the fork (½ day)

```bash
cd RGB-Agent
git remote add upstream https://github.com/alexisfox7/PRO-LONG.git
git fetch upstream
git checkout -b prolong-autumn upstream/main      # fresh from 9d2f2d4
```

Keep `texttool-sandboxed-python` for provenance (it is the branch the submodule pointer records
and what produced `evaluation_results_autumn/`). Move the pointer only once the new branch runs.
Put the adaptation in a **new sibling**, `research/autumn/`, so the diff against upstream stays
one readable directory; `research/arc-agi-3/` is touched only by import.

### Phase 1 — the Autumn harness (2–3 days)

Reuse unchanged (2010 lines, zero ARC deps): `agent/base.py`, `codex_agent.py`,
`codex_events.py`, `action_queue.py`, `utils/`, `metrics/`.

Write (~600 lines, replacing 1212):

* `research/autumn/env.py` — `AutumnPlanningEnv(BaseEnv)` over `Branch`. `reset()` branches
  `(program, seed, prefix)` and asserts the start; `step()` returns the observation dict
  `GameRunner` expects; `done` = goal reached | terminated | cap.
* `research/autumn/prompts.py` — the Autumn system prompt. ACTION1–7 → `left/right/up/down/
  noop/click ROW COL`; hex palette → colour-name grids; "solve all levels / score = level
  cleared" → "reach this goal within N actions". Keep the shape: ~30 lines, log markers,
  `actions.json` contract, "parse it programmatically".
* `research/autumn/action_queue.py` — Autumn verbs + `click ROW COL` parsing, **row-first**.
  Memory `click-arg-order-swap`: the interpreter's `click(col,row)` is x,y while agent actions
  are row-major; `planning_v2._action` is the reference implementation.
* `research/autumn/runner.py` — trimmed `GameRunner`: same log format, same queue drain, same
  `_wait_for_plan` retry discipline; level/score/WIN machinery replaced by
  `goal_test(grids, executed, new_grid)`, plus the empty-list study round of §4.
* `research/autumn/rig.py` — imports (never copies) `eval_curated_plan` / `eval_curated_online`
  for the loader, goal configuration, floor filter, live test and authoritative re-score.
  Reuse `cc_autumn/autumn-code/rig/curated.py`'s `LABELS` map verbatim: opaque world codes so
  the workspace directory name is not itself a hint.
* `research/autumn/launch.py` — replaces `swarm.py`: enumerate the 86 problems, one session
  each, checkpoint/resume per problem. No per-game pass, no artifact to copy.
* Codex provider wiring: add the `-c model_providers.*` / `-c model_provider` overrides from
  F7 to `_build_codex_args`, pass the proxy key instead of `CODEX_API_KEY`, and drop the
  `reasoning_effort="none"` default.

### Phase 2 — sanitised corpus exporter (½ day, mostly deleted)

`offline_learning/scripts/export_agent_corpus.py` is a thin shell around the `icl` arm's loader,
not a new export:

```python
trs = icl_context.load_pool_transitions(game)   # train_d* only, 9-frame context, header stripped
icl_context.assert_matches_launch(game, trs, artifact_root)
```

Then write those 60 transitions under `<out>/<LABEL>/` as files the agent can grep — one JSON
per transition (`state`, `action`, `next_state`, `context[9]`) plus an `index.csv` — with a
README describing only the *format*. `<LABEL>` is the opaque world code, never the game name.

Because the loader never opens `MANIFEST.json`, never touches `test_d*`, and strips the
observation header itself, leaks 1, 2, 3 and 5 are structurally absent. What remains is the
backstop assertion: the export contains none of the 15 English game names, no `human_game`, no
`dynamics`, no `selection_note` substring, no `Task:`/`Step:` header. Fail the export, not the
run. Test in `tests/test_agent_corpus.py`, including a byte-level check that the exported set is
the same 60 transitions `icl_context.build_icl_block` renders for that game — which is what
makes `agent − icl` a controlled contrast rather than an assertion.

### Phase 3 — scoring and reporting (1 day)

Emit one row per problem in the **online evaluator's `rows` shape** (`task_uid`, `tier`,
`goal_presentation`, `action_cap`, `random_floor`, and an `agent` cell of
`{status, attempts, pass_rate, pass_any}`) so nothing downstream needs a special case. Success
is `execute_and_score` on the recorded action list, never the live verdict. Then teach
`report_planning_v2_online.py` to fill the `Agent` column (currently hard-coded `\textemdash`
in `tex_results`) and extend `--check` to assert the agent run shares the problem set and the
per-problem budgets, as it already does for the NLWM pair.

### Phase 4 — run

1. Free dry run: enumerate, replay every prefix, confirm 86 starts reproduce and every
   reference plan still satisfies its goal on this path. Zero paid calls.
2. **Pilot on one game** — `bt3gb`/ice, 8 problems: the reference game, spans L1–L4, caps 2–50.
   The pilot's real question is not the score but whether deepseek-v4-flash sustains a long
   agentic session reliably (§8).
3. Full 15 games / 86 problems.
4. `agent-nodata`, then `lw25` / `no-log`.

---

## 6. Comparability checklist

- [ ] 86 `task_uid`s identical to `logs/2026-09-03/planning_v2_online_ds_percap_nl`
- [ ] `--cap-mode per-problem`; 0 rows differing in budget from that run
- [ ] `--goal-presentation nl`; goal frame absent from the workspace on every row
- [ ] `--max-floor 0.95`, floors measured at the same budget
- [ ] model `deepseek/deepseek-v4-flash`, **same provider pin**, and an **explicit**
      `model_reasoning_effort` calibrated against the planner's reasoning-token volume —
      unset means zero reasoning (F12); `model_context_window` pinned (F13)
- [ ] training corpus = the same 60 transitions `icl_context.load_pool_transitions` gives the
      `icl` arm, byte-identical, leak assertions passing
- [ ] success = `execute_and_score` on the recorded action list
- [ ] study rounds bounded and recorded per problem; no study round ever advanced the engine
- [ ] agent never reached this repo, the web, or another problem's workspace (audit, not hope —
      `cc_autumn/autumn-code/rig/audit.py` is the precedent worth reusing)
- [ ] tokens + cost per problem recorded **for all four arms**, since inference compute is the
      one axis that is not matched — and the single phase means the agent re-pays per problem
      what `lmwm` paid once per game

---

## 7. Cost and infrastructure

* **Cost.** At $0.087/M in, $0.174/M out, $0.017/M cached and ≈$0.009 per measured agent call:
  ~10–20 agent calls per problem for the planning itself ≈ $0.10–0.20, plus the corpus reading
  the two-phase design amortised across a game and the single phase pays in every session — a
  23k–159k-token pool, so call it +$0.30–1.50 — giving **$0.4–1.7 per problem, $35–150 for the
  86**. Budget **$50–200 for the whole arm**, ablations included. Still two orders cheaper than
  the Opus-5 agentic run, which is itself a reportable fact about what the arm costs; the single
  phase just makes that comparison slightly less flattering than the two-phase one did, and that
  should be said rather than absorbed.
* **Containers.** `docker` on this host is a shim to rootless podman (`~/.local/bin/docker`);
  `docker info` works. `codex_agent.py` shells out to `docker run` and should work unmodified;
  memory `rgb-agent-rootless-setup` has the prior art if not.
* **Egress.** Keep the `--internal` network + squid lockdown and change the allowlist to the
  proxy/OpenRouter (F10). If the lockdown has to be opted out of
  (`CODEX_DOCKER_NETWORK=host`), the audit becomes the only evidence the agent stayed inside
  its workspace — so then it is not optional.
* **Images.** `docker/codex-sandbox` installs `@openai/codex` via npm inside the image; host
  npm is not needed. Build once, and pin the codex version in the image so a CLI upgrade cannot
  silently change the arm mid-run (0.153.4 is already out; 0.151.0 is what F6–F8 were measured
  against).
* **No subscription limits.** Unlike the Opus-5 arms this is a pay-per-token API key, so none
  of the 5-hour-window ops in memory `claude-reflection-proxy` apply.

---

## 8. Risks

1. **deepseek-v4-flash may not sustain long agentic sessions.** It drove the tool loop
   correctly in a trivial test; 86 multi-round planning sessions over megabytes of CSV is a
   different ask, and the single phase makes each session longer still. This is the pilot's
   actual question. Mitigations already in the harness:
   `agent_retries` + `_RETRY_NUDGE` on a missing/malformed `actions.json`, and the runner
   treating an unparseable plan as a failed round rather than a crashed run. If it cannot hold
   a session, the honest report is *that* — a cheap model that can plan one-shot but cannot run
   an agent loop is itself a result about where the compute goes.
2. **Context growth, sharpened by the single phase.** Codex re-sends the conversation each
   turn, and corpus reading now shares a session with planning instead of having its own. The
   exported pool is 23k–159k tokens (`logic_gates` worst); deepseek-v4-flash has a 1M window and
   the prompt says grep, don't read. The pilot's second question is whether a session pulls the
   whole pool into context on turn 1 and then carries it for the rest of the problem — which
   would quietly turn the arm into `icl` with extra steps.
3. **Provider pin.** `parasail/fp8,novita/fp8,alibaba/fp8` are fp8-quantised third-party hosts.
   If the injecting proxy proves awkward, the fallback is unpinned OpenRouter routing — but
   then the agent and the planner are being served by different hardware, which must be stated
   rather than glossed.
