"""Score LLM uncertainty for post-dedup unanswered questions per step.

Walks a stepwise_eb_learn run directory and, for each step:
  1. Extracts the post-dedup unanswered question list from trim_log.json
     (parsed from the selection prompt — same wording the run actually used).
  2. Extracts default_knowledge and current_beliefs from the same prompt.
  3. Asks an LLM, one question at a time, to predict the answer and rate
     its uncertainty on a 0.0–1.0 scale given those beliefs/knowledge.
  4. Writes uncertainty_scores.json into the step directory.

A combined viewer file (uncertainty_viewer.html) is written at the run root
with all per-step results embedded so it can be opened directly in a browser.

Usage:
    uv run score_uncertainty.py <run_dir> [--episode 0] \\
        [--model openrouter/google/gemini-2.5-flash] [--max-concurrent 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import litellm
from omegaconf import OmegaConf

from balrog.utils import setup_environment
from llm_utils import build_llm_input, extract_llm_response_text, extract_xml_key
from improve import _get_response_cost

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing the selection prompt
# ---------------------------------------------------------------------------


_SECTION_RE = {
    "default_knowledge": re.compile(
        r"=== DEFAULT KNOWLEDGE ===\n(.*?)\n=== END DEFAULT KNOWLEDGE ===",
        re.DOTALL,
    ),
    "beliefs": re.compile(
        r"=== CURRENT BELIEFS ===\n(.*?)\n=== END CURRENT BELIEFS ===",
        re.DOTALL,
    ),
}

_Q_RE = re.compile(r"^Q(\d+):\s+(.+?)\s+->\s+UNANSWERED\s*$", re.MULTILINE)


def parse_selection_prompt(prompt: str) -> dict:
    """Pull default_knowledge, beliefs, and post-dedup unanswered Qs out."""
    default_knowledge = ""
    beliefs = ""
    m = _SECTION_RE["default_knowledge"].search(prompt)
    if m:
        default_knowledge = m.group(1).strip()
    m = _SECTION_RE["beliefs"].search(prompt)
    if m:
        beliefs = m.group(1).strip()

    questions: list[dict] = []
    for match in _Q_RE.finditer(prompt):
        questions.append(
            {
                "q_number": int(match.group(1)),
                "question": match.group(2).strip(),
            }
        )
    return {
        "default_knowledge": default_knowledge,
        "beliefs": beliefs,
        "questions": questions,
    }


# ---------------------------------------------------------------------------
# Uncertainty prompt + scoring
# ---------------------------------------------------------------------------


def build_uncertainty_prompt(
    question: str, beliefs: str, default_knowledge: str
) -> str:
    default_knowledge_section = ""
    if default_knowledge:
        default_knowledge_section = f"""
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===
"""
    beliefs_section = f"""
=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty — no beliefs yet)"}
=== END CURRENT BELIEFS ==="""

    return f"""You are evaluating how uncertain an agent should be about answering a yes/no question, given only its current background knowledge and beliefs about an environment.

{default_knowledge_section}
{beliefs_section}

=== QUESTION ===
{question}
=== END QUESTION ===

Based ONLY on the default knowledge and current beliefs above, do two things:

1. Give your best guess for the answer: YES, NO, or UNKNOWN.
2. Rate your uncertainty on an integer scale from 1 to 5:
   - 1 = least uncertain / most certain — the beliefs/knowledge fully settle the question.
   - 2 = mostly certain — strong support, only minor doubt.
   - 3 = moderately uncertain — can lean one way but evidence is weak.
   - 4 = mostly uncertain — only a faint hint, mostly guessing.
   - 5 = most uncertain / least certain — no basis at all to prefer YES over NO.

Do not invent new facts beyond what is stated in the beliefs/knowledge.

Format your response as:
<think>
Brief reasoning: what's known, what's missing.
</think>
<answer>YES or NO or UNKNOWN</answer>
<uncertainty>an integer in {1, 2, 3, 4, 5}</uncertainty>"""


def parse_uncertainty_response(text: str) -> dict:
    answer = extract_xml_key(text, "answer")
    uncertainty_raw = extract_xml_key(text, "uncertainty")
    think = extract_xml_key(text, "think")

    uncertainty: int | None = None
    if uncertainty_raw is not None:
        m = re.search(r"[-+]?\d+", uncertainty_raw)
        if m:
            try:
                value = int(m.group(0))
                if value < 1:
                    value = 1
                elif value > 5:
                    value = 5
                uncertainty = value
            except ValueError:
                uncertainty = None

    norm_answer: str | None = None
    if answer is not None:
        val = answer.strip().upper()
        if "YES" in val and "NO" not in val:
            norm_answer = "YES"
        elif "NO" in val and "YES" not in val and "UNKNOWN" not in val:
            norm_answer = "NO"
        elif "UNKNOWN" in val:
            norm_answer = "UNKNOWN"
        else:
            norm_answer = val
    return {
        "answer": norm_answer,
        "uncertainty": uncertainty,
        "reasoning": (think or "").strip(),
    }


async def call_llm(
    model: str,
    prompt: str,
    temperature: float | None,
) -> tuple[str, float]:
    api_kwargs = {
        "model": model,
        "input": build_llm_input(prompt),
        "num_retries": 5,
    }
    if temperature is not None:
        api_kwargs["temperature"] = temperature
    response = await asyncio.to_thread(litellm.responses, **api_kwargs)
    text = extract_llm_response_text(response)
    # Strip the litellm provider prefix for cost lookup — "openrouter/google/..." -> "google/..."
    bare_model = model.split("/", 1)[1] if "/" in model else model
    try:
        cost = _get_response_cost(response, bare_model)
    except Exception:
        cost = 0.0
    return text, cost


async def score_step(
    step_dir: Path,
    model: str,
    temperature: float | None,
    sem: asyncio.Semaphore,
) -> dict:
    trim_log_path = step_dir / "trim_log.json"
    if not trim_log_path.exists():
        log.warning(f"[{step_dir.name}] no trim_log.json — skipping")
        return {
            "step_name": step_dir.name,
            "skipped": True,
            "reason": "no trim_log.json",
        }

    trim_log = json.loads(trim_log_path.read_text())
    selection = trim_log.get("selection") or {}
    sel_prompt = selection.get("prompt") or ""
    if not sel_prompt:
        log.warning(f"[{step_dir.name}] empty selection prompt — skipping")
        return {
            "step_name": step_dir.name,
            "skipped": True,
            "reason": "no selection prompt",
        }

    parsed = parse_selection_prompt(sel_prompt)
    questions = parsed["questions"]
    if not questions:
        log.info(f"[{step_dir.name}] no unanswered post-dedup questions")
        return {
            "step_name": step_dir.name,
            "skipped": True,
            "reason": "no post-dedup unanswered questions",
            "default_knowledge": parsed["default_knowledge"],
            "beliefs": parsed["beliefs"],
            "questions": [],
        }

    async def score_one(q: dict) -> dict:
        prompt = build_uncertainty_prompt(
            question=q["question"],
            beliefs=parsed["beliefs"],
            default_knowledge=parsed["default_knowledge"],
        )
        async with sem:
            try:
                text, cost = await call_llm(model, prompt, temperature)
                parsed_resp = parse_uncertainty_response(text)
                return {
                    "q_number": q["q_number"],
                    "question": q["question"],
                    "uncertainty": parsed_resp["uncertainty"],
                    "predicted_answer": parsed_resp["answer"],
                    "reasoning": parsed_resp["reasoning"],
                    "prompt": prompt,
                    "response": text,
                    "cost": cost,
                    "error": None,
                }
            except Exception as e:
                log.error(
                    f"[{step_dir.name}] Q{q['q_number']} scoring failed: {e}",
                    exc_info=False,
                )
                return {
                    "q_number": q["q_number"],
                    "question": q["question"],
                    "uncertainty": None,
                    "predicted_answer": None,
                    "reasoning": "",
                    "prompt": prompt,
                    "response": "",
                    "cost": 0.0,
                    "error": str(e),
                }

    log.info(
        f"[{step_dir.name}] scoring {len(questions)} unanswered questions via {model}"
    )
    results = await asyncio.gather(*(score_one(q) for q in questions))
    total_cost = sum(r["cost"] for r in results)
    return {
        "step_name": step_dir.name,
        "skipped": False,
        "model": model,
        "default_knowledge": parsed["default_knowledge"],
        "beliefs": parsed["beliefs"],
        "questions": results,
        "total_cost": total_cost,
    }


def resolve_model(args: argparse.Namespace, run_dir: Path) -> str:
    if args.model:
        return args.model
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(
            f"--model not given and no config.yaml at {cfg_path}; "
            f"pass --model litellm-style-id (e.g. openrouter/google/gemini-2.5-flash)"
        )
    cfg = OmegaConf.load(cfg_path)
    client_name = cfg.client.client_name
    model_id = cfg.client.model_id
    if client_name == "vllm":
        return f"hosted_vllm/{model_id}"
    return f"{client_name}/{model_id}"


def find_episode_dir(run_dir: Path, episode: int) -> Path:
    cand = run_dir / f"episode_{episode}"
    if cand.is_dir():
        return cand
    episodes = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not episodes:
        raise SystemExit(f"No episode_* directories under {run_dir}")
    raise SystemExit(
        f"episode_{episode} not found under {run_dir}; "
        f"available: {[p.name for p in episodes]}"
    )


def find_step_dirs(episode_dir: Path) -> list[Path]:
    return sorted(
        p for p in episode_dir.iterdir() if p.is_dir() and p.name.startswith("step_")
    )


VIEWER_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Uncertainty Viewer — __TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 0; }
  header { padding: 12px 16px; border-bottom: 1px solid #ccc;
           background: rgba(0,0,0,0.04); position: sticky; top: 0; z-index: 5; }
  h1 { font-size: 16px; margin: 0 0 6px 0; }
  .subtle { color: #666; font-size: 12px; }
  main { padding: 16px; }
  select { font-size: 14px; padding: 4px 8px; }
  details { margin: 6px 0; border: 1px solid #ccc; border-radius: 6px;
            padding: 6px 10px; background: rgba(0,0,0,0.02); }
  details > summary { cursor: pointer; user-select: none; outline: none; }
  details[open] { background: rgba(0,0,0,0.04); }
  .q-row { display: flex; gap: 10px; align-items: baseline; }
  .q-score { font-variant-numeric: tabular-nums; font-weight: 600;
             min-width: 4em; text-align: right; }
  .q-bar { display: inline-block; height: 8px; background: #ddd;
           border-radius: 4px; overflow: hidden; vertical-align: middle;
           width: 120px; margin: 0 6px; }
  .q-bar > span { display: block; height: 100%; background: #d0552b; }
  .answer-pill { display: inline-block; padding: 1px 6px; border-radius: 9px;
                 font-size: 11px; font-weight: 600; margin-left: 6px; }
  .pill-YES { background: #d4f5d4; color: #1f6f1f; }
  .pill-NO { background: #fcdada; color: #8b1f1f; }
  .pill-UNKNOWN { background: #eee; color: #555; }
  .pill-null { background: #f3d6a8; color: #6a4a14; }
  pre { white-space: pre-wrap; background: rgba(0,0,0,0.05);
        padding: 8px; border-radius: 4px; max-height: 50vh; overflow: auto;
        font-size: 12px; }
  .panel-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .panel-row > div { min-width: 0; }
  .belief-block { margin-top: 8px; }
  .belief-block details { background: rgba(0,0,0,0.01); }
  .qbody { margin-top: 8px; }
  .meta { font-size: 12px; color: #555; margin-top: 4px; }
  .step-summary { display: flex; gap: 14px; flex-wrap: wrap; align-items: baseline;
                  font-size: 12px; color: #555; margin: 6px 0; }
</style>
</head>
<body>
<header>
  <h1>Uncertainty scores — <span id="title">__TITLE__</span></h1>
  <div class="subtle">
    Step: <select id="stepSelect"></select>
    <span id="stepCost" class="subtle"></span>
  </div>
</header>
<main id="content"></main>

<script>
const DATA = __DATA_JSON__;

function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "style") e.setAttribute("style", v);
      else if (k.startsWith("on")) e[k] = v;
      else e.setAttribute(k, v);
    }
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function answerPill(ans) {
  const cls = "answer-pill pill-" + (ans || "null");
  return el("span", { class: cls }, ans || "—");
}

function renderQuestion(q) {
  const score = q.uncertainty;
  const scoreStr = (score == null) ? "—" : `${score}/5`;
  const pct = (score == null) ? 0 : Math.max(0, Math.min(5, score)) / 5 * 100;
  const bar = el("span", { class: "q-bar" }, el("span", { style: `width: ${pct}%` }));
  const summary = el(
    "summary", { class: "q-row" },
    el("span", { class: "q-score" }, scoreStr),
    bar,
    el("span", null, `Q${q.q_number}: ${q.question}`),
    answerPill(q.predicted_answer)
  );

  const meta = el("div", { class: "meta" },
    q.error
      ? `error: ${q.error}`
      : (q.cost != null ? `cost: $${q.cost.toFixed(6)}` : "")
  );

  const reasoning = q.reasoning
    ? el("details", null,
        el("summary", null, "reasoning"),
        el("pre", null, q.reasoning))
    : null;

  const panel = el("div", { class: "panel-row qbody" },
    el("div", null,
      el("strong", null, "Prompt"),
      el("pre", null, q.prompt || "")),
    el("div", null,
      el("strong", null, "Response"),
      el("pre", null, q.response || ""))
  );

  return el("details", null, summary, meta, reasoning, panel);
}

function renderStep(stepData) {
  const content = document.getElementById("content");
  content.innerHTML = "";
  if (stepData.skipped) {
    content.appendChild(el("div", null,
      `Skipped: ${stepData.reason}`));
    return;
  }
  const beliefs = stepData.beliefs || "(empty)";
  const dk = stepData.default_knowledge || "(none)";
  content.appendChild(
    el("div", { class: "step-summary" },
      el("span", null, `model: ${stepData.model || "?"}`),
      el("span", null, `${stepData.questions.length} questions`),
      el("span", null, `total cost: $${(stepData.total_cost || 0).toFixed(6)}`),
    )
  );
  content.appendChild(el("div", { class: "belief-block" },
    el("details", null,
      el("summary", null, "Current beliefs"),
      el("pre", null, beliefs))));
  content.appendChild(el("div", { class: "belief-block" },
    el("details", null,
      el("summary", null, "Default knowledge"),
      el("pre", null, dk))));

  // Sort by uncertainty descending (highest uncertainty first)
  const qs = [...stepData.questions].sort((a, b) => {
    const ua = a.uncertainty == null ? -1 : a.uncertainty;
    const ub = b.uncertainty == null ? -1 : b.uncertainty;
    return ub - ua;
  });
  for (const q of qs) content.appendChild(renderQuestion(q));
}

function init() {
  const sel = document.getElementById("stepSelect");
  DATA.steps.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    const tag = s.skipped ? " (skipped)" : ` (${s.questions.length} qs)`;
    opt.textContent = s.step_name + tag;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", () => {
    renderStep(DATA.steps[sel.value]);
  });
  if (DATA.steps.length) renderStep(DATA.steps[0]);
}
init();
</script>
</body>
</html>
"""


def write_viewer(out_path: Path, title: str, steps: list[dict]) -> None:
    payload = {"title": title, "steps": steps}
    html = VIEWER_HTML_TEMPLATE.replace("__TITLE__", title).replace(
        "__DATA_JSON__", json.dumps(payload)
    )
    out_path.write_text(html)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path,
                   help="stepwise_eb_learn run directory (contains config.yaml + episode_*)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--model", default=None,
                   help="litellm model id; defaults to client.client_name/model_id from config.yaml")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--max-concurrent", type=int, default=32,
        help="global concurrency cap shared across ALL steps — every in-flight "
             "LLM call counts against this limit",
    )
    p.add_argument("--steps", default=None,
                   help="comma-separated step indices to score (e.g. '0,2'); default: all")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite existing uncertainty_scores.json for each step")
    p.add_argument("--viewer-out", type=Path, default=None,
                   help="where to write the combined HTML viewer; "
                        "default: <run_dir>/uncertainty_viewer.html")
    args = p.parse_args()

    if not args.run_dir.is_dir():
        log.error(f"run_dir does not exist: {args.run_dir}")
        return 1

    setup_environment(original_cwd=str(Path(__file__).resolve().parent))
    model = resolve_model(args, args.run_dir)
    log.info(f"Using model: {model}")

    episode_dir = find_episode_dir(args.run_dir, args.episode)
    step_dirs = find_step_dirs(episode_dir)
    if not step_dirs:
        log.error(f"No step_* dirs under {episode_dir}")
        return 1

    selected_step_indices: set[int] | None = None
    if args.steps:
        selected_step_indices = {int(s) for s in args.steps.split(",") if s.strip()}

    async def run_all() -> list[dict]:
        # One global semaphore — all LLM calls across all steps share it,
        # so total in-flight requests are bounded but every prompt that can
        # run does run.
        sem = asyncio.Semaphore(max(1, args.max_concurrent))
        results_by_idx: dict[int, dict] = {}
        tasks: list[asyncio.Task] = []

        async def run_one(i: int, step_dir: Path) -> None:
            out_path = step_dir / "uncertainty_scores.json"
            result = await score_step(
                step_dir=step_dir,
                model=model,
                temperature=args.temperature,
                sem=sem,
            )
            out_path.write_text(json.dumps(result, indent=2))
            log.info(
                f"[{step_dir.name}] wrote {out_path.name} "
                f"(cost=${result.get('total_cost', 0):.6f})"
            )
            results_by_idx[i] = result

        for i, step_dir in enumerate(step_dirs):
            if selected_step_indices is not None and i not in selected_step_indices:
                log.info(f"Skipping {step_dir.name} (not in --steps)")
                existing = step_dir / "uncertainty_scores.json"
                if existing.exists():
                    results_by_idx[i] = json.loads(existing.read_text())
                continue
            out_path = step_dir / "uncertainty_scores.json"
            if out_path.exists() and not args.overwrite:
                log.info(f"[{step_dir.name}] uncertainty_scores.json exists — reusing")
                results_by_idx[i] = json.loads(out_path.read_text())
                continue
            tasks.append(asyncio.create_task(run_one(i, step_dir)))

        if tasks:
            await asyncio.gather(*tasks)
        return [results_by_idx[i] for i in sorted(results_by_idx)]

    results = asyncio.run(run_all())

    viewer_path = args.viewer_out or (args.run_dir / "uncertainty_viewer.html")
    title = f"{args.run_dir.name} / episode_{args.episode}"
    write_viewer(viewer_path, title, results)
    log.info(f"Wrote viewer: {viewer_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
