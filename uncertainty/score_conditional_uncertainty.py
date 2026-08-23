"""Score conditional uncertainty: for each post-dedup unanswered question
Q_i, ask the LLM (in ONE prompt per Q_i) to rate its 1–5 uncertainty for
every OTHER question Q_j assuming Q_i has been resolved.

Produces an N×N matrix per step (diagonal undefined) and saves it to
``conditional_uncertainty.json`` in each step directory. Writes a combined
HTML heatmap viewer at the run root.

Usage:
    uv run uncertainty/score_conditional_uncertainty.py <run_dir> [--episode 0] \\
        [--model ...] [--max-concurrent 8] [--overwrite]
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
from explore.improve import _get_response_cost
from score_uncertainty import (
    find_episode_dir,
    find_step_dirs,
    parse_selection_prompt,
    resolve_model,
    score_step as score_uncertainty_step,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt / parsing
# ---------------------------------------------------------------------------


def _indent_block(text: str, prefix: str = "    ") -> str:
    if not text:
        return f"{prefix}(no reasoning recorded)"
    lines = text.rstrip().splitlines() or [text.rstrip()]
    return "\n".join(prefix + ln for ln in lines)


def build_conditional_prompt(
    questions: list[dict],
    focus_index: int,  # 0-based index into questions
    beliefs: str,
    default_knowledge: str,
    baselines: list[dict | None] | None = None,
) -> str:
    """Build a prompt that asks the LLM, for each non-focus question, to
    rate the CONDITIONAL uncertainty assuming the focus question has been
    resolved.

    The unconditional uncertainty is *not* requested here — it is sourced
    separately from ``uncertainty_scores.json`` (one prompt per question,
    focus-independent), to avoid the per-focus drift in unconditional
    scores that the joint prompt exhibits. Downstream marginal info gain
    is computed as ``unconditional - conditional`` using the authoritative
    unconditional from the dedicated pass.

    If ``baselines`` is provided, each entry corresponds to ``questions[i]``
    and contains the authoritative unconditional pass output for that
    question:
        {"uncertainty": int, "predicted_answer": str, "reasoning": str}
    These get rendered into a BASELINE block so the conditional model is
    anchored to the same target reading the unconditional pass committed
    to, and so it can enforce cond ≤ baseline target-by-target.
    """

    focus_q = questions[focus_index]

    # Re-number questions 1..N within the prompt so they're contiguous,
    # independent of the original q_number from the trim_log.
    all_qs_block = "\n".join(
        f"Q{i + 1}: {q['question']}" for i, q in enumerate(questions)
    )
    other_blocks = "\n".join(
        f"  <q n=\"Q{i + 1}\"><conditional>?</conditional></q>"
        for i in range(len(questions))
        if i != focus_index
    )

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

    baseline_section = ""
    if baselines is not None and any(b is not None for b in baselines):
        parts: list[str] = []
        for i, q in enumerate(questions):
            if i == focus_index:
                continue
            b = baselines[i] if i < len(baselines) else None
            if b is None or b.get("uncertainty") is None:
                parts.append(
                    f"Q{i + 1}: (no baseline available — score conditionally with "
                    f"no upper bound)"
                )
                continue
            u = b["uncertainty"]
            ans = b.get("predicted_answer") or "?"
            reasoning = (b.get("reasoning") or "").strip()
            parts.append(
                f"Q{i + 1}: baseline_unconditional = {u}/5  "
                f"(best-guess answer: {ans})\n"
                f"  reasoning used to derive this baseline:\n"
                f"{_indent_block(reasoning, prefix='    ')}"
            )
        baseline_section = (
            "\n=== BASELINE UNCONDITIONAL UNCERTAINTY (authoritative) ===\n"
            "For each target question, the following baseline unconditional "
            "uncertainty has already been independently established by reasoning "
            "about that target alone (no focus question, just the default "
            "knowledge + beliefs above). These baselines are AUTHORITATIVE — use "
            "the SAME interpretation of each target question that the baseline "
            "reasoning uses, and treat the baseline score as the strict upper "
            "bound on your conditional score for that target.\n\n"
            + "\n\n".join(parts)
            + "\n=== END BASELINE UNCONDITIONAL UNCERTAINTY ===\n"
        )

    rules_lines = [
        "  - The conditional score measures absolute remaining uncertainty about "
        "the target's YES/NO answer assuming the focus has been resolved.",
        "  - If beliefs already settle the target, the conditional score should "
        "be 1 (not higher, since adding info cannot increase certainty).",
        "  - The conditional score for a target must NEVER exceed that target's "
        "baseline_unconditional shown above — learning the focus answer can only "
        "reduce or keep uncertainty the same; it can never increase it. If you "
        "find yourself wanting to score conditional > baseline, you are using a "
        "different interpretation of the target than the baseline used; "
        "re-interpret the target the same way the baseline reasoning does.",
        "  - If the focus is informative about the target, the conditional score "
        "should be strictly below the baseline. If the focus is irrelevant to "
        "the target, the conditional score should equal the baseline exactly.",
    ]
    if baselines is None or not any(b is not None for b in baselines):
        # Fall back to the un-anchored rule wording when no baselines are present.
        rules_lines[2] = (
            "  - The conditional score must never exceed your unconditional "
            "uncertainty about the target given only the beliefs above — learning "
            "the focus answer can only reduce or keep uncertainty the same; it "
            "can never increase it."
        )
        rules_lines[3] = (
            "  - If the focus is informative about the target, the conditional "
            "score should be strictly below the unconditional baseline. If the "
            "focus is irrelevant to the target, the conditional score should "
            "equal the unconditional baseline."
        )
    rules_block = "\n".join(rules_lines)

    return f"""You are evaluating how answering one yes/no question would change your uncertainty about other yes/no questions, given only your current background knowledge and beliefs about an environment.

{default_knowledge_section}
{beliefs_section}

=== ALL QUESTIONS ===
{all_qs_block}
=== END ALL QUESTIONS ===
{baseline_section}
=== FOCUS QUESTION ===
Q{focus_index + 1}: {focus_q['question']}
=== END FOCUS QUESTION ===

For EACH OTHER question (every question that is NOT the focus question), give a CONDITIONAL uncertainty score on the integer scale 1..5:

  1 = least uncertain / most certain — fully settled.
  2 = mostly certain — strong support, only minor doubt.
  3 = moderately uncertain — can lean one way but evidence is weak.
  4 = mostly uncertain — only a faint hint, mostly guessing.
  5 = most uncertain / least certain — no basis at all to prefer YES over NO.

CONDITIONAL uncertainty = your absolute uncertainty about the target question's YES/NO answer given the default knowledge, current beliefs, AND the assumption that the focus question has just been resolved (you now know its true YES/NO answer, though it has not been revealed to you — consider implications across both possible resolutions).

Important consistency rules:
{rules_block}

Do not invent new facts beyond what is stated in the beliefs / default knowledge or what would follow logically from resolving the focus question.

Format your response exactly as:
<think>
Brief reasoning. For each target question, state whether the focus resolution adds anything about the target, and how that compares to the baseline.
</think>
<scores>
{other_blocks}
</scores>"""


_SCORE_BLOCK_RE = re.compile(
    r"<q\b[^>]*\bn\s*=\s*[\"']?Q?(\d+)[\"']?[^>]*>(.*?)</q>",
    re.IGNORECASE | re.DOTALL,
)
_UNCERTAINTY_INNER_RE = re.compile(
    r"<uncertainty[^>]*>\s*([-+]?\d+)\s*</uncertainty>",
    re.IGNORECASE,
)
_UNCONDITIONAL_INNER_RE = re.compile(
    r"<unconditional[^>]*>\s*([-+]?\d+)\s*</unconditional>",
    re.IGNORECASE,
)
_CONDITIONAL_INNER_RE = re.compile(
    r"<conditional[^>]*>\s*([-+]?\d+)\s*</conditional>",
    re.IGNORECASE,
)


def _clip_score(v: int) -> int:
    return max(1, min(5, v))


def parse_conditional_response(text: str, num_questions: int, focus_index: int) -> dict:
    """Parse per-target uncertainty scores from the response.

    Each <q n="QK">...</q> block is expected to contain BOTH:
      <unconditional>X</unconditional> and <conditional>Y</conditional>.

    For backward compatibility we also accept a single <uncertainty>Z</uncertainty>
    (older prompt shape) and treat Z as both unconditional and conditional.

    Returns:
        {
            "unconditional": dict[int, int|None],
            "conditional":   dict[int, int|None],
            "marginal":      dict[int, int|None],  # = unconditional - conditional
                                                   # (>=0 when consistency holds;
                                                   # we clamp to 0 if the model
                                                   # outputs cond > uncond).
            "reasoning": str,
            "missing": list[int],  # missing either score
        }
    """
    unc: dict[int, int | None] = {}
    cond: dict[int, int | None] = {}
    for m in _SCORE_BLOCK_RE.finditer(text):
        try:
            q_num = int(m.group(1))
        except ValueError:
            continue
        idx = q_num - 1
        if idx < 0 or idx >= num_questions or idx == focus_index:
            continue
        inner = m.group(2)

        u_match = _UNCONDITIONAL_INNER_RE.search(inner)
        c_match = _CONDITIONAL_INNER_RE.search(inner)
        # backward compat
        if u_match is None and c_match is None:
            legacy = _UNCERTAINTY_INNER_RE.search(inner)
            if legacy is not None:
                try:
                    v = _clip_score(int(legacy.group(1)))
                    unc[idx] = v
                    cond[idx] = v
                except ValueError:
                    pass
            continue
        if u_match is not None:
            try:
                unc[idx] = _clip_score(int(u_match.group(1)))
            except ValueError:
                pass
        if c_match is not None:
            try:
                cond[idx] = _clip_score(int(c_match.group(1)))
            except ValueError:
                pass

    expected = {i for i in range(num_questions) if i != focus_index}
    missing = sorted(
        i for i in expected if unc.get(i) is None or cond.get(i) is None
    )
    for i in expected:
        unc.setdefault(i, None)
        cond.setdefault(i, None)

    marginal: dict[int, int | None] = {}
    for i in expected:
        u = unc[i]
        c = cond[i]
        if u is None or c is None:
            marginal[i] = None
        else:
            # Enforce non-negative info gain: if the model violates the
            # consistency rule (cond > unc), we clamp to 0 so the marginal
            # remains interpretable as "how much did the focus reduce
            # uncertainty about the target?"
            marginal[i] = max(0, u - c)

    return {
        "unconditional": unc,
        "conditional": cond,
        "marginal": marginal,
        "reasoning": (extract_xml_key(text, "think") or "").strip(),
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


async def call_llm(
    model: str, prompt: str, temperature: float | None
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
    bare_model = model.split("/", 1)[1] if "/" in model else model
    try:
        cost = _get_response_cost(response, bare_model)
    except Exception:
        cost = 0.0
    return text, cost


# ---------------------------------------------------------------------------
# Step orchestration
# ---------------------------------------------------------------------------


def _baselines_from_payload(
    payload: dict | None, questions: list[dict]
) -> list[dict | None]:
    """Build per-question baseline dicts (in `questions` order) from an
    uncertainty_scores.json payload. Each entry is:
        {"uncertainty": int|None, "predicted_answer": str|None, "reasoning": str}
    or None if no matching question was found.
    Match is by exact question text (q_number can drift between passes).
    """
    if not payload:
        return [None] * len(questions)
    by_text: dict[str, dict] = {
        q.get("question"): q for q in payload.get("questions", [])
    }
    out: list[dict | None] = []
    for q in questions:
        src = by_text.get(q["question"])
        if src is None:
            out.append(None)
            continue
        u_raw = src.get("uncertainty")
        try:
            u = int(u_raw) if u_raw is not None else None
        except (TypeError, ValueError):
            u = None
        out.append(
            {
                "uncertainty": u,
                "predicted_answer": src.get("predicted_answer"),
                "reasoning": (src.get("reasoning") or "").strip(),
            }
        )
    return out


def _load_authoritative_unconditional(
    step_dir: Path, questions: list[dict]
) -> tuple[list[int | None], list[dict | None]]:
    """Load per-question unconditional baselines from uncertainty_scores.json.

    Returns:
        (unconditional_by_index, baselines_by_index)
        - unconditional_by_index[i] = integer score for ``questions[i]`` or None
        - baselines_by_index[i] = full baseline dict (see _baselines_from_payload)
          or None if the question wasn't found / file missing.
    """
    path = step_dir / "uncertainty_scores.json"
    payload: dict | None = None
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except Exception as e:
            log.warning(f"[{step_dir.name}] failed to read {path.name}: {e}")
            payload = None
    baselines = _baselines_from_payload(payload, questions)
    scores = [b["uncertainty"] if b else None for b in baselines]
    missing = sum(1 for s in scores if s is None)
    if missing:
        log.warning(
            f"[{step_dir.name}] uncertainty_scores.json missing baseline for "
            f"{missing}/{len(questions)} question(s)"
        )
    return scores, baselines


async def score_step(
    step_dir: Path,
    model: str,
    temperature: float | None,
    sem: asyncio.Semaphore,
    *,
    ensure_unconditional: bool = True,
    overwrite_unconditional: bool = False,
) -> dict:
    """Run the conditional pass for one step.

    If ``ensure_unconditional`` is True and the step's uncertainty_scores.json
    is missing or skipped, the unconditional pass for this step is run inline
    (sharing the same global semaphore so it parallelizes with everything else)
    and its output is written before the conditional prompts fire. This is what
    lets a single asyncio.run launch the whole pipeline as one big concurrent
    batch.
    """
    trim_log_path = step_dir / "trim_log.json"
    if not trim_log_path.exists():
        log.warning(f"[{step_dir.name}] no trim_log.json — skipping")
        return {"step_name": step_dir.name, "skipped": True, "reason": "no trim_log.json"}

    trim_log = json.loads(trim_log_path.read_text())
    selection = trim_log.get("selection") or {}
    sel_prompt = selection.get("prompt") or ""
    if not sel_prompt:
        return {
            "step_name": step_dir.name,
            "skipped": True,
            "reason": "no selection prompt",
        }

    parsed = parse_selection_prompt(sel_prompt)
    questions = parsed["questions"]
    if len(questions) < 2:
        return {
            "step_name": step_dir.name,
            "skipped": True,
            "reason": f"only {len(questions)} unanswered question(s) — nothing to condition on",
            "default_knowledge": parsed["default_knowledge"],
            "beliefs": parsed["beliefs"],
            "questions": questions,
        }

    # Authoritative per-question unconditional from the separate one-prompt-per-question
    # pass. Same value populates every focus row's ``unconditional[target]`` so
    # within-step dispersion in unconditional is removed by construction.
    unc_path = step_dir / "uncertainty_scores.json"
    unc_payload: dict | None = None
    if unc_path.exists() and not overwrite_unconditional:
        try:
            unc_payload = json.loads(unc_path.read_text())
        except Exception as e:
            log.warning(f"[{step_dir.name}] failed to read {unc_path.name}: {e}")
            unc_payload = None
        if unc_payload is not None and unc_payload.get("skipped"):
            # Treat skipped baselines as missing so we'll attempt them again.
            unc_payload = None

    if unc_payload is None and ensure_unconditional:
        log.info(
            f"[{step_dir.name}] no usable baselines — running unconditional pass inline"
        )
        unc_payload = await score_uncertainty_step(
            step_dir=step_dir,
            model=model,
            temperature=temperature,
            sem=sem,
        )
        unc_path.write_text(json.dumps(unc_payload, indent=2))

    baselines = _baselines_from_payload(unc_payload, questions)
    auth_unc = [b["uncertainty"] if b else None for b in baselines]

    def _unc_row(focus_idx: int) -> list[int | None]:
        # Authoritative unconditional, but null out the diagonal (focus's own slot).
        out: list[int | None] = []
        for j, v in enumerate(auth_unc):
            out.append(None if j == focus_idx else v)
        return out

    async def score_one(focus_idx: int) -> dict:
        prompt = build_conditional_prompt(
            questions=questions,
            focus_index=focus_idx,
            beliefs=parsed["beliefs"],
            default_knowledge=parsed["default_knowledge"],
            baselines=baselines,
        )
        unc_row = _unc_row(focus_idx)
        async with sem:
            try:
                text, cost = await call_llm(model, prompt, temperature)
                parsed_resp = parse_conditional_response(
                    text=text,
                    num_questions=len(questions),
                    focus_index=focus_idx,
                )
                cond_list = [
                    parsed_resp["conditional"].get(j) for j in range(len(questions))
                ]
                # Marginal uses authoritative unconditional. Clamp cond to <= unc
                # so the consistency rule (cond ≤ uncond) holds — when the model
                # outputs cond > uncond, we treat that as a parsing/judgment
                # error and floor the marginal at 0.
                marg_list: list[int | None] = []
                for j in range(len(questions)):
                    if j == focus_idx:
                        marg_list.append(None)
                        continue
                    u = unc_row[j]
                    c = cond_list[j]
                    if u is None or c is None:
                        marg_list.append(None)
                    else:
                        marg_list.append(max(0, u - c))
                return {
                    "focus_index": focus_idx,
                    "focus_q_number": questions[focus_idx]["q_number"],
                    "focus_question": questions[focus_idx]["question"],
                    # `scores` retained for backward compatibility with the
                    # per-step conditional viewer; it mirrors `conditional`.
                    "scores": cond_list,
                    "unconditional": unc_row,
                    "conditional": cond_list,
                    "marginal": marg_list,
                    "reasoning": parsed_resp["reasoning"],
                    "missing_indices": parsed_resp["missing"],
                    "prompt": prompt,
                    "response": text,
                    "cost": cost,
                    "error": None,
                }
            except Exception as e:
                log.error(
                    f"[{step_dir.name}] focus={focus_idx} failed: {e}",
                    exc_info=False,
                )
                return {
                    "focus_index": focus_idx,
                    "focus_q_number": questions[focus_idx]["q_number"],
                    "focus_question": questions[focus_idx]["question"],
                    "scores": [None] * len(questions),
                    "unconditional": unc_row,
                    "conditional": [None] * len(questions),
                    "marginal": [None] * len(questions),
                    "reasoning": "",
                    "missing_indices": [
                        j for j in range(len(questions)) if j != focus_idx
                    ],
                    "prompt": prompt,
                    "response": "",
                    "cost": 0.0,
                    "error": str(e),
                }

    log.info(
        f"[{step_dir.name}] running {len(questions)} conditional prompts "
        f"(N={len(questions)})"
    )
    rows = await asyncio.gather(*(score_one(i) for i in range(len(questions))))
    total_cost = sum(r["cost"] for r in rows)
    return {
        "step_name": step_dir.name,
        "skipped": False,
        "model": model,
        "default_knowledge": parsed["default_knowledge"],
        "beliefs": parsed["beliefs"],
        "questions": questions,
        "matrix": rows,
        "total_cost": total_cost,
    }


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


VIEWER_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Conditional Uncertainty — __TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 0; }
  header { padding: 12px 16px; border-bottom: 1px solid #ccc;
           background: rgba(0,0,0,0.04); position: sticky; top: 0; z-index: 5; }
  h1 { font-size: 16px; margin: 0 0 6px 0; }
  .subtle { color: #666; font-size: 12px; }
  main { padding: 14px; }
  select { font-size: 14px; padding: 4px 8px; }
  .legend { display: inline-flex; gap: 8px; align-items: center; font-size: 12px;
            margin-left: 16px; }
  .legend .swatch { display: inline-block; width: 14px; height: 14px;
                    border: 1px solid #888; }
  table.heatmap { border-collapse: collapse; font-size: 12px;
                  font-variant-numeric: tabular-nums; }
  table.heatmap th, table.heatmap td { border: 1px solid #ddd; padding: 4px 6px;
                                       text-align: center; }
  table.heatmap th.q-label { text-align: left; max-width: 360px;
                             white-space: nowrap; overflow: hidden;
                             text-overflow: ellipsis; cursor: pointer; }
  table.heatmap th.col-num { width: 32px; }
  table.heatmap td.cell { width: 28px; height: 24px; cursor: pointer;
                          font-weight: 600; }
  table.heatmap td.diag { background: repeating-linear-gradient(45deg,
                          #ddd, #ddd 4px, #eee 4px, #eee 8px); color: #888; }
  table.heatmap td.missing { background: #f8e2e2; color: #777; }
  /* Uncertainty palette (1..5): green = certain, red = uncertain */
  table.heatmap td.cell.s1 { background: #4d9b59; color: white; }
  table.heatmap td.cell.s2 { background: #a6d96a; color: #103510; }
  table.heatmap td.cell.s3 { background: #ffffbf; color: #555; }
  table.heatmap td.cell.s4 { background: #fdae61; color: #4a2c0c; }
  table.heatmap td.cell.s5 { background: #d7191c; color: white; }
  /* Marginal-info palette (0..4): grey = no info, blue = high info gain */
  table.heatmap td.cell.m0 { background: #f0f0f0; color: #888; }
  table.heatmap td.cell.m1 { background: #c6dbef; color: #1a3552; }
  table.heatmap td.cell.m2 { background: #6baed6; color: white; }
  table.heatmap td.cell.m3 { background: #3182bd; color: white; }
  table.heatmap td.cell.m4 { background: #08519c; color: white; }
  table.heatmap tr.selected th, table.heatmap tr.selected td { outline: 2px solid #2a73f5; }

  .detail { margin-top: 16px; padding: 10px;
            border: 1px solid #bbb; border-radius: 6px;
            background: rgba(0,0,0,0.02); }
  .detail h3 { margin: 0 0 6px 0; font-size: 14px; }
  pre { white-space: pre-wrap; background: rgba(0,0,0,0.05);
        padding: 8px; border-radius: 4px; max-height: 60vh; overflow: auto;
        font-size: 12px; }
  .panel-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
               margin-top: 8px; }
  .panel-row > div { min-width: 0; }
  details { margin: 6px 0; }
  details > summary { cursor: pointer; }
  .step-summary { display: flex; gap: 14px; flex-wrap: wrap; align-items: baseline;
                  font-size: 12px; color: #555; margin: 6px 0; }
</style>
</head>
<body>
<header>
  <h1>Conditional uncertainty — <span id="title">__TITLE__</span></h1>
  <div class="subtle">
    Step: <select id="stepSelect"></select>
    &nbsp;&nbsp;Metric: <select id="metricSelect">
      <option value="marginal" selected>marginal info gain (uncond − cond)</option>
      <option value="conditional">conditional uncertainty</option>
      <option value="unconditional">unconditional uncertainty</option>
    </select>
    <span class="legend" id="legend"></span>
    <span id="stepCost" class="subtle"></span>
  </div>
  <div class="subtle" id="metricDesc">
    Rows = focus question (resolved). Columns = target question.
    Cell = marginal info gain — how much resolving the focus reduces uncertainty about the target.
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

const METRIC_INFO = {
  marginal: {
    label: "marginal info gain",
    desc: "Cell = how much resolving the focus reduces uncertainty about the target " +
          "(unconditional − conditional). 0 = focus tells us nothing new; 4 = focus fully resolves.",
    field: "marginal",
    legend: [
      ["#f0f0f0", "0"], ["#c6dbef", "1"], ["#6baed6", "2"],
      ["#3182bd", "3"], ["#08519c", "4"],
    ],
    cls: v => `cell m${Math.max(0, Math.min(4, v))}`,
  },
  conditional: {
    label: "conditional uncertainty",
    desc: "Cell = absolute uncertainty about the target AFTER assuming the focus is resolved. " +
          "1 = certain, 5 = totally uncertain.",
    field: "conditional",
    legend: [
      ["#4d9b59", "1"], ["#a6d96a", "2"], ["#ffffbf", "3"],
      ["#fdae61", "4"], ["#d7191c", "5"],
    ],
    cls: v => `cell s${Math.max(1, Math.min(5, v))}`,
  },
  unconditional: {
    label: "unconditional uncertainty",
    desc: "Cell = absolute uncertainty about the target given ONLY current beliefs " +
          "(focus is ignored). Each row shows the same values, since the focus doesn't enter. " +
          "1 = certain, 5 = totally uncertain.",
    field: "unconditional",
    legend: [
      ["#4d9b59", "1"], ["#a6d96a", "2"], ["#ffffbf", "3"],
      ["#fdae61", "4"], ["#d7191c", "5"],
    ],
    cls: v => `cell s${Math.max(1, Math.min(5, v))}`,
  },
};

function getCellValue(row, j, metric) {
  const info = METRIC_INFO[metric];
  const arr = row[info.field];
  if (!arr) {
    // Backward-compat: legacy data only has `scores` (= conditional)
    if (metric === "conditional") return row.scores ? row.scores[j] : null;
    return null;
  }
  return arr[j];
}

function renderLegend(metric) {
  const wrap = document.getElementById("legend");
  wrap.innerHTML = "";
  wrap.appendChild(el("span", null, `${METRIC_INFO[metric].label}:`));
  for (const [color, label] of METRIC_INFO[metric].legend) {
    wrap.appendChild(el("span", { class: "swatch", style: `background:${color}` }));
    wrap.appendChild(el("span", null, label));
  }
}

function renderStep(stepData, metric) {
  const content = document.getElementById("content");
  content.innerHTML = "";
  document.getElementById("metricDesc").textContent =
    "Rows = focus question. Columns = target question. " +
    METRIC_INFO[metric].desc;
  renderLegend(metric);

  if (stepData.skipped) {
    content.appendChild(el("div", null, `Skipped: ${stepData.reason}`));
    return;
  }

  const questions = stepData.questions;
  const matrix = stepData.matrix;
  const N = questions.length;
  const info = METRIC_INFO[metric];

  // Pre-compute mean of the chosen metric per row.
  const rowMeans = matrix.map(r => {
    const xs = [];
    for (let j = 0; j < N; j++) {
      const v = getCellValue(r, j, metric);
      if (typeof v === "number") xs.push(v);
    }
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
  });

  content.appendChild(
    el("div", { class: "step-summary" },
      el("span", null, `model: ${stepData.model || "?"}`),
      el("span", null, `${N} questions × ${N - 1} targets each`),
      el("span", null, `total cost: $${(stepData.total_cost || 0).toFixed(6)}`),
    )
  );
  content.appendChild(el("details", null,
    el("summary", null, "Current beliefs"),
    el("pre", null, stepData.beliefs || "(empty)")));
  content.appendChild(el("details", null,
    el("summary", null, "Default knowledge"),
    el("pre", null, stepData.default_knowledge || "(none)")));

  const table = el("table", { class: "heatmap" });
  const thead = el("thead");
  const headRow = el("tr", null, el("th", null, "focus \\ target"));
  for (let j = 0; j < N; j++) {
    headRow.appendChild(el("th", {
      class: "col-num", title: questions[j].question,
    }, `Q${j + 1}`));
  }
  headRow.appendChild(el("th", { class: "col-num" }, "mean"));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (let i = 0; i < N; i++) {
    const row = matrix[i];
    const tr = el("tr", { "data-focus": String(i) });
    const labelTh = el("th", {
      class: "q-label", title: questions[i].question,
      onclick: () => selectRow(i),
    }, `Q${i + 1}: ${questions[i].question}`);
    tr.appendChild(labelTh);
    for (let j = 0; j < N; j++) {
      if (j === i) {
        tr.appendChild(el("td", { class: "diag" }, "·"));
        continue;
      }
      const v = getCellValue(row, j, metric);
      if (v == null) {
        tr.appendChild(el("td", { class: "missing", title: "missing/parse error" }, "—"));
        continue;
      }
      // Tooltip shows all three values for context.
      const u = getCellValue(row, j, "unconditional");
      const c = getCellValue(row, j, "conditional");
      const m = getCellValue(row, j, "marginal");
      const tip =
        `focus Q${i + 1} → target Q${j + 1}\n` +
        `unconditional: ${u ?? "?"}\n` +
        `conditional:   ${c ?? "?"}\n` +
        `marginal gain: ${m ?? "?"}`;
      tr.appendChild(el("td", {
        class: info.cls(v),
        title: tip,
        onclick: () => selectRow(i),
      }, String(v)));
    }
    const rm = rowMeans[i];
    tr.appendChild(el("td",
      { class: "cell", style: "background:transparent;font-weight:400;color:#555;" },
      rm == null ? "—" : rm.toFixed(2)));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  content.appendChild(table);

  const detailWrap = el("div", { id: "detail-wrap" });
  content.appendChild(detailWrap);

  function selectRow(i) {
    for (const tr of tbody.querySelectorAll("tr")) {
      tr.classList.toggle("selected", tr.getAttribute("data-focus") === String(i));
    }
    const row = matrix[i];
    detailWrap.innerHTML = "";
    const card = el("div", { class: "detail" });
    card.appendChild(el("h3", null, `Focus Q${i + 1}: ${questions[i].question}`));
    if (row.error) {
      card.appendChild(el("div", { style: "color:#a00;" }, `error: ${row.error}`));
    }
    const numParsed = (() => {
      let n = 0;
      for (let j = 0; j < N; j++) {
        if (j === i) continue;
        if (getCellValue(row, j, metric) != null) n++;
      }
      return n;
    })();
    card.appendChild(el("div", { class: "subtle" },
      `cost: $${(row.cost || 0).toFixed(6)} · ` +
      `parsed (${metric}): ${numParsed} / ${N - 1}` +
      (row.missing_indices && row.missing_indices.length
        ? ` · missing: ${row.missing_indices.map(x => "Q" + (x + 1)).join(", ")}`
        : "")
    ));
    if (row.reasoning) {
      card.appendChild(el("details", null,
        el("summary", null, "reasoning"),
        el("pre", null, row.reasoning)));
    }
    card.appendChild(el("div", { class: "panel-row" },
      el("div", null,
        el("strong", null, "Prompt"),
        el("pre", null, row.prompt || "")),
      el("div", null,
        el("strong", null, "Response"),
        el("pre", null, row.response || ""))));
    detailWrap.appendChild(card);
  }

  if (N > 0) selectRow(0);
}

function currentMetric() {
  return document.getElementById("metricSelect").value;
}

function currentStepIdx() {
  return Number(document.getElementById("stepSelect").value);
}

function init() {
  const stepSel = document.getElementById("stepSelect");
  DATA.steps.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    const tag = s.skipped ? " (skipped)" :
      ` (${(s.questions || []).length}×${(s.questions || []).length - 1})`;
    opt.textContent = s.step_name + tag;
    stepSel.appendChild(opt);
  });
  const metricSel = document.getElementById("metricSelect");
  const rerender = () => renderStep(DATA.steps[currentStepIdx()], currentMetric());
  stepSel.addEventListener("change", rerender);
  metricSel.addEventListener("change", rerender);
  if (DATA.steps.length) rerender();
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--max-concurrent", type=int, default=32,
        help="global concurrency cap shared across ALL steps and ALL phases "
             "(unconditional + conditional) — every in-flight LLM call counts "
             "against this limit",
    )
    p.add_argument("--steps", default=None,
                   help="comma-separated step indices (e.g. '0,2'); default: all")
    p.add_argument("--overwrite", action="store_true",
                   help="re-run conditional pass even if conditional_uncertainty.json "
                        "exists; does NOT re-run unconditional pass")
    p.add_argument("--overwrite-unconditional", action="store_true",
                   help="also re-run the unconditional pass for each step "
                        "(overwrites uncertainty_scores.json)")
    p.add_argument(
        "--viewer-only",
        action="store_true",
        help="skip scoring; rewrite the HTML viewer from existing "
             "conditional_uncertainty.json files",
    )
    p.add_argument("--viewer-out", type=Path, default=None)
    args = p.parse_args()

    if not args.run_dir.is_dir():
        log.error(f"run_dir does not exist: {args.run_dir}")
        return 1

    if args.viewer_only:
        episode_dir = find_episode_dir(args.run_dir, args.episode)
        step_dirs = find_step_dirs(episode_dir)
        results: list[dict] = []
        for step_dir in step_dirs:
            f = step_dir / "conditional_uncertainty.json"
            if f.exists():
                results.append(json.loads(f.read_text()))
        if not results:
            log.error("No conditional_uncertainty.json files found")
            return 1
        viewer_path = args.viewer_out or (
            args.run_dir / "conditional_uncertainty_viewer.html"
        )
        title = f"{args.run_dir.name} / episode_{args.episode}"
        write_viewer(viewer_path, title, results)
        log.info(f"Wrote viewer: {viewer_path}")
        return 0

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
        # ONE global semaphore for every LLM call in the pipeline (unconditional
        # baselines + conditional focus prompts), across every step. This lets
        # all per-step work fan out in parallel and saturates the rate limit
        # without per-step serial gaps.
        sem = asyncio.Semaphore(max(1, args.max_concurrent))
        results_by_idx: dict[int, dict] = {}
        tasks: list[asyncio.Task] = []

        async def run_one(i: int, step_dir: Path) -> None:
            out_path = step_dir / "conditional_uncertainty.json"
            result = await score_step(
                step_dir=step_dir,
                model=model,
                temperature=args.temperature,
                sem=sem,
                ensure_unconditional=True,
                overwrite_unconditional=args.overwrite_unconditional,
            )
            out_path.write_text(json.dumps(result, indent=2))
            log.info(
                f"[{step_dir.name}] wrote {out_path.name} "
                f"(cost=${result.get('total_cost', 0):.6f})"
            )
            results_by_idx[i] = result

        for i, step_dir in enumerate(step_dirs):
            out_path = step_dir / "conditional_uncertainty.json"
            if selected_step_indices is not None and i not in selected_step_indices:
                log.info(f"Skipping {step_dir.name} (not in --steps)")
                if out_path.exists():
                    results_by_idx[i] = json.loads(out_path.read_text())
                continue
            if out_path.exists() and not args.overwrite:
                log.info(f"[{step_dir.name}] {out_path.name} exists — reusing")
                results_by_idx[i] = json.loads(out_path.read_text())
                continue
            tasks.append(asyncio.create_task(run_one(i, step_dir)))

        if tasks:
            log.info(
                f"Dispatching {len(tasks)} step(s) in parallel "
                f"(max_concurrent={args.max_concurrent} LLM calls)"
            )
            await asyncio.gather(*tasks)
        return [results_by_idx[i] for i in sorted(results_by_idx)]

    results = asyncio.run(run_all())

    viewer_path = args.viewer_out or (
        args.run_dir / "conditional_uncertainty_viewer.html"
    )
    title = f"{args.run_dir.name} / episode_{args.episode}"
    write_viewer(viewer_path, title, results)
    log.info(f"Wrote viewer: {viewer_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
