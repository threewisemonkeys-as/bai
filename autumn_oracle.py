"""Oracle for AutumnBench DSL programs.

Direct LLM-based question answering against the Autumn DSL source, so
question/experiment selection strategies in stepwise_eb_learn can be
evaluated without running the game. The oracle reads the .sexp program
plus the Autumn DSL standard-library reference, and answers yes/no/unknown
questions about the program.

Layout under data/oracle/:
    autumn_dsl_docs.md              shared DSL reference
    <task>/summary.md               one-time NL summary of the program
    <task>/ground_truth_beliefs.md  LLM draft + human edit
    <task>/answers.jsonl            per-question oracle cache (one record per line)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Literal, TypedDict

import litellm

from explore.improve import _get_response_cost
from llm_utils import build_llm_input, extract_llm_response_text, extract_xml_key

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DSL_DOCS_PATH = REPO_ROOT / "data" / "oracle" / "autumn_dsl_docs.md"
PROGRAMS_DIR = (
    REPO_ROOT
    / "MARAProtocol"
    / "python_examples"
    / "autumnbench"
    / "example_benchmark"
    / "programs"
)
ORACLE_DIR = REPO_ROOT / "data" / "oracle"

STRONG_MODEL = "openrouter/openai/gpt-5.5-pro"


# ---------------------------------------------------------------------------
# Paths and file loading
# ---------------------------------------------------------------------------

def task_program_path(task: str) -> Path:
    return PROGRAMS_DIR / f"{task}.sexp"


def task_oracle_dir(task: str) -> Path:
    p = ORACLE_DIR / task
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_dsl_docs() -> str:
    if not DSL_DOCS_PATH.exists():
        raise FileNotFoundError(
            f"Autumn DSL docs not found at {DSL_DOCS_PATH}. "
            "Re-clone from the Autumn.cpp wiki to data/oracle/autumn_dsl_docs.md."
        )
    return DSL_DOCS_PATH.read_text()


def load_program_source(task: str) -> str:
    path = task_program_path(task)
    if not path.exists():
        raise FileNotFoundError(f"Autumn DSL program not found at {path}")
    return path.read_text()


def _normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _question_cache_key(q: str) -> str:
    return hashlib.sha256(_normalize_question(q).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Low-level LLM call
# ---------------------------------------------------------------------------

async def _strong_llm_call(
    prompt: str,
    *,
    model: str = STRONG_MODEL,
    temperature: float | None = None,
) -> tuple[str, float]:
    input_data = build_llm_input(prompt)
    kwargs: dict = {"model": model, "input": input_data, "num_retries": 5}
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = await asyncio.to_thread(litellm.responses, **kwargs)
    text = extract_llm_response_text(response)
    model_id = model.split("/", 1)[1] if "/" in model else model
    cost = _get_response_cost(response, model_id)
    return text, cost


# ---------------------------------------------------------------------------
# Step 1: summarize_program
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """You will be given an AutumnBench game program written in the Autumn DSL.
Your job is to produce a natural-language summary of the game so that a future reader can understand its rules without parsing the source.

For background, here is the Autumn standard-library reference:
<autumn_dsl_docs>
{dsl_docs}
</autumn_dsl_docs>

Here is the game program:
<program>
{program_src}
</program>

Produce a thorough, organized NL summary covering:
1. **Objects**: every object defined, its fields, default appearance, and what each field controls.
2. **Initial state**: where each object starts and what state it starts in.
3. **Dynamics**: every `on (...)` handler and every `initnext` clause — under what condition does state change, and how?
4. **Player actions**: the full set of input events the program responds to (click on which objects, arrow keys, etc.) and what each does.
5. **Edge cases / boundaries**: any clamping, modular arithmetic, conditions guarding state changes (e.g. "gas cannot go below 0").
6. **Win/lose / progression**: any reward, terminal, or progression signal the program implicitly defines.

Be precise. Quote DSL expressions inline when helpful. Do not invent rules the program does not have.

Write the summary inside <summary>...</summary> tags."""


async def summarize_program(
    task: str,
    *,
    model: str = STRONG_MODEL,
    force: bool = False,
) -> tuple[str, float]:
    out_path = task_oracle_dir(task) / "summary.md"
    if out_path.exists() and not force:
        return out_path.read_text(), 0.0
    prompt = _SUMMARY_PROMPT.format(
        dsl_docs=load_dsl_docs(),
        program_src=load_program_source(task),
    )
    text, cost = await _strong_llm_call(prompt, model=model)
    summary = (extract_xml_key(text, "summary") or text).strip()
    out_path.write_text(summary + "\n")
    return summary, cost


# ---------------------------------------------------------------------------
# Step 2: derive_ground_truth_beliefs
# ---------------------------------------------------------------------------

_GT_BELIEFS_PROMPT = """You are establishing the ground-truth set of beliefs about an AutumnBench game.
These beliefs will be used to evaluate whether a learning agent's discovered beliefs converge to the truth.

For background, here is the Autumn standard-library reference:
<autumn_dsl_docs>
{dsl_docs}
</autumn_dsl_docs>

Here is the program:
<program>
{program_src}
</program>

Here is a natural-language summary of the program:
<summary>
{summary}
</summary>

Enumerate a comprehensive set of atomic beliefs that an ideal learning agent would converge to about this game. Each belief should be:
- **Atomic**: one fact per bullet, not compound. Split conjunctions.
- **Operational**: phrased as something the agent could test by interacting with the game, not abstract design intent.
- **Specific**: name the object, action, or state involved.
- **True per the program source** — not a guess, not generic game-design wisdom.

Cover at least: each object's role, what each player action does, every conditional rule (especially boundary/guard conditions), and any non-obvious dynamics (e.g. things that auto-update each tick).

Format your answer as a markdown bullet list (one `- ` line per belief) inside <beliefs>...</beliefs> tags. Aim for completeness over conciseness — typically 15-40 beliefs depending on game complexity."""


_GT_HEADER = (
    "<!--\n"
    "Ground-truth beliefs for AutumnBench task '{task}'.\n"
    "Drafted by an LLM; REVIEW AND EDIT before relying on this file.\n"
    "Lines starting with `- ` are parsed as individual beliefs by\n"
    "judge_beliefs_against_gt. Everything else is treated as commentary.\n"
    "-->\n\n"
)


async def derive_ground_truth_beliefs(
    task: str,
    *,
    model: str = STRONG_MODEL,
    force: bool = False,
) -> tuple[str, float]:
    out_path = task_oracle_dir(task) / "ground_truth_beliefs.md"
    if out_path.exists() and not force:
        return out_path.read_text(), 0.0
    summary, summary_cost = await summarize_program(task, model=model, force=force)
    prompt = _GT_BELIEFS_PROMPT.format(
        dsl_docs=load_dsl_docs(),
        program_src=load_program_source(task),
        summary=summary,
    )
    text, cost = await _strong_llm_call(prompt, model=model)
    beliefs = (extract_xml_key(text, "beliefs") or text).strip()
    out_path.write_text(_GT_HEADER.format(task=task) + beliefs + "\n")
    return out_path.read_text(), cost + summary_cost


# ---------------------------------------------------------------------------
# Step 3: answer_question
# ---------------------------------------------------------------------------

class OracleAnswer(TypedDict):
    answer: Literal["yes", "no", "unknown"]
    rationale: str
    cost: float
    cached: bool
    raw: str
    prompt: str


_ANSWER_PROMPT = """You are an oracle answering yes/no questions about an AutumnBench game.

Background — the Autumn standard library:
<autumn_dsl_docs>
{dsl_docs}
</autumn_dsl_docs>

Game program (this is ground truth — every rule is here):
<program>
{program_src}
</program>

Natural-language summary of the program:
<summary>
{summary}
</summary>

Question:
<question>{question}</question>

Determine the answer by direct reference to the program. Allowed answers:
- **yes** — the program clearly implies the answer is yes.
- **no**  — the program clearly implies the answer is no.
- **unknown** — the question is not decidable from the program alone (e.g. asks about something the program does not model, or is ambiguous).

Reason concisely. Cite specific DSL constructs (object name, on-clause, etc.) when relevant.

Respond in this exact format:
<rationale>1-3 sentences referencing the program</rationale>
<answer>yes|no|unknown</answer>"""


def _load_answers_cache(task: str) -> dict[str, dict]:
    path = task_oracle_dir(task) / "answers.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "key" in rec:
                out[rec["key"]] = rec
    return out


def _append_answer_cache(task: str, record: dict) -> None:
    path = task_oracle_dir(task) / "answers.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _normalize_oracle_label(raw: str) -> Literal["yes", "no", "unknown"]:
    r = raw.strip().lower()
    if r in ("yes", "no", "unknown"):
        return r  # type: ignore[return-value]
    if r.startswith("y"):
        return "yes"
    if r.startswith("n"):
        return "no"
    return "unknown"


async def answer_question(
    task: str,
    question: str,
    *,
    model: str = STRONG_MODEL,
    semaphore: asyncio.Semaphore | None = None,
    use_cache: bool = True,
    program_src: str | None = None,
    summary: str | None = None,
    dsl_docs: str | None = None,
) -> OracleAnswer:
    key = _question_cache_key(question)
    if use_cache:
        cached = _load_answers_cache(task).get(key)
        if cached:
            return OracleAnswer(
                answer=_normalize_oracle_label(cached["answer"]),
                rationale=cached.get("rationale", ""),
                cost=0.0,
                cached=True,
                raw=cached.get("raw", ""),
                prompt=cached.get("prompt", ""),
            )

    if program_src is None:
        program_src = load_program_source(task)
    if dsl_docs is None:
        dsl_docs = load_dsl_docs()
    if summary is None:
        summary_path = task_oracle_dir(task) / "summary.md"
        if summary_path.exists():
            summary = summary_path.read_text()
        else:
            summary, _ = await summarize_program(task, model=model)

    prompt = _ANSWER_PROMPT.format(
        dsl_docs=dsl_docs,
        program_src=program_src,
        summary=summary,
        question=question.strip(),
    )

    async def _call() -> tuple[str, float]:
        return await _strong_llm_call(prompt, model=model)

    if semaphore is not None:
        async with semaphore:
            text, cost = await _call()
    else:
        text, cost = await _call()

    rationale = (extract_xml_key(text, "rationale") or "").strip()
    label = _normalize_oracle_label(extract_xml_key(text, "answer") or "")
    record = {
        "key": key,
        "question": question.strip(),
        "answer": label,
        "rationale": rationale,
        "raw": text,
        "model": model,
    }
    if use_cache:
        _append_answer_cache(task, record)
    return OracleAnswer(
        answer=label,
        rationale=rationale,
        cost=cost,
        cached=False,
        raw=text,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Belief scoring against summary (no ground truth required)
# ---------------------------------------------------------------------------

_SUMMARY_SCORE_PROMPT = """You are scoring a learning agent's beliefs about an AutumnBench game on a 0-10 scale, using the program and its NL summary as ground truth.

Game program:
<program>
{program_src}
</program>

Natural-language summary of the program (use this as the rubric — the agent should ideally end up believing what the summary says):
<summary>
{summary}
</summary>

Agent's current beliefs:
<agent_beliefs>
{agent_beliefs}
</agent_beliefs>

Rate the beliefs on a 0-10 integer scale:
- 0  — beliefs are absent or entirely unrelated to the game.
- 3  — beliefs touch on the game but are mostly vague or wrong.
- 5  — beliefs capture some real mechanics but miss many key dynamics or contain notable errors.
- 7  — beliefs capture most of the important dynamics correctly, with minor gaps or imprecisions.
- 10 — beliefs comprehensively and accurately describe the game's objects, dynamics, player actions, and edge cases.

Penalize *incorrect* claims about the game heavily — a wrong belief is worse than a missing one. Reward beliefs that are specific and operational over generic platitudes.

Respond in this exact format:
<rationale>2-4 sentences: what the beliefs got right, what they got wrong, what they missed.</rationale>
<score>integer 0-10</score>"""


async def score_beliefs_against_summary(
    task: str,
    agent_beliefs_text: str,
    *,
    model: str = STRONG_MODEL,
    summary: str | None = None,
    program_src: str | None = None,
) -> dict:
    """Score agent beliefs 0-10 against the program + summary.

    Returns: {"score": int, "rationale": str, "cost": float, "raw": str}
    Score is -1 if the LLM response could not be parsed.
    """
    if program_src is None:
        program_src = load_program_source(task)
    if summary is None:
        summary_path = task_oracle_dir(task) / "summary.md"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No summary at {summary_path}. "
                f"Run `uv run scripts/bootstrap_oracle.py {task}` first."
            )
        summary = summary_path.read_text()

    prompt = _SUMMARY_SCORE_PROMPT.format(
        program_src=program_src,
        summary=summary,
        agent_beliefs=agent_beliefs_text.strip() or "(empty)",
    )
    text, cost = await _strong_llm_call(prompt, model=model)

    rationale = (extract_xml_key(text, "rationale") or "").strip()
    score_raw = (extract_xml_key(text, "score") or "").strip()
    m = re.search(r"\d+", score_raw)
    score = int(m.group(0)) if m else -1
    if 0 <= score <= 10:
        pass
    elif score > 10:
        score = 10
    elif score < 0 and m is not None:
        score = 0

    return {
        "score": score,
        "rationale": rationale,
        "cost": cost,
        "raw": text,
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Step 4: judge_beliefs_against_gt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are evaluating how well a learning agent's current beliefs about an AutumnBench game match a ground-truth set of beliefs.

Ground-truth beliefs (authoritative):
<ground_truth>
{gt_beliefs}
</ground_truth>

Agent's current beliefs:
<agent_beliefs>
{agent_beliefs}
</agent_beliefs>

For each ground-truth belief, decide whether the agent's beliefs:
- **entailed** — the agent clearly believes this (explicit statement, or trivial paraphrase).
- **contradicted** — the agent believes something incompatible with this.
- **missing** — the agent's beliefs are silent on this point.

Be strict about entailment — paraphrases count, but vague generalities do not.

Respond as a JSON array inside <verdicts>...</verdicts>. Emit one object per ground-truth belief, in the same order they appear, with these exact keys:
{{"gt": "<the GT belief text>", "verdict": "entailed|contradicted|missing", "evidence": "<short quote or paraphrase from agent beliefs, or empty string>"}}"""


def _parse_gt_belief_lines(gt_text: str) -> list[str]:
    out = []
    for line in gt_text.splitlines():
        s = line.strip()
        if s.startswith(("- ", "* ")):
            out.append(s[2:].strip())
    return out


async def judge_beliefs_against_gt(
    task: str,
    agent_beliefs_text: str,
    *,
    model: str = STRONG_MODEL,
    gt_beliefs_text: str | None = None,
) -> dict:
    """Score agent beliefs against ground-truth beliefs.

    Recall:      entailed / n_ground_truth
    Precision:   entailed / (entailed + contradicted)  (only over GT beliefs the agent took a position on)
    F1:          harmonic mean of the above
    """
    if gt_beliefs_text is None:
        gt_path = task_oracle_dir(task) / "ground_truth_beliefs.md"
        if not gt_path.exists():
            raise FileNotFoundError(
                f"No ground-truth beliefs at {gt_path}. "
                f"Run `uv run scripts/bootstrap_oracle.py {task}` first."
            )
        gt_beliefs_text = gt_path.read_text()

    gt_beliefs = _parse_gt_belief_lines(gt_beliefs_text)
    if not gt_beliefs:
        raise ValueError(
            f"No bulleted beliefs parsed from ground-truth file for task '{task}'."
        )

    prompt = _JUDGE_PROMPT.format(
        gt_beliefs="\n".join(f"- {b}" for b in gt_beliefs),
        agent_beliefs=agent_beliefs_text.strip(),
    )
    text, cost = await _strong_llm_call(prompt, model=model)

    verdicts_blob = extract_xml_key(text, "verdicts") or text
    try:
        verdicts = json.loads(verdicts_blob.strip())
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", verdicts_blob, re.DOTALL)
        verdicts = json.loads(match.group(0)) if match else []

    n_gt = len(gt_beliefs)
    n_entailed = sum(1 for v in verdicts if v.get("verdict") == "entailed")
    n_contradicted = sum(1 for v in verdicts if v.get("verdict") == "contradicted")
    n_missing = n_gt - n_entailed - n_contradicted

    recall = n_entailed / n_gt if n_gt else 0.0
    n_positions = n_entailed + n_contradicted
    precision = n_entailed / n_positions if n_positions else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "per_belief": verdicts,
        "n_ground_truth": n_gt,
        "n_entailed": n_entailed,
        "n_contradicted": n_contradicted,
        "n_missing": n_missing,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost": cost,
        "raw": text,
    }
