"""Theory-driven exploration (Plan B: theory-seeded crux questions + theory-entropy scoring).

This module implements the building blocks for using *competing world-models*
("theories") to drive which question the agent should investigate next, as an
alternative to the B-difference value-of-information scorer in
``question_scoring.py``.

Pipeline (stateless, regenerated at each selection point):

  1. ``generate_theories``      -> M ranked <world_knowledge>-style theories, each
                                   with a prior weight derived from its rank.
  2. ``generate_crux_questions``-> binary questions designed to split the theories
                                   (added to the question bank before scoring).
  3. ``score_questions_theory_entropy`` -> scores each unanswered question by the
                                   mutual information I(answer; theory) under the
                                   theory posterior: how much learning its YES/NO
                                   answer would collapse the distribution over
                                   theories. High score == good discriminator.

All LLM calls route through ``mixed_improve._llm_call`` (mock-mode / logging /
cost parity) and answer prediction reuses ``question_scoring._predict_answer_vector``
so a theory's predicted answers are produced exactly like belief-conditioned
predictions elsewhere in the codebase.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from llm_utils import extract_xml_key
from explore.mixed_improve import _llm_call

if TYPE_CHECKING:
    from explore.stepwise_eb_learn_improve import EBQAPair


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Theory:
    """A candidate world-model: a <world_knowledge>-style block + prior weight."""

    world_knowledge: str
    rationale: str = ""
    rank: int = 0  # 1 = most likely (as ranked by the generator)
    likelihood: str = ""  # raw likelihood phrase from the generator, if any
    weight: float = 0.0  # normalized prior probability over the ensemble


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _extract_block(text: str, key: str) -> str:
    """Extract <key>...</key>, tolerating a missing/garbled closing tag.

    The generator sometimes emits malformed closing tags (e.g. ``</rationate>``
    or ``</ration-ale>``), so when the exact closing tag is absent we fall back
    to reading up to the next opening tag or end of segment.
    """
    exact = extract_xml_key(text, key)
    if exact is not None:
        return exact.strip()
    open_tag = f"<{key}>"
    start = text.find(open_tag)
    if start == -1:
        return ""
    rest = text[start + len(open_tag) :]
    # Cut at the next opening tag of any kind, or a closing tag.
    m = re.search(r"<\s*/?\s*[a-zA-Z]", rest)
    return (rest[: m.start()] if m else rest).strip()


def parse_theories(text: str) -> list[Theory]:
    """Parse <theory rank=.. likelihood=..> blocks from a generator response.

    Lenient: keyed off each ``<theory`` opening tag, reading until the next
    ``<theory`` opening (or end of text), and tolerant of malformed inner
    closing tags.
    """
    theories: list[Theory] = []
    openings = list(re.finditer(r"<theory\b([^>]*)>", text, re.IGNORECASE))
    for idx, match in enumerate(openings):
        attrs = match.group(1)
        end = openings[idx + 1].start() if idx + 1 < len(openings) else len(text)
        body = text[match.end() : end]

        rank_m = re.search(r'rank\s*=\s*["\']?\s*(\d+)', attrs, re.IGNORECASE)
        like_m = re.search(
            r'likelihood\s*=\s*["\']([^"\']*)["\']', attrs, re.IGNORECASE
        )
        rank = int(rank_m.group(1)) if rank_m else (idx + 1)

        # Accept both <content> (current generation prompts) and
        # <world_knowledge> (older saved dumps / legacy responses).
        wk = _extract_block(body, "content") or _extract_block(body, "world_knowledge")
        if not wk:
            continue  # a theory without a body is unusable
        theories.append(
            Theory(
                world_knowledge=wk,
                rationale=_extract_block(body, "rationale"),
                rank=rank,
                likelihood=like_m.group(1).strip() if like_m else "",
            )
        )
    # Stable order by rank (generator order is already most->least likely).
    theories.sort(key=lambda t: t.rank)
    return theories


def parse_crux_questions(text: str) -> list[str]:
    """Parse <q>...</q> entries from a crux-question response."""
    questions: list[str] = []
    for m in re.finditer(r"<q\b[^>]*>(.*?)</q>", text, re.IGNORECASE | re.DOTALL):
        q = m.group(1).strip()
        # Strip a possible nested <question> wrapper.
        inner = extract_xml_key(q, "question")
        if inner is not None:
            q = inner.strip()
        if q:
            questions.append(q)
    return questions


# ---------------------------------------------------------------------------
# Prior weights
# ---------------------------------------------------------------------------


def assign_rank_weights(theories: list[Theory], decay: float = 0.6) -> None:
    """Assign normalized prior weights by rank: w_r proportional to decay^(rank-1).

    Mutates ``theories`` in place. ``decay`` in (0, 1]; smaller == more mass on
    the top-ranked theory. decay == 1.0 yields a uniform prior.
    """
    if not theories:
        return
    raw = [decay ** max(0, t.rank - 1) for t in theories]
    total = sum(raw) or 1.0
    for t, r in zip(theories, raw):
        t.weight = r / total


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _dk_section(default_knowledge: str) -> str:
    if not default_knowledge:
        return ""
    return f"\n=== DEFAULT KNOWLEDGE ===\n{default_knowledge}\n=== END DEFAULT KNOWLEDGE ===\n"


async def generate_theories(
    config: DictConfig,
    *,
    beliefs: str,
    default_knowledge: str,
    steps_context: str = "",
    current_observation: str | None = None,
    current_image=None,
    steps_context_images: list | None = None,
    num_theories: int = 5,
    decay: float = 0.6,
    goal: str = "",
    goal_aware: bool = True,
    seed_questions: list[str] | None = None,
) -> tuple[list[Theory], float, dict]:
    """Generate ``num_theories`` ranked theories of how the environment works.

    When ``goal`` is provided AND ``goal_aware`` is True, each theory is required
    to be goal-aware: it must state a hypothesis about the win/progress condition
    AND the strategy (what to do, and why) for achieving it — not just local
    mechanics. With ``goal_aware=False`` (e.g. dynamics-only exploration) no
    win/progress framing is emitted regardless of ``goal``. Returns
    (theories, cost, log). Theories carry normalized prior weights.
    """
    _show_goal = bool(goal_aware and goal and goal.strip())
    images: list = []
    if steps_context_images:
        images.extend(steps_context_images)
    if current_image is not None:
        images.append(current_image)

    obs_section = ""
    if current_observation:
        obs_section = f"\n=== CURRENT STATE (text) ===\n{current_observation}\n=== END CURRENT STATE ===\n"
    history_section = ""
    if steps_context:
        history_section = f"\n=== RECENT HISTORY OF STATES AND ACTIONS ===\n{steps_context}\n=== END RECENT HISTORY ===\n"
    image_note = (
        " The current game state is shown in the attached image(s)." if images else ""
    )
    goal_section = (
        f"\n=== GOAL / WIN CONDITION ===\n{goal.strip()}\n=== END GOAL ===\n"
        if _show_goal else ""
    )
    goal_requirement = (
        "\n\nEach theory MUST be GOAL-AWARE — not just a description of local mechanics. "
        "Each theory must explicitly state:\n"
        "  (a) WIN/PROGRESS CONDITION: what it hypothesizes is required to make progress / "
        "score / complete the level;\n"
        "  (b) STRATEGY: the concrete sequence or kind of actions that should advance the goal "
        "under this theory, and why;\n"
        "  (c) the minimal mechanics the strategy relies on.\n"
        "The theories should DISAGREE about what advances the goal, so that acting on them and "
        "observing whether progress is made will tell you which strategy is right."
        if _show_goal else ""
    )
    seed_section = ""
    seed_requirement = ""
    _seeds = [q.strip() for q in (seed_questions or []) if q and q.strip()]
    if _seeds:
        qlist = "\n".join(f"- {q}" for q in _seeds)
        seed_section = (
            "\n=== OPEN QUESTIONS PROBING POSSIBLY-UNMODELED MECHANISMS ===\n"
            "These open questions hypothesize mechanisms that your current world knowledge "
            "may NOT yet account for:\n"
            f"{qlist}\n"
            "=== END OPEN QUESTIONS ===\n"
        )
        seed_requirement = (
            "\n\nAt least one theory MUST take a hypothesized mechanism from the OPEN "
            "QUESTIONS as TRUE — assume that mechanism is real, and explain concretely how it "
            "would change how the game behaves. Do NOT dismiss such a mechanism merely "
            "because the current world knowledge does not mention it; the purpose is to give "
            "the ensemble a member that PREDICTS YES for that question, so that acting can "
            "test it."
        )

    task_objective = (
        "how this game works and what is required to make progress / complete levels. Each "
        "theory must be a self-contained explanation about the game's mechanics and win condition"
        if _show_goal else
        "how this game works. Each theory must be a self-contained explanation about the game's "
        "mechanics"
    )

    prompt = f"""You are an agent trying to understand how a game works. Below is the fixed \
information about the environment, everything you currently believe, and the current state.
{_dk_section(default_knowledge)}{goal_section}
=== CURRENT WORLD KNOWLEDGE (what you believe so far) ===
{beliefs.strip() if beliefs and beliefs.strip() else "(no world knowledge accumulated yet)"}
=== END CURRENT WORLD KNOWLEDGE ==={history_section}{obs_section}{seed_section}
{image_note}

Your task: propose exactly {num_theories} DISTINCT theories of {task_objective}. The theories \
should be genuinely different hypotheses (not rephrasings of each other), each consistent with \
the default knowledge and the current state, though they may go beyond or contradict the \
current world knowledge where you think it may be wrong.{goal_requirement}{seed_requirement}

Order the theories from MOST likely to LEAST likely.

Respond in exactly this format:
<theories>
<theory rank="1" likelihood="<rough probability or short phrase>">
<content>
- ...
</content>
<rationale>Why ranked here.</rationale>
</theory>
... (ranks 2..{num_theories})
</theories>"""

    text, cost = await _llm_call(config, prompt, images=images or None)
    theories = parse_theories(text)[:num_theories]
    assign_rank_weights(theories, decay=decay)
    log = {
        "kind": "generate_theories",
        "num_requested": num_theories,
        "num_parsed": len(theories),
        "decay": decay,
        "prompt": prompt,
        "response": text,
        "theories": [
            {
                "rank": t.rank,
                "likelihood": t.likelihood,
                "weight": t.weight,
                "world_knowledge": t.world_knowledge,
                "rationale": t.rationale,
            }
            for t in theories
        ],
    }
    if not theories:
        logging.warning("[theory_exploration] generate_theories parsed 0 theories")
    return theories, cost, log


async def generate_crux_questions(
    config: DictConfig,
    *,
    theories: list[Theory],
    beliefs: str,
    default_knowledge: str,
    num_crux: int = 5,
) -> tuple[list[str], float, dict]:
    """Generate binary questions whose answers best discriminate the theories."""
    if len(theories) < 2:
        return (
            [],
            0.0,
            {
                "kind": "generate_crux_questions",
                "note": "need >=2 theories",
                "questions": [],
            },
        )

    theory_blocks = "\n\n".join(
        f"THEORY {t.rank} (prior {t.weight:.2f}):\n{t.world_knowledge.strip()}"
        for t in theories
    )
    prompt = f"""You are designing experiments to distinguish between competing theories of how a game works.
{_dk_section(default_knowledge)}
=== CURRENT WORLD KNOWLEDGE ===
{beliefs.strip() if beliefs and beliefs.strip() else "(none yet)"}
=== END CURRENT WORLD KNOWLEDGE ===

=== COMPETING THEORIES ===
{theory_blocks}
=== END COMPETING THEORIES ===

Propose up to {num_crux} CRUX questions: binary (YES/NO) questions about the game whose answer \
would most sharply distinguish between the theories above — i.e. the theories disagree on the \
answer. Each question must be answerable by direct observation after a concrete action in the \
game, and phrased so a single YES/NO settles it. Prefer questions that split the higher-prior \
theories. Do NOT ask questions that all theories would answer the same way.

Respond in exactly this format:
<crux_questions>
<q>First binary question?</q>
<q>Second binary question?</q>
...
</crux_questions>"""

    text, cost = await _llm_call(config, prompt)
    questions = parse_crux_questions(text)[:num_crux]
    log = {
        "kind": "generate_crux_questions",
        "num_requested": num_crux,
        "num_parsed": len(questions),
        "prompt": prompt,
        "response": text,
        "questions": questions,
    }
    return questions, cost, log


# ---------------------------------------------------------------------------
# Mutual-information scoring over the theory posterior
# ---------------------------------------------------------------------------


def _entropy_bits(weights: list[float]) -> float:
    """Shannon entropy (bits) of a (not necessarily normalized) weight list."""
    total = sum(weights)
    if total <= 0:
        return 0.0
    h = 0.0
    for w in weights:
        p = w / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def mutual_information(weights: list[float], p_yes_per_theory: list[float]) -> float:
    """I(answer ; theory) in bits for one binary question.

    weights: prior weight of each theory (will be normalized).
    p_yes_per_theory: P(answer = YES | theory_k), in [0, 1] (0.5 == theory
        is agnostic / UNKNOWN under that theory).

    MI = H(theory) - E_{answer}[ H(theory | answer) ].
    """
    assert len(weights) == len(p_yes_per_theory)
    total = sum(weights) or 1.0
    w = [x / total for x in weights]

    h_prior = _entropy_bits(w)
    # Marginal P(YES) and the two posterior weight vectors.
    yes_weights = [wk * pk for wk, pk in zip(w, p_yes_per_theory)]
    no_weights = [wk * (1.0 - pk) for wk, pk in zip(w, p_yes_per_theory)]
    p_yes = sum(yes_weights)
    p_no = sum(no_weights)

    h_post = p_yes * _entropy_bits(yes_weights) + p_no * _entropy_bits(no_weights)
    return max(0.0, h_prior - h_post)


async def score_questions_theory_entropy(
    config: DictConfig,
    *,
    theories: list[Theory],
    qa_pairs: list["EBQAPair"],
    candidate_indices: list[int] | None = None,
    default_knowledge: str = "",
    max_concurrent: int = 8,
) -> tuple[dict[int, float], float, dict]:
    """Score unanswered questions by expected information gain over theories.

    For each theory we predict its YES/NO/UNKNOWN answer to every unanswered
    question (one prediction call per theory, reusing
    ``question_scoring._predict_answer_vector``), then score each candidate
    question by the mutual information between its answer and the theory
    identity under the theories' prior weights.

    Returns: (scores_by_index, total_cost, log). Scores are in bits (>= 0);
    higher == better discriminator. Mirrors the return shape of
    ``question_scoring.score_questions_b_diff`` so it can drop into the same
    call sites.
    """
    import asyncio

    from explore.question_scoring import _predict_answer_vector  # lazy: avoid import cycle

    unanswered_indices = [i for i, qa in enumerate(qa_pairs) if qa.answer is None]
    if candidate_indices is None:
        score_indices = list(unanswered_indices)
    else:
        unanswered_set = set(unanswered_indices)
        seen: set[int] = set()
        score_indices = []
        for i in candidate_indices:
            if i in unanswered_set and i not in seen:
                seen.add(i)
                score_indices.append(i)

    log: dict = {
        "kind": "score_questions_theory_entropy",
        "num_theories": len(theories),
        "default_knowledge_length": len(default_knowledge),
        "num_unanswered_scored": len(score_indices),
        "theory_weights": [t.weight for t in theories],
        "per_question": [],
    }

    if not score_indices:
        log["note"] = "no unanswered candidate questions to score"
        return {}, 0.0, log
    if len(theories) < 2:
        log["note"] = "fewer than 2 theories — cannot discriminate; scores = 0"
        return {i: 0.0 for i in score_indices}, 0.0, log

    # Predict every theory's answers over the candidate questions in one call
    # per theory. We predict over the candidate set (the questions we want to
    # rank); MI is computed per candidate independently.
    questions = [qa_pairs[i].question for i in score_indices]
    weights = [t.weight for t in theories]

    sem = asyncio.Semaphore(max(1, max_concurrent))

    async def predict_for_theory(t: Theory):
        async with sem:
            answers, cost, prompt, response, _artifact = await _predict_answer_vector(
                config, t.world_knowledge, questions, default_knowledge
            )
        return answers, cost, prompt, response

    results = await asyncio.gather(*(predict_for_theory(t) for t in theories))

    total_cost = 0.0
    # matrix[theory_k][candidate_j] = P(YES | theory_k) in {1.0, 0.0, 0.5}
    matrix: list[list[float]] = []
    theory_logs = []
    for t, (answers, cost, prompt, response) in zip(theories, results):
        total_cost += cost
        # _predict_answer_vector already returns 1.0/0.0/0.5 floats.
        matrix.append(list(answers))
        theory_logs.append(
            {
                "rank": t.rank,
                "weight": t.weight,
                "answers": list(answers),
                "prompt": prompt,
                "response": response,
            }
        )
    log["theory_predictions"] = theory_logs

    scores: dict[int, float] = {}
    for j, qa_idx in enumerate(score_indices):
        p_yes_per_theory = [matrix[k][j] for k in range(len(theories))]
        mi = mutual_information(weights, p_yes_per_theory)
        scores[qa_idx] = mi
        log["per_question"].append(
            {
                "idx": qa_idx,
                "question": qa_pairs[qa_idx].question,
                "source_step": qa_pairs[qa_idx].source_step,
                "score": mi,
                "p_yes_per_theory": p_yes_per_theory,
            }
        )

    log["per_question"].sort(key=lambda d: d["score"], reverse=True)
    log["total_cost"] = total_cost
    logging.info(
        f"[theory_exploration] scored {len(scores)} questions over "
        f"{len(theories)} theories; cost=${total_cost:.6f}"
    )
    return scores, total_cost, log
