"""Stepwise EB-learning: experiment-driven per-step learning.

Like stepwise B-learning but more experiment-driven: the agent generates
questions about the environment, designs experiments to answer them, and
updates Q from trajectory evidence.  No critical moments — improvement is
via beliefs/perception (Tracks 1a, 1b) and QA (Track 2) only.
"""

import asyncio
import csv
import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from balrog.agents import AgentFactory
from balrog.client import (
    set_mock_mode as set_client_mock_mode,
    set_mock_action_provider,
)
from balrog.environments import make_env
from balrog.utils import get_unique_seed

from explore import get_default_knowledge, override_temperature, evolve_logger
from mixed_improve import (
    QAPair,
    _llm_call,
    _run_perception_on_observation,
    set_mock_mode,
)
from b_learn_improve import (
    qa_forward_pass,
    qa_get_feedback,
    serialize_qa_feedback_results,
    _improve_with_perception_validation_conversational,
    _improve_beliefs_only_conversational,
    _prepend_rejection_notice,
)
from stepwise_b_learn_improve import (
    parse_submit_signal,
    build_perception_followup_message,
    build_perception_with_analysis_prompt,
    build_qa_followup_message,
    _build_obs_section,
    _build_execution_report_section,
)
from stepwise_eb_learn_improve import (
    EBQAPair,
    serialize_eb_qa_pairs,
    deserialize_eb_qa_pairs,
    eb_qa_to_qa,
    generate_questions_from_steps,
    formulate_experiment_from_question,
    update_qa_from_trajectory,
    deduplicate_qa_pairs,
    select_qa_pairs_for_experiment,
    trim_qa_pairs,
    trim_qa_pairs_scored,
)
from llm_utils import extract_xml_key
from stepwise_b_learn import (
    format_steps_context,
    _compose_obs_text,
    _refresh_buffer_with_perception,
    _sample_observations_from_buffer,
    _histories_for_samples,
    _inject_beliefs,
    _flush_improve_progress,
)
from stepwise_explore import (
    load_perception_fn,
    apply_perception,
    apply_perception_with_history,
)
from run_utils import setup_run, improve_logging, _update_summary_json


def _extract_xml_attr(attrs: str, name: str) -> str | None:
    match = re.search(
        rf"""\b{name}\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))""",
        attrs,
        re.IGNORECASE,
    )
    if not match:
        return None
    return next(group for group in match.groups() if group is not None)


def _normalize_q_ref(raw: str | None, max_index: int) -> int | None:
    if raw is None:
        return None
    match = re.search(r"(?:\bQ\s*)?(\d+)", raw, re.IGNORECASE)
    if not match:
        return None
    idx = int(match.group(1)) - 1
    if 0 <= idx < max_index:
        return idx
    return None


def _parse_q_tag_indices(text: str, max_index: int) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"<q\b(?P<attrs>[^>]*)/?>", text or "", re.IGNORECASE):
        attrs = match.group("attrs")
        idx = _normalize_q_ref(_extract_xml_attr(attrs, "n"), max_index)
        if idx is None:
            idx = _normalize_q_ref(_extract_xml_attr(attrs, "source_index"), max_index)
        if idx is not None and idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


# ---------------------------------------------------------------------------
# EB-local prompt templates: teach the LLM that perceive() takes a list of raw
# observations (most recent N, chronological) rather than a single string.
# Scoped to stepwise_eb_learn so stepwise_b_learn's prompts are unaffected.
# ---------------------------------------------------------------------------

def _eb_perception_instructions(include_policy: bool = True) -> str:
    belief_ref = "world knowledge and policy" if include_policy else "world knowledge"
    return f"""For the perception module:
- It should be a valid Python function `perceive(observation_history: list[str]) -> str`.
- Input `observation_history` is a list of the most recent raw environment observations from the current episode, in chronological order. `observation_history[-1]` is the current observation; earlier entries are prior steps.
- Output should only contain features important for decision-making in the environment.
- Ensure the output does not exceed 2000 characters. Remove features that the agent does not use for decision-making.
- The output should be consistent with the current {belief_ref} and should not make any additional or contradictory assumptions to them.
- Ensure that the perception module is working correctly — that it is correctly extracting the intended information from the raw environment state and presenting it clearly.
"""


def _eb_beliefs_block_template(include_policy: bool) -> str:
    """Return the <updated_beliefs>...</updated_beliefs> schema block."""
    if include_policy:
        return """<updated_beliefs>
<world_knowledge>
- [fact about mechanics, environmental properties, cause-and-effect relationships, etc ...]
- ...
</world_knowledge>
<policy>
- [what to do in specific situations, priorities, strategies for completing the objective etc ...]
- ...
</policy>
</updated_beliefs>"""
    return """<updated_beliefs>
<world_knowledge>
- [fact about mechanics, environmental properties, cause-and-effect relationships, etc ...]
- ...
</world_knowledge>
</updated_beliefs>"""


def _build_eb_response_format(include_policy: bool = True) -> str:
    beliefs_block = _eb_beliefs_block_template(include_policy)
    return f"""Format your response as:
<think>
Analyze the step sequence and determine what needs to change.
</think>

{beliefs_block}

<updated_perception>
```python
def perceive(observation_history: list[str]) -> str:
    # Your implementation here. observation_history[-1] is the current observation;
    # earlier entries are prior observations from the same episode (capped).
    pass
```
</updated_perception>

<status>CONTINUE or SUBMIT</status>

Set status to SUBMIT if you believe your current beliefs and perception are sufficient given the available evidence. Set status to CONTINUE if you want to receive re-evaluation results and iterate further. When in doubt, prefer CONTINUE."""


def _build_eb_beliefs_only_response_format(include_policy: bool = True) -> str:
    beliefs_block = _eb_beliefs_block_template(include_policy)
    think_line = (
        "Analyze the step sequence and determine what world knowledge and policy need to change."
        if include_policy
        else "Analyze the step sequence and determine what world knowledge needs to change."
    )
    return f"""Format your response as:
<think>
{think_line}
</think>

{beliefs_block}

<perception_analysis>
Analysis of how the perception module could be improved:
- What extracted information was misleading or incorrect?
- What kind of information can be extracted that would help the agent make better decisions?
</perception_analysis>"""


def _build_beliefs_section_guidance(include_policy: bool = True) -> str:
    """Inline guidance describing how to structure the beliefs output."""
    if include_policy:
        return """For beliefs:
- Overall the beliefs should be split into two sections:
  * <world_knowledge>: Facts about how the environment works — mechanics, properties, cause-and-effect relationships.
  * <policy>: Tactical approaches — what to do in specific situations, priorities, strategies for completing the objective.
- Correct any wrong or misleading beliefs in either section.
- Both sections should be consise, made up of a few brief points, merging any redundant or stale information.
- They should be grounded in the evidence present in the step sequence, only containing inferences from what we have observed so far."""
    return """For beliefs:
- Beliefs should consist of a single <world_knowledge> section containing facts about how the environment works — mechanics, properties, cause-and-effect relationships.
- Correct any wrong or misleading beliefs.
- Keep it consise, made up of a few brief points, merging any redundant or stale information.
- They should be grounded in the evidence present in the step sequence, only containing inferences from what we have observed so far."""


EB_PERCEPTION_ONLY_RESPONSE_FORMAT = """Format your response as:
<think>
Analyze the perception input/output examples and determine what the perception module should extract differently.
</think>

<updated_perception>
```python
def perceive(observation_history: list[str]) -> str:
    # Your implementation here. observation_history[-1] is the current observation;
    # earlier entries are prior observations from the same episode (capped).
    pass
```
</updated_perception>

<status>CONTINUE or SUBMIT</status>

Set status to SUBMIT if you believe your current perception module is extracting information well. Set status to CONTINUE if you want to see updated examples and iterate further. When in doubt, prefer CONTINUE."""


@dataclass
class StepwiseEBLearnConfig:
    n_environment_steps: int
    max_perception_iterations: "int | list[list[int]]"  # Track 1b turns (int or schedule)
    max_qa_iterations: "int | list[list[int]]"          # Track 2 turns (int or schedule)
    max_qa_per_forward: int
    max_answered_qa_pairs: int
    max_unanswered_qa_pairs: int
    num_questions: int                 # Questions per generation step
    num_sample_obs: int
    explore_temp: float
    artifact_update_interval: int
    improve_interval: int
    experiment_interval: int
    max_steps_context_chars: int
    max_images_context: int = 10
    perception_history_window: int = 10
    perception_input_tail: int = 2
    hide_obs_when_image: bool = False
    question_gen_current_state_only: bool = False
    include_policy: bool = True
    question_scoring_method: str = "b_diff_light"
    question_scoring_max_concurrent: int = 8
    mock_mode: bool = False


def _resolve_schedule(value: "int | list[list[int]]", global_step: int) -> int:
    """Resolve a schedule value based on global_step.

    If value is an int, return it directly.
    If value is a list of [step_threshold, count] pairs, return the count
    for the first range that contains global_step. The last entry acts as
    the default (its threshold is ignored).

    Example: [[10, 10], [20, 5], [0, 3]]
      - steps 0-9:  10
      - steps 10-19: 5
      - steps 20+:   3
    """
    if isinstance(value, int):
        return value
    cumulative = 0
    for i, entry in enumerate(value):
        threshold, count = entry[0], entry[1]
        if i == len(value) - 1:
            return count
        if global_step < cumulative + threshold:
            return count
        cumulative += threshold
    return value[-1][1]


# ---------------------------------------------------------------------------
# Mock mode helpers. In mock mode every LLM call is short-circuited at its
# respective client layer:
#   - Improve/QA/experiment LLM calls go through mixed_improve._llm_call /
#     _llm_call_conversational, gated by mixed_improve.set_mock_mode().
#   - Agent LLM calls go through balrog.client.LLMClientWrapper.generate,
#     gated by balrog.client.set_mock_mode() + set_mock_action_provider().
# In both cases the real prompt is still constructed and logged; only the
# network call is replaced with a synthesized response. Gated end-to-end by
# StepwiseEBLearnConfig.mock_mode.
# ---------------------------------------------------------------------------


def _mock_available_actions(env) -> list[str]:
    """Return a list of valid action strings for the given env."""
    actions = getattr(env, "language_action_space", None)
    if actions is None and hasattr(env, "env"):
        actions = getattr(env.env, "language_action_space", None)
    # Some wrappers expose a Strings object (crafter); unwrap if needed.
    values = getattr(actions, "_values", None)
    if values is not None:
        actions = values
    if not actions:
        actions = ["wait"]
    try:
        return list(actions)
    except TypeError:
        return ["wait"]


# ---------------------------------------------------------------------------
# Image helpers: trajectory_buffer entries carry obs PIL images under the
# "image" key. JSON serialization strips them; lookups match by step number.
# ---------------------------------------------------------------------------


def _buffer_for_json(trajectory_buffer: list[dict]) -> list[dict]:
    """Return a copy of trajectory_buffer safe for JSON serialization (drops PIL images)."""
    return [
        {k: v for k, v in e.items() if k not in ("image", "result_image")}
        for e in trajectory_buffer
    ]


def _save_prompt_images(images: list, step_dir: Path, subdir: str) -> list[str]:
    """Save each PIL image under ``step_dir/subdir/image_N.png`` (1-indexed).

    Returns relative paths (relative to ``step_dir``) for inclusion in log JSON,
    so the viz can render them alongside the prompt via the same ``(image K)``
    numbering used in the prompt text.
    """
    if not images:
        return []
    out_dir = step_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_paths: list[str] = []
    for i, img in enumerate(images, 1):
        if img is None:
            continue
        rel = f"{subdir}/image_{i}.png"
        try:
            img.save(step_dir / rel)
        except Exception as e:
            evolve_logger.warning(f"Failed to save prompt image {i} to {rel}: {e}")
            continue
        rel_paths.append(rel)
    return rel_paths


def _images_for_sample_obs(
    trajectory_buffer: list[dict],
    sample_obs: list[tuple[str, int]],
) -> list:
    """Return the PIL image for each (raw_obs, step_num) sample, aligned by index.

    Uses the trajectory_buffer's stored pre-action image for the matching step.
    """
    images = []
    for _raw_obs, step_num in sample_obs:
        img = None
        for entry in trajectory_buffer:
            if entry.get("episode_boundary"):
                continue
            if entry.get("action") is None:
                continue
            if entry.get("step") == step_num:
                img = entry.get("image")
                break
        images.append(img)
    return images


def _images_for_steps_context(
    trajectory_buffer: list[dict],
    steps_context_text: str,
    max_images: int | None = None,
) -> tuple[str, list]:
    """Return ``(annotated_text, images)`` for the step blocks in ``steps_context_text``.

    Each ``<pre_state>`` opening tag is annotated with ``(image K)`` and, when a
    block also contains ``<post_state>``, that tag likewise. ``image K`` is
    1-indexed and refers to position K in the returned images list so the LLM
    can cross-reference the textual step with the attached screenshot. Pre-action
    images come from each buffer entry's ``image`` field; post-action images
    (for ``<post_state>``) come from ``result_image``.
    """
    if not steps_context_text:
        return steps_context_text, []
    images: list = []
    seen: set[int] = set()
    entries_by_step = {
        e.get("step"): e
        for e in trajectory_buffer
        if not e.get("episode_boundary") and e.get("action") is not None
    }
    image_slots: list[tuple[int, str]] = []

    for m in re.finditer(r'<step n="(\d+)">(.*?)</step>', steps_context_text, re.DOTALL):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)

        entry = entries_by_step.get(n)
        if entry is None:
            continue

        block_inner = m.group(2)
        if entry.get("image") is not None:
            image_slots.append((n, "pre"))
        if "<post_state>" in block_inner and entry.get("result_image") is not None:
            image_slots.append((n, "post"))

    if max_images is None:
        keep_slots = set(image_slots)
    elif max_images <= 0:
        keep_slots = set()
    else:
        keep_slots = set(image_slots[-max_images:])

    seen.clear()

    def annotate_block(m: re.Match) -> str:
        n = int(m.group(1))
        block_inner = m.group(2)
        if n in seen:
            return m.group(0)
        seen.add(n)

        entry = entries_by_step.get(n)
        if entry is None:
            return m.group(0)

        new_inner = block_inner
        img = entry.get("image")
        if img is not None and (n, "pre") in keep_slots:
            images.append(img)
            idx = len(images)
            new_inner = new_inner.replace(
                "<pre_state>", f"<pre_state> (image {idx})", 1
            )

        if "<post_state>" in block_inner:
            result_img = entry.get("result_image")
            if result_img is not None and (n, "post") in keep_slots:
                images.append(result_img)
                idx = len(images)
                new_inner = new_inner.replace(
                    "<post_state>", f"<post_state> (image {idx})", 1
                )

        return f'<step n="{n}">{new_inner}</step>'

    annotated = re.sub(
        r'<step n="(\d+)">(.*?)</step>',
        annotate_block,
        steps_context_text,
        flags=re.DOTALL,
    )
    return annotated, images


# ---------------------------------------------------------------------------
# Artifact saving helpers
# ---------------------------------------------------------------------------


def _save_step_artifacts_eb(
    step_dir: Path,
    beliefs: str,
    perception: str,
    qa_pairs: list[EBQAPair],
    feedback_history: list[dict],
    extraction_log: dict | None = None,
    experiment_log: dict | None = None,
    trim_log: dict | None = None,
):
    """Save all artifacts for a completed step."""
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "beliefs.txt").write_text(beliefs)
    (step_dir / "perception.py").write_text(perception)
    with open(step_dir / "qa_pairs.json", "w") as f:
        json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)
    if feedback_history:
        with open(step_dir / "feedback_history.json", "w") as f:
            json.dump(feedback_history, f, indent=4, default=str)
    if extraction_log:
        with open(step_dir / "extraction_log.json", "w") as f:
            json.dump(extraction_log, f, indent=4, default=str)
    if experiment_log:
        with open(step_dir / "experiment_log.json", "w") as f:
            json.dump(experiment_log, f, indent=4, default=str)
    if trim_log:
        with open(step_dir / "trim_log.json", "w") as f:
            json.dump(trim_log, f, indent=4, default=str)


def _save_step_log_eb(
    step_dir: Path,
    step: int,
    global_step: int,
    action: str | None,
    reward: float,
    done: bool,
    episode_return: float,
    agent_cost: float,
    extract_cost: float,
    improve_cost: float,
    experiment_cost: float,
    trim_cost: float,
    num_qa: int,
    num_unanswered: int,
    did_gen_questions: bool = False,
    did_formulate_experiment: bool = False,
    did_trim: bool = False,
    active_experiment: str | None = None,
    phase: str = "complete",
    env_info: dict | None = None,
):
    """Write a per-step JSON log with action, costs, and artifact counts."""
    step_log = {
        "step": step,
        "global_step": global_step,
        "phase": phase,
        "action": action,
        "reward": reward,
        "done": done,
        "episode_return_so_far": episode_return,
        "agent_step_cost": agent_cost,
        "extract_cost": extract_cost,
        "improve_cost": improve_cost,
        "experiment_cost": experiment_cost,
        "trim_cost": trim_cost,
        "step_total_cost": agent_cost + extract_cost + improve_cost + experiment_cost + trim_cost,
        "num_qa_pairs": num_qa,
        "num_answered_questions": num_qa - num_unanswered,
        "num_unanswered_questions": num_unanswered,
        "did_gen_questions": did_gen_questions,
        "did_formulate_experiment": did_formulate_experiment,
        "did_trim": did_trim,
        "active_experiment": active_experiment,
    }
    # Persist environment-specific info (e.g. ARC-AGI game_id, levels, state)
    if env_info:
        step_log["env_info"] = env_info
    with open(step_dir / "step_log.json", "w") as f:
        json.dump(step_log, f, indent=4)


def _save_episode_artifacts_eb(
    episode_dir: Path,
    beliefs: str,
    perception: str,
    qa_pairs: list[EBQAPair],
    trajectory_buffer: list[dict] | None = None,
    past_experiments: list[str] | None = None,
):
    """Save all artifacts for a completed episode."""
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "beliefs.txt").write_text(beliefs)
    (episode_dir / "perception.py").write_text(perception)
    with open(episode_dir / "qa_pairs.json", "w") as f:
        json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)
    if trajectory_buffer is not None:
        with open(episode_dir / "trajectory_buffer.json", "w") as f:
            json.dump(_buffer_for_json(trajectory_buffer), f, indent=2, default=str)
    if past_experiments is not None:
        with open(episode_dir / "past_experiments.json", "w") as f:
            json.dump(past_experiments, f, indent=4)


def _find_last_completed_episode_eb(
    output_dir: str,
) -> tuple[int, str, str, list[EBQAPair]]:
    """Find the last completed episode directory and restore EB state.

    Returns: (last_episode, beliefs, perception, qa_pairs)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return -1, "", "", []

    episode_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("episode_"):
            try:
                ep_num = int(item.name.split("_")[1])
                if (item / "beliefs.txt").exists():
                    episode_dirs.append((ep_num, item))
            except (ValueError, IndexError):
                continue

    if not episode_dirs:
        return -1, "", "", []

    episode_dirs.sort(key=lambda x: x[0])
    last_ep, last_dir = episode_dirs[-1]

    beliefs = (last_dir / "beliefs.txt").read_text()

    perception = ""
    perc_file = last_dir / "perception.py"
    if perc_file.exists():
        perception = perc_file.read_text()

    qa_pairs: list[EBQAPair] = []
    qa_file = last_dir / "qa_pairs.json"
    if qa_file.exists():
        try:
            qa_pairs = deserialize_eb_qa_pairs(json.loads(qa_file.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass

    evolve_logger.info(f"Resuming from episode {last_ep} in {last_dir}")
    return last_ep, beliefs, perception, qa_pairs


# ---------------------------------------------------------------------------
# Inner improve loop (conversational only, no moments)
# ---------------------------------------------------------------------------


def _run_improve_loop_eb(
    config: DictConfig,
    eb_config: StepwiseEBLearnConfig,
    beliefs: str,
    perception: str,
    qa_pairs: list[EBQAPair],
    trajectory_buffer: list[dict],
    default_knowledge: str,
    step: int,
    global_step: int = 0,
    step_dir: Path | None = None,
) -> tuple[str, str, list[EBQAPair], float, list[dict]]:
    """Run the 2-track adaptive improve loop (beliefs/perception + QA).

    Track 1a: Steps-based beliefs improvement (1 turn)
    Track 1b: Perception improvement from analysis (multi-turn)
    Track 2:  QA-based improvement (multi-turn, answered questions only)

    Returns: (beliefs, perception, qa_pairs, total_cost, feedback_history)
    """
    total_cost = 0.0
    feedback_history: list[dict] = []
    tag = f"[g{global_step}]"

    max_perception_iters = _resolve_schedule(eb_config.max_perception_iterations, global_step)
    max_qa_iters = _resolve_schedule(eb_config.max_qa_iterations, global_step)

    include_policy = eb_config.include_policy
    perception_instructions = _eb_perception_instructions(include_policy)
    eb_response_format = _build_eb_response_format(include_policy)
    beliefs_only_response_format = _build_eb_beliefs_only_response_format(include_policy)
    beliefs_section_guidance = _build_beliefs_section_guidance(include_policy)
    policy_task_phrase = (
        "world knowledge, policy, and perception module"
        if include_policy
        else "world knowledge and perception module"
    )

    hist_window = eb_config.perception_history_window
    display_tail = eb_config.perception_input_tail
    steps_context = format_steps_context(
        trajectory_buffer, perception, eb_config.max_steps_context_chars,
        history_window=hist_window,
        hide_raw_obs_when_image=eb_config.hide_obs_when_image,
    )
    sample_obs = _sample_observations_from_buffer(
        trajectory_buffer, eb_config.num_sample_obs,
    )
    sample_obs_histories = _histories_for_samples(trajectory_buffer, sample_obs)
    steps_context, steps_context_images = _images_for_steps_context(
        trajectory_buffer, steps_context,
    )
    sample_obs_images = _images_for_sample_obs(trajectory_buffer, sample_obs)
    track1b_sample_obs = (
        sample_obs[-eb_config.max_images_context:]
        if eb_config.max_images_context > 0
        else []
    )
    track1b_sample_obs_histories = (
        sample_obs_histories[-len(track1b_sample_obs):]
        if track1b_sample_obs
        else []
    )
    track1b_sample_obs_images = (
        sample_obs_images[-len(track1b_sample_obs):]
        if track1b_sample_obs
        else []
    )

    num_answered = sum(1 for q in qa_pairs if q.answer is not None)
    evolve_logger.info(
        f"{tag} Improve loop: perception={max_perception_iters}, "
        f"qa={max_qa_iters} iters, "
        f"{len(qa_pairs)} QA ({num_answered} answered), "
        f"{len(steps_context)} chars context"
    )

    try:

        # ========================================
        # Track 1a: Steps-based beliefs improvement
        # ========================================
        if steps_context:
            track1a_record = {"track": "steps_beliefs", "step": step, "global_step": global_step, "turns": []}

            steps_beliefs_prompt = f"""We are interacting with an environment and trying to figure out how it works. We maintain beliefs about the environment to guide the agent's decisions.

The agent receives the following default instructions/knowledge:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

We maintain the following current beliefs about the environment:
=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

Below is the actual sequence of the agent's recent interactions with the environment.
Each step shows: the pre-action observation, the perception module's output on it, the agent's reasoning, and the action taken.
Each ``<pre_state>`` (and ``<post_state>``, when present) is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the observation after the last action of an episode segment.

=== SEQUENCE OF STEPS ===
{steps_context}
=== END SEQUENCE OF STEPS ===

Your task is to:
1. Analyze the step sequence.
2. Update our beliefs about the environment based on confirmed knowledge from the steps.

Provide analysis highlighting:
- Belief learning: What can we infer from the observations and how should we update our current beliefs so that they more accurately reflect how the world works.
- Perception analysis: What information was presented in output of the perception module. What part of that information was helpful, what information was misleading / incorrect and what additional information would have helped if extracted by the perception module?

{beliefs_section_guidance}

{perception_instructions}

{beliefs_only_response_format}"""

            beliefs, turn_cost, _, response_text = asyncio.run(
                _improve_beliefs_only_conversational(
                    config=config,
                    beliefs=beliefs,
                    conversation_history=[],
                    user_message=steps_beliefs_prompt,
                    images=steps_context_images,
                )
            )
            total_cost += turn_cost

            perception_analysis = extract_xml_key(response_text, "perception_analysis") or ""

            turn_record = {"turn": 1, "cost": turn_cost, "prompt": steps_beliefs_prompt, "response": response_text}
            track1a_record["turns"].append(turn_record)

            evolve_logger.info(f"{tag}     Track 1a done (cost: ${turn_cost:.6f}, perception_analysis: {len(perception_analysis)} chars)")

            feedback_history.append(track1a_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)
        else:
            evolve_logger.info(f"{tag}     Track 1a: No steps context, skipping")
            perception_analysis = ""

        # ========================================
        # Track 1b: Perception improvement guided by beliefs analysis
        # ========================================
        if track1b_sample_obs:
            track1b_record = {"track": "perception_from_analysis", "step": step, "global_step": global_step, "turns": []}
            pre_perception_track1b = perception
            obs_section_1b = _build_obs_section(
                perception, track1b_sample_obs,
                sample_histories=track1b_sample_obs_histories, history_window=hist_window,
                display_tail=display_tail,
            )

            perception_from_analysis_prompt = build_perception_with_analysis_prompt(
                beliefs=beliefs,
                perception=perception,
                default_knowledge=default_knowledge,
                obs_section=obs_section_1b,
                perception_analysis=perception_analysis,
                max_iterations=max_perception_iters,
                perception_instructions=perception_instructions,
                response_format=EB_PERCEPTION_ONLY_RESPONSE_FORMAT,
            )

            perception_conv_1b: list[dict] = []
            prev_obs_section_1b = obs_section_1b
            prev_validation_error_1b: str | None = None

            for turn in range(max_perception_iters):
                evolve_logger.info(f"{tag}     Track 1b (perception from analysis) turn {turn + 1}/{max_perception_iters}")

                message = perception_from_analysis_prompt if turn == 0 else build_perception_followup_message(
                    perception, track1b_sample_obs, prev_obs_section_1b,
                    current_turn=turn + 1, max_turns=max_perception_iters,
                    sample_histories=track1b_sample_obs_histories, history_window=hist_window,
                    display_tail=display_tail,
                    perception_instructions=perception_instructions,
                    response_format=EB_PERCEPTION_ONLY_RESPONSE_FORMAT,
                )
                if prev_validation_error_1b:
                    message = _prepend_rejection_notice(message, prev_validation_error_1b)

                # Attach sample images on the first turn only; subsequent turns rely on history.
                turn_images = track1b_sample_obs_images if turn == 0 else None

                _beliefs_unused, perception, turn_cost, perception_conv_1b, response_text, prev_validation_error_1b = asyncio.run(
                    _improve_with_perception_validation_conversational(
                        config=config,
                        beliefs=beliefs,
                        perception=perception,
                        conversation_history=perception_conv_1b,
                        user_message=message,
                        sample_observations=track1b_sample_obs,
                        images=turn_images,
                        sample_histories=track1b_sample_obs_histories,
                        history_window=hist_window,
                    )
                )
                total_cost += turn_cost

                turn_record = {"turn": turn + 1, "cost": turn_cost, "prompt": message, "response": response_text}
                submitted = parse_submit_signal(response_text)
                turn_record["submitted"] = submitted
                track1b_record["turns"].append(turn_record)

                evolve_logger.info(
                    f"{tag}     Track 1b turn {turn + 1} done "
                    f"(cost: ${turn_cost:.6f}, submit: {submitted})"
                )

                if submitted:
                    break

                prev_obs_section_1b = _build_obs_section(
                    perception, track1b_sample_obs,
                    sample_histories=track1b_sample_obs_histories, history_window=hist_window,
                    display_tail=display_tail,
                )

            feedback_history.append(track1b_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

            # Rebuild steps_context if perception changed during Track 1b
            if perception != pre_perception_track1b:
                steps_context = format_steps_context(
                    trajectory_buffer, perception, eb_config.max_steps_context_chars,
                    history_window=hist_window,
                    hide_raw_obs_when_image=eb_config.hide_obs_when_image,
                )
                steps_context, steps_context_images = _images_for_steps_context(
                    trajectory_buffer, steps_context,
                )
        else:
            evolve_logger.info(f"{tag}     Track 1b: No sample observations, skipping")

        # ========================================
        # Track 2: QA-based conversational improvement
        # ========================================
        # Filter to answered questions only for forward/feedback evaluation
        answered_qa = [q for q in qa_pairs if q.answer is not None]
        if not answered_qa:
            evolve_logger.info(f"{tag}     Track 2: No answered questions, skipping")
        else:
            track2_record = {"track": "qa", "step": step, "global_step": global_step, "turns": []}

            # Convert to QAPair for existing forward/feedback functions
            qa_for_eval = [eb_qa_to_qa(q) for q in answered_qa]

            # Initial QA evaluation
            evolve_logger.info(f"{tag}     Track 2: Initial QA forward pass on {len(qa_for_eval)} answered questions...")
            qa_fwd_results, qa_fwd_cost, qa_fwd_prompts, qa_fwd_responses = asyncio.run(
                qa_forward_pass(
                    config=config,
                    beliefs=beliefs,
                    qa_pairs=qa_for_eval,
                    max_per_batch=eb_config.max_qa_per_forward,
                )
            )
            total_cost += qa_fwd_cost

            qa_fb_results, qa_fb_cost, qa_fb_prompts, qa_fb_responses = asyncio.run(
                qa_get_feedback(
                    config=config,
                    qa_forward_results=qa_fwd_results,
                    max_per_batch=eb_config.max_qa_per_forward,
                )
            )
            total_cost += qa_fb_cost

            qa_correct = [fr for fr in qa_fb_results if fr.verdict == "CORRECT"]
            qa_incorrect = [fr for fr in qa_fb_results if fr.verdict == "INCORRECT"]
            qa_inconclusive = [fr for fr in qa_fb_results if fr.verdict == "INCONCLUSIVE"]
            qa_actionable = [fr for fr in qa_fb_results if fr.verdict != "INCONCLUSIVE"]

            evolve_logger.info(
                f"{tag}     Track 2: Initial eval: {len(qa_correct)} correct, "
                f"{len(qa_incorrect)} incorrect, {len(qa_inconclusive)} inconclusive"
            )

            track2_record["initial_correct"] = len(qa_correct)
            track2_record["initial_incorrect"] = len(qa_incorrect)
            track2_record["qa_forward_cost"] = qa_fwd_cost
            track2_record["qa_feedback_cost"] = qa_fb_cost
            track2_record["qa_forward_prompt"] = "\n---\n".join(qa_fwd_prompts)
            track2_record["qa_forward_response"] = "\n---\n".join(qa_fwd_responses)
            track2_record["qa_feedback_prompt"] = "\n---\n".join(qa_fb_prompts)
            track2_record["qa_feedback_response"] = "\n---\n".join(qa_fb_responses)
            track2_record["qa_feedback_details"] = serialize_qa_feedback_results(qa_fb_results)

            pre_track_perception = perception
            if qa_actionable and qa_incorrect:
                # Build initial QA improvement prompt
                qa_blocks = []
                for i, fr in enumerate(qa_actionable, 1):
                    actual = "YES" if fr.forward.qa_pair.answer else "NO"
                    qa_blocks.append(
                        f"<qa_feedback n=\"{i}\">\n"
                        f"<question>{fr.forward.qa_pair.question}</question>\n"
                        f"<correct_answer>{actual}</correct_answer>\n"
                        f"<evidence>{fr.forward.qa_pair.evidence}</evidence>\n"
                        f"<predicted_answer>{fr.forward.predicted_answer}</predicted_answer>\n"
                        f"<agent_reasoning>{fr.forward.reasoning}</agent_reasoning>\n"
                        f"<verdict>{fr.verdict}</verdict>\n"
                        f"<feedback>{fr.feedback}</feedback>\n"
                        f"</qa_feedback>"
                    )
                qa_text = "\n\n".join(qa_blocks)

                execution_report_section = _build_execution_report_section(
                    perception, sample_obs,
                    sample_histories=sample_obs_histories, history_window=hist_window,
                )

                initial_qa_prompt = f"""You are improving an agent's knowledge and perception based on testing its understanding of the environment via question-answering.

The agent receives the following default instructions/knowledge:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

=== CURRENT PERCEPTION MODULE ===
{perception if perception else "(empty - no perception module yet)"}
=== END CURRENT PERCEPTION MODULE ===
{execution_report_section}
We tested the agent's understanding by asking it factual questions about the environment.
The agent answered based only on its current world knowledge.

Results: {len(qa_correct)} correct, {len(qa_incorrect)} incorrect out of {len(qa_fb_results)} evaluated.

<qa_feedback_results>
{qa_text}
</qa_feedback_results>

Each ``<pre_state>`` (and ``<post_state>``, when present) in the sequence below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the observation after the last action of an episode segment.

=== SEQUENCE OF STEPS (for additional context) ===
{steps_context if steps_context else "(no steps recorded yet)"}
=== END SEQUENCE OF STEPS ===

Your task: Based on the QA feedback, improve the agent's {policy_task_phrase}.

For INCORRECT predictions, analyze:
1. Was the agent's world knowledge missing the relevant fact? If so, add it.
2. Was the agent's world knowledge wrong? If so, correct it.
3. Does the perception module need to extract different information to support this knowledge? If so, update it.

Guidelines:
- Keep each beliefs section to at most 8 concise bullet points. Merge redundant points.
- Balance safety with progress toward the objective.
- Remove beliefs that are contradicted by the QA evidence.

This is a multi-turn conversation. After each response, the QA pairs will be re-evaluated with your updated beliefs/perception. You can iterate up to {max_qa_iters} turns.

{perception_instructions}

{eb_response_format}"""

                qa_conversation: list[dict] = []
                prev_qa_correct = len(qa_correct)
                prev_qa_incorrect = len(qa_incorrect)
                prev_validation_error_2: str | None = None

                for turn in range(max_qa_iters):
                    evolve_logger.info(f"{tag}     Track 2 turn {turn + 1}/{max_qa_iters}")

                    message = initial_qa_prompt if turn == 0 else build_qa_followup_message(
                        qa_fb_results, prev_qa_correct, prev_qa_incorrect,
                        response_format=eb_response_format,
                    )
                    if prev_validation_error_2:
                        message = _prepend_rejection_notice(message, prev_validation_error_2)

                    # Initial turn's prompt contains both steps_context and sample_obs
                    # raw states — attach their images. Followups re-reference via history.
                    turn_images = None
                    if turn == 0:
                        turn_images = list(steps_context_images) + list(sample_obs_images)

                    beliefs, perception, turn_cost, qa_conversation, response_text, prev_validation_error_2 = asyncio.run(
                        _improve_with_perception_validation_conversational(
                            config=config,
                            beliefs=beliefs,
                            perception=perception,
                            conversation_history=qa_conversation,
                            user_message=message,
                            sample_observations=sample_obs if sample_obs else None,
                            images=turn_images,
                            sample_histories=sample_obs_histories if sample_obs else None,
                            history_window=hist_window,
                        )
                    )
                    total_cost += turn_cost

                    turn_record = {"turn": turn + 1, "cost": turn_cost, "prompt": message, "response": response_text}
                    submitted = parse_submit_signal(response_text)
                    turn_record["submitted"] = submitted
                    track2_record["turns"].append(turn_record)

                    evolve_logger.info(
                        f"{tag}     Track 2 turn {turn + 1} done "
                        f"(cost: ${turn_cost:.6f}, submit: {submitted})"
                    )

                    if submitted:
                        evolve_logger.info(f"{tag}     Track 2: LLM submitted after {turn + 1} turn(s)")
                        break

                    # Re-evaluate QA for next turn (unless this is the last turn)
                    if turn + 1 < max_qa_iters:
                        prev_qa_correct = sum(1 for fr in qa_fb_results if fr.verdict == "CORRECT")
                        prev_qa_incorrect = sum(1 for fr in qa_fb_results if fr.verdict == "INCORRECT")

                        qa_fwd_results, qa_fwd_cost, _, _ = asyncio.run(
                            qa_forward_pass(
                                config=config,
                                beliefs=beliefs,
                                qa_pairs=qa_for_eval,
                                max_per_batch=eb_config.max_qa_per_forward,
                            )
                        )
                        total_cost += qa_fwd_cost

                        qa_fb_results, qa_fb_cost, _, _ = asyncio.run(
                            qa_get_feedback(
                                config=config,
                                qa_forward_results=qa_fwd_results,
                                max_per_batch=eb_config.max_qa_per_forward,
                            )
                        )
                        total_cost += qa_fb_cost

                        new_correct = sum(1 for fr in qa_fb_results if fr.verdict == "CORRECT")
                        new_incorrect = sum(1 for fr in qa_fb_results if fr.verdict == "INCORRECT")
                        evolve_logger.info(
                            f"{tag}     Track 2 re-eval: {new_correct} correct "
                            f"({new_correct - prev_qa_correct:+d}), "
                            f"{new_incorrect} incorrect ({new_incorrect - prev_qa_incorrect:+d})"
                        )
            else:
                evolve_logger.info(f"{tag}     Track 2: No incorrect QA, skipping improvement")

            feedback_history.append(track2_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

            # Rebuild steps_context if perception changed during Track 2
            if perception != pre_track_perception:
                steps_context = format_steps_context(
                    trajectory_buffer, perception, eb_config.max_steps_context_chars,
                    history_window=hist_window,
                    hide_raw_obs_when_image=eb_config.hide_obs_when_image,
                )
                steps_context, steps_context_images = _images_for_steps_context(
                    trajectory_buffer, steps_context,
                )

    except Exception as e:
        evolve_logger.error(f"{tag}     Improve loop failed: {e}")
        logging.exception("Improve loop failed")
        feedback_history.append({"error": str(e), "step": step, "global_step": global_step})
        if step_dir is not None:
            _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

    return beliefs, perception, qa_pairs, total_cost, feedback_history


# ---------------------------------------------------------------------------
# Core per-step episode loop
# ---------------------------------------------------------------------------


async def select_tied_b_diff_question(
    config: DictConfig,
    *,
    qa_pairs: list[EBQAPair],
    tied_source_indices: list[int],
    top_score: float,
    beliefs: str,
    default_knowledge: str,
) -> tuple[int | None, float, dict]:
    """Use an LLM to choose one target question from tied top b-diff scores."""
    if len(tied_source_indices) <= 1:
        return (
            tied_source_indices[0] if tied_source_indices else None,
            0.0,
            {
                "executed": False,
                "reason": "no_top_score_tie",
                "top_score": top_score,
                "candidate_source_indices": tied_source_indices,
            },
        )

    tied_questions_text = "\n".join(
        f'Q{i + 1} (score={top_score:.6f}, source_step={qa_pairs[i].source_step}): '
        f"{qa_pairs[i].question}"
        for i in tied_source_indices
    )
    default_knowledge_section = ""
    if default_knowledge:
        default_knowledge_section = f"""
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===
"""
    prompt = f"""You are selecting the next experiment target for an agent learning an environment.

{default_knowledge_section}

=== CURRENT BELIEFS ===
{beliefs}
=== END CURRENT BELIEFS ===

=== AVAILABLE QUESTIONS ===
{tied_questions_text}
=== AVAILABLE QUESTIONS ===

Select questions that will -
1. be most valuable to answer next
2. cover distinct aspects of the environment


Use each question's Q number in the <q n="..."> attribute. Format your response as:
<think>
Which questions should be selected?
</think>
<selected_question>
<q n="Q1" />
</selected_question>"""

    text, cost = await _llm_call(config, prompt)
    selected_text = extract_xml_key(text, "selected_question") or ""
    tied_set = set(tied_source_indices)
    selected_source_index: int | None = None
    for idx in _parse_q_tag_indices(selected_text, len(qa_pairs)):
        if idx in tied_set:
            selected_source_index = idx
            break
    if selected_source_index is None:
        for match in re.finditer(r"\d+", selected_text):
            idx = int(match.group(0)) - 1
            if idx in tied_set:
                selected_source_index = idx
                break

    parse_error = None
    if selected_source_index is None:
        selected_source_index = tied_source_indices[0]
        parse_error = "No valid tied source_index parsed; fell back to first ranked tied question"

    tie_break_log = {
        "executed": True,
        "reason": "multiple_top_score_tie",
        "top_score": top_score,
        "candidate_source_indices": tied_source_indices,
        "candidate_questions": [
            {
                "source_index": i,
                "question": qa_pairs[i].question,
                "source_step": qa_pairs[i].source_step,
                "score": top_score,
            }
            for i in tied_source_indices
        ],
        "selected_source_index": selected_source_index,
        "selected_question": qa_pairs[selected_source_index].question,
        "prompt": prompt,
        "response": text,
    }
    if parse_error:
        tie_break_log["parse_error"] = parse_error
    return selected_source_index, cost, tie_break_log


def run_stepwise_eb_learn_episode(
    config: DictConfig,
    eb_config: StepwiseEBLearnConfig,
    beliefs: str,
    perception: str,
    qa_pairs: list[EBQAPair],
    current_experiment: str | None,
    default_knowledge: str,
    output_dir: str,
    episode_idx: int = 0,
    global_step_start: int = 0,
    max_episode_steps: int | None = None,
    trajectory_buffer: list[dict] | None = None,
    past_experiments: list[str] | None = None,
    cumulative_cost_offset: float = 0.0,
) -> tuple[str, str, list[EBQAPair], str | None, dict, int, list[dict], list[str]]:
    """Run a single episode with per-step EB-learning.

    Returns:
        (beliefs, perception, qa_pairs, current_experiment, episode_stats, steps_taken,
         trajectory_buffer, past_experiments)
    """
    # --- Setup environment and agent ---
    env_name = config.envs.names.split("-")[0]

    if env_name == "arc_agi":
        from arc_agi_env import make_arc_env
        task = config.tasks.arc_agi_tasks[0]
        env = make_arc_env(task, config)
    elif env_name == "autumn":
        from autumn_env import make_autumn_env
        task = config.tasks.autumn_tasks[0]
        env = make_autumn_env(task, config)
    else:
        tasks = config.tasks[f"{env_name}_tasks"]
        task = tasks[0]
        env = make_env(env_name, task, config)
    agent_factory = AgentFactory(config)
    agent = agent_factory.create_agent()
    agent.reset()

    # In mock mode, install a closure that samples a random valid action from
    # *this* episode's env. balrog.client's mock hook calls it when synthesizing
    # the agent's LLM response. Set only when mock_mode is on; the flag itself
    # is toggled once in stepwise_eb_learn() below.
    if eb_config.mock_mode:
        set_mock_action_provider(lambda: random.choice(_mock_available_actions(env)))

    seed = config.envs.env_kwargs.seed
    if seed is None:
        seed = get_unique_seed(process_num=0, episode_idx=episode_idx)
    random.seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)

    # Inform agent of death/respawn if this is not the first episode
    if episode_idx > 0:
        obs["text"]["short_term_context"] = (
            "The previous episode was terminated and you have respawned.\n\n"
            + obs["text"]["short_term_context"]
        )

    # Setup instruction prompt with beliefs
    _inject_beliefs(config, agent, env, env_name, task, beliefs)
    agent.experiment_goal = current_experiment

    # Save raw initial obs before apply_perception modifies long_term_context in-place
    _pre_action_raw_long = obs["text"]["long_term_context"]
    _pre_action_raw_short = obs["text"].get("short_term_context", "")
    _pre_action_image = obs.get("image")  # PIL Image or None

    # Per-episode raw observation history for history-aware perception modules.
    raw_obs_history: list[str] = [_pre_action_raw_long]

    # Setup perception
    perception_fn = load_perception_fn(perception)
    if perception_fn is not None:
        apply_perception_with_history(
            obs, perception_fn, raw_obs_history, eb_config.perception_history_window
        )

    # Build initial obs_text (with perception applied) for the first buffer entry
    _pre_action_obs_text = _compose_obs_text(
        obs["text"]["short_term_context"],
        obs["text"]["long_term_context"],
    )

    # Episode tracking
    max_steps = env.max_steps if config.eval.get("max_steps_per_episode") is None else config.eval.max_steps_per_episode
    if max_episode_steps is not None:
        max_steps = min(max_steps, max_episode_steps)
    episode_log: dict = {
        "task": task,
        "action_frequency": defaultdict(int),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
    }

    trajectory_buffer = trajectory_buffer if trajectory_buffer is not None else []
    # Insert episode boundary marker
    if trajectory_buffer:
        trajectory_buffer.append({
            "step": None,
            "episode_boundary": True,
            "episode_idx": episode_idx,
            "obs_text": "",
            "raw_long_term_context": "",
            "action": None,
            "reward": 0.0,
            "reasoning": "",
            "done": True,
        })
    total_learn_cost = 0.0
    episode_return = 0.0
    cumulative_step_cost = 0.0
    step_extraction_log: dict | None = None
    step_experiment_log: dict | None = None
    past_experiments = past_experiments if past_experiments is not None else []
    if current_experiment and current_experiment not in past_experiments:
        past_experiments.append(current_experiment)

    # CSV logging
    ep_dir = Path(output_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = ep_dir / "trajectory.csv"

    pbar = tqdm(total=max_steps, desc=f"Stepwise EB-learn ep {episode_idx}", leave=False, dynamic_ncols=True)
    feedback_history: list[dict] = []

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, escapechar="\u02d8", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Step", "Action", "Reasoning", "Observation", "Auxiliary_Observation", "Reward", "Done"])

        action = None
        step = 0
        result_obs_text: str | None = None
        new_raw_short: str = ""
        done = False

        for step in range(max_steps):
            global_step = global_step_start + step

            # Per-step directory
            step_dir = ep_dir / f"step_{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            step_extract_cost = 0.0
            step_improve_cost = 0.0
            step_experiment_cost = 0.0
            step_trim_cost = 0.0
            step_extraction_log = None
            step_experiment_log = None
            step_trim_log: dict | None = None
            did_trim_step = False
            step_feedback_records: list[dict] = []
            num_unanswered = sum(1 for q in qa_pairs if q.answer is None)

            # Write preliminary step_log
            _save_step_log_eb(
                step_dir=step_dir, step=step, global_step=global_step,
                action=None, reward=0.0, done=False, episode_return=episode_return,
                agent_cost=0.0, extract_cost=0.0, improve_cost=0.0, experiment_cost=0.0,
                trim_cost=0.0,
                num_qa=len(qa_pairs), num_unanswered=num_unanswered,
                did_gen_questions=False, did_formulate_experiment=False,
                active_experiment=current_experiment, phase="started",
            )

            # --- Question generation + experiment formulation ---
            should_gen_experiments = step % eb_config.experiment_interval == 0
            did_gen_questions = False
            did_formulate_experiment = False

            if should_gen_experiments:
                evolve_logger.info(f"[g{global_step}] Generating questions...")
                exp_steps_context = format_steps_context(
                    trajectory_buffer, perception, eb_config.max_steps_context_chars,
                    history_window=eb_config.perception_history_window,
                    hide_raw_obs_when_image=eb_config.hide_obs_when_image,
                    include_trailing_state=False,
                )
                q_steps_context, q_steps_context_images = _images_for_steps_context(
                    trajectory_buffer, exp_steps_context,
                )
                exp_steps_context, exp_steps_context_images = _images_for_steps_context(
                    trajectory_buffer,
                    exp_steps_context,
                    max_images=eb_config.max_images_context,
                )
                q_steps_context = (
                    "" if eb_config.question_gen_current_state_only else q_steps_context
                )
                q_steps_context_images = (
                    [] if eb_config.question_gen_current_state_only else q_steps_context_images
                )
                with improve_logging(step_dir):
                    # Step 1: Generate questions
                    new_questions, q_cost, q_prompt, q_response = asyncio.run(
                        generate_questions_from_steps(
                            config=config,
                            beliefs=beliefs,
                            perception_code=perception,
                            steps_context=q_steps_context,
                            current_qa=qa_pairs,
                            current_observation=_pre_action_raw_long,
                            current_aux_observation=_pre_action_raw_short,
                            default_knowledge=default_knowledge,
                            num_questions=eb_config.num_questions,
                            current_step=global_step,
                            current_image=_pre_action_image,
                            steps_context_images=q_steps_context_images,
                            hide_raw_obs=eb_config.hide_obs_when_image,
                            include_recent_history=not eb_config.question_gen_current_state_only,
                        )
                    )
                    qa_pairs.extend(new_questions)
                    step_experiment_cost += q_cost
                    total_learn_cost += q_cost
                    did_gen_questions = True

                    evolve_logger.info(
                        f"[g{global_step}] Generated {len(new_questions)} questions — cost: ${q_cost:.6f}"
                    )

                    # Step 2: De-duplicate the maintained bank, then select a
                    # capped probe subset for this experiment prompt. Unlike
                    # the old trim path, low-scoring questions remain in the
                    # maintained bank for future scoring/projection coverage.
                    selection_cost_total = 0.0
                    qa_pairs_for_experiment = list(qa_pairs)
                    selected_source_indices = list(range(len(qa_pairs)))
                    scoring_method = eb_config.question_scoring_method

                    qa_pairs, dedup_cost, dedup_log = asyncio.run(
                        deduplicate_qa_pairs(
                            config=config,
                            current_qa=qa_pairs,
                        )
                    )
                    selection_cost_total += dedup_cost

                    (
                        qa_pairs_for_experiment,
                        selected_source_indices,
                        select_cost,
                        selection_log,
                    ) = asyncio.run(
                        select_qa_pairs_for_experiment(
                            config=config,
                            current_qa=qa_pairs,
                            max_answered_qa_pairs=eb_config.max_answered_qa_pairs,
                            max_unanswered_qa_pairs=eb_config.max_unanswered_qa_pairs,
                            default_knowledge=default_knowledge,
                            beliefs=beliefs,
                        )
                    )
                    selection_cost_total += select_cost

                    scoring_log: dict | None = None
                    ranked_unanswered_indices: list[int] = []
                    target_experiment_question_index: int | None = None
                    target_experiment_question_source_index: int | None = None
                    if scoring_method in ("b_diff_full", "b_diff_light"):
                        from question_scoring import score_questions_b_diff

                        candidate_indices = [
                            i for i in selected_source_indices
                            if qa_pairs[i].answer is None
                        ]
                        method_suffix = (
                            "full" if scoring_method == "b_diff_full" else "light"
                        )
                        scores, score_cost, scoring_log = asyncio.run(
                            score_questions_b_diff(
                                config=config,
                                beliefs=beliefs,
                                qa_pairs=qa_pairs,
                                method=method_suffix,
                                include_policy=eb_config.include_policy,
                                max_concurrent=eb_config.question_scoring_max_concurrent,
                                candidate_indices=candidate_indices,
                                default_knowledge=default_knowledge,
                            )
                        )
                        selection_cost_total += score_cost

                        ranked_unanswered_indices = sorted(
                            candidate_indices,
                            key=lambda i: (scores.get(i, 0.0), qa_pairs[i].source_step),
                            reverse=True,
                        )
                        tied_top_indices: list[int] = []
                        tie_break_log: dict = {
                            "executed": False,
                            "reason": "no_unanswered_candidates",
                            "candidate_source_indices": [],
                        }
                        selected_tied_source_index: int | None = None
                        if ranked_unanswered_indices:
                            top_score = scores.get(ranked_unanswered_indices[0], 0.0)
                            tied_top_indices = [
                                i
                                for i in ranked_unanswered_indices
                                if scores.get(i, 0.0) == top_score
                            ]
                            if len(tied_top_indices) > 1:
                                (
                                    selected_tied_source_index,
                                    tie_break_cost,
                                    tie_break_log,
                                ) = asyncio.run(
                                    select_tied_b_diff_question(
                                        config=config,
                                        qa_pairs=qa_pairs,
                                        tied_source_indices=tied_top_indices,
                                        top_score=top_score,
                                        beliefs=beliefs,
                                        default_knowledge=default_knowledge,
                                    )
                                )
                                selection_cost_total += tie_break_cost
                            else:
                                selected_tied_source_index = ranked_unanswered_indices[0]
                                tie_break_log = {
                                    "executed": False,
                                    "reason": "no_top_score_tie",
                                    "top_score": top_score,
                                    "candidate_source_indices": tied_top_indices,
                                    "selected_source_index": selected_tied_source_index,
                                    "selected_question": qa_pairs[selected_tied_source_index].question,
                                }
                        selected_answered_indices = [
                            i for i in selected_source_indices
                            if qa_pairs[i].answer is not None
                        ]
                        ranked_source_indices = (
                            ranked_unanswered_indices + selected_answered_indices
                        )
                        qa_pairs_for_experiment = [
                            qa_pairs[i] for i in ranked_source_indices
                        ]
                        selected_source_indices = ranked_source_indices
                        if selected_tied_source_index is not None:
                            target_experiment_question_source_index = selected_tied_source_index
                            target_experiment_question_index = selected_source_indices.index(
                                selected_tied_source_index
                            )

                        scoring_log["ranked_unanswered"] = [
                            {
                                "idx": i,
                                "question": qa_pairs[i].question,
                                "source_step": qa_pairs[i].source_step,
                                "score": scores.get(i, 0.0),
                            }
                            for i in ranked_unanswered_indices
                        ]
                        scoring_log["selected_probe_indices"] = candidate_indices
                        scoring_log["selected_probe_questions"] = [
                            qa_pairs[i].question for i in candidate_indices
                        ]
                        scoring_log["tie_break"] = tie_break_log
                        scoring_log["projection_question_count"] = sum(
                            1 for q in qa_pairs if q.answer is None
                        )
                        scoring_log["target_experiment_question_source_index"] = (
                            target_experiment_question_source_index
                        )
                        scoring_log["target_experiment_question"] = (
                            qa_pairs[target_experiment_question_source_index].question
                            if target_experiment_question_source_index is not None
                            else None
                        )
                    elif scoring_method != "llm_trim":
                        raise ValueError(
                            f"Unknown question_scoring_method: {scoring_method!r}"
                        )

                    step_trim_log = {
                        "method": f"probe_selection_{scoring_method}",
                        "pre_trim_count": dedup_log.get("pre_dedup_count"),
                        "post_trim_count": len(qa_pairs),
                        "pre_trim_answered": dedup_log.get("pre_dedup_answered"),
                        "post_trim_answered": sum(
                            1 for q in qa_pairs if q.answer is not None
                        ),
                        "pre_trim_unanswered": dedup_log.get("pre_dedup_unanswered"),
                        "post_trim_unanswered": sum(
                            1 for q in qa_pairs if q.answer is None
                        ),
                        "max_answered_qa_pairs": eb_config.max_answered_qa_pairs,
                        "max_unanswered_qa_pairs": eb_config.max_unanswered_qa_pairs,
                        "dropped_count": dedup_log.get("dropped_count", 0),
                        "total_cost": selection_cost_total,
                        "dedup": dedup_log,
                        "selection": selection_log,
                        "scoring": scoring_log,
                        "maintained_bank_preserved": True,
                        "experiment_source_indices": selected_source_indices,
                        "target_experiment_prompt_index": target_experiment_question_index,
                        "target_experiment_source_index": target_experiment_question_source_index,
                        "target_experiment_question": (
                            qa_pairs[target_experiment_question_source_index].question
                            if target_experiment_question_source_index is not None
                            else None
                        ),
                        "experiment_questions": [
                            {
                                "source_index": i,
                                "question": qa_pairs[i].question,
                                "answer": qa_pairs[i].answer,
                                "source_step": qa_pairs[i].source_step,
                            }
                            for i in selected_source_indices
                        ],
                    }
                    step_trim_cost = selection_cost_total
                    total_learn_cost += selection_cost_total
                    did_trim_step = bool(dedup_log.get("dropped_count", 0))

                    with open(step_dir / "trim_log.json", "w") as f:
                        json.dump(step_trim_log, f, indent=4, default=str)
                    with open(step_dir / "question_selection_log.json", "w") as f:
                        json.dump(step_trim_log, f, indent=4, default=str)
                    if scoring_log is not None:
                        method_suffix = (
                            "full" if scoring_method == "b_diff_full" else "light"
                        )
                        scoring_artifact = {
                            "step": global_step,
                            "method": method_suffix,
                            "source": "online_probe_selection",
                            "did_trim": False,
                            "num_qa_before_trim": step_trim_log.get("pre_trim_count"),
                            "num_qa_after_trim": step_trim_log.get("post_trim_count"),
                            "num_answered_before_trim": step_trim_log.get("pre_trim_answered"),
                            "num_answered_after_trim": step_trim_log.get("post_trim_answered"),
                            "num_unanswered_before_trim": step_trim_log.get("pre_trim_unanswered"),
                            "num_unanswered_after_trim": step_trim_log.get("post_trim_unanswered"),
                            "cap_answered": step_trim_log.get("max_answered_qa_pairs"),
                            "cap_unanswered": step_trim_log.get("max_unanswered_qa_pairs"),
                            "dropped_count": step_trim_log.get("dropped_count"),
                            "cost_usd": selection_cost_total,
                            "ranked_unanswered": scoring_log.get("ranked_unanswered", []),
                            "kept_unanswered_questions": [
                                qa_pairs[i].question for i in ranked_unanswered_indices
                            ],
                            "dropped_unanswered_questions": [],
                            "tie_break": scoring_log.get("tie_break"),
                            "scoring_log": scoring_log,
                            "selection_log": selection_log,
                            "dedup_log": dedup_log,
                        }
                        with open(
                            step_dir / f"scoring_online_{method_suffix}.json", "w"
                        ) as f:
                            json.dump(scoring_artifact, f, indent=4, default=str)
                    with open(step_dir / "qa_pairs.json", "w") as f:
                        json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)

                    num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
                    evolve_logger.info(
                        f"[g{global_step}] Question selection done — "
                        f"bank QA: {len(qa_pairs)} ({num_unanswered} unanswered), "
                        f"experiment prompt QA: {len(qa_pairs_for_experiment)}, "
                        f"cost: ${selection_cost_total:.6f}"
                    )

                    # Step 3: Formulate experiment from question
                    evolve_logger.info(f"[g{global_step}] Formulating experiment from questions...")
                    experiment_plan, q_idx, e_cost, e_prompt, e_response = asyncio.run(
                        formulate_experiment_from_question(
                            config=config,
                            beliefs=beliefs,
                            perception_code=perception,
                            steps_context=exp_steps_context,
                            current_qa=qa_pairs_for_experiment,
                            current_experiment=current_experiment,
                            current_observation=_pre_action_raw_long,
                            current_aux_observation=_pre_action_raw_short,
                            default_knowledge=default_knowledge,
                            current_image=_pre_action_image,
                            steps_context_images=exp_steps_context_images,
                            hide_raw_obs=eb_config.hide_obs_when_image,
                            target_question_index=target_experiment_question_index,
                        )
                    )
                    step_experiment_cost += e_cost
                    total_learn_cost += e_cost

                    if experiment_plan is not None:
                        # Move old active experiment to past if it exists
                        if current_experiment and current_experiment not in past_experiments:
                            past_experiments.append(current_experiment)
                        current_experiment = experiment_plan
                        did_formulate_experiment = True

                    # Experiment prompts use the capped trajectory-image
                    # sequence plus the current pre-action image. Question
                    # prompts keep their own sequence so the saved prompt
                    # images match the exact attachments used for each call.
                    exp_prompt_images = list(exp_steps_context_images or [])
                    if _pre_action_image is not None:
                        exp_prompt_images.append(_pre_action_image)
                    exp_image_paths = _save_prompt_images(
                        exp_prompt_images, step_dir, "experiment_log_images",
                    )
                    if eb_config.question_gen_current_state_only:
                        q_image_paths = _save_prompt_images(
                            [_pre_action_image] if _pre_action_image is not None else [],
                            step_dir,
                            "question_gen_log_images",
                        )
                    else:
                        q_prompt_images = list(q_steps_context_images or [])
                        if _pre_action_image is not None:
                            q_prompt_images.append(_pre_action_image)
                        if q_prompt_images == exp_prompt_images:
                            q_image_paths = exp_image_paths
                        else:
                            q_image_paths = _save_prompt_images(
                                q_prompt_images,
                                step_dir,
                                "question_gen_log_images",
                            )

                    step_experiment_log = {
                        "question_gen_prompt": q_prompt,
                        "question_gen_response": q_response,
                        "question_gen_image_paths": q_image_paths,
                        "new_questions": [q.question for q in new_questions],
                        "experiment_prompt": e_prompt,
                        "experiment_response": e_response,
                        "experiment_image_paths": exp_image_paths,
                        "experiment_plan": experiment_plan,
                        "selected_question_index": q_idx,
                        "selected_question_source_index": (
                            selected_source_indices[q_idx]
                            if q_idx is not None
                            and 0 <= q_idx < len(selected_source_indices)
                            else None
                        ),
                        "target_question_index": target_experiment_question_index,
                        "target_question_source_index": target_experiment_question_source_index,
                        "target_question": (
                            qa_pairs[target_experiment_question_source_index].question
                            if target_experiment_question_source_index is not None
                            else None
                        ),
                        "question_selection_method": scoring_method,
                        "qa_pairs_for_experiment": serialize_eb_qa_pairs(
                            qa_pairs_for_experiment
                        ),
                        "qa_pairs_for_experiment_source_indices": selected_source_indices,
                        "qa_pairs_at_formulation": serialize_eb_qa_pairs(qa_pairs),
                    }

                # Update current experiment and inject into agent
                agent.experiment_goal = current_experiment

                # Write experiment artifacts immediately
                with open(step_dir / "experiment_log.json", "w") as f:
                    json.dump(step_experiment_log, f, indent=4, default=str)
                with open(step_dir / "qa_pairs.json", "w") as f:
                    json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)

                evolve_logger.info(
                    f"[g{global_step}] Experiment: {'new' if did_formulate_experiment else 'kept'} — "
                    f"cost: ${step_experiment_cost:.6f}"
                )

            num_unanswered = sum(1 for q in qa_pairs if q.answer is None)

            # Update step_log: experiment phase done, agent about to act
            _save_step_log_eb(
                step_dir=step_dir, step=step, global_step=global_step,
                action=None, reward=0.0, done=False, episode_return=episode_return,
                agent_cost=0.0, extract_cost=0.0, improve_cost=0.0, experiment_cost=step_experiment_cost,
                trim_cost=step_trim_cost,
                num_qa=len(qa_pairs), num_unanswered=num_unanswered,
                did_gen_questions=did_gen_questions, did_formulate_experiment=did_formulate_experiment,
                did_trim=did_trim_step,
                active_experiment=current_experiment, phase="acting",
            )

            # --- Agent acts ---
            # Even in mock mode we call agent.act() so the full prompt is
            # constructed and logged; balrog.client's mock hook short-circuits
            # the API call and returns a synthesized response containing a
            # random valid action from the action provider installed below.
            if eb_config.hide_obs_when_image and _pre_action_image is not None:
                obs["text"]["long_term_context"] = ""
            response = agent.act(obs, prev_action=action)
            action = response.completion
            reasoning = response.reasoning if hasattr(response, "reasoning") else ""

            episode_log["action_frequency"][action] += 1
            episode_log["input_tokens"] += response.input_tokens
            episode_log["output_tokens"] += response.output_tokens
            episode_log["total_cost"] += response.cost
            agent_step_cost = response.cost

            # --- Environment step ---
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except ValueError as e:
                logging.warning(f"[g{global_step}] Invalid action: {action} — {e}")
                if config.eval.feedback_on_invalid_action:
                    obs["text"]["long_term_context"] = (
                        f"\n\nYour previous output did not contain a valid action. Retry\n\n"
                        f"Observation:\n{obs['text']['long_term_context']}"
                    )
                terminated = False
                truncated = False
                reward = 0.0

            done = terminated or truncated
            episode_return += reward

            # Save raw new obs BEFORE applying perception
            new_raw_long = obs["text"]["long_term_context"]
            new_raw_short = obs["text"].get("short_term_context", "")

            # Grow the per-episode raw-obs history with this step's post-action obs.
            raw_obs_history.append(new_raw_long)

            # Apply perception to new obs
            if perception_fn is not None:
                apply_perception_with_history(
                    obs, perception_fn, raw_obs_history, eb_config.perception_history_window
                )

            result_obs_text = _compose_obs_text(
                obs["text"]["short_term_context"],
                obs["text"]["long_term_context"],
            )

            # Capture agent messages
            try:
                agent_messages = [
                    {"role": m.role, "content": m.content}
                    for m in agent.last_messages
                ]
            except Exception:
                agent_messages = []
            agent_messages.append({
                "role": "assistant",
                "content": reasoning,
                "action": action,
            })

            with open(step_dir / "agent_messages.json", "w") as amf:
                json.dump(agent_messages, amf, indent=2, default=str)

            # Save observation images if available
            if _pre_action_image is not None:
                try:
                    _pre_action_image.save(step_dir / "obs_before.png")
                except Exception:
                    pass
            _post_action_image = obs.get("image")
            if _post_action_image is not None:
                try:
                    _post_action_image.save(step_dir / "obs_after.png")
                except Exception:
                    pass

            # Write CSV row immediately
            csv_writer.writerow([step, action, reasoning, _pre_action_obs_text, _pre_action_raw_short, reward, done])
            csv_file.flush()

            # Update step_log with action/reward/done
            _save_step_log_eb(
                step_dir=step_dir, step=step, global_step=global_step,
                action=action, reward=reward, done=done, episode_return=episode_return,
                agent_cost=agent_step_cost, extract_cost=0.0, improve_cost=0.0, experiment_cost=step_experiment_cost,
                trim_cost=step_trim_cost,
                num_qa=len(qa_pairs), num_unanswered=num_unanswered,
                did_gen_questions=did_gen_questions, did_formulate_experiment=did_formulate_experiment,
                did_trim=did_trim_step,
                active_experiment=current_experiment, phase="extracting",
                env_info=info if isinstance(info, dict) else None,
            )

            # Append buffer entry (image is the pre-action obs image;
            # result_image is the post-action obs image).
            trajectory_buffer.append({
                "step": global_step,
                "obs_text": _pre_action_obs_text,
                "raw_long_term_context": _pre_action_raw_long,
                "raw_short_term_context": _pre_action_raw_short,
                "result_raw_long_term_context": new_raw_long,
                "result_raw_short_term_context": new_raw_short,
                "image": _pre_action_image,
                "result_image": _post_action_image,
                "action": action,
                "reward": reward,
                "reasoning": reasoning,
                "done": False,
            })

            if done:
                trajectory_buffer.append({
                    "step": global_step + 1,
                    "obs_text": result_obs_text,
                    "raw_long_term_context": new_raw_long,
                    "raw_short_term_context": new_raw_short,
                    "image": _post_action_image,
                    "action": None,
                    "reward": 0.0,
                    "reasoning": "",
                    "done": True,
                })

            pbar.update(1)

            evolve_logger.info(
                f"[g{global_step}|ep{episode_idx}|s{step}] "
                f"action={action!r}  reward={reward:.2f}  return={episode_return:.2f}  "
                f"done={done}  agent_cost=${agent_step_cost:.6f}"
            )

            # --- Determine what to do this step ---
            steps_in = step + 1
            should_update_artifacts = (
                (steps_in % eb_config.artifact_update_interval == 0)
                or done
            )
            should_improve = (
                (steps_in % eb_config.improve_interval == 0)
                or done
            )

            # --- Artifact update (update Q from trajectory) ---
            if should_update_artifacts and len(trajectory_buffer) > 0:
                evolve_logger.info(f"[g{global_step}] Updating Q from {len(trajectory_buffer)} buffered steps...")
                steps_context = format_steps_context(
                    trajectory_buffer, perception, eb_config.max_steps_context_chars,
                    history_window=eb_config.perception_history_window,
                    hide_raw_obs_when_image=eb_config.hide_obs_when_image,
                )
                steps_context, steps_context_images_update = _images_for_steps_context(
                    trajectory_buffer,
                    steps_context,
                    max_images=eb_config.max_images_context,
                )
                with improve_logging(step_dir):
                    qa_pairs, extract_cost, step_extraction_log = asyncio.run(
                        update_qa_from_trajectory(
                            config=config,
                            current_qa=qa_pairs,
                            steps_context=steps_context,
                            current_step=global_step,
                            steps_context_images=steps_context_images_update,
                            hide_raw_obs=eb_config.hide_obs_when_image,
                        )
                    )
                    step_extract_cost = extract_cost
                    total_learn_cost += extract_cost

                # Save images attached to the extraction prompt so the viz can
                # render them alongside.
                step_extraction_log["prompt_image_paths"] = _save_prompt_images(
                    steps_context_images_update, step_dir, "extraction_log_images",
                )

                # Write extraction artifacts immediately
                with open(step_dir / "extraction_log.json", "w") as f:
                    json.dump(step_extraction_log, f, indent=4, default=str)
                with open(step_dir / "qa_pairs.json", "w") as f:
                    json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)

                num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
                evolve_logger.info(
                    f"[g{global_step}] Q update done — "
                    f"QA: {len(qa_pairs)} ({num_unanswered} unanswered), cost: ${extract_cost:.6f}"
                )

                # Update step_log: extraction done, improve starting
                _save_step_log_eb(
                    step_dir=step_dir, step=step, global_step=global_step,
                    action=action, reward=reward, done=done, episode_return=episode_return,
                    agent_cost=agent_step_cost, extract_cost=step_extract_cost, improve_cost=0.0,
                    experiment_cost=step_experiment_cost, trim_cost=step_trim_cost,
                    num_qa=len(qa_pairs), num_unanswered=num_unanswered,
                    did_gen_questions=did_gen_questions, did_formulate_experiment=did_formulate_experiment,
                    did_trim=did_trim_step,
                    active_experiment=current_experiment, phase="improving",
                    env_info=info if isinstance(info, dict) else None,
                )

            # --- Improve loop (beliefs/perception + QA) ---
            if should_improve:
                _perc_iters = _resolve_schedule(eb_config.max_perception_iterations, global_step)
                _qa_iters = _resolve_schedule(eb_config.max_qa_iterations, global_step)
                evolve_logger.info(
                    f"[g{global_step}] Running improve loop (perception={_perc_iters}, "
                    f"qa={_qa_iters} iters)..."
                )
                pre_improve_perception = perception
                with improve_logging(step_dir):
                    beliefs, perception, qa_pairs, improve_cost, iter_records = _run_improve_loop_eb(
                        config=config,
                        eb_config=eb_config,
                        beliefs=beliefs,
                        perception=perception,
                        qa_pairs=qa_pairs,
                        trajectory_buffer=trajectory_buffer,
                        default_knowledge=default_knowledge,
                        step=step,
                        global_step=global_step,
                        step_dir=step_dir,
                    )
                    step_improve_cost = improve_cost
                    total_learn_cost += improve_cost
                    step_feedback_records = iter_records
                    feedback_history.extend(iter_records)

                perception_changed = perception != pre_improve_perception

                # Reload perception after improvement
                perception_fn = load_perception_fn(perception)

                # Re-apply updated perception to current obs for the agent's next step
                if not done:
                    obs["text"]["long_term_context"] = new_raw_long
                    obs["text"]["short_term_context"] = new_raw_short
                    if perception_fn is not None:
                        apply_perception_with_history(
                            obs, perception_fn, raw_obs_history, eb_config.perception_history_window
                        )

                # Rebuild all buffered observations with the latest perception
                if perception_changed:
                    _refresh_buffer_with_perception(
                        trajectory_buffer, perception_fn,
                        history_window=eb_config.perception_history_window,
                    )

                # Inject updated beliefs for next step
                if not done:
                    _inject_beliefs(config, agent, env, env_name, task, beliefs)

                evolve_logger.info(
                    f"[g{global_step}] Improve done — cost: ${improve_cost:.6f}"
                )

                # Write qa immediately after improve
                with open(step_dir / "qa_pairs.json", "w") as f:
                    json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)

            # --- Carry forward pre-action vars ---
            if not done:
                _pre_action_raw_long = new_raw_long
                _pre_action_raw_short = new_raw_short
                _pre_action_image = obs.get("image")
                _pre_action_obs_text = _compose_obs_text(
                    obs["text"]["short_term_context"],
                    obs["text"]["long_term_context"],
                )

            # --- Per-step artifact save ---
            step_total_cost = agent_step_cost + step_extract_cost + step_improve_cost + step_experiment_cost + step_trim_cost
            cumulative_step_cost += step_total_cost

            did_learn = should_update_artifacts or should_improve or should_gen_experiments
            if did_learn:
                _save_step_artifacts_eb(
                    step_dir, beliefs, perception, qa_pairs,
                    step_feedback_records,
                    extraction_log=step_extraction_log,
                    experiment_log=step_experiment_log,
                    trim_log=step_trim_log,
                )

            num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
            _save_step_log_eb(
                step_dir=step_dir, step=step, global_step=global_step,
                action=action, reward=reward, done=done, episode_return=episode_return,
                agent_cost=agent_step_cost, extract_cost=step_extract_cost,
                improve_cost=step_improve_cost, experiment_cost=step_experiment_cost,
                trim_cost=step_trim_cost,
                num_qa=len(qa_pairs), num_unanswered=num_unanswered,
                did_gen_questions=did_gen_questions, did_formulate_experiment=did_formulate_experiment,
                did_trim=did_trim_step,
                active_experiment=current_experiment, phase="complete",
                env_info=info if isinstance(info, dict) else None,
            )

            # Per-step summary update
            _update_summary_json(
                output_dir=os.path.dirname(output_dir),
                step=global_step,
                step_cost=step_total_cost,
                cumulative_cost=cumulative_cost_offset + cumulative_step_cost,
                rollout_stats={
                    "episode_idx": episode_idx,
                    "episode_step": step,
                    "action": action,
                    "reward": reward,
                    "episode_return": episode_return,
                    "done": done,
                    "num_qa_pairs": len(qa_pairs),
                    "num_unanswered_questions": num_unanswered,
                    "did_extract": should_update_artifacts,
                    "did_improve": should_improve,
                    "did_gen_questions": did_gen_questions,
                    "did_formulate_experiment": did_formulate_experiment,
                    "did_trim": did_trim_step,
                },
            )

            if done:
                evolve_logger.info(
                    f"[g{global_step}] Episode {episode_idx} DONE — "
                    f"return={episode_return:.2f}, steps={step + 1}"
                )
                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
                pbar.set_postfix_str("DONE")
                break

        # Write terminal row with the post-action state from the last completed step,
        # regardless of whether the episode ended via `done` or by hitting max_steps.
        # This lets the viewer show the state *after* the final action.
        if result_obs_text is not None:
            csv_writer.writerow([step + 1, "", "", result_obs_text, new_raw_short, 0.0, done])
            csv_file.flush()

    if pbar.n < pbar.total:
        pbar.update(pbar.total - pbar.n)
    pbar.close()

    # Finalize episode stats
    episode_log["episode_return"] = episode_return
    episode_log["num_steps"] = step + 1
    episode_log["failed_candidates"] = env.failed_candidates
    episode_log.update(env.get_stats())
    episode_log["seed"] = seed
    episode_log["total_learn_cost"] = total_learn_cost
    episode_log["cumulative_step_cost"] = cumulative_step_cost
    episode_log["num_qa_pairs"] = len(qa_pairs)
    episode_log["num_answered_questions"] = sum(1 for q in qa_pairs if q.answer is not None)
    episode_log["num_unanswered_questions"] = sum(1 for q in qa_pairs if q.answer is None)


    json_filename = ep_dir / "episode_log.json"
    with open(json_filename, "w") as f:
        json.dump(episode_log, f, indent=4, default=str)

    env.close()

    evolve_logger.info(
        f"Episode {episode_idx} complete — return: {episode_return:.2f}, "
        f"steps: {step + 1}, "
        f"learn cost: ${total_learn_cost:.4f}, agent cost: ${episode_log['total_cost']:.4f}"
    )

    return beliefs, perception, qa_pairs, current_experiment, episode_log, step + 1, trajectory_buffer, past_experiments


# ---------------------------------------------------------------------------
# Outer orchestrator
# ---------------------------------------------------------------------------


def stepwise_eb_learn(
    eb_config: StepwiseEBLearnConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
):
    """Run stepwise EB-learning: experiment-driven per-step improvement across episodes."""
    evolve_logger.info("Starting stepwise EB-learning")

    # In mock mode, intercept all LLM calls at their client layer — both the
    # mixed_improve _llm_call path (improve/QA/experiment) and the BALROG
    # LLMClientWrapper.generate path (agent). Real prompts are still fully
    # constructed and logged; only the API call is short-circuited.
    set_mock_mode(bool(eb_config.mock_mode))
    set_client_mock_mode(bool(eb_config.mock_mode))

    # Check for resume
    last_ep, beliefs, perception, qa_pairs = _find_last_completed_episode_eb(output_dir)
    start_episode = last_ep + 1

    current_experiment: str | None = None
    trajectory_buffer: list[dict] = []
    past_experiments: list[str] = []
    global_steps_used = 0
    if start_episode > 0:
        num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
        evolve_logger.info(f"Resuming from episode {start_episode} ({len(qa_pairs)} QA, {num_unanswered} unanswered)")
        # Recover active experiment from last episode's step logs
        last_ep_dir = Path(output_dir) / f"episode_{last_ep}"
        for step_dir in sorted(last_ep_dir.glob("step_*"), reverse=True):
            sl_file = step_dir / "step_log.json"
            if sl_file.exists():
                try:
                    sl = json.loads(sl_file.read_text())
                    current_experiment = sl.get("active_experiment")
                    break
                except (json.JSONDecodeError, TypeError):
                    pass
        # Recover global step count from episode logs
        for ep_idx in range(start_episode):
            ep_log_file = Path(output_dir) / f"episode_{ep_idx}" / "episode_log.json"
            if ep_log_file.exists():
                try:
                    ep_data = json.loads(ep_log_file.read_text())
                    global_steps_used += ep_data.get("num_steps", 0)
                except (json.JSONDecodeError, TypeError):
                    pass
        # Recover trajectory buffer and past experiments from last episode
        traj_file = Path(output_dir) / f"episode_{last_ep}" / "trajectory_buffer.json"
        if traj_file.exists():
            try:
                trajectory_buffer = json.loads(traj_file.read_text())
            except (json.JSONDecodeError, TypeError):
                trajectory_buffer = []
        past_exp_file = Path(output_dir) / f"episode_{last_ep}" / "past_experiments.json"
        if past_exp_file.exists():
            try:
                past_experiments = json.loads(past_exp_file.read_text())
            except (json.JSONDecodeError, TypeError):
                past_experiments = []
    else:
        # Load initial state
        if (beliefs_path := config.eval.get("beliefs_path", None)) is not None:
            beliefs = Path(beliefs_path).read_text()
            evolve_logger.info(f"Loaded initial beliefs from: {beliefs_path}")
        else:
            beliefs = ""
        if (perception_path := config.eval.get("perception_path", None)) is not None:
            perception = Path(perception_path).read_text()
            evolve_logger.info(f"Loaded initial perception from: {perception_path}")
        else:
            perception = ""
        qa_pairs = []

    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge: {len(default_knowledge)} chars")

    evolve_logger.info(f"Stepwise EB-learn config:")
    evolve_logger.info(f"  Total env steps: {eb_config.n_environment_steps}")
    evolve_logger.info(f"  Improve iterations: perception={eb_config.max_perception_iterations}, qa={eb_config.max_qa_iterations} (schedule or fixed)")
    evolve_logger.info(f"  Artifact update interval: {eb_config.artifact_update_interval}")
    evolve_logger.info(f"  Improve interval: {eb_config.improve_interval}")
    evolve_logger.info(f"  Experiment interval: {eb_config.experiment_interval}")
    evolve_logger.info(f"  Num questions per gen: {eb_config.num_questions}")
    evolve_logger.info(f"  Max answered QA pairs: {eb_config.max_answered_qa_pairs}")
    evolve_logger.info(f"  Max unanswered QA pairs: {eb_config.max_unanswered_qa_pairs}")
    evolve_logger.info(f"  Question scoring method: {eb_config.question_scoring_method}")
    evolve_logger.info(f"  Question scoring max concurrent: {eb_config.question_scoring_max_concurrent}")
    evolve_logger.info(f"  Max steps context chars: {eb_config.max_steps_context_chars}")
    evolve_logger.info(f"  Max images context: {eb_config.max_images_context}")
    evolve_logger.info(f"  Explore temp: {eb_config.explore_temp}")
    evolve_logger.info(f"  Question gen current-state only: {eb_config.question_gen_current_state_only}")
    evolve_logger.info(f"  Include policy: {eb_config.include_policy}")
    if eb_config.mock_mode:
        evolve_logger.info(f"  MOCK MODE: enabled — no LLM calls; random actions and artifact perturbations")

    cumulative_cost = 0.0
    episode_idx = start_episode

    while global_steps_used < eb_config.n_environment_steps:
        remaining_steps = eb_config.n_environment_steps - global_steps_used

        num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
        evolve_logger.info(f"\n{'='*80}")
        evolve_logger.info(
            f"STEPWISE EB-LEARN EPISODE {episode_idx} "
            f"(global steps: {global_steps_used}/{eb_config.n_environment_steps}, "
            f"remaining: {remaining_steps})"
        )
        evolve_logger.info(f"QA pairs: {len(qa_pairs)} ({num_unanswered} unanswered), Experiment: {current_experiment or 'none'}")
        evolve_logger.info(f"{'='*80}")

        episode_dir = Path(output_dir) / f"episode_{episode_idx}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save input state
        (episode_dir / "input_beliefs.txt").write_text(beliefs)
        (episode_dir / "input_perception.py").write_text(perception)
        with override_temperature(config, eb_config.explore_temp):
            beliefs, perception, qa_pairs, current_experiment, episode_log, steps_taken, trajectory_buffer, past_experiments = (
                run_stepwise_eb_learn_episode(
                    config=config,
                    eb_config=eb_config,
                    beliefs=beliefs,
                    perception=perception,
                    qa_pairs=qa_pairs,
                    current_experiment=current_experiment,
                    default_knowledge=default_knowledge,
                    output_dir=str(episode_dir),
                    episode_idx=episode_idx,
                    global_step_start=global_steps_used,
                    max_episode_steps=remaining_steps,
                    trajectory_buffer=trajectory_buffer,
                    past_experiments=past_experiments,
                    cumulative_cost_offset=cumulative_cost,
                )
            )

        global_steps_used += steps_taken

        # Save episode artifacts
        _save_episode_artifacts_eb(
            episode_dir, beliefs, perception, qa_pairs,
            trajectory_buffer=trajectory_buffer,
            past_experiments=past_experiments,
        )

        episode_cost = episode_log.get("total_cost", 0.0) + episode_log.get("total_learn_cost", 0.0)
        cumulative_cost += episode_cost
        evolve_logger.info(
            f"[g{global_steps_used}] Episode {episode_idx} done — "
            f"cost: ${episode_cost:.4f}, cumulative: ${cumulative_cost:.4f}, "
            f"steps: {global_steps_used}/{eb_config.n_environment_steps}"
        )

        episode_idx += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    run_name_suffix = f"{config.agent.type}_{config.client.model_id.replace('/', '_')}_stepwise_eb_learn"

    original_cwd, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=config.eval.resume_from,
        output_dir_base=config.eval.output_dir,
        logger_name="evolve",
    )

    evolve_cfg = config.eval.evolve

    legacy_max_total_qa_pairs = evolve_cfg.get("max_total_qa_pairs", 50)
    eb_config = StepwiseEBLearnConfig(
        n_environment_steps=evolve_cfg.get("n_environment_steps", 100),
        max_perception_iterations=evolve_cfg.get(
            "max_perception_iterations",
            evolve_cfg.get("max_improve_iterations", evolve_cfg.get("num_improve_iterations", 5)),
        ),
        max_qa_iterations=evolve_cfg.get(
            "max_qa_iterations",
            evolve_cfg.get("max_improve_iterations", evolve_cfg.get("num_improve_iterations", 5)),
        ),
        max_qa_per_forward=evolve_cfg.get("max_qa_per_forward", 10),
        max_answered_qa_pairs=evolve_cfg.get(
            "max_answered_qa_pairs", legacy_max_total_qa_pairs
        ),
        max_unanswered_qa_pairs=evolve_cfg.get(
            "max_unanswered_qa_pairs", legacy_max_total_qa_pairs
        ),
        num_questions=evolve_cfg.get("num_questions", 5),
        num_sample_obs=evolve_cfg.get("num_sample_obs", 3),
        explore_temp=evolve_cfg.get("explore_temp", 1.0),
        artifact_update_interval=evolve_cfg.get("artifact_update_interval", 1),
        improve_interval=evolve_cfg.get("improve_interval", 5),
        experiment_interval=evolve_cfg.get("experiment_interval", 10),
        max_steps_context_chars=evolve_cfg.get("max_steps_context_chars", 50000),
        max_images_context=evolve_cfg.get("max_images_context", 10),
        perception_history_window=evolve_cfg.get("perception_history_window", 10),
        perception_input_tail=evolve_cfg.get("perception_input_tail", 2),
        hide_obs_when_image=evolve_cfg.get("hide_obs_when_image", False),
        question_gen_current_state_only=evolve_cfg.get("question_gen_current_state_only", False),
        include_policy=evolve_cfg.get("include_policy", True),
        question_scoring_method=evolve_cfg.get("question_scoring_method", "b_diff_light"),
        question_scoring_max_concurrent=evolve_cfg.get("question_scoring_max_concurrent", 8),
        mock_mode=evolve_cfg.get("mock_mode", False),
    )

    stepwise_eb_learn(
        eb_config=eb_config,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
