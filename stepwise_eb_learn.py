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
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# The GEPA/legacy experience-learning pipeline lives under prototypes/perc_invdyn
# and its modules import one another by bare name (run via uv from repo root), so
# that directory must be on sys.path. We add it here but import gepa_optimize /
# validate lazily inside _relearn_frontier_eb so a missing `gepa` install only
# fails runs that actually use question_scoring_method == "gepa_frontier".
_PERC_INVDYN_DIR = str((Path(__file__).resolve().parent / "prototypes" / "perc_invdyn"))
if _PERC_INVDYN_DIR not in sys.path:
    sys.path.insert(0, _PERC_INVDYN_DIR)

import hydra
import numpy as np
from balrog.agents import AgentFactory
from balrog.client import (
    set_mock_action_provider,
)
from balrog.client import (
    set_mock_mode as set_client_mock_mode,
)
from balrog.environments import make_env
from balrog.utils import get_unique_seed
from omegaconf import DictConfig
from tqdm import tqdm

from b_learn_improve import (
    _improve_beliefs_only_conversational,
    _improve_with_perception_validation_conversational,
    qa_forward_pass,
    qa_get_feedback,
    serialize_qa_feedback_results,
)
from explore import evolve_logger, get_default_knowledge
from goal_prompts import append_agent_goal, is_goal_aware, resolve_agent_goal
from llm_utils import extract_xml_key, extract_xml_kv
from mixed_improve import (
    QAPair,
    _llm_call,
    _run_perception_on_observation,
    set_meta_temperature,
    set_mock_mode,
)
from multi_theory_exploration import (
    deserialize_theories,
    init_theory_ensemble,
    merge_falsifications,
    refill_theories,
    select_discriminating_action,
    select_goal_action,
    serialize_theories,
    update_theory_posterior,
)
from theory_exploration import Theory
from run_utils import (
    _update_summary_json,
    improve_logging,
    is_minihack_success_episode,
    setup_run,
)
from stepwise_b_learn import (
    _compose_obs_text,
    _flush_improve_progress,
    _histories_for_samples,
    _inject_beliefs,
    _refresh_buffer_with_perception,
    _sample_observations_from_buffer,
    collect_perception_runtime_errors,
    format_perception_runtime_errors,
    format_steps_context,
)
from stepwise_b_learn_improve import (
    _build_execution_report_section,
    _build_obs_section,
    build_perception_with_analysis_prompt,
    build_qa_followup_message,
    parse_submit_signal,
)
from stepwise_eb_learn_improve import (
    EBQAPair,
    TRAJECTORY_REASONING_NOTE,
    deduplicate_qa_pairs,
    deserialize_eb_qa_pairs,
    eb_qa_to_qa,
    formulate_experiment_for_question,
    formulate_experiment_from_question,
    generate_questions_from_steps,
    score_experiments_against_questions,
    select_qa_pairs_and_formulate_experiments,
    select_qa_pairs_for_experiment,
    serialize_eb_qa_pairs,
    trim_qa_pairs,
    trim_qa_pairs_scored,
    update_qa_from_trajectory,
)
from stepwise_explore import (
    _perceive_signature_mode,
    apply_perception,
    apply_perception_with_history,
    load_perception_fn,
)

INVALID_ACTION_RETRY_MESSAGE = "Your previous action was not formatted correctly. Retry"


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
    match = re.fullmatch(r"\s*(?:Q\s*)?(\d+)\s*", raw, re.IGNORECASE)
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
- perceive() must work on EVERY observation the environment produces, including the first observation of an episode (which may include introductory text around the grid) and single-frame histories. It must never raise an exception and must never return a blank/empty string — if parsing fails, degrade gracefully and still return a useful non-empty description.
"""


_BELIEFS_COMPLETE_NOTE = (
    "Output the COMPLETE beliefs inside <updated_beliefs>: restate every belief you "
    "want to keep, not just what changed. This block fully replaces the current "
    "beliefs, so anything you omit will be lost. Never submit an empty "
    "<world_knowledge> unless you genuinely intend to discard all current beliefs."
)


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


_PERCEPTION_DIFF_GUIDANCE = """Provide your perception change as a **unified diff** that patches the CURRENT PERCEPTION MODULE shown above. Use standard `diff -u` syntax with `@@ ... @@` hunk headers; line numbers in the headers are ignored — context lines are matched verbatim against the current source, so quote them exactly. Keep each diff minimal: include only the lines you are changing plus a few context lines around them. Example:

```diff
@@ -10,5 +10,7 @@
 def perceive(observation_history: list[str]) -> str:
     obs = observation_history[-1]
-    return obs[:100]
+    parsed = parse_grid(obs)
+    return f"player={parsed.player} target={parsed.target}"
```

If you need to rewrite the perception module from scratch, you may instead emit the full module inside a ```python ... ``` block."""


def _build_eb_response_format(include_policy: bool = True) -> str:
    beliefs_block = _eb_beliefs_block_template(include_policy)
    return f"""Format your response as:
<think>
Analyze the step sequence and determine what needs to change.
</think>

{beliefs_block}

<updated_perception>
{_PERCEPTION_DIFF_GUIDANCE}
</updated_perception>

<status>CONTINUE or SUBMIT</status>

Set status to SUBMIT if you believe your current beliefs and perception are sufficient given the available evidence otherwise set status to CONTINUE."""


def _build_eb_qa_response_format(
    include_policy: bool = True, perception_enabled: bool = True
) -> str:
    beliefs_block = _eb_beliefs_block_template(include_policy)
    if not perception_enabled:
        return f"""Format your response as:
<think>
Analyze the feedback and determine what needs to change.
</think>

{beliefs_block}

{_BELIEFS_COMPLETE_NOTE}

<status>CONTINUE or SUBMIT</status>

Set status to SUBMIT if you believe your current beliefs are sufficient given the available evidence otherwise set status to CONTINUE."""
    return f"""Format your response as:
<think>
Analyze the feedback and determine what needs to change.
</think>

{beliefs_block}

{_BELIEFS_COMPLETE_NOTE}

If perception needs to change, put a unified diff inside <updated_perception> (see format below). Otherwise set the block exactly to KEEP_UNCHANGED.

<updated_perception>
KEEP_UNCHANGED
</updated_perception>

When you do change perception:
{_PERCEPTION_DIFF_GUIDANCE}

<status>CONTINUE or SUBMIT</status>

Set status to SUBMIT if you believe your current beliefs and perception are sufficient given the available evidence otherwise set status to CONTINUE."""


def _build_eb_beliefs_only_response_format(
    include_policy: bool = True, include_perception_analysis: bool = True
) -> str:
    beliefs_block = _eb_beliefs_block_template(include_policy)
    think_line = (
        "Analyze the step sequence and determine what world knowledge and policy need to change."
        if include_policy
        else "Analyze the step sequence and determine what world knowledge needs to change."
    )
    if not include_perception_analysis:
        return f"""Format your response as:
<think>
{think_line}
</think>

{beliefs_block}

{_BELIEFS_COMPLETE_NOTE}"""
    return f"""Format your response as:
<think>
{think_line}
</think>

{beliefs_block}

{_BELIEFS_COMPLETE_NOTE}

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


EB_PERCEPTION_ONLY_RESPONSE_FORMAT = f"""Format your response as:
<think>
Analyze the perception input/output examples and determine what the perception module should extract differently.
</think>

<updated_perception>
{_PERCEPTION_DIFF_GUIDANCE}
</updated_perception>

<status>CONTINUE or SUBMIT</status>

Set status to SUBMIT if you believe your current perception module is extracting information well otherwise set status to CONTINUE."""


@dataclass
class StepwiseEBLearnConfig:
    n_environment_steps: int
    max_perception_iterations: (
        "int | list[list[int]]"  # Track 1b turns (int or schedule)
    )
    max_qa_iterations: "int | list[list[int]]"  # Track 2 turns (int or schedule)
    max_qa_per_forward: int
    max_answered_qa_pairs: int
    max_unanswered_qa_pairs: int
    trim_unanswered_at_selection: bool
    num_questions: int  # Questions per generation step
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
    perception_enabled: bool = True
    question_scoring_method: str = "b_diff_light"
    question_scoring_max_concurrent: int = 8
    # Plan B (theory_entropy scoring): regenerate competing theories each
    # selection point, seed the bank with crux questions, and rank unanswered
    # questions by mutual information with the theory posterior.
    num_theories: int = 5
    num_crux_questions: int = 5
    # When true, the theory generator sees only the current state (text + image),
    # not the recent step history — to test whether withholding the trajectory
    # changes the hypotheses it proposes.
    theory_gen_current_state_only: bool = False
    theory_weight_decay: float = 0.6  # prior over theories: w_r ~ decay^(rank-1)
    # MI-residual theory seeding (lever #1): after generating the initial
    # theories, score the unanswered bank; any question the whole ensemble is
    # agnostic about (all-UNK -> MI 0) probes a mechanism no theory models, so
    # regenerate theories seeded with up to this many such "residual" questions
    # (each required to be ASSUMED-true by >=1 theory), giving the ensemble a
    # member that predicts it -> the question becomes discriminable/selectable.
    # 0 == disabled (original behavior).
    num_theory_seed_questions: int = 0
    # Plan A (question_scoring_method == "theory_disagreement"): a *persistent*
    # theory ensemble drives action selection (discriminating action) and is
    # reweighted from pre-registered predictions each step. See
    # multi_theory_exploration.py.
    theory_violation_penalty: float = 0.7  # violated theory keeps (1-p) of its weight
    theory_min_weight: float = 0.02  # drop theories below this posterior weight
    num_candidate_actions: int = 4  # candidate actions the selector compares
    # Plan A explore->exploit switch: once the ensemble has survived
    # ``exploit_stable_streak`` consecutive steps without an all-violated wipe
    # (and holds >= ``exploit_min_theories`` theories), switch from running
    # discriminating experiments to acting toward the goal under the MAP theory.
    exploit_enabled: bool = True
    exploit_stable_streak: int = 2
    exploit_min_theories: int = 2
    experiment_selection_mode: str = "single"  # "single" | "score_topk"
    experiment_scoring_max_concurrent: int = 8
    score_topk_filter_questions: bool = False
    critical_transitions_enabled: bool = False
    critical_id_min_for_perception: int = 3
    mock_mode: bool = False
    frozen_eval_after_learn: bool = False
    frozen_eval_envs: list[str] | None = None
    frozen_eval_max_steps: int = 501
    frozen_eval_minihack_goal: str | None = None
    frozen_eval_arc_agi_goal: str | None = None
    autumn_eval_after_learn: bool = False
    autumn_eval_task_types: list[str] | None = None
    autumn_eval_max_steps: int = 501
    autumn_eval_render_mode: str = "text"
    frozen_eval_autumn_planning_goal: str | None = None
    # --- GEPA/legacy frontier mode (question_scoring_method == "gepa_frontier") ---
    # Maintain a frontier (set of competing {perception, world_knowledge}
    # candidates) learned from the collected trajectory via an inverse-dynamics
    # objective, and use the candidates' disagreement to formulate experiments.
    frontier_learner: str = "gepa"  # "gepa" (pareto) | "legacy" (greedy P/B) | "legacy_pop" (pop)
    # Coordinate-aware inverse dynamics: keep click coordinates as distinct
    # learnable targets (ARC ACTION6 x y / autumn click row col) and build
    # hard-negative click choice sets. No-ops on move-only action sets.
    frontier_click_aware: bool = True
    frontier_size: int = 3
    frontier_relearn_interval: int = 10  # env steps between relearns
    frontier_min_buffer: int = 12  # skip relearn below this many usable transitions
    frontier_max_metric_calls: int = 80  # GEPA budget
    frontier_legacy_rounds: int = 6  # legacy greedy P/B alternations
    frontier_pop_size: int = 4  # legacy_pop: population / frontier size
    frontier_pop_rounds: int = 6  # legacy_pop: mutate+select rounds
    # legacy_pop image mode: ARC obs are images with no text grid to parse, so
    # learn B only via image inverse-dynamics (predictor sees the frames).
    # "auto" = on when the buffer carries images; "on"/"off" force it.
    frontier_image_mode: str = "auto"
    frontier_image_max_transitions: int = 16  # cap image transitions (vision cost)
    frontier_k_choices: int = 5
    frontier_train_n: int = 14
    frontier_val_n: int = 12
    frontier_test_n: int = 10
    frontier_fd_scorer: str = "none"  # "none" | "textdiff" | "judge"
    frontier_fd_weight: float = 0.5
    frontier_concurrency: int = 8
    # Default to the deployed-agent model (GEPA's weak task LM) and the same
    # model for reflection when unset; both resolve to config.client.model_id.
    frontier_task_model: str | None = None
    frontier_reflection_model: str | None = None


def _format_prior_attempts(prior_attempts: list[dict]) -> str:
    """Render a short per-turn outcome log to seed the next turn's prompt.

    Conversation history is reset between outer turns; this block stands in
    for the (deliberately discarded) chat memory so the LLM still knows which
    of its prior diffs were accepted vs. rejected.
    """
    if not prior_attempts:
        return ""
    lines = ["=== PRIOR PERCEPTION ATTEMPTS THIS LOOP ==="]
    for entry in prior_attempts:
        n = entry["turn"]
        if entry.get("validated"):
            tag_ = "perception updated" if entry.get("changed") else "no change"
            lines.append(f"Turn {n}: accepted ({tag_})")
        else:
            err = (entry.get("error") or "").replace("\n", " ").strip()
            if len(err) > 220:
                err = err[:220] + "..."
            lines.append(
                f"Turn {n}: rejected — {err}. Previous perception was preserved."
            )
    lines.append("=== END PRIOR ATTEMPTS ===")
    return "\n".join(lines)


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


def _critical_obs_from_buffer(
    trajectory_buffer: list[dict],
    n: int,
) -> list[tuple[str, int]]:
    """Return the middle and latest critical transitions, in chronological order.

    Shape matches ``_sample_observations_from_buffer``: ``list[(raw_obs, step)]``.
    Only entries with ``critical=True``, a non-empty raw observation, and a real
    action (i.e. not a terminal/episode-boundary marker) are eligible.
    """
    if n <= 0 or not trajectory_buffer:
        return []

    valid = [
        (e["raw_long_term_context"], e["step"])
        for e in trajectory_buffer
        if e.get("critical") is True
        and e.get("raw_long_term_context", "").strip()
        and not e.get("episode_boundary")
        and e.get("action") is not None
    ]
    if not valid:
        return []

    if len(valid) == 1:
        return [valid[-1]]

    middle = valid[len(valid) // 2]
    latest = valid[-1]
    return [middle] if middle[1] == latest[1] else [middle, latest]


def _middle_last_observations_from_buffer(
    trajectory_buffer: list[dict],
    n: int,
) -> list[tuple[str, int]]:
    """Return up to two uniformly sampled observations: middle and latest.

    This is EB Track 1b's fallback sampler. It intentionally omits the start
    observation so perception improvement sees only a representative middle
    state and the latest state.
    """
    if n <= 0 or not trajectory_buffer:
        return []

    valid = [
        (e["raw_long_term_context"], e["step"])
        for e in trajectory_buffer
        if e.get("raw_long_term_context", "").strip()
        and not e.get("episode_boundary")
        and e.get("action") is not None
    ]
    if not valid:
        return []

    max_samples = min(n, 2)
    if max_samples == 1 or len(valid) == 1:
        return [valid[-1]]

    middle = valid[len(valid) // 2]
    latest = valid[-1]
    return [middle] if middle[1] == latest[1] else [middle, latest]


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


def _save_eval_agent_messages(
    step_dir: Path, agent, reasoning: str, action: str
) -> list[dict]:
    """Persist the exact agent messages sent to the LLM for frozen eval steps."""
    try:
        runtime_messages = list(getattr(agent, "last_messages", []) or [])
    except Exception:
        runtime_messages = []

    image_dir = step_dir / "agent_message_images"
    agent_messages: list[dict] = []
    for i, message in enumerate(runtime_messages, 1):
        record = {
            "role": getattr(message, "role", "unknown"),
            "content": getattr(message, "content", ""),
        }
        attachment = getattr(message, "attachment", None)
        if attachment is not None:
            attachments = (
                attachment if isinstance(attachment, (list, tuple)) else [attachment]
            )
            attachment_paths = []
            for j, img in enumerate(attachments, 1):
                rel_path = (
                    f"agent_message_images/message_{i:03d}_attachment_{j:02d}.png"
                )
                try:
                    image_dir.mkdir(parents=True, exist_ok=True)
                    img.save(step_dir / rel_path)
                    attachment_paths.append(rel_path)
                except Exception as exc:
                    record.setdefault("attachment_errors", []).append(str(exc))
            if attachment_paths:
                record["attachment_paths"] = attachment_paths
                if len(attachment_paths) == 1:
                    record["attachment_path"] = attachment_paths[0]
        agent_messages.append(record)

    agent_messages.append(
        {
            "role": "assistant",
            "content": reasoning,
            "action": action,
        }
    )

    with open(step_dir / "agent_messages.json", "w") as amf:
        json.dump(agent_messages, amf, indent=2, default=str)
    return agent_messages


def _restore_agent_history_events(agent, history_events: list[dict] | None) -> None:
    """Seed a freshly-created BALROG agent with prior prompt-builder events."""
    if not history_events:
        return
    events = getattr(getattr(agent, "prompt_builder", None), "_events", None)
    if events is None:
        return
    events.clear()
    events.extend(history_events)


def _append_pending_agent_action_for_history(agent, action: str | None) -> None:
    """Record the final action of an episode before carrying history forward."""
    if not action:
        return
    prompt_builder = getattr(agent, "prompt_builder", None)
    events = getattr(prompt_builder, "_events", None)
    if events is None:
        return
    last_event = events[-1] if events else None
    if (
        isinstance(last_event, dict)
        and last_event.get("type") == "action"
        and last_event.get("action") == action
    ):
        return
    try:
        prompt_builder.update_action(action)
    except Exception:
        pass


def _snapshot_agent_history_events(agent) -> list[dict]:
    """Return the current prompt-builder event buffer for the next episode."""
    events = getattr(getattr(agent, "prompt_builder", None), "_events", None)
    if events is None:
        return []
    return list(events)


def _safe_image_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_").lower() or "image"


def _save_eval_labeled_images(
    step_dir: Path, prefix: str, images: list[dict] | None
) -> list[str]:
    rel_paths = []
    for i, item in enumerate(images or [], 1):
        image = item.get("image") if isinstance(item, dict) else None
        if image is None:
            continue
        label = _safe_image_label(str(item.get("label", f"image_{i}")))
        rel_path = f"{prefix}_{i:02d}_{label}.png"
        try:
            image.save(step_dir / rel_path)
            rel_paths.append(rel_path)
        except Exception:
            pass
    return rel_paths


def _save_eval_step_images(
    step_dir: Path,
    before_image,
    after_image,
    before_images: list[dict] | None = None,
    after_images: list[dict] | None = None,
) -> dict[str, bool | list[str]]:
    """Save pre/post observation images for frozen eval viewer inspection."""
    flags: dict[str, bool | list[str]] = {
        "has_obs_before": False,
        "has_obs_after": False,
    }
    if before_image is not None:
        try:
            before_image.save(step_dir / "obs_before.png")
            flags["has_obs_before"] = True
        except Exception:
            pass
    if after_image is not None:
        try:
            after_image.save(step_dir / "obs_after.png")
            flags["has_obs_after"] = True
        except Exception:
            pass
    before_paths = _save_eval_labeled_images(step_dir, "obs_before", before_images)
    after_paths = _save_eval_labeled_images(step_dir, "obs_after", after_images)
    if before_paths:
        flags["has_obs_before"] = True
        flags["obs_before_image_paths"] = before_paths
    if after_paths:
        flags["has_obs_after"] = True
        flags["obs_after_image_paths"] = after_paths
    return flags


def _run_perception_on_planning_state(perception_fn, state_text: str) -> str:
    try:
        if _perceive_signature_mode(perception_fn) == "history":
            return str(perception_fn([state_text]))
        return str(perception_fn(state_text))
    except Exception as e:
        logging.warning(f"Planning state perception module failed: {e}")
        return f"Perception code failed with error -\n{e}"


def _apply_autumn_planning_perception(obs: dict, perception_fn) -> dict | None:
    planning_eval = obs.get("planning_eval") or {}
    current_state = planning_eval.get("current_state_text")
    goal_state = planning_eval.get("goal_state_text")
    if perception_fn is None or not current_state or not goal_state:
        return None

    current_output = _run_perception_on_planning_state(perception_fn, current_state)
    goal_output = _run_perception_on_planning_state(perception_fn, goal_state)
    section = (
        f"\n{'=' * 10} Start of current-state features from Perception Module {'=' * 10}\n"
        f"{current_output}\n\n"
        f"{'=' * 10} End of current-state features from Perception Module {'=' * 10}\n\n"
        f"{'=' * 10} Start of goal-state features from Perception Module {'=' * 10}\n"
        f"{goal_output}\n\n"
        f"{'=' * 10} End of goal-state features from Perception Module {'=' * 10}\n"
    )
    obs["text"]["short_term_context"] = (
        section
        + "\n"
        + f"{'=' * 10} Start of Auxilliary Observation {'=' * 10}\n"
        + obs["text"].get("short_term_context", "")
        + f"\n\n{'=' * 10} End of Auxilliary Observation {'=' * 10}"
    )
    return {
        "current_state_input": current_state,
        "goal_state_input": goal_state,
        "current_state_perception": current_output,
        "goal_state_perception": goal_output,
    }


def _images_for_sample_obs(
    trajectory_buffer: list[dict],
    sample_obs: list[tuple[str, int]],
    include_result_images: bool = False,
    post_image_only: bool = False,
) -> list:
    """Return the PIL image for each (raw_obs, step_num) sample, aligned by index.

    Uses the trajectory_buffer's stored pre-action image for the matching step.
    When include_result_images is True, returns interleaved [before, after, before, after, ...]
    pairs for each sample (after falls back to before if result_image is None).
    When post_image_only is True, returns only each sample's post-action image.
    """
    images = []
    for _raw_obs, step_num in sample_obs:
        img = None
        result_img = None
        for entry in trajectory_buffer:
            if entry.get("episode_boundary"):
                continue
            if entry.get("action") is None:
                continue
            if entry.get("step") == step_num:
                img = entry.get("image")
                result_img = entry.get("result_image")
                break
        if post_image_only:
            images.append(result_img if result_img is not None else img)
        elif include_result_images:
            images.append(img)
            images.append(result_img if result_img is not None else img)
        else:
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

    for m in re.finditer(
        r'<step n="(\d+)">(.*?)</step>', steps_context_text, re.DOTALL
    ):
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


def _select_residual_seed_questions(
    scoring_log: dict, k: int
) -> list[str]:
    """Pick up to ``k`` unanswered questions the current theory ensemble cannot
    predict — every theory answered UNKNOWN (p_yes == 0.5), so MI == 0. These
    are out-of-ensemble probes: their answer is decided by a mechanism no theory
    models. Returns their question texts (newest source_step first), to be fed
    back into theory generation as seeds (lever #1, MI-residual variant)."""
    if k <= 0:
        return []
    residual = []
    for d in scoring_log.get("per_question", []):
        pys = d.get("p_yes_per_theory") or []
        if not pys:
            continue
        all_unknown = all(abs(float(p) - 0.5) < 1e-9 for p in pys)
        if all_unknown and float(d.get("score", 1.0)) <= 1e-9:
            residual.append(d)
    # Newest first (a fresh anomaly is the most useful seed); stable otherwise.
    residual.sort(key=lambda d: (d.get("source_step") or 0), reverse=True)
    return [d["question"] for d in residual[:k]]


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
    active_experiment_question: str | None = None,
    phase: str = "complete",
    env_info: dict | None = None,
    critical_cost: float = 0.0,
    did_critical_id: bool = False,
    critical: bool | None = None,
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
        "critical_cost": critical_cost,
        "step_total_cost": (
            agent_cost
            + extract_cost
            + improve_cost
            + experiment_cost
            + trim_cost
            + critical_cost
        ),
        "num_qa_pairs": num_qa,
        "num_answered_questions": num_qa - num_unanswered,
        "num_unanswered_questions": num_unanswered,
        "did_gen_questions": did_gen_questions,
        "did_formulate_experiment": did_formulate_experiment,
        "did_trim": did_trim,
        "did_critical_id": did_critical_id,
        "critical": critical,
        "active_experiment": active_experiment,
        "active_experiment_question": active_experiment_question,
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
    theories: list[Theory] | None = None,
    frontier: list[dict] | None = None,
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
    if theories is not None:
        with open(episode_dir / "theories.json", "w") as f:
            json.dump(serialize_theories(theories), f, indent=4)
    if frontier is not None:
        with open(episode_dir / "frontier.json", "w") as f:
            json.dump(frontier, f, indent=2, default=str)


def _find_last_completed_episode_eb(
    output_dir: str,
) -> tuple[int, str, str, list[EBQAPair], list[Theory], list[dict]]:
    """Find the last completed episode directory and restore EB state.

    Returns: (last_episode, beliefs, perception, qa_pairs, theories, frontier)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return -1, "", "", [], [], []

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
        return -1, "", "", [], [], []

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

    theories: list[Theory] = []
    theory_file = last_dir / "theories.json"
    if theory_file.exists():
        try:
            theories = deserialize_theories(json.loads(theory_file.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass

    frontier: list[dict] = []
    frontier_file = last_dir / "frontier.json"
    if frontier_file.exists():
        try:
            frontier = json.loads(frontier_file.read_text())
        except (json.JSONDecodeError, TypeError):
            pass

    evolve_logger.info(f"Resuming from episode {last_ep} in {last_dir}")
    return last_ep, beliefs, perception, qa_pairs, theories, frontier


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

    max_perception_iters = _resolve_schedule(
        eb_config.max_perception_iterations, global_step
    )
    max_qa_iters = _resolve_schedule(eb_config.max_qa_iterations, global_step)

    # Dynamics mode has no win condition: drop the <policy> section (its guidance
    # is framed around "completing the objective") and any progress/objective
    # wording from the belief-audit prompt.
    goal_aware = is_goal_aware(config)
    include_policy = eb_config.include_policy and goal_aware
    perception_enabled = eb_config.perception_enabled
    perception_instructions = (
        _eb_perception_instructions(include_policy) if perception_enabled else ""
    )
    eb_response_format = _build_eb_response_format(include_policy)
    qa_response_format = _build_eb_qa_response_format(include_policy, perception_enabled)
    beliefs_only_response_format = _build_eb_beliefs_only_response_format(
        include_policy, include_perception_analysis=perception_enabled
    )
    beliefs_section_guidance = _build_beliefs_section_guidance(include_policy)
    if perception_enabled:
        policy_task_phrase = (
            "world knowledge, policy, and perception module"
            if include_policy
            else "world knowledge and perception module"
        )
    else:
        policy_task_phrase = (
            "world knowledge and policy" if include_policy else "world knowledge"
        )
    safety_progress_line = (
        "- Balance safety with progress toward the objective.\n"
        if goal_aware else ""
    )

    hist_window = eb_config.perception_history_window
    display_tail = eb_config.perception_input_tail
    steps_context = format_steps_context(
        trajectory_buffer,
        perception,
        eb_config.max_steps_context_chars,
        history_window=hist_window,
        hide_raw_obs_when_image=eb_config.hide_obs_when_image,
    )
    if eb_config.critical_transitions_enabled:
        crits = _critical_obs_from_buffer(
            trajectory_buffer,
            eb_config.num_sample_obs,
        )
        if len(crits) >= eb_config.critical_id_min_for_perception:
            sample_obs = crits[: eb_config.num_sample_obs]
            evolve_logger.info(
                f"{tag} Track 1b sampling: using {len(sample_obs)} critical transitions "
                f"(>= min {eb_config.critical_id_min_for_perception})"
            )
        else:
            sampled = _middle_last_observations_from_buffer(
                trajectory_buffer,
                eb_config.num_sample_obs,
            )
            seen = {s for _, s in crits}
            padded = crits + [(o, s) for (o, s) in sampled if s not in seen]
            sample_obs = padded[: eb_config.num_sample_obs]
            evolve_logger.info(
                f"{tag} Track 1b sampling: {len(crits)} critical transitions "
                f"(< min {eb_config.critical_id_min_for_perception}); padded to "
                f"{len(sample_obs)} with middle/latest samples"
            )
    else:
        sample_obs = _middle_last_observations_from_buffer(
            trajectory_buffer,
            eb_config.num_sample_obs,
        )
    sample_obs_histories = _histories_for_samples(trajectory_buffer, sample_obs)
    steps_context, steps_context_images = _images_for_steps_context(
        trajectory_buffer,
        steps_context,
        max_images=eb_config.max_images_context,
    )
    sample_obs_images = _images_for_sample_obs(trajectory_buffer, sample_obs)
    track1b_sample_obs = (
        sample_obs[-eb_config.max_images_context :]
        if eb_config.max_images_context > 0
        else []
    )
    track1b_sample_obs_histories = (
        sample_obs_histories[-len(track1b_sample_obs) :] if track1b_sample_obs else []
    )
    # When critical_transitions_enabled, attach only the post-action image for
    # each sampled transition.
    use_post_images = eb_config.critical_transitions_enabled and bool(
        track1b_sample_obs
    )
    if use_post_images:
        track1b_sample_obs_images = _images_for_sample_obs(
            trajectory_buffer,
            track1b_sample_obs,
            post_image_only=True,
        )
    else:
        track1b_sample_obs_images = (
            sample_obs_images[-len(track1b_sample_obs) :] if track1b_sample_obs else []
        )

    num_answered = sum(1 for q in qa_pairs if q.answer is not None)
    evolve_logger.info(
        f"{tag} Improve loop: perception={max_perception_iters}, "
        f"qa={max_qa_iters} iters, "
        f"{len(qa_pairs)} QA ({num_answered} answered), "
        f"{len(steps_context)} chars context"
    )

    # Replay the current perception over the buffer and surface any runtime
    # crashes to the improve prompts. These are the errors the agent silently
    # hit during rollout (failed perceive() => no features for that step).
    def _runtime_perception_errors(current_perception: str) -> list[dict]:
        if not (perception_enabled and current_perception.strip()):
            return []
        return collect_perception_runtime_errors(
            trajectory_buffer, current_perception, history_window=hist_window
        )

    runtime_perc_errors = _runtime_perception_errors(perception)
    if runtime_perc_errors:
        evolve_logger.info(
            f"{tag} Perception runtime errors detected on "
            f"{len(runtime_perc_errors)} buffered observation(s); feeding back "
            f"into improve prompts"
        )
    runtime_errors_block = format_perception_runtime_errors(runtime_perc_errors)
    runtime_errors_section = (
        "\n=== RUNTIME PERCEPTION ERRORS ===\n"
        f"{runtime_errors_block}\n"
        "=== END RUNTIME PERCEPTION ERRORS ===\n"
        if runtime_errors_block
        else ""
    )

    try:
        # ========================================
        # Track 1a: Steps-based beliefs improvement
        # ========================================
        if steps_context:
            track1a_record = {
                "track": "steps_beliefs",
                "step": step,
                "global_step": global_step,
                "turns": [],
            }

            perception_analysis_bullet = (
                "\n- Perception analysis: What information was presented in output of the perception module. What part of that information was helpful, what information was misleading / incorrect and what additional information would have helped if extracted by the perception module?"
                if perception_enabled
                else ""
            )
            step_shows_line = (
                "Each step shows: the pre-action observation, the perception module's output on it, the agent's reasoning, and the action taken."
                if perception_enabled
                else "Each step shows: the pre-action observation, the agent's reasoning, and the action taken."
            )
            steps_beliefs_prompt = f"""We are interacting with an environment and trying to figure out how it works. We maintain beliefs about the environment to guide future actions.

We receive the following default instructions/knowledge:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

We maintain the following current beliefs about the environment:
=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

Below is the actual sequence of the agent's recent interactions with the environment.
{step_shows_line}
Each ``<pre_state>`` (and ``<post_state>``, when present) is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the observation after the last action of an episode segment.

{TRAJECTORY_REASONING_NOTE}

=== SEQUENCE OF STEPS ===
{steps_context}
=== END SEQUENCE OF STEPS ===
{runtime_errors_section}
Your task is to:
1. Analyze the step sequence.
2. Update our beliefs about the environment based on confirmed knowledge from the steps.

Provide analysis highlighting:
- Belief learning: What can we infer from the observations and how should we update our current beliefs so that they more accurately reflect how the world works.{perception_analysis_bullet}

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

            perception_analysis = (
                extract_xml_key(response_text, "perception_analysis") or ""
            )

            turn_record = {
                "turn": 1,
                "cost": turn_cost,
                "prompt": steps_beliefs_prompt,
                "response": response_text,
            }
            track1a_record["turns"].append(turn_record)

            evolve_logger.info(
                f"{tag}     Track 1a done (cost: ${turn_cost:.6f}, perception_analysis: {len(perception_analysis)} chars)"
            )

            feedback_history.append(track1a_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)
        else:
            evolve_logger.info(f"{tag}     Track 1a: No steps context, skipping")
            perception_analysis = ""

        # ========================================
        # Track 1b: Perception improvement guided by beliefs analysis
        # ========================================
        if perception_enabled and track1b_sample_obs:
            track1b_record = {
                "track": "perception_from_analysis",
                "step": step,
                "global_step": global_step,
                "turns": [],
            }
            pre_perception_track1b = perception
            prior_attempts_1b: list[dict] = []

            for turn in range(max_perception_iters):
                evolve_logger.info(
                    f"{tag}     Track 1b (perception from analysis) turn {turn + 1}/{max_perception_iters}"
                )

                # Each turn is self-contained: rebuild the obs section from the
                # current perception and prepend a short outcome log so the LLM
                # knows which prior diffs were accepted/rejected without us
                # carrying the full chat history.
                obs_section_1b = _build_obs_section(
                    perception,
                    track1b_sample_obs,
                    sample_histories=track1b_sample_obs_histories,
                    history_window=hist_window,
                    display_tail=display_tail,
                    post_image_only=use_post_images,
                )
                # Recompute runtime errors against the current perception each
                # turn so the LLM sees whether its previous fix resolved them.
                turn_runtime_errors = _runtime_perception_errors(perception)
                turn_errors_block = format_perception_runtime_errors(
                    turn_runtime_errors
                )
                prior_block = _format_prior_attempts(prior_attempts_1b)
                analysis_parts = [
                    part
                    for part in (
                        prior_block,
                        turn_errors_block,
                        perception_analysis,
                    )
                    if part and part.strip()
                ]
                combined_analysis = "\n\n".join(analysis_parts).strip()

                # Validate any accepted change against the inputs that crashed
                # at runtime (one representative per unique error), in addition
                # to the regular samples — otherwise a "fix" can pass validation
                # while still failing on the exact observation that broke it.
                val_sample_obs = list(track1b_sample_obs)
                val_sample_hists = list(track1b_sample_obs_histories)
                if turn_runtime_errors:
                    failing_reps: list[dict] = []
                    seen_errors: set[str] = set()
                    for err_entry in turn_runtime_errors:
                        if err_entry["error"] in seen_errors:
                            continue
                        seen_errors.add(err_entry["error"])
                        failing_reps.append(err_entry)
                        if len(failing_reps) >= 2:
                            break
                    val_sample_obs = [
                        (e["raw_obs"], e["step"]) for e in failing_reps
                    ] + val_sample_obs
                    val_sample_hists = [
                        e["history"] for e in failing_reps
                    ] + val_sample_hists

                message = build_perception_with_analysis_prompt(
                    beliefs=beliefs,
                    perception=perception,
                    default_knowledge=default_knowledge,
                    obs_section=obs_section_1b,
                    perception_analysis=combined_analysis,
                    max_iterations=max_perception_iters,
                    perception_instructions=perception_instructions,
                    response_format=EB_PERCEPTION_ONLY_RESPONSE_FORMAT,
                )

                prev_perception_1b = perception
                (
                    _beliefs_unused,
                    perception,
                    turn_cost,
                    _conv_unused,
                    response_text,
                    validation_error_1b,
                ) = asyncio.run(
                    _improve_with_perception_validation_conversational(
                        config=config,
                        beliefs=beliefs,
                        perception=perception,
                        conversation_history=[],
                        user_message=message,
                        sample_observations=val_sample_obs,
                        images=track1b_sample_obs_images,
                        sample_histories=val_sample_hists,
                        history_window=hist_window,
                        extraction_mode="diff",
                    )
                )
                total_cost += turn_cost

                prior_attempts_1b.append(
                    {
                        "turn": turn + 1,
                        "validated": validation_error_1b is None,
                        "error": validation_error_1b,
                        "changed": perception != prev_perception_1b,
                    }
                )

                turn_record = {
                    "turn": turn + 1,
                    "cost": turn_cost,
                    "prompt": message,
                    "response": response_text,
                    "validated": validation_error_1b is None,
                    "validation_error": validation_error_1b,
                    "perception_changed": perception != prev_perception_1b,
                }
                submitted = parse_submit_signal(response_text)
                turn_record["submitted"] = submitted
                track1b_record["turns"].append(turn_record)

                evolve_logger.info(
                    f"{tag}     Track 1b turn {turn + 1} done "
                    f"(cost: ${turn_cost:.6f}, submit: {submitted}, "
                    f"validated: {validation_error_1b is None})"
                )

                if submitted:
                    break

            feedback_history.append(track1b_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

            # Rebuild steps_context if perception changed during Track 1b
            if perception != pre_perception_track1b:
                steps_context = format_steps_context(
                    trajectory_buffer,
                    perception,
                    eb_config.max_steps_context_chars,
                    history_window=hist_window,
                    hide_raw_obs_when_image=eb_config.hide_obs_when_image,
                )
                steps_context, steps_context_images = _images_for_steps_context(
                    trajectory_buffer,
                    steps_context,
                    max_images=eb_config.max_images_context,
                )
        elif not perception_enabled:
            evolve_logger.info(f"{tag}     Track 1b: perception disabled, skipping")
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
            track2_record = {
                "track": "qa",
                "step": step,
                "global_step": global_step,
                "turns": [],
            }

            # Convert to QAPair for existing forward/feedback functions
            qa_for_eval = [eb_qa_to_qa(q) for q in answered_qa]

            # Initial QA evaluation
            evolve_logger.info(
                f"{tag}     Track 2: Initial QA forward pass on {len(qa_for_eval)} answered questions..."
            )
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
            qa_inconclusive = [
                fr for fr in qa_fb_results if fr.verdict == "INCONCLUSIVE"
            ]
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
            track2_record["qa_feedback_details"] = serialize_qa_feedback_results(
                qa_fb_results
            )

            pre_track_perception = perception
            # Run the belief audit whenever there is actionable evidence — not only
            # when a prediction was wrong. A belief can be wrong/unsupported even when
            # the agent predicted the (narrow) question correctly, so gating on
            # qa_incorrect would skip exactly those falsifying-but-correctly-predicted
            # items. The prompt lets the model SUBMIT unchanged when nothing needs fixing.
            if qa_actionable:
                # Build initial QA improvement prompt
                qa_blocks = []
                for i, fr in enumerate(qa_actionable, 1):
                    actual = "YES" if fr.forward.qa_pair.answer else "NO"
                    qa_blocks.append(
                        f'<qa_feedback n="{i}">\n'
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

                if perception_enabled:
                    execution_report_section = _build_execution_report_section(
                        perception,
                        sample_obs,
                        sample_histories=sample_obs_histories,
                        history_window=hist_window,
                        display_tail=display_tail,
                    )
                    perception_module_section = f"""
=== CURRENT PERCEPTION MODULE ===
{perception if perception else "(empty - no perception module yet)"}
=== END CURRENT PERCEPTION MODULE ===
{execution_report_section}"""
                    incorrect_analysis_points = """1. Was the agent's world knowledge missing the relevant fact? If so, add it.
2. Was the agent's world knowledge wrong? If so, correct it.
3. Does the perception module need to extract different information to support this knowledge? If so, update it."""
                    reeval_note = "your updated beliefs/perception"
                    improve_subject = "knowledge and perception"
                else:
                    perception_module_section = ""
                    incorrect_analysis_points = """1. Was the agent's world knowledge missing the relevant fact? If so, add it.
2. Was the agent's world knowledge wrong? If so, correct it."""
                    reeval_note = "your updated beliefs"
                    improve_subject = "knowledge"

                initial_qa_prompt = f"""You are improving an agent's {improve_subject} based on testing its understanding of the environment via question-answering.

The agent receives the following default instructions/knowledge:
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===
{perception_module_section}
We tested the agent's understanding by asking it factual questions about the environment.
The agent answered based only on its current world knowledge.

Results: {len(qa_correct)} correct, {len(qa_incorrect)} incorrect out of {len(qa_fb_results)} evaluated.

<qa_feedback_results>
{qa_text}
</qa_feedback_results>

Each ``<pre_state>`` (and ``<post_state>``, when present) in the sequence below is annotated with an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached to this message — use these to cross-reference the textual observation with the actual visual state. ``<pre_state>`` is the observation before the step's action; ``<post_state>`` is the observation after the last action of an episode segment.

{TRAJECTORY_REASONING_NOTE}

=== SEQUENCE OF STEPS (for additional context) ===
{steps_context if steps_context else "(no steps recorded yet)"}
=== END SEQUENCE OF STEPS ===

Your task: Reconcile the agent's {policy_task_phrase} with the QA evidence below.

Treat each question's <correct_answer> and <evidence> as ground truth about how the environment behaves — it was derived directly from the observed trajectory, and it overrides the current beliefs wherever they disagree. This holds even for items the agent predicted CORRECTLY: a belief can still be wrong, vague, or unsupported even when it happened to yield the right yes/no answer (e.g. a narrow or compound question the agent answers correctly while still holding a wrong underlying model). Audit the evidence itself, not just the verdicts.

Belief audit — go through EVERY current belief and check it against the FULL set of evidence above (both correct and incorrect items):
- If the evidence contradicts a belief, REMOVE or REPLACE it. Do not keep a refuted belief alive by bolting on an exception or qualifier — if the core claim is wrong, rewrite the claim or mark the mechanism as unknown.
- If a belief is only weakly supported (it was assumed or inferred, not actually observed), weaken it to match what was observed, or drop it.
- If the evidence reveals a fact that no belief captures, add it.

For INCORRECT predictions specifically, analyze:
{incorrect_analysis_points}

Guidelines:
- Keep each beliefs section to at most 8 concise bullet points. Merge redundant points.
{safety_progress_line}- Prefer replacing a belief over qualifying it: a hedge that preserves a wrong frame ("X is the goal, but only sometimes / needs more") is worse than restating the claim correctly or admitting the mechanism is unknown.
- Remove beliefs that are contradicted by the QA evidence.
- If, after auditing every belief against the evidence, nothing needs to change, leave the beliefs unchanged and SUBMIT.

This is a multi-turn conversation. After each response, the QA pairs will be re-evaluated with {reeval_note}. You can iterate up to {max_qa_iters} turns.

{perception_instructions}

{qa_response_format}"""

                prev_qa_correct = len(qa_correct)
                prev_qa_incorrect = len(qa_incorrect)
                prior_attempts_2: list[dict] = []

                for turn in range(max_qa_iters):
                    evolve_logger.info(
                        f"{tag}     Track 2 turn {turn + 1}/{max_qa_iters}"
                    )

                    if turn == 0:
                        base_message = initial_qa_prompt
                    else:
                        base_message = build_qa_followup_message(
                            qa_fb_results,
                            prev_qa_correct,
                            prev_qa_incorrect,
                            response_format=qa_response_format,
                            perception_enabled=perception_enabled,
                            current_beliefs=beliefs,
                        )

                    # The prior-attempts block reports accepted/rejected perception
                    # diffs; it is meaningless (and references perception) in
                    # beliefs-only mode, so suppress it when perception is disabled.
                    prior_block = (
                        _format_prior_attempts(prior_attempts_2)
                        if perception_enabled
                        else ""
                    )
                    message = (
                        f"{prior_block}\n\n{base_message}" if prior_block else base_message
                    )

                    # History is reset each turn, so always re-attach the images
                    # referenced by the prompt.
                    turn_images = list(steps_context_images) + list(sample_obs_images)

                    prev_perception_2 = perception
                    if perception_enabled:
                        (
                            beliefs,
                            perception,
                            turn_cost,
                            _conv_unused,
                            response_text,
                            validation_error_2,
                        ) = asyncio.run(
                            _improve_with_perception_validation_conversational(
                                config=config,
                                beliefs=beliefs,
                                perception=perception,
                                conversation_history=[],
                                user_message=message,
                                sample_observations=sample_obs if sample_obs else None,
                                images=turn_images,
                                sample_histories=sample_obs_histories
                                if sample_obs
                                else None,
                                history_window=hist_window,
                                allow_keep_perception=True,
                                extraction_mode="diff",
                            )
                        )
                    else:
                        beliefs, turn_cost, _conv_unused, response_text = asyncio.run(
                            _improve_beliefs_only_conversational(
                                config=config,
                                beliefs=beliefs,
                                conversation_history=[],
                                user_message=message,
                                images=turn_images,
                            )
                        )
                        validation_error_2 = None
                    total_cost += turn_cost

                    prior_attempts_2.append(
                        {
                            "turn": turn + 1,
                            "validated": validation_error_2 is None,
                            "error": validation_error_2,
                            "changed": perception != prev_perception_2,
                        }
                    )

                    turn_record = {
                        "turn": turn + 1,
                        "cost": turn_cost,
                        "prompt": message,
                        "response": response_text,
                        "validated": validation_error_2 is None,
                        "validation_error": validation_error_2,
                        "perception_changed": perception != prev_perception_2,
                    }
                    submitted = parse_submit_signal(response_text)
                    turn_record["submitted"] = submitted
                    track2_record["turns"].append(turn_record)

                    evolve_logger.info(
                        f"{tag}     Track 2 turn {turn + 1} done "
                        f"(cost: ${turn_cost:.6f}, submit: {submitted}, "
                        f"validated: {validation_error_2 is None})"
                    )

                    if submitted:
                        evolve_logger.info(
                            f"{tag}     Track 2: LLM submitted after {turn + 1} turn(s)"
                        )
                        break

                    # Re-evaluate QA for next turn (unless this is the last turn)
                    if turn + 1 < max_qa_iters:
                        prev_qa_correct = sum(
                            1 for fr in qa_fb_results if fr.verdict == "CORRECT"
                        )
                        prev_qa_incorrect = sum(
                            1 for fr in qa_fb_results if fr.verdict == "INCORRECT"
                        )

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

                        new_correct = sum(
                            1 for fr in qa_fb_results if fr.verdict == "CORRECT"
                        )
                        new_incorrect = sum(
                            1 for fr in qa_fb_results if fr.verdict == "INCORRECT"
                        )
                        evolve_logger.info(
                            f"{tag}     Track 2 re-eval: {new_correct} correct "
                            f"({new_correct - prev_qa_correct:+d}), "
                            f"{new_incorrect} incorrect ({new_incorrect - prev_qa_incorrect:+d})"
                        )
            else:
                evolve_logger.info(
                    f"{tag}     Track 2: No actionable QA, skipping improvement"
                )

            feedback_history.append(track2_record)
            if step_dir is not None:
                _flush_improve_progress(step_dir, feedback_history, beliefs, perception)

            # Rebuild steps_context if perception changed during Track 2
            if perception != pre_track_perception:
                steps_context = format_steps_context(
                    trajectory_buffer,
                    perception,
                    eb_config.max_steps_context_chars,
                    history_window=hist_window,
                    hide_raw_obs_when_image=eb_config.hide_obs_when_image,
                )
                steps_context, steps_context_images = _images_for_steps_context(
                    trajectory_buffer,
                    steps_context,
                    max_images=eb_config.max_images_context,
                )

    except Exception as e:
        evolve_logger.error(f"{tag}     Improve loop failed: {e}")
        logging.exception("Improve loop failed")
        feedback_history.append(
            {"error": str(e), "step": step, "global_step": global_step}
        )
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
        f"Q{pos + 1} (score={top_score:.6f}, source_step={qa_pairs[src_i].source_step}): "
        f"{qa_pairs[src_i].question}"
        for pos, src_i in enumerate(tied_source_indices)
    )
    default_knowledge_section = ""
    if default_knowledge:
        default_knowledge_section = f"""
=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===
"""
    n_tied = len(tied_source_indices)
    selection_criterion = (
        "help achieve the overall objective for the environment"
        if is_goal_aware(config)
        else "deepen understanding of how the environment works"
    )
    prompt = f"""You are selecting the next question target for learning about how a game works.

{default_knowledge_section}

=== CURRENT BELIEFS ===
{beliefs}
=== END CURRENT BELIEFS ===

=== AVAILABLE QUESTIONS ===
{tied_questions_text}
=== AVAILABLE QUESTIONS ===

Select questions that will
1. That will {selection_criterion}
2. Cover distinct aspects of the environment

Use each question's Q number in the <q n="..."> attribute. Format your response as:
<think>
Which questions should be selected?
</think>
<selected_question>
<q n="Q1" />
</selected_question>"""

    text, cost = await _llm_call(config, prompt)
    selected_text = extract_xml_key(text, "selected_question") or ""
    selected_source_index: int | None = None
    # Q numbers in the prompt are 1-based positions in tied_source_indices (Q1, Q2, ...)
    for pos in _parse_q_tag_indices(selected_text, n_tied):
        selected_source_index = tied_source_indices[pos]
        break
    if selected_source_index is None:
        for match in re.finditer(
            r"(?<![A-Za-z_])(?:Q\s*)?(\d+)(?![A-Za-z_])",
            selected_text,
            re.IGNORECASE,
        ):
            pos = int(match.group(1)) - 1
            if 0 <= pos < n_tied:
                selected_source_index = tied_source_indices[pos]
                break

    parse_error = None
    if selected_source_index is None:
        selected_source_index = tied_source_indices[0]
        parse_error = (
            "No valid tied source_index parsed; fell back to first ranked tied question"
        )

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


async def identify_critical_transition(
    *,
    config: DictConfig,
    beliefs: str,
    perception_code: str,
    default_knowledge: str,
    steps_context: str,
    steps_context_images: list,
    current_experiment: str | None = None,
    current_experiment_question: str | None = None,
    perception_enabled: bool = True,
) -> tuple[bool, str, float, str, str]:
    """Decide whether the current state warrants pausing to update artifacts.

    The state is "critical" when either (1) the recent transitions are
    surprising or revealing given current beliefs + perception output, or
    (2) the current experiment is stale and should be replaced with a new
    one. Critical states trigger belief/perception updates and new experiment
    selection downstream.

    Returns: (is_critical, reason, cost, prompt, response_text).
    """
    if perception_enabled:
        update_target = "beliefs/perception"
        criterion1 = "(1) The recent state transitions demonstrate something new or surprising about the environment given current beliefs + perception module output (e.g. a post-state violates current beliefs, exposes a gap in the perception module, or reveals something the agent did not previously know)."
        surprise_basis = "given current beliefs and perception output"
        gap_phrase = "belief or perception gap"
    else:
        update_target = "beliefs"
        criterion1 = "(1) The recent state transitions demonstrate something new or surprising about the environment given current beliefs (e.g. a post-state violates current beliefs or reveals something the agent did not previously know)."
        surprise_basis = "given current beliefs"
        gap_phrase = "belief gap"
    default_knowledge_section = ""
    if default_knowledge:
        default_knowledge_section = f"""=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===
"""

    if current_experiment:
        question_line = (
            f"Question being investigated:\n{current_experiment_question}\n\n"
            if current_experiment_question
            else ""
        )
        current_experiment_section = f"""=== CURRENT EXPERIMENT ===
{question_line}Experiment plan:
{current_experiment}
=== END CURRENT EXPERIMENT ===
"""
    else:
        current_experiment_section = """=== CURRENT EXPERIMENT ===
(none - no experiment currently active)
=== END CURRENT EXPERIMENT ===
"""

    image_legend = ""
    if steps_context_images:
        image_legend = (
            "Each ``<pre_state>`` (and ``<post_state>``, when present) is annotated with "
            "an ``(image K)`` marker referring to the K-th (1-indexed) screenshot attached "
            "to this message — use these to cross-reference the textual observation with "
            "the actual visual state. ``<pre_state>`` is the observation before the step's "
            "action; ``<post_state>`` is the observation after the last action of an episode "
            "segment.\n\n"
        )

    prompt = f"""We are interacting with an environment and trying to figure out how it works. The agent is currently pursuing an experiment to answer a specific question about the environment. After every action we decide whether the current state is *critical* — meaning we should pause to update {update_target} and (potentially) replace the current experiment — or *uninformative* — meaning we should keep executing the current experiment.

Mark the state as critical if EITHER of the following holds:
{criterion1}
(2) The current experiment is stale and needs to be replaced with a new one — for example, the question has effectively been answered by the evidence gathered so far, the experiment is no longer relevant given what we now believe, or the experiment is not making progress and a different question would be more valuable to investigate now.

Otherwise — if the recent transitions are predictable from current beliefs AND the current experiment is still worth executing — mark it not critical.

{default_knowledge_section}
=== CURRENT BELIEFS ===
{beliefs if beliefs else "(empty - no beliefs yet)"}
=== END CURRENT BELIEFS ===

{current_experiment_section}
{image_legend}=== SEQUENCE OF STEPS ===
{steps_context}
=== END SEQUENCE OF STEPS ===

Focus on the *most recent transitions* (the last few steps shown above): the agent's pre-action observations, the actions taken, and the resulting post-states. Also consider whether continuing to pursue the current experiment is still the best use of the agent's effort given what has been observed.

Decide:
- Were the recent transitions surprising or revealing {surprise_basis}, or were they predictable?
- Is the current experiment still worth pursuing, or has it become stale (answered, irrelevant, or stalled)?

Respond exactly in this format:
<analysis>
Brief analysis covering: (a) whether the recent transitions are surprising vs predictable given current beliefs (identify the specific {gap_phrase} if any); (b) whether the current experiment is still worth pursuing or has become stale.
</analysis>
<critical>yes</critical>
<reason>One sentence on why critical (and which of the two criteria applies), or why not.</reason>

Use <critical>yes</critical> or <critical>no</critical>."""

    text, cost = await _llm_call(config, prompt, images=steps_context_images)
    parsed = extract_xml_kv(text, ["analysis", "critical", "reason"])
    raw_critical = (parsed.get("critical") or "").strip().lower()
    if raw_critical.startswith("yes"):
        is_critical = True
    elif raw_critical.startswith("no"):
        is_critical = False
    else:
        evolve_logger.warning(
            f"identify_critical_transition: could not parse <critical> tag "
            f"(got {raw_critical!r}); defaulting to True (fail-open)."
        )
        is_critical = True
    reason = (parsed.get("reason") or "").strip()
    return is_critical, reason, cost, prompt, text


# ---------------------------------------------------------------------------
# GEPA/legacy frontier relearn (question_scoring_method == "gepa_frontier").
# Learns a frontier (set of competing {perception, world_knowledge} candidates)
# from the collected trajectory via an inverse-dynamics objective, mirroring
# prototypes/perc_invdyn/explore_loop.py::relearn(). The frontier is then the
# "set of possible Bs": its top candidate feeds the agent (B + P) and the
# candidates' disagreement drives experiment formulation.
# ---------------------------------------------------------------------------
def _clean_frontier_perception(text: str) -> str:
    """Strip markdown fences / prose from a GEPA candidate's perception code."""
    import gepa_optimize as G  # lazy: only needed in frontier mode

    return G._clean_component("perception", text or "")


def _normalize_action(raw: str) -> str:
    """Canonical action label for the inverse-dynamics target.

    Single-token (move) actions collapse to the verb. Parametric actions
    (e.g. ARC ``ACTION6 x=10 y=20`` or autumn ``click 3 4``) keep their
    coordinates as ``"<verb> <n1> <n2> ..."`` so clicks at different cells are
    distinct learnable targets. Integers are read from the ARGS only (so the
    digit in a verb like ``ACTION6`` is never mistaken for a coordinate), and an
    uninstantiated template (``click ROW COL``) with no concrete numbers
    collapses back to the verb.
    """
    parts = str(raw).split()
    if not parts:
        return ""
    verb = parts[0]
    nums = re.findall(r"-?\d+", " ".join(parts[1:]))
    if not nums:
        return verb
    return verb + " " + " ".join(nums)


def _is_parametric_action(canonical: str) -> bool:
    """True for a coordinate-bearing (click) action in canonical form."""
    return len(canonical.split()) > 1


def _bake_choices_clickaware(transitions, pool, k, rng):
    """Choice sets for inverse dynamics with coordinate-aware HARD NEGATIVES.

    For a click transition the distractors are preferentially *other observed
    click locations* (same verb first, then any other click), so the MCQ cannot
    be solved by verb alone and the learner is forced to localize the click.
    Move transitions fall back to generic distractors (identical to the shared
    ``make_choices`` behavior). Emits the same ``{tr, choices}`` baked dicts the
    GEPA adapter / ``eval_on`` consume."""
    click_actions = [a for a in pool if _is_parametric_action(a)]
    noncl = [a for a in pool if not _is_parametric_action(a)]
    baked = []
    for tr in transitions:
        true = tr.action
        if _is_parametric_action(true):
            verb = true.split()[0]
            same = [a for a in click_actions if a != true and a.split()[0] == verb]
            other = [a for a in click_actions if a != true and a.split()[0] != verb]
            rng.shuffle(same)
            rng.shuffle(other)
            spare = [a for a in noncl]
            rng.shuffle(spare)
            distractors = same + other + spare
        else:
            distractors = [a for a in pool if a != true]
            rng.shuffle(distractors)
        choices = [true] + distractors[: max(0, k - 1)]
        rng.shuffle(choices)
        baked.append({"tr": tr, "choices": choices})
    return baked


def _balanced_split_by_verb(transitions, holdout_n, rng):
    """Like validate_beliefs.balanced_split but balances by VERB, not by exact
    action — so a held-out split stays representative when many click
    coordinates are singletons. Returns (rest, holdout)."""
    by_verb = defaultdict(list)
    for t in transitions:
        by_verb[t.action.split()[0]].append(t)
    for v in by_verb.values():
        rng.shuffle(v)
    verbs = list(by_verb)
    holdout = []
    i = 0
    while len(holdout) < holdout_n and any(by_verb[v] for v in verbs):
        v = verbs[i % len(verbs)]
        if by_verb[v]:
            holdout.append(by_verb[v].pop())
        i += 1
    rest = [t for v in verbs for t in by_verb[v]]
    rng.shuffle(rest)
    return rest, holdout


def _transitions_from_buffer(trajectory_buffer: list[dict], click_aware: bool = True):
    """Build inverse-dynamics Transitions (X_t, action, X_{t+1}) from the rollout
    buffer. Each acted entry already carries pre- and post-action raw
    observations, so no consecutive-entry pairing is needed (episode boundaries
    have action=None and are skipped).

    ``click_aware`` keeps coordinates on parametric actions (see
    ``_normalize_action``); when False, every action collapses to its verb."""
    from validate import Transition  # lazy: only needed in frontier mode

    transitions = []
    for e in trajectory_buffer:
        if e.get("episode_boundary") or not e.get("action"):
            continue
        x_t = (e.get("raw_long_term_context") or "").strip()
        x_t1 = (e.get("result_raw_long_term_context") or "").strip()
        if not x_t or not x_t1:
            continue
        raw = str(e["action"])
        label = _normalize_action(raw) if click_aware else raw.split()[0]
        if not label:
            continue
        transitions.append(Transition(x_t, x_t1, label))
    return transitions


def _img_transitions_from_buffer(trajectory_buffer: list[dict], click_aware: bool = True):
    """Build IMAGE inverse-dynamics transitions (frame_t, action, frame_{t+1})
    from the buffer's pre/post-action PIL images. For image-native envs (ARC)
    whose text obs has no grid. Entries lacking either image are skipped."""
    from legacy_pop import ImgTransition  # lazy: only in frontier image mode

    transitions = []
    for e in trajectory_buffer:
        if e.get("episode_boundary") or not e.get("action"):
            continue
        img_t = e.get("image")
        img_t1 = e.get("result_image")
        if img_t is None or img_t1 is None:
            continue
        raw = str(e["action"])
        label = _normalize_action(raw) if click_aware else raw.split()[0]
        if not label:
            continue
        transitions.append(ImgTransition(label, img_t, img_t1))
    return transitions


def _relearn_frontier_eb(
    config: DictConfig,
    eb_config: StepwiseEBLearnConfig,
    trajectory_buffer: list[dict],
    env_name: str,
    current_frontier: list[dict],
    rng: random.Random,
):
    """Run GEPA (or legacy greedy) on the buffer; return (frontier, metric, info, cost).

    frontier = up to ``frontier_size`` distinct top-val candidates (best first).
    metric   = best candidate's inverse-dynamics accuracy on a held-out test
               split that neither optimization nor selection ever touched.
    On any skip (too little data) the current frontier is returned unchanged.
    """
    if eb_config.mock_mode:
        # Mock mode short-circuits LLM calls at the mixed_improve / BALROG client
        # layers, but the GEPA/validate pipeline uses its own client that is NOT
        # gated. Synthesize a small frontier instead of issuing real API calls so
        # the branch plumbing (re-inject + experiment + persistence) is exercised.
        synth = [
            {"perception": "", "world_knowledge": f"mock world knowledge (candidate {i + 1})"}
            for i in range(max(2, eb_config.frontier_size))
        ]
        return synth, 0.0, {"mock": True, "frontier_size": len(synth)}, 0.0

    import gepa  # lazy: only needed in frontier mode
    import gepa_optimize as G
    from validate import make_config

    click_aware = eb_config.frontier_click_aware

    # --- Image-native B learner (ARC: obs is an image with no text grid) ---
    has_images = any(
        e.get("image") is not None and e.get("result_image") is not None
        for e in trajectory_buffer
    )
    image_mode = (eb_config.frontier_learner == "legacy_pop") and (
        eb_config.frontier_image_mode == "on"
        or (eb_config.frontier_image_mode == "auto" and has_images)
    )
    if image_mode:
        from legacy_pop import run_legacy_pop_img

        img_trans = _img_transitions_from_buffer(
            trajectory_buffer, click_aware=click_aware
        )
        cap = eb_config.frontier_image_max_transitions
        if len(img_trans) > cap:  # bound vision cost: keep the most recent
            img_trans = img_trans[-cap:]
        img_pool = sorted({t.action for t in img_trans})
        if len(img_pool) < 2 or len(img_trans) < eb_config.frontier_min_buffer:
            return (
                current_frontier, None,
                {"skipped": f"image transitions={len(img_trans)} "
                            f"distinct_actions={len(img_pool)}"},
                0.0,
            )
        k = eb_config.frontier_k_choices
        _bake_img = _bake_choices_clickaware if click_aware else (
            lambda tr, p, kk, r: G.bake_choices(tr, p, kk, r)
        )
        baked = _bake_img(img_trans, img_pool, k, rng)  # val == train (tied)
        task_cfg = make_config(
            eb_config.frontier_task_model or config.client.model_id,
            config.client.client_name,
        )
        sem = asyncio.Semaphore(eb_config.frontier_concurrency)
        pop_frontier, pcost = asyncio.run(
            run_legacy_pop_img(
                task_cfg, baked, sem, seed=0,
                rounds=eb_config.frontier_pop_rounds,
                pop_size=eb_config.frontier_pop_size,
            )
        )
        frontier = pop_frontier[: eb_config.frontier_size]
        top = frontier[0]
        info = {
            "learner": "legacy_pop", "image_mode": True, "click_aware": click_aware,
            "transitions": len(img_trans), "pool": img_pool,
            "train": len(baked), "val": len(baked),
            "pop_size": eb_config.frontier_pop_size,
            "distinct_B": len({c["world_knowledge"].strip() for c in frontier}),
            "val_accs": [round(c.get("val_acc", 0.0), 3) for c in frontier],
            "cost": round(pcost, 4), "frontier_size": len(frontier),
        }
        return frontier, top.get("val_acc"), info, pcost

    transitions = _transitions_from_buffer(trajectory_buffer, click_aware=click_aware)
    pool = sorted({t.action for t in transitions})
    # Need >=2 distinct actions to form a non-trivial inverse-dynamics choice
    # set. In click-aware mode distinct *coordinates* count (a pure-click game
    # like ft09 has one verb but many discriminable click targets), so gate on
    # the number of distinct full actions, not verbs.
    n_verbs = len({a.split()[0] for a in pool})
    if len(pool) < 2 or len(transitions) < eb_config.frontier_min_buffer:
        return (
            current_frontier,
            None,
            {
                "skipped": f"transitions={len(transitions)} "
                f"distinct_actions={len(pool)} verbs={n_verbs}"
            },
            0.0,
        )

    # Coordinate-aware mode balances the split by verb (clicks are mostly
    # singletons) and bakes HARD-NEGATIVE click choice sets; otherwise use the
    # shared verb-only split/baker.
    _split = _balanced_split_by_verb if click_aware else (
        lambda d, n, r: G.balanced_split(d, n, 10**9, r)
    )
    _bake = _bake_choices_clickaware if click_aware else (
        lambda tr, p, kk, r: G.bake_choices(tr, p, kk, r)
    )

    data = list(transitions)
    rng.shuffle(data)
    k = eb_config.frontier_k_choices
    # Low-data regime: val = train = the FULL experience so far, no held-out test
    # split and no separate test eval. This matches the legacy_pop prototype that
    # validated this approach; carving disjoint val/test starves the tiny buffer
    # and makes pareto-by-example selection meaningless (val too small).
    train_tr = val_tr = test_tr = data
    if not train_tr:
        return current_frontier, None, {"skipped": "empty split"}, 0.0

    client_name = config.client.client_name
    task_cfg = make_config(
        eb_config.frontier_task_model or config.client.model_id, client_name
    )
    refl_cfg = make_config(
        eb_config.frontier_reflection_model or config.client.model_id, client_name
    )

    test = _bake(test_tr, pool, k, rng)
    cost = 0.0

    if eb_config.frontier_learner == "legacy":
        # Greedy P/B loop -> a single best candidate (frontier of size 1). Uses
        # val for selection; metric is measured on the untouched test split.
        # NOTE: run_legacy_loop bakes its own choice sets via the shared
        # make_choices (generic distractors), so it does not get the
        # coordinate-aware hard negatives that the GEPA branch / test split do.
        sem = asyncio.Semaphore(eb_config.frontier_concurrency)
        best_code, best_beliefs, lcost = asyncio.run(
            G.run_legacy_loop(
                task_cfg, train_tr, val_tr, pool, k, sem, rng,
                eb_config.frontier_legacy_rounds, start_code="",
            )
        )
        cost += lcost
        frontier = [{"perception": best_code, "world_knowledge": best_beliefs}]
        metric, _ = asyncio.run(G.eval_on(task_cfg, best_code, best_beliefs, test))
        info = {
            "learner": "legacy", "click_aware": click_aware,
            "transitions": len(transitions), "pool": pool, "n_verbs": n_verbs,
            "train": len(train_tr), "val": len(val_tr), "test": len(test),
            "cost": round(cost, 4), "frontier_size": len(frontier),
        }
        return frontier, metric, info, cost

    if eb_config.frontier_learner == "legacy_pop":
        # Population of {P,B} candidates, each mutated by legacy's G1 failure-
        # directed update, selected by pareto-by-example on a fixed-choice val
        # set -> a DIVERSE frontier (the disagreement that drives experiments)
        # at lower cost than GEPA's reflective search. (prototype: legacy_pop.py)
        from legacy_pop import run_legacy_pop

        val = _bake(val_tr, pool, k, rng)
        sem = asyncio.Semaphore(eb_config.frontier_concurrency)
        pop_frontier, pcost = asyncio.run(
            run_legacy_pop(
                task_cfg, train_tr, val, pool, k, sem, seed=0,
                rounds=eb_config.frontier_pop_rounds,
                pop_size=eb_config.frontier_pop_size, start_code="",
            )
        )
        cost += pcost
        # run_legacy_pop returns dicts already shaped {perception, world_knowledge, val_acc}
        frontier = pop_frontier[: eb_config.frontier_size]
        top = frontier[0]
        # No held-out test eval: metric = top candidate's (in-sample) val accuracy.
        metric = top.get("val_acc")
        info = {
            "learner": "legacy_pop", "click_aware": click_aware,
            "transitions": len(transitions), "pool": pool, "n_verbs": n_verbs,
            "train": len(train_tr), "val": len(val),
            "pop_size": eb_config.frontier_pop_size,
            "distinct_B": len({c["world_knowledge"].strip() for c in frontier}),
            "distinct_P": len({c["perception"].strip() for c in frontier}),
            "val_accs": [round(c.get("val_acc", 0.0), 3) for c in frontier],
            "cost": round(cost, 4), "frontier_size": len(frontier),
        }
        return frontier, metric, info, cost

    # GEPA (pareto) branch -- mirrors explore_loop.relearn.
    train = _bake(train_tr, pool, k, rng)
    val = _bake(val_tr, pool, k, rng)
    adapter = G.InvDynAdapter(
        task_cfg, pool, concurrency=eb_config.frontier_concurrency,
        fd_scorer=eb_config.frontier_fd_scorer, fd_weight=eb_config.frontier_fd_weight,
    )
    seed_candidate = {"perception": "", "world_knowledge": ""}
    result = gepa.optimize(
        seed_candidate=seed_candidate, trainset=train, valset=val, adapter=adapter,
        reflection_lm=G.make_reflection_lm(refl_cfg),
        reflection_prompt_template=G.build_reflection_templates(env_name),
        candidate_selection_strategy="pareto", module_selector="round_robin",
        reflection_minibatch_size=min(len(train), 12),
        max_metric_calls=eb_config.frontier_max_metric_calls,
        display_progress_bar=False, seed=0, cache_evaluation=True,
        raise_on_exception=False, track_best_outputs=True,
    )
    cost += adapter.total_cost
    # frontier = distinct candidates ranked by val aggregate score (best first).
    order = sorted(
        range(len(result.candidates)),
        key=lambda i: result.val_aggregate_scores[i], reverse=True,
    )
    frontier, seen = [], set()
    for i in order:
        c = result.candidates[i]
        key = (c.get("perception", ""), c.get("world_knowledge", ""))
        if key in seen:
            continue
        seen.add(key)
        frontier.append(c)
        if len(frontier) >= eb_config.frontier_size:
            break

    metric, _ = asyncio.run(
        G.eval_on(
            task_cfg,
            G._clean_component("perception", result.best_candidate.get("perception", "")),
            result.best_candidate.get("world_knowledge", ""),
            test,
        )
    )
    info = {
        "learner": "gepa", "click_aware": click_aware,
        "transitions": len(transitions), "pool": pool, "n_verbs": n_verbs,
        "train": len(train), "val": len(val), "test": len(test),
        "candidates": result.num_candidates,
        "metric_calls": result.total_metric_calls,
        "cost": round(cost, 4), "frontier_size": len(frontier),
    }
    return frontier, metric, info, cost


def run_stepwise_eb_learn_episode(
    config: DictConfig,
    eb_config: StepwiseEBLearnConfig,
    beliefs: str,
    perception: str,
    qa_pairs: list[EBQAPair],
    current_experiment: str | None,
    current_experiment_question: str | None,
    default_knowledge: str,
    output_dir: str,
    episode_idx: int = 0,
    global_step_start: int = 0,
    max_episode_steps: int | None = None,
    trajectory_buffer: list[dict] | None = None,
    past_experiments: list[str] | None = None,
    agent_history_events: list[dict] | None = None,
    cumulative_cost_offset: float = 0.0,
    theories: list[Theory] | None = None,
    frontier: list[dict] | None = None,
) -> tuple[
    str,
    str,
    list[EBQAPair],
    str | None,
    str | None,
    dict,
    int,
    list[dict],
    list[str],
    list[dict],
    list[Theory],
    list[dict],
]:
    """Run a single episode with per-step EB-learning.

    Returns:
        (beliefs, perception, qa_pairs, current_experiment,
         current_experiment_question, episode_stats, steps_taken, trajectory_buffer,
         past_experiments, agent_history_events, theories, frontier)

    ``frontier`` carries the GEPA/legacy "set of possible Bs" (only used when
    ``question_scoring_method == "gepa_frontier"``; otherwise empty).

    ``theories`` carries the Plan A persistent theory ensemble (only used when
    ``question_scoring_method == "theory_disagreement"``; otherwise empty).
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
    _restore_agent_history_events(agent, agent_history_events)

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
    agent_goal = resolve_agent_goal(config)
    # Dynamics mode has no win condition: theory generation and action selection
    # must not inject any progress/win framing, and exploit is disabled.
    goal_aware = is_goal_aware(config)

    # ARC-AGI exposes a game-specific (and sometimes mid-episode-changing) action
    # set. The run-level default_knowledge was built with the generic full action
    # list, so rebuild it from the live env's actions. This keeps every prompt
    # that embeds default_knowledge (experiment generation, question scoring,
    # belief improvement) referencing only actions the agent can actually take.
    if env_name == "arc_agi":
        live_actions = getattr(env, "language_action_space", None)
        if live_actions:
            default_knowledge = append_agent_goal(
                get_default_knowledge(config, available_actions=list(live_actions)),
                agent_goal,
            )

    _inject_beliefs(config, agent, env, env_name, task, beliefs, agent_goal=agent_goal)
    agent.experiment_goal = current_experiment

    # Save raw initial obs before apply_perception modifies long_term_context in-place
    _pre_action_raw_long = obs["text"]["long_term_context"]
    _pre_action_raw_short = obs["text"].get("short_term_context", "")
    _pre_action_image = obs.get("image")  # PIL Image or None

    # Per-episode raw observation history for history-aware perception modules.
    raw_obs_history: list[str] = [_pre_action_raw_long]

    # Setup perception
    perception_fn = (load_perception_fn(perception) if eb_config.perception_enabled else None)
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
    max_steps = (
        env.max_steps
        if config.eval.get("max_steps_per_episode") is None
        else config.eval.max_steps_per_episode
    )
    if max_episode_steps is not None:
        max_steps = min(max_steps, max_episode_steps)
    episode_log: dict = {
        "task": task,
        "agent_goal": agent_goal,
        "action_frequency": defaultdict(int),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
    }

    trajectory_buffer = trajectory_buffer if trajectory_buffer is not None else []
    # Insert episode boundary marker
    if trajectory_buffer:
        trajectory_buffer.append(
            {
                "step": None,
                "episode_boundary": True,
                "episode_idx": episode_idx,
                "obs_text": "",
                "raw_long_term_context": "",
                "action": None,
                "reward": 0.0,
                "reasoning": "",
                "done": True,
            }
        )
    total_learn_cost = 0.0
    episode_return = 0.0
    cumulative_step_cost = 0.0
    step_extraction_log: dict | None = None
    step_experiment_log: dict | None = None
    past_experiments = past_experiments if past_experiments is not None else []
    if current_experiment and current_experiment not in past_experiments:
        past_experiments.append(current_experiment)

    # --- Plan A (theory_disagreement) persistent state ---
    # ``theories`` is the carried posterior over world-models; the pending_*
    # vars hold the discriminating action's pre-registered predictions so the
    # next observation can reweight the ensemble. ``theory_needs_regen`` starts
    # True so the ensemble is generated before the first action.
    plan_a_mode = eb_config.question_scoring_method == "theory_disagreement"
    theories = theories if theories is not None else []
    theory_needs_regen = True
    pending_predictions: dict[int, str] = {}
    pending_action_plan: str | None = None
    # Accumulated falsification evidence (claim + observed contradiction) for
    # every theory ruled out this episode. Fed back into regeneration so fresh
    # theories must explain what actually happened and cannot re-propose a
    # just-falsified mechanic. Bounded to the most recent entries.
    falsification_memory: list[dict] = []
    falsification_memory_cap = 16
    # Explore->exploit switch state. ``model_stable_streak`` counts consecutive
    # steps the ensemble survived without an all-violated wipe; once it reaches
    # ``exploit_stable_streak`` (and >= exploit_min_theories remain), the loop
    # acts toward the goal under the MAP theory instead of running experiments.
    model_stable_streak = 0
    exploit_mode = False

    # --- GEPA/legacy frontier mode (gepa_frontier) persistent state ---
    # ``frontier`` is the carried "set of possible Bs": a ranked list of
    # competing {perception, world_knowledge} candidates learned from experience.
    # Re-fit on a cadence; the top candidate feeds the agent's B + P.
    frontier_mode = eb_config.question_scoring_method == "gepa_frontier"
    frontier = frontier if frontier is not None else []
    frontier_metric: float | None = None

    # CSV logging
    ep_dir = Path(output_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = ep_dir / "trajectory.csv"

    pbar = tqdm(
        total=max_steps,
        desc=f"Stepwise EB-learn ep {episode_idx}",
        leave=False,
        dynamic_ncols=True,
    )
    feedback_history: list[dict] = []

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(
            csv_file, escapechar="\u02d8", quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(
            [
                "Step",
                "Action",
                "Reasoning",
                "Observation",
                "Auxiliary_Observation",
                "Reward",
                "Done",
            ]
        )

        action = None
        step = 0
        result_obs_text: str | None = None
        new_raw_short: str = ""
        done = False
        # Critical-transition gate (see eb_config.critical_transitions_enabled).
        # Carried across iterations: a critical decision in step n gates the
        # next step's experiment-generation; init True so step 0 runs
        # experiment-gen before the agent's first action.
        critical_flag_for_experiment_gen = True

        for step in range(max_steps):
            global_step = global_step_start + step

            # Per-step directory
            step_dir = ep_dir / f"step_{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            step_extract_cost = 0.0
            step_improve_cost = 0.0
            step_experiment_cost = 0.0
            step_trim_cost = 0.0
            step_critical_cost = 0.0
            step_extraction_log = None
            step_experiment_log = None
            step_trim_log: dict | None = None
            did_trim_step = False
            critical_this_step = False
            did_critical_id_this_step = False
            critical_id_log: dict | None = None
            step_feedback_records: list[dict] = []
            num_unanswered = sum(1 for q in qa_pairs if q.answer is None)

            # Write preliminary step_log
            _save_step_log_eb(
                step_dir=step_dir,
                step=step,
                global_step=global_step,
                action=None,
                reward=0.0,
                done=False,
                episode_return=episode_return,
                agent_cost=0.0,
                extract_cost=0.0,
                improve_cost=0.0,
                experiment_cost=0.0,
                trim_cost=0.0,
                num_qa=len(qa_pairs),
                num_unanswered=num_unanswered,
                did_gen_questions=False,
                did_formulate_experiment=False,
                active_experiment=current_experiment,
                phase="started",
                active_experiment_question=current_experiment_question,
            )

            # --- Question generation + experiment formulation ---
            if eb_config.critical_transitions_enabled:
                should_gen_experiments = critical_flag_for_experiment_gen
            else:
                should_gen_experiments = step % eb_config.experiment_interval == 0
            did_gen_questions = False
            did_formulate_experiment = False

            if plan_a_mode:
                # --- Plan A: persistent theory ensemble -> discriminating action ---
                # Runs every step (the discriminating action is a single action);
                # the theory ensemble is only (re)generated on cold start, on a
                # surprise flagged by the previous posterior update, or when it
                # has shrunk below N. Replaces question-gen / scoring / formulation.
                evolve_logger.info(
                    f"[g{global_step}] Plan A: selecting discriminating action "
                    f"(ensemble size {len(theories)})..."
                )
                pa_steps_context = format_steps_context(
                    trajectory_buffer,
                    perception,
                    eb_config.max_steps_context_chars,
                    history_window=eb_config.perception_history_window,
                    hide_raw_obs_when_image=eb_config.hide_obs_when_image,
                    include_trailing_state=False,
                )
                pa_steps_context, pa_steps_context_images = _images_for_steps_context(
                    trajectory_buffer,
                    pa_steps_context,
                    max_images=eb_config.max_images_context,
                )
                theory_gen_log = None
                action_obj = None
                sel_log = None
                with improve_logging(step_dir):
                    # --- Decide explore vs exploit from the carried posterior ---
                    # ``theories`` is reindexed by weight after each update, so
                    # theories[0] is the current MAP theory. Enter exploit once the
                    # ensemble has been stable (no all-violated wipe) for
                    # ``exploit_stable_streak`` steps; stay until a single
                    # violation/wipe drops the streak to 0 (hysteresis).
                    map_t = theories[0] if theories else None
                    can_exploit = (
                        eb_config.exploit_enabled
                        and goal_aware
                        and map_t is not None
                        and len(theories) >= eb_config.exploit_min_theories
                    )
                    if exploit_mode:
                        exploit_mode = can_exploit and model_stable_streak >= 1
                    else:
                        exploit_mode = (
                            can_exploit
                            and model_stable_streak >= eb_config.exploit_stable_streak
                        )

                    if exploit_mode and map_t is not None:
                        # EXPLOIT: act toward the goal under the MAP theory (no
                        # regeneration; the MAP prediction keeps the posterior live).
                        action_obj, sel_cost, sel_log = asyncio.run(
                            select_goal_action(
                                config=config,
                                map_theory=map_t,
                                beliefs=beliefs,
                                default_knowledge=default_knowledge,
                                goal=agent_goal,
                                goal_aware=goal_aware,
                                steps_context=pa_steps_context,
                                current_observation=_pre_action_raw_long,
                                current_image=_pre_action_image,
                                steps_context_images=pa_steps_context_images,
                            )
                        )
                        step_experiment_cost += sel_cost
                        total_learn_cost += sel_cost
                    else:
                        # EXPLORE: (re)generate theories then run a discriminating
                        # experiment. Genuine cold start only when there is nothing
                        # to learn from yet; once falsification evidence exists,
                        # route regeneration through refill so fresh theories see
                        # what was ruled out (init has no falsification memory).
                        if not theories and not falsification_memory:
                            theories, t_cost, theory_gen_log = asyncio.run(
                                init_theory_ensemble(
                                    config=config,
                                    beliefs=beliefs,
                                    default_knowledge=default_knowledge,
                                    num_theories=eb_config.num_theories,
                                    decay=eb_config.theory_weight_decay,
                                    steps_context=pa_steps_context,
                                    current_observation=_pre_action_raw_long,
                                    current_image=_pre_action_image,
                                    steps_context_images=pa_steps_context_images,
                                    goal=agent_goal,
                                    goal_aware=goal_aware,
                                )
                            )
                            step_experiment_cost += t_cost
                            total_learn_cost += t_cost
                            theory_needs_regen = False
                        elif theory_needs_regen or len(theories) < eb_config.num_theories:
                            theories, t_cost, theory_gen_log = asyncio.run(
                                refill_theories(
                                    config=config,
                                    theories=theories,
                                    beliefs=beliefs,
                                    default_knowledge=default_knowledge,
                                    num_theories=eb_config.num_theories,
                                    steps_context=pa_steps_context,
                                    current_observation=_pre_action_raw_long,
                                    current_image=_pre_action_image,
                                    steps_context_images=pa_steps_context_images,
                                    falsifications=falsification_memory,
                                    goal=agent_goal,
                                    goal_aware=goal_aware,
                                )
                            )
                            step_experiment_cost += t_cost
                            total_learn_cost += t_cost
                            theory_needs_regen = False

                        action_obj, sel_cost, sel_log = asyncio.run(
                            select_discriminating_action(
                                config=config,
                                theories=theories,
                                beliefs=beliefs,
                                default_knowledge=default_knowledge,
                                steps_context=pa_steps_context,
                                current_observation=_pre_action_raw_long,
                                current_image=_pre_action_image,
                                steps_context_images=pa_steps_context_images,
                                num_candidate_actions=eb_config.num_candidate_actions,
                                goal=agent_goal,
                                goal_aware=goal_aware,
                            )
                        )
                        step_experiment_cost += sel_cost
                        total_learn_cost += sel_cost

                if action_obj is not None:
                    if current_experiment and current_experiment not in past_experiments:
                        past_experiments.append(current_experiment)
                    current_experiment = action_obj.plan
                    current_experiment_question = (
                        "goal action (Plan A exploit)"
                        if exploit_mode else "discriminating action (Plan A)"
                    )
                    pending_predictions = dict(action_obj.predictions)
                    pending_action_plan = action_obj.plan
                    did_formulate_experiment = True
                    evolve_logger.info(
                        f"[g{global_step}] Plan A {'EXPLOIT' if exploit_mode else 'explore'} "
                        f"action: {action_obj.plan[:110]}"
                    )
                else:
                    evolve_logger.warning(
                        f"[g{global_step}] Plan A: no action "
                        f"(mode={'exploit' if exploit_mode else 'explore'}, "
                        f"theories={len(theories)}); keeping current experiment."
                    )
                    pending_predictions = {}
                    pending_action_plan = None

                with open(step_dir / "theory_log.json", "w") as f:
                    json.dump(
                        {
                            "theories": serialize_theories(theories),
                            "theory_generation": theory_gen_log,
                            "select_action": sel_log,
                        },
                        f,
                        indent=4,
                        default=str,
                    )
                step_experiment_log = {
                    "mode": "theory_disagreement",
                    "active_experiment": current_experiment,
                    "active_experiment_question": current_experiment_question,
                    "num_theories": len(theories),
                    "theory_weights": {t.rank: t.weight for t in theories},
                    "selected_action": (
                        {
                            "plan": action_obj.plan,
                            "rationale": action_obj.rationale,
                            "predictions": action_obj.predictions,
                            "candidate_actions": action_obj.candidate_actions,
                        }
                        if action_obj is not None
                        else None
                    ),
                }
                agent.experiment_goal = current_experiment
                with open(step_dir / "experiment_log.json", "w") as f:
                    json.dump(step_experiment_log, f, indent=4, default=str)
                evolve_logger.info(
                    f"[g{global_step}] Plan A experiment set — "
                    f"cost: ${step_experiment_cost:.6f}"
                )

            elif frontier_mode:
                # --- GEPA/legacy frontier: learn a set of possible Bs from the
                # collected trajectory, then formulate a discriminating
                # experiment from their disagreement. RELEARN runs on a cadence
                # (it is expensive); the EXPERIMENT is reformulated each step
                # from the current frontier. ---
                fr_steps_context = format_steps_context(
                    trajectory_buffer,
                    perception,
                    eb_config.max_steps_context_chars,
                    history_window=eb_config.perception_history_window,
                    hide_raw_obs_when_image=eb_config.hide_obs_when_image,
                    include_trailing_state=False,
                )
                fr_steps_context, fr_steps_context_images = _images_for_steps_context(
                    trajectory_buffer,
                    fr_steps_context,
                    max_images=eb_config.max_images_context,
                )
                relearn_info: dict | None = None
                with improve_logging(step_dir):
                    # RELEARN (cadence) -- refit the frontier on the buffer.
                    if step % eb_config.frontier_relearn_interval == 0:
                        evolve_logger.info(
                            f"[g{global_step}] Frontier relearn "
                            f"({eb_config.frontier_learner}) on buffer..."
                        )
                        relearn_rng = random.Random(seed + global_step)
                        new_frontier, frontier_metric, relearn_info, fr_cost = (
                            _relearn_frontier_eb(
                                config=config,
                                eb_config=eb_config,
                                trajectory_buffer=trajectory_buffer,
                                env_name=env_name,
                                current_frontier=frontier,
                                rng=relearn_rng,
                            )
                        )
                        step_experiment_cost += fr_cost
                        total_learn_cost += fr_cost
                        if new_frontier and not relearn_info.get("skipped"):
                            frontier = new_frontier
                            # Top candidate feeds the agent's B + P; re-inject
                            # into the live agent (mid-episode update).
                            top = frontier[0]
                            beliefs = top.get("world_knowledge", "") or beliefs
                            if eb_config.perception_enabled:
                                new_perc = _clean_frontier_perception(
                                    top.get("perception", "")
                                )
                                if new_perc.strip():
                                    perception = new_perc
                                    perception_fn = load_perception_fn(perception)
                            _inject_beliefs(
                                config, agent, env, env_name, task, beliefs,
                                agent_goal=agent_goal,
                            )
                            evolve_logger.info(
                                f"[g{global_step}] Frontier updated: "
                                f"{len(frontier)} candidates, id_acc={frontier_metric} "
                                f"| {relearn_info}"
                            )
                        else:
                            evolve_logger.info(
                                f"[g{global_step}] Frontier relearn skipped: "
                                f"{relearn_info.get('skipped')}"
                            )

                    # EXPERIMENT -- formulate a discriminating experiment from the
                    # competing Bs (wrapped as theories). Falls back to keeping the
                    # current experiment when <2 distinct candidates exist.
                    fr_theories = [
                        Theory(world_knowledge=c.get("world_knowledge", ""), rank=i + 1)
                        for i, c in enumerate(frontier)
                        if (c.get("world_knowledge", "") or "").strip()
                    ]
                    action_obj = None
                    sel_log = None
                    if len(fr_theories) >= 2:
                        action_obj, sel_cost, sel_log = asyncio.run(
                            select_discriminating_action(
                                config=config,
                                theories=fr_theories,
                                beliefs=beliefs,
                                default_knowledge=default_knowledge,
                                steps_context=fr_steps_context,
                                current_observation=_pre_action_raw_long,
                                current_image=_pre_action_image,
                                steps_context_images=fr_steps_context_images,
                                num_candidate_actions=eb_config.num_candidate_actions,
                                goal=agent_goal,
                                goal_aware=goal_aware,
                            )
                        )
                        step_experiment_cost += sel_cost
                        total_learn_cost += sel_cost

                if action_obj is not None:
                    if current_experiment and current_experiment not in past_experiments:
                        past_experiments.append(current_experiment)
                    current_experiment = action_obj.plan
                    current_experiment_question = "discriminating experiment (frontier)"
                    did_formulate_experiment = True
                    evolve_logger.info(
                        f"[g{global_step}] Frontier experiment: {action_obj.plan[:110]}"
                    )

                agent.experiment_goal = current_experiment
                with open(step_dir / "frontier.json", "w") as f:
                    json.dump(
                        {
                            "frontier": frontier,
                            "metric": frontier_metric,
                            "relearn": relearn_info,
                        },
                        f, indent=2, default=str,
                    )
                step_experiment_log = {
                    "mode": "gepa_frontier",
                    "active_experiment": current_experiment,
                    "active_experiment_question": current_experiment_question,
                    "frontier_size": len(frontier),
                    "frontier_metric": frontier_metric,
                    "selected_action": (
                        {
                            "plan": action_obj.plan,
                            "rationale": action_obj.rationale,
                            "predictions": action_obj.predictions,
                            "candidate_actions": action_obj.candidate_actions,
                        }
                        if action_obj is not None
                        else None
                    ),
                }
                with open(step_dir / "experiment_log.json", "w") as f:
                    json.dump(step_experiment_log, f, indent=4, default=str)
                evolve_logger.info(
                    f"[g{global_step}] Frontier experiment set — "
                    f"cost: ${step_experiment_cost:.6f}"
                )

            elif should_gen_experiments:
                evolve_logger.info(f"[g{global_step}] Generating questions...")
                exp_steps_context = format_steps_context(
                    trajectory_buffer,
                    perception,
                    eb_config.max_steps_context_chars,
                    history_window=eb_config.perception_history_window,
                    hide_raw_obs_when_image=eb_config.hide_obs_when_image,
                    include_trailing_state=False,
                )
                q_steps_context, q_steps_context_images = _images_for_steps_context(
                    trajectory_buffer,
                    exp_steps_context,
                )
                exp_steps_context, exp_steps_context_images = _images_for_steps_context(
                    trajectory_buffer,
                    exp_steps_context,
                    max_images=eb_config.max_images_context,
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
                            include_recent_history=True,
                        )
                    )
                    qa_pairs.extend(new_questions)
                    step_experiment_cost += q_cost
                    total_learn_cost += q_cost
                    did_gen_questions = True

                    evolve_logger.info(
                        f"[g{global_step}] Generated {len(new_questions)} questions — cost: ${q_cost:.6f}"
                    )

                    # Plan B: theory generation + theory-seeded crux questions.
                    # Stateless — regenerate competing theories from the current
                    # state/beliefs each selection point, then add crux questions
                    # (designed to split the theories) to the bank before dedup.
                    theories: list = []
                    theory_gen_log = None
                    crux_log = None
                    if eb_config.question_scoring_method == "theory_entropy":
                        from theory_exploration import (
                            generate_crux_questions,
                            generate_theories,
                        )

                        _theory_state_only = eb_config.theory_gen_current_state_only

                        def _gen_theories(seed_questions=None):
                            return generate_theories(
                                config=config,
                                beliefs=beliefs,
                                default_knowledge=default_knowledge,
                                steps_context=(
                                    "" if _theory_state_only else q_steps_context
                                ),
                                current_observation=_pre_action_raw_long,
                                current_image=_pre_action_image,
                                steps_context_images=(
                                    None if _theory_state_only else q_steps_context_images
                                ),
                                num_theories=eb_config.num_theories,
                                decay=eb_config.theory_weight_decay,
                                goal_aware=goal_aware,
                                seed_questions=seed_questions,
                            )

                        theories, theory_cost, theory_gen_log = asyncio.run(
                            _gen_theories()
                        )
                        step_experiment_cost += theory_cost

                        # Lever #1 (MI-residual): score the unanswered bank
                        # against this initial ensemble; any question every
                        # theory is agnostic about (all-UNK -> MI 0) probes a
                        # mechanism no theory models. Regenerate theories seeded
                        # with those residual questions so the new ensemble has a
                        # member that predicts them -> they become selectable.
                        theory_seed_log: dict | None = None
                        _num_seed = eb_config.num_theory_seed_questions
                        if _num_seed > 0 and len(theories) >= 2:
                            from theory_exploration import (
                                score_questions_theory_entropy,
                            )

                            _resid_cands = [
                                i
                                for i, qa in enumerate(qa_pairs)
                                if qa.answer is None
                            ]
                            _pre_scores, _pre_cost, _pre_log = asyncio.run(
                                score_questions_theory_entropy(
                                    config=config,
                                    theories=theories,
                                    qa_pairs=qa_pairs,
                                    candidate_indices=_resid_cands,
                                    default_knowledge=default_knowledge,
                                    max_concurrent=(
                                        eb_config.question_scoring_max_concurrent
                                    ),
                                )
                            )
                            step_experiment_cost += _pre_cost
                            total_learn_cost += _pre_cost
                            _seed_qs = _select_residual_seed_questions(
                                _pre_log, _num_seed
                            )
                            theory_seed_log = {
                                "residual_prescore": _pre_log,
                                "seed_questions": _seed_qs,
                                "initial_theories": theory_gen_log,
                                "seed_theories": None,
                                "num_added": 0,
                            }
                            if _seed_qs:
                                (
                                    _seeded,
                                    _seed_cost,
                                    _seed_gen_log,
                                ) = asyncio.run(_gen_theories(seed_questions=_seed_qs))
                                step_experiment_cost += _seed_cost
                                total_learn_cost += _seed_cost
                                # Use the seeded ensemble (same context + the
                                # residual mechanisms now covered). It is ranked
                                # 1..M with proper decay weights, so the covered
                                # theory carries real weight (unlike appending).
                                if _seeded:
                                    theories = _seeded
                                    theory_gen_log = _seed_gen_log
                                    theory_seed_log["seed_theories"] = _seed_gen_log
                                    theory_seed_log["num_added"] = len(_seeded)
                                evolve_logger.info(
                                    f"[g{global_step}] MI-residual seeding: "
                                    f"{len(_seed_qs)} residual Q -> regenerated "
                                    f"{len(theories)} theories"
                                )
                            else:
                                evolve_logger.info(
                                    f"[g{global_step}] MI-residual seeding: no "
                                    f"all-UNK residual questions; ensemble "
                                    f"unchanged"
                                )
                            with open(
                                step_dir / "theory_seed_log.json", "w"
                            ) as f:
                                json.dump(theory_seed_log, f, indent=4, default=str)
                        total_learn_cost += theory_cost

                        crux_qs, crux_cost, crux_log = asyncio.run(
                            generate_crux_questions(
                                config=config,
                                theories=theories,
                                beliefs=beliefs,
                                default_knowledge=default_knowledge,
                                num_crux=eb_config.num_crux_questions,
                            )
                        )
                        step_experiment_cost += crux_cost
                        total_learn_cost += crux_cost
                        for cq in crux_qs:
                            qa_pairs.append(
                                EBQAPair(
                                    question=cq,
                                    answer=None,
                                    evidence="",
                                    source_step=global_step,
                                )
                            )
                        evolve_logger.info(
                            f"[g{global_step}] Theories: {len(theories)} generated; "
                            f"crux questions added: {len(crux_qs)} — "
                            f"cost: ${theory_cost + crux_cost:.6f}"
                        )
                        with open(step_dir / "theory_log.json", "w") as f:
                            json.dump(
                                {
                                    "theories": theory_gen_log,
                                    "crux_questions": crux_log,
                                },
                                f,
                                indent=4,
                                default=str,
                            )

                    # Step 2: De-duplicate the maintained bank, then select a
                    # capped probe subset for this experiment prompt. Unlike
                    # the old trim path, low-scoring questions remain in the
                    # maintained bank for future scoring/projection coverage.
                    selection_cost_total = 0.0
                    qa_pairs_for_experiment = list(qa_pairs)
                    selected_source_indices = list(range(len(qa_pairs)))
                    scoring_method = eb_config.question_scoring_method
                    selection_mode = eb_config.experiment_selection_mode
                    if selection_mode not in ("single", "score_topk"):
                        raise ValueError(
                            f"Unknown experiment_selection_mode: {selection_mode!r}"
                        )
                    if scoring_method not in (
                        "b_diff_full",
                        "b_diff_light",
                        "llm_trim",
                        "theory_entropy",
                    ):
                        raise ValueError(
                            f"Unknown question_scoring_method: {scoring_method!r}"
                        )
                    combined_candidate_experiments: list[dict] = []
                    destructive_unanswered_dropped_count = 0
                    destructive_unanswered_dropped_questions: list[str] = []

                    qa_pairs, dedup_cost, dedup_log = asyncio.run(
                        deduplicate_qa_pairs(
                            config=config,
                            current_qa=qa_pairs,
                        )
                    )
                    selection_cost_total += dedup_cost

                    if selection_mode == "score_topk":
                        (
                            qa_pairs_for_experiment,
                            selected_source_indices,
                            combined_candidate_experiments,
                            select_cost,
                            selection_log,
                        ) = asyncio.run(
                            select_qa_pairs_and_formulate_experiments(
                                config=config,
                                current_qa=qa_pairs,
                                max_unanswered_qa_pairs=eb_config.max_unanswered_qa_pairs,
                                beliefs=beliefs,
                                perception_code=perception,
                                steps_context=exp_steps_context,
                                current_observation=_pre_action_raw_long,
                                current_aux_observation=_pre_action_raw_short,
                                default_knowledge=default_knowledge,
                                current_image=_pre_action_image,
                                steps_context_images=exp_steps_context_images,
                                hide_raw_obs=eb_config.hide_obs_when_image,
                                filter_questions=eb_config.score_topk_filter_questions,
                            )
                        )
                    else:
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

                    if eb_config.trim_unanswered_at_selection:
                        selected_unanswered_indices = set(selected_source_indices)
                        kept_source_indices = [
                            i
                            for i, qa in enumerate(qa_pairs)
                            if qa.answer is not None or i in selected_unanswered_indices
                        ]
                        dropped_unanswered_indices = [
                            i
                            for i, qa in enumerate(qa_pairs)
                            if qa.answer is None
                            and i not in selected_unanswered_indices
                        ]
                        destructive_unanswered_dropped_count = len(
                            dropped_unanswered_indices
                        )
                        destructive_unanswered_dropped_questions = [
                            qa_pairs[i].question for i in dropped_unanswered_indices
                        ]

                        source_to_filtered_index = {
                            source_idx: filtered_idx
                            for filtered_idx, source_idx in enumerate(
                                kept_source_indices
                            )
                        }
                        selected_source_indices = [
                            source_to_filtered_index[i]
                            for i in selected_source_indices
                            if i in source_to_filtered_index
                        ]
                        for cand in combined_candidate_experiments:
                            old_source_index = cand.get("source_index")
                            if old_source_index not in source_to_filtered_index:
                                cand["source_index"] = None
                                continue
                            cand["source_index"] = source_to_filtered_index[
                                old_source_index
                            ]
                        qa_pairs = [qa_pairs[i] for i in kept_source_indices]
                        qa_pairs_for_experiment = [
                            qa_pairs[i] for i in selected_source_indices
                        ]
                        selection_log["destructive_unanswered_filter"] = {
                            "enabled": True,
                            "kept_source_indices_before_filter": kept_source_indices,
                            "dropped_unanswered_source_indices_before_filter": (
                                dropped_unanswered_indices
                            ),
                            "dropped_unanswered_questions": (
                                destructive_unanswered_dropped_questions
                            ),
                            "dropped_unanswered_count": (
                                destructive_unanswered_dropped_count
                            ),
                            "post_filter_count": len(qa_pairs),
                            "post_filter_unanswered": sum(
                                1 for q in qa_pairs if q.answer is None
                            ),
                        }

                    scoring_log: dict | None = None
                    ranked_unanswered_indices: list[int] = []
                    target_experiment_question_index: int | None = None
                    target_experiment_question_source_index: int | None = None
                    # b_diff selects a single target question for the
                    # single-mode formulation prompt; score_topk replaces that
                    # with experiment-level scoring, so b_diff is skipped.
                    if selection_mode == "single" and scoring_method in (
                        "b_diff_full",
                        "b_diff_light",
                        "theory_entropy",
                    ):
                        candidate_indices = [
                            i
                            for i in selected_source_indices
                            if qa_pairs[i].answer is None
                        ]
                        if scoring_method == "theory_entropy":
                            from theory_exploration import (
                                score_questions_theory_entropy,
                            )

                            scores, score_cost, scoring_log = asyncio.run(
                                score_questions_theory_entropy(
                                    config=config,
                                    theories=theories,
                                    qa_pairs=qa_pairs,
                                    candidate_indices=candidate_indices,
                                    default_knowledge=default_knowledge,
                                    max_concurrent=eb_config.question_scoring_max_concurrent,
                                )
                            )
                        else:
                            from question_scoring import score_questions_b_diff

                            method_suffix = (
                                "full" if scoring_method == "b_diff_full" else "light"
                            )
                            scores, score_cost, scoring_log = asyncio.run(
                                score_questions_b_diff(
                                    config=config,
                                    beliefs=beliefs,
                                    qa_pairs=qa_pairs,
                                    method=method_suffix,
                                    include_policy=eb_config.include_policy and goal_aware,
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
                                selected_tied_source_index = ranked_unanswered_indices[
                                    0
                                ]
                                tie_break_log = {
                                    "executed": False,
                                    "reason": "no_top_score_tie",
                                    "top_score": top_score,
                                    "candidate_source_indices": tied_top_indices,
                                    "selected_source_index": selected_tied_source_index,
                                    "selected_question": qa_pairs[
                                        selected_tied_source_index
                                    ].question,
                                }
                        selected_answered_indices = [
                            i
                            for i in selected_source_indices
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
                            target_experiment_question_source_index = (
                                selected_tied_source_index
                            )
                            target_experiment_question_index = (
                                selected_source_indices.index(
                                    selected_tied_source_index
                                )
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

                    step_trim_log = {
                        "method": f"probe_selection_{scoring_method}_{selection_mode}",
                        "experiment_selection_mode": selection_mode,
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
                        "dropped_count": (
                            dedup_log.get("dropped_count", 0)
                            + destructive_unanswered_dropped_count
                        ),
                        "selection_dropped_unanswered_count": (
                            destructive_unanswered_dropped_count
                        ),
                        "selection_dropped_unanswered_questions": (
                            destructive_unanswered_dropped_questions
                        ),
                        "total_cost": selection_cost_total,
                        "dedup": dedup_log,
                        "selection": selection_log,
                        "scoring": scoring_log,
                        "maintained_bank_preserved": (
                            not eb_config.trim_unanswered_at_selection
                        ),
                        "trim_unanswered_at_selection": (
                            eb_config.trim_unanswered_at_selection
                        ),
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
                    did_trim_step = bool(step_trim_log.get("dropped_count", 0))

                    with open(step_dir / "trim_log.json", "w") as f:
                        json.dump(step_trim_log, f, indent=4, default=str)
                    with open(step_dir / "question_selection_log.json", "w") as f:
                        json.dump(step_trim_log, f, indent=4, default=str)
                    if scoring_log is not None:
                        if scoring_method == "theory_entropy":
                            method_suffix = "theory_entropy"
                        elif scoring_method == "b_diff_full":
                            method_suffix = "full"
                        else:
                            method_suffix = "light"
                        scoring_artifact = {
                            "step": global_step,
                            "method": method_suffix,
                            "source": "online_probe_selection",
                            "did_trim": False,
                            "num_qa_before_trim": step_trim_log.get("pre_trim_count"),
                            "num_qa_after_trim": step_trim_log.get("post_trim_count"),
                            "num_answered_before_trim": step_trim_log.get(
                                "pre_trim_answered"
                            ),
                            "num_answered_after_trim": step_trim_log.get(
                                "post_trim_answered"
                            ),
                            "num_unanswered_before_trim": step_trim_log.get(
                                "pre_trim_unanswered"
                            ),
                            "num_unanswered_after_trim": step_trim_log.get(
                                "post_trim_unanswered"
                            ),
                            "cap_answered": step_trim_log.get("max_answered_qa_pairs"),
                            "cap_unanswered": step_trim_log.get(
                                "max_unanswered_qa_pairs"
                            ),
                            "dropped_count": step_trim_log.get("dropped_count"),
                            "cost_usd": selection_cost_total,
                            "ranked_unanswered": scoring_log.get(
                                "ranked_unanswered", []
                            ),
                            "kept_unanswered_questions": [
                                qa_pairs[i].question for i in ranked_unanswered_indices
                            ],
                            "dropped_unanswered_questions": (
                                destructive_unanswered_dropped_questions
                            ),
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

                    # Step 3: Formulate experiment(s).
                    # In "single" mode formulation is now driven entirely by
                    # question selection (no separate gating LLM call): the
                    # scorer above picks a single target question, and we
                    # formulate a fresh experiment for it *iff* it differs from
                    # the question the active experiment is already pursuing.
                    # If the selected question is the same (or no unanswered
                    # candidate was selected), the active experiment is kept
                    # verbatim. In "score_topk" mode Step 2 already selected
                    # top-k questions and wrote their fresh experiment plans in
                    # one prompt; here we score each candidate against the full
                    # unanswered bank and pick the one with the most YES
                    # verdicts.
                    experiment_scoring_log: dict | None = None
                    if selection_mode == "single" and scoring_method in (
                        "b_diff_full",
                        "b_diff_light",
                        "theory_entropy",
                    ):
                        # Selection-driven formulation: the scorer above already
                        # picked a single target question. Formulate a fresh
                        # experiment for it iff it differs from the active
                        # experiment's question; otherwise keep the current
                        # experiment. No separate gating LLM call / null return.
                        experiment_plan = None
                        q_idx = target_experiment_question_index
                        e_cost = 0.0
                        e_prompt = ""
                        e_response = ""
                        selected_question_text = None

                        selected_q_text = (
                            qa_pairs[target_experiment_question_source_index].question
                            if target_experiment_question_source_index is not None
                            else None
                        )
                        active_q_norm = (
                            current_experiment_question.strip().lower()
                            if current_experiment_question
                            else None
                        )
                        is_new_question = selected_q_text is not None and (
                            current_experiment is None
                            or active_q_norm is None
                            or selected_q_text.strip().lower() != active_q_norm
                        )

                        if is_new_question:
                            evolve_logger.info(
                                f"[g{global_step}] New question selected; "
                                f"formulating experiment for it..."
                            )
                            experiment_plan, e_cost, e_prompt, e_response = (
                                asyncio.run(
                                    formulate_experiment_for_question(
                                        config=config,
                                        beliefs=beliefs,
                                        perception_code=perception,
                                        steps_context=exp_steps_context,
                                        target_question=selected_q_text,
                                        current_observation=_pre_action_raw_long,
                                        current_aux_observation=_pre_action_raw_short,
                                        default_knowledge=default_knowledge,
                                        current_image=_pre_action_image,
                                        steps_context_images=exp_steps_context_images,
                                        hide_raw_obs=eb_config.hide_obs_when_image,
                                    )
                                )
                            )
                            step_experiment_cost += e_cost
                            total_learn_cost += e_cost

                            if experiment_plan:
                                # Move old active experiment to past if it exists
                                if (
                                    current_experiment
                                    and current_experiment not in past_experiments
                                ):
                                    past_experiments.append(current_experiment)
                                current_experiment = experiment_plan
                                current_experiment_question = selected_q_text
                                selected_question_text = selected_q_text
                                did_formulate_experiment = True
                            else:
                                evolve_logger.warning(
                                    f"[g{global_step}] Formulation returned no plan; "
                                    f"keeping current experiment."
                                )
                        else:
                            reason = (
                                "no unanswered candidate selected"
                                if selected_q_text is None
                                else "selected question unchanged"
                            )
                            evolve_logger.info(
                                f"[g{global_step}] Keeping current experiment ({reason})."
                            )
                            selected_question_text = current_experiment_question
                    elif selection_mode == "single":
                        # Legacy single-mode path for scorers that do not select
                        # a single target question (e.g. llm_trim): the gated
                        # formulation call performs its own question selection and
                        # may return null to keep the current experiment.
                        evolve_logger.info(
                            f"[g{global_step}] Formulating experiment from questions..."
                        )
                        experiment_plan, q_idx, e_cost, e_prompt, e_response = (
                            asyncio.run(
                                formulate_experiment_from_question(
                                    config=config,
                                    beliefs=beliefs,
                                    perception_code=perception,
                                    steps_context=exp_steps_context,
                                    current_qa=qa_pairs_for_experiment,
                                    current_experiment=current_experiment,
                                    current_experiment_question=current_experiment_question,
                                    current_observation=_pre_action_raw_long,
                                    current_aux_observation=_pre_action_raw_short,
                                    default_knowledge=default_knowledge,
                                    current_image=_pre_action_image,
                                    steps_context_images=exp_steps_context_images,
                                    hide_raw_obs=eb_config.hide_obs_when_image,
                                    target_question_index=target_experiment_question_index,
                                )
                            )
                        )
                        step_experiment_cost += e_cost
                        total_learn_cost += e_cost

                        if experiment_plan is not None:
                            selected_question_text = (
                                qa_pairs_for_experiment[q_idx].question
                                if q_idx is not None
                                and 0 <= q_idx < len(qa_pairs_for_experiment)
                                else None
                            )
                            # Move old active experiment to past if it exists
                            if (
                                current_experiment
                                and current_experiment not in past_experiments
                            ):
                                past_experiments.append(current_experiment)
                            current_experiment = experiment_plan
                            current_experiment_question = selected_question_text
                            did_formulate_experiment = True
                        else:
                            selected_question_text = None
                    else:
                        # score_topk mode
                        candidates: list[dict] = list(combined_candidate_experiments)
                        topk_unanswered_order = [
                            src_idx
                            for src_idx in selected_source_indices
                            if qa_pairs[src_idx].answer is None
                        ]
                        evolve_logger.info(
                            f"[g{global_step}] Scoring {len(candidates)} combined selected/formulated candidate experiments..."
                        )

                        if current_experiment:
                            active_source_index: int | None = None
                            if current_experiment_question:
                                key = current_experiment_question.strip().lower()
                                for i, qa in enumerate(qa_pairs):
                                    if qa.question.strip().lower() == key:
                                        active_source_index = i
                                        break
                            candidates.append(
                                {
                                    "kind": "active",
                                    "source_index": active_source_index,
                                    "question": current_experiment_question,
                                    "plan": current_experiment,
                                    "topk_rank": len(topk_unanswered_order),
                                    "formulation_prompt": None,
                                    "formulation_response": None,
                                    "formulation_cost": 0.0,
                                }
                            )

                        # Score every candidate against the full unanswered bank.
                        unanswered_pool_indices = [
                            i for i, qa in enumerate(qa_pairs) if qa.answer is None
                        ]
                        unanswered_pool_qa = [
                            qa_pairs[i] for i in unanswered_pool_indices
                        ]

                        if candidates:
                            (
                                scores_per_candidate,
                                per_candidate_yes,
                                unified_score_cost,
                                unified_score_prompt,
                                unified_score_response,
                                scoring_parsed_ok,
                            ) = asyncio.run(
                                score_experiments_against_questions(
                                    config=config,
                                    candidates=candidates,
                                    unanswered_qa=unanswered_pool_qa,
                                    unanswered_source_indices=unanswered_pool_indices,
                                    beliefs=beliefs,
                                    default_knowledge=default_knowledge,
                                )
                            )
                        else:
                            scores_per_candidate = []
                            per_candidate_yes = []
                            unified_score_cost = 0.0
                            unified_score_prompt = ""
                            unified_score_response = ""
                            scoring_parsed_ok = True

                        step_experiment_cost += unified_score_cost
                        total_learn_cost += unified_score_cost

                        # All candidates are scored in one prompt; attach the
                        # shared prompt/response/cost to the first candidate so
                        # the per-candidate viz still surfaces them once, and
                        # zero out the rest to avoid double-counting.
                        for i, cand in enumerate(candidates):
                            cand["score"] = scores_per_candidate[i]
                            cand["per_question_yes_source_indices"] = sorted(
                                src for src, v in per_candidate_yes[i].items() if v
                            )
                            cand["score_prompt"] = (
                                unified_score_prompt if i == 0 else None
                            )
                            cand["score_response"] = (
                                unified_score_response if i == 0 else None
                            )
                            cand["score_cost"] = unified_score_cost if i == 0 else 0.0

                        # Pick winner: highest score, ties broken by topk_rank
                        # (earliest topk position wins; active candidate goes
                        # last because its topk_rank is len(topk)).
                        winner_index: int | None = None
                        winner_fallback_reason: str | None = None
                        if candidates:
                            winner_index = max(
                                range(len(candidates)),
                                key=lambda i: (
                                    candidates[i]["score"],
                                    -candidates[i]["topk_rank"],
                                ),
                            )
                            # If the scoring LLM response was unparseable, keep
                            # the current active experiment rather than picking
                            # an arbitrary fresh candidate via tie-break.
                            if not scoring_parsed_ok:
                                active_idx = next(
                                    (
                                        i
                                        for i, c in enumerate(candidates)
                                        if c["kind"] == "active"
                                    ),
                                    None,
                                )
                                if active_idx is not None:
                                    winner_index = active_idx
                                    winner_fallback_reason = "scoring_parse_failed"
                                    evolve_logger.warning(
                                        f"[g{global_step}] Experiment scoring response "
                                        "unparseable — falling back to active experiment."
                                    )

                        experiment_scoring_log = {
                            "mode": "score_topk",
                            "parsed_ok": scoring_parsed_ok,
                            "winner_fallback_reason": winner_fallback_reason,
                            "topk_source_indices": list(selected_source_indices),
                            "topk_unanswered_source_indices": topk_unanswered_order,
                            "unanswered_pool_source_indices": unanswered_pool_indices,
                            "candidates": [
                                {
                                    "kind": c["kind"],
                                    "source_index": c["source_index"],
                                    "question": c["question"],
                                    "plan": c["plan"],
                                    "score": c.get("score", 0),
                                    "per_question_yes_source_indices": (
                                        c.get("per_question_yes_source_indices", [])
                                    ),
                                    "topk_rank": c["topk_rank"],
                                    "formulation_cost": c["formulation_cost"],
                                    "score_cost": c.get("score_cost", 0.0),
                                    "formulation_prompt": c["formulation_prompt"],
                                    "formulation_response": c["formulation_response"],
                                    "score_prompt": c.get("score_prompt"),
                                    "score_response": c.get("score_response"),
                                }
                                for c in candidates
                            ],
                            "winner_index": winner_index,
                            "winner_kind": (
                                candidates[winner_index]["kind"]
                                if winner_index is not None
                                else None
                            ),
                            "winner_source_index": (
                                candidates[winner_index]["source_index"]
                                if winner_index is not None
                                else None
                            ),
                            "winner_score": (
                                candidates[winner_index]["score"]
                                if winner_index is not None
                                else None
                            ),
                            "selection_formulation_cost": select_cost,
                            "score_cost": sum(
                                c.get("score_cost", 0.0) for c in candidates
                            ),
                            "total_cost": (
                                select_cost
                                + sum(c.get("score_cost", 0.0) for c in candidates)
                            ),
                        }

                        # Apply the winner.
                        experiment_plan = None
                        q_idx = None
                        selected_question_text = None
                        e_prompt = selection_log.get("prompt", "")
                        e_response = selection_log.get("response", "")
                        e_cost = experiment_scoring_log["total_cost"]
                        if winner_index is not None:
                            winner = candidates[winner_index]
                            if winner["kind"] == "fresh":
                                if (
                                    current_experiment
                                    and current_experiment not in past_experiments
                                    and current_experiment != winner["plan"]
                                ):
                                    past_experiments.append(current_experiment)
                                current_experiment = winner["plan"]
                                current_experiment_question = winner["question"]
                                experiment_plan = winner["plan"]
                                selected_question_text = winner["question"]
                                if (
                                    winner["source_index"] is not None
                                    and winner["source_index"]
                                    in selected_source_indices
                                ):
                                    q_idx = selected_source_indices.index(
                                        winner["source_index"]
                                    )
                                did_formulate_experiment = True
                            else:
                                # active candidate won — keep current
                                selected_question_text = current_experiment_question
                                experiment_plan = current_experiment
                                if (
                                    winner["source_index"] is not None
                                    and winner["source_index"]
                                    in selected_source_indices
                                ):
                                    q_idx = selected_source_indices.index(
                                        winner["source_index"]
                                    )
                        evolve_logger.info(
                            f"[g{global_step}] Scored {len(candidates)} candidate experiments — "
                            f"winner: "
                            f"{candidates[winner_index]['kind'] if winner_index is not None else 'none'} "
                            f"score="
                            f"{candidates[winner_index]['score'] if winner_index is not None else 'n/a'}, "
                            f"cost: ${experiment_scoring_log['total_cost']:.6f}"
                        )

                    # Experiment prompts use the capped trajectory-image
                    # sequence plus the current pre-action image. Question
                    # prompts keep their own sequence so the saved prompt
                    # images match the exact attachments used for each call.
                    exp_prompt_images = list(exp_steps_context_images or [])
                    if _pre_action_image is not None:
                        exp_prompt_images.append(_pre_action_image)
                    exp_image_paths = _save_prompt_images(
                        exp_prompt_images,
                        step_dir,
                        "experiment_log_images",
                    )
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
                        "selected_question": selected_question_text,
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
                        "active_experiment": current_experiment,
                        "active_experiment_question": current_experiment_question,
                        "experiment_selection_mode": selection_mode,
                        "question_selection_method": scoring_method,
                        "qa_pairs_for_experiment": serialize_eb_qa_pairs(
                            qa_pairs_for_experiment
                        ),
                        "qa_pairs_for_experiment_source_indices": selected_source_indices,
                        "qa_pairs_at_formulation": serialize_eb_qa_pairs(qa_pairs),
                        "experiment_scoring": experiment_scoring_log,
                    }

                # Update current experiment and inject into agent
                agent.experiment_goal = current_experiment

                # Write experiment artifacts immediately
                with open(step_dir / "experiment_log.json", "w") as f:
                    json.dump(step_experiment_log, f, indent=4, default=str)
                if experiment_scoring_log is not None:
                    with open(step_dir / "experiment_scoring_log.json", "w") as f:
                        json.dump(
                            experiment_scoring_log,
                            f,
                            indent=4,
                            default=str,
                        )
                with open(step_dir / "qa_pairs.json", "w") as f:
                    json.dump(serialize_eb_qa_pairs(qa_pairs), f, indent=4)

                evolve_logger.info(
                    f"[g{global_step}] Experiment: {'new' if did_formulate_experiment else 'kept'} — "
                    f"cost: ${step_experiment_cost:.6f}"
                )

            num_unanswered = sum(1 for q in qa_pairs if q.answer is None)

            # Update step_log: experiment phase done, agent about to act
            _save_step_log_eb(
                step_dir=step_dir,
                step=step,
                global_step=global_step,
                action=None,
                reward=0.0,
                done=False,
                episode_return=episode_return,
                agent_cost=0.0,
                extract_cost=0.0,
                improve_cost=0.0,
                experiment_cost=step_experiment_cost,
                trim_cost=step_trim_cost,
                num_qa=len(qa_pairs),
                num_unanswered=num_unanswered,
                did_gen_questions=did_gen_questions,
                did_formulate_experiment=did_formulate_experiment,
                did_trim=did_trim_step,
                active_experiment=current_experiment,
                phase="acting",
                active_experiment_question=current_experiment_question,
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
            invalid_action = False
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except ValueError as e:
                logging.warning(f"[g{global_step}] Invalid action: {action} — {e}")
                invalid_action = True
                if config.eval.feedback_on_invalid_action:
                    obs["text"]["long_term_context"] = (
                        f"\n\n{INVALID_ACTION_RETRY_MESSAGE}\n\n"
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

            # Apply perception to new obs (skip on invalid action — obs unchanged,
            # re-applying would nest the Auxiliary Observation block again)
            if perception_fn is not None and not invalid_action:
                apply_perception_with_history(
                    obs,
                    perception_fn,
                    raw_obs_history,
                    eb_config.perception_history_window,
                )

            result_obs_text = _compose_obs_text(
                obs["text"]["short_term_context"],
                obs["text"]["long_term_context"],
            )

            # Capture agent messages
            try:
                agent_messages = [
                    {"role": m.role, "content": m.content} for m in agent.last_messages
                ]
            except Exception:
                agent_messages = []
            agent_messages.append(
                {
                    "role": "assistant",
                    "content": reasoning,
                    "action": action,
                }
            )

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
            csv_writer.writerow(
                [
                    step,
                    action,
                    reasoning,
                    _pre_action_obs_text,
                    _pre_action_raw_short,
                    reward,
                    done,
                ]
            )
            csv_file.flush()

            # Update step_log with action/reward/done
            _save_step_log_eb(
                step_dir=step_dir,
                step=step,
                global_step=global_step,
                action=action,
                reward=reward,
                done=done,
                episode_return=episode_return,
                agent_cost=agent_step_cost,
                extract_cost=0.0,
                improve_cost=0.0,
                experiment_cost=step_experiment_cost,
                trim_cost=step_trim_cost,
                num_qa=len(qa_pairs),
                num_unanswered=num_unanswered,
                did_gen_questions=did_gen_questions,
                did_formulate_experiment=did_formulate_experiment,
                did_trim=did_trim_step,
                active_experiment=current_experiment,
                phase="extracting",
                active_experiment_question=current_experiment_question,
                env_info=info if isinstance(info, dict) else None,
            )

            # Append buffer entry (image is the pre-action obs image;
            # result_image is the post-action obs image).
            trajectory_buffer.append(
                {
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
                }
            )

            if done:
                trajectory_buffer.append(
                    {
                        "step": global_step + 1,
                        "obs_text": result_obs_text,
                        "raw_long_term_context": new_raw_long,
                        "raw_short_term_context": new_raw_short,
                        "image": _post_action_image,
                        "action": None,
                        "reward": 0.0,
                        "reasoning": "",
                        "done": True,
                    }
                )

            pbar.update(1)

            evolve_logger.info(
                f"[g{global_step}|ep{episode_idx}|s{step}] "
                f"action={action!r}  reward={reward:.2f}  return={episode_return:.2f}  "
                f"done={done}  agent_cost=${agent_step_cost:.6f}"
            )

            # --- Critical-transition identification ---
            # When enabled, decide whether this transition is critical
            # (post-state surprising / revealing given current beliefs +
            # perception). The flag gates artifact-update + improve in this
            # step and experiment-gen in the next step.
            if eb_config.critical_transitions_enabled:
                crit_steps_context = format_steps_context(
                    trajectory_buffer,
                    perception,
                    eb_config.max_steps_context_chars,
                    history_window=eb_config.perception_history_window,
                    hide_raw_obs_when_image=eb_config.hide_obs_when_image,
                    include_trailing_state=True,
                )
                crit_steps_context, crit_images = _images_for_steps_context(
                    trajectory_buffer,
                    crit_steps_context,
                    max_images=eb_config.max_images_context,
                )
                with improve_logging(step_dir):
                    (
                        critical_this_step,
                        critical_reason,
                        c_cost,
                        c_prompt,
                        c_response,
                    ) = asyncio.run(
                        identify_critical_transition(
                            config=config,
                            beliefs=beliefs,
                            perception_code=perception,
                            default_knowledge=default_knowledge,
                            steps_context=crit_steps_context,
                            steps_context_images=crit_images,
                            current_experiment=current_experiment,
                            current_experiment_question=current_experiment_question,
                            perception_enabled=eb_config.perception_enabled,
                        )
                    )
                step_critical_cost = c_cost
                total_learn_cost += c_cost
                did_critical_id_this_step = True

                # Tag the most recent non-terminal entry (the one we just
                # appended for this transition; on `done`, that's at index -2
                # because we also appended a terminal marker).
                tag_idx = -2 if done else -1
                if abs(tag_idx) <= len(trajectory_buffer):
                    trajectory_buffer[tag_idx]["critical"] = critical_this_step

                critical_id_log = {
                    "global_step": global_step,
                    "critical": critical_this_step,
                    "reason": critical_reason,
                    "cost_usd": c_cost,
                    "prompt": c_prompt,
                    "response": c_response,
                    "prompt_image_paths": _save_prompt_images(
                        crit_images,
                        step_dir,
                        "critical_id_log_images",
                    ),
                }
                with open(step_dir / "critical_id_log.json", "w") as f:
                    json.dump(critical_id_log, f, indent=4, default=str)

                evolve_logger.info(
                    f"[g{global_step}] critical_id: {critical_this_step} "
                    f"(cost: ${c_cost:.6f}) — {critical_reason[:120]}"
                )

            # --- Plan A: reweight the theory posterior from the observed outcome ---
            # The discriminating action registered a per-theory prediction; mark
            # each consistent/violated against the real post-action state and
            # update weights. A violated MAP theory sets the surprise flag, which
            # gates ensemble regeneration at the next step's selection.
            if plan_a_mode and pending_predictions:
                with improve_logging(step_dir):
                    theories, theory_update_log = asyncio.run(
                        update_theory_posterior(
                            config=config,
                            theories=theories,
                            predictions=pending_predictions,
                            action_taken=pending_action_plan or (action or ""),
                            observed_outcome=new_raw_long,
                            observed_image=_post_action_image,
                            default_knowledge=default_knowledge,
                            violation_penalty=eb_config.theory_violation_penalty,
                            min_weight=eb_config.theory_min_weight,
                        )
                    )
                upd_cost = theory_update_log.get("cost", 0.0)
                step_critical_cost += upd_cost
                total_learn_cost += upd_cost
                theory_needs_regen = bool(theory_update_log.get("surprise", False))
                # Stability streak for the explore->exploit switch: reset on an
                # all-violated wipe, otherwise the ensemble survived this step.
                if theory_update_log.get("all_violated"):
                    model_stable_streak = 0
                else:
                    model_stable_streak += 1
                # Accumulate this step's falsification evidence and keep only the
                # most recent entries so regeneration is grounded but the prompt
                # stays bounded.
                new_falsifications = theory_update_log.get("falsifications", []) or []
                if new_falsifications:
                    merge_falsifications(
                        falsification_memory,
                        new_falsifications,
                        falsification_memory_cap,
                    )
                with open(step_dir / "theory_update_log.json", "w") as f:
                    json.dump(theory_update_log, f, indent=4, default=str)
                with open(step_dir / "falsification_memory.json", "w") as f:
                    json.dump(falsification_memory, f, indent=4, default=str)
                evolve_logger.info(
                    f"[g{global_step}] Plan A posterior: surprise="
                    f"{theory_needs_regen}, dropped="
                    f"{theory_update_log.get('num_dropped', 0)}, "
                    f"survivors={len(theories)}, "
                    f"stable_streak={model_stable_streak}, "
                    f"falsif_mem={len(falsification_memory)} (cost: ${upd_cost:.6f})"
                )
                pending_predictions = {}
                pending_action_plan = None

            # --- Determine what to do this step ---
            steps_in = step + 1
            if eb_config.critical_transitions_enabled:
                should_update_artifacts = critical_this_step or done
                should_improve = critical_this_step or done
                # Carry-forward: gates experiment-gen at the next iteration's
                # top. On `done`, the loop breaks; the next episode call
                # re-initializes the flag to True.
                critical_flag_for_experiment_gen = critical_this_step
            else:
                should_update_artifacts = (
                    steps_in % eb_config.artifact_update_interval == 0
                ) or done
                should_improve = (steps_in % eb_config.improve_interval == 0) or done

            # In gepa_frontier mode the frontier relearn IS the B/P learner. The
            # legacy QA/improve loop would redundantly RE-learn and overwrite the
            # frontier's beliefs/perception each step (it runs later in the step),
            # so the frontier's learned B/P would be discarded and the agent would
            # act on QA-loop beliefs instead. Disable it so legacy_pop/GEPA owns B/P.
            if frontier_mode:
                should_improve = False
                should_update_artifacts = False

            # --- Artifact update (update Q from trajectory) ---
            if should_update_artifacts and len(trajectory_buffer) > 0:
                evolve_logger.info(
                    f"[g{global_step}] Updating Q from {len(trajectory_buffer)} buffered steps..."
                )
                # For ARC-AGI the raw <pre_state>/<post_state> bodies are huge
                # numeric grids that the image + perception output already
                # cover, so force-strip them from the QA-update prompt.
                qa_hide_raw_obs = (
                    eb_config.hide_obs_when_image or env_name == "arc_agi"
                )
                steps_context = format_steps_context(
                    trajectory_buffer,
                    perception,
                    eb_config.max_steps_context_chars,
                    history_window=eb_config.perception_history_window,
                    hide_raw_obs_when_image=qa_hide_raw_obs,
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
                            hide_raw_obs=qa_hide_raw_obs,
                        )
                    )
                    step_extract_cost = extract_cost
                    total_learn_cost += extract_cost

                # Save images attached to the extraction prompt so the viz can
                # render them alongside.
                step_extraction_log["prompt_image_paths"] = _save_prompt_images(
                    steps_context_images_update,
                    step_dir,
                    "extraction_log_images",
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
                    step_dir=step_dir,
                    step=step,
                    global_step=global_step,
                    action=action,
                    reward=reward,
                    done=done,
                    episode_return=episode_return,
                    agent_cost=agent_step_cost,
                    extract_cost=step_extract_cost,
                    improve_cost=0.0,
                    experiment_cost=step_experiment_cost,
                    trim_cost=step_trim_cost,
                    num_qa=len(qa_pairs),
                    num_unanswered=num_unanswered,
                    did_gen_questions=did_gen_questions,
                    did_formulate_experiment=did_formulate_experiment,
                    did_trim=did_trim_step,
                    active_experiment=current_experiment,
                    phase="improving",
                    active_experiment_question=current_experiment_question,
                    env_info=info if isinstance(info, dict) else None,
                    critical_cost=step_critical_cost,
                    did_critical_id=did_critical_id_this_step,
                    critical=critical_this_step if did_critical_id_this_step else None,
                )

            # --- Improve loop (beliefs/perception + QA) ---
            if should_improve:
                _perc_iters = _resolve_schedule(
                    eb_config.max_perception_iterations, global_step
                )
                _qa_iters = _resolve_schedule(eb_config.max_qa_iterations, global_step)
                evolve_logger.info(
                    f"[g{global_step}] Running improve loop (perception={_perc_iters}, "
                    f"qa={_qa_iters} iters)..."
                )
                pre_improve_perception = perception
                with improve_logging(step_dir):
                    beliefs, perception, qa_pairs, improve_cost, iter_records = (
                        _run_improve_loop_eb(
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
                    )
                    step_improve_cost = improve_cost
                    total_learn_cost += improve_cost
                    step_feedback_records = iter_records
                    feedback_history.extend(iter_records)

                perception_changed = perception != pre_improve_perception

                # Reload perception after improvement
                perception_fn = (load_perception_fn(perception) if eb_config.perception_enabled else None)

                # Re-apply updated perception to current obs for the agent's next step.
                # Invalid-action feedback keeps the observation unchanged, so the
                # short-term context is already perception-wrapped from the prior
                # prompt; wrapping it again nests the auxiliary observation block.
                if not done:
                    obs["text"]["long_term_context"] = new_raw_long
                    obs["text"]["short_term_context"] = new_raw_short
                    if perception_fn is not None and not invalid_action:
                        apply_perception_with_history(
                            obs,
                            perception_fn,
                            raw_obs_history,
                            eb_config.perception_history_window,
                        )

                # Rebuild all buffered observations with the latest perception
                if perception_changed:
                    _refresh_buffer_with_perception(
                        trajectory_buffer,
                        perception_fn,
                        history_window=eb_config.perception_history_window,
                    )

                # Inject updated beliefs for next step
                if not done:
                    _inject_beliefs(
                        config,
                        agent,
                        env,
                        env_name,
                        task,
                        beliefs,
                        agent_goal=agent_goal,
                    )

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
            step_total_cost = (
                agent_step_cost
                + step_extract_cost
                + step_improve_cost
                + step_experiment_cost
                + step_trim_cost
                + step_critical_cost
            )
            cumulative_step_cost += step_total_cost

            did_learn = (
                should_update_artifacts
                or should_improve
                or should_gen_experiments
                or did_critical_id_this_step
            )
            if did_learn:
                _save_step_artifacts_eb(
                    step_dir,
                    beliefs,
                    perception,
                    qa_pairs,
                    step_feedback_records,
                    extraction_log=step_extraction_log,
                    experiment_log=step_experiment_log,
                    trim_log=step_trim_log,
                )

            num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
            _save_step_log_eb(
                step_dir=step_dir,
                step=step,
                global_step=global_step,
                action=action,
                reward=reward,
                done=done,
                episode_return=episode_return,
                agent_cost=agent_step_cost,
                extract_cost=step_extract_cost,
                improve_cost=step_improve_cost,
                experiment_cost=step_experiment_cost,
                trim_cost=step_trim_cost,
                num_qa=len(qa_pairs),
                num_unanswered=num_unanswered,
                did_gen_questions=did_gen_questions,
                did_formulate_experiment=did_formulate_experiment,
                did_trim=did_trim_step,
                active_experiment=current_experiment,
                phase="complete",
                active_experiment_question=current_experiment_question,
                env_info=info if isinstance(info, dict) else None,
                critical_cost=step_critical_cost,
                did_critical_id=did_critical_id_this_step,
                critical=critical_this_step if did_critical_id_this_step else None,
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
                    "did_critical_id": did_critical_id_this_step,
                    "critical": critical_this_step
                    if did_critical_id_this_step
                    else None,
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
            csv_writer.writerow(
                [step + 1, "", "", result_obs_text, new_raw_short, 0.0, done]
            )
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
    episode_log["num_answered_questions"] = sum(
        1 for q in qa_pairs if q.answer is not None
    )
    episode_log["num_unanswered_questions"] = sum(
        1 for q in qa_pairs if q.answer is None
    )

    json_filename = ep_dir / "episode_log.json"
    with open(json_filename, "w") as f:
        json.dump(episode_log, f, indent=4, default=str)

    _append_pending_agent_action_for_history(agent, action)
    agent_history_events = _snapshot_agent_history_events(agent)

    env.close()

    evolve_logger.info(
        f"Episode {episode_idx} complete — return: {episode_return:.2f}, "
        f"steps: {step + 1}, "
        f"learn cost: ${total_learn_cost:.4f}, agent cost: ${episode_log['total_cost']:.4f}"
    )

    return (
        beliefs,
        perception,
        qa_pairs,
        current_experiment,
        current_experiment_question,
        episode_log,
        step + 1,
        trajectory_buffer,
        past_experiments,
        agent_history_events,
        theories,
        frontier,
    )


def _run_frozen_autumn_task_eval(
    *,
    config: DictConfig,
    eb_config: StepwiseEBLearnConfig,
    beliefs: str,
    perception: str,
    program: str,
    task_type: str,
    output_dir: Path,
) -> dict:
    """Score learned EB artifacts on one AutumnBench evaluation task."""
    from autumn_env import AutumnBenchEnvWrapper

    autumn_kwargs = getattr(getattr(config, "envs", None), "autumn_kwargs", None)

    def _cfg(name: str, default):
        if autumn_kwargs is None:
            return default
        value = getattr(autumn_kwargs, name, None)
        return default if value is None else value

    seed = getattr(config.envs.env_kwargs, "seed", None)
    if seed is None:
        seed = get_unique_seed(process_num=0, episode_idx=0)
    env_kwargs = {
        "env_name": program,
        "task_type": task_type,
        "max_episode_steps": eb_config.autumn_eval_max_steps,
        "max_interaction_steps": int(
            _cfg("max_interaction_steps", _cfg("max_episode_steps", 300))
        ),
        "seed": seed,
        "stack_frames": bool(_cfg("stack_frames", False)),
        "skip_frames": bool(_cfg("skip_frames", False)),
        "render_mode": eb_config.autumn_eval_render_mode,
        "logging_path": str(output_dir / "autumn_env_logs"),
    }
    data_dir = _cfg("data_dir", None)
    if data_dir is not None:
        env_kwargs["data_dir"] = str(data_dir)
    env = AutumnBenchEnvWrapper(**env_kwargs)

    agent = AgentFactory(config).create_agent()
    agent.reset()

    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs, info = env.reset(seed=seed)

    artifact_eval_cfg = getattr(config, "artifact_eval", None)
    start_in_test_phase = bool(
        getattr(artifact_eval_cfg, "autumn_eval_start_in_test_phase", False)
    )
    transition_info = None
    if start_in_test_phase and task_type in {"mfp", "cd", "planning"}:
        obs, reward, terminated, truncated, info = env.step("go-to-test")
        transition_info = {
            "action": "go-to-test",
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "env_stats": env.get_stats(),
        }
        with open(output_dir / "autumn_eval_transition.json", "w") as f:
            json.dump(transition_info, f, indent=4, default=str)

    eval_goal = _frozen_autumn_eval_goal(task_type, config, eb_config)
    _inject_beliefs(
        config,
        agent,
        env,
        "autumn",
        program,
        beliefs,
        goal_override=eval_goal,
    )

    perception_fn = (load_perception_fn(perception) if eb_config.perception_enabled else None)
    raw_obs_history = [
        (obs.get("planning_eval") or {}).get(
            "current_state_text", obs["text"]["long_term_context"]
        )
    ]
    planning_perception = None
    if perception_fn is not None:
        planning_perception = _apply_autumn_planning_perception(obs, perception_fn)
        if planning_perception is None:
            apply_perception_with_history(
                obs, perception_fn, raw_obs_history, eb_config.perception_history_window
            )

    if eb_config.mock_mode:
        set_mock_action_provider(lambda: random.choice(_mock_available_actions(env)))

    trajectory: list[dict] = []
    episode_return = 0.0
    total_cost = 0.0
    input_tokens = 0
    output_tokens = 0
    action = None
    done = False
    step = 0

    for step in range(eb_config.autumn_eval_max_steps):
        step_dir = output_dir / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        raw_before = obs["text"]["long_term_context"]
        aux_before = obs["text"].get("short_term_context", "")
        before_image = obs.get("image")
        before_images = obs.get("images", [])
        planning_perception_before = planning_perception

        response = agent.act(obs, prev_action=action)
        action = response.completion
        reasoning = getattr(response, "reasoning", "")
        total_cost += getattr(response, "cost", 0.0)
        input_tokens += getattr(response, "input_tokens", 0)
        output_tokens += getattr(response, "output_tokens", 0)
        _save_eval_agent_messages(step_dir, agent, reasoning, action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward

        raw_after = obs["text"]["long_term_context"]
        aux_after = obs["text"].get("short_term_context", "")
        after_image = obs.get("image")
        after_images = obs.get("images", [])
        image_flags = _save_eval_step_images(
            step_dir,
            before_image,
            after_image,
            before_images=before_images,
            after_images=after_images,
        )
        raw_obs_history.append(
            (obs.get("planning_eval") or {}).get("current_state_text", raw_after)
        )
        planning_perception = None
        if perception_fn is not None:
            planning_perception = _apply_autumn_planning_perception(obs, perception_fn)
            if planning_perception is None:
                apply_perception_with_history(
                    obs,
                    perception_fn,
                    raw_obs_history,
                    eb_config.perception_history_window,
                )

        trajectory.append(
            {
                "step": step,
                "action": action,
                "reasoning": reasoning,
                "reward": reward,
                "done": done,
                "info": info,
                "phase": env.get_stats().get("phase"),
                "pre_observation": raw_before,
                "pre_auxiliary_observation": aux_before,
                "post_observation": raw_after,
                "post_auxiliary_observation": aux_after,
                "planning_perception": planning_perception_before,
                **image_flags,
            }
        )

        if done:
            break

    result = {
        "program": program,
        "task_type": task_type,
        "eval_goal": eval_goal,
        "episode_return": episode_return,
        "num_steps": step + 1,
        "done": done,
        "failed_candidates": env.failed_candidates,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": total_cost,
        "env_stats": env.get_stats(),
    }
    if transition_info is not None:
        result["pre_eval_transition"] = transition_info
    with open(output_dir / "episode_log.json", "w") as f:
        json.dump(result, f, indent=4, default=str)
    with open(output_dir / "trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2, default=str)
    env.close()
    return result


def run_frozen_autumn_evaluation(
    *,
    config: DictConfig,
    eb_config: StepwiseEBLearnConfig,
    beliefs: str,
    perception: str,
    output_dir: str,
) -> dict:
    """Run AutumnBench test phases with learned artifacts frozen."""
    task_types = eb_config.autumn_eval_task_types or ["mfp", "cd", "planning"]
    programs = list(config.tasks.autumn_tasks)
    eval_root = Path(output_dir) / "autumn_frozen_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    aggregate_reward = 0.0
    total_cost = 0.0
    for program in programs:
        for task_type in task_types:
            run_key = f"{program}_{task_type}"
            run_dir = eval_root / run_key
            evolve_logger.info(f"Running frozen Autumn eval: {run_key}")
            try:
                result = _run_frozen_autumn_task_eval(
                    config=config,
                    eb_config=eb_config,
                    beliefs=beliefs,
                    perception=perception,
                    program=program,
                    task_type=task_type,
                    output_dir=run_dir,
                )
            except Exception as e:
                logging.exception("Frozen Autumn eval failed")
                result = {
                    "program": program,
                    "task_type": task_type,
                    "error": str(e),
                    "episode_return": 0.0,
                }
            results[run_key] = result
            aggregate_reward += float(result.get("episode_return", 0.0))
            total_cost += float(result.get("total_cost", 0.0))

    summary = {
        "aggregate_reward": aggregate_reward,
        "num_tasks": len(results),
        "total_cost": total_cost,
        "results": results,
    }
    with open(eval_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)
    evolve_logger.info(
        f"Frozen Autumn eval complete: aggregate_reward={aggregate_reward:.3f}, "
        f"num_tasks={len(results)}, cost=${total_cost:.4f}"
    )
    return summary


def _frozen_eval_goal(
    env_name: str, config: DictConfig, eb_config: StepwiseEBLearnConfig
) -> str:
    if env_name == "minihack":
        return eb_config.frozen_eval_minihack_goal or resolve_agent_goal(config)
    if env_name == "arc_agi":
        return eb_config.frozen_eval_arc_agi_goal or resolve_agent_goal(config)
    return resolve_agent_goal(config)


def _frozen_autumn_eval_goal(
    task_type: str, config: DictConfig, eb_config: StepwiseEBLearnConfig
) -> str:
    if task_type == "planning" and eb_config.frozen_eval_autumn_planning_goal:
        return eb_config.frozen_eval_autumn_planning_goal
    return resolve_agent_goal(config)


def _make_frozen_eval_env(env_name: str, task: str, config: DictConfig):
    if env_name == "arc_agi":
        from arc_agi_env import make_arc_env

        return make_arc_env(task, config)
    return make_env(env_name, task, config)


def _run_frozen_task_eval(
    *,
    config: DictConfig,
    eb_config: StepwiseEBLearnConfig,
    env_name: str,
    task: str,
    beliefs: str,
    perception: str,
    output_dir: Path,
    eval_goal: str,
    seed_index: int = 0,
) -> dict:
    """Run one non-learning evaluation episode with learned artifacts frozen."""
    env = _make_frozen_eval_env(env_name, task, config)
    agent = AgentFactory(config).create_agent()
    agent.reset()

    seed = getattr(config.envs.env_kwargs, "seed", None)
    if seed is None:
        seed = get_unique_seed(process_num=0, episode_idx=seed_index)
    random.seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)

    _inject_beliefs(
        config,
        agent,
        env,
        env_name,
        task,
        beliefs,
        goal_override=eval_goal,
    )

    perception_fn = (load_perception_fn(perception) if eb_config.perception_enabled else None)
    raw_obs_history = [obs["text"]["long_term_context"]]
    if perception_fn is not None:
        apply_perception_with_history(
            obs, perception_fn, raw_obs_history, eb_config.perception_history_window
        )

    if eb_config.mock_mode:
        set_mock_action_provider(lambda: random.choice(_mock_available_actions(env)))

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory: list[dict] = []
    episode_return = 0.0
    total_cost = 0.0
    input_tokens = 0
    output_tokens = 0
    action_frequency: dict[str, int] = defaultdict(int)
    failed_actions: list[str] = []
    action = None
    done = False
    step = 0
    final_info = info

    for step in range(eb_config.frozen_eval_max_steps):
        step_dir = output_dir / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        raw_before = raw_obs_history[-1]
        aux_before = obs["text"].get("short_term_context", "")
        before_image = obs.get("image")

        response = agent.act(obs, prev_action=action)
        action = response.completion
        reasoning = getattr(response, "reasoning", "")
        total_cost += getattr(response, "cost", 0.0)
        input_tokens += getattr(response, "input_tokens", 0)
        output_tokens += getattr(response, "output_tokens", 0)
        action_frequency[action] += 1
        _save_eval_agent_messages(step_dir, agent, reasoning, action)

        invalid_action = False
        try:
            obs, reward, terminated, truncated, final_info = env.step(action)
        except ValueError as e:
            logging.warning(f"Frozen eval invalid action: {action} — {e}")
            invalid_action = True
            failed_actions.append(action)
            if config.eval.feedback_on_invalid_action:
                obs["text"]["long_term_context"] = (
                    f"\n\n{INVALID_ACTION_RETRY_MESSAGE}\n\n"
                    f"Observation:\n{obs['text']['long_term_context']}"
                )
            reward = 0.0
            terminated = False
            truncated = False
            final_info = {"invalid_action_error": str(e)}

        done = terminated or truncated
        episode_return += reward

        raw_after = obs["text"]["long_term_context"]
        aux_after = obs["text"].get("short_term_context", "")
        after_image = obs.get("image")
        image_flags = _save_eval_step_images(step_dir, before_image, after_image)
        raw_obs_history.append(raw_after)
        if perception_fn is not None and not invalid_action:
            apply_perception_with_history(
                obs, perception_fn, raw_obs_history, eb_config.perception_history_window
            )

        trajectory.append(
            {
                "step": step,
                "action": action,
                "reasoning": reasoning,
                "reward": reward,
                "done": done,
                "info": final_info,
                "pre_observation": raw_before,
                "pre_auxiliary_observation": aux_before,
                "post_observation": raw_after,
                "post_auxiliary_observation": aux_after,
                **image_flags,
            }
        )

        if done:
            break

    env_stats = {}
    if hasattr(env, "get_stats"):
        try:
            env_stats = env.get_stats()
        except Exception:
            logging.exception("Failed to collect frozen eval env stats")

    result = {
        "env_name": env_name,
        "task": task,
        "eval_goal": eval_goal,
        "episode_return": episode_return,
        "num_steps": len(trajectory),
        "done": done,
        "final_info": final_info,
        "failed_actions": failed_actions,
        "failed_candidates": getattr(env, "failed_candidates", []),
        "action_frequency": dict(action_frequency),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": total_cost,
        "env_stats": env_stats,
    }
    with open(output_dir / "episode_log.json", "w") as f:
        json.dump(result, f, indent=4, default=str)
    with open(output_dir / "trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2, default=str)
    if hasattr(env, "close"):
        env.close()
    return result


def run_frozen_environment_evaluation(
    *,
    config: DictConfig,
    eb_config: StepwiseEBLearnConfig,
    beliefs: str,
    perception: str,
    output_dir: str,
) -> dict:
    """Run frozen post-learning evaluation for MiniHack and ARC-AGI-3."""
    active_env = config.envs.names.split("-")[0]
    env_names = eb_config.frozen_eval_envs or [active_env]
    eval_root = Path(output_dir) / "frozen_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    aggregate_reward = 0.0
    total_cost = 0.0
    seed_index = 0
    for env_name in env_names:
        if env_name not in ("minihack", "arc_agi"):
            evolve_logger.info(
                f"Skipping generic frozen eval for unsupported env: {env_name}"
            )
            continue

        tasks = list(config.tasks[f"{env_name}_tasks"])
        eval_goal = _frozen_eval_goal(env_name, config, eb_config)
        for task in tasks:
            run_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{env_name}_{task}")
            run_dir = eval_root / run_key
            evolve_logger.info(f"Running frozen eval: {run_key}")
            try:
                result = _run_frozen_task_eval(
                    config=config,
                    eb_config=eb_config,
                    env_name=env_name,
                    task=task,
                    beliefs=beliefs,
                    perception=perception,
                    output_dir=run_dir,
                    eval_goal=eval_goal,
                    seed_index=seed_index,
                )
            except Exception as e:
                logging.exception("Frozen eval failed")
                result = {
                    "env_name": env_name,
                    "task": task,
                    "eval_goal": eval_goal,
                    "error": str(e),
                    "episode_return": 0.0,
                }
            results[run_key] = result
            aggregate_reward += float(result.get("episode_return", 0.0))
            total_cost += float(result.get("total_cost", 0.0))
            seed_index += 1

    summary = {
        "aggregate_reward": aggregate_reward,
        "num_tasks": len(results),
        "total_cost": total_cost,
        "results": results,
    }
    with open(eval_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)
    evolve_logger.info(
        f"Frozen eval complete: aggregate_reward={aggregate_reward:.3f}, "
        f"num_tasks={len(results)}, cost=${total_cost:.4f}"
    )
    return summary


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

    # Route all meta/learning LLM calls through eb_config.explore_temp.
    # The BALROG agent continues to use config.client.generate_kwargs.temperature.
    set_meta_temperature(eb_config.explore_temp)

    # Check for resume
    last_ep, beliefs, perception, qa_pairs, theories, frontier = (
        _find_last_completed_episode_eb(output_dir)
    )
    start_episode = last_ep + 1

    current_experiment: str | None = None
    current_experiment_question: str | None = None
    trajectory_buffer: list[dict] = []
    past_experiments: list[str] = []
    agent_history_events: list[dict] = []
    global_steps_used = 0
    if start_episode > 0:
        num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
        evolve_logger.info(
            f"Resuming from episode {start_episode} ({len(qa_pairs)} QA, {num_unanswered} unanswered)"
        )
        # Recover active experiment from last episode's step logs
        last_ep_dir = Path(output_dir) / f"episode_{last_ep}"
        for step_dir in sorted(last_ep_dir.glob("step_*"), reverse=True):
            sl_file = step_dir / "step_log.json"
            if sl_file.exists():
                try:
                    sl = json.loads(sl_file.read_text())
                    current_experiment = sl.get("active_experiment")
                    current_experiment_question = sl.get("active_experiment_question")
                    break
                except (json.JSONDecodeError, TypeError):
                    pass
        if current_experiment and not current_experiment_question:
            for step_dir in sorted(last_ep_dir.glob("step_*"), reverse=True):
                exp_file = step_dir / "experiment_log.json"
                if exp_file.exists():
                    try:
                        exp_log = json.loads(exp_file.read_text())
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if exp_log.get("active_experiment") == current_experiment:
                        current_experiment_question = exp_log.get(
                            "active_experiment_question"
                        ) or exp_log.get("selected_question")
                        break
                    if exp_log.get("experiment_plan") == current_experiment:
                        current_experiment_question = exp_log.get("selected_question")
                        break
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
        past_exp_file = (
            Path(output_dir) / f"episode_{last_ep}" / "past_experiments.json"
        )
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

    agent_goal = resolve_agent_goal(config)
    default_knowledge = append_agent_goal(get_default_knowledge(config), agent_goal)
    evolve_logger.info(f"Default knowledge: {len(default_knowledge)} chars")
    evolve_logger.info(f"Agent goal: {agent_goal}")

    evolve_logger.info(f"Stepwise EB-learn config:")
    evolve_logger.info(f"  Total env steps: {eb_config.n_environment_steps}")
    evolve_logger.info(
        f"  Improve iterations: perception={eb_config.max_perception_iterations}, qa={eb_config.max_qa_iterations} (schedule or fixed)"
    )
    evolve_logger.info(
        f"  Artifact update interval: {eb_config.artifact_update_interval}"
    )
    evolve_logger.info(f"  Improve interval: {eb_config.improve_interval}")
    evolve_logger.info(f"  Experiment interval: {eb_config.experiment_interval}")
    evolve_logger.info(f"  Num questions per gen: {eb_config.num_questions}")
    evolve_logger.info(f"  Max answered QA pairs: {eb_config.max_answered_qa_pairs}")
    evolve_logger.info(
        f"  Max unanswered QA pairs: {eb_config.max_unanswered_qa_pairs}"
    )
    evolve_logger.info(
        f"  Trim unanswered at selection: {eb_config.trim_unanswered_at_selection}"
    )
    evolve_logger.info(
        f"  Question scoring method: {eb_config.question_scoring_method}"
    )
    evolve_logger.info(
        f"  Question scoring max concurrent: {eb_config.question_scoring_max_concurrent}"
    )
    evolve_logger.info(
        f"  Max steps context chars: {eb_config.max_steps_context_chars}"
    )
    evolve_logger.info(f"  Max images context: {eb_config.max_images_context}")
    evolve_logger.info(f"  Explore temp: {eb_config.explore_temp}")
    evolve_logger.info("  Question gen trajectory context: enabled")
    evolve_logger.info(f"  Include policy: {eb_config.include_policy}")
    if eb_config.mock_mode:
        evolve_logger.info(
            f"  MOCK MODE: enabled — no LLM calls; random actions and artifact perturbations"
        )

    cumulative_cost = 0.0
    episode_idx = start_episode
    env_name = config.envs.names.split("-")[0]

    while global_steps_used < eb_config.n_environment_steps:
        remaining_steps = eb_config.n_environment_steps - global_steps_used

        num_unanswered = sum(1 for q in qa_pairs if q.answer is None)
        evolve_logger.info(f"\n{'=' * 80}")
        evolve_logger.info(
            f"STEPWISE EB-LEARN EPISODE {episode_idx} "
            f"(global steps: {global_steps_used}/{eb_config.n_environment_steps}, "
            f"remaining: {remaining_steps})"
        )
        evolve_logger.info(
            f"QA pairs: {len(qa_pairs)} ({num_unanswered} unanswered), Experiment: {current_experiment or 'none'}"
        )
        evolve_logger.info(f"{'=' * 80}")

        episode_dir = Path(output_dir) / f"episode_{episode_idx}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save input state
        (episode_dir / "input_beliefs.txt").write_text(beliefs)
        (episode_dir / "input_perception.py").write_text(perception)
        (
            beliefs,
            perception,
            qa_pairs,
            current_experiment,
            current_experiment_question,
            episode_log,
            steps_taken,
            trajectory_buffer,
            past_experiments,
            agent_history_events,
            theories,
            frontier,
        ) = run_stepwise_eb_learn_episode(
            config=config,
            eb_config=eb_config,
            beliefs=beliefs,
            perception=perception,
            qa_pairs=qa_pairs,
            current_experiment=current_experiment,
            current_experiment_question=current_experiment_question,
            default_knowledge=default_knowledge,
            output_dir=str(episode_dir),
            episode_idx=episode_idx,
            global_step_start=global_steps_used,
            max_episode_steps=remaining_steps,
            trajectory_buffer=trajectory_buffer,
            past_experiments=past_experiments,
            agent_history_events=agent_history_events,
            cumulative_cost_offset=cumulative_cost,
            theories=theories,
            frontier=frontier,
        )

        global_steps_used += steps_taken

        # Save episode artifacts
        _save_episode_artifacts_eb(
            episode_dir,
            beliefs,
            perception,
            qa_pairs,
            trajectory_buffer=trajectory_buffer,
            past_experiments=past_experiments,
            theories=theories,
            frontier=frontier,
        )

        episode_cost = episode_log.get("total_cost", 0.0) + episode_log.get(
            "total_learn_cost", 0.0
        )
        cumulative_cost += episode_cost
        evolve_logger.info(
            f"[g{global_steps_used}] Episode {episode_idx} done — "
            f"cost: ${episode_cost:.4f}, cumulative: ${cumulative_cost:.4f}, "
            f"steps: {global_steps_used}/{eb_config.n_environment_steps}"
        )

        if is_minihack_success_episode(env_name, episode_log):
            evolve_logger.info(
                f"[g{global_steps_used}] MiniHack task success reached; "
                "stopping run before starting another episode."
            )
            break

        episode_idx += 1

    if (
        eb_config.autumn_eval_after_learn
        and config.envs.names.split("-")[0] == "autumn"
    ):
        run_frozen_autumn_evaluation(
            config=config,
            eb_config=eb_config,
            beliefs=beliefs,
            perception=perception,
            output_dir=output_dir,
        )
    frozen_eval_envs = eb_config.frozen_eval_envs or [config.envs.names.split("-")[0]]
    if eb_config.frozen_eval_after_learn and any(
        env_name in ("minihack", "arc_agi") for env_name in frozen_eval_envs
    ):
        run_frozen_environment_evaluation(
            config=config,
            eb_config=eb_config,
            beliefs=beliefs,
            perception=perception,
            output_dir=output_dir,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_stepwise_eb_config(config: DictConfig) -> StepwiseEBLearnConfig:
    """Build the EB config from Hydra config using the main script defaults."""
    evolve_cfg = config.eval.evolve

    legacy_max_total_qa_pairs = evolve_cfg.get("max_total_qa_pairs", 50)
    frozen_eval_envs = evolve_cfg.get("frozen_eval_envs", None)
    return StepwiseEBLearnConfig(
        n_environment_steps=evolve_cfg.get("n_environment_steps", 100),
        max_perception_iterations=evolve_cfg.get(
            "max_perception_iterations",
            evolve_cfg.get(
                "max_improve_iterations", evolve_cfg.get("num_improve_iterations", 5)
            ),
        ),
        max_qa_iterations=evolve_cfg.get(
            "max_qa_iterations",
            evolve_cfg.get(
                "max_improve_iterations", evolve_cfg.get("num_improve_iterations", 5)
            ),
        ),
        max_qa_per_forward=evolve_cfg.get("max_qa_per_forward", 10),
        max_answered_qa_pairs=evolve_cfg.get(
            "max_answered_qa_pairs", legacy_max_total_qa_pairs
        ),
        max_unanswered_qa_pairs=evolve_cfg.get(
            "max_unanswered_qa_pairs", legacy_max_total_qa_pairs
        ),
        trim_unanswered_at_selection=evolve_cfg.get(
            "trim_unanswered_at_selection", False
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
        question_gen_current_state_only=evolve_cfg.get(
            "question_gen_current_state_only", False
        ),
        include_policy=evolve_cfg.get("include_policy", True),
        perception_enabled=evolve_cfg.get("perception_enabled", True),
        question_scoring_method=evolve_cfg.get(
            "question_scoring_method", "b_diff_light"
        ),
        question_scoring_max_concurrent=evolve_cfg.get(
            "question_scoring_max_concurrent", 8
        ),
        num_theories=evolve_cfg.get("num_theories", 5),
        num_crux_questions=evolve_cfg.get("num_crux_questions", 5),
        num_theory_seed_questions=evolve_cfg.get("num_theory_seed_questions", 0),
        theory_gen_current_state_only=evolve_cfg.get(
            "theory_gen_current_state_only", False
        ),
        theory_weight_decay=evolve_cfg.get("theory_weight_decay", 0.6),
        theory_violation_penalty=evolve_cfg.get("theory_violation_penalty", 0.7),
        theory_min_weight=evolve_cfg.get("theory_min_weight", 0.02),
        num_candidate_actions=evolve_cfg.get("num_candidate_actions", 4),
        exploit_enabled=evolve_cfg.get("exploit_enabled", True),
        exploit_stable_streak=evolve_cfg.get("exploit_stable_streak", 2),
        exploit_min_theories=evolve_cfg.get("exploit_min_theories", 2),
        experiment_selection_mode=evolve_cfg.get("experiment_selection_mode", "single"),
        experiment_scoring_max_concurrent=evolve_cfg.get(
            "experiment_scoring_max_concurrent", 8
        ),
        score_topk_filter_questions=evolve_cfg.get(
            "score_topk_filter_questions", False
        ),
        critical_transitions_enabled=evolve_cfg.get(
            "critical_transitions_enabled", False
        ),
        critical_id_min_for_perception=evolve_cfg.get(
            "critical_id_min_for_perception", 3
        ),
        mock_mode=evolve_cfg.get("mock_mode", False),
        frozen_eval_after_learn=evolve_cfg.get("frozen_eval_after_learn", False),
        frozen_eval_envs=(
            list(frozen_eval_envs) if frozen_eval_envs is not None else None
        ),
        frozen_eval_max_steps=evolve_cfg.get("frozen_eval_max_steps", 501),
        frozen_eval_minihack_goal=evolve_cfg.get(
            "frozen_eval_minihack_goal",
            StepwiseEBLearnConfig.frozen_eval_minihack_goal,
        ),
        frozen_eval_arc_agi_goal=evolve_cfg.get(
            "frozen_eval_arc_agi_goal",
            StepwiseEBLearnConfig.frozen_eval_arc_agi_goal,
        ),
        autumn_eval_after_learn=evolve_cfg.get("autumn_eval_after_learn", False),
        autumn_eval_task_types=list(
            evolve_cfg.get("autumn_eval_task_types", ["mfp", "cd", "planning"])
        ),
        autumn_eval_max_steps=evolve_cfg.get("autumn_eval_max_steps", 501),
        autumn_eval_render_mode=evolve_cfg.get("autumn_eval_render_mode", "text"),
        frozen_eval_autumn_planning_goal=evolve_cfg.get(
            "frozen_eval_autumn_planning_goal",
            StepwiseEBLearnConfig.frozen_eval_autumn_planning_goal,
        ),
        frontier_learner=evolve_cfg.get("frontier_learner", "gepa"),
        frontier_click_aware=evolve_cfg.get("frontier_click_aware", True),
        frontier_size=evolve_cfg.get("frontier_size", 3),
        frontier_relearn_interval=evolve_cfg.get("frontier_relearn_interval", 10),
        frontier_min_buffer=evolve_cfg.get("frontier_min_buffer", 12),
        frontier_max_metric_calls=evolve_cfg.get("frontier_max_metric_calls", 80),
        frontier_legacy_rounds=evolve_cfg.get("frontier_legacy_rounds", 6),
        frontier_pop_size=evolve_cfg.get("frontier_pop_size", 4),
        frontier_pop_rounds=evolve_cfg.get("frontier_pop_rounds", 6),
        frontier_image_mode=evolve_cfg.get("frontier_image_mode", "auto"),
        frontier_image_max_transitions=evolve_cfg.get(
            "frontier_image_max_transitions", 16
        ),
        frontier_k_choices=evolve_cfg.get("frontier_k_choices", 5),
        frontier_train_n=evolve_cfg.get("frontier_train_n", 14),
        frontier_val_n=evolve_cfg.get("frontier_val_n", 12),
        frontier_test_n=evolve_cfg.get("frontier_test_n", 10),
        frontier_fd_scorer=evolve_cfg.get("frontier_fd_scorer", "none"),
        frontier_fd_weight=evolve_cfg.get("frontier_fd_weight", 0.5),
        frontier_concurrency=evolve_cfg.get("frontier_concurrency", 8),
        frontier_task_model=evolve_cfg.get("frontier_task_model", None),
        frontier_reflection_model=evolve_cfg.get("frontier_reflection_model", None),
    )


@hydra.main(
    config_path="BALROG/balrog/config", config_name="config", version_base="1.1"
)
@hydra.main(
    config_path="BALROG/balrog/config", config_name="config", version_base="1.1"
)
def main(config: DictConfig):
    run_name_suffix = f"{config.agent.type}_{config.client.model_id.replace('/', '_')}_stepwise_eb_learn"

    original_cwd, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=config.eval.resume_from,
        output_dir_base=config.eval.output_dir,
        logger_name="evolve",
    )

    eb_config = build_stepwise_eb_config(config)

    stepwise_eb_learn(
        eb_config=eb_config,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
