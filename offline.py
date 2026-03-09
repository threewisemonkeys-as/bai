import asyncio
import csv as csv_module
import json
import logging
import os
import sys
from pathlib import Path

import hydra
import litellm
from omegaconf import DictConfig

from run_utils import setup_run, improve_logging, _update_summary_json
from improve import (
    _get_response_cost,
    get_episode_outcome_header,
    trim_to_model_context_lim,
)
from llm_utils import build_llm_input, extract_llm_response_text, extract_xml_kv, validate_response_fields


logger = logging.getLogger("offline")


def _get_model_name(config: DictConfig) -> str:
    if config.client.client_name == "vllm":
        return f"hosted_vllm/{config.client.model_id}"
    return f"{config.client.client_name}/{config.client.model_id}"


# ---------------------------------------------------------------------------
# 1. Load trajectory batch
# ---------------------------------------------------------------------------

def load_trajectory_batch(
    trajectories_dir: str,
    batch_index: int,
    batch_size: int,
) -> list[tuple[str, str, str]]:
    """Load a batch of trajectories, cycling if batch exceeds total.

    Args:
        trajectories_dir: Directory containing .csv trajectory files
        batch_index: 0-based batch index
        batch_size: Number of trajectories per batch

    Returns:
        List of (csv_path, outcome_header, traj_text) tuples
    """
    traj_dir = Path(trajectories_dir)
    csv_files = sorted(traj_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No .csv trajectory files found in {trajectories_dir}")

    total = len(csv_files)
    start = (batch_index * batch_size) % total
    batch = []
    for i in range(batch_size):
        idx = (start + i) % total
        csv_path = str(csv_files[idx])
        outcome_header = get_episode_outcome_header(csv_path)
        traj_text = csv_files[idx].read_text()
        batch.append((csv_path, outcome_header, traj_text))

    return batch


# ---------------------------------------------------------------------------
# 2. Find last completed offline step
# ---------------------------------------------------------------------------

def find_last_completed_offline_step(
    output_dir: str,
) -> tuple[int, str, str]:
    """Find last completed step and extract world_knowledge + policy.

    Returns:
        (last_step, world_knowledge, policy)
        Returns (0, "", "") if no completed steps found.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0, "", ""

    step_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            try:
                step_num = int(item.name.split("_")[1])
                beliefs_file = item / "beliefs.txt"
                if beliefs_file.exists():
                    step_dirs.append((step_num, beliefs_file))
            except (ValueError, IndexError):
                continue

    if not step_dirs:
        return 0, "", ""

    step_dirs.sort(key=lambda x: x[0])
    last_step_num, last_beliefs_file = step_dirs[-1]
    beliefs_content = last_beliefs_file.read_text()

    parsed = extract_xml_kv(beliefs_content, ["world_knowledge", "policy"])
    world_knowledge = parsed.get("world_knowledge", "").strip()
    policy = parsed.get("policy", "").strip()

    return last_step_num, world_knowledge, policy


# ---------------------------------------------------------------------------
# 3. Summarize a single trajectory (offline, async)
# ---------------------------------------------------------------------------

async def _summarize_trajectory_offline_async(
    config: DictConfig,
    outcome_header: str,
    traj_text: str,
    traj_path: str,
    current_knowledge: str,
) -> tuple[str, float]:
    """Summarize a single trajectory for offline learning.

    Focuses on mechanics, cause-effect, patterns, and contradictions
    with current understanding. No perception module references.
    """
    model_name = _get_model_name(config)
    traj_text = trim_to_model_context_lim(traj_text, config.client.model_id)

    prompt = f"""We are analyzing interactions with an environment to understand how it works.

Our current understanding of the environment:
=== CURRENT KNOWLEDGE ===
{current_knowledge if current_knowledge else "(empty - no knowledge yet)"}
=== END OF CURRENT KNOWLEDGE ===

Here is an episode outcome and the interaction transcript:
=== EPISODE OUTCOME ===
{outcome_header}
=== END OF EPISODE OUTCOME ===
=== INTERACTION TRANSCRIPT ===
{traj_text}
=== END OF INTERACTION TRANSCRIPT ===

Analyze this interaction and provide a summary focusing on:
1. **Mechanics**: What rules or mechanics of the environment are revealed by this interaction?
2. **Cause-effect**: What actions led to what outcomes? What are the causal relationships?
3. **Patterns**: Are there recurring patterns in how the environment responds?
4. **Contradictions**: Does anything in this interaction contradict our current understanding?
5. **Key learnings**: What are the most important takeaways for improving future performance?

Format your response in XML style as:
<think>
Analyze the interaction carefully, tracing cause and effect.
</think>
<summary>
Your concise summary covering the points above.
</summary>
"""

    input_data = build_llm_input(prompt)

    logging.info(f"Calling LLM to summarize trajectory: {traj_path}")
    logging.info(f"Offline summary prompt:\n{prompt}")

    response = await asyncio.to_thread(
        litellm.responses,
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    response_text = extract_llm_response_text(response)
    logging.info(f"Offline summary LLM response for {traj_path}:\n{response_text}")

    cost = _get_response_cost(response, config.client.model_id)
    response_dict = extract_xml_kv(response_text, ["summary"])

    if not validate_response_fields(response_dict, response_text, ["summary"]):
        return "", cost
    return response_dict["summary"], cost


# ---------------------------------------------------------------------------
# 4. Summarize a batch of trajectories (parallel async)
# ---------------------------------------------------------------------------

def _load_summary_cache(output_dir: str) -> dict[str, str]:
    """Load the trajectory summary cache from the run's output directory."""
    cache_path = Path(output_dir) / "summary_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_summary_cache(output_dir: str, cache: dict[str, str]):
    """Save the trajectory summary cache to the run's output directory."""
    cache_path = Path(output_dir) / "summary_cache.json"
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def summarize_trajectory_batch(
    config: DictConfig,
    batch: list[tuple[str, str, str]],
    current_knowledge: str,
    summary_cache: dict[str, str] | None = None,
) -> tuple[str, float, dict[str, str]]:
    """Summarize a batch of trajectories in parallel, skipping cached ones.

    Args:
        config: Configuration
        batch: List of (csv_path, outcome_header, traj_text) tuples
        current_knowledge: Current world knowledge
        summary_cache: Optional dict mapping csv_path -> cached summary text.
            Trajectories found in the cache are not re-summarized.

    Returns:
        (combined_summaries_xml, total_cost, updated_cache)
    """
    if summary_cache is None:
        summary_cache = {}

    # Split batch into cached and uncached
    uncached_indices = []
    for i, (csv_path, _, _) in enumerate(batch):
        if csv_path not in summary_cache:
            uncached_indices.append(i)

    cached_count = len(batch) - len(uncached_indices)
    if cached_count > 0:
        logging.info(f"Summary cache: {cached_count} cached, {len(uncached_indices)} to summarize")

    # Summarize only uncached trajectories
    total_cost = 0.0
    if uncached_indices:
        async def _gather():
            tasks = [
                _summarize_trajectory_offline_async(
                    config,
                    batch[i][1],  # outcome_header
                    batch[i][2],  # traj_text
                    batch[i][0],  # csv_path
                    current_knowledge,
                )
                for i in uncached_indices
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_gather())

        for idx, (summary, cost) in zip(uncached_indices, results):
            csv_path = batch[idx][0]
            summary_cache[csv_path] = summary
            total_cost += cost

    # Build combined XML from all batch items (preserving batch order)
    summaries_xml = "<episode_summaries>\n"
    for i, (csv_path, _, _) in enumerate(batch):
        summary = summary_cache.get(csv_path, "")
        summaries_xml += f"<episode_summary episode_idx={i+1}>\n{summary.strip()}\n</episode_summary>\n"
    summaries_xml += "</episode_summaries>"

    return summaries_xml, total_cost, summary_cache


# ---------------------------------------------------------------------------
# 5. Improve world knowledge only
# ---------------------------------------------------------------------------

def improve_world_knowledge(
    config: DictConfig,
    current_knowledge: str,
    summaries: str,
) -> tuple[str, float]:
    """Use LLM to update world knowledge from trajectory summaries.

    Returns:
        (updated_knowledge, cost)
    """
    model_name = _get_model_name(config)

    prompt = f"""We are building an understanding of how an environment works by analyzing interactions.

Our current world knowledge:
=== CURRENT WORLD KNOWLEDGE ===
{current_knowledge if current_knowledge else "(empty - no knowledge yet)"}
=== END OF CURRENT WORLD KNOWLEDGE ===

We have analyzed a batch of recent interactions:
{summaries}

Your task is to update our world knowledge based on the new evidence from these interaction summaries.

For world knowledge:
- Include mechanics, environmental properties, and cause-and-effect relationships.
- Each point should be concise (a few sentences).
- Correct any wrong or misleading knowledge based on new evidence.
- Add newly discovered mechanics or properties.
- Remove or modify knowledge that is contradicted by evidence.
- Keep knowledge grounded in observed evidence.

Format your response in XML style as:
<think>
Analyze what we have learned and how to update our knowledge.
</think>
<world_knowledge>
- [fact about mechanics, environmental properties, cause-and-effect relationships, etc.]
- ...
</world_knowledge>
"""

    input_data = build_llm_input(prompt)
    logging.info(f"Improve world knowledge prompt:\n{prompt}")

    response = litellm.responses(
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    response_text = extract_llm_response_text(response)
    logging.info(f"Improve world knowledge LLM response:\n{response_text}")

    cost = _get_response_cost(response, config.client.model_id)
    response_dict = extract_xml_kv(response_text, ["world_knowledge"])

    if not validate_response_fields(response_dict, response_text, ["world_knowledge"]):
        return current_knowledge, cost

    return response_dict["world_knowledge"].strip(), cost


# ---------------------------------------------------------------------------
# 6. Improve knowledge and policy together
# ---------------------------------------------------------------------------

def improve_knowledge_and_policy(
    config: DictConfig,
    current_knowledge: str,
    current_policy: str,
    summaries: str,
) -> tuple[str, str, float]:
    """Use LLM to update both world knowledge and policy from trajectory summaries.

    Returns:
        (updated_knowledge, updated_policy, cost)
    """
    model_name = _get_model_name(config)

    prompt = f"""We are building an understanding of how an environment works and developing a strategy by analyzing interactions.

Our current world knowledge:
=== CURRENT WORLD KNOWLEDGE ===
{current_knowledge if current_knowledge else "(empty - no knowledge yet)"}
=== END OF CURRENT WORLD KNOWLEDGE ===

Our current policy:
=== CURRENT POLICY ===
{current_policy if current_policy else "(empty - no policy yet)"}
=== END OF CURRENT POLICY ===

We have analyzed a batch of recent interactions:
{summaries}

Your task is to update both our world knowledge and policy based on the new evidence.

For world knowledge:
- Include mechanics, environmental properties, and cause-and-effect relationships.
- Each point should be concise (a few sentences).
- Correct any wrong or misleading knowledge based on new evidence.
- Add newly discovered mechanics or properties.
- Keep knowledge grounded in observed evidence.

For policy:
- Define what to do in specific situations, priorities, and strategies for completing the objective.
- Each point should be concise and actionable.
- Update strategies based on what worked and what failed in the interactions.
- Remove or modify strategies that are contradicted by evidence.

Format your response in XML style as:
<think>
Analyze what we have learned and how to update knowledge and policy.
</think>
<world_knowledge>
- [fact about mechanics, environmental properties, cause-and-effect relationships, etc.]
- ...
</world_knowledge>
<policy>
- [what to do in specific situations, priorities, strategies, etc.]
- ...
</policy>
"""

    input_data = build_llm_input(prompt)
    logging.info(f"Improve knowledge and policy prompt:\n{prompt}")

    response = litellm.responses(
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    response_text = extract_llm_response_text(response)
    logging.info(f"Improve knowledge and policy LLM response:\n{response_text}")

    cost = _get_response_cost(response, config.client.model_id)
    response_dict = extract_xml_kv(response_text, ["world_knowledge", "policy"])

    updated_knowledge = current_knowledge
    updated_policy = current_policy

    if "world_knowledge" in response_dict:
        updated_knowledge = response_dict["world_knowledge"].strip()
    if "policy" in response_dict:
        updated_policy = response_dict["policy"].strip()

    return updated_knowledge, updated_policy, cost


# ---------------------------------------------------------------------------
# 7. Improve policy from provided world knowledge (reflection mode)
# ---------------------------------------------------------------------------

def improve_policy_from_knowledge(
    config: DictConfig,
    world_knowledge: str,
    objective: str,
    current_policy: str,
    step: int,
    summaries: str = "",
) -> tuple[str, float]:
    """Use LLM to improve policy through reflection on world knowledge.

    Includes explicit self-critique step for gaps, wrong assumptions, edge cases.
    Optionally incorporates trajectory summaries as additional evidence.

    Returns:
        (updated_policy, cost)
    """
    model_name = _get_model_name(config)

    if summaries:
        summaries_section = f"""
We have also analyzed a batch of recent interactions with the environment:
{summaries}
"""
        summaries_critique = "- Do the interaction summaries reveal situations the current policy handles poorly?\n"
        summaries_improve = "- Incorporate lessons from the interaction summaries where applicable.\n"
    else:
        summaries_section = ""
        summaries_critique = ""
        summaries_improve = ""

    prompt = f"""We have the following knowledge about an environment:
=== WORLD KNOWLEDGE ===
{world_knowledge}
=== END OF WORLD KNOWLEDGE ===

The objective is: {objective}

Our current policy (iteration {step}):
=== CURRENT POLICY ===
{current_policy if current_policy else "(empty - no policy yet)"}
=== END OF CURRENT POLICY ===
{summaries_section}
Your task is to improve the policy through careful analysis and self-critique.

Step 1: Self-critique the current policy
- What gaps exist in the current policy?
- What assumptions might be wrong?
- What edge cases are not handled?
- What situations could arise where the current policy would fail?
- Are there contradictions between the policy and the world knowledge?
{summaries_critique}
Step 2: Generate an improved policy
- Address the gaps and issues identified in the self-critique.
- The policy should define what to do in specific situations, priorities, and strategies.
- Each point should be concise and actionable.
- The policy should be consistent with the world knowledge.
- Focus on strategies that directly help achieve the objective.
{summaries_improve}
Format your response in XML style as:
<think>
Self-critique: Identify gaps, wrong assumptions, edge cases, and contradictions in the current policy.
Then reason about how to improve the policy.
</think>
<policy>
- [what to do in specific situations, priorities, strategies, etc.]
- ...
</policy>
"""

    input_data = build_llm_input(prompt)
    logging.info(f"Improve policy from knowledge prompt:\n{prompt}")

    response = litellm.responses(
        model=model_name,
        input=input_data,
        num_retries=5,
    )

    response_text = extract_llm_response_text(response)
    logging.info(f"Improve policy from knowledge LLM response:\n{response_text}")

    cost = _get_response_cost(response, config.client.model_id)
    response_dict = extract_xml_kv(response_text, ["policy"])

    if not validate_response_fields(response_dict, response_text, ["policy"]):
        return current_policy, cost

    return response_dict["policy"].strip(), cost


# ---------------------------------------------------------------------------
# Helper: format beliefs.txt content
# ---------------------------------------------------------------------------

def _load_initial_beliefs(path: str) -> tuple[str, str]:
    """Load world_knowledge and policy from a beliefs file.

    Accepts files with <world_knowledge>/<policy> XML tags or plain text
    (treated as world_knowledge).

    Returns:
        (world_knowledge, policy)
    """
    content = Path(path).read_text()
    parsed = extract_xml_kv(content, ["world_knowledge", "policy"])
    wk = parsed.get("world_knowledge", "").strip()
    pol = parsed.get("policy", "").strip()
    # If no XML tags found, treat the whole file as world knowledge
    if not wk and not pol:
        wk = content.strip()
    return wk, pol


def _format_beliefs(world_knowledge: str, policy: str = "") -> str:
    """Format world knowledge and optional policy as beliefs.txt content."""
    parts = [f"<world_knowledge>\n{world_knowledge}\n</world_knowledge>"]
    if policy:
        parts.append(f"<policy>\n{policy}\n</policy>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helper: update offline summary.json
# ---------------------------------------------------------------------------

def _update_offline_summary(output_dir: str, step: int, step_cost: float, cumulative_cost: float):
    """Update summary.json with offline step data."""
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {"steps": []}

    summary["steps"].append({
        "step": step,
        "step_cost": step_cost,
        "cumulative_cost": cumulative_cost,
    })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)


# ---------------------------------------------------------------------------
# 8. Main loop: knowledge-only mode
# ---------------------------------------------------------------------------

def offline_knowledge(config: DictConfig, output_dir: str):
    """Offline learning loop: extract world knowledge from trajectories."""
    offline_cfg = config.offline
    trajectories_dir = offline_cfg.trajectories_dir
    num_steps = offline_cfg.num_steps
    batch_size = offline_cfg.batch_size
    num_refinements = offline_cfg.get("num_refinements", 1)

    if not trajectories_dir:
        raise ValueError("offline.trajectories_dir must be set for 'knowledge' mode")

    # Check for resume
    last_step, world_knowledge, _ = find_last_completed_offline_step(output_dir)
    start_step = last_step + 1

    if last_step > 0:
        logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
    else:
        # Load initial beliefs if provided
        initial_path = offline_cfg.get("initial_beliefs_path", None)
        if initial_path:
            world_knowledge, _ = _load_initial_beliefs(initial_path)
            logger.info(f"Loaded initial world knowledge from: {initial_path}")
        logger.info("Starting fresh offline knowledge extraction")

    cumulative_cost = 0.0
    summary_cache = _load_summary_cache(output_dir)
    logger.info(f"Starting offline knowledge loop: {num_steps} steps, batch_size={batch_size}, num_refinements={num_refinements}")
    logger.info(f"Summary cache: {len(summary_cache)} entries loaded")

    for step in range(start_step, num_steps + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"OFFLINE KNOWLEDGE STEP {step}/{num_steps}")
        logger.info(f"{'='*60}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Save input state
        (step_output_dir / "input_beliefs.txt").write_text(
            _format_beliefs(world_knowledge)
        )

        # Load trajectory batch (cycling)
        batch_index = step - 1
        batch = load_trajectory_batch(trajectories_dir, batch_index, batch_size)
        logger.info(f"Loaded {len(batch)} trajectories for step {step}")

        # Summarize batch
        with improve_logging(step_output_dir):
            logging.info(f"Step {step} offline knowledge improve logs")

            summaries, summary_cost, summary_cache = summarize_trajectory_batch(
                config, batch, world_knowledge, summary_cache
            )
            (step_output_dir / "summaries.txt").write_text(summaries)
            _save_summary_cache(output_dir, summary_cache)
            logger.info(f"Summarization cost: ${summary_cost:.6f}")

            # Iterative refinement
            improve_cost = 0.0
            for r in range(num_refinements):
                logger.info(f"Refinement pass {r+1}/{num_refinements}")
                world_knowledge, cost = improve_world_knowledge(
                    config, world_knowledge, summaries
                )
                improve_cost += cost

        # Save outputs
        beliefs_text = _format_beliefs(world_knowledge)
        (step_output_dir / "beliefs.txt").write_text(beliefs_text)

        step_cost = summary_cost + improve_cost
        cumulative_cost += step_cost
        logger.info(f"Step {step} cost: ${step_cost:.6f}, cumulative: ${cumulative_cost:.6f}")
        logger.info(f"Updated world knowledge:\n{world_knowledge}")

        _update_offline_summary(output_dir, step, step_cost, cumulative_cost)


# ---------------------------------------------------------------------------
# 9. Main loop: knowledge + policy mode
# ---------------------------------------------------------------------------

def offline_both(config: DictConfig, output_dir: str):
    """Offline learning loop: extract world knowledge and policy from trajectories."""
    offline_cfg = config.offline
    trajectories_dir = offline_cfg.trajectories_dir
    num_steps = offline_cfg.num_steps
    batch_size = offline_cfg.batch_size
    num_refinements = offline_cfg.get("num_refinements", 1)

    if not trajectories_dir:
        raise ValueError("offline.trajectories_dir must be set for 'both' mode")

    # Check for resume
    last_step, world_knowledge, policy = find_last_completed_offline_step(output_dir)
    start_step = last_step + 1

    if last_step > 0:
        logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
    else:
        # Load initial beliefs if provided
        initial_path = offline_cfg.get("initial_beliefs_path", None)
        if initial_path:
            world_knowledge, policy = _load_initial_beliefs(initial_path)
            logger.info(f"Loaded initial beliefs from: {initial_path}")
        logger.info("Starting fresh offline knowledge+policy extraction")

    cumulative_cost = 0.0
    summary_cache = _load_summary_cache(output_dir)
    logger.info(f"Starting offline both loop: {num_steps} steps, batch_size={batch_size}, num_refinements={num_refinements}")
    logger.info(f"Summary cache: {len(summary_cache)} entries loaded")

    for step in range(start_step, num_steps + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"OFFLINE BOTH STEP {step}/{num_steps}")
        logger.info(f"{'='*60}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Save input state
        (step_output_dir / "input_beliefs.txt").write_text(
            _format_beliefs(world_knowledge, policy)
        )

        # Load trajectory batch (cycling)
        batch_index = step - 1
        batch = load_trajectory_batch(trajectories_dir, batch_index, batch_size)
        logger.info(f"Loaded {len(batch)} trajectories for step {step}")

        # Summarize batch
        with improve_logging(step_output_dir):
            logging.info(f"Step {step} offline both improve logs")

            summaries, summary_cost, summary_cache = summarize_trajectory_batch(
                config, batch, world_knowledge, summary_cache
            )
            (step_output_dir / "summaries.txt").write_text(summaries)
            _save_summary_cache(output_dir, summary_cache)
            logger.info(f"Summarization cost: ${summary_cost:.6f}")

            # Iterative refinement
            improve_cost = 0.0
            for r in range(num_refinements):
                logger.info(f"Refinement pass {r+1}/{num_refinements}")
                world_knowledge, policy, cost = improve_knowledge_and_policy(
                    config, world_knowledge, policy, summaries
                )
                improve_cost += cost

        # Save outputs
        beliefs_text = _format_beliefs(world_knowledge, policy)
        (step_output_dir / "beliefs.txt").write_text(beliefs_text)

        step_cost = summary_cost + improve_cost
        cumulative_cost += step_cost
        logger.info(f"Step {step} cost: ${step_cost:.6f}, cumulative: ${cumulative_cost:.6f}")
        logger.info(f"Updated world knowledge:\n{world_knowledge}")
        logger.info(f"Updated policy:\n{policy}")

        _update_offline_summary(output_dir, step, step_cost, cumulative_cost)


# ---------------------------------------------------------------------------
# 10. Main loop: policy-from-knowledge mode
# ---------------------------------------------------------------------------

def offline_policy(config: DictConfig, output_dir: str):
    """Offline learning loop: refine policy from provided world knowledge via reflection.

    When offline.trajectories_dir is set, trajectory batches are loaded and summarized
    each step, and the summaries are fed into the policy improvement prompt alongside
    the world knowledge. Otherwise, pure reflection (no trajectory data).
    """
    offline_cfg = config.offline
    world_knowledge_path = offline_cfg.world_knowledge_path
    objective = offline_cfg.objective
    num_steps = offline_cfg.num_steps
    num_refinements = offline_cfg.get("num_refinements", 1)
    trajectories_dir = offline_cfg.get("trajectories_dir", None)
    batch_size = offline_cfg.get("batch_size", 5)
    use_trajectories = bool(trajectories_dir)

    if not world_knowledge_path:
        raise ValueError("offline.world_knowledge_path must be set for 'policy' mode")
    if not objective:
        raise ValueError("offline.objective must be set for 'policy' mode")

    # Load world knowledge from file
    wk_path = Path(world_knowledge_path)
    if not wk_path.exists():
        raise FileNotFoundError(f"World knowledge file not found: {world_knowledge_path}")

    wk_content = wk_path.read_text()
    # Try to extract from XML tags if present, otherwise use raw content
    parsed = extract_xml_kv(wk_content, ["world_knowledge"])
    world_knowledge = parsed.get("world_knowledge", wk_content).strip()

    # Check for resume
    last_step, _, policy = find_last_completed_offline_step(output_dir)
    start_step = last_step + 1

    if last_step > 0:
        logger.info(f"Resuming from step {start_step} (found {last_step} completed steps)")
    else:
        # Load initial policy from beliefs file if provided
        initial_path = offline_cfg.get("initial_beliefs_path", None)
        if initial_path:
            _, policy = _load_initial_beliefs(initial_path)
            logger.info(f"Loaded initial policy from: {initial_path}")
        else:
            policy = ""
        logger.info("Starting fresh offline policy refinement")

    cumulative_cost = 0.0
    logger.info(f"Starting offline policy loop: {num_steps} steps, num_refinements={num_refinements}")
    logger.info(f"Objective: {objective}")
    if use_trajectories:
        logger.info(f"Trajectory-grounded mode: loading batches from {trajectories_dir} (batch_size={batch_size})")
    else:
        logger.info("Pure reflection mode (no trajectory data)")

    summary_cache = _load_summary_cache(output_dir) if use_trajectories else {}
    if use_trajectories:
        logger.info(f"Summary cache: {len(summary_cache)} entries loaded")

    for step in range(start_step, num_steps + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"OFFLINE POLICY STEP {step}/{num_steps}")
        logger.info(f"{'='*60}")

        step_output_dir = Path(output_dir) / f"step_{step}"
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Save input state
        (step_output_dir / "input_beliefs.txt").write_text(
            _format_beliefs(world_knowledge, policy)
        )

        with improve_logging(step_output_dir):
            logging.info(f"Step {step} offline policy improve logs")

            # Optionally summarize trajectory batch
            summaries = ""
            summary_cost = 0.0
            if use_trajectories:
                batch_index = step - 1
                batch = load_trajectory_batch(trajectories_dir, batch_index, batch_size)
                logger.info(f"Loaded {len(batch)} trajectories for step {step}")

                summaries, summary_cost, summary_cache = summarize_trajectory_batch(
                    config, batch, world_knowledge, summary_cache
                )
                (step_output_dir / "summaries.txt").write_text(summaries)
                _save_summary_cache(output_dir, summary_cache)
                logger.info(f"Summarization cost: ${summary_cost:.6f}")

            # Iterative refinement
            improve_cost = 0.0
            for r in range(num_refinements):
                logger.info(f"Refinement pass {r+1}/{num_refinements}")
                policy, cost = improve_policy_from_knowledge(
                    config, world_knowledge, objective, policy, step,
                    summaries=summaries,
                )
                improve_cost += cost

        # Save outputs
        beliefs_text = _format_beliefs(world_knowledge, policy)
        (step_output_dir / "beliefs.txt").write_text(beliefs_text)

        step_cost = summary_cost + improve_cost
        cumulative_cost += step_cost
        logger.info(f"Step {step} cost: ${step_cost:.6f}, cumulative: ${cumulative_cost:.6f}")
        logger.info(f"Updated policy:\n{policy}")

        _update_offline_summary(output_dir, step, step_cost, cumulative_cost)


# ---------------------------------------------------------------------------
# 11. Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    offline_cfg = config.offline
    mode = offline_cfg.mode

    model_slug = config.client.model_id.replace("/", "_")
    run_name_suffix = f"offline_{mode}_{model_slug}"

    _, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=offline_cfg.resume_from,
        output_dir_base=offline_cfg.output_dir,
        logger_name="offline",
    )

    logger.info(f"Offline mode: {mode}")

    match mode:
        case "knowledge":
            offline_knowledge(config, output_dir)
        case "both":
            offline_both(config, output_dir)
        case "policy":
            offline_policy(config, output_dir)
        case _:
            raise ValueError(f"Unsupported offline mode: {mode}. Must be 'knowledge', 'both', or 'policy'.")

    logger.info("Offline learning complete.")


if __name__ == "__main__":
    main()
