"""Simple tool-less agent baseline: per-step env interaction with a plain LLM.

Mirrors the outer shape of openhands_stepwise.py — single orchestrator owns
a rolling message history across the whole run; episodes are only visible to
the agent via a 'respawned' prepend on the user message. Each env-step posts
one user message (raw obs + image + valid actions) into the history and the
agent responds with the same brief plan plus ``<|ACTION|>...<|END|>``-style
action marker used by the robust CoT agent in stepwise_eb_learn.py. The
orchestrator parses the final action marker and forwards it to env.step(...).

No workspace, no tools — just LLM <-> env.
"""

import base64
import csv
import io
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import litellm
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from balrog.environments import make_env
from balrog.utils import get_unique_seed

from explore import evolve_logger, get_default_knowledge
from goal_prompts import append_agent_goal, resolve_agent_goal
from run_utils import setup_run, _update_summary_json, is_minihack_success_episode


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class SimpleStepwiseConfig:
    n_environment_steps: int
    history_window: int                  # Max user/assistant turn pairs retained in the running message list
    max_image_history: int               # Max recent user turns whose images stay attached
    model: str                           # e.g. "anthropic/claude-sonnet-4-5-20250929"
    api_key_env: str                     # env var name holding the key, e.g. "ANTHROPIC_API_KEY"
    base_url: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    hide_obs_when_image: bool = False
    mock_mode: bool = False
    dynamics_history_path: str | None = None
    history_checkpoint_interval: int = 0


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _pil_to_data_url(img) -> str | None:
    if img is None:
        return None
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")
    except Exception as e:  # noqa: BLE001
        evolve_logger.warning(f"Failed to encode PIL image: {e}")
        return None
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _save_step_dir_image(img, step_dir: Path, filename: str) -> None:
    if img is None:
        return
    try:
        img.save(step_dir / filename)
    except Exception as e:  # noqa: BLE001
        evolve_logger.warning(f"Failed to save {filename} in {step_dir}: {e}")


# ---------------------------------------------------------------------------
# Env dispatch (matches openhands_stepwise._make_env_for_config)
# ---------------------------------------------------------------------------


def _make_env_for_config(config: DictConfig):
    env_name = config.envs.names.split("-")[0]
    if env_name == "arc_agi":
        from arc_agi_env import make_arc_env
        task = config.tasks.arc_agi_tasks[0]
        return env_name, task, make_arc_env(task, config)
    if env_name == "autumn":
        from autumn_env import make_autumn_env
        task = config.tasks.autumn_tasks[0]
        return env_name, task, make_autumn_env(task, config)
    tasks = config.tasks[f"{env_name}_tasks"]
    task = tasks[0]
    return env_name, task, make_env(env_name, task, config)


def _valid_actions(env) -> list[str]:
    space = getattr(env, "language_action_space", None)
    if space is None and hasattr(env, "env"):
        space = getattr(env.env, "language_action_space", None)
    values = getattr(space, "_values", None)
    if values is not None:
        space = values
    if not space:
        return []
    try:
        return list(space)
    except TypeError:
        return []


def _fallback_action(env, valid_actions: list[str]) -> str:
    default = getattr(env, "default_action", None)
    if default is not None and (not valid_actions or default in valid_actions):
        return default
    if valid_actions:
        return valid_actions[0]
    return default or "wait"


def _compose_obs_text(short_term: str, long_term: str) -> str:
    """Format a pre-action observation block matching stepwise_eb_learn's
    trajectory.csv "Observation" column so the viz trajectory tab renders
    runs from this baseline identically to EB runs."""
    sep = "=" * 10
    return (
        f"{short_term}\n\n"
        f"{sep} Start of Direct Observation {sep}\n"
        f"{long_term}\n\n"
        f"{sep} End of Direct Observation {sep}"
    )


def _user_content_text(user_content: list[dict]) -> str:
    """Extract the text portion of a multi-modal user content list."""
    for part in user_content:
        if isinstance(part, dict) and part.get("type") == "text":
            return part.get("text", "") or ""
    return ""


def _flatten_history_for_log(
    history: list[dict], final_action: str | None, step_dir: Path | None = None
) -> list[dict]:
    """Serialize the rolling message history into a JSON-safe shape for the viz
    agent-messages panel. Text content passes through; multi-modal parts are
    flattened (text preserved, images replaced with an ``[image attached]``
    marker). ``final_action`` is attached to the last assistant entry."""
    out: list[dict] = []
    image_dir = step_dir / "agent_message_images" if step_dir is not None else None
    for i, msg in enumerate(history, 1):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        record: dict = {"role": role}
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            attachment_paths: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        t = part.get("text", "") or ""
                        if t:
                            parts.append(t)
                    elif part.get("type") == "image_url":
                        parts.append("[image attached]")
                        data_url = (part.get("image_url") or {}).get("url", "")
                        if image_dir is not None and data_url.startswith("data:image/"):
                            try:
                                _, encoded = data_url.split(",", 1)
                                rel_path = (
                                    f"agent_message_images/"
                                    f"message_{i:03d}_attachment_{len(attachment_paths) + 1:02d}.png"
                                )
                                image_dir.mkdir(parents=True, exist_ok=True)
                                (step_dir / rel_path).write_bytes(
                                    base64.b64decode(encoded)
                                )
                                attachment_paths.append(rel_path)
                            except Exception as e:  # noqa: BLE001
                                record.setdefault("attachment_errors", []).append(str(e))
            text = "\n".join(parts)
            if attachment_paths:
                record["attachment_paths"] = attachment_paths
                if len(attachment_paths) == 1:
                    record["attachment_path"] = attachment_paths[0]
        else:
            text = str(content)
        record["content"] = text
        out.append(record)
    if out and final_action is not None and out[-1].get("role") == "assistant":
        out[-1]["action"] = final_action
    return out


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

_ACTION_INSTRUCTIONS = """
First create (if not present) or update your plan from the previous steps and presented the updated plan in -
<plan>
<reasoning>
thinking about what to do next and how to do it.
</reasoning>
<goal>
high level subtask you are trying to achieve currently. update this as you complete subtasks, achieve goals or are given new goals.
</goal>
<history>
updated history with what happened. keep this numbered and summarise the past as you go along to keep the list short.
</history>
<steps>
updated next steps to achieve subtask and eventually the main task.
</steps>
</plan>


Finally you must choose exactly one of the listed actions and output it strictly in the following format -
<|ACTION|>YOUR_CHOSEN_ACTION<|END|>

Replace YOUR_CHOSEN_ACTION with the chosen action.
Keep the plan very brief.
""".strip()


def _build_user_content(
    obs: dict,
    is_first_step: bool,
    instruction_prompt: str,
    default_knowledge: str,
    respawn_notice: bool,
    previous_terminal_obs: dict | None,
    episode_idx: int,
    step_idx: int,
    global_step: int,
    invalid_action_feedback: str | None = None,
) -> list[dict]:
    long_ctx = obs.get("text", {}).get("long_term_context", "") or ""
    short_ctx = obs.get("text", {}).get("short_term_context", "") or ""
    image = obs.get("image")

    lines: list[str] = []

    if is_first_step:
        if instruction_prompt:
            lines.append("=== ENVIRONMENT INSTRUCTIONS ===")
            lines.append(instruction_prompt.strip())
            lines.append("=== END ENVIRONMENT INSTRUCTIONS ===")
        if default_knowledge and default_knowledge.strip() != instruction_prompt.strip():
            lines.append("\n=== DEFAULT KNOWLEDGE ===")
            lines.append(default_knowledge.strip())
            lines.append("=== END DEFAULT KNOWLEDGE ===")

    lines.append(
        f"\n=== STEP (episode={episode_idx}, step={step_idx}, global_step={global_step}) ==="
    )
    if respawn_notice:
        lines.append("The previous episode was terminated and you have respawned.")

    previous_terminal_image = None
    if respawn_notice and previous_terminal_obs is not None:
        terminal_long = (
            previous_terminal_obs.get("text", {}).get("long_term_context", "") or ""
        )
        terminal_short = (
            previous_terminal_obs.get("text", {}).get("short_term_context", "") or ""
        )
        previous_terminal_image = previous_terminal_obs.get("image")
        lines.append(
            "\nPrevious Terminal Observation "
            "(after the action that ended the previous episode):"
        )
        if terminal_short:
            lines.append(terminal_short)
        if terminal_long:
            lines.append(terminal_long)
        if previous_terminal_image is not None:
            lines.append(
                "\nThe previous terminal observation image is attached before "
                "the current observation image."
            )

    if invalid_action_feedback:
        lines.append("\n[invalid_previous_action]")
        lines.append(invalid_action_feedback)

    lines.append("\nCurrent Observation:")
    if short_ctx:
        lines.append(short_ctx)
    lines.append(long_ctx)

    lines.append("\n" + _ACTION_INSTRUCTIONS)

    text = "\n".join(lines)
    content: list[dict] = [{"type": "text", "text": text}]

    previous_terminal_data_url = _pil_to_data_url(previous_terminal_image)
    if previous_terminal_data_url is not None:
        content.append(
            {"type": "image_url", "image_url": {"url": previous_terminal_data_url}}
        )

    data_url = _pil_to_data_url(image)
    if data_url is not None:
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    return content


def _build_persistent_env_message(
    instruction_prompt: str,
    default_knowledge: str,
) -> dict:
    lines: list[str] = []
    if instruction_prompt:
        lines.append("=== ENVIRONMENT INSTRUCTIONS ===")
        lines.append(instruction_prompt.strip())
        lines.append("=== END ENVIRONMENT INSTRUCTIONS ===")
    if default_knowledge and default_knowledge.strip() != instruction_prompt.strip():
        lines.append("\n=== DEFAULT KNOWLEDGE ===")
        lines.append(default_knowledge.strip())
        lines.append("=== END DEFAULT KNOWLEDGE ===")
    return {"role": "system", "content": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_ACTION_RE = re.compile(r"<\|ACTION\|>(.*?)<\|END\|>", re.DOTALL)


def _parse_action(text: str) -> str | None:
    if not text:
        return None
    m = _ACTION_RE.search(text)
    if m is not None:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Mock-mode: stub litellm.completion. Keep prompt construction / logging /
# summary-emission paths live; skip the network call. Mirrors
# openhands_stepwise.py's mock_mode semantics.
# ---------------------------------------------------------------------------


_MOCK_ACTION_PROVIDER = None


def set_mock_action_provider(fn) -> None:
    """Install a zero-arg callable returning a valid action string for the
    current env. Called by the patched litellm.completion when it needs to
    synthesize the agent's LLM response."""
    global _MOCK_ACTION_PROVIDER
    _MOCK_ACTION_PROVIDER = fn


def _pick_mock_action(env) -> str:
    candidates = _valid_actions(env)
    if not candidates:
        return getattr(env, "default_action", "wait")
    return random.choice(candidates)


def _install_litellm_mock() -> None:
    import uuid

    from litellm import ModelResponse, Choices
    from litellm.types.utils import Message as LLMMsg

    def _mock_completion(*, messages, model, **kwargs):  # noqa: ARG001
        action = (
            _MOCK_ACTION_PROVIDER() if _MOCK_ACTION_PROVIDER is not None else "wait"
        )
        content = (
            "<plan>\n"
            "<reasoning>[mock] selecting a random valid action.</reasoning>\n"
            "<goal>[mock] explore the current environment.</goal>\n"
            "<history>[mock] smoke-test rollout.</history>\n"
            "<steps>[mock] continue with one valid action.</steps>\n"
            "</plan>\n\n"
            f"<|ACTION|>{action}<|END|>"
        )
        msg = LLMMsg(role="assistant", content=content)
        return ModelResponse(
            id=f"mock-{uuid.uuid4().hex[:8]}",
            choices=[Choices(index=0, message=msg, finish_reason="stop")],
            model=model,
            object="chat.completion",
            created=0,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    litellm.completion = _mock_completion

    if not getattr(litellm, "_simple_stepwise_mock_cost_installed", False):
        _orig_cost = getattr(litellm, "completion_cost", None)

        def _zero_cost(*args, **kwargs):  # noqa: ARG001
            return 0.0

        litellm.completion_cost = _zero_cost
        litellm._simple_stepwise_mock_cost_installed = True  # type: ignore[attr-defined]
        litellm._simple_stepwise_mock_cost_orig = _orig_cost  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# LLM call + history management
# ---------------------------------------------------------------------------


def _extract_usage(resp) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    if usage is None:
        return tokens

    def _g(name: str) -> int:
        if isinstance(usage, dict):
            return int(usage.get(name, 0) or 0)
        return int(getattr(usage, name, 0) or 0)

    tokens["prompt_tokens"] = _g("prompt_tokens")
    tokens["completion_tokens"] = _g("completion_tokens")
    # LiteLLM's cache fields are provider-specific; try a few names.
    tokens["cache_read_tokens"] = _g("cache_read_input_tokens") or _g("cache_read_tokens")
    tokens["cache_write_tokens"] = (
        _g("cache_creation_input_tokens") or _g("cache_write_tokens")
    )
    tokens["reasoning_tokens"] = _g("reasoning_tokens")
    return tokens


def _call_llm(
    sc: SimpleStepwiseConfig,
    api_key: str,
    messages: list[dict],
) -> tuple[str, dict[str, int], float]:
    kwargs: dict[str, Any] = {
        "model": sc.model,
        "messages": messages,
        "api_key": api_key,
    }
    if sc.base_url:
        kwargs["base_url"] = sc.base_url
    if sc.temperature is not None:
        kwargs["temperature"] = sc.temperature
    if sc.max_output_tokens is not None:
        kwargs["max_tokens"] = sc.max_output_tokens

    resp = litellm.completion(**kwargs)
    choice = resp.choices[0]
    msg = choice.message
    content = getattr(msg, "content", "") or ""
    tokens = _extract_usage(resp)
    try:
        cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception as e:  # noqa: BLE001
        evolve_logger.debug(f"completion_cost failed: {e}")
        cost = 0.0
    return content, tokens, cost


def _trim_history(history: list[dict], max_user_turns: int) -> None:
    """Keep system messages plus the most recent ``max_user_turns`` user turns
    (and every message that comes after the cut). System messages are always
    retained. Runs in-place."""
    if max_user_turns <= 0:
        return
    user_positions = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_positions) <= max_user_turns:
        return
    cut_at = user_positions[-max_user_turns]
    system_msgs = [m for m in history[:cut_at] if m.get("role") == "system"]
    tail = history[cut_at:]
    history[:] = system_msgs + tail


def _trim_image_history(history: list[dict], max_image_turns: int) -> None:
    """Keep image attachments only on the most recent user turns."""
    user_positions_with_images: list[int] = []
    for i, msg in enumerate(history):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in msg["content"]
        ):
            user_positions_with_images.append(i)

    keep = (
        set(user_positions_with_images[-max_image_turns:])
        if max_image_turns > 0
        else set()
    )
    for i in user_positions_with_images:
        if i in keep:
            continue
        msg = history[i]
        msg["content"] = [
            part
            for part in msg["content"]
            if not (isinstance(part, dict) and part.get("type") == "image_url")
        ]


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


def _run_episode(
    config: DictConfig,
    sc: SimpleStepwiseConfig,
    api_key: str,
    history: list[dict],
    episode_idx: int,
    episode_dir: Path,
    default_knowledge: str,
    global_step_start: int,
    max_episode_steps: int | None,
    is_first_step_in_run: bool,
    cumulative_cost_start: float,
    previous_terminal_observation: dict | None = None,
) -> tuple[dict, int, float, dict | None]:
    env_name, task, env = _make_env_for_config(config)

    if sc.mock_mode:
        set_mock_action_provider(lambda: _pick_mock_action(env))

    seed = config.envs.env_kwargs.seed
    if seed is None:
        seed = get_unique_seed(process_num=0, episode_idx=episode_idx)
    random.seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)

    max_steps = (
        env.max_steps
        if config.eval.get("max_steps_per_episode") is None
        else config.eval.max_steps_per_episode
    )
    if max_episode_steps is not None:
        max_steps = min(max_steps, max_episode_steps)

    try:
        env_instruction_prompt = env.get_instruction_prompt()
    except Exception:  # noqa: BLE001
        env_instruction_prompt = default_knowledge
    agent_goal = resolve_agent_goal(config)
    instruction_prompt = append_agent_goal(env_instruction_prompt, agent_goal)
    if default_knowledge.strip() == (env_instruction_prompt or "").strip():
        default_knowledge = ""

    if is_first_step_in_run and episode_idx == 0:
        history.insert(
            0,
            _build_persistent_env_message(
                instruction_prompt=instruction_prompt or "",
                default_knowledge=default_knowledge or "",
            ),
        )

    episode_log: dict = {
        "task": task,
        "env_name": env_name,
        "episode_idx": episode_idx,
        "seed": seed,
        "agent_goal": agent_goal,
        "action_frequency": defaultdict(int),
        "num_action_parse_failures": 0,
        "num_default_action_fallbacks": 0,
        "episode_return": 0.0,
    }

    pbar = tqdm(total=max_steps, desc=f"SimpleAgent ep {episode_idx}", leave=False, dynamic_ncols=True)

    episode_return = 0.0
    step = 0
    done = False
    respawn_notice = episode_idx > 0
    cumulative_cost = cumulative_cost_start
    feedback_on_invalid = bool(config.eval.get("feedback_on_invalid_action", False))
    pending_invalid_feedback: str | None = None
    terminal_observation: dict | None = None

    csv_path = episode_dir / "trajectory.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file, escapechar="˘", quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow([
        "Step", "Action", "Reasoning", "Observation",
        "Auxiliary_Observation", "Reward", "Done",
    ])

    try:
        for step in range(max_steps):
            global_step = global_step_start + step
            step_dir = episode_dir / f"step_{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            raw_long = obs.get("text", {}).get("long_term_context", "") or ""
            raw_short = obs.get("text", {}).get("short_term_context", "") or ""
            prompt_obs = obs
            if sc.hide_obs_when_image and obs.get("image") is not None:
                prompt_obs = {
                    **obs,
                    "text": {
                        **(obs.get("text") or {}),
                        "long_term_context": "",
                    },
                }
            _save_step_dir_image(obs.get("image"), step_dir, "obs_before.png")

            valid_actions = _valid_actions(env)

            user_content = _build_user_content(
                obs=prompt_obs,
                is_first_step=False,
                instruction_prompt="",
                default_knowledge="",
                respawn_notice=respawn_notice and step == 0,
                previous_terminal_obs=(
                    previous_terminal_observation if step == 0 else None
                ),
                episode_idx=episode_idx,
                step_idx=step,
                global_step=global_step,
                invalid_action_feedback=pending_invalid_feedback,
            )
            pending_invalid_feedback = None
            history.append({"role": "user", "content": user_content})
            _trim_image_history(history, sc.max_image_history)

            _t0 = time.time()
            parse_failed = False
            try:
                assistant_text, step_tokens, step_cost = _call_llm(sc, api_key, history)
            except Exception as e:  # noqa: BLE001
                evolve_logger.error(f"[g{global_step}] litellm.completion raised: {e}")
                assistant_text = ""
                step_tokens = {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "cache_read_tokens": 0, "cache_write_tokens": 0,
                    "reasoning_tokens": 0,
                }
                step_cost = 0.0
            dt = time.time() - _t0

            history.append({"role": "assistant", "content": assistant_text})

            parsed = _parse_action(assistant_text)
            if parsed is None:
                parse_failed = True
                episode_log["num_action_parse_failures"] += 1
                action = _fallback_action(env, valid_actions)
                reasoning = "(no <|ACTION|>...<|END|> marker found in response; fell back to default action)"
                episode_log["num_default_action_fallbacks"] += 1
                evolve_logger.warning(
                    f"[g{global_step}] no <|ACTION|>...<|END|> marker in LLM response "
                    f"(run took {dt:.1f}s); using default action {action!r}."
                )
            else:
                action = parsed
                reasoning = assistant_text

            _trim_history(history, sc.history_window)
            _trim_image_history(history, sc.max_image_history)

            invalid_action_this_step = False
            try:
                obs_next, reward, terminated, truncated, info = env.step(action)
            except ValueError as e:
                logging.warning(f"[g{global_step}] Invalid action: {action} — {e}")
                invalid_action_this_step = True
                if feedback_on_invalid:
                    pending_invalid_feedback = (
                        f"Your previous response selected action {action!r}, which the "
                        f"environment rejected as invalid ({e}). The observation below "
                        "is unchanged from the previous step — retry with one of the "
                        "valid actions listed."
                    )
                terminated = False
                truncated = False
                reward = 0.0
                obs_next = obs

            done = bool(terminated or truncated)
            terminal_observation = (
                obs_next if done and isinstance(obs_next, dict) else None
            )
            episode_return += float(reward)
            cumulative_cost += step_cost
            episode_log["action_frequency"][action] += 1

            if isinstance(obs_next, dict):
                _save_step_dir_image(obs_next.get("image"), step_dir, "obs_after.png")

            step_log = {
                "step": step,
                "global_step": global_step,
                "episode_idx": episode_idx,
                "action": action,
                "reasoning": reasoning,
                "assistant_response": assistant_text,
                "action_parse_failed": parse_failed,
                "reward": float(reward),
                "done": done,
                "episode_return_so_far": episode_return,
                "invalid_action": invalid_action_this_step,
                "step_cost": step_cost,
                "cumulative_cost": cumulative_cost,
                "tokens": step_tokens,
                "pre_action_obs_short": raw_short,
                "pre_action_obs_long": raw_long,
                # EB-schema aliases so viz/stepwise_eb_learn renders this run.
                # Non-agent cost buckets are always zero for this baseline.
                "agent_step_cost": step_cost,
                "step_total_cost": step_cost,
                "extract_cost": 0.0,
                "improve_cost": 0.0,
                "experiment_cost": 0.0,
                "trim_cost": 0.0,
                "input_tokens": step_tokens.get("prompt_tokens", 0),
                "output_tokens": step_tokens.get("completion_tokens", 0),
                "phase": "complete",
                "num_qa_pairs": 0,
                "num_unanswered_questions": 0,
            }
            if isinstance(info, dict):
                step_log["env_info"] = info
            with open(step_dir / "step_log.json", "w") as f:
                json.dump(step_log, f, indent=4, default=str)

            # Per-step agent_messages.json: full rolling history snapshot
            # (post-trim). Text content is preserved; image parts are replaced
            # with an ``[image attached]`` marker. The last assistant entry
            # carries the parsed action for the viz trajectory panel.
            agent_messages = _flatten_history_for_log(history, action, step_dir=step_dir)
            with open(step_dir / "agent_messages.json", "w") as amf:
                json.dump(agent_messages, amf, indent=2, default=str)

            if (
                sc.history_checkpoint_interval > 0
                and global_step > 0
                and global_step % sc.history_checkpoint_interval == 0
            ):
                with open(step_dir / "dynamics_history.json", "w") as dhf:
                    json.dump(history, dhf, indent=2, default=str)

            csv_writer.writerow([
                step, action, reasoning,
                _compose_obs_text(raw_short, raw_long),
                raw_short, float(reward), done,
            ])
            csv_file.flush()

            pbar.update(1)
            evolve_logger.info(
                f"[g{global_step}|ep{episode_idx}|s{step}] "
                f"action={action!r}  reward={float(reward):.2f}  "
                f"return={episode_return:.2f}  done={done}  "
                f"step_cost=${step_cost:.4f}  cum_cost=${cumulative_cost:.4f}"
            )

            _update_summary_json(
                output_dir=str(episode_dir.parent),
                step=global_step,
                step_cost=step_cost,
                cumulative_cost=cumulative_cost,
                rollout_stats={
                    "episode_idx": episode_idx,
                    "episode_step": step,
                    "action": action,
                    "reward": float(reward),
                    "episode_return": episode_return,
                    "done": done,
                    "tokens": step_tokens,
                    "invalid_action": invalid_action_this_step,
                    "action_parse_failed": parse_failed,
                },
            )

            obs = obs_next
            respawn_notice = False

            if done:
                evolve_logger.info(
                    f"[g{global_step}] Episode {episode_idx} DONE — "
                    f"return={episode_return:.2f}, steps={step + 1}"
                )
                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
                pbar.set_postfix_str("DONE")
                break

    finally:
        pbar.close()
        try:
            csv_file.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    episode_log["num_steps"] = step + 1
    episode_log["episode_return"] = episode_return
    try:
        episode_log["env_stats"] = env.get_stats()
    except Exception:  # noqa: BLE001
        pass
    episode_log["failed_candidates"] = list(getattr(env, "failed_candidates", []))
    with open(episode_dir / "episode_log.json", "w") as f:
        json.dump(episode_log, f, indent=4, default=str)

    return episode_log, step + 1, cumulative_cost, terminal_observation


# ---------------------------------------------------------------------------
# Outer orchestrator
# ---------------------------------------------------------------------------


def stepwise_simple(
    sc: SimpleStepwiseConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
) -> None:
    evolve_logger.info("Starting stepwise simple-agent baseline")

    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge: {len(default_knowledge)} chars")

    if sc.mock_mode:
        _install_litellm_mock()
        evolve_logger.info(
            "Mock mode: prompts/logs are constructed normally; "
            "litellm.completion is stubbed to synthesize a brief plan plus "
            "a <|ACTION|>...<|END|> response with a random valid env action "
            "each step."
        )

    api_key = os.getenv(sc.api_key_env)
    if not api_key:
        if sc.mock_mode:
            api_key = "mock-api-key-not-used"
        else:
            raise RuntimeError(
                f"Env var {sc.api_key_env!r} is not set. "
                "Simple baseline requires an LLM API key."
            )

    history: list[dict] = []
    if sc.dynamics_history_path:
        seed_path = Path(sc.dynamics_history_path)
        if not seed_path.is_absolute():
            seed_path = Path(original_cwd) / seed_path
        try:
            with open(seed_path) as f:
                seed_history = json.load(f)
            if isinstance(seed_history, list):
                history.extend(seed_history)
                evolve_logger.info(
                    f"Loaded {len(seed_history)} seed history messages from "
                    f"{seed_path}"
                )
            else:
                evolve_logger.warning(
                    f"Seed history at {seed_path} is not a list; ignoring."
                )
        except (OSError, json.JSONDecodeError) as e:
            evolve_logger.warning(
                f"Failed to load dynamics_history_path={seed_path}: {e}"
            )

    episode_idx = 0
    global_steps_used = 0
    cumulative_cost = 0.0
    is_first_step_in_run = True
    previous_terminal_observation: dict | None = None
    env_name = config.envs.names.split("-")[0]

    evolve_logger.info(
        f"Simple agent ready (model={sc.model}, history_window={sc.history_window}, "
        f"max_image_history={sc.max_image_history}, "
        f"hide_obs_when_image={sc.hide_obs_when_image}, mock_mode={sc.mock_mode})"
    )

    while global_steps_used < sc.n_environment_steps:
        remaining = sc.n_environment_steps - global_steps_used
        episode_dir = Path(output_dir) / f"episode_{episode_idx}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        evolve_logger.info(
            f"\n{'=' * 80}\nSIMPLE-AGENT EPISODE {episode_idx} "
            f"(global steps: {global_steps_used}/{sc.n_environment_steps}, "
            f"remaining: {remaining})\n{'=' * 80}"
        )

        (
            episode_log,
            steps_taken,
            cumulative_cost,
            previous_terminal_observation,
        ) = _run_episode(
            config=config,
            sc=sc,
            api_key=api_key,
            history=history,
            episode_idx=episode_idx,
            episode_dir=episode_dir,
            default_knowledge=default_knowledge,
            global_step_start=global_steps_used,
            max_episode_steps=remaining,
            is_first_step_in_run=is_first_step_in_run,
            cumulative_cost_start=cumulative_cost,
            previous_terminal_observation=previous_terminal_observation,
        )
        is_first_step_in_run = False
        global_steps_used += steps_taken

        with open(episode_dir / "dynamics_history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

        episode_idx += 1

        evolve_logger.info(
            f"[g{global_steps_used}] Episode {episode_idx - 1} done — "
            f"return={episode_log.get('episode_return', 0.0):.2f}, "
            f"steps={steps_taken}"
        )

        if is_minihack_success_episode(env_name, episode_log):
            evolve_logger.info(
                f"[g{global_steps_used}] MiniHack task success reached; "
                "stopping run before starting another episode."
            )
            break

    with open(Path(output_dir) / "dynamics_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    run_name_suffix = f"simple_{config.envs.names}_stepwise"
    original_cwd, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=config.eval.resume_from,
        output_dir_base=config.eval.output_dir,
        logger_name="evolve",
    )

    scfg = config.eval.get("simple", {}) or {}
    simple_history_window = scfg.get("history_window", None)
    simple_max_image_history = scfg.get("max_image_history", None)
    evolve_logger.info(f"Agent goal: {resolve_agent_goal(config)}")
    sc = SimpleStepwiseConfig(
        n_environment_steps=int(
            scfg.get("n_environment_steps", config.eval.evolve.get("n_environment_steps", 20))
        ),
        history_window=int(
            config.agent.get("max_text_history", 20)
            if simple_history_window is None
            else simple_history_window
        ),
        max_image_history=int(
            config.agent.get("max_image_history", 1)
            if simple_max_image_history is None
            else simple_max_image_history
        ),
        model=str(scfg.get("model", "anthropic/claude-sonnet-4-5-20250929")),
        api_key_env=str(scfg.get("api_key_env", "ANTHROPIC_API_KEY")),
        base_url=scfg.get("base_url", None),
        temperature=scfg.get("temperature", None),
        max_output_tokens=scfg.get("max_output_tokens", None),
        hide_obs_when_image=bool(
            scfg.get(
                "hide_obs_when_image",
                config.eval.evolve.get("hide_obs_when_image", False),
            )
        ),
        mock_mode=bool(scfg.get("mock_mode", False)),
        dynamics_history_path=scfg.get("dynamics_history_path", None),
        history_checkpoint_interval=int(scfg.get("history_checkpoint_interval", 0)),
    )

    stepwise_simple(
        sc=sc,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
