"""OpenHands coding-agent baseline: per-step env interaction inside one long-lived Conversation.

Mirrors the outer shape of stepwise_eb_learn.py — single orchestrator owns the
Conversation and workspace across the whole run; episodes are only visible to
the agent via a 'respawned' prepend on the user message. Each env-step posts
one user message (raw obs + image + valid actions) into the conversation and
waits for a `submit_action` tool call, which pauses the conversation so the
orchestrator can forward the action to env.step(...).

Workspace layout (orchestrator-owned only):
    <workspace>/
      last_n_steps.json        — rolling buffer of the last N committed steps
      obs_images/step_NNN.png  — pre-action image per step, if env emits one

Everything else in the workspace is agent-managed. We do not tell the agent
the files exist; it can discover and use them however it wants.
"""

import base64
import csv
import io
import json
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from collections.abc import Sequence
from omegaconf import DictConfig
from pydantic import Field
from tqdm import tqdm

from openhands.sdk import (
    Agent,
    Conversation,
    ImageContent,
    LLM,
    Message,
    TextContent,
    Tool,
)
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.tools.preset.default import get_default_tools

from balrog.environments import make_env
from balrog.utils import get_unique_seed

from explore import evolve_logger, get_default_knowledge
from run_utils import setup_run, _update_summary_json


# ---------------------------------------------------------------------------
# submit_action: the agent's only env-coupled tool. Pauses the conversation
# so the orchestrator can read the committed action and advance the env.
# ---------------------------------------------------------------------------


class SubmitAction(Action):
    action: str = Field(
        description=(
            "The exact action string to execute in the environment. "
            "Must be one of the valid actions for the current step."
        )
    )
    reasoning: str = Field(
        default="",
        description="Brief rationale for the chosen action (optional).",
    )


class SubmitActionObservation(Observation):
    pass


class _ActionSink:
    """Cross-call state hand-off between orchestrator and tool executor."""

    def __init__(self) -> None:
        self.action: str | None = None
        self.reasoning: str = ""
        self.call_count: int = 0

    def reset(self) -> None:
        self.action = None
        self.reasoning = ""

    def record(self, action: str, reasoning: str) -> None:
        self.action = action
        self.reasoning = reasoning
        self.call_count += 1


class SubmitActionExecutor(ToolExecutor):
    def __init__(self, sink: _ActionSink) -> None:
        self._sink = sink

    def __call__(self, action: SubmitAction, conversation=None) -> SubmitActionObservation:
        self._sink.record(action.action, action.reasoning)
        # FIFOLock is reentrant; pausing from inside the tool is safe and causes
        # run() to exit at the next loop iteration after this step completes.
        if conversation is not None:
            try:
                conversation.pause()
            except Exception as e:  # noqa: BLE001
                evolve_logger.warning(f"submit_action: pause() failed: {e}")
        return SubmitActionObservation.from_text(
            f"Action '{action.action}' committed; control returned to environment."
        )


_SUBMIT_ACTION_DESCRIPTION = """Commit a single environment action for this step.

Call this tool exactly once per step when you have decided what to do. After it
returns, the environment advances and you will receive the next observation in a
following user message. The action string must match one of the valid actions
listed in the current user message.
"""


# Module-level sink: the orchestrator holds a reference and resets it per step;
# the tool class reaches in via SubmitActionTool.create. A module-level
# definition (not a factory-local class) is required so Pydantic's discriminated
# union captures `kind="SubmitActionTool"` at class-creation time, which the
# OpenHands ToolDefinition registry validates against.
_SUBMIT_ACTION_SINK = _ActionSink()


class SubmitActionTool(ToolDefinition[SubmitAction, SubmitActionObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["SubmitActionTool"]:  # noqa: ARG003
        return [
            cls(
                action_type=SubmitAction,
                observation_type=SubmitActionObservation,
                description=_SUBMIT_ACTION_DESCRIPTION,
                executor=SubmitActionExecutor(_SUBMIT_ACTION_SINK),
            )
        ]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class OpenHandsStepwiseConfig:
    n_environment_steps: int
    history_window: int                  # N entries retained in last_n_steps.json
    max_iteration_per_step: int          # cap on OpenHands inner iterations per env step
    openhands_model: str                 # e.g. "anthropic/claude-sonnet-4-5-20250929"
    openhands_api_key_env: str           # env var name holding the key, e.g. "ANTHROPIC_API_KEY"
    openhands_base_url: str | None = None
    openhands_temperature: float | None = None
    openhands_max_output_tokens: int | None = None
    enable_browser_tools: bool = False
    mock_mode: bool = False


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


def _save_obs_image(img, workspace: Path, step_num: int) -> str | None:
    if img is None:
        return None
    out_dir = workspace / "obs_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = f"obs_images/step_{step_num:04d}.png"
    try:
        img.save(workspace / rel)
    except Exception as e:  # noqa: BLE001
        evolve_logger.warning(f"Failed to save obs image for step {step_num}: {e}")
        return None
    return rel


def _save_step_dir_image(img, step_dir: Path, filename: str) -> None:
    """Write the PIL image into the per-step directory under the given name.

    Expected by viz.stepwise_eb_viewer_data (obs_before.png / obs_after.png).
    """
    if img is None:
        return
    try:
        img.save(step_dir / filename)
    except Exception as e:  # noqa: BLE001
        evolve_logger.warning(f"Failed to save {filename} in {step_dir}: {e}")


# ---------------------------------------------------------------------------
# Rolling step buffer
# ---------------------------------------------------------------------------


def _write_last_n_steps(workspace: Path, entries: list[dict]) -> None:
    path = workspace / "last_n_steps.json"
    with open(path, "w") as f:
        json.dump(entries, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Env dispatch (matches stepwise_eb_learn.py:1045-1058)
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


# ---------------------------------------------------------------------------
# Per-step user message construction
# ---------------------------------------------------------------------------


def _build_user_message(
    env,
    obs: dict,
    is_first_step: bool,
    instruction_prompt: str,
    default_knowledge: str,
    respawn_notice: bool,
    episode_idx: int,
    step_idx: int,
    global_step: int,
    invalid_action_feedback: str | None = None,
) -> Message:
    long_ctx = obs.get("text", {}).get("long_term_context", "") or ""
    short_ctx = obs.get("text", {}).get("short_term_context", "") or ""
    image = obs.get("image")

    lines: list[str] = []

    if is_first_step:
        lines.append(
            "You are an agent interacting with an environment. On each of your turns "
            "you will receive the current raw observation. "
            "You must commit exactly one action per turn by calling the `submit_action` "
            "tool. Between commits you are free to use the workspace (bash, file edits, "
            "etc.)."
        )
        if instruction_prompt:
            lines.append("\n=== ENVIRONMENT INSTRUCTIONS ===")
            lines.append(instruction_prompt.strip())
            lines.append("=== END ENVIRONMENT INSTRUCTIONS ===")
        if default_knowledge and default_knowledge.strip() != instruction_prompt.strip():
            lines.append("\n=== DEFAULT KNOWLEDGE ===")
            lines.append(default_knowledge.strip())
            lines.append("=== END DEFAULT KNOWLEDGE ===")

    header_prefix = ""
    if respawn_notice:
        header_prefix = "The previous episode was terminated and you have respawned.\n\n"

    lines.append(
        f"\n=== STEP (episode={episode_idx}, step={step_idx}, global_step={global_step}) ==="
    )
    if header_prefix:
        lines.append(header_prefix.rstrip())

    if invalid_action_feedback:
        lines.append("\n[invalid_previous_action]")
        lines.append(invalid_action_feedback)

    lines.append("\nCurrent Observation:")
    if short_ctx:
        lines.append(short_ctx)
    lines.append(long_ctx)

    lines.append(
        "\nCommit your chosen action by calling `submit_action(action=..., reasoning=...)`."
    )

    text = "\n".join(lines)
    content: list[Any] = [TextContent(text=text)]

    data_url = _pil_to_data_url(image)
    if data_url is not None:
        content.append(ImageContent(image_urls=[data_url]))

    return Message(role="user", content=content)


# ---------------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------------


def _build_openhands_llm(oh_config: OpenHandsStepwiseConfig) -> LLM:
    api_key = os.getenv(oh_config.openhands_api_key_env)
    if not api_key:
        if oh_config.mock_mode:
            api_key = "mock-api-key-not-used"
        else:
            raise RuntimeError(
                f"Env var {oh_config.openhands_api_key_env!r} is not set. "
                "OpenHands baseline requires an LLM API key for the coding agent."
            )
    kwargs: dict[str, Any] = {
        "model": oh_config.openhands_model,
        "api_key": api_key,
        "usage_id": "openhands_stepwise",
    }
    if oh_config.openhands_base_url:
        kwargs["base_url"] = oh_config.openhands_base_url
    if oh_config.openhands_temperature is not None:
        kwargs["temperature"] = oh_config.openhands_temperature
    if oh_config.openhands_max_output_tokens is not None:
        kwargs["max_output_tokens"] = oh_config.openhands_max_output_tokens
    return LLM(**kwargs)


# ---------------------------------------------------------------------------
# Mock-mode support: exercise the full OpenHands conversation loop end-to-end,
# but intercept the LiteLLM transport call so no API requests are made. The
# synthesized response is an assistant message with a tool_call to
# `submit_action`, so the agent/tool/sink/pause machinery runs as normal.
# Mirrors stepwise_eb_learn.py's mock_mode semantics (run everything except
# the network call).
# ---------------------------------------------------------------------------


_MOCK_ACTION_PROVIDER = None  # set per-episode via set_mock_action_provider


def set_mock_action_provider(fn) -> None:
    """Install a zero-arg callable returning a valid action string for the
    current env. Called by the patched litellm_completion when it needs to
    synthesize the agent's LLM response."""
    global _MOCK_ACTION_PROVIDER
    _MOCK_ACTION_PROVIDER = fn


def _valid_actions(env) -> list[str]:
    """Return the list of valid action strings for ``env``, unwrapping one
    level of wrapper if the outer object does not expose ``language_action_space``
    (MiniHack exposes it on ``env.env``). Also handles the Strings wrapper
    (``_values``) used by crafter. Mirrors ``stepwise_eb_learn._mock_available_actions``."""
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


def _pick_mock_action(env) -> str:
    candidates = _valid_actions(env)
    if not candidates:
        return getattr(env, "default_action", "wait")
    return random.choice(candidates)


def _fallback_action(env, valid_actions: list[str]) -> str:
    """Pick a safe action when the agent did not submit one. Prefer the env's
    declared default, but only if it is actually in the valid action list —
    otherwise fall back to the first valid action. Many MiniHack tasks reject
    the literal ``"wait"`` default."""
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


def _message_text(user_msg: Message) -> str:
    """Extract the text portion of an OpenHands multi-modal Message."""
    out = []
    for part in getattr(user_msg, "content", []) or []:
        text = getattr(part, "text", None)
        if text:
            out.append(text)
    return "\n".join(out)


def _flatten_conversation_events(conversation, final_action: str | None) -> list[dict]:
    """Walk ``conversation._state.events`` and serialize each LLM-convertible
    event into a ``{role, content, ...}`` dict for the viz agent-messages
    panel. ``final_action`` is attached to the most recent ActionEvent entry.
    Best-effort: returns [] if the state is not accessible."""
    from openhands.sdk.event.base import LLMConvertibleEvent
    from openhands.sdk.event.llm_convertible import (
        ActionEvent,
        AgentErrorEvent,
        MessageEvent,
        ObservationEvent,
        SystemPromptEvent,
        UserRejectObservation,
    )
    from openhands.sdk.llm import content_to_str

    try:
        events = list(conversation._state.events)  # noqa: SLF001
    except Exception as e:  # noqa: BLE001
        evolve_logger.debug(f"_flatten_conversation_events: events access failed: {e}")
        return []

    out: list[dict] = []
    last_action_idx: int | None = None
    for ev in events:
        if not isinstance(ev, LLMConvertibleEvent):
            # Skip non-LLM events like PauseEvent — they're internal control
            # flow, not part of the conversation the model sees.
            continue
        if isinstance(ev, SystemPromptEvent):
            text = getattr(ev.system_prompt, "text", "") or ""
            out.append({"role": "system", "content": text})
        elif isinstance(ev, MessageEvent):
            llm_msg = ev.llm_message
            parts: list[str] = []
            for p in getattr(llm_msg, "content", []) or []:
                t = getattr(p, "text", None)
                if t:
                    parts.append(t)
                elif hasattr(p, "image_urls"):
                    parts.append("[image attached]")
            out.append(
                {
                    "role": getattr(llm_msg, "role", ev.source),
                    "content": "\n".join(parts),
                }
            )
        elif isinstance(ev, ActionEvent):
            thought_parts: list[str] = []
            reasoning = getattr(ev, "reasoning_content", "") or ""
            if reasoning:
                thought_parts.append(f"[reasoning]\n{reasoning}")
            thought_text = "\n".join(
                getattr(t, "text", "") or "" for t in getattr(ev, "thought", []) or []
            ).strip()
            if thought_text:
                thought_parts.append(f"[thought]\n{thought_text}")
            tool_name = ev.tool_name
            args_str = ""
            try:
                args_str = getattr(ev.tool_call, "arguments", "") or ""
            except Exception:  # noqa: BLE001
                pass
            thought_parts.append(f"[tool_call: {tool_name}]")
            if args_str:
                thought_parts.append(args_str)
            entry: dict = {
                "role": "assistant",
                "content": "\n".join(thought_parts),
                "tool_name": tool_name,
            }
            out.append(entry)
            last_action_idx = len(out) - 1
        elif isinstance(ev, ObservationEvent):
            try:
                obs_text = "".join(content_to_str(ev.observation.to_llm_content))
            except Exception:  # noqa: BLE001
                obs_text = str(ev.observation)
            out.append(
                {"role": "tool", "content": obs_text, "tool_name": ev.tool_name}
            )
        elif isinstance(ev, AgentErrorEvent):
            out.append(
                {
                    "role": "tool",
                    "content": getattr(ev, "error", "") or str(ev),
                    "tool_name": ev.tool_name,
                }
            )
        elif isinstance(ev, UserRejectObservation):
            out.append(
                {
                    "role": "tool",
                    "content": getattr(ev, "rejection_reason", "") or str(ev),
                    "tool_name": ev.tool_name,
                }
            )
        else:
            out.append({"role": "unknown", "content": str(ev)})

    if last_action_idx is not None and final_action is not None:
        out[last_action_idx]["action"] = final_action
    return out


def _read_conv_metrics(conversation) -> tuple[float, dict[str, int]]:
    """Snapshot accumulated cost and token usage from the conversation's stats.
    Returns (accumulated_cost, token_usage_dict). Best-effort: returns zeros if
    the stats path is not available (e.g. mock mode with no real responses)."""
    try:
        state = conversation._state  # noqa: SLF001
        combined = state.stats.get_combined_metrics()
        usage = combined.accumulated_token_usage
        tokens: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        }
        if usage is not None:
            tokens["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
            tokens["completion_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
            tokens["cache_read_tokens"] = int(getattr(usage, "cache_read_tokens", 0) or 0)
            tokens["cache_write_tokens"] = int(getattr(usage, "cache_write_tokens", 0) or 0)
            tokens["reasoning_tokens"] = int(getattr(usage, "reasoning_tokens", 0) or 0)
        return float(combined.accumulated_cost or 0.0), tokens
    except Exception as e:  # noqa: BLE001
        evolve_logger.debug(f"_read_conv_metrics failed: {e}")
        return 0.0, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        }


def _extract_tool_names(tools) -> list[str]:
    names: list[str] = []
    if not tools:
        return names
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else getattr(t, "function", None)
        name = None
        if isinstance(fn, dict):
            name = fn.get("name")
        elif fn is not None:
            name = getattr(fn, "name", None)
        if name:
            names.append(name)
    return names


def _mock_litellm_completion(*, messages, model, tools=None, **kwargs):
    """Stand-in for litellm.completion: returns a ModelResponse whose assistant
    message calls `submit_action` with a random valid env action. When
    submit_action isn't available among the tools (shouldn't happen in our
    flow), falls back to an empty-content text response."""
    import json
    import uuid
    from litellm import ModelResponse, Choices
    from litellm.types.utils import Message as LLMMsg
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Function,
    )

    tool_names = _extract_tool_names(tools)
    want_submit = "submit_action" in tool_names

    if want_submit:
        action = (
            _MOCK_ACTION_PROVIDER() if _MOCK_ACTION_PROVIDER is not None else "wait"
        )
        tool_call = ChatCompletionMessageToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=Function(
                name="submit_action",
                arguments=json.dumps(
                    {"action": action, "reasoning": "(mock llm: random valid action)"}
                ),
            ),
        )
        msg = LLMMsg(role="assistant", content=None, tool_calls=[tool_call])
        finish_reason = "tool_calls"
    else:
        msg = LLMMsg(role="assistant", content="(mock-mode noop)")
        finish_reason = "stop"

    return ModelResponse(
        id=f"mock-{uuid.uuid4().hex[:8]}",
        choices=[Choices(index=0, message=msg, finish_reason=finish_reason)],
        model=model,
        object="chat.completion",
        created=0,
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


def _install_litellm_mock() -> None:
    """Monkeypatch the binding used inside openhands.sdk.llm.llm._transport_call.
    Also replace any `completion_cost` path with a zero-cost stub to avoid
    pricing lookups on zero-token synthetic responses."""
    import openhands.sdk.llm.llm as _llm_module

    _llm_module.litellm_completion = _mock_litellm_completion

    # Silence any cost-lookup path that may not tolerate zero-token mocks.
    try:
        import litellm

        if not getattr(litellm, "_oh_stepwise_mock_cost_installed", False):
            _orig_cost = getattr(litellm, "completion_cost", None)

            def _zero_cost(*args, **kwargs):  # noqa: ARG001
                return 0.0

            litellm.completion_cost = _zero_cost
            litellm._oh_stepwise_mock_cost_installed = True  # type: ignore[attr-defined]
            litellm._oh_stepwise_mock_cost_orig = _orig_cost  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        evolve_logger.warning(f"Could not stub litellm.completion_cost: {e}")


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


def _run_episode(
    config: DictConfig,
    oh_config: OpenHandsStepwiseConfig,
    conversation: Conversation,
    sink: _ActionSink,
    workspace: Path,
    episode_idx: int,
    episode_dir: Path,
    default_knowledge: str,
    rolling_buffer: list[dict],
    global_step_start: int,
    max_episode_steps: int | None,
    is_first_step_in_run: bool,
) -> tuple[dict, int, bool]:
    """Run one episode; returns (episode_log, steps_taken, first_step_flag_after)."""
    env_name, task, env = _make_env_for_config(config)

    # In mock mode, register a closure that hands the patched
    # litellm_completion a random valid action from *this* episode's env.
    if oh_config.mock_mode:
        set_mock_action_provider(lambda: _pick_mock_action(env))

    seed = config.envs.env_kwargs.seed
    if seed is None:
        seed = get_unique_seed(process_num=0, episode_idx=episode_idx)
    random.seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)

    max_steps = env.max_steps if config.eval.get("max_steps_per_episode") is None else config.eval.max_steps_per_episode
    if max_episode_steps is not None:
        max_steps = min(max_steps, max_episode_steps)

    try:
        instruction_prompt = env.get_instruction_prompt()
    except Exception:
        instruction_prompt = default_knowledge

    episode_log: dict = {
        "task": task,
        "env_name": env_name,
        "episode_idx": episode_idx,
        "seed": seed,
        "action_frequency": defaultdict(int),
        "num_submit_action_calls": 0,
        "num_default_action_fallbacks": 0,
        "episode_return": 0.0,
    }

    pbar = tqdm(total=max_steps, desc=f"OpenHands ep {episode_idx}", leave=False, dynamic_ncols=True)

    episode_return = 0.0
    step = 0
    done = False
    respawn_notice = episode_idx > 0
    first_step_flag_after = False  # becomes True if run ends without sending any step
    feedback_on_invalid = bool(config.eval.get("feedback_on_invalid_action", False))
    pending_invalid_feedback: str | None = None

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
            image_rel = _save_obs_image(obs.get("image"), workspace, global_step)
            _save_step_dir_image(obs.get("image"), step_dir, "obs_before.png")

            valid_actions = _valid_actions(env)

            # Full pipeline (real or mocked LLM): build user message, run the
            # conversation until submit_action fires and pauses it, then read
            # the recorded action from the sink.
            assert conversation is not None and sink is not None
            sink.reset()
            call_count_before = sink.call_count
            user_msg = _build_user_message(
                env=env,
                obs=obs,
                is_first_step=(
                    is_first_step_in_run and step == 0 and episode_idx == 0
                ),
                instruction_prompt=instruction_prompt or "",
                default_knowledge=default_knowledge or "",
                respawn_notice=respawn_notice and step == 0,
                episode_idx=episode_idx,
                step_idx=step,
                global_step=global_step,
                invalid_action_feedback=pending_invalid_feedback,
            )
            pending_invalid_feedback = None
            conversation.send_message(user_msg)
            cost_before, tokens_before = _read_conv_metrics(conversation)
            _t0 = time.time()
            try:
                conversation.run()
            except Exception as e:  # noqa: BLE001
                evolve_logger.error(
                    f"[g{global_step}] conversation.run() raised: {e}"
                )
            dt = time.time() - _t0
            cost_after, tokens_after = _read_conv_metrics(conversation)
            step_cost = max(0.0, cost_after - cost_before)
            step_tokens = {
                k: max(0, tokens_after.get(k, 0) - tokens_before.get(k, 0))
                for k in ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                          "cache_write_tokens", "reasoning_tokens")
            }
            call_count_after = sink.call_count
            if sink.action is not None and call_count_after > call_count_before:
                action = sink.action
                reasoning = sink.reasoning
            else:
                action = _fallback_action(env, valid_actions)
                reasoning = "(no submit_action this step: fell back to default action)"
                episode_log["num_default_action_fallbacks"] += 1
                evolve_logger.warning(
                    f"[g{global_step}] submit_action was not called this step "
                    f"(run() took {dt:.1f}s); using default action '{action}'."
                )
            episode_log["num_submit_action_calls"] += (
                1 if call_count_after > call_count_before else 0
            )

            # Clear pause so the next run() can transition cleanly.
            with conversation._state:  # noqa: SLF001 — reaching in intentionally
                if conversation._state.execution_status == ConversationExecutionStatus.PAUSED:
                    conversation._state.execution_status = ConversationExecutionStatus.IDLE

            # Step env.
            invalid_action_this_step = False
            try:
                obs_next, reward, terminated, truncated, info = env.step(action)
            except ValueError as e:
                logging.warning(f"[g{global_step}] Invalid action: {action} — {e}")
                invalid_action_this_step = True
                if feedback_on_invalid:
                    pending_invalid_feedback = (
                        f"Your previous submit_action call used action {action!r}, "
                        f"which the environment rejected as invalid ({e}). "
                        "The observation below is unchanged from the previous step — "
                        "retry with one of the valid actions listed."
                    )
                terminated = False
                truncated = False
                reward = 0.0
                obs_next = obs

            done = bool(terminated or truncated)
            episode_return += float(reward)
            episode_log["action_frequency"][action] += 1

            if isinstance(obs_next, dict):
                _save_step_dir_image(obs_next.get("image"), step_dir, "obs_after.png")

            # Rolling buffer update (last N).
            entry = {
                "step": global_step,
                "episode_idx": episode_idx,
                "pre_action_obs_long": raw_long,
                "pre_action_obs_short": raw_short,
                "pre_action_image_path": image_rel,
                "action": action,
                "reasoning": reasoning,
                "reward": float(reward),
                "done": done,
                "info": info if isinstance(info, dict) else None,
            }
            rolling_buffer.append(entry)
            if len(rolling_buffer) > oh_config.history_window:
                del rolling_buffer[: len(rolling_buffer) - oh_config.history_window]
            _write_last_n_steps(workspace, rolling_buffer)

            # Per-step log.
            step_log = {
                "step": step,
                "global_step": global_step,
                "episode_idx": episode_idx,
                "action": action,
                "reasoning": reasoning,
                "reward": float(reward),
                "done": done,
                "episode_return_so_far": episode_return,
                "submit_action_called": call_count_after > call_count_before,
                "invalid_action": invalid_action_this_step,
                "step_cost": step_cost,
                "cumulative_cost": cost_after,
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

            # Per-step agent_messages.json: full conversation snapshot built
            # from ``conversation._state.events``. Covers system prompt, user
            # observations, agent thought/tool_calls, tool results, and errors.
            # The last ActionEvent carries the parsed action for the viz
            # trajectory panel.
            agent_messages = _flatten_conversation_events(conversation, action)
            with open(step_dir / "agent_messages.json", "w") as amf:
                json.dump(agent_messages, amf, indent=2, default=str)

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
                f"step_cost=${step_cost:.4f}  cum_cost=${cost_after:.4f}"
            )

            _update_summary_json(
                output_dir=str(episode_dir.parent),
                step=global_step,
                step_cost=step_cost,
                cumulative_cost=cost_after,
                rollout_stats={
                    "episode_idx": episode_idx,
                    "episode_step": step,
                    "action": action,
                    "reward": float(reward),
                    "episode_return": episode_return,
                    "done": done,
                    "tokens": step_tokens,
                    "invalid_action": invalid_action_this_step,
                },
            )

            obs = obs_next
            respawn_notice = False  # only prepend on the first step after a respawn

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

    return episode_log, step + 1, first_step_flag_after


# ---------------------------------------------------------------------------
# Outer orchestrator
# ---------------------------------------------------------------------------


def stepwise_openhands(
    oh_config: OpenHandsStepwiseConfig,
    config: DictConfig,
    original_cwd: str,
    output_dir: str,
) -> None:
    evolve_logger.info("Starting stepwise OpenHands baseline")

    workspace = Path(output_dir) / "openhands_ws"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "obs_images").mkdir(parents=True, exist_ok=True)
    _write_last_n_steps(workspace, [])

    default_knowledge = get_default_knowledge(config)
    evolve_logger.info(f"Default knowledge: {len(default_knowledge)} chars")

    # Install LLM transport mock up-front if requested, before building LLM.
    if oh_config.mock_mode:
        _install_litellm_mock()
        evolve_logger.info(
            "Mock mode: full OpenHands conversation pipeline is exercised; "
            "the LiteLLM transport call is stubbed to synthesize a submit_action "
            "tool call with a random valid env action each step."
        )

    llm = _build_openhands_llm(oh_config)
    tools = get_default_tools(enable_browser=oh_config.enable_browser_tools)
    sink = _SUBMIT_ACTION_SINK
    register_tool("SubmitActionTool", SubmitActionTool)
    tools = list(tools) + [Tool(name="SubmitActionTool")]
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt_filename=str(Path(__file__).parent / "openhands_empty_system_prompt.j2"),
    )
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=oh_config.max_iteration_per_step,
        visualizer=None,
    )
    evolve_logger.info(
        f"Conversation ready (model={oh_config.openhands_model}, "
        f"workspace={workspace}, tools={[t.name for t in tools]}, "
        f"mock_mode={oh_config.mock_mode})"
    )

    rolling_buffer: list[dict] = []
    episode_idx = 0
    global_steps_used = 0
    cumulative_cost = 0.0
    is_first_step_in_run = True

    try:
        while global_steps_used < oh_config.n_environment_steps:
            remaining = oh_config.n_environment_steps - global_steps_used
            episode_dir = Path(output_dir) / f"episode_{episode_idx}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            evolve_logger.info(
                f"\n{'=' * 80}\nOPENHANDS EPISODE {episode_idx} "
                f"(global steps: {global_steps_used}/{oh_config.n_environment_steps}, "
                f"remaining: {remaining})\n{'=' * 80}"
            )

            episode_log, steps_taken, _ = _run_episode(
                config=config,
                oh_config=oh_config,
                conversation=conversation,
                sink=sink,
                workspace=workspace,
                episode_idx=episode_idx,
                episode_dir=episode_dir,
                default_knowledge=default_knowledge,
                rolling_buffer=rolling_buffer,
                global_step_start=global_steps_used,
                max_episode_steps=remaining,
                is_first_step_in_run=is_first_step_in_run,
            )
            is_first_step_in_run = False
            global_steps_used += steps_taken
            episode_idx += 1

            evolve_logger.info(
                f"[g{global_steps_used}] Episode {episode_idx - 1} done — "
                f"return={episode_log.get('episode_return', 0.0):.2f}, "
                f"steps={steps_taken}"
            )
    finally:
        if conversation is not None:
            try:
                conversation.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    run_name_suffix = f"openhands_{config.envs.names}_stepwise"
    original_cwd, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=config.eval.resume_from,
        output_dir_base=config.eval.output_dir,
        logger_name="evolve",
    )

    ohcfg = config.eval.get("openhands", {}) or {}
    oh_config = OpenHandsStepwiseConfig(
        n_environment_steps=int(ohcfg.get("n_environment_steps", config.eval.evolve.get("n_environment_steps", 20))),
        history_window=int(ohcfg.get("history_window", 20)),
        max_iteration_per_step=int(ohcfg.get("max_iteration_per_step", 30)),
        openhands_model=str(ohcfg.get("model", "anthropic/claude-sonnet-4-5-20250929")),
        openhands_api_key_env=str(ohcfg.get("api_key_env", "ANTHROPIC_API_KEY")),
        openhands_base_url=ohcfg.get("base_url", None),
        openhands_temperature=ohcfg.get("temperature", None),
        openhands_max_output_tokens=ohcfg.get("max_output_tokens", None),
        enable_browser_tools=bool(ohcfg.get("enable_browser_tools", False)),
        mock_mode=bool(ohcfg.get("mock_mode", False)),
    )

    stepwise_openhands(
        oh_config=oh_config,
        config=config,
        original_cwd=original_cwd,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
