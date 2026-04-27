"""Shared run-level agent goal resolution and prompt formatting."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


AGENT_GOAL_PERFORMANCE = "Your goal is to get as far as possible in the game."
AGENT_GOAL_DYNAMICS = (
    "Your goal is to understand how the environment works."
)

# TODO: customize these env-specific eval goals. Used by ExploreEval phase 2.
AGENT_GOAL_MINIHACK_EVAL = (
    "Your goal is to reach the stairs down ('>')"
)
AGENT_GOAL_ARC_AGI_EVAL = (
    "Your goal is to get as far as possible in the game."
)
AGENT_GOAL_AUTUMN_EVAL = (
    "You are shown the current state grid and target goal grid. Your task is to execute actions that transform the current state to the goal state."
)

AGENT_GOAL_TEXTS = {
    "performance": AGENT_GOAL_PERFORMANCE,
    "dynamics": AGENT_GOAL_DYNAMICS,
    "minihack_eval": AGENT_GOAL_MINIHACK_EVAL,
    "arc_agi_eval": AGENT_GOAL_ARC_AGI_EVAL,
    "autumn_eval": AGENT_GOAL_AUTUMN_EVAL,
}


def eval_goal_mode_for_env(env_name: str) -> str:
    """Return the AGENT_GOAL_TEXTS key for the given env's eval-phase goal.

    `env_name` may include an env-suffix (e.g. ``autumn-foo``); only the
    prefix before the first ``-`` is used.
    """
    base = (env_name or "").split("-", 1)[0]
    mode = f"{base}_eval"
    if mode not in AGENT_GOAL_TEXTS:
        raise ValueError(
            f"No eval goal mode registered for env {env_name!r} "
            f"(looked up {mode!r}); known: {sorted(AGENT_GOAL_TEXTS)}"
        )
    return mode


def resolve_agent_goal(config: DictConfig) -> str:
    """Resolve the run-level goal text from config.

    Precedence:
    1. ``eval.agent_goal_text`` exact override.
    2. ``eval.agent_goal_mode`` key into ``AGENT_GOAL_TEXTS``.
    3. ``performance`` default for backwards-compatible behavior.
    """
    explicit = OmegaConf.select(config, "eval.agent_goal_text", default=None)
    if explicit:
        return str(explicit).strip()

    mode = str(
        OmegaConf.select(config, "eval.agent_goal_mode", default="performance")
        or "performance"
    ).strip()
    try:
        return AGENT_GOAL_TEXTS[mode]
    except KeyError as exc:
        allowed = ", ".join(sorted(AGENT_GOAL_TEXTS))
        raise ValueError(
            f"Unknown eval.agent_goal_mode={mode!r}; expected one of: {allowed}"
        ) from exc


def format_agent_goal_block(agent_goal: str | None) -> str:
    goal = (agent_goal or "").strip()
    if not goal:
        return ""
    return goal


def append_agent_goal(prompt: str, agent_goal: str | None) -> str:
    """Append the standardized run-level goal block to a prompt."""
    pieces = []
    base = (prompt or "").strip()
    if base:
        pieces.append(base)
    goal_block = format_agent_goal_block(agent_goal)
    if goal_block:
        pieces.append(goal_block)
    return "\n\n".join(pieces)
