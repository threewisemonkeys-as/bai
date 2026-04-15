"""ARC-AGI 3 environment wrapper compatible with BALROG's EnvWrapper interface."""

import logging
import re
from typing import Any, Optional

import numpy as np
from PIL import Image

import arc_agi
from arcengine import GameAction, GameState
from arcengine.enums import FrameDataRaw

from arc_agi_prompts import get_arc_instruction_prompt

logger = logging.getLogger(__name__)


def _pretty_print_frames(frames: list) -> str:
    """Render frame layers in the row-by-row format used by reference ARC-AGI 3 agents.

    Each grid layer is printed as:
        <grid_0>
          [0, 0, 5, ...]
          [0, 8, 8, ...]
          ...
        </grid_0>
    """
    lines = []
    for i, frame in enumerate(frames):
        lines.append(f"<grid_{i}>")
        if hasattr(frame, "tolist"):
            grid = frame.tolist()
        else:
            grid = frame
        for row in grid:
            lines.append(f"{row}")
        lines.append(f"</grid_{i}>")
        lines.append("")
    return "\n".join(lines)


# ARC-AGI 3 canonical 16-color palette as RGB tuples (index 0-15)
# Matches the game engine's color indices from the reference repo
_PALETTE = {
    0: (255, 255, 255),  # white
    1: (204, 204, 204),  # off-white
    2: (153, 153, 153),  # light-gray
    3: (102, 102, 102),  # gray
    4: (51, 51, 51),     # dark-gray
    5: (0, 0, 0),        # black
    6: (229, 58, 163),   # magenta
    7: (255, 123, 204),  # pink
    8: (249, 60, 49),    # red
    9: (30, 147, 255),   # blue
    10: (136, 216, 241), # light-blue
    11: (255, 220, 0),   # yellow
    12: (255, 133, 27),  # orange
    13: (146, 18, 49),   # maroon
    14: (79, 204, 48),   # green
    15: (163, 86, 214),  # purple
}

_IMAGE_SCALE = 2  # 64px -> 128px, matching reference repo


def _frames_to_pil_image(frame_data: FrameDataRaw) -> Optional[Image.Image]:
    """Convert FrameDataRaw frames to a scaled PIL Image.

    Renders the last frame layer using the 16-color palette and scales
    up with nearest-neighbor interpolation to preserve crisp pixel art.
    Multiple frame layers are stacked horizontally.
    """
    if not frame_data.frame:
        return None

    images = []
    for frame in frame_data.frame:
        h, w = frame.shape[:2]
        img_array = np.zeros((h, w, 3), dtype=np.uint8)

        for val, rgb in _PALETTE.items():
            mask = frame == val
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            img_array[mask] = rgb

        img = Image.fromarray(img_array, mode="RGB")
        # Scale up with nearest-neighbor to keep pixel art crisp
        img = img.resize(
            (w * _IMAGE_SCALE, h * _IMAGE_SCALE), Image.NEAREST
        )
        images.append(img)

    if len(images) == 1:
        return images[0]

    # Stack multiple layers horizontally with a separator
    sep_width = 4
    total_w = sum(im.width for im in images) + sep_width * (len(images) - 1)
    max_h = max(im.height for im in images)
    combined = Image.new("RGB", (total_w, max_h), (40, 40, 40))
    x_offset = 0
    for im in images:
        combined.paste(im, (x_offset, 0))
        x_offset += im.width + sep_width

    return combined


def _render_frames_as_text(frame_data: FrameDataRaw) -> str:
    """Convert all frame layers to pretty-printed grid text.

    Uses the row-by-row format from the reference ARC-AGI 3 agents,
    which is more readable than flat JSON for 64x64 grids.
    """
    if not frame_data.frame:
        return "(no frame data)"

    return _pretty_print_frames(frame_data.frame)


def _build_action_strings(available_action_ids: list[int]) -> list[str]:
    """Build the list of valid action strings from available action IDs."""
    actions = []
    for aid in available_action_ids:
        try:
            ga = GameAction.from_id(aid)
            if ga == GameAction.RESET:
                continue  # Don't expose RESET as a player action
            if ga.is_complex():
                # Complex actions need coordinates — add the base name
                actions.append(ga.name)
            else:
                actions.append(ga.name)
        except ValueError:
            continue
    return actions


def _parse_action(action_str: str) -> tuple[Optional[GameAction], Optional[dict]]:
    """Parse a string action into a GameAction and optional data dict.

    Supports formats:
      "ACTION1"
      "ACTION6 x=10 y=20"
      "ACTION6 10 20"
    """
    action_str = action_str.strip().upper()

    # Extract action name (first word)
    parts = action_str.split(None, 1)
    action_name = parts[0]

    try:
        ga = GameAction.from_name(action_name)
    except ValueError:
        return None, None

    data = None
    if ga.is_complex() and len(parts) > 1:
        remainder = parts[1]
        # Try "x=N y=M" format
        match = re.match(r"x\s*=\s*(\d+)\s+y\s*=\s*(\d+)", remainder, re.IGNORECASE)
        if match:
            data = {"x": int(match.group(1)), "y": int(match.group(2))}
        else:
            # Try "N M" format
            nums = re.findall(r"\d+", remainder)
            if len(nums) >= 2:
                data = {"x": int(nums[0]), "y": int(nums[1])}

    return ga, data


class ArcAgiEnvWrapper:
    """Wraps an ARC-AGI 3 environment to match BALROG's EnvWrapper interface.

    Supports the ARC-AGI 3 scorecard system for tracking results across games.
    """

    def __init__(
        self,
        game_id: str,
        max_episode_steps: int = 200,
        seed: int = 0,
        scorecard_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ):
        self.game_id = game_id
        self._max_steps = max_episode_steps
        self._seed = seed

        self._arcade = arc_agi.Arcade()

        # Scorecard support — matching the reference Swarm pattern
        self._scorecard_id = scorecard_id
        self._tags = tags or []
        if scorecard_id is None:
            try:
                self._scorecard_id = self._arcade.open_scorecard(
                    tags=self._tags
                )
                logger.info(
                    f"Opened scorecard {self._scorecard_id} for {game_id}"
                )
            except Exception as e:
                logger.warning(f"Could not open scorecard: {e}")
                self._scorecard_id = None

        self._env = self._arcade.make(
            game_id,
            seed=seed,
            scorecard_id=self._scorecard_id,
        )

        self.failed_candidates: list[str] = []
        self._steps_taken = 0
        self._prev_levels_completed = 0
        self._last_frame_data: Optional[FrameDataRaw] = None
        self._available_actions: list[str] = []

        # Set up from initial observation
        initial = self._env.observation_space
        if initial is not None:
            self._last_frame_data = initial
            self._prev_levels_completed = initial.levels_completed
            self._available_actions = _build_action_strings(initial.available_actions)

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def language_action_space(self) -> list[str]:
        return self._available_actions

    @property
    def default_action(self) -> str:
        if self._available_actions:
            return self._available_actions[0]
        return "ACTION1"

    def _make_obs(self, frame_data: FrameDataRaw) -> dict:
        """Convert FrameDataRaw to a BALROG-style observation dict.

        Matches the observation format used by reference ARC-AGI 3 agents:
        - long_term_context: pretty-printed grid (row-by-row)
        - short_term_context: state, score, action count
        - image: scaled PIL image of the frame
        """
        long_term = _render_frames_as_text(frame_data)

        short_term = (
            f"State: {frame_data.state.name}\n"
            f"Levels completed: {frame_data.levels_completed}/{frame_data.win_levels}\n"
            f"Action count: {self._steps_taken}"
        )

        current_image = _frames_to_pil_image(frame_data)

        return {
            "text": {
                "long_term_context": long_term,
                "short_term_context": short_term,
            },
            "image": current_image,
        }

    def reset(self, seed: Optional[int] = None, **kwargs) -> tuple[dict, dict]:
        """Reset the environment. Returns (obs, info)."""
        self._steps_taken = 0
        self.failed_candidates = []

        frame_data = self._env.reset()
        if frame_data is None:
            # Fallback — recreate the environment
            self._env = self._arcade.make(self.game_id, seed=seed or self._seed)
            frame_data = self._env.observation_space

        self._last_frame_data = frame_data
        self._prev_levels_completed = frame_data.levels_completed if frame_data else 0
        if frame_data:
            self._available_actions = _build_action_strings(frame_data.available_actions)

        obs = self._make_obs(frame_data) if frame_data else {
            "text": {"long_term_context": "(reset failed)", "short_term_context": ""},
            "image": None,
        }

        info = {
            "game_id": self.game_id,
            "win_levels": frame_data.win_levels if frame_data else 0,
        }

        return obs, info

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        """Take a step. Returns (obs, reward, terminated, truncated, info)."""
        self._steps_taken += 1

        # Parse and validate action
        ga, data = _parse_action(action)
        if ga is None:
            self.failed_candidates.append(action)
            ga = GameAction.from_name(self.default_action)
            data = None

        # Execute action
        frame_data = self._env.step(ga, data=data)
        if frame_data is None:
            # Step failed — return current state with no reward
            obs = self._make_obs(self._last_frame_data) if self._last_frame_data else {
                "text": {"long_term_context": "(step failed)", "short_term_context": ""},
                "image": None,
            }
            return obs, 0.0, False, self._steps_taken >= self._max_steps, {}

        self._last_frame_data = frame_data
        self._available_actions = _build_action_strings(frame_data.available_actions)

        # Compute reward
        reward = 0.0
        level_delta = frame_data.levels_completed - self._prev_levels_completed
        if level_delta > 0:
            reward += 0.1 * level_delta
        if frame_data.state == GameState.WIN:
            reward += 1.0
        self._prev_levels_completed = frame_data.levels_completed

        # Termination
        terminated = frame_data.state in (GameState.WIN, GameState.GAME_OVER)
        truncated = (not terminated) and (self._steps_taken >= self._max_steps)

        obs = self._make_obs(frame_data)
        info = {
            "game_id": self.game_id,
            "levels_completed": frame_data.levels_completed,
            "win_levels": frame_data.win_levels,
            "state": frame_data.state.value,
        }

        return obs, reward, terminated, truncated, info

    def check_action_validity(self, candidate_action: str) -> str:
        """Validate an action string. Returns valid action or default."""
        ga, _ = _parse_action(candidate_action)
        if ga is not None and ga.name in self._available_actions:
            return candidate_action
        self.failed_candidates.append(candidate_action)
        return self.default_action

    def get_instruction_prompt(self, instructions=None) -> str:
        return get_arc_instruction_prompt(self._available_actions)

    def get_stats(self) -> dict:
        stats = {
            "game_id": self.game_id,
            "steps_taken": self._steps_taken,
            "scorecard_id": self._scorecard_id,
        }
        if self._last_frame_data:
            stats["levels_completed"] = self._last_frame_data.levels_completed
            stats["win_levels"] = self._last_frame_data.win_levels
            stats["final_state"] = self._last_frame_data.state.value
        return stats

    def close_scorecard(self) -> Optional[dict]:
        """Close the scorecard and return results.

        Matches the reference Swarm.close_scorecard pattern.
        """
        if self._scorecard_id is None:
            return None
        try:
            result = self._arcade.close_scorecard(self._scorecard_id)
            logger.info(f"Closed scorecard {self._scorecard_id}")
            card_id = self._scorecard_id
            self._scorecard_id = None
            if result is not None and hasattr(result, "model_dump"):
                return result.model_dump()
            return {"scorecard_id": card_id}
        except Exception as e:
            logger.warning(f"Could not close scorecard: {e}")
            return None

    def close(self):
        """Close the scorecard if one is open."""
        self.close_scorecard()


def make_arc_env(
    game_id: str,
    config,
    scorecard_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> ArcAgiEnvWrapper:
    """Factory function to create an ARC-AGI environment wrapper.

    Args:
        game_id: ARC-AGI game identifier (e.g. "ls20")
        config: Hydra config (uses config.envs.arc_agi_kwargs if present)
        scorecard_id: Optional existing scorecard ID to attach to
        tags: Optional tags for the scorecard (e.g. ["agent", "stepwise_eb"])

    Returns:
        ArcAgiEnvWrapper instance
    """
    kwargs: dict[str, Any] = {}

    arc_kwargs = getattr(getattr(config, "envs", None), "arc_agi_kwargs", None)
    if arc_kwargs is not None:
        kwargs["max_episode_steps"] = getattr(arc_kwargs, "max_episode_steps", 200)

    seed = getattr(getattr(config, "envs", None), "env_kwargs", None)
    if seed is not None:
        seed_val = getattr(seed, "seed", None)
        if seed_val is not None:
            kwargs["seed"] = seed_val

    if scorecard_id is not None:
        kwargs["scorecard_id"] = scorecard_id
    if tags is not None:
        kwargs["tags"] = tags

    return ArcAgiEnvWrapper(game_id=game_id, **kwargs)
