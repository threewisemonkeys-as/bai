"""AutumnBench environment wrapper compatible with BALROG's EnvWrapper interface.

Wraps MARAProtocol's concrete_envs (InteractiveEnvironment / CDSliderEnvironment /
MFP / Planning) and exposes the dict-style (obs, reward, terminated, truncated, info)
interface used by stepwise_eb_learn / arc_agi_env.
"""

import io
import json
import logging
import os
from typing import Any, Optional

from PIL import Image

from generated.mara import mara_environment_pb2 as env_pb2
from python_examples.autumnbench.concrete_envs import (
    CDSliderEnvironment,
    InteractiveEnvironment,
)
from python_examples.autumnbench.env_utils import parse_grid, render_grid

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(
        __import__(
            "python_examples.autumnbench.concrete_envs", fromlist=["x"]
        ).__file__
    ),
    "example_benchmark",
)


INSTRUCTION_PROMPT = """You are interacting with a new environment.
Each observation is a 2D grid of color-name strings (rows x cols). At each step you choose an action.

Actions:
- left / right / up / down — move (meaning depends on the game's dynamics, which you need to discover)
- click ROW COL — click grid cell at (row, col), 0-indexed
- noop — wait one step
- go-to-test — end the interactive phase (only used when you are confident you understand the dynamics)
- quit — give up

Your goal is to interact with the environment to understand its underlying dynamics rules.
"""


def _obs_to_balrog(
    obs_pb: env_pb2.Observation,
    step_count: int,
    task_type: str,
    text_grid: Optional[str] = None,
) -> dict:
    """Convert protobuf Observation into BALROG's dict-shape observation.

    The grid is JSON-encoded as a 2D list of color-name strings. In text mode
    the inner env packs that JSON into ``obs_pb.text_data``. In image mode the
    inner env only sets an optional instruction header on ``text_data`` and
    puts PNG bytes in ``image_data`` — in that case the caller passes a
    pre-rendered ``text_grid`` so both modalities reach the prompt.
    """
    header = obs_pb.text_data or ""
    if text_grid is not None:
        long_term = header + text_grid if header else text_grid
    else:
        long_term = header
    short_term = f"Task: {task_type}\nStep: {step_count}"

    image = None
    if obs_pb.image_data:
        try:
            image = Image.open(io.BytesIO(obs_pb.image_data)).convert("RGB")
        except Exception:
            image = None

    return {
        "text": {
            "long_term_context": long_term,
            "short_term_context": short_term,
        },
        "image": image,
    }


def _build_action_strings(action_space: list, grid_size: int) -> list[str]:
    """Flatten MARAProtocol's Action list into LLM-facing action strings.

    Replaces the templated 'click [0-N] [0-N]' with the literal instructions
    the LLM can emit, and keeps the directional / noop / go-to-test / quit
    entries as-is.
    """
    out = []
    for act in action_space:
        t = act.text_data
        if t.startswith("click ["):
            out.append(f"click ROW COL  (ROW, COL in 0..{grid_size - 1})")
        else:
            out.append(t)
    return out


class AutumnBenchEnvWrapper:
    """BALROG-compatible wrapper around a MARAProtocol autumnbench environment.

    Phase 1 (task_type='interactive'): rule-discovery phase only. Matches what
    stepwise_eb_learn optimizes (beliefs + perception over environment dynamics).
    CD-slider / MFP / Planning can be added later as separate phases.
    """

    def __init__(
        self,
        env_name: str,
        task_type: str = "interactive",
        max_episode_steps: int = 200,
        seed: int = 0,
        stack_frames: bool = False,
        skip_frames: bool = False,
        render_mode: str = "text",
        data_dir: str = _DEFAULT_DATA_DIR,
        logging_path: str = "./logs/autumn",
    ):
        self.env_name = env_name
        self.task_type = task_type
        self._seed = seed
        self._max_steps = max_episode_steps
        # Wrapper-level render_mode semantics:
        #   "text"  — text only; inner env in text mode; no image attached.
        #   "image" — dual mode (text + image); inner env in image mode (so
        #             PNG bytes are populated) and we synthesize the text grid
        #             ourselves so both modalities reach the prompt.
        self._render_mode = render_mode
        self._inner_render_mode = "image" if render_mode == "image" else "text"

        os.makedirs(logging_path, exist_ok=True)

        if task_type == "interactive":
            self._env = InteractiveEnvironment(
                env_name=env_name,
                stack_frames=stack_frames,
                skip_frames=skip_frames,
                render_mode=self._inner_render_mode,
                logging_path=logging_path,
                data_dir=data_dir,
                seed=seed,
            )
        elif task_type == "cd":
            self._env = CDSliderEnvironment(
                env_name=env_name,
                render_mode=self._inner_render_mode,
                stack_frames=stack_frames,
                skip_frames=skip_frames,
                logging_path=logging_path,
                data_dir=data_dir,
                seed=seed,
            )
        else:
            raise ValueError(
                f"Unsupported task_type={task_type!r}. Supported: 'interactive', 'cd'."
            )

        self._steps_taken = 0
        self._grid_size = 16
        self._last_reward = 0.0
        self._last_info: dict = {}
        self.failed_candidates: list[str] = []

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def _render_text_grid(self) -> Optional[str]:
        """JSON-encode the current grid as a 2D color-name matrix.

        Mirrors what concrete_envs.InteractiveEnvironment.get_observation
        embeds in text_data in text mode, so we can attach it ourselves when
        the inner env is running in image mode.
        """
        try:
            render_str = self._env.interpreter.render_all()
            render_dict = json.loads(render_str)
            self._grid_size = render_dict.get("GRID_SIZE", self._grid_size)
            matrix = render_grid(
                render_dict,
                background_color=self._env.interpreter.get_background(),
                color_dict=self._env.color_dict_str_to_int,
            )
            return json.dumps(matrix)
        except Exception as e:
            logger.warning(f"Failed to synthesize text grid: {e}")
            return None

    @property
    def language_action_space(self) -> list[str]:
        try:
            acts = self._env.get_action_space()
        except Exception:
            return ["left", "right", "up", "down", "noop", "go-to-test", "quit"]
        return _build_action_strings(acts, self._grid_size)

    @property
    def default_action(self) -> str:
        return "noop"

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> tuple[dict, dict]:
        self._steps_taken = 0
        self._last_reward = 0.0
        self.failed_candidates = []
        if seed is not None:
            self._env.seed = seed
            self._seed = seed
        self._env.reset()

        obs_pb = self._env.get_observation()
        try:
            _, self._grid_size = parse_grid(self._env.interpreter.render_all())
        except Exception:
            pass

        text_grid = (
            self._render_text_grid() if self._inner_render_mode == "image" else None
        )
        obs = _obs_to_balrog(
            obs_pb, self._steps_taken, self.task_type, text_grid=text_grid
        )
        info = {"env_name": self.env_name, "task_type": self.task_type}
        return obs, info

    def _coerce_action(self, action: str) -> str:
        """Clip to the first word set the interpreter actually accepts.

        The LLM may emit 'click 3 4' or 'click ROW COL' — pass first-line,
        literal strings through. Unknown strings become noop (with a failed
        candidate record).
        """
        first = action.strip().split("\n")[0].strip()
        head = first.split(" ", 1)[0].lower()
        if head in {"left", "right", "up", "down", "noop", "quit", "reset"}:
            return head
        if head == "go-to-test":
            return "go-to-test"
        if head.startswith("click"):
            # preserve full 'click X Y'
            return first
        if head in {"fault!", "i"}:
            # "I found the change!" / "Fault!"  (CD state transitions)
            return first
        # Unknown — record and fall back
        self.failed_candidates.append(action)
        return self.default_action

    def step(
        self, action: str
    ) -> tuple[dict, float, bool, bool, dict]:
        self._steps_taken += 1
        coerced = self._coerce_action(action)
        obs_pb, reward, terminated, info = self._env.step(
            env_pb2.Action(text_data=coerced)
        )
        self._last_reward = float(reward) if reward is not None else 0.0
        self._last_info = dict(info) if info else {}

        truncated = (not terminated) and (self._steps_taken >= self._max_steps)
        text_grid = (
            self._render_text_grid() if self._inner_render_mode == "image" else None
        )
        obs = _obs_to_balrog(
            obs_pb, self._steps_taken, self.task_type, text_grid=text_grid
        )
        return obs, self._last_reward, bool(terminated), truncated, self._last_info

    def check_action_validity(self, candidate_action: str) -> str:
        coerced = self._coerce_action(candidate_action)
        return coerced

    def get_instruction_prompt(self, instructions=None) -> str:
        return INSTRUCTION_PROMPT

    def get_stats(self) -> dict:
        return {
            "env_name": self.env_name,
            "task_type": self.task_type,
            "steps_taken": self._steps_taken,
            "last_reward": self._last_reward,
            "last_info": self._last_info,
        }

    def close(self):
        pass


def make_autumn_env(
    env_name: str,
    config,
) -> AutumnBenchEnvWrapper:
    kwargs: dict[str, Any] = {}
    autumn_kwargs = getattr(getattr(config, "envs", None), "autumn_kwargs", None)
    if autumn_kwargs is not None:
        for k in (
            "task_type",
            "max_episode_steps",
            "stack_frames",
            "skip_frames",
            "render_mode",
            "data_dir",
            "logging_path",
        ):
            v = getattr(autumn_kwargs, k, None)
            if v is not None:
                kwargs[k] = v

    seed_cfg = getattr(getattr(config, "envs", None), "env_kwargs", None)
    if seed_cfg is not None:
        seed_val = getattr(seed_cfg, "seed", None)
        if seed_val is not None:
            kwargs["seed"] = seed_val

    return AutumnBenchEnvWrapper(env_name=env_name, **kwargs)
