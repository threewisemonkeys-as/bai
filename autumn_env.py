"""AutumnBench environment wrapper compatible with BALROG's EnvWrapper interface.

Wraps MARAProtocol's concrete_envs (InteractiveEnvironment / CDSliderEnvironment /
MFP / Planning) and exposes the dict-style (obs, reward, terminated, truncated, info)
interface used by stepwise_eb_learn / arc_agi_env.
"""

import io
import json
import logging
import os
import re
from typing import Any, Optional

from PIL import Image

from generated.mara import mara_environment_pb2 as env_pb2
from generated.mara import mara_environment_service_pb2 as env_service_pb2
from python_examples.autumnbench.concrete_envs import (
    CDSliderEnvironment,
    InteractiveEnvironment,
)
from python_examples.autumnbench.env_utils import parse_grid, render_grid
from python_examples.autumnbench.environment_interfaces import (
    MARACompositeAutumnChangeDetectionServicer,
    MARACompositeAutumnPlanningServicer,
)
from python_examples.autumnbench.environment_interfaces_mfp import (
    MARACompositeAutumnMFPServicer,
)

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
Each observation is a 2D grid of color-name strings (rows x cols). At each step you choose one of the currently available actions listed in the auxiliary observation.

Interactive actions:
- left / right / up / down — move (meaning depends on the game's dynamics, which you need to discover)
- click ROW COL — click grid cell at (row, col), 0-indexed
- noop — wait one step
- reset — reset the interactive phase to the initial state when available
- quit — give up

Evaluation actions depend on the task:
- mfp — use step/rewind to inspect the trajectory, then choose_option_N to fill the masked final state.
- cd — interact with the changed environment, use I found the change! when behavior diverges from learned normal dynamics, then choose_frame_N and Submit choice.
- planning — use movement/click/noop actions to reach the provided goal state in the highlighted mask. Exploration-only actions are not available.

During exploration, your goal is to understand the underlying dynamics. During evaluation, use what you learned during exploration to maximize task reward.
"""


def _obs_to_balrog(
    obs_pb: env_pb2.Observation,
    step_count: int,
    task_type: str,
    text_grid: Optional[str] = None,
    phase: Optional[str] = None,
    action_strings: Optional[list[str]] = None,
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
    short_term_lines = [f"Task: {task_type}", f"Step: {step_count}"]
    if phase:
        short_term_lines.append(f"Phase: {phase}")
    if action_strings:
        short_term_lines.append("Available actions now:")
        short_term_lines.extend(f"- {action}" for action in action_strings)
    short_term = "\n".join(short_term_lines)

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
    the LLM can emit, and keeps the directional / noop / reset / quit
    entries as-is.
    """
    out = []
    for act in action_space:
        t = act.text_data
        if t == "go-to-test":
            continue
        if t.startswith("click ["):
            out.append(f"click ROW COL  (ROW, COL in 0..{grid_size - 1})")
        else:
            out.append(t)
    return out


class AutumnBenchEnvWrapper:
    """BALROG-compatible wrapper around a MARAProtocol autumnbench environment.

    ``task_type='interactive'`` exposes rule discovery only, matching what
    stepwise_eb_learn optimizes. ``mfp``, ``cd``, and ``planning`` wrap MARA's
    composite AutumnBench environments, which run interactive exploration and
    then transition into the corresponding scored evaluation phase.
    """

    def __init__(
        self,
        env_name: str,
        task_type: str = "interactive",
        max_episode_steps: int = 200,
        max_interaction_steps: int = 200,
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
        self._max_interaction_steps = max_interaction_steps
        self._is_composite = task_type in {"mfp", "cd", "planning"}
        self._env_stack_frames = stack_frames
        self._env_skip_frames = skip_frames
        self._data_dir = data_dir
        self._logging_path = logging_path
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
        elif task_type == "cd_test":
            self._env = CDSliderEnvironment(
                env_name=env_name,
                render_mode=self._inner_render_mode,
                stack_frames=stack_frames,
                skip_frames=skip_frames,
                logging_path=logging_path,
                data_dir=data_dir,
                seed=seed,
            )
        elif task_type == "mfp":
            self._env = MARACompositeAutumnMFPServicer()
        elif task_type == "cd":
            self._env = MARACompositeAutumnChangeDetectionServicer()
        elif task_type == "planning":
            self._env = MARACompositeAutumnPlanningServicer()
        else:
            raise ValueError(
                f"Unsupported task_type={task_type!r}. Supported: "
                "'interactive', 'mfp', 'cd', 'planning', 'cd_test'."
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
            acts = self._query_action_space()
        except Exception:
            return [
                "left",
                "right",
                "up",
                "down",
                "noop",
                "reset",
                "quit",
            ]
        return _build_action_strings(acts, self._grid_size)

    def _query_action_space(self) -> list[env_pb2.Action]:
        if self._is_composite:
            request = env_service_pb2.SpaceQueryRequest()
            request.reactive_query.SetInParent()
            response = self._env.QuerySpaces(request, None)
            return list(response.reactive_response.action_space.available_actions)
        return self._env.get_action_space()

    def _initialize_composite(self) -> None:
        init_req = env_service_pb2.InitializeRequest(
            env_type=env_pb2.REACTIVE,
            config={
                "env_name": self.env_name,
                "max_interaction_steps": str(self._max_interaction_steps),
                "stack_frames": str(self._env_stack_frames).lower(),
                "skip_frames": str(self._env_skip_frames).lower(),
                "render_mode": self._inner_render_mode,
                "logging_path": self._logging_path,
                "seed": str(self._seed),
                "data_dir": self._data_dir,
            },
        )
        self._env.Initialize(init_req, None)

    def _phase_name(self) -> str:
        if hasattr(self._env, "transiting_state"):
            return str(getattr(self._env, "transiting_state"))
        if hasattr(self._env, "transiting"):
            return str(getattr(self._env, "transiting"))
        if self.task_type == "interactive":
            return "Interactive"
        if self.task_type == "cd_test":
            return "Change"
        return self.task_type

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
            self._seed = seed
            if not self._is_composite and hasattr(self._env, "seed"):
                self._env.seed = seed
        if self._is_composite:
            self._initialize_composite()
            reset_response = self._env.Reset(env_service_pb2.ResetRequest(), None)
            obs_pb = reset_response.initial_observation
        else:
            self._env.reset()
            obs_pb = self._env.get_observation()
            try:
                _, self._grid_size = parse_grid(self._env.interpreter.render_all())
            except Exception:
                pass

        text_grid = (
            self._render_text_grid()
            if (not self._is_composite and self._inner_render_mode == "image")
            else None
        )
        obs = _obs_to_balrog(
            obs_pb,
            self._steps_taken,
            self.task_type,
            text_grid=text_grid,
            phase=self._phase_name(),
            action_strings=self.language_action_space,
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
        if first in self.language_action_space:
            return first
        head = first.split(" ", 1)[0].lower()
        lowered = first.lower()
        if head in {
            "left",
            "right",
            "up",
            "down",
            "noop",
            "quit",
            "reset",
            "step",
            "rewind",
        }:
            return head
        if head == "go-to-test":
            self.failed_candidates.append(action)
            return self.default_action
        if head.startswith("click"):
            # preserve full 'click X Y'
            return first
        if match := re.search(r"\bchoose[_ -]?option[_ -]?(\d+)\b", lowered):
            return f"choose_option_{match.group(1)}"
        if match := re.search(r"\bchoose[_ -]?frame[_ -]?(\d+)\b", lowered):
            return f"choose_frame_{match.group(1)}"
        if "submit choice" in lowered:
            return "Submit choice"
        if "found the change" in lowered:
            return "I found the change!"
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
        if self._is_composite:
            step_response = self._env.Step(
                env_service_pb2.StepRequest(action=env_pb2.Action(text_data=coerced)),
                None,
            )
            obs_pb = step_response.observation
            reward = step_response.reward
            terminated = step_response.is_terminal
            info = dict(step_response.info)
        else:
            obs_pb, reward, terminated, info = self._env.step(
                env_pb2.Action(text_data=coerced)
            )
        self._last_reward = float(reward) if reward is not None else 0.0
        self._last_info = dict(info) if info else {}

        truncated = (not terminated) and (self._steps_taken >= self._max_steps)
        text_grid = (
            self._render_text_grid()
            if (not self._is_composite and self._inner_render_mode == "image")
            else None
        )
        obs = _obs_to_balrog(
            obs_pb,
            self._steps_taken,
            self.task_type,
            text_grid=text_grid,
            phase=self._phase_name(),
            action_strings=self.language_action_space,
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
            "phase": self._phase_name(),
            "steps_taken": self._steps_taken,
            "last_reward": self._last_reward,
            "last_info": self._last_info,
        }

    def close(self):
        if self._is_composite:
            try:
                self._env.Close(env_service_pb2.CloseRequest(), None)
            except Exception:
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
            "max_interaction_steps",
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
