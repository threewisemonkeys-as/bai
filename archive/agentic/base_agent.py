from dataclasses import dataclass
from pathlib import Path
import json
import logging

from dotenv import load_dotenv
import litellm
import gymnasium as gym
import minihack
from PIL import Image
import numpy as np
from nle import nethack

from utils import (
    crop_central_component,
    image_to_base64,
    extract_xml_kv,
)

load_dotenv()


ACTIONS = tuple(
    list(nethack.CompassCardinalDirection)
    + [
        nethack.Command.PICKUP,
        nethack.Command.QUAFF,
        nethack.Command.ZAP,
        nethack.Command.EAT,
        nethack.Command.PICKUP,
        nethack.Command.PRAY,
        nethack.Command.WEAR,
        nethack.Command.KICK,
    ]
)


@dataclass
class AgentConfig:
    env_name: str
    max_iter: int

    log_dir: Path | None = None

    model: str = "openai/gpt-5"


class BaseAgent:
    """Base class for all agents with common functionality."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._setup_logger()
        self._setup_environment()

    def _setup_logger(self):
        """Initialize logger with optional file handler."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if self.config.log_dir is not None:
            self.config.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.config.log_dir / "runner.log"
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging initialized. Log file: {log_file}")

    def _setup_environment(self):
        """Initialize the Gym environment."""
        self.env = gym.make(
            self.config.env_name,
            observation_keys=("glyphs", "chars", "colors", "pixel"),
            actions=ACTIONS,
        )

    def _extract_response_text(self, response) -> str:
        """Extract text from LLM response with error handling.

        Args:
            response: LiteLLM response object

        Returns:
            Extracted response text

        Raises:
            RuntimeError: If response format is invalid
        """
        try:
            response_output_text = response.output[-1].content[0].text
            self.logger.info(f"Response text: {response_output_text}")
            return response_output_text
        except AttributeError as e:
            error_msg = f"Error in response-\n {response}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _parse_experiment(self, response_dict: dict, response_output_text: str) -> list[int]:
        """Parse and validate experiment actions from response.

        Args:
            response_dict: Parsed XML response as dictionary
            response_output_text: Original response text for error messages

        Returns:
            List of action integers

        Raises:
            RuntimeError: If experiment is missing or malformed
        """
        if "experiment" not in response_dict:
            error_msg = f"Experiment missing from response -\n{response_output_text}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            experiment_actions = json.loads(response_dict["experiment"])
        except json.JSONDecodeError as e:
            error_msg = f"Experiment not formatted correctly in response -\n{response_output_text}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        self.logger.info(f"Parsed experiment actions: {experiment_actions}")
        return experiment_actions

    def _validate_response_fields(self, response_dict: dict, response_output_text: str, required_fields: list[str]):
        """Validate that required fields exist in response.

        Args:
            response_dict: Parsed XML response as dictionary
            response_output_text: Original response text for error messages
            required_fields: List of required field names

        Raises:
            RuntimeError: If any required field is missing
        """
        for field in required_fields:
            if field not in response_dict:
                error_msg = f"{field.capitalize()} missing from response -\n{response_output_text}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

    def _build_input_with_images(self, prompt: str, images: list[np.ndarray]) -> list[dict]:
        """Build LLM input with text prompt and images.

        Args:
            prompt: Text prompt for the LLM
            images: List of image arrays to include

        Returns:
            Formatted input list for LiteLLM
        """
        content = [
            {
                "type": "input_text",
                "text": prompt,
            }
        ]

        for image in images:
            img_b64 = image_to_base64(image)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img_b64}"
            })

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def _call_llm(self, input: list[dict], num_retries: int = 2, previous_response_id: str | None = None):
        """Call LiteLLM API with retry logic.

        Args:
            input: Formatted input for LiteLLM
            num_retries: Number of retry attempts
            previous_response_id: Optional ID for conversation continuity

        Returns:
            LiteLLM response object
        """
        kwargs = {
            "model": self.config.model,
            "input": input,
            "num_retries": num_retries,
        }

        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id

        response = litellm.responses(**kwargs)
        self.logger.info(f"Model response received")
        return response

    def _execute_actions(self, exp_actions: list[int], iter_idx: int) -> tuple[dict, bool, list[int]]:
        """Execute a sequence of actions in the environment.

        Args:
            exp_actions: List of actions to execute
            iter_idx: Current iteration index for logging

        Returns:
            Tuple of (final_observation, done_flag, taken_actions)
        """
        if len(exp_actions) == 0:
            raise RuntimeError(f"Cannot execute empty sequence of actions")

        taken_actions = []
        done = False

        for step, a in enumerate(exp_actions):
            obs, reward, done, truncated, info = self.env.step(a)
            self.logger.info(f"Iter {iter_idx}, Step {step}: took action {a}, got reward: {reward}, done: {done}, info: {info}")
            taken_actions.append(a)
            if done:
                break

        return obs, done, taken_actions # pyright: ignore[reportPossiblyUnboundVariable]

    def _save_image(self, image: np.ndarray, iter_idx: int, stage: str):
        """Save an image if logging is enabled.

        Args:
            image: Image array to save
            iter_idx: Current iteration index
            stage: Stage name (e.g., "start" or "end")
        """
        if self.config.log_dir is not None:
            img_path = self.config.log_dir / f"iter_{iter_idx}_{stage}.png"
            Image.fromarray(image).save(img_path)
            self.logger.info(f"Saved iter {iter_idx} {stage} image to {img_path}")

    def process_image_obs(self, image_obs: np.ndarray) -> np.ndarray:
        """Process raw image observations.

        Args:
            image_obs: Raw image observation array

        Returns:
            Processed image observation
        """
        image_obs = crop_central_component(image_obs)
        return image_obs

    def run(self, history_id: str | None):
        """Run the agent. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")