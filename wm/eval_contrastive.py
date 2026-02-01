import asyncio
import enum
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import litellm
import nle.dataset
import nle.dataset.db
import numpy as np
from jinja2 import Template
from nle.env.tasks import NetHackChallenge
from nle.nethack import tty_render
from nle.language_wrapper.wrappers.nle_language_wrapper import NLELanguageWrapper
from utils import (
    build_llm_input,
    extract_llm_response_text,
    extract_xml_kv,
    validate_response_fields,
)


def setup_logger(log_dir: Path, name: str = "wm_eval") -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Args:
        log_dir: Directory to save log files
        name: Logger name

    Returns:
        Configured logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler for detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"wm_eval_{timestamp}.log", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Global logger - will be initialized when WMEvaluator is created
logger = logging.getLogger("wm_eval")



STATE_PRED_PROMPT_TEMPLATE = """We are playing a game where we control a character in a world.
We currently know the following about the world -

{{ h }}

Given that we know this, our task is to determine whether the transition presented below is correct.
You will be shown the game screen before an action is taken, the action that was taken and a possible game screen after the action is taken.
Your task is to determine whether the after screen displays the correct consequence of taking the given action.

Before screen -

{{ before_screen }}

Action taken - {{ action_taken }}

After screen -

{{ after_screen }}

Determine whether the after screen is the correct result of taking the given action on the before screen.
Present your judgement at the end of your answer in an xml format -
<answer>
Yes/No
</answer>
"""


ACTION_PRED_PROMPT_TEMPLATE = """We are playing a game where we control a character in a world.
We currently know the following about the world -

{{ h }}

Given that we know this, our task is to determine whether the transition presented below is correct.
You will be shown the game screen before an action is taken, the action that was taken and a possible game screen after the action is taken.
Your task is to determine whether the action presented is the correct one that will result in the before screen transitioning to the after screen.

Before screen -

{{ before_screen }}

Action taken - {{ action_taken }}

After screen -

{{ after_screen }}

Determine whether the action taken is the correct one that results in the before screen transitioning to the after screen..
Present your judgement at the end of your answer in an xml format -
<answer>
Yes/No
</answer>
"""


def print_before_after(before, after):
    """Print before and after game screens with action taken."""
    logger.debug("Printing before/after transition")
    action = int(before["keypresses"][0])

    before_screen = tty_render(
        before["tty_chars"], before["tty_colors"], before["tty_cursor"]
    )
    after_screen = tty_render(
        after["tty_chars"], after["tty_colors"], after["tty_cursor"]
    )
    logger.debug(f"Action taken: {action}")
    print(
        f"Before scene -\n{before_screen}\n\nAction taken - {action}\n\nAfter Scene -\n{after_screen}"
    )


def render_transition(transition):
    """
    Render before and after screens from a transition.

    Args:
        transition: Tuple of (before, after) game states

    Returns:
        Tuple of (before_screen, after_screen) as strings
    """
    logger.debug("Rendering transition")
    before, after = transition
    before_screen = tty_render(
        before["tty_chars"], before["tty_colors"], before["tty_cursor"]
    )
    after_screen = tty_render(
        after["tty_chars"], after["tty_colors"], after["tty_cursor"]
    )
    logger.debug("Transition rendered successfully")
    return before_screen, after_screen


NetHackKPtoActions = {
    a.value: a for a in NetHackChallenge().actions
}


class EvalMode(enum.Enum):
    STATE = "state"
    ACTION = "action"


@dataclass
class WMEvalConfig:
    model: str
    dataset_path: str | Path
    log_dir: str | Path
    eval_mode: EvalMode
    num_instances: int | None = None


class WMEvaluator:
    """
    Evaluator for world model hypothesis testing on NetHack transitions.
    """

    def __init__(
        self,
        config: WMEvalConfig,
    ):
        """
        Initialize the evaluator with dataset and configuration.

        Args:
            config: Configuration for evaluation
        """
        logger.info("Initializing WMEvaluator")
        logger.info(
            f"Config: model={config.model}, dataset_path={config.dataset_path}, log_dir={config.log_dir}"
        )

        self.config = config
        self._dbfilename = "ttyrecs.db"

        logger.debug(f"Checking for database: {self._dbfilename}")
        if not nle.dataset.db.exists(self._dbfilename):
            logger.info(
                f"Database not found. Creating new database: {self._dbfilename}"
            )
            nle.dataset.db.create(self._dbfilename)
            logger.info(f"Adding NLE data directory: {self.config.dataset_path}")
            nle.dataset.add_nledata_directory(
                self.config.dataset_path, "taster-dataset", self._dbfilename
            )
        else:
            logger.debug("Database already exists")

        logger.debug("Connecting to database")
        self._db_conn = nle.dataset.db.connect(filename=self._dbfilename)
        num_games = nle.dataset.db.count_games("taster-dataset", conn=self._db_conn)
        logger.info(f'NLD AA "Taster" Dataset has {num_games} games.')
        print(f'NLD AA "Taster" Dataset has {num_games} games.')

        logger.debug("Creating TtyrecDataset with batch_size=1, seq_length=2")
        self.dataset = nle.dataset.TtyrecDataset(
            "taster-dataset",
            batch_size=1,
            seq_length=2,
            dbfilename=self._dbfilename,
        )

        logger.info("Loading and dataset")
        _data = defaultdict(list)
        for i, batch in enumerate(self.dataset):
            if config.num_instances is not None and i == config.num_instances :
                logger.debug(f"Reached limit of num instance {config.num_instances}, stopping data loading")
                break

            for k, v in batch.items():
                _data[k].extend(v)

        logger.debug("Mapping keypresses to action indices")
        _data["keypresses"] = [
            [NLELanguageWrapper.all_nle_action_map[NetHackKPtoActions[kp]][0] for kp in kpd] for kpd in _data["keypresses"]
        ]

        logger.debug("Inverting data structure to list of transition pairs")
        invert = lambda dict_data: [
            dict(zip(dict_data.keys(), vals)) for vals in zip(*dict_data.values())
        ]
        _data = [invert(d) for d in invert(_data)]
        self.correct_data = _data
        logger.info(f"Loaded {len(self.correct_data)} correct transitions")

        # Create incorrect_data

        if config.eval_mode == EvalMode.STATE:
            self.wm_prompt_template = STATE_PRED_PROMPT_TEMPLATE
            # same first timestep, different second timestep
            logger.info("Creating incorrect transitions by shuffling second timesteps")
            self.incorrect_data = []
            second_timesteps = [pair[1] for pair in self.correct_data]
            shuffled_indices = np.random.permutation(len(second_timesteps))
            for i, pair in enumerate(self.correct_data):
                incorrect_second = second_timesteps[shuffled_indices[i]]
                self.incorrect_data.append([pair[0], incorrect_second])
        elif config.eval_mode == EvalMode.ACTION:
            self.wm_prompt_template = ACTION_PRED_PROMPT_TEMPLATE
            # same timesteps different action
            logger.info("Creating incorrect transitions by shuffling actions")
            self.incorrect_data = []
            for pair in self.correct_data:
                # Create a copy of the transition with a different action
                incorrect_pair = [pair[0].copy(), pair[1].copy()]
                # Get current action and choose a different one
                current_action = incorrect_pair[0]["keypresses"]
                available_actions = [
                    a[0] for a in NLELanguageWrapper.all_nle_action_map.values() if a[0] != current_action
                ]
                new_action = str(np.random.choice(available_actions))
                incorrect_pair[0]["keypresses"] = new_action
                self.incorrect_data.append(incorrect_pair)

        logger.info(f"Created {len(self.incorrect_data)} incorrect transitions")
        logger.debug("Combining correct and incorrect data, then shuffling")
        self.data = [(i, 1) for i in self.correct_data] + [
            (i, 0) for i in self.incorrect_data
        ]
        random.shuffle(self.data)
        logger.info(
            f"Total evaluation dataset size: {len(self.data)} (50% correct, 50% incorrect)"
        )
        logger.info("WMEvaluator initialization complete")

    async def _process_single_transition(
        self,
        idx: int,
        transition,
        label: int,
        h: str,
    ) -> tuple[bool, int, dict]:
        """
        Process a single transition asynchronously.

        Args:
            idx: Index of the transition
            transition: Transition data
            label: Ground truth label (1=correct, 0=incorrect)
            h: Hypothesis to evaluate
            return_details: If True, return detailed result dict as third element

        Returns:
            Tuple of (is_correct, idx, details_dict)
        """
        logger.debug(f"Processing transition {idx + 1}")
        logger.debug(f"Ground truth label: {'correct' if label == 1 else 'incorrect'}")


        before_screen, after_screen = render_transition(transition)
        action_taken = transition[0]["keypresses"]
        logger.debug(f"Action taken: {action_taken}")

        logger.debug("Rendering prompt from template")
        prompt = Template(self.wm_prompt_template).render(
            h=h,
            before_screen=before_screen,
            action_taken=action_taken,
            after_screen=after_screen,
        )
        logger.debug(f"Prompt length: {len(prompt)} characters")
        # logger.info(f"Prompt -\n{prompt}")

        logger.debug(f"Building LLM input for model: {self.config.model}")
        input_data = build_llm_input(prompt)

        logger.info(f"Calling LLM for transition {idx + 1}")
        try:
            # Run the blocking litellm.responses call in a thread pool
            response = await asyncio.to_thread(
                litellm.responses,
                model=self.config.model,
                input=input_data,
                num_retries=5,
            )
            logger.debug("LLM response received successfully")
        except Exception as e:
            logger.error(f"LLM call failed for transition {idx + 1}: {e}")
            raise

        response_output_text = extract_llm_response_text(response)
        logger.debug(f"Response text length: {len(response_output_text)} characters")
        logger.debug(f"Response preview: {response_output_text[:200]}...")

        logger.debug("Extracting XML answer field")
        response_dict = extract_xml_kv(response_output_text, ["answer"])
        validate_response_fields(response_dict, response_output_text, ["answer"])

        answer_text = response_dict["answer"].strip().lower()
        logger.debug(f"Extracted answer text: '{answer_text}'")

        if "yes" in answer_text:
            answer_bool = True
            logger.debug("Parsed answer as: YES (transition is correct)")
        elif "no" in answer_text:
            answer_bool = False
            logger.debug("Parsed answer as: NO (transition is incorrect)")
        else:
            logger.error(f"Could not parse answer from text: '{answer_text}'")
            raise RuntimeError(
                f"Could not get answer from answer text -\n{answer_text}"
            )

        is_correct = bool(label) == answer_bool

        if is_correct:
            logger.debug("✓ Model prediction matches ground truth")
        else:
            logger.debug("✗ Model prediction does NOT match ground truth")

        details = {
            "is_correct": is_correct,
            "before_screen": before_screen,
            "action_taken": str(action_taken),
            "after_screen": after_screen,
            "ground_truth": "correct" if label == 1 else "incorrect",
            "predicted": "correct" if answer_bool else "incorrect",
        }
        return is_correct, idx, details


    async def _eval_hypothesis_async(
        self, h: str, batch_size: int = 1,
    ):
        """
        Evaluate a set of hypotheses on the transition dataset asynchronously.

        Args:
            h: Hypothesis to evaluate
            batch_size: Number of concurrent API requests to process at once

        Returns:
            Tuple of (accuracy, detailed_results)
        """
        logger.info("=" * 80)
        logger.info("Starting hypothesis evaluation (async with batching)")
        logger.info(f"Evaluating on {len(self.data)} transitions")
        logger.info(f"Batch size: {batch_size} concurrent requests")

        results: list = [None] * len(self.data)
        detailed_results = []
        correct_count = 0
        incorrect_count = 0

        # Process data in batches
        for batch_start in range(0, len(self.data), batch_size):
            batch_end = min(batch_start + batch_size, len(self.data))
            batch = self.data[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start // batch_size + 1}: transitions {batch_start + 1}-{batch_end}"
            )

            # Create tasks for the current batch
            tasks = [
                self._process_single_transition(
                    batch_start + i, transition, label, h
                )
                for i, (transition, label) in enumerate(batch)
            ]

            # Execute all tasks in the batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in batch_results:
                if isinstance(result, BaseException):
                    logger.error(f"Task failed with exception: {result}")
                    continue
                    
                is_correct, idx, details = result
                detailed_results.append(details)

                results[idx] = is_correct

                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1

            # Log progress
            processed = batch_end
            current_accuracy = correct_count / processed
            logger.info(
                f"Progress: {processed}/{len(self.data)} | Current accuracy: {current_accuracy:.4f}"
            )

        if len(results) == 0:
            raise RuntimeError("Did not eval everything!")

        if None in results:
            logger.warning(
                f" {len([r for r in results if r is None])} / {len(results)} results were None"
            )
            results = [r for r in results if r is not None]

        accuracy = sum(results) / len(results)
        logger.info("=" * 80)
        logger.info("Evaluation complete")
        logger.info(f"Total transitions evaluated: {len(results)}")
        logger.info(f"Correct predictions: {correct_count}")
        logger.info(f"Incorrect predictions: {incorrect_count}")
        logger.info(f"Final accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        logger.info("=" * 80)

        return accuracy, detailed_results

    def eval_hypothesis(
        self, h: str, batch_size: int = 1
    ):
        """
        Evaluate a set of hypotheses on the transition dataset.

        Args:
            h: Hypothesis to evaluate
            batch_size: Number of concurrent API requests to process at once (default: 1)

        Returns:
            Tuple of (accuracy, detailed_results)
        """
        return asyncio.run(self._eval_hypothesis_async(h, batch_size))




def test():
    """
    Test function for debugging WMEvaluator initialization.
    """
    e = WMEvaluator(
        WMEvalConfig(
            model="",
            dataset_path="nle_data/nld-aa-taster/nle_data",
            log_dir="logs/",
            eval_mode=EvalMode.ACTION,
            num_instances=1
        )
    )

    breakpoint()


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "test": test,
        }
    )
