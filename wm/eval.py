from collections import defaultdict
import random
from pathlib import Path
from dataclasses import dataclass
import json
import logging
from datetime import datetime
import asyncio
from typing import List, Tuple

from jinja2 import Template
import numpy as np
import nle.dataset
import nle.dataset.db
from nle.env.tasks import NetHackChallenge
from nle.nethack import tty_render
import litellm

from utils import (
    build_llm_input,
    extract_llm_response_text,
    extract_xml_kv,
    validate_response_fields
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
    file_handler = logging.FileHandler(
        log_dir / f"wm_eval_{timestamp}.log",
        mode='a'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Global logger - will be initialized when WMEvaluator is created
logger = logging.getLogger("wm_eval")


ACTION_MAP = {
    a.value: i for i, a in enumerate(
        NetHackChallenge().actions
    )
}

WM_PROMPT_TEMPLATE = """We are playing a game where we control a character in a world.
We currently know the following about the world - 

{% for h in hypothesis_list %}
- {{ h }}
{% endfor %}

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
    print(f"Before scene -\n{before_screen}\n\nAction taken - {action}\n\nAfter Scene -\n{after_screen}")

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


@dataclass
class WMEvalConfig:
    model: str
    dataset_path: str | Path
    log_dir: str | Path


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
        logger.info(f"Config: model={config.model}, dataset_path={config.dataset_path}, log_dir={config.log_dir}")

        self.config = config
        self._dbfilename = "ttyrecs.db"

        logger.debug(f"Checking for database: {self._dbfilename}")
        if not nle.dataset.db.exists(self._dbfilename):
            logger.info(f"Database not found. Creating new database: {self._dbfilename}")
            nle.dataset.db.create(self._dbfilename)
            logger.info(f"Adding NLE data directory: {self.config.dataset_path}")
            nle.dataset.add_nledata_directory(self.config.dataset_path, "taster-dataset", self._dbfilename)
        else:
            logger.debug("Database already exists")

        logger.debug("Connecting to database")
        self._db_conn = nle.dataset.db.connect(filename=self._dbfilename)
        num_games = nle.dataset.db.count_games('taster-dataset', conn=self._db_conn)
        logger.info(f"NLD AA \"Taster\" Dataset has {num_games} games.")
        print(f"NLD AA \"Taster\" Dataset has {num_games} games.")

        logger.debug("Creating TtyrecDataset with batch_size=32, seq_length=2")
        self.dataset = nle.dataset.TtyrecDataset(
            "taster-dataset",
            batch_size=32,
            seq_length=2,
            dbfilename=self._dbfilename,
        )

        logger.info("Loading and processing dataset batches")
        _data = defaultdict(list)
        for i, batch in enumerate(self.dataset):
            if i == 2:
                logger.debug(f"Reached batch limit (16), stopping data loading")
                break

            logger.debug(f"Processing batch {i}")
            for k, v in batch.items():
                _data[k].extend(v)

        logger.debug("Mapping keypresses to action indices")
        _data["keypresses"] = [np.array([ACTION_MAP[kp] for kp in kpd]) for kpd in _data["keypresses"]]

        logger.debug("Inverting data structure to list of transition pairs")
        invert = lambda dict_data: [dict(zip(dict_data.keys(), vals)) for vals in zip(*dict_data.values())]
        _data = [invert(d) for d in invert(_data)]
        self.correct_data = _data
        logger.info(f"Loaded {len(self.correct_data)} correct transitions")

        # Create incorrect_data: same first timestep, different second timestep
        logger.info("Creating incorrect transitions by shuffling second timesteps")
        self.incorrect_data = []
        second_timesteps = [pair[1] for pair in self.correct_data]
        shuffled_indices = np.random.permutation(len(second_timesteps))
        for i, pair in enumerate(self.correct_data):
            incorrect_second = second_timesteps[shuffled_indices[i]]
            self.incorrect_data.append([pair[0], incorrect_second])
        logger.info(f"Created {len(self.incorrect_data)} incorrect transitions")

        logger.debug("Combining correct and incorrect data, then shuffling")
        self.data = [(i, 1) for i in self.correct_data] + [(i, 0) for i in self.incorrect_data]
        random.shuffle(self.data)
        logger.info(f"Total evaluation dataset size: {len(self.data)} (50% correct, 50% incorrect)")
        logger.info("WMEvaluator initialization complete")
        

    async def _process_single_transition(
        self,
        idx: int,
        transition,
        label: int,
        hset: list[str],
        subset_size: int | None = None
    ) -> Tuple[bool, int]:
        """
        Process a single transition asynchronously.

        Args:
            idx: Index of the transition
            transition: Transition data
            label: Ground truth label (1=correct, 0=incorrect)
            hset: List of hypothesis strings
            subset_size: If provided, sample this many hypotheses

        Returns:
            Tuple of (is_correct, idx)
        """
        logger.debug(f"Processing transition {idx + 1}")
        logger.debug(f"Ground truth label: {'correct' if label == 1 else 'incorrect'}")

        # Sample a subset of hypotheses for this iteration if subset_size is specified
        if subset_size is not None:
            current_hset = random.sample(hset, min(subset_size, len(hset)))
            logger.debug(f"Sampled {len(current_hset)} hypotheses for this iteration")
        else:
            current_hset = hset

        before_screen, after_screen = render_transition(transition)
        action_taken = int(transition[0]["keypresses"])
        logger.debug(f"Action taken: {action_taken}")

        logger.debug("Rendering prompt from template")
        prompt = Template(WM_PROMPT_TEMPLATE).render(
            hypothesis_list=current_hset,
            before_screen=before_screen,
            action_taken=action_taken,
            after_screen=after_screen,
        )
        logger.debug(f"Prompt length: {len(prompt)} characters")

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
            raise RuntimeError(f"Could not get answer from answer text -\n{answer_text}")

        is_correct = bool(label) == answer_bool

        if is_correct:
            logger.debug("✓ Model prediction matches ground truth")
        else:
            logger.debug("✗ Model prediction does NOT match ground truth")

        return is_correct, idx

    async def _eval_hypothesis_async(
        self,
        hset: list[str],
        subset_size: int | None = None,
        batch_size: int = 10
    ):
        """
        Evaluate a set of hypotheses on the transition dataset asynchronously.

        Args:
            hset: List of hypothesis strings to evaluate
            subset_size: If provided, sample this many hypotheses randomly at each iteration.
                        If None, use all hypotheses.
            batch_size: Number of concurrent API requests to process at once

        Returns:
            Accuracy score (float between 0 and 1)
        """
        logger.info("=" * 80)
        logger.info("Starting hypothesis evaluation (async with batching)")
        logger.info(f"Total number of hypotheses available: {len(hset)}")
        if subset_size is not None:
            logger.info(f"Will sample {subset_size} hypotheses per iteration")
        else:
            logger.info(f"Using all {len(hset)} hypotheses for each iteration")
        logger.info(f"Evaluating on {len(self.data)} transitions")
        logger.info(f"Batch size: {batch_size} concurrent requests")

        results = [None] * len(self.data)
        correct_count = 0
        incorrect_count = 0

        # Process data in batches
        for batch_start in range(0, len(self.data), batch_size):
            batch_end = min(batch_start + batch_size, len(self.data))
            batch = self.data[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start // batch_size + 1}: transitions {batch_start + 1}-{batch_end}")

            # Create tasks for the current batch
            tasks = [
                self._process_single_transition(
                    batch_start + i,
                    transition,
                    label,
                    hset,
                    subset_size
                )
                for i, (transition, label) in enumerate(batch)
            ]

            # Execute all tasks in the batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                    continue

                is_correct, idx = result
                results[idx] = is_correct

                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1

            # Log progress
            processed = batch_end
            current_accuracy = correct_count / processed
            logger.info(f"Progress: {processed}/{len(self.data)} | Current accuracy: {current_accuracy:.4f}")

        if None in results or len(results) == 0:
            logger.error("Some evaluations were not performed!")
            raise RuntimeError(f"Did not eval everything!")

        accuracy = sum(results) / len(results)
        logger.info("=" * 80)
        logger.info("Evaluation complete")
        logger.info(f"Total transitions evaluated: {len(results)}")
        logger.info(f"Correct predictions: {correct_count}")
        logger.info(f"Incorrect predictions: {incorrect_count}")
        logger.info(f"Final accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        logger.info("=" * 80)

        return accuracy

    def eval_hypothesis(
        self,
        hset: list[str],
        subset_size: int | None = None,
        batch_size: int = 10
    ):
        """
        Evaluate a set of hypotheses on the transition dataset.

        Args:
            hset: List of hypothesis strings to evaluate
            subset_size: If provided, sample this many hypotheses randomly at each iteration.
                        If None, use all hypotheses.
            batch_size: Number of concurrent API requests to process at once (default: 10)

        Returns:
            Accuracy score (float between 0 and 1)
        """
        return asyncio.run(self._eval_hypothesis_async(hset, subset_size, batch_size))
        


def experiment(
    model: str,
    hstar_path: str | Path,
    log_dir: str | Path,
    batch_size: int = 20,
):
    """
    Run experiment evaluating hypothesis subsets at different percentages.

    Args:
        model: LLM model identifier
        hstar_path: Path to file containing hypothesis set (one per line)
        log_dir: Directory for saving logs and results
        batch_size: Number of concurrent API requests to process at once (default: 10)
    """
    hstar_path = Path(hstar_path)
    log_dir = Path(log_dir)

    # Set up logger at the entry point
    global logger
    logger = setup_logger(log_dir)

    logger.info("*" * 80)
    logger.info("STARTING EXPERIMENT")
    logger.info(f"Model: {model}")
    logger.info(f"Hypothesis file: {hstar_path}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("*" * 80)

    logger.info("Creating WMEvaluator instance")
    e = WMEvaluator(
            WMEvalConfig(
                model=model,
                dataset_path="nle_data/nld-aa-taster/nle_data",
                log_dir=log_dir,
            )
    )

    logger.info(f"Reading hypothesis set from: {hstar_path}")
    hstar = hstar_path.read_text().splitlines()
    logger.info(f"Loaded {len(hstar)} hypotheses")

    results = {}
    percentages = list(range(10, 101, 20))
    logger.info(f"Will evaluate at percentages: {percentages}")

    for pct in percentages:
        subset_size = int(len(hstar) * pct / 100)
        logger.info("-" * 80)
        logger.info(f"EVALUATING {pct}% SUBSET")
        logger.info(f"Subset size: {subset_size} hypotheses (will be sampled differently per iteration)")

        print(f"Evaluating {pct}% subset (size={subset_size})...")
        result = e.eval_hypothesis(hstar, subset_size=subset_size, batch_size=batch_size)
        print(f"Result for {pct}%: {result:.4f}")

        results[pct] = result
        logger.info(f"Result for {pct}%: {result:.4f}")

    results_path = log_dir / "results.json"
    logger.info("-" * 80)
    logger.info(f"Saving results to: {results_path}")
    json.dump(results, open(results_path, "w"), indent=2)
    logger.info("Results saved successfully")

    logger.info("*" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("Final Results Summary:")
    for pct, acc in results.items():
        logger.info(f"  {pct}%: {acc:.4f}")
    logger.info("*" * 80)


def test():
    """
    Test function for debugging WMEvaluator initialization.
    """
    e = WMEvaluator(
            WMEvalConfig(
                model="",
                dataset_path="nle_data/nld-aa-taster/nle_data",
                log_dir="logs/",
            )
    )

    breakpoint()


if __name__ == '__main__':
    import fire

    logger.info("Starting eval.py script")
    logger.debug(f"Available commands: experiment, test")

    fire.Fire(
        {
            "experiment": experiment,
            "test": test,
        }
    )
