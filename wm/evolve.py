import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import litellm
from utils import (
    build_llm_input,
    extract_llm_response_text,
    extract_xml_kv,
    validate_response_fields,
)


def setup_logger(log_dir: Path, name: str = "wm_evolve") -> logging.Logger:
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
    file_handler = logging.FileHandler(log_dir / f"wm_evolve_{timestamp}.log", mode="a")
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


# Global logger - will be initialized in main
logger = logging.getLogger("wm_evolve")


def improve(
    model: str,
    beliefs: str,
    accuracy: float,
    detailed_results: list[dict],
    num_examples: int = 10,
) -> str:
    """Improve beliefs based on evaluation results.

    Args:
        model: LLM model identifier
        beliefs: Current set of beliefs as a string
        accuracy: Accuracy score from evaluation (0.0 to 1.0)
        detailed_results: List of detailed evaluation results for each transition
        num_examples: Number of example transitions to include in prompt

    Returns:
        Updated beliefs as a string
    """
    logger.info("=" * 80)
    logger.info("Starting belief improvement")
    logger.info(f"Current accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    logger.info(f"Number of current beliefs: {len(beliefs.splitlines()) if beliefs else 0}")

    # Separate correct and incorrect predictions
    correct_results = [r for r in detailed_results if r["is_correct"]]
    incorrect_results = [r for r in detailed_results if not r["is_correct"]]

    logger.info(f"Correct predictions: {len(correct_results)}")
    logger.info(f"Incorrect predictions: {len(incorrect_results)}")

    # Sample examples (prioritize incorrect ones)
    import random
    num_incorrect = min(num_examples // 2 + num_examples % 2, len(incorrect_results))
    num_correct = min(num_examples - num_incorrect, len(correct_results))

    # Adjust if we don't have enough incorrect examples
    if num_incorrect < num_examples // 2 and len(correct_results) > num_correct:
        num_correct = min(num_examples - num_incorrect, len(correct_results))

    sampled_incorrect = random.sample(incorrect_results, num_incorrect) if incorrect_results else []
    sampled_correct = random.sample(correct_results, num_correct) if correct_results else []

    # Build examples string
    examples_str = ""

    if sampled_incorrect:
        examples_str += "Examples where we made INCORRECT predictions:\n\n"
        for i, result in enumerate(sampled_incorrect, 1):
            examples_str += f"Example {i} (WRONG):\n"
            examples_str += f"Before screen:\n{result['before_screen']}\n\n"
            examples_str += f"Action taken: {result['action_taken']}\n\n"
            examples_str += f"After screen:\n{result['after_screen']}\n\n"
            examples_str += f"Ground truth: This transition is {result['ground_truth']}\n"
            examples_str += f"We predicted: This transition is {result['predicted']}\n"
            examples_str += "-" * 40 + "\n\n"

    if sampled_correct:
        examples_str += "Examples where we made CORRECT predictions:\n\n"
        for i, result in enumerate(sampled_correct, 1):
            examples_str += f"Example {i} (CORRECT):\n"
            examples_str += f"Before screen:\n{result['before_screen']}\n\n"
            examples_str += f"Action taken: {result['action_taken']}\n\n"
            examples_str += f"After screen:\n{result['after_screen']}\n\n"
            examples_str += f"Ground truth: This transition is {result['ground_truth']}\n"
            examples_str += f"We predicted: This transition is {result['predicted']}\n"
            examples_str += "-" * 40 + "\n\n"

    logger.debug(f"Sampled {len(sampled_incorrect)} incorrect and {len(sampled_correct)} correct examples")


    prompt = f"""We are playing a game (NetHack) and trying to learn how the game world works.
Currently we have the following set of beliefs about the game world -

{beliefs if beliefs else '(empty)'}

We evaluated these beliefs on a dataset of game transitions and achieved an accuracy of {accuracy:.4f} ({accuracy * 100:.2f}%).

The evaluation works by showing the model a game state, an action taken, and a resulting game state,
then asking whether the transition is correct based on the beliefs.

Here are some specific examples from our evaluation:

{examples_str}

Think about how we need to update our beliefs to improve accuracy. Consider:
- Which beliefs might be incorrect or incomplete based on the examples above?
- What patterns do you see in the incorrect predictions?
- What new beliefs might help better predict transitions?
- How can we make beliefs more specific and actionable?

Format your response in XML style as -
<think>
Think about what might be wrong with current beliefs and how to improve them based on the examples.
</think>
<beliefs>
- updated belief
- updated belief
...
</beliefs>"""


    logger.debug(f"Prompt length: {len(prompt)} characters")

    # Build input for LLM
    input_data = build_llm_input(prompt)

    # Call LLM
    logger.info(f"Calling LLM ({model}) to generate updated beliefs with prompt -\n{prompt}")
    response = litellm.responses(
        model=model,
        input=input_data,
        num_retries=5,
    )

    # Extract response text
    response_text = extract_llm_response_text(response)
    logger.debug(f"Response text length: {len(response_text)} characters")

    # Extract beliefs from XML
    response_dict = extract_xml_kv(response_text, ["beliefs", "think"])
    validate_response_fields(response_dict, response_text, ["beliefs"])

    updated_beliefs = response_dict["beliefs"].strip()

    logger.info("Updated beliefs:")
    for line in updated_beliefs.splitlines():
        logger.info(f"  {line}")

    logger.info(f"Number of updated beliefs: {len(updated_beliefs.splitlines())}")
    logger.info("=" * 80)

    return updated_beliefs


@dataclass
class EvolveConfig:
    """Configuration for evolution loop."""
    num_steps: int
    model: str
    dataset_path: str | Path
    log_dir: str | Path
    batch_size: int = 10
    eval_mode: str = "action"  # "action" or "state"


def evolve(config: EvolveConfig):
    """Run evolution loop to improve beliefs.

    Args:
        config: Evolution configuration
    """
    from wm.eval_hset import WMEvaluator, WMEvalConfig, EvalMode

    # Set up logger
    global logger
    logger = setup_logger(Path(config.log_dir))

    # Also set up the wm_eval logger so we can see eval.py logs
    eval_logger = setup_logger(Path(config.log_dir), name="wm_eval")

    logger.info("*" * 80)
    logger.info("STARTING EVOLUTION")
    logger.info(f"Model: {config.model}")
    logger.info(f"Number of steps: {config.num_steps}")
    logger.info(f"Dataset path: {config.dataset_path}")
    logger.info(f"Log directory: {config.log_dir}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Eval mode: {config.eval_mode}")
    logger.info("*" * 80)

    # Initialize evaluator
    logger.info("Creating WMEvaluator instance")
    eval_config = WMEvalConfig(
        model=config.model,
        dataset_path=config.dataset_path,
        log_dir=config.log_dir,
        eval_mode=EvalMode.ACTION if config.eval_mode == "action" else EvalMode.STATE,
    )
    evaluator = WMEvaluator(eval_config)

    # Initialize empty beliefs
    h = ""

    # Evolution loop
    for step in range(1, config.num_steps + 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP {step}/{config.num_steps}")
        logger.info("=" * 80)

        # Create step directory
        step_dir = Path(config.log_dir) / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Log current beliefs
        logger.info(f"Current beliefs: {h if h else '(empty)'}")

        # Evaluate current beliefs
        logger.info(f"Evaluating beliefs on dataset...")
        if h:
            hset = h.splitlines()
        else:
            hset = []

        accuracy, detailed_results = evaluator.eval_hypothesis(
            hset=hset,
            subset_size=None,  # Use all beliefs
            batch_size=config.batch_size,
            return_details=True,  # Get detailed results for improvement
        )

        logger.info(f"Step {step} accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Save accuracy result
        accuracy_file = step_dir / "accuracy.txt"
        accuracy_file.write_text(f"{accuracy:.4f}\n")
        logger.info(f"Saved accuracy to {accuracy_file}")

        # Save current beliefs
        beliefs_file = step_dir / "beliefs_before.txt"
        beliefs_file.write_text(h if h else "(empty)\n")
        logger.info(f"Saved current beliefs to {beliefs_file}")

        # Improve beliefs based on detailed results
        logger.info("Improving beliefs...")
        h = improve(
            model=config.model,
            beliefs=h,
            accuracy=accuracy,
            detailed_results=detailed_results,
            num_examples=10,  # Number of examples to show to LLM
        )

        # Save updated beliefs
        updated_beliefs_file = step_dir / "beliefs_after.txt"
        updated_beliefs_file.write_text(h)
        logger.info(f"Saved updated beliefs to {updated_beliefs_file}")

    logger.info("\n" + "*" * 80)
    logger.info("EVOLUTION COMPLETE")
    logger.info("*" * 80)


def run_evolve(
    num_steps: int,
    model: str,
    dataset_path: str = "nle_data/nld-aa-taster/nle_data",
    log_dir: str = "logs/evolution",
    batch_size: int = 10,
    eval_mode: str = "action",
):
    """Run evolution loop to improve beliefs.

    Args:
        num_steps: Number of evolution steps
        model: LLM model identifier (e.g., "anthropic/claude-3-5-sonnet-20241022")
        dataset_path: Path to NLE dataset
        log_dir: Directory for logs and results
        batch_size: Number of concurrent API requests
        eval_mode: Evaluation mode ("action" or "state")
    """
    config = EvolveConfig(
        num_steps=num_steps,
        model=model,
        dataset_path=dataset_path,
        log_dir=log_dir,
        batch_size=batch_size,
        eval_mode=eval_mode,
    )
    evolve(config)


if __name__ == "__main__":
    import fire

    fire.Fire(run_evolve)
