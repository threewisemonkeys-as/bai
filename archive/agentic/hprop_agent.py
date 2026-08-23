
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jinja2 import Template

from base_agent import BaseAgent, AgentConfig, ACTIONS
from utils import extract_xml_kv


PROPOSAL_PROMPT_TEMPLATE = """We currently know the following about this world -
{% for h in hypothesis_list %}
- {{ h }}
{% endfor %}

We do not know anything else about this world apart from the above.
Propose a hypothesis and an experiment to evaluate the hypothesis.
The hypothesis should be a concise statement describing some aspect of how the world works
The experiment should be a list of not more than 10 actions.

Present the proposal in an xml format -
<hypothesis>
proposed hypothesis statement
</hypothesis>
<experiment>
[a1, a2, a3 ...]
</experiment>
"""

VERIFIER_PROMPT_TEMPLATE = """We currently know the following about this world -
{% for h in hypothesis_list %}
- {{ h }}
{% endfor %}

We do not know anything else about this world apart from the above.
You are presented two images of the world, one before conducting the experiment and one after.
Your task is to determine whether the result of the experiment supports the hypothesis.

Proposed hypothesis: {{ proposed_h }}
Conducted experiment (sequence of actions): {{ experiment }}
Whether episode ended at the end of the experiment: {{ done }}

Present your judgement at the end of your answer in an xml format -
<answer>
Yes/No
</answer>
"""


SEED_HYPOTHESIS = [
    "We control the character",
    "Actions are specified by integers",
    f"There are {len(ACTIONS)} possible actions (numbered 0 - {len(ACTIONS) - 1})",
]


@dataclass
class HypothesisSet:
    hypothesis: list[str] = list()

    def add(self, h: str):
        self.hypothesis.append(h)

    def append(self, h: str):
        self.hypothesis.append(h)

    def __iter__(self):
        return iter(self.hypothesis)

    def __str__(self):
        return '\n'.join(f"- {h}" for h in self.hypothesis)


class HPropAgent(BaseAgent):
    """Agent that uses hypothesis-driven scientific method."""

    def __init__(self, config: AgentConfig, hypothesis: HypothesisSet | None = None):
        super().__init__(config)
        self.hypothesis_list = hypothesis if hypothesis is not None else HypothesisSet()
        self.logger.info(f"Initialized with seed hypotheses: {self.hypothesis_list}")



    def get_action_from_image(self, image_obs: np.ndarray) -> tuple[str, list[int]]:
        """Propose a hypothesis and experiment based on current observations.

        Returns:
            Tuple of (hypothesis, experiment_actions)
        """
        prompt = Template(PROPOSAL_PROMPT_TEMPLATE).render(hypothesis_list=self.hypothesis_list)
        input = self._build_input_with_images(prompt, [image_obs])
        response = self._call_llm(input)
        response_output_text = self._extract_response_text(response)

        response_dict = extract_xml_kv(response_output_text, ["hypothesis", "experiment"])
        self._validate_response_fields(response_dict, response_output_text, ["hypothesis", "experiment"])

        experiment_actions = self._parse_experiment(response_dict, response_output_text)
        hypothesis = response_dict["hypothesis"].strip()
        self.logger.info(f"Parsed hypothesis: {hypothesis}")

        return hypothesis, experiment_actions

    def update_hypothesis(self, image_obs: np.ndarray, h: str, e: list[int],
                         new_image_obs: np.ndarray, done: bool):
        """Verify if hypothesis is supported by experimental results.

        Args:
            image_obs: Image before experiment
            h: Proposed hypothesis
            e: Experiment actions taken
            new_image_obs: Image after experiment
            done: Whether episode ended
        """
        prompt = Template(VERIFIER_PROMPT_TEMPLATE).render(
            hypothesis_list=self.hypothesis_list,
            proposed_h=h,
            experiment=e,
            done=done,
        )
        input = self._build_input_with_images(prompt, [image_obs, new_image_obs])
        response = self._call_llm(input, num_retries=5)
        response_output_text = self._extract_response_text(response)

        response_dict = extract_xml_kv(response_output_text, ["answer"])
        self._validate_response_fields(response_dict, response_output_text, ["answer"])

        answer_text = response_dict["answer"].strip().lower()
        if "yes" in answer_text:
            self.logger.info(f"Hypothesis deemed supported, will be added to hypothesis list")
            self.hypothesis_list.append(h)
        elif "no" in answer_text:
            self.logger.info(f"Hypothesis deemed unsupported, will not be added to hypothesis list")
        else:
            error_msg = f"Could not determine Yes/No from answer text -\n{answer_text}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)


    def run(self, history_id: str | None = None):
        """Run the agent's main loop."""
        self.logger.info(f"Starting run")
        obs, info = self.env.reset()

        for iter_idx in range(self.config.max_iter):
            self.logger.info(f"=== Iter {iter_idx}/{self.config.max_iter} ===")
            image_obs = self.process_image_obs(obs["pixel"])

            self._save_image(image_obs, iter_idx, "start")

            hypothesis, exp_actions = self.get_action_from_image(image_obs)

            obs, done, taken_actions = self._execute_actions(exp_actions, iter_idx)

            new_image_obs = self.process_image_obs(obs["pixel"])
            self.update_hypothesis(image_obs, hypothesis, taken_actions, new_image_obs, done)
            hlist_str = '\n'.join(self.hypothesis_list)
            self.logger.info(f"End of Iter {iter_idx}. Hypothesis list -\n{hlist_str}")

            self._save_image(new_image_obs, iter_idx, "end")

            if done:
                obs, info = self.env.reset()

    

def main():

    num_runs = 1
    max_iter = 10
    envs = [
        "MiniHack-Eat-Distr-v0",
        "MiniHack-Pray-Distr-v0",
        "MiniHack-Wear-Distr-v0",
        "MiniHack-LockedDoor-v0",
        "MiniHack-LavaCross-Full-v0", 
    ]
    hset = HypothesisSet(SEED_HYPOTHESIS)

    for env in envs:
        for run_idx in range(num_runs):
            runner = HPropAgent(
                AgentConfig(
                    env_name=env,
                    max_iter=max_iter,
                    log_dir=Path(f"logs/hprop_agent/{env}/run{run_idx}"),
                ),
                hset
            )
            runner.run()
    

if __name__ == '__main__':
    main()
