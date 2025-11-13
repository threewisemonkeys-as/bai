
from pathlib import Path

import numpy as np
from jinja2 import Template

from base_agent import BaseAgent, AgentConfig, ACTIONS
from utils import extract_xml_kv


INIT_PROMPT = f"""We currently know the following about this world -

- We control the character
- Actions are specified by integers
- There are {len(ACTIONS)} possible actions (numbered 0 - {len(ACTIONS) - 1})

We do not know anything else about this world apart from the above.
Our task is to find out how this world works.
We will do this by interacting with the world and observing the results of the interactions.
The interactions will be specified as experiments which are a list of actions.
The observations will be images of the world before and after the experiment.
We want to build a list of facts in addition to those given above where each fact should be a concise statement describing some aspect of how the world works.
The experiment should be a list of not more than 10 actions.

At each step specify the facts so far and propose an experiment in an xml format -
<facts>
statements about the world from our interactions so far
</facts>
<experiment>
[a1, a2, a3 ...]
</experiment>
"""


CONT_PROMPT_TEMPLATE = """Our task is to build an understanding of how this world works.
We just carried out the following actions: {{ taken_actions }}
The image of the world before and after these actions were carried out is also presented.
Think about what this tells us about how this world works.

Update the facts and propose the next experiment in an xml format -
<facts>
statements about the world from our interactions so far
</facts>
<experiment>
[a1, a2, a3 ...]
</experiment>
"""


class SimpleAgent(BaseAgent):
    """Agent that builds understanding through conversation-based fact accumulation."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)


    def get_init_action_from_image(self, image_obs: np.ndarray, history_id: str | None = None) -> tuple[list[int], str]:
        """Get initial action proposal from image observation.

        Returns:
            Tuple of (experiment_actions, response_id)
        """
        input = self._build_input_with_images(INIT_PROMPT, [image_obs])
        response = self._call_llm(input, previous_response_id=history_id)
        response_output_text = self._extract_response_text(response)

        response_dict = extract_xml_kv(response_output_text, ["experiment"])
        experiment_actions = self._parse_experiment(response_dict, response_output_text)

        return experiment_actions, response.id




    def get_action_from_image(self, image_obs: np.ndarray, new_image_obs: np.ndarray,
                              taken_actions: list[int], history_id: str) -> tuple[list[int], str]:
        """Get next action proposal based on previous actions and observations.

        Args:
            image_obs: Image before actions
            new_image_obs: Image after actions
            taken_actions: List of actions taken
            history_id: Conversation history ID

        Returns:
            Tuple of (experiment_actions, response_id)
        """
        prompt = Template(CONT_PROMPT_TEMPLATE).render(taken_actions=taken_actions)
        input = self._build_input_with_images(prompt, [image_obs, new_image_obs])
        response = self._call_llm(input, previous_response_id=history_id)
        response_output_text = self._extract_response_text(response)

        response_dict = extract_xml_kv(response_output_text, ["experiment"])
        experiment_actions = self._parse_experiment(response_dict, response_output_text)

        return experiment_actions, response.id

    def run(self, history_id: str | None):
        """Run the agent's main loop."""
        self.logger.info(f"Starting run")
        obs, info = self.env.reset()

        image_obs = self.process_image_obs(obs["pixel"])
        exp_actions, history_id = self.get_init_action_from_image(image_obs, history_id=history_id)

        for iter_idx in range(self.config.max_iter):
            self.logger.info(f"=== Iter {iter_idx}/{self.config.max_iter} ===")

            self._save_image(image_obs, iter_idx, "start")

            obs, done, taken_actions = self._execute_actions(exp_actions, iter_idx)

            new_image_obs = self.process_image_obs(obs["pixel"])
            self._save_image(new_image_obs, iter_idx, "end")

            exp_actions, history_id = self.get_action_from_image(
                image_obs,
                new_image_obs,
                taken_actions,
                history_id=history_id
            )

            image_obs = new_image_obs
            if done:
                obs, info = self.env.reset()

        return history_id
    

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
    history_id = None
    for env in envs:
        for run_idx in range(num_runs):
            runner = SimpleAgent(
                AgentConfig(
                    env_name=env,
                    max_iter=max_iter,
                    log_dir=Path(f"logs/simple_agent/{env}/run{run_idx}"),
                ),
            )
            runner.run(history_id)
    

if __name__ == '__main__':
    main()
