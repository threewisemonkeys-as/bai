from collections import defaultdict
import random
from pathlib import Path
from dataclasses import dataclass

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
    action = int(before["keypresses"][0])

    before_screen = tty_render(
        before["tty_chars"], before["tty_colors"], before["tty_cursor"]
    )
    after_screen = tty_render(
        after["tty_chars"], after["tty_colors"], after["tty_cursor"]   
    )
    print(f"Before scene -\n{before_screen}\n\nAction taken - {action}\n\nAfter Scene -\n{after_screen}")

def render_transition(transition):
    before, after = transition
    before_screen = tty_render(
        before["tty_chars"], before["tty_colors"], before["tty_cursor"]
    )
    after_screen = tty_render(
        after["tty_chars"], after["tty_colors"], after["tty_cursor"]   
    )
    return before_screen, after_screen


@dataclass
class WMEvalConfig:
    model: str
    dataset_path: str | Path
    log_dir: str | Path


class WMEvaluator:

    def __init__(
        self,
        config: WMEvalConfig,
    ):
        self.config = config
        self._dbfilename = "ttyrecs.db"

        if not nle.dataset.db.exists(self._dbfilename):
            nle.dataset.db.create(self._dbfilename)
            nle.dataset.add_nledata_directory(self.config.dataset_path, "taster-dataset", self._dbfilename)
        
        self._db_conn = nle.dataset.db.connect(filename=self._dbfilename)
        print(f"NLD AA \"Taster\" Dataset has {nle.dataset.db.count_games('taster-dataset', conn=self._db_conn)} games.")
        self.dataset = nle.dataset.TtyrecDataset(
            "taster-dataset",
            batch_size=32,
            seq_length=2,
            dbfilename=self._dbfilename,
        )



        _data = defaultdict(list)
        for i, batch in enumerate(self.dataset):
            if i == 16:
                break

            for k, v in batch.items():
                _data[k].extend(v)

        _data["keypresses"] = [np.array([ACTION_MAP[kp] for kp in kpd]) for kpd in _data["keypresses"]]
        invert = lambda dict_data: [dict(zip(dict_data.keys(), vals)) for vals in zip(*dict_data.values())]
        _data = [invert(d) for d in invert(_data)]
        self.correct_data = _data

        # Create incorrect_data: same first timestep, different second timestep
        self.incorrect_data = []
        second_timesteps = [pair[1] for pair in self.correct_data]
        shuffled_indices = np.random.permutation(len(second_timesteps))
        for i, pair in enumerate(self.correct_data):
            incorrect_second = second_timesteps[shuffled_indices[i]]
            self.incorrect_data.append([pair[0], incorrect_second])
            
        self.data = [(i, 1) for i in self.correct_data] + [(i, 0) for i in self.incorrect_data]
        random.shuffle(self.data)
        

    def eval_hypothesis(
        self,
        hset: list[str]
    ):
        results = []

        for transition, label in self.data:
            before_screen, after_screen = render_transition(transition)
            action_taken = int(transition[0]["keypresses"][0])
            
            prompt = Template(WM_PROMPT_TEMPLATE).render(
                hypothesis_list=hset,
                before_screen=before_screen,
                action_taken=action_taken,
                after_screen=after_screen,
            )

            input = build_llm_input(prompt)
            response = litellm.responses(
                model=self.config.model,
                input=input,
                num_retries=5,
            )
            response_output_text = extract_llm_response_text(response)

            response_dict = extract_xml_kv(response_output_text, ["answer"])
            validate_response_fields(response_dict, response_output_text, ["answer"])

            answer_text = response_dict["answer"].strip().lower()
            if "yes" in answer_text:
                answer_bool = True
            elif "no" in answer_text:
                answer_bool = False
            else:
                raise RuntimeError(f"Could not get answer from answer text -\n{answer_text}")

            results.append(bool(label) == answer_bool)
        

        if len(results) == 0:
            raise RuntimeError(f"Did not eval anything!")
        
        return sum(results) / len(results)
        
    def eval_experiment(self, hstar: list[str]):
        results = {}

        percentages = list(range(10, 101, 10))

        for pct in percentages:
            subset_size = int(len(hstar) * pct / 100)
            subset = random.sample(hstar, subset_size)

            print(f"Evaluating {pct}% subset (size={subset_size})...")
            result = self.eval_hypothesis(subset)
            results[pct] = result

        return results


if __name__ == '__main__':
    e = WMEvaluator(
            WMEvalConfig(
                model="",
                dataset_path="/Users/ays57/Documents/opus/bai/nle_data/nld-aa-taster/nle_data",
                log_dir="logs/",
            )
    )

    breakpoint()
        