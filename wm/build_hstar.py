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



PROMPT_TEMPLATE = """We are building a list of hypothesis for the game of NetHack
We have the following list of commands that we can take in the game.
Each command is specified as Command name: command int
Here command name describes what the command does and the command int is the id that we submit to the game when taking the action.

{% for c in command_list %}
{{ c[0] }}: {{ c[1] }}
{% endfor %}


Given this list of command, think what the mechanics of the game might be.
Finally output a list of hyptoehsis that describe aspects of the game in the format: 
- hypothesis 1
- hypothesis 2
...
Explicitly talk about the command ints when talking about commands and do not use their informal names.
Use whatever you already know about NetHack while building this list.
"""

def main(
    model: str,
    output: str | Path,
):
    output = Path(output)

    actions = [(str(a), a.value) for a in NetHackChallenge().actions]
    prompt = Template(PROMPT_TEMPLATE).render(command_list=actions)

    print(prompt)

    input = build_llm_input(prompt)
    response = litellm.responses(
        model=model,
        input=input,
    )
    response_text = extract_llm_response_text(response)

    output.write_text(response_text)
    print(f"Wrote to {output}")

if __name__ == '__main__':
    import fire
    fire.Fire(main)
