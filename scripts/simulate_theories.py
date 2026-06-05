"""Simulate "5 theories" generation at chosen steps of a stepwise_eb_learn log.

For each requested step we reconstruct the exact context the model had at that
point in the run:

  * DEFAULT KNOWLEDGE  - the environment instruction prompt (extracted verbatim
                         from the step's logged question_gen_prompt).
  * WORLD KNOWLEDGE    - the step's beliefs.txt (<world_knowledge> block).
  * CURRENT STATE      - obs_before.png for the step (attached as an image), and
                         the perception module output if a non-empty
                         perception.py exists for the step.

We then ask the LLM to propose 5 distinct theories of how the game works, each
written as a <world_knowledge>-style block, ordered most -> least likely.

Usage:
  uv run scripts/simulate_theories.py \
      --episode-dir logs/.../episode_0 \
      --steps 5 9 12 \
      --out scripts/theory_sims

The model/client are read from the run's config.yaml (sibling of the episode
dir) unless overridden with --model.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running as `uv run scripts/simulate_theories.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf
from PIL import Image

# Reuse the production LLM call so model selection, the Responses API path, and
# cost accounting are identical to the real run.
from mixed_improve import _llm_call, _run_perception_on_observation, extract_perception_input

DK_START = "=== DEFAULT KNOWLEDGE ==="
DK_END = "=== END DEFAULT KNOWLEDGE ==="


def extract_default_knowledge(step_dir: Path) -> str:
    """Pull the verbatim DEFAULT KNOWLEDGE block from the step's logged prompt."""
    exp_log = step_dir / "experiment_log.json"
    if exp_log.exists():
        d = json.loads(exp_log.read_text())
        for key in ("question_gen_prompt", "experiment_prompt"):
            prompt = d.get(key) or ""
            i = prompt.find(DK_START)
            j = prompt.find(DK_END)
            if i != -1 and j != -1:
                return prompt[i + len(DK_START):j].strip()
    # Fallback: rebuild from the prompt module (generic action set + run goal).
    from explore import append_agent_goal, get_default_knowledge  # noqa
    raise RuntimeError(
        f"Could not extract DEFAULT KNOWLEDGE from {exp_log}; no logged prompt found."
    )


def get_perception_output(episode_dir: Path, step_dir: Path, step: int) -> str:
    """Run the step's perception module on its raw observation, if one exists."""
    perc_file = step_dir / "perception.py"
    if not perc_file.exists():
        return ""
    perception = perc_file.read_text()
    if not perception.strip():
        return ""  # No perception module at this step.
    buf_file = episode_dir / "trajectory_buffer.json"
    if not buf_file.exists():
        return ""
    buf = json.loads(buf_file.read_text())
    entry = next((e for e in buf if e.get("step") == step), None)
    if entry is None:
        return ""
    raw_obs = entry.get("raw_long_term_context") or entry.get("obs_text") or ""
    raw_obs = extract_perception_input(raw_obs)
    return _run_perception_on_observation(perception, raw_obs)


def build_theory_prompt(default_knowledge: str, world_knowledge: str, perception_output: str) -> str:
    perception_section = ""
    if perception_output.strip():
        perception_section = f"""
=== PERCEPTION MODULE OUTPUT (current state) ===
{perception_output}
=== END PERCEPTION MODULE OUTPUT ===
"""
    world_section = world_knowledge.strip() or "(no world knowledge accumulated yet)"

    return f"""You are an agent trying to understand how a game works. Below is the fixed \
information you were given about the environment, everything you currently believe about \
how the game works, and the current state of the game (provided as an image{', plus a perception module summary of that state' if perception_output.strip() else ''}).

=== DEFAULT KNOWLEDGE ===
{default_knowledge}
=== END DEFAULT KNOWLEDGE ===

=== CURRENT WORLD KNOWLEDGE (what you believe so far) ===
{world_section}
=== END CURRENT WORLD KNOWLEDGE ===
{perception_section}
The current game state is shown in the attached image.

Your task: propose exactly 5 DISTINCT theories of how this game works and what is \
required to make progress / complete levels. Each theory must be a self-contained \
explanation about the game's mechanics and win condition. The theories should \
be genuinely different hypotheses (not rephrasings of each other), and each must be \
consistent with the default knowledge and the current state, though they may go beyond \
or even contradict the current world knowledge where you think it may be wrong.

Order the theories from MOST likely to LEAST likely in your judgement.

Respond in exactly this format:

<theories>
<theory rank="1" likelihood="<your rough probability or short phrase>">
<content>
- ...
- ...
</content>
<rationale>Why you ranked this here.</rationale>
</theory>
<theory rank="2" likelihood="...">
<content>
- ...
</content>
<rationale>...</rationale>
</theory>
... (ranks 3, 4, 5)
</theories>
"""


async def run_step(config, episode_dir: Path, step: int, out_dir: Path) -> dict:
    step_dir = episode_dir / f"step_{step:03d}"
    if not step_dir.exists():
        raise FileNotFoundError(f"Step dir not found: {step_dir}")

    default_knowledge = extract_default_knowledge(step_dir)
    beliefs_file = step_dir / "beliefs.txt"
    world_knowledge = beliefs_file.read_text() if beliefs_file.exists() else ""
    perception_output = get_perception_output(episode_dir, step_dir, step)

    img_file = step_dir / "obs_before.png"
    image = Image.open(img_file).convert("RGB") if img_file.exists() else None

    prompt = build_theory_prompt(default_knowledge, world_knowledge, perception_output)

    print(f"[step {step}] calling model (image={'yes' if image else 'no'}, "
          f"perception={'yes' if perception_output.strip() else 'no'}) ...", flush=True)
    text, cost = await _llm_call(config, prompt, images=[image] if image else None)

    record = {
        "step": step,
        "model": f"{config.client.client_name}/{config.client.model_id}",
        "had_image": image is not None,
        "had_perception": bool(perception_output.strip()),
        "default_knowledge": default_knowledge,
        "world_knowledge": world_knowledge,
        "perception_output": perception_output,
        "prompt": prompt,
        "response": text,
        "cost_usd": cost,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"step_{step:03d}_theories.json").write_text(json.dumps(record, indent=2))
    (out_dir / f"step_{step:03d}_theories.txt").write_text(
        f"=== STEP {step} | model {record['model']} | cost ${cost:.4f} ===\n\n{text}\n"
    )
    print(f"[step {step}] done (cost ${cost:.4f}) -> {out_dir}/step_{step:03d}_theories.*", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True, type=Path)
    ap.add_argument("--steps", required=True, nargs="+", type=int)
    ap.add_argument("--out", type=Path, default=Path("scripts/theory_sims"))
    ap.add_argument("--model", default=None,
                    help="Override model as client_name/model_id, e.g. openrouter/google/gemini-2.5-flash")
    args = ap.parse_args()

    episode_dir = args.episode_dir.resolve()
    config_file = episode_dir.parent / "config.yaml"
    if not config_file.exists():
        sys.exit(f"config.yaml not found at {config_file}")
    config = OmegaConf.load(config_file)

    if args.model:
        client_name, model_id = args.model.split("/", 1)
        config.client.client_name = client_name
        config.client.model_id = model_id

    out_dir = args.out.resolve()
    for step in args.steps:
        asyncio.run(run_step(config, episode_dir, step, out_dir))


if __name__ == "__main__":
    main()
