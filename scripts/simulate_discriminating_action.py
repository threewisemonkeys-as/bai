"""Offline validation of Plan A's discriminating-action selection.

Reconstructs the exact state context at a chosen step of a stepwise_eb_learn
run (reusing the helpers from ``simulate_theories.py``), parses the competing
theories that the one-shot brainstorm produced there, then exercises the Plan A
primitives in ``multi_theory_exploration.py``:

  * ``select_discriminating_action`` — the key check: at the step-9 gemini ARC
    case, does the discriminator actually pick an action that *tests* the
    correct (target-pattern) theory, rather than letting the agent fixate?
  * ``refill_theories`` (optional, ``--refill``) — demo of replenishing the
    ensemble after dropping theories.

This does NOT touch the live loop; it just calls the new functions with the
production ``_llm_call`` so model selection / mock-mode / cost are identical to
a real run.

Usage:
  uv run scripts/simulate_discriminating_action.py \
      --episode-dir logs/.../episode_0 --step 9 \
      --out scripts/discriminating_sims

The theories are taken from a sibling ``scripts/theory_sims/step_<NNN>_theories.json``
dump by default (produced by simulate_theories.py); override with --theory-dump.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf
from PIL import Image

from scripts.simulate_theories import (  # reuse exact context reconstruction
    extract_default_knowledge,
    get_perception_output,
)
from theory_exploration import assign_rank_weights, parse_theories

import multi_theory_exploration as M


async def run(config, episode_dir: Path, step: int, theory_dump: Path,
              out_dir: Path, decay: float, refill: bool) -> dict:
    step_dir = episode_dir / f"step_{step:03d}"
    if not step_dir.exists():
        sys.exit(f"Step dir not found: {step_dir}")

    # --- Reconstruct the state context for this step. ---
    default_knowledge = extract_default_knowledge(step_dir)
    beliefs_file = step_dir / "beliefs.txt"
    beliefs = beliefs_file.read_text() if beliefs_file.exists() else ""
    perception_output = get_perception_output(episode_dir, step_dir, step)
    obs_text = perception_output or None

    img_file = step_dir / "obs_before.png"
    image = Image.open(img_file).convert("RGB") if img_file.exists() else None

    # --- Load + weight the competing theories from the offline dump. ---
    if not theory_dump.exists():
        sys.exit(f"Theory dump not found: {theory_dump} (run simulate_theories.py first)")
    dump = json.loads(theory_dump.read_text())
    theories = parse_theories(dump["response"])
    assign_rank_weights(theories, decay=decay)
    M.reindex_ranks(theories)
    print(f"[step {step}] loaded {len(theories)} theories "
          f"(weights={[round(t.weight, 3) for t in theories]})", flush=True)

    total_cost = 0.0

    # --- The headline check: discriminating-action selection. ---
    action, sel_cost, sel_log = await M.select_discriminating_action(
        config,
        theories=theories,
        beliefs=beliefs,
        default_knowledge=default_knowledge,
        current_observation=obs_text,
        current_image=image,
    )
    total_cost += sel_cost
    if action is None:
        print(f"[step {step}] select_discriminating_action returned no action.", flush=True)
    else:
        print(f"\n[step {step}] SELECTED DISCRIMINATING ACTION:\n{action.plan}\n", flush=True)
        print(f"  rationale: {action.rationale}", flush=True)
        for rank, outcome in sorted(action.predictions.items()):
            print(f"  theory {rank} predicts: {outcome}", flush=True)

    refill_log = None
    if refill and theories:
        # Demo: drop the lowest-weight half, then refill back to the original N.
        target_n = len(theories)
        theories.sort(key=lambda t: t.weight, reverse=True)
        del theories[(target_n + 1) // 2:]
        M.renormalize(theories)
        M.reindex_ranks(theories)
        theories, refill_cost, refill_log = await M.refill_theories(
            config,
            theories=theories,
            beliefs=beliefs,
            default_knowledge=default_knowledge,
            num_theories=target_n,
            current_observation=obs_text,
            current_image=image,
        )
        total_cost += refill_cost
        print(f"\n[step {step}] after refill: {len(theories)} theories "
              f"(weights={[round(t.weight, 3) for t in theories]})", flush=True)

    record = {
        "step": step,
        "model": f"{config.client.client_name}/{config.client.model_id}",
        "theory_dump": str(theory_dump),
        "decay": decay,
        "select_action_log": sel_log,
        "selected_action": (
            {
                "plan": action.plan,
                "rationale": action.rationale,
                "predictions": action.predictions,
                "candidate_actions": action.candidate_actions,
            }
            if action else None
        ),
        "refill_log": refill_log,
        "total_cost_usd": total_cost,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"step_{step:03d}_discriminate.json").write_text(
        json.dumps(record, indent=2, default=str)
    )
    print(f"\n[step {step}] done (cost ${total_cost:.4f}) -> "
          f"{out_dir}/step_{step:03d}_discriminate.json", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True, type=Path)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--theory-dump", type=Path, default=None,
                    help="Defaults to scripts/theory_sims/step_<NNN>_theories.json")
    ap.add_argument("--out", type=Path, default=Path("scripts/discriminating_sims"))
    ap.add_argument("--decay", type=float, default=0.6, help="Rank-prior decay.")
    ap.add_argument("--refill", action="store_true",
                    help="Also demo refill_theories after dropping half the ensemble.")
    ap.add_argument("--model", default=None,
                    help="Override model as client_name/model_id.")
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

    theory_dump = args.theory_dump or (
        Path("scripts/theory_sims") / f"step_{args.step:03d}_theories.json"
    )
    asyncio.run(run(config, episode_dir, args.step, theory_dump.resolve(),
                    args.out.resolve(), args.decay, args.refill))


if __name__ == "__main__":
    main()
