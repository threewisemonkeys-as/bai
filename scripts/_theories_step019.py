"""Ad-hoc: ask the LLM for theories of how ft09 works, given the default
knowledge + the beliefs captured at step_019 of the planB_ft09_v3 run.

Reuses theory_exploration.generate_theories so the prompt is built exactly the
way the real loop builds it (DK + current beliefs, no history / no live state).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf
from PIL import Image

import mixed_improve
from theory_exploration import generate_theories

STEP_DIR = (
    "logs/planB_ft09_v3/2026-06-03_15-31-19_robust_cot_google_gemini-2.5-flash_"
    "stepwise_eb_learn/episode_0/step_019"
)

# Default knowledge exactly as used in this run (extracted from the step's
# scoring_online_theory_entropy.json prediction prompt).
DEFAULT_KNOWLEDGE = """You are playing a game.

The game is played on a 64x64 grid of integer values (0-15), each representing a color:
  0: white
  1: off-white
  2: light-gray
  3: gray
  4: dark-gray
  5: black
  6: magenta
  7: pink
  8: red
  9: blue
  10: light-blue
  11: yellow
  12: orange
  13: maroon
  14: green
  15: purple

Available actions:
<actions>
ACTION6 x=<int> y=<int>: Complex action - click/select the cell at coordinates x,y (each in 0-63)
</actions>

- ACTION6 requires x and y coordinates (e.g. "ACTION6 x=32 y=16").

Your goal is to progress as much as possible in the game."""


# ARC-AGI 16-color palette (RGB), matching arc_agi_env._PALETTE.
_PALETTE = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}
_RGB_TO_IDX = {rgb: idx for idx, rgb in _PALETTE.items()}


def reconstruct_raw_state(img: Image.Image) -> str:
    """Recover the 64x64 int grid from the rendered image and format it as the
    env's long_term_context (<grid_0> ... </grid_0>, one python list per row)."""
    px = img.load()
    w, h = img.size
    sx, sy = w // 64, h // 64  # integer upscale factor (=2)
    def to_idx(rgb):
        if rgb in _RGB_TO_IDX:
            return _RGB_TO_IDX[rgb]
        # nearest palette color (guards against any anti-aliasing)
        nearest = min(_RGB_TO_IDX, key=lambda k: sum((a - b) ** 2 for a, b in zip(k, rgb)))
        return _RGB_TO_IDX[nearest]

    rows = []
    for r in range(64):
        row = [to_idx(px[c * sx, r * sy][:3]) for c in range(64)]
        rows.append(row)
    lines = ["<grid_0>"] + [f"{row}" for row in rows] + ["</grid_0>", ""]
    return "\n".join(lines)


def main():
    with open(f"{STEP_DIR}/beliefs.txt") as f:
        beliefs = f.read()

    # Pre-action state image for this step.
    state_image = Image.open(f"{STEP_DIR}/obs_before.png").convert("RGB")

    # Reconstruct the raw text state (long_term_context) from the image. The
    # env renders the 64x64 int grid with a fixed 16-color palette at scale 2
    # (nearest-neighbor), so downsampling + inverse-palette recovers the grid
    # exactly, and we re-emit it in the env's <grid_0> row-by-row format.
    raw_state = reconstruct_raw_state(state_image)

    config = OmegaConf.create(
        {"client": {"client_name": "openrouter", "model_id": "google/gemini-2.5-flash"}}
    )
    mixed_improve._MOCK_MODE = False
    mixed_improve._META_TEMPERATURE = 0.0

    theories, cost, log = asyncio.run(
        generate_theories(
            config=config,
            beliefs=beliefs,
            default_knowledge=DEFAULT_KNOWLEDGE,
            steps_context="",             # no history
            current_observation=raw_state,  # raw grid text state
            current_image=state_image,     # plus the state image
            steps_context_images=None,
            num_theories=10,
            decay=0.6,
        )
    )

    print("=" * 80)
    print("PROMPT SENT")
    print("=" * 80)
    print(log["prompt"])
    print()
    print("=" * 80)
    print(f"PARSED {len(theories)} THEORIES (cost=${cost:.4f})")
    print("=" * 80)
    for t in theories:
        print(f"\n--- THEORY rank={t.rank} likelihood={t.likelihood!r} weight={t.weight:.3f} ---")
        print(t.world_knowledge)
        if t.rationale:
            print(f"[rationale] {t.rationale}")


if __name__ == "__main__":
    main()
