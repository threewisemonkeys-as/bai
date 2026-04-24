"""Experiment matrix launcher.

Cross-product of {script, env, model} cells with per-cell Hydra overrides.
Each cell is launched as a subprocess. Edit the MATRIX section below to
change dimensions or tuning; use CLI flags to pick subsets at launch time.

Usage:
    uv run launch.py --log-dir logs/matrix --dry-run
    uv run launch.py --log-dir logs/matrix --scripts eb_learn --envs minihack,autumn
    uv run launch.py --log-dir logs/matrix --models gemini-2.5-flash --parallel 2

Outputs land under <log_dir>/<timestamp>/<cell_name>/ with stdout.log,
stderr.log, and cmd.txt (the exact command that was run). Override the
timestamp subdir with --tag.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# MATRIX — edit here to change dimensions, model IDs, or eb_learn tuning.
# ---------------------------------------------------------------------------

SCRIPT_FILES = {
    "eb_learn":  "stepwise_eb_learn.py",
    # "openhands": "openhands_stepwise.py",
    # "simple":    "simple_stepwise.py",
}

ENVS = [
    "minihack",
    "arc_agi",
    "autumn",
]

# model → provider model ID. All scripts route via OpenRouter; to use a
# direct provider, change the id here and adjust `model_overrides` below.
# "mock" uses a placeholder id — it's never called since mock_mode shorts
# out all LLM requests.
MODEL_IDS = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    # "sonnet-4.6":       "anthropic/claude-sonnet-4.6",
    # "mock":             "google/gemini-2.5-flash",
}
MODELS = list(MODEL_IDS)


def model_overrides(model: str, script: str) -> dict:
    """Hydra overrides for (model, script). `mock` additionally flips each
    script's mock_mode flag; the mock step budget for eb_learn lives in
    EB_LEARN_OVERRIDES (for openhands/simple it's set here)."""
    mid = MODEL_IDS[model]
    is_mock = model == "mock"
    if script == "eb_learn":
        ov: dict = {
            "client.client_name": "openrouter",
            "client.model_id":    mid,
        }
        if is_mock:
            ov["eval.evolve.mock_mode"] = True
        return ov
    ns = f"eval.{script}"  # eval.openhands | eval.simple
    # Set mock_mode explicitly — config.yaml defaults it to true for these
    # scripts, so non-mock models must flip it off or they'll never call the LLM.
    ov = {
        f"{ns}.model":       f"openrouter/{mid}",
        f"{ns}.api_key_env": "OPENROUTER_API_KEY",
        f"{ns}.mock_mode":   is_mock,
    }
    if is_mock:
        ov[f"{ns}.n_environment_steps"] = 5
    return ov

# eb_learn-only defaults and per-cell tuning. Per-cell overrides are layered
# on top of EB_LEARN_DEFAULT so shared behavior, including question scoring,
# is not lost when an env/model only customizes step counts or image settings.
EB_LEARN_DEFAULT = {
    "eval.evolve.n_environment_steps": 10,
    "eval.evolve.hide_obs_when_image": False,
    "agent.max_image_history":         0,
    "eval.evolve.question_scoring_method": "b_diff_full",
    "eval.evolve.question_scoring_max_concurrent": 8,
    "eval.evolve.max_unanswered_qa_pairs": 20,
}
EB_LEARN_OVERRIDES: dict[tuple[str, str], dict] = {
    ("minihack", "gemini-2.5-flash"): {
        "eval.evolve.n_environment_steps": 50,
        "eval.evolve.hide_obs_when_image": False,
        "agent.max_image_history":         0,
    },
    ("minihack", "sonnet-4.6"): {
        "eval.evolve.n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": False,
        "agent.max_image_history":         0,
    },
    ("arc_agi", "gemini-2.5-flash"): {
        "eval.evolve.n_environment_steps": 50,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_image_history":         4,
    },
    ("arc_agi", "sonnet-4.6"): {
        "eval.evolve.n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_image_history":         4,
    },
    ("autumn", "gemini-2.5-flash"): {
        "eval.evolve.n_environment_steps": 100,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_image_history":         4,
        "eval.evolve.autumn_eval_after_learn": True,
        "eval.evolve.autumn_eval_max_steps": 501,
    },
    ("autumn", "sonnet-4.6"): {
        "eval.evolve.n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_image_history":         4,
    },
    ("minihack", "mock"): {
        "eval.evolve.n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": False,
        "agent.max_image_history":         0,
    },
    ("arc_agi", "mock"): {
        "eval.evolve.n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_image_history":         4,
    },
    ("autumn", "mock"): {
        "eval.evolve.n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_image_history":         4,
    },
}


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    script: str
    env: str
    model: str
    overrides: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.script}__{self.env}__{self.model}".replace(".", "p")


# Hydra key for "number of env steps" per script — used by --num-steps.
NUM_STEPS_KEYS = {
    "eb_learn":  "eval.evolve.n_environment_steps",
    "openhands": "eval.openhands.n_environment_steps",
    "simple":    "eval.simple.n_environment_steps",
}


def build_cells() -> list[Cell]:
    cells: list[Cell] = []
    for script, env, model in itertools.product(SCRIPT_FILES, ENVS, MODELS):
        ov: dict = {"envs.names": env, **model_overrides(model, script)}
        if script == "eb_learn":
            ov.update({
                **EB_LEARN_DEFAULT,
                **EB_LEARN_OVERRIDES.get((env, model), {}),
            })
        cells.append(Cell(script=script, env=env, model=model, overrides=ov))
    return cells


# ---------------------------------------------------------------------------
# Filtering & launch
# ---------------------------------------------------------------------------

def filter_cells(cells, scripts, envs, models) -> list[Cell]:
    def match(values, v):
        return values is None or v in values
    return [c for c in cells
            if match(scripts, c.script)
            and match(envs,    c.env)
            and match(models,  c.model)]


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def build_cmd(cell: Cell, output_dir: Path) -> list[str]:
    overrides = {**cell.overrides, "eval.output_dir": str(output_dir)}
    args = [f"{k}={_fmt(v)}" for k, v in overrides.items()]
    return ["uv", "run", SCRIPT_FILES[cell.script], *args]


def run_cell(cell: Cell, root: Path) -> tuple[Cell, int]:
    out_dir = root / cell.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(cell, out_dir)
    (out_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")
    with open(out_dir / "stdout.log", "w") as so, open(out_dir / "stderr.log", "w") as se:
        rc = subprocess.run(cmd, stdout=so, stderr=se).returncode
    return cell, rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_csv(s):
    return None if s is None else [x.strip() for x in s.split(",") if x.strip()]


def _validate(values, valid, label):
    if values is None:
        return
    bad = [v for v in values if v not in valid]
    if bad:
        sys.exit(f"Unknown {label}: {bad}. Valid: {sorted(valid)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scripts", type=parse_csv, default=None,
                   help=f"Subset of {list(SCRIPT_FILES)} (comma-separated)")
    p.add_argument("--envs", type=parse_csv, default=None,
                   help=f"Subset of {ENVS}")
    p.add_argument("--models", type=parse_csv, default=None,
                   help=f"Subset of {MODELS}")
    p.add_argument("--log-dir", type=Path, required=True,
                   help="Root directory; cells land in <log_dir>/<tag>/<cell_name>/")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Global override for n_environment_steps across all cells")
    p.add_argument("--parallel", type=int, default=1,
                   help="Max concurrent cells (default: 1 = serial)")
    p.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"),
                   help="Subdir name under --log-dir (default: timestamp)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved matrix and exit")
    args = p.parse_args()

    _validate(args.scripts, SCRIPT_FILES, "script")
    _validate(args.envs,    ENVS,         "env")
    _validate(args.models,  MODELS,       "model")

    cells = filter_cells(build_cells(), args.scripts, args.envs, args.models)
    if not cells:
        sys.exit("No cells matched filters.")

    if args.num_steps is not None:
        for c in cells:
            c.overrides[NUM_STEPS_KEYS[c.script]] = args.num_steps

    root = args.log_dir / args.tag
    print(f"Matrix root: {root}")
    print(f"Cells ({len(cells)}):")
    for c in cells:
        print(f"  - {c.name}")
        for k, v in c.overrides.items():
            print(f"      {k}={_fmt(v)}")

    if args.dry_run:
        return

    root.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futures = [ex.submit(run_cell, c, root) for c in cells]
        for fut in as_completed(futures):
            cell, rc = fut.result()
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[{status}] {cell.name}")
            if rc != 0:
                failed.append(cell.name)

    if failed:
        sys.exit(f"{len(failed)} cell(s) failed: {failed}")


if __name__ == "__main__":
    main()
