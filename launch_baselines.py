"""Baseline experiment matrix launcher.

Runs the OpenHands and simple stepwise baselines over the same env/model matrix
shape as launch.py. Each cell is launched as a subprocess with per-cell Hydra
overrides, and all selected cells run in parallel by default.

Usage:
    uv run launch_baselines.py --log-dir logs/baselines --dry-run
    uv run launch_baselines.py --log-dir logs/baselines --scripts openhands
    uv run launch_baselines.py --log-dir logs/baselines --envs minihack,autumn
    uv run launch_baselines.py --log-dir logs/baselines --parallel 2

Outputs land under <log_dir>/<timestamp>/<cell_name>/ with stdout.log,
stderr.log, and cmd.txt. Override the timestamp subdir with --tag.
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
# MATRIX -- keep this aligned with launch.py unless intentionally changed.
# ---------------------------------------------------------------------------

SCRIPT_FILES = {
    "openhands": "openhands_stepwise.py",
    "simple": "simple_stepwise.py",
}

ENVS = [
    "minihack",
    "arc_agi",
    "autumn",
]

# model -> provider model ID. All real runs route through OpenRouter, matching
# launch.py. Add "mock" here if you want fast smoke tests without LLM calls.
MODEL_IDS = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    # "sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "mock": "google/gemini-2.5-flash",
}
MODELS = list(MODEL_IDS)


def model_overrides(model: str, script: str) -> dict:
    """Hydra overrides for (model, baseline script)."""
    mid = MODEL_IDS[model]
    is_mock = model == "mock"
    ns = f"eval.{script}"  # eval.openhands | eval.simple

    overrides = {
        f"{ns}.model": f"openrouter/{mid}",
        f"{ns}.api_key_env": "OPENROUTER_API_KEY",
        # Baseline config.yaml defaults mock_mode to true for smoke tests, so
        # real model cells must explicitly disable it.
        f"{ns}.mock_mode": is_mock,
    }
    if is_mock:
        overrides[f"{ns}.n_environment_steps"] = 5
    return overrides


# Comparable EB-launch settings mirrored onto the baselines. EB-only learning
# controls (question scoring, QA caps, post-learning eval flags) have no
# baseline equivalent and are intentionally not included here.
BASELINE_DEFAULT = {
    "n_environment_steps": 10,
    "eval.evolve.hide_obs_when_image": False,
    "agent.max_text_history": 4,
    "agent.max_image_history": 0,
    "agent.max_cot_history": 4,
}

BASELINE_OVERRIDES: dict[tuple[str, str], dict] = {
    ("minihack", "gemini-2.5-flash"): {
        "n_environment_steps": 50,
        "eval.evolve.hide_obs_when_image": False,
        "agent.max_text_history": 4,
        "agent.max_image_history": 0,
    },
    ("minihack", "sonnet-4.6"): {
        "n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": False,
        "agent.max_text_history": 4,
        "agent.max_image_history": 0,
    },
    ("arc_agi", "gemini-2.5-flash"): {
        "n_environment_steps": 50,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_text_history": 4,
        "agent.max_image_history": 4,
    },
    ("arc_agi", "sonnet-4.6"): {
        "n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_text_history": 4,
        "agent.max_image_history": 4,
    },
    ("autumn", "gemini-2.5-flash"): {
        "n_environment_steps": 100,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_text_history": 4,
        "agent.max_image_history": 4,
    },
    ("autumn", "sonnet-4.6"): {
        "n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_text_history": 4,
        "agent.max_image_history": 4,
    },
    ("minihack", "mock"): {
        "n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": False,
        "agent.max_text_history": 4,
        "agent.max_image_history": 0,
    },
    ("arc_agi", "mock"): {
        "n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_text_history": 4,
        "agent.max_image_history": 4,
    },
    ("autumn", "mock"): {
        "n_environment_steps": 5,
        "eval.evolve.hide_obs_when_image": True,
        "agent.max_text_history": 4,
        "agent.max_image_history": 4,
    },
}

NUM_STEPS_KEYS = {
    "openhands": "eval.openhands.n_environment_steps",
    "simple": "eval.simple.n_environment_steps",
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


def _baseline_env_overrides(env: str, model: str, script: str) -> dict:
    merged = {
        **BASELINE_DEFAULT,
        **BASELINE_OVERRIDES.get((env, model), {}),
    }
    steps = merged.pop("n_environment_steps")
    overrides = {
        NUM_STEPS_KEYS[script]: steps,
        **merged,
    }
    if script == "simple":
        overrides["eval.simple.hide_obs_when_image"] = merged[
            "eval.evolve.hide_obs_when_image"
        ]
        overrides["eval.simple.history_window"] = merged["agent.max_text_history"]
        overrides["eval.simple.max_image_history"] = merged["agent.max_image_history"]
    return overrides


def build_cells() -> list[Cell]:
    cells: list[Cell] = []
    for script, env, model in itertools.product(SCRIPT_FILES, ENVS, MODELS):
        overrides = {
            "envs.names": env,
            **model_overrides(model, script),
            **_baseline_env_overrides(env, model, script),
        }
        cells.append(Cell(script=script, env=env, model=model, overrides=overrides))
    return cells


# ---------------------------------------------------------------------------
# Filtering and launch
# ---------------------------------------------------------------------------


def parse_csv(s):
    return None if s is None else [x.strip() for x in s.split(",") if x.strip()]


def _validate(values, valid, label):
    if values is None:
        return
    bad = [v for v in values if v not in valid]
    if bad:
        sys.exit(f"Unknown {label}: {bad}. Valid: {sorted(valid)}")


def filter_cells(cells, scripts, envs, models) -> list[Cell]:
    def match(values, value):
        return values is None or value in values

    return [
        c
        for c in cells
        if match(scripts, c.script)
        and match(envs, c.env)
        and match(models, c.model)
    ]


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scripts",
        type=parse_csv,
        default=None,
        help=f"Subset of {list(SCRIPT_FILES)} (comma-separated)",
    )
    parser.add_argument("--envs", type=parse_csv, default=None, help=f"Subset of {ENVS}")
    parser.add_argument(
        "--models",
        type=parse_csv,
        default=None,
        help=f"Subset of {MODELS}",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Root directory; cells land in <log_dir>/<tag>/<cell_name>/",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Global override for n_environment_steps across all cells",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Max concurrent cells (default: all selected cells)",
    )
    parser.add_argument(
        "--tag",
        default=time.strftime("%Y%m%d-%H%M%S"),
        help="Subdir name under --log-dir (default: timestamp)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved matrix and exit",
    )
    args = parser.parse_args()

    _validate(args.scripts, SCRIPT_FILES, "script")
    _validate(args.envs, ENVS, "env")
    _validate(args.models, MODELS, "model")

    cells = filter_cells(build_cells(), args.scripts, args.envs, args.models)
    if not cells:
        sys.exit("No cells matched filters.")

    if args.num_steps is not None:
        for cell in cells:
            cell.overrides[NUM_STEPS_KEYS[cell.script]] = args.num_steps

    root = args.log_dir / args.tag
    max_workers = len(cells) if args.parallel is None else max(1, args.parallel)

    print(f"Matrix root: {root}")
    print(f"Cells ({len(cells)}), parallel={max_workers}:")
    for cell in cells:
        print(f"  - {cell.name}")
        for key, value in cell.overrides.items():
            print(f"      {key}={_fmt(value)}")

    if args.dry_run:
        return

    root.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_cell, cell, root) for cell in cells]
        for future in as_completed(futures):
            cell, rc = future.result()
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[{status}] {cell.name}")
            if rc != 0:
                failed.append(cell.name)

    if failed:
        sys.exit(f"{len(failed)} cell(s) failed: {failed}")


if __name__ == "__main__":
    main()
