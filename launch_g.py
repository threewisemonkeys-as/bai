"""Greedy (G) experiment launcher.

Single-phase wrapper that runs the existing stepwise scripts with the
``performance`` agent goal. Restricted to ``minihack`` and ``arc_agi``
(``autumn`` is intrinsically two-phase and excluded by spec).

Outputs land under ``<log_dir>/<tag>/<cell_name>/`` with stdout.log,
stderr.log, and cmd.txt — same shape as ``launch.py`` /
``launch_baselines.py``.

Usage:
    uv run launch_g.py --log-dir logs/greedy --dry-run
    uv run launch_g.py --log-dir logs/greedy --scripts simple --envs minihack
    uv run launch_g.py --log-dir logs/greedy --scripts openhands,simple \
        --envs minihack --repeats 5 --seed-base 1700 --parallel 10
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

import launch as _eb_launch
import launch_baselines as _baseline_launch


SCRIPT_FILES = {
    "eb_learn":  "stepwise_eb_learn.py",
    "simple":    "simple_stepwise.py",
    "openhands": "openhands_stepwise.py",
}

ENVS = ["minihack", "arc_agi"]  # autumn excluded by spec.

MODEL_IDS = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "mock":             "google/gemini-2.5-flash",
}
MODELS = list(MODEL_IDS)


NUM_STEPS_KEYS = {
    "eb_learn":  "eval.evolve.n_environment_steps",
    "openhands": "eval.openhands.n_environment_steps",
    "simple":    "eval.simple.n_environment_steps",
}


@dataclass
class Cell:
    script: str
    env: str
    model: str
    overrides: dict = field(default_factory=dict)
    repeat_idx: int | None = None

    @property
    def name(self) -> str:
        base = f"{self.script}__{self.env}__{self.model}".replace(".", "p")
        if self.repeat_idx is None:
            return base
        return f"{base}__repeat_{self.repeat_idx:03d}"

    @property
    def config_name(self) -> str:
        return f"{self.script}__{self.env}__{self.model}".replace(".", "p")


def _cell_overrides(script: str, env: str, model: str) -> dict:
    if script == "eb_learn":
        ov: dict = {
            "envs.names": env,
            **_eb_launch.model_overrides(model, "eb_learn"),
            **_eb_launch.EB_LEARN_DEFAULT,
            **_eb_launch.EB_LEARN_OVERRIDES.get((env, model), {}),
        }
        # Greedy runs should rely on the LLM probe selector for experiment
        # generation rather than the B-diff question scoring path.
        ov["eval.evolve.question_scoring_method"] = "llm_trim"
    else:
        ov = {
            "envs.names": env,
            **_baseline_launch.model_overrides(model, script),
            **_baseline_launch._baseline_env_overrides(env, model, script),
        }
    ov["eval.agent_goal_mode"] = "performance"
    return ov


def build_cells() -> list[Cell]:
    cells: list[Cell] = []
    for script, env, model in itertools.product(SCRIPT_FILES, ENVS, MODELS):
        cells.append(Cell(
            script=script,
            env=env,
            model=model,
            overrides=_cell_overrides(script, env, model),
        ))
    return cells


def parse_csv(s):
    return None if s is None else [x.strip() for x in s.split(",") if x.strip()]


def _validate(values, valid, label):
    if values is None:
        return
    bad = [v for v in values if v not in valid]
    if bad:
        sys.exit(f"Unknown {label}: {bad}. Valid: {sorted(valid)}")


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


def apply_extra_overrides(cells: list[Cell], extra_overrides: list[str] | None) -> None:
    if not extra_overrides:
        return
    for raw in extra_overrides:
        if "=" not in raw:
            sys.exit(f"Invalid override {raw!r}; expected key=value.")
        key, value = raw.split("=", 1)
        if not key:
            sys.exit(f"Invalid override {raw!r}; override key is empty.")
        for cell in cells:
            cell.overrides[key] = value


def _has_override(extra_overrides: list[str] | None, key: str) -> bool:
    if not extra_overrides:
        return False
    return any(raw.split("=", 1)[0] == key for raw in extra_overrides if "=" in raw)


def expand_repeats(cells: list[Cell], repeats: int, seed_base: int | None) -> list[Cell]:
    if repeats < 1:
        sys.exit("--repeats must be >= 1.")
    if repeats == 1 and seed_base is None:
        return cells

    expanded: list[Cell] = []
    for cell in cells:
        for repeat_idx in range(repeats):
            overrides = dict(cell.overrides)
            if seed_base is not None:
                overrides["envs.env_kwargs.seed"] = seed_base + repeat_idx
            expanded.append(Cell(
                script=cell.script,
                env=cell.env,
                model=cell.model,
                overrides=overrides,
                repeat_idx=repeat_idx,
            ))
    return expanded


def run_cell(cell: Cell, root: Path) -> tuple[Cell, int]:
    out_dir = root / cell.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(cell, out_dir)
    (out_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")
    with open(out_dir / "stdout.log", "w") as so, open(out_dir / "stderr.log", "w") as se:
        rc = subprocess.run(cmd, stdout=so, stderr=se).returncode
    return cell, rc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scripts", type=parse_csv, default=None,
                   help=f"Subset of {list(SCRIPT_FILES)}")
    p.add_argument("--envs", type=parse_csv, default=None, help=f"Subset of {ENVS}")
    p.add_argument("--models", type=parse_csv, default=None, help=f"Subset of {MODELS}")
    p.add_argument("--log-dir", type=Path, required=True,
                   help="Root directory; cells land in <log_dir>/<tag>/<cell_name>/")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Global override for n_environment_steps across all cells")
    p.add_argument("--parallel", type=int, default=1,
                   help="Max concurrent cells (default: 1)")
    p.add_argument("--repeats", type=int, default=1,
                   help="Number of repeated runs per selected config (default: 1)")
    p.add_argument("--seed-base", type=int, default=None,
                   help="If set, use envs.env_kwargs.seed=seed_base+repeat_idx")
    p.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--override", action="append", default=None,
                   help="Extra Hydra override key=value. May be repeated.")
    p.add_argument("--dry-run", action="store_true")
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
    apply_extra_overrides(cells, args.override)
    if args.seed_base is not None and _has_override(args.override, "envs.env_kwargs.seed"):
        sys.exit("Use either --seed-base or --override envs.env_kwargs.seed=..., not both.")
    cells = expand_repeats(cells, args.repeats, args.seed_base)

    root = args.log_dir / args.tag
    print(f"Greedy matrix root: {root}")
    print(f"Runs ({len(cells)}), repeats={args.repeats}, parallel={max(1, args.parallel)}:")
    for c in cells:
        print(f"  - {c.name}")
        if c.repeat_idx is not None:
            print(f"      config={c.config_name}")
            print(f"      repeat_idx={c.repeat_idx}")
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
