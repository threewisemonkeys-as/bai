"""ExploreEval (EE) two-phase experiment launcher.

For each (script, env, model) cell, runs two subprocesses sequentially:

  Phase 1 (explore): the underlying stepwise script with the *dynamics* agent
  goal, saving artifacts (beliefs+perception, message history, or workspace).

  Phase 2 (eval): the matching ``eval_*_artifacts.py`` script with the
  env-specific *eval* goal, loading the phase-1 artifacts. Defaults to the
  final artifact; ``--eval-steps`` selects intermediate ones.

Outputs land under ``<log_dir>/<tag>/<cell_name>/{phase1_learn,phase2_eval}/``.

Usage:
    uv run launch_ee.py --log-dir logs/ee --dry-run
    uv run launch_ee.py --log-dir logs/ee --scripts simple --envs minihack
    uv run launch_ee.py --log-dir logs/ee --eval-steps 10,20,final \
        --checkpoint-interval 10
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
from goal_prompts import eval_goal_mode_for_env


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

SCRIPT_PAIRS: dict[str, tuple[str, str]] = {
    "eb_learn":  ("stepwise_eb_learn.py",  "eval_stepwise_eb_artifacts.py"),
    "simple":    ("simple_stepwise.py",    "eval_simple_artifacts.py"),
    "openhands": ("openhands_stepwise.py", "eval_openhands_artifacts.py"),
}

ENVS = ["minihack", "arc_agi", "autumn"]

MODEL_IDS = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "mock":             "google/gemini-2.5-flash",
}
MODELS = list(MODEL_IDS)


PHASE2_DEFAULTS = {
    "repeats":          5,
    "parallel_workers": 10,
    "seed_base":        1700,
    "autumn_max_steps": 100,
}


NUM_STEPS_KEYS = {
    "eb_learn":  "eval.evolve.n_environment_steps",
    "openhands": "eval.openhands.n_environment_steps",
    "simple":    "eval.simple.n_environment_steps",
}

CHECKPOINT_KEYS = {
    "simple":    "eval.simple.history_checkpoint_interval",
    "openhands": "eval.openhands.workspace_checkpoint_interval",
    # eb_learn snapshots per step natively.
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


def _phase1_overrides(script: str, env: str, model: str) -> dict:
    if script == "eb_learn":
        ov: dict = {
            "envs.names": env,
            **_eb_launch.model_overrides(model, "eb_learn"),
            **_eb_launch.EB_LEARN_DEFAULT,
            **_eb_launch.EB_LEARN_OVERRIDES.get((env, model), {}),
        }
        # Phase 2 handles the eval; suppress the in-script frozen evals so the
        # dynamics-goal agent doesn't accidentally run them with the wrong goal.
        ov["eval.evolve.autumn_eval_after_learn"] = False
        ov["eval.evolve.frozen_eval_after_learn"] = False
    else:
        ov = {
            "envs.names": env,
            **_baseline_launch.model_overrides(model, script),
            **_baseline_launch._baseline_env_overrides(env, model, script),
        }
    ov["eval.agent_goal_mode"] = "dynamics"
    return ov


def build_cells() -> list[Cell]:
    cells: list[Cell] = []
    for script, env, model in itertools.product(SCRIPT_PAIRS, ENVS, MODELS):
        cells.append(Cell(
            script=script,
            env=env,
            model=model,
            overrides=_phase1_overrides(script, env, model),
        ))
    return cells


# ---------------------------------------------------------------------------
# Phase 2 override construction
# ---------------------------------------------------------------------------

def _phase2_overrides(
    cell: Cell,
    *,
    source_run: Path,
    output_dir: Path,
    eval_steps: list[str],
    eval_every_n: int | None,
    repeats: int,
    parallel_workers: int,
    seed_base: int,
) -> list[str]:
    args = [
        f"+artifact_eval.source_run={source_run}",
        f"+artifact_eval.steps=[{','.join(eval_steps)}]",
        f"+artifact_eval.repeats={repeats}",
        f"+artifact_eval.parallel_workers={parallel_workers}",
        f"+artifact_eval.seed_base={seed_base}",
        f"+artifact_eval.agent_goal_mode={eval_goal_mode_for_env(cell.env)}",
        f"eval.output_dir={output_dir}",
    ]
    if eval_every_n is not None:
        args.append(f"+artifact_eval.every_n={eval_every_n}")
    if cell.model == "mock":
        args.append("+artifact_eval.mock_mode=true")
    if cell.env == "autumn":
        args.append("+artifact_eval.autumn_eval_task_types=[planning]")
        args.append(f"+artifact_eval.autumn_eval_max_steps={PHASE2_DEFAULTS['autumn_max_steps']}")
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _phase1_cmd(cell: Cell, output_dir: Path) -> list[str]:
    overrides = {**cell.overrides, "eval.output_dir": str(output_dir)}
    args = [f"{k}={_fmt(v)}" for k, v in overrides.items()]
    script_file, _ = SCRIPT_PAIRS[cell.script]
    return ["uv", "run", script_file, *args]


def _phase2_cmd(cell: Cell, args: list[str]) -> list[str]:
    _, eval_script = SCRIPT_PAIRS[cell.script]
    return ["uv", "run", eval_script, *args]


def _resolve_phase1_run_dir(phase1_root: Path) -> Path:
    """Return the single hydra-injected child dir under phase1_learn/.

    setup_run writes ``<output_dir_base>/<timestamp>_<run_name_suffix>``. If
    multiple children exist (retried run), pick the most recently modified.
    """
    children = [p for p in phase1_root.iterdir() if p.is_dir()]
    if not children:
        raise FileNotFoundError(f"No phase-1 run dir found under {phase1_root}")
    children.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return children[0]


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(
    cell: Cell,
    root: Path,
    *,
    eval_steps: list[str],
    eval_every_n: int | None,
    repeats: int,
    parallel_workers: int,
    seed_base: int,
) -> tuple[Cell, str, int]:
    cell_root = root / cell.name
    cell_root.mkdir(parents=True, exist_ok=True)

    # Phase 1
    phase1_dir = cell_root / "phase1_learn"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    phase1_cmd = _phase1_cmd(cell, phase1_dir)
    (cell_root / "phase1_cmd.txt").write_text(" ".join(phase1_cmd) + "\n")
    with open(cell_root / "phase1_stdout.log", "w") as so, \
         open(cell_root / "phase1_stderr.log", "w") as se:
        rc = subprocess.run(phase1_cmd, stdout=so, stderr=se).returncode
    if rc != 0:
        return cell, "phase1", rc

    # Phase 2
    try:
        source_run = _resolve_phase1_run_dir(phase1_dir)
    except FileNotFoundError as e:
        with open(cell_root / "phase2_stderr.log", "w") as se:
            se.write(f"Could not resolve phase-1 run dir: {e}\n")
        return cell, "phase2-resolve", 1

    phase2_dir = cell_root / "phase2_eval"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    phase2_args = _phase2_overrides(
        cell,
        source_run=source_run,
        output_dir=phase2_dir,
        eval_steps=eval_steps,
        eval_every_n=eval_every_n,
        repeats=repeats,
        parallel_workers=parallel_workers,
        seed_base=seed_base,
    )
    phase2_cmd = _phase2_cmd(cell, phase2_args)
    (cell_root / "phase2_cmd.txt").write_text(" ".join(phase2_cmd) + "\n")
    with open(cell_root / "phase2_stdout.log", "w") as so, \
         open(cell_root / "phase2_stderr.log", "w") as se:
        rc = subprocess.run(phase2_cmd, stdout=so, stderr=se).returncode
    if rc != 0:
        return cell, "phase2", rc
    return cell, "ok", 0


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


def filter_cells(cells, scripts, envs, models) -> list[Cell]:
    def match(values, v):
        return values is None or v in values
    return [c for c in cells
            if match(scripts, c.script)
            and match(envs,    c.env)
            and match(models,  c.model)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scripts", type=parse_csv, default=None,
                   help=f"Subset of {list(SCRIPT_PAIRS)}")
    p.add_argument("--envs", type=parse_csv, default=None, help=f"Subset of {ENVS}")
    p.add_argument("--models", type=parse_csv, default=None, help=f"Subset of {MODELS}")
    p.add_argument("--log-dir", type=Path, required=True,
                   help="Root directory; cells land under <log_dir>/<tag>/<cell>/")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Phase-1 n_environment_steps override across all cells")
    p.add_argument("--checkpoint-interval", type=int, default=0,
                   help="Snapshot phase-1 simple history / openhands workspace every N steps")
    p.add_argument("--eval-steps", type=parse_csv, default=["final"],
                   help="Phase-2 +artifact_eval.steps list (default: final)")
    p.add_argument("--eval-every-n", type=int, default=None,
                   help="Phase-2 +artifact_eval.every_n")
    p.add_argument("--repeats", type=int, default=PHASE2_DEFAULTS["repeats"],
                   help=f"Phase-2 repeats per artifact (default: {PHASE2_DEFAULTS['repeats']})")
    p.add_argument("--phase2-parallel", type=int, default=PHASE2_DEFAULTS["parallel_workers"],
                   help="Phase-2 internal parallel_workers per cell")
    p.add_argument("--seed-base", type=int, default=PHASE2_DEFAULTS["seed_base"],
                   help="Phase-2 +artifact_eval.seed_base")
    p.add_argument("--parallel", type=int, default=1,
                   help="Max concurrent cells (default: 1)")
    p.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    _validate(args.scripts, SCRIPT_PAIRS, "script")
    _validate(args.envs,    ENVS,         "env")
    _validate(args.models,  MODELS,       "model")

    cells = filter_cells(build_cells(), args.scripts, args.envs, args.models)
    if not cells:
        sys.exit("No cells matched filters.")

    if args.num_steps is not None:
        for c in cells:
            c.overrides[NUM_STEPS_KEYS[c.script]] = args.num_steps

    if args.checkpoint_interval > 0:
        for c in cells:
            key = CHECKPOINT_KEYS.get(c.script)
            if key is not None:
                c.overrides[key] = args.checkpoint_interval

    root = args.log_dir / args.tag
    print(f"EE matrix root: {root}")
    print(f"Cells ({len(cells)}):")
    for c in cells:
        print(f"  - {c.name}")
        for k, v in c.overrides.items():
            print(f"      {k}={_fmt(v)}")
    print(f"Phase-2: steps={args.eval_steps} every_n={args.eval_every_n} "
          f"repeats={args.repeats} parallel={args.phase2_parallel} "
          f"seed_base={args.seed_base}")

    if args.dry_run:
        return

    root.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futures = [
            ex.submit(
                run_cell,
                c,
                root,
                eval_steps=args.eval_steps,
                eval_every_n=args.eval_every_n,
                repeats=args.repeats,
                parallel_workers=args.phase2_parallel,
                seed_base=args.seed_base,
            )
            for c in cells
        ]
        for fut in as_completed(futures):
            cell, stage, rc = fut.result()
            if rc == 0:
                print(f"[OK]   {cell.name}")
            else:
                print(f"[FAIL:{stage}({rc})] {cell.name}")
                failed.append((cell.name, stage))

    if failed:
        sys.exit(f"{len(failed)} cell(s) failed: {failed}")


if __name__ == "__main__":
    main()
