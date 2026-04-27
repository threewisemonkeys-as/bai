"""Launch the six artifact-eval jobs for the apr24 run in parallel.

Evaluates learned checkpoints vs an empty baseline across ARC-AGI,
MiniHack, and Autumn.  All six jobs run concurrently; stdout/stderr
for each lands in <out>/<job_name>/{stdout,stderr}.log.

Usage:
    uv run launch_eval.py
    uv run launch_eval.py --mock                        # 1 repeat, mock LLM
    uv run launch_eval.py --out logs/dev/my_eval
    uv run launch_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Source runs
# ---------------------------------------------------------------------------

_ROOT = "logs/dev/apr24/20260424-190547"
_STEM = "2026-04-24_19-05-52_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn"

ARC_RUN = f"{_ROOT}/eb_learn__arc_agi__gemini-2p5-flash/{_STEM}"
MH_RUN  = f"{_ROOT}/eb_learn__minihack__gemini-2p5-flash/{_STEM}"
AUT_RUN = f"{_ROOT}/eb_learn__autumn__gemini-2p5-flash/{_STEM}"

DEFAULT_OUT = "logs/dev/apr25/artifact_eval_apr24_20260424-190547"


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

def build_jobs(out: str, mock: bool) -> list[dict]:
    mock_args = ["+artifact_eval.mock_mode=true"] if mock else []

    def job(name: str, repeats: int, *extra: str) -> dict:
        output_dir = next(a.split("=", 1)[1] for a in extra if a.startswith("eval.output_dir="))
        n = 1 if mock else repeats
        cmd = [
            "uv", "run", "eval_stepwise_eb_artifacts.py",
            f"+artifact_eval.repeats={n}",
            "+artifact_eval.parallel_workers=10",
            *extra,
            *mock_args,
        ]
        return {"name": name, "cmd": cmd, "output_dir": output_dir}

    return [
        job("arc/g049", 5,
            f"+artifact_eval.source_run={ARC_RUN}",
            "+artifact_eval.steps=[49]",
            "+artifact_eval.seed_base=1100",
            f"eval.output_dir={out}/arc_agi/g049"),
        job("minihack/g049", 10,
            f"+artifact_eval.source_run={MH_RUN}",
            "+artifact_eval.steps=[49]",
            "+artifact_eval.seed_base=1300",
            f"eval.output_dir={out}/minihack/g049"),
        job("autumn/g097", 5,
            f"+artifact_eval.source_run={AUT_RUN}",
            "+artifact_eval.steps=[97]",
            "+artifact_eval.seed_base=1500",
            "+artifact_eval.autumn_eval_task_types=[planning]",
            "+artifact_eval.autumn_eval_max_steps=100",
            f"eval.output_dir={out}/autumn/g097"),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_job(job: dict) -> tuple[str, int]:
    log_dir = Path(job["output_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "cmd.txt").write_text(" ".join(job["cmd"]) + "\n")
    with (
        open(log_dir / "stdout.log", "w") as so,
        open(log_dir / "stderr.log", "w") as se,
    ):
        rc = subprocess.run(job["cmd"], stdout=so, stderr=se).returncode
    return job["name"], rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"Output root (default: {DEFAULT_OUT})")
    p.add_argument("--mock", action="store_true",
                   help="1 repeat + mock LLM — fast smoke test")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands and exit without running")
    args = p.parse_args()

    jobs = build_jobs(args.out, args.mock)

    print(f"Output root : {args.out}")
    print(f"Jobs ({len(jobs)}):")
    for j in jobs:
        print(f"  {j['name']}  ->  {j['output_dir']}")
        if args.dry_run:
            print(f"    {' '.join(j['cmd'])}")
    if args.dry_run:
        return

    t0 = time.monotonic()
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {ex.submit(run_job, j): j["name"] for j in jobs}
        print(f"\nLaunched {len(jobs)} jobs.", flush=True)
        for fut in as_completed(futures):
            name, rc = fut.result()
            elapsed = time.monotonic() - t0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{status:12s}] {name}  ({elapsed:.0f}s)", flush=True)
            if rc != 0:
                failed.append(name)

    total = time.monotonic() - t0
    print(f"\nDone in {total:.0f}s — {len(jobs) - len(failed)}/{len(jobs)} succeeded.")
    if failed:
        print("Failed:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
