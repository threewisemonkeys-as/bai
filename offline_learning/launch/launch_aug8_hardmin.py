"""Five hard-min rex_pure runs (dq8gc, bt3gb, n2ntd, s2kt7, 83wkq) with the
gpt-oss-20b@groq evaluator (effort=low) + deepseek-v4-flash reflection, max-nodes 30.

Config is canonical: one fixed hard-min flag set (composite min + contrastive-fd +
cfd-hard-decoys + id-set-loss + rex_pure C=5 + analyze-mistakes, context-k 9,
train 30 / test 50), and ONLY each game's data flags are transplanted from its
source launch.json (so the optimizer config is identical across games; no drift
between the aug5 and clean_sweep sources). Old prototypes/perc_invdyn/ data paths
are repointed to offline_learning/. s2kt7 uses the seed5 (non-degenerate) data.

RESUMABLE + re-runnable: for each game, if test_summary_rexpure_seed1.json exists
the game is skipped (done); else if a resume_state.json checkpoint exists the run
is relaunched with --resume (continues from its last node); else it starts fresh.
Re-invoke this script after any interruption to pick up exactly where it left off.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "logs/aug8_hardmin_gptoss20b"
TASK_PIN = "groq"                       # gpt-oss-20b served by groq
REFL_PIN = "deepseek,baidu,fireworks"
DATA_FLAGS = ["--run", "--test-run", "--context-source-run",
              "--test-context-source-run", "--actions"]

SOURCES = {
    "dq8gc": "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50_fulltestctx_phase1/dq8gc_seed1",
    "bt3gb": "logs/aug5_rexpure/bt3gb_strat30_seed1",
    "n2ntd": "logs/aug5_rexpure/n2ntd_seed1",
    "s2kt7": "logs/aug5_rexpure/s2kt7_seed5data",
    "83wkq": "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50_fulltestctx_phase2/83wkq_seed1",
}


def repoint(tok: str) -> str:
    return tok.replace(str(ROOT / "prototypes/perc_invdyn"), str(ROOT / "offline_learning"))


def data_args(game: str) -> list[str]:
    cmd = json.loads((ROOT / SOURCES[game] / "launch.json").read_text())["cmd"]
    out = []
    for flag in DATA_FLAGS:
        if flag in cmd:
            out += [flag, repoint(cmd[cmd.index(flag) + 1])]
    return out


def build_cmd(game: str, outd: Path, resume: bool) -> list[str]:
    cmd = [
        sys.executable, str(ROOT / "offline_learning/rexpure_optimize.py"),
        *data_args(game),
        "--train-n", "30", "--test-n", "50", "--k-choices", "5",
        "--context-k", "9", "--concurrency", "12", "--seed", "1",
        "--max-nodes", "30",
        "--fd-scorer", "none",
        "--task-model", "openai/gpt-oss-20b",
        "--task-provider-order", TASK_PIN,
        "--task-reasoning-json", '{"effort": "low"}',
        "--reflection-model", "deepseek/deepseek-v4-flash",
        "--reflection-provider-order", REFL_PIN,
        "--reflection-hedge-delay", "120", "--reflection-timeout", "300",
        "--analyze-mistakes",
        "--start-perception", str(ROOT / "offline_learning/autumn_seed_perception.py"),
        "--id-set-loss", "--id-eps", "0.1",
        "--composite", "min", "--contrastive-fd", "--cfd-hard-decoys",
        "--rex-c", "5",
        "--out-dir", str(outd),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def main() -> None:
    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=TASK_PIN)
    for game in SOURCES:
        outd = OUT_ROOT / f"{game}_seed1"
        done = (outd / "test_summary_rexpure_seed1.json").exists()
        if done:
            print(f"skip  {game}: already complete ({outd.name}/test_summary_rexpure_seed1.json)")
            continue
        ckpt = (outd / "rexpure_run_seed1" / "resume_state.json").exists()
        outd.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(game, outd, resume=ckpt)
        mode = "RESUME" if ckpt else "fresh"
        with (outd / "stdout.txt").open("a" if ckpt else "w") as f:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        (outd / "launch.json").write_text(json.dumps(
            {"game": game, "arm": "hardmin_gptoss20b_groq_dsv4flash_refl",
             "pid": p.pid, "cmd": cmd, "mode": mode, "max_nodes": 30,
             "task_pin": TASK_PIN, "source": SOURCES[game]}, indent=2) + "\n")
        (outd / "pid").write_text(f"{p.pid}\n")
        print(f"{mode:6} {game}: pid={p.pid} -> {outd}")


if __name__ == "__main__":
    main()
