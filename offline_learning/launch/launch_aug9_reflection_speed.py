"""A/B for the two reflection speedups against the aug8 hard-min config.

logs/aug8_hardmin_gptoss20b took 55-100 min per game for 30 nodes, and roughly half of
that wall sits in the reflection stack: per node the loop makes ONE diagnosis call and
ONE proposer call, both serial and both blocking the child eval.

Replaying that run's OWN logged prompts (logs/aug8_reflection_probe) showed the cause is
NOT a slow endpoint -- dsv4flash serves ~105 tok/s, in line with the field. It is that
dsv4flash spends 76-97% of its output on HIDDEN REASONING: ~10.6k reasoning tokens to
emit a ~320-token diagnosis. Measured medians on the real prompts:

  dsv4flash (aug8 default)          93.5s   $0.0023/call   6/6 valid
  dsv4flash + reasoning disabled    17.5s   $0.0006/call   6/6 valid
  ling-3.0-flash @novita nothink     2.8s   $0.0004/call   6/6 valid
  gpt-oss-120b @cerebras +low        1.7s   $0.0071/call   6/6 valid

Reflection is only ~12% of the run's DOLLAR cost ($0.05-0.10 of $0.44-0.81), so even the
3x-pricier gpt-oss arm leaves reflection well under F.

Arms (each = the exact aug8 canonical config with ONE thing changed):

  control      aug8 config re-run for matched provider conditions (dsv4flash, B=1)
  dsnothink    SAME model, thinking off -- the one-flag fix, isolates the reasoning waste
  gptoss120b   fastest proposer (cerebras)
  ling30flash  cheapest + near-fastest proposer
  batch3       aug8 reflection model, --propose-batch 3 (concurrent iterations)
  fastbatch3   gpt-oss-120b + --propose-batch 3

Node budget, scoring, data and F model are identical across arms, so test accuracy is
comparable and the only expected difference is wall-clock (plus whatever quality delta
the reflection model itself brings).

BLOCKED DESIGN: --seeds runs every arm on the same (game, seed) cells. A seed fixes the
train split AND the selector RNG, so within a block the arms start from identical data and
an identical first draw -- the delta is paired and the block absorbs the between-seed
variance that swamps a single-run comparison (repro noise on an UNCHANGED config is ~0.03;
5 games x 5 seeds = 25 blocks puts the paired SE near 0.02).

The speedup pays for its own validation: control is ~90 min/run, dsnothink ~35, gptoss120b
~25, so the same 25 blocks costs ~37 h of serial compute instead of ~94 h.

Runs fan out as subprocesses; skip-if-complete, so re-invoking resumes the matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "logs/aug9_reflection_speed"
SRC_ROOT = ROOT / "logs/aug8_hardmin_gptoss20b"
GAMES = ["bt3gb", "n2ntd"]

DSV4 = "deepseek/deepseek-v4-flash"
DSV4_PIN = "deepseek,baidu,fireworks"
NOTHINK = '{"enabled": false}'
EFFLOW = '{"effort": "low"}'

EFFMED = '{"effort": "medium"}'

# arm -> (reflection model, provider order, reasoning json or None, hedge_s, timeout_s, extra)
# hedge/timeout are PER ARM: a hedge below the model's own median call time fires on most
# calls and just doubles spend, so each is set from that arm's measured median
# (logs/aug8_reflection_probe, logs/aug9_fast_tier_probe): dsv4flash-think 93.5s,
# dsv4flash-nothink 17.5s, gpt-oss-120b+med 3.3s, gpt-5.6-luna-pro 55.2s.
ARMS = {
    "control":     (DSV4, DSV4_PIN, None,    120, 300, []),
    "dsnothink":   (DSV4, DSV4_PIN, NOTHINK,  30, 120, []),
    "gptoss120b":  ("openai/gpt-oss-120b", "cerebras,groq,sambanova", EFFLOW, 30, 120, []),
    "gptoss120bmed": ("openai/gpt-oss-120b", "cerebras,groq,sambanova", EFFMED, 30, 120, []),
    "lunapro":     ("openai/gpt-5.6-luna-pro", "", EFFMED, 150, 400, []),
    "ling30flash": ("inclusionai/ling-3.0-flash", "novita,deepinfra", NOTHINK, 30, 120, []),
    "batch3":      (DSV4, DSV4_PIN, None, 120, 300, ["--propose-batch", "3"]),
    "batch5":      (DSV4, DSV4_PIN, None, 120, 300, ["--propose-batch", "5"]),
    "fastbatch3":  ("openai/gpt-oss-120b", "cerebras,groq,sambanova", EFFLOW, 30, 120,
                    ["--propose-batch", "3"]),
}


def build_cmd(game: str, arm: str, seed: int, outd: Path, resume: bool) -> list[str]:
    """Start from the game's own aug8 launch.json so data flags / scoring config are
    byte-identical, then swap only the reflection routing (+ arm extras)."""
    cmd = list(json.loads((SRC_ROOT / f"{game}_seed1/launch.json").read_text())["cmd"])
    cmd[0] = sys.executable
    model, pin, reasoning, hedge, timeout, extra = ARMS[arm]
    cmd[cmd.index("--reflection-model") + 1] = model
    cmd[cmd.index("--reflection-provider-order") + 1] = pin
    cmd[cmd.index("--out-dir") + 1] = str(outd)
    cmd[cmd.index("--seed") + 1] = str(seed)
    cmd[cmd.index("--reflection-hedge-delay") + 1] = str(hedge)
    cmd[cmd.index("--reflection-timeout") + 1] = str(timeout)
    if reasoning is not None:
        cmd += ["--reflection-reasoning-json", reasoning]
    cmd += extra
    if resume:
        cmd.append("--resume")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--seeds", default="1", help="comma-separated; each (game, seed) is a block")
    ap.add_argument("--max-parallel", type=int, default=6,
                    help="subprocesses in flight; each opens --concurrency F requests")
    ap.add_argument("--extra-flags", default="",
                    help="space-separated flags appended to every run (e.g. "
                         "'--no-analysis-memo' to keep an arm a single-variable change "
                         "against a pre-memo baseline)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER="groq")  # F stays gpt-oss-20b@groq
    live: list[subprocess.Popen] = []
    for seed in [int(x) for x in args.seeds.split(",")]:
        for game in args.games.split(","):
            for arm in args.arms.split(","):
                outd = OUT_ROOT / f"{game}_s{seed}_{arm}"
                if (outd / f"test_summary_rexpure_seed{seed}.json").exists():
                    print(f"skip  {game}/s{seed}/{arm}: already complete")
                    continue
                resume = (outd / f"rexpure_run_seed{seed}" / "resume_state.json").exists()
                cmd = build_cmd(game, arm, seed, outd, resume) + args.extra_flags.split()
                if args.dry_run:
                    print(f"[dry-run] {game}/s{seed}/{arm}:\n  {' '.join(cmd)}\n")
                    continue
                while len([p for p in live if p.poll() is None]) >= args.max_parallel:
                    time.sleep(15)
                outd.mkdir(parents=True, exist_ok=True)
                with (outd / "stdout.txt").open("a" if resume else "w") as f:
                    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                live.append(p)
                (outd / "launch.json").write_text(json.dumps(
                    {"game": game, "seed": seed, "arm": arm, "pid": p.pid, "cmd": cmd,
                     "mode": "RESUME" if resume else "fresh"}, indent=2) + "\n")
                (outd / "pid").write_text(f"{p.pid}\n")
                print(f"{'RESUME' if resume else 'fresh':6} {game}/s{seed}/{arm}: "
                      f"pid={p.pid} -> {outd}")


if __name__ == "__main__":
    main()
