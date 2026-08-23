"""Eval-call (task-model F) cost/speed benchmark: candidates to replace gpt-oss-120b.

Each arm replays the IDENTICAL candidate-evaluation workload from the aug4_mixed
bt3gb run -- warm-start its SHIPPED beliefs+perception with --max-nodes 1,
so the optimizer evaluates only the seed candidate (ID + contrastive-FD on the split),
proposes nothing, and ships the seed into the test50 ID eval + raw-frame ref. Only
the task model + its provider pin vary; prompts, transitions, concurrency (48),
hedge/timeout defaults are shared. Reflection model is set to the same candidate
but is never called at a 1-node (seed-only) budget.

Measures per arm: wall time, real OpenRouter cost (usage.include passthrough),
test50 ID-set accuracy (decode-quality gate), hedge/retry counts.

Arms (fastest hosting per 2026-08-06 OpenRouter endpoint scan):
  gpt-oss-120b @ cerebras,groq,sambanova   -- current prod config (baseline)
  gpt-oss-20b  @ groq                      -- only fast host; $0.075/$0.30 per M
  qwen3.7-flash @ alibaba                  -- only host; $0.03/$0.13 per M, 1M ctx
  nemotron-3-nano-30b-a3b @ crusoe,deepinfra,novita -- all three at $0.05/$0.20
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_RUN = ROOT / "logs/aug4_mixed/bt3gb_strat30_seed1"
OUT_ROOT = ROOT / "logs/aug6_evalmodel_bench"

ARMS = [
    ("gptoss120b_cerebras", "openai/gpt-oss-120b", "cerebras,groq,sambanova"),
    ("gptoss20b_groq", "openai/gpt-oss-20b", "groq"),
    ("qwen37flash_alibaba", "qwen/qwen3.7-flash", "alibaba"),
    ("nemotron3nano_multi", "nvidia/nemotron-3-nano-30b-a3b", "crusoe,deepinfra,novita"),
]

PARSE = {
    "test_acc": r"CLEAN test acc \(inverse\) = ([\d.]+)",
    "id_set": r"ID-set: hit_rate=([\d.]+) mean_set_size=([\d.]+) mean_loss=([\d.]+) strict=([\d.]+)",
    "f_cost": r"F \(task_lm\) cost=\$([\d.]+) \((\d+) fresh F evals",
    "requests": r"LLM requests=(\d+) hedged=(\d+) hedge_wins=(\d+) retries=(\d+)",
}


def run_arm(name: str, model: str, pin: str) -> dict:
    outd = OUT_ROOT / name
    if outd.exists():
        raise SystemExit(f"refusing to overwrite existing run: {outd}")
    cmd = list(json.loads((SRC_RUN / "launch.json").read_text())["cmd"])
    cmd[0] = sys.executable
    cmd[cmd.index("--out-dir") + 1] = str(outd)
    cmd[cmd.index("--task-model") + 1] = model
    cmd[cmd.index("--reflection-model") + 1] = model
    # seed-only warm-start: budget = 1 node. Handle both the historical
    # (--max-metric-calls) and current (--max-nodes) budget flag names.
    _bi = cmd.index("--max-metric-calls") if "--max-metric-calls" in cmd else cmd.index("--max-nodes")
    cmd[_bi], cmd[_bi + 1] = "--max-nodes", "1"
    cmd[cmd.index("--task-provider-order") + 1] = pin
    cmd[cmd.index("--reflection-provider-order") + 1] = pin
    cmd[cmd.index("--start-perception") + 1] = str(
        SRC_RUN / "best_perception_gepa_seed1.py"
    )
    cmd += ["--start-beliefs", str(SRC_RUN / "best_beliefs_gepa_seed1.txt")]
    outd.mkdir(parents=True)
    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=pin)
    t0 = time.perf_counter()
    with (outd / "stdout.txt").open("w") as stdout:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
    wall = time.perf_counter() - t0
    text = (outd / "stdout.txt").read_text()
    res = {
        "arm": name,
        "model": model,
        "pin": pin,
        "wall_s": round(wall, 1),
        "returncode": proc.returncode,
    }
    for key, pat in PARSE.items():
        m = re.search(pat, text)
        res[key] = [float(g) for g in m.groups()] if m else None
    (outd / "launch.json").write_text(
        json.dumps({"cmd": cmd, "pin": pin, "result": res}, indent=2) + "\n"
    )
    return res


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    results = []
    for name, model, pin in ARMS:
        print(f"[bench] running {name} ({model} @ {pin}) ...", flush=True)
        res = run_arm(name, model, pin)
        results.append(res)
        print(f"[bench] {name}: {res}", flush=True)
        (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'arm':<22} {'wall(s)':>8} {'F cost($)':>10} {'F evals':>8} "
          f"{'test_acc':>9} {'strict':>7} {'hedged':>7} {'retries':>8}")
    for r in results:
        cost, evals = (r["f_cost"] or [float("nan")] * 2)[:2]
        acc = (r["test_acc"] or [float("nan")])[0]
        strict = (r["id_set"] or [float("nan")] * 4)[3]
        hedged = retries = float("nan")
        if r["requests"]:
            hedged, retries = r["requests"][1], r["requests"][3]
        print(f"{r['arm']:<22} {r['wall_s']:>8.0f} {cost:>10.4f} {evals:>8.0f} "
              f"{acc:>9.2f} {strict:>7.2f} {hedged:>7.0f} {retries:>8.0f}")


if __name__ == "__main__":
    main()
