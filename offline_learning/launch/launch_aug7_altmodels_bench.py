"""Stage-B bench: can any OpenRouter model/provider beat gpt-oss-120b@cerebras
+effort-low on the eval-call role at BOTH cost and wall?

Stage A (probe_evalmodel_candidates.py over the 52 paper-competitive models from
the 2026-08-07 full-catalog endpoint scan) left five survivors in the Cerebras
latency class (p50 <= ~3s on the real windowed set-ID prompt):

  gpt-oss-20b @ groq + effort low     1.2s  $0.00035/call
  llama-4-scout @ groq                1.6s  $0.00049
  llama-3.1-8b @ groq                 1.7s  $0.00020
  gpt-5.6-luna + effort none          1.8s  $0.00044 (83-token answers)
  ling-3.0-flash + nothink            3.0s  $0.00024

Each runs the standard eval-only workload (aug6 bt3gb saved cmd sanitized for the
current rexpure CLI: --max-nodes 1 seed-only + test50 ID eval), decode quality is
the gate. Fresh gpt-oss-120b+low re-baseline included because the CLI budget flag
changed (--max-metric-calls -> --max-nodes) since the aug7_qwen_nothink numbers.
Reasoning overrides go through the newly wired --task-reasoning-json flags (no
wrapper). Results -> logs/aug7_altmodels_bench/.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _launch_util import sanitize_rexpure_cmd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC_LAUNCH = ROOT / "logs/aug6_evalmodel_bench/gptoss120b_cerebras/launch.json"
OUT_ROOT = ROOT / "logs/aug7_altmodels_bench"

EFFLOW = '{"effort": "low"}'
EFFNONE = '{"effort": "none"}'
NOTHINK = '{"enabled": false}'

# (name, model, provider order ("" = unpinned), reasoning json or None)
ARMS = [
    ("gptoss120b_efflow_rebase", "openai/gpt-oss-120b", "cerebras,groq,sambanova", EFFLOW),
    ("gptoss20b_groq_low", "openai/gpt-oss-20b", "groq", EFFLOW),
    ("llama4scout_groq", "meta-llama/llama-4-scout", "groq", None),
    ("llama31_8b_groq", "meta-llama/llama-3.1-8b-instruct", "groq", None),
    ("gpt56luna_effnone", "openai/gpt-5.6-luna", "", EFFNONE),
    ("ling30flash_nothink", "inclusionai/ling-3.0-flash", "novita,deepinfra", NOTHINK),
]

PARSE = {
    "test_acc": r"CLEAN test acc \(inverse\) = ([\d.]+)",
    "id_set": r"ID-set: hit_rate=([\d.]+) mean_set_size=([\d.]+) mean_loss=([\d.]+) strict=([\d.]+)",
    "f_cost": r"F \(task_lm\) cost=\$([\d.]+) \((\d+) fresh F evals",
    "requests": r"LLM requests=(\d+) hedged=(\d+) hedge_wins=(\d+) retries=(\d+)",
}


def build_cmd(outd: Path, model: str, pin: str, reasoning: str | None) -> list[str]:
    cmd = sanitize_rexpure_cmd(list(json.loads(SRC_LAUNCH.read_text())["cmd"]))
    cmd = [a.replace("/prototypes/perc_invdyn/", "/offline_learning/") for a in cmd]
    cmd[0] = sys.executable
    # budget flag renamed since the aug6 cmds were saved; value 1 = seed-only either way
    cmd[cmd.index("--max-metric-calls")] = "--max-nodes"
    cmd[cmd.index("--out-dir") + 1] = str(outd)
    cmd[cmd.index("--task-model") + 1] = model
    cmd[cmd.index("--reflection-model") + 1] = model
    cmd[cmd.index("--task-provider-order") + 1] = pin
    cmd[cmd.index("--reflection-provider-order") + 1] = pin
    if reasoning is not None:
        cmd += ["--task-reasoning-json", reasoning,
                "--reflection-reasoning-json", reasoning]
    return cmd


def run_arm(name: str, model: str, pin: str, reasoning: str | None) -> dict:
    outd = OUT_ROOT / name
    if outd.exists():
        raise SystemExit(f"refusing to overwrite existing run: {outd}")
    cmd = build_cmd(outd, model, pin, reasoning)
    outd.mkdir(parents=True)
    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=pin)
    t0 = time.perf_counter()
    with (outd / "stdout.txt").open("w") as stdout:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                              stdout=stdout, stderr=subprocess.STDOUT)
    res = {
        "arm": name, "model": model, "pin": pin, "reasoning": reasoning,
        "wall_s": round(time.perf_counter() - t0, 1), "returncode": proc.returncode,
    }
    text = (outd / "stdout.txt").read_text()
    for key, pat in PARSE.items():
        m = re.search(pat, text)
        res[key] = [float(g) for g in m.groups()] if m else None
    (outd / "launch.json").write_text(
        json.dumps({"cmd": cmd, "result": res}, indent=2) + "\n"
    )
    return res


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    results = []
    for name, model, pin, reasoning in ARMS:
        done = OUT_ROOT / name / "launch.json"
        if done.exists():
            results.append(json.loads(done.read_text())["result"])
            print(f"[bench] {name}: already done, reusing result", flush=True)
            continue
        print(f"[bench] running {name} ...", flush=True)
        res = run_arm(name, model, pin, reasoning)
        results.append(res)
        print(f"[bench] {name}: {res}", flush=True)
        (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'arm':<26} {'wall(s)':>8} {'F cost($)':>10} {'test_acc':>9} "
          f"{'strict':>7} {'hedged':>7} {'retries':>8}")
    for r in results:
        cost = (r["f_cost"] or [float("nan")])[0]
        acc = (r["test_acc"] or [float("nan")])[0]
        strict = (r["id_set"] or [float("nan")] * 4)[3]
        hedged = retries = float("nan")
        if r["requests"]:
            hedged, retries = r["requests"][1], r["requests"][3]
        print(f"{r['arm']:<26} {r['wall_s']:>8.0f} {cost:>10.4f} {acc:>9.2f} "
              f"{strict:>7.2f} {hedged:>7.0f} {retries:>8.0f}")


if __name__ == "__main__":
    main()
