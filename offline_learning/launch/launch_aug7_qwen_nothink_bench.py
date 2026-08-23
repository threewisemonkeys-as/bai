"""Does disabling qwen3.7-flash's hidden thinking fix its eval-call wall time?

Follow-up to launch_aug6_evalmodel_bench.py (qwen matched gpt-oss-120b quality at
-75% cost but 14x wall; probes traced the latency to ~2.2k hidden thinking tokens
per call, disabled by OpenRouter `reasoning: {"effort": "none"}`, which litellm's
bridge drops -- see _nothink_wrapper.py).

Three arms, same eval-only workload as aug6 (warm-started shipped B/P,
--max-nodes 1, test50 ID eval), now on the post-refactor rexpure CLI
(saved gepa cmds sanitized via _launch_util.sanitize_rexpure_cmd -- the CLI
change makes yesterday's absolute numbers non-comparable, hence the re-baseline):

  gptoss120b_cerebras  -- current prod config, re-baselined under rexpure
  qwen37flash_default  -- thinking at provider default (yesterday's config)
  qwen37flash_nothink  -- + reasoning {"effort": "none"} via the wrapper

Results -> logs/aug7_qwen_nothink_bench/.
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
SRC = ROOT / "logs/aug6_evalmodel_bench"
OUT_ROOT = ROOT / "logs/aug7_qwen_nothink_bench"
WRAPPER = Path(__file__).resolve().parent / "_nothink_wrapper.py"

ARMS = [
    # (name, source aug6 arm, reasoning override, cmd flag overrides)
    ("gptoss120b_cerebras", "gptoss120b_cerebras", None, {}),
    ("qwen37flash_default", "qwen37flash_alibaba", None, {}),
    ("qwen37flash_nothink", "qwen37flash_alibaba", {"effort": "none"}, {}),
    # does the 3.2x wall gap close when the 11-15s/call latency is hidden by
    # more parallelism? (hedge stays at the 30s default for clean attribution)
    ("qwen37flash_nothink_c96", "qwen37flash_alibaba", {"effort": "none"},
     {"--concurrency": "96"}),
    ("qwen37flash_nothink_c128", "qwen37flash_alibaba", {"effort": "none"},
     {"--concurrency": "128"}),
    # gpt-oss can't disable reasoning ("mandatory for this endpoint", 400) but
    # accepts effort low: probe showed reasoning tokens 511->302, cost -11%/call.
    ("gptoss120b_efflow", "gptoss120b_cerebras", {"effort": "low"}, {}),
]

PARSE = {
    "test_acc": r"CLEAN test acc \(inverse\) = ([\d.]+)",
    "id_set": r"ID-set: hit_rate=([\d.]+) mean_set_size=([\d.]+) mean_loss=([\d.]+) strict=([\d.]+)",
    "f_cost": r"F \(task_lm\) cost=\$([\d.]+) \((\d+) fresh F evals",
    "requests": r"LLM requests=(\d+) hedged=(\d+) hedge_wins=(\d+) retries=(\d+)",
}


def run_arm(name: str, src_arm: str, reasoning: dict | None, overrides: dict) -> dict:
    outd = OUT_ROOT / name
    if outd.exists():
        raise SystemExit(f"refusing to overwrite existing run: {outd}")
    src_launch = json.loads((SRC / src_arm / "launch.json").read_text())
    cmd = sanitize_rexpure_cmd(list(src_launch["cmd"]))
    # saved cmds predate the refactor that moved prototypes/perc_invdyn -> offline_learning
    cmd = [a.replace("/prototypes/perc_invdyn/", "/offline_learning/") for a in cmd]
    cmd[0] = sys.executable
    cmd[cmd.index("--out-dir") + 1] = str(outd)
    for flag, value in overrides.items():
        cmd[cmd.index(flag) + 1] = value
    env = dict(os.environ, OPENROUTER_PROVIDER_ORDER=src_launch["pin"])
    if reasoning is not None:
        cmd[1] = str(WRAPPER)
        env["REASONING_OVERRIDE_JSON"] = json.dumps(reasoning)
    outd.mkdir(parents=True)
    t0 = time.perf_counter()
    with (outd / "stdout.txt").open("w") as stdout:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=subprocess.STDOUT,
        )
    wall = time.perf_counter() - t0
    text = (outd / "stdout.txt").read_text()
    res = {
        "arm": name, "reasoning_override": reasoning, "pin": src_launch["pin"],
        "wall_s": round(wall, 1), "returncode": proc.returncode,
    }
    for key, pat in PARSE.items():
        m = re.search(pat, text)
        res[key] = [float(g) for g in m.groups()] if m else None
    (outd / "launch.json").write_text(
        json.dumps({"cmd": cmd, "result": res, "source": src_arm}, indent=2) + "\n"
    )
    return res


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    results = []
    for name, src_arm, reasoning, overrides in ARMS:
        done = OUT_ROOT / name / "launch.json"
        if done.exists():  # resume: keep completed arms, rerun only missing ones
            res = json.loads(done.read_text())["result"]
            print(f"[bench] {name}: already done, reusing result", flush=True)
            results.append(res)
            continue
        print(f"[bench] running {name} ...", flush=True)
        res = run_arm(name, src_arm, reasoning, overrides)
        results.append(res)
        print(f"[bench] {name}: {res}", flush=True)
        (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'arm':<22} {'wall(s)':>8} {'F cost($)':>10} {'test_acc':>9} "
          f"{'strict':>7} {'hedged':>7}")
    for r in results:
        cost = (r["f_cost"] or [float("nan")])[0]
        acc = (r["test_acc"] or [float("nan")])[0]
        strict = (r["id_set"] or [float("nan")] * 4)[3]
        hedged = r["requests"][1] if r["requests"] else float("nan")
        print(f"{r['arm']:<22} {r['wall_s']:>8.0f} {cost:>10.4f} {acc:>9.2f} "
              f"{strict:>7.2f} {hedged:>7.0f}")


if __name__ == "__main__":
    main()
