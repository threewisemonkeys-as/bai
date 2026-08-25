"""Run the two learners on HUMAN-ORIGIN training data, matched to their artificial runs.

The question: the aug9/aug10 results were produced on hand-curated synthetic drives
(`clean_data3`). Does the same learner, with the same config, reach the same place when
the transitions come from real human play (`basis_data.zip`, replayed by
`offline_learning/human_replay.py`)?

Both learners now default to the CURATED human dataset (`informative_curated`, the
2026-08-24 variant of record for the 15-game selection: the unified3 recipe -- horizon 8,
--oov noop, coverage/under-fill guards -- with the drives picked by hand per game from
`offline_learning/curated_drives.json`; see HUMAN_DATA_METHODOLOGY.md): one shared
60-target train pool + identical 50-target test set per game, so a rexpure-vs-worldcoder
head-to-head is on the SAME transitions. `--variant informative_unified3` is the
uncurated (ranked-drive) control arm.
For matched consumption rexpure's reference `--train-n 30` is lifted to the 60-target pool
(worldcoder has no `--train-n` and already consumes the whole pool), and worldcoder's
`--collapse-action-params` is dropped so both learners score the end-of-run inverse-dynamics
test at click-LOCATION level (wc collapse is test-ID-only, so this leaves its program
synthesis untouched). Results land in `logs/2026-08-24/human_curated/` (--out-root to
change; use `--variant informative_unified3 --out-root logs/2026-08-24/human_unified3` for
the control arm). Pass `--variant informative_unified --out-root
logs/2026-08-11/human_unified` to reproduce the 2026-08-11 arms, or `--variant
informative` / `informative_wc` for the old reference-sized ones.

So the ONLY thing that changes is the data. Each command is rebuilt from the reference
run's own `launch.json` -- every flag is carried over byte-identically and only the four
data-path flags, `--out-dir` (and, for the unified variant, rexpure's `--train-n`) swapped:

  rexpure     reference logs/archive/2026-08-11/batch3_consolidated/<game>_s1_batch3
              (s2kt7 falls back to 83wkq's; the 10 games added 2026-08-24 fall back to
               bt3gb's -- the batch3 non-data flags are identical across games)
  worldcoder  reference logs/archive/2026-08-11/wc_seed1_consolidated/<game>_s1_wc

Human slices are 2 rows (one scored target each), so temporal context always comes from
`--context-source-run` / `--test-context-source-run`. Those two flags are therefore
passed even for the reference commands that did not use them (bt3gb/83wkq/s2kt7 under
worldcoder pointed at self-contained clean_data3 slices instead).

Reflection-model arm (2026-08-24): `--reflection-model claude-opus-5 --reflection-client vllm`
swaps ONLY the reflection/analysis LLM of either learner for Claude served by the local
CLI proxy (`scripts/claude_cli_proxy.py`, start it with `scripts/claude_proxy_ctl.sh start`);
the reference's reflection routing flags (provider order, hedge, timeout) are replaced by
proxy-appropriate ones (no pin, no hedge -- a hedge would be a second full CLI call --
timeout 900 s) and HOSTED_VLLM_API_BASE/KEY are exported to the run. rexpure's task model F
is untouched, so the arm isolates the reflection model. Use a separate --out-root.

Usage:
    uv run python offline_learning/launch/launch_human_origin.py --learner rexpure
    uv run python offline_learning/launch/launch_human_origin.py --learner worldcoder
    uv run python offline_learning/launch/launch_human_origin.py --learner worldcoder \
        --reflection-model claude-opus-5 --reflection-client vllm \
        --out-root logs/2026-08-24/human_curated_opus5
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
DATA = ROOT / "offline_learning/human_data"
OUT_ROOT = ROOT / "logs/2026-08-24/human_curated"   # overridable via --out-root
# the 15-game selection of experimental_plan.md (2026-08-24)
GAMES = ["eahcw", "egg", "bt3gb", "dq8gc", "7xf97", "n2ntd", "va6fq", "s2kt7",
         "colour_lines", "SET", "diffusion", "dino", "f5w3n", "logic_gates", "7www9"]

# reference runs were archived 2026-08-11 (they lived at logs/{batch3,wc_seed1}_consolidated)
REF = {
    "rexpure": (ROOT / "logs/archive/2026-08-11/batch3_consolidated", "{game}_s1_batch3",
                "offline_learning/rexpure_optimize.py"),
    "worldcoder": (ROOT / "logs/archive/2026-08-11/wc_seed1_consolidated", "{game}_s1_wc",
                   "offline_learning/worldcoder_optimize.py"),
}
# both learners read the CURATED shared pool (flat 60 train / 50 test, hand-picked drives).
# --variant informative_unified3 = ranked-drive control; informative / informative_wc =
# the old reference-sized arms.
VARIANT = {"rexpure": "informative_curated", "worldcoder": "informative_curated"}
# the unified pool size; rexpure's reference --train-n is lifted to this so both learners
# consume the whole shared pool (worldcoder has no --train-n). Every 60/50 flat-pool
# variant (informative_unified* and informative_curated) gets the unified treatment
# (train-n lift + wc uncollapse).
UNIFIED_TRAIN_N = 60


def is_unified(variant: str) -> bool:
    return variant.startswith("informative_unified") or variant == "informative_curated"

DATA_FLAGS = ("--run", "--context-source-run", "--test-run",
              "--test-context-source-run", "--out-dir", "--actions")
# value-taking reference flags replaced wholesale when --reflection-model is given
REFLECTION_FLAGS = ("--reflection-model", "--reflection-client", "--reflection-provider-order",
                    "--reflection-hedge-delay", "--reflection-timeout",
                    "--reflection-reasoning-json", "--analysis-reasoning-json")
PROXY_BASE = "http://127.0.0.1:8000/v1"

# s2kt7 has no batch3 launch.json; it is the 83wkq config with s2kt7 data. The 10 games
# added 2026-08-24 have no reference run at all: they inherit FALLBACK's config -- safe
# because the batch3 non-data flags are byte-identical across games (verified 2026-08-24),
# and the only per-game wc flags (--collapse-action-params, --max-proposals 50-55) are
# dropped / irrelevant under the unified variant.
S2KT7_REXPURE_FROM = "83wkq"
FALLBACK = "bt3gb"


def ref_src(learner: str, game: str) -> str:
    """Which reference run supplies this game's non-data flags."""
    root, pat, _ = REF[learner]
    if (root / pat.format(game=game) / "launch.json").exists():
        return game
    if game == "s2kt7" and learner == "rexpure":
        return S2KT7_REXPURE_FROM
    return FALLBACK


def ref_env(learner: str, game: str) -> dict[str, str]:
    """Leading KEY=VALUE tokens recorded in the reference cmd (e.g. s2kt7's
    LLM_TIMEOUT_S / LLM_HEDGE_DELAY_S), so the human run inherits them too."""
    root, pat, _script = REF[learner]
    p = root / pat.format(game=ref_src(learner, game)) / "launch.json"
    if not p.exists():
        return {}
    env = {}
    for tok in json.loads(p.read_text())["cmd"]:
        if tok.endswith(".py"):
            break
        if "=" in tok and not tok.startswith("-"):
            k, _, v = tok.partition("=")
            env[k] = v
    return env


def ref_cmd(learner: str, game: str) -> list[str]:
    root, pat, script = REF[learner]
    p = root / pat.format(game=ref_src(learner, game)) / "launch.json"
    cmd = list(json.loads(p.read_text())["cmd"])
    # some launch.jsons record an env-var prefix and/or `uv run python`; normalise to
    # a bare interpreter invocation of the script, keeping every real flag.
    i = next(i for i, x in enumerate(cmd) if x.endswith(".py"))
    return [sys.executable, str(ROOT / script)] + cmd[i + 1:]


def build(learner: str, game: str, outd: Path, variant: str,
          reflection_model: str | None = None, reflection_client: str | None = None,
          reflection_timeout: float = 900.0) -> list[str]:
    cmd = ref_cmd(learner, game)
    # value-taking flags to drop: data paths always; for the unified variant also rexpure's
    # --train-n cap (re-added below at the shared pool size); with a reflection override
    # every reflection routing flag (re-added below for the new endpoint).
    drop = set(DATA_FLAGS) | ({"--train-n"} if is_unified(variant) else set())
    if reflection_model:
        drop |= set(REFLECTION_FLAGS)
    # valueless (store_true) flags to drop: for the unified variant un-collapse worldcoder's
    # test ID so BOTH learners score inverse-dynamics at click-location level. wc collapse is
    # test-ID-only (the program is always fit on full 'click ROW COL'), so dropping it does
    # not change wc's learning -- only the end-of-run ID protocol.
    drop_flag = ({"--collapse-action-params"}
                 if is_unified(variant) and learner == "worldcoder" else set())
    stripped, i = [], 2
    while i < len(cmd):
        if cmd[i] in drop:
            i += 2
            continue
        if cmd[i] in drop_flag:
            i += 1
            continue
        stripped.append(cmd[i])
        i += 1
    if is_unified(variant) and learner == "rexpure":
        stripped += ["--train-n", str(UNIFIED_TRAIN_N)]
    if reflection_model:
        stripped += ["--reflection-model", reflection_model,
                     "--reflection-provider-order", "",      # explicit no-pin
                     "--reflection-hedge-delay", "0",         # a hedge = a 2nd full call
                     "--reflection-timeout", str(reflection_timeout)]
        if reflection_client:
            stripped += ["--reflection-client", reflection_client]
    paths = json.loads((DATA / game / variant / "dataset_paths.json").read_text())
    return cmd[:2] + [
        "--run", paths["run"],
        "--context-source-run", paths["context_source_run"],
        "--test-run", paths["test_run"],
        "--test-context-source-run", paths["test_context_source_run"],
        "--actions", paths["actions"],
        "--out-dir", str(outd),
    ] + stripped


def done_marker(learner: str, outd: Path) -> Path:
    return outd / ("test_summary_rexpure_seed1.json" if learner == "rexpure"
                   else "test_summary_wc_seed1.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", required=True, choices=sorted(REF))
    ap.add_argument("--variant", help="override the dataset variant (e.g. raw)")
    ap.add_argument("--suffix", default="", help="appended to the out-dir name")
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="tree for <learner><suffix>/<game>_s1 out-dirs")
    ap.add_argument("--reflection-model", default=None,
                    help="swap the reflection LLM (e.g. claude-opus-5 via the CLI proxy)")
    ap.add_argument("--reflection-client", default=None,
                    help="litellm provider for the reflection model; 'vllm' = OpenAI-compatible "
                         "endpoint at --proxy-base (the Claude CLI proxy)")
    ap.add_argument("--reflection-timeout", type=float, default=900.0)
    ap.add_argument("--proxy-base", default=PROXY_BASE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.reflection_client == "vllm":
        import urllib.request
        try:
            urllib.request.urlopen(args.proxy_base.rsplit("/", 1)[0] + "/healthz", timeout=5)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"reflection proxy not reachable at {args.proxy_base}: {exc} "
                             "(start it: bash offline_learning/scripts/claude_proxy_ctl.sh start)")

    # The groq pin exists for rexpure's TASK model (gpt-oss-20b@groq). worldcoder has no
    # task LLM -- exporting it there pins its reflection model to a provider that does not
    # serve it, and every iteration 404s. The reference wc runs never set it.
    env = dict(os.environ)
    if args.learner == "rexpure":
        env["OPENROUTER_PROVIDER_ORDER"] = "groq"
    if args.reflection_client == "vllm":
        env["HOSTED_VLLM_API_BASE"] = args.proxy_base
        env["HOSTED_VLLM_API_KEY"] = "local-proxy"
    live: list[subprocess.Popen] = []
    for game in args.games.split(","):
        variant = args.variant or VARIANT[args.learner]
        outd = Path(args.out_root) / (args.learner + args.suffix) / f"{game}_s1"
        if done_marker(args.learner, outd).exists():
            print(f"skip  {game}: already complete")
            continue
        cmd = build(args.learner, game, outd, variant, args.reflection_model,
                    args.reflection_client, args.reflection_timeout)
        if args.dry_run:
            print(f"[dry-run] {game}:\n  {' '.join(cmd)}\n")
            continue
        while len([p for p in live if p.poll() is None]) >= args.max_parallel:
            time.sleep(15)
        outd.mkdir(parents=True, exist_ok=True)
        run_env = dict(env, **ref_env(args.learner, game))
        with (outd / "stdout.txt").open("w") as f:
            p = subprocess.Popen(cmd, cwd=ROOT, env=run_env, stdin=subprocess.DEVNULL,
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        live.append(p)
        (outd / "launch.json").write_text(json.dumps(
            {"game": game, "learner": args.learner, "variant": variant,
             "reflection_model": args.reflection_model, "reflection_client": args.reflection_client,
             "pid": p.pid, "cmd": cmd,
             "env": {**ref_env(args.learner, game),
                     **({"HOSTED_VLLM_API_BASE": args.proxy_base} if args.reflection_client == "vllm" else {})}},
            indent=2) + "\n")
        print(f"start {args.learner}/{game}: pid={p.pid} -> {outd}", flush=True)
    for p in live:
        p.wait()
    print("all done")


if __name__ == "__main__":
    main()
