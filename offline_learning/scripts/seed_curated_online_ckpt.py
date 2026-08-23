#!/usr/bin/env python3
"""Copy the ARTIFACT-INDEPENDENT arms of a curated ONLINE eval into another run's checkpoint.

`eval_curated_online.py` resumes from `<out>.ckpt.jsonl`, keyed `problem|arm|attempt`. Two of
its three arms do not read the rexpure artifacts at all:

  raw   `llm_rollout("raw", ...)` builds its prompt from PLAN_RAW_TMPL + raw_transcript and
        never touches `perceive` or `beliefs` (eval_coverage_online.llm_rollout, the `arm ==
        "raw"` branch), so its rollout distribution is identical between two runs that differ
        only in those artifacts.
  wc    a deterministic program search over `worldcoder/<game>_s1`, which the ablation root
        exposes as a symlink to the very same file -- so a re-run would reproduce it exactly.

Seeding them lets an A/B (e.g. the no-perception ablation vs its learned-P reference) share
ONE control sample instead of paying for a second one, and makes the `raw` column literally
identical in both reports rather than merely equal in distribution. The `lmwm` arm is never
copied -- it is the arm under test.

    uv run python offline_learning/scripts/seed_curated_online_ckpt.py \
        --from logs/2026-08-18/curated/eval/online.ckpt.jsonl \
        --to   logs/2026-08-19/noperc_ablation/curated_eval/online.ckpt.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SHARED_ARMS = ("raw", "wc")


def load_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--arms", default=",".join(SHARED_ARMS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    arms = set(filter(None, a.arms.split(",")))
    assert "lmwm" not in arms, "lmwm is the arm under test and must never be copied"
    src, dst = Path(a.src), Path(a.dst)
    have = load_keys(dst)

    copied, skipped = Counter(), Counter()
    lines = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        arm = rec["key"].split("|")[1]
        if arm not in arms:
            continue
        if rec["key"] in have:
            skipped[arm] += 1
            continue
        lines.append(line)
        copied[arm] += 1

    print(f"from {src} -> {dst}")
    for arm in sorted(arms):
        print(f"  {arm}: copy {copied[arm]}, already present {skipped[arm]}")
    if a.dry_run:
        print("[dry-run] nothing written")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a") as fh:
        for line in lines:
            fh.write(line + "\n")
    print(f"wrote {len(lines)} rollout(s); {dst} now has {len(load_keys(dst))} key(s)")


if __name__ == "__main__":
    main()
