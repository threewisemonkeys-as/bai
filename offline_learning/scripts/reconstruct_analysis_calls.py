#!/usr/bin/env python3
"""Best-effort reconstruction of the --analyze-mistakes diagnosis calls for runs that
PREDATE analysis-call logging (analysis_calls.jsonl did not exist yet).

What is and isn't recoverable
-----------------------------
The diagnosis call's RAW response was never persisted, and the exact diagnosis PROMPT
is not deterministically rebuildable from the old-format logs. BUT the call's OUTPUT
SUBSTANCE survives: make_reflective_dataset injected each per-mistake diagnosis into the
proposer feedback as `ANALYSIS (inverse-dynamics mistake): ...` / `ANALYSIS (forward-
prediction mistake): ...` blocks (with the COMMON ROOT CAUSE synthesis prepended to the
first one). Those blocks are saved verbatim in process_log.jsonl's `feedback`.

So this script scans process_log.jsonl, pulls every embedded ANALYSIS block per
(iteration, component), and writes an analysis_calls.jsonl whose `response` is the
recovered diagnosis output and whose `prompt` is a placeholder explaining it was not
logged. Records are flagged `reconstructed: true` so the viz labels them as partial.

  uv run python offline_learning/scripts/reconstruct_analysis_calls.py \
      --run-dir logs/clean_sweep_fwd/dq8gc_seed1_swap_fdreflect_analyze/rexpure_run_seed1

Then rebuild the viz; the "Mistake-analysis call(s)" panel will populate from this file.
Re-running gepa_optimize.py with the patched code logs the real prompt+response instead.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Matches an embedded diagnosis block and captures its kind + body up to the next section
# marker ("===", a following "ANALYSIS (", or end of the field).
_ANALYSIS_RE = re.compile(
    r"ANALYSIS \((inverse-dynamics|forward-prediction) mistake\):\s*\n"
    r"(.*?)(?=\n===|\nANALYSIS \(|\Z)",
    re.DOTALL,
)

_PROMPT_PLACEHOLDER = (
    "(The original diagnosis prompt was not logged for this run -- it predates "
    "analysis_calls.jsonl. The RESPONSE below is the diagnosis OUTPUT recovered verbatim "
    "from the ANALYSIS blocks embedded in the proposer feedback (process_log.jsonl). "
    "Re-run gepa_optimize.py with --analyze-mistakes on the patched code to capture the "
    "exact prompt + raw response.)"
)


def _iter_strings(obj):
    """Yield every string leaf in a nested dict/list (the reflective-dataset entry)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def extract_analyses(feedback) -> dict[str, list[dict]]:
    """reflective_dataset -> {component: [{kind, ti, text}, ...]} of recovered ANALYSIS blocks."""
    out: dict[str, list[dict]] = {}
    if not isinstance(feedback, dict):
        return out
    for comp, entries in feedback.items():
        blocks = []
        for ti, entry in enumerate(entries or []):
            for s in _iter_strings(entry):
                for m in _ANALYSIS_RE.finditer(s):
                    kind = "inv" if m.group(1).startswith("inverse") else "fwd"
                    body = m.group(2).strip()
                    if body:
                        blocks.append({"kind": kind, "ti": ti, "text": body})
        if blocks:
            out[comp] = blocks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="…/rexpure_run_seed<N> (or historical gepa_run_seed<N>) directory")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing analysis_calls.jsonl (refuse by default if it looks "
        "like real logged data, i.e. has no reconstructed records)",
    )
    args = ap.parse_args()
    gdir = Path(args.run_dir)
    proc_path = gdir / "process_log.jsonl"
    out_path = gdir / "analysis_calls.jsonl"
    if not proc_path.exists():
        raise SystemExit(f"no process_log.jsonl in {gdir}")

    if out_path.exists() and not args.force:
        # guard: don't clobber a real logged file
        try:
            first = json.loads(out_path.open().readline() or "{}")
        except json.JSONDecodeError:
            first = {}
        if not first.get("reconstructed"):
            raise SystemExit(
                f"{out_path} already exists and is not reconstructed data; refusing to "
                f"overwrite (use --force)."
            )

    records, n_blocks = [], 0
    for line in proc_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        it = rec.get("i")
        per_comp = extract_analyses(rec.get("feedback"))
        for comp, blocks in per_comp.items():
            n_blocks += len(blocks)
            # one combined record per (iteration, component): the diagnosis call diagnosed
            # all shown mistakes for the component together (combined mode).
            body = "\n\n".join(
                f"[mistake {b['ti']} · {b['kind']}]\n{b['text']}" for b in blocks
            )
            records.append({
                "iteration": it,
                "ts": rec.get("ts", ""),
                "component": comp,
                "mode": "combined",
                "kind": "combined",
                "ti": None,
                "n_cases": len(blocks),
                "prompt": _PROMPT_PLACEHOLDER,
                "response": body,
                "cost": None,
                "reconstructed": True,
            })

    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(
        f"[reconstruct] wrote {len(records)} analysis records "
        f"({n_blocks} recovered ANALYSIS blocks) -> {out_path}"
    )


if __name__ == "__main__":
    main()
