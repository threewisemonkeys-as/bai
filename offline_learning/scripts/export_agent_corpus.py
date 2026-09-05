#!/usr/bin/env python3
"""The world model's training pool, on disk, for the PRO-LONG agent arm to grep.

The `agent` arm is the coding agent handed the SAME offline data the NLWM world model
was fit on and the `icl` arm reads in its prompt. That only means something if all
three see the same bytes, so this exporter does not walk the pool directory: it calls
`icl_context.load_pool_transitions`, the function the `icl` arm uses, and writes what
comes back. Everything the parity argument needs -- train_d* only, 9-frame context
backfilled from the train drives, the `Task:/Step:` observation header stripped -- is
that function's behaviour, not this file's.

Choosing the loader over the directory also closes four of the five leaks in the plan
by construction rather than by scrubbing:

  1. MANIFEST.json's `human_game` (the world's real English name) -- never opened.
  2. MANIFEST.json's `selection_note` (a hand-written dynamics summary; the answer
     key in prose) -- never opened.
  3. drive metadata's `task_id` (the name again) -- never opened.
  5. test_d* and drives/test_d* (the learner's held-out targets) -- never read.

Leak 4, the observation header, is stripped inside the loader because the learner
strips it. `_assert_clean` re-checks all five over the bytes actually written, because
a structural argument that is never tested is a belief.

    uv run python offline_learning/scripts/export_agent_corpus.py --game bt3gb --out /tmp/corpus
    uv run python offline_learning/scripts/export_agent_corpus.py --game all --out corpora/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _p in (REPO, REPO / "offline_learning", REPO / "cc_autumn/autumn-code/rig"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import icl_context  # noqa: E402

DEFAULT_ARTIFACT_ROOT = REPO / "logs/2026-08-24/human_curated"

_TOKEN_RE = re.compile(r"[^a-z0-9]+")

# The opaque world codes the agent's workspace is named with, so the directory name is
# not itself a hint. Imported, never copied: `curated.py` is where the battery defines
# them and a second table is a second answer.
try:
    from curated import LABELS  # type: ignore
except ImportError:                                     # pragma: no cover - rig absent
    LABELS = {}

# Everything the export must not contain. The English names are the answer to "what
# game is this?"; the rest are the metadata fields that carry it.
_FORBIDDEN_SUBSTRINGS = (
    "human_game", "selection_note", "task_id", "dataset_paths",
    "dynamics", "informative_curated",
    # the header stripped at load; its presence means the strip did not run
    "Task:", "Phase:", "Available actions now:",
)

README = """\
# Recorded transitions

{n} transitions recorded from this environment by a human player, one JSON file each.

    <id>.json   {{"id", "action", "state", "next_state", "context"}}
    index.csv   id,action,file -- one row per transition

`state` and `next_state` are JSON 2-D arrays of colour-name strings: the grid before
and after `action`. `context` is up to {k} earlier `[state, action]` pairs from the same
session, oldest first, so behaviour that depends on what came before is visible.

Entries are independent samples drawn from several sessions. Consecutive ids are NOT
consecutive in time -- do not read the directory as one trajectory.

Parse these programmatically. The set is {mb:.1f} MB; reading it all into context is
neither necessary nor useful.
"""


def _obs_header_present(text: str) -> bool:
    """The `Task:/Step:/Phase:` block the learners strip at load."""
    head = text[:400]
    return "Task:" in head and "Step:" in head


def export_game(game: str, out_root: Path, *, pool: str = icl_context.DEFAULT_POOL,
                data_root: Path | None = None,
                artifact_root: Path | None = DEFAULT_ARTIFACT_ROOT,
                label: str | None = None) -> dict:
    """Write `game`'s training pool under `out_root/<LABEL>/`. Returns the manifest."""
    transitions = icl_context.load_pool_transitions(game, pool=pool, data_root=data_root)
    check = ({"checked": False, "reason": "no artifact root given"} if artifact_root is None
             else icl_context.assert_matches_launch(game, transitions, Path(artifact_root)))

    label = label or LABELS.get(game) or game.upper()
    out = Path(out_root) / label
    if out.exists():
        shutil.rmtree(out)
    (out / "drives").mkdir(parents=True)

    rows, total = [], 0
    for i, t in enumerate(transitions):
        tid = f"t{i:03d}"
        record = {
            "id": tid,
            "action": t.action,
            "state": t.x_t,
            "next_state": t.x_t1,
            "context": [[state, action] for state, action in (t.ctx_prev or [])],
        }
        path = out / "drives" / f"{tid}.json"
        text = json.dumps(record)
        path.write_text(text)
        total += len(text)
        rows.append({"id": tid, "action": t.action, "file": f"drives/{tid}.json"})

    with open(out / "drives" / "index.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "action", "file"])
        writer.writeheader()
        writer.writerows(rows)

    context_k = max((len(t.ctx_prev or []) for t in transitions), default=0)
    readme = README.format(n=len(transitions), k=context_k, mb=total / 1e6)
    (out / "drives" / "README.md").write_text(readme)

    manifest = {
        "label": label, "n_transitions": len(transitions), "context_k": context_k,
        "bytes": total, "match_check": check,
    }
    _assert_clean(out, game, manifest, readme)
    return manifest


def _assert_clean(out: Path, game: str, manifest: dict, readme: str) -> None:
    """Fail the export, not the run.

    Two different things are being guarded, and conflating them makes the guard useless.

    The *data* is machine-generated from the pool, so it is scanned: for the metadata
    fields that carry the world's identity, and for the name of any of the fifteen
    worlds -- naming even the wrong one narrows the field.

    The *README* is ours, so it is pinned instead. It cannot be scanned for names,
    because four of the fifteen worlds are spelled with ordinary English words (`SET`,
    `egg`, `dino`, `diffusion`) and prose about a data set trips on them; the first
    version of this file rejected its own README over the word "set". Pinning it to the
    rendered template is the stronger check anyway: the text cannot drift at all.
    """
    names = {n.lower() for n in LABELS} | {game.lower()}
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        low = text.lower()
        if path.name == "README.md":
            if text != readme:
                raise AssertionError(f"{path.relative_to(out)} is not the pinned template")
            continue
        for needle in _FORBIDDEN_SUBSTRINGS:
            if needle.lower() in low:
                raise AssertionError(
                    f"{path.relative_to(out)} contains forbidden metadata {needle!r}")
        # Tokenise on every non-alphanumeric run, then match the name as a bounded
        # token sequence. Splitting on whitespace alone is not enough: the leak this
        # guard exists for would arrive inside JSON, where the name carries punctuation
        # (`"bt3gb"}`) and never equals a bare token. Joining also lets a multi-word
        # name (`logic_gates`, `colour_lines`) match as one unit.
        tokens = f" {' '.join(_TOKEN_RE.split(low))} "
        for name in names:
            needle = f" {' '.join(_TOKEN_RE.split(name))} "
            if needle in tokens:
                raise AssertionError(
                    f"{path.relative_to(out)} names the world ({name!r})")
        if path.suffix == ".json" and _obs_header_present(text):
            raise AssertionError(f"{path.relative_to(out)} still carries the obs header")
    if manifest["n_transitions"] == 0:
        raise AssertionError(f"{game}: exported an empty corpus")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, help="a game name, or 'all'")
    ap.add_argument("--out", required=True, help="output root; <out>/<LABEL>/ per game")
    ap.add_argument("--pool", default=icl_context.DEFAULT_POOL)
    ap.add_argument("--data-root", default=str(icl_context.DEFAULT_DATA_ROOT))
    ap.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    args = ap.parse_args()

    root = Path(args.data_root)
    games = ([p.name for p in sorted(root.iterdir()) if (p / args.pool).is_dir()]
             if args.game == "all" else [args.game])

    out_root = Path(args.out)
    manifests = {}
    for game in games:
        manifest = export_game(game, out_root, pool=args.pool, data_root=root,
                               artifact_root=Path(args.artifact_root))
        manifests[game] = manifest
        check = manifest["match_check"]
        flag = "ok" if check.get("checked") else f"UNCHECKED ({check.get('reason')})"
        print(f"{manifest['label']:8s} {manifest['n_transitions']:3d} transitions  "
              f"k={manifest['context_k']}  {manifest['bytes']/1e6:5.1f} MB  {flag}")

    (out_root / "export.json").write_text(json.dumps(manifests, indent=1))
    print(f"\n{len(manifests)} game(s) -> {out_root}")


if __name__ == "__main__":
    main()
