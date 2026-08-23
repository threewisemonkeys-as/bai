#!/usr/bin/env python3
"""Simulator-grounded rescoring of the test50 inverse-dynamics eval.

Exact-match ID scoring marks a predicted action wrong whenever it differs from
the recorded action, even when several actions map s_t to the same s_t+1 (noop
aliasing, handler-less moves, quiet clicks). This script re-judges every
persisted prediction in the Autumn engine: reset the game (seed 0), replay the
recorded drive prefix to reach s_t, execute the predicted action, and count the
prediction correct iff the rendered next grid equals the recorded s_t+1 grid.

The engine state at s_t is reconstructed from the full source drives behind the
test50 slices (logs/fulltraj_context_* manifests, ada85's reconstructed srcs,
self-contained slices, or logs/test50_sim_rescore_inputs for the reconstructed
games). Every drive is first verified against the engine frame-by-frame and
every slice against its drive verbatim, so a recorded action re-executed at its
own step provably reproduces the recorded next state; sim scoring is therefore
a strict superset of exact matching.

Predictions come from logs/id_eval_test50_raw_vs_learned.json — no LLM calls.
A bare "click" prediction (parameter-collapsed games) is judged correct iff
clicking SOME cell reproduces s_t+1 (cells enumerated, non-background first).

    uv run python offline_learning/scripts/rescore_test50_id_sim.py
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_forward_modes as efm
import test50_sim_tools as T
from clean_sweep import GAMES
from validate import load_transitions
from validate_beliefs import balanced_split

MANIFEST_ROOTS = [
    REPO / "logs/fulltraj_context_remaining_autumn_inputs",
    REPO / "logs/fulltraj_context_remaining_autumn_phase2_inputs",
    REPO / "logs/test50_sim_rescore_inputs",
]
ADA85_SRC = (
    REPO / "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50/ada85_seed1/"
    "test50_fulltraj_sources_reconstructed"
)
DIRECTIONS = {"left", "right", "up", "down", "noop"}


def drive_csv_for(source_root: Path) -> Path:
    for cand in (source_root / "episode_0/trajectory.csv", source_root / "trajectory.csv"):
        if cand.exists():
            return cand
    hits = list(source_root.rglob("trajectory.csv"))
    if len(hits) != 1:
        raise FileNotFoundError(f"no unique trajectory.csv under {source_root}")
    return hits[0]


def resolve_sources(game: str, data_root: Path) -> dict[str, list[Path]]:
    """episode dir name -> candidate full source drive CSVs (row Steps index the
    drive). Usually one candidate; when several verified drives contain the slice
    verbatim (ada85's paired srcs), all are kept and branch outcomes must agree."""
    for root in MANIFEST_ROOTS:
        manifest = root / game / "test50_context_manifest.json"
        if manifest.exists():
            return {
                f"episode_{row['curated_episode']}": [drive_csv_for(REPO / row["source_root"])]
                for row in json.loads(manifest.read_text())
            }
    episodes = sorted(
        (data_root / game / "test50").glob("episode_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    candidates = []
    if game == "ada85":
        candidates = [drive_csv_for(src) for src in sorted(ADA85_SRC.glob("src*"))]
    mapping = {}
    for ep in episodes:
        ep_csv = ep / "trajectory.csv"
        slice_rows = T.read_rows(ep_csv)
        matches = [
            c for c in candidates
            if not T.match_slice_to_drive(slice_rows, T.read_rows(c))
        ]
        if not matches and int(slice_rows[0]["Step"]) == 0:
            matches = [ep_csv]  # the slice is its own full drive
        if not matches:
            raise RuntimeError(
                f"{game}/{ep.name}: no candidate drive matches and the slice "
                f"does not start at step 0; no manifest available"
            )
        mapping[ep.name] = matches
    return mapping


def split_with_provenance(game: str, data_root: Path, test_dir: str, seed: int,
                          context_k: int, test_n: int):
    """reproduce_test_split with (episode_csv, row_index) provenance attached."""
    action_csv, keep_params, _mmc, _rounds = GAMES[game]
    whitelist = set(filter(None, action_csv.split(","))) or None
    train = load_transitions([data_root / game / "train"], whitelist, context_k=context_k)
    test = load_transitions([data_root / game / test_dir], whitelist, context_k=context_k)

    # Re-walk the test CSVs with load_transitions' exact skip logic to attach
    # provenance; pairwise-assert to catch any ordering drift.
    prov = []
    import csv as _csv
    for csv_path in (data_root / game / test_dir).glob("episode_*/trajectory.csv"):
        rows = list(_csv.DictReader(csv_path.open()))
        for i in range(len(rows) - 1):
            r, nxt = rows[i], rows[i + 1]
            action = (r.get("Action") or "").strip()
            obs, obs_next = r.get("Observation") or "", nxt.get("Observation") or ""
            done = (r.get("Done") or "").strip().lower() in ("true", "1")
            if done or not action or not obs.strip() or not obs_next.strip():
                continue
            if whitelist is not None and action.split()[0] not in whitelist:
                continue
            prov.append((csv_path, i, action, obs, obs_next))
    if len(prov) != len(test):
        raise AssertionError(f"{game}: provenance walk found {len(prov)} vs {len(test)}")
    for tr, (csv_path, i, action, obs, obs_next) in zip(test, prov):
        # load_transitions canonicalizes click args (storage (col,row) -> logical
        # (row,col)); the provenance walk reads raw CSV actions, so canonicalize
        # before comparing (same convention as the drive-action check below).
        if tr.x_t != obs or tr.x_t1 != obs_next or tr.action != action:
            raise AssertionError(f"{game}: provenance misaligned at {csv_path} row {i}")
        tr._prov = (csv_path, i)
        tr._orig_action = action

    if not keep_params:
        efm.collapse_actions(train)
        efm.collapse_actions(test)
    rng = random.Random(seed)
    train_pool = list(train)
    rng.shuffle(train_pool)
    test_pool = list(test)
    rng.shuffle(test_pool)
    _, selected = balanced_split(test_pool, test_n, 10**9, rng)
    return selected


class DriveSim:
    """Branch executor over one verified source drive."""

    def __init__(self, game: str, drive_csv: Path):
        self.game = game
        self.drive_csv = drive_csv
        self.rows = T.read_rows(drive_csv)
        self.actions = [(r.get("Action") or "").strip() for r in self.rows]
        self.problems = T.verify_drive_in_sim(game, self.rows)
        self._cache: dict[tuple[int, str], str | None] = {}

    def outcome(self, t: int, action: str) -> str | None:
        """Rendered grid after replaying the drive to step t then taking `action`."""
        key = (t, action)
        if key not in self._cache:
            env, obs = T.make_env(self.game, max_steps=t + 5)
            for k in range(t):
                obs, _, term, _, _ = env.step(self.actions[k])
                if term:
                    self._cache[key] = None
                    return None
            # `action` is a candidate/predicted action, already row-major, so it is
            # passed to the engine unchanged (stored clicks are `click ROW COL`).
            obs, _, _, _, _ = env.step(action)
            self._cache[key] = T.obs_grid(obs)
        return self._cache[key]

    def click_enum(self, t: int, target: str, start_grid: list) -> tuple[bool, str | None]:
        """Does clicking ANY cell reproduce the target grid? Non-background first."""
        cells = sorted(
            ((r, c) for r in range(len(start_grid)) for c in range(len(start_grid[0]))),
            key=lambda rc: start_grid[rc[0]][rc[1]] == "black",
        )
        for r, c in cells:
            if grids_equal(self.outcome(t, f"click {r} {c}"), target):
                return True, f"click {r} {c}"
        return False, None


def grids_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return a == b or json.loads(a) == json.loads(b)


def rescore_game(game: str, result: dict, config: dict, data_root: Path) -> dict:
    transitions = split_with_provenance(
        game, data_root, config["test_dir"], config["seed"],
        config["context_k"], config["test_n"],
    )
    rows_by_idx: dict[int, list[dict]] = {}
    for row in result["rows"]:
        rows_by_idx.setdefault(row["idx"], []).append(row)
    if len(transitions) != len(rows_by_idx):
        raise AssertionError(f"{game}: split size {len(transitions)} != rows {len(rows_by_idx)}")

    sources = resolve_sources(game, data_root)
    sims: dict[Path, DriveSim] = {}
    slice_ok: dict[str, list[str]] = {}
    for ep_name, drive_csvs in sources.items():
        slice_rows = T.read_rows(data_root / game / "test50" / ep_name / "trajectory.csv")
        problems = []
        for drive_csv in drive_csvs:
            if drive_csv not in sims:
                sims[drive_csv] = DriveSim(game, drive_csv)
            problems += T.match_slice_to_drive(slice_rows, sims[drive_csv].rows)
        slice_ok[ep_name] = problems

    slice_cache: dict[Path, list[dict]] = {}
    out_rows, n_enum = [], 0
    for idx, tr in enumerate(transitions):
        csv_path, row_i = tr._prov
        ep_name = csv_path.parent.name
        ep_sims = [sims[p] for p in sources[ep_name] if not sims[p].problems]
        sim = ep_sims[0] if ep_sims else sims[sources[ep_name][0]]
        if csv_path not in slice_cache:
            slice_cache[csv_path] = T.read_rows(csv_path)
        t = int(slice_cache[csv_path][row_i]["Step"])
        target = T.grid_json(tr.x_t1)
        replay_ok = bool(ep_sims) and not slice_ok[ep_name]
        stored = rows_by_idx[idx][0]["raw_grid_start"]
        if json.loads(efm.canonical_grid(tr.x_t)) != json.loads(stored):
            raise AssertionError(f"{game} idx {idx}: split drifted from eval rows")
        # ground truth really is the drive action at t. tr.action is canonicalized
        # (row,col) by load_transitions; the drive's raw action is interpreter-conv,
        # so canonicalize it before comparing.
        drive_act = sim.actions[t]
        if (drive_act.split()[0] if not GAMES[game][1] else drive_act) != tr.action:
            raise AssertionError(
                f"{game} idx {idx}: drive action {sim.actions[t]!r} (canon {drive_act!r}) "
                f"!= truth {tr.action!r}"
            )

        def branch(action: str):
            """Outcome grids under every candidate drive; None marks disagreement."""
            grids = [s.outcome(t, action) for s in ep_sims]
            agree = all(grids_equal(g, grids[0]) or g == grids[0] for g in grids[1:])
            return grids[0], agree

        for row in rows_by_idx[idx]:
            pred, truth = row["pred"], row["truth"]
            if row["truth"] != tr.action:
                raise AssertionError(f"{game} idx {idx}: truth mismatch vs eval row")
            entry = {
                "idx": idx, "mode": row["mode"], "truth": truth,
                "truth_full": tr._orig_action, "pred": pred,
                "exact": float(row["exact"]),
                "episode": ep_name, "drive_step": t,
                "drives": [str(s.drive_csv) for s in ep_sims],
                "replay_ok": replay_ok,
            }
            if pred is None:
                entry.update(sim_correct=False, sim_method="no-pred")
            elif pred == truth:
                entry.update(sim_correct=True, sim_method="exact-match")
            elif not replay_ok:
                entry.update(sim_correct=pred == truth, sim_method="replay-broken")
            elif pred in DIRECTIONS or pred.startswith("click "):
                grid, agree = branch(pred)
                if agree:
                    entry.update(sim_correct=grids_equal(grid, target),
                                 sim_method="sim-exec")
                else:
                    entry.update(sim_correct=pred == truth,
                                 sim_method="ambiguous-source")
            elif pred == "click":
                start_grid = json.loads(T.grid_json(tr.x_t))
                results = [s.click_enum(t, target, start_grid) for s in ep_sims]
                n_enum += len(ep_sims)
                if all(ok == results[0][0] for ok, _ in results):
                    entry.update(sim_correct=results[0][0],
                                 sim_method=f"click-enum:{results[0][1]}")
                else:
                    entry.update(sim_correct=pred == truth,
                                 sim_method="ambiguous-source")
            else:
                entry.update(sim_correct=False, sim_method=f"unknown-action:{pred!r}")
            out_rows.append(entry)

    summary = {}
    # mode-agnostic: raw/learned from the LLM eval, program from the WC arm, etc.
    for mode in sorted({r["mode"] for r in out_rows}):
        rows = [r for r in out_rows if r["mode"] == mode]
        summary[mode] = {
            "n": len(rows),
            "exact": sum(r["exact"] for r in rows) / len(rows),
            "sim": sum(r["sim_correct"] for r in rows) / len(rows),
            "flipped": sum(1 for r in rows if r["sim_correct"] and not r["exact"]),
            "replay_broken": sum(1 for r in rows if not r["replay_ok"]),
            "ambiguous": sum(1 for r in rows if r["sim_method"] == "ambiguous-source"),
        }
    return {
        "game": game, "n": len(transitions),
        "sources": {ep: [str(p) for p in ps] for ep, ps in sources.items()},
        "drive_problems": {str(p): s.problems for p, s in sims.items() if s.problems},
        "slice_problems": {ep: v for ep, v in slice_ok.items() if v},
        "click_enumerations": n_enum,
        "summary": summary, "rows": out_rows,
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Test50 ID: exact-match vs simulator-grounded rescoring", "",
        "A prediction is sim-correct iff executing it in the Autumn engine at the",
        "recorded state s_t reproduces the recorded s_t+1 grid (drives replayed from",
        "seed-0 reset; bare `click` judged by enumerating cells). Exact-match rows are",
        "sim-correct by construction, so sim >= exact everywhere; the delta is the",
        "action-aliasing unfairness of exact matching.", "",
    ]
    # mode-agnostic columns: whatever modes the source eval carried (raw/learned/program).
    modes = sorted({m for res in payload["results"] for m in res["summary"]})
    lines += [
        "| game | n | "
        + " | ".join(f"{m} exact | {m} sim | {m} Δ" for m in modes)
        + " | flipped " + "/".join(modes) + " |",
        "|---|---:|" + "---:|" * (3 * len(modes)) + "---:|",
    ]
    agg = {m: {"exact": [], "sim": []} for m in modes}
    for res in payload["results"]:
        s = res["summary"]
        for m in modes:
            agg[m]["exact"].append(s[m]["exact"])
            agg[m]["sim"].append(s[m]["sim"])
        lines.append(
            f"| {res['game']} | {res['n']} "
            + "".join(
                f"| {s[m]['exact']:.3f} | {s[m]['sim']:.3f} "
                f"| +{s[m]['sim'] - s[m]['exact']:.3f} "
                for m in modes
            )
            + "| " + "/".join(str(s[m]["flipped"]) for m in modes) + " |"
        )
    if payload["results"]:
        n = len(payload["results"])
        lines.append(
            "| **average** | — "
            + "".join(
                f"| {sum(agg[m]['exact'])/n:.3f} | {sum(agg[m]['sim'])/n:.3f} "
                f"| +{(sum(agg[m]['sim']) - sum(agg[m]['exact']))/n:.3f} "
                for m in modes
            )
            + "| — |"
        )
    if payload.get("skipped"):
        lines += ["", "Skipped (no verified source drives yet): "
                  + ", ".join(f"{g} ({why})" for g, why in payload["skipped"].items())]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-json", type=Path,
                    default=REPO / "logs/id_eval_test50_raw_vs_learned.json")
    ap.add_argument("--out", type=Path,
                    default=REPO / "logs/id_eval_test50_sim_rescore")
    ap.add_argument("--games", type=str, default="",
                    help="comma-separated subset; default = all games in the ID json")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    payload_in = json.loads(args.id_json.read_text())
    config = payload_in["config"]
    data_root = Path(config["data_root"])
    wanted = set(filter(None, args.games.split(",")))

    out_json = args.out.with_suffix(".json")
    results, skipped = [], {}
    if args.resume and out_json.exists():
        prior = json.loads(out_json.read_text())
        results = prior.get("results", [])
    done = {r["game"] for r in results}

    started = time.time()
    for result in payload_in["results"]:
        game = result["game"]
        if wanted and game not in wanted:
            continue
        if game in done:
            print(f"[resume] {game}", flush=True)
            continue
        t0 = time.time()
        try:
            res = rescore_game(game, result, config, data_root)
        except (RuntimeError, FileNotFoundError) as exc:
            skipped[game] = str(exc)
            print(f"[skip] {game}: {exc}", flush=True)
            continue
        results.append(res)
        s = res["summary"]
        print(
            f"[done] {game}: "
            + " ".join(f"{m} {s[m]['exact']:.3f}->{s[m]['sim']:.3f}"
                       f"(+{s[m]['flipped']} flipped)" for m in sorted(s))
            + f" ({res['click_enumerations']} enums, {time.time()-t0:.1f}s)",
            flush=True,
        )
        payload = {
            "config": {
                "id_source": str(args.id_json), "seed": config["seed"],
                "test_n": config["test_n"], "test_dir": config["test_dir"],
                "context_k": config["context_k"],
                "protocol": "seed-0 drive replay to s_t; predicted action executed; "
                            "exact rendered-grid match against recorded s_t+1",
            },
            "elapsed_seconds": time.time() - started,
            "results": results, "skipped": skipped,
        }
        out_json.write_text(json.dumps(payload, indent=2))

    payload = {
        "config": {
            "id_source": str(args.id_json), "seed": config["seed"],
            "test_n": config["test_n"], "test_dir": config["test_dir"],
            "context_k": config["context_k"],
            "protocol": "seed-0 drive replay to s_t; predicted action executed; "
                        "exact rendered-grid match against recorded s_t+1",
        },
        "elapsed_seconds": time.time() - started,
        "results": results, "skipped": skipped,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    report = render_report(payload)
    args.out.with_suffix(".md").write_text(report)
    print(report, flush=True)
    print(f"wrote {out_json} and {args.out.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
