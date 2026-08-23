"""Write aggregate artifacts for completed Autumn full-trajectory GEPA runs.

The older pilot batch predates persisted held-out forward-dynamics traces.  Its
held-out FD fields therefore remain null except for ada85, which has a separate
full-trajectory test50 re-evaluation.  Validation ID/FD components are recovered
from the per-candidate prediction sidecar for the selected candidate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = ROOT / "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50"
PHASE_ROOTS = [
    ROOT
    / "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50_fulltestctx_phase1",
    ROOT
    / "logs/clean_sweep_gepa_cd3_fulltraj_aug30_mb15_test50_fulltestctx_phase2",
]
OUTPUT_ROOT = ROOT / "logs/autumn_gepa_fulltraj_aug30_completed_results"
REFRESH_ID = ROOT / "logs/id_eval_test50_raw_vs_learned.json"
REFRESH_FD = ROOT / "logs/forward_eval_test50_raw_vs_learned.json"


def content_hash(*parts: str) -> str:
    digest = hashlib.md5()
    for part in parts:
        digest.update((part or "").encode("utf-8"))
        digest.update(bytes([0]))
    return digest.hexdigest()[:16]


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def selected_validation_metrics(run_dir: Path) -> dict:
    gepa_dir = run_dir / "gepa_run_seed1"
    log_text = (gepa_dir / "run_log.txt").read_text(errors="replace")
    selected = re.findall(
        r"Best program as per aggregate score on valset:\s*(\d+)", log_text
    )
    best_scores = re.findall(r"Best score on valset:\s*([0-9.]+)", log_text)
    if not selected:
        return {
            "selected_candidate_index": None,
            "val_id": None,
            "val_fd_exact": None,
            "val_composite": float(best_scores[-1]) if best_scores else None,
        }

    candidate_index = int(selected[-1])
    candidates = load_json(gepa_dir / "candidates.json")
    candidate = candidates[candidate_index]
    candidate_hash = content_hash(
        candidate.get("perception", ""), candidate.get("world_knowledge", "")
    )

    # A candidate may be evaluated on overlapping minibatches. Retaining its
    # latest record per transition reconstructs its final 30-item validation set.
    latest_by_transition: dict[str, dict] = {}
    with (gepa_dir / "predictions.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("cand_hash") == candidate_hash:
                latest_by_transition[record["tr_hash"]] = record
    records = list(latest_by_transition.values())
    if not records:
        return {
            "selected_candidate_index": candidate_index,
            "val_id": None,
            "val_fd_exact": None,
            "val_composite": float(best_scores[-1]) if best_scores else None,
        }

    return {
        "selected_candidate_index": candidate_index,
        "val_n": len(records),
        "val_id": sum(row["id_score"] for row in records) / len(records),
        "val_fd_exact": sum(row["fd_score"] for row in records) / len(records),
        "val_composite": sum(row["score"] for row in records) / len(records),
    }


def common_run_metrics(run_dir: Path) -> dict:
    gepa_dir = run_dir / "gepa_run_seed1"
    raw_path = run_dir / "test_trace_raw_seed1.json"
    candidates_path = gepa_dir / "candidates.json"
    result = selected_validation_metrics(run_dir)
    result.update(
        {
            "raw_frame_test_id": load_json(raw_path).get("acc")
            if raw_path.exists()
            else None,
            "candidate_count": len(load_json(candidates_path))
            if candidates_path.exists()
            else None,
        }
    )
    return result


def pilot_rows() -> tuple[list[dict], list[dict]]:
    completed: list[dict] = []
    incomplete: list[dict] = []
    if not PILOT_ROOT.exists():
        return completed, incomplete

    for run_dir in sorted(PILOT_ROOT.glob("*_seed1*")):
        game = run_dir.name.split("_seed1", 1)[0]
        trace_path = run_dir / "test_trace_gepa_seed1.json"
        best_path = run_dir / "best_perception_gepa_seed1.py"
        if not (trace_path.exists() and best_path.exists()):
            incomplete.append(
                {
                    "game": game,
                    "batch": "pilot",
                    "run_dir": str(run_dir),
                    "reason": "no completed learned held-out test trace",
                }
            )
            continue

        trace = load_json(trace_path)
        row = {
            "game": game,
            "seed": 1,
            "batch": "pilot",
            "run_dir": str(run_dir),
            "train_n": 30,
            "train_minibatch_n": 15,
            "train_context": "full_trajectory",
            "test_context": "episode_split",
            "test_n": len(trace.get("records", [])),
            "test_id": trace.get("acc"),
            "test_fd_exact": None,
            "test_fd_partial": None,
            "test_composite_id_fd_exact": None,
            "fd_persistence": "missing_historical_test_trace",
            "run_cost_usd": None,
            "test_eval_cost_usd": None,
            "elapsed_seconds": None,
        }
        row.update(common_run_metrics(run_dir))

        # ada85 was explicitly re-evaluated after the episode-boundary bug was
        # fixed, with both exact and partial FD persisted.
        full_summary_path = run_dir / "test50_fulltraj_summary_seed1.json"
        if full_summary_path.exists():
            summary = load_json(full_summary_path)
            row.update(
                {
                    "test_context": "full_trajectory",
                    "test_n": summary["test_n"],
                    "test_id": summary["inverse_accuracy"],
                    "test_fd_exact": summary["fd_exact"],
                    "test_fd_partial": summary["fd_partial"],
                    "test_composite_id_fd_exact": summary[
                        "composite_id_fd_exact"
                    ],
                    "fd_persistence": "complete_exact_and_partial",
                    "test_eval_cost_usd": summary["total_cost"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "original_episode_split_test_id": summary[
                        "original_truncated_context_inverse_accuracy"
                    ],
                }
            )
        completed.append(row)
    return completed, incomplete


def phase_rows() -> tuple[list[dict], list[dict]]:
    completed: list[dict] = []
    incomplete: list[dict] = []
    for phase_number, phase_root in enumerate(PHASE_ROOTS, start=1):
        if not phase_root.exists():
            continue
        for run_dir in sorted(phase_root.glob("*_seed1")):
            game = run_dir.name.removesuffix("_seed1")
            summary_path = run_dir / "test_summary_gepa_seed1.json"
            if not summary_path.exists():
                incomplete.append(
                    {
                        "game": game,
                        "batch": f"remaining_phase{phase_number}",
                        "run_dir": str(run_dir),
                        "reason": "run has not written final test summary",
                    }
                )
                continue

            summary = load_json(summary_path)
            stdout_path = run_dir / "stdout.txt"
            stdout = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
            footer = re.findall(
                r"GEPA \(pareto \+ strong reflection\)\s+"
                r"[0-9.]+\s+([0-9.]+)\s+([0-9]+)",
                stdout,
            )
            # current rexpure logs report "nodes explored = N"; older gepa/rexpure
            # logs reported "total metric calls = N" (== N/train_n nodes). Accept both.
            nodes = re.findall(r"(?:nodes explored|total metric calls)\s*=\s*(\d+)", stdout)
            row = {
                "game": game,
                "seed": 1,
                "batch": f"remaining_phase{phase_number}",
                "run_dir": str(run_dir),
                "train_n": 30,
                "train_minibatch_n": 15,
                "train_context": "full_trajectory",
                "test_context": "full_trajectory",
                "test_n": summary["n_test"],
                "test_id": summary["inverse_accuracy"],
                "test_fd_exact": summary["forward_score"]
                if summary.get("forward_scorer") == "exact"
                else None,
                "test_fd_partial": summary["forward_score"]
                if summary.get("forward_scorer") == "textdiff"
                else None,
                "test_composite_id_fd_exact": (
                    (summary["inverse_accuracy"] + summary["forward_score"]) / 2
                    if summary.get("forward_scorer") == "exact"
                    else None
                ),
                "fd_persistence": f"complete_{summary['forward_scorer']}",
                "run_cost_usd": float(footer[-1][0]) if footer else None,
                "test_eval_cost_usd": None,
                "elapsed_seconds": int(footer[-1][1]) if footer else None,
                "nodes_explored": int(nodes[-1]) if nodes else None,
            }
            row.update(common_run_metrics(run_dir))
            completed.append(row)
    return completed, incomplete



def apply_refreshed_test50(completed: list[dict]) -> bool:
    """Overlay the unified test50 ID/FD evaluation when it is available."""
    if not (REFRESH_ID.exists() and REFRESH_FD.exists()):
        return False

    id_payload = load_json(REFRESH_ID)
    fd_payload = load_json(REFRESH_FD)
    id_by_game = {row["game"]: row for row in id_payload["results"]}
    fd_by_game = {row["game"]: row for row in fd_payload["results"]}
    overrides = id_payload.get("config", {}).get("artifact_overrides", {})
    refreshed = False

    for row in completed:
        game = row["game"]
        if game not in overrides or game not in id_by_game or game not in fd_by_game:
            continue
        inverse = id_by_game[game]
        forward = fd_by_game[game]
        if Path(inverse["artifact_dir"]) != Path(row["run_dir"]):
            raise ValueError(f"{game}: refreshed ID does not use completed run artifact")
        if Path(forward["artifact_dir"]) != Path(row["run_dir"]):
            raise ValueError(f"{game}: refreshed FD does not use completed run artifact")

        row["historical_test_metrics"] = {
            key: row.get(key)
            for key in (
                "test_context", "test_n", "raw_frame_test_id", "test_id",
                "test_fd_exact", "test_fd_partial",
                "test_composite_id_fd_exact", "test_eval_cost_usd",
            )
        }
        raw_id = inverse["summary"]["raw"]
        learned_id = inverse["summary"]["learned"]
        learned_fd = forward["summary"]["learned"]
        row.update(
            {
                "test_context": "test50_slice_bounded_k9",
                "test_n": inverse["test_n"],
                "raw_frame_test_id": raw_id["exact"],
                "test_id": learned_id["exact"],
                "test_fd_exact": learned_fd["exact"],
                "test_fd_partial": learned_fd["partial"],
                "test_composite_id_fd_exact": (
                    learned_id["exact"] + learned_fd["exact"]
                ) / 2,
                "fd_persistence": "refreshed_exact_partial_prompts_responses",
                "test_eval_cost_usd": (
                    raw_id.get("cost", 0.0)
                    + learned_id.get("cost", 0.0)
                    + sum(
                        float(item.get("cost", 0.0))
                        for item in forward["rows"]
                        if item["mode"] == "learned"
                    )
                ),
                "test_eval_source": (
                    "id_eval_test50_raw_vs_learned.json + "
                    "forward_eval_test50_raw_vs_learned.json"
                ),
            }
        )
        refreshed = True
    return refreshed

def format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def write_outputs(completed: list[dict], incomplete: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    refreshed = apply_refreshed_test50(completed)
    generated_at = datetime.now(ZoneInfo("America/New_York")).isoformat()
    completed.sort(key=lambda row: (row["batch"], row["game"]))

    payload = {
        "generated_at": generated_at,
        "selection_objective": "0.5 * inverse_dynamics + 0.5 * forward_dynamics_exact",
        "completed_run_count": len(completed),
        "completed_runs": completed,
        "incomplete_run_directories": incomplete,
        "notes": [
            "All learned test metrics use the exported newer GEPA perception and beliefs.",
            (
                "Test metrics were refreshed with the common test50 slice-bounded "
                "K=9 protocol." if refreshed else
                "Refreshed common-protocol test50 artifacts were not available."
            ),
            "Raw ID and learned ID use the same held-out transitions and fixed action choices.",
            "FD partial is diagnostic only and was not part of GEPA candidate selection.",
            "Each refreshed row retains its prior values under historical_test_metrics.",
        ]
    }
    (OUTPUT_ROOT / "completed_results.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    columns = [
        "game",
        "seed",
        "batch",
        "train_n",
        "train_minibatch_n",
        "train_context",
        "test_context",
        "val_n",
        "val_id",
        "val_fd_exact",
        "val_composite",
        "test_n",
        "raw_frame_test_id",
        "test_id",
        "test_fd_exact",
        "test_fd_partial",
        "test_composite_id_fd_exact",
        "candidate_count",
        "nodes_explored",
        "run_cost_usd",
        "test_eval_cost_usd",
        "elapsed_seconds",
        "fd_persistence",
        "test_eval_source",
        "run_dir",
    ]
    with (OUTPUT_ROOT / "completed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)

    lines = [
        "# Completed Autumn full-trajectory GEPA results",
        "",
        f"Generated: {generated_at}",
        "",
        "Selection objective: `0.5 × ID + 0.5 × FD-exact`.",
        "",
        "| Game | Batch | Test context | Val ID | Val FD | Val composite | Test n | Raw ID | Learned ID | Test FD exact | Test FD partial | Test composite |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in completed:
        lines.append(
            "| {game} | {batch} | {test_context} | {val_id} | {val_fd} | "
            "{val_comp} | {test_n} | {raw_id} | {test_id} | {fd_exact} | "
            "{fd_partial} | {test_comp} |".format(
                game=row["game"],
                batch=row["batch"],
                test_context=row["test_context"],
                val_id=format_metric(row.get("val_id")),
                val_fd=format_metric(row.get("val_fd_exact")),
                val_comp=format_metric(row.get("val_composite")),
                test_n=row["test_n"],
                raw_id=format_metric(row.get("raw_frame_test_id")),
                test_id=format_metric(row.get("test_id")),
                fd_exact=format_metric(row.get("test_fd_exact")),
                fd_partial=format_metric(row.get("test_fd_partial")),
                test_comp=format_metric(row.get("test_composite_id_fd_exact")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Learned metrics use each run exported newer GEPA perception and beliefs.",
            "- All displayed test metrics use the common test50 slice-bounded K=9 protocol; prior values remain in JSON under historical_test_metrics.",
            "- Raw ID and learned ID use the same held-out transitions and fixed action choices.",
            "- Raw-frame FD is not rerun here; the separate unified forward report retains its existing raw reference.",
            "- FD-partial is diagnostic only and was not used in candidate selection.",
            "",
            "## Incomplete run directories excluded",
            "",
        ]
    )
    if incomplete:
        for row in incomplete:
            lines.append(
                f"- `{row['game']}` ({row['batch']}): {row['reason']} — `{row['run_dir']}`"
            )
    else:
        lines.append("None.")
    lines.append("")
    (OUTPUT_ROOT / "completed_results.md").write_text("\n".join(lines))


def main() -> None:
    pilot_completed, pilot_incomplete = pilot_rows()
    phase_completed, phase_incomplete = phase_rows()
    completed = pilot_completed + phase_completed
    incomplete = pilot_incomplete + phase_incomplete
    write_outputs(completed, incomplete)
    print(f"wrote {len(completed)} completed runs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
