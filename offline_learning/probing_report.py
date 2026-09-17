"""Coverage-aware summaries and paired comparisons for a frozen probing run."""
from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from probing_common import collision_summary, file_digest, read_jsonl, write_json
from probing_eval import prepared_protocol


# Matches human_replay.GAMES and the paper's names without importing the simulator.
GAME_NAMES = {
    "bt3gb": "Ice", "dq8gc": "Disease", "n2ntd": "Mario", "s2kt7": "Ants",
    "83wkq": "Particles", "eahcw": "Paint", "7www9": "Magnets", "7xf97": "Grow",
    "va6fq": "Sand", "f5w3n": "Space Invaders", "egg": "Egg",
    "colour_lines": "Colour Lines", "diffusion": "Diffusion", "dino": "Dino",
    "logic_gates": "Logic Gates", "SET": "SET",
}


def display_name(game: str) -> str:
    """The English environment name used in paper figures and result tables."""
    return GAME_NAMES.get(game, game)


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        strata = ["all"]
        if row["kind"] == "forward":
            strata.append("changed" if row["changed"] else "unchanged")
        if not row["seen_in_train"]:
            strata.append("train_unseen_state")
        for checkpoint in row["checkpoints"]:
            for stratum in strata:
                key = (row["kind"], row["arm"], checkpoint["label"], row["mode"],
                       row["horizon"], row["game"], stratum)
                grouped[key].append((row, checkpoint))
    result = []
    for key, entries in sorted(grouped.items()):
        kind, arm, checkpoint, mode, horizon, game, stratum = key
        items = [r for r, _ in entries]
        status = Counter(r["status"] for r in items)
        scored = [r for r in items if r["metrics"] is not None]
        complete = len(scored) == len(items)
        fields = sorted({k for r in scored for k, v in r["metrics"].items()
                         if isinstance(v, (float, int))})
        averages = {field: sum(r["metrics"][field] for r in scored) / len(scored) for field in fields}
        distinct = defaultdict(list)
        for row in scored:
            distinct[row["target_grid_sha256"]].append(row["metrics"]["exact"])
        representations = [r for r in items if "representation" in r and not r["representation_errors"]]
        reconstruction_bound = collision_summary([r["representation"] for r in representations],
            [r["target"] for r in representations]) if representations else None
        result.append({"kind": kind, "arm": arm, "checkpoint": checkpoint,
            "mode": mode, "horizon": horizon, "game": game, "stratum": stratum,
            "iteration": entries[0][1].get("iteration"),
            "expected": len(items), "scored": len(scored), "complete": complete,
            "users": len({r["user_id"] for r in items}),
            "unique_target_grids": len({r["target_grid_sha256"] for r in items}),
            "status": dict(status), "metrics": averages if complete else None,
            "partial_metrics": averages,
            "unique_state_exact": sum(sum(vals) / len(vals) for vals in distinct.values()) / len(distinct)
                                  if kind == "reconstruction" and complete and distinct else None,
            "parse_failures": sum(bool(r["metrics"].get("parse_error")) for r in scored),
            "perception_failures": sum(bool(r["representation_errors"]) for r in items),
            "representation_audit_queries": len(representations),
            "mean_representation_characters": sum(len(r["representation"]) for r in representations)
                                              / len(representations) if representations else None,
            "empirical_reconstruction_bound": reconstruction_bound["empirical_reconstruction_bound"]
                                              if reconstruction_bound else None,
            "ambiguous_representations": reconstruction_bound["ambiguous_outputs"]
                                         if reconstruction_bound else None,
            "native_static_fraction": sum(r["native_static"] for r in items) / len(items)
                                      if mode == "learned_native" else None,
            "truncated": sum((r.get("call") or {}).get("finish_reason") == "length" for r in items)})
    return result


def macro_summary(summary: list[dict], games: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in summary:
        if row["stratum"] == "all":
            groups[(row["kind"], row["arm"], row["checkpoint"], row["mode"], row["horizon"])].append(row)
    macro = []
    for key, rows in sorted(groups.items()):
        complete = {r["game"] for r in rows if r["complete"]} == set(games)
        fields = sorted({field for row in rows for field in (row["metrics"] or {})})
        macro.append({"kind": key[0], "arm": key[1], "checkpoint": key[2],
                      "mode": key[3], "horizon": key[4], "complete": complete,
                      "expected_games": len(games), "complete_games": sum(r["complete"] for r in rows),
                      "metrics": {field: sum(r["metrics"][field] for r in rows) / len(games)
                                  for field in fields} if complete else None})
    return macro


def paired_comparison(rows: list[dict], arm: str, mode: str, horizon: int,
                      metric: str, games: list[str], *, draws=2000, seed=0,
                      control_arm: str | None = None) -> dict:
    """Shared Exp(1) user weights preserve repeated users across games.

    This is a descriptive cluster-weight sensitivity interval for fixed artifacts
    and fixed games, not uncertainty over training seeds or the population of games.
    """
    left, right = {}, {}
    for row in rows:
        if row["mode"] != mode or row["horizon"] != horizon:
            continue
        labels = {c["label"] for c in row["checkpoints"]}
        key = (row["game"], row["query_id"])
        if row["arm"] == arm and "final" in labels:
            left[key] = row
        if row["arm"] == (control_arm or arm) and ("final" if control_arm else "first_working") in labels:
            right[key] = row
    base = {"arm": arm, "comparison": f"final minus {control_arm + '/final' if control_arm else 'first_working'}",
            "mode": mode, "horizon": horizon, "metric": metric,
            "interval_method": "global-user exponential-weight sensitivity interval",
            "conditioning": "fixed saved training runs and fixed benchmark games", "draws": draws, "seed": seed}
    if not left or left.keys() != right.keys():
        return {**base, "complete": False, "reason": "missing or unmatched query coverage"}
    by_game = defaultdict(list)
    for key in left:
        a, b = left[key], right[key]
        if a["metrics"] is None or b["metrics"] is None:
            return {**base, "complete": False, "reason": "pending or failed provider responses"}
        if a["user_id"] != b["user_id"] or a["target_grid_sha256"] != b["target_grid_sha256"]:
            raise ValueError("paired query provenance mismatch")
        by_game[key[0]].append((a["user_id"], a["metrics"][metric] - b["metrics"][metric]))
    if set(by_game) != set(games):
        return {**base, "complete": False, "reason": "incomplete game coverage"}
    per_game = {game: sum(delta for _, delta in items) / len(items) for game, items in by_game.items()}
    users = sorted({user for items in by_game.values() for user, _ in items})
    rng, samples = random.Random(seed), []
    for _ in range(draws):
        weights = {user: rng.expovariate(1.0) for user in users}
        values = [sum(weights[u] * delta for u, delta in items) / sum(weights[u] for u, _ in items)
                  for items in by_game.values()]
        samples.append(sum(values) / len(values))
    samples.sort()
    interval = None
    if samples and all(len({u for u, _ in items}) > 1 for items in by_game.values()):
        interval = [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]]
    return {**base, "complete": True, "delta": sum(per_game.values()) / len(games),
            "per_game": per_game, "user_clusters": len(users), "interval_95": interval}


def report(out: Path, *, plots=True, draws=2000) -> dict:
    prepared, protocol = prepared_protocol(out)
    rows = list(read_jsonl(out / "results.jsonl"))
    expected_ids = {j["job_id"] for j in read_jsonl(out / "jobs.jsonl")}
    expected = prepared["jobs"]
    if (len(rows) != expected or len(expected_ids) != expected
            or {r["job_id"] for r in rows} != expected_ids):
        raise ValueError("result coverage does not match the prepared study")
    if any(r["protocol_sha256"] != protocol["protocol_sha256"] for r in rows):
        raise ValueError("mixed evaluation protocols in results")
    games, reference = sorted(protocol["games"], key=display_name), protocol["reference_arm"]
    summary = summarize(rows)
    macro = macro_summary(summary, games)
    comparisons = []
    for h in sorted({r["horizon"] for r in rows if r["kind"] == "forward"}):
        comparisons.append(paired_comparison(rows, reference, "learned_raw", h, "change_f1", games, draws=draws))
    comparisons.append(paired_comparison(rows, reference, "learned", 0, "exact", games, draws=draws))
    for arm in sorted({r["arm"] for r in rows} - {reference, "baseline"}):
        if any(r["horizon"] == 4 for r in rows):
            comparisons.append(paired_comparison(rows, arm, "learned_raw", 4, "change_f1", games,
                                                draws=draws, control_arm=reference))
    calls = {r["request_key"]: r["call"] for r in rows if r.get("call")}
    cost_values = [(c.get("usage") or {}).get("cost") for c in calls.values()]
    named_summary = [{**r, "game": display_name(r["game"]), "game_id": r["game"]} for r in summary]
    for comparison in comparisons:
        if "per_game" in comparison:
            comparison["per_game"] = {display_name(game): value
                                      for game, value in comparison["per_game"].items()}
    points = multistep_points(summary, reference)
    payload = {"protocol_sha256": protocol["protocol_sha256"],
               "results_sha256": file_digest(out / "results.jsonl"),
               "reporter_sha256": file_digest(Path(__file__)), "coverage": dict(Counter(r["status"] for r in rows)),
               "unique_calls": len(calls), "reported_cost": sum(float(c) for c in cost_values if c is not None),
               "calls_without_cost": sum(c is None for c in cost_values),
               "per_game": named_summary, "macro": macro, "paired_comparisons": comparisons,
               "fd1_vs_multistep": points}
    write_json(out / "summary.json", payload)
    flat = []
    for row in named_summary:
        flat.append({**{k: v for k, v in row.items() if k not in {"status", "metrics", "partial_metrics"}},
                     **(row["metrics"] or {})})
    fields = sorted({key for row in flat for key in row})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        writer.writerows(flat)
    with (out / "fd1_vs_multistep.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, ["game", "game_id", "arm", "checkpoint", "horizon",
                                    "fd1_change_f1", "fdh_change_f1"])
        writer.writeheader()
        writer.writerows(points)
    lines = ["# Probing analysis", "", f"Population: **{protocol['evaluation_population']}**. "
             "These are conditional forecasts on recorded human actions and fixed saved training runs.", "",
             "Native feature scores have candidate-dependent targets. Raw-target scores compare the same grids.", "",
             "Environments: " + ", ".join(display_name(game) for game in games) + ".", "",
             "Result tables use English environment names in `game`; `game_id` retains the source identifier.", "",
             "## Coverage", "", " | ".join(f"{k}: {v}" for k, v in payload["coverage"].items()), "",
             f"Unique stored model requests: {len(calls)}; reported cost: ${payload['reported_cost']:.4f}; "
             f"requests without reported cost: {payload['calls_without_cost']}.", "",
             "## Macro scores", "", "Incomplete groups have no headline score; partial means are only in summary.json.", "",
             "| task | arm | checkpoint | mode | horizon | complete games | exact | change F1 |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    for row in macro:
        m = row["metrics"] or {}
        lines.append(f"| {row['kind']} | {row['arm']} | {row['checkpoint']} | {row['mode']} | "
                     f"{row['horizon']} | {row['complete_games']}/{row['expected_games']} | "
                     f"{fmt(m.get('exact'))} | {fmt(m.get('change_f1'))} |")
    lines += ["", "## Paired comparisons", "",
              "Intervals vary user-cluster weights jointly across games. They do not estimate training-seed "
              "uncertainty or establish a causal effect. Per-game details and denominators are in summary.json.", ""]
    for c in comparisons:
        description = f"{c['arm']}, {c['mode']}, h={c['horizon']}, {c['comparison']}"
        lines.append(f"- {description}: " + (f"delta={c['delta']:+.3f}, interval={c['interval_95']}"
                     if c["complete"] else c["reason"]))
    lines += ["", "## FD-1 versus FD-h", "",
              f"The combined scatter contains {len(points)} environment/checkpoint/horizon pairs. "
              "Each point pairs FD-1 with FD-h for the same environment and checkpoint; color and marker "
              "indicate the longer horizon. Both axes measure raw-grid change F1. "
              "Only complete groups with a matching FD-1 score are included.", "",
              "[Scatter plot](fd1_vs_multistep.png) · [PDF](fd1_vs_multistep.pdf) · "
              "[Plotted values](fd1_vs_multistep.csv)"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plots:
        figures(summary, games, reference, out)
    return payload


def multistep_points(summary: list[dict], reference: str) -> list[dict]:
    """Pair each longer horizon with FD-1 from the same environment and checkpoint."""
    forward = [r for r in summary if r["stratum"] == "all" and r["complete"]
               and r["kind"] == "forward" and r["arm"] == reference and r["mode"] == "learned_raw"]
    one = {(r["game"], r["checkpoint"]): r for r in forward if r["horizon"] == 1}
    points = []
    for row in forward:
        first = one.get((row["game"], row["checkpoint"]))
        if row["horizon"] <= 1 or first is None:
            continue
        points.append({"game": display_name(row["game"]), "game_id": row["game"],
                       "arm": reference, "checkpoint": row["checkpoint"], "horizon": row["horizon"],
                       "fd1_change_f1": first["metrics"]["change_f1"],
                       "fdh_change_f1": row["metrics"]["change_f1"]})
    return sorted(points, key=lambda r: (r["game"], r["checkpoint"], r["horizon"]))


def figures(summary: list[dict], games: list[str], reference: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [r for r in summary if r["stratum"] == "all" and r["complete"]]
    columns, rows_n = min(3, len(games)), math.ceil(len(games) / 3)

    def panels(name, ylabel, plot):
        fig, axes = plt.subplots(rows_n, columns, figsize=(4.2 * columns, 3.2 * rows_n), squeeze=False)
        for ax, game in zip(axes.flat, games):
            plot(ax, [r for r in selected if r["game"] == game])
            ax.set(title=display_name(game), ylabel=ylabel, ylim=(-0.03, 1.03))
            ax.grid(alpha=0.2)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7)
        for ax in list(axes.flat)[len(games):]:
            ax.set_visible(False)
        fig.suptitle("Completed groups only; see REPORT.md for coverage", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)

    def horizon(ax, data):
        specs = [(reference, label, "learned_raw") for label in ("first_working", "p25", "p50", "p75", "final")]
        specs += [("baseline", mode, mode) for mode in ("raw", "lossless", "copy")]
        for arm, checkpoint, mode in specs:
            curve = sorted((r for r in data if r["kind"] == "forward" and r["arm"] == arm
                            and r["checkpoint"] == checkpoint and r["mode"] == mode), key=lambda r: r["horizon"])
            if curve:
                ax.plot([r["horizon"] for r in curve], [r["metrics"]["change_f1"] for r in curve],
                        marker="o", label=f"{arm}/{checkpoint}")
        ax.set_xlabel("Forecast horizon")

    def native_horizon(ax, data):
        for checkpoint in ("first_working", "p25", "p50", "p75", "final"):
            curve = sorted((r for r in data if r["arm"] == reference and r["mode"] == "learned_native"
                            and r["checkpoint"] == checkpoint), key=lambda r: r["horizon"])
            if curve:
                ax.plot([r["horizon"] for r in curve], [r["metrics"]["exact"] for r in curve],
                        marker="o", label=checkpoint)
        ax.set_xlabel("Forecast horizon (targets vary with P)")

    panels("forward_horizons", "Raw-grid change F1", horizon)
    panels("native_forward_horizons", "Native exact match", native_horizon)

    points = multistep_points(summary, reference)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    markers = ("o", "s", "^", "D", "v", "P", "X")
    for index, h in enumerate(sorted({r["horizon"] for r in points})):
        future = [r for r in points if r["horizon"] == h]
        ax.scatter([r["fd1_change_f1"] for r in future], [r["fdh_change_f1"] for r in future],
                   label=f"h={h}", marker=markers[index % len(markers)], s=55,
                   alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="FD-h = FD-1", zorder=1)
    ax.set(title="FD-1 versus FD-h across environments", xlabel="FD-1 change F1", ylabel="FD-h change F1",
           xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect="equal")
    ax.grid(alpha=0.2)
    ax.legend(title="Forecast horizon", fontsize=9)
    n_games = len({r["game_id"] for r in points})
    fig.text(0.5, 0.015, f"{n_games} environments · {len(points)} paired points · completed groups only",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fd1_vs_multistep.pdf")
    fig.savefig(out / "fd1_vs_multistep.png", dpi=150)
    plt.close(fig)
