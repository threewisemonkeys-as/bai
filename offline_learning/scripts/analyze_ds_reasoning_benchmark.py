"""Offline, corrected analysis for the DeepSeek reasoning benchmark records."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def median(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return statistics.median(values) if values else None


def mean(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return statistics.mean(values) if values else None


def percentile(values, fraction):
    values = sorted(value for value in values if isinstance(value, (int, float)))
    if not values:
        return None
    index = fraction * (len(values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def is_valid(record):
    checks = record["automatic_quality"]
    return all(checks.get(key, True) for key in
               ("nonempty", "has_fence", "compiles", "all_tags", "under_word_limits"))


def arm_summary(records):
    latency = [record["elapsed_s"] for record in records]
    ttft = [record["performance"].get("ttft_reasoning_s") or
            record["performance"].get("ttft_content_s") for record in records]
    content_ttft = [record["performance"].get("ttft_content_s") for record in records]
    completion = [record["usage"].get("output_tokens") for record in records]
    reasoning = [record["usage"].get("reasoning_tokens") for record in records]
    visible = [(record["usage"].get("output_tokens") or 0) -
               (record["usage"].get("reasoning_tokens") or 0) for record in records]
    corrected_tps = [(record["usage"].get("output_tokens") or 0) /
                     record["performance"]["generation_s"] for record in records]
    cache = [record["usage"].get("cache_hit_tokens") or 0 for record in records]
    inputs = [record["usage"].get("input_tokens") or 0 for record in records]
    cache_ratios = [cached / supplied if supplied else 0 for cached, supplied in zip(cache, inputs)]
    return {
        "n": len(records),
        "latency_s": {
            "mean": mean(latency), "median": median(latency),
            "p90": percentile(latency, .9), "max": max(latency),
        },
        "ttft_reasoning_s_median": median(ttft),
        "ttft_content_s_median": median(content_ttft),
        "completion_tokens_median": median(completion),
        "reasoning_tokens_median": median(reasoning),
        "visible_tokens_median": median(visible),
        "reasoning_share_of_completion_median": median([
            reason / total if total else None for reason, total in zip(reasoning, completion)
        ]),
        "corrected_generated_tokens_per_s": {
            "mean": mean(corrected_tps), "median": median(corrected_tps),
        },
        "cache_hit_tokens_median": median(cache),
        "cache_hit_ratio_median": median(cache_ratios),
        "aggregate_cache_hit_ratio": sum(cache) / sum(inputs) if sum(inputs) else None,
        "empty_outputs": sum(not record["text"] for record in records),
        "completion_cap_hits": sum((record["usage"].get("output_tokens") or 0) >= 32768
                                   for record in records),
        "automatic_valid": sum(is_valid(record) for record in records),
        "cost_reported_usd": sum(record["usage"].get("cost") or 0 for record in records),
    }


def paired(records, left, right):
    lookup = {(record["case_id"], record["backend"], record["effort"]): record
              for record in records}
    ratios = []
    differences = []
    left_wins = 0
    for case_id in sorted({record["case_id"] for record in records}):
        a = lookup[(case_id, *left)]["elapsed_s"]
        b = lookup[(case_id, *right)]["elapsed_s"]
        ratios.append(a / b)
        differences.append(a - b)
        left_wins += a < b
    return {
        "left": f"{left[0]}_{left[1]}", "right": f"{right[0]}_{right[1]}",
        "median_latency_ratio_left_over_right": median(ratios),
        "geometric_mean_latency_ratio_left_over_right": math.exp(mean([math.log(x) for x in ratios])),
        "median_latency_difference_s_left_minus_right": median(differences),
        "left_faster_cases": left_wins, "pairs": len(ratios),
    }


def main(args):
    root = Path(args.out_dir)
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for record in records:
        groups[f"{record['backend']}_{record['effort']}"] .append(record)

    breakdown = {}
    for dimension in ("kind", "component"):
        breakdown[dimension] = {}
        for value in sorted({record[dimension] for record in records}):
            breakdown[dimension][value] = {
                arm: arm_summary([record for record in arm_records if record[dimension] == value])
                for arm, arm_records in groups.items()
            }

    provider_metadata = Counter()
    regions = Counter()
    attempts = Counter()
    fingerprints = Counter()
    for record in records:
        if record["backend"] == "openrouter":
            metadata = record["router_metadata"]
            selected = [endpoint.get("provider")
                        for endpoint in metadata.get("endpoints", {}).get("available", [])
                        if endpoint.get("selected")]
            provider_metadata.update(selected or [metadata.get("summary", "unknown")])
            regions.update([metadata.get("region", "unknown")])
            attempts.update([str(metadata.get("attempt", "unknown"))])
        else:
            fingerprints.update([record["response_metadata"].get("system_fingerprint", "unknown")])

    result = {
        "note": "completion_tokens includes reasoning_tokens; corrected TPS does not add them again",
        "arms": {arm: arm_summary(arm_records) for arm, arm_records in groups.items()},
        "breakdown": breakdown,
        "paired": [
            paired(records, ("openrouter", "high"), ("openrouter", "low")),
            paired(records, ("direct", "high"), ("direct", "low")),
            paired(records, ("direct", "low"), ("openrouter", "low")),
            paired(records, ("direct", "high"), ("openrouter", "high")),
        ],
        "metadata": {
            "openrouter_selected_providers": dict(provider_metadata),
            "openrouter_regions": dict(regions),
            "openrouter_attempts": dict(attempts),
            "direct_system_fingerprints": dict(fingerprints),
        },
    }
    destination = root / "analysis_corrected.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    main(parser.parse_args())
