#!/usr/bin/env python3
"""Export curated stepwise EB runs for the static GitHub Pages viewer."""

import argparse

from stepwise_eb_viewer_data import export_static_report


def main():
    parser = argparse.ArgumentParser(description="Export a stepwise EB run to static JSON files")
    parser.add_argument("log_dir", help="Path to a log directory or parent folder containing one")
    parser.add_argument(
        "--output-root",
        default="data/stepwise_eb_runs",
        help="Directory that holds curated exported runs (default: data/stepwise_eb_runs)",
    )
    parser.add_argument("--run-id", default=None, help="Stable run id used in the published URL/data path")
    parser.add_argument("--title", default=None, help="Display title shown in the static viewer")
    parser.add_argument("--description", default=None, help="Optional short description shown in the run picker")
    args = parser.parse_args()

    result = export_static_report(
        log_dir=args.log_dir,
        export_root=args.output_root,
        run_id=args.run_id,
        title=args.title,
        description=args.description,
    )

    record = result["run_record"]
    print(f"Exported run '{record['id']}' to {result['run_dir']}")
    print(f"Resolved log dir: {result['resolved_log_dir']}")
    print(f"Updated run index: {result['index_path']}")
    print(
        f"Viewer summary: {record['episodes']} episodes, "
        f"{record['steps']} steps, total cost ${record['total_cost']:.4f}"
    )


if __name__ == "__main__":
    main()
