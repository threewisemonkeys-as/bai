# Viz Module

This directory contains the stepwise EB visualization stack.

## Structure

- `visualize_stepwise_eb_learn.py`
  Local dynamic server. It serves the frontend and reads a selected log directory directly from disk.
- `export_stepwise_eb_static.py`
  Static export entrypoint. It converts one local run into a curated, publishable snapshot under `data/stepwise_eb_runs/`.
- `stepwise_eb_viewer_data.py`
  Shared data-loading and export logic used by both the local server and the static exporter.
- `stepwise_eb_learn/`
  Shared frontend assets for both dynamic and static modes.

## Frontend Files

- `stepwise_eb_learn/index.html`
  Page shell, CSS, and frontend mode configuration.
- `stepwise_eb_learn/app.js`
  UI rendering, tab behavior, data fetch logic, run selection, and shared display code.

If you want to change how a tab displays data, this is usually a one-file change in `stepwise_eb_learn/app.js`.

## Data Flow

Dynamic mode:

1. Run `python viz/visualize_stepwise_eb_learn.py <log_dir>`.
2. The server reads logs from disk through `stepwise_eb_viewer_data.py`.
3. The frontend fetches data from the local `/api/...` endpoints.

Static mode:

1. Run `python viz/export_stepwise_eb_static.py <log_dir> --run-id <id>`.
2. The exporter writes:
   - `data/stepwise_eb_runs/index.json`
   - `data/stepwise_eb_runs/<run-id>/report.json`
   - `data/stepwise_eb_runs/<run-id>/step_details/*.json`
   - `data/stepwise_eb_runs/<run-id>/trajectories/*.json`
   - `data/stepwise_eb_runs/<run-id>/combined_trajectory.json`
   - `data/stepwise_eb_runs/<run-id>/qa_timeline.json`
   - `data/stepwise_eb_runs/<run-id>/experiment_timeline.json`
3. The committed static page loads one of those curated runs from `data/`.

## Commands

Local dynamic viewer:

```bash
python viz/visualize_stepwise_eb_learn.py logs/dev/apr8/2026-04-08_16-48-43_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn
```

Static export:

```bash
python viz/export_stepwise_eb_static.py \
  logs/dev/apr8/2026-04-08_16-48-43_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn \
  --run-id apr8-gemini-flash \
  --title "Apr 8 Gemini 2.5 Flash"
```

## Editing Guide

- Display-only change: edit `viz/stepwise_eb_learn/app.js`
- Display + styling change: edit `viz/stepwise_eb_learn/app.js` and `viz/stepwise_eb_learn/index.html`
- New data field needed by the frontend: also edit `viz/stepwise_eb_viewer_data.py`

## Notes

- Dynamic mode is still intended for local/private use because it accepts a filesystem path and reads directly from disk.
- Static mode is the publishable path for GitHub Pages.
