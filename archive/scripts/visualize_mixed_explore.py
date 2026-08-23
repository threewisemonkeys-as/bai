#!/usr/bin/env python3
"""Visualize mixed_explore.py logs with a local web server.

Usage:
    python visualize_mixed_explore.py <log_dir> [--port PORT] [--open-browser]

Serves an interactive viewer for mixed_explore.py log directories.
Trajectories and heavy data are loaded on-demand to keep the UI fast.
"""

import argparse
import csv
import io
import json
import os
import sys
import threading
import traceback
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


def read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except (FileNotFoundError, IsADirectoryError):
        return ""


def read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def find_rollout_files(rollout_dir):
    """Walk subdirectories to find .csv and .json files."""
    json_file = None
    csv_file = None
    for root, dirs, files in os.walk(rollout_dir):
        for f in files:
            if f.endswith(".json"):
                json_file = os.path.join(root, f)
            if f.endswith(".csv"):
                csv_file = os.path.join(root, f)
    return json_file, csv_file


def _sort_key(name):
    for prefix in ("baseline_", "experiment_"):
        if name.startswith(prefix):
            try:
                return (0 if prefix == "baseline_" else 1, int(name.replace(prefix, "")))
            except ValueError:
                break
    return (2, 999)


def load_log_dir_light(log_dir):
    """Load log directory without trajectory data or heavy artifacts."""
    summary = read_json(os.path.join(log_dir, "summary.json"))
    config = read_file(os.path.join(log_dir, "config.yaml"))

    # Build step cost lookup from summary.json
    step_costs = {}
    cumulative_costs = {}
    if summary and "steps" in summary:
        for s in summary["steps"]:
            step_costs[s.get("step")] = s.get("step_cost", 0)
            cumulative_costs[s.get("step")] = s.get("cumulative_cost", 0)

    step_dirs = []
    for name in os.listdir(log_dir):
        if name.startswith("step_") and os.path.isdir(os.path.join(log_dir, name)):
            try:
                idx = int(name.replace("step_", ""))
                step_dirs.append((idx, name))
            except ValueError:
                pass
    step_dirs.sort()

    steps = []
    for idx, name in step_dirs:
        step_path = os.path.join(log_dir, name)

        # Gather rollout names
        rollouts_dir = os.path.join(step_path, "rollouts")
        rollout_names = []
        if os.path.isdir(rollouts_dir):
            for rname in sorted(os.listdir(rollouts_dir), key=_sort_key):
                rpath = os.path.join(rollouts_dir, rname)
                if os.path.isdir(rpath):
                    # Check for experiment.txt
                    exp_text = read_file(os.path.join(rpath, "experiment.txt"))
                    json_file, _ = find_rollout_files(rpath)
                    meta = read_json(json_file) if json_file else None
                    rollout_names.append({
                        "name": rname,
                        "experiment_text": exp_text,
                        "meta": meta,
                    })

        rollout_stats = read_json(os.path.join(step_path, "rollout_stats.json"))

        step_data = {
            "index": idx,
            "input_beliefs": read_file(os.path.join(step_path, "input_beliefs.txt")),
            "input_perception": read_file(os.path.join(step_path, "input_perception.py")),
            "beliefs": read_file(os.path.join(step_path, "beliefs.txt")),
            "perception": read_file(os.path.join(step_path, "perception.py")),
            "rollout_stats": rollout_stats,
            "rollouts": rollout_names,
            "step_cost": step_costs.get(idx, 0),
            "cumulative_cost": cumulative_costs.get(idx, 0),
            "has_experiments": os.path.exists(os.path.join(step_path, "experiments.json")),
            "has_input_experiments": os.path.exists(os.path.join(step_path, "input_experiments.json")),
            "has_scoring": os.path.exists(os.path.join(step_path, "scoring_history.json")),
            "has_knowledge": os.path.exists(os.path.join(step_path, "qa_pairs.json"))
                or os.path.exists(os.path.join(step_path, "critical_moments.json")),
            "has_improve_log": os.path.exists(os.path.join(step_path, "improve.log")),
        }
        steps.append(step_data)

    total_cost = 0
    if steps:
        total_cost = steps[-1].get("cumulative_cost", 0)

    return {
        "config": config,
        "summary": summary,
        "steps": steps,
        "total_cost": total_cost,
        "log_dir_name": os.path.basename(log_dir),
    }


def load_trajectory(log_dir, step_idx, rollout_name):
    """Load a single trajectory CSV on demand."""
    rollout_dir = os.path.join(log_dir, f"step_{step_idx}", "rollouts", rollout_name)
    if not os.path.isdir(rollout_dir):
        return []

    _, csv_file = find_rollout_files(rollout_dir)
    if not csv_file:
        return []

    text = read_file(csv_file)
    if not text:
        return []

    steps = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        steps.append({
            "step": row.get("Step", ""),
            "action": row.get("Action", ""),
            "reasoning": row.get("Reasoning", ""),
            "observation": row.get("Observation", ""),
            "reward": row.get("Reward", ""),
            "done": row.get("Done", ""),
        })
    return steps


def load_step_knowledge(log_dir, step_idx):
    """Load knowledge artifacts for a step on demand."""
    step_path = os.path.join(log_dir, f"step_{step_idx}")
    return {
        "qa_pairs": read_json(os.path.join(step_path, "qa_pairs.json")) or [],
        "critical_moments": read_json(os.path.join(step_path, "critical_moments.json")) or [],
        "new_qa_pairs": read_json(os.path.join(step_path, "new_qa_pairs.json")) or [],
        "new_critical_moments": read_json(os.path.join(step_path, "new_critical_moments.json")) or [],
        "per_trajectory_extractions": read_json(os.path.join(step_path, "per_trajectory_extractions.json")) or [],
    }


def load_scoring(log_dir, step_idx):
    """Load scoring history for a step on demand."""
    step_path = os.path.join(log_dir, f"step_{step_idx}")
    return read_json(os.path.join(step_path, "scoring_history.json")) or []


def load_improve_log(log_dir, step_idx):
    """Load improve.log text for a step on demand."""
    step_path = os.path.join(log_dir, f"step_{step_idx}")
    return read_file(os.path.join(step_path, "improve.log"))


def load_experiments(log_dir, step_idx):
    """Load input and output experiments for a step."""
    step_path = os.path.join(log_dir, f"step_{step_idx}")
    return {
        "input_experiments": read_json(os.path.join(step_path, "input_experiments.json")) or [],
        "experiments": read_json(os.path.join(step_path, "experiments.json")) or [],
    }


def _extract_perception_example(step_path):
    """Extract a single (raw_observation, perception_output) example from mid-trajectory.

    Looks through rollout CSVs for the step and returns the first example found
    from the middle of a trajectory, or None if no examples are available.
    """
    PERC_START = "========== Start of features from Perception Module =========="
    PERC_END = "========== End of features from Perception Module =========="
    OBS_START = "========== Start of Direct Observation =========="
    OBS_END = "========== End of Direct Observation =========="

    rollouts_dir = os.path.join(step_path, "rollouts")
    if not os.path.isdir(rollouts_dir):
        return None

    csv_files = []
    for root, dirs, files in os.walk(rollouts_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    csv_files.sort()

    for csv_path in csv_files:
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f, escapechar="\u02d8", quoting=csv.QUOTE_MINIMAL)
                rows = list(reader)
        except Exception:
            continue

        if len(rows) < 2:
            continue

        # Pick from middle of trajectory
        mid_idx = len(rows) // 2
        obs_text = rows[mid_idx].get("Observation", "")
        if PERC_START not in obs_text:
            continue

        # Extract between markers
        perc_start = obs_text.find(PERC_START)
        perc_end = obs_text.find(PERC_END)
        obs_start = obs_text.find(OBS_START)
        obs_end = obs_text.find(OBS_END)

        if perc_start >= 0 and perc_end > perc_start and obs_start >= 0 and obs_end > obs_start:
            perc_out = obs_text[perc_start + len(PERC_START):perc_end].strip()
            raw_obs = obs_text[obs_start + len(OBS_START):obs_end].strip()
            step_num = rows[mid_idx].get("Step", str(mid_idx))
            return {
                "step_num": step_num,
                "raw_observation": raw_obs,
                "perception_output": perc_out,
            }

    return None


def load_perception_history(log_dir, step_idx):
    """Load perception improvement history for a step."""
    step_path = os.path.join(log_dir, f"step_{step_idx}")
    history = read_json(os.path.join(step_path, "perception_history.json"))
    input_perception = read_file(os.path.join(step_path, "input_perception.py"))
    output_perception = read_file(os.path.join(step_path, "perception.py"))
    example = _extract_perception_example(step_path)
    return {
        "input_perception": input_perception,
        "output_perception": output_perception,
        "history": history or [],
        "example": example,
    }


# ============= HTTP Server =============

LOG_DIR = None
DATA = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default logging

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path == "/":
                self._serve_html()
            elif path == "/api/data":
                self._json_response(DATA)
            elif path == "/api/trajectory":
                step = int(params.get("step", [0])[0])
                rollout = params.get("rollout", [""])[0]
                traj = load_trajectory(LOG_DIR, step, rollout)
                self._json_response(traj)
            elif path == "/api/improve_log":
                step = int(params.get("step", [0])[0])
                text = load_improve_log(LOG_DIR, step)
                self._json_response({"text": text})
            elif path == "/api/step_knowledge":
                step = int(params.get("step", [0])[0])
                self._json_response(load_step_knowledge(LOG_DIR, step))
            elif path == "/api/scoring":
                step = int(params.get("step", [0])[0])
                self._json_response(load_scoring(LOG_DIR, step))
            elif path == "/api/experiments":
                step = int(params.get("step", [0])[0])
                self._json_response(load_experiments(LOG_DIR, step))
            elif path == "/api/perception_history":
                step = int(params.get("step", [0])[0])
                self._json_response(load_perception_history(LOG_DIR, step))
            else:
                self.send_error(404)
        except Exception as e:
            err = f"[visualize_mixed_explore] request failed for {self.path}: {e}\n{traceback.format_exc()}"
            print(err, file=sys.stderr, flush=True)
            try:
                self._json_response({"error": str(e), "path": self.path})
            except Exception:
                try:
                    self.send_error(500, explain=str(e))
                except Exception:
                    pass

    def _json_response(self, obj):
        data = json.dumps(obj, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_html(self):
        html = get_html().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)


def get_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mixed Explore Log Viewer</title>
<style>
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --surface2: #21262d;
  --border: #30363d;
  --text: #e6edf3;
  --text-muted: #8b949e;
  --accent: #58a6ff;
  --accent2: #3fb950;
  --accent3: #d29922;
  --danger: #f85149;
  --font-mono: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.5;
}
.app { display: flex; flex-direction: column; height: 100vh; }

/* Top bar */
.topbar {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 12px 24px; display: flex; align-items: center; gap: 24px;
  flex-shrink: 0;
}
.topbar-title { font-size: 16px; font-weight: 700; }
.topbar-info { font-size: 13px; color: var(--text-muted); }
.topbar-cost { font-size: 13px; color: var(--accent3); font-weight: 600; }

.content-area { display: flex; flex: 1; overflow: hidden; }

/* Sidebar */
.sidebar {
  width: 220px; min-width: 220px; background: var(--surface);
  border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow-y: auto;
}
.sidebar h2 {
  padding: 16px; font-size: 14px; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border);
}
.sidebar-item {
  padding: 8px 16px; cursor: pointer; font-size: 13px;
  border-left: 3px solid transparent; transition: all 0.15s;
  display: flex; align-items: center; justify-content: space-between;
}
.sidebar-item:hover { background: var(--surface2); }
.sidebar-item.active {
  background: var(--surface2); border-left-color: var(--accent); color: var(--accent);
}
.sidebar-cost {
  font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);
}
.sidebar-item.active .sidebar-cost { color: var(--accent); }

/* Main */
.main { flex: 1; overflow-y: auto; padding: 24px; }
.main h1 { font-size: 22px; margin-bottom: 20px; }

/* Tabs */
.tabs { display: flex; gap: 0; margin-bottom: 20px; flex-wrap: wrap; }
.tab {
  padding: 8px 16px; cursor: pointer; font-size: 13px; color: var(--text-muted);
  background: var(--surface); border: 1px solid var(--border); transition: all 0.15s;
}
.tab:first-child { border-radius: 6px 0 0 6px; }
.tab:last-child { border-radius: 0 6px 6px 0; }
.tab.active { background: var(--accent); color: var(--bg); border-color: var(--accent); }

/* Cards */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; margin-bottom: 16px;
}
.card-header {
  padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 14px;
  font-weight: 600; display: flex; align-items: center; justify-content: space-between;
  cursor: pointer; user-select: none;
}
.card-header:hover { background: var(--surface2); }
.card-header .toggle { color: var(--text-muted); font-size: 12px; }
.card-body { padding: 16px; }
.card-body.collapsed { display: none; }

pre {
  background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
  padding: 12px; font-family: var(--font-mono); font-size: 12px; line-height: 1.6;
  overflow-x: auto; white-space: pre-wrap; word-wrap: break-word;
  max-height: 500px; overflow-y: auto;
}

/* Stats */
.stats-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin-bottom: 20px;
}
.stat-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 16px;
}
.stat-label {
  font-size: 11px; color: var(--text-muted); text-transform: uppercase;
  letter-spacing: 0.5px; margin-bottom: 4px;
}
.stat-value { font-size: 24px; font-weight: 700; }
.stat-value.green { color: var(--accent2); }
.stat-value.blue { color: var(--accent); }
.stat-value.yellow { color: var(--accent3); }
.stat-value.red { color: var(--danger); }

/* Table */
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.data-table th {
  text-align: left; padding: 8px 12px; background: var(--surface2);
  border-bottom: 1px solid var(--border); font-weight: 600;
  color: var(--text-muted); font-size: 11px; text-transform: uppercase;
}
.data-table td {
  padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top;
}
.data-table tr:hover { background: var(--surface2); }
.data-table tr.clickable { cursor: pointer; }
.data-table tr.new-item { background: rgba(63,185,80,0.08); }
.data-table tr.new-item td:first-child { border-left: 3px solid var(--accent2); }

/* Trajectory */
.traj-step {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; margin-bottom: 8px;
}
.traj-step-header {
  padding: 10px 16px; display: flex; align-items: center; gap: 12px;
  cursor: pointer; user-select: none; font-size: 13px;
}
.traj-step-header:hover { background: var(--surface2); }
.traj-step-num {
  background: var(--accent); color: var(--bg); width: 28px; height: 28px;
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; flex-shrink: 0;
}
.traj-step-num.death { background: var(--danger); color: white; }
.traj-step-num.success { background: var(--accent2); color: white; }
.traj-step-action {
  font-family: var(--font-mono); background: var(--surface2);
  padding: 2px 8px; border-radius: 4px; font-size: 12px;
}
.traj-step-reward {
  margin-left: auto; font-family: var(--font-mono); font-size: 12px; color: var(--text-muted);
}
.traj-step-body { display: none; padding: 0 16px 16px; }
.traj-step-body.open { display: block; }
.traj-section { margin-top: 12px; }
.traj-section-label {
  font-size: 11px; text-transform: uppercase; color: var(--text-muted);
  margin-bottom: 4px; letter-spacing: 0.5px;
}

/* Charts */
.chart-container { margin-bottom: 24px; }
.chart-title { font-size: 14px; font-weight: 600; margin-bottom: 12px; }
.bar-chart {
  display: flex; align-items: flex-end; gap: 4px; height: 200px;
  padding: 0 8px; border-bottom: 1px solid var(--border);
}
.bar-group {
  flex: 1; display: flex; flex-direction: column; align-items: center;
  height: 100%; justify-content: flex-end;
}
.bar {
  width: 100%; max-width: 30px; border-radius: 3px 3px 0 0;
  transition: height 0.3s; min-height: 2px;
}
.bar.blue { background: var(--accent); }
.bar.green { background: var(--accent2); }
.bar.yellow { background: var(--accent3); }
.bar.red { background: var(--danger); }
.bar-label { font-size: 10px; color: var(--text-muted); margin-top: 6px; }
.bar-value { font-size: 10px; color: var(--text-muted); margin-bottom: 2px; white-space: nowrap; }
.chart-legend { display: flex; gap: 16px; margin-top: 8px; font-size: 12px; color: var(--text-muted); }
.legend-item { display: flex; align-items: center; gap: 4px; }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; }

/* Scoring */
.score-correct { color: var(--accent2); }
.score-incorrect { color: var(--danger); }
.score-item {
  padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 13px;
}
.score-item:last-child { border-bottom: none; }

/* Side-by-side diff */
.side-by-side {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}
.side-by-side > div > h3 {
  font-size: 13px; color: var(--text-muted); margin-bottom: 8px;
  text-transform: uppercase; letter-spacing: 0.5px;
}

/* Step nav */
.step-nav { display: flex; gap: 8px; margin-bottom: 16px; align-items: center; }
.step-nav-btn {
  background: var(--surface2); border: 1px solid var(--border); color: var(--text);
  padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px;
}
.step-nav-btn:hover { background: var(--border); }
.step-nav-btn:disabled { opacity: 0.3; cursor: not-allowed; }

.back-btn {
  background: var(--surface2); border: 1px solid var(--border); color: var(--text);
  padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px;
  margin-bottom: 16px; display: inline-flex; align-items: center; gap: 6px;
}
.back-btn:hover { background: var(--border); }

.loading {
  color: var(--text-muted); font-style: italic; padding: 20px; text-align: center;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--border); }

/* Per-traj extraction */
.extraction-section {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; margin-bottom: 8px;
}
.extraction-header {
  padding: 10px 16px; cursor: pointer; user-select: none; font-size: 13px;
  display: flex; align-items: center; gap: 10px;
}
.extraction-header:hover { background: var(--surface2); }
.extraction-body { display: none; padding: 12px 16px; border-top: 1px solid var(--border); }
.extraction-body.open { display: block; }
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="topbar-title" id="topbar-title">Mixed Explore Viewer</div>
    <div class="topbar-info" id="topbar-dir"></div>
    <div class="topbar-cost" id="topbar-cost"></div>
  </div>
  <div class="content-area">
    <nav class="sidebar" id="sidebar">
      <h2>Steps</h2>
      <div id="step-list"></div>
    </nav>
    <div class="main" id="main-content">
      <div class="loading">Loading data...</div>
    </div>
  </div>
</div>

<script>
let DATA = null;
let currentStep = null;
let currentTab = 'overview';
const trajCache = {};
const knowledgeCache = {};
const scoringCache = {};
const experimentsCache = {};
const improveLogCache = {};

async function init() {
  try {
    const resp = await fetch('/api/data');
    if (!resp.ok) throw new Error('/api/data returned ' + resp.status);
    DATA = await resp.json();
    document.getElementById('topbar-dir').textContent = DATA.log_dir_name;
    document.getElementById('topbar-cost').textContent = 'Total cost: $' + (DATA.total_cost || 0).toFixed(4);
    buildSidebar();
    if (DATA.steps.length > 0) {
      showStep(0);
    } else {
      document.getElementById('main-content').innerHTML = '<h1>No steps found</h1>';
    }
  } catch (e) {
    document.getElementById('main-content').innerHTML =
      '<h1>Viewer Error</h1><div class="card"><div class="card-body"><pre>' + esc(String(e && e.stack ? e.stack : e)) + '</pre></div></div>';
  }
}

function buildSidebar() {
  const stepList = document.getElementById('step-list');
  stepList.innerHTML = '';
  DATA.steps.forEach((step, i) => {
    const el = document.createElement('div');
    el.className = 'sidebar-item';
    el.dataset.step = i;
    el.innerHTML = 'Step ' + step.index + '<span class="sidebar-cost">$' + (step.step_cost || 0).toFixed(3) + '</span>';
    el.onclick = () => { currentTab = 'overview'; showStep(i); };
    stepList.appendChild(el);
  });
}

function showStep(idx) {
  currentStep = idx;
  updateSidebar();
  renderStep(idx);
}

function updateSidebar() {
  document.querySelectorAll('.sidebar-item').forEach(el => {
    el.classList.toggle('active', parseInt(el.dataset.step) === currentStep);
  });
}

function esc(s) {
  if (s == null) return '';
  const div = document.createElement('div');
  div.textContent = String(s);
  return div.innerHTML;
}

function collapsible(title, content, startOpen) {
  const cls = startOpen ? '' : 'collapsed';
  const arrow = startOpen ? '&#9660;' : '&#9654;';
  return '<div class="card">' +
    '<div class="card-header" onclick="toggleCard(this)">' + title + ' <span class="toggle">' + arrow + '</span></div>' +
    '<div class="card-body ' + cls + '">' + content + '</div>' +
    '</div>';
}

function toggleCard(header) {
  const body = header.nextElementSibling;
  const toggle = header.querySelector('.toggle');
  body.classList.toggle('collapsed');
  toggle.innerHTML = body.classList.contains('collapsed') ? '&#9654;' : '&#9660;';
}

function toggleTrajStep(el) { el.nextElementSibling.classList.toggle('open'); }

function toggleExtraction(el) { el.nextElementSibling.classList.toggle('open'); }

// ============ STEP RENDERING ============
function renderStep(idx) {
  const step = DATA.steps[idx];
  const mc = document.getElementById('main-content');
  const totalSteps = DATA.steps.length;

  let html = '<div class="step-nav">' +
    '<button class="step-nav-btn" onclick="currentTab=\\'overview\\';showStep(' + (idx-1) + ')" ' + (idx===0?'disabled':'') + '>&#8592; Prev</button>' +
    '<h1 style="margin:0;flex:1">Step ' + step.index + '</h1>' +
    '<span style="font-size:13px;color:var(--text-muted)">Cost: $' + (step.step_cost||0).toFixed(4) + ' | Cumulative: $' + (step.cumulative_cost||0).toFixed(4) + '</span>' +
    '<button class="step-nav-btn" onclick="currentTab=\\'overview\\';showStep(' + (idx+1) + ')" ' + (idx>=totalSteps-1?'disabled':'') + '>Next &#8594;</button>' +
    '</div>';

  html += '<div class="tabs">';
  var tabs = [
    ['overview', 'Overview'],
    ['rollouts', 'Rollouts'],
    ['knowledge', 'Knowledge'],
    ['scoring', 'Scoring'],
    ['perception', 'Perception'],
    ['experiments', 'Experiments'],
    ['logs', 'Logs']
  ];
  tabs.forEach(function(t) {
    html += '<div class="tab ' + (currentTab===t[0]?'active':'') + '" onclick="currentTab=\\'' + t[0] + '\\';renderStep(' + idx + ')">' + t[1] + '</div>';
  });
  html += '</div>';

  if (currentTab === 'overview') html += renderOverview(step);
  else if (currentTab === 'rollouts') html += renderRollouts(step, idx);
  else if (currentTab === 'knowledge') html += renderKnowledgePlaceholder(step);
  else if (currentTab === 'scoring') html += renderScoringPlaceholder(step);
  else if (currentTab === 'perception') html += renderPerceptionPlaceholder(step);
  else if (currentTab === 'experiments') html += renderExperimentsPlaceholder(step);
  else if (currentTab === 'logs') html += renderLogsPlaceholder(step);

  mc.innerHTML = html;

  // Trigger async loads for lazy tabs
  if (currentTab === 'knowledge') loadKnowledge(step.index);
  else if (currentTab === 'scoring') loadScoring(step.index);
  else if (currentTab === 'perception') loadPerceptionHistory(step.index);
  else if (currentTab === 'experiments') loadExperiments(step.index);
  else if (currentTab === 'logs') loadImproveLog(step.index);
}

// ============ OVERVIEW TAB ============
function renderOverview(step) {
  let html = '';

  // Rollout stats summary
  const rs = step.rollout_stats || {};
  let totalRollouts = 0, successCount = 0, failCount = 0, errorCount = 0;
  const entries = Object.entries(rs);
  entries.forEach(function(e) {
    var v = e[1];
    totalRollouts++;
    var summary = v.summary || {};
    Object.values(summary).forEach(function(env) {
      if (env.avg_prog >= 1) successCount++;
    });
  });
  failCount = totalRollouts - successCount;

  html += '<div class="stats-grid">' +
    '<div class="stat-card"><div class="stat-label">Rollouts</div><div class="stat-value blue">' + totalRollouts + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Successful</div><div class="stat-value green">' + successCount + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Failed</div><div class="stat-value red">' + failCount + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Step Cost</div><div class="stat-value yellow">$' + (step.step_cost||0).toFixed(4) + '</div></div>' +
    '</div>';

  // Side-by-side beliefs
  html += '<div class="side-by-side">' +
    '<div><h3>Input Beliefs</h3><pre>' + esc(step.input_beliefs || '(empty)') + '</pre></div>' +
    '<div><h3>Output Beliefs</h3><pre>' + esc(step.beliefs || '(empty)') + '</pre></div>' +
    '</div>';

  // Side-by-side perception
  html += '<div class="side-by-side" style="margin-top:16px">' +
    '<div><h3>Input Perception</h3><pre>' + esc(step.input_perception || '(empty)') + '</pre></div>' +
    '<div><h3>Output Perception</h3><pre>' + esc(step.perception || '(empty)') + '</pre></div>' +
    '</div>';

  return html;
}

// ============ ROLLOUTS TAB ============
function renderRollouts(step, stepIdx) {
  let html = '';

  if (!step.rollouts || step.rollouts.length === 0) {
    return '<div class="card"><div class="card-body">No rollouts found.</div></div>';
  }

  html += '<div class="card"><div class="card-header">Rollouts (' + step.rollouts.length + ')</div><div class="card-body">' +
    '<table class="data-table">' +
    '<tr><th>#</th><th>Name</th><th>Experiment</th><th>Steps</th><th>Prog</th><th>Cost</th><th>End</th></tr>';

  step.rollouts.forEach(function(r, ri) {
    var m = r.meta || {};
    var endReasons = {0:'timeout',1:'death',2:'success'};
    var endReason = endReasons[m.end_reason] || '?';
    var endColor = m.end_reason === 2 ? 'accent2' : m.end_reason === 1 ? 'danger' : 'accent3';
    var expSnip = r.experiment_text ? (r.experiment_text.length > 60 ? r.experiment_text.slice(0,60)+'...' : r.experiment_text) : '';

    html += '<tr class="clickable" onclick="showRolloutDetail(' + stepIdx + ',' + ri + ')">' +
      '<td>' + ri + '</td>' +
      '<td>' + esc(r.name) + '</td>' +
      '<td title="' + esc(r.experiment_text) + '">' + esc(expSnip) + '</td>' +
      '<td>' + (m.num_steps || '-') + '</td>' +
      '<td>' + ((m.progression||0)*100).toFixed(0) + '%</td>' +
      '<td>$' + (m.total_cost||0).toFixed(4) + '</td>' +
      '<td style="color:var(--' + endColor + ')">' + endReason + '</td>' +
      '</tr>';
  });

  html += '</table></div></div>';
  return html;
}

// ============ ROLLOUT DETAIL ============
async function showRolloutDetail(stepIdx, rolloutIdx) {
  var step = DATA.steps[stepIdx];
  var rollout = step.rollouts[rolloutIdx];
  var mc = document.getElementById('main-content');

  var html = '<button class="back-btn" onclick="renderStep(' + stepIdx + ')">&#8592; Back to Step ' + step.index + '</button>';
  html += '<h1>' + esc(rollout.name) + '</h1>';

  if (rollout.experiment_text) {
    html += '<div class="card"><div class="card-header">Experiment</div><div class="card-body"><pre>' + esc(rollout.experiment_text) + '</pre></div></div>';
  }

  if (rollout.meta) {
    var m = rollout.meta;
    var endReasons = {0:'timeout',1:'death',2:'success'};
    var endColor = m.end_reason === 2 ? 'green' : m.end_reason === 1 ? 'red' : 'yellow';
    html += '<div class="stats-grid">' +
      '<div class="stat-card"><div class="stat-label">Task</div><div class="stat-value" style="font-size:16px">' + esc(m.task) + '</div></div>' +
      '<div class="stat-card"><div class="stat-label">Steps</div><div class="stat-value blue">' + m.num_steps + '</div></div>' +
      '<div class="stat-card"><div class="stat-label">Progression</div><div class="stat-value green">' + ((m.progression||0)*100).toFixed(0) + '%</div></div>' +
      '<div class="stat-card"><div class="stat-label">End Reason</div><div class="stat-value ' + endColor + '">' + (endReasons[m.end_reason]||'?') + '</div></div>' +
      '<div class="stat-card"><div class="stat-label">Cost</div><div class="stat-value yellow">$' + (m.total_cost||0).toFixed(4) + '</div></div>' +
      '<div class="stat-card"><div class="stat-label">Tokens</div><div class="stat-value" style="font-size:16px">' + ((m.input_tokens||0).toLocaleString()) + ' in / ' + ((m.output_tokens||0).toLocaleString()) + ' out</div></div>' +
      '</div>';

    if (m.action_frequency) {
      var af = Object.entries(m.action_frequency).sort(function(a,b){ return b[1]-a[1]; });
      html += '<div class="card"><div class="card-header" onclick="toggleCard(this)">Action Frequency <span class="toggle">&#9660;</span></div><div class="card-body">';
      af.forEach(function(e) {
        html += '<span style="display:inline-block;margin:2px 4px;padding:3px 10px;background:var(--surface2);border-radius:4px;font-size:13px;font-family:var(--font-mono)">' + esc(e[0]) + ': ' + e[1] + '</span>';
      });
      html += '</div></div>';
    }
  }

  html += '<div id="trajectory-container"><div class="loading">Loading trajectory...</div></div>';
  mc.innerHTML = html;

  // Fetch trajectory on demand
  var cacheKey = step.index + '_' + rollout.name;
  var traj = trajCache[cacheKey];
  if (!traj) {
    var resp = await fetch('/api/trajectory?step=' + step.index + '&rollout=' + encodeURIComponent(rollout.name));
    traj = await resp.json();
    trajCache[cacheKey] = traj;
  }

  var tc = document.getElementById('trajectory-container');
  if (!traj || traj.length === 0) {
    tc.innerHTML = '<div class="card"><div class="card-body">No trajectory data available.</div></div>';
    return;
  }

  var tHtml = '<div class="card"><div class="card-header">Trajectory (' + traj.length + ' steps)</div><div class="card-body">';
  traj.forEach(function(t, ti) {
    var doneClass = t.done === 'True' ? (parseFloat(t.reward) > 0 ? 'success' : 'death') : '';
    tHtml += '<div class="traj-step">' +
      '<div class="traj-step-header" onclick="toggleTrajStep(this)">' +
        '<div class="traj-step-num ' + doneClass + '">' + t.step + '</div>' +
        '<div class="traj-step-action">' + esc(t.action) + '</div>' +
        '<div class="traj-step-reward">r=' + t.reward + (t.done === 'True' ? ' (DONE)' : '') + '</div>' +
      '</div>' +
      '<div class="traj-step-body">' +
        '<div class="traj-section"><div class="traj-section-label">Observation</div><pre>' + esc(t.observation) + '</pre></div>' +
        '<div class="traj-section"><div class="traj-section-label">Reasoning</div><pre>' + esc(t.reasoning) + '</pre></div>' +
      '</div>' +
    '</div>';
  });
  tHtml += '</div></div>';
  tc.innerHTML = tHtml;
}

// ============ KNOWLEDGE TAB ============
function renderKnowledgePlaceholder(step) {
  return '<div id="knowledge-container"><div class="loading">Loading knowledge data...</div></div>';
}

async function loadKnowledge(stepIdx) {
  var container = document.getElementById('knowledge-container');
  if (!container) return;

  var data = knowledgeCache[stepIdx];
  if (!data) {
    try {
      var resp = await fetch('/api/step_knowledge?step=' + stepIdx);
      data = await resp.json();
      knowledgeCache[stepIdx] = data;
    } catch(e) {
      container.innerHTML = '<pre>' + esc(String(e)) + '</pre>';
      return;
    }
  }

  var html = '';

  // Build sets of new items for highlighting
  var newQaSet = new Set();
  (data.new_qa_pairs || []).forEach(function(q, i) {
    newQaSet.add(JSON.stringify({question: q.question, answer: q.answer}));
  });
  var newMomentSet = new Set();
  (data.new_critical_moments || []).forEach(function(m) {
    newMomentSet.add(JSON.stringify({state: m.state, goal: m.goal}));
  });

  // Q&A Pairs
  var qaPairs = data.qa_pairs || [];
  html += '<div class="card"><div class="card-header" onclick="toggleCard(this)">Q&A Pairs (' + qaPairs.length + ') <span class="toggle">&#9660;</span></div><div class="card-body">';
  if (qaPairs.length === 0) {
    html += '<div style="color:var(--text-muted)">No Q&A pairs.</div>';
  } else {
    html += '<table class="data-table"><tr><th>Question</th><th>Answer</th><th>Evidence</th><th>Source Step</th></tr>';
    qaPairs.forEach(function(q) {
      var isNew = newQaSet.has(JSON.stringify({question: q.question, answer: q.answer}));
      html += '<tr class="' + (isNew ? 'new-item' : '') + '">' +
        '<td>' + esc(q.question) + '</td>' +
        '<td><strong>' + esc(q.answer) + '</strong></td>' +
        '<td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(q.evidence) + '">' + esc(q.evidence) + '</td>' +
        '<td>' + esc(q.source_step) + '</td>' +
        '</tr>';
    });
    html += '</table>';
  }
  html += '</div></div>';

  // Critical Moments
  var moments = data.critical_moments || [];
  html += '<div class="card"><div class="card-header" onclick="toggleCard(this)">Critical Moments (' + moments.length + ') <span class="toggle">&#9660;</span></div><div class="card-body">';
  if (moments.length === 0) {
    html += '<div style="color:var(--text-muted)">No critical moments.</div>';
  } else {
    html += '<table class="data-table"><tr><th>State</th><th>Goal</th><th>Good Actions</th><th>Bad Actions</th><th>Evidence</th><th>Source Step</th></tr>';
    moments.forEach(function(m) {
      var isNew = newMomentSet.has(JSON.stringify({state: m.state, goal: m.goal}));
      var goodActs = (m.good_actions || []).join(', ') || '-';
      var badActs = (m.bad_actions || []).join(', ') || '-';
      html += '<tr class="' + (isNew ? 'new-item' : '') + '">' +
        '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(m.state) + '">' + esc(m.state) + '</td>' +
        '<td>' + esc(m.goal) + '</td>' +
        '<td style="color:#4caf50">' + esc(goodActs) + '</td>' +
        '<td style="color:#f44336">' + esc(badActs) + '</td>' +
        '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(m.evidence) + '">' + esc(m.evidence) + '</td>' +
        '<td>' + esc(m.source_step) + '</td>' +
        '</tr>';
    });
    html += '</table>';
  }
  html += '</div></div>';

  // Per-trajectory extractions
  var perTraj = data.per_trajectory_extractions || [];
  if (perTraj.length > 0) {
    html += '<div class="card"><div class="card-header" onclick="toggleCard(this)">Per-Trajectory Extractions (' + perTraj.length + ') <span class="toggle">&#9654;</span></div><div class="card-body collapsed">';
    perTraj.forEach(function(ext, i) {
      var label = ext.trajectory || ext.rollout || ('Trajectory ' + i);
      html += '<div class="extraction-section">' +
        '<div class="extraction-header" onclick="toggleExtraction(this)">' +
        '<span style="color:var(--accent);font-weight:600">' + esc(String(label)) + '</span>' +
        '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
        '</div>' +
        '<div class="extraction-body"><pre>' + esc(JSON.stringify(ext, null, 2)) + '</pre></div>' +
        '</div>';
    });
    html += '</div></div>';
  }

  // Legend
  html += '<div style="font-size:12px;color:var(--text-muted);margin-top:8px">' +
    '<span style="display:inline-block;width:12px;height:12px;background:rgba(63,185,80,0.15);border-left:3px solid var(--accent2);margin-right:6px;vertical-align:middle"></span>' +
    'Highlighted rows are newly added in this step' +
    '</div>';

  container.innerHTML = html;
}

// ============ SCORING TAB ============
function renderScoringPlaceholder(step) {
  return '<div id="scoring-container"><div class="loading">Loading scoring data...</div></div>';
}

async function loadScoring(stepIdx) {
  var container = document.getElementById('scoring-container');
  if (!container) return;

  var data = scoringCache[stepIdx];
  if (!data) {
    try {
      var resp = await fetch('/api/scoring?step=' + stepIdx);
      data = await resp.json();
      scoringCache[stepIdx] = data;
    } catch(e) {
      container.innerHTML = '<pre>' + esc(String(e)) + '</pre>';
      return;
    }
  }

  if (!data || data.length === 0) {
    container.innerHTML = '<div class="card"><div class="card-body" style="color:var(--text-muted)">No scoring history found for this step.</div></div>';
    return;
  }

  var html = '';

  // Chart of scores across iterations
  var maxScore = 1;
  html += '<div class="chart-container"><div class="chart-title">Scoring History</div><div class="bar-chart">';
  data.forEach(function(iter) {
    var qaH = (iter.qa_score || 0) * 100;
    var momH = (iter.moment_score || 0) * 100;
    html += '<div class="bar-group">' +
      '<div class="bar-value">Q:' + ((iter.qa_score||0)*100).toFixed(0) + '% M:' + ((iter.moment_score||0)*100).toFixed(0) + '%</div>' +
      '<div style="display:flex;gap:2px;align-items:flex-end;height:100%;width:100%;justify-content:center;">' +
        '<div class="bar blue" style="height:' + qaH + '%;max-width:14px;" title="QA Score: ' + ((iter.qa_score||0)*100).toFixed(1) + '%"></div>' +
        '<div class="bar green" style="height:' + momH + '%;max-width:14px;" title="Moment Score: ' + ((iter.moment_score||0)*100).toFixed(1) + '%"></div>' +
      '</div>' +
      '<div class="bar-label">Iter ' + (iter.iteration != null ? iter.iteration : '?') + '</div>' +
    '</div>';
  });
  html += '</div>' +
    '<div class="chart-legend">' +
    '<div class="legend-item"><div class="legend-dot" style="background:var(--accent)"></div>QA Score</div>' +
    '<div class="legend-item"><div class="legend-dot" style="background:var(--accent2)"></div>Moment Score</div>' +
    '</div></div>';

  // Summary stats
  html += '<div class="stats-grid">';
  data.forEach(function(iter) {
    html += '<div class="stat-card">' +
      '<div class="stat-label">Iteration ' + (iter.iteration != null ? iter.iteration : '?') + '</div>' +
      '<div style="font-size:13px;margin-top:4px">' +
        'QA: <strong class="' + ((iter.qa_score||0) >= 0.5 ? 'score-correct' : 'score-incorrect') + '">' + ((iter.qa_score||0)*100).toFixed(0) + '%</strong> (' + (iter.num_qa||0) + ' items)<br>' +
        'Moments: <strong class="' + ((iter.moment_score||0) >= 0.5 ? 'score-correct' : 'score-incorrect') + '">' + ((iter.moment_score||0)*100).toFixed(0) + '%</strong> (' + (iter.num_moments||0) + ' items)' +
      '</div></div>';
  });
  html += '</div>';

  // Per-iteration details
  data.forEach(function(iter) {
    var details = iter.details || [];
    if (details.length === 0) return;

    var detailHtml = '<table class="data-table"><tr><th>Type</th><th>ID</th><th>Question/State</th><th>Predicted</th><th>Expected</th><th>Result</th></tr>';
    details.forEach(function(d) {
      var resultText, resultClass;
      if (d.type === 'moment') {
        var score = d.score != null ? d.score : (d.correct ? 1 : 0);
        if (score > 0) { resultText = '+1 (good)'; resultClass = 'score-correct'; }
        else if (score < 0) { resultText = '-1 (bad)'; resultClass = 'score-incorrect'; }
        else { resultText = '0 (neutral)'; resultClass = ''; }
        var goodActs = (d.good_actions || []).join(', ') || '-';
        var badActs = (d.bad_actions || []).join(', ') || '-';
        var expected = 'Good: [' + goodActs + '] Bad: [' + badActs + ']';
      } else {
        resultText = d.correct ? 'Correct' : 'Wrong';
        resultClass = d.correct ? 'score-correct' : 'score-incorrect';
        var expected = d.actual != null ? String(d.actual) : '';
      }
      var questionOrState = d.question || d.state || '';
      detailHtml += '<tr>' +
        '<td>' + esc(d.type || '') + '</td>' +
        '<td>' + esc(d.id != null ? String(d.id) : '') + '</td>' +
        '<td style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(questionOrState) + '">' + esc(questionOrState) + '</td>' +
        '<td>' + esc(d.predicted != null ? String(d.predicted) : '') + '</td>' +
        '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(expected) + '">' + esc(expected) + '</td>' +
        '<td class="' + resultClass + '"><strong>' + resultText + '</strong></td>' +
        '</tr>';
    });
    detailHtml += '</table>';

    var correct = details.filter(function(d){ return d.correct; }).length;
    var total = details.length;
    html += collapsible('Iteration ' + (iter.iteration != null ? iter.iteration : '?') + ' Details (' + correct + '/' + total + ' correct)', detailHtml, false);
  });

  container.innerHTML = html;
}

// ============ EXPERIMENTS TAB ============
function renderExperimentsPlaceholder(step) {
  return '<div id="experiments-container"><div class="loading">Loading experiments data...</div></div>';
}

async function loadExperiments(stepIdx) {
  var container = document.getElementById('experiments-container');
  if (!container) return;

  var data = experimentsCache[stepIdx];
  if (!data) {
    try {
      var resp = await fetch('/api/experiments?step=' + stepIdx);
      data = await resp.json();
      experimentsCache[stepIdx] = data;
    } catch(e) {
      container.innerHTML = '<pre>' + esc(String(e)) + '</pre>';
      return;
    }
  }

  var html = '';

  // Input experiments
  var inputExps = data.input_experiments || [];
  html += '<div class="card"><div class="card-header" onclick="toggleCard(this)">Input Experiments (' + inputExps.length + ') <span class="toggle">&#9660;</span></div><div class="card-body">';
  if (inputExps.length === 0) {
    html += '<div style="color:var(--text-muted)">No input experiments.</div>';
  } else {
    inputExps.forEach(function(exp, i) {
      var text = typeof exp === 'string' ? exp : JSON.stringify(exp, null, 2);
      html += '<div style="background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:12px;margin-bottom:8px;font-size:13px">' +
        '<span style="color:var(--accent);font-weight:600">#' + (i+1) + '</span> ' +
        '<pre style="margin-top:8px;border:none;padding:0;background:transparent">' + esc(text) + '</pre></div>';
    });
  }
  html += '</div></div>';

  // Output experiments (generated for next step)
  var outputExps = data.experiments || [];
  html += '<div class="card"><div class="card-header" onclick="toggleCard(this)">Output Experiments for Next Step (' + outputExps.length + ') <span class="toggle">&#9660;</span></div><div class="card-body">';
  if (outputExps.length === 0) {
    html += '<div style="color:var(--text-muted)">No output experiments generated.</div>';
  } else {
    outputExps.forEach(function(exp, i) {
      var text = typeof exp === 'string' ? exp : JSON.stringify(exp, null, 2);
      html += '<div style="background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:12px;margin-bottom:8px;font-size:13px">' +
        '<span style="color:var(--accent2);font-weight:600">#' + (i+1) + '</span> ' +
        '<pre style="margin-top:8px;border:none;padding:0;background:transparent">' + esc(text) + '</pre></div>';
    });
  }
  html += '</div></div>';

  container.innerHTML = html;
}

// ============ LOGS TAB ============
function renderLogsPlaceholder(step) {
  return '<div id="logs-container"><div class="loading">Loading improve.log...</div></div>';
}

async function loadImproveLog(stepIdx) {
  var container = document.getElementById('logs-container');
  if (!container) return;

  var data = improveLogCache[stepIdx];
  if (!data) {
    try {
      var resp = await fetch('/api/improve_log?step=' + stepIdx);
      data = await resp.json();
      improveLogCache[stepIdx] = data;
    } catch(e) {
      container.innerHTML = '<pre>' + esc(String(e)) + '</pre>';
      return;
    }
  }

  var text = data.text || '';
  if (!text) {
    container.innerHTML = '<div class="card"><div class="card-body" style="color:var(--text-muted)">No improve.log found for this step.</div></div>';
    return;
  }

  container.innerHTML = '<div class="card"><div class="card-header">improve.log</div><div class="card-body">' +
    '<pre style="max-height:800px">' + esc(text) + '</pre></div></div>';
}

// ============ PERCEPTION TAB ============
var perceptionCache = {};

function renderPerceptionPlaceholder(step) {
  return '<div id="perception-container"><div class="loading">Loading perception history...</div></div>';
}

function _percToggle(id) {
  var el = document.getElementById(id);
  if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

function _percSectionCard(title, bodyHtml, opts) {
  opts = opts || {};
  var collapsed = opts.collapsed !== false;
  var accent = opts.accent || 'var(--accent)';
  var bodyId = opts.id || ('perc-sec-' + Math.random().toString(36).substring(2, 8));
  var badge = opts.badge ? ' <span style="background:' + accent + ';color:var(--bg);font-size:11px;padding:2px 8px;border-radius:10px;margin-left:8px;font-weight:600">' + opts.badge + '</span>' : '';
  var html = '<div class="card">';
  html += '<div class="card-header" onclick="_percToggle(\\''+bodyId+'\\');" style="cursor:pointer;border-left:3px solid '+accent+'">';
  html += '<span>' + title + badge + '</span>';
  html += '<span class="toggle" style="color:var(--text-muted)">&#9660;</span>';
  html += '</div>';
  html += '<div id="'+bodyId+'" class="card-body" style="display:'+(collapsed?'none':'block')+'">';
  html += bodyHtml;
  html += '</div></div>';
  return html;
}

// Render a single-candidate iteration pipeline
function _renderPercIterationPipeline(iterEntries, idPrefix) {
  var html = '';
  iterEntries.forEach(function(h, idx) {
    var uid = idPrefix + '-' + idx;
    var scoreText = h.perc_moment_score != null ? (h.perc_moment_score * 100).toFixed(1) + '%' : '-';
    var scoreColor = h.perc_moment_score != null
      ? (h.perc_moment_score >= 0.75 ? 'var(--accent2)' : h.perc_moment_score >= 0.5 ? 'var(--accent3)' : 'var(--danger)')
      : 'var(--text-muted)';

    // ── Iteration header ──
    html += '<div style="display:flex;align-items:center;gap:12px;margin:20px 0 12px">';
    html += '<div style="background:var(--accent);color:var(--bg);width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;flex-shrink:0">' + h.iteration + '</div>';
    html += '<div style="font-size:15px;font-weight:600">Iteration ' + h.iteration + '</div>';
    html += '<div style="font-size:12px;color:var(--text-muted)">$' + (h.cost || 0).toFixed(6) + '</div>';
    html += '<div style="margin-left:auto;font-size:13px;font-weight:600;color:' + scoreColor + '">Moment Score: ' + scoreText + '</div>';
    html += '</div>';

    // ── Phase 1: Scoring ──
    if (h.scoring_prompt || h.scoring_response || h.scoring_feedback) {
      if (h.scoring_prompt) {
        html += _percSectionCard(
          '&#9312; Scoring Prompt',
          '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">Prompt sent to LLM to predict actions from perception output alone</div>' +
          '<pre style="max-height:500px">' + esc(h.scoring_prompt) + '</pre>',
          { accent: 'var(--accent3)', collapsed: true, id: uid + '-score-prompt' }
        );
      }
      if (h.scoring_response) {
        html += _percSectionCard(
          '&#9312; Scoring Response',
          '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">LLM\\'s action predictions based on perception output</div>' +
          '<pre style="max-height:400px">' + esc(h.scoring_response) + '</pre>',
          { accent: 'var(--accent3)', collapsed: true, id: uid + '-score-resp' }
        );
      }
      if (h.scoring_feedback) {
        var feedbackLines = h.scoring_feedback.split('\\n');
        var summaryLine = feedbackLines[0] || '';
        html += _percSectionCard(
          '&#9312; Test Results &mdash; ' + esc(summaryLine),
          '<pre style="max-height:500px;white-space:pre-wrap">' + esc(h.scoring_feedback) + '</pre>',
          { accent: scoreColor, collapsed: false, id: uid + '-score-fb' }
        );
      }
    }

    // ── Phase 2: Improvement prompt ──
    if (h.improve_prompt) {
      html += _percSectionCard(
        '&#9313; Improvement Prompt',
        '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">Full prompt sent to LLM to generate improved perception code</div>' +
        '<pre style="max-height:600px">' + esc(h.improve_prompt) + '</pre>',
        { accent: 'var(--accent)', collapsed: true, id: uid + '-improve-prompt' }
      );
    }

    // ── Phase 3: Reasoning ──
    if (h.reasoning) {
      html += _percSectionCard(
        '&#9314; LLM Reasoning',
        '<pre style="max-height:400px;white-space:pre-wrap">' + esc(h.reasoning) + '</pre>',
        { accent: '#bc8cff', collapsed: false, id: uid + '-reasoning' }
      );
    }

    // ── Phase 4: Updated code ──
    html += _percSectionCard(
      '&#9315; Updated Perception Code',
      '<pre style="max-height:400px">' + esc(h.perception || '(empty)') + '</pre>',
      { accent: 'var(--accent2)', badge: (h.perception || '').length + ' chars', collapsed: true, id: uid + '-code' }
    );

    // ── Phase 5: I/O example ──
    if (h.io_example) {
      var ioe = h.io_example;
      var ioeBody = '<div class="side-by-side">';
      ioeBody += '<div><div style="font-size:12px;text-transform:uppercase;color:var(--accent);letter-spacing:0.5px;margin-bottom:8px;font-weight:600">Raw Observation (input)</div>';
      ioeBody += '<pre style="max-height:350px">' + esc(ioe.raw_observation) + '</pre></div>';
      ioeBody += '<div><div style="font-size:12px;text-transform:uppercase;color:var(--accent2);letter-spacing:0.5px;margin-bottom:8px;font-weight:600">Perception Output</div>';
      ioeBody += '<pre style="max-height:350px">' + esc(ioe.perception_output || '(empty)') + '</pre></div>';
      ioeBody += '</div>';
      html += _percSectionCard(
        '&#9316; I/O After Update',
        ioeBody,
        { accent: '#bc8cff', badge: 'Step ' + ioe.step_num, collapsed: true, id: uid + '-io' }
      );
    }

    // ── Connector ──
    if (idx < iterEntries.length - 1) {
      html += '<div style="display:flex;justify-content:center;padding:4px 0"><div style="width:2px;height:24px;background:var(--border)"></div></div>';
    }
  });
  return html;
}

// Render a moment score bar chart for a list of entries
function _renderPercScoreChart(entries, title) {
  if (!entries.length || !entries.some(function(h) { return h.perc_moment_score != null; })) return '';
  var html = '<div class="card"><div class="card-header" style="border-left:3px solid var(--accent3)">' + title + '</div><div class="card-body">';
  html += '<div style="display:flex;align-items:flex-end;gap:8px;height:80px;padding-bottom:4px">';
  entries.forEach(function(h) {
    var pct = h.perc_moment_score != null ? (h.perc_moment_score * 100) : 0;
    var color = pct >= 75 ? 'var(--accent2)' : pct >= 50 ? 'var(--accent3)' : 'var(--danger)';
    html += '<div style="flex:1;display:flex;flex-direction:column;align-items:center;height:100%;justify-content:flex-end">';
    html += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:2px">' + pct.toFixed(0) + '%</div>';
    html += '<div style="width:100%;max-width:40px;height:' + Math.max(pct * 0.7, 2) + 'px;background:' + color + ';border-radius:3px 3px 0 0"></div>';
    html += '<div style="font-size:10px;color:var(--text-muted);margin-top:4px">Iter ' + h.iteration + '</div>';
    html += '</div>';
  });
  html += '</div></div></div>';
  return html;
}

var _percActiveCandidate = {};
function _percShowCandidate(stepIdx, candIdx) {
  _percActiveCandidate[stepIdx] = candIdx;
  // Hide all candidate panels, show selected
  var panels = document.querySelectorAll('.perc-cand-panel-' + stepIdx);
  panels.forEach(function(p) { p.style.display = 'none'; });
  var active = document.getElementById('perc-cand-' + stepIdx + '-' + candIdx);
  if (active) active.style.display = 'block';
  // Update tab styling
  var tabs = document.querySelectorAll('.perc-cand-tab-' + stepIdx);
  tabs.forEach(function(t) {
    t.style.background = 'var(--surface)';
    t.style.color = 'var(--text-muted)';
    t.style.borderColor = 'var(--border)';
  });
  var activeTab = document.getElementById('perc-cand-tab-' + stepIdx + '-' + candIdx);
  if (activeTab) {
    activeTab.style.background = 'var(--accent)';
    activeTab.style.color = 'var(--bg)';
    activeTab.style.borderColor = 'var(--accent)';
  }
}

async function loadPerceptionHistory(stepIdx) {
  var container = document.getElementById('perception-container');
  if (!container) return;

  var data = perceptionCache[stepIdx];
  if (!data) {
    try {
      var resp = await fetch('/api/perception_history?step=' + stepIdx);
      data = await resp.json();
      perceptionCache[stepIdx] = data;
    } catch(e) {
      container.innerHTML = '<pre>' + esc(String(e)) + '</pre>';
      return;
    }
  }

  var html = '';
  var history = data.history || [];

  // Detect multi-candidate: check for entries with "candidate" field or a selection entry
  var isMultiCandidate = history.some(function(h) { return h.candidate != null || h.selected_candidate != null; });
  var selectionEntry = null;
  var iterEntries = history;
  if (isMultiCandidate) {
    selectionEntry = history.filter(function(h) { return h.selected_candidate != null; })[0] || null;
    iterEntries = history.filter(function(h) { return h.candidate != null && h.iteration != null; });
  }

  // ── Overview stats ──
  var totalCost = 0;
  iterEntries.forEach(function(h) { totalCost += (h.cost || 0); });

  if (isMultiCandidate && selectionEntry) {
    var numCands = selectionEntry.num_candidates || 1;
    html += '<div class="stats-grid">';
    html += '<div class="stat-card"><div class="stat-label">Candidates</div><div class="stat-value blue">' + numCands + '</div></div>';
    html += '<div class="stat-card"><div class="stat-label">Selected</div><div class="stat-value green">#' + selectionEntry.selected_candidate + '</div></div>';
    html += '<div class="stat-card"><div class="stat-label">Best Score</div><div class="stat-value green">' + (selectionEntry.final_score != null ? (selectionEntry.final_score * 100).toFixed(1) + '%' : '-') + '</div></div>';
    html += '<div class="stat-card"><div class="stat-label">Total Cost</div><div class="stat-value yellow">$' + totalCost.toFixed(4) + '</div></div>';
    html += '</div>';
  } else {
    var lastScore = iterEntries.length > 0 && iterEntries[iterEntries.length-1].perc_moment_score != null
      ? (iterEntries[iterEntries.length-1].perc_moment_score * 100).toFixed(1) + '%' : '-';
    var firstScore = iterEntries.length > 0 && iterEntries[0].perc_moment_score != null
      ? (iterEntries[0].perc_moment_score * 100).toFixed(1) + '%' : '-';
    html += '<div class="stats-grid">';
    html += '<div class="stat-card"><div class="stat-label">Iterations</div><div class="stat-value blue">' + iterEntries.length + '</div></div>';
    html += '<div class="stat-card"><div class="stat-label">First Moment Score</div><div class="stat-value yellow">' + firstScore + '</div></div>';
    html += '<div class="stat-card"><div class="stat-label">Final Moment Score</div><div class="stat-value green">' + lastScore + '</div></div>';
    html += '<div class="stat-card"><div class="stat-label">Total Cost</div><div class="stat-value yellow">$' + totalCost.toFixed(4) + '</div></div>';
    html += '</div>';
  }

  // ── Input perception code ──
  html += _percSectionCard(
    'Input Perception Code',
    '<pre style="max-height:400px">' + esc(data.input_perception || '(empty)') + '</pre>',
    { accent: 'var(--text-muted)', collapsed: true, id: 'perc-input-code' }
  );

  // ── Perception I/O example (from rollout CSVs) ──
  if (data.example) {
    var ex = data.example;
    var exBody = '<div class="side-by-side">';
    exBody += '<div><div style="font-size:12px;text-transform:uppercase;color:var(--accent);letter-spacing:0.5px;margin-bottom:8px;font-weight:600">Raw Observation (input)</div>';
    exBody += '<pre style="max-height:400px">' + esc(ex.raw_observation) + '</pre></div>';
    exBody += '<div><div style="font-size:12px;text-transform:uppercase;color:var(--accent2);letter-spacing:0.5px;margin-bottom:8px;font-weight:600">Perception Output</div>';
    exBody += '<pre style="max-height:400px">' + esc(ex.perception_output) + '</pre></div>';
    exBody += '</div>';
    html += _percSectionCard(
      'Perception I/O Example (input perception)',
      exBody,
      { accent: 'var(--accent)', badge: 'Step ' + ex.step_num, collapsed: false, id: 'perc-io-example' }
    );
  }

  // ── No history ──
  if (iterEntries.length === 0) {
    html += '<div class="card"><div class="card-body" style="color:var(--text-muted)">No perception improvement iterations recorded for this step.</div></div>';
    html += _percSectionCard(
      'Output Perception Code',
      '<pre style="max-height:400px">' + esc(data.output_perception || '(empty)') + '</pre>',
      { accent: 'var(--accent2)', collapsed: false, id: 'perc-output-final' }
    );
    container.innerHTML = html;
    return;
  }

  // ── Multi-candidate or single-candidate rendering ──
  if (isMultiCandidate) {
    // Group entries by candidate
    var candGroups = {};
    iterEntries.forEach(function(h) {
      var c = h.candidate;
      if (!candGroups[c]) candGroups[c] = [];
      candGroups[c].push(h);
    });
    var candIds = Object.keys(candGroups).sort(function(a,b) { return a - b; });
    var selectedCand = selectionEntry ? selectionEntry.selected_candidate : 1;

    // Candidate comparison chart: final scores side by side
    html += '<div class="card"><div class="card-header" style="border-left:3px solid var(--accent3)">Candidate Final Scores</div><div class="card-body">';
    html += '<div style="display:flex;align-items:flex-end;gap:12px;height:100px;padding-bottom:4px">';
    candIds.forEach(function(cid) {
      var cEntries = candGroups[cid];
      var lastEntry = cEntries[cEntries.length - 1];
      var pct = lastEntry.perc_moment_score != null ? (lastEntry.perc_moment_score * 100) : 0;
      var isSelected = parseInt(cid) === selectedCand;
      var barColor = isSelected ? 'var(--accent2)' : 'var(--accent)';
      var borderStyle = isSelected ? 'border:2px solid var(--accent2)' : '';
      html += '<div style="flex:1;display:flex;flex-direction:column;align-items:center;height:100%;justify-content:flex-end">';
      html += '<div style="font-size:12px;font-weight:600;color:' + (isSelected ? 'var(--accent2)' : 'var(--text-muted)') + ';margin-bottom:2px">' + pct.toFixed(1) + '%</div>';
      html += '<div style="width:100%;max-width:50px;height:' + Math.max(pct * 0.8, 2) + 'px;background:' + barColor + ';border-radius:3px 3px 0 0;' + borderStyle + '"></div>';
      html += '<div style="font-size:11px;margin-top:4px;font-weight:' + (isSelected ? '700' : '400') + ';color:' + (isSelected ? 'var(--accent2)' : 'var(--text-muted)') + '">';
      html += '#' + cid + (isSelected ? ' &#10003;' : '') + '</div>';
      html += '</div>';
    });
    html += '</div></div></div>';

    // Candidate tabs
    html += '<div style="margin:24px 0 12px;font-size:16px;font-weight:700;color:var(--text)">Candidate Pipelines</div>';
    html += '<div style="display:flex;gap:0;margin-bottom:16px;flex-wrap:wrap">';
    candIds.forEach(function(cid, ti) {
      var isSelected = parseInt(cid) === selectedCand;
      var isFirst = ti === 0;
      var isLast = ti === candIds.length - 1;
      var cEntries = candGroups[cid];
      var lastEntry = cEntries[cEntries.length - 1];
      var finalPct = lastEntry.perc_moment_score != null ? (lastEntry.perc_moment_score * 100).toFixed(1) + '%' : '-';
      html += '<div id="perc-cand-tab-' + stepIdx + '-' + cid + '" class="perc-cand-tab-' + stepIdx + '" ';
      html += 'onclick="_percShowCandidate(' + stepIdx + ',' + cid + ')" ';
      html += 'style="padding:8px 16px;cursor:pointer;font-size:13px;';
      html += 'border:1px solid ' + (isFirst ? 'var(--accent)' : 'var(--border)') + ';';
      html += 'background:' + (isFirst ? 'var(--accent)' : 'var(--surface)') + ';';
      html += 'color:' + (isFirst ? 'var(--bg)' : 'var(--text-muted)') + ';';
      html += 'border-radius:' + (isFirst ? '6px 0 0 6px' : isLast ? '0 6px 6px 0' : '0') + ';';
      html += 'transition:all 0.15s">';
      html += 'Candidate #' + cid + ' (' + finalPct + ')';
      if (isSelected) html += ' &#10003;';
      html += '</div>';
    });
    html += '</div>';

    // Candidate panels
    candIds.forEach(function(cid, ti) {
      var cEntries = candGroups[cid];
      html += '<div id="perc-cand-' + stepIdx + '-' + cid + '" class="perc-cand-panel-' + stepIdx + '" style="display:' + (ti === 0 ? 'block' : 'none') + '">';
      html += _renderPercScoreChart(cEntries, 'Candidate #' + cid + ' — Moment Score Across Iterations');
      html += _renderPercIterationPipeline(cEntries, 'perc-c' + cid);
      html += '</div>';
    });

  } else {
    // Single candidate: render directly
    html += _renderPercScoreChart(iterEntries, 'Moment Score Across Iterations');
    html += '<div style="margin:24px 0 12px;font-size:16px;font-weight:700;color:var(--text)">Improvement Pipeline</div>';
    html += _renderPercIterationPipeline(iterEntries, 'perc');
  }

  // ── Final output ──
  html += '<div style="display:flex;align-items:center;gap:12px;margin:20px 0 12px">';
  html += '<div style="background:var(--accent2);color:var(--bg);width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;flex-shrink:0">&#10003;</div>';
  html += '<div style="font-size:15px;font-weight:600;color:var(--accent2)">Final Output Perception</div>';
  if (isMultiCandidate && selectionEntry) {
    html += '<div style="font-size:12px;color:var(--text-muted);margin-left:8px">(from candidate #' + selectionEntry.selected_candidate + ')</div>';
  }
  html += '</div>';
  html += _percSectionCard(
    'Final Perception Code (end of step)',
    '<pre style="max-height:400px">' + esc(data.output_perception || '(empty)') + '</pre>',
    { accent: 'var(--accent2)', collapsed: false, id: 'perc-final-output' }
  );

  container.innerHTML = html;

  // Auto-show first candidate tab
  if (isMultiCandidate && candIds && candIds.length > 0) {
    _percShowCandidate(stepIdx, parseInt(candIds[0]));
  }
}

init();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize mixed_explore.py logs")
    parser.add_argument("log_dir", help="Path to the mixed_explore log directory")
    parser.add_argument("--port", type=int, default=8766, help="Port (default: 8766)")
    parser.add_argument("--open-browser", action="store_true", help="Open browser automatically")
    args = parser.parse_args()

    global LOG_DIR, DATA

    LOG_DIR = os.path.abspath(args.log_dir)
    if not os.path.isdir(LOG_DIR):
        print(f"Error: {LOG_DIR} is not a directory")
        sys.exit(1)

    print(f"Loading logs from {LOG_DIR}...")
    DATA = load_log_dir_light(LOG_DIR)
    print(f"  Found {len(DATA['steps'])} steps")
    print(f"  Total cost: ${DATA['total_cost']:.4f}")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nServing at {url}")
    print("Press Ctrl+C to stop\n")

    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
