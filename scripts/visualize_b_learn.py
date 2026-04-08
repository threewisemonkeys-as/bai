#!/usr/bin/env python3
"""Visualize b_learn.py logs with a local web server.

Usage:
    python visualize_b_learn.py <log_dir> [--port PORT] [--open-browser]

Serves an interactive viewer for b_learn.py log directories.
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
            "has_feedback": os.path.exists(os.path.join(step_path, "feedback_history.json")),
            "has_knowledge": os.path.exists(os.path.join(step_path, "critical_moments.json")),
            "has_qa": os.path.exists(os.path.join(step_path, "qa_pairs.json")),
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
        "critical_moments": read_json(os.path.join(step_path, "critical_moments.json")) or [],
        "new_critical_moments": read_json(os.path.join(step_path, "new_moments.json")) or [],
        "per_trajectory_moments": read_json(os.path.join(step_path, "per_trajectory_moments.json")) or [],
        "qa_pairs": read_json(os.path.join(step_path, "qa_pairs.json")) or [],
        "new_qa_pairs": read_json(os.path.join(step_path, "new_qa_pairs.json")) or [],
        "per_trajectory_qa": read_json(os.path.join(step_path, "per_trajectory_qa.json")) or [],
        "extraction_prompts": read_json(os.path.join(step_path, "extraction_prompts.json")) or [],
    }


def load_feedback(log_dir, step_idx):
    """Load feedback history for a step on demand."""
    step_path = os.path.join(log_dir, f"step_{step_idx}")
    return read_json(os.path.join(step_path, "feedback_history.json")) or []


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
            elif path == "/api/feedback":
                step = int(params.get("step", [0])[0])
                self._json_response(load_feedback(LOG_DIR, step))
            elif path == "/api/experiments":
                step = int(params.get("step", [0])[0])
                self._json_response(load_experiments(LOG_DIR, step))
            else:
                self.send_error(404)
        except Exception as e:
            err = f"[visualize_b_learn] request failed for {self.path}: {e}\n{traceback.format_exc()}"
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
<title>B-Learn Log Viewer</title>
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
  --purple: #bc8cff;
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
.stat-value.purple { color: var(--purple); }

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
.bar.purple { background: var(--purple); }
.bar-label { font-size: 10px; color: var(--text-muted); margin-top: 6px; }
.bar-value { font-size: 10px; color: var(--text-muted); margin-bottom: 2px; white-space: nowrap; }
.chart-legend { display: flex; gap: 16px; margin-top: 8px; font-size: 12px; color: var(--text-muted); }
.legend-item { display: flex; align-items: center; gap: 4px; }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; }

/* Side-by-side diff */
.side-by-side {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}
.side-by-side > div > h3 {
  font-size: 13px; color: var(--text-muted); margin-bottom: 8px;
  text-transform: uppercase; letter-spacing: 0.5px;
}

/* Verdict badges */
.verdict { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
.verdict-correct { background: rgba(63,185,80,0.15); color: var(--accent2); }
.verdict-incorrect { background: rgba(248,81,73,0.15); color: var(--danger); }
.verdict-inconclusive { background: rgba(210,153,34,0.15); color: var(--accent3); }

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

/* Extraction sections */
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

/* Feedback item */
.feedback-item {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; margin-bottom: 8px;
}
.feedback-item-header {
  padding: 10px 16px; cursor: pointer; user-select: none; font-size: 13px;
  display: flex; align-items: center; gap: 12px;
}
.feedback-item-header:hover { background: var(--surface2); }
.feedback-item-body { display: none; padding: 12px 16px; border-top: 1px solid var(--border); }
.feedback-item-body.open { display: block; }
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="topbar-title" id="topbar-title">B-Learn Viewer</div>
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
const feedbackCache = {};
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

function toggleFeedbackItem(el) { el.nextElementSibling.classList.toggle('open'); }

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
    ['feedback', 'Feedback Loop'],
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
  else if (currentTab === 'feedback') html += renderFeedbackPlaceholder(step);
  else if (currentTab === 'experiments') html += renderExperimentsPlaceholder(step);
  else if (currentTab === 'logs') html += renderLogsPlaceholder(step);

  mc.innerHTML = html;

  // Trigger async loads for lazy tabs
  if (currentTab === 'knowledge') loadKnowledge(step.index);
  else if (currentTab === 'feedback') loadFeedback(step.index);
  else if (currentTab === 'experiments') loadExperiments(step.index);
  else if (currentTab === 'logs') loadImproveLog(step.index);
}

// ============ OVERVIEW TAB ============
function renderOverview(step) {
  let html = '';

  // Rollout stats summary
  const rs = step.rollout_stats || {};
  let totalRollouts = 0, successCount = 0;
  const entries = Object.entries(rs);
  entries.forEach(function(e) {
    var v = e[1];
    totalRollouts++;
    var summary = v.summary || {};
    Object.values(summary).forEach(function(env) {
      if (env.avg_prog >= 1) successCount++;
    });
  });
  var failCount = totalRollouts - successCount;

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

  // Build set of new moments for highlighting
  var newMomentSet = new Set();
  (data.new_critical_moments || []).forEach(function(m) {
    newMomentSet.add(JSON.stringify({state: m.state, goal: m.goal}));
  });

  // Build set of new QA pairs for highlighting
  var newQASet = new Set();
  (data.new_qa_pairs || []).forEach(function(q) {
    newQASet.add(q.question);
  });

  var moments = data.critical_moments || [];
  var qaPairs = data.qa_pairs || [];

  // Stats grid
  html += '<div class="stats-grid">' +
    '<div class="stat-card"><div class="stat-label">Total Moments</div><div class="stat-value blue">' + moments.length + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">New Moments</div><div class="stat-value green">' + (data.new_critical_moments || []).length + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Total QA Pairs</div><div class="stat-value purple">' + qaPairs.length + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">New QA Pairs</div><div class="stat-value green">' + (data.new_qa_pairs || []).length + '</div></div>' +
    '</div>';

  // QA Pairs table
  html += '<div class="card"><div class="card-header" onclick="toggleCard(this)">QA Pairs (' + qaPairs.length + ') <span class="toggle">&#9660;</span></div><div class="card-body">';
  if (qaPairs.length === 0) {
    html += '<div style="color:var(--text-muted)">No QA pairs.</div>';
  } else {
    html += '<table class="data-table"><tr><th>Question</th><th>Answer</th><th>Evidence</th><th>Source Step</th></tr>';
    qaPairs.forEach(function(q) {
      var isNew = newQASet.has(q.question);
      var answer = q.answer === true ? 'YES' : q.answer === false ? 'NO' : String(q.answer);
      var answerColor = q.answer ? 'var(--accent2)' : 'var(--danger)';
      html += '<tr class="' + (isNew ? 'new-item' : '') + '">' +
        '<td style="max-width:300px" title="' + esc(q.question) + '">' + esc(q.question) + '</td>' +
        '<td style="color:' + answerColor + ';font-weight:600">' + answer + '</td>' +
        '<td style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(q.evidence) + '">' + esc(q.evidence) + '</td>' +
        '<td>' + esc(q.source_step) + '</td>' +
        '</tr>';
    });
    html += '</table>';
  }
  html += '</div></div>';

  // Critical Moments table
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

  // Per-trajectory extractions (moments + QA)
  var perTrajMoments = data.per_trajectory_moments || [];
  var perTrajQA = data.per_trajectory_qa || [];
  var numPerTraj = Math.max(perTrajMoments.length, perTrajQA.length);
  if (numPerTraj > 0) {
    html += '<div class="card"><div class="card-header" onclick="toggleCard(this)">Per-Trajectory Extractions (' + numPerTraj + ' trajectories) <span class="toggle">&#9654;</span></div><div class="card-body collapsed">';
    for (var ti = 0; ti < numPerTraj; ti++) {
      var trajMoments = perTrajMoments[ti] || [];
      var trajQA = perTrajQA[ti] || [];
      var label = 'Trajectory ' + ti + ' (' + trajMoments.length + ' moments, ' + trajQA.length + ' QA pairs)';
      var combined = {};
      if (trajMoments.length > 0) combined.moments = trajMoments;
      if (trajQA.length > 0) combined.qa_pairs = trajQA;
      html += '<div class="extraction-section">' +
        '<div class="extraction-header" onclick="toggleExtraction(this)">' +
        '<span style="color:var(--accent);font-weight:600">' + esc(label) + '</span>' +
        '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
        '</div>' +
        '<div class="extraction-body"><pre>' + esc(JSON.stringify(combined, null, 2)) + '</pre></div>' +
        '</div>';
    }
    html += '</div></div>';
  }

  // Extraction prompts/responses
  var extractionPrompts = data.extraction_prompts || [];
  if (extractionPrompts.length > 0) {
    html += '<div class="card"><div class="card-header" onclick="toggleCard(this)">Extraction Prompts & Responses (' + extractionPrompts.length + ' trajectories) <span class="toggle">&#9654;</span></div><div class="card-body collapsed">';
    extractionPrompts.forEach(function(rec, idx) {
      var trajLabel = rec.trajectory ? rec.trajectory.split('/').pop() : ('Trajectory ' + idx);

      // Summary prompt/response
      if (rec.summary_prompt) {
        html += '<div class="extraction-section">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--accent);font-weight:600">Summary Generation — ' + esc(trajLabel) + '</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body">' +
          '<div style="margin-bottom:8px;font-weight:600;color:var(--accent)">Prompt:</div>' +
          '<pre style="max-height:400px;overflow:auto">' + esc(rec.summary_prompt) + '</pre>' +
          '<div style="margin:8px 0;font-weight:600;color:var(--accent2)">Response:</div>' +
          '<pre style="max-height:400px;overflow:auto">' + esc(rec.summary_response || '(no response)') + '</pre>' +
          '</div></div>';
      }

      // Extraction (QA + moments) prompt/response
      if (rec.extraction_prompt) {
        html += '<div class="extraction-section">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--accent);font-weight:600">QA & Moments Extraction — ' + esc(trajLabel) + '</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body">' +
          '<div style="margin-bottom:8px;font-weight:600;color:var(--accent)">Prompt:</div>' +
          '<pre style="max-height:400px;overflow:auto">' + esc(rec.extraction_prompt) + '</pre>' +
          '<div style="margin:8px 0;font-weight:600;color:var(--accent2)">Response:</div>' +
          '<pre style="max-height:400px;overflow:auto">' + esc(rec.extraction_response || '(no response)') + '</pre>' +
          '</div></div>';
      }
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

// ============ FEEDBACK LOOP TAB ============
function renderFeedbackPlaceholder(step) {
  return '<div id="feedback-container"><div class="loading">Loading feedback data...</div></div>';
}

async function loadFeedback(stepIdx) {
  var container = document.getElementById('feedback-container');
  if (!container) return;

  var data = feedbackCache[stepIdx];
  if (!data) {
    try {
      var resp = await fetch('/api/feedback?step=' + stepIdx);
      data = await resp.json();
      feedbackCache[stepIdx] = data;
    } catch(e) {
      container.innerHTML = '<pre>' + esc(String(e)) + '</pre>';
      return;
    }
  }

  if (!data || data.length === 0) {
    container.innerHTML = '<div class="card"><div class="card-body" style="color:var(--text-muted)">No feedback history found for this step.</div></div>';
    return;
  }

  var html = '';

  // Compute totals across iterations
  var totalIterCost = 0;
  data.forEach(function(iter) {
    totalIterCost += (iter.summary_cost || 0) +
      (iter.qa_forward_cost || 0) + (iter.qa_feedback_cost || 0) + (iter.qa_improve_cost || 0) +
      (iter.moment_forward_cost || 0) + (iter.moment_feedback_cost || 0) + (iter.moment_improve_cost || 0);
  });

  var lastIter = data[data.length - 1];
  html += '<div class="stats-grid">' +
    '<div class="stat-card"><div class="stat-label">Iterations</div><div class="stat-value blue">' + data.length + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Final Moment Correct</div><div class="stat-value green">' + (lastIter.moment_num_correct || 0) + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Final Moment Incorrect</div><div class="stat-value red">' + (lastIter.moment_num_incorrect || 0) + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Final QA Correct</div><div class="stat-value green">' + (lastIter.qa_num_correct || 0) + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Final QA Incorrect</div><div class="stat-value red">' + (lastIter.qa_num_incorrect || 0) + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Total Loop Cost</div><div class="stat-value yellow">$' + totalIterCost.toFixed(4) + '</div></div>' +
    '</div>';

  // Chart: moment correct/incorrect across iterations
  var hasMomentData = data.some(function(iter) { return iter.moment_num_moments; });
  var hasQAData = data.some(function(iter) { return iter.qa_num_pairs; });

  if (hasMomentData) {
    html += '<div class="chart-container"><div class="chart-title">Moment Verdicts Across Iterations</div><div class="bar-chart">';
    data.forEach(function(iter) {
      var total = (iter.moment_num_correct||0) + (iter.moment_num_incorrect||0) + (iter.moment_num_inconclusive||0);
      if (total === 0) total = 1;
      var correctPct = ((iter.moment_num_correct||0) / total * 100);
      var incorrectPct = ((iter.moment_num_incorrect||0) / total * 100);
      var inconclusivePct = ((iter.moment_num_inconclusive||0) / total * 100);
      html += '<div class="bar-group">' +
        '<div class="bar-value" style="font-size:9px">' + (iter.moment_num_correct||0) + '/' + (iter.moment_num_incorrect||0) + '/' + (iter.moment_num_inconclusive||0) + '</div>' +
        '<div style="display:flex;flex-direction:column;gap:1px;align-items:center;width:100%;max-width:30px">' +
          '<div class="bar green" style="height:' + Math.max(correctPct * 1.5, 2) + 'px;max-width:30px;border-radius:3px"></div>' +
          '<div class="bar red" style="height:' + Math.max(incorrectPct * 1.5, 2) + 'px;max-width:30px;border-radius:3px"></div>' +
          '<div class="bar yellow" style="height:' + Math.max(inconclusivePct * 1.5, 2) + 'px;max-width:30px;border-radius:3px"></div>' +
        '</div>' +
        '<div class="bar-label">Iter ' + (iter.iteration || '?') + '</div>' +
      '</div>';
    });
    html += '</div>' +
      '<div class="chart-legend">' +
      '<div class="legend-item"><div class="legend-dot" style="background:var(--accent2)"></div>Correct</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:var(--danger)"></div>Incorrect</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:var(--accent3)"></div>Inconclusive</div>' +
      '</div></div>';
  }

  if (hasQAData) {
    html += '<div class="chart-container"><div class="chart-title">QA Verdicts Across Iterations</div><div class="bar-chart">';
    data.forEach(function(iter) {
      var total = (iter.qa_num_correct||0) + (iter.qa_num_incorrect||0) + (iter.qa_num_inconclusive||0);
      if (total === 0) total = 1;
      var correctPct = ((iter.qa_num_correct||0) / total * 100);
      var incorrectPct = ((iter.qa_num_incorrect||0) / total * 100);
      var inconclusivePct = ((iter.qa_num_inconclusive||0) / total * 100);
      html += '<div class="bar-group">' +
        '<div class="bar-value" style="font-size:9px">' + (iter.qa_num_correct||0) + '/' + (iter.qa_num_incorrect||0) + '/' + (iter.qa_num_inconclusive||0) + '</div>' +
        '<div style="display:flex;flex-direction:column;gap:1px;align-items:center;width:100%;max-width:30px">' +
          '<div class="bar green" style="height:' + Math.max(correctPct * 1.5, 2) + 'px;max-width:30px;border-radius:3px"></div>' +
          '<div class="bar red" style="height:' + Math.max(incorrectPct * 1.5, 2) + 'px;max-width:30px;border-radius:3px"></div>' +
          '<div class="bar yellow" style="height:' + Math.max(inconclusivePct * 1.5, 2) + 'px;max-width:30px;border-radius:3px"></div>' +
        '</div>' +
        '<div class="bar-label">Iter ' + (iter.iteration || '?') + '</div>' +
      '</div>';
    });
    html += '</div>' +
      '<div class="chart-legend">' +
      '<div class="legend-item"><div class="legend-dot" style="background:var(--accent2)"></div>Correct</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:var(--danger)"></div>Incorrect</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:var(--accent3)"></div>Inconclusive</div>' +
      '</div></div>';
  }

  // Per-iteration details
  data.forEach(function(iter, iterIdx) {
    var iterHtml = '';

    // Cost breakdown
    var iterCosts = [];
    if (iter.summary_cost) iterCosts.push('Summary: $' + iter.summary_cost.toFixed(6));
    if (iter.qa_forward_cost) iterCosts.push('QA Fwd: $' + iter.qa_forward_cost.toFixed(6));
    if (iter.qa_feedback_cost) iterCosts.push('QA FB: $' + iter.qa_feedback_cost.toFixed(6));
    if (iter.qa_improve_cost) iterCosts.push('QA Imp: $' + iter.qa_improve_cost.toFixed(6));
    if (iter.moment_forward_cost) iterCosts.push('Mom Fwd: $' + iter.moment_forward_cost.toFixed(6));
    if (iter.moment_feedback_cost) iterCosts.push('Mom FB: $' + iter.moment_feedback_cost.toFixed(6));
    if (iter.moment_improve_cost) iterCosts.push('Mom Imp: $' + iter.moment_improve_cost.toFixed(6));

    if (iterCosts.length > 0) {
      iterHtml += '<div style="margin-bottom:12px;font-size:12px;color:var(--text-muted)">' +
        'Costs: ' + iterCosts.join(' | ') + '</div>';
    }

    // Summary improvement section
    if (iter.summary_cost) {
      iterHtml += '<div style="margin-bottom:16px;padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:6px">' +
        '<div style="font-size:13px;font-weight:600;margin-bottom:4px;color:var(--accent)">3.3 Summary-based Improvement</div>' +
        '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">Cost: $' + iter.summary_cost.toFixed(6) + '</div>';

      if (iter.summary_improve_prompt) {
        iterHtml += '<div class="extraction-section">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--text-muted);font-weight:600">Improve Prompt</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body"><pre style="max-height:600px">' + esc(iter.summary_improve_prompt) + '</pre></div>' +
          '</div>';
      }
      if (iter.summary_improve_response) {
        iterHtml += '<div class="extraction-section">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--accent2);font-weight:600">Improve Response</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body"><pre style="max-height:600px">' + esc(iter.summary_improve_response) + '</pre></div>' +
          '</div>';
      }
      iterHtml += '</div>';
    }

    // QA feedback section
    var qaDetails = iter.qa_feedback_details || [];
    if (qaDetails.length > 0 || iter.qa_num_pairs) {
      iterHtml += '<div style="margin-bottom:16px;padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:6px">' +
        '<div style="font-size:13px;font-weight:600;margin-bottom:8px;color:var(--purple)">3.4 QA-based Improvement</div>';

      iterHtml += '<div class="stats-grid" style="margin-bottom:12px">' +
        '<div class="stat-card"><div class="stat-label">QA Pairs Tested</div><div class="stat-value purple" style="font-size:18px">' + (iter.qa_num_pairs||0) + '</div></div>' +
        '<div class="stat-card"><div class="stat-label">Correct</div><div class="stat-value green" style="font-size:18px">' + (iter.qa_num_correct||0) + '</div></div>' +
        '<div class="stat-card"><div class="stat-label">Incorrect</div><div class="stat-value red" style="font-size:18px">' + (iter.qa_num_incorrect||0) + '</div></div>' +
        '<div class="stat-card"><div class="stat-label">Inconclusive</div><div class="stat-value yellow" style="font-size:18px">' + (iter.qa_num_inconclusive||0) + '</div></div>' +
        '</div>';

      // QA feedback prompts and responses (collapsible)
      var qaFbPrompts = iter.qa_feedback_prompts || [];
      var qaFbResponses = iter.qa_feedback_responses || [];
      if (qaFbPrompts.length > 0) {
        iterHtml += '<div class="extraction-section" style="margin-bottom:12px">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--text-muted);font-weight:600">Feedback Prompt' + (qaFbPrompts.length > 1 ? 's (' + qaFbPrompts.length + ' batches)' : '') + '</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body">';
        qaFbPrompts.forEach(function(p, pi) {
          if (qaFbPrompts.length > 1) {
            iterHtml += '<div style="font-size:11px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px">Batch ' + (pi+1) + '</div>';
          }
          iterHtml += '<pre style="max-height:600px">' + esc(p) + '</pre>';
        });
        iterHtml += '</div></div>';
      }
      if (qaFbResponses.length > 0) {
        iterHtml += '<div class="extraction-section" style="margin-bottom:12px">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--accent2);font-weight:600">Feedback Response' + (qaFbResponses.length > 1 ? 's (' + qaFbResponses.length + ' batches)' : '') + '</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body">';
        qaFbResponses.forEach(function(r, ri) {
          if (qaFbResponses.length > 1) {
            iterHtml += '<div style="font-size:11px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px">Batch ' + (ri+1) + '</div>';
          }
          iterHtml += '<pre style="max-height:600px">' + esc(r) + '</pre>';
        });
        iterHtml += '</div></div>';
      }

      // QA improve prompt and response (collapsible)
      if (iter.qa_improve_prompt) {
        iterHtml += '<div class="extraction-section" style="margin-bottom:12px">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--text-muted);font-weight:600">Improve Prompt</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body"><pre style="max-height:600px">' + esc(iter.qa_improve_prompt) + '</pre></div>' +
          '</div>';
      }
      if (iter.qa_improve_response) {
        iterHtml += '<div class="extraction-section" style="margin-bottom:12px">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--accent2);font-weight:600">Improve Response</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body"><pre style="max-height:600px">' + esc(iter.qa_improve_response) + '</pre></div>' +
          '</div>';
      }

      qaDetails.forEach(function(d) {
        var verdictClass = d.verdict === 'CORRECT' ? 'verdict-correct' :
                           d.verdict === 'INCORRECT' ? 'verdict-incorrect' : 'verdict-inconclusive';
        var qSnip = d.question ? (d.question.length > 80 ? d.question.slice(0,80) + '...' : d.question) : '';

        iterHtml += '<div class="feedback-item">' +
          '<div class="feedback-item-header" onclick="toggleFeedbackItem(this)">' +
            '<span class="verdict ' + verdictClass + '">' + esc(d.verdict) + '</span>' +
            '<span style="font-family:var(--font-mono);font-size:12px;background:var(--surface2);padding:2px 8px;border-radius:4px">' + esc(d.predicted_answer || '') + '</span>' +
            '<span style="color:var(--text-muted);font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(d.question) + '">' + esc(qSnip) + '</span>' +
            '<span style="color:var(--text-muted);font-size:11px">&#9654;</span>' +
          '</div>' +
          '<div class="feedback-item-body">' +
            '<div style="margin-bottom:12px"><div style="font-size:11px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px">Question</div><pre style="max-height:100px">' + esc(d.question || '') + '</pre></div>' +
            '<div style="margin-bottom:12px;display:flex;gap:16px">' +
              '<div style="flex:1"><div style="font-size:11px;text-transform:uppercase;color:var(--accent2);margin-bottom:4px">Correct Answer</div><span style="font-weight:600">' + esc(d.correct_answer || '') + '</span></div>' +
              '<div style="flex:1"><div style="font-size:11px;text-transform:uppercase;color:var(--accent);margin-bottom:4px">Predicted Answer</div><span style="font-weight:600">' + esc(d.predicted_answer || '') + '</span></div>' +
            '</div>' +
            '<div style="margin-bottom:12px"><div style="font-size:11px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px">Evidence</div><pre style="max-height:150px">' + esc(d.evidence || '') + '</pre></div>' +
            '<div style="margin-bottom:12px"><div style="font-size:11px;text-transform:uppercase;color:var(--accent);margin-bottom:4px">Reasoning</div><pre style="max-height:150px">' + esc(d.reasoning || '') + '</pre></div>' +
            '<div><div style="font-size:11px;text-transform:uppercase;color:var(--accent3);margin-bottom:4px">Feedback</div><pre style="max-height:150px">' + esc(d.feedback || '') + '</pre></div>' +
          '</div></div>';
      });
      iterHtml += '</div>';
    }

    // Moment feedback section
    var momentDetails = iter.moment_feedback_details || [];
    if (momentDetails.length > 0 || iter.moment_num_moments) {
      iterHtml += '<div style="margin-bottom:16px;padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:6px">' +
        '<div style="font-size:13px;font-weight:600;margin-bottom:8px;color:var(--accent)">3.5 Moment-based Improvement</div>';

      iterHtml += '<div class="stats-grid" style="margin-bottom:12px">' +
        '<div class="stat-card"><div class="stat-label">Moments Tested</div><div class="stat-value blue" style="font-size:18px">' + (iter.moment_num_moments||0) + '</div></div>' +
        '<div class="stat-card"><div class="stat-label">Correct</div><div class="stat-value green" style="font-size:18px">' + (iter.moment_num_correct||0) + '</div></div>' +
        '<div class="stat-card"><div class="stat-label">Incorrect</div><div class="stat-value red" style="font-size:18px">' + (iter.moment_num_incorrect||0) + '</div></div>' +
        '<div class="stat-card"><div class="stat-label">Inconclusive</div><div class="stat-value yellow" style="font-size:18px">' + (iter.moment_num_inconclusive||0) + '</div></div>' +
        '</div>';

      // Feedback prompts and responses (collapsible)
      var fbPrompts = iter.moment_feedback_prompts || [];
      var fbResponses = iter.moment_feedback_responses || [];
      if (fbPrompts.length > 0) {
        iterHtml += '<div class="extraction-section" style="margin-bottom:12px">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--text-muted);font-weight:600">Feedback Prompt' + (fbPrompts.length > 1 ? 's (' + fbPrompts.length + ' batches)' : '') + '</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body">';
        fbPrompts.forEach(function(p, pi) {
          if (fbPrompts.length > 1) {
            iterHtml += '<div style="font-size:11px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px">Batch ' + (pi+1) + '</div>';
          }
          iterHtml += '<pre style="max-height:600px">' + esc(p) + '</pre>';
        });
        iterHtml += '</div></div>';
      }
      if (fbResponses.length > 0) {
        iterHtml += '<div class="extraction-section" style="margin-bottom:12px">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--accent2);font-weight:600">Feedback Response' + (fbResponses.length > 1 ? 's (' + fbResponses.length + ' batches)' : '') + '</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body">';
        fbResponses.forEach(function(r, ri) {
          if (fbResponses.length > 1) {
            iterHtml += '<div style="font-size:11px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px">Batch ' + (ri+1) + '</div>';
          }
          iterHtml += '<pre style="max-height:600px">' + esc(r) + '</pre>';
        });
        iterHtml += '</div></div>';
      }

      // Improve prompt and response (collapsible)
      if (iter.moment_improve_prompt) {
        iterHtml += '<div class="extraction-section" style="margin-bottom:12px">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--text-muted);font-weight:600">Improve Prompt</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body"><pre style="max-height:600px">' + esc(iter.moment_improve_prompt) + '</pre></div>' +
          '</div>';
      }
      if (iter.moment_improve_response) {
        iterHtml += '<div class="extraction-section" style="margin-bottom:12px">' +
          '<div class="extraction-header" onclick="toggleExtraction(this)">' +
          '<span style="color:var(--accent2);font-weight:600">Improve Response</span>' +
          '<span style="margin-left:auto;color:var(--text-muted);font-size:12px">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body"><pre style="max-height:600px">' + esc(iter.moment_improve_response) + '</pre></div>' +
          '</div>';
      }

      momentDetails.forEach(function(d) {
        var verdictClass = d.verdict === 'CORRECT' ? 'verdict-correct' :
                           d.verdict === 'INCORRECT' ? 'verdict-incorrect' : 'verdict-inconclusive';
        var goalSnip = d.moment_goal ? (d.moment_goal.length > 60 ? d.moment_goal.slice(0,60) + '...' : d.moment_goal) : '';

        iterHtml += '<div class="feedback-item">' +
          '<div class="feedback-item-header" onclick="toggleFeedbackItem(this)">' +
            '<span class="verdict ' + verdictClass + '">' + esc(d.verdict) + '</span>' +
            '<span style="font-family:var(--font-mono);font-size:12px;background:var(--surface2);padding:2px 8px;border-radius:4px">' + esc(d.predicted_action || '') + '</span>' +
            '<span style="color:var(--text-muted);font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(d.moment_goal) + '">' + esc(goalSnip) + '</span>' +
            '<span style="color:var(--text-muted);font-size:11px">&#9654;</span>' +
          '</div>' +
          '<div class="feedback-item-body">';

        iterHtml += '<div style="margin-bottom:12px">' +
          '<div style="font-size:11px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px;letter-spacing:0.5px">State</div>' +
          '<pre style="max-height:200px">' + esc(d.moment_state || '') + '</pre></div>';

        iterHtml += '<div style="margin-bottom:12px">' +
          '<div style="font-size:11px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px;letter-spacing:0.5px">Goal</div>' +
          '<pre style="max-height:100px">' + esc(d.moment_goal || '') + '</pre></div>';

        var goodActs = (d.moment_good_actions || []).join(', ') || '-';
        var badActs = (d.moment_bad_actions || []).join(', ') || '-';
        iterHtml += '<div style="margin-bottom:12px;display:flex;gap:16px">' +
          '<div style="flex:1"><div style="font-size:11px;text-transform:uppercase;color:var(--accent2);margin-bottom:4px;letter-spacing:0.5px">Good Actions</div>' +
          '<span style="font-family:var(--font-mono);font-size:12px">' + esc(goodActs) + '</span></div>' +
          '<div style="flex:1"><div style="font-size:11px;text-transform:uppercase;color:var(--danger);margin-bottom:4px;letter-spacing:0.5px">Bad Actions</div>' +
          '<span style="font-family:var(--font-mono);font-size:12px">' + esc(badActs) + '</span></div>' +
          '</div>';

        if (d.perception_output) {
          iterHtml += '<div style="margin-bottom:12px">' +
            '<div style="font-size:11px;text-transform:uppercase;color:var(--purple);margin-bottom:4px;letter-spacing:0.5px">Perception Output</div>' +
            '<pre style="max-height:200px">' + esc(d.perception_output) + '</pre></div>';
        }

        iterHtml += '<div style="margin-bottom:12px">' +
          '<div style="font-size:11px;text-transform:uppercase;color:var(--accent);margin-bottom:4px;letter-spacing:0.5px">Predicted Action</div>' +
          '<span style="font-family:var(--font-mono);font-size:13px;font-weight:600">' + esc(d.predicted_action || '') + '</span></div>';

        if (d.reasoning) {
          iterHtml += '<div style="margin-bottom:12px">' +
            '<div style="font-size:11px;text-transform:uppercase;color:var(--accent);margin-bottom:4px;letter-spacing:0.5px">Reasoning</div>' +
            '<pre style="max-height:200px">' + esc(d.reasoning) + '</pre></div>';
        }

        iterHtml += '<div>' +
          '<div style="font-size:11px;text-transform:uppercase;color:var(--accent3);margin-bottom:4px;letter-spacing:0.5px">Feedback</div>' +
          '<pre style="max-height:200px">' + esc(d.feedback || '') + '</pre></div>';

        iterHtml += '</div></div>';
      });
      iterHtml += '</div>';
    }

    // Build iteration title summary
    var titleParts = [];
    if (iter.summary_cost) titleParts.push('summary');
    if (iter.qa_num_pairs) titleParts.push('' + (iter.qa_num_correct||0) + '/' + (iter.qa_num_incorrect||0) + ' QA');
    if (iter.moment_num_moments) titleParts.push('' + (iter.moment_num_correct||0) + '/' + (iter.moment_num_incorrect||0) + ' moments');
    var iterTitle = 'Iteration ' + (iter.iteration || '?') +
      (titleParts.length > 0 ? ' &mdash; ' + titleParts.join(', ') : '');
    html += collapsible(iterTitle, iterHtml, iterIdx === data.length - 1);
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
    html += '<div style="color:var(--text-muted)">No input experiments (initial step).</div>';
  } else {
    inputExps.forEach(function(exp, i) {
      var text = typeof exp === 'string' ? exp : JSON.stringify(exp, null, 2);
      html += '<div style="background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:12px;margin-bottom:8px;font-size:13px">' +
        '<span style="color:var(--accent);font-weight:600">#' + (i+1) + '</span> ' +
        '<pre style="margin-top:8px;border:none;padding:0;background:transparent">' + esc(text) + '</pre></div>';
    });
  }
  html += '</div></div>';

  // Output experiments
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

init();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize b_learn.py logs")
    parser.add_argument("log_dir", help="Path to the b_learn log directory")
    parser.add_argument("--port", type=int, default=8767, help="Port (default: 8767)")
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
