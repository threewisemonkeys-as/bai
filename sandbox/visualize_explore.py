#!/usr/bin/env python3
"""Visualize explore.py logs with a local web server.

Usage:
    python visualize_explore.py <log_dir> [--port PORT]

Opens a browser with an interactive viewer for explore.py log directories.
Trajectories are loaded on-demand to keep the UI fast.
"""

import argparse
import csv
import io
import json
import os
import sys
import threading
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
    json_file = None
    csv_file = None
    for root, dirs, files in os.walk(rollout_dir):
        for f in files:
            if f.endswith(".json"):
                json_file = os.path.join(root, f)
            if f.endswith(".csv"):
                csv_file = os.path.join(root, f)
    return json_file, csv_file


def _sort_key(name, phase_type):
    prefix = "hypothesis_" if phase_type == "explore" else "baseline_"
    try:
        return int(name.replace(prefix, ""))
    except ValueError:
        return 999


def load_phase_light(phase_dir, phase_type):
    """Load phase data WITHOUT trajectory CSVs (lightweight)."""
    phase = {
        "beliefs": read_file(os.path.join(phase_dir, "beliefs.txt")),
        "perception": read_file(os.path.join(phase_dir, "perception.py")),
        "rollout_stats": read_json(os.path.join(phase_dir, "rollout_stats.json")),
        "hypotheses": read_json(os.path.join(phase_dir, "hypotheses.json")),
        "rollouts": [],
    }

    rollouts_dir = os.path.join(phase_dir, "rollouts")
    if not os.path.isdir(rollouts_dir):
        return phase

    rollout_names = sorted(os.listdir(rollouts_dir), key=lambda x: _sort_key(x, phase_type))
    for name in rollout_names:
        rollout_path = os.path.join(rollouts_dir, name)
        if not os.path.isdir(rollout_path):
            continue

        json_file, _ = find_rollout_files(rollout_path)
        meta = read_json(json_file) if json_file else None

        hypothesis_text = ""
        hyp_file = os.path.join(rollout_path, "hypothesis.txt")
        if os.path.exists(hyp_file):
            hypothesis_text = read_file(hyp_file)

        phase["rollouts"].append({
            "name": name,
            "hypothesis": hypothesis_text,
            "meta": meta,
        })

    return phase


def load_log_dir_light(log_dir):
    """Load log directory without trajectory data."""
    data = {
        "config": read_file(os.path.join(log_dir, "config.yaml")),
        "steps": [],
    }

    step_dirs = []
    for name in os.listdir(log_dir):
        if name.startswith("step_") and os.path.isdir(os.path.join(log_dir, name)):
            try:
                idx = int(name.replace("step_", ""))
                step_dirs.append((idx, name))
            except ValueError:
                pass
    step_dirs.sort()

    for idx, name in step_dirs:
        step_path = os.path.join(log_dir, name)
        step = {
            "index": idx,
            "input_beliefs": read_file(os.path.join(step_path, "input_beliefs.txt")),
            "input_perception": read_file(os.path.join(step_path, "input_perception.txt")),
            "baseline": load_phase_light(os.path.join(step_path, "baseline"), "baseline"),
            "explore": load_phase_light(os.path.join(step_path, "explore"), "explore"),
        }
        data["steps"].append(step)

    return data


def load_trajectory(log_dir, step_idx, phase, rollout_name):
    """Load a single trajectory CSV on demand."""
    step_dir = os.path.join(log_dir, f"step_{step_idx}", phase, "rollouts", rollout_name)
    if not os.path.isdir(step_dir):
        return []

    _, csv_file = find_rollout_files(step_dir)
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


def _extract_rollout_label(text):
    """Extract rollout name (baseline_N or hypothesis_N) from a log line."""
    for part in text.split("/"):
        if "baseline_" in part or "hypothesis_" in part:
            return part
    return ""


def parse_improve_log(log_path):
    """Parse an improve.log into structured sections for display."""
    text = read_file(log_path)
    if not text:
        return {"summaries": [], "hypothesis_summaries": [], "improve_prompt": "", "improve_response": ""}

    import re
    lines = text.split("\n")

    sections = []
    current_marker = None
    current_body = []

    timestamp_re = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - INFO - (.*)$")

    for line in lines:
        m = timestamp_re.match(line)
        if m:
            if current_marker:
                sections.append((current_marker, "\n".join(current_body).strip()))
            current_marker = m.group(1).strip()
            current_body = []
        else:
            current_body.append(line)

    if current_marker:
        sections.append((current_marker, "\n".join(current_body).strip()))

    result = {
        "summaries": [],            # Instructions/perception summaries
        "hypothesis_summaries": [],  # Hypothesis-specific summaries (explore phase)
        "improve_prompt": "",
        "improve_response": "",
    }

    # Index summaries by rollout for matching prompts to responses
    summary_by_rollout = {}   # rollout -> dict
    hyp_summary_by_rollout = {}

    for i, (marker, body) in enumerate(sections):
        # --- Instructions/perception summary prompt ---
        if marker.startswith("Instructions/perception summary prompt"):
            rollout = ""
            if i > 0:
                rollout = _extract_rollout_label(sections[i - 1][0])
            entry = {"rollout": rollout, "prompt": body, "response": ""}
            result["summaries"].append(entry)
            summary_by_rollout[rollout] = entry

        # --- Instructions/perception summary response ---
        elif marker.startswith("Instructions/perception summary LLM response for"):
            rollout = _extract_rollout_label(marker)
            if rollout in summary_by_rollout:
                summary_by_rollout[rollout]["response"] = body
            else:
                entry = {"rollout": rollout, "prompt": "", "response": body}
                result["summaries"].append(entry)
                summary_by_rollout[rollout] = entry

        # --- Hypothesis summary prompt ---
        elif marker.startswith("Hypothesis summary prompt"):
            rollout = ""
            if i > 0:
                rollout = _extract_rollout_label(sections[i - 1][0])
            entry = {"rollout": rollout, "prompt": body, "response": ""}
            result["hypothesis_summaries"].append(entry)
            hyp_summary_by_rollout[rollout] = entry

        # --- Hypothesis summary response ---
        elif marker.startswith("Hypothesis summary LLM response for"):
            rollout = _extract_rollout_label(marker)
            if rollout in hyp_summary_by_rollout:
                hyp_summary_by_rollout[rollout]["response"] = body
            else:
                entry = {"rollout": rollout, "prompt": "", "response": body}
                result["hypothesis_summaries"].append(entry)
                hyp_summary_by_rollout[rollout] = entry

        # --- Improve prompt (handles both "Improve step prompt" and "Baseline improve step prompt") ---
        elif "improve step prompt" in marker.lower():
            result["improve_prompt"] = body

        # --- Improve response ---
        elif "improve step llm response" in marker.lower():
            result["improve_response"] = body

    return result


def compute_meta_stats(data):
    stats = []
    for step in data["steps"]:
        s = {"step": step["index"]}

        for phase_key in ("baseline", "explore"):
            ph_stats = step[phase_key]["rollout_stats"] or {}
            ph_steps = []
            ph_progs = []
            ph_costs = []
            for v in ph_stats.values():
                for env_data in v.get("summary", {}).values():
                    ph_steps.append(env_data.get("avg_steps", 0))
                    ph_progs.append(env_data.get("avg_prog", 0))
                    ph_costs.append(env_data.get("total_cost", 0))
            s[f"{phase_key}_avg_steps"] = sum(ph_steps) / len(ph_steps) if ph_steps else 0
            s[f"{phase_key}_avg_prog"] = sum(ph_progs) / len(ph_progs) if ph_progs else 0
            s[f"{phase_key}_total_cost"] = sum(ph_costs)
            s[f"{phase_key}_num_rollouts"] = len(ph_stats)

        s["step_total_cost"] = s["baseline_total_cost"] + s["explore_total_cost"]
        stats.append(s)

    cum = 0
    for s in stats:
        cum += s["step_total_cost"]
        s["cumulative_cost"] = cum

    return stats


def search_logs(log_dir, data, query, scopes, max_results=500):
    """Search across trajectories, beliefs, and hypotheses. Returns matches."""
    if not query:
        return {"results": [], "total": 0, "truncated": False}

    q = query.lower()
    results = []
    total = 0

    for step in data["steps"]:
        step_idx = step["index"]

        for phase_key in ("baseline", "explore"):
            phase = step[phase_key]

            # --- Search beliefs ---
            if "beliefs" in scopes and phase.get("beliefs"):
                for line_no, line in enumerate(phase["beliefs"].split("\n"), 1):
                    if q in line.lower():
                        total += 1
                        if len(results) < max_results:
                            results.append({
                                "type": "beliefs",
                                "step": step_idx,
                                "phase": phase_key,
                                "context": line.strip(),
                                "line": line_no,
                            })

            # --- Search hypotheses ---
            if "hypotheses" in scopes and phase.get("hypotheses"):
                for hi, hyp in enumerate(phase["hypotheses"]):
                    if q in hyp.lower():
                        total += 1
                        if len(results) < max_results:
                            results.append({
                                "type": "hypotheses",
                                "step": step_idx,
                                "phase": phase_key,
                                "context": hyp[:200],
                                "hyp_index": hi,
                            })

            # --- Search rollout hypothesis text ---
            if "hypotheses" in scopes:
                for r in phase.get("rollouts", []):
                    if r.get("hypothesis") and q in r["hypothesis"].lower():
                        total += 1
                        if len(results) < max_results:
                            results.append({
                                "type": "hypotheses",
                                "step": step_idx,
                                "phase": phase_key,
                                "rollout": r["name"],
                                "context": r["hypothesis"][:200],
                            })

            # --- Search trajectory actions / reasoning ---
            if "actions" in scopes or "reasoning" in scopes:
                for r in phase.get("rollouts", []):
                    rollout_name = r["name"]
                    traj = load_trajectory(log_dir, step_idx, phase_key, rollout_name)
                    for t in traj:
                        if "actions" in scopes and q in (t.get("action") or "").lower():
                            total += 1
                            if len(results) < max_results:
                                results.append({
                                    "type": "actions",
                                    "step": step_idx,
                                    "phase": phase_key,
                                    "rollout": rollout_name,
                                    "traj_step": t.get("step", ""),
                                    "context": t["action"],
                                })
                        if "reasoning" in scopes and q in (t.get("reasoning") or "").lower():
                            total += 1
                            if len(results) < max_results:
                                snippet = t["reasoning"]
                                # Extract ~200 chars around the match
                                idx = snippet.lower().find(q)
                                start = max(0, idx - 80)
                                end = min(len(snippet), idx + len(query) + 120)
                                context = ("..." if start > 0 else "") + snippet[start:end] + ("..." if end < len(snippet) else "")
                                results.append({
                                    "type": "reasoning",
                                    "step": step_idx,
                                    "phase": phase_key,
                                    "rollout": rollout_name,
                                    "traj_step": t.get("step", ""),
                                    "context": context,
                                })

    return {"results": results, "total": total, "truncated": total > max_results}


# ============= HTTP Server =============

LOG_DIR = None
DATA = None
META = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default logging

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_html()
        elif path == "/api/data":
            self._json_response({"data": DATA, "meta": META})
        elif path == "/api/trajectory":
            step = int(params.get("step", [0])[0])
            phase = params.get("phase", ["baseline"])[0]
            rollout = params.get("rollout", [""])[0]
            traj = load_trajectory(LOG_DIR, step, phase, rollout)
            self._json_response(traj)
        elif path == "/api/improve_log":
            step = int(params.get("step", [0])[0])
            phase = params.get("phase", ["baseline"])[0]
            log_path = os.path.join(LOG_DIR, f"step_{step}", phase, "improve.log")
            result = parse_improve_log(log_path)
            self._json_response(result)
        elif path == "/api/search":
            query = params.get("q", [""])[0]
            scope_str = params.get("scopes", ["actions,reasoning,beliefs,hypotheses"])[0]
            scopes = set(scope_str.split(","))
            results = search_logs(LOG_DIR, DATA, query, scopes)
            self._json_response(results)
        else:
            self.send_error(404)

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
<title>Explore Log Viewer</title>
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
.app { display: flex; height: 100vh; }

/* Sidebar */
.sidebar {
  width: 230px; min-width: 230px; background: var(--surface);
  border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow-y: auto;
}
.sidebar h2 {
  padding: 16px; font-size: 14px; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border);
}
.sidebar-item {
  padding: 8px 16px; cursor: pointer; font-size: 14px;
  border-left: 3px solid transparent; transition: all 0.15s;
  display: flex; align-items: center; gap: 8px;
}
.sidebar-item:hover { background: var(--surface2); }
.sidebar-item.active {
  background: var(--surface2); border-left-color: var(--accent); color: var(--accent);
}
.sidebar-item .badge {
  margin-left: auto; background: var(--surface2); padding: 2px 6px;
  border-radius: 10px; font-size: 11px; color: var(--text-muted);
}
.sidebar-item.active .badge { background: rgba(88,166,255,0.15); color: var(--accent); }
.sidebar-section {
  padding: 12px 16px 4px; font-size: 11px; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.5px;
}

/* Main */
.main { flex: 1; overflow-y: auto; padding: 24px; }
.main h1 { font-size: 22px; margin-bottom: 20px; }

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
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
.rollout-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.rollout-table th {
  text-align: left; padding: 8px 12px; background: var(--surface2);
  border-bottom: 1px solid var(--border); font-weight: 600;
  color: var(--text-muted); font-size: 11px; text-transform: uppercase;
}
.rollout-table td {
  padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top;
}
.rollout-table tr:hover { background: var(--surface2); }
.rollout-table tr.clickable { cursor: pointer; }

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

/* Hypothesis */
.hyp-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 12px 16px; margin-bottom: 8px; font-size: 13px;
}
.hyp-idx { font-weight: 700; color: var(--accent); margin-right: 8px; }

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

/* Phase tabs */
.phase-tabs { display: flex; gap: 0; margin-bottom: 16px; }
.phase-tab {
  padding: 8px 16px; cursor: pointer; font-size: 13px; color: var(--text-muted);
  background: var(--surface); border: 1px solid var(--border); transition: all 0.15s;
}
.phase-tab:first-child { border-radius: 6px 0 0 6px; }
.phase-tab:last-child { border-radius: 0 6px 6px 0; }
.phase-tab.active { background: var(--accent); color: var(--bg); border-color: var(--accent); }

/* Buttons */
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

/* Nav buttons */
.step-nav { display: flex; gap: 8px; margin-bottom: 16px; align-items: center; }
.step-nav-btn {
  background: var(--surface2); border: 1px solid var(--border); color: var(--text);
  padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px;
}
.step-nav-btn:hover { background: var(--border); }
.step-nav-btn:disabled { opacity: 0.3; cursor: not-allowed; }

/* Hypothesis stats in explore table */
.hyp-stats-row { background: var(--bg); }
.hyp-stats-row td { font-size: 12px; color: var(--text-muted); padding: 4px 12px; }

/* Improve log panels */
.improve-panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; margin-bottom: 8px;
}
.improve-panel-header {
  padding: 10px 16px; cursor: pointer; user-select: none; font-size: 13px;
  display: flex; align-items: center; gap: 10px;
  border-bottom: 1px solid transparent;
}
.improve-panel-header:hover { background: var(--surface2); }
.improve-panel-header .ip-toggle { color: var(--text-muted); font-size: 12px; margin-left: auto; }
.improve-panel-header .ip-label {
  font-weight: 600; font-size: 12px; padding: 2px 8px; border-radius: 4px;
}
.ip-label.prompt { background: rgba(88,166,255,0.15); color: var(--accent); }
.ip-label.response { background: rgba(63,185,80,0.15); color: var(--accent2); }
.improve-panel-body { display: none; padding: 12px 16px; border-top: 1px solid var(--border); }
.improve-panel-body.open { display: block; }
.improve-panel-body pre { max-height: 600px; }

/* Search */
.search-box {
  padding: 12px 16px; border-bottom: 1px solid var(--border);
}
.search-box input {
  width: 100%; padding: 6px 10px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 4px; color: var(--text); font-size: 13px; outline: none;
}
.search-box input:focus { border-color: var(--accent); }
.search-box input::placeholder { color: var(--text-muted); }
.search-scopes {
  display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
}
.search-scopes label {
  font-size: 11px; color: var(--text-muted); display: flex; align-items: center; gap: 3px; cursor: pointer;
}
.search-scopes input[type="checkbox"] { accent-color: var(--accent); width: 13px; height: 13px; }
.search-info {
  padding: 8px 16px; font-size: 11px; color: var(--text-muted); border-bottom: 1px solid var(--border);
}
.search-result {
  padding: 10px 16px; border-bottom: 1px solid var(--border); cursor: pointer;
  font-size: 13px; transition: background 0.1s;
}
.search-result:hover { background: var(--surface2); }
.search-result-header {
  display: flex; align-items: center; gap: 6px; margin-bottom: 4px; flex-wrap: wrap;
}
.sr-badge {
  font-size: 10px; padding: 1px 6px; border-radius: 3px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.3px;
}
.sr-badge.actions { background: rgba(88,166,255,0.15); color: var(--accent); }
.sr-badge.reasoning { background: rgba(210,153,34,0.15); color: var(--accent3); }
.sr-badge.beliefs { background: rgba(63,185,80,0.15); color: var(--accent2); }
.sr-badge.hypotheses { background: rgba(248,81,73,0.15); color: var(--danger); }
.sr-loc { font-size: 11px; color: var(--text-muted); }
.search-result-context {
  font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);
  line-height: 1.5; white-space: pre-wrap; word-break: break-word;
  max-height: 60px; overflow: hidden;
}
.search-result-context mark {
  background: rgba(88,166,255,0.25); color: var(--text); border-radius: 2px;
  padding: 0 1px;
}
</style>
</head>
<body>
<div class="app">
  <nav class="sidebar" id="sidebar">
    <h2>Explore Viewer</h2>
    <div class="search-box">
      <input type="text" id="search-input" placeholder="Search logs..." onkeydown="if(event.key==='Enter')doSearch()">
      <div class="search-scopes">
        <label><input type="checkbox" id="sc-actions" checked> Actions</label>
        <label><input type="checkbox" id="sc-reasoning" checked> Reasoning</label>
        <label><input type="checkbox" id="sc-beliefs" checked> Beliefs</label>
        <label><input type="checkbox" id="sc-hypotheses" checked> Hypotheses</label>
      </div>
    </div>
    <div class="sidebar-section">Overview</div>
    <div class="sidebar-item active" data-view="meta" onclick="showView('meta')">Meta Statistics</div>
    <div class="sidebar-item" data-view="config" onclick="showView('config')">Config</div>
    <div class="sidebar-section">Steps</div>
    <div id="step-list"></div>
  </nav>
  <div class="main" id="main-content">
    <div class="loading">Loading data...</div>
  </div>
</div>

<script>
let DATA = null;
let META = null;
let currentView = 'meta';
let currentStep = null;
let stepPhase = 'input';
const trajCache = {};

async function init() {
  const resp = await fetch('/api/data');
  const json = await resp.json();
  DATA = json.data;
  META = json.meta;
  buildSidebar();
  renderMeta();
}

function buildSidebar() {
  const stepList = document.getElementById('step-list');
  stepList.innerHTML = '';
  DATA.steps.forEach((step, i) => {
    const el = document.createElement('div');
    el.className = 'sidebar-item';
    el.dataset.view = 'step';
    el.dataset.step = i;
    const numHyp = step.explore.rollouts ? step.explore.rollouts.length : 0;
    el.innerHTML = `Step ${step.index} <span class="badge">${numHyp}h</span>`;
    el.onclick = () => showStep(i);
    stepList.appendChild(el);
  });
}

function showView(view) {
  currentView = view; currentStep = null; updateSidebar();
  if (view === 'meta') renderMeta();
  else if (view === 'config') renderConfig();
}

function showStep(idx) {
  currentView = 'step'; currentStep = idx; updateSidebar(); renderStep(idx);
}

function updateSidebar() {
  document.querySelectorAll('.sidebar-item').forEach(el => {
    el.classList.remove('active');
    if (currentView === 'step' && el.dataset.view === 'step' && parseInt(el.dataset.step) === currentStep)
      el.classList.add('active');
    else if (el.dataset.view === currentView && currentView !== 'step')
      el.classList.add('active');
  });
}

function esc(s) {
  if (s == null) return '';
  const div = document.createElement('div');
  div.textContent = String(s);
  return div.innerHTML;
}

// ============ META ============
function renderMeta() {
  const mc = document.getElementById('main-content');
  if (!META || META.length === 0) { mc.innerHTML = '<h1>No data</h1>'; return; }
  const totalSteps = META.length;
  const totalCost = META[META.length - 1].cumulative_cost;
  const maxSteps = Math.max(...META.map(s => Math.max(s.baseline_avg_steps, s.explore_avg_steps)), 1);
  const maxCost = Math.max(...META.map(s => s.step_total_cost), 0.001);

  let html = '<h1>Meta Statistics</h1>';

  html += `<div class="stats-grid">
    <div class="stat-card"><div class="stat-label">Total Steps</div><div class="stat-value blue">${totalSteps}</div></div>
    <div class="stat-card"><div class="stat-label">Total Cost</div><div class="stat-value yellow">$${totalCost.toFixed(2)}</div></div>
    <div class="stat-card"><div class="stat-label">Avg Baseline Steps (last)</div><div class="stat-value green">${META[META.length-1].baseline_avg_steps.toFixed(1)}</div></div>
    <div class="stat-card"><div class="stat-label">Avg Explore Steps (last)</div><div class="stat-value blue">${META[META.length-1].explore_avg_steps.toFixed(1)}</div></div>
  </div>`;

  // Steps chart
  html += chartHtml('Average Steps per Rollout', META, maxSteps,
    s => s.baseline_avg_steps, s => s.explore_avg_steps,
    (b,e) => `${b.toFixed(0)} / ${e.toFixed(0)}`,
    'blue', 'green', 'Baseline', 'Explore');

  // Cost chart
  html += chartHtml('Cost per Step', META, maxCost,
    s => s.baseline_total_cost, s => s.explore_total_cost,
    (b,e) => `$${(b+e).toFixed(3)}`,
    'yellow', 'red', 'Baseline Cost', 'Explore Cost');

  // Cumulative cost
  const maxCum = Math.max(...META.map(s => s.cumulative_cost), 0.001);
  html += `<div class="chart-container"><div class="chart-title">Cumulative Cost</div><div class="bar-chart">`;
  META.forEach(s => {
    const h = (s.cumulative_cost / maxCum) * 100;
    html += `<div class="bar-group">
      <div class="bar-value">$${s.cumulative_cost.toFixed(2)}</div>
      <div style="display:flex;gap:2px;align-items:flex-end;height:100%;width:100%;justify-content:center;">
        <div class="bar yellow" style="height:${h}%;max-width:24px;"></div>
      </div>
      <div class="bar-label">${s.step}</div>
    </div>`;
  });
  html += `</div></div>`;

  // Progression (if any)
  if (META.some(s => s.baseline_avg_prog > 0 || s.explore_avg_prog > 0)) {
    const maxP = Math.max(...META.map(s => Math.max(s.baseline_avg_prog, s.explore_avg_prog)), 0.01);
    html += chartHtml('Average Progression', META, maxP,
      s => s.baseline_avg_prog, s => s.explore_avg_prog,
      (b,e) => `${(b*100).toFixed(0)}/${(e*100).toFixed(0)}%`,
      'blue', 'green', 'Baseline', 'Explore');
  }

  mc.innerHTML = html;
}

function chartHtml(title, data, maxVal, getA, getB, fmtLabel, colorA, colorB, legendA, legendB) {
  let html = `<div class="chart-container"><div class="chart-title">${title}</div><div class="bar-chart">`;
  data.forEach(s => {
    const a = getA(s), b = getB(s);
    const ah = (a / maxVal) * 100, bh = (b / maxVal) * 100;
    html += `<div class="bar-group">
      <div class="bar-value">${fmtLabel(a, b)}</div>
      <div style="display:flex;gap:2px;align-items:flex-end;height:100%;width:100%;justify-content:center;">
        <div class="bar ${colorA}" style="height:${ah}%;max-width:14px;" title="${legendA}: ${a.toFixed(4)}"></div>
        <div class="bar ${colorB}" style="height:${bh}%;max-width:14px;" title="${legendB}: ${b.toFixed(4)}"></div>
      </div>
      <div class="bar-label">${s.step}</div>
    </div>`;
  });
  html += `</div><div class="chart-legend">
    <div class="legend-item"><div class="legend-dot" style="background:var(--${colorA})"></div>${legendA}</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--${colorB})"></div>${legendB}</div>
  </div></div>`;
  return html;
}

// ============ CONFIG ============
function renderConfig() {
  document.getElementById('main-content').innerHTML =
    `<h1>Configuration</h1><div class="card"><div class="card-body"><pre>${esc(DATA.config)}</pre></div></div>`;
}

// ============ STEP ============
function renderStep(idx) {
  const step = DATA.steps[idx];
  const mc = document.getElementById('main-content');
  const totalSteps = DATA.steps.length;

  let html = `<div class="step-nav">
    <button class="step-nav-btn" onclick="showStep(${idx-1})" ${idx===0?'disabled':''}>&#8592; Prev</button>
    <h1 style="margin:0;flex:1">Step ${step.index}</h1>
    <button class="step-nav-btn" onclick="showStep(${idx+1})" ${idx>=totalSteps-1?'disabled':''}>Next &#8594;</button>
  </div>`;

  html += `<div class="phase-tabs">
    <div class="phase-tab ${stepPhase==='input'?'active':''}" onclick="setPhase('input',${idx})">Input State</div>
    <div class="phase-tab ${stepPhase==='baseline'?'active':''}" onclick="setPhase('baseline',${idx})">Baseline</div>
    <div class="phase-tab ${stepPhase==='explore'?'active':''}" onclick="setPhase('explore',${idx})">Explore</div>
  </div>`;

  if (stepPhase === 'input') html += renderInputPhase(step);
  else if (stepPhase === 'baseline') html += renderPhaseView(step, 'baseline');
  else html += renderPhaseView(step, 'explore');

  mc.innerHTML = html;
}

function setPhase(phase, idx) { stepPhase = phase; renderStep(idx); }

function renderInputPhase(step) {
  let html = '';
  html += collapsible('Input Beliefs', `<pre>${esc(step.input_beliefs || '(empty - initial step)')}</pre>`, true);
  html += collapsible('Input Perception Module', `<pre>${esc(step.input_perception || '(empty - initial step)')}</pre>`, true);
  return html;
}

function renderPhaseView(step, phaseKey) {
  const phase = step[phaseKey];
  let html = '';

  const stats = phase.rollout_stats || {};
  const entries = Object.entries(stats).sort((a,b) => {
    const ai = parseInt((a[0].match(/\\d+/) || [0])[0]);
    const bi = parseInt((b[0].match(/\\d+/) || [0])[0]);
    return ai - bi;
  });

  let totalRollouts = entries.length;
  let totalStepsSum = 0, totalCost = 0;
  entries.forEach(([k, v]) => {
    Object.values(v.summary || {}).forEach(env => {
      totalStepsSum += env.avg_steps || 0;
      totalCost += env.total_cost || 0;
    });
  });

  html += `<div class="stats-grid">
    <div class="stat-card"><div class="stat-label">Rollouts</div><div class="stat-value blue">${totalRollouts}</div></div>
    <div class="stat-card"><div class="stat-label">Avg Steps</div><div class="stat-value green">${totalRollouts > 0 ? (totalStepsSum/totalRollouts).toFixed(1) : '-'}</div></div>
    <div class="stat-card"><div class="stat-label">Total Cost</div><div class="stat-value yellow">$${totalCost.toFixed(4)}</div></div>
  </div>`;

  // Rollouts table
  const isExplore = phaseKey === 'explore';
  html += `<div class="card"><div class="card-header">Rollouts</div><div class="card-body">
    <table class="rollout-table">
    <tr><th>#</th><th>Name</th>${isExplore ? '<th>Hypothesis</th>' : ''}<th>Steps</th><th>Prog</th><th>Cost</th><th>End</th></tr>`;

  phase.rollouts.forEach((r, ri) => {
    const m = r.meta || {};
    const endReasons = {0:'timeout',1:'death',2:'success'};
    const endReason = endReasons[m.end_reason] || '?';
    const endColor = m.end_reason === 2 ? 'accent2' : m.end_reason === 1 ? 'danger' : 'accent3';
    const hypSnip = r.hypothesis ? (r.hypothesis.length > 80 ? r.hypothesis.slice(0,80)+'...' : r.hypothesis) : '';

    html += `<tr class="clickable" onclick="showRollout(${currentStep},'${phaseKey}',${ri})">
      <td>${ri}</td><td>${esc(r.name)}</td>
      ${isExplore ? `<td title="${esc(r.hypothesis)}">${esc(hypSnip)}</td>` : ''}
      <td>${m.num_steps || '-'}</td>
      <td>${((m.progression||0)*100).toFixed(0)}%</td>
      <td>$${(m.total_cost||0).toFixed(4)}</td>
      <td style="color:var(--${endColor})">${endReason}</td>
    </tr>`;
  });

  html += '</table></div></div>';

  html += collapsible('Updated Beliefs', `<pre>${esc(phase.beliefs)}</pre>`);
  html += collapsible('Updated Perception Module', `<pre>${esc(phase.perception)}</pre>`);

  if (phaseKey === 'baseline' && phase.hypotheses) {
    let hHtml = '';
    phase.hypotheses.forEach((h, i) => {
      hHtml += `<div class="hyp-card"><span class="hyp-idx">#${i+1}</span>${esc(h)}</div>`;
    });
    html += collapsible(`Generated Hypotheses (${phase.hypotheses.length})`, hHtml);
  }

  // Placeholder for improve log - loaded on demand
  html += `<div id="improve-log-container" data-step="${step.index}" data-phase="${phaseKey}">
    <div class="card"><div class="card-header" onclick="loadImproveLog(this)" style="cursor:pointer">
      Summarisation &amp; Improve Logs <span class="toggle" style="color:var(--text-muted);font-size:12px">&#9654; Click to load</span>
    </div><div class="card-body collapsed" id="improve-log-body"><div class="loading">Loading...</div></div></div>
  </div>`;

  return html;
}

// ============ ROLLOUT ============
async function showRollout(stepIdx, phaseKey, rolloutIdx) {
  const step = DATA.steps[stepIdx];
  const phase = step[phaseKey];
  const rollout = phase.rollouts[rolloutIdx];
  const mc = document.getElementById('main-content');

  let html = `<button class="back-btn" onclick="renderStep(${stepIdx})">&#8592; Back to Step ${step.index} / ${phaseKey}</button>`;
  html += `<h1>${esc(rollout.name)}</h1>`;

  if (rollout.hypothesis)
    html += `<div class="card"><div class="card-header">Hypothesis</div><div class="card-body"><pre>${esc(rollout.hypothesis)}</pre></div></div>`;

  if (rollout.meta) {
    const m = rollout.meta;
    const endReasons = {0:'timeout',1:'death',2:'success'};
    const endColor = m.end_reason === 2 ? 'green' : m.end_reason === 1 ? 'red' : 'yellow';
    html += `<div class="stats-grid">
      <div class="stat-card"><div class="stat-label">Task</div><div class="stat-value" style="font-size:16px">${esc(m.task)}</div></div>
      <div class="stat-card"><div class="stat-label">Steps</div><div class="stat-value blue">${m.num_steps}</div></div>
      <div class="stat-card"><div class="stat-label">Progression</div><div class="stat-value green">${(m.progression*100).toFixed(0)}%</div></div>
      <div class="stat-card"><div class="stat-label">End Reason</div><div class="stat-value ${endColor}">${endReasons[m.end_reason]||'?'}</div></div>
      <div class="stat-card"><div class="stat-label">Cost</div><div class="stat-value yellow">$${m.total_cost.toFixed(4)}</div></div>
      <div class="stat-card"><div class="stat-label">Tokens</div><div class="stat-value" style="font-size:16px">${(m.input_tokens||0).toLocaleString()} in / ${(m.output_tokens||0).toLocaleString()} out</div></div>
    </div>`;

    if (m.action_frequency) {
      const af = Object.entries(m.action_frequency).sort((a,b) => b[1]-a[1]);
      html += `<div class="card"><div class="card-header" onclick="toggleCard(this)">Action Frequency <span class="toggle">&#9660;</span></div><div class="card-body">`;
      af.forEach(([a,c]) => {
        html += `<span style="display:inline-block;margin:2px 4px;padding:3px 10px;background:var(--surface2);border-radius:4px;font-size:13px;font-family:var(--font-mono)">${esc(a)}: ${c}</span>`;
      });
      html += '</div></div>';
    }
  }

  html += '<div id="trajectory-container"><div class="loading">Loading trajectory...</div></div>';
  mc.innerHTML = html;

  // Fetch trajectory on demand
  const cacheKey = `${step.index}_${phaseKey}_${rollout.name}`;
  let traj = trajCache[cacheKey];
  if (!traj) {
    const resp = await fetch(`/api/trajectory?step=${step.index}&phase=${phaseKey}&rollout=${encodeURIComponent(rollout.name)}`);
    traj = await resp.json();
    trajCache[cacheKey] = traj;
  }

  const tc = document.getElementById('trajectory-container');
  if (!traj || traj.length === 0) {
    tc.innerHTML = '<div class="card"><div class="card-body">No trajectory data available.</div></div>';
    return;
  }

  let tHtml = `<div class="card"><div class="card-header">Trajectory (${traj.length} steps)</div><div class="card-body">`;
  traj.forEach((t, ti) => {
    const doneClass = t.done === 'True' ? (parseFloat(t.reward) > 0 ? 'success' : 'death') : '';
    tHtml += `<div class="traj-step">
      <div class="traj-step-header" onclick="toggleTrajStep(this)">
        <div class="traj-step-num ${doneClass}">${t.step}</div>
        <div class="traj-step-action">${esc(t.action)}</div>
        <div class="traj-step-reward">r=${t.reward}${t.done === 'True' ? ' (DONE)' : ''}</div>
      </div>
      <div class="traj-step-body">
        <div class="traj-section"><div class="traj-section-label">Observation</div><pre>${esc(t.observation)}</pre></div>
        <div class="traj-section"><div class="traj-section-label">Reasoning</div><pre>${esc(t.reasoning)}</pre></div>
      </div>
    </div>`;
  });
  tHtml += '</div></div>';
  tc.innerHTML = tHtml;
}

function toggleTrajStep(el) { el.nextElementSibling.classList.toggle('open'); }

function collapsible(title, content, startOpen) {
  const cls = startOpen ? '' : 'collapsed';
  const arrow = startOpen ? '&#9660;' : '&#9654;';
  return `<div class="card">
    <div class="card-header" onclick="toggleCard(this)">${title} <span class="toggle">${arrow}</span></div>
    <div class="card-body ${cls}">${content}</div>
  </div>`;
}

function toggleCard(header) {
  const body = header.nextElementSibling;
  const toggle = header.querySelector('.toggle');
  body.classList.toggle('collapsed');
  toggle.innerHTML = body.classList.contains('collapsed') ? '&#9654;' : '&#9660;';
}

// ============ IMPROVE LOG ============
const improveCache = {};

async function loadImproveLog(headerEl) {
  const body = headerEl.nextElementSibling;
  const toggle = headerEl.querySelector('.toggle');

  // If already loaded, just toggle
  if (body.dataset.loaded === 'true') {
    body.classList.toggle('collapsed');
    toggle.innerHTML = body.classList.contains('collapsed') ? '&#9654; Click to load' : '&#9660;';
    return;
  }

  body.classList.remove('collapsed');
  toggle.innerHTML = '&#9660;';

  const container = headerEl.closest('#improve-log-container') || headerEl.parentElement.parentElement;
  const step = container.dataset.step;
  const phase = container.dataset.phase;
  const cacheKey = `${step}_${phase}`;

  let data = improveCache[cacheKey];
  if (!data) {
    const resp = await fetch(`/api/improve_log?step=${step}&phase=${phase}`);
    data = await resp.json();
    improveCache[cacheKey] = data;
  }

  let html = '';

  // Trajectory summaries
  if (data.summaries && data.summaries.length > 0) {
    html += '<h3 style="font-size:14px;margin-bottom:12px;color:var(--text-muted)">Trajectory Summaries</h3>';
    data.summaries.forEach((s, i) => {
      const label = s.rollout || `Episode ${i+1}`;
      html += improvePanel(`${label} - Summary Prompt`, s.prompt, 'prompt');
      html += improvePanel(`${label} - Summary Response`, s.response, 'response');
    });
  }

  // Hypothesis summaries (explore phase)
  if (data.hypothesis_summaries && data.hypothesis_summaries.length > 0) {
    html += '<h3 style="font-size:14px;margin:20px 0 12px;color:var(--text-muted)">Hypothesis Summaries</h3>';
    data.hypothesis_summaries.forEach((s, i) => {
      const label = s.rollout || `Hypothesis ${i+1}`;
      html += improvePanel(`${label} - Hypothesis Prompt`, s.prompt, 'prompt');
      html += improvePanel(`${label} - Hypothesis Response`, s.response, 'response');
    });
  }

  // Improve step
  if (data.improve_prompt || data.improve_response) {
    html += '<h3 style="font-size:14px;margin:20px 0 12px;color:var(--text-muted)">Improve Step</h3>';
    if (data.improve_prompt)
      html += improvePanel('Improve Prompt', data.improve_prompt, 'prompt');
    if (data.improve_response)
      html += improvePanel('Improve Response', data.improve_response, 'response');
  }

  if (!html) html = '<div style="color:var(--text-muted);font-style:italic">No improve log data found.</div>';

  body.innerHTML = html;
  body.dataset.loaded = 'true';
}

function improvePanel(title, content, type) {
  if (!content) return '';
  const lines = content.split('\\n').length;
  const chars = content.length;
  const sizeHint = `${lines} lines, ${(chars/1024).toFixed(1)}KB`;
  return `<div class="improve-panel">
    <div class="improve-panel-header" onclick="toggleImprovePanel(this)">
      <span class="ip-label ${type}">${type === 'prompt' ? 'PROMPT' : 'RESPONSE'}</span>
      <span>${esc(title)}</span>
      <span style="font-size:11px;color:var(--text-muted)">(${sizeHint})</span>
      <span class="ip-toggle">&#9654;</span>
    </div>
    <div class="improve-panel-body"><pre>${esc(content)}</pre></div>
  </div>`;
}

function toggleImprovePanel(header) {
  const body = header.nextElementSibling;
  const toggle = header.querySelector('.ip-toggle');
  body.classList.toggle('open');
  toggle.innerHTML = body.classList.contains('open') ? '&#9660;' : '&#9654;';
}

// ============ SEARCH ============
let searchAbort = null;

async function doSearch() {
  const q = document.getElementById('search-input').value.trim();
  if (!q) return;

  const scopes = [];
  if (document.getElementById('sc-actions').checked) scopes.push('actions');
  if (document.getElementById('sc-reasoning').checked) scopes.push('reasoning');
  if (document.getElementById('sc-beliefs').checked) scopes.push('beliefs');
  if (document.getElementById('sc-hypotheses').checked) scopes.push('hypotheses');
  if (scopes.length === 0) return;

  currentView = 'search';
  currentStep = null;
  updateSidebar();

  const mc = document.getElementById('main-content');
  mc.innerHTML = '<h1>Search Results</h1><div class="loading">Searching...</div>';

  try {
    const resp = await fetch(`/api/search?q=${encodeURIComponent(q)}&scopes=${scopes.join(',')}`);
    const data = await resp.json();
    renderSearchResults(q, data.results, scopes, data.total, data.truncated);
  } catch(e) {
    mc.innerHTML = `<h1>Search Results</h1><div class="loading">Search failed: ${e.message}</div>`;
  }
}

function highlightMatch(text, query) {
  if (!text || !query) return esc(text);
  const escaped = esc(text);
  const qEsc = esc(query);
  // Case-insensitive highlight
  const re = new RegExp('(' + qEsc.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi');
  return escaped.replace(re, '<mark>$1</mark>');
}

function renderSearchResults(query, results, scopes, total, truncated) {
  const mc = document.getElementById('main-content');
  let html = `<h1>Search Results</h1>`;
  html += `<div style="margin-bottom:16px;font-size:13px;color:var(--text-muted)">`;
  html += `Found <strong style="color:var(--text)">${total}</strong> match${total!==1?'es':''} for `;
  html += `<strong style="color:var(--accent)">"${esc(query)}"</strong> in ${scopes.join(', ')}`;
  if (truncated) html += ` (showing first ${results.length})`;
  html += `</div>`;

  const shown = results;

  // Group by type for summary
  const counts = {};
  results.forEach(r => { counts[r.type] = (counts[r.type]||0)+1; });
  html += `<div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap">`;
  for (const [type, count] of Object.entries(counts)) {
    html += `<span class="sr-badge ${type}" style="font-size:12px;padding:4px 10px">${type}: ${count}</span>`;
  }
  html += `</div>`;

  shown.forEach((r, i) => {
    const loc = `Step ${r.step} / ${r.phase}` + (r.rollout ? ` / ${r.rollout}` : '') + (r.traj_step ? ` #${r.traj_step}` : '') + (r.line ? ` L${r.line}` : '');
    html += `<div class="search-result" onclick="navigateResult(${JSON.stringify(r).replace(/"/g, '&quot;')})">
      <div class="search-result-header">
        <span class="sr-badge ${r.type}">${r.type}</span>
        <span class="sr-loc">${esc(loc)}</span>
      </div>
      <div class="search-result-context">${highlightMatch(r.context, query)}</div>
    </div>`;
  });

  if (results.length === 0) {
    html += `<div style="padding:40px;text-align:center;color:var(--text-muted)">No results found.</div>`;
  }

  mc.innerHTML = html;
}

function navigateResult(r) {
  // Find the step index in DATA.steps
  const stepIdx = DATA.steps.findIndex(s => s.index === r.step);
  if (stepIdx < 0) return;

  if (r.type === 'beliefs') {
    stepPhase = r.phase;
    showStep(stepIdx);
  } else if (r.type === 'hypotheses' && !r.rollout) {
    stepPhase = r.phase;
    showStep(stepIdx);
  } else if (r.rollout) {
    // Navigate to the specific rollout
    const phase = DATA.steps[stepIdx][r.phase];
    const ri = phase.rollouts.findIndex(ro => ro.name === r.rollout);
    if (ri >= 0) {
      currentView = 'step';
      currentStep = stepIdx;
      stepPhase = r.phase;
      showRollout(stepIdx, r.phase, ri);
    } else {
      stepPhase = r.phase;
      showStep(stepIdx);
    }
  } else {
    showStep(stepIdx);
  }
}

init();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize explore.py logs")
    parser.add_argument("log_dir", help="Path to the explore log directory")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    global LOG_DIR, DATA, META

    LOG_DIR = os.path.abspath(args.log_dir)
    if not os.path.isdir(LOG_DIR):
        print(f"Error: {LOG_DIR} is not a directory")
        sys.exit(1)

    print(f"Loading logs from {LOG_DIR}...")
    DATA = load_log_dir_light(LOG_DIR)
    print(f"  Found {len(DATA['steps'])} steps")

    META = compute_meta_stats(DATA)
    print(f"  Computed meta statistics")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nServing at {url}")
    print("Press Ctrl+C to stop\n")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
