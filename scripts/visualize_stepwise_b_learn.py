#!/usr/bin/env python3
"""Visualize stepwise_b_learn.py logs with a local web server.

Usage:
    python visualize_stepwise_b_learn.py [log_dir] [--port PORT] [--open-browser]

Serves an interactive step-centric viewer for stepwise_b_learn.py log directories.
The log directory can be specified on the command line or entered in the browser.
Multiple tabs can view different log directories simultaneously.
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
from urllib.parse import urlparse, parse_qs, quote


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


def load_log_dir(log_dir):
    """Load the full step-centric log structure."""
    summary = read_json(os.path.join(log_dir, "summary.json"))
    config = read_file(os.path.join(log_dir, "config.yaml"))

    # Build a flat list of all steps across all episodes
    all_steps = []
    episodes = []

    episode_dirs = []
    for name in os.listdir(log_dir):
        if name.startswith("episode_") and os.path.isdir(os.path.join(log_dir, name)):
            try:
                idx = int(name.replace("episode_", ""))
                episode_dirs.append((idx, name))
            except ValueError:
                pass
    episode_dirs.sort()

    for ep_idx, ep_name in episode_dirs:
        ep_path = os.path.join(log_dir, ep_name)
        ep_log = read_json(os.path.join(ep_path, "episode_log.json"))

        # Also try old format
        if not ep_log:
            for f in os.listdir(ep_path):
                if f.endswith(".json") and "_run_" in f:
                    ep_log = read_json(os.path.join(ep_path, f))
                    if ep_log and "episode_return" in ep_log:
                        break

        ep_info = {
            "index": ep_idx,
            "path": ep_path,
            "log": ep_log or {},
            "input_beliefs": read_file(os.path.join(ep_path, "input_beliefs.txt")),
            "input_perception": read_file(os.path.join(ep_path, "input_perception.py")),
        }
        episodes.append(ep_info)

        # Scan step directories
        step_dirs = []
        for sname in os.listdir(ep_path):
            if sname.startswith("step_") and os.path.isdir(os.path.join(ep_path, sname)):
                try:
                    s_idx = int(sname.replace("step_", ""))
                    step_dirs.append((s_idx, sname))
                except ValueError:
                    pass
        step_dirs.sort()

        for s_idx, s_name in step_dirs:
            s_path = os.path.join(ep_path, s_name)
            step_log = read_json(os.path.join(s_path, "step_log.json")) or {}
            all_steps.append({
                "episode_idx": ep_idx,
                "step": s_idx,
                "global_step": step_log.get("global_step", len(all_steps)),
                "action": step_log.get("action", ""),
                "reward": step_log.get("reward", 0),
                "done": step_log.get("done", False),
                "episode_return_so_far": step_log.get("episode_return_so_far", 0),
                "agent_step_cost": step_log.get("agent_step_cost", 0),
                "extract_cost": step_log.get("extract_cost", 0),
                "improve_cost": step_log.get("improve_cost", 0),
                "experiment_cost": step_log.get("experiment_cost", 0),
                "step_total_cost": step_log.get("step_total_cost", 0),
                "num_qa_pairs": step_log.get("num_qa_pairs", 0),
                "num_moments": step_log.get("num_moments", 0),
                "num_experiments": step_log.get("num_experiments", 0),
                "has_beliefs": os.path.exists(os.path.join(s_path, "beliefs.txt")),
                "has_feedback": os.path.exists(os.path.join(s_path, "feedback_history.json")),
                "has_improve_log": os.path.exists(os.path.join(s_path, "improve.log")),
                "has_extraction_log": os.path.exists(os.path.join(s_path, "extraction_log.json")),
                "has_experiment_log": os.path.exists(os.path.join(s_path, "experiment_log.json")),
                "has_agent_messages": os.path.exists(os.path.join(s_path, "agent_messages.json")),
                "did_gen_experiments": step_log.get("did_gen_experiments", False),
                "active_experiment": step_log.get("active_experiment", None),
                "phase": step_log.get("phase", "complete"),
                "path": s_path,
            })

    total_cost = 0
    if summary and "steps" in summary and len(summary["steps"]) > 0:
        total_cost = sum(s.get("step_cost", 0) for s in summary["steps"])

    return {
        "config": config,
        "summary": summary,
        "episodes": [{"index": e["index"], "log": e["log"],
                       "input_beliefs": e["input_beliefs"],
                       "input_perception": e["input_perception"]} for e in episodes],
        "steps": all_steps,
        "total_cost": total_cost,
        "log_dir_name": os.path.basename(log_dir),
    }


def load_step_detail(log_dir, episode_idx, step_idx):
    """Load full detail for a specific step."""
    s_path = os.path.join(log_dir, f"episode_{episode_idx}", f"step_{step_idx:03d}")
    if not os.path.isdir(s_path):
        return {"error": f"Step dir not found: {s_path}"}
    return {
        "beliefs": read_file(os.path.join(s_path, "beliefs.txt")),
        "perception": read_file(os.path.join(s_path, "perception.py")),
        "qa_pairs": read_json(os.path.join(s_path, "qa_pairs.json")) or [],
        "critical_moments": read_json(os.path.join(s_path, "critical_moments.json")) or [],
        "experiments": read_json(os.path.join(s_path, "experiments.json")) or [],
        "feedback_history": read_json(os.path.join(s_path, "feedback_history.json")) or [],
        "extraction_log": read_json(os.path.join(s_path, "extraction_log.json")) or {},
        "experiment_log": read_json(os.path.join(s_path, "experiment_log.json")) or {},
        "agent_messages": read_json(os.path.join(s_path, "agent_messages.json")) or [],
        "improve_log": read_file(os.path.join(s_path, "improve.log")),
        "step_log": read_json(os.path.join(s_path, "step_log.json")) or {},
    }


def load_trajectory(log_dir, episode_idx):
    """Load trajectory CSV for an episode."""
    ep_dir = os.path.join(log_dir, f"episode_{episode_idx}")
    csv_file = os.path.join(ep_dir, "trajectory.csv")
    if not os.path.isfile(csv_file):
        # Fallback: find any CSV
        for f in os.listdir(ep_dir):
            if f.endswith(".csv"):
                csv_file = os.path.join(ep_dir, f)
                break
        else:
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
            "auxiliary_observation": row.get("Auxiliary_Observation", ""),
            "reward": row.get("Reward", ""),
            "done": row.get("Done", ""),
        })
    return steps


def load_combined_trajectory(log_dir):
    """Load trajectories from all episodes, with episode boundary markers."""
    episode_dirs = []
    for name in os.listdir(log_dir):
        if name.startswith("episode_") and os.path.isdir(os.path.join(log_dir, name)):
            try:
                idx = int(name.replace("episode_", ""))
                episode_dirs.append((idx, name))
            except ValueError:
                pass
    episode_dirs.sort()

    combined = []
    global_step = 0
    for ep_idx, ep_name in episode_dirs:
        if combined:
            combined.append({
                "episode_boundary": True,
                "episode_idx": ep_idx,
                "global_step": global_step,
            })
        ep_traj = load_trajectory(log_dir, ep_idx)
        for t in ep_traj:
            t["episode_idx"] = ep_idx
            t["global_step"] = global_step
            global_step += 1
            combined.append(t)
    return combined


def load_experiment_timeline(log_dir):
    """Collect experiment history across all episodes and steps."""
    episode_dirs = []
    for name in os.listdir(log_dir):
        if name.startswith("episode_") and os.path.isdir(os.path.join(log_dir, name)):
            try:
                idx = int(name.replace("episode_", ""))
                episode_dirs.append((idx, name))
            except ValueError:
                pass
    episode_dirs.sort()

    timeline = []
    seen_experiments = set()
    for ep_idx, ep_name in episode_dirs:
        ep_path = os.path.join(log_dir, ep_name)
        step_dirs = []
        for sname in os.listdir(ep_path):
            if sname.startswith("step_") and os.path.isdir(os.path.join(ep_path, sname)):
                try:
                    s_idx = int(sname.replace("step_", ""))
                    step_dirs.append((s_idx, sname))
                except ValueError:
                    pass
        step_dirs.sort()

        for s_idx, s_name in step_dirs:
            s_path = os.path.join(ep_path, s_name)
            exp_log = read_json(os.path.join(s_path, "experiment_log.json"))
            step_log = read_json(os.path.join(s_path, "step_log.json")) or {}
            if not exp_log:
                continue
            new_exps = exp_log.get("new_experiments", [])
            old_exps = exp_log.get("old_experiments", [])
            first_seen = []
            for e in new_exps:
                if e not in seen_experiments:
                    seen_experiments.add(e)
                    first_seen.append(e)
            timeline.append({
                "episode_idx": ep_idx,
                "step": s_idx,
                "global_step": step_log.get("global_step", s_idx),
                "old_experiments": old_exps,
                "new_experiments": new_exps,
                "first_seen": first_seen,
                "active": exp_log.get("new_experiments", [None])[0] if new_exps else None,
                "cumulative_count": len(seen_experiments),
            })
    return timeline


# ============= HTTP Server =============


def resolve_log_dir(raw_path):
    """Resolve and validate a log directory path.

    If the directory has no episode_* subdirs, use the last subdirectory.
    Returns the resolved absolute path or raises ValueError.
    """
    log_dir = os.path.abspath(raw_path)
    if not os.path.isdir(log_dir):
        raise ValueError(f"Not a directory: {log_dir}")

    has_episodes = any(
        name.startswith("episode_") and os.path.isdir(os.path.join(log_dir, name))
        for name in os.listdir(log_dir)
    )
    if not has_episodes:
        subdirs = sorted(
            name for name in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, name))
        )
        if not subdirs:
            raise ValueError(f"No subdirectories in: {log_dir}")
        log_dir = os.path.join(log_dir, subdirs[-1])

    return log_dir


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _get_log_dir(self, params):
        """Extract and resolve log_dir from query params."""
        raw = params.get("log_dir", [None])[0]
        if not raw:
            raise ValueError("Missing log_dir parameter")
        return resolve_log_dir(raw)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path == "/":
                self._serve_html()
            elif path == "/api/data":
                log_dir = self._get_log_dir(params)
                data = load_log_dir(log_dir)
                self._json_response(data)
            elif path == "/api/step_detail":
                log_dir = self._get_log_dir(params)
                ep = int(params.get("episode", [0])[0])
                step = int(params.get("step", [0])[0])
                self._json_response(load_step_detail(log_dir, ep, step))
            elif path == "/api/trajectory":
                log_dir = self._get_log_dir(params)
                ep = int(params.get("episode", [0])[0])
                self._json_response(load_trajectory(log_dir, ep))
            elif path == "/api/combined_trajectory":
                log_dir = self._get_log_dir(params)
                self._json_response(load_combined_trajectory(log_dir))
            elif path == "/api/experiment_timeline":
                log_dir = self._get_log_dir(params)
                self._json_response(load_experiment_timeline(log_dir))
            else:
                self.send_error(404)
        except Exception as e:
            err = f"[visualize] request failed for {self.path}: {e}\n{traceback.format_exc()}"
            print(err, file=sys.stderr, flush=True)
            try:
                self._json_response({"error": str(e)})
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
<title>Stepwise B-Learn Viewer</title>
<meta id="log-dir-meta">
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --surface2: #21262d; --border: #30363d;
  --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff; --accent2: #3fb950;
  --accent3: #d29922; --danger: #f85149; --purple: #bc8cff;
  --font-mono: 'SF Mono', 'Fira Code', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }
.app { display: flex; flex-direction: column; height: 100vh; }
.topbar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; gap: 24px; flex-shrink: 0; }
.topbar-title { font-size: 16px; font-weight: 700; }
.topbar-info { font-size: 13px; color: var(--text-muted); }
.topbar-cost { font-size: 13px; color: var(--accent3); font-weight: 600; }
.topbar-btn { background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; margin-left: auto; }
.topbar-btn:hover { background: var(--accent); color: var(--bg); }
.content-area { display: flex; flex: 1; overflow: hidden; }
.sidebar { width: 280px; min-width: 280px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow-y: auto; }
.sidebar h2 { padding: 12px 16px; font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }
.step-item { padding: 6px 12px; cursor: pointer; font-size: 12px; border-left: 3px solid transparent; transition: all 0.15s; display: flex; align-items: center; gap: 8px; }
.step-item:hover { background: var(--surface2); }
.step-item.active { background: var(--surface2); border-left-color: var(--accent); color: var(--accent); }
.step-item.ep-boundary { border-top: 1px solid var(--border); margin-top: 4px; padding-top: 10px; }
.step-item .gs { color: var(--text-muted); font-family: var(--font-mono); font-size: 11px; min-width: 30px; }
.step-item .act { font-family: var(--font-mono); font-size: 11px; background: var(--surface2); padding: 1px 6px; border-radius: 3px; max-width: 80px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.step-item .rw { font-family: var(--font-mono); font-size: 11px; margin-left: auto; }
.step-item .rw.pos { color: var(--accent2); }
.step-item .rw.neg { color: var(--danger); }
.step-item .rw.zero { color: var(--text-muted); }
.step-item .learn-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--purple); flex-shrink: 0; }
.step-item .done-marker { color: var(--danger); font-weight: 700; font-size: 11px; }
.step-item .phase-badge { font-size: 10px; padding: 1px 5px; border-radius: 3px; background: rgba(210,153,34,0.2); color: var(--accent3); font-weight: 600; white-space: nowrap; }
.ep-header { padding: 8px 16px; font-size: 12px; font-weight: 600; color: var(--accent); background: var(--bg); border-bottom: 1px solid var(--border); }
.main { flex: 1; overflow-y: auto; padding: 24px; }
pre { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 12px; font-family: var(--font-mono); font-size: 12px; line-height: 1.6; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; max-height: 500px; overflow-y: auto; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 20px; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
.stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
.stat-value { font-size: 20px; font-weight: 700; }
.stat-value.green { color: var(--accent2); } .stat-value.blue { color: var(--accent); }
.stat-value.yellow { color: var(--accent3); } .stat-value.red { color: var(--danger); }
.stat-value.purple { color: var(--purple); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 16px; }
.card-header { padding: 10px 16px; border-bottom: 1px solid var(--border); font-size: 13px; font-weight: 600; display: flex; align-items: center; justify-content: space-between; cursor: pointer; user-select: none; }
.card-header:hover { background: var(--surface2); }
.card-body { padding: 12px 16px; }
.card-body.collapsed { display: none; }
.tabs { display: flex; gap: 0; margin-bottom: 16px; flex-wrap: wrap; }
.tab { padding: 6px 14px; cursor: pointer; font-size: 12px; color: var(--text-muted); background: var(--surface); border: 1px solid var(--border); transition: all 0.15s; }
.tab:first-child { border-radius: 6px 0 0 6px; } .tab:last-child { border-radius: 0 6px 6px 0; }
.tab.active { background: var(--accent); color: var(--bg); border-color: var(--accent); }
.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.data-table th { text-align: left; padding: 6px 10px; background: var(--surface2); border-bottom: 1px solid var(--border); font-weight: 600; color: var(--text-muted); font-size: 11px; text-transform: uppercase; }
.data-table td { padding: 6px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
.data-table tr:hover { background: var(--surface2); }
.side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.side-by-side > div > h3 { font-size: 12px; color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.verdict { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
.verdict-correct { background: rgba(63,185,80,0.15); color: var(--accent2); }
.verdict-incorrect { background: rgba(248,81,73,0.15); color: var(--danger); }
.verdict-inconclusive { background: rgba(210,153,34,0.15); color: var(--accent3); }
.extraction-section { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 8px; }
.extraction-header { padding: 8px 14px; cursor: pointer; user-select: none; font-size: 12px; display: flex; align-items: center; gap: 8px; }
.extraction-header:hover { background: var(--surface2); }
.extraction-body { display: none; padding: 10px 14px; border-top: 1px solid var(--border); }
.extraction-body.open { display: block; }
.traj-step { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 6px; }
.traj-step-header { padding: 8px 12px; display: flex; align-items: center; gap: 10px; cursor: pointer; user-select: none; font-size: 12px; }
.traj-step-header:hover { background: var(--surface2); }
.traj-step-num { background: var(--accent); color: var(--bg); width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; flex-shrink: 0; }
.traj-step-num.death { background: var(--danger); } .traj-step-num.success { background: var(--accent2); }
.traj-step-action { font-family: var(--font-mono); background: var(--surface2); padding: 2px 6px; border-radius: 4px; font-size: 11px; }
.traj-step-reward { margin-left: auto; font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); }
.traj-step-body { display: none; padding: 0 12px 12px; }
.traj-step-body.open { display: block; }
.traj-section { margin-top: 8px; }
.traj-section-label { font-size: 11px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 2px; }
.loading { color: var(--text-muted); font-style: italic; padding: 20px; text-align: center; }
.msg-bubble { padding: 10px 14px; margin-bottom: 8px; border-radius: 8px; font-size: 13px; max-width: 90%; }
.msg-user { background: var(--surface2); border: 1px solid var(--border); align-self: flex-start; }
.msg-assistant { background: rgba(88,166,255,0.1); border: 1px solid rgba(88,166,255,0.3); align-self: flex-end; margin-left: auto; }
.msg-role { font-size: 10px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; font-weight: 600; }
.chart-bar-group { display: flex; align-items: flex-end; gap: 2px; }
.landing { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
.landing h1 { font-size: 28px; font-weight: 700; }
.landing p { color: var(--text-muted); font-size: 14px; max-width: 500px; text-align: center; }
.landing-form { display: flex; gap: 8px; width: 100%; max-width: 600px; }
.landing-form input { flex: 1; background: var(--surface); border: 1px solid var(--border); color: var(--text); padding: 10px 14px; border-radius: 6px; font-size: 14px; font-family: var(--font-mono); }
.landing-form input:focus { outline: none; border-color: var(--accent); }
.landing-form button { background: var(--accent); color: var(--bg); border: none; padding: 10px 20px; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; white-space: nowrap; }
.landing-form button:hover { opacity: 0.9; }
.landing-error { color: var(--danger); font-size: 13px; }
.topbar-dir-input { background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 4px 10px; border-radius: 4px; font-size: 12px; font-family: var(--font-mono); width: 300px; }
.topbar-dir-input:focus { outline: none; border-color: var(--accent); }
.topbar-new-tab { background: var(--surface2); border: 1px solid var(--border); color: var(--text-muted); padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.topbar-new-tab:hover { background: var(--accent); color: var(--bg); }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 4px; }
</style>
</head>
<body>

<div class="landing" id="landing-page">
  <h1>Stepwise B-Learn Viewer</h1>
  <p>Enter the path to a log directory to visualize. Each browser tab can view a different directory.</p>
  <div class="landing-form">
    <input type="text" id="landing-input" placeholder="/path/to/logs/run_dir" autofocus>
    <button onclick="loadFromLanding()">Load</button>
  </div>
  <div class="landing-error" id="landing-error"></div>
</div>

<div class="app" id="viewer-app" style="display:none">
  <div class="topbar">
    <div class="topbar-title">Stepwise B-Learn Viewer</div>
    <div class="topbar-info" id="topbar-dir"></div>
    <div class="topbar-cost" id="topbar-cost"></div>
    <button class="topbar-new-tab" onclick="openNewTab()" title="Open a new tab to view a different log directory">+ New Tab</button>
    <button class="topbar-btn" onclick="reloadData()" title="Reload data from disk">Reload</button>
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
let selectedStepIdx = null;
let currentTab = 'overview';
const detailCache = {};
const trajCache = {};
let combinedTrajCache = null;
let expTimelineCache = null;

// Get log_dir from URL params
const urlParams = new URLSearchParams(window.location.search);
const LOG_DIR = urlParams.get('log_dir');

function apiUrl(path, extraParams) {
  const p = new URLSearchParams({ log_dir: LOG_DIR, ...extraParams });
  return path + '?' + p.toString();
}

function loadFromLanding() {
  const input = document.getElementById('landing-input');
  const val = input.value.trim();
  if (!val) { document.getElementById('landing-error').textContent = 'Please enter a path'; return; }
  window.location.href = '/?log_dir=' + encodeURIComponent(val);
}

function openNewTab() {
  window.open('/', '_blank');
}

// Handle enter key on landing input
document.getElementById('landing-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') loadFromLanding();
});

async function init() {
  if (!LOG_DIR) {
    document.getElementById('landing-page').style.display = '';
    document.getElementById('viewer-app').style.display = 'none';
    return;
  }
  document.getElementById('landing-page').style.display = 'none';
  document.getElementById('viewer-app').style.display = '';

  try {
    const resp = await fetch(apiUrl('/api/data'));
    DATA = await resp.json();
    if (DATA.error) {
      document.getElementById('main-content').innerHTML = '<pre style="color:var(--danger)">' + esc(DATA.error) + '</pre>';
      return;
    }
    document.getElementById('topbar-dir').textContent = DATA.log_dir_name;
    document.title = DATA.log_dir_name + ' — Stepwise B-Learn Viewer';
    document.getElementById('topbar-cost').textContent = 'Total cost: $' + (DATA.total_cost || 0).toFixed(4);
    buildSidebar();
    if (DATA.steps.length > 0) showStep(selectedStepIdx != null ? selectedStepIdx : 0);
    else document.getElementById('main-content').innerHTML = '<h1>No steps found</h1>';
  } catch (e) {
    document.getElementById('main-content').innerHTML = '<pre>' + esc(e) + '</pre>';
  }
}

async function reloadData() {
  // Clear caches
  for (const k of Object.keys(detailCache)) delete detailCache[k];
  for (const k of Object.keys(trajCache)) delete trajCache[k];
  combinedTrajCache = null;
  expTimelineCache = null;
  await init();
}

function buildSidebar() {
  const list = document.getElementById('step-list');
  list.innerHTML = '';
  let lastEp = -1;
  DATA.steps.forEach((s, i) => {
    if (s.episode_idx !== lastEp) {
      lastEp = s.episode_idx;
      const epMeta = DATA.episodes.find(e => e.index === s.episode_idx);
      const epLog = epMeta ? epMeta.log : {};
      const ret = epLog.episode_return != null ? ' (r=' + Number(epLog.episode_return).toFixed(1) + ')' : '';
      const hdr = document.createElement('div');
      hdr.className = 'ep-header';
      hdr.textContent = 'Episode ' + s.episode_idx + ret;
      list.appendChild(hdr);
    }
    const el = document.createElement('div');
    el.className = 'step-item' + (s.step === 0 ? ' ep-boundary' : '');
    el.dataset.idx = i;
    const rVal = Number(s.reward);
    const rCls = rVal > 0 ? 'pos' : rVal < 0 ? 'neg' : 'zero';
    const learnDot = (s.improve_cost > 0 || s.extract_cost > 0 || s.experiment_cost > 0) ? '<span class="learn-dot" title="learning occurred"></span>' : '';
    const doneMark = s.done ? '<span class="done-marker">END</span>' : '';
    const isInProgress = s.phase && s.phase !== 'complete';
    const phaseLabels = { started: 'starting', acting: 'acting', extracting: 'extracting', improving: 'improving' };
    const phaseBadge = isInProgress ? '<span class="phase-badge" title="step in progress">' + (phaseLabels[s.phase] || s.phase) + '</span>' : '';
    const actionText = s.action || (isInProgress ? '...' : '');
    el.innerHTML = '<span class="gs">g' + s.global_step + '</span>' +
      '<span class="act" title="' + esc(actionText) + '">' + esc(actionText) + '</span>' +
      learnDot + doneMark + phaseBadge +
      '<span class="rw ' + rCls + '">' + (isInProgress && !s.action ? '' : rVal.toFixed(2)) + '</span>';
    el.onclick = () => { currentTab = 'overview'; showStep(i); };
    list.appendChild(el);
  });
}

function showStep(idx) {
  selectedStepIdx = idx;
  document.querySelectorAll('.step-item').forEach(el => {
    el.classList.toggle('active', parseInt(el.dataset.idx) === idx);
  });
  renderStep(idx);
}

function esc(s) { if (s == null) return ''; const d = document.createElement('div'); d.textContent = String(s); return d.innerHTML; }
function toggleCard(h) { const b = h.nextElementSibling; const t = h.querySelector('.toggle'); b.classList.toggle('collapsed'); t.innerHTML = b.classList.contains('collapsed') ? '&#9654;' : '&#9660;'; }
function toggleBody(h) { h.nextElementSibling.classList.toggle('open'); }
function collapsible(title, content, open) {
  return '<div class="card"><div class="card-header" onclick="toggleCard(this)">' + title +
    ' <span class="toggle">' + (open ? '&#9660;' : '&#9654;') + '</span></div>' +
    '<div class="card-body ' + (open ? '' : 'collapsed') + '">' + content + '</div></div>';
}
function promptResponseBlock(label, prompt, response) {
  let h = '';
  if (prompt) {
    h += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)"><span style="color:var(--text-muted)">' + esc(label) + ' Prompt</span><span style="margin-left:auto;font-size:11px">&#9654;</span></div><div class="extraction-body"><pre style="max-height:400px">' + esc(prompt) + '</pre></div></div>';
  }
  if (response) {
    h += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)"><span style="color:var(--accent2)">' + esc(label) + ' Response</span><span style="margin-left:auto;font-size:11px">&#9654;</span></div><div class="extraction-body"><pre style="max-height:400px">' + esc(response) + '</pre></div></div>';
  }
  return h;
}

const TABS = [
  ['overview','Overview'],
  ['artifacts','Artifacts'],
  ['experiments','Experiments'],
  ['feedback','Feedback'],
  ['agent_messages','Agent Messages'],
  ['trajectory','Trajectory'],
  ['combined_trajectory','Cross-Episode Traj'],
  ['experiment_timeline','Experiment Timeline'],
  ['logs','Improve Log'],
];

function renderStep(idx) {
  const s = DATA.steps[idx];
  const mc = document.getElementById('main-content');
  const total = DATA.steps.length;

  const stepIsInProgress = s.phase && s.phase !== 'complete';
  const phaseColors = { started: 'var(--accent3)', acting: 'var(--accent)', extracting: 'var(--purple)', improving: 'var(--accent2)' };
  const phaseColor = phaseColors[s.phase] || 'var(--text-muted)';
  const phasePill = stepIsInProgress
    ? ' <span style="font-size:11px;padding:2px 8px;border-radius:10px;background:rgba(210,153,34,0.15);color:' + phaseColor + ';font-weight:600;vertical-align:middle">' + (s.phase || '') + '</span>'
    : '';

  let html = '<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">' +
    '<button style="background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px" onclick="showStep(' + Math.max(0,idx-1) + ')" ' + (idx===0?'disabled':'') + '>&#8592;</button>' +
    '<h1 style="margin:0;font-size:18px">Step ' + s.step + phasePill + ' <span style="color:var(--text-muted);font-size:14px;font-weight:400">ep' + s.episode_idx + ' | global ' + s.global_step + '</span></h1>' +
    '<span style="font-size:12px;color:var(--text-muted);margin-left:auto">action: <b>' + esc(s.action || '…') + '</b> | reward: ' + (s.action ? Number(s.reward).toFixed(2) : '—') + ' | cost: $' + Number(s.step_total_cost).toFixed(4) + '</span>' +
    '<button style="background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px" onclick="showStep(' + Math.min(total-1,idx+1) + ')" ' + (idx>=total-1?'disabled':'') + '>&#8594;</button>' +
    '</div>';

  html += '<div class="tabs">';
  TABS.forEach(t => {
    html += '<div class="tab ' + (currentTab===t[0]?'active':'') + '" onclick="currentTab=\\'' + t[0] + '\\';renderStep(' + idx + ')">' + t[1] + '</div>';
  });
  html += '</div>';

  if (currentTab === 'overview') html += '<div id="overview-container"><div class="loading">Loading...</div></div>';
  else if (currentTab === 'artifacts') html += '<div id="artifacts-container"><div class="loading">Loading...</div></div>';
  else if (currentTab === 'experiments') html += '<div id="experiments-container"><div class="loading">Loading...</div></div>';
  else if (currentTab === 'feedback') html += '<div id="feedback-container"><div class="loading">Loading...</div></div>';
  else if (currentTab === 'agent_messages') html += '<div id="agent-messages-container"><div class="loading">Loading...</div></div>';
  else if (currentTab === 'trajectory') html += '<div id="trajectory-container"><div class="loading">Loading...</div></div>';
  else if (currentTab === 'combined_trajectory') html += '<div id="combined-trajectory-container"><div class="loading">Loading...</div></div>';
  else if (currentTab === 'experiment_timeline') html += '<div id="experiment-timeline-container"><div class="loading">Loading...</div></div>';
  else if (currentTab === 'logs') html += '<div id="logs-container"><div class="loading">Loading...</div></div>';

  mc.innerHTML = html;

  // Overview needs detail for beliefs/perception
  if (currentTab === 'overview') loadStepDetailForTab(s.episode_idx, s.step, 'overview', s);
  else if (currentTab === 'artifacts') loadStepDetailForTab(s.episode_idx, s.step, 'artifacts');
  else if (currentTab === 'experiments') loadStepDetailForTab(s.episode_idx, s.step, 'experiments');
  else if (currentTab === 'feedback') loadStepDetailForTab(s.episode_idx, s.step, 'feedback');
  else if (currentTab === 'agent_messages') loadStepDetailForTab(s.episode_idx, s.step, 'agent_messages');
  else if (currentTab === 'trajectory') loadTrajectory(s.episode_idx, s.step);
  else if (currentTab === 'combined_trajectory') loadCombinedTrajectory(s.global_step);
  else if (currentTab === 'experiment_timeline') loadExperimentTimeline(s.global_step);
  else if (currentTab === 'logs') loadStepDetailForTab(s.episode_idx, s.step, 'logs');
}

async function loadStepDetailForTab(epIdx, stepIdx, tab, stepOverview) {
  const key = epIdx + '_' + stepIdx;
  let data = detailCache[key];
  if (!data) {
    try {
      const resp = await fetch(apiUrl('/api/step_detail', { episode: epIdx, step: stepIdx }));
      data = await resp.json();
      detailCache[key] = data;
    } catch(e) {
      const c = document.getElementById(tab + '-container') || document.getElementById('overview-container');
      if (c) c.innerHTML = '<pre>' + esc(e) + '</pre>';
      return;
    }
  }

  if (tab === 'overview') renderOverview(data, stepOverview);
  else if (tab === 'artifacts') renderArtifacts(data);
  else if (tab === 'experiments') renderExperiments(data);
  else if (tab === 'feedback') renderFeedback(data);
  else if (tab === 'agent_messages') renderAgentMessages(data);
  else if (tab === 'logs') renderLogs(data);
}

// ============ OVERVIEW (beliefs + perception + stats + cost chart) ============

function renderOverview(data, s) {
  const c = document.getElementById('overview-container');
  if (!c) return;
  let html = '';

  // Stats cards
  html += '<div class="stats-grid">' +
    '<div class="stat-card"><div class="stat-label">Action</div><div class="stat-value blue" style="font-size:16px;font-family:var(--font-mono)">' + esc(s.action) + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Reward</div><div class="stat-value ' + (s.reward > 0 ? 'green' : s.reward < 0 ? 'red' : '') + '">' + Number(s.reward).toFixed(2) + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Return So Far</div><div class="stat-value ' + (s.episode_return_so_far > 0 ? 'green' : '') + '">' + Number(s.episode_return_so_far).toFixed(2) + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Done</div><div class="stat-value ' + (s.done ? 'red' : 'green') + '">' + (s.done ? 'YES' : 'NO') + '</div></div>' +
    '</div>';

  html += '<div class="stats-grid">' +
    '<div class="stat-card"><div class="stat-label">Agent Cost</div><div class="stat-value yellow" style="font-size:16px">$' + Number(s.agent_step_cost).toFixed(4) + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Extract Cost</div><div class="stat-value yellow" style="font-size:16px">$' + Number(s.extract_cost).toFixed(4) + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Improve Cost</div><div class="stat-value yellow" style="font-size:16px">$' + Number(s.improve_cost).toFixed(4) + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Experiment Cost</div><div class="stat-value yellow" style="font-size:16px">$' + Number(s.experiment_cost).toFixed(4) + '</div></div>' +
    '</div>';

  html += '<div class="stats-grid">' +
    '<div class="stat-card"><div class="stat-label">QA Pairs</div><div class="stat-value purple">' + s.num_qa_pairs + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Moments</div><div class="stat-value blue">' + s.num_moments + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Experiments</div><div class="stat-value green">' + s.num_experiments + '</div></div>' +
    '<div class="stat-card"><div class="stat-label">Step Total Cost</div><div class="stat-value yellow">$' + Number(s.step_total_cost).toFixed(4) + '</div></div>' +
    '</div>';

  // Active experiment (generated at start of step, given to agent before it acted)
  if (s.active_experiment) {
    html += '<div class="card" style="margin-bottom:16px;border-left:3px solid var(--accent2)">' +
      '<div class="card-header" onclick="toggleCard(this)">Active Experiment (used this step) <span style="font-size:11px;color:var(--text-muted);font-weight:400">' + (s.did_gen_experiments ? 'generated at start of this step' : 'carried over from previous step') + '</span> <span class="toggle">&#9660;</span></div>' +
      '<div class="card-body"><pre style="max-height:none">' + esc(s.active_experiment) + '</pre></div></div>';
  }

  // Beliefs and Perception
  html += '<div class="side-by-side" style="margin-bottom:16px"><div><h3>Beliefs</h3><pre>' + esc(data.beliefs || '(empty)') + '</pre></div>';
  html += '<div><h3>Perception</h3><pre>' + esc(data.perception || '(empty)') + '</pre></div></div>';

  // Cost over time chart
  html += renderCostChart();

  c.innerHTML = html;
}

function renderCostChart() {
  if (!DATA.steps || DATA.steps.length < 2) return '';
  let html = '<div class="card"><div class="card-header" onclick="toggleCard(this)">Cost Over Steps <span class="toggle">&#9660;</span></div><div class="card-body">';
  const maxCost = Math.max(...DATA.steps.map(s => s.step_total_cost), 0.0001);
  html += '<div style="display:flex;align-items:flex-end;gap:2px;height:120px;border-bottom:1px solid var(--border)">';
  DATA.steps.forEach((s, i) => {
    const h = Math.max((s.step_total_cost / maxCost) * 100, 1);
    const isSelected = i === selectedStepIdx;
    const color = isSelected ? 'var(--accent)' : (s.improve_cost > 0 ? 'var(--purple)' : 'var(--surface2)');
    html += '<div style="flex:1;height:' + h + '%;background:' + color + ';border-radius:2px 2px 0 0;cursor:pointer;min-width:2px" title="g' + s.global_step + ': $' + s.step_total_cost.toFixed(4) + '" onclick="showStep(' + i + ')"></div>';
  });
  html += '</div>';
  html += '<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-muted);margin-top:4px"><span>g0</span><span>g' + DATA.steps[DATA.steps.length-1].global_step + '</span></div>';
  html += '</div></div>';
  return html;
}

// ============ ARTIFACTS (QA/moments with extraction prompts) ============

function renderArtifacts(data) {
  const c = document.getElementById('artifacts-container');
  if (!c) return;
  let html = '';

  // Extraction log (QA + moments updating prompts/responses)
  const extLog = data.extraction_log || {};
  if (extLog.extract_prompt || extLog.extract_response) {
    let extHtml = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">' +
      'New QA: ' + (extLog.new_qa_count || 0) + ' | New Moments: ' + (extLog.new_moments_count || 0) + '</div>';
    extHtml += promptResponseBlock('Extraction', extLog.extract_prompt, extLog.extract_response);
    html += collapsible('Knowledge Extraction (QA + Moments Updating)', extHtml, false);
  }

  // QA Pairs
  const qa = data.qa_pairs || [];
  if (qa.length > 0) {
    let qaHtml = '<table class="data-table"><tr><th>Question</th><th>Answer</th><th>Evidence</th><th>Src Step</th></tr>';
    qa.forEach(q => {
      const ans = q.answer === true ? 'YES' : q.answer === false ? 'NO' : String(q.answer);
      const col = q.answer ? 'var(--accent2)' : 'var(--danger)';
      qaHtml += '<tr><td style="max-width:280px">' + esc(q.question) + '</td><td style="color:' + col + ';font-weight:600">' + ans + '</td><td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(q.evidence) + '">' + esc(q.evidence) + '</td><td>' + esc(q.source_step) + '</td></tr>';
    });
    qaHtml += '</table>';
    html += collapsible('QA Pairs (' + qa.length + ')', qaHtml, false);
  }

  // Critical moments
  const mom = data.critical_moments || [];
  if (mom.length > 0) {
    let momHtml = '<table class="data-table"><tr><th>State</th><th>Goal</th><th>Good</th><th>Bad</th><th>Src Step</th></tr>';
    mom.forEach(m => {
      momHtml += '<tr><td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(m.state) + '">' + esc(m.state) + '</td>' +
        '<td>' + esc(m.goal) + '</td><td style="color:var(--accent2)">' + esc((m.good_actions||[]).join(', ')) + '</td>' +
        '<td style="color:var(--danger)">' + esc((m.bad_actions||[]).join(', ')) + '</td><td>' + esc(m.source_step) + '</td></tr>';
    });
    momHtml += '</table>';
    html += collapsible('Critical Moments (' + mom.length + ')', momHtml, false);
  }

  if (!html) html = '<div style="color:var(--text-muted);padding:20px">No artifact data for this step.</div>';
  c.innerHTML = html;
}

// ============ EXPERIMENTS (old, prompt/response, new) ============

function renderExperiments(data) {
  const c = document.getElementById('experiments-container');
  if (!c) return;
  let html = '';
  const expLog = data.experiment_log || {};
  const experiments = data.experiments || [];
  const stepMeta = DATA.steps[selectedStepIdx] || {};
  const didGen = stepMeta.did_gen_experiments;

  // Timing note
  html += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">' +
    (didGen
      ? 'Experiments were <strong style="color:var(--accent2)">generated at the start of this step</strong> (before the agent acted). The first experiment in the new list was set as the agent&#39;s goal for this step.'
      : 'No experiment generation this step \u2014 the agent used the experiment carried over from the previous generation cycle.') +
    '</div>';

  if (expLog.old_experiments || expLog.prompt || expLog.response || expLog.new_experiments) {
    // Experiments before generation (what was active going into this step)
    if (expLog.old_experiments && expLog.old_experiments.length > 0) {
      let oldHtml = '<ul style="margin:0;padding-left:20px">';
      expLog.old_experiments.forEach(e => { oldHtml += '<li style="margin-bottom:4px;font-size:13px">' + esc(e) + '</li>'; });
      oldHtml += '</ul>';
      html += collapsible('Experiments before generation (from prior cycle)', oldHtml, false);
    }

    // Generation prompt/response
    if (expLog.prompt || expLog.response) {
      let genHtml = promptResponseBlock('Experiment Generation', expLog.prompt, expLog.response);
      html += collapsible('Experiment Generation LLM Call (ran at start of step, shown current state + history)', genHtml, false);
    }

    // New experiments — first one was used as agent goal for this step
    const newExps = expLog.new_experiments || experiments;
    if (newExps && newExps.length > 0) {
      let newHtml = '<div style="font-size:11px;color:var(--text-muted);margin-bottom:8px">The first experiment below was set as the agent&#39;s active goal for this step.</div>';
      newHtml += '<ul style="margin:0;padding-left:20px">';
      newExps.forEach((e, i) => {
        const isActive = i === 0;
        newHtml += '<li style="margin-bottom:6px;font-size:13px;color:' + (isActive ? 'var(--accent2)' : 'var(--text)') + '">' +
          (isActive ? '<strong>[active this step] </strong>' : '') + esc(e) + '</li>';
      });
      newHtml += '</ul>';
      html += collapsible('Experiments generated this step (' + newExps.length + ')', newHtml, true);
    }
  } else if (experiments.length > 0) {
    // No generation log — just show what experiments existed (carried over)
    let expHtml = '<ul style="margin:0;padding-left:20px">';
    experiments.forEach(e => { expHtml += '<li style="margin-bottom:4px;font-size:13px">' + esc(e) + '</li>'; });
    expHtml += '</ul>';
    html += collapsible('Current Experiments (' + experiments.length + ')', expHtml, true);
  }

  const hasContent = experiments.length > 0 || expLog.old_experiments || expLog.prompt || expLog.new_experiments;
  if (!hasContent) html += '<div style="color:var(--text-muted);padding:20px">No experiment data for this step.</div>';
  c.innerHTML = html;
}

// ============ FEEDBACK (with progress chart) ============

function isNewTrackFormat(fb) {
  // New format: each entry has a "track" field
  return fb.length > 0 && fb[0].track != null;
}

function renderFeedback(data) {
  const c = document.getElementById('feedback-container');
  if (!c) return;
  const fb = data.feedback_history || [];
  if (fb.length === 0) { c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No feedback history for this step.</div>'; return; }

  // Detect new vs old format
  if (isNewTrackFormat(fb)) {
    renderFeedbackNew(c, fb);
  } else {
    renderFeedbackLegacy(c, fb);
  }
}

function getTrackMeta(track) {
  if (track === 'steps_beliefs')            return { label: 'Track 1b: Steps Beliefs',               color: 'var(--accent)'  };
  if (track === 'perception_from_analysis') return { label: 'Track 1c: Perception (from Analysis)',  color: 'var(--accent3)' };
  if (track === 'steps')                    return { label: 'Track 1: Steps-based',                  color: 'var(--accent)'  };
  if (track === 'qa')                       return { label: 'Track 2: QA',                           color: 'var(--purple)'  };
  if (track === 'moments')                  return { label: 'Track 3: Moments',                      color: 'var(--accent2)' };
  return { label: 'Track: ' + track, color: 'var(--text-muted)' };
}

function renderFeedbackNew(c, fb) {
  let html = '';

  // Progress chart for conversational tracks
  html += renderFeedbackProgressChartNew(fb);

  fb.forEach(trackRecord => {
    const track = trackRecord.track || 'unknown';
    const turns = trackRecord.turns || [];
    const { label: trackLabel, color: trackColor } = getTrackMeta(track);

    let trackHtml = '';
    if (trackRecord.global_step != null) trackHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:8px">Global step: ' + trackRecord.global_step + ' | Env step: ' + trackRecord.step + '</div>';

    // Track summary stats
    if (track === 'qa' || track === 'moments') {
      const initCorrect = trackRecord.initial_correct || 0;
      const initIncorrect = trackRecord.initial_incorrect || 0;
      trackHtml += '<div style="font-size:12px;margin-bottom:8px">Initial eval: <span style="color:var(--accent2)">' + initCorrect + ' correct</span>, <span style="color:var(--danger)">' + initIncorrect + ' incorrect</span></div>';
    }

    // QA feedback details (initial eval)
    if (track === 'qa' && trackRecord.qa_feedback_details && trackRecord.qa_feedback_details.length > 0) {
      let detailHtml = '<table class="data-table"><tr><th>Question</th><th>Correct</th><th>Predicted</th><th>Verdict</th><th>Feedback</th></tr>';
      trackRecord.qa_feedback_details.forEach(d => {
        const vClass = d.verdict === 'CORRECT' ? 'verdict-correct' : d.verdict === 'INCORRECT' ? 'verdict-incorrect' : 'verdict-inconclusive';
        detailHtml += '<tr><td style="max-width:200px">' + esc(d.question || (d.forward && d.forward.qa_pair ? d.forward.qa_pair.question : '')) + '</td>' +
          '<td>' + esc(d.correct_answer || '') + '</td>' +
          '<td>' + esc(d.predicted_answer || (d.forward ? d.forward.predicted_answer : '')) + '</td>' +
          '<td><span class="verdict ' + vClass + '">' + esc(d.verdict) + '</span></td>' +
          '<td style="max-width:200px">' + esc(d.feedback) + '</td></tr>';
      });
      detailHtml += '</table>';
      trackHtml += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)"><span>QA Feedback Details</span><span style="margin-left:auto;font-size:11px">&#9654;</span></div><div class="extraction-body">' + detailHtml + '</div></div>';
      trackHtml += promptResponseBlock('QA Forward', trackRecord.qa_forward_prompt, trackRecord.qa_forward_response);
      trackHtml += promptResponseBlock('QA Feedback', trackRecord.qa_feedback_prompt, trackRecord.qa_feedback_response);
    }

    // Moment feedback details (initial eval)
    if (track === 'moments' && trackRecord.moment_feedback_details && trackRecord.moment_feedback_details.length > 0) {
      let detailHtml = '<table class="data-table"><tr><th>Goal</th><th>Predicted</th><th>Good</th><th>Bad</th><th>Verdict</th><th>Feedback</th></tr>';
      trackRecord.moment_feedback_details.forEach(d => {
        const vClass = d.verdict === 'CORRECT' ? 'verdict-correct' : d.verdict === 'INCORRECT' ? 'verdict-incorrect' : 'verdict-inconclusive';
        const moment = d.forward ? d.forward.moment : {};
        detailHtml += '<tr><td>' + esc(moment.goal || d.goal || '') + '</td>' +
          '<td>' + esc(d.predicted_action || (d.forward ? d.forward.predicted_action : '')) + '</td>' +
          '<td style="color:var(--accent2)">' + esc((moment.good_actions||[]).join(', ')) + '</td>' +
          '<td style="color:var(--danger)">' + esc((moment.bad_actions||[]).join(', ')) + '</td>' +
          '<td><span class="verdict ' + vClass + '">' + esc(d.verdict) + '</span></td>' +
          '<td style="max-width:200px">' + esc(d.feedback) + '</td></tr>';
      });
      detailHtml += '</table>';
      trackHtml += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)"><span>Moment Feedback Details</span><span style="margin-left:auto;font-size:11px">&#9654;</span></div><div class="extraction-body">' + detailHtml + '</div></div>';
    }

    // Conversation turns
    if (turns.length > 0) {
      let turnsHtml = '<div style="margin-top:8px">';
      const totalCost = turns.reduce((s, t) => s + (t.cost || 0), 0);
      turnsHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:12px">' + turns.length + ' turn(s), total cost: $' + totalCost.toFixed(4) + '</div>';

      turns.forEach(t => {
        const submitBadge = t.submitted
          ? '<span style="background:rgba(63,185,80,0.15);color:var(--accent2);padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">SUBMITTED</span>'
          : '<span style="background:rgba(88,166,255,0.1);color:var(--accent);padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">CONTINUE</span>';

        // Turn header
        turnsHtml += '<div class="extraction-section" style="margin-bottom:8px">' +
          '<div class="extraction-header" onclick="toggleBody(this)" style="padding:8px 12px">' +
            '<span style="font-weight:600;color:' + trackColor + '">Turn ' + t.turn + '</span>' +
            '<span style="color:var(--accent3);font-family:var(--font-mono);font-size:11px;margin-left:8px">$' + (t.cost || 0).toFixed(4) + '</span>' +
            submitBadge +
            '<span style="margin-left:auto;font-size:11px;color:var(--text-muted)">&#9654;</span>' +
          '</div>' +
          '<div class="extraction-body">';

        // Prompt bubble
        turnsHtml += '<div style="margin-bottom:10px">' +
          '<div style="font-size:10px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px;font-weight:600">Prompt</div>' +
          '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:10px 14px">' +
          '<pre style="max-height:400px;margin:0;border:none;padding:0;background:transparent">' + (t.prompt ? esc(t.prompt) : '<span style="color:var(--text-muted)">(not recorded — re-run with latest code to capture prompts)</span>') + '</pre>' +
          '</div></div>';

        // Response bubble
        turnsHtml += '<div style="margin-bottom:6px">' +
          '<div style="font-size:10px;text-transform:uppercase;color:var(--accent2);margin-bottom:4px;font-weight:600">Response</div>' +
          '<div style="background:rgba(88,166,255,0.05);border:1px solid rgba(88,166,255,0.2);border-radius:8px;padding:10px 14px">' +
          '<pre style="max-height:400px;margin:0;border:none;padding:0;background:transparent">' + (t.response ? esc(t.response) : '<span style="color:var(--text-muted)">(not recorded — re-run with latest code to capture responses)</span>') + '</pre>' +
          '</div></div>';

        turnsHtml += '</div></div>';
      });
      turnsHtml += '</div>';
      trackHtml += turnsHtml;
    }

    if (trackRecord.error) trackHtml += '<pre style="color:var(--danger)">' + esc(trackRecord.error) + '</pre>';

    html += '<div style="margin-bottom:16px;padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:6px;border-left:3px solid ' + trackColor + '">' +
      '<div style="font-size:13px;font-weight:600;color:' + trackColor + ';margin-bottom:8px">' + trackLabel + '</div>' +
      trackHtml + '</div>';
  });
  c.innerHTML = html;
}

function renderFeedbackProgressChartNew(fb) {
  // For new format: show per-track turn cost progression
  const trackData = [];
  fb.forEach(tr => {
    if (tr.turns && tr.turns.length > 1) {
      trackData.push({
        track: tr.track,
        turns: tr.turns,
        initial_correct: tr.initial_correct,
        initial_incorrect: tr.initial_incorrect,
      });
    }
  });

  if (trackData.length === 0) return '';

  let html = '<div class="card" style="margin-bottom:16px"><div class="card-header" onclick="toggleCard(this)">Conversation Progress <span class="toggle">&#9660;</span></div><div class="card-body">';

  trackData.forEach(td => {
    const { label: trackLabel, color: trackColor } = getTrackMeta(td.track);

    html += '<div style="margin-bottom:16px"><div style="font-size:12px;font-weight:600;margin-bottom:8px;color:' + trackColor + '">' + trackLabel + ' (' + td.turns.length + ' turns)</div>';

    // Cost per turn bar chart
    const maxCost = Math.max(...td.turns.map(t => t.cost || 0), 0.0001);
    html += '<div style="display:flex;gap:8px;align-items:flex-end;height:80px">';
    td.turns.forEach(t => {
      const h = Math.max(Math.round(((t.cost || 0) / maxCost) * 70), 2);
      const color = t.submitted ? 'var(--accent2)' : trackColor;
      html += '<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:0">' +
        '<div style="font-size:9px;color:var(--text-muted);margin-bottom:2px">$' + (t.cost||0).toFixed(3) + '</div>' +
        '<div style="width:100%;min-width:20px;height:' + h + 'px;background:' + color + ';border-radius:3px" title="Turn ' + t.turn + ': $' + (t.cost||0).toFixed(4) + (t.submitted ? ' (SUBMITTED)' : '') + '"></div>' +
        '<div style="font-size:10px;color:var(--text-muted);margin-top:4px">T' + t.turn + '</div>' +
        '</div>';
    });
    html += '</div></div>';
  });

  html += '</div></div>';
  return html;
}

// Legacy feedback rendering for old format (iteration-based)
function renderFeedbackLegacy(c, fb) {
  let html = '';

  // Progress chart: QA and moment accuracy across iterations
  html += renderFeedbackProgressChartLegacy(fb);

  fb.forEach(iter => {
    let iterHtml = '';
    if (iter.global_step != null) iterHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:8px">Global step: ' + iter.global_step + ' | Env step: ' + iter.step + '</div>';

    // Track 1
    if (iter.steps_cost) {
      iterHtml += '<div style="margin-bottom:12px;padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:6px">';
      iterHtml += '<div style="font-size:12px;font-weight:600;color:var(--accent);margin-bottom:4px">Track 1: Steps-based ($' + iter.steps_cost.toFixed(4) + ')</div>';
      iterHtml += promptResponseBlock('Steps Improve', iter.steps_prompt, iter.steps_response);
      iterHtml += '</div>';
    }

    // Track 2
    if (iter.qa_num_pairs) {
      iterHtml += '<div style="margin-bottom:12px;padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:6px">';
      iterHtml += '<div style="font-size:12px;font-weight:600;color:var(--purple);margin-bottom:4px">Track 2: QA (' + (iter.qa_num_correct||0) + ' correct, ' + (iter.qa_num_incorrect||0) + ' incorrect, ' + (iter.qa_num_inconclusive||0) + ' inconclusive)</div>';

      // QA feedback details
      if (iter.qa_feedback_details && iter.qa_feedback_details.length > 0) {
        let detailHtml = '<table class="data-table"><tr><th>Question</th><th>Correct</th><th>Predicted</th><th>Verdict</th><th>Feedback</th></tr>';
        iter.qa_feedback_details.forEach(d => {
          const vClass = d.verdict === 'CORRECT' ? 'verdict-correct' : d.verdict === 'INCORRECT' ? 'verdict-incorrect' : 'verdict-inconclusive';
          detailHtml += '<tr><td style="max-width:200px">' + esc(d.question || (d.forward && d.forward.qa_pair ? d.forward.qa_pair.question : '')) + '</td>' +
            '<td>' + esc(d.correct_answer || '') + '</td>' +
            '<td>' + esc(d.predicted_answer || (d.forward ? d.forward.predicted_answer : '')) + '</td>' +
            '<td><span class="verdict ' + vClass + '">' + esc(d.verdict) + '</span></td>' +
            '<td style="max-width:200px">' + esc(d.feedback) + '</td></tr>';
        });
        detailHtml += '</table>';
        iterHtml += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)"><span>QA Feedback Details</span><span style="margin-left:auto;font-size:11px">&#9654;</span></div><div class="extraction-body">' + detailHtml + '</div></div>';
      }

      iterHtml += promptResponseBlock('QA Forward', iter.qa_forward_prompt, iter.qa_forward_response);
      iterHtml += promptResponseBlock('QA Feedback', iter.qa_feedback_prompt, iter.qa_feedback_response);
      iterHtml += promptResponseBlock('QA Improve', iter.qa_improve_prompt, iter.qa_improve_response);
      iterHtml += '</div>';
    }

    // Track 3
    if (iter.moment_num_moments) {
      iterHtml += '<div style="margin-bottom:12px;padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:6px">';
      iterHtml += '<div style="font-size:12px;font-weight:600;color:var(--accent);margin-bottom:4px">Track 3: Moments (' + (iter.moment_num_correct||0) + ' correct, ' + (iter.moment_num_incorrect||0) + ' incorrect, ' + (iter.moment_num_inconclusive||0) + ' inconclusive)</div>';

      // Moment feedback details
      if (iter.moment_feedback_details && iter.moment_feedback_details.length > 0) {
        let detailHtml = '<table class="data-table"><tr><th>Goal</th><th>Predicted</th><th>Good</th><th>Bad</th><th>Verdict</th><th>Feedback</th></tr>';
        iter.moment_feedback_details.forEach(d => {
          const vClass = d.verdict === 'CORRECT' ? 'verdict-correct' : d.verdict === 'INCORRECT' ? 'verdict-incorrect' : 'verdict-inconclusive';
          const moment = d.forward ? d.forward.moment : {};
          detailHtml += '<tr><td>' + esc(moment.goal || d.goal || '') + '</td>' +
            '<td>' + esc(d.predicted_action || (d.forward ? d.forward.predicted_action : '')) + '</td>' +
            '<td style="color:var(--accent2)">' + esc((moment.good_actions||[]).join(', ')) + '</td>' +
            '<td style="color:var(--danger)">' + esc((moment.bad_actions||[]).join(', ')) + '</td>' +
            '<td><span class="verdict ' + vClass + '">' + esc(d.verdict) + '</span></td>' +
            '<td style="max-width:200px">' + esc(d.feedback) + '</td></tr>';
        });
        detailHtml += '</table>';
        iterHtml += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)"><span>Moment Feedback Details</span><span style="margin-left:auto;font-size:11px">&#9654;</span></div><div class="extraction-body">' + detailHtml + '</div></div>';
      }

      iterHtml += promptResponseBlock('Moment Improve', iter.moment_improve_prompt, iter.moment_improve_response);
      iterHtml += '</div>';
    }

    if (iter.error) iterHtml += '<pre style="color:var(--danger)">' + esc(iter.error) + '</pre>';

    html += collapsible('Iteration ' + (iter.iteration || '?'), iterHtml, fb.length === 1);
  });
  c.innerHTML = html;
}

function renderFeedbackProgressChartLegacy(fb) {
  // Collect QA and moment accuracy per iteration
  const qaData = [];
  const momentData = [];
  fb.forEach((iter, i) => {
    if (iter.qa_num_pairs) {
      const total = (iter.qa_num_correct||0) + (iter.qa_num_incorrect||0) + (iter.qa_num_inconclusive||0);
      qaData.push({ iter: i+1, correct: iter.qa_num_correct||0, incorrect: iter.qa_num_incorrect||0, inconclusive: iter.qa_num_inconclusive||0, total: total });
    }
    if (iter.moment_num_moments) {
      const total = (iter.moment_num_correct||0) + (iter.moment_num_incorrect||0) + (iter.moment_num_inconclusive||0);
      momentData.push({ iter: i+1, correct: iter.moment_num_correct||0, incorrect: iter.moment_num_incorrect||0, inconclusive: iter.moment_num_inconclusive||0, total: total });
    }
  });

  if (qaData.length === 0 && momentData.length === 0) return '';

  let html = '<div class="card" style="margin-bottom:16px"><div class="card-header" onclick="toggleCard(this)">Feedback Progress Across Iterations <span class="toggle">&#9660;</span></div><div class="card-body">';

  // Build stacked bar charts
  function stackedBars(items, label, correctColor, incorrectColor, inconclusiveColor) {
    if (items.length === 0) return '';
    let h = '<div style="margin-bottom:16px"><div style="font-size:12px;font-weight:600;margin-bottom:8px">' + label + '</div>';
    const maxTotal = Math.max(...items.map(d => d.total), 1);
    h += '<div style="display:flex;gap:12px;align-items:flex-end;height:100px">';
    items.forEach(d => {
      const cH = Math.round((d.correct / maxTotal) * 90);
      const iH = Math.round((d.incorrect / maxTotal) * 90);
      const uH = Math.round((d.inconclusive / maxTotal) * 90);
      h += '<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:0">' +
        '<div style="display:flex;flex-direction:column-reverse;width:100%;min-width:20px">' +
        '<div style="height:' + cH + 'px;background:' + correctColor + ';border-radius:0 0 3px 3px" title="Correct: ' + d.correct + '"></div>' +
        '<div style="height:' + iH + 'px;background:' + incorrectColor + '" title="Incorrect: ' + d.incorrect + '"></div>' +
        '<div style="height:' + uH + 'px;background:' + inconclusiveColor + ';border-radius:3px 3px 0 0" title="Inconclusive: ' + d.inconclusive + '"></div>' +
        '</div>' +
        '<div style="font-size:10px;color:var(--text-muted);margin-top:4px">It ' + d.iter + '</div>' +
        '</div>';
    });
    h += '</div>';
    // Legend
    h += '<div style="display:flex;gap:16px;margin-top:8px;font-size:11px">' +
      '<span><span style="display:inline-block;width:10px;height:10px;background:' + correctColor + ';border-radius:2px;vertical-align:middle"></span> Correct</span>' +
      '<span><span style="display:inline-block;width:10px;height:10px;background:' + incorrectColor + ';border-radius:2px;vertical-align:middle"></span> Incorrect</span>' +
      '<span><span style="display:inline-block;width:10px;height:10px;background:' + inconclusiveColor + ';border-radius:2px;vertical-align:middle"></span> Inconclusive</span>' +
      '</div>';
    h += '</div>';
    return h;
  }

  html += stackedBars(qaData, 'QA Pairs Accuracy', 'var(--accent2)', 'var(--danger)', 'var(--accent3)');
  html += stackedBars(momentData, 'Critical Moments Accuracy', 'var(--accent2)', 'var(--danger)', 'var(--accent3)');

  html += '</div></div>';
  return html;
}

// ============ AGENT MESSAGES ============

function renderAgentMessages(data) {
  const c = document.getElementById('agent-messages-container');
  if (!c) return;
  const msgs = data.agent_messages || [];
  if (msgs.length === 0) { c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No agent messages for this step.</div>'; return; }

  let html = '<div style="display:flex;flex-direction:column;gap:8px">';
  msgs.forEach((m, i) => {
    const isAssistant = m.role === 'assistant';
    const bubbleClass = isAssistant ? 'msg-assistant' : 'msg-user';
    const roleLabel = m.role || 'unknown';
    const content = m.content || '';
    const isResponse = isAssistant && m.action !== undefined;

    let extra = '';
    if (isResponse && m.action) {
      extra = '<div style="margin-top:8px;padding:6px 10px;background:var(--accent);color:#fff;border-radius:4px;font-family:monospace;font-size:0.9em">' +
        '<strong>Action:</strong> ' + esc(m.action) + '</div>';
    }

    html += '<div class="msg-bubble ' + bubbleClass + '">' +
      '<div class="msg-role">' + esc(isResponse ? 'assistant (response)' : roleLabel) +
      ' (message ' + (i+1) + '/' + msgs.length + ', ' + content.length + ' chars)</div>' +
      '<pre style="max-height:none;margin:0;border:none;padding:0;background:transparent">' + esc(content) + '</pre>' +
      extra +
      '</div>';
  });
  html += '</div>';
  c.innerHTML = html;
}

// ============ LOGS ============

function renderLogs(data) {
  const c = document.getElementById('logs-container');
  if (!c) return;
  if (!data.improve_log) { c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No improve log for this step.</div>'; return; }
  c.innerHTML = '<pre style="max-height:none">' + esc(data.improve_log) + '</pre>';
}

// ============ TRAJECTORY ============

async function loadTrajectory(epIdx, highlightStep) {
  const c = document.getElementById('trajectory-container');
  if (!c) return;
  let traj = trajCache[epIdx];
  if (!traj) {
    try {
      const resp = await fetch(apiUrl('/api/trajectory', { episode: epIdx }));
      traj = await resp.json();
      trajCache[epIdx] = traj;
    } catch(e) { c.innerHTML = '<pre>' + esc(e) + '</pre>'; return; }
  }
  if (!traj || traj.length === 0) { c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No trajectory data.</div>'; return; }

  let html = '';
  traj.forEach((t, i) => {
    const isHighlighted = (parseInt(t.step) === highlightStep);
    const doneClass = t.done === 'True' ? (parseFloat(t.reward) > 0 ? 'success' : 'death') : '';
    const border = isHighlighted ? 'border-color:var(--accent)' : '';
    html += '<div class="traj-step" style="' + border + '">' +
      '<div class="traj-step-header" onclick="toggleBody(this)">' +
        '<div class="traj-step-num ' + doneClass + '">' + t.step + '</div>' +
        '<div class="traj-step-action">' + esc(t.action) + '</div>' +
        '<div class="traj-step-reward">r=' + t.reward + (t.done === 'True' ? ' (DONE)' : '') + '</div>' +
      '</div>' +
      '<div class="traj-step-body' + (isHighlighted ? ' open' : '') + '">' +
        '<div class="traj-section"><div class="traj-section-label">Observation</div><pre>' + esc(t.observation) + '</pre></div>' +
        (t.auxiliary_observation ? '<div class="traj-section"><div class="traj-section-label">Auxiliary Observation</div><pre>' + esc(t.auxiliary_observation) + '</pre></div>' : '') +
        '<div class="traj-section"><div class="traj-section-label">Reasoning</div><pre>' + esc(t.reasoning) + '</pre></div>' +
      '</div></div>';
  });
  c.innerHTML = html;
  // Scroll to highlighted step
  if (highlightStep != null) {
    setTimeout(() => {
      const el = c.querySelector('.traj-step[style*="accent"]');
      if (el) el.scrollIntoView({behavior:'smooth', block:'center'});
    }, 100);
  }
}

// ============ COMBINED TRAJECTORY (cross-episode) ============

async function loadCombinedTrajectory(highlightGlobalStep) {
  const c = document.getElementById('combined-trajectory-container');
  if (!c) return;
  if (!combinedTrajCache) {
    try {
      const resp = await fetch(apiUrl('/api/combined_trajectory'));
      combinedTrajCache = await resp.json();
    } catch(e) { c.innerHTML = '<pre>' + esc(e) + '</pre>'; return; }
  }
  const traj = combinedTrajCache;
  if (!traj || traj.length === 0) { c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No trajectory data.</div>'; return; }

  let html = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">' +
    'Combined trajectory across all episodes. Episode boundaries are marked. Global step numbers shown.' +
    '</div>';

  traj.forEach((t, i) => {
    if (t.episode_boundary) {
      html += '<div style="padding:10px 16px;margin:8px 0;background:var(--surface);border:1px solid var(--accent3);border-radius:6px;font-size:12px;font-weight:600;color:var(--accent3);text-align:center">' +
        '\\u2014\\u2014 Episode ' + t.episode_idx + ' starts \\u2014\\u2014</div>';
      return;
    }
    const gs = t.global_step;
    const isHighlighted = (gs === highlightGlobalStep);
    const reward = parseFloat(t.reward) || 0;
    const doneClass = t.done === 'True' ? (reward > 0 ? 'success' : 'death') : '';
    const border = isHighlighted ? 'border-color:var(--accent)' : '';
    html += '<div class="traj-step" style="' + border + '" data-gs="' + gs + '">' +
      '<div class="traj-step-header" onclick="toggleBody(this)">' +
        '<div class="traj-step-num ' + doneClass + '">' + gs + '</div>' +
        '<span style="font-size:10px;color:var(--text-muted);margin-right:4px">ep' + t.episode_idx + '</span>' +
        '<div class="traj-step-action">' + esc(t.action) + '</div>' +
        '<div class="traj-step-reward">r=' + t.reward + (t.done === 'True' ? ' (DONE)' : '') + '</div>' +
      '</div>' +
      '<div class="traj-step-body' + (isHighlighted ? ' open' : '') + '">' +
        '<div class="traj-section"><div class="traj-section-label">Observation</div><pre>' + esc(t.observation) + '</pre></div>' +
        (t.auxiliary_observation ? '<div class="traj-section"><div class="traj-section-label">Auxiliary Observation</div><pre>' + esc(t.auxiliary_observation) + '</pre></div>' : '') +
        '<div class="traj-section"><div class="traj-section-label">Reasoning</div><pre>' + esc(t.reasoning) + '</pre></div>' +
      '</div></div>';
  });
  c.innerHTML = html;
  if (highlightGlobalStep != null) {
    setTimeout(() => {
      const el = c.querySelector('.traj-step[style*="accent"]');
      if (el) el.scrollIntoView({behavior:'smooth', block:'center'});
    }, 100);
  }
}

// ============ EXPERIMENT TIMELINE ============

async function loadExperimentTimeline(highlightGlobalStep) {
  const c = document.getElementById('experiment-timeline-container');
  if (!c) return;
  if (!expTimelineCache) {
    try {
      const resp = await fetch(apiUrl('/api/experiment_timeline'));
      expTimelineCache = await resp.json();
    } catch(e) { c.innerHTML = '<pre>' + esc(e) + '</pre>'; return; }
  }
  const timeline = expTimelineCache;
  if (!timeline || timeline.length === 0) {
    c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No experiment generation events found.</div>';
    return;
  }

  let html = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">' +
    'Experiment generation events across all episodes, showing how the experiment pool grows over time. ' +
    'With cross-episode history, past experiments from earlier episodes are now visible to the LLM.' +
    '</div>';

  // Cumulative chart
  if (timeline.length > 1) {
    const maxCount = Math.max(...timeline.map(t => t.cumulative_count), 1);
    html += '<div class="card" style="margin-bottom:16px"><div class="card-header" onclick="toggleCard(this)">Cumulative Unique Experiments <span class="toggle">&#9660;</span></div><div class="card-body">';
    html += '<div style="display:flex;align-items:flex-end;gap:4px;height:100px;border-bottom:1px solid var(--border)">';
    timeline.forEach((t, i) => {
      const h = Math.max(Math.round((t.cumulative_count / maxCount) * 90), 2);
      const isNear = t.global_step === highlightGlobalStep;
      const color = isNear ? 'var(--accent)' : 'var(--accent2)';
      html += '<div style="flex:1;height:' + h + '%;background:' + color + ';border-radius:2px 2px 0 0;min-width:4px;cursor:pointer" ' +
        'title="g' + t.global_step + ' (ep' + t.episode_idx + '): ' + t.cumulative_count + ' unique experiments' +
        (t.first_seen.length > 0 ? ', +' + t.first_seen.length + ' new' : '') + '" ' +
        'onclick="showStep(' + DATA.steps.findIndex(s => s.global_step === ' + t.global_step + ') + ')"></div>';
    });
    html += '</div>';
    html += '<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-muted);margin-top:4px"><span>g' + timeline[0].global_step + '</span><span>g' + timeline[timeline.length-1].global_step + '</span></div>';
    html += '</div></div>';
  }

  // Per-event detail
  let lastEp = -1;
  timeline.forEach((t, i) => {
    if (t.episode_idx !== lastEp) {
      lastEp = t.episode_idx;
      html += '<div style="padding:8px 16px;font-size:12px;font-weight:600;color:var(--accent);background:var(--bg);border-bottom:1px solid var(--border);margin-top:8px">Episode ' + t.episode_idx + '</div>';
    }

    const isHighlighted = t.global_step === highlightGlobalStep;
    const borderStyle = isHighlighted ? 'border-left:3px solid var(--accent)' : 'border-left:3px solid var(--surface2)';

    let eventHtml = '<div style="padding:10px 14px;margin-bottom:6px;background:var(--surface);border:1px solid var(--border);border-radius:6px;' + borderStyle + '">';
    eventHtml += '<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">';
    eventHtml += '<span style="font-size:12px;font-weight:600;color:var(--accent)">g' + t.global_step + '</span>';
    eventHtml += '<span style="font-size:11px;color:var(--text-muted)">ep' + t.episode_idx + ' step ' + t.step + '</span>';
    eventHtml += '<span style="font-size:11px;color:var(--accent2);font-weight:600">' + t.cumulative_count + ' total unique</span>';
    if (t.first_seen.length > 0) {
      eventHtml += '<span style="font-size:11px;padding:2px 8px;background:rgba(63,185,80,0.15);color:var(--accent2);border-radius:4px;font-weight:600">+' + t.first_seen.length + ' new</span>';
    }
    eventHtml += '</div>';

    // Active experiment
    if (t.active) {
      eventHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">ACTIVE:</div>';
      eventHtml += '<div style="font-size:12px;padding:6px 10px;background:var(--bg);border:1px solid var(--accent2);border-radius:4px;margin-bottom:8px">' + esc(t.active) + '</div>';
    }

    // New experiments generated
    if (t.new_experiments.length > 0) {
      eventHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">GENERATED (' + t.new_experiments.length + '):</div>';
      eventHtml += '<ul style="margin:0 0 8px 20px;font-size:12px">';
      t.new_experiments.forEach(e => {
        const isNew = t.first_seen.includes(e);
        eventHtml += '<li style="margin-bottom:2px;color:' + (isNew ? 'var(--accent2)' : 'var(--text-muted)') + '">' +
          (isNew ? '<strong>[NEW] </strong>' : '<span style="opacity:0.6">[seen before] </span>') + esc(e) + '</li>';
      });
      eventHtml += '</ul>';
    }

    // Old experiments (what was passed in)
    if (t.old_experiments.length > 0) {
      eventHtml += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)"><span style="font-size:11px;color:var(--text-muted)">Input experiments (' + t.old_experiments.length + ')</span><span style="margin-left:auto;font-size:11px">&#9654;</span></div>';
      eventHtml += '<div class="extraction-body"><ul style="margin:0;padding-left:20px;font-size:12px">';
      t.old_experiments.forEach(e => { eventHtml += '<li style="margin-bottom:2px">' + esc(e) + '</li>'; });
      eventHtml += '</ul></div></div>';
    }

    eventHtml += '</div>';
    html += eventHtml;
  });

  c.innerHTML = html;
}

init();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize stepwise_b_learn.py logs")
    parser.add_argument("log_dir", nargs="?", default=None, help="Path to log directory (optional — can be set in browser)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--open-browser", action="store_true", help="Open browser automatically")
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"

    if args.log_dir:
        try:
            resolved = resolve_log_dir(args.log_dir)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Log dir resolved to: {resolved}")
        data = load_log_dir(resolved)
        print(f"Found {len(data['episodes'])} episodes, {len(data['steps'])} steps")
        url += f"?log_dir={quote(resolved, safe='')}"

    print(f"Serving at {url}")

    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
