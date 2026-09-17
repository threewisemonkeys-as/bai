"""Prepare and execute direct forward-prediction and reconstruction probes.

Ground-truth targets stay in scorer records. The API adapter receives only the
prompt and frozen model configuration. Preparing jobs never imports a model client.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from probing_common import (
    REPO, SCHEMA_VERSION, collision_summary, decode_lossless, digest, file_digest,
    forward_prompt, grid_scores, json_text, lossless_encoding, native_scores, parse_grid,
    perceive, perception_runtime, read_json, read_jsonl, reconstruction_prompt, write_json,
)
from probing_formats import DEFAULT_FORMAT, GridContract, get_format
from probing_manifest import verify_manifest

FORWARD_MODES = {"learned_raw", "learned_native", "raw", "lossless", "copy"}
RECONSTRUCTION_MODES = {"learned", "raw", "lossless", "constant", "hash", "shuffled",
                        "lossless_inverse", "train_mode"}


def artifact_groups(run: dict, labels: set[str], *, reconstruction=False):
    groups = {}
    for checkpoint in run["checkpoints"]:
        if labels and checkpoint["label"] not in labels:
            continue
        idx = str(checkpoint["candidate_idx"])
        artifact = run["artifacts"][idx]
        key = (artifact["perception_sha256"],
               None if reconstruction else artifact["knowledge_sha256"])
        group = groups.setdefault(key, {"artifact": artifact, "checkpoints": []})
        group["checkpoints"].append(checkpoint)
    return groups.values()


def study_jobs(manifest: dict, *, cache_dir: Path, forward_modes: set[str],
               reconstruction_modes: set[str], checkpoint_labels: set[str],
               examples: int | None = None, arms: set[str] | None = None,
               grid_format: str | None = None):
    if forward_modes - FORWARD_MODES or reconstruction_modes - RECONSTRUCTION_MODES:
        raise ValueError("unknown probe mode")
    for game, dataset in sorted(manifest["games"].items()):
        print(f"[{game}] preparing probe prompts and representation controls", flush=True)
        frames = dataset["frames"]
        demo_ids = dataset["example_frame_ids"]
        if examples is not None:
            if not 0 <= examples <= len(demo_ids):
                raise ValueError("example count exceeds the frozen demonstration set")
            demo_ids = demo_ids[:examples]
        required = set(demo_ids)
        for window in dataset["windows"]:
            required.update(window["context_frame_ids"] + window["future_frame_ids"])
        required.update(q["frame_id"] for q in dataset["reconstruction_queries"])
        observations = {fid: frames[fid]["observation"] for fid in sorted(required)}
        representations = {}
        # One contract per game: the wire format bound to this game's grid shape,
        # background and full colour vocabulary, shared by the prompts and the scorer.
        contract = None
        if grid_format is not None:
            contract = GridContract.for_grids(
                grid_format, (parse_grid(frames[fid]["grid"]) for fid in sorted(frames)),
                dataset["program"]["background"])

        def outputs(artifact):
            key = artifact["perception_sha256"]
            if key not in representations:
                representations[key] = perceive(artifact["perception"], observations,
                    timeout=manifest["config"]["perception_runtime"]["frame_timeout_s"],
                    cache_dir=cache_dir)
            return representations[key]

        def rendered(mode, artifact=None):
            if mode in {"learned", "learned_raw", "learned_native", "shuffled"}:
                return outputs(artifact)
            fn = {"raw": lambda x: x, "copy": lambda x: x,
                  "lossless": lossless_encoding, "lossless_inverse": lossless_encoding,
                  "constant": lambda x: "(constant)", "hash": lambda x: digest(x),
                  "train_mode": lambda x: "(constant)"}[mode]
            return {fid: {"output": fn(frames[fid]["grid"]), "error": None} for fid in required}

        def metadata(mode, run=None, group=None):
            return {"game": game, "arm": run["arm"] if run else "baseline",
                    "mode": mode, "training_seed": run["seed"] if run else None,
                    "checkpoints": group["checkpoints"] if group else [{"label": mode}],
                    "perception_sha256": group["artifact"]["perception_sha256"] if group else None,
                    "knowledge_sha256": group["artifact"]["knowledge_sha256"] if group else None,
                    "example_frame_ids": demo_ids, "background": dataset["program"]["background"],
                    "grid_contract": contract.as_dict() if contract else None}

        def preset(grid_text):
            """A deterministic baseline must answer in the same wire format the model
            is asked for, or the contract's parser would score the control as zero."""
            return contract.render(parse_grid(grid_text)) if contract else grid_text

        def errors_for(values, ids):
            errors = [values[fid]["error"] for fid in ids if values[fid]["error"]]
            if any(not values[fid]["output"] for fid in ids):
                errors.append("empty perception output")
            return sorted(set(errors))

        def forward(mode, run=None, group=None):
            artifact = group["artifact"] if group else None
            values = rendered(mode, artifact)
            native = mode == "learned_native"
            demos = [] if native else [(values[f]["output"], frames[f]["grid"]) for f in demo_ids]
            for window in dataset["windows"]:
                ctx = window["context_frame_ids"]
                for h in manifest["config"]["horizons"]:
                    target_id = window["future_frame_ids"][h - 1]
                    input_ids = ctx + ([] if native else demo_ids)
                    used = input_ids + ([target_id] if native else [])
                    errors = errors_for(values, used)
                    prompt = "" if mode == "copy" else forward_prompt(
                        [values[f]["output"] for f in ctx], window["past_actions"],
                        window["future_actions"][:h], artifact["knowledge"] if artifact else "",
                        demos, native=native, contract=None if native else contract)
                    job = {**metadata(mode, run, group), "kind": "forward", "horizon": h,
                        "query_id": window["id"], "user_id": window["user_id"],
                        "drive_id": window["drive_id"], "row_index": window["row_index"],
                        "target_space": "native" if native else "raw",
                        "target": values[target_id]["output"] if native else frames[target_id]["grid"],
                        "start": values[ctx[-1]]["output"] if native else frames[ctx[-1]]["grid"],
                        "prompt": prompt, "representation_errors": errors,
                        "example_frame_ids": [] if native else demo_ids,
                        "input_frame_ids": input_ids, "target_frame_id": target_id,
                        "past_actions": window["past_actions"], "future_actions": window["future_actions"][:h],
                        "changed": window["changed"][str(h)],
                        "seen_in_train": window["target_seen_in_train"][str(h)],
                        "context_seen_in_train": window["context_seen_in_train"],
                        "target_grid_sha256": frames[target_id]["grid_sha256"],
                        "native_static": values[ctx[-1]]["output"] == values[target_id]["output"] if native else None}
                    if mode == "copy":
                        job["preset_response"] = preset(frames[ctx[-1]]["grid"])
                    yield job

        def reconstruct(mode, run=None, group=None):
            artifact = group["artifact"] if group else None
            values = rendered(mode, artifact)
            demos = [(values[f]["output"], frames[f]["grid"]) for f in demo_ids]
            queries = dataset["reconstruction_queries"]
            order = list(range(len(queries)))
            random.Random(f"probe-shuffle:{manifest['config']['sample_seed']}:{game}").shuffle(order)
            donors = {idx: order[(j + 1) % len(order)] for j, idx in enumerate(order)}
            if mode == "shuffled" and len(queries) < 2:
                raise ValueError("shuffled control requires at least two queries")
            training_grids = Counter(frames[fid]["grid"] for item in dataset["training_target_positions"]["train"]
                                     for fid in item["frame_ids"])
            train_mode = max(sorted(training_grids), key=training_grids.get)
            audit = collision_summary([values[q["frame_id"]]["output"] for q in queries],
                                      [frames[q["frame_id"]]["grid"] for q in queries])
            for i, query in enumerate(queries):
                fid = query["frame_id"]
                donor = queries[donors[i]] if mode == "shuffled" else query
                donor_id = donor["frame_id"]
                errors = errors_for(values, [donor_id, *demo_ids])
                z = values[donor_id]["output"]
                job = {**metadata(mode, run, group), "kind": "reconstruction", "horizon": 0,
                    "query_id": query["id"], "user_id": query["user_id"],
                    "drive_id": query["drive_id"], "row_index": query["row_index"],
                    "knowledge_sha256": None, "target_space": "raw", "target": frames[fid]["grid"],
                    "prompt": reconstruction_prompt(z, demos, contract),
                    "representation_errors": errors,
                    "input_frame_ids": [donor_id, *demo_ids], "target_frame_id": fid,
                    "target_grid_sha256": frames[fid]["grid_sha256"],
                    "seen_in_train": query["seen_in_train"], "representation": z,
                    "collision_audit": audit,
                    "control_donor_query_id": donor["id"] if mode == "shuffled" else None,
                    "control_changes_representation": z != values[fid]["output"] if mode == "shuffled" else None}
                if mode == "lossless_inverse":
                    job["preset_response"] = preset(decode_lossless(z))
                    job["prompt"] = ""
                elif mode == "train_mode":
                    job["preset_response"] = preset(train_mode)
                    job["prompt"] = ""
                yield job

        for mode in sorted(forward_modes - {"learned_raw", "learned_native"}):
            yield from forward(mode)
        for mode in sorted(reconstruction_modes - {"learned", "shuffled"}):
            yield from reconstruct(mode)
        for run in dataset["runs"]:
            if arms and run["arm"] not in arms:
                continue
            for group in artifact_groups(run, checkpoint_labels):
                for mode in sorted(forward_modes & {"learned_raw", "learned_native"}):
                    yield from forward(mode, run, group)
            for group in artifact_groups(run, checkpoint_labels, reconstruction=True):
                for mode in sorted(reconstruction_modes & {"learned", "shuffled"}):
                    yield from reconstruct(mode, run, group)


def prepare(manifest: dict, out: Path, config: dict, *, check_sources=True) -> dict:
    verify_manifest(manifest, check_sources=check_sources)
    runtime = manifest["config"]["perception_runtime"]
    if perception_runtime(runtime["frame_timeout_s"]) != runtime:
        raise ValueError("perception runtime changed; rebuild the manifest in the intended environment")
    known_arms = {r["arm"] for g in manifest["games"].values() for r in g["runs"]}
    known_checkpoints = {c["label"] for g in manifest["games"].values() for r in g["runs"] for c in r["checkpoints"]}
    if set(config["arms"]) - known_arms or set(config["checkpoints"]) - known_checkpoints:
        raise ValueError("unknown arm or checkpoint filter")
    if not config["forward_modes"] and not config["reconstruction_modes"]:
        raise ValueError("select at least one probe mode")
    if (config["max_prompt_chars"] < 1 or config["model"]["max_output_tokens"] < 1
            or config["model"]["timeout_s"] <= 0):
        raise ValueError("prompt, output and timeout budgets must be positive")
    # A protocol without the key is a pre-format run: keep the frozen JSON-only wording
    # and the strict parser so its saved numbers stay reproducible. Recorded on a copy,
    # so preparing a run never mutates the caller's config.
    grid_format = config.get("grid_format", DEFAULT_FORMAT)
    if grid_format is not None:
        get_format(grid_format)
    config = {**config, "grid_format": grid_format}
    # probing_formats.py defines the wire format and its parser, so an edit to it changes
    # what a given answer scores; it belongs in the protocol identity like the rest.
    implementation = {name: file_digest(Path(__file__).with_name(name)) for name in
                      ("probing_eval.py", "probing_common.py", "probing_perception.py",
                       "probing_formats.py")}
    protocol = {"schema_version": SCHEMA_VERSION, "manifest_sha256": manifest["manifest_sha256"],
                "config": config, "implementation": implementation,
                "reference_arm": manifest["reference_arm"], "games": sorted(manifest["games"]),
                "source_verification": check_sources,
                "replay_verified": all(g["audit"]["replay_status"] == "verified"
                                       for g in manifest["games"].values()),
                "evaluation_population": manifest["config"]["split"]}
    protocol["protocol_sha256"] = digest(protocol)
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("output directory belongs to a different protocol; choose a new directory")
    write_json(protocol_path, protocol)
    counts, request_keys = Counter(), set()
    max_chars, sum_chars = 0, 0
    tmp = out / "jobs.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        for job in study_jobs(manifest, cache_dir=out / "perception_cache",
                forward_modes=set(config["forward_modes"]),
                reconstruction_modes=set(config["reconstruction_modes"]),
                checkpoint_labels=set(config["checkpoints"]), examples=config["examples"],
                grid_format=grid_format,
                arms=set(config["arms"])):
            job["protocol_sha256"] = protocol["protocol_sha256"]
            job["job_id"] = digest(job)
            job["request_key"] = digest({"prompt": job["prompt"], "model": config["model"],
                "manifest": manifest["manifest_sha256"], "implementation": implementation})
            if len(job["prompt"]) > config["max_prompt_chars"]:
                raise ValueError(f"prompt exceeds character budget ({job['game']}, {job['mode']}); "
                                 "choose a supported common example/history budget")
            fh.write(json_text(job) + "\n")
            counts[f"{job['kind']}/{job['arm']}/{job['mode']}"] += 1
            if "preset_response" not in job and not job["representation_errors"]:
                if job["request_key"] not in request_keys:
                    sum_chars += len(job["prompt"])
                request_keys.add(job["request_key"])
                max_chars = max(max_chars, len(job["prompt"]))
    tmp.replace(out / "jobs.jsonl")
    cached = sum((out / "calls" / f"{key}.json").exists() for key in request_keys)
    summary = {"jobs": sum(counts.values()), "by_mode": dict(counts),
               "unique_model_requests": len(request_keys), "cached_requests": cached,
               "uncached_requests": len(request_keys) - cached,
               "max_prompt_characters": max_chars, "unique_prompt_characters": sum_chars,
               "jobs_sha256": file_digest(out / "jobs.jsonl"),
               "protocol_sha256": protocol["protocol_sha256"]}
    write_json(out / "prepared.json", summary)
    return summary


class OpenRouterClient:
    """Direct chat endpoint, as used by eval_test50_idfd; explicit caps and usage.

    No hidden retry/hedging layer and no process-global model configuration. The
    adapter sees no labels, raw targets, or source/user metadata.
    """
    def __init__(self, config: dict):
        from dotenv import load_dotenv
        import httpx
        load_dotenv(REPO / ".env")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required for execution")
        self.config = config
        self.http = httpx.AsyncClient(timeout=config["timeout_s"],
                                     headers={"Authorization": f"Bearer {key}"})

    async def __call__(self, prompt: str) -> dict:
        body = {"model": self.config["id"], "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config["temperature"], "max_tokens": self.config["max_output_tokens"],
                "usage": {"include": True}}
        if self.config["reasoning"]:
            body["reasoning"] = self.config["reasoning"]
        if self.config["provider"]:
            body["provider"] = {"order": self.config["provider"].split(","), "allow_fallbacks": False}
        response = await self.http.post("https://openrouter.ai/api/v1/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        choice = data["choices"][0]
        return {"response": choice["message"].get("content") or "", "usage": data.get("usage", {}),
                "response_id": data.get("id"), "returned_model": data.get("model"),
                "returned_provider": data.get("provider"), "finish_reason": choice.get("finish_reason"),
                "client_version": importlib.metadata.version("httpx")}

    async def close(self):
        await self.http.aclose()


def load_call(path: Path, key: str, prompt: str) -> dict | None:
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("request_key") != key or result.get("prompt_sha256") != digest(prompt):
        raise ValueError("call cache identity mismatch")
    return result


def prepared_protocol(out: Path) -> tuple[dict, dict]:
    prepared, protocol = read_json(out / "prepared.json"), read_json(out / "protocol.json")
    body = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    if (digest(body) != protocol.get("protocol_sha256")
            or prepared.get("protocol_sha256") != protocol["protocol_sha256"]):
        raise ValueError("prepared protocol identity changed")
    if file_digest(out / "jobs.jsonl") != prepared["jobs_sha256"]:
        raise ValueError("prepared jobs changed")
    return prepared, protocol


async def execute(out: Path, caller, *, concurrency=4, attempts=2, max_calls=1000,
                  retry_errors=False) -> dict:
    if concurrency < 1 or attempts < 1 or max_calls < 0:
        raise ValueError("invalid request budget")
    _prepared, protocol = prepared_protocol(out)
    pending, visited = [], set()
    # Store request keys only; stream the potentially large prompts a second time.
    for job in read_jsonl(out / "jobs.jsonl"):
        if "preset_response" in job or job["representation_errors"]:
            continue
        key = job["request_key"]
        if key in visited:
            continue
        visited.add(key)
        prior = load_call(out / "calls" / f"{key}.json", key, job["prompt"])
        if prior is None or (retry_errors and prior["status"] == "provider_error"):
            pending.append(key)
    if len(pending) > max_calls:
        raise ValueError(f"{len(pending)} uncached requests exceed --max-calls={max_calls}; "
                         "filter checkpoints/modes or set the intended budget")
    pending = set(pending)
    queue = asyncio.Queue(maxsize=concurrency * 2)
    counters = Counter()

    async def worker():
        while (job := await queue.get()) is not None:
            key, started = job["request_key"], time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            prior = load_call(out / "calls" / f"{key}.json", key, job["prompt"])
            history = list(prior.get("attempts", [])) if prior else []
            result = {"status": "provider_error"}
            for attempt in range(attempts):
                tick = time.monotonic()
                try:
                    response = await caller(job["prompt"])
                    if not isinstance(response.get("response"), str):
                        raise ValueError("client response must be text")
                    history.append({"status": "ok", "elapsed_s": time.monotonic() - tick,
                                    "usage": response.get("usage", {})})
                    result = {**response, "status": "completed"}
                    break
                except Exception as exc:
                    history.append({"status": "error", "type": type(exc).__name__,
                                    "error": str(exc)[:500], "elapsed_s": time.monotonic() - tick})
                    counters["failed_attempts"] += 1
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    if status_code is not None and 400 <= status_code < 500 and status_code not in {408, 409, 429}:
                        break
                    if attempt + 1 < attempts:
                        await asyncio.sleep(min(2 ** attempt, 4))
            result.update({"request_key": key, "prompt_sha256": digest(job["prompt"]),
                           "attempts": history, "elapsed_s": time.monotonic() - started,
                           "started_at": started_at,
                           "completed_at": datetime.now(timezone.utc).isoformat(),
                           "model_config": protocol["config"]["model"]})
            write_json(out / "calls" / f"{key}.json", result)
            counters[result["status"]] += 1
            if sum(counters[s] for s in ("completed", "provider_error")) % 25 == 0:
                print(f"completed={counters['completed']} provider_errors={counters['provider_error']}", flush=True)
            queue.task_done()

    async def produce():
        for job in read_jsonl(out / "jobs.jsonl"):
            if job["request_key"] in pending:
                pending.remove(job["request_key"])
                await queue.put(job)
        for _ in range(concurrency):
            await queue.put(None)

    # A persistence failure must also cancel a producer blocked on a full queue.
    async with asyncio.TaskGroup() as tasks:
        for _ in range(concurrency):
            tasks.create_task(worker())
        tasks.create_task(produce())
    write_json(out / "execution.json", dict(counters))
    return dict(counters)


def score_jobs(out: Path) -> dict:
    prepared_protocol(out)
    counts = Counter()
    tmp = out / "results.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        for job in read_jsonl(out / "jobs.jsonl"):
            call = load_call(out / "calls" / f"{job['request_key']}.json", job["request_key"], job["prompt"])
            if job["representation_errors"]:
                status, response = "perception_error", ""
            elif "preset_response" in job:
                status, response = "deterministic_baseline", job["preset_response"]
            elif call:
                status, response = call["status"], call.get("response", "")
            else:
                status, response = "pending", ""
            metrics = None
            if status not in {"pending", "provider_error"}:
                if job["target_space"] == "native":
                    metrics = native_scores(response, job["target"])
                else:
                    spec = job.get("grid_contract")
                    metrics = grid_scores(response, job["target"], start=job.get("start"),
                        background=job["background"],
                        tag="reconstruction" if job["kind"] == "reconstruction" else "next_state",
                        contract=GridContract.from_dict(spec) if spec else None)
                if status == "perception_error":
                    metrics = {k: 0.0 if isinstance(v, (int, float)) else v for k, v in metrics.items()}
            row = {k: v for k, v in job.items() if k not in {"prompt", "preset_response"}}
            row.update({"status": status, "metrics": metrics,
                        "response": response, "call": call})
            fh.write(json_text(row) + "\n")
            counts[status] += 1
    tmp.replace(out / "results.jsonl")
    write_json(out / "coverage.json", dict(counts))
    return dict(counts)
