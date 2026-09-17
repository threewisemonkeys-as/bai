"""Build a frozen probe dataset from the saved human-data REx runs.

Uses the original build_data() for selection, while attaching source positions to
the loader's existing Transition objects. No model calls are made here.
"""
from __future__ import annotations

import contextlib
import csv
import importlib
import io
import json
import random
import re
import subprocess
from collections import Counter, defaultdict, deque
from pathlib import Path

from probing_common import (
    REPO, SCHEMA_VERSION, digest, file_digest, incumbent_checkpoints,
    observation_grid, parse_grid, perceive, perception_runtime, read_json, read_jsonl,
)

csv.field_size_limit(10_000_000)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def terminal(row: dict) -> bool:
    return row.get("Done", "").strip().lower() in {"true", "1"}


def edge_error(rows: list[dict], i: int, whitelist: set[str]) -> str | None:
    """A terminal last observation may be a target, but never an outgoing state."""
    if terminal(rows[i]):
        return "terminal"
    if int(rows[i + 1]["Step"]) != int(rows[i]["Step"]) + 1:
        return "step_gap_or_reset"
    action = rows[i].get("Action", "").strip()
    if not action or action.split()[0] not in whitelist:
        return "missing_or_unknown_action"
    if not re.fullmatch(r"(?:left|right|up|down|noop|click \d+ \d+)", action):
        return "malformed_action"
    if action.startswith("click "):
        grid = parse_grid(observation_grid(rows[i]["Observation"])[1])
        r, c = map(int, action.split()[1:])
        if r >= len(grid) or c >= len(grid[0]):
            return "out_of_bounds_click"
    return None


def eligible_starts(rows: list[dict], context_k: int, horizon: int,
                    whitelist: set[str]) -> tuple[list[int], dict]:
    if context_k < 0 or horizon < 1:
        raise ValueError("history must be nonnegative and horizon positive")
    reasons = Counter()
    starts = []
    edges = [edge_error(rows, i, whitelist) for i in range(len(rows) - 1)]
    for t in range(context_k, len(rows) - horizon):
        invalid = next((reason for reason in edges[t - context_k:t + horizon] if reason), None)
        if invalid:
            reasons[invalid] += 1
        else:
            starts.append(t)
    return starts, dict(reasons)


def balanced_sample(items: list[dict], n: int, rng: random.Random,
                    *, separation: int = 0) -> list[dict]:
    """Round-robin users, then drives within each user; independent of P and scores."""
    grouped = defaultdict(lambda: defaultdict(list))
    for item in items:
        grouped[item["user_id"]][item["drive_id"]].append(item)
    users = sorted(grouped)
    rng.shuffle(users)
    queues = {}
    for user in users:
        drives = sorted(grouped[user])
        rng.shuffle(drives)
        queues[user] = deque()
        for drive in drives:
            values = grouped[user][drive]
            rng.shuffle(values)
            queues[user].append(deque(values))
    chosen, positions = [], defaultdict(list)
    users = deque(users)
    while users and len(chosen) < n:
        user = users.popleft()
        drives = queues[user]
        queue = drives.popleft()
        picked = None
        while queue:
            item = queue.popleft()
            if all(abs(item["row_index"] - old) >= separation
                   for old in positions[item["drive_id"]]):
                picked = item
                break
        if queue:
            drives.append(queue)
        if drives:
            users.append(user)
        if picked is not None:
            chosen.append(picked)
            positions[picked["drive_id"]].append(picked["row_index"])
    return chosen


def transition_content(tr) -> dict:
    return {"x_t": tr.x_t, "x_t1": tr.x_t1, "action": tr.action,
            "ctx_prev": tr.ctx_prev, "ctx_next": tr.ctx_next}


class ManifestBuilder:
    def __init__(self, *, repo=REPO, cache_dir: Path | None = None, replay=True,
                 context_k=9, horizons=(1, 2, 4, 8), windows=50, reconstruction=100,
                 examples=8, sample_seed=0, split="test", nonoverlapping=False,
                 perception_timeout=2.0):
        self.repo = Path(repo).resolve()
        self.cache_dir = cache_dir
        self.replay = replay
        self.timeout = perception_timeout
        self.sources = {}
        self.config = {"context_k": context_k, "horizons": sorted(set(horizons)),
                       "windows_per_game": windows, "reconstruction_per_game": reconstruction,
                       "examples": examples, "sample_seed": sample_seed, "split": split,
                       "sampling": "user_then_drive_round_robin",
                       "nonoverlapping": nonoverlapping, "replay": replay,
                       "perception_runtime": perception_runtime(perception_timeout)}
        if (context_k < 0 or not horizons or min(horizons) < 1 or windows < 1
                or reconstruction < 1 or examples < 0 or perception_timeout <= 0
                or split not in {"train", "test"}):
            raise ValueError("invalid probe dataset configuration")

    def relative(self, path: Path) -> str:
        path = path.resolve()
        return str(path.relative_to(self.repo)) if path.is_relative_to(self.repo) else str(path)

    def source(self, path: Path) -> str:
        name = self.relative(path)
        if name not in self.sources:
            self.sources[name] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
        return name

    def rebuild(self, run_dir: Path):
        import rexpure_optimize as optimizer
        import invdyn_core as core

        cmd = read_json(run_dir / "launch.json")["cmd"]
        first = next(i for i, arg in enumerate(cmd) if str(arg).endswith(".py"))
        args = optimizer.build_parser().parse_args([str(x) for x in cmd[first + 1:]])
        if args.keep_obs_metadata or args.collapse_action_params:
            raise ValueError("v1 requires stripped metadata and fully parameterized actions")
        original = optimizer.load_transitions

        def tracked_loader(dirs, whitelist, context_k=0):
            transitions = original(dirs, whitelist, context_k=context_k)
            positions = []
            # Match the original loader's iteration order, including its unsorted glob.
            for directory in dirs:
                for path in directory.glob("episode_*/trajectory.csv"):
                    self.source(path)
                    rows = read_rows(path)
                    for i, (row, nxt) in enumerate(zip(rows, rows[1:])):
                        action = row.get("Action", "").strip()
                        obs, following = row.get("Observation", ""), nxt.get("Observation", "")
                        if terminal(row) or not action or not obs.strip() or not following.strip():
                            continue
                        if whitelist and action.split()[0] not in whitelist:
                            continue
                        positions.append((path, i, row, nxt))
            if len(positions) != len(transitions):
                raise ValueError("source instrumentation differs from original loader")
            for tr, (path, i, row, nxt) in zip(transitions, positions):
                if (tr.x_t, tr.action, tr.x_t1) != (row["Observation"], row["Action"].strip(),
                                                  nxt["Observation"]):
                    raise ValueError("instrumented source order differs from original loader")
                tr.probe_source = {"csv": str(path), "row_index": i, "step": int(row["Step"])}
            return transitions

        optimizer.load_transitions = tracked_loader
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                data = optimizer.build_data(args, random.Random(args.seed))
        finally:
            optimizer.load_transitions = original
        run_state = run_dir / f"rexpure_run_seed{args.seed}" / "resume_state.json"
        state = read_json(run_state)
        old = core._train_fingerprint(data[0])
        if not state.get("train_fingerprint") or old != state["train_fingerprint"]:
            raise ValueError(f"saved train fingerprint mismatch in {run_dir}")
        self.source(run_dir / "launch.json")
        self.source(run_state)
        return args, data, state

    def check_batches(self, run_dir: Path, seed: int, train: list, candidates: list) -> dict:
        path = run_dir / f"rexpure_run_seed{seed}" / "resume_batches.jsonl"
        self.source(path)
        seen = set()
        expected = [{"x_t": i["tr"].x_t, "x_t1": i["tr"].x_t1,
                     "prev_raw": [x for x, _ in i["tr"].ctx_prev],
                     "nxt_raw": [x for _, x in i["tr"].ctx_next]} for i in train]
        for batch in read_jsonl(path):
            idx = batch["cand_idx"]
            if idx in seen or idx not in range(len(candidates)):
                raise ValueError("duplicate or unknown candidate in saved batches")
            seen.add(idx)
            records = batch.get("trajectories", [])
            if len(records) != len(expected):
                raise ValueError("saved batch length differs from rebuilt training split")
            for j, (record, exp) in enumerate(zip(records, expected)):
                window = record.get("win")
                if window is None or any(window.get(k) != v for k, v in exp.items()):
                    raise ValueError(f"saved raw training context mismatch: {run_dir.name}, "
                                     f"candidate {idx}, item {j}")
                tr = train[j]["tr"]
                if ([pair[1] for pair in window["prev"]] != [a for _, a in tr.ctx_prev]
                        or [pair[0] for pair in window["nxt"]] != [a for a, _ in tr.ctx_next]):
                    raise ValueError(f"saved context actions differ from rebuilt training item {j}")
        if seen != set(range(len(candidates))):
            raise ValueError("saved batches do not cover all candidates")
        return {"candidates_checked": len(seen), "items_per_candidate": len(expected),
                "raw_contexts_match_saved_batches": True}

    def replay_drive(self, program: str, seed: int, grids: list[str], actions: list[str]) -> dict:
        if not self.replay:
            return {"status": "not_run"}
        from autumn_env import AutumnBenchEnvWrapper

        # The compiled interpreter and Autumn standard library determine replay
        # semantics; a repository revision or uv.lock alone does not identify them.
        for name in ("concrete_envs", "env_utils", "interpreter_module", "autumnstdlib"):
            module = importlib.import_module(f"python_examples.autumnbench.{name}")
            self.source(Path(module.__file__))
        env = AutumnBenchEnvWrapper(env_name=program, task_type="interactive", seed=seed,
                                    max_episode_steps=len(actions) + 8, render_mode="text")
        try:
            obs, _ = env.reset(seed=seed)
            current = observation_grid(obs["text"]["long_term_context"])[1]
            if current != grids[0]:
                raise ValueError(f"initial replay grid mismatch for {program}, seed {seed}")
            for i, action in enumerate(actions):
                obs, _, terminated, truncated, _ = env.step(action)
                current = observation_grid(obs["text"]["long_term_context"])[1]
                if current != grids[i + 1]:
                    raise ValueError(f"replay mismatch for {program}, seed {seed}, row {i + 1}")
                if (terminated or truncated) and i + 1 < len(actions):
                    raise ValueError("simulator terminated before recorded drive ended")
        finally:
            env.close()
        return {"status": "verified", "frames": len(grids)}

    def build_game(self, game: str, runs: list[tuple[str, Path]]) -> dict:
        from program_meta import background, resolve

        frames, drives, drive_rows, runs_out = {}, {}, {}, []
        train_exposure = set()
        split_signature = None
        dataset_identity = None
        target_positions = {"train": [], "test": []}

        def frame(observation):
            original, canonical = observation_grid(observation)
            fid = digest(original)
            frames.setdefault(fid, {"observation": original, "grid": canonical,
                                    "grid_sha256": digest(canonical)})
            return fid

        for arm_index, (label, run_dir) in enumerate(runs):
            print(f"[{game}/{label}] reconstructing split and checking saved batches", flush=True)
            args, data, state = self.rebuild(run_dir)
            train, test, _pool, _k, whitelist, transitions, _idn = data
            signature = digest([[transition_content(i["tr"]) for i in batch]
                                for batch in (train, test)])
            if split_signature is not None and signature != split_signature:
                raise ValueError(f"{game}: {label} does not share the reference split/context")
            split_signature = signature
            dataset = Path(args.run.split(",")[0]).parent
            manifest_path = dataset / "MANIFEST.json"
            self.source(manifest_path)
            dataset_manifest = read_json(manifest_path)
            identity = digest(dataset_manifest)
            if dataset_identity is not None and identity != dataset_identity:
                raise ValueError(f"{game}: arms reference different dataset manifests")
            dataset_identity = identity
            program = dataset_manifest["program"]
            program_path = resolve(program)
            self.source(program_path)
            if arm_index == 0:
                train_users = {d["user_id"] for d in dataset_manifest["drives"]["train"]}
                test_users = {d["user_id"] for d in dataset_manifest["drives"]["test"]}
                if train_users & test_users:
                    raise ValueError(f"{game}: train and test users overlap")
                for split, roots in (("train", args.context_source_run),
                                     ("test", args.test_context_source_run)):
                    if not roots:
                        raise ValueError("human probe datasets require full context-source drives")
                    for root in map(Path, roots.split(",")):
                        match = re.fullmatch(rf"{split}_d(\d+)", root.name)
                        if not match:
                            raise ValueError(f"unknown human drive name: {root}")
                        meta = dataset_manifest["drives"][split][int(match[1])]
                        paths = sorted(root.glob("episode_*/trajectory.csv"))
                        if len(paths) != 1:
                            raise ValueError("expected one reset segment per human source drive")
                        path = paths[0]
                        self.source(path)
                        rows = read_rows(path)
                        if len(rows) < 2 or int(rows[0]["Step"]) != 0:
                            raise ValueError("source drive must include its reset state")
                        ids = [frame(row["Observation"]) for row in rows]
                        actions = [row.get("Action", "").strip() for row in rows[:-1]]
                        errors = [edge_error(rows, i, whitelist) for i in range(len(rows) - 1)]
                        if any(e in {"step_gap_or_reset", "terminal", "missing_or_unknown_action",
                                     "malformed_action"} for e in errors):
                            raise ValueError(f"source drive is not a contiguous reset segment: {path}")
                        drive_id = f"{game}/{root.name}"
                        drive_rows[drive_id] = rows
                        drives[drive_id] = {"id": drive_id, "split": split,
                            "user_id": digest("probe-user:" + str(meta["user_id"])),
                            "session_id": digest({"user": meta["user_id"], "task": meta["task_id"],
                                                  "seed": meta["seed"], "segment": meta["seg_idx"]}),
                            "seed": int(meta["seed"]), "segment": meta["seg_idx"],
                            "source": self.relative(path), "frame_ids": ids,
                            "actions": actions, "steps": [int(row["Step"]) for row in rows],
                            "replay": self.replay_drive(program, int(meta["seed"]),
                                [frames[fid]["grid"] for fid in ids], actions)}

            # Verify selected sliced rows verbatim against their corresponding source drive.
            for split, batch in (("train", train), ("test", test)):
                for instance in batch:
                    tr = instance["tr"]
                    source = tr.probe_source
                    path = Path(source["csv"])
                    drive_id = f"{game}/{path.parent.parent.name}"
                    if drive_id not in drives or drives[drive_id]["split"] != split:
                        raise ValueError("slice does not map to a declared source drive")
                    rows = drive_rows[drive_id]
                    indices = [i for i, row in enumerate(rows) if int(row["Step"]) == source["step"]]
                    if len(indices) != 1 or indices[0] + 1 >= len(rows):
                        raise ValueError("ambiguous or missing source position")
                    pos = indices[0]
                    sliced = read_rows(path)
                    j = source["row_index"]
                    if sliced[j:j + 2] != rows[pos:pos + 2]:
                        raise ValueError(f"slice rows differ from drive at {path}")
                    if arm_index == 0:
                        target_positions[split].append({"drive_id": drive_id, "row_index": pos,
                            "slice": self.relative(path), "action": tr.action,
                            "frame_ids": [frame(tr.x_t), frame(tr.x_t1)]})

            cp_dir = run_dir / f"rexpure_run_seed{args.seed}"
            for name in ("candidates.jsonl", "process_log.jsonl"):
                self.source(cp_dir / name)
            candidates = list(read_jsonl(cp_dir / "candidates.jsonl"))
            candidates.sort(key=lambda c: c["idx"])
            process = list(read_jsonl(cp_dir / "process_log.jsonl"))
            batch_audit = self.check_batches(run_dir, args.seed, train, candidates)
            train_frame_ids = sorted({frame(x) for i in train for x in (i["tr"].x_t, i["tr"].x_t1)})

            def working(candidate):
                values = perceive(candidate.get("perception", ""),
                                  {fid: frames[fid]["observation"] for fid in train_frame_ids},
                                  timeout=self.timeout, cache_dir=self.cache_dir)
                errors = [v["error"] for v in values.values() if v["error"]]
                outputs = [v["output"] for v in values.values()]
                return {"working": not errors and all(outputs) and len(set(outputs)) > 1,
                        "n_frames": len(outputs), "errors": len(errors),
                        "unique_outputs": len(set(outputs)), "error_examples": sorted(set(errors))[:3]}

            checkpoints, inventory = incumbent_checkpoints(candidates, process, state["it"],
                                                           working, final_only=arm_index > 0)
            pp = run_dir / f"best_perception_rexpure_seed{args.seed}.py"
            kp = run_dir / f"best_beliefs_rexpure_seed{args.seed}.txt"
            self.source(pp)
            self.source(kp)
            shipped = [c["idx"] for c in candidates if c.get("perception", "") == pp.read_text()
                       and c.get("world_knowledge", "") == kp.read_text()]
            if shipped != [checkpoints[-1]["candidate_idx"]]:
                raise ValueError(f"{run_dir}: shipped P/K differs from unique training incumbent")
            selected = sorted({c["candidate_idx"] for c in checkpoints})
            artifacts = {}
            for idx in selected:
                c = candidates[idx]
                code, knowledge = c.get("perception", ""), c.get("world_knowledge", "")
                artifacts[str(idx)] = {"perception": code, "knowledge": knowledge,
                                      "perception_sha256": digest(code),
                                      "knowledge_sha256": digest(knowledge)}
            runs_out.append({"arm": label, "run_dir": self.relative(run_dir), "seed": args.seed,
                "original_train_fingerprint": state["train_fingerprint"],
                "split_content_sha256": signature, "batch_audit": batch_audit,
                "checkpoints": checkpoints, "inventory": inventory, "artifacts": artifacts,
                "training_objective": {"fd_scorer": args.fd_scorer,
                    "contrastive_fd": args.contrastive_fd, "composite": args.composite,
                    "no_id": args.no_id, "no_beliefs": args.no_beliefs,
                    "no_perception": args.no_perception},
                "evaluator": {"model": args.task_model, "client": args.client,
                    "provider": args.task_provider_order or "",
                    "reasoning": json.loads(args.task_reasoning_json or "{}")},
                "training_cost": state.get("cost", {})})
            for i in train:
                tr = i["tr"]
                exposed = [tr.x_t, tr.x_t1] + [x for x, _ in tr.ctx_prev]
                exposed += [x for _, x in tr.ctx_next] + list(i.get("cfd_options", []))
                train_exposure.update(frames[frame(x)]["grid_sha256"] for x in exposed)
            # Conservative: every training-pool frame was eligible to be a decoy.
            train_exposure.update(frames[frame(x)]["grid_sha256"] for tr in transitions
                                  for x in (tr.x_t, tr.x_t1))

        sample_seed = self.config["sample_seed"]
        max_h = max(self.config["horizons"])
        k, split = self.config["context_k"], self.config["split"]
        possible, exclusions, state_queries = [], {}, []
        for drive_id, drive in sorted(drives.items()):
            if drive["split"] != split:
                continue
            starts, invalid = eligible_starts(drive_rows[drive_id], k, max_h, whitelist)
            exclusions[drive_id] = invalid
            for t in starts:
                possible.append({"drive_id": drive_id, "user_id": drive["user_id"], "row_index": t})
            for t in range(len(drive["frame_ids"])):
                state_queries.append({"drive_id": drive_id, "user_id": drive["user_id"], "row_index": t})
        chosen = balanced_sample(possible, self.config["windows_per_game"],
                                 random.Random(f"probe-windows:{sample_seed}:{game}"),
                                 separation=k + max_h + 1 if self.config["nonoverlapping"] else 0)
        if not chosen:
            raise ValueError(f"no eligible windows for {game}")
        windows = []
        for item in chosen:
            d, t = drives[item["drive_id"]], item["row_index"]
            ctx, future = d["frame_ids"][t - k:t + 1], d["frame_ids"][t + 1:t + max_h + 1]
            windows.append({**item, "id": digest({"game": game, **item, "k": k, "h": max_h}),
                            "context_frame_ids": ctx, "past_actions": d["actions"][t - k:t],
                            "future_frame_ids": future, "future_actions": d["actions"][t:t + max_h],
                            "context_seen_in_train": any(frames[f]["grid_sha256"] in train_exposure for f in ctx),
                            "target_seen_in_train": {str(h): frames[future[h - 1]]["grid_sha256"] in train_exposure
                                                     for h in self.config["horizons"]},
                            "changed": {str(h): frames[ctx[-1]]["grid"] != frames[future[h - 1]]["grid"]
                                        for h in self.config["horizons"]}})
        queries = balanced_sample(state_queries, self.config["reconstruction_per_game"],
                                  random.Random(f"probe-reconstruction:{sample_seed}:{game}"))
        for q in queries:
            fid = drives[q["drive_id"]]["frame_ids"][q["row_index"]]
            q.update({"id": digest({"game": game, "reconstruction": q.copy()}), "frame_id": fid,
                      "seen_in_train": frames[fid]["grid_sha256"] in train_exposure})
        # Freeze demonstrations using training targets only, without consulting test queries.
        demo_by_grid = {}
        for item in target_positions["train"]:
            for fid in item["frame_ids"]:
                demo_by_grid.setdefault(frames[fid]["grid_sha256"], fid)
        demo_ids = [demo_by_grid[key] for key in sorted(demo_by_grid)]
        random.Random(f"probe-examples:{sample_seed}:{game}").shuffle(demo_ids)
        if len(demo_ids) < self.config["examples"]:
            raise ValueError(f"{game}: only {len(demo_ids)} unique training states for demonstrations")
        example_ids = demo_ids[:self.config["examples"]]
        test_grid_hashes = {frames[f]["grid_sha256"] for d in drives.values() if d["split"] == "test"
                            for f in d["frame_ids"]}
        return {"game": game, "program": {"id": program, "source": self.relative(program_path),
                                          "background": background(program)},
                "frames": frames, "drives": drives, "runs": runs_out, "windows": windows,
                "reconstruction_queries": queries, "example_frame_ids": example_ids,
                "training_target_positions": target_positions,
                "train_exposure_grid_hashes": sorted(train_exposure),
                "audit": {"split_content_sha256": split_signature,
                    "train_test_users_disjoint": True,
                    "exposure_scope": "saved raw batch contexts plus all training-pool decoy frames; "
                                      "reflection derives from these batches; external program reads not audited",
                    "test_action_vocabulary_shared_during_training": True,
                    "historical_test_use": "retrospective; not a fresh confirmation set",
                    "test_unique_grids_seen_in_train": len(test_grid_hashes & train_exposure),
                    "test_unique_grids": len(test_grid_hashes),
                    "eligible_windows": len(possible), "sampled_windows": len(windows),
                    "window_shortfall": max(0, self.config["windows_per_game"] - len(windows)),
                    "excluded_windows": exclusions,
                    "reconstruction_queries": len(queries),
                    "train_users": len({d["user_id"] for d in drives.values() if d["split"] == "train"}),
                    "test_users": len({d["user_id"] for d in drives.values() if d["split"] == "test"}),
                    "replay_status": "verified" if self.replay else "not_run"}}

    def build(self, roots: list[tuple[str, Path]], games: list[str], train_seed=1) -> dict:
        if not roots or len({label for label, _ in roots}) != len(roots):
            raise ValueError("provide distinct artifact-arm labels")
        payload = {"schema_version": SCHEMA_VERSION, "config": self.config,
                   "reference_arm": roots[0][0], "games": {}}
        for game in games:
            print(f"[{game}] auditing data, replay, and checkpoints", flush=True)
            runs = [(label, root / "rexpure" / f"{game}_s{train_seed}") for label, root in roots]
            payload["games"][game] = self.build_game(game, runs)
            a = payload["games"][game]["audit"]
            print(f"[{game}] {a['sampled_windows']} windows / {a['eligible_windows']} eligible; "
                  f"{a['test_users']} test users; replay={a['replay_status']}", flush=True)
        for path in ("offline_learning/validate.py", "offline_learning/rexpure_optimize.py",
                     "offline_learning/invdyn_core.py", "offline_learning/probing_common.py",
                     "offline_learning/probing_perception.py", "offline_learning/probing_manifest.py",
                     "autumn_env.py", "program_meta.py", "uv.lock",
                     "MARAProtocol/python_examples/autumnbench/concrete_envs.py",
                     "MARAProtocol/python_examples/autumnbench/env_utils.py"):
            p = self.repo / path
            if p.exists():
                self.source(p)
        try:
            payload["git_revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        except subprocess.CalledProcessError:
            payload["git_revision"] = None
        payload["sources"] = self.sources
        payload["manifest_sha256"] = digest(payload)
        return payload


def verify_manifest(manifest: dict, repo=REPO, *, check_sources=True) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported probe manifest version")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest(body) != manifest.get("manifest_sha256"):
        raise ValueError("probe manifest content hash mismatch")
    if check_sources:
        for name, record in manifest["sources"].items():
            path = Path(name) if Path(name).is_absolute() else Path(repo) / name
            if not path.is_file() or file_digest(path) != record["sha256"]:
                raise ValueError(f"manifest source changed or missing: {name}")
