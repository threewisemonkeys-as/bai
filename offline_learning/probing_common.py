"""Deterministic, network-free primitives shared by the probing CLIs.

The observed state is a rectangular colour-name grid. Recorder metadata is never
part of a prediction target. Prompt constructors deliberately have no target arg.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[1]


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def digest(value) -> str:
    data = value.encode() if isinstance(value, str) else json_text(value).encode()
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, value) -> None:
    """Atomic writes, including deterministic gzip headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json_text(value) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{i}") from exc


def parse_grid(value) -> list[list[str]]:
    grid = json.loads(value) if isinstance(value, str) else value
    if (not isinstance(grid, list) or not grid or not isinstance(grid[0], list)
            or not grid[0]):
        raise ValueError("expected a nonempty grid")
    width = len(grid[0])
    if any(not isinstance(row, list) or len(row) != width
           or any(not isinstance(c, str) or not c for c in row) for row in grid):
        raise ValueError("expected a rectangular colour-name grid")
    return grid


def observation_grid(observation: str) -> tuple[str, str]:
    """Return (original grid serialization, canonical grid); only inputs allow headers.

    Preserve the original spacing for P, matching strip_autumn_obs_metadata. Model
    outputs instead go through parse_prediction, which never extracts arbitrary
    grid fragments from surrounding explanations.
    """
    marker = "========== Start of Direct Observation =========="
    text = observation.split(marker, 1)[-1] if marker in observation else observation
    start = text.find("[[")
    if start < 0:
        raise ValueError("observation has no colour grid")
    grid, end = json.JSONDecoder().raw_decode(text[start:])
    parse_grid(grid)
    return text[start:start + end], json_text(grid)


def response_body(response: str, tag: str) -> str:
    text = response.strip()
    matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    if matches:
        if len(matches) != 1:
            raise ValueError(f"expected one <{tag}> answer")
        return matches[0].strip()
    if f"<{tag}>" in text or f"</{tag}>" in text:
        raise ValueError(f"incomplete <{tag}> answer")
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if not match:
            raise ValueError("malformed code fence")
        return match[1].strip()
    return text


def parse_prediction(response: str, tag="next_state") -> list[list[str]]:
    return parse_grid(response_body(response, tag))


def set_scores(pred: set, truth: set) -> dict:
    if not pred and not truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    hit = len(pred & truth)
    precision = hit / len(pred) if pred else 0.0
    recall = hit / len(truth) if truth else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * hit / (len(pred) + len(truth))}


def grid_scores(response: str, target: str, *, start: str | None = None,
                background: str = "black", tag="next_state", contract=None) -> dict:
    """Score a raw-grid answer.

    With a contract the answer is read in its wire format by that format's tolerant
    parser; without one this is the frozen strict JSON parse, so saved runs keep
    reproducing their published numbers. Either way the parsed grid is compared against
    the untouched target by the same metrics.
    """
    truth = parse_grid(target)
    keys = ["exact", "cell_accuracy", "foreground_precision", "foreground_recall",
            "foreground_f1"]
    if start is not None:
        keys += ["change_precision", "change_recall", "change_f1"]
    try:
        pred = (contract.parse(response, tag) if contract is not None
                else parse_prediction(response, tag))
        if (len(pred), len(pred[0])) != (len(truth), len(truth[0])):
            raise ValueError("predicted grid dimensions differ from target")
    except (ValueError, TypeError) as exc:
        return {**dict.fromkeys(keys, 0.0), "parse_error": str(exc)}
    result = {"exact": float(pred == truth), "parse_error": None,
              "cell_accuracy": sum(p == t for pr, tr in zip(pred, truth)
                                   for p, t in zip(pr, tr)) / (len(truth) * len(truth[0]))}

    def cells(grid, comparison):
        return {(r, c, val) for r, row in enumerate(grid) for c, val in enumerate(row)
                if val != comparison(r, c)}

    fg = set_scores(cells(pred, lambda r, c: background),
                    cells(truth, lambda r, c: background))
    result.update({f"foreground_{key}": val for key, val in fg.items()})
    if start is not None:
        initial = parse_grid(start)
        if (len(initial), len(initial[0])) != (len(truth), len(truth[0])):
            raise ValueError("start and target dimensions differ")
        delta = set_scores(cells(pred, lambda r, c: initial[r][c]),
                           cells(truth, lambda r, c: initial[r][c]))
        result.update({f"change_{key}": val for key, val in delta.items()})
    return result


def native_scores(response: str, target: str) -> dict:
    try:
        pred = response_body(response, "next_state")
    except ValueError as exc:
        return {"exact": 0.0, "parse_error": str(exc)}
    return {"exact": float(bool(pred) and pred.strip() == target.strip()),
            "parse_error": "empty prediction" if not pred else None}


def lossless_encoding(grid: str) -> str:
    """Generic run-length encoding, including every cell and the dimensions."""
    cells = parse_grid(grid)
    rows = []
    for row in cells:
        runs = []
        for colour in row:
            if runs and runs[-1][0] == colour:
                runs[-1][1] += 1
            else:
                runs.append([colour, 1])
        rows.append(runs)
    return json_text({"height": len(cells), "width": len(cells[0]), "row_runs": rows})


def decode_lossless(encoded: str) -> str:
    obj = json.loads(encoded)
    rows = [[colour for colour, n in runs for _ in range(n)] for runs in obj["row_runs"]]
    grid = parse_grid(rows)
    if (len(grid), len(grid[0])) != (obj["height"], obj["width"]):
        raise ValueError("incorrect run-length dimensions")
    return json_text(grid)


def collision_summary(outputs: list[str], targets: list[str]) -> dict:
    if len(outputs) != len(targets) or not outputs:
        raise ValueError("collision audit requires equally sized, nonempty lists")
    counts = defaultdict(Counter)
    for z, x in zip(outputs, targets):
        counts[z][x] += 1
    return {"n": len(outputs), "unique_outputs": len(counts),
            "unique_targets": len(set(targets)),
            "ambiguous_outputs": sum(len(c) > 1 for c in counts.values()),
            "empirical_reconstruction_bound": sum(max(c.values()) for c in counts.values())
            / len(outputs)}


def perception_runtime(timeout: float) -> dict:
    return {"python": sys.version, "hash_seed": "0", "frame_timeout_s": timeout,
            "worker_sha256": file_digest(Path(__file__).with_name("probing_perception.py"))}


def perceive(code: str, observations: dict[str, str], *, timeout=2.0,
             cache_dir: Path | None = None) -> dict[str, dict]:
    """Execute P in two fresh pinned subprocesses; cache only content-addressed results.

    Fresh exec namespaces per frame match validate.run_perceive. The second process
    detects nondeterminism on these inputs. These subprocesses are resource isolation,
    not a security sandbox for untrusted programs.
    """
    runtime = perception_runtime(timeout)
    key = digest({"code": code, "runtime": runtime})
    path = cache_dir / f"{key}.json.gz" if cache_dir else None
    saved = read_json(path) if path and path.exists() else {"key": key, "outputs": {}}
    if saved.get("key") != key:
        raise ValueError("perception cache identity mismatch")
    missing = {digest(obs): obs for obs in observations.values()
               if digest(obs) not in saved["outputs"]}
    if missing:
        payload = json_text({"code": code, "observations": missing, "timeout": timeout})
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        worker = Path(__file__).with_name("probing_perception.py")

        def run():
            try:
                proc = subprocess.run([sys.executable, str(worker)], input=payload,
                                      text=True, capture_output=True, env=env,
                                      timeout=10 + len(missing) * timeout)
                if proc.returncode:
                    raise RuntimeError(f"perception worker exited {proc.returncode}")
                values = json.loads(proc.stdout)
                if set(values) != set(missing):
                    raise ValueError("perception worker returned incorrect frame IDs")
                return values
            except (subprocess.TimeoutExpired, ValueError, RuntimeError) as exc:
                return {fid: {"output": "", "error": str(exc)} for fid in missing}

        first, second = run(), run()
        for fid in missing:
            if first[fid] != second[fid]:
                first[fid] = {"output": "", "error": "nondeterministic perception output"}
        saved["outputs"].update(first)
        if path:
            write_json(path, saved)
    return {fid: saved["outputs"][digest(obs)] for fid, obs in observations.items()}


def incumbent_checkpoints(candidates: list[dict], process: list[dict], last_iteration: int,
                         working, *, final_only=False) -> tuple[list[dict], list[dict]]:
    """REx argmax breaks ties toward the earliest admitted candidate, not latest."""
    by_id = {c["idx"]: c for c in candidates}
    if sorted(by_id) != list(range(len(candidates))):
        raise ValueError("candidate IDs must be unique and contiguous from zero")
    born = {0: 0}
    iterations = set()
    for row in process:
        it = int(row["i"])
        if it in iterations:
            raise ValueError(f"duplicate search iteration {it}")
        iterations.add(it)
        idx = row.get("new_idx")
        if idx is not None:
            if idx in born or idx not in by_id:
                raise ValueError("candidate admission mismatch")
            born[idx] = it
            if not math.isclose(float(row["new_score"]), float(by_id[idx]["train_score"]),
                                rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("candidate score differs from process log")
    if set(born) != set(by_id) or max(born.values()) > last_iteration:
        raise ValueError("incomplete candidate/iteration mapping")
    if iterations and max(iterations) > last_iteration:
        raise ValueError("process log extends beyond saved final iteration")

    def incumbent(it):
        eligible = [c for c in candidates if born[c["idx"]] <= it]
        return max(eligible, key=lambda c: (float(c["train_score"]), -c["idx"]))

    selected, statuses = [], {}

    def add(label, it, candidate):
        idx = candidate["idx"]
        if idx not in statuses:
            statuses[idx] = working(candidate)
        selected.append({"label": label, "iteration": it, "candidate_idx": idx,
                         "created_iteration": born[idx], "validity": statuses[idx]})

    if not final_only:
        for it in sorted(set(born.values())):
            c = incumbent(it)
            if c["idx"] not in statuses:
                statuses[c["idx"]] = working(c)
            if statuses[c["idx"]]["working"]:
                add("first_working", it, c)
                break
        if not selected:
            raise ValueError("no working incumbent on the training frames")
        for pct in (25, 50, 75):
            it = math.ceil(last_iteration * pct / 100)
            add(f"p{pct}", it, incumbent(it))
    add("final", last_iteration, incumbent(last_iteration))
    inventory = [{"idx": c["idx"], "created_iteration": born[c["idx"]],
                  "train_score": c["train_score"], "parents": c.get("parents", []),
                  "validity": statuses.get(c["idx"])} for c in candidates]
    return selected, inventory


SCHEMA_PROMPT = """States are rectangular grids of colour names. Grid rows are ordered
top to bottom and columns left to right, with zero-based (row, column) coordinates.
Actions are left, right, up, down, noop, or click ROW COL. Each action is one tick;
passive dynamics may advance even on noop. These are format conventions, not rules
about how objects move. Use only the supplied observations and knowledge to infer rules.
"""


def example_text(examples: list[tuple[str, str]], contract=None) -> str:
    """Worked representation/grid pairs, with the grids in the contract's wire format."""
    if not examples:
        return "No representation-to-grid examples are supplied."
    render = (lambda x: contract.render(parse_grid(x))) if contract else (lambda x: x)
    return "\n\n".join(f"EXAMPLE {i + 1}\nRepresentation:\n{z}\nRaw grid:\n{render(x)}"
                       for i, (z, x) in enumerate(examples))


def forward_prompt(states: list[str], past_actions: list[str], future_actions: list[str],
                   knowledge: str, examples: list[tuple[str, str]], *, native=False,
                   contract=None) -> str:
    """The forecast prompt. Without a contract this is the frozen JSON-only wording."""
    if len(states) != len(past_actions) + 1 or not future_actions:
        raise ValueError("forecast history/action alignment mismatch")
    if native and contract is not None:
        raise ValueError("native forecasts do not use a raw-grid contract")
    transcript = []
    for i, state in enumerate(states):
        t = i - len(past_actions)
        transcript.append(f"STATE t{t:+d}{' (CURRENT)' if t == 0 else ''}:\n{state}")
        if i < len(past_actions):
            transcript.append(f"Action to next state: {past_actions[i]}")
    if contract is not None:
        ask = ("Predict the complete raw grid after ALL actions, including unchanged "
               f"content, as {contract.contract_line}.\nReturn only your prediction inside "
               "<next_state> and </next_state>, with nothing else between the tags. "
               "Do not return a plan.")
    else:
        output = ("the complete feature string in EXACTLY the current representation's format"
                  if native else
                  "the complete raw grid as a JSON array of rows of colour strings")
        ask = (f"Predict {output} after ALL actions, including unchanged content.\n"
               "Return only <next_state>YOUR PREDICTION</next_state>. Do not return a plan.")
    return (f"Predict the state after the supplied {len(future_actions)} actions.\n"
            f"{SCHEMA_PROMPT}\nWORLD KNOWLEDGE:\n{knowledge or '(empty)'}\n\n"
            "REPRESENTATION EXAMPLES (training states):\n"
            f"{example_text(examples, contract)}\n\n"
            + "\n".join(transcript)
            + "\n\nFUTURE ACTIONS, in order:\n"
            + "\n".join(f"{i + 1}. {a}" for i, a in enumerate(future_actions))
            + "\n\n" + ask)


def reconstruction_prompt(representation: str, examples: list[tuple[str, str]],
                          contract=None) -> str:
    """The reconstruction prompt. Without a contract this is the frozen JSON-only wording."""
    if contract is not None:
        schema = contract.schema_line
        ask = (f"Return every cell of the grid, including background, as "
               f"{contract.contract_line}.\nPut only that inside <reconstruction> and "
               "</reconstruction>, with nothing else between the tags. Do not explain.")
    else:
        schema = ("The raw output format is a JSON array of rows of colour-name strings; "
                  "coordinates, when present, are zero-based (row, column).")
        ask = ("Return every cell, including background, in "
               "<reconstruction>COMPLETE JSON GRID</reconstruction>. Do not explain.")
    return ("Reconstruct the current observed raw grid from its representation.\n"
            f"{schema}\n\n"
            f"PAIRED TRAINING EXAMPLES:\n{example_text(examples, contract)}\n\n"
            f"QUERY REPRESENTATION:\n{representation}\n\n" + ask)
