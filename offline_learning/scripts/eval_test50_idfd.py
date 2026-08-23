#!/usr/bin/env python3
"""Evaluate test50 ID and refresh learned FD for newer completed artifacts.

The existing forward JSON supplies the exact test50 selection and raw FD rows.
This runner evaluates raw + learned ID for every game, refreshes learned FD only
for games overridden by completed_results.json, and persists exact prompts and
full responses. It checkpoints after every game and can resume safely.
"""
from __future__ import annotations

import os as _bos, sys as _bsys  # offline_learning/ on sys.path (flat-import the kept libs)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import argparse
import asyncio
import json
import logging
import os

import httpx
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_forward_modes as efm
import invdyn_core as go
import explore.mixed_improve as llm_backend
from forward_objective import textdiff_delta_f1
from invdyn_core import (
    DEFAULT_KNOWLEDGE, INV_WINDOW_TMPL, _extract_action,
    _inverse_transcript, build_window, exact_match_f1,
    predict_action_from_window, predict_next_state_from_window,
)
from validate import load_transitions, make_config
from validate_beliefs import PREDICT_TMPL as ID_TWO_STATE_TMPL, make_choices


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _reasoning_config():
    """Reasoning cap from env: EVAL_REASONING_EFFORT or EVAL_REASONING_MAX_TOKENS."""
    effort = os.environ.get("EVAL_REASONING_EFFORT")
    if effort:
        return {"effort": effort}
    rmax = os.environ.get("EVAL_REASONING_MAX_TOKENS")
    if rmax:
        return {"max_tokens": int(rmax)}
    return None


async def _openrouter_direct_call(config, prompt, reasoning):
    """litellm drops OpenRouter's `reasoning` config on both the responses and
    chat paths (verified 2026-07-23), so capped calls bypass it entirely."""
    body = {
        "model": llm_backend._get_model_name(config).removeprefix("openrouter/"),
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": reasoning,
        "usage": {"include": True},
    }
    pin = llm_backend._openrouter_provider_pin()
    if pin:
        body["provider"] = pin
    if llm_backend._META_TEMPERATURE is not None:
        body["temperature"] = llm_backend._META_TEMPERATURE
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    async with asyncio.timeout(1860):
        async with httpx.AsyncClient(timeout=1800) as client:
            resp = await client.post(OPENROUTER_CHAT_URL, json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"openrouter: {data['error']}")
    text = data["choices"][0]["message"].get("content") or ""
    logging.info("LLM response:\n%s", text)
    cost = float((data.get("usage") or {}).get("cost") or 0.0)
    return text, cost


async def evaluator_llm_call(config, prompt, images=None):
    """Use one provider attempt per outer retry, avoiding extreme tail calls."""
    if llm_backend._MOCK_MODE:
        return await llm_backend._llm_call(config, prompt, images=images)
    model_name = llm_backend._get_model_name(config)
    reasoning = _reasoning_config()
    if reasoning and not images and model_name.startswith("openrouter/"):
        logging.info("LLM prompt (direct, reasoning=%s):\n%s", reasoning, prompt)
        return await _openrouter_direct_call(config, prompt, reasoning)
    input_data = llm_backend.build_llm_input(prompt, images=images)
    logging.info("LLM prompt (images=%d):\n%s", len(images or []), prompt)
    api_kwargs = {
        "model": model_name,
        "input": input_data,
        "num_retries": 0,
        "timeout": 1800,
    }
    if model_name.startswith("openrouter/"):
        api_kwargs["extra_body"] = {"usage": {"include": True}}
        pin = llm_backend._openrouter_provider_pin()
        if pin:  # top-level kwarg: litellm drops a provider key inside extra_body
            api_kwargs["provider"] = pin
    if llm_backend._META_TEMPERATURE is not None:
        api_kwargs["temperature"] = llm_backend._META_TEMPERATURE
    async with asyncio.timeout(1860):
        response = await llm_backend.litellm.aresponses(**api_kwargs)
    text = llm_backend.extract_llm_response_text(response)
    logging.info("LLM response:\n%s", text)
    cost = llm_backend._get_response_cost(response, config.client.model_id)
    return text, cost


# The imported window helpers resolve their caller from gepa_optimize globals.
_llm_call = evaluator_llm_call
go._llm_call = evaluator_llm_call


def load_overrides(path: Path, games: set[str]) -> dict[str, str]:
    payload = json.loads(path.read_text())
    out = {}
    for row in payload.get("completed_runs", []):
        game, run_dir = row.get("game"), Path(row.get("run_dir", ""))
        needed = [
            run_dir / "best_perception_gepa_seed1.py",
            run_dir / "best_beliefs_gepa_seed1.txt",
        ]
        if game in games and all(p.exists() for p in needed):
            out[game] = str(run_dir)
    return out


def game_action_pool(game, data_root, test_dir, context_k):
    action_csv, keep_params, _mmc, _rounds = efm.GAMES[game]
    whitelist = set(filter(None, action_csv.split(","))) or None
    transitions = load_transitions(
        [data_root / game / "train", data_root / game / test_dir],
        whitelist, context_k=context_k,
    )
    if not keep_params:
        efm.collapse_actions(transitions)
    return sorted({tr.action for tr in transitions})


def fixed_choices(game, transitions, pool, seed, k):
    rng = random.Random(f"test50-id:{seed}:{game}")
    return [make_choices(tr.action, pool, k, rng) for tr in transitions]


def format_choices(choices):
    return "\n".join(f"- {choice}" for choice in choices)


def raw_id_prompt(tr, choices):
    return ID_TWO_STATE_TMPL.format(
        beliefs="(empty)", z_t=tr.x_t[:6000] or "(empty)",
        z_t1=tr.x_t1[:6000] or "(empty)",
        choices=format_choices(choices),
    )


def learned_id_prompt(win, beliefs, choices):
    return INV_WINDOW_TMPL.format(
        beliefs=beliefs.strip() or "(empty)",
        default_knowledge=DEFAULT_KNOWLEDGE,
        transcript=_inverse_transcript(win),
        choices=format_choices(choices),
    )


async def retry_raw_id(cfg, sem, prompt, choices, attempts):
    errors, total_cost, text = [], 0.0, ""
    for attempt in range(1, attempts + 1):
        try:
            async with sem:
                text, cost = await _llm_call(cfg, prompt)
            text = text or ""
            total_cost += float(cost or 0.0)
            if text.strip():
                return _extract_action(text, choices), text, total_cost, errors
            errors.append(f"attempt {attempt}: empty response")
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        if attempt < attempts:
            await asyncio.sleep(min(2**attempt, 8))
    return _extract_action(text, choices), text, total_cost, errors


async def retry_learned_id(cfg, sem, win, beliefs, choices, attempts):
    errors, total_cost = [], 0.0
    last = (None, "", 0.0, learned_id_prompt(win, beliefs, choices))
    for attempt in range(1, attempts + 1):
        try:
            last = await predict_action_from_window(
                cfg, win, beliefs, choices, sem
            )
            pred, response, cost, prompt = last
            total_cost += float(cost or 0.0)
            if (response or "").strip():
                return pred, response, total_cost, prompt, errors
            errors.append(f"attempt {attempt}: empty response")
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        if attempt < attempts:
            await asyncio.sleep(min(2**attempt, 8))
    pred, response, _cost, prompt = last
    return pred, response, total_cost, prompt, errors


async def retry_learned_fd(cfg, sem, win, action, beliefs, attempts):
    errors, total_cost = [], 0.0
    last = ("", 0.0, "", "")
    for attempt in range(1, attempts + 1):
        try:
            last = await predict_next_state_from_window(
                cfg, win, action, beliefs, sem
            )
            pred, cost, response, prompt = last
            total_cost += float(cost or 0.0)
            if (response or "").strip():
                return pred, total_cost, response, prompt, errors
            errors.append(f"attempt {attempt}: empty response")
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        if attempt < attempts:
            await asyncio.sleep(min(2**attempt, 8))
    pred, _cost, response, prompt = last
    return pred, total_cost, response, prompt, errors


async def evaluate_game(
    transitions, choices_by_item, artifact_dir, cfg,
    concurrency, attempts, refresh_fd,
):
    code = (artifact_dir / "best_perception_gepa_seed1.py").read_text()
    beliefs_path = artifact_dir / "best_beliefs_gepa_seed1.txt"
    beliefs = beliefs_path.read_text() if beliefs_path.exists() else ""
    windows = [build_window(code, tr) for tr in transitions]
    sem = asyncio.Semaphore(concurrency)

    async def inverse_one(idx):
        tr, choices = transitions[idx], choices_by_item[idx]
        win, perception_error = windows[idx]
        prompt = raw_id_prompt(tr, choices)
        raw_result, learned_result = await asyncio.gather(
            retry_raw_id(cfg, sem, prompt, choices, attempts),
            retry_learned_id(cfg, sem, win, beliefs, choices, attempts),
        )
        raw_pred, raw_response, raw_cost, raw_errors = raw_result
        lpred, lresponse, lcost, lprompt, lerrors = learned_result
        common = {
            "idx": idx, "truth": tr.action, "choices": choices,
            "raw_grid_start": efm.canonical_grid(tr.x_t),
            "raw_grid_target": efm.canonical_grid(tr.x_t1),
        }
        return [
            {
                **common, "mode": "raw", "z_t": tr.x_t[:6000],
                "z_t1": tr.x_t1[:6000], "pred": raw_pred,
                "prompt": prompt, "response": raw_response,
                "correct": raw_pred == tr.action,
                "exact": float(raw_pred == tr.action), "cost": raw_cost,
                "retry_errors": raw_errors, "perception_error": None,
            },
            {
                **common, "mode": "learned", "z_t": win["z_t"],
                "z_t1": win["z_t1"], "pred": lpred,
                "prompt": lprompt, "response": lresponse,
                "correct": lpred == tr.action,
                "exact": float(lpred == tr.action), "cost": lcost,
                "retry_errors": lerrors,
                "perception_error": perception_error,
            },
        ]

    async def forward_one(idx):
        tr = transitions[idx]
        win, perception_error = windows[idx]
        pred, cost, response, prompt, errors = await retry_learned_fd(
            cfg, sem, win, tr.action, beliefs, attempts
        )
        start, target = win["z_t"], win["z_t1"]
        return {
            "mode": "learned", "action": tr.action,
            "start": start, "target": target, "pred": pred,
            "prompt": prompt, "response": response,
            "changed": start.strip() != target.strip(),
            "exact": exact_match_f1(pred, target),
            "partial": textdiff_delta_f1(start, pred, target),
            "stale_exact": exact_match_f1(start, target),
            "stale_partial": textdiff_delta_f1(start, start, target),
            "perception_error": perception_error, "cost": cost,
            "retry_errors": errors, "idx": idx,
        }

    groups = await asyncio.gather(
        *(inverse_one(i) for i in range(len(transitions)))
    )
    id_rows = [row for group in groups for row in group]
    fd_rows = (
        await asyncio.gather(
            *(forward_one(i) for i in range(len(transitions)))
        )
        if refresh_fd else []
    )
    return id_rows, fd_rows, beliefs


def summarize_id(rows):
    return {
        "n": len(rows),
        "exact": sum(float(r["exact"]) for r in rows) / len(rows),
        "cost": sum(float(r.get("cost", 0.0)) for r in rows),
        "empty_responses": sum(not (r.get("response") or "").strip() for r in rows),
        "retried_items": sum(bool(r.get("retry_errors")) for r in rows),
        "action_counts": dict(Counter(r["truth"] for r in rows)),
    }


def make_payload(config, results, elapsed):
    aggregate = {}
    if results:
        for mode in ("raw", "learned"):
            rows = [
                row for result in results for row in result["rows"]
                if row["mode"] == mode
            ]
            aggregate[mode] = summarize_id(rows)
    return {
        "config": config, "elapsed_seconds": elapsed,
        "results": results, "aggregate": aggregate,
    }


def render_report(id_payload, forward):
    fd_by_game = {r["game"]: r for r in forward["results"]}
    lines = [
        "# Test50 inverse + forward dynamics: raw vs learned", "",
        "ID and FD use the same selected test50 transitions. Composite exact is",
        "the unweighted mean of ID exact and FD exact within a mode.", "",
        "| game | artifact | n | raw ID | raw FD | raw composite | learned ID | learned FD | learned composite | ID cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overrides = id_payload["config"]["artifact_overrides"]
    for result in id_payload["results"]:
        game, fd = result["game"], fd_by_game[result["game"]]["summary"]
        rid, lid = result["summary"]["raw"]["exact"], result["summary"]["learned"]["exact"]
        rfd, lfd = fd["raw"]["exact"], fd["learned"]["exact"]
        source = "newer" if game in overrides else "original"
        cost = result["summary"]["raw"]["cost"] + result["summary"]["learned"]["cost"]
        lines.append(
            f"| {game} | {source} | {result['test_n']} | {rid:.3f} | "
            f"{rfd:.3f} | {(rid+rfd)/2:.3f} | {lid:.3f} | {lfd:.3f} | "
            f"{(lid+lfd)/2:.3f} | USD {cost:.4f} |"
        )
    return "\n".join(lines) + "\n"


async def main_async(args):
    started = time.time()
    forward = json.loads(args.forward_json.read_text())
    games = list(forward["config"]["games"])
    order = {game: i for i, game in enumerate(games)}
    overrides = load_overrides(args.completed_results, set(games))
    fd_by_game = {r["game"]: r for r in forward["results"]}

    results = []
    if args.resume and args.id_out.with_suffix(".json").exists():
        prior = json.loads(args.id_out.with_suffix(".json").read_text())
        if prior.get("config", {}).get("artifact_overrides") == overrides:
            results = prior.get("results", [])
    completed = {r["game"] for r in results}
    config = {
        "games": games, "seed": forward["config"]["seed"],
        "test_n": forward["config"]["test_n"],
        "test_dir": forward["config"]["test_dir"],
        "context_k": forward["config"]["context_k"],
        "k_choices": args.k_choices,
        "task_model": forward["config"]["task_model"],
        "client": forward["config"]["client"],
        "data_root": forward["config"]["data_root"],
        "forward_source": str(args.forward_json),
        "artifact_overrides_source": str(args.completed_results),
        "artifact_overrides": overrides,
        "protocol": "same test50 selection and slice-bounded K=9 context as forward eval",
    }
    cfg = make_config(config["task_model"], config["client"])
    data_root = Path(config["data_root"])

    for game in games:
        if game in completed:
            print(f"[resume] {game}: already complete", flush=True)
            continue
        base = fd_by_game[game]
        artifact_dir = Path(overrides.get(game, base["artifact_dir"]))
        transitions, train_total, test_total = efm.reproduce_test_split(
            game, data_root, config["test_dir"], config["seed"],
            config["context_k"], config["test_n"],
        )
        raw_fd = {
            r["idx"]: r for r in base["rows"] if r["mode"] == "raw"
        }
        for idx, tr in enumerate(transitions):
            if raw_fd[idx]["action"] != tr.action:
                raise AssertionError(f"{game}: FD action mismatch at {idx}")
            if raw_fd[idx]["start"] != efm.canonical_grid(tr.x_t):
                raise AssertionError(f"{game}: FD raw-state mismatch at {idx}")
        pool = game_action_pool(
            game, data_root, config["test_dir"], config["context_k"]
        )
        choices = fixed_choices(
            game, transitions, pool, config["seed"], args.k_choices
        )
        refresh_fd = game in overrides
        print(
            f"[start] {game}: n={len(transitions)} "
            f"artifact={'newer' if refresh_fd else 'original'} "
            f"calls={len(transitions)*(3 if refresh_fd else 2)}",
            flush=True,
        )
        game_started = time.time()
        id_rows, learned_fd, beliefs = await evaluate_game(
            transitions, choices, artifact_dir, cfg,
            args.concurrency, args.attempts, refresh_fd,
        )
        if refresh_fd:
            raw_rows = [r for r in base["rows"] if r["mode"] == "raw"]
            base["rows"] = raw_rows + learned_fd
            base["artifact_dir"] = str(artifact_dir)
            base["summary"] = {
                "raw": efm.summarize(raw_rows),
                "learned": efm.summarize(learned_fd),
            }
            base["artifact_source"] = "newer_completed_results"
        else:
            base["artifact_source"] = "original_forward_eval"

        raw_rows = [r for r in id_rows if r["mode"] == "raw"]
        learned_rows = [r for r in id_rows if r["mode"] == "learned"]
        result = {
            "game": game, "seed": config["seed"],
            "artifact_dir": str(artifact_dir),
            "artifact_source": base["artifact_source"],
            "train_total": train_total, "test_total": test_total,
            "test_n": len(transitions), "beliefs": beliefs,
            "action_pool": pool,
            "summary": {
                "raw": summarize_id(raw_rows),
                "learned": summarize_id(learned_rows),
            },
            "rows": id_rows,
        }
        results.append(result)
        results.sort(key=lambda r: order[r["game"]])
        checkpoint = make_payload(config, results, time.time()-started)
        args.id_out.with_suffix(".json").write_text(
            json.dumps(checkpoint, indent=2)
        )
        # Persist refreshed FD rows alongside the ID checkpoint. Without this,
        # resuming after an interruption could skip a completed game ID while
        # silently losing that game newly evaluated FD rows.
        forward["aggregate"] = {
            mode: efm.aggregate(forward["results"], mode)
            for mode in ("raw", "learned")
        }
        forward["config"]["artifact_overrides"] = overrides
        forward["config"]["artifact_overrides_source"] = str(
            args.completed_results
        )
        forward["refresh_elapsed_seconds"] = time.time()-started
        args.forward_json.write_text(
            json.dumps(forward, indent=2, default=str)
        )
        print(
            f"[done] {game}: ID raw/learned="
            f"{result['summary']['raw']['exact']:.3f}/"
            f"{result['summary']['learned']['exact']:.3f} "
            f"FD learned={base['summary']['learned']['exact']:.3f} "
            f"seconds={time.time()-game_started:.1f}", flush=True,
        )

    results.sort(key=lambda r: order[r["game"]])
    forward["results"].sort(key=lambda r: order[r["game"]])
    forward["aggregate"] = {
        mode: efm.aggregate(forward["results"], mode)
        for mode in ("raw", "learned")
    }
    forward["config"]["artifact_overrides"] = overrides
    forward["config"]["artifact_overrides_source"] = str(args.completed_results)
    forward["refresh_elapsed_seconds"] = time.time()-started
    args.forward_json.write_text(json.dumps(forward, indent=2, default=str))
    args.forward_json.with_suffix(".md").write_text(efm.render_report(forward))

    payload = make_payload(config, results, time.time()-started)
    args.id_out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    report = render_report(payload, forward)
    args.id_out.with_suffix(".md").write_text(report)
    print(report, flush=True)
    print(
        f"wrote {args.id_out.with_suffix('.json')}, "
        f"{args.id_out.with_suffix('.md')}, and refreshed {args.forward_json}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--forward-json", type=Path,
        default=REPO / "logs/forward_eval_test50_raw_vs_learned.json",
    )
    ap.add_argument(
        "--completed-results", type=Path,
        default=REPO / "logs/autumn_gepa_fulltraj_aug30_completed_results/completed_results.json",
    )
    ap.add_argument(
        "--id-out", type=Path,
        default=REPO / "logs/id_eval_test50_raw_vs_learned",
    )
    ap.add_argument("--k-choices", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    args.id_out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
