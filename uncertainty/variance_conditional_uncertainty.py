"""Aggregate conditional uncertainty matrices across steps and compute, for
each (focus_question, target_question) pair, the variance of marginal
information gain across all steps where the pair appears.

Marginal information gain = unconditional_uncertainty - conditional_uncertainty
where both are scored on the same 1..5 scale. This cancels the model's
reading-noise about the baseline absolute uncertainty (i.e. step-to-step
swings in how settled the target feels on its own) and isolates how much
the focus question's resolution would actually reduce uncertainty.

Reads per-step ``conditional_uncertainty.json`` files produced by
``score_conditional_uncertainty.py`` and writes:
  - ``variance_conditional_uncertainty.json`` at the run root: aggregated data
  - ``variance_conditional_uncertainty_viewer.html`` at the run root: viewer

Questions are matched across steps by exact question text (since q_number
shifts as new questions get added/answered/deduped).

Usage:
    uv run uncertainty/variance_conditional_uncertainty.py <run_dir> [--episode 0] \\
        [--min-occurrences 2] [--metric marginal|conditional]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

from score_uncertainty import find_episode_dir, find_step_dirs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def collect_pair_observations(
    step_results: list[dict], metric: str
) -> dict:
    """Walk per-step matrices and bucket scores by (focus_text, target_text).

    Args:
        step_results: list of per-step conditional_uncertainty.json payloads.
        metric: which per-target value to summarize. One of:
            "marginal"     — info gain = unconditional - conditional (0..4)
            "conditional"  — conditional uncertainty (1..5)
            "unconditional"— unconditional uncertainty (1..5)

    Each observation includes all three values when present so the viewer
    can show context (e.g. mean unconditional alongside marginal variance).
    """
    questions: list[str] = []
    qid: dict[str, int] = {}

    def ensure_q(text: str) -> int:
        if text in qid:
            return qid[text]
        idx = len(questions)
        questions.append(text)
        qid[text] = idx
        return idx

    pairs: dict[tuple[int, int], list[dict]] = {}
    question_steps: dict[str, list[str]] = {}

    for step in step_results:
        if step.get("skipped"):
            continue
        step_name = step["step_name"]
        qs = step.get("questions", [])
        local_to_global: list[int] = []
        for q in qs:
            gi = ensure_q(q["question"])
            local_to_global.append(gi)
            question_steps.setdefault(q["question"], []).append(step_name)

        matrix = step.get("matrix", [])
        for row in matrix:
            f_local = row["focus_index"]
            f_global = local_to_global[f_local]
            cond_list = row.get("conditional") or row.get("scores") or []
            unc_list = row.get("unconditional") or []
            marg_list = row.get("marginal") or []
            n_local = len(cond_list)
            for j_local in range(n_local):
                if j_local == f_local:
                    continue
                cond_v = cond_list[j_local] if j_local < len(cond_list) else None
                unc_v = unc_list[j_local] if j_local < len(unc_list) else None
                marg_v = marg_list[j_local] if j_local < len(marg_list) else None
                if marg_v is None and unc_v is not None and cond_v is not None:
                    marg_v = max(0, unc_v - cond_v)

                if metric == "marginal":
                    primary = marg_v
                elif metric == "conditional":
                    primary = cond_v
                elif metric == "unconditional":
                    primary = unc_v
                else:
                    raise ValueError(f"unknown metric: {metric}")
                if primary is None:
                    continue
                t_global = local_to_global[j_local]
                obs = pairs.setdefault((f_global, t_global), [])
                obs.append(
                    {
                        "step_name": step_name,
                        "score": int(primary),
                        "unconditional": unc_v,
                        "conditional": cond_v,
                        "marginal": marg_v,
                        "step_focus_q_number": row.get("focus_q_number"),
                        "step_target_q_number": qs[j_local].get("q_number"),
                    }
                )

    return {
        "questions": questions,
        "question_id": qid,
        "pairs": pairs,
        "question_steps": question_steps,
    }


def summarize_pairs(
    collected: dict, min_occurrences: int
) -> list[dict]:
    questions = collected["questions"]
    rows: list[dict] = []
    for (f, t), obs in collected["pairs"].items():
        if len(obs) < min_occurrences:
            continue
        scores = [o["score"] for o in obs]
        mean = statistics.fmean(scores)
        var = statistics.pvariance(scores) if len(scores) >= 2 else 0.0
        rows.append(
            {
                "focus_index": f,
                "target_index": t,
                "focus_question": questions[f],
                "target_question": questions[t],
                "n": len(scores),
                "mean": mean,
                "variance": var,
                "stdev": var ** 0.5,
                "min": min(scores),
                "max": max(scores),
                "observations": obs,
            }
        )
    rows.sort(key=lambda r: (-r["variance"], -r["n"]))
    return rows


def summarize_focus_questions(pair_rows: list[dict]) -> list[dict]:
    """Per focus question: aggregate variance across its target pairs.

    Mean variance is computed weighted by number of observations.
    """
    by_focus: dict[int, list[dict]] = {}
    for row in pair_rows:
        by_focus.setdefault(row["focus_index"], []).append(row)
    out: list[dict] = []
    for f_idx, rows in by_focus.items():
        total_n = sum(r["n"] for r in rows)
        if total_n == 0:
            continue
        wmean_var = sum(r["variance"] * r["n"] for r in rows) / total_n
        out.append(
            {
                "focus_index": f_idx,
                "focus_question": rows[0]["focus_question"],
                "num_target_pairs": len(rows),
                "total_observations": total_n,
                "weighted_mean_variance": wmean_var,
                "max_variance": max(r["variance"] for r in rows),
                "max_variance_target": max(rows, key=lambda r: r["variance"])[
                    "target_question"
                ],
            }
        )
    out.sort(key=lambda r: -r["weighted_mean_variance"])
    return out


VIEWER_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Conditional Uncertainty Variance — __TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 0; }
  header { padding: 12px 16px; border-bottom: 1px solid #ccc;
           background: rgba(0,0,0,0.04); position: sticky; top: 0; z-index: 5; }
  h1 { font-size: 16px; margin: 0 0 6px 0; }
  .subtle { color: #666; font-size: 12px; }
  main { padding: 14px; }
  .controls { display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap;
              font-size: 12px; margin: 4px 0; }
  .controls label { display: inline-flex; gap: 6px; align-items: baseline; }
  input[type="number"] { width: 70px; }
  button { font-size: 12px; padding: 2px 8px; }

  .section { margin-top: 18px; }
  .section h2 { font-size: 14px; margin: 0 0 6px 0; }

  table { border-collapse: collapse; font-size: 12px;
          font-variant-numeric: tabular-nums; }
  th, td { border: 1px solid #ddd; padding: 4px 6px; text-align: center; }
  th.q-label, td.q-label { text-align: left; max-width: 340px;
                           white-space: nowrap; overflow: hidden;
                           text-overflow: ellipsis; cursor: pointer; }
  th.col-num { width: 32px; }

  /* Heatmap colored by variance (0..4) where 4 = max possible for a 1..5 scale
     (range of 4 around mean of 3). */
  td.var-cell { width: 30px; height: 24px; cursor: pointer; font-weight: 600; }
  td.var-diag { background: repeating-linear-gradient(45deg,
                #ddd, #ddd 4px, #eee 4px, #eee 8px); color: #888; }
  td.var-missing { background: #fafafa; color: #bbb; }
  /* variance buckets (low -> high) */
  td.v0 { background: #e9f6e9; color: #1e4f1e; }
  td.v1 { background: #b9e4b9; color: #143614; }
  td.v2 { background: #ffe18a; color: #5a3d05; }
  td.v3 { background: #ffb066; color: #4a2206; }
  td.v4 { background: #ff7a7a; color: #5a0a0a; }
  td.v5 { background: #c41a1a; color: #fff; }
  tr.selected th, tr.selected td { outline: 2px solid #2a73f5; }

  /* Per-pair detail */
  .detail { margin-top: 14px; padding: 10px;
            border: 1px solid #bbb; border-radius: 6px;
            background: rgba(0,0,0,0.02); }
  .detail h3 { margin: 0 0 6px 0; font-size: 14px; }
  .obs-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
             font-size: 12px; }
  .chip { display: inline-flex; align-items: baseline; gap: 4px;
          padding: 2px 6px; border-radius: 10px; font-weight: 600;
          font-variant-numeric: tabular-nums; }
  .chip .stepname { font-weight: 400; color: #555; font-size: 11px; }
  .chip-s0 { background: #e0e0e0; color: #555; }
  .chip-s1 { background: #c7e9b4; color: #143614; }
  .chip-s2 { background: #ffffbf; color: #5a4a05; }
  .chip-s3 { background: #fdae61; color: #4a2c0c; }
  .chip-s4 { background: #d7191c; color: white; }
  .chip-s5 { background: #99000d; color: white; }

  .pair-list { width: 100%; }
  .pair-list th, .pair-list td { text-align: left; padding: 4px 8px;
                                 border-bottom: 1px solid #eee; border-top: none;
                                 border-left: none; border-right: none; }
  .pair-list th { background: rgba(0,0,0,0.04); position: sticky; top: 0; }
  .pair-list .num { text-align: right; font-variant-numeric: tabular-nums; }
  .pair-list tr.hover:hover { background: rgba(0,0,0,0.04); cursor: pointer; }

  details { margin: 6px 0; }
  details > summary { cursor: pointer; }
  pre { white-space: pre-wrap; background: rgba(0,0,0,0.05);
        padding: 8px; border-radius: 4px; max-height: 50vh; overflow: auto;
        font-size: 12px; }
  .legend { display: inline-flex; gap: 8px; align-items: center; font-size: 12px;
            margin-left: 16px; }
  .legend .swatch { display: inline-block; width: 14px; height: 14px;
                    border: 1px solid #888; }
</style>
</head>
<body>
<header>
  <h1>Variance across steps — <span id="title">__TITLE__</span></h1>
  <div class="subtle" id="metric-banner">
    For each (focus → target) pair, the chosen per-pair score is observed in
    every step where both questions appear. Cells / variance values below
    summarize that distribution across steps.
  </div>
  <div class="controls">
    <label>min occurrences:
      <input type="number" id="minOcc" value="2" min="2" step="1" />
    </label>
    <label>min focus appearances:
      <input type="number" id="minFocus" value="2" min="1" step="1" />
    </label>
    <button id="apply">apply</button>
    <span class="legend">
      <span>variance:</span>
      <span class="swatch" style="background:#e9f6e9"></span>0
      <span class="swatch" style="background:#b9e4b9"></span>~0.3
      <span class="swatch" style="background:#ffe18a"></span>~0.8
      <span class="swatch" style="background:#fdae61"></span>~1.5
      <span class="swatch" style="background:#ff7a7a"></span>~2.5
      <span class="swatch" style="background:#c41a1a"></span>≥4
    </span>
  </div>
</header>
<main id="content"></main>

<script>
const DATA = __DATA_JSON__;

function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "style") e.setAttribute("style", v);
      else if (k.startsWith("on")) e[k] = v;
      else e.setAttribute(k, v);
    }
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function varBucket(v) {
  // 0 .. 4 mapped to 0..5 buckets
  if (v <= 0.05) return 0;
  if (v <= 0.5) return 1;
  if (v <= 1.0) return 2;
  if (v <= 2.0) return 3;
  if (v <= 3.5) return 4;
  return 5;
}

function buildIndex() {
  // pair_key -> pair row
  const m = new Map();
  for (const r of DATA.pair_rows) {
    m.set(r.focus_index + "->" + r.target_index, r);
  }
  return m;
}

const PAIR_BY_KEY = buildIndex();

function renderHeatmap(container, minOcc, includedFocusIndices, includedTargetIndices) {
  // Build heatmap restricted to the given focus + target sets
  const focusList = includedFocusIndices;
  const targetList = includedTargetIndices;
  const table = el("table");
  const thead = el("thead");
  const headRow = el("tr",
    null,
    el("th", null, "focus \\ target"),
  );
  for (const t of targetList) {
    headRow.appendChild(el("th", {
      class: "col-num",
      title: DATA.questions[t],
    }, `Q${t + 1}`));
  }
  headRow.appendChild(el("th", { class: "col-num" }, "row mean var"));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const f of focusList) {
    const tr = el("tr", { "data-focus": String(f) });
    tr.appendChild(el("th", {
      class: "q-label",
      title: DATA.questions[f],
    }, `Q${f + 1}: ${DATA.questions[f]}`));

    const rowVariances = [];
    for (const t of targetList) {
      if (f === t) {
        tr.appendChild(el("td", { class: "var-diag" }, "·"));
        continue;
      }
      const r = PAIR_BY_KEY.get(f + "->" + t);
      if (!r || r.n < minOcc) {
        tr.appendChild(el("td", { class: "var-missing",
          title: r ? `only ${r.n} observation(s)` : "no observations" }, "—"));
        continue;
      }
      rowVariances.push(r.variance);
      const b = varBucket(r.variance);
      tr.appendChild(el("td", {
        class: `var-cell v${b}`,
        title: `focus Q${f + 1} → target Q${t + 1}\n` +
               `n=${r.n}, mean=${r.mean.toFixed(2)}, var=${r.variance.toFixed(2)}\n` +
               `range: [${r.min}, ${r.max}]`,
        onclick: () => showPair(r),
      }, r.variance.toFixed(2)));
    }
    if (rowVariances.length === 0) {
      tr.appendChild(el("td", { class: "var-missing" }, "—"));
    } else {
      const mv = rowVariances.reduce((a, b) => a + b, 0) / rowVariances.length;
      tr.appendChild(el("td", { style: "font-weight:400;color:#444;" }, mv.toFixed(2)));
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);
}

function showPair(r) {
  const wrap = document.getElementById("detail-wrap");
  wrap.innerHTML = "";
  const card = el("div", { class: "detail" });
  card.appendChild(el("h3", null,
    `Focus Q${r.focus_index + 1}  →  Target Q${r.target_index + 1}`));
  card.appendChild(el("div", { class: "subtle" }, `focus: ${r.focus_question}`));
  card.appendChild(el("div", { class: "subtle" }, `target: ${r.target_question}`));
  card.appendChild(el("div", { class: "subtle",
    style: "margin-top:6px;" },
    `n=${r.n} · mean=${r.mean.toFixed(2)} · var=${r.variance.toFixed(2)} ` +
    `· stdev=${r.stdev.toFixed(2)} · range=[${r.min}, ${r.max}]`));

  const obsSorted = [...r.observations].sort(
    (a, b) => a.step_name.localeCompare(b.step_name));
  const obsRow = el("div", { class: "obs-row", style: "margin-top:6px;" });
  for (const o of obsSorted) {
    const sub = (o.unconditional != null && o.conditional != null)
      ? ` (u=${o.unconditional}, c=${o.conditional})`
      : "";
    obsRow.appendChild(el("span", { class: `chip chip-s${o.score}` },
      String(o.score),
      el("span", { class: "stepname" }, o.step_name + sub)));
  }
  card.appendChild(obsRow);
  wrap.appendChild(card);
}

function renderPairList(container, rows) {
  const table = el("table", { class: "pair-list" });
  const head = el("thead");
  head.appendChild(el("tr", null,
    el("th", { class: "num" }, "#"),
    el("th", null, "focus question"),
    el("th", null, "target question"),
    el("th", { class: "num" }, "n"),
    el("th", { class: "num" }, "mean"),
    el("th", { class: "num" }, "var"),
    el("th", { class: "num" }, "range"),
    el("th", null, "scores by step"),
  ));
  table.appendChild(head);
  const tbody = el("tbody");
  rows.forEach((r, i) => {
    const obsSorted = [...r.observations].sort(
      (a, b) => a.step_name.localeCompare(b.step_name));
    const chips = el("div", { class: "obs-row" });
    for (const o of obsSorted) {
      const stepShort = o.step_name.replace(/^step_/, "");
      chips.appendChild(el("span", { class: `chip chip-s${o.score}`,
        title: o.step_name },
        String(o.score),
        el("span", { class: "stepname" }, stepShort)));
    }
    const tr = el("tr", { class: "hover", onclick: () => showPair(r) },
      el("td", { class: "num" }, String(i + 1)),
      el("td", null, `Q${r.focus_index + 1}: ${r.focus_question}`),
      el("td", null, `Q${r.target_index + 1}: ${r.target_question}`),
      el("td", { class: "num" }, String(r.n)),
      el("td", { class: "num" }, r.mean.toFixed(2)),
      el("td", { class: "num" }, r.variance.toFixed(2)),
      el("td", { class: "num" }, `[${r.min}, ${r.max}]`),
      el("td", null, chips),
    );
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);
}

function renderFocusTable(container, focusRows) {
  const table = el("table", { class: "pair-list" });
  const head = el("thead");
  head.appendChild(el("tr", null,
    el("th", { class: "num" }, "#"),
    el("th", null, "focus question"),
    el("th", { class: "num" }, "target pairs"),
    el("th", { class: "num" }, "obs"),
    el("th", { class: "num" }, "weighted mean var"),
    el("th", { class: "num" }, "max var"),
    el("th", null, "max-var target"),
  ));
  table.appendChild(head);
  const tbody = el("tbody");
  focusRows.forEach((r, i) => {
    const tr = el("tr", null,
      el("td", { class: "num" }, String(i + 1)),
      el("td", null, `Q${r.focus_index + 1}: ${r.focus_question}`),
      el("td", { class: "num" }, String(r.num_target_pairs)),
      el("td", { class: "num" }, String(r.total_observations)),
      el("td", { class: "num" }, r.weighted_mean_variance.toFixed(3)),
      el("td", { class: "num" }, r.max_variance.toFixed(2)),
      el("td", null, r.max_variance_target),
    );
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);
}

function render() {
  const minOcc = Math.max(2, parseInt(document.getElementById("minOcc").value) || 2);
  const minFocus = Math.max(1, parseInt(document.getElementById("minFocus").value) || 1);

  // Filter pair rows by min occurrences
  const filteredRows = DATA.pair_rows.filter(r => r.n >= minOcc);

  // Recompute per-focus summary on filtered rows
  const byFocus = new Map();
  for (const r of filteredRows) {
    if (!byFocus.has(r.focus_index)) byFocus.set(r.focus_index, []);
    byFocus.get(r.focus_index).push(r);
  }
  const focusRows = [];
  for (const [f, rows] of byFocus.entries()) {
    if (rows.length < minFocus) continue;
    const totalN = rows.reduce((a, r) => a + r.n, 0);
    const wmv = rows.reduce((a, r) => a + r.variance * r.n, 0) / totalN;
    const mx = rows.reduce((a, r) => r.variance > a.variance ? r : a, rows[0]);
    focusRows.push({
      focus_index: f,
      focus_question: rows[0].focus_question,
      num_target_pairs: rows.length,
      total_observations: totalN,
      weighted_mean_variance: wmv,
      max_variance: mx.variance,
      max_variance_target: mx.target_question,
    });
  }
  focusRows.sort((a, b) => b.weighted_mean_variance - a.weighted_mean_variance);

  const includedFocus = focusRows.map(r => r.focus_index);
  // For columns: any question that ever appears as a target with enough obs
  const includedTargetSet = new Set();
  for (const r of filteredRows) {
    if (byFocus.has(r.focus_index) && byFocus.get(r.focus_index).length >= minFocus) {
      includedTargetSet.add(r.target_index);
    }
  }
  // Sort target columns by total variance they participate in (most-variant first)
  const targetVarSum = new Map();
  for (const r of filteredRows) {
    targetVarSum.set(r.target_index,
      (targetVarSum.get(r.target_index) || 0) + r.variance);
  }
  const includedTargets = [...includedTargetSet].sort(
    (a, b) => (targetVarSum.get(b) || 0) - (targetVarSum.get(a) || 0));

  const content = document.getElementById("content");
  content.innerHTML = "";

  // 1) Focus question summary
  const sec1 = el("div", { class: "section" });
  sec1.appendChild(el("h2", null,
    `Per focus question (${focusRows.length} shown · ` +
    `min ${minFocus} target pair(s) at ≥${minOcc} obs each)`));
  renderFocusTable(sec1, focusRows);
  content.appendChild(sec1);

  // 2) Pair heatmap (rows: focus, cols: targets)
  const sec2 = el("div", { class: "section" });
  sec2.appendChild(el("h2", null,
    `Variance heatmap — ${includedFocus.length} focus rows × ` +
    `${includedTargets.length} target cols`));
  sec2.appendChild(el("div", { class: "subtle" },
    "Cell value = variance of conditional uncertainty score across steps. " +
    "Click any cell or row in the list below to see per-step scores."));
  const heatWrap = el("div", { style: "overflow:auto; margin-top:6px;" });
  sec2.appendChild(heatWrap);
  renderHeatmap(heatWrap, minOcc, includedFocus, includedTargets);
  content.appendChild(sec2);

  // 3) Pair detail panel
  const sec3 = el("div", { class: "section" });
  sec3.appendChild(el("h2", null, "Pair detail"));
  sec3.appendChild(el("div", { id: "detail-wrap" }));
  content.appendChild(sec3);

  // 4) Top-variance pair list
  const sec4 = el("div", { class: "section" });
  sec4.appendChild(el("h2", null,
    `All pairs (${filteredRows.length}), sorted by variance`));
  renderPairList(sec4, filteredRows);
  content.appendChild(sec4);
}

function init() {
  const metric = DATA.metric || "marginal";
  const banner = document.getElementById("metric-banner");
  const desc = {
    marginal:
      "Metric: marginal information gain = unconditional - conditional uncertainty " +
      "(0..4). 0 = focus tells us nothing new; 4 = focus fully resolves the target. " +
      "Variance across steps is what you want to be small when the model's pairwise " +
      "reasoning is stable.",
    conditional:
      "Metric: conditional uncertainty (1..5) — absolute posterior uncertainty about " +
      "the target after the focus has been resolved.",
    unconditional:
      "Metric: unconditional uncertainty (1..5) — uncertainty about the target given " +
      "only the current beliefs (focus ignored).",
  }[metric];
  if (desc) banner.textContent = desc;

  document.getElementById("apply").addEventListener("click", render);
  document.getElementById("minOcc").addEventListener("change", render);
  document.getElementById("minFocus").addEventListener("change", render);
  render();
}
init();
</script>
</body>
</html>
"""


def write_viewer(out_path: Path, title: str, payload: dict) -> None:
    html = VIEWER_HTML_TEMPLATE.replace("__TITLE__", title).replace(
        "__DATA_JSON__", json.dumps(payload)
    )
    out_path.write_text(html)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="minimum number of steps a pair must appear in to be summarized",
    )
    p.add_argument(
        "--metric",
        choices=["marginal", "conditional", "unconditional"],
        default="marginal",
        help="which per-target value to summarize (default: marginal info gain)",
    )
    p.add_argument("--viewer-out", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    if not args.run_dir.is_dir():
        log.error(f"run_dir does not exist: {args.run_dir}")
        return 1

    episode_dir = find_episode_dir(args.run_dir, args.episode)
    step_dirs = find_step_dirs(episode_dir)
    if not step_dirs:
        log.error(f"No step_* dirs under {episode_dir}")
        return 1

    step_results: list[dict] = []
    for sd in step_dirs:
        f = sd / "conditional_uncertainty.json"
        if not f.exists():
            log.info(f"[{sd.name}] no conditional_uncertainty.json — skipping")
            continue
        step_results.append(json.loads(f.read_text()))
    if not step_results:
        log.error("No conditional_uncertainty.json found in any step")
        return 1

    log.info(f"Loaded {len(step_results)} step results; metric={args.metric}")
    collected = collect_pair_observations(step_results, metric=args.metric)
    log.info(
        f"Found {len(collected['questions'])} unique question texts across steps; "
        f"{len(collected['pairs'])} (focus,target) pairs observed at least once"
    )

    pair_rows = summarize_pairs(collected, min_occurrences=args.min_occurrences)
    log.info(
        f"{len(pair_rows)} pairs occur in ≥{args.min_occurrences} steps"
    )
    focus_rows = summarize_focus_questions(pair_rows)

    payload = {
        "title": f"{args.run_dir.name} / episode_{args.episode}",
        "metric": args.metric,
        "min_occurrences": args.min_occurrences,
        "num_steps": len(step_results),
        "questions": collected["questions"],
        "question_steps": collected["question_steps"],
        "pair_rows": pair_rows,
        "focus_rows": focus_rows,
    }

    json_out = args.json_out or (
        args.run_dir / "variance_conditional_uncertainty.json"
    )
    json_out.write_text(json.dumps(payload, indent=2))
    log.info(f"Wrote {json_out}")

    viewer_out = args.viewer_out or (
        args.run_dir / "variance_conditional_uncertainty_viewer.html"
    )
    write_viewer(viewer_out, payload["title"], payload)
    log.info(f"Wrote viewer: {viewer_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
