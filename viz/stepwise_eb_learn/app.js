let DATA = null;
let selectedStepIdx = null;
let currentTab = "overview";
const detailCache = {};
const trajCache = {};
let combinedTrajCache = null;
let expTimelineCache = null;
let qaTimelineCache = null;
let staticIndex = null;
let currentStaticRun = null;
let currentStaticRunBase = null;

const VIEWER_CONFIG = window.STEPWISE_EB_VIEWER_CONFIG || {};
if (VIEWER_CONFIG.dataIndexPath && !/^https?:\/\//.test(VIEWER_CONFIG.dataIndexPath)) {
  VIEWER_CONFIG.dataIndexPath = new URL(VIEWER_CONFIG.dataIndexPath, window.location.href).toString();
}
const TABS = [
  ["overview", "Overview"],
  ["experiments", "Questions & Experiments"],
  ["agent_messages", "Agent Messages"],
  ["artifacts", "Artifacts"],
  ["feedback", "Feedback"],
  ["trajectory", "Trajectory"],
  ["combined_trajectory", "Cross-Episode Traj"],
  ["qa_timeline", "Q&A Timeline"],
  ["experiment_timeline", "Experiment Timeline"],
  ["logs", "Improve Log"],
];

function currentParams() {
  return new URLSearchParams(window.location.search);
}

function currentLogDir() {
  return currentParams().get("log_dir");
}

function currentRunId() {
  return currentParams().get("run");
}

function clearCaches() {
  for (const key of Object.keys(detailCache)) delete detailCache[key];
  for (const key of Object.keys(trajCache)) delete trajCache[key];
  combinedTrajCache = null;
  expTimelineCache = null;
  qaTimelineCache = null;
}

function esc(s) {
  if (s == null) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

function setLandingError(msg) {
  document.getElementById("landing-error").textContent = msg || "";
}

function navigateWith(updates) {
  const url = new URL(window.location.href);
  for (const [key, value] of Object.entries(updates)) {
    if (value == null || value === "") url.searchParams.delete(key);
    else url.searchParams.set(key, value);
  }
  window.location.href = url.toString();
}

function openNewTab() {
  const url = new URL(window.location.href);
  url.search = "";
  window.open(url.toString(), "_blank");
}

function toggleCard(header) {
  const body = header.nextElementSibling;
  const toggle = header.querySelector(".toggle");
  body.classList.toggle("collapsed");
  if (toggle) toggle.innerHTML = body.classList.contains("collapsed") ? "&#9654;" : "&#9660;";
}

function toggleBody(header) {
  header.nextElementSibling.classList.toggle("open");
}

function collapsible(title, content, open) {
  return '<div class="card"><div class="card-header" onclick="toggleCard(this)">' + title +
    ' <span class="toggle">' + (open ? "&#9660;" : "&#9654;") + '</span></div>' +
    '<div class="card-body ' + (open ? "" : "collapsed") + '">' + content + "</div></div>";
}

function promptResponseBlock(label, prompt, response, imageOpts) {
  let html = "";
  if (prompt) {
    const imgs = imageOpts ? promptImagesHtml(prompt, imageOpts) : "";
    html += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)"><span style="color:var(--text-muted)">' +
      esc(label) + ' Prompt</span><span style="margin-left:auto;font-size:11px">&#9654;</span></div><div class="extraction-body">' +
      imgs +
      '<pre style="max-height:400px">' + esc(prompt) + "</pre></div></div>";
  }
  if (response) {
    html += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)"><span style="color:var(--accent2)">' +
      esc(label) + ' Response</span><span style="margin-left:auto;font-size:11px">&#9654;</span></div><div class="extraction-body"><pre style="max-height:400px">' +
      esc(response) + "</pre></div></div>";
  }
  return html;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("HTTP " + response.status + " for " + url);
  }
  return await response.json();
}

function isDynamicMode() {
  return !!currentLogDir();
}

function isStaticMode() {
  return !!currentRunId();
}

function apiUrl(path, extraParams) {
  const params = new URLSearchParams({ log_dir: currentLogDir(), ...extraParams });
  return path + "?" + params.toString();
}

function staticUrl(relativePath) {
  return new URL(relativePath, currentStaticRunBase).toString();
}

function staticTrajectoryPath(epIdx) {
  return "trajectories/episode_" + String(epIdx).padStart(3, "0") + ".json";
}

function staticStepDetailPath(epIdx, stepIdx) {
  return "step_details/ep_" + String(epIdx).padStart(3, "0") + "_step_" + String(stepIdx).padStart(3, "0") + ".json";
}

function stepImageUrl(epIdx, stepIdx, name) {
  if (isDynamicMode()) return apiUrl("/api/step_image", { episode: epIdx, step: stepIdx, name: name });
  return staticUrl("images/ep_" + String(epIdx).padStart(3, "0") + "_step_" + String(stepIdx).padStart(3, "0") + "_" + name);
}

function stepLocalImageUrl(epIdx, stepIdx, relPath) {
  if (isDynamicMode()) return apiUrl("/api/step_image", { episode: epIdx, step: stepIdx, name: relPath });
  return staticUrl("images/ep_" + String(epIdx).padStart(3, "0") + "_step_" + String(stepIdx).padStart(3, "0") + "/" + relPath);
}

function obsImageHtml(epIdx, stepIdx, data, step) {
  const hasBefore = (data && data.has_obs_before) || (step && step.has_obs_before);
  const hasAfter = (data && data.has_obs_after) || (step && step.has_obs_after);
  if (!hasBefore && !hasAfter) return "";
  let html = '<div style="display:flex;gap:12px;margin:8px 0;flex-wrap:wrap">';
  if (hasBefore) {
    html += '<div style="text-align:center"><div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">Before action</div>' +
      '<img src="' + stepImageUrl(epIdx, stepIdx, "obs_before.png") + '" style="max-width:256px;image-rendering:pixelated;border:1px solid var(--border);border-radius:4px" /></div>';
  }
  if (hasAfter) {
    html += '<div style="text-align:center"><div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">After action</div>' +
      '<img src="' + stepImageUrl(epIdx, stepIdx, "obs_after.png") + '" style="max-width:256px;image-rendering:pixelated;border:1px solid var(--border);border-radius:4px" /></div>';
  }
  html += '</div>';
  return html;
}

function resolveStepByGlobal(gs) {
  if (!DATA || !DATA.steps) return null;
  for (const s of DATA.steps) {
    if (s.global_step === gs) return s;
  }
  return null;
}

function makePromptImagePart(slot, stepMeta, kind, label, accent) {
  if (!stepMeta) return null;
  const isAfter = kind === "after";
  const hasImage = isAfter ? stepMeta.has_obs_after : stepMeta.has_obs_before;
  if (!hasImage) return null;
  return {
    slot: slot,
    label: label,
    url: stepImageUrl(stepMeta.episode_idx, stepMeta.step, isAfter ? "obs_after.png" : "obs_before.png"),
    accent: !!accent,
  };
}

function explicitPromptImageParts(imagePaths, opts) {
  opts = opts || {};
  const stepMeta = opts.stepMeta || opts.currentStep || (DATA && DATA.steps ? DATA.steps[selectedStepIdx] : null);
  if (!stepMeta || !Array.isArray(imagePaths) || imagePaths.length === 0) return [];
  const prefix = opts.labelPrefix || "Image";
  return imagePaths.map((path, i) => {
    const match = /(?:^|\/)image_(\d+)\.png$/i.exec(path || "");
    const slot = match ? parseInt(match[1], 10) : i + 1;
    return {
      slot: slot,
      label: prefix + " " + slot,
      url: stepLocalImageUrl(stepMeta.episode_idx, stepMeta.step, path),
      accent: !!opts.accentCurrent && i === imagePaths.length - 1,
    };
  });
}

function parseNumberedPromptImages(promptText, opts) {
  if (!promptText) return [];
  opts = opts || {};
  const parts = [];
  const seenSlots = new Set();

  const stepBlockRe = /<step\s+n="(\d+)">([\s\S]*?)<\/step>/g;
  let stepMatch;
  while ((stepMatch = stepBlockRe.exec(promptText)) !== null) {
    const gs = parseInt(stepMatch[1], 10);
    const stepMeta = resolveStepByGlobal(gs);
    if (!stepMeta) continue;
    const body = stepMatch[2];
    const preRe = /<(?:pre_state|raw_state)>\s*\(image\s+(\d+)\)/g;
    let preMatch;
    while ((preMatch = preRe.exec(body)) !== null) {
      const slot = parseInt(preMatch[1], 10);
      if (!seenSlots.has(slot)) {
        const part = makePromptImagePart(slot, stepMeta, "before", "Image " + slot + " - g" + gs + " pre", false);
        if (part) parts.push(part);
        seenSlots.add(slot);
      }
    }
    const postRe = /<(?:post_state|resulting_state)>\s*\(image\s+(\d+)\)/g;
    let postMatch;
    while ((postMatch = postRe.exec(body)) !== null) {
      const slot = parseInt(postMatch[1], 10);
      if (!seenSlots.has(slot)) {
        const part = makePromptImagePart(slot, stepMeta, "after", "Image " + slot + " - g" + gs + " result", false);
        if (part) parts.push(part);
        seenSlots.add(slot);
      }
    }
  }

  if (opts.currentStep) {
    const currentBlockRe = /=== CURRENT STATE[^\n]* ===[\s\S]*?<(?:pre_state|raw_state)>\s*\(image\s+(\d+)\)/;
    const currentMatch = currentBlockRe.exec(promptText);
    if (currentMatch) {
      const slot = parseInt(currentMatch[1], 10);
      if (!seenSlots.has(slot)) {
        const part = makePromptImagePart(
          slot,
          opts.currentStep,
          "before",
          "Image " + slot + " - current (g" + opts.currentStep.global_step + ")",
          true
        );
        if (part) parts.push(part);
        seenSlots.add(slot);
      }
    }
  }

  parts.sort((a, b) => a.slot - b.slot);
  return parts;
}

function parseSamplePromptImages(promptText) {
  if (!promptText) return [];
  const parts = [];
  const sampleRe = /<(?:perception_example|execution_sample)\s+step="(\d+)">/g;
  let sampleIdx = 0;
  let sampleMatch;
  while ((sampleMatch = sampleRe.exec(promptText)) !== null) {
    const gs = parseInt(sampleMatch[1], 10);
    const stepMeta = resolveStepByGlobal(gs);
    const part = makePromptImagePart(
      Number.MAX_SAFE_INTEGER,
      stepMeta,
      "before",
      "Sample " + (sampleIdx + 1) + " - g" + gs,
      false
    );
    if (part) {
      part.sampleIdx = sampleIdx;
      parts.push(part);
      sampleIdx += 1;
    }
  }
  return parts;
}

// Build a thumbnail strip for the images sent alongside a prompt. Reconstructs
// the actual attachment sequence from numbered image tags in the prompt plus
// sampled observation blocks for prompts that attach perception examples.
function promptImagesHtml(promptText, opts) {
  opts = opts || {};
  let parts = explicitPromptImageParts(opts.imagePaths, opts);
  if (parts.length === 0) {
    parts = parseNumberedPromptImages(promptText, opts);
    const samples = parseSamplePromptImages(promptText);
    samples.forEach((part) => parts.push(part));
  }
  if (parts.length === 0) return "";
  let html = '<div style="margin:6px 0 10px"><div style="font-size:10px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px;font-weight:600">Images attached (' +
    parts.length + ')</div>' +
    '<div style="display:flex;gap:6px;flex-wrap:wrap;padding:8px;background:var(--surface2);border:1px solid var(--border);border-radius:4px">';
  parts.forEach((p) => {
    const borderColor = p.accent ? "var(--accent)" : "var(--border)";
    html += '<div style="text-align:center"><div style="font-size:10px;color:var(--text-muted);margin-bottom:2px">' + esc(p.label) + '</div>' +
      '<img src="' + p.url + '" style="max-width:112px;image-rendering:pixelated;border:1px solid ' + borderColor + ';border-radius:3px" /></div>';
  });
  html += '</div></div>';
  return html;
}

async function fetchReport() {
  if (isDynamicMode()) return await fetchJson(apiUrl("/api/data"));
  return await fetchJson(staticUrl("report.json"));
}

async function fetchStepDetail(epIdx, stepIdx) {
  if (isDynamicMode()) return await fetchJson(apiUrl("/api/step_detail", { episode: epIdx, step: stepIdx }));
  return await fetchJson(staticUrl(staticStepDetailPath(epIdx, stepIdx)));
}

async function fetchTrajectory(epIdx) {
  if (isDynamicMode()) return await fetchJson(apiUrl("/api/trajectory", { episode: epIdx }));
  return await fetchJson(staticUrl(staticTrajectoryPath(epIdx)));
}

async function fetchCombinedTrajectory() {
  if (isDynamicMode()) return await fetchJson(apiUrl("/api/combined_trajectory"));
  return await fetchJson(staticUrl("combined_trajectory.json"));
}

async function fetchQATimeline() {
  if (isDynamicMode()) return await fetchJson(apiUrl("/api/qa_timeline"));
  return await fetchJson(staticUrl("qa_timeline.json"));
}

async function fetchExperimentTimeline() {
  if (isDynamicMode()) return await fetchJson(apiUrl("/api/experiment_timeline"));
  return await fetchJson(staticUrl("experiment_timeline.json"));
}

async function loadRunIndex() {
  if (!VIEWER_CONFIG.dataIndexPath) return;
  staticIndex = await fetchJson(VIEWER_CONFIG.dataIndexPath);
}

function runLabel(run) {
  if (!run) return "";
  const title = run.title || run.log_dir_name || run.id;
  const summary = [];
  if (run.episodes != null) summary.push(run.episodes + " ep");
  if (run.steps != null) summary.push(run.steps + " steps");
  return summary.length ? title + " (" + summary.join(", ") + ")" : title;
}

function populateRunSelectors() {
  if (!staticIndex || !Array.isArray(staticIndex.runs)) return;

  const landingSelect = document.getElementById("landing-run-select");
  const topbarSelect = document.getElementById("topbar-run-select");
  const current = currentRunId() || "";

  const options = ['<option value="">Select a run</option>'];
  staticIndex.runs.forEach((run) => {
    const selected = run.id === current ? " selected" : "";
    options.push('<option value="' + esc(run.id) + '"' + selected + ">" + esc(runLabel(run)) + "</option>");
  });

  landingSelect.innerHTML = options.join("");
  topbarSelect.innerHTML = options.join("");
  landingSelect.value = current;
  topbarSelect.value = current;
  topbarSelect.onchange = function () {
    if (this.value) navigateWith({ run: this.value, log_dir: null });
  };
  landingSelect.onchange = updateLandingRunMeta;
  updateLandingRunMeta();
}

function updateLandingRunMeta() {
  const meta = document.getElementById("landing-run-meta");
  if (!staticIndex || !Array.isArray(staticIndex.runs)) {
    meta.textContent = "";
    return;
  }
  const selected = document.getElementById("landing-run-select").value;
  const run = staticIndex.runs.find((item) => item.id === selected);
  if (!run) {
    meta.textContent = staticIndex.runs.length ? "Choose one of the exported runs." : "No exported runs found yet.";
    return;
  }
  const parts = [];
  if (run.description) parts.push(run.description);
  parts.push((run.episodes || 0) + " episodes");
  parts.push((run.steps || 0) + " steps");
  if (run.total_cost != null) parts.push("total cost $" + Number(run.total_cost).toFixed(4));
  meta.textContent = parts.join(" | ");
}

function configureLanding() {
  const dynamicPanel = document.getElementById("dynamic-panel");
  const staticPanel = document.getElementById("static-panel");
  const allowDynamic = !!VIEWER_CONFIG.allowDynamicInput;
  const allowStatic = !!(staticIndex && Array.isArray(staticIndex.runs));
  dynamicPanel.classList.toggle("hidden", !allowDynamic);
  staticPanel.classList.toggle("hidden", !allowStatic);
}

function showLanding() {
  document.getElementById("landing-page").style.display = "";
  document.getElementById("viewer-app").style.display = "none";
}

function showViewer() {
  document.getElementById("landing-page").style.display = "none";
  document.getElementById("viewer-app").style.display = "";
}

function updateTopbar() {
  const topbarDir = document.getElementById("topbar-dir");
  const topbarCost = document.getElementById("topbar-cost");
  const runWrap = document.getElementById("topbar-run-wrap");

  if (isStaticMode() && currentStaticRun) {
    topbarDir.textContent = currentStaticRun.title || DATA.log_dir_name;
    runWrap.style.display = "inline-flex";
    document.getElementById("topbar-run-select").value = currentStaticRun.id;
  } else {
    topbarDir.textContent = DATA.log_dir_name;
    runWrap.style.display = "none";
  }
  topbarCost.textContent = "Total cost: $" + (DATA.total_cost || 0).toFixed(4);

  const title = isStaticMode() && currentStaticRun
    ? (currentStaticRun.title || DATA.log_dir_name)
    : DATA.log_dir_name;
  document.title = title + " - Stepwise EB-Learn Viewer";
}

function loadDynamicFromLanding() {
  const input = document.getElementById("landing-input");
  const value = input.value.trim();
  if (!value) {
    setLandingError("Please enter a path.");
    return;
  }
  navigateWith({ log_dir: value, run: null });
}

function loadStaticFromLanding() {
  const runId = document.getElementById("landing-run-select").value;
  if (!runId) {
    setLandingError("Please choose a published run.");
    return;
  }
  navigateWith({ run: runId, log_dir: null });
}

async function reloadData() {
  clearCaches();
  DATA = null;
  await init();
}

async function init() {
  setLandingError("");
  try {
    if (VIEWER_CONFIG.dataIndexPath) {
      await loadRunIndex();
      populateRunSelectors();
    }
  } catch (e) {
    setLandingError("Failed to load published run index: " + e.message);
  }

  configureLanding();

  if (!isDynamicMode() && !isStaticMode()) {
    showLanding();
    return;
  }

  if (isStaticMode()) {
    if (!staticIndex || !Array.isArray(staticIndex.runs)) {
      showLanding();
      setLandingError("Published run index is not available.");
      return;
    }
    currentStaticRun = staticIndex.runs.find((run) => run.id === currentRunId()) || null;
    if (!currentStaticRun) {
      showLanding();
      setLandingError("Unknown published run: " + currentRunId());
      return;
    }
    currentStaticRunBase = new URL(currentStaticRun.path, VIEWER_CONFIG.dataIndexPath).toString();
  } else {
    currentStaticRun = null;
    currentStaticRunBase = null;
  }

  showViewer();
  document.getElementById("main-content").innerHTML = '<div class="loading">Loading data...</div>';

  try {
    DATA = await fetchReport();
    if (DATA.error) {
      document.getElementById("main-content").innerHTML = '<pre style="color:var(--danger)">' + esc(DATA.error) + "</pre>";
      return;
    }
    updateTopbar();
    buildSidebar();
    if (DATA.steps.length > 0) showStep(selectedStepIdx != null ? selectedStepIdx : 0);
    else document.getElementById("main-content").innerHTML = "<h1>No steps found</h1>";
  } catch (e) {
    document.getElementById("main-content").innerHTML = "<pre>" + esc(e.message || e) + "</pre>";
  }
}

document.getElementById("landing-input").addEventListener("keydown", function (e) {
  if (e.key === "Enter") loadDynamicFromLanding();
});

function buildSidebar() {
  const list = document.getElementById("step-list");
  list.innerHTML = "";
  let lastEp = -1;
  DATA.steps.forEach((step, i) => {
    if (step.episode_idx !== lastEp) {
      lastEp = step.episode_idx;
      const epMeta = DATA.episodes.find((episode) => episode.index === step.episode_idx);
      const epLog = epMeta ? epMeta.log : {};
      const ret = epLog.episode_return != null ? " (r=" + Number(epLog.episode_return).toFixed(1) + ")" : "";
      const hdr = document.createElement("div");
      hdr.className = "ep-header";
      hdr.textContent = "Episode " + step.episode_idx + ret;
      list.appendChild(hdr);
    }

    const el = document.createElement("div");
    el.className = "step-item" + (step.step === 0 ? " ep-boundary" : "");
    el.dataset.idx = i;
    const rewardVal = Number(step.reward);
    const rewardClass = rewardVal > 0 ? "pos" : rewardVal < 0 ? "neg" : "zero";

    let dotColor = "";
    let dotTitle = "";
    if (step.improve_cost > 0) {
      dotColor = "var(--purple)";
      dotTitle = "improve loop ran";
    } else if (step.did_trim) {
      dotColor = "var(--accent2)";
      dotTitle = "Q&A trimmed";
    } else if (step.extract_cost > 0) {
      dotColor = "var(--accent)";
      dotTitle = "Q&A extraction";
    } else if (step.did_gen_questions) {
      dotColor = "var(--accent3)";
      dotTitle = "questions generated";
    }

    const statusDot = dotColor ? '<span class="status-dot" style="background:' + dotColor + '" title="' + dotTitle + '"></span>' : "";
    const doneMark = step.done ? '<span class="done-marker">END</span>' : "";
    const isInProgress = step.phase && step.phase !== "complete";
    const phaseLabels = { started: "starting", acting: "acting", extracting: "extracting", improving: "improving" };
    const phaseBadge = isInProgress ? '<span class="phase-badge">' + (phaseLabels[step.phase] || step.phase) + "</span>" : "";
    const actionText = step.action || (isInProgress ? "..." : "");

    // Show level progress for ARC-AGI steps
    const ei2 = step.env_info || {};
    const lvlBadge = (ei2.levels_completed != null && ei2.win_levels)
      ? '<span style="font-size:9px;color:var(--text-muted);margin-left:2px" title="levels completed">' + ei2.levels_completed + '/' + ei2.win_levels + '</span>'
      : '';

    el.innerHTML = '<span class="gs">g' + step.global_step + "</span>" +
      '<span class="act" title="' + esc(actionText) + '">' + esc(actionText) + "</span>" +
      statusDot + doneMark + phaseBadge + lvlBadge +
      '<span class="rw ' + rewardClass + '">' + (isInProgress && !step.action ? "" : rewardVal.toFixed(2)) + "</span>";
    el.onclick = () => {
      currentTab = "overview";
      showStep(i);
    };
    list.appendChild(el);
  });
}

function showStep(idx) {
  selectedStepIdx = idx;
  document.querySelectorAll(".step-item").forEach((el) => {
    el.classList.toggle("active", parseInt(el.dataset.idx, 10) === idx);
  });
  renderStep(idx);
}

function renderStep(idx) {
  const step = DATA.steps[idx];
  const mc = document.getElementById("main-content");
  const total = DATA.steps.length;
  const stepIsInProgress = step.phase && step.phase !== "complete";
  const phaseColors = { started: "var(--accent3)", acting: "var(--accent)", extracting: "var(--purple)", improving: "var(--accent2)" };
  const phaseColor = phaseColors[step.phase] || "var(--text-muted)";
  const phasePill = stepIsInProgress
    ? ' <span style="font-size:11px;padding:2px 8px;border-radius:10px;background:rgba(210,153,34,0.15);color:' + phaseColor + ';font-weight:600;vertical-align:middle">' + (step.phase || "") + "</span>"
    : "";

  // Build env info badge (ARC-AGI: game_id, levels, state)
  const ei = step.env_info || {};
  let envBadge = "";
  if (ei.game_id) {
    const stateColor = ei.state === "WIN" ? "var(--accent2)" : ei.state === "GAME_OVER" ? "#e55" : "var(--text-muted)";
    envBadge = ' <span style="font-size:11px;color:var(--text-muted);font-weight:400">| ' +
      esc(ei.game_id) +
      (ei.levels_completed != null ? " lvl " + ei.levels_completed + "/" + (ei.win_levels || "?") : "") +
      (ei.state ? ' <span style="color:' + stateColor + '">' + esc(ei.state) + "</span>" : "") +
      "</span>";
  }

  let html = '<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">' +
    '<button style="background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px" onclick="showStep(' + Math.max(0, idx - 1) + ')" ' + (idx === 0 ? "disabled" : "") + ">&#8592;</button>" +
    '<h1 style="margin:0;font-size:18px">Step ' + step.step + phasePill + ' <span style="color:var(--text-muted);font-size:14px;font-weight:400">ep' + step.episode_idx + " | global " + step.global_step + envBadge + "</span></h1>" +
    '<span style="font-size:12px;color:var(--text-muted);margin-left:auto">action: <b>' + esc(step.action || "...") + "</b> | reward: " + (step.action ? Number(step.reward).toFixed(2) : "—") + " | cost: $" + Number(step.step_total_cost).toFixed(4) + "</span>" +
    '<button style="background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px" onclick="showStep(' + Math.min(total - 1, idx + 1) + ')" ' + (idx >= total - 1 ? "disabled" : "") + ">&#8594;</button>" +
    "</div>";

  html += '<div class="tabs">';
  TABS.forEach((tab) => {
    html += '<div class="tab ' + (currentTab === tab[0] ? "active" : "") + '" onclick="currentTab=\'' + tab[0] + "';renderStep(" + idx + ')">' + tab[1] + "</div>";
  });
  html += "</div>";

  const containerId = currentTab.replace(/_/g, "-") + "-container";
  html += '<div id="' + containerId + '"><div class="loading">Loading...</div></div>';
  mc.innerHTML = html;

  if (currentTab === "overview") loadStepDetailForTab(step.episode_idx, step.step, "overview", step);
  else if (currentTab === "artifacts") loadStepDetailForTab(step.episode_idx, step.step, "artifacts");
  else if (currentTab === "experiments") loadStepDetailForTab(step.episode_idx, step.step, "experiments");
  else if (currentTab === "feedback") loadStepDetailForTab(step.episode_idx, step.step, "feedback");
  else if (currentTab === "agent_messages") loadStepDetailForTab(step.episode_idx, step.step, "agent_messages");
  else if (currentTab === "trajectory") loadTrajectory(step.episode_idx, step.step);
  else if (currentTab === "combined_trajectory") loadCombinedTrajectory(step.global_step);
  else if (currentTab === "qa_timeline") loadQATimeline(step.global_step);
  else if (currentTab === "experiment_timeline") loadExperimentTimeline(step.global_step);
  else if (currentTab === "logs") loadStepDetailForTab(step.episode_idx, step.step, "logs");
}

async function loadStepDetailForTab(epIdx, stepIdx, tab, stepOverview) {
  const key = epIdx + "_" + stepIdx;
  let data = detailCache[key];
  if (!data) {
    try {
      data = await fetchStepDetail(epIdx, stepIdx);
      detailCache[key] = data;
    } catch (e) {
      const c = document.getElementById(tab.replace(/_/g, "-") + "-container");
      if (c) c.innerHTML = "<pre>" + esc(e.message || e) + "</pre>";
      return;
    }
  }
  if (tab === "overview") renderOverview(data, stepOverview);
  else if (tab === "artifacts") renderArtifacts(data);
  else if (tab === "experiments") renderExperiments(data);
  else if (tab === "feedback") renderFeedback(data);
  else if (tab === "agent_messages") renderAgentMessages(data);
  else if (tab === "logs") renderLogs(data);
}

function renderOverview(data, step) {
  const c = document.getElementById("overview-container");
  if (!c) return;
  let html = "";

  // Agent LLM Call section at top with bigger boxes — show only Current Observation from prompt
  const msgs = data.agent_messages || [];
  if (msgs.length > 0) {
    let lastUser = null;
    let lastAssistant = null;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (!lastAssistant && msgs[i].role === "assistant") lastAssistant = msgs[i];
      if (!lastUser && msgs[i].role === "user") lastUser = msgs[i];
      if (lastUser && lastAssistant) break;
    }
    if (lastUser || lastAssistant) {
      let agentHtml = "";
      if (lastUser) {
        // Extract only the "Current Observation" section from the prompt
        const fullContent = lastUser.content || "";
        let observationContent = fullContent;
        const obsStart = fullContent.indexOf("Current Observation:");
        if (obsStart !== -1) {
          // Start after the "Current Observation:" header line
          let obsText = fullContent.substring(obsStart + "Current Observation:".length);
          // Find the end — cut before agent instruction boilerplate
          const endMarkers = ["\n\nTips -", "\n\nCurrent experimental goal:", "\n\nFirst create (if not present)", "\n\nYou always have to output", "\n\nFinally you must choose"];
          let endIdx = obsText.length;
          for (const marker of endMarkers) {
            const idx = obsText.indexOf(marker);
            if (idx !== -1 && idx < endIdx) endIdx = idx;
          }
          observationContent = obsText.substring(0, endIdx).trim();
        }
        agentHtml += '<div style="margin-bottom:10px"><div style="font-weight:600;font-size:12px;color:var(--text-muted);margin-bottom:4px">Current Observation</div>' +
          '<pre style="max-height:none;font-size:11px;padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:4px;white-space:pre-wrap;word-break:break-word">' + esc(observationContent) + "</pre></div>";
      }
      if (lastAssistant) {
        const actionBadge = lastAssistant.action ? ' <span style="display:inline-block;margin-left:8px;padding:2px 8px;background:var(--accent);color:#fff;border-radius:4px;font-family:var(--font-mono);font-size:11px">' + esc(lastAssistant.action) + "</span>" : "";
        agentHtml += '<div><div style="font-weight:600;font-size:12px;color:var(--accent2);margin-bottom:4px">Model Response' + actionBadge + "</div>" +
          '<pre style="max-height:500px;overflow:auto;font-size:11px;padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:4px;white-space:pre-wrap;word-break:break-word">' + esc(lastAssistant.content || "") + "</pre></div>";
      }
      html += collapsible("Agent LLM Call", agentHtml, true);
    }
  }

  // Observation images
  const imgHtml = obsImageHtml(step.episode_idx, step.step, data, step);
  if (imgHtml) {
    html += collapsible("Observation Images", imgHtml, true);
  }

  // Active Experiment with selected questions
  if (step.active_experiment) {
    const genLabel = step.did_formulate_experiment ? "formulated at start of this step" : "carried over from previous step";
    let expContent = "";
    // Show selected question if available
    const expLog = data.experiment_log || {};
    const selectedQIdx = expLog.selected_question_index;
    if (selectedQIdx != null) {
      const qa = data.qa_pairs || [];
      const selectedQ = selectedQIdx < qa.length ? qa[selectedQIdx] : null;
      if (selectedQ) {
        expContent += '<div style="font-size:12px;margin-bottom:8px;padding:8px 10px;background:var(--bg);border:1px solid var(--accent3);border-radius:4px">' +
          '<span style="color:var(--accent3);font-weight:600">Selected Question (Q' + (selectedQIdx + 1) + '):</span> ' + esc(selectedQ.question) + "</div>";
      }
    }
    expContent += '<pre style="max-height:none">' + esc(step.active_experiment) + "</pre>";
    html += '<div class="card" style="margin-bottom:16px;border-left:3px solid var(--accent2)">' +
      '<div class="card-header" onclick="toggleCard(this)">Active Experiment <span style="font-size:11px;color:var(--text-muted);font-weight:400">' + genLabel + '</span> <span class="toggle">&#9660;</span></div>' +
      '<div class="card-body">' + expContent + "</div></div>";
  }

  html += '<div style="margin-bottom:16px"><h3 style="font-size:12px;color:var(--text-muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">Beliefs</h3><pre>' + esc(data.beliefs || "(empty)") + "</pre></div>";
  html += '<div style="margin-bottom:16px"><h3 style="font-size:12px;color:var(--text-muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">Perception</h3><pre>' + esc(data.perception || "(empty)") + "</pre></div>";

  // Cost section — below perception
  let costHtml = '<table class="data-table"><tr><th>Category</th><th>This Step</th><th>Cumulative</th></tr>';
  let cumAgent = 0, cumExtract = 0, cumImprove = 0, cumExperiment = 0, cumTotal = 0;
  for (let i = 0; i <= selectedStepIdx; i++) {
    const s = DATA.steps[i];
    cumAgent += s.agent_step_cost || 0;
    cumExtract += s.extract_cost || 0;
    cumImprove += s.improve_cost || 0;
    cumExperiment += s.experiment_cost || 0;
    cumTotal += s.step_total_cost || 0;
  }
  costHtml += '<tr><td>Agent</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + Number(step.agent_step_cost).toFixed(4) + '</td><td style="font-family:var(--font-mono);color:var(--text-muted)">$' + cumAgent.toFixed(4) + "</td></tr>";
  costHtml += '<tr><td>Extraction</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + Number(step.extract_cost).toFixed(4) + '</td><td style="font-family:var(--font-mono);color:var(--text-muted)">$' + cumExtract.toFixed(4) + "</td></tr>";
  costHtml += '<tr><td>Improve</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + Number(step.improve_cost).toFixed(4) + '</td><td style="font-family:var(--font-mono);color:var(--text-muted)">$' + cumImprove.toFixed(4) + "</td></tr>";
  costHtml += '<tr><td>Experiment</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + Number(step.experiment_cost).toFixed(4) + '</td><td style="font-family:var(--font-mono);color:var(--text-muted)">$' + cumExperiment.toFixed(4) + "</td></tr>";
  costHtml += '<tr style="font-weight:600"><td>Total</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + Number(step.step_total_cost).toFixed(4) + '</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + cumTotal.toFixed(4) + "</td></tr>";
  costHtml += "</table>";
  html += collapsible("Cost Breakdown", costHtml, true);
  html += renderCostChart();

  c.innerHTML = html;
}

function renderCostChart() {
  if (!DATA.steps || DATA.steps.length < 2) return "";
  let html = '<div class="card"><div class="card-header" onclick="toggleCard(this)">Cost Over Steps <span class="toggle">&#9660;</span></div><div class="card-body">';
  const maxCost = Math.max(...DATA.steps.map((step) => step.step_total_cost), 0.0001);
  html += '<div style="display:flex;align-items:flex-end;gap:2px;height:120px;border-bottom:1px solid var(--border)">';
  DATA.steps.forEach((step, i) => {
    const height = Math.max((step.step_total_cost / maxCost) * 100, 1);
    const isSelected = i === selectedStepIdx;
    const color = isSelected ? "var(--accent)" : (step.improve_cost > 0 ? "var(--purple)" : "var(--surface2)");
    html += '<div style="flex:1;height:' + height + '%;background:' + color + ';border-radius:2px 2px 0 0;cursor:pointer;min-width:2px" title="g' + step.global_step + ": $" + step.step_total_cost.toFixed(4) + '" onclick="showStep(' + i + ')"></div>';
  });
  html += "</div>";
  html += '<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-muted);margin-top:4px"><span>g0</span><span>g' + DATA.steps[DATA.steps.length - 1].global_step + "</span></div>";
  html += "</div></div>";
  return html;
}

function renderArtifacts(data) {
  const c = document.getElementById("artifacts-container");
  if (!c) return;
  let html = "";

  const extLog = data.extraction_log || {};
  if (extLog.prompt || extLog.response) {
    let extHtml = "";
    if (extLog.prev_count != null) {
      extHtml += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">' +
        "Q: " + extLog.prev_count + " → " + extLog.new_count +
        " | Unanswered: " + extLog.prev_unanswered + " → " + extLog.new_unanswered +
        " | Newly answered: " + (extLog.newly_answered || 0) + "</div>";
    }
    extHtml += promptResponseBlock("Q Update", extLog.prompt, extLog.response, {
      imagePaths: extLog.prompt_image_paths,
      stepMeta: DATA.steps[selectedStepIdx] || {},
    });
    html += collapsible("Q&A Update from Trajectory", extHtml, false);
  }

  const trimLog = data.trim_log || {};
  if (trimLog.prompt || trimLog.response) {
    let trimHtml = "";
    if (trimLog.pre_trim_count != null) {
      trimHtml += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">' +
        "Q: " + trimLog.pre_trim_count + " → " + (trimLog.post_trim_count != null ? trimLog.post_trim_count : "?") +
        " (dropped " + (trimLog.dropped_count || 0) + ", limit: " + (trimLog.max_total_qa_pairs || "?") + ")" + "</div>";
    }
    trimHtml += promptResponseBlock("Q Trim", trimLog.prompt, trimLog.response);
    html += collapsible("Q&A Trim", trimHtml, false);
  }

  const qa = data.qa_pairs || [];
  if (qa.length > 0) {
    const answered = qa.filter((item) => item.answer !== null);
    const unanswered = qa.filter((item) => item.answer === null);
    let qaHtml = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">' +
      answered.length + " answered, " + unanswered.length + " unanswered</div>";
    qaHtml += '<table class="data-table"><tr><th>#</th><th>Question</th><th>Answer</th><th>Evidence</th><th>Src Step</th></tr>';
    qa.forEach((item, i) => {
      let answer;
      let verdictClass;
      if (item.answer === null || item.answer === undefined) {
        answer = "UNANSWERED";
        verdictClass = "verdict-unanswered";
      } else if (item.answer === true) {
        answer = "YES";
        verdictClass = "verdict-correct";
      } else {
        answer = "NO";
        verdictClass = "verdict-incorrect";
      }
      qaHtml += '<tr><td style="color:var(--text-muted)">Q' + (i + 1) + "</td>" +
        '<td style="max-width:280px">' + esc(item.question) + "</td>" +
        '<td><span class="verdict ' + verdictClass + '">' + answer + "</span></td>" +
        '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(item.evidence) + '">' + esc(item.evidence || "") + "</td>" +
        "<td>" + esc(item.source_step) + "</td></tr>";
    });
    qaHtml += "</table>";
    html += collapsible("Questions (" + qa.length + ": " + answered.length + " answered, " + unanswered.length + " unanswered)", qaHtml, true);
  }

  if (!html) html = '<div style="color:var(--text-muted);padding:20px">No artifact data for this step.</div>';
  c.innerHTML = html;
}

function renderExperiments(data) {
  const c = document.getElementById("experiments-container");
  if (!c) return;
  let html = "";
  const expLog = data.experiment_log || {};
  const stepMeta = DATA.steps[selectedStepIdx] || {};
  const didGenQ = stepMeta.did_gen_questions;
  const didFormulate = stepMeta.did_formulate_experiment;

  html += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">' +
    (didGenQ
      ? 'Questions were <strong style="color:var(--accent3)">generated at the start of this step</strong>. ' +
        (didFormulate
          ? 'A new <strong style="color:var(--accent2)">experiment was formulated</strong> from an unanswered question.'
          : "The current experiment was <strong>kept</strong> (LLM returned null).")
      : "No question generation this step — the agent used the experiment carried over from the previous cycle.") +
    "</div>";

  // Step 1: Question Generation — Prompt & Response
  if (expLog.question_gen_prompt || expLog.question_gen_response) {
    let qGenHtml = "";
    if (expLog.question_gen_prompt) {
      qGenHtml += promptImagesHtml(expLog.question_gen_prompt, {
        currentStep: stepMeta,
        stepMeta: stepMeta,
        imagePaths: expLog.question_gen_image_paths,
        accentCurrent: true,
      });
      qGenHtml += '<div style="margin-bottom:10px"><div style="font-weight:600;font-size:12px;color:var(--text-muted);margin-bottom:4px">Prompt</div>' +
        '<pre style="max-height:400px;overflow:auto;font-size:11px;padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:4px;white-space:pre-wrap;word-break:break-word">' + esc(expLog.question_gen_prompt) + "</pre></div>";
    }
    if (expLog.question_gen_response) {
      qGenHtml += '<div><div style="font-weight:600;font-size:12px;color:var(--accent3);margin-bottom:4px">Response</div>' +
        '<pre style="max-height:400px;overflow:auto;font-size:11px;padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:4px;white-space:pre-wrap;word-break:break-word">' + esc(expLog.question_gen_response) + "</pre></div>";
    }
    html += collapsible("Step 1: Question Generation — Prompt & Response", qGenHtml, false);
  }

  // Step 2: New Questions Generated
  if (expLog.new_questions && expLog.new_questions.length > 0) {
    let qHtml = '<ul style="margin:0;padding-left:20px">';
    expLog.new_questions.forEach((question) => {
      qHtml += '<li style="margin-bottom:4px;font-size:13px;color:var(--accent3)">' + esc(question) + "</li>";
    });
    qHtml += "</ul>";
    html += collapsible("Step 2: New Questions Generated (" + expLog.new_questions.length + ")", qHtml, true);
  }

  // Step 3: All Available Questions Now (question bank)
  const qa = data.qa_pairs || [];
  if (qa.length > 0) {
    const answered = qa.filter((item) => item.answer !== null);
    const unanswered = qa.filter((item) => item.answer === null);
    let qaHtml = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">' +
      answered.length + " answered, " + unanswered.length + " unanswered</div>";
    qaHtml += '<table class="data-table"><tr><th>#</th><th>Question</th><th>Answer</th><th>Evidence</th><th>Src Step</th></tr>';
    qa.forEach((item, i) => {
      let answer;
      let verdictClass;
      if (item.answer === null || item.answer === undefined) {
        answer = "UNANSWERED";
        verdictClass = "verdict-unanswered";
      } else if (item.answer === true) {
        answer = "YES";
        verdictClass = "verdict-correct";
      } else {
        answer = "NO";
        verdictClass = "verdict-incorrect";
      }
      qaHtml += '<tr><td style="color:var(--text-muted)">Q' + (i + 1) + "</td>" +
        '<td style="max-width:280px">' + esc(item.question) + "</td>" +
        '<td><span class="verdict ' + verdictClass + '">' + answer + "</span></td>" +
        '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(item.evidence) + '">' + esc(item.evidence || "") + "</td>" +
        "<td>" + esc(item.source_step) + "</td></tr>";
    });
    qaHtml += "</table>";
    html += collapsible("Step 3: All Available Questions (" + qa.length + ": " + answered.length + " answered, " + unanswered.length + " unanswered)", qaHtml, false);
  }

  // Step 4: Experiment Formulation — Prompt & Response
  if (expLog.experiment_prompt || expLog.experiment_response) {
    let expHtml = "";
    if (expLog.experiment_prompt) {
      expHtml += promptImagesHtml(expLog.experiment_prompt, {
        currentStep: stepMeta,
        stepMeta: stepMeta,
        imagePaths: expLog.experiment_image_paths,
        accentCurrent: true,
      });
      expHtml += '<div style="margin-bottom:10px"><div style="font-weight:600;font-size:12px;color:var(--text-muted);margin-bottom:4px">Prompt</div>' +
        '<pre style="max-height:400px;overflow:auto;font-size:11px;padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:4px;white-space:pre-wrap;word-break:break-word">' + esc(expLog.experiment_prompt) + "</pre></div>";
    }
    if (expLog.experiment_response) {
      expHtml += '<div><div style="font-weight:600;font-size:12px;color:var(--accent2);margin-bottom:4px">Response</div>' +
        '<pre style="max-height:400px;overflow:auto;font-size:11px;padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:4px;white-space:pre-wrap;word-break:break-word">' + esc(expLog.experiment_response) + "</pre></div>";
    }
    html += collapsible("Step 4: Experiment Formulation — Prompt & Response", expHtml, false);
  }

  // Step 5: Selected Question & Formulated Experiment
  if (expLog.experiment_plan || expLog.selected_question_index != null) {
    let resultHtml = "";
    const qIdx = expLog.selected_question_index;
    if (qIdx != null && qIdx < qa.length) {
      resultHtml += '<div style="font-size:12px;margin-bottom:8px;padding:8px 10px;background:var(--bg);border:1px solid var(--accent3);border-radius:4px">' +
        '<span style="color:var(--accent3);font-weight:600">Selected Question (Q' + (qIdx + 1) + '):</span> ' + esc(qa[qIdx].question) + "</div>";
    }
    if (expLog.experiment_plan) {
      resultHtml += '<div style="font-size:12px;padding:10px 14px;background:var(--bg);border:1px solid var(--accent2);border-radius:6px">' +
        '<strong style="color:var(--accent2)">Formulated Experiment:</strong> ' + esc(expLog.experiment_plan) + "</div>";
    } else {
      resultHtml += '<div style="color:var(--text-muted);font-size:12px;padding:8px 12px;background:var(--surface);border-radius:4px">LLM chose to keep the current experiment (returned null).</div>';
    }
    html += collapsible("Step 5: Selected Question & Formulated Experiment", resultHtml, true);
  }

  if (!html) html = '<div style="color:var(--text-muted);padding:20px">No experiment data for this step.</div>';
  c.innerHTML = html;
}

function getTrackMeta(track) {
  if (track === "steps_beliefs") return { label: "Track 1a: Steps Beliefs", color: "var(--accent)" };
  if (track === "perception_from_analysis") return { label: "Track 1b: Perception (from Analysis)", color: "var(--accent3)" };
  if (track === "qa") return { label: "Track 2: QA", color: "var(--purple)" };
  return { label: "Track: " + track, color: "var(--text-muted)" };
}

function renderFeedback(data) {
  const c = document.getElementById("feedback-container");
  if (!c) return;
  const fb = data.feedback_history || [];
  if (fb.length === 0) {
    c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No feedback history for this step.</div>';
    return;
  }

  let html = "";

  fb.forEach((trackRecord) => {
    const track = trackRecord.track || "unknown";
    const turns = trackRecord.turns || [];
    const meta = getTrackMeta(track);

    let trackHtml = "";
    if (trackRecord.global_step != null) {
      trackHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:8px">Global step: ' + trackRecord.global_step + " | Env step: " + trackRecord.step + "</div>";
    }

    if (track === "qa") {
      trackHtml += '<div style="font-size:12px;margin-bottom:8px">Initial eval: <span style="color:var(--accent2)">' +
        (trackRecord.initial_correct || 0) + ' correct</span>, <span style="color:var(--danger)">' +
        (trackRecord.initial_incorrect || 0) + " incorrect</span></div>";
    }

    if (track === "qa" && trackRecord.qa_feedback_details && trackRecord.qa_feedback_details.length > 0) {
      let detailHtml = '<table class="data-table"><tr><th>Question</th><th>Correct</th><th>Predicted</th><th>Verdict</th><th>Feedback</th></tr>';
      trackRecord.qa_feedback_details.forEach((detail) => {
        const verdictClass = detail.verdict === "CORRECT" ? "verdict-correct" :
          detail.verdict === "INCORRECT" ? "verdict-incorrect" : "verdict-inconclusive";
        detailHtml += '<tr><td style="max-width:200px">' + esc(detail.question || (detail.forward && detail.forward.qa_pair ? detail.forward.qa_pair.question : "")) + "</td>" +
          "<td>" + esc(detail.correct_answer || "") + "</td>" +
          "<td>" + esc(detail.predicted_answer || (detail.forward ? detail.forward.predicted_answer : "")) + "</td>" +
          '<td><span class="verdict ' + verdictClass + '">' + esc(detail.verdict) + "</span></td>" +
          '<td style="max-width:200px">' + esc(detail.feedback) + "</td></tr>";
      });
      detailHtml += "</table>";
      trackHtml += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)"><span>QA Feedback Details</span><span style="margin-left:auto;font-size:11px">&#9654;</span></div><div class="extraction-body">' + detailHtml + "</div></div>";
      trackHtml += promptResponseBlock("QA Forward", trackRecord.qa_forward_prompt, trackRecord.qa_forward_response);
      trackHtml += promptResponseBlock("QA Feedback", trackRecord.qa_feedback_prompt, trackRecord.qa_feedback_response);
    }

    if (turns.length > 0) {
      let turnsHtml = '<div style="margin-top:8px">';
      const totalCost = turns.reduce((sum, turn) => sum + (turn.cost || 0), 0);
      turnsHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:12px">' + turns.length + " turn(s), total cost: $" + totalCost.toFixed(4) + "</div>";
      turns.forEach((turn) => {
        const submitBadge = turn.submitted
          ? '<span style="background:rgba(63,185,80,0.15);color:var(--accent2);padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">SUBMITTED</span>'
          : '<span style="background:rgba(88,166,255,0.1);color:var(--accent);padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">CONTINUE</span>';

        const turnImagesHtml = promptImagesHtml(turn.prompt || "");

        turnsHtml += '<div class="extraction-section" style="margin-bottom:8px">' +
          '<div class="extraction-header" onclick="toggleBody(this)" style="padding:8px 12px">' +
          '<span style="font-weight:600;color:' + meta.color + '">Turn ' + turn.turn + '</span>' +
          '<span style="color:var(--accent3);font-family:var(--font-mono);font-size:11px;margin-left:8px">$' + (turn.cost || 0).toFixed(4) + "</span>" +
          submitBadge +
          '<span style="margin-left:auto;font-size:11px;color:var(--text-muted)">&#9654;</span>' +
          "</div>" +
          '<div class="extraction-body">' +
          turnImagesHtml +
          '<div style="margin-bottom:10px"><div style="font-size:10px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px;font-weight:600">Prompt</div>' +
          '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:10px 14px">' +
          '<pre style="max-height:400px;margin:0;border:none;padding:0;background:transparent">' + esc(turn.prompt || "") + "</pre></div></div>" +
          '<div style="margin-bottom:6px"><div style="font-size:10px;text-transform:uppercase;color:var(--accent2);margin-bottom:4px;font-weight:600">Response</div>' +
          '<div style="background:rgba(88,166,255,0.05);border:1px solid rgba(88,166,255,0.2);border-radius:8px;padding:10px 14px">' +
          '<pre style="max-height:400px;margin:0;border:none;padding:0;background:transparent">' + esc(turn.response || "") + "</pre></div></div>" +
          "</div></div>";
      });
      turnsHtml += "</div>";
      trackHtml += turnsHtml;
    }

    if (trackRecord.error) trackHtml += '<pre style="color:var(--danger)">' + esc(trackRecord.error) + "</pre>";

    html += '<div style="margin-bottom:16px;padding:12px;background:var(--bg);border:1px solid var(--border);border-radius:6px;border-left:3px solid ' + meta.color + '">' +
      '<div style="font-size:13px;font-weight:600;color:' + meta.color + ';margin-bottom:8px">' + meta.label + "</div>" +
      trackHtml + "</div>";

    // After the QA track, show QA answering performance chart
    if (track === "qa") {
      html += renderQAPerformanceChart(trackRecord, turns);
    }
  });

  c.innerHTML = html;
}

function renderQAPerformanceChart(trackRecord, turns) {
  if (!turns || turns.length === 0) return "";

  let scoreData = [];
  // Initial scores
  scoreData.push({ label: "Init", correct: trackRecord.initial_correct || 0, incorrect: trackRecord.initial_incorrect || 0 });
  // Extract scores from each turn's response
  turns.forEach((turn) => {
    const resp = turn.response || "";
    const correctMatch = resp.match(/(\d+)\s*correct/i);
    const incorrectMatch = resp.match(/(\d+)\s*incorrect/i);
    if (correctMatch || incorrectMatch) {
      scoreData.push({
        label: "T" + turn.turn,
        correct: correctMatch ? parseInt(correctMatch[1]) : 0,
        incorrect: incorrectMatch ? parseInt(incorrectMatch[1]) : 0,
      });
    }
  });
  if (scoreData.length <= 1) return "";

  const maxScore = Math.max(...scoreData.map((d) => d.correct + d.incorrect), 1);
  let html = '<div class="card" style="margin-bottom:16px"><div class="card-header" onclick="toggleCard(this)">QA Answering Performance Per Turn <span class="toggle">&#9660;</span></div><div class="card-body">';
  html += '<div style="display:flex;align-items:flex-end;gap:8px;height:100px">';
  scoreData.forEach((d) => {
    const correctH = Math.max(Math.round((d.correct / maxScore) * 85), 1);
    const incorrectH = Math.max(Math.round((d.incorrect / maxScore) * 85), 1);
    html += '<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:0">' +
      '<div style="display:flex;flex-direction:column-reverse;height:85px;width:100%;min-width:24px">' +
      '<div style="height:' + correctH + 'px;background:var(--accent2);border-radius:0 0 2px 2px"></div>' +
      '<div style="height:' + incorrectH + 'px;background:var(--danger);border-radius:2px 2px 0 0"></div>' +
      "</div>" +
      '<div style="font-size:10px;color:var(--text-muted);margin-top:4px">' + d.label + "</div>" +
      '<div style="font-size:10px;color:var(--accent2);font-weight:600">' + d.correct + '/' + (d.correct + d.incorrect) + "</div></div>";
  });
  html += "</div>";
  html += '<div style="display:flex;gap:16px;margin-top:8px;font-size:11px"><span><span style="display:inline-block;width:10px;height:10px;background:var(--accent2);border-radius:2px;vertical-align:middle"></span> Correct</span><span><span style="display:inline-block;width:10px;height:10px;background:var(--danger);border-radius:2px;vertical-align:middle"></span> Incorrect</span></div>';
  html += "</div></div>";
  return html;
}

function renderAgentMessages(data) {
  const c = document.getElementById("agent-messages-container");
  if (!c) return;
  const msgs = data.agent_messages || [];
  if (msgs.length === 0) {
    c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No agent messages for this step.</div>';
    return;
  }

  let html = '<div style="display:flex;flex-direction:column;gap:8px">';
  msgs.forEach((msg, i) => {
    const isAssistant = msg.role === "assistant";
    const bubbleClass = isAssistant ? "msg-assistant" : "msg-user";
    const content = msg.content || "";
    const isResponse = isAssistant && msg.action !== undefined;
    let extra = "";
    if (isResponse && msg.action) {
      extra = '<div style="margin-top:8px;padding:6px 10px;background:var(--accent);color:#fff;border-radius:4px;font-family:monospace;font-size:0.9em"><strong>Action:</strong> ' + esc(msg.action) + "</div>";
    }
    html += '<div class="msg-bubble ' + bubbleClass + '">' +
      '<div class="msg-role">' + esc(isResponse ? "assistant (response)" : msg.role || "unknown") +
      " (message " + (i + 1) + "/" + msgs.length + ", " + content.length + " chars)</div>" +
      '<pre style="max-height:none;margin:0;border:none;padding:0;background:transparent">' + esc(content) + "</pre>" +
      extra + "</div>";
  });
  html += "</div>";
  c.innerHTML = html;
}

function renderLogs(data) {
  const c = document.getElementById("logs-container");
  if (!c) return;
  if (!data.improve_log) {
    c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No improve log for this step.</div>';
    return;
  }
  c.innerHTML = '<pre style="max-height:none">' + esc(data.improve_log) + "</pre>";
}

async function loadTrajectory(epIdx, highlightStep) {
  const c = document.getElementById("trajectory-container");
  if (!c) return;
  let traj = trajCache[epIdx];
  if (!traj) {
    try {
      traj = await fetchTrajectory(epIdx);
      trajCache[epIdx] = traj;
    } catch (e) {
      c.innerHTML = "<pre>" + esc(e.message || e) + "</pre>";
      return;
    }
  }
  if (!traj || traj.length === 0) {
    c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No trajectory data.</div>';
    return;
  }

  let html = "";
  traj.forEach((t) => {
    const isHighlighted = parseInt(t.step, 10) === highlightStep;
    const doneClass = t.done === "True" ? (parseFloat(t.reward) > 0 ? "success" : "death") : "";
    const border = isHighlighted ? "border-color:var(--accent)" : "";
    html += '<div class="traj-step" style="' + border + '">' +
      '<div class="traj-step-header" onclick="toggleBody(this)">' +
      '<div class="traj-step-num ' + doneClass + '">' + t.step + "</div>" +
      '<div class="traj-step-action">' + esc(t.action) + "</div>" +
      '<div class="traj-step-reward">r=' + t.reward + (t.done === "True" ? " (DONE)" : "") + "</div>" +
      "</div>" +
      '<div class="traj-step-body' + (isHighlighted ? " open" : "") + '">' +
      '<div class="traj-section"><div class="traj-section-label">Observation</div><pre>' + esc(t.observation) + "</pre></div>" +
      '<div class="traj-section"><div class="traj-section-label">Reasoning</div><pre>' + esc(t.reasoning) + "</pre></div>" +
      "</div></div>";
  });
  c.innerHTML = html;
}

async function loadCombinedTrajectory(highlightGlobalStep) {
  const c = document.getElementById("combined-trajectory-container");
  if (!c) return;
  if (!combinedTrajCache) {
    try {
      combinedTrajCache = await fetchCombinedTrajectory();
    } catch (e) {
      c.innerHTML = "<pre>" + esc(e.message || e) + "</pre>";
      return;
    }
  }
  const traj = combinedTrajCache;
  if (!traj || traj.length === 0) {
    c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No trajectory data.</div>';
    return;
  }

  let html = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">Combined trajectory across all episodes. Episode boundaries are marked.</div>';
  traj.forEach((t) => {
    if (t.episode_boundary) {
      html += '<div style="padding:10px 16px;margin:8px 0;background:var(--surface);border:1px solid var(--accent3);border-radius:6px;font-size:12px;font-weight:600;color:var(--accent3);text-align:center">—— Episode ' + t.episode_idx + " starts ——</div>";
      return;
    }
    const gs = t.global_step;
    const isHighlighted = gs === highlightGlobalStep;
    const doneClass = t.done === "True" ? (parseFloat(t.reward) > 0 ? "success" : "death") : "";
    const border = isHighlighted ? "border-color:var(--accent)" : "";
    html += '<div class="traj-step" style="' + border + '" data-gs="' + gs + '">' +
      '<div class="traj-step-header" onclick="toggleBody(this)">' +
      '<div class="traj-step-num ' + doneClass + '">' + gs + "</div>" +
      '<span style="font-size:10px;color:var(--text-muted);margin-right:4px">ep' + t.episode_idx + "</span>" +
      '<div class="traj-step-action">' + esc(t.action) + "</div>" +
      '<div class="traj-step-reward">r=' + t.reward + (t.done === "True" ? " (DONE)" : "") + "</div>" +
      "</div>" +
      '<div class="traj-step-body' + (isHighlighted ? " open" : "") + '">' +
      '<div class="traj-section"><div class="traj-section-label">Observation</div><pre>' + esc(t.observation) + "</pre></div>" +
      '<div class="traj-section"><div class="traj-section-label">Reasoning</div><pre>' + esc(t.reasoning) + "</pre></div>" +
      "</div></div>";
  });
  c.innerHTML = html;
}

async function loadQATimeline(highlightGlobalStep) {
  const c = document.getElementById("qa-timeline-container");
  if (!c) return;
  if (!qaTimelineCache) {
    try {
      qaTimelineCache = await fetchQATimeline();
    } catch (e) {
      c.innerHTML = "<pre>" + esc(e.message || e) + "</pre>";
      return;
    }
  }
  const timeline = qaTimelineCache;
  if (!timeline || timeline.length === 0) {
    c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No Q&A data found.</div>';
    return;
  }

  let html = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">Evolution of Q&A pairs over time. Green = answered, yellow = unanswered.</div>';
  const maxTotal = Math.max(...timeline.map((item) => item.total), 1);
  html += '<div class="card"><div class="card-header" onclick="toggleCard(this)">Questions Over Time <span class="toggle">&#9660;</span></div><div class="card-body">';
  html += '<div style="display:flex;align-items:flex-end;gap:2px;height:140px;border-bottom:1px solid var(--border)">';
  timeline.forEach((item) => {
    const answeredH = Math.round((item.answered / maxTotal) * 120);
    const unansweredH = Math.round((item.unanswered / maxTotal) * 120);
    const isNear = item.global_step === highlightGlobalStep;
    const opacity = isNear ? "1" : "0.7";
    const border = isNear ? "2px solid var(--accent)" : "none";
    html += '<div style="flex:1;min-width:3px;display:flex;flex-direction:column-reverse;opacity:' + opacity + ";border:" + border + ';border-radius:2px" title="g' + item.global_step + ": " + item.answered + " answered, " + item.unanswered + ' unanswered">' +
      '<div style="height:' + answeredH + 'px;background:var(--accent2);border-radius:0 0 2px 2px"></div>' +
      '<div style="height:' + unansweredH + 'px;background:var(--accent3);border-radius:2px 2px 0 0"></div>' +
      "</div>";
  });
  html += "</div>";
  html += '<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-muted);margin-top:4px"><span>g' + timeline[0].global_step + "</span><span>g" + timeline[timeline.length - 1].global_step + "</span></div>";
  html += '<div style="display:flex;gap:16px;margin-top:8px;font-size:11px"><span><span style="display:inline-block;width:10px;height:10px;background:var(--accent2);border-radius:2px;vertical-align:middle"></span> Answered</span><span><span style="display:inline-block;width:10px;height:10px;background:var(--accent3);border-radius:2px;vertical-align:middle"></span> Unanswered</span></div>';
  html += "</div></div>";

  // Per-step details: new questions and full question list in dropdowns
  let lastEp = -1;
  timeline.forEach((item, idx) => {
    if (item.episode_idx !== lastEp) {
      lastEp = item.episode_idx;
      html += '<div style="padding:8px 16px;font-size:12px;font-weight:600;color:var(--accent);background:var(--bg);border-bottom:1px solid var(--border);margin-top:8px">Episode ' + item.episode_idx + "</div>";
    }

    const isHighlighted = item.global_step === highlightGlobalStep;
    const borderStyle = isHighlighted ? "border-left:3px solid var(--accent)" : "border-left:3px solid var(--surface2)";
    let stepHtml = '<div style="padding:10px 14px;margin-bottom:6px;background:var(--surface);border:1px solid var(--border);border-radius:6px;' + borderStyle + '">';
    stepHtml += '<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">';
    stepHtml += '<span style="font-size:12px;font-weight:600;color:var(--accent)">g' + item.global_step + "</span>";
    stepHtml += '<span style="font-size:11px;color:var(--text-muted)">ep' + item.episode_idx + " step " + item.step + "</span>";
    stepHtml += '<span style="font-size:11px;color:var(--accent2)">' + item.answered + " answered</span>";
    stepHtml += '<span style="font-size:11px;color:var(--accent3)">' + item.unanswered + " unanswered</span>";
    stepHtml += "</div>";

    const newQ = item.new_questions || [];
    if (newQ.length > 0) {
      stepHtml += '<div class="extraction-section" style="margin-bottom:4px"><div class="extraction-header" onclick="toggleBody(this)" style="padding:6px 10px">' +
        '<span style="font-size:11px;color:var(--accent3);font-weight:600">New Questions (+' + newQ.length + ')</span>' +
        '<span style="margin-left:auto;font-size:11px">&#9654;</span></div>' +
        '<div class="extraction-body"><ul style="margin:0;padding-left:18px;font-size:12px">';
      newQ.forEach((q) => {
        stepHtml += '<li style="margin-bottom:2px;color:var(--accent3)">' + esc(q) + "</li>";
      });
      stepHtml += "</ul></div></div>";
    }

    const allQ = item.all_questions || [];
    if (allQ.length > 0) {
      stepHtml += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)" style="padding:6px 10px">' +
        '<span style="font-size:11px;color:var(--text-muted);font-weight:600">All Questions (' + allQ.length + ')</span>' +
        '<span style="margin-left:auto;font-size:11px">&#9654;</span></div>' +
        '<div class="extraction-body"><table class="data-table" style="font-size:11px"><tr><th>#</th><th>Question</th><th>Status</th></tr>';
      allQ.forEach((q, qi) => {
        const status = q.answer === null || q.answer === undefined
          ? '<span class="verdict verdict-unanswered">UNANSWERED</span>'
          : q.answer === true
            ? '<span class="verdict verdict-correct">YES</span>'
            : '<span class="verdict verdict-incorrect">NO</span>';
        stepHtml += "<tr><td>Q" + (qi + 1) + "</td><td>" + esc(q.question) + "</td><td>" + status + "</td></tr>";
      });
      stepHtml += "</table></div></div>";
    }

    stepHtml += "</div>";
    html += stepHtml;
  });

  c.innerHTML = html;
}

async function loadExperimentTimeline(highlightGlobalStep) {
  const c = document.getElementById("experiment-timeline-container");
  if (!c) return;
  if (!expTimelineCache) {
    try {
      expTimelineCache = await fetchExperimentTimeline();
    } catch (e) {
      c.innerHTML = "<pre>" + esc(e.message || e) + "</pre>";
      return;
    }
  }
  const timeline = expTimelineCache;
  if (!timeline || timeline.length === 0) {
    c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No experiment events found.</div>';
    return;
  }

  let html = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">Experiment formulation events. Each event shows the selected question and the experiment formulated to investigate it.</div>';
  if (timeline.length > 1) {
    const maxQ = Math.max(...timeline.map((item) => item.cumulative_questions), 1);
    html += '<div class="card" style="margin-bottom:16px"><div class="card-header" onclick="toggleCard(this)">Cumulative Questions Generated <span class="toggle">&#9660;</span></div><div class="card-body">';
    html += '<div style="display:flex;align-items:flex-end;gap:4px;height:100px;border-bottom:1px solid var(--border)">';
    timeline.forEach((item) => {
      const height = Math.max(Math.round((item.cumulative_questions / maxQ) * 90), 2);
      const isNear = item.global_step === highlightGlobalStep;
      const color = isNear ? "var(--accent)" : "var(--accent3)";
      html += '<div style="flex:1;height:' + height + '%;background:' + color + ';border-radius:2px 2px 0 0;min-width:4px" title="g' + item.global_step + ": " + item.cumulative_questions + " total questions, " + item.cumulative_experiments + ' experiments"></div>';
    });
    html += "</div>";
    html += '<div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-muted);margin-top:4px"><span>g' + timeline[0].global_step + "</span><span>g" + timeline[timeline.length - 1].global_step + "</span></div>";
    html += "</div></div>";
  }

  let lastEp = -1;
  timeline.forEach((item) => {
    if (item.episode_idx !== lastEp) {
      lastEp = item.episode_idx;
      html += '<div style="padding:8px 16px;font-size:12px;font-weight:600;color:var(--accent);background:var(--bg);border-bottom:1px solid var(--border);margin-top:8px">Episode ' + item.episode_idx + "</div>";
    }

    const isHighlighted = item.global_step === highlightGlobalStep;
    const borderStyle = isHighlighted ? "border-left:3px solid var(--accent)" : "border-left:3px solid var(--surface2)";
    let eventHtml = '<div style="padding:10px 14px;margin-bottom:6px;background:var(--surface);border:1px solid var(--border);border-radius:6px;' + borderStyle + '">';
    eventHtml += '<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">';
    eventHtml += '<span style="font-size:12px;font-weight:600;color:var(--accent)">g' + item.global_step + "</span>";
    eventHtml += '<span style="font-size:11px;color:var(--text-muted)">ep' + item.episode_idx + " step " + item.step + "</span>";
    if (item.experiment_plan) {
      eventHtml += '<span style="font-size:11px;padding:2px 8px;background:rgba(63,185,80,0.15);color:var(--accent2);border-radius:4px;font-weight:600">new experiment</span>';
    }
    eventHtml += "</div>";

    // Show selected question
    if (item.selected_question_index != null) {
      const qText = item.selected_question_text || ("Q" + (item.selected_question_index + 1));
      eventHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">SELECTED QUESTION (Q' + (item.selected_question_index + 1) + "):</div>";
      eventHtml += '<div style="font-size:12px;padding:6px 10px;margin-bottom:8px;background:var(--bg);border:1px solid var(--accent3);border-radius:4px;color:var(--accent3)">' + esc(qText) + "</div>";
    }

    if (item.experiment_plan) {
      eventHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">FORMULATED EXPERIMENT:</div>';
      eventHtml += '<div style="font-size:12px;padding:6px 10px;background:var(--bg);border:1px solid var(--accent2);border-radius:4px">' + esc(item.experiment_plan) + "</div>";
    }
    eventHtml += "</div>";
    html += eventHtml;
  });

  c.innerHTML = html;
}

init();
