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
let dynamicRuns = null;
let currentDynamicRun = null;

const VIEWER_CONFIG = window.STEPWISE_EB_VIEWER_CONFIG || {};
if (VIEWER_CONFIG.dataIndexPath && !/^https?:\/\//.test(VIEWER_CONFIG.dataIndexPath)) {
  VIEWER_CONFIG.dataIndexPath = new URL(VIEWER_CONFIG.dataIndexPath, window.location.href).toString();
}
const TABS = [
  ["overview", "Overview"],
  ["experiments", "Questions & Experiments"],
  ["theories", "Theories"],
  ["frontier", "Frontier (B/P)"],
  ["agent_messages", "Agent Messages"],
  ["artifacts", "Artifacts"],
  ["feedback", "Feedback"],
  ["trajectory", "Trajectory"],
  ["combined_trajectory", "Cross-Episode Traj"],
  ["qa_timeline", "Q&A Timeline"],
  ["experiment_timeline", "Experiment Timeline"],
  ["logs", "Improve Log"],
];

// Tabs that only make sense for stepwise_eb_learn runs (they depend on
// belief/perception/QA/experiment artifacts that the simple + openhands
// baselines never produce). Hidden automatically when those artifacts are
// absent; see isEBRun / visibleTabs below.
const EB_ONLY_TABS = new Set([
  "experiments", "artifacts", "feedback",
  "qa_timeline", "experiment_timeline", "logs", "frontier",
]);

function isEBRun() {
  if (!DATA || !Array.isArray(DATA.steps)) return false;
  return DATA.steps.some((s) =>
    s.has_experiment_log || s.has_extraction_log ||
    s.has_improve_log || s.has_beliefs || s.has_trim_log ||
    s.has_question_selection_log || s.has_critical_id_log ||
    s.did_critical_id
  );
}

// The Theories tab is meaningful only for Plan B (theory_entropy) runs, which
// write theory_log.json per step. Hidden otherwise.
function hasTheoryData() {
  if (!DATA || !Array.isArray(DATA.steps)) return false;
  return DATA.steps.some((s) => s.has_theory_log);
}

// The Frontier tab is meaningful only for gepa_frontier runs, which learn a
// set of competing {perception, world_knowledge} candidates and write a
// non-empty frontier.json. Hidden otherwise.
function hasFrontierData() {
  if (!DATA || !Array.isArray(DATA.steps)) return false;
  return DATA.steps.some((s) => s.has_frontier);
}

function visibleTabs() {
  let tabs = isEBRun() ? TABS : TABS.filter((t) => !EB_ONLY_TABS.has(t[0]));
  if (!hasTheoryData()) tabs = tabs.filter((t) => t[0] !== "theories");
  if (!hasFrontierData()) tabs = tabs.filter((t) => t[0] !== "frontier");
  return tabs;
}

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

// Render a unified-diff payload (with or without ```diff fences) as a colored
// per-line block: `@@` headers in accent, `+` lines green, `-` lines red.
function renderUnifiedDiffPayload(diffText) {
  let body = (diffText || "").trim();
  body = body.replace(/^```(?:diff|patch)?\s*\n?/i, "").replace(/\n?```\s*$/, "");
  const lines = body.split("\n");
  let html = '<div style="font-family:var(--font-mono);font-size:12px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:8px 12px;overflow-x:auto;white-space:pre;line-height:1.5">';
  for (const line of lines) {
    let style;
    if (line.startsWith("@@")) {
      style = "color:var(--accent3);font-weight:600";
    } else if (line.startsWith("+++") || line.startsWith("---")) {
      style = "color:var(--text-muted)";
    } else if (line.charAt(0) === "+") {
      style = "color:#56d364;background:rgba(63,185,80,0.10)";
    } else if (line.charAt(0) === "-") {
      style = "color:#f85149;background:rgba(248,81,73,0.10)";
    } else {
      style = "color:var(--text)";
    }
    html += '<div style="' + style + '">' + esc(line || " ") + "</div>";
  }
  html += "</div>";
  return html;
}

// Render an LLM response, special-casing <updated_perception>...</updated_perception>
// when its body looks like a unified diff. Other text is rendered as a normal <pre>.
function renderResponseBody(response) {
  const fallback = '<pre style="max-height:400px;margin:0;border:none;padding:0;background:transparent">' + esc(response || "") + "</pre>";
  if (!response) return fallback;
  const upRe = /<updated_perception>([\s\S]*?)<\/updated_perception>/;
  const m = upRe.exec(response);
  if (!m) return fallback;
  const inner = m[1] || "";
  const isDiff = /```diff/i.test(inner) || /(^|\n)\s*@@[^\n]*@@/.test(inner);
  if (!isDiff) return fallback;
  const before = response.substring(0, m.index);
  const after = response.substring(m.index + m[0].length);
  let html = "";
  if (before.trim()) {
    html += '<pre style="max-height:none;margin:0 0 6px 0;border:none;padding:0;background:transparent">' + esc(before) + "</pre>";
  }
  html += '<div style="font-size:10px;text-transform:uppercase;color:var(--accent3);margin:4px 0;font-weight:600">&lt;updated_perception&gt; (unified diff)</div>';
  html += renderUnifiedDiffPayload(inner);
  if (after.trim()) {
    html += '<pre style="max-height:none;margin:6px 0 0 0;border:none;padding:0;background:transparent">' + esc(after) + "</pre>";
  }
  return html;
}

// Render a small validated/rejected/changed badge for a Track 1b/Track 2 turn.
// Returns empty string when the turn was logged before the diff-mode rollout
// (legacy turns don't carry `validated`).
function validationBadgesHtml(turn) {
  if (!turn || turn.validated === undefined || turn.validated === null) return "";
  if (turn.validated) {
    const changedTxt = turn.perception_changed ? "perception updated" : "no change";
    return '<span style="background:rgba(63,185,80,0.15);color:var(--accent2);padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin-left:6px" title="Perception code validated successfully">VALIDATED · ' +
      esc(changedTxt) + "</span>";
  }
  const errFull = turn.validation_error || "";
  const errShort = errFull.split("\n")[0].slice(0, 80);
  return '<span title="' + esc(errFull) +
    '" style="background:rgba(248,81,73,0.15);color:var(--danger);padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin-left:6px">REJECTED' +
    (errShort ? " · " + esc(errShort) : "") + "</span>";
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
  return !currentLogDir() && !!currentRunId();
}

function apiUrl(path, extraParams) {
  const base = { log_dir: currentLogDir() };
  const run = currentRunId();
  if (run) base.run = run;
  const params = new URLSearchParams({ ...base, ...extraParams });
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

  function addPart(slot, stepMeta, kind, label, accent) {
    if (seenSlots.has(slot)) return;
    const part = makePromptImagePart(slot, stepMeta, kind, label, accent);
    if (part) parts.push(part);
    seenSlots.add(slot);
  }

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
      addPart(slot, stepMeta, "before", "Image " + slot + " - g" + gs + " pre", false);
    }
    const postRe = /<(?:post_state|resulting_state)>\s*\(image\s+(\d+)\)/g;
    let postMatch;
    while ((postMatch = postRe.exec(body)) !== null) {
      const slot = parseInt(postMatch[1], 10);
      addPart(slot, stepMeta, "after", "Image " + slot + " - g" + gs + " result", false);
    }
  }

  const perceptionBlockRe = /<perception_example\s+step="(\d+)">([\s\S]*?)<\/perception_example>/g;
  let perceptionMatch;
  while ((perceptionMatch = perceptionBlockRe.exec(promptText)) !== null) {
    const gs = parseInt(perceptionMatch[1], 10);
    const stepMeta = resolveStepByGlobal(gs);
    if (!stepMeta) continue;
    const body = perceptionMatch[2];
    const beforeRe = /<before_action\s+\(image\s+(\d+)\)>/g;
    let beforeMatch;
    while ((beforeMatch = beforeRe.exec(body)) !== null) {
      const slot = parseInt(beforeMatch[1], 10);
      addPart(slot, stepMeta, "before", "Image " + slot + " - g" + gs + " before", false);
    }
    const afterRe = /<after_action\s+\(image\s+(\d+)\)>/g;
    let afterMatch;
    while ((afterMatch = afterRe.exec(body)) !== null) {
      const slot = parseInt(afterMatch[1], 10);
      addPart(slot, stepMeta, "after", "Image " + slot + " - g" + gs + " after", false);
    }
  }

  if (opts.currentStep) {
    const currentBlockRe = /=== CURRENT STATE[^\n]* ===[\s\S]*?<(?:pre_state|raw_state)>\s*\(image\s+(\d+)\)/;
    const currentMatch = currentBlockRe.exec(promptText);
    if (currentMatch) {
      const slot = parseInt(currentMatch[1], 10);
      addPart(
        slot,
        opts.currentStep,
        "before",
        "Image " + slot + " - current (g" + opts.currentStep.global_step + ")",
        true
      );
    }
  }

  parts.sort((a, b) => a.slot - b.slot);
  return parts;
}

function parseSamplePromptImages(promptText) {
  if (!promptText) return [];
  const parts = [];
  const seenSteps = new Set();
  const sampleRe = /<(?:perception_example|execution_sample)\s+step="(\d+)">/g;
  let sampleIdx = 0;
  let sampleMatch;
  while ((sampleMatch = sampleRe.exec(promptText)) !== null) {
    const gs = parseInt(sampleMatch[1], 10);
    if (seenSteps.has(gs)) continue;
    seenSteps.add(gs);
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
    if (parts.length === 0) {
      const samples = parseSamplePromptImages(promptText);
      samples.forEach((part) => parts.push(part));
    }
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

function messageAttachmentPaths(msg) {
  if (!msg) return [];
  const paths = [];
  const seen = new Set();
  function addPath(path) {
    if (!path || seen.has(path)) return;
    seen.add(path);
    paths.push(path);
  }
  addPath(msg.attachment_path);
  if (Array.isArray(msg.attachment_paths)) {
    msg.attachment_paths.forEach((path) => {
      addPath(path);
    });
  }
  if (paths.length === 0 && typeof msg.content === "string" && msg.content.includes("[image attached]")) {
    addPath("obs_before.png");
  }
  return paths;
}

function messageAttachmentsHtml(msg, stepMeta) {
  const paths = messageAttachmentPaths(msg);
  if (paths.length === 0 || !stepMeta) return "";
  let html = '<div style="margin:8px 0 10px"><div style="font-size:10px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px;font-weight:600">Message images (' +
    paths.length + ')</div><div style="display:flex;gap:6px;flex-wrap:wrap;padding:8px;background:var(--surface2);border:1px solid var(--border);border-radius:4px">';
  paths.forEach((path, i) => {
    html += '<div style="text-align:center"><div style="font-size:10px;color:var(--text-muted);margin-bottom:2px">Attachment ' + (i + 1) + '</div>' +
      '<img src="' + stepLocalImageUrl(stepMeta.episode_idx, stepMeta.step, path) + '" style="max-width:160px;image-rendering:pixelated;border:1px solid var(--border);border-radius:3px" /></div>';
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

async function loadDynamicRuns() {
  const logDir = currentLogDir();
  if (!logDir) return null;
  const params = new URLSearchParams({ log_dir: logDir });
  return await fetchJson("/api/runs?" + params.toString());
}

function dynamicRunLabel(run) {
  if (!run) return "";
  const summary = [];
  if (run.episodes) summary.push(run.episodes + " ep");
  if (run.steps) summary.push(run.steps + " steps");
  const name = run.name || run.id || "(this directory)";
  return summary.length ? name + " (" + summary.join(", ") + ")" : name;
}

function populateDynamicRunSelectors() {
  if (!dynamicRuns || !Array.isArray(dynamicRuns.runs)) return;
  const landingSelect = document.getElementById("landing-dynamic-run-select");
  const topbarSelect = document.getElementById("topbar-run-select");
  const current = currentRunId() || "";

  const options = dynamicRuns.runs.length > 1
    ? ['<option value="">Select a run</option>']
    : [];
  dynamicRuns.runs.forEach((run) => {
    const value = run.id || "";
    const selected = value === current ? " selected" : "";
    options.push('<option value="' + esc(value) + '"' + selected + ">" + esc(dynamicRunLabel(run)) + "</option>");
  });

  if (landingSelect) {
    landingSelect.innerHTML = options.join("");
    landingSelect.value = current;
    landingSelect.onchange = updateLandingDynamicRunMeta;
    updateLandingDynamicRunMeta();
  }
  if (topbarSelect) {
    topbarSelect.innerHTML = options.join("");
    topbarSelect.value = current;
    topbarSelect.onchange = function () {
      navigateWith({ run: this.value || null });
    };
  }
}

function updateLandingDynamicRunMeta() {
  const meta = document.getElementById("landing-dynamic-run-meta");
  if (!meta || !dynamicRuns) return;
  const subtitle = document.getElementById("dynamic-runs-subtitle");
  if (subtitle) subtitle.textContent = "Found " + dynamicRuns.runs.length +
    " run(s) under " + dynamicRuns.log_dir + ". Pick one to open.";
  const selected = document.getElementById("landing-dynamic-run-select").value;
  const run = dynamicRuns.runs.find((item) => (item.id || "") === selected);
  if (!run) { meta.textContent = ""; return; }
  meta.textContent = run.path;
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
  const dynamicRunsPanel = document.getElementById("dynamic-runs-panel");
  const staticPanel = document.getElementById("static-panel");
  const allowDynamic = !!VIEWER_CONFIG.allowDynamicInput;
  const allowStatic = !!(staticIndex && Array.isArray(staticIndex.runs));
  const showDynamicRuns = !!(dynamicRuns && Array.isArray(dynamicRuns.runs) && dynamicRuns.runs.length);
  dynamicPanel.classList.toggle("hidden", !allowDynamic || showDynamicRuns);
  dynamicRunsPanel.classList.toggle("hidden", !showDynamicRuns);
  staticPanel.classList.toggle("hidden", !allowStatic || showDynamicRuns);
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
  const topbarSelect = document.getElementById("topbar-run-select");

  if (isStaticMode() && currentStaticRun) {
    topbarDir.textContent = currentStaticRun.title || DATA.log_dir_name;
    runWrap.style.display = "inline-flex";
    topbarSelect.value = currentStaticRun.id;
  } else if (isDynamicMode() && dynamicRuns && dynamicRuns.runs.length > 1) {
    topbarDir.textContent = dynamicRuns.log_dir;
    runWrap.style.display = "inline-flex";
    topbarSelect.value = currentRunId() || "";
  } else {
    topbarDir.textContent = DATA.log_dir_name;
    runWrap.style.display = "none";
  }
  const criticalSteps = DATA.steps.filter((s) => s.did_critical_id && s.critical === true).length;
  const classifiedSteps = DATA.steps.filter((s) => s.did_critical_id).length;
  topbarCost.textContent = "Total cost: $" + (DATA.total_cost || 0).toFixed(4) +
    (classifiedSteps ? " | critical " + criticalSteps + "/" + classifiedSteps : "");

  let title = DATA.log_dir_name;
  if (isStaticMode() && currentStaticRun) title = currentStaticRun.title || title;
  else if (isDynamicMode() && currentDynamicRun && currentDynamicRun.name) title = currentDynamicRun.name;
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

function loadDynamicRunFromLanding() {
  const select = document.getElementById("landing-dynamic-run-select");
  const runId = select ? select.value : "";
  if (!runId && dynamicRuns && dynamicRuns.runs.length > 1) {
    setLandingError("Please choose a run.");
    return;
  }
  navigateWith({ run: runId || null });
}

function clearLogDirFromLanding() {
  navigateWith({ log_dir: null, run: null });
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

  dynamicRuns = null;
  currentDynamicRun = null;
  if (isDynamicMode()) {
    try {
      dynamicRuns = await loadDynamicRuns();
    } catch (e) {
      configureLanding();
      showLanding();
      setLandingError("Failed to list runs: " + e.message);
      return;
    }

    const runs = (dynamicRuns && dynamicRuns.runs) || [];
    if (runs.length === 0) {
      configureLanding();
      showLanding();
      setLandingError("No runs found under " + (dynamicRuns ? dynamicRuns.log_dir : currentLogDir()));
      return;
    }

    const requested = currentRunId() || "";
    if (runs.length === 1) {
      // Exactly one run: auto-select (log_dir may be the run itself, in which case id="").
      currentDynamicRun = runs[0];
      if ((runs[0].id || "") !== requested) {
        navigateWith({ run: runs[0].id || null });
        return;
      }
    } else {
      currentDynamicRun = runs.find((r) => (r.id || "") === requested) || null;
      if (!currentDynamicRun) {
        populateDynamicRunSelectors();
        configureLanding();
        showLanding();
        return;
      }
    }
    populateDynamicRunSelectors();
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
    const criticalBadge = step.did_critical_id
      ? '<span class="critical-badge ' + (step.critical ? "yes" : "no") + '" title="' + esc(step.critical_reason || (step.critical ? "critical transition" : "non-critical transition")) + '">' + (step.critical ? "C" : "skip") + "</span>"
      : "";
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
      statusDot + criticalBadge + doneMark + phaseBadge + lvlBadge +
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
  const criticalPill = step.did_critical_id
    ? ' <span style="font-size:11px;padding:2px 8px;border-radius:10px;background:' + (step.critical ? "rgba(63,185,80,0.15)" : "rgba(139,148,158,0.14)") + ';color:' + (step.critical ? "var(--accent2)" : "var(--text-muted)") + ';font-weight:700;vertical-align:middle">' + (step.critical ? "critical" : "non-critical") + "</span>"
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
    '<h1 style="margin:0;font-size:18px">Step ' + step.step + phasePill + criticalPill + ' <span style="color:var(--text-muted);font-size:14px;font-weight:400">ep' + step.episode_idx + " | global " + step.global_step + envBadge + "</span></h1>" +
    '<span style="font-size:12px;color:var(--text-muted);margin-left:auto">action: <b>' + esc(step.action || "...") + "</b> | reward: " + (step.action ? Number(step.reward).toFixed(2) : "—") + " | cost: $" + Number(step.step_total_cost).toFixed(4) + "</span>" +
    '<button style="background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px" onclick="showStep(' + Math.min(total - 1, idx + 1) + ')" ' + (idx >= total - 1 ? "disabled" : "") + ">&#8594;</button>" +
    "</div>";

  const tabsToShow = visibleTabs();
  if (!tabsToShow.some((t) => t[0] === currentTab)) {
    currentTab = "overview";
  }
  html += '<div class="tabs">';
  tabsToShow.forEach((tab) => {
    html += '<div class="tab ' + (currentTab === tab[0] ? "active" : "") + '" onclick="currentTab=\'' + tab[0] + "';renderStep(" + idx + ')">' + tab[1] + "</div>";
  });
  html += "</div>";

  const containerId = currentTab.replace(/_/g, "-") + "-container";
  html += '<div id="' + containerId + '"><div class="loading">Loading...</div></div>';
  mc.innerHTML = html;

  if (currentTab === "overview") loadStepDetailForTab(step.episode_idx, step.step, "overview", step);
  else if (currentTab === "artifacts") loadStepDetailForTab(step.episode_idx, step.step, "artifacts");
  else if (currentTab === "experiments") loadStepDetailForTab(step.episode_idx, step.step, "experiments");
  else if (currentTab === "theories") loadStepDetailForTab(step.episode_idx, step.step, "theories");
  else if (currentTab === "frontier") loadStepDetailForTab(step.episode_idx, step.step, "frontier");
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
  else if (tab === "theories") renderTheories(data);
  else if (tab === "frontier") renderFrontier(data);
  else if (tab === "feedback") renderFeedback(data);
  else if (tab === "agent_messages") renderAgentMessages(data);
  else if (tab === "logs") renderLogs(data);
}

function scoringArtifactSummaryRows(artifact) {
  const rows = [];
  const isProbeSelection = artifact.source === "online_probe_selection";
  if (artifact.source) rows.push(["Source", artifact.source]);
  if (artifact.method) rows.push(["Method", artifact.method]);
  if (artifact.qa_source) rows.push(["QA Source", artifact.qa_source]);
  if (artifact.num_qa != null) rows.push(["Questions", artifact.num_qa]);
  if (artifact.num_qa_before_trim != null) rows.push([isProbeSelection ? "Questions Before Dedup" : "Questions Before Trim", artifact.num_qa_before_trim]);
  if (artifact.num_qa_after_trim != null) rows.push([isProbeSelection ? "Questions After Dedup" : "Questions After Trim", artifact.num_qa_after_trim]);
  if (artifact.num_unanswered_scored != null) rows.push(["Unanswered Scored", artifact.num_unanswered_scored]);
  if (artifact.num_unanswered_projection != null) rows.push(["Projection Questions", artifact.num_unanswered_projection]);
  if (artifact.num_unanswered_before_trim != null) rows.push([isProbeSelection ? "Unanswered Before Dedup" : "Unanswered Before Trim", artifact.num_unanswered_before_trim]);
  if (artifact.num_unanswered_after_trim != null) rows.push([isProbeSelection ? "Unanswered After Dedup" : "Unanswered After Trim", artifact.num_unanswered_after_trim]);
  if (artifact.cap_unanswered != null) rows.push(["Unanswered Cap", artifact.cap_unanswered]);
  if (artifact.maintained_bank_preserved != null) rows.push(["Maintained Bank Preserved", artifact.maintained_bank_preserved ? "yes" : "no"]);
  if (artifact.overlap_at_k != null) rows.push(["Overlap@k", Number(artifact.overlap_at_k).toFixed(3)]);
  if (artifact.cost_usd != null) rows.push(["Cost", "$" + Number(artifact.cost_usd).toFixed(4)]);
  if (artifact.did_trim != null) rows.push([isProbeSelection ? "Trimmed Maintained Bank" : "Did Trim", artifact.did_trim ? "yes" : "no"]);
  return rows;
}

function renderScoringRankedTable(entries, keptQuestions, droppedQuestions) {
  const kept = new Set(keptQuestions || []);
  const dropped = new Set(droppedQuestions || []);
  if (!Array.isArray(entries) || entries.length === 0) return '<div style="color:var(--text-muted)">No ranked unanswered questions recorded.</div>';
  let html = '<table class="data-table"><tr><th>Rank</th><th>Question</th><th>Score</th><th>Src Step</th><th>Status</th></tr>';
  entries.forEach((entry, idx) => {
    let status = "scored";
    if (kept.has(entry.question)) status = "kept";
    else if (dropped.has(entry.question)) status = "dropped";
    html += '<tr><td>' + (idx + 1) + '</td><td style="max-width:420px">' + esc(entry.question) + '</td><td style="font-family:var(--font-mono)">' +
      Number(entry.score || 0).toFixed(4) + '</td><td>' + esc(entry.source_step) + '</td><td>' + esc(status) + '</td></tr>';
  });
  html += "</table>";
  return html;
}

function renderTieBreakSection(tieBreak) {
  if (!tieBreak || typeof tieBreak !== "object") return "";
  if (!tieBreak.executed && !tieBreak.reason && !Array.isArray(tieBreak.candidate_questions)) return "";

  let html = '<table class="data-table" style="margin-bottom:12px">';
  html += '<tr><th style="width:180px">Executed</th><td>' + (tieBreak.executed ? "yes" : "no") + "</td></tr>";
  if (tieBreak.reason) html += '<tr><th>Reason</th><td>' + esc(tieBreak.reason) + "</td></tr>";
  if (tieBreak.top_score != null) html += '<tr><th>Top Score</th><td style="font-family:var(--font-mono)">' + Number(tieBreak.top_score).toFixed(6) + "</td></tr>";
  if (tieBreak.selected_source_index != null) html += '<tr><th>Selected Bank #</th><td>Q' + (tieBreak.selected_source_index + 1) + "</td></tr>";
  if (tieBreak.selected_question) html += '<tr><th>Selected Question</th><td>' + esc(tieBreak.selected_question) + "</td></tr>";
  html += "</table>";

  const candidates = Array.isArray(tieBreak.candidate_questions) ? tieBreak.candidate_questions : [];
  if (candidates.length) {
    html += '<div style="font-weight:600;font-size:12px;color:var(--text-muted);margin:12px 0 6px">Tied Top Questions</div>';
    html += '<table class="data-table" style="margin-bottom:12px"><tr><th>Bank #</th><th>Question</th><th>Score</th><th>Src Step</th></tr>';
    candidates.forEach((item) => {
      html += '<tr><td>' + (item.source_index != null ? "Q" + (item.source_index + 1) : "") +
        '</td><td style="max-width:420px">' + esc(item.question || "") +
        '</td><td style="font-family:var(--font-mono)">' + Number(item.score || 0).toFixed(6) +
        '</td><td>' + esc(item.source_step) + "</td></tr>";
    });
    html += "</table>";
  }

  if (tieBreak.prompt || tieBreak.response) {
    html += promptResponseBlock("Tie-break Selection", tieBreak.prompt, tieBreak.response);
  }
  return html;
}

function renderQuestionSnapshotTable(items, options) {
  const opts = options || {};
  const sourceIndices = Array.isArray(opts.sourceIndices) ? opts.sourceIndices : null;
  if (!Array.isArray(items) || items.length === 0) {
    return '<div style="color:var(--text-muted)">No questions recorded.</div>';
  }
  let html = '<table class="data-table"><tr><th>#</th>';
  if (sourceIndices) html += "<th>Bank #</th>";
  html += "<th>Question</th><th>Answer</th><th>Evidence</th><th>Src Step</th></tr>";
  items.forEach((item, i) => {
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
    html += '<tr><td style="color:var(--text-muted)">Q' + (i + 1) + "</td>";
    if (sourceIndices) {
      const src = sourceIndices[i];
      html += '<td style="color:var(--text-muted)">' + (src != null ? "Q" + (src + 1) : "") + "</td>";
    }
    html += '<td style="max-width:320px">' + esc(item.question || "") + "</td>" +
      '<td><span class="verdict ' + verdictClass + '">' + answer + "</span></td>" +
      '<td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(item.evidence || "") + '">' + esc(item.evidence || "") + "</td>" +
      "<td>" + esc(item.source_step) + "</td></tr>";
  });
  html += "</table>";
  return html;
}

function orderedQaForDisplay(data, qa) {
  const items = Array.isArray(qa) ? qa : [];
  const preordered = items === data.qa_pairs && Array.isArray(data.qa_pairs_ordered) && data.qa_pairs_ordered.length === items.length
    ? data.qa_pairs_ordered
    : null;
  if (preordered) {
    return {
      items: preordered,
      sourceIndices: Array.isArray(data.qa_pairs_ordered_source_indices) ? data.qa_pairs_ordered_source_indices : null,
    };
  }

  const expLog = data.experiment_log || {};
  const qSelectLog = data.question_selection_log || data.trim_log || {};
  let order = Array.isArray(expLog.qa_pairs_for_experiment_source_indices)
    ? expLog.qa_pairs_for_experiment_source_indices
    : null;
  if (!order || order.length === 0) {
    order = Array.isArray(qSelectLog.experiment_source_indices) ? qSelectLog.experiment_source_indices : null;
  }
  if (!order || order.length === 0) return { items: items, sourceIndices: null };

  const seen = new Set();
  const orderedItems = [];
  const sourceIndices = [];
  order.forEach((idx) => {
    if (!Number.isInteger(idx) || idx < 0 || idx >= items.length || seen.has(idx)) return;
    seen.add(idx);
    orderedItems.push(items[idx]);
    sourceIndices.push(idx);
  });
  items.forEach((item, idx) => {
    if (seen.has(idx)) return;
    orderedItems.push(item);
    sourceIndices.push(idx);
  });
  return { items: orderedItems, sourceIndices: sourceIndices };
}

function renderAnswerVectorTable(questionLog) {
  const others = Array.isArray(questionLog.other_questions) ? questionLog.other_questions : [];
  const aPos = Array.isArray(questionLog.a_pos) ? questionLog.a_pos : [];
  const aNeg = Array.isArray(questionLog.a_neg) ? questionLog.a_neg : [];
  if (!others.length) return '<div style="color:var(--text-muted)">No paired questions for answer-vector comparison.</div>';
  let html = '<table class="data-table"><tr><th>#</th><th>Other Question</th><th>a_pos</th><th>a_neg</th><th>Diff</th></tr>';
  others.forEach((question, idx) => {
    const pos = aPos[idx];
    const neg = aNeg[idx];
    const diff = pos !== neg;
    html += '<tr><td>' + (idx + 1) + '</td><td style="max-width:420px">' + esc(question) + '</td><td style="font-family:var(--font-mono)">' + esc(pos) +
      '</td><td style="font-family:var(--font-mono)">' + esc(neg) + '</td><td>' + (diff ? '<span class="verdict verdict-incorrect">changed</span>' : '<span class="verdict verdict-correct">same</span>') + '</td></tr>';
  });
  html += "</table>";
  return html;
}

function lcsDiffRows(aLines, bLines) {
  const m = aLines.length;
  const n = bLines.length;
  const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (aLines[i] === bLines[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
      else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const rows = [];
  let i = 0;
  let j = 0;
  while (i < m && j < n) {
    if (aLines[i] === bLines[j]) {
      rows.push({ type: "same", text: aLines[i] });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      rows.push({ type: "only_neg", text: aLines[i] });
      i++;
    } else {
      rows.push({ type: "only_pos", text: bLines[j] });
      j++;
    }
  }
  while (i < m) {
    rows.push({ type: "only_neg", text: aLines[i] });
    i++;
  }
  while (j < n) {
    rows.push({ type: "only_pos", text: bLines[j] });
    j++;
  }
  return rows;
}

function renderBeliefDiff(questionLog) {
  const bPos = String(questionLog.b_pos || "");
  const bNeg = String(questionLog.b_neg || "");
  if (!bPos && !bNeg) return '<div style="color:var(--text-muted)">No rewritten beliefs recorded.</div>';
  if (bPos === bNeg) return '<div style="color:var(--text-muted)">No difference between b_pos and b_neg.</div>';

  const rows = lcsDiffRows(bNeg.split("\n"), bPos.split("\n"));
  let html = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">Lines only in <span style="color:var(--accent3)">b_pos</span> are marked with <strong>+</strong>; lines only in <span style="color:var(--accent2)">b_neg</span> are marked with <strong>-</strong>.</div>';
  html += '<table class="data-table"><tr><th style="width:60px">Type</th><th>Line</th></tr>';
  rows.forEach((row) => {
    let label = "=";
    let bg = "transparent";
    let color = "var(--text)";
    if (row.type === "only_pos") {
      label = "+";
      bg = "rgba(35, 180, 90, 0.12)";
      color = "var(--accent3)";
    } else if (row.type === "only_neg") {
      label = "-";
      bg = "rgba(220, 110, 110, 0.12)";
      color = "#d87";
    }
    html += '<tr><td style="font-family:var(--font-mono);color:' + color + ';background:' + bg + '">' + label + '</td>' +
      '<td style="white-space:pre-wrap;font-family:var(--font-mono);background:' + bg + '">' + esc(row.text) + '</td></tr>';
  });
  html += "</table>";
  return html;
}

function renderScoringQuestionLog(questionLog, index) {
  const summary = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:10px">' +
    'score <span style="font-family:var(--font-mono);color:var(--accent3)">' + Number(questionLog.score || 0).toFixed(4) + '</span>' +
    ' | cost <span style="font-family:var(--font-mono);color:var(--text-muted)">$' + Number(questionLog.cost || 0).toFixed(4) + '</span>' +
    ' | src step ' + esc(questionLog.source_step) + '</div>';
  const beliefHtml =
    '<div style="margin-bottom:12px"><div style="font-weight:600;font-size:12px;color:var(--accent3);margin-bottom:4px">b_pos</div><pre style="max-height:320px">' +
    esc(questionLog.b_pos || "") + '</pre></div>' +
    '<div style="margin-bottom:12px"><div style="font-weight:600;font-size:12px;color:var(--accent2);margin-bottom:4px">b_neg</div><pre style="max-height:320px">' +
    esc(questionLog.b_neg || "") + '</pre></div>';
  const beliefDiffHtml = renderBeliefDiff(questionLog);
  const promptHtml =
    promptResponseBlock("Rewrite b_pos", questionLog.rewrite_pos && questionLog.rewrite_pos.prompt, questionLog.rewrite_pos && questionLog.rewrite_pos.response) +
    promptResponseBlock("Rewrite b_neg", questionLog.rewrite_neg && questionLog.rewrite_neg.prompt, questionLog.rewrite_neg && questionLog.rewrite_neg.response) +
    promptResponseBlock("Predict a_pos", questionLog.predict_pos && questionLog.predict_pos.prompt, questionLog.predict_pos && questionLog.predict_pos.response) +
    promptResponseBlock("Predict a_neg", questionLog.predict_neg && questionLog.predict_neg.prompt, questionLog.predict_neg && questionLog.predict_neg.response);
  const vectorHtml = renderAnswerVectorTable(questionLog);
  return collapsible(
    'Q' + (index + 1) + ': ' + esc(questionLog.question),
    summary + beliefHtml + collapsible("Belief Diff (b_pos vs b_neg)", beliefDiffHtml, true) + collapsible("Answer Vector Diff", vectorHtml, true) + collapsible("Prompt / Response Chain", promptHtml || '<div style="color:var(--text-muted)">No prompt artifacts recorded.</div>', false),
    false,
  );
}

function renderScoringArtifact(title, artifact) {
  if (!artifact) return "";
  let html = "";
  const summaryRows = scoringArtifactSummaryRows(artifact);
  if (summaryRows.length) {
    html += '<table class="data-table" style="margin-bottom:12px">';
    summaryRows.forEach((row) => {
      html += '<tr><th style="width:220px">' + esc(row[0]) + '</th><td>' + esc(row[1]) + '</td></tr>';
    });
    html += "</table>";
  }
  if (Array.isArray(artifact.kept_unanswered_questions) && artifact.kept_unanswered_questions.length) {
    const keptLabel = artifact.source === "online_probe_selection" ? "Selected Probe Questions" : "Kept Unanswered";
    html += '<div style="margin-bottom:10px"><div style="font-weight:600;font-size:12px;color:var(--accent2);margin-bottom:4px">' + keptLabel + '</div><ol style="margin:0 0 0 20px">';
    artifact.kept_unanswered_questions.forEach((q) => { html += '<li>' + esc(q) + '</li>'; });
    html += "</ol></div>";
  }
  if (Array.isArray(artifact.dropped_unanswered_questions) && artifact.dropped_unanswered_questions.length) {
    html += '<div style="margin-bottom:10px"><div style="font-weight:600;font-size:12px;color:#d87; margin-bottom:4px">Dropped Unanswered</div><ol style="margin:0 0 0 20px">';
    artifact.dropped_unanswered_questions.forEach((q) => { html += '<li>' + esc(q) + '</li>'; });
    html += "</ol></div>";
  }
  if (Array.isArray(artifact.b_diff_top_k_questions) && artifact.b_diff_top_k_questions.length) {
    html += '<div style="margin-bottom:10px"><div style="font-weight:600;font-size:12px;color:var(--accent3);margin-bottom:4px">B-diff Top-k</div><ol style="margin:0 0 0 20px">';
    artifact.b_diff_top_k_questions.forEach((q) => { html += '<li>' + esc(q) + '</li>'; });
    html += "</ol></div>";
  }
  if (Array.isArray(artifact.llm_trim_kept_unanswered) && artifact.llm_trim_kept_unanswered.length) {
    html += '<div style="margin-bottom:10px"><div style="font-weight:600;font-size:12px;color:var(--purple);margin-bottom:4px">Live LLM-Kept Unanswered</div><ol style="margin:0 0 0 20px">';
    artifact.llm_trim_kept_unanswered.forEach((q) => { html += '<li>' + esc(q) + '</li>'; });
    html += "</ol></div>";
  }

  html += collapsible(
    "Ranked Unanswered Questions",
    renderScoringRankedTable(
      artifact.ranked_unanswered || [],
      artifact.kept_unanswered_questions,
      artifact.dropped_unanswered_questions,
    ),
    true,
  );

  const tieBreakHtml = renderTieBreakSection(artifact.tie_break || (artifact.scoring_log && artifact.scoring_log.tie_break));
  if (tieBreakHtml) {
    html += collapsible("Top-Score Tie-break", tieBreakHtml, true);
  }

  if (Array.isArray(artifact.experiment_questions) && artifact.experiment_questions.length) {
    let subsetHtml = '<table class="data-table"><tr><th>#</th><th>Bank #</th><th>Question</th><th>Answer</th><th>Src Step</th></tr>';
    artifact.experiment_questions.forEach((item, idx) => {
      const ans = item.answer === null || item.answer === undefined ? "UNANSWERED" : (item.answer ? "YES" : "NO");
      subsetHtml += '<tr><td>Q' + (idx + 1) + '</td><td>' +
        (item.source_index != null ? "Q" + (item.source_index + 1) : "") +
        '</td><td style="max-width:420px">' + esc(item.question || "") +
        '</td><td>' + esc(ans) + '</td><td>' + esc(item.source_step) + "</td></tr>";
    });
    subsetHtml += "</table>";
    html += collapsible("Experiment Prompt Question Subset", subsetHtml, true);
  }

  const selectionLog = artifact.selection_log || {};
  if (selectionLog.prompt || selectionLog.response || Array.isArray(selectionLog.selected_questions)) {
    let selHtml = "";
    if (Array.isArray(selectionLog.selected_questions) && selectionLog.selected_questions.length) {
      selHtml += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">Selected ' +
        selectionLog.selected_questions.length + " questions for experiment formulation.</div>";
      selHtml += '<table class="data-table" style="margin-bottom:12px"><tr><th>#</th><th>Bank #</th><th>Question</th><th>Answer</th><th>Src Step</th></tr>';
      selectionLog.selected_questions.forEach((item, idx) => {
        const ans = item.answer === null || item.answer === undefined ? "UNANSWERED" : (item.answer ? "YES" : "NO");
        selHtml += '<tr><td>Q' + (idx + 1) + '</td><td>' +
          (item.source_index != null ? "Q" + (item.source_index + 1) : "") +
          '</td><td style="max-width:420px">' + esc(item.question || "") +
          '</td><td>' + esc(ans) + '</td><td>' + esc(item.source_step) + "</td></tr>";
      });
      selHtml += "</table>";
    }
    if (selectionLog.prompt || selectionLog.response) {
      selHtml += promptResponseBlock("Probe Selection", selectionLog.prompt, selectionLog.response);
    }
    html += collapsible("Probe Selection", selHtml, false);
  }

  const dedupLog = artifact.dedup_log || {};
  if (dedupLog.prompt || dedupLog.response || Array.isArray(dedupLog.dropped_questions)) {
    let dedupHtml = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">Dropped duplicates: ' +
      (dedupLog.dropped_count || 0) + "</div>";
    if (Array.isArray(dedupLog.dropped_questions) && dedupLog.dropped_questions.length) {
      dedupHtml += '<ol style="margin:0 0 12px 20px">';
      dedupLog.dropped_questions.forEach((q) => { dedupHtml += '<li>' + esc(q) + "</li>"; });
      dedupHtml += "</ol>";
    }
    if (dedupLog.prompt || dedupLog.response) {
      dedupHtml += promptResponseBlock("De-duplication", dedupLog.prompt, dedupLog.response);
    }
    html += collapsible("Question Bank De-duplication", dedupHtml, false);
  }

  const scoreLog = artifact.scoring_log || {};
  const perQuestion = Array.isArray(scoreLog.per_question) ? scoreLog.per_question : [];
  let perQuestionHtml = "";
  perQuestion.forEach((questionLog, index) => {
    perQuestionHtml += renderScoringQuestionLog(questionLog, index);
  });
  html += collapsible(
    "Per-Question Artifacts (" + perQuestion.length + ")",
    perQuestionHtml || '<div style="color:var(--text-muted)">No per-question scoring artifacts recorded.</div>',
    false,
  );

  return collapsible(title, html, true);
}

function renderExperimentSelectionScoring(data) {
  const artifacts = data.question_scoring || {};
  let html = "";
  html += renderScoringArtifact("Online Scoring (full)", artifacts.online_full);
  html += renderScoringArtifact("Online Scoring (light)", artifacts.online_light);
  html += renderScoringArtifact("Online Scoring (theory-entropy)", artifacts.online_theory_entropy);
  if (html) return html;

  const qSelectLog = data.question_selection_log || {};
  const trimLog = data.trim_log || {};
  const selectionLog = qSelectLog.selection || trimLog.selection || {};
  const dedupLog = qSelectLog.dedup || trimLog.dedup || {};
  if (!(dedupLog.prompt || dedupLog.response || selectionLog.prompt || selectionLog.response)) {
    return "";
  }

  let selectionHtml = "";
  if (dedupLog.prompt || dedupLog.response) {
    selectionHtml += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">De-dup dropped ' +
      (dedupLog.dropped_count || 0) + " duplicate questions.</div>";
    selectionHtml += promptResponseBlock("De-duplication", dedupLog.prompt, dedupLog.response);
  }
  if (selectionLog.prompt || selectionLog.response) {
    selectionHtml += promptResponseBlock("Probe Selection", selectionLog.prompt, selectionLog.response);
  }
  return selectionHtml;
}

function experimentScoringLog(data) {
  const direct = data.experiment_scoring_log;
  if (direct && typeof direct === "object" && Object.keys(direct).length > 0) return direct;
  const expLog = data.experiment_log || {};
  const nested = expLog.experiment_scoring;
  if (nested && typeof nested === "object" && Object.keys(nested).length > 0) return nested;
  return null;
}

function sourceIndexLabel(idx) {
  return idx != null ? "Q" + (idx + 1) : "";
}

function renderSourceQuestionCoverage(indices, qaBank) {
  if (!Array.isArray(indices) || indices.length === 0) {
    return '<div style="color:var(--text-muted)">No YES-covered questions recorded.</div>';
  }
  let html = '<table class="data-table"><tr><th>Bank #</th><th>Question</th><th>Src Step</th></tr>';
  indices.forEach((idx) => {
    const qa = Array.isArray(qaBank) && Number.isInteger(idx) ? qaBank[idx] : null;
    html += '<tr><td>' + esc(sourceIndexLabel(idx)) + '</td><td style="max-width:520px">' +
      esc((qa && qa.question) || "") + '</td><td>' + esc(qa && qa.source_step) + "</td></tr>";
  });
  html += "</table>";
  return html;
}

function renderExperimentCandidateScoring(data) {
  const scoreLog = experimentScoringLog(data);
  if (!scoreLog || scoreLog.mode !== "score_topk") return "";

  const expLog = data.experiment_log || {};
  const qaBank = expLog.qa_pairs_at_formulation || data.qa_pairs || [];
  const candidates = Array.isArray(scoreLog.candidates) ? scoreLog.candidates : [];
  const winnerIndex = scoreLog.winner_index;

  let html = '<table class="data-table" style="margin-bottom:12px">';
  html += '<tr><th style="width:220px">Mode</th><td>' + esc(scoreLog.mode) + "</td></tr>";
  html += '<tr><th>Candidates Scored</th><td>' + candidates.length + "</td></tr>";
  if (scoreLog.topk_unanswered_source_indices) html += '<tr><th>Fresh Candidate Targets</th><td>' + scoreLog.topk_unanswered_source_indices.map(sourceIndexLabel).join(", ") + "</td></tr>";
  if (scoreLog.unanswered_pool_source_indices) html += '<tr><th>Scored Against Pool</th><td>' + scoreLog.unanswered_pool_source_indices.length + " unanswered questions</td></tr>";
  if (scoreLog.winner_kind) html += '<tr><th>Winner</th><td><strong style="color:var(--accent2)">' + esc(scoreLog.winner_kind) + "</strong>" +
    (scoreLog.winner_source_index != null ? " / bank " + esc(sourceIndexLabel(scoreLog.winner_source_index)) : "") +
    (scoreLog.winner_score != null ? " / score " + Number(scoreLog.winner_score || 0).toFixed(0) : "") + "</td></tr>";
  if (scoreLog.total_cost != null) html += '<tr><th>Candidate Formulation + Scoring Cost</th><td style="font-family:var(--font-mono)">$' + Number(scoreLog.total_cost || 0).toFixed(4) + "</td></tr>";
  html += "</table>";

  if (!candidates.length) {
    html += '<div style="color:var(--text-muted)">No candidate experiments recorded.</div>';
    return html;
  }

  const sharedScorePrompt = candidates.map((c) => c && c.score_prompt).find((p) => p) || "";
  const sharedScoreResponse = candidates.map((c) => c && c.score_response).find((r) => r) || "";
  if (sharedScorePrompt || sharedScoreResponse) {
    html += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:6px">All candidates are scored in one unified LLM call against the unanswered question pool — the prompt/response below is shared across every candidate.</div>';
    html += promptResponseBlock("YES Coverage Scoring (shared)", sharedScorePrompt, sharedScoreResponse);
  }

  html += '<table class="data-table" style="margin-bottom:12px"><tr><th></th><th>Kind</th><th>Bank #</th><th>Top-k Rank</th><th>Score</th><th>YES Coverage</th><th>Cost</th><th>Question</th></tr>';
  candidates.forEach((cand, idx) => {
    const isWinner = idx === winnerIndex;
    const yesCount = Array.isArray(cand.per_question_yes_source_indices) ? cand.per_question_yes_source_indices.length : 0;
    const cost = Number(cand.formulation_cost || 0) + Number(cand.score_cost || 0);
    html += '<tr style="' + (isWinner ? "background:rgba(63,185,80,0.08)" : "") + '">' +
      '<td>' + (isWinner ? '<span class="verdict verdict-correct">winner</span>' : "") + '</td>' +
      '<td>' + esc(cand.kind || "") + '</td>' +
      '<td>' + esc(sourceIndexLabel(cand.source_index)) + '</td>' +
      '<td>' + (cand.topk_rank != null ? cand.topk_rank + 1 : "") + '</td>' +
      '<td style="font-family:var(--font-mono)">' + Number(cand.score || 0).toFixed(0) + '</td>' +
      '<td>' + yesCount + '</td>' +
      '<td style="font-family:var(--font-mono)">$' + cost.toFixed(4) + '</td>' +
      '<td style="max-width:420px">' + esc(cand.question || "") + "</td></tr>";
  });
  html += "</table>";

  candidates.forEach((cand, idx) => {
    const isWinner = idx === winnerIndex;
    let candHtml = "";
    candHtml += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">' +
      "kind: " + esc(cand.kind || "") +
      (cand.source_index != null ? " | bank " + esc(sourceIndexLabel(cand.source_index)) : "") +
      (cand.topk_rank != null ? " | top-k rank " + (cand.topk_rank + 1) : "") +
      " | score " + Number(cand.score || 0).toFixed(0) +
      " | cost $" + (Number(cand.formulation_cost || 0) + Number(cand.score_cost || 0)).toFixed(4) +
      "</div>";
    if (cand.question) {
      candHtml += '<div style="font-size:12px;padding:8px 10px;background:var(--bg);border:1px solid var(--border);border-radius:4px;margin-bottom:8px">' +
        '<strong style="color:var(--accent3)">Question:</strong> ' + esc(cand.question) + "</div>";
    }
    candHtml += '<div style="font-weight:600;font-size:12px;color:var(--text-muted);margin:12px 0 6px">Plan</div>';
    candHtml += '<pre style="max-height:none">' + esc(cand.plan || "") + "</pre>";
    candHtml += collapsible(
      "YES Coverage (" + (Array.isArray(cand.per_question_yes_source_indices) ? cand.per_question_yes_source_indices.length : 0) + ")",
      renderSourceQuestionCoverage(cand.per_question_yes_source_indices, qaBank),
      true,
    );
    candHtml += promptResponseBlock("Candidate Formulation", cand.formulation_prompt, cand.formulation_response);

    html += collapsible(
      (isWinner ? "Winning Candidate: " : "Candidate " + (idx + 1) + ": ") +
        esc(cand.kind || "candidate") +
        (cand.source_index != null ? " " + esc(sourceIndexLabel(cand.source_index)) : ""),
      candHtml,
      isWinner,
    );
  });

  return html;
}

function parseQuestionBlockFromPrompt(prompt) {
  if (!prompt) return [];
  const match = String(prompt).match(/=== CURRENT QUESTIONS ===\n([\s\S]*?)\n=== END CURRENT QUESTIONS ===/);
  if (!match) return [];
  return match[1]
    .trim()
    .split(/\n(?=Q\d+:\s)/)
    .map((chunk) => {
      const parsed = chunk.match(/^Q(\d+):\s*([\s\S]*?)(?:\s*->\s*(YES|NO|UNANSWERED|UNKNOWN)(?:\s*\(evidence:\s*([\s\S]*)\))?)?$/);
      if (!parsed) return null;
      let evidence = parsed[4] || "";
      if (evidence.endsWith(")")) evidence = evidence.slice(0, -1);
      let answer = null;
      if (parsed[3] === "YES") answer = true;
      else if (parsed[3] === "NO") answer = false;
      return {
        original_index: Number(parsed[1]) - 1,
        question: parsed[2].trim(),
        answer: answer,
        evidence: evidence,
        source_step: "",
      };
    })
    .filter(Boolean);
}

function filterDroppedQuestions(items, droppedIndices) {
  if (!Array.isArray(items) || !items.length) return [];
  const dropped = new Set(Array.isArray(droppedIndices) ? droppedIndices : []);
  return items.filter((item, idx) => !dropped.has(item.original_index != null ? item.original_index : idx));
}

function afterDedupQuestionsFromLog(items, dedupLog) {
  if (!Array.isArray(items) || !items.length) return [];
  const log = dedupLog || {};
  const dropped = new Set(Array.isArray(log.dropped_indices) ? log.dropped_indices : []);
  const replacementsByFirstIndex = new Map();
  const groups = Array.isArray(log.replacement_groups) ? log.replacement_groups : [];

  groups.forEach((group) => {
    const memberIndices = Array.isArray(group.member_indices)
      ? group.member_indices.filter((idx) => Number.isInteger(idx))
      : [];
    if (!memberIndices.length) return;
    replacementsByFirstIndex.set(Math.min(...memberIndices), {
      replacement_question: group.replacement_question || "",
      member_questions: Array.isArray(group.member_questions) ? group.member_questions : [],
      member_indices: memberIndices,
      answer_preserved: group.answer_preserved,
    });
  });

  return items
    .map((item, idx) => {
      const originalIndex = item.original_index != null ? item.original_index : idx;
      if (dropped.has(originalIndex)) return null;
      const replacement = replacementsByFirstIndex.get(originalIndex);
      if (!replacement) return item;
      const replacementQuestion = replacement.replacement_question || item.question || "";
      return {
        ...item,
        question: replacementQuestion,
        retained_original_question: item.question || "",
        dedup_member_questions: replacement.member_questions,
        dedup_member_indices: replacement.member_indices,
        dedup_answer_preserved: replacement.answer_preserved,
      };
    })
    .filter(Boolean);
}

function questionItemsFromIndices(items, indices) {
  if (!Array.isArray(items) || !items.length || !Array.isArray(indices)) return [];
  return indices
    .map((idx) => {
      if (!Number.isInteger(idx) || idx < 0 || idx >= items.length) return null;
      return items[idx];
    })
    .filter(Boolean);
}

function answerLabelForQuestion(item) {
  if (!item || item.answer === null || item.answer === undefined) return "UNANSWERED";
  if (item.answer === true) return "YES";
  if (item.answer === false) return "NO";
  return String(item.answer || "UNANSWERED");
}

function topKQuestionLines(items, indices) {
  if (!Array.isArray(items) || !Array.isArray(indices)) return "";
  return indices
    .map((idx) => {
      if (!Number.isInteger(idx)) return null;
      const item = items[idx];
      if (!item) return null;
      return "Q" + (idx + 1) + ": " + (item.question || "") + " -> " + answerLabelForQuestion(item);
    })
    .filter(Boolean)
    .join("\n");
}

function renderTopKPromptResponse(selectionLog, candidateQuestions) {
  const log = selectionLog || {};
  if (log.prompt || log.response) {
    return promptResponseBlock("Top-k Selection", log.prompt, log.response);
  }

  const candidateIndices = Array.isArray(log.candidate_source_indices) ? log.candidate_source_indices : [];
  const selectedIndices = Array.isArray(log.selected_source_indices) ? log.selected_source_indices : [];
  const candidateText = topKQuestionLines(candidateQuestions, candidateIndices);
  const selectedText = selectedIndices.length
    ? selectedIndices.map((idx) => '<q n="Q' + (idx + 1) + '" />').join("\n")
    : "NONE";

  const prompt = [
    "No LLM prompt was sent for this top-k selection.",
    "",
    "Reason: " + (log.note || "selection was resolved without an LLM call"),
    "Unanswered questions: " + (log.pre_selection_unanswered != null ? log.pre_selection_unanswered : "?"),
    "Unanswered cap: " + (log.max_unanswered_qa_pairs != null ? log.max_unanswered_qa_pairs : "?"),
    "",
    "Equivalent candidate input:",
    "=== AVAILABLE QUESTIONS ===",
    candidateText || "(none)",
    "=== END AVAILABLE QUESTIONS ===",
  ].join("\n");

  const response = [
    "Deterministic top-k shortcut response:",
    "<selected_questions>",
    selectedText,
    "</selected_questions>",
  ].join("\n");

  return promptResponseBlock("Top-k Selection", prompt, response);
}

function renderSimpleQuestionList(questions) {
  if (!Array.isArray(questions) || questions.length === 0) {
    return '<div style="color:var(--text-muted)">No questions recorded.</div>';
  }
  let html = '<ol style="margin:0 0 0 20px">';
  questions.forEach((question) => {
    html += '<li style="margin-bottom:4px;font-size:13px">' + esc(question) + "</li>";
  });
  html += "</ol>";
  return html;
}

function renderExperimentQuestionTable(items, options) {
  const opts = options || {};
  if (!Array.isArray(items) || items.length === 0) {
    return '<div style="color:var(--text-muted)">No questions recorded.</div>';
  }
  let html = '<table class="data-table"><tr><th>#</th>';
  if (opts.showOriginalIndex) html += "<th>Original #</th>";
  html += "<th>Question</th><th>Answer</th><th>Evidence</th><th>Src Step</th></tr>";
  items.forEach((item, idx) => {
    let answer = "UNANSWERED";
    let verdictClass = "verdict-unanswered";
    if (item.answer === true) {
      answer = "YES";
      verdictClass = "verdict-correct";
    } else if (item.answer === false) {
      answer = "NO";
      verdictClass = "verdict-incorrect";
    } else if (typeof item.answer === "string" && item.answer) {
      answer = item.answer;
    }
    html += '<tr><td style="color:var(--text-muted)">Q' + (idx + 1) + "</td>";
    if (opts.showOriginalIndex) {
      html += '<td style="color:var(--text-muted)">' + (item.original_index != null ? "Q" + (item.original_index + 1) : "") + "</td>";
    }
    let questionHtml = '<div>' + esc(item.question || "") + "</div>";
    if (item.retained_original_question && item.retained_original_question !== item.question) {
      questionHtml += '<div style="margin-top:4px;font-size:11px;color:var(--text-muted)">Retained original: ' +
        esc(item.retained_original_question) + "</div>";
    }
    if (Array.isArray(item.dedup_member_questions) && item.dedup_member_questions.length) {
      const members = item.dedup_member_questions.map((question, memberIdx) => {
        const originalIdx = Array.isArray(item.dedup_member_indices) ? item.dedup_member_indices[memberIdx] : null;
        const label = originalIdx != null ? "Q" + (originalIdx + 1) + ": " : "";
        return esc(label + question);
      }).join("<br>");
      questionHtml += '<details style="margin-top:4px;font-size:11px;color:var(--text-muted)"><summary>Replacement group (' +
        item.dedup_member_questions.length + ")</summary><div style=\"margin-top:4px;line-height:1.4\">" + members + "</div></details>";
    }
    html += '<td style="max-width:420px">' + questionHtml + "</td>" +
      '<td><span class="verdict ' + verdictClass + '">' + esc(answer) + "</span></td>" +
      '<td style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(item.evidence || "") + '">' + esc(item.evidence || "") + "</td>" +
      "<td>" + esc(item.source_step) + "</td></tr>";
  });
  html += "</table>";
  return html;
}

function renderSelectionQuestionTable(items) {
  if (!Array.isArray(items) || items.length === 0) {
    return '<div style="color:var(--text-muted)">No selected questions recorded.</div>';
  }
  const hasSourceIndex = items.some((item) => item && typeof item === "object" && item.source_index != null);
  let html = '<table class="data-table"><tr><th>#</th>';
  if (hasSourceIndex) html += "<th>Bank #</th>";
  html += "<th>Question</th><th>Src Step</th></tr>";
  items.forEach((item, idx) => {
    html += '<tr><td>Q' + (idx + 1) + "</td>";
    if (hasSourceIndex) {
      html += '<td style="color:var(--text-muted)">' + (item.source_index != null ? "Q" + (item.source_index + 1) : "") + "</td>";
    }
    html += '<td style="max-width:520px">' + esc(item.question || item) +
      '</td><td>' + esc(item.source_step) + "</td></tr>";
  });
  html += "</table>";
  return html;
}

function primaryOnlineScoringArtifact(data) {
  const artifacts = data.question_scoring || {};
  return artifacts.online_full || artifacts.online_light || artifacts.online_theory_entropy || null;
}

function renderScoringPipelineSection(artifact, scoringLog) {
  const scoring = scoringLog || (artifact && artifact.scoring_log) || {};
  let html = "";

  const perQuestion = Array.isArray(scoring.per_question) ? scoring.per_question : [];
  let perQuestionHtml = "";
  perQuestion.forEach((questionLog, index) => {
    perQuestionHtml += renderScoringQuestionLog(questionLog, index);
  });
  html += '<div style="font-weight:600;font-size:12px;color:var(--text-muted);margin:14px 0 6px">Per-Question Artifacts (' + perQuestion.length + ")</div>";
  html += perQuestionHtml || '<div style="color:var(--text-muted);margin-bottom:12px">No per-question scoring artifacts recorded.</div>';

  html += '<div style="font-weight:600;font-size:12px;color:var(--text-muted);margin:14px 0 6px">Ranked Questions with Scores</div>';
  html += renderScoringRankedTable(
    scoring.ranked_unanswered || (artifact && artifact.ranked_unanswered) || [],
    scoring.selected_probe_questions || (artifact && artifact.kept_unanswered_questions),
    artifact && artifact.dropped_unanswered_questions,
  );
  const tieBreakHtml = renderTieBreakSection(scoring.tie_break || (artifact && artifact.tie_break));
  if (tieBreakHtml) {
    html += '<div style="font-weight:600;font-size:12px;color:var(--text-muted);margin:14px 0 6px">Top-Score Tie-break</div>';
    html += tieBreakHtml;
  }
  return html;
}

function renderCriticalDecision(data, step) {
  const log = data.critical_id_log || {};
  const hasLog = !!(step && step.did_critical_id) || !!(log.prompt || log.response);
  if (!hasLog) return "";
  const isCritical = step && step.critical === true;
  const statusColor = isCritical ? "var(--accent2)" : "var(--text-muted)";
  const statusBg = isCritical ? "rgba(63,185,80,0.12)" : "rgba(139,148,158,0.12)";
  const reason = log.reason || (step && step.critical_reason) || "";
  const cost = step && step.critical_cost != null ? step.critical_cost : log.cost_usd;
  let html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:12px">';
  html += '<div class="stat-card"><div class="stat-label">Decision</div><div class="stat-value" style="font-size:18px;color:' + statusColor + '">' + (isCritical ? "Critical" : "Non-critical") + "</div></div>";
  html += '<div class="stat-card"><div class="stat-label">Gate</div><div class="stat-value" style="font-size:18px;color:' + statusColor + '">' + (isCritical ? "Learn" : "Skip learn") + "</div></div>";
  if (cost != null) {
    html += '<div class="stat-card"><div class="stat-label">Critical ID Cost</div><div class="stat-value yellow" style="font-size:18px">$' + Number(cost || 0).toFixed(4) + "</div></div>";
  }
  html += "</div>";
  if (reason) {
    html += '<div style="font-size:12px;padding:8px 10px;background:' + statusBg + ';border:1px solid var(--border);border-radius:4px;margin-bottom:10px">' +
      '<strong style="color:' + statusColor + '">Reason:</strong> ' + esc(reason) + "</div>";
  }
  html += promptResponseBlock("Critical Identification", log.prompt, log.response, {
    imagePaths: log.prompt_image_paths,
    labelPrefix: "Image",
    accentCurrent: true,
    currentStep: step,
  });
  return '<div class="card" style="margin-bottom:16px;border-left:3px solid ' + statusColor + '">' +
    '<div class="card-header" onclick="toggleCard(this)">Critical Transition Gate <span style="font-size:11px;color:' + statusColor + ';font-weight:700">' + (isCritical ? "critical" : "non-critical") + '</span> <span class="toggle">&#9660;</span></div>' +
    '<div class="card-body">' + html + "</div></div>";
}

function renderOverview(data, step) {
  const c = document.getElementById("overview-container");
  if (!c) return;
  let html = "";

  // Observation images
  const imgHtml = obsImageHtml(step.episode_idx, step.step, data, step);
  if (imgHtml) {
    html += collapsible("Observation Images", imgHtml, true);
  }

  // Agent LLM Call section with bigger boxes — show only Current Observation from prompt
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
          messageAttachmentsHtml(lastUser, step) +
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

  html += renderCriticalDecision(data, step);

  // Active Experiment with selected questions
  if (step.active_experiment) {
    const genLabel = step.did_formulate_experiment ? "formulated at start of this step" : "carried over from previous step";
    let expContent = "";
    // Show selected question if available
    const expLog = data.experiment_log || {};
    const selectedQIdx = expLog.selected_question_index;
    if (selectedQIdx != null) {
      const qa = expLog.qa_pairs_for_experiment || expLog.qa_pairs_at_formulation || data.qa_pairs || [];
      const selectedQ = selectedQIdx < qa.length ? qa[selectedQIdx] : null;
      const sourceIdx = expLog.selected_question_source_index;
      const sourceLabel = sourceIdx != null ? ' <span style="color:var(--text-muted)">(bank Q' + (sourceIdx + 1) + ")</span>" : "";
      if (selectedQ) {
        expContent += '<div style="font-size:12px;margin-bottom:8px;padding:8px 10px;background:var(--bg);border:1px solid var(--accent3);border-radius:4px">' +
          '<span style="color:var(--accent3);font-weight:600">Selected Question (prompt Q' + (selectedQIdx + 1) + '):</span>' + sourceLabel + " " + esc(selectedQ.question) + "</div>";
      }
    } else if (expLog.selected_question || expLog.active_experiment_question) {
      const sourceIdx = expLog.selected_question_source_index;
      const sourceLabel = sourceIdx != null ? ' <span style="color:var(--text-muted)">(bank Q' + (sourceIdx + 1) + ")</span>" : "";
      expContent += '<div style="font-size:12px;margin-bottom:8px;padding:8px 10px;background:var(--bg);border:1px solid var(--accent3);border-radius:4px">' +
        '<span style="color:var(--accent3);font-weight:600">Experiment Question:</span>' + sourceLabel + " " + esc(expLog.selected_question || expLog.active_experiment_question) + "</div>";
    }
    expContent += '<pre style="max-height:none">' + esc(step.active_experiment) + "</pre>";
    html += '<div class="card" style="margin-bottom:16px;border-left:3px solid var(--accent2)">' +
      '<div class="card-header" onclick="toggleCard(this)">Active Experiment <span style="font-size:11px;color:var(--text-muted);font-weight:400">' + genLabel + '</span> <span class="toggle">&#9660;</span></div>' +
      '<div class="card-body">' + expContent + "</div></div>";
  }

  const ebRun = isEBRun();
  if (ebRun) {
    html += '<div style="margin-bottom:16px"><h3 style="font-size:12px;color:var(--text-muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">Beliefs</h3><pre>' + esc(data.beliefs || "(empty)") + "</pre></div>";
    html += '<div style="margin-bottom:16px"><h3 style="font-size:12px;color:var(--text-muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">Perception</h3><pre>' + esc(data.perception || "(empty)") + "</pre></div>";
  }

  // Cost section — below perception
  let costHtml = '<table class="data-table"><tr><th>Category</th><th>This Step</th><th>Cumulative</th></tr>';
  let cumAgent = 0, cumExtract = 0, cumImprove = 0, cumExperiment = 0, cumCritical = 0, cumTotal = 0;
  for (let i = 0; i <= selectedStepIdx; i++) {
    const s = DATA.steps[i];
    cumAgent += s.agent_step_cost || 0;
    cumExtract += s.extract_cost || 0;
    cumImprove += s.improve_cost || 0;
    cumExperiment += s.experiment_cost || 0;
    cumCritical += s.critical_cost || 0;
    cumTotal += s.step_total_cost || 0;
  }
  const agentLabel = ebRun ? "Agent" : "LLM Call";
  costHtml += '<tr><td>' + agentLabel + '</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + Number(step.agent_step_cost).toFixed(4) + '</td><td style="font-family:var(--font-mono);color:var(--text-muted)">$' + cumAgent.toFixed(4) + "</td></tr>";
  if (ebRun) {
    if (step.did_critical_id || cumCritical > 0) {
      costHtml += '<tr><td>Critical ID</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + Number(step.critical_cost || 0).toFixed(4) + '</td><td style="font-family:var(--font-mono);color:var(--text-muted)">$' + cumCritical.toFixed(4) + "</td></tr>";
    }
    costHtml += '<tr><td>Extraction</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + Number(step.extract_cost).toFixed(4) + '</td><td style="font-family:var(--font-mono);color:var(--text-muted)">$' + cumExtract.toFixed(4) + "</td></tr>";
    costHtml += '<tr><td>Improve</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + Number(step.improve_cost).toFixed(4) + '</td><td style="font-family:var(--font-mono);color:var(--text-muted)">$' + cumImprove.toFixed(4) + "</td></tr>";
    costHtml += '<tr><td>Experiment</td><td style="font-family:var(--font-mono);color:var(--accent3)">$' + Number(step.experiment_cost).toFixed(4) + '</td><td style="font-family:var(--font-mono);color:var(--text-muted)">$' + cumExperiment.toFixed(4) + "</td></tr>";
  }
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
    const color = isSelected ? "var(--accent)" : (step.improve_cost > 0 ? "var(--purple)" : (step.critical === false ? "var(--border)" : (step.did_critical_id ? "var(--accent2)" : "var(--surface2)")));
    const title = "g" + step.global_step + ": $" + step.step_total_cost.toFixed(4) +
      (step.did_critical_id ? " | " + (step.critical ? "critical" : "non-critical") : "");
    html += '<div style="flex:1;height:' + height + '%;background:' + color + ';border-radius:2px 2px 0 0;cursor:pointer;min-width:2px" title="' + esc(title) + '" onclick="showStep(' + i + ')"></div>';
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
    const ordered = orderedQaForDisplay(data, qa);
    const answered = ordered.items.filter((item) => item.answer !== null);
    const unanswered = ordered.items.filter((item) => item.answer === null);
    let qaHtml = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">' +
      answered.length + " answered, " + unanswered.length + " unanswered";
    if (ordered.sourceIndices) qaHtml += " | ordered by experiment selection";
    qaHtml += "</div>";
    qaHtml += renderQuestionSnapshotTable(ordered.items, { sourceIndices: ordered.sourceIndices });
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
  const qSelectLog = data.question_selection_log || data.trim_log || {};
  const dedupLog = qSelectLog.dedup || {};
  const selectionLog = qSelectLog.selection || {};
  const scoringArtifact = primaryOnlineScoringArtifact(data);
  const scoringLog = qSelectLog.scoring || (scoringArtifact && scoringArtifact.scoring_log) || {};
  const expScoringLog = experimentScoringLog(data);
  const selectionMode = expLog.experiment_selection_mode || qSelectLog.experiment_selection_mode || (expScoringLog && expScoringLog.mode) || "single";
  const stepMeta = DATA.steps[selectedStepIdx] || {};
  const didGenQ = stepMeta.did_gen_questions;
  const didFormulate = stepMeta.did_formulate_experiment;
  const beforeDedupQuestions = parseQuestionBlockFromPrompt(dedupLog.prompt);
  const afterDedupQuestions = afterDedupQuestionsFromLog(beforeDedupQuestions, dedupLog);
  const scoringCandidateIndices = scoringLog.candidate_indices || scoringLog.projection_unanswered_indices;
  let selectedForScoring = [];
  if (Array.isArray(qSelectLog.experiment_questions) && qSelectLog.experiment_questions.length) {
    selectedForScoring = qSelectLog.experiment_questions;
  } else if (Array.isArray(selectionLog.selected_questions) && selectionLog.selected_questions.length) {
    selectedForScoring = selectionLog.selected_questions;
  } else {
    selectedForScoring = questionItemsFromIndices(afterDedupQuestions, scoringCandidateIndices);
  }
  if (!selectedForScoring.length && Array.isArray(scoringLog.per_question)) {
    selectedForScoring = scoringLog.per_question.map((item) => ({
      question: item.question,
      source_step: item.source_step,
    }));
  }

  html += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">' +
    (didGenQ
      ? 'Questions were <strong style="color:var(--accent3)">generated at the start of this step</strong>. ' +
        (selectionMode === "score_topk"
          ? ('Mode: <strong style="color:var(--accent3)">score_topk</strong>. Candidate experiments were formulated for the selected top-k questions, scored against the unanswered bank, and the ' +
            ((expScoringLog && expScoringLog.winner_kind === "active")
              ? "active experiment won, so it was <strong>kept</strong>."
              : (didFormulate ? 'best fresh candidate became the <strong style="color:var(--accent2)">active experiment</strong>.' : "current experiment was <strong>kept</strong>.")))
          : (didFormulate
            ? 'A new <strong style="color:var(--accent2)">experiment was formulated</strong> from an unanswered question.'
            : "The current experiment was <strong>kept</strong> (LLM returned null)."))
      : "No question generation this step — the agent used the experiment carried over from the previous cycle.") +
    "</div>";

  if (expLog.question_gen_prompt || expLog.question_gen_response || (expLog.new_questions && expLog.new_questions.length)) {
    let qGenHtml = "";
    qGenHtml += promptResponseBlock("Question Generation", expLog.question_gen_prompt, expLog.question_gen_response, {
      currentStep: stepMeta,
      stepMeta: stepMeta,
      imagePaths: expLog.question_gen_image_paths,
      accentCurrent: true,
    });
    qGenHtml += '<div style="font-weight:600;font-size:12px;color:var(--text-muted);margin:12px 0 6px">New Questions Generated (' +
      ((expLog.new_questions && expLog.new_questions.length) || 0) + ")</div>";
    qGenHtml += renderSimpleQuestionList(expLog.new_questions || []);
    html += collapsible("1. Question Generation", qGenHtml, true);
  }

  if (beforeDedupQuestions.length || dedupLog.prompt || dedupLog.response || afterDedupQuestions.length) {
    let dedupHtml = "";
    dedupHtml += '<div style="font-weight:600;font-size:12px;color:var(--text-muted);margin-bottom:6px">All Questions Before Dedup (' +
      (dedupLog.pre_dedup_count != null ? dedupLog.pre_dedup_count : beforeDedupQuestions.length) + ")</div>";
    dedupHtml += renderExperimentQuestionTable(beforeDedupQuestions, { showOriginalIndex: false });
    dedupHtml += '<div style="font-size:12px;color:var(--text-muted);margin:10px 0">Dropped duplicates: ' + (dedupLog.dropped_count || 0) + "</div>";
    dedupHtml += promptResponseBlock("Dedup", dedupLog.prompt, dedupLog.response);
    dedupHtml += '<div style="font-weight:600;font-size:12px;color:var(--text-muted);margin:14px 0 6px">Questions After Dedup (' +
      (dedupLog.post_dedup_count != null ? dedupLog.post_dedup_count : afterDedupQuestions.length) + ")</div>";
    dedupHtml += renderExperimentQuestionTable(afterDedupQuestions, { showOriginalIndex: true });
    html += collapsible("2. Deduplication", dedupHtml, true);
  }

  if (selectionLog.prompt || selectionLog.response || selectedForScoring.length) {
    let topKHtml = "";
    topKHtml += renderTopKPromptResponse(selectionLog, afterDedupQuestions);
    if (selectionLog.note) {
      topKHtml += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">' + esc(selectionLog.note) + "</div>";
    }
    topKHtml += '<div style="font-weight:600;font-size:12px;color:var(--text-muted);margin:12px 0 6px">' +
      (selectionMode === "score_topk" ? "Selected Questions for Candidate Experiments" : "Selected Questions for Scoring") + " (" +
      selectedForScoring.length + ")</div>";
    topKHtml += renderSelectionQuestionTable(selectedForScoring);
    html += collapsible("3. Top-k Selection", topKHtml, true);
  }

  if (scoringArtifact || scoringLog.method || Array.isArray(scoringLog.per_question)) {
    html += collapsible("4. Question Scoring", renderScoringPipelineSection(scoringArtifact, scoringLog), true);
  }

  const tieBreakHtml = renderTieBreakSection(scoringLog.tie_break || (scoringArtifact && scoringArtifact.tie_break));
  if (tieBreakHtml) {
    html += collapsible("5. Top-Score Tie-break", tieBreakHtml, true);
  }

  const candidateScoringHtml = renderExperimentCandidateScoring(data);
  if (candidateScoringHtml) {
    html += collapsible("6. Candidate Experiment Scoring", candidateScoringHtml, true);
  }

  const qaBank = expLog.qa_pairs_at_formulation || data.qa_pairs || [];
  const qaForExperiment = expLog.qa_pairs_for_experiment || qSelectLog.experiment_questions || qaBank;

  if (expLog.target_question || expLog.selected_question || expLog.selected_question_index != null || expLog.selected_question_source_index != null) {
    let selectedHtml = "";
    const qIdx = expLog.selected_question_index;
    let selectedQ = null;
    if (qIdx != null && qIdx < qaForExperiment.length) selectedQ = qaForExperiment[qIdx];
    if (!selectedQ && expLog.selected_question) selectedQ = { question: expLog.selected_question };
    if (!selectedQ && expLog.target_question) selectedQ = { question: expLog.target_question };
    if (selectedQ) {
      const sourceIdx = expLog.selected_question_source_index != null ? expLog.selected_question_source_index : expLog.target_question_source_index;
      const sourceLabel = sourceIdx != null ? ' <span style="color:var(--text-muted)">(bank Q' + (sourceIdx + 1) + ")</span>" : "";
      const promptLabel = qIdx != null ? "prompt Q" + (qIdx + 1) : "target";
      selectedHtml += '<div style="font-size:13px;padding:10px 12px;background:var(--bg);border:1px solid var(--accent3);border-radius:4px">' +
        '<span style="color:var(--accent3);font-weight:600">Selected Question (' + promptLabel + '):</span>' +
        sourceLabel + " " + esc(selectedQ.question || "") + "</div>";
    }
    html += collapsible((selectionMode === "score_topk" ? "7. Winning Experiment Question" : "6. Selected Question"), selectedHtml || '<div style="color:var(--text-muted)">No selected question recorded.</div>', true);
  }

  if (selectionMode !== "score_topk" && (expLog.experiment_prompt || expLog.experiment_response)) {
    const expHtml = promptResponseBlock("Experiment Formulation", expLog.experiment_prompt, expLog.experiment_response, {
      currentStep: stepMeta,
      stepMeta: stepMeta,
      imagePaths: expLog.experiment_image_paths,
      accentCurrent: true,
    });
    html += collapsible("7. Experiment Formulation", expHtml, true);
  }

  if (expLog.experiment_plan || expLog.experiment_response) {
    let resultHtml = "";
    if (expLog.experiment_plan) {
      const resultLabel = selectionMode === "score_topk" && expScoringLog && expScoringLog.winner_kind === "active"
        ? "Kept Active Experiment"
        : "Formulated Experiment";
      resultHtml += '<div style="font-size:12px;padding:10px 14px;background:var(--bg);border:1px solid var(--accent2);border-radius:6px">' +
        '<strong style="color:var(--accent2)">' + resultLabel + ':</strong> ' + esc(expLog.experiment_plan) + "</div>";
    } else {
      resultHtml += '<div style="color:var(--text-muted);font-size:12px;padding:8px 12px;background:var(--surface);border-radius:4px">LLM chose to keep the current experiment (returned null).</div>';
    }
    html += collapsible(selectionMode === "score_topk" ? "8. Experiment Result" : "8. Formulated Experiment", resultHtml, true);
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

// Render one YES/NO/UNKNOWN prediction cell from a p_yes probability:
// 1.0 -> YES (green), 0.0 -> NO (red), anything else (0.5) -> "?" (muted).
function theoryPredCell(p) {
  if (p == null) return '<td style="text-align:center;color:var(--text-muted)">·</td>';
  if (p >= 0.999) return '<td style="text-align:center;background:rgba(63,185,80,0.18);color:var(--accent2);font-weight:600">Y</td>';
  if (p <= 0.001) return '<td style="text-align:center;background:rgba(229,85,85,0.18);color:#e55;font-weight:600">N</td>';
  return '<td style="text-align:center;color:var(--text-muted)">?</td>';
}

function renderTheories(data) {
  const c = document.getElementById("theories-container");
  if (!c) return;
  const tlog = data.theory_log || {};
  const gen = tlog.theories || {};
  const cruxGen = tlog.crux_questions || {};
  const theories = Array.isArray(gen.theories) ? gen.theories : [];
  const cruxQuestions = Array.isArray(cruxGen.questions) ? cruxGen.questions : [];
  const scoring = (data.question_scoring || {}).online_theory_entropy || null;
  const scoreLog = (scoring && scoring.scoring_log) || {};
  const perQuestion = Array.isArray(scoreLog.per_question) ? scoreLog.per_question : [];
  const weights = Array.isArray(scoreLog.theory_weights) ? scoreLog.theory_weights : theories.map((t) => t.weight);

  if (!theories.length && !cruxQuestions.length && !perQuestion.length) {
    c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No theory data recorded for this step.</div>';
    return;
  }

  const cruxSet = new Set(cruxQuestions);
  const selectedQ = scoreLog.target_experiment_question;
  const maxWeight = theories.reduce((m, t) => Math.max(m, t.weight || 0), 0) || 1;

  let html = "";

  // Summary banner.
  html += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">' +
    '<strong style="color:var(--accent3)">Theory-entropy selection (Plan B).</strong> ' +
    theories.length + ' theories generated (rank-weighted, decay=' + (gen.decay != null ? gen.decay : "?") + '), ' +
    cruxQuestions.length + ' crux questions seeded. ' +
    'Questions are scored by mutual information I(answer; theory) — high when the theories disagree on the answer.' +
    (selectedQ ? ' Selected target: <span style="color:var(--accent3)">' + esc(selectedQ) + '</span>' : '') +
    "</div>";

  // 1. Theory ensemble — ranked cards with weight bars.
  if (theories.length) {
    let tHtml = "";
    theories.forEach((t, i) => {
      const w = t.weight || 0;
      const barPct = Math.round((w / maxWeight) * 100);
      tHtml += '<div style="border:1px solid var(--border);border-radius:5px;margin-bottom:10px;overflow:hidden">';
      tHtml += '<div style="display:flex;align-items:center;gap:10px;padding:6px 10px;background:var(--surface2);border-bottom:1px solid var(--border)">' +
        '<span style="font-weight:700;color:var(--accent3)">T' + (i + 1) + '</span>' +
        '<span style="font-size:11px;color:var(--text-muted)">rank ' + (t.rank != null ? t.rank : i + 1) +
        (t.likelihood ? ' &middot; ' + esc(t.likelihood) : "") + '</span>' +
        '<div style="flex:1;height:10px;background:var(--bg);border-radius:5px;overflow:hidden;max-width:240px;margin-left:auto">' +
        '<div style="height:100%;width:' + barPct + '%;background:var(--accent3)"></div></div>' +
        '<span style="font-family:var(--font-mono);font-size:12px;font-weight:600">w=' + w.toFixed(3) + '</span>' +
        "</div>";
      tHtml += '<div style="padding:8px 10px"><pre style="margin:0;white-space:pre-wrap;font-size:12px">' +
        esc(t.world_knowledge || "") + "</pre>";
      if (t.rationale) {
        tHtml += '<div style="margin-top:6px;font-size:12px;color:var(--text-muted)"><em>Rationale:</em> ' + esc(t.rationale) + "</div>";
      }
      tHtml += "</div></div>";
    });
    html += collapsible("1. Theory Ensemble (" + theories.length + ")", tHtml, true);
  }

  // 2. Crux questions seeded from the theory disagreements.
  if (cruxQuestions.length) {
    let cqHtml = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:6px">Questions designed to split the ensemble; added to the unanswered bank for scoring.</div>';
    cqHtml += '<table class="data-table"><tr><th>#</th><th>Crux Question</th></tr>';
    cruxQuestions.forEach((q, i) => {
      cqHtml += '<tr><td>' + (i + 1) + '</td><td>' + esc(q) + "</td></tr>";
    });
    cqHtml += "</table>";
    cqHtml += promptResponseBlock("Crux Generation", cruxGen.prompt, cruxGen.response);
    html += collapsible("2. Seeded Crux Questions (" + cruxQuestions.length + ")", cqHtml, true);
  }

  // 3. MI ranking + per-theory prediction matrix (the "why" behind selection).
  if (perQuestion.length) {
    const ranked = perQuestion.slice().sort((a, b) => (b.score || 0) - (a.score || 0));
    const nTheories = weights.length || (ranked[0] && ranked[0].p_yes_per_theory ? ranked[0].p_yes_per_theory.length : 0);
    let mHtml = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:6px">' +
      'Each row is a candidate question; columns T1..T' + nTheories + ' are each theory\'s predicted answer ' +
      '(<span style="color:var(--accent2);font-weight:600">Y</span>=yes, <span style="color:#e55;font-weight:600">N</span>=no, ?=unknown). ' +
      'MI is highest when weighted predictions are most split.</div>';
    mHtml += '<table class="data-table"><tr><th>Rank</th><th>MI (bits)</th>';
    for (let k = 0; k < nTheories; k++) {
      mHtml += '<th title="weight ' + (weights[k] != null ? weights[k].toFixed(3) : "?") + '">T' + (k + 1) + "</th>";
    }
    mHtml += "<th>Question</th></tr>";
    ranked.forEach((row, idx) => {
      const isCrux = cruxSet.has(row.question);
      const isSel = selectedQ && row.question === selectedQ;
      const rowStyle = isSel ? ' style="background:rgba(210,153,34,0.12)"' : "";
      mHtml += "<tr" + rowStyle + "><td>" + (idx + 1) + '</td><td style="font-family:var(--font-mono)">' +
        Number(row.score || 0).toFixed(4) + "</td>";
      const preds = row.p_yes_per_theory || [];
      for (let k = 0; k < nTheories; k++) mHtml += theoryPredCell(preds[k]);
      mHtml += '<td style="max-width:420px">' + esc(row.question) +
        (isCrux ? ' <span style="font-size:10px;color:var(--accent3);font-weight:700">[CRUX]</span>' : "") +
        (isSel ? ' <span style="font-size:10px;color:var(--accent3);font-weight:700">&#9733; SELECTED</span>' : "") +
        "</td></tr>";
    });
    mHtml += "</table>";
    if (scoring && scoring.tie_break) {
      const tb = renderTieBreakSection(scoring.tie_break);
      if (tb) mHtml += '<div style="margin-top:10px">' + tb + "</div>";
    }
    html += collapsible("3. Theory-Entropy Scoring (MI ranking)", mHtml, true);
  }

  // 4. Raw generation prompt/response for auditing.
  if (gen.prompt || gen.response) {
    html += collapsible("4. Theory Generation Prompt & Response",
      promptResponseBlock("Theory Generation", gen.prompt, gen.response), false);
  }

  c.innerHTML = html;
}

function renderFrontier(data) {
  const c = document.getElementById("frontier-container");
  if (!c) return;
  const fr = data.frontier || {};
  const candidates = Array.isArray(fr.frontier) ? fr.frontier : [];
  if (!candidates.length) {
    c.innerHTML = '<div style="color:var(--text-muted);padding:20px">No frontier recorded for this step (only populated under <code>question_scoring_method=gepa_frontier</code> on relearn steps).</div>';
    return;
  }

  const metric = fr.metric;
  const relearn = fr.relearn;
  const relearnObj = (relearn && typeof relearn === "object") ? relearn : null;
  const didRelearn = relearnObj != null || relearn === true;
  let html = "";

  // Summary banner.
  html += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">' +
    '<strong style="color:var(--accent3)">GEPA/legacy frontier.</strong> ' +
    candidates.length + ' competing {perception, world_knowledge} candidate(s) learned from the trajectory via an inverse-dynamics objective. ' +
    'Relearned this step: <strong>' + (didRelearn ? "yes" : "no") + "</strong>. " +
    (metric != null ? 'Best held-out metric (id_acc): <strong style="color:var(--accent2)">' + Number(metric).toFixed(3) + "</strong>." : "") +
    "</div>";

  // Relearn metadata table (gepa_frontier writes a rich relearn object).
  if (relearnObj) {
    const order = ["learner", "click_aware", "transitions", "n_verbs", "train", "val", "test", "candidates", "frontier_size", "metric_calls", "cost"];
    const keys = order.filter((k) => relearnObj[k] != null).concat(
      Object.keys(relearnObj).filter((k) => order.indexOf(k) < 0 && k !== "pool")
    );
    let rows = "";
    keys.forEach((k) => {
      let v = relearnObj[k];
      if (k === "cost" && typeof v === "number") v = "$" + Number(v).toFixed(4);
      else if (typeof v === "boolean") v = v ? "yes" : "no";
      else if (typeof v === "object") v = JSON.stringify(v);
      rows += '<tr><td style="padding:3px 10px;color:var(--text-muted);border-bottom:1px solid var(--border)">' + esc(k) + '</td><td style="padding:3px 10px;border-bottom:1px solid var(--border)"><b>' + esc(v) + "</b></td></tr>";
    });
    html += '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px">' +
      '<table style="font-size:12px;border:1px solid var(--border);border-radius:4px;border-collapse:collapse;background:var(--surface)">' + rows + "</table>";
    if (Array.isArray(relearnObj.pool) && relearnObj.pool.length) {
      html += '<div style="flex:1;min-width:240px"><div style="font-size:12px;color:var(--text-muted);margin-bottom:4px">action pool (inverse-dynamics targets)</div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:4px">' +
        relearnObj.pool.map((a) => '<span style="font-size:11px;padding:2px 6px;border-radius:4px;background:var(--surface2);border:1px solid var(--border)">' + esc(a) + "</span>").join("") +
        "</div></div>";
    }
    html += "</div>";
  }

  // Best metric across candidates that carry their own val_acc.
  const accs = candidates.map((cand) => cand.val_acc).filter((v) => v != null);
  const bestAcc = accs.length ? Math.max.apply(null, accs) : null;

  candidates.forEach((cand, i) => {
    const perception = cand.perception || "";
    const worldKnowledge = cand.world_knowledge || "";
    const valAcc = cand.val_acc;
    const isBest = valAcc != null && bestAcc != null && valAcc === bestAcc;
    const accPill = valAcc != null
      ? ' <span style="font-size:11px;padding:2px 8px;border-radius:10px;background:' + (isBest ? "rgba(63,185,80,0.18)" : "rgba(139,148,158,0.14)") + ';color:' + (isBest ? "var(--accent2)" : "var(--text-muted)") + ';font-weight:700">val_acc ' + Number(valAcc).toFixed(3) + (isBest ? " (best)" : "") + "</span>"
      : "";
    const emptyP = !perception.trim();
    const emptyB = !worldKnowledge.trim();

    html += '<div style="border:1px solid var(--border);border-radius:6px;margin-bottom:14px;background:var(--surface)">' +
      '<div style="padding:8px 12px;border-bottom:1px solid var(--border);font-weight:600">Candidate ' + i + accPill + "</div>" +
      '<div style="padding:12px">';
    // world_knowledge (beliefs)
    html += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:4px">world_knowledge (B)' + (emptyB ? ' — <em>empty (seed)</em>' : "") + "</div>";
    if (!emptyB) {
      html += '<pre style="white-space:pre-wrap;background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:10px;margin:0 0 12px;font-size:12px;max-height:260px;overflow:auto">' + esc(worldKnowledge) + "</pre>";
    }
    // perception (code) — collapsible
    html += '<details' + (emptyP ? "" : " open") + '><summary style="cursor:pointer;font-size:12px;color:var(--text-muted)">perception.py (P) — ' + (emptyP ? "empty (seed)" : (perception.split("\n").length + " lines")) + "</summary>";
    if (!emptyP) {
      html += '<pre style="white-space:pre;background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:10px;margin:8px 0 0;font-size:12px;max-height:360px;overflow:auto">' + esc(perception) + "</pre>";
    }
    html += "</details>";
    html += "</div></div>";
  });

  c.innerHTML = html;
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
        const validationBadge = validationBadgesHtml(turn);

        const turnImagesHtml = promptImagesHtml(turn.prompt || "");

        turnsHtml += '<div class="extraction-section" style="margin-bottom:8px">' +
          '<div class="extraction-header" onclick="toggleBody(this)" style="padding:8px 12px">' +
          '<span style="font-weight:600;color:' + meta.color + '">Turn ' + turn.turn + '</span>' +
          '<span style="color:var(--accent3);font-family:var(--font-mono);font-size:11px;margin-left:8px">$' + (turn.cost || 0).toFixed(4) + "</span>" +
          submitBadge +
          validationBadge +
          '<span style="margin-left:auto;font-size:11px;color:var(--text-muted)">&#9654;</span>' +
          "</div>" +
          '<div class="extraction-body">' +
          turnImagesHtml +
          '<div style="margin-bottom:10px"><div style="font-size:10px;text-transform:uppercase;color:var(--text-muted);margin-bottom:4px;font-weight:600">Prompt</div>' +
          '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:10px 14px">' +
          '<pre style="max-height:400px;margin:0;border:none;padding:0;background:transparent">' + esc(turn.prompt || "") + "</pre></div></div>" +
          '<div style="margin-bottom:6px"><div style="font-size:10px;text-transform:uppercase;color:var(--accent2);margin-bottom:4px;font-weight:600">Response</div>' +
          '<div style="background:rgba(88,166,255,0.05);border:1px solid rgba(88,166,255,0.2);border-radius:8px;padding:10px 14px">' +
          renderResponseBody(turn.response || "") + "</div></div>" +
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
  const stepMeta = DATA && DATA.steps ? DATA.steps[selectedStepIdx] : null;
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
      messageAttachmentsHtml(msg, stepMeta) +
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
        '<span style="margin-left:auto;font-size:11px">&#9660;</span></div>' +
        '<div class="extraction-body open"><ul style="margin:0;padding-left:18px;font-size:12px">';
      newQ.forEach((q) => {
        stepHtml += '<li style="margin-bottom:2px;color:var(--accent3)">' + esc(q) + "</li>";
      });
      stepHtml += "</ul></div></div>";
    }

    const allQ = item.all_questions || [];
    if (allQ.length > 0) {
      const hasSourceIndex = allQ.some((q) => q.source_index != null);
      stepHtml += '<div class="extraction-section"><div class="extraction-header" onclick="toggleBody(this)" style="padding:6px 10px">' +
        '<span style="font-size:11px;color:var(--text-muted);font-weight:600">All Questions (' + allQ.length + ')</span>' +
        '<span style="margin-left:auto;font-size:11px">&#9660;</span></div>' +
        '<div class="extraction-body open"><table class="data-table" style="font-size:11px"><tr><th>#</th>' +
        (hasSourceIndex ? "<th>Bank #</th>" : "") + '<th>Question</th><th>Status</th></tr>';
      allQ.forEach((q, qi) => {
        const status = q.answer === null || q.answer === undefined
          ? '<span class="verdict verdict-unanswered">UNANSWERED</span>'
          : q.answer === true
            ? '<span class="verdict verdict-correct">YES</span>'
            : '<span class="verdict verdict-incorrect">NO</span>';
        stepHtml += "<tr><td>Q" + (qi + 1) + "</td>" +
          (hasSourceIndex ? "<td>" + (q.source_index != null ? "Q" + (q.source_index + 1) : "") + "</td>" : "") +
          "<td>" + esc(q.question) + "</td><td>" + status + "</td></tr>";
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

  let html = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:4px">Experiment events. Score-top-k events show candidate scoring and whether a fresh candidate or the already-active experiment won.</div>';
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
    if (item.experiment_selection_mode) {
      eventHtml += '<span style="font-size:11px;padding:2px 8px;background:rgba(80,120,220,0.15);color:var(--accent3);border-radius:4px;font-weight:600">' + esc(item.experiment_selection_mode) + "</span>";
    }
    if (item.did_formulate_experiment) {
      eventHtml += '<span style="font-size:11px;padding:2px 8px;background:rgba(63,185,80,0.15);color:var(--accent2);border-radius:4px;font-weight:600">new experiment</span>';
    } else if (item.experiment_plan) {
      eventHtml += '<span style="font-size:11px;padding:2px 8px;background:rgba(139,148,158,0.15);color:var(--text-muted);border-radius:4px;font-weight:600">kept active</span>';
    }
    if (item.candidate_count) {
      eventHtml += '<span style="font-size:11px;color:var(--text-muted)">' + item.candidate_count + " candidates</span>";
    }
    if (item.winner_kind) {
      eventHtml += '<span style="font-size:11px;color:var(--text-muted)">winner: <strong style="color:var(--accent2)">' + esc(item.winner_kind) + "</strong>" +
        (item.winner_score != null ? " score " + Number(item.winner_score || 0).toFixed(0) : "") + "</span>";
    }
    eventHtml += "</div>";

    // Show selected question
    if (item.selected_question_index != null) {
      const qText = item.selected_question_text || ("Q" + (item.selected_question_index + 1));
      const sourceLabel = item.selected_question_source_index != null
        ? " / bank Q" + (item.selected_question_source_index + 1)
        : "";
      eventHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">SELECTED QUESTION (prompt Q' + (item.selected_question_index + 1) + sourceLabel + "):</div>";
      eventHtml += '<div style="font-size:12px;padding:6px 10px;margin-bottom:8px;background:var(--bg);border:1px solid var(--accent3);border-radius:4px;color:var(--accent3)">' + esc(qText) + "</div>";
    }

    if (item.experiment_plan) {
      eventHtml += '<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">' + (item.did_formulate_experiment ? "FORMULATED EXPERIMENT:" : "ACTIVE EXPERIMENT WON:") + "</div>";
      eventHtml += '<div style="font-size:12px;padding:6px 10px;background:var(--bg);border:1px solid var(--accent2);border-radius:4px">' + esc(item.experiment_plan) + "</div>";
    }
    eventHtml += "</div>";
    html += eventHtml;
  });

  c.innerHTML = html;
}

init();
