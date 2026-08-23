"""Build the 55-game characterisation report (Markdown + HTML) from scripts/autumn55_static.py + probe results.

Usage (repo root):  uv run scripts/autumn55_make_report.py
Reads  scripts/autumn55_probe_results_s1.json (produce with BASE_SEED=1 uv run scripts/autumn55_probe.py)
Writes notes/autumn55_game_characteristics.md and .cache/autumn55/autumn55_report.html
"""
import html, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(ROOT, ".cache", "autumn55")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, f"{ROOT}/Autumn.cpp/build")
sys.path.insert(0, f"{ROOT}/MARAProtocol/python_examples/autumnbench")
import autumn55_static as S  # noqa
import interpreter_module as im  # noqa
from autumnstdlib import autumnstdlib  # noqa

PROG_DIR = os.environ.get("AUTUMN55_PROG_DIR") or os.path.join(ROOT, "autumn_programs")  # tracked .sexp sources
PROBE = json.load(open(os.path.join(ROOT, "scripts", "autumn55_probe_results_s1.json")))
P = {r["name"]: r for r in PROBE}
MD_OUT = os.path.join(ROOT, "notes", "autumn55_game_characteristics.md")
HTML_OUT = os.path.join(SP, "autumn55_report.html")

TIER_LABEL = {"A": "A · core", "B": "B · secondary", "S": "S · stochastic", "X": "X · exclude"}
TIER_ORDER = ["A", "B", "S", "X"]

COLOR_DICT = {}
try:
    import yaml
    COLOR_DICT = yaml.safe_load(open("/home/ays57/bai/MARAProtocol/python_examples/autumnbench/example_benchmark/color_dict.yaml"))
except Exception:
    pass


def fmt(v, pct=False):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def stoch_emp(r):
    n, a = r.get("noop_div_step_s01"), r.get("act_div_step_s01")
    if n == 0:
        return "init"
    if n is not None and n > 0:
        return f"step (t={n})"
    if a is not None and a >= 0:
        return f"event (t={a})"
    return "none"


# ---------- thumbnails (initial frame, seed 1; or after 20 random steps if the start is near-empty) ----------
def render_cells(name):
    prog = open(f"{PROG_DIR}/{name}.sexp").read()
    it = im.Interpreter()
    it.run_script(prog, autumnstdlib, "", 1)
    it.set_verbose(False)

    def grab():
        d = json.loads(it.render_all())
        G = d.pop("GRID_SIZE")
        cells = {}
        for v in d.values():
            for e in v:
                x, y = e["position"]["x"], e["position"]["y"]
                if 0 <= x < G and 0 <= y < G:
                    cells[(x, y)] = e["color"].lower()
        return G, cells
    bg = it.get_background() or "black"
    G, cells = grab()
    if len([c for c in cells.values() if c != bg]) < 3:
        import random
        rng = random.Random(7)
        for _ in range(20):
            if rng.random() < 0.4:
                getattr(it, rng.choice(["left", "right", "up", "down"]))()
            else:
                it.click(rng.randrange(G), rng.randrange(G))
            it.step()
            it.render_all()
        G, cells = grab()
    return G, cells, bg


def svg_thumb(name, size=56):
    try:
        G, cells, bg = render_cells(name)
    except Exception:
        return f'<svg width="{size}" height="{size}" viewBox="0 0 1 1"><rect width="1" height="1" fill="#444"/></svg>'
    rects = []
    for (x, y), c in cells.items():
        if c == bg:
            continue
        rects.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{html.escape(c)}"/>')
    return (f'<svg class="thumb" width="{size}" height="{size}" viewBox="0 0 {G} {G}" shape-rendering="crispEdges" '
            f'role="img" aria-label="{name} initial frame"><rect width="{G}" height="{G}" fill="{html.escape(bg)}"/>{"".join(rects)}</svg>')


# ---------- rows ----------
def row(g):
    r = P.get(g["name"], {})
    ok = r.get("ok", False)
    return dict(
        name=g["name"], id=g["id"] or "", grid=g["grid"], bg=g["bg"], tier=g["tier"],
        stoch=g["stoch"], stoch_emp=stoch_emp(r) if ok else "crash",
        passive=g["passive"], pas0=r.get("passive_change_rate_0_40"), pas40=r.get("passive_change_rate_40_80"),
        noopch=r.get("branch_noop_changed_frac"),
        hidden=g["hidden"], relational=g["relational"], quant=g["quant"], mode=g["mode"],
        inputs=g["inputs"], click=g["click"],
        arr1=r.get("arrow_observable_h1"), arr3=r.get("arrow_observable_h3"),
        clk1=r.get("click_observable_h1"), clk3=r.get("click_observable_h3"),
        cobj=r.get("click_on_object_observable_h1"), cemp=r.get("click_on_empty_observable_h1"),
        arrd=r.get("arrow_distinct_outcomes_mean"), clkd=r.get("click_distinct_outcomes_mean"),
        cells=r.get("mean_nonbg_cells"), maxcells=r.get("max_nonbg_cells"), colors=r.get("distinct_colors"),
        objs=r.get("max_objects"), off=r.get("max_offscreen_cells"), dim=g["dim"],
        mech=g["mech"], why=g["why"], hidden_note=g["hidden_note"], bugs=g["bugs"], prior=g["prior"],
        stoch_note=g["stoch_note"], passive_note=g["passive_note"], rel_note=g["rel_note"], irrev=g["irrev"], ok=ok,
    )


ROWS = [row(g) for g in S.G]
ROWS.sort(key=lambda x: (TIER_ORDER.index(x["tier"]), x["name"].lower()))
N_BENCH = sum(1 for g in S.G if g["id"])
STOCH_PROGS = sorted((g["name"] for g in S.G if g["stoch"] != "none"), key=str.lower)

# ---------- markdown ----------
def md_table(rows, cols):
    head = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for x in rows:
        body += "| " + " | ".join(fmt(c[1](x)) for c in cols) + " |\n"
    return head + body


MAIN_COLS = [
    ("Game", lambda x: f"**{x['name']}**"), ("Bench ID", lambda x: x["id"] or "–"), ("Tier", lambda x: x["tier"]),
    ("Grid", lambda x: x["grid"]), ("Bg", lambda x: x["bg"]),
    ("Stoch (src)", lambda x: x["stoch"]), ("Stoch (probe)", lambda x: x["stoch_emp"]),
    ("Passive", lambda x: x["passive"]), ("Hidden", lambda x: x["hidden"]), ("Relational", lambda x: x["relational"]),
    ("Quant", lambda x: x["quant"]), ("Mode", lambda x: x["mode"]), ("Inputs", lambda x: x["inputs"]), ("Click", lambda x: x["click"]),
]
VIS_COLS = [
    ("Game", lambda x: f"**{x['name']}**"), ("Tier", lambda x: x["tier"]),
    ("Arrow vis h1", lambda x: x["arr1"]), ("Arrow vis h3", lambda x: x["arr3"]), ("Arrow distinct/5", lambda x: x["arrd"]),
    ("Click vis h1", lambda x: x["clk1"]), ("Click vis h3", lambda x: x["clk3"]), ("Click-on-obj h1", lambda x: x["cobj"]),
    ("Click-on-empty h1", lambda x: x["cemp"]), ("Click distinct/8", lambda x: x["clkd"]),
    ("Passive 0-40", lambda x: x["pas0"]), ("Passive 40-80", lambda x: x["pas40"]), ("Noop-change (in play)", lambda x: x["noopch"]),
]
SIZE_COLS = [
    ("Game", lambda x: f"**{x['name']}**"), ("Tier", lambda x: x["tier"]), ("Grid", lambda x: x["grid"]),
    ("Mean cells", lambda x: x["cells"]), ("Max cells", lambda x: x["maxcells"]), ("Colours", lambda x: x["colors"]),
    ("Max objects", lambda x: x["objs"]), ("Max off-screen cells", lambda x: x["off"]), ("Dim", lambda x: x["dim"]),
]


def per_game_md(x):
    s = f"- **{x['name']}**"
    if x["id"]:
        s += f" (`{x['id']}`)"
    s += f" — {x['grid']}×{x['grid']}, {x['bg']} bg. {x['mech']}"
    bits = []
    if x["hidden"] != "none":
        bits.append(f"hidden: {x['hidden_note']}")
    if x["stoch"] != "none":
        bits.append(f"stochastic: {x['stoch_note'] or x['stoch']}")
    if x["bugs"]:
        bits.append(f"quirk: {x['bugs']}")
    if x["prior"]:
        bits.append(f"prior: {x['prior']}")
    if bits:
        s += " _(" + "; ".join(bits) + ")_"
    if x["ok"]:
        s += (f" Probe: arrows visible {fmt(x['arr1'])}/{fmt(x['arr3'])} (h1/h3), clicks {fmt(x['clk1'])}/{fmt(x['clk3'])}, "
              f"passive {fmt(x['pas0'])}, noop-change in play {fmt(x['noopch'])}, ~{fmt(x['cells'])} cells, {x['colors']} colours.")
    s += f" **Why {x['tier']}:** {x['why']}."
    return s + "\n"


md = []
md.append("# AutumnBench 55-program characterisation for offline belief + perception learning\n")
md.append("_Generated 2026-08-22 from `autumn_programs_55.zip` (all 55 `.sexp` sources read in full) plus an empirical probe on the real interpreter "
          "(`scripts/autumn55_probe.py`, results `scripts/autumn55_probe_results_s1.json`). Static labels live in `scripts/autumn55_static.py`; "
          "this file is produced by `scripts/autumn55_make_report.py`._\n")
md.append("""
## TL;DR

**Recommended core set (Tier A, 19 games).** Keep the 10 benchmark worlds that already work in the pipeline — `ice` (reference), `paint` (control),
`lights`, `hatch`, `disease`, `mario`, `coins`, `grow`, `sand`, `bbq` — and add **nine new worlds** that each cover a characteristic the current set lacks:

| Add | Fills this gap | Grid |
|---|---|---|
| `pacman` | state-dependent autonomous dynamics (ghosts chase every 3rd tick) + agent + walls; cannot terminate (capture bug) | 10 |
| `charge` | hidden accumulating resource (energy, never rendered) + timer + gravity; revealed only through jump height | 7 |
| `lightning_rod` | banded hidden counter + threshold-gated click + 20-tick countdown — the temporal complement of `bbq` | 9 |
| `gameOfLife` | neighbourhood-counting rule; dynamics only on a button = a perfectly observable "step" action with a large effect | 16 |
| `blicket` | causal-structure discovery: a disjunctive 2-factor rule with visibility toggles | 11 |
| `arc_slack` | `disease`'s invisible click-to-select, generalised to 7 objects + 2 buttons + deletion | 16 |
| `balls` | autonomous bouncing physics with a latent velocity (needs a 2-frame window) and a toggleable obstacle | 12 |
| `chomp` | turn-taking hidden mode (rendered), geometric quadrant selection, irreversible removal, game-over state | 7 |
| `twiddle` | 9-colour 2×2 permutation rule on a 3×3 grid — the smallest perception/belief sanity check in the corpus | 3 |

Suggested order if budget is limited: `pacman`, `charge`, `lightning_rod`, `gameOfLife`, `blicket`, `arc_slack`, then `balls`, `chomp`, `twiddle`.

**Tier B (20)** are sound but second-choice (large static clutter, delayed effects, reset-time randomness, non-black backgrounds, or already-measured
weak testbeds). **Tier S (8)** are genuinely stochastic and must be scored distributionally (or on their deterministic sub-rules) — never by exact next-frame match.
**Tier X (8)** are excluded: one crashes, one is unplayable at 50×50 with click-independent random dynamics, others are degenerate or near-stateless.

**Two interpreter facts discovered while probing (both affect existing tooling):**

1. **`render_all()` is not side-effect free.** The collision primitives (`isFreePos`, `isFreeExcept`, every `*NoCollision` move, `nextLiquid`) read an
   occupancy set that is rebuilt *only* inside `renderAll()`; `step()` never refreshes it. The benchmark harness renders after every step, so the "true" game
   is the render-every-step one. Any replay that steps without rendering (e.g. `offline_learning/scripts/game_profile.py::_step`, or a prefix replay that
   only renders the final frame) silently plays a different game: after `left`, `right` is blocked by the object's own stale cells (`mario`, `magnets`,
   `block_breaker`, `lightning_rod` all reproduce this). `curated_plan.py` renders before and after each step and is safe. Rule: call `render_all()` after
   every `step()`, always.
2. **Seed 0 makes `uniformChoice` return the first element, every time** (generalising the known `randomPositions`→(0,0) artifact). At seed 0 `kaleidoscope`
   paints only gold, `tetris` spawns only the first piece, `crystallization` particles all drift the same way off-grid, `minesweeper` is ~all mines. Never generate
   data or measure stochastic games at seed 0. Affected programs (any `uniformChoice`/`randomPositions`): """ + ", ".join(f"`{n}`" for n in STOCH_PROGS) + ".\n")

md.append("""
## Characteristics and why each matters for this pipeline

The pipeline learns a perception module P (code) and beliefs B (text) from logged transitions with inverse-dynamics (ID), forward (FD) and contrastive-FD
objectives over a K-step window, then evaluates by multistep planning and the benchmark's MFP / CD / planning tasks. Each characteristic below is a known
failure or success lever of that setup.

| Characteristic | Definition (how it was measured) | Why it matters here |
|---|---|---|
| **Stochasticity** | From source: `uniformChoice`/`randomPositions` sites, classified as *init* (reset only), *event* (on a trigger), *step* (every tick). From probe: first step at which seeds 1 vs 2 diverge under a noop-only policy (`nDiv`) and under a fixed random action sequence (`aDiv`). | Exact-frame FD punishes a correct model; ID/FD ceilings are depressed. *init*-only randomness is fine within a seed but is memorisable (leak) across a single-seed dataset. |
| **Passive dynamics** | From source: none / *scripted* (timers, oscillators — state-independent) / *conditional* (only player-created objects evolve, e.g. falling water) / *autonomous* (always running, state-dependent). From probe: frame-change rate under noops in steps 0–40 and 40–80, and the noop-change rate at in-play states. | Static windows are FD freebies; timer rhythms let ID be gamed by padding; scripted motion is learnable by a counter, autonomous state-dependent motion needs a real model. |
| **Partial observability / hidden state** | From source: unrendered variables (modes, counters, velocities, secret bits, occluded or off-screen objects) and how they are recoverable — from action history (memory), by exploration, or never. | Memory-recoverable latents test the belief window; exploration-recoverable ones test directed exploration; never-recoverable ones (off-screen blobs, secret coin flips) cap every objective. |
| **Action visibility / latency** | Probe: at 16 branch points along a random trajectory, replay the prefix and branch with noop, each arrow, and 8 clicks (4 fixed corners/centre + 2 on occupied cells + 2 on empty cells). `vis h1` = fraction of branches whose t+1 frame differs from the noop branch; `vis h3` = same by t+3. `arrow distinct/5` = mean number of distinct t+1 frames among {noop, 4 arrows} (aliasing); `click distinct/8` = same among the 8 clicks (does click *location* matter). | An action with no visible effect at t+1 is unobservable to h=1 ID (the dominant failure in the GEPA sweep); h3 > h1 means a multistep window is required; low `distinct` means actions alias (ID ceiling < 1). |
| **Relational rules** | From source: adjacency contagion, containment, collision, neighbour counting, graph adjacency, symmetry. | These are what perception code has to compute; where raw frames suffice, P cannot pay off. |
| **Quantitative rules** | From source: counters, thresholds, timers, velocities (and whether they are rendered exactly, in bands, or not at all). | Banded or hidden counters need beliefs that track integers over many steps. |
| **Mode-dependent inputs** | From source: the same key means different things depending on a (possibly hidden) mode. | Forces conditional beliefs; combined with a hidden mode it creates invisible transitions. |
| **Observation size** | Probe: mean/max non-background cells, distinct colours, objects, and cells rendered off-grid (invisible but live). | Perception only pays when raw observations overwhelm the decoder; off-screen cells are unrecoverable state. |
| **Irreversibility / absorbing states** | From source: deletions, deaths, game-over, auto-resets. | Absorbing states truncate useful data; resets create discontinuities a model must explain. |
| **Benchmark membership** | Whether the program is one of the 20 in the public manifest (each of those has an MFP + CD + planning task). | New worlds need curated tasks authored (see the curated-planning procedure) before they can be scored on benchmark metrics. |

""")

md.append("## Master table — source-level characteristics\n\nTier: A core · B secondary · S stochastic tier · X exclude. Inputs: L/R/U/D arrows, C click. "
          "Click: what the click location means (global = ignored, button = fixed cells, object = on game objects, free = anywhere).\n\n")
md.append(md_table(ROWS, MAIN_COLS))
md.append("\n## Master table — probe: action visibility and passive dynamics (seed 1, render-every-step)\n\n"
          "`vis h1/h3`: fraction of counterfactual branches where the action changed the t+1 / t+3 frame relative to a noop. "
          "`distinct`: mean distinct next frames among 5 arrow options / 8 click options. `Passive`: noop-only frame-change rate. "
          "`Noop-change (in play)`: at branch points of the random trajectory, how often a noop still changes the frame.\n\n")
md.append(md_table(ROWS, VIS_COLS))
md.append("\n## Master table — observation size (random policy, 80 steps, seed 1)\n\n")
md.append(md_table(ROWS, SIZE_COLS))

for t in TIER_ORDER:
    md.append(f"\n## Tier {t}: {TIER_LABEL[t].split('· ')[1]}\n\n")
    if t == "A":
        md.append("Deterministic, every latent is recoverable from history or exploration, action effects are visible within 3 steps, "
                  "and the rules need real perception or belief content.\n\n")
    if t == "B":
        md.append("Sound worlds that lose to a Tier-A sibling on cost or signal: heavy static clutter, long action→effect chains, reset-time randomness, "
                  "non-black backgrounds, or a measured weak testbed in the earlier audit. Use when a specific characteristic is needed.\n\n")
    if t == "S":
        md.append("Fresh randomness enters during play. Score with a distributional metric (likelihood of the logged frame, set-based credit) or only on the "
                  "deterministic sub-rules (e.g. crystallisation contagion, ant chasing, rock flight); never by exact next-frame match. Always seed ≠ 0.\n\n")
    if t == "X":
        md.append("Excluded: unrunnable, degenerate, near-stateless, or dominated by a better sibling.\n\n")
    for x in ROWS:
        if x["tier"] == t:
            md.append(per_game_md(x))

md.append("""
## Practical notes for adding new worlds

- **Tasks.** Only the 20 manifest programs have MFP/CD/planning instances. A new world needs a curated planning ladder (frame goals) and, if wanted, CD/MFP
  instances — the curated-planning procedure (5 diagnostics + failure catalogue) applies directly. `gameOfLife`, `chomp`, `twiddle`, `blicket` have natural
  exact-frame goals; `pacman` (eat all pellets), `charge` (reach the stop), `lightning_rod` (light the rod) have natural predicate goals.
- **Backgrounds.** `mario` (white), `balloon` (skyblue), `egg`/`tictactoe` (gray) need the non-black goal-renderer fix; everything in Tier A except `mario` is black.
- **Click surfaces.** Random exploration finds nothing in `lights_new`/`logic_gates` (8 clickable cells of 576), `balloon` (~15), `bottle` (3 cells), `hatch`
  (only bottom/over-feather shells do anything). Directed exploration or authored drives are required there.
- **Off-screen state.** `gravity` (186 cells), `wind` (57), `gravity_3` (42), `kaleidoscope` (36), `SET` (45 template cards) keep live objects outside the
  grid; `coins`/`grow`/`rink` let the agent walk off. Those states are unrecoverable from frames — prefer drives that stay on-grid.
- **Click args are (col,row)** at the interpreter; the MARA env for `disease` transposes them (known).
- **Replay faithfully:** render after every step (see TL;DR #1), and never at seed 0 for any program in the stochastic list.

## Method

- Static: every `.sexp` read in full; randomness sites grepped (`uniformChoice`, `randomPositions` are the interpreter's only RNG entry points); stdlib semantics
  (`nextLiquid`, `nextSolid`, `*NoCollision`, `isFreeExcept`) read from `Autumn.cpp/autumnstdlib/stdlib.sexp`.
- Probe (`scripts/autumn55_probe.py`): per game, seed 1 base; determinism check (same seed + actions twice); seed 1 vs 2 vs 3 divergence under noop-only and
  random policies; 80-step noop run and 80-step random run (40 % arrows / 60 % uniform clicks, same action RNG for all games); 16 counterfactual branch points
  with 13 branches each (noop, 4 arrows, 8 clicks), prefix replayed from reset with `render_all()` after every step, outcomes compared at t+1 and t+3.
  Interpreter: `Autumn.cpp/build/interpreter_module` with the harness stdlib `MARAProtocol/python_examples/autumnbench/autumnstdlib.py`.
- One program (`exp_particles`) raises inside the interpreter and is reported as a crash.
""")

os.makedirs(os.path.dirname(MD_OUT), exist_ok=True)
open(MD_OUT, "w").write("".join(md))
print("wrote", MD_OUT)

# ---------- HTML ----------
def cell(v, kind=None):
    if v is None:
        return '<td class="num na">–</td>'
    if isinstance(v, bool):
        return f'<td class="bool">{"yes" if v else "no"}</td>'
    if isinstance(v, float):
        shade = ""
        if kind == "vis":
            shade = f' style="--v:{v:.2f}"'
        return f'<td class="num vis"{shade} data-v="{v:.3f}">{v:.2f}</td>'
    return f'<td data-v="{html.escape(str(v))}">{html.escape(str(v))}</td>'


def level_chip(v):
    return f'<td data-v="{["none","low","med","high"].index(v) if v in ("none","low","med","high") else 0}"><span class="lvl lvl-{v}">{v}</span></td>'


thumbs = {x["name"]: svg_thumb(x["name"]) for x in ROWS}
trs = []
for x in ROWS:
    trs.append(
        f'<tr class="tier-{x["tier"]}" data-tier="{x["tier"]}" data-name="{x["name"]}">'
        f'<td class="thumbcell">{thumbs[x["name"]]}</td>'
        f'<td class="name" data-v="{x["name"]}"><span class="gname">{x["name"]}</span>{"<span class=bench>"+x["id"]+"</span>" if x["id"] else ""}</td>'
        f'<td data-v="{TIER_ORDER.index(x["tier"])}"><span class="tier tier-{x["tier"]}">{x["tier"]}</span></td>'
        f'<td class="num" data-v="{x["grid"]}">{x["grid"]}</td>'
        f'<td data-v="{x["stoch"]}"><span class="lvl st-{x["stoch"].split("+")[0]}">{x["stoch"]}</span><span class="sub">{html.escape(x["stoch_emp"])}</span></td>'
        f'<td data-v="{x["passive"]}">{x["passive"]}<span class="sub">noop {fmt(x["pas0"])} · play {fmt(x["noopch"])}</span></td>'
        + level_chip(x["hidden"]) + level_chip(x["relational"]) + level_chip(x["quant"])
        + cell(x["mode"]) + f'<td class="mono">{html.escape(x["inputs"])}</td><td>{html.escape(x["click"])}</td>'
        + cell(x["arr1"], "vis") + cell(x["arr3"], "vis") + cell(x["clk1"], "vis") + cell(x["clk3"], "vis") + cell(x["arrd"]) + cell(x["clkd"])
        + cell(x["cells"]) + cell(x["colors"]) + cell(x["off"])
        + f'<td class="why">{html.escape(x["why"])}</td></tr>'
    )

notes_html = []
for t in TIER_ORDER:
    notes_html.append(f'<section class="tiersec" id="tier-{t}"><h3><span class="tier tier-{t}">{t}</span> {html.escape(TIER_LABEL[t].split("· ")[1])}</h3><ul class="notes">')
    for x in ROWS:
        if x["tier"] != t:
            continue
        bits = []
        if x["hidden"] != "none":
            bits.append("<b>hidden:</b> " + html.escape(x["hidden_note"]))
        if x["stoch"] != "none":
            bits.append("<b>stochastic:</b> " + html.escape(x["stoch_note"] or x["stoch"]))
        if x["bugs"]:
            bits.append("<b>quirk:</b> " + html.escape(x["bugs"]))
        if x["prior"]:
            bits.append("<b>prior:</b> " + html.escape(x["prior"]))
        probe = (f'arrows {fmt(x["arr1"])}/{fmt(x["arr3"])} · clicks {fmt(x["clk1"])}/{fmt(x["clk3"])} · passive {fmt(x["pas0"])} · '
                 f'in-play noop-change {fmt(x["noopch"])} · ~{fmt(x["cells"])} cells · {x["colors"]} colours') if x["ok"] else "crashes in interpreter"
        notes_html.append(
            f'<li id="g-{x["name"]}">{thumbs[x["name"]]}<div><div class="nh"><span class="gname">{x["name"]}</span>'
            f'{"<span class=bench>"+x["id"]+"</span>" if x["id"] else ""}<span class="meta">{x["grid"]}×{x["grid"]} · {x["bg"]}</span></div>'
            f'<p>{html.escape(x["mech"])}</p>'
            + (f'<p class="bits">{" · ".join(bits)}</p>' if bits else "")
            + f'<p class="probe">{probe}</p><p class="why"><b>Why {x["tier"]}:</b> {html.escape(x["why"])}.</p></div></li>')
    notes_html.append("</ul></section>")

stoch_list = ", ".join(f"<code>{n}</code>" for n in STOCH_PROGS)
ADD = [("pacman", "state-dependent autonomous dynamics (ghosts chase every 3rd tick) + agent + walls; cannot terminate", 10),
       ("charge", "hidden accumulating resource + timer + gravity, revealed only through jump height", 7),
       ("lightning_rod", "banded hidden counter, threshold-gated click, 20-tick countdown — the temporal complement of bbq", 9),
       ("gameOfLife", "neighbourhood-counting rule; the step button is a perfectly observable action with a large effect", 16),
       ("blicket", "causal-structure discovery: a disjunctive 2-factor rule with visibility toggles", 11),
       ("arc_slack", "disease's invisible click-to-select generalised to 7 objects + 2 buttons + deletion", 16),
       ("balls", "autonomous bouncing physics with a latent velocity and a toggleable obstacle", 12),
       ("chomp", "turn-taking mode, geometric quadrant selection, irreversible removal, game-over", 7),
       ("twiddle", "9-colour 2×2 permutation rule on 3×3 — the smallest perception/belief sanity check", 3)]
add_rows = "".join(f'<tr><td class="mono"><a href="#g-{n}">{n}</a></td><td>{html.escape(d)}</td><td class="num">{g}</td></tr>' for n, d, g in ADD)

page = f"""<title>Autumn 55 Testbed Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@90,500;90,600;90,700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#F6F7F9;--surface:#FFFFFF;--ink:#151922;--muted:#5C6570;--line:#D9DEE5;--accent:#2C56CF;--accent-ink:#FFFFFF;--row:#FBFCFD;
--A:#1E8A5A;--A-bg:#E3F4EA;--B:#A8720F;--B-bg:#FBF0D6;--S:#6E45C8;--S-bg:#EEE7FB;--X:#8B3F3F;--X-bg:#F7E4E4;
--lvl-none:#E9ECF0;--lvl-low:#DCE6F7;--lvl-med:#B9CBF2;--lvl-high:#7F9FE6;--vis0:#EEF1F5;--vis1:#2C56CF;--code:#EEF1F5;--thumb-line:#C9CFD8}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0F1216;--surface:#161A20;--ink:#E7EAEF;--muted:#98A2AF;--line:#2A313B;--accent:#86A6FF;--accent-ink:#0F1216;--row:#13171C;
--A:#4FCB8E;--A-bg:#153626;--B:#E4AD3F;--B-bg:#3A2C10;--S:#B394FF;--S-bg:#2A1F47;--X:#E08585;--X-bg:#3E1F1F;
--lvl-none:#232932;--lvl-low:#25324A;--lvl-med:#2F4A7D;--lvl-high:#4A6FC0;--vis0:#1C2129;--vis1:#86A6FF;--code:#1C2129;--thumb-line:#2F3742}}}}
:root[data-theme="dark"]{{--bg:#0F1216;--surface:#161A20;--ink:#E7EAEF;--muted:#98A2AF;--line:#2A313B;--accent:#86A6FF;--accent-ink:#0F1216;--row:#13171C;
--A:#4FCB8E;--A-bg:#153626;--B:#E4AD3F;--B-bg:#3A2C10;--S:#B394FF;--S-bg:#2A1F47;--X:#E08585;--X-bg:#3E1F1F;
--lvl-none:#232932;--lvl-low:#25324A;--lvl-med:#2F4A7D;--lvl-high:#4A6FC0;--vis0:#1C2129;--vis1:#86A6FF;--code:#1C2129;--thumb-line:#2F3742}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Source Serif 4",Georgia,"Times New Roman",serif;font-size:17px;line-height:1.55}}
h1,h2,h3,.eyebrow,th,.tier,.lvl,.gname,.bench,.chip{{font-family:Archivo,"Helvetica Neue",Arial,sans-serif;font-stretch:90%}}
code,.mono,.num,.sub,.probe,pre{{font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}}
.wrap{{max-width:74ch;margin:0 auto;padding:0 24px}}
.wide{{max-width:1500px;margin:0 auto;padding:0 24px}}
header{{padding:56px 0 28px;border-bottom:1px solid var(--line)}}
.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:600;color:var(--muted)}}
h1{{font-size:40px;line-height:1.08;margin:10px 0 14px;font-weight:700;letter-spacing:-.01em;text-wrap:balance}}
h2{{font-size:26px;margin:44px 0 14px;font-weight:700;text-wrap:balance}}
h3{{font-size:20px;margin:30px 0 10px;font-weight:600}}
p{{margin:0 0 14px}} .lede{{font-size:19px;color:var(--muted)}}
a{{color:var(--accent)}} a:focus-visible,button:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
code{{background:var(--code);padding:1px 5px;border-radius:3px;font-size:.88em}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600;background:var(--surface);position:sticky;top:0;z-index:2;cursor:pointer;user-select:none;white-space:nowrap}}
th.sorted::after{{content:" ▾";color:var(--accent)}} th.sorted.asc::after{{content:" ▴"}}
td.num{{text-align:right}} td.na{{color:var(--muted)}}
td.vis{{background:color-mix(in srgb,var(--vis1) calc(var(--v,0)*38%),var(--vis0));}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}}
tbody tr:nth-child(even){{background:var(--row)}}
.tier{{display:inline-block;min-width:22px;text-align:center;padding:1px 6px;border-radius:4px;font-weight:700;font-size:12px;color:var(--surface)}}
.tier-A{{background:var(--A)}} .tier-B{{background:var(--B)}} .tier-S{{background:var(--S)}} .tier-X{{background:var(--X)}}
h3 .tier{{font-size:14px;vertical-align:middle;margin-right:6px}}
.lvl{{display:inline-block;padding:1px 7px;border-radius:4px;font-size:12px;font-weight:600}}
.lvl-none{{background:var(--lvl-none);color:var(--muted)}} .lvl-low{{background:var(--lvl-low)}} .lvl-med{{background:var(--lvl-med)}} .lvl-high{{background:var(--lvl-high);color:#fff}}
.st-none{{background:var(--lvl-none);color:var(--muted)}} .st-init{{background:var(--B-bg);color:var(--B)}} .st-event{{background:var(--S-bg);color:var(--S)}} .st-step{{background:var(--S);color:#fff}}
.sub{{display:block;font-size:11px;color:var(--muted);margin-top:2px}}
.gname{{font-weight:600;font-size:14px}} .bench{{display:inline-block;margin-left:6px;font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:3px;padding:0 4px;vertical-align:1px}}
td.why{{min-width:260px;max-width:380px;font-size:13px;color:var(--muted);font-family:"Source Serif 4",Georgia,serif}}
.thumbcell{{width:64px}} .thumb{{display:block;border:1px solid var(--thumb-line);border-radius:2px}}
.filters{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0}}
.chip{{border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:999px;padding:4px 12px;font-size:13px;font-weight:600;cursor:pointer}}
.chip[aria-pressed="true"]{{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}}
.filters input{{font:inherit;font-size:14px;padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink)}}
.count{{color:var(--muted);font-size:13px;margin-left:auto}}
.callout{{border:1px solid var(--line);border-left:4px solid var(--accent);background:var(--surface);padding:14px 18px;border-radius:6px;margin:18px 0}}
.callout h3{{margin:0 0 8px;font-size:17px}}
.callout ol{{margin:0;padding-left:20px}} .callout li{{margin-bottom:8px}}
.notes{{list-style:none;padding:0;margin:0;display:grid;gap:12px}}
.notes li{{display:grid;grid-template-columns:56px 1fr;gap:14px;padding:12px 14px;background:var(--surface);border:1px solid var(--line);border-radius:6px}}
.notes p{{margin:0 0 6px;font-size:15px}} .notes .bits,.notes .why{{font-size:14px;color:var(--muted)}} .notes .probe{{font-size:12px;color:var(--muted)}}
.nh{{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:4px}} .meta{{font-size:12px;color:var(--muted)}}
.defs th{{position:static;cursor:default}} .defs td{{font-size:14px}} .defs td:first-child{{white-space:nowrap;font-weight:600;font-family:Archivo,sans-serif;font-stretch:90%}}
.addtab td:first-child{{white-space:nowrap}}
.legend{{font-size:13px;color:var(--muted);margin:8px 0 0}}
footer{{padding:40px 0 60px;color:var(--muted);font-size:14px}}
@media (prefers-reduced-motion:no-preference){{.chip,tbody tr{{transition:background .15s}}}}
</style>
<header><div class="wrap"><div class="eyebrow">AutumnBench · 55 programs · read 2026-08-22</div>
<h1>Which Autumn worlds deserve a belief-and-perception learner?</h1>
<p class="lede">Every <code>.sexp</code> in <code>autumn_programs_55.zip</code> read in full and probed on the real interpreter: stochasticity, passive dynamics, hidden state,
action visibility, relational and quantitative rules, observation size — and a recommendation per game.</p></div></header>

<main>
<div class="wrap">
<h2>Recommendation</h2>
<p><b>Keep the ten benchmark worlds already in the pipeline</b> — <code>ice</code> (reference), <code>paint</code> (control), <code>lights</code>, <code>hatch</code>,
<code>disease</code>, <code>mario</code>, <code>coins</code>, <code>grow</code>, <code>sand</code>, <code>bbq</code> — and <b>add nine new worlds</b>, each chosen because it
carries a characteristic the current set does not:</p>
<div class="tablewrap"><table class="addtab"><thead><tr><th>Add</th><th>Fills this gap</th><th>Grid</th></tr></thead><tbody>{add_rows}</tbody></table></div>
<p class="legend">Suggested order under a budget: pacman, charge, lightning_rod, gameOfLife, blicket, arc_slack; then balls, chomp, twiddle.</p>
<p>That gives a 19-game <b>Tier A</b>. <b>Tier B</b> (20) is sound but second choice — static clutter, long action→effect chains, reset-time randomness, non-black
backgrounds, or a weak measured testbed. <b>Tier S</b> (8) is genuinely stochastic: score distributionally or on the deterministic sub-rules, never by exact
next-frame match. <b>Tier X</b> (8) is excluded: one crashes, one is a 50×50 click-independent random process, the rest are degenerate or near-stateless.</p>

<div class="callout"><h3>Two interpreter facts that affect existing tooling</h3><ol>
<li><b><code>render_all()</code> is not side-effect free.</b> <code>isFreePos</code>, <code>isFreeExcept</code>, every <code>*NoCollision</code> move and <code>nextLiquid</code>
read an occupancy set that is rebuilt only inside <code>renderAll()</code>; <code>step()</code> never refreshes it. The benchmark harness renders after every step, so that is the
true game. A replay that steps without rendering plays a different one: after <code>left</code>, <code>right</code> is blocked by the object's own stale cells
(<code>mario</code>, <code>magnets</code>, <code>block_breaker</code>, <code>lightning_rod</code> all reproduce it). <code>curated_plan.py</code> renders each step and is safe;
<code>offline_learning/scripts/game_profile.py::_step</code> does not. Rule: <code>render_all()</code> after every <code>step()</code>.</li>
<li><b>Seed 0 makes <code>uniformChoice</code> return the first element every time</b> (the known <code>randomPositions</code>→(0,0) artifact is a special case).
At seed 0 <code>kaleidoscope</code> paints only gold, <code>tetris</code> spawns one piece type, <code>crystallization</code> drifts off-grid, <code>minesweeper</code> is nearly all mines.
Never generate or measure at seed 0 for: {stoch_list}.</li></ol></div>

<h2>What each characteristic means here</h2>
<p>The learner fits perception code and belief text from logged transitions with inverse-dynamics, forward and contrastive-forward objectives over a K-step window,
then is judged by multistep planning and the benchmark's MFP / CD / planning tasks. Each column is a known lever of that setup.</p>
<div class="tablewrap"><table class="defs"><thead><tr><th>Characteristic</th><th>How it was read / measured</th><th>Why it matters</th></tr></thead><tbody>
<tr><td>Stochasticity</td><td>Source: <code>uniformChoice</code>/<code>randomPositions</code> sites → init (reset only) / event (on a trigger) / step (every tick). Probe: first step where seeds 1 vs 2 diverge under noops and under one fixed random action sequence.</td><td>Exact-frame FD punishes a correct model; init-only randomness is fine within a seed but memorisable across a single-seed dataset.</td></tr>
<tr><td>Passive dynamics</td><td>Source: none / scripted (timers, oscillators) / conditional (only player-created objects evolve) / autonomous (state-dependent, always on). Probe: noop frame-change rate (steps 0–40) and noop-change rate at in-play states.</td><td>Static windows are FD freebies; timer rhythms let ID be gamed; autonomous state-dependent motion needs a real model.</td></tr>
<tr><td>Hidden state</td><td>Source: unrendered variables — modes, counters, velocities, secret bits, occluded or off-screen objects — and whether they are recoverable from action history, by exploration, or never.</td><td>Memory-recoverable latents test the belief window; exploration-recoverable ones test directed exploration; never-recoverable ones cap every objective.</td></tr>
<tr><td>Action visibility</td><td>Probe: 16 branch points on a random trajectory; replay the prefix, branch with noop / 4 arrows / 8 clicks (corners, centre, 2 on objects, 2 on empty). <i>vis h1</i> = fraction whose t+1 frame differs from the noop branch; <i>h3</i> = by t+3. <i>distinct</i> = mean distinct next frames among the 5 arrow options / 8 click positions.</td><td>No visible effect at t+1 = unobservable to one-step ID (the dominant sweep failure); h3 ≫ h1 = needs a window; low distinct = aliasing, ID ceiling &lt; 1; click-distinct ≈ 1 = click location is irrelevant.</td></tr>
<tr><td>Relational rules</td><td>Source: contagion, containment, collision, neighbour counting, graph adjacency, symmetry.</td><td>What perception code has to compute; where raw frames suffice, P cannot pay off.</td></tr>
<tr><td>Quantitative rules</td><td>Source: counters, thresholds, timers, velocities — rendered exactly, in bands, or not at all.</td><td>Banded or hidden counters need beliefs that track integers over many steps.</td></tr>
<tr><td>Mode-dependent inputs</td><td>Source: the same key means different things depending on a (possibly hidden) mode.</td><td>Forces conditional beliefs; with a hidden mode it creates invisible transitions.</td></tr>
<tr><td>Observation size</td><td>Probe: mean non-background cells, colours, objects, and live cells rendered off-grid.</td><td>Perception pays only when raw frames overwhelm the decoder; off-screen cells are unrecoverable state.</td></tr>
</tbody></table></div>
</div>

<div class="wide">
<h2>All 55 programs</h2>
<p class="legend">Click a header to sort; chips filter by tier; the thumbnail is the real initial frame (seed 1; a few frames in for worlds that start empty).
Visibility cells are shaded by value. Probe numbers use seed 1 with a render after every step.</p>
<div class="filters" role="group" aria-label="Filter by tier">
<button class="chip" data-f="all" aria-pressed="true">All</button><button class="chip" data-f="A" aria-pressed="false">A core</button>
<button class="chip" data-f="B" aria-pressed="false">B secondary</button><button class="chip" data-f="S" aria-pressed="false">S stochastic</button>
<button class="chip" data-f="X" aria-pressed="false">X exclude</button><input id="q" type="search" placeholder="Find a game…" aria-label="Find a game"><span class="count" id="count"></span></div>
<div class="tablewrap"><table id="master"><thead><tr>
<th></th><th data-k="name">Game</th><th data-k="tier">Tier</th><th data-k="grid">Grid</th><th data-k="stoch">Stochastic</th><th data-k="passive">Passive</th>
<th data-k="hidden">Hidden</th><th data-k="rel">Relational</th><th data-k="quant">Quant</th><th data-k="mode">Mode</th><th>Inputs</th><th>Click</th>
<th data-k="arr1" title="arrows change t+1 frame">Arrow h1</th><th data-k="arr3" title="arrows change t+3 frame">Arrow h3</th><th data-k="clk1">Click h1</th><th data-k="clk3">Click h3</th>
<th data-k="arrd" title="distinct next frames among noop+4 arrows">Arrow distinct/5</th><th data-k="clkd" title="distinct next frames among 8 click positions">Click distinct/8</th>
<th data-k="cells">Cells</th><th data-k="colors">Colours</th><th data-k="off">Off-grid</th><th>Why</th></tr></thead>
<tbody>{"".join(trs)}</tbody></table></div>
</div>

<div class="wrap">
<h2>Per-game notes</h2>
{"".join(notes_html)}

<h2>Practical notes for adding worlds</h2>
<ul>
<li><b>Tasks.</b> Only the 20 manifest programs have MFP/CD/planning instances; a new world needs a curated planning ladder (the existing curated-planning procedure applies). <code>gameOfLife</code>, <code>chomp</code>, <code>twiddle</code>, <code>blicket</code> have natural exact-frame goals; <code>pacman</code> (eat all pellets), <code>charge</code> (touch the stop), <code>lightning_rod</code> (light the rod) have natural predicate goals.</li>
<li><b>Backgrounds.</b> <code>mario</code> (white), <code>balloon</code> (skyblue), <code>egg</code>/<code>tictactoe</code> (gray) need the non-black goal-renderer fix; every other Tier-A world is black.</li>
<li><b>Click surfaces.</b> Random play finds nothing in <code>lights_new</code>/<code>logic_gates</code> (8 clickable cells of 576), <code>balloon</code> (~15), <code>bottle</code> (3), and only bottom or over-feather shells in <code>hatch</code>; those need directed exploration or authored drives.</li>
<li><b>Off-grid state.</b> <code>gravity</code> (186 cells), <code>wind</code> (57), <code>gravity_3</code> (42), <code>kaleidoscope</code> (36), <code>SET</code> (45) keep live objects outside the grid; <code>coins</code>, <code>grow</code>, <code>rink</code> let the agent walk off. Unrecoverable from frames — prefer drives that stay on-grid.</li>
<li><b>Click args are (col,row)</b> at the interpreter; the MARA env for <code>disease</code> transposes them.</li>
</ul>

<h2>Method</h2>
<p>Static: every source read in full; the interpreter's only RNG entry points are <code>uniformChoice</code> and <code>randomPositions</code>; stdlib semantics
(<code>nextLiquid</code>, <code>nextSolid</code>, <code>*NoCollision</code>, <code>isFreeExcept</code>) taken from <code>Autumn.cpp/autumnstdlib/stdlib.sexp</code>.
Probe (<code>scripts/autumn55_probe.py</code>): seed-1 base; determinism check; seed 1/2/3 divergence under noop and random policies; 80-step noop and random runs
(40 % arrows / 60 % uniform clicks, one action RNG shared by all games); 16 counterfactual branch points × 13 branches (noop, 4 arrows, 8 clicks), prefix replayed from
reset with <code>render_all()</code> after every step, compared at t+1 and t+3. <code>exp_particles</code> raises inside the interpreter and is reported as a crash.
The Markdown twin of this page is <code>notes/autumn55_game_characteristics.md</code>.</p>
</div>
</main>
<footer><div class="wrap">Autumn 55 testbed atlas · static read + interpreter probe · 2026-08-22</div></footer>
<script>
(function(){{
 const tbl=document.getElementById('master'),tb=tbl.tBodies[0],rows=[...tb.rows],chips=[...document.querySelectorAll('.chip')],q=document.getElementById('q'),cnt=document.getElementById('count');
 let f='all';
 function apply(){{const s=q.value.trim().toLowerCase();let n=0;rows.forEach(r=>{{const ok=(f==='all'||r.dataset.tier===f)&&(!s||r.dataset.name.toLowerCase().includes(s));r.style.display=ok?'':'none';if(ok)n++;}});cnt.textContent=n+' of '+rows.length;}}
 chips.forEach(c=>c.addEventListener('click',()=>{{f=c.dataset.f;chips.forEach(x=>x.setAttribute('aria-pressed',x===c));apply();}}));
 q.addEventListener('input',apply);apply();
 const ths=[...tbl.tHead.rows[0].cells];
 ths.forEach((th,i)=>{{if(!th.dataset.k)return;th.addEventListener('click',()=>{{const asc=th.classList.contains('sorted')&&!th.classList.contains('asc');ths.forEach(t=>t.classList.remove('sorted','asc'));th.classList.add('sorted');if(asc)th.classList.add('asc');
  const val=r=>{{const c=r.cells[i];const v=c.dataset.v!==undefined?c.dataset.v:c.textContent.trim();const n=parseFloat(v);return isNaN(n)?v.toLowerCase():n;}};
  rows.sort((a,b)=>{{const x=val(a),y=val(b);if(typeof x==='number'&&typeof y==='number')return asc?x-y:y-x;if(typeof x==='number')return -1;if(typeof y==='number')return 1;return asc?x.localeCompare(y):y.localeCompare(x);}});
  rows.forEach(r=>tb.appendChild(r));}});}});
}})();
</script>
"""
open(HTML_OUT, "w").write(page)
print("wrote", HTML_OUT, len(page) // 1024, "KB")
