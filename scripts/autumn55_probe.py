"""Empirical probe over all 55 Autumn programs in autumn_programs/ (the tracked .sexp sources).

Usage (repo root):  BASE_SEED=1 uv run scripts/autumn55_probe.py [game ...]
Robust sweep: run one game per process (PROBE_OUT=<file> ... autumn55_probe.py <game>) under
xargs -P and merge; a worker segfault inside Pool.map hangs the pool forever.
Writes scripts/autumn55_probe_results_s{BASE_SEED}.json.  Programs are read from autumn_programs/ (override with AUTUMN55_PROG_DIR).

Per game measures (all via the real interpreter + harness stdlib):
  - crash-freeness, grid size, background
  - determinism (same seed + same actions twice -> identical)
  - stochasticity: first divergence step between BASE_SEED and BASE_SEED+1 under
    (a) noop-only policy, (b) random-action policy  (-1 = never diverged)
  - passive dynamics: frame-change rate under noops (first 40 / last 40 steps)
  - state size: mean non-background cells, distinct colours, max objects, off-screen cells
  - counterfactual action observability (prefix replay, branch at sampled t):
      arrows / clicks that change the t+1 frame vs the noop branch (h=1),
      and the t+3 frame (h=3, delayed effects); arrow aliasing (# distinct
      t+1 frames among noop+4 arrows); click-location sensitivity (# distinct
      t+1 frames among 8 click positions); click-on-object vs click-on-empty.

IMPORTANT: render_all() is called after EVERY step.  The interpreter's collision primitives
(isFreePos / *NoCollision / nextLiquid) read an occupancy set that only renderAll() rebuilds;
stepping without rendering plays a different game from the benchmark harness.
Never use BASE_SEED=0: seed 0 makes uniformChoice return the first element every call.
"""
import json, os, random, sys, time, traceback
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROG_DIR = os.environ.get("AUTUMN55_PROG_DIR") or os.path.join(ROOT, "autumn_programs")  # tracked .sexp sources
BASE = int(os.environ.get("BASE_SEED", "1"))
OUT = os.environ.get("PROBE_OUT") or os.path.join(ROOT, "scripts", f"autumn55_probe_results_s{BASE}.json")
sys.path.insert(0, f"{ROOT}/Autumn.cpp/build")
sys.path.insert(0, f"{ROOT}/MARAProtocol/python_examples/autumnbench")
import interpreter_module as im  # noqa
from autumnstdlib import autumnstdlib  # noqa

T_BASE = 80
BRANCH_TS = list(range(0, T_BASE, 5))  # 16 branch points
ARROWS = ["left", "right", "up", "down"]


def load(prog, seed):
    it = im.Interpreter()
    it.run_script(prog, autumnstdlib, "", seed)
    try:
        it.set_verbose(False)
    except Exception:
        pass
    return it


def frame(it):
    d = json.loads(it.render_all())
    G = d.pop("GRID_SIZE", 0)
    cells, off, nobj = {}, 0, 0
    for name, elems in d.items():
        nobj += 1
        for e in elems:
            x, y, c = e["position"]["x"], e["position"]["y"], e["color"].lower()
            if 0 <= x < G and 0 <= y < G:
                cells[(x, y)] = c  # later objects overwrite earlier (render order)
            else:
                off += 1
    return frozenset(cells.items()), off, nobj, G


def apply(it, a):
    if a == "noop":
        return
    if a.startswith("click"):
        _, x, y = a.split()
        it.click(int(x), int(y))
    else:
        getattr(it, a)()


def run(prog, seed, actions):
    it = load(prog, seed)
    frames = [frame(it)]  # frame() renders -> cache fresh every step
    for a in actions:
        apply(it, a)
        it.step()
        frames.append(frame(it))
    return it, frames


def rand_actions(G, n, rng):
    acts = []
    for _ in range(n):
        if rng.random() < 0.4:
            acts.append(rng.choice(ARROWS))
        else:
            acts.append(f"click {rng.randrange(G)} {rng.randrange(G)}")
    return acts


def first_div(fa, fb):
    for i, (a, b) in enumerate(zip(fa, fb)):
        if a[0] != b[0]:
            return i
    return -1


def change_stats(frames):
    ch = [frames[i][0] != frames[i + 1][0] for i in range(len(frames) - 1)]
    ncell = [len(frames[i][0] ^ frames[i + 1][0]) for i in range(len(frames) - 1)]
    return ch, ncell


def probe(name):
    # silence interpreter prints
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    t0 = time.time()
    prog = open(f"{PROG_DIR}/{name}.sexp").read()
    res = {"name": name}
    try:
        it = load(prog, BASE)
        f0 = frame(it)
        G = f0[3]
        res["grid"] = G
        res["background"] = it.get_background()
        res["n_objects_init"] = f0[2]
        res["init_cells"] = len(f0[0])
        rng = random.Random(123)
        acts = rand_actions(G, T_BASE, rng)
        noops = ["noop"] * T_BASE

        # determinism
        _, fa = run(prog, BASE, acts)
        _, fb = run(prog, BASE, acts)
        res["deterministic_replay"] = first_div(fa, fb) == -1

        # stochasticity across seeds
        _, n0 = run(prog, BASE, noops)
        _, n1 = run(prog, BASE + 1, noops)
        _, n2 = run(prog, BASE + 2, noops)
        _, a1 = run(prog, BASE + 1, acts)
        res["noop_div_step_s01"] = first_div(n0, n1)
        res["noop_div_step_s02"] = first_div(n0, n2)
        res["act_div_step_s01"] = first_div(fa, a1)

        # passive dynamics (noop policy, seed 0)
        ch, nc = change_stats(n0)
        res["passive_change_rate_0_40"] = sum(ch[:40]) / 40
        res["passive_change_rate_40_80"] = sum(ch[40:]) / 40
        res["passive_cells_changed_mean"] = sum(nc) / len(nc)
        res["passive_ever_changes"] = any(ch)

        # random-policy run stats (seed 0)
        ch, nc = change_stats(fa)
        res["random_change_rate"] = sum(ch) / len(ch)
        res["random_cells_changed_mean"] = sum(nc) / len(nc)
        res["mean_nonbg_cells"] = sum(len(f[0]) for f in fa) / len(fa)
        res["max_nonbg_cells"] = max(len(f[0]) for f in fa)
        cols = set()
        for f in fa:
            cols |= {c for _, c in f[0]}
        res["distinct_colors"] = len(cols)
        res["colors"] = sorted(cols)
        res["max_objects"] = max(f[2] for f in fa)
        res["mean_offscreen_cells"] = sum(f[1] for f in fa) / len(fa)
        res["max_offscreen_cells"] = max(f[1] for f in fa)

        # counterfactual branches
        arrow_obs1 = arrow_obs3 = arrow_tot = 0
        click_obs1 = click_obs3 = click_tot = 0
        cobj_obs1 = cobj_tot = cemp_obs1 = cemp_tot = 0
        arrow_alias, click_distinct, all_distinct = [], [], []
        noop_vs_prev_changed = 0
        for t in BRANCH_TS:
            prefix = acts[:t]
            base = fa[t][0]
            occ = [p for p, _ in base]
            emp = [(x, y) for x in range(G) for y in range(G) if (x, y) not in set(occ)]
            crng = random.Random(1000 + t)
            clicks = [(0, 0), (G // 2, G // 2), (G - 1, G - 1), (G - 1, 0)]
            cobj = crng.sample(occ, min(2, len(occ))) if occ else []
            cemp = crng.sample(emp, min(2, len(emp))) if emp else []
            clicks = clicks + cobj + cemp
            branches = ["noop"] + ARROWS + [f"click {x} {y}" for x, y in clicks]
            out1, out3 = {}, {}
            for b in branches:
                it = load(prog, BASE)
                it.render_all()  # render after EVERY step: collision primitives read the render cache
                for a in prefix:
                    apply(it, a)
                    it.step()
                    it.render_all()
                apply(it, b)
                it.step()
                out1[b] = frame(it)[0]
                it.step()
                it.render_all()
                it.step()
                out3[b] = frame(it)[0]
            if out1["noop"] != base:
                noop_vs_prev_changed += 1
            for a in ARROWS:
                arrow_tot += 1
                arrow_obs1 += out1[a] != out1["noop"]
                arrow_obs3 += (out1[a] != out1["noop"]) or (out3[a] != out3["noop"])
            arrow_alias.append(len({out1[b] for b in ["noop"] + ARROWS}))
            cl = [b for b in branches if b.startswith("click")]
            for b in cl:
                click_tot += 1
                click_obs1 += out1[b] != out1["noop"]
                click_obs3 += (out1[b] != out1["noop"]) or (out3[b] != out3["noop"])
            for x, y in cobj:
                cobj_tot += 1
                cobj_obs1 += out1[f"click {x} {y}"] != out1["noop"]
            for x, y in cemp:
                cemp_tot += 1
                cemp_obs1 += out1[f"click {x} {y}"] != out1["noop"]
            click_distinct.append(len({out1[b] for b in cl}))
            all_distinct.append(len(set(out1.values())))
        res["arrow_observable_h1"] = arrow_obs1 / arrow_tot
        res["arrow_observable_h3"] = arrow_obs3 / arrow_tot
        res["click_observable_h1"] = click_obs1 / click_tot
        res["click_observable_h3"] = click_obs3 / click_tot
        res["click_on_object_observable_h1"] = cobj_obs1 / cobj_tot if cobj_tot else None
        res["click_on_empty_observable_h1"] = cemp_obs1 / cemp_tot if cemp_tot else None
        res["arrow_distinct_outcomes_mean"] = sum(arrow_alias) / len(arrow_alias)  # max 5
        res["click_distinct_outcomes_mean"] = sum(click_distinct) / len(click_distinct)  # max 8
        res["all_distinct_outcomes_mean"] = sum(all_distinct) / len(all_distinct)  # max 13
        res["branch_noop_changed_frac"] = noop_vs_prev_changed / len(BRANCH_TS)
        res["ok"] = True
    except Exception as e:  # noqa
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {e}"[:300]
        res["trace"] = traceback.format_exc()[-600:]
    res["secs"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    names = sorted(f[:-5] for f in os.listdir(PROG_DIR) if f.endswith(".sexp"))
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    if len(names) == 1:  # one game inline (use this from a per-game driver: a segfaulting worker would hang Pool.map)
        results = [probe(names[0])]
    else:
        with Pool(min(12, os.cpu_count() or 4)) as p:
            results = p.map(probe, names, chunksize=1)
    json.dump(results, open(OUT, "w"), indent=1)
    for r in results:
        print(r["name"], "OK" if r["ok"] else "FAIL " + r.get("error", ""), r["secs"], file=sys.stderr)
