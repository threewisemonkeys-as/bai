"""Render each clean_data3/<game>/train set as a WHOLE-TRAJECTORY filmstrip with the
curated training transitions highlighted (like the old build_dataset_viz.py --highlight).

The clean_data3 train set is physically stored as many short non-contiguous slices (one
episode per slice, so load_transitions windows stop at boundaries). For eyeballing it is
clearer to show the FULL original clean_data2 trajectory and gold-mark which transitions
were selected into the curated set -- i.e. "which frames are selected for training", in
context of the whole trajectory.

    uv run python prototypes/perc_invdyn/build_clean_data3_viz.py            # all games
    uv run python prototypes/perc_invdyn/build_clean_data3_viz.py dq8gc ada85
"""
import csv, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_dataset_viz import render_png_b64           # reuse the exact renderer/palette
from clean_data3_tools import grid_at, classify

WL = {
    "27vwc": "noop,click", "7www9": "left,right,up,down,noop", "7xf97": "left,right,up,down,noop,click",
    "83wkq": "noop,click", "ada85": "noop,click", "aw9wd": "noop,click",
    "bt3gb": "left,right,up,down,noop,click", "dgg2c": "left,right,up,down,noop,click",
    "dq8gc": "left,right,up,down,noop,click", "e3v6m": "left,right,up,down,noop,click",
    "eahcw": "left,right,up,down,noop,click", "f5w3n": "left,right,up,noop",
    "ice": "left,right,up,down,noop,click", "n2ntd": "left,right,up,down,noop,click",
    "nrdf6": "noop,click", "ntq4y": "noop,click", "qfsvc": "left,right,up,down,noop,click",
    "qqm74": "left,right,up,down,noop,click", "s2kt7": "noop,click", "va6fq": "noop,click",
    "vqjh6": "left,right,up,down,noop,click",
    # ARC-AGI-3 games
    "ls20": "ACTION1,ACTION2,ACTION3,ACTION4", "ft09": "ACTION6", "tn36": "ACTION6",
    "vc33": "ACTION6", "sp80": "ACTION1,ACTION2,ACTION3,ACTION4,ACTION5,ACTION6",
}

CSS = """
body{background:#0c0e12;color:#cdd3da;font:13px ui-monospace,Menlo,monospace;margin:16px}
h2{margin:0 0 4px} .sub{color:#8a93a0;margin-bottom:14px}
.strip{display:flex;flex-wrap:wrap;gap:10px}
figure{margin:0;background:#151921;border:1px solid #232a33;border-radius:6px;padding:6px;position:relative;width:150px}
figure.sel{border:2px solid #ffd45a;background:#211d12;box-shadow:0 0 0 2px rgba(255,212,90,.25)}
.badge{position:absolute;top:8px;left:8px;background:#ffd45a;color:#1a1407;font-size:10px;font-weight:700;padding:1px 5px;border-radius:3px;letter-spacing:.3px}
img{display:block;image-rendering:pixelated;width:138px;border:1px solid #2a3340}
.noimg{width:138px;height:100px;display:flex;align-items:center;justify-content:center;color:#556}
figcaption{padding:5px 2px 1px;line-height:1.5;font-size:11px}
.s{color:#7fd1ff} .act{color:#ffd45a;font-weight:700} .dim{color:#6b7480}
.chg{color:#7fd99a;font-size:10px;display:block;margin-top:2px;word-break:break-word}
.nc{color:#b56;font-size:10px;display:block;margin-top:2px}
"""


def selected_source_steps(game: str, wl: set) -> set:
    """Original Step values whose outgoing transition is a scored target in the curated set."""
    tdir = HERE / "clean_data3" / game / "train"
    steps = set()
    for ep in tdir.glob("episode_*"):
        rows = list(csv.reader((ep / "trajectory.csv").open()))[1:]
        for i, r in enumerate(rows[:-1]):          # last frame in a slice has no scored pair
            a = (r[1] or "").strip()
            if a and a.split()[0] in wl:
                steps.add(r[0])                     # Step string of the source frame
    return steps


def render_game(game: str):
    wl = {w.strip() for w in WL.get(game, "noop,click").split(",") if w.strip()}
    # prefer a regenerated full trajectory (clean_data3/<game>/train_regen) when present
    # (e.g. bt3gb, whose original rollout never demonstrated ice stacking); else clean_data2.
    regen = HERE / "clean_data3" / game / "train_regen" / "episode_0" / "trajectory.csv"
    full = regen if regen.exists() else (
        HERE / "clean_data2" / game / "train" / "episode_0" / "trajectory.csv")
    if not full.exists():
        return None
    csv.field_size_limit(10 ** 7)
    rows = list(csv.reader(full.open()))[1:]
    sel_steps = selected_source_steps(game, wl)

    cards, actcount = [], Counter()
    for i, r in enumerate(rows):
        step, action, obs = r[0], (r[1] or "").strip(), r[3]
        b64, shape = render_png_b64(obs)
        img = (f'<img src="data:image/png;base64,{b64}">' if b64
               else '<div class="noimg">no grid</div>')
        nxt = rows[i + 1][3] if i + 1 < len(rows) else None
        chg = classify(grid_at(obs), grid_at(nxt)) if nxt is not None else ""
        chg_html = ("" if not chg else
                    f'<span class=nc>{chg}</span>' if chg == "NO_CHANGE" else f'<span class=chg>{chg}</span>')
        sel = step in sel_steps
        if sel:
            actcount[action] += 1
        act_html = (f'<span class=act>{action}</span>' if sel
                    else f'<span class=dim>{action or "(terminal)"}</span>')
        badge = '<span class=badge>&#9733; TRAIN</span>' if sel else ''
        cards.append(
            f'<figure class="{"sel" if sel else ""}">{badge}{img}<figcaption>'
            f'<span class=s>step {step}</span> &rarr; {act_html}{chg_html}</figcaption></figure>')

    hist = " ".join(f"{a}:{c}" for a, c in sorted(actcount.items()))
    html = (f"<!doctype html><meta charset=utf8><title>{game} clean_data3 train selection</title>"
            f"<style>{CSS}</style>"
            f"<h2>{game} &mdash; full trajectory ({len(rows)} frames), "
            f"{len(sel_steps)} transitions selected into the curated clean_data3 train set</h2>"
            f"<div class=sub>Each frame shows the action taken FROM it (producing the next). "
            f"<b style='color:#ffd45a'>Gold-bordered &#9733;TRAIN frames are the scored curated targets.</b> "
            f"Green text = what changes; red = NO_CHANGE. Selected by action: <b style='color:#ffd45a'>{hist}</b></div>"
            f"<div class=strip>{''.join(cards)}</div>")
    out = HERE / "clean_data3" / game / "train" / "viz.html"
    out.write_text(html)
    return out, len(rows), len(sel_steps)


def main():
    games = sys.argv[1:] or sorted(p.name for p in (HERE / "clean_data3").iterdir()
                                   if (p / "train").is_dir())
    for g in games:
        res = render_game(g)
        if res:
            out, nf, ns = res
            print(f"{g:8s} -> {out}  (full traj {nf} frames, {ns} selected)")
        else:
            print(f"{g:8s} -> SKIP (no clean_data2 trajectory)")


if __name__ == "__main__":
    main()
