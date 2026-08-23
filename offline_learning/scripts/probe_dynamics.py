"""Probe autumn/arc games: which actions visibly change the grid.

Used to pick movement-responsive games for clean-data generation (avoid games
where a fixed sequence degenerates into all no-ops / aliased transitions).
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[2]
if str(_BAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAI_ROOT))


def _autumn_grid(obs_cell: str):
    i = obs_cell.find("[[")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(obs_cell)):
        if obs_cell[j] == "[":
            depth += 1
        elif obs_cell[j] == "]":
            depth -= 1
            if depth == 0:
                return json.loads(obs_cell[i:j + 1])
    return None


def probe_autumn(game, seq):
    from autumn_env import AutumnBenchEnvWrapper
    env = AutumnBenchEnvWrapper(env_name=game, task_type="interactive",
                                max_episode_steps=300, seed=0, render_mode="text")
    obs, _ = env.reset(seed=0)

    def cell(o):
        return o["text"]["short_term_context"] + "\n" + o["text"]["long_term_context"]

    cur = obs
    per_action = {}
    changed = 0
    total = 0
    for a in seq:
        g0 = _autumn_grid(cell(cur))
        nxt, r, term, trunc, info = env.step(a)
        g1 = _autumn_grid(cell(nxt))
        ch = (g0 != g1)
        akey = a.split()[0]
        per_action.setdefault(akey, [0, 0])
        per_action[akey][1] += 1
        per_action[akey][0] += int(ch)
        changed += int(ch)
        total += 1
        cur = nxt
        if term:
            break
    env.close()
    return changed, total, per_action


def probe_arc(game, seq):
    from arc_agi_env import ArcAgiEnvWrapper
    env = ArcAgiEnvWrapper(game_id=game, max_episode_steps=300, seed=0)
    obs, _ = env.reset(seed=0)

    def grids(o):
        txt = o["text"]["long_term_context"] + o["text"]["short_term_context"]
        return re.findall(r"<grid_\d+>(.*?)</grid_\d+>", txt, re.DOTALL)

    cur = obs
    per_action = {}
    changed = 0
    total = 0
    for a in seq:
        g0 = grids(cur)
        nxt, r, term, trunc, info = env.step(a)
        g1 = grids(nxt)
        ch = (g0 != g1)
        akey = a.split()[0]
        per_action.setdefault(akey, [0, 0])
        per_action[akey][1] += 1
        per_action[akey][0] += int(ch)
        changed += int(ch)
        total += 1
        cur = nxt
        if term:
            break
    env.close()
    return changed, total, per_action


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["autumn", "arc"])
    ap.add_argument("games")
    args = ap.parse_args()
    autumn_seq = (["up", "down", "left", "right", "noop"] * 2 +
                  ["click 8 8", "up", "down", "click 4 4", "left", "right"])
    arc_seq = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5",
               "ACTION6 x=20 y=20", "ACTION6 x=40 y=40", "ACTION1", "ACTION2", "ACTION3"]
    for game in args.games.split(","):
        game = game.strip()
        try:
            if args.kind == "autumn":
                ch, tot, pa = probe_autumn(game, autumn_seq)
            else:
                ch, tot, pa = probe_arc(game, arc_seq)
            pas = " ".join(f"{k}={v[0]}/{v[1]}" for k, v in pa.items())
            print(f"{game}: changed {ch}/{tot} | {pas}")
        except Exception as e:
            print(f"{game}: ERROR {type(e).__name__}: {e}")
