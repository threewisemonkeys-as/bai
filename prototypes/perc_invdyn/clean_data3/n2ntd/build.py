import csv, shutil
from pathlib import Path

G = "n2ntd"
SRC = Path(f"prototypes/perc_invdyn/clean_data2/{G}/train/episode_0/trajectory.csv")
OUT = Path(f"prototypes/perc_invdyn/clean_data3/{G}")
rows = list(csv.DictReader(SRC.open())); fields = list(rows[0].keys())
by_step = {int(r["Step"]): r for r in rows}

# Each inner list = consecutive ORIGINAL step numbers -> one episode slice.
# Annotations = which dynamic each internal target pair exercises.
EPISODES = [
    # enemy patrol LEFT + gravity contrast (Mario on floor -> does NOT fall on noop)
    [0, 1],                       # 0->1 noop: enemy moves left; Mario on floor stays put
    # JUMP from floor (up = -4 rows)
    [3, 4],                       # 3->4 up: Mario row11->row7 (jump from floor)
    # gravity FALL on noop + enemy patrol DIRECTION FLIP at right end
    [12, 13, 14],                 # 12->13 noop: Mario falls, enemy reaches right edge;
                                  # 13->14 noop: Mario falls, enemy FLIPS to leftward
    # bullet rises into enemy -> bullet+enemy KILL (collision)
    [27, 28, 29],                 # 27->28 noop: bullet rises to enemy; 28->29 noop: KILL
    # click NEGATIVE (ammo==0 -> no-op) + pure noop NEGATIVE (nothing moves)
    [30, 31, 32],                 # 30->31 click(ammo0)=NO_CHANGE; 31->32 noop=NO_CHANGE
    # clean LEFT moves on floor (enemy already dead)
    [34, 35, 36],                 # 34->35, 35->36 left: Mario col-1
    # LEFT then JUMP from floor
    [38, 39, 40],                 # 38->39 left col-1; 39->40 up jump row11->row7
    # RIGHT moves; first pair also collects a COIN (gold vanishes at Mario)
    [42, 43, 44, 45],             # 42->43 right + COIN pickup; 43->44, 44->45 clean right
    # click FIRE positive + bullet RISE + bullet FREEZE under step + down no-op
    [49, 50, 51, 52, 53, 54],     # 49->50 click FIRE (bullet appears); 50->51,51->52 rise;
                                  # 52->53 noop FREEZE (no move, below step); 53->54 down=NO_CHANGE
]

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "train").mkdir(parents=True)
for ei, steps in enumerate(EPISODES):
    d = OUT / "train" / f"episode_{ei}"; d.mkdir()
    with (d / "trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in steps:
            assert s in by_step, f"missing step {s}"
            w.writerow(by_step[s])

shutil.copytree(f"prototypes/perc_invdyn/clean_data2/{G}/test", OUT / "test")
shutil.copy(f"prototypes/perc_invdyn/clean_data2/{G}/dynamics.txt", OUT / "dynamics.txt")
print("built", OUT, "episodes:", len(EPISODES))
