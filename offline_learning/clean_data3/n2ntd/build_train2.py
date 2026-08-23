import csv, shutil
from pathlib import Path

G = "n2ntd"
SRC = Path(f"prototypes/perc_invdyn/clean_data3/{G}/train_regen2/episode_0/trajectory.csv")
OUT = Path(f"prototypes/perc_invdyn/clean_data3/{G}/train2")

csv.field_size_limit(10_000_000)
rows = list(csv.DictReader(SRC.open())); fields = list(rows[0].keys())
by_step = {int(r["Step"]): r for r in rows}

# Each inner list = consecutive steps of train_regen2's fresh drive -> one episode slice.
# Annotations = which dynamic(s) each internal target pair exercises (fresh vs train/ + test50/).
EPISODES = [
    # D1 left (2 clean, fresh col2->1->0) + BLOCKED negative at col0 (enemy-alive context,
    # distinct history from train's col0-blocked / test50's col0,col11-blocked instances)
    [6, 7, 8, 9],
    # D8 enemy patrol LEFT-EDGE FLIP at t=22 (fresh cycle; train used t13, test50 used
    # t4/t13/t31) + D2 right (onto coin(9,1), pickup itself NOT scored to avoid duplicating
    # test50 ep9's exact nudge-blocked pair)
    [21, 22, 23],
    # D7 gravity: mario leaves row10-platform support (col3) and lands on the floor
    [27, 28],
    # D3 JUMP floor col3 (open air, no platform underneath) + D7 fall + D3 MIDAIR-UP NEGATIVE
    # (fresh column vs test50's col11 midair-up case) -- same resulting frame as noop would give
    [28, 29, 30, 31],
    # D4 FIRE (floor, fresh col5) -> D9 bullet RISE x2 -> D9 FREEZE (fresh location (9,5),
    # vs train's (9,4) and test50's (9,6))
    [35, 36, 37, 38, 39],
    # D3 JUMP from row8-PLATFORM (row7->row3, col6) + D7 fall + D2 right landing on
    # coin(4,7) + D10 PICKUP resolve (nudge succeeds, row4->row5 -- contrast vs EP6 below)
    [43, 44, 45, 46, 47],
    # D2 right onto row6-platform (rest) + D2 right onto coin(5,9) + D10 PICKUP resolve
    # (nudge BLOCKED by row6 platform -- contrastive negative vs EP5's nudge-succeeds) +
    # D2 right (cleansing) + D3 NOVEL JUMP from row6-platform (highest platform; row5->row1,
    # col10) -- never demonstrated in train/ or test50/
    [47, 48, 49, 50, 51, 52],
    # D7 fall lands back on row6-platform + D5 "down" CONFIRM-rest negative (down==noop,
    # enemy still alive/patrolling in background)
    [55, 56, 57],
    # D9 bullet rise x2 (in-window, fired earlier from the row6-platform height) -> D11 KILL
    # at col10 with the enemy at its right patrol edge (origin10, cols9-11) -- fresh column/
    # configuration vs train (col7) and test50 (cols3,7,11,1)
    [65, 66, 67, 68],
    # D4 FIRE positive (ammo>0, 3rd bullet) -> D9 rise x2 -> D4 AMMO-0 NEGATIVE (same click
    # action, ammo now 0 -> true no-op) -- direct positive/negative contrast in one slice
    [71, 72, 73, 74, 75],
]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
for ei, steps in enumerate(EPISODES):
    d = OUT / f"episode_{ei}"; d.mkdir()
    with (d / "trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in steps:
            assert s in by_step, f"missing step {s}"
            w.writerow(by_step[s])

n_targets = sum(len(s) - 1 for s in EPISODES)
print("built", OUT, "episodes:", len(EPISODES), "scored targets:", n_targets)
