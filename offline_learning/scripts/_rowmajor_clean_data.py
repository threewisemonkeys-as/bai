"""One-off: convert clean_data click actions + embedded help text to ROW-MAJOR.

Transformation (provably consistent; does NOT re-run the env):
- Action-column `click A B` (old col-first: col=A,row=B) -> `click B A` (row-major
  label for the SAME physical cell). Verified: for old data the visible change
  sits at grid[B][A]; relabelling to `click B A` makes the row-major reading
  (row=B,col=A) point at that same cell.
- Embedded action-help line `click X Y (...; column/x first, ...)` -> the new
  row-major help line (preserving the per-game grid bound 0..N).
- viz.html display labels `<b>click A B</b>` -> `<b>click B A</b>`.

Safeguards:
- Skips any CSV already converted (contains the new help text) -> idempotent and
  protects the manually-converted games (ada85, va6fq).
- Skips games whose gepa_optimize process is live-reading them (SKIP_ACTIVE).
- Atomic writes (temp + os.replace).

Run with --apply to write; default is a dry run.
"""
import csv, io, re, sys, glob, os

# games with a live gepa_optimize run reading clean_data/<game> -- leave untouched
SKIP_ACTIVE = {"nrdf6", "ntq4y", "qqm74", "vqjh6"}
NEW_HELP_MARK = "ROW first, then COL"

HELP_RE = re.compile(
    r"click X Y\s+\(X=column, Y=row, both in 0\.\.(\d+); column/x first, then row/y\)"
)
def help_sub(text):
    return HELP_RE.sub(
        r"click ROW COL  (ROW first, then COL, both in 0..\1; "
        r"matches the (row, col) order the perception reports)",
        text,
    )

CLICK_RE = re.compile(r"^click (\d+) (\d+)$")
def swap_action(a):
    m = CLICK_RE.match(a.strip())
    return f"click {m.group(2)} {m.group(1)}" if m else a

VIZ_RE = re.compile(r"(<b>click )(\d+) (\d+)(</b>)")
def viz_sub(text):
    return VIZ_RE.sub(lambda m: f"{m.group(1)}{m.group(3)} {m.group(2)}{m.group(4)}", text)

def atomic_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        fh.write(data)
    os.replace(tmp, path)

def game_of(path, root):
    return os.path.relpath(path, root).split(os.sep)[1]

apply = "--apply" in sys.argv
root = os.path.dirname(os.path.abspath(__file__))

csv_files = sorted(glob.glob(os.path.join(root, "clean_data/*/episode_*/trajectory.csv")))
html_files = sorted(glob.glob(os.path.join(root, "clean_data/*/viz.html")))

tot_acts = tot_csv = tot_viz = 0
skipped_active = set()
skipped_done = set()
converted_games = set()

for path in csv_files:
    rel = os.path.relpath(path, root)
    g = game_of(path, root)
    if g.endswith("_bak"):
        continue
    if g in SKIP_ACTIVE:
        skipped_active.add(g); continue
    raw = open(path).read()
    if NEW_HELP_MARK in raw:
        skipped_done.add(g); continue
    converted_games.add(g)
    rows = list(csv.reader(io.StringIO(raw)))
    acts = 0
    for r in rows:
        for j, field in enumerate(r):
            r[j] = help_sub(field)
        if len(r) > 1:
            na = swap_action(r[1])
            if na != r[1]:
                acts += 1; r[1] = na
    out = io.StringIO(); csv.writer(out).writerows(rows)
    tot_acts += acts; tot_csv += 1
    if apply:
        atomic_write(path, out.getvalue())
    print(f"  {rel}: {acts} click swap(s)")

for path in html_files:
    g = game_of(path, root)
    # Only touch viz for games whose CSV we just converted in this run. Games that
    # were skipped (already converted by hand, or live) already have matching viz.
    if g not in converted_games:
        continue
    raw = open(path).read()
    new = viz_sub(help_sub(raw))
    if new != raw:
        tot_viz += 1
        if apply:
            atomic_write(path, new)
        print(f"  {os.path.relpath(path, root)}: viz updated")

print(f"\n{'APPLIED' if apply else 'DRY RUN'}: {tot_csv} CSVs converted "
      f"({tot_acts} click swaps), {tot_viz} viz files updated")
print(f"  skipped (already converted): {', '.join(sorted(skipped_done)) or 'none'}")
print(f"  skipped (live gepa run): {', '.join(sorted(skipped_active)) or 'none'}")
