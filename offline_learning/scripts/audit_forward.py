"""Render every forward prediction from diag_forward_results.json in a human-auditable
form: the three abstract states, the TRUE change (z_t->z_t1) and the PREDICTED change
(z_t->z_hat) as token-level difflib edits, and the textdiff score. Lets us eyeball
whether textdiff's number matches how close Ẑ actually is to the true next state."""
import difflib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = json.load(open(HERE / "diag_forward_results.json"))["rows"]


def toks(s):
    return re.findall(r"\d+|[A-Za-z]+|[^\s\w]", s or "")


def change(a, b, cap=600):
    """Compact token-level diff a->b: '-x' removed, '+y' added (first cap chars)."""
    ta, tb = toks(a), toks(b)
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    rem, add = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            rem += ta[i1:i2]
            add += tb[j1:j2]
    s = f"-[{' '.join(rem)}]  +[{' '.join(add)}]"
    return s[:cap] + ("…" if len(s) > cap else "")


def short(s, cap=240):
    s = s or "(empty)"
    return s[:cap] + ("…" if len(s) > cap else "")


envs = sys.argv[1].split(",") if len(sys.argv) > 1 else None
for i, r in enumerate(rows):
    if envs and r["env"] not in envs:
        continue
    print(f"\n#{i:3d} [{r['env']}] action={r['action']!r} moved={r['moved']} "
          f"td={r['td_real']:.3f} judge={r['judge_real']:.2f}")
    print(f"  z_t  : {short(r['z_t'])}")
    print(f"  z_t1 : {short(r['z_t1'])}   <- TRUE next")
    print(f"  z_hat: {short(r['z_hat'])}   <- PREDICTED")
    print(f"  Δtrue (z_t→z_t1): {change(r['z_t'], r['z_t1'])}")
    print(f"  Δpred (z_t→z_hat): {change(r['z_t'], r['z_hat'])}")
