"""Shared helper for the archival launch_* scripts: they replay saved launch.json
commands that targeted the (removed) gepa_optimize.py, so before running one we
swap the script to rexpure_optimize.py and strip the gepa-only flags that the
REx-pure CLI does not accept (pareto / gated-rex / val-carve / minibatch knobs)."""
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
REXPURE = str(_HERE / "rexpure_optimize.py")

# gepa-only flags that take a VALUE (strip the flag AND the following token).
_VALUE_FLAGS = {"--selector", "--val-n", "--reflection-minibatch", "--legacy-rounds"}
# gepa-only boolean flags (strip the flag only).
_BOOL_FLAGS = {
    "--no-tie-train-val", "--tie-train-val", "--stratified-split",
    "--swap-click-train", "--no-swap-click-train", "--accept-ties",
    "--no-accept-ties", "--fresh-traces", "--compare", "--legacy-only",
}


def sanitize_rexpure_cmd(cmd, script_path=REXPURE):
    """Return a copy of `cmd` retargeted at rexpure_optimize.py with all gepa-only
    flags (and their values) removed. Any arg ending in 'gepa_optimize.py' is
    rewritten to `script_path`."""
    out, i = [], 0
    while i < len(cmd):
        arg = cmd[i]
        if isinstance(arg, str) and arg.endswith("gepa_optimize.py"):
            out.append(script_path)
        elif arg in _VALUE_FLAGS:
            i += 1  # also skip the value token
        elif arg in _BOOL_FLAGS:
            pass  # drop the flag
        else:
            out.append(arg)
        i += 1
    return out
