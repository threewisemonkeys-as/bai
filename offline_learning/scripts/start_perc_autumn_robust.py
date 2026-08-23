import json
from collections import Counter


def perceive(observation_history: list[str]) -> str:
    """Robust autumn-grid perception.

    The autumn observation embeds the grid as ONE contiguous JSON 2D array of
    colour-name strings (not split one row per text line); the first row is not
    necessarily black. Parse it directly via the first "[[" .. last "]]" span,
    then report the step and every cell whose colour differs from the dominant
    (background) colour, so the action between two consecutive states is
    recoverable from the two summaries. Never raises, never returns empty.
    """
    obs = observation_history[-1]
    try:
        step = None
        for line in obs.split("\n"):
            line = line.strip()
            if line.startswith("Step:"):
                try:
                    step = int(line.split(":", 1)[1].strip())
                except Exception:
                    step = None
                break

        s = obs.find("[[")
        e = obs.rfind("]]") + 2
        if s == -1 or e <= 1:
            return f"step={step}; grid=unavailable"
        grid = json.loads(obs[s:e])

        flat = [c for row in grid for c in row]
        if not flat:
            return f"step={step}; grid=empty"
        bg = Counter(flat).most_common(1)[0][0]

        cells = []
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell != bg:
                    cells.append((r, c, cell))

        out = f"step={step}; bg={bg}; n_fg={len(cells)}; fg={cells}"
        if len(out) > 1900:  # keep well under the 2000-char budget
            out = f"step={step}; bg={bg}; n_fg={len(cells)}; fg={cells[:60]} ...(truncated)"
        return out
    except Exception as ex:  # noqa: BLE001 -- must never raise
        return f"step=unknown; parse_failed={type(ex).__name__}"
