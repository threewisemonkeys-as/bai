"""Static facts about an Autumn program read straight from its .sexp source.

Replaces the per-game literal tables (`mechanics._BG`, `mechanics_rules.BG/SIZE`) that
KeyError'd on any game they did not list and silently assumed a black background. Both
facts are declared in the source: `(= GRID_SIZE n)` and, optionally, `(= background "c")`
(the interpreter renders empty cells black when no background is declared -- every
benchmark world except N2NTD/QQM74 relies on that default).

Names resolve the way the rest of the tooling spells them: a lower-case game code
(`dq8gc`, `n2ntd`), a benchmark id (`DQ8GC`), `ice`, or a zip-sourced name (`rink`,
`balloon`). The program directory is the harness one (see tools/install_autumn_programs.py).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_BAI_ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = _BAI_ROOT / "MARAProtocol/python_examples/autumnbench/example_benchmark/programs"

_BACKGROUND_RE = re.compile(r'\(=\s*background\s+"([^"]+)"\s*\)')
_GRID_SIZE_RE = re.compile(r"\(=\s*GRID_SIZE\s+(\d+)\s*\)")
DEFAULT_BACKGROUND = "black"


def resolve(name: str) -> Path:
    """Path of the .sexp for a game code / benchmark id / zip name (exact, upper, lower)."""
    for cand in (name, name.upper(), name.lower()):
        p = PROGRAMS / f"{cand}.sexp"
        if p.is_file():
            return p
    raise KeyError(f"no Autumn program for {name!r} in {PROGRAMS} "
                   f"(zip-sourced games need tools/install_autumn_programs.py)")


@lru_cache(maxsize=None)
def source(name: str) -> str:
    return resolve(name).read_text()


@lru_cache(maxsize=None)
def background(name: str) -> str:
    """The declared background colour, or the interpreter default ("black")."""
    m = _BACKGROUND_RE.search(source(name))
    return m.group(1).lower() if m else DEFAULT_BACKGROUND


@lru_cache(maxsize=None)
def grid_size(name: str) -> int:
    m = _GRID_SIZE_RE.search(source(name))
    if not m:
        raise ValueError(f"{resolve(name).name} declares no GRID_SIZE")
    return int(m.group(1))


class DerivedTable(dict):
    """A dict that derives a missing game's entry from its program on first access, so
    existing `TABLE[game]` call sites keep working for every installed program."""

    def __init__(self, derive):
        super().__init__()
        self._derive = derive

    def __missing__(self, game):
        value = self._derive(game)
        self[game] = value
        return value

    def get(self, game, default=None):  # dict.get bypasses __missing__; derive here too
        try:
            return self[game]
        except KeyError:
            return default
