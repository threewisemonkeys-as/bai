"""Unified-diff parsing and application for LLM-produced perception patches.

The applier is intentionally forgiving about line numbers:
- `@@ -X,Y +A,B @@` headers are consumed only to delimit hunks; the line
  numbers themselves are ignored.
- Each hunk is matched as a (context + removed) substring of the source; if
  it occurs exactly once, it is replaced with (context + added).
- If the "before" block is missing or ambiguous, application fails with a
  message intended to be surfaced back to the LLM as a validation error.

This matches how aider-style search/replace patches behave while accepting
standard unified-diff syntax produced by `diff -u` or LLMs trained on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Hunk:
    before: str
    after: str


_OPEN_FENCE_RE = re.compile(
    r"^```(?:diff|patch|python|py)?[ \t]*\n?", re.IGNORECASE
)
_TRAILING_FENCE_RE = re.compile(r"\n?```\s*$")


def strip_code_fences(text: str) -> str:
    """Remove a single pair of opening/closing triple-backtick fences."""
    text = text.strip()
    text = _OPEN_FENCE_RE.sub("", text, count=1)
    text = _TRAILING_FENCE_RE.sub("", text)
    return text.strip("\n")


def looks_like_unified_diff(text: str) -> bool:
    """Return True if the text appears to contain a unified diff hunk."""
    if not text:
        return False
    for line in text.splitlines():
        s = line.lstrip()
        if s.startswith("@@") and s.rstrip().endswith("@@"):
            return True
        if s.startswith("@@ "):
            return True
    return False


def parse_unified_diff(patch: str) -> list[Hunk]:
    """Parse a unified diff string into hunks.

    `--- a/...` / `+++ b/...` file headers and `@@ ... @@` line ranges are
    consumed as delimiters; only ` ` (context), `-` (removed) and `+` (added)
    body lines contribute to the hunk content.
    """
    hunks: list[Hunk] = []
    before_lines: list[str] = []
    after_lines: list[str] = []
    in_hunk = False

    def flush() -> None:
        nonlocal before_lines, after_lines
        if before_lines or after_lines:
            hunks.append(
                Hunk(
                    before="\n".join(before_lines),
                    after="\n".join(after_lines),
                )
            )
        before_lines = []
        after_lines = []

    for raw_line in patch.splitlines():
        if raw_line.startswith("@@"):
            if in_hunk:
                flush()
            in_hunk = True
            continue
        if raw_line.startswith("--- ") or raw_line.startswith("+++ "):
            continue
        if not in_hunk:
            continue
        if raw_line.startswith("\\"):
            continue  # "\ No newline at end of file"
        if raw_line.startswith("-"):
            before_lines.append(raw_line[1:])
        elif raw_line.startswith("+"):
            after_lines.append(raw_line[1:])
        elif raw_line.startswith(" "):
            before_lines.append(raw_line[1:])
            after_lines.append(raw_line[1:])
        elif raw_line == "":
            # Some LLMs drop the leading space on blank context lines.
            before_lines.append("")
            after_lines.append("")

    if in_hunk:
        flush()
    return hunks


def apply_unified_diff(source: str, patch: str) -> tuple[str | None, str | None]:
    """Apply a unified diff to ``source``.

    Returns ``(result, error)``. On success ``error`` is None.
    """
    hunks = parse_unified_diff(patch)
    if not hunks:
        return None, (
            "No hunks found in patch. Expected at least one `@@ ... @@` "
            "header followed by context / `-` / `+` lines."
        )

    result = source
    for i, hunk in enumerate(hunks, 1):
        if hunk.before == hunk.after:
            continue
        if not hunk.before:
            if result and not result.endswith("\n"):
                result = result + "\n"
            result = result + hunk.after
            continue
        count = result.count(hunk.before)
        if count == 0:
            preview = hunk.before[:300]
            if len(hunk.before) > 300:
                preview += "..."
            return None, (
                f"Hunk {i}: 'before' block (context + removed lines) not "
                f"found in source. The context probably doesn't match the "
                f"actual source verbatim. Block was:\n{preview}"
            )
        if count > 1:
            return None, (
                f"Hunk {i}: 'before' block matches {count} locations in the "
                f"source. Include more surrounding context to disambiguate."
            )
        result = result.replace(hunk.before, hunk.after, 1)

    return result, None
