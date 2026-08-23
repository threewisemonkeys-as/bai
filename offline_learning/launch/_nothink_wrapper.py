"""Run rexpure_optimize.py with an OpenRouter `reasoning` override injected into
every litellm call of THIS process.

Why: litellm's responses->completion bridge whitelists request fields and drops
`reasoning` (and OpenrouterConfig clobbers user extra_body), so hybrid-thinking
models like qwen3.7-flash cannot be switched out of thinking mode through the
harness's litellm.aresponses path. The bridge handler does
`acompletion(**litellm_completion_request)`, and a top-level `reasoning` kwarg on
acompletion DOES reach OpenRouter (validated 2026-08-07, probe round 2) -- so we
patch the bridge's request transform to add it there.

Usage (in place of rexpure_optimize.py, same args):
    REASONING_OVERRIDE_JSON='{"effort": "none"}' \
        python _nothink_wrapper.py --run ... [rexpure args]

The override applies to ALL LLM calls in the process (fine for single-model
eval-only runs; do not use for mixed-model runs).
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

REXPURE = Path(__file__).resolve().parents[1] / "rexpure_optimize.py"


def _patch_bridge(reasoning: dict) -> None:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig as Cfg,
    )

    orig = Cfg.transform_responses_api_request_to_chat_completion_request

    def patched(*args, **kwargs):
        req = orig(*args, **kwargs)
        req["reasoning"] = reasoning
        return req

    Cfg.transform_responses_api_request_to_chat_completion_request = staticmethod(patched)


def main() -> None:
    reasoning = json.loads(os.environ["REASONING_OVERRIDE_JSON"])
    _patch_bridge(reasoning)
    print(f"[nothink-wrapper] reasoning override active: {reasoning}", flush=True)
    sys.path.insert(0, str(REXPURE.parent))  # rexpure imports its siblings bare
    sys.argv = [str(REXPURE)] + sys.argv[1:]
    runpy.run_path(str(REXPURE), run_name="__main__")


if __name__ == "__main__":
    main()
