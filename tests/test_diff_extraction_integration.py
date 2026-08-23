"""End-to-end tests for the diff-mode extraction path in
_improve_with_perception_validation_conversational.

Monkey-patches the LLM call so the test can drive scripted responses through
the real extraction / diff-apply / validate pipeline.
"""

import asyncio
from unittest.mock import patch

import explore.b_learn_improve as b_learn_improve


BASE_PERCEPTION = """import re

def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    return obs[:100]
"""


def _fake_call(responses):
    """Return an async stand-in for _llm_call_conversational that yields scripted responses.

    Replays the last response indefinitely once the iterator is exhausted, so
    retry loops that exceed the scripted list don't crash the test.
    """
    pending = list(responses)
    last = [pending[-1]] if pending else [""]

    async def fake(config, history, message, images=None):
        text = pending.pop(0) if pending else last[0]
        last[0] = text
        return (
            text,
            0.0,
            list(history)
            + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": text},
            ],
        )

    return fake


def _run(perception, response_text, **kwargs):
    fake = _fake_call([response_text])
    with patch.object(b_learn_improve, "_llm_call_conversational", fake):
        return asyncio.run(
            b_learn_improve._improve_with_perception_validation_conversational(
                config=None,
                beliefs="",
                perception=perception,
                conversation_history=[],
                user_message="(test prompt)",
                sample_observations=[("test obs", 0)],
                extraction_mode="diff",
                **kwargs,
            )
        )


def test_diff_applied_successfully():
    response = """<think>tweak return slice</think>
<updated_perception>
```diff
@@ -2,4 +2,4 @@
 def perceive(observation_history: list[str]) -> str:
     obs = observation_history[-1] if observation_history else ""
-    return obs[:100]
+    return obs[:200]
```
</updated_perception>
<status>CONTINUE</status>"""
    beliefs, perception, _cost, _hist, _resp, err = _run(BASE_PERCEPTION, response)
    assert err is None, err
    assert "obs[:200]" in perception
    assert "obs[:100]" not in perception


def test_diff_with_bad_context_rejected():
    response = """<updated_perception>
```diff
@@ -1,3 +1,3 @@
 def perceive(observation_history: list[str]) -> str:
-    return "this text is not in the source"
+    return "anything"
```
</updated_perception>"""
    _b, perception, _c, _h, _r, err = _run(BASE_PERCEPTION, response)
    assert err is not None
    assert "not found" in err.lower() or "could not apply" in err.lower()
    # On failure, perception must be reverted to the original.
    assert perception == BASE_PERCEPTION


def test_full_code_escape_hatch_in_diff_mode():
    response = """<updated_perception>
```python
import re

def perceive(observation_history: list[str]) -> str:
    return "rewritten"
```
</updated_perception>"""
    _b, perception, _c, _h, _r, err = _run(BASE_PERCEPTION, response)
    assert err is None, err
    assert "rewritten" in perception
    assert "obs[:100]" not in perception


def test_empty_base_with_diff_is_rejected():
    response = """<updated_perception>
```diff
@@ -1 +1 @@
-old
+new
```
</updated_perception>"""
    _b, perception, _c, _h, _r, err = _run("", response)
    assert err is not None
    assert "no existing" in err.lower() or "diff" in err.lower()


def test_empty_base_with_full_code_accepted():
    response = """<updated_perception>
```python
def perceive(observation_history: list[str]) -> str:
    return "fresh"
```
</updated_perception>"""
    _b, perception, _c, _h, _r, err = _run("", response)
    assert err is None, err
    assert "fresh" in perception


def test_keep_unchanged_in_diff_mode():
    response = """<updated_perception>KEEP_UNCHANGED</updated_perception>"""
    _b, perception, _c, _h, _r, err = _run(
        BASE_PERCEPTION, response, allow_keep_perception=True
    )
    assert err is None, err
    assert perception == BASE_PERCEPTION


def test_diff_passes_validation_after_apply():
    """Diff applies cleanly AND resulting code passes validate_perception_code."""
    response = """<updated_perception>
```diff
@@ -2,4 +2,4 @@
 def perceive(observation_history: list[str]) -> str:
     obs = observation_history[-1] if observation_history else ""
-    return obs[:100]
+    return "OK:" + obs[:80]
```
</updated_perception>"""
    _b, perception, _c, _h, _r, err = _run(BASE_PERCEPTION, response)
    assert err is None, err
    assert "OK:" in perception


def test_diff_yielding_broken_code_fails_validation():
    """Apply diff produces syntactically broken code → validation rejects it."""
    response = """<updated_perception>
```diff
@@ -2,4 +2,4 @@
 def perceive(observation_history: list[str]) -> str:
     obs = observation_history[-1] if observation_history else ""
-    return obs[:100]
+    return obs[:100  # broken (unterminated
```
</updated_perception>"""
    # Single attempt only to keep the test quick (max_retries=1).
    _b, perception, _c, _h, _r, err = _run(BASE_PERCEPTION, response, max_retries=1)
    assert err is not None
    assert perception == BASE_PERCEPTION


if __name__ == "__main__":
    import sys

    failed = 0
    tests = [obj for name, obj in globals().items() if name.startswith("test_")]
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR {fn.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
