"""Subprocess entry point: same single-frame semantics as validate.run_perceive."""
from __future__ import annotations

import contextlib
import json
import os
import signal
import sys


def evaluate(code: str, observations: dict[str, str], timeout: float) -> dict:
    def expire(_sig, _frame):
        raise TimeoutError("perceive exceeded frame timeout")

    signal.signal(signal.SIGALRM, expire)
    results = {}
    with open(os.devnull, "w") as sink:
        for fid, observation in observations.items():
            try:
                signal.setitimer(signal.ITIMER_REAL, timeout)
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    if not code.strip():
                        value = ""
                    else:
                        namespace = {}
                        exec(code, namespace)
                        fn = namespace.get("perceive")
                        if not callable(fn):
                            raise ValueError("no callable perceive()")
                        value = fn([observation])
                        value = value if isinstance(value, str) else str(value)
                if len(value) > 250_000:
                    raise ValueError("perception output exceeds 250000 characters")
                results[fid] = {"output": value, "error": None}
            except Exception as exc:
                results[fid] = {"output": "", "error": f"{type(exc).__name__}: {exc}"}
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    return results


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    result = evaluate(payload["code"], payload["observations"], payload["timeout"])
    json.dump(result, sys.stdout)
