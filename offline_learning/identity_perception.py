"""IDENTITY perception -- the no-perception ablation's fixed P.

Used by `rexpure_optimize.py --no-perception`, which learns ONLY the world-knowledge
(belief) block and freezes P here. `perceive()` returns the raw observation verbatim,
so every place the pipeline would show F "the features" (the K-step inverse/forward
window transcripts, the contrastive-FD candidates, the belief-proposer's reflective
dataset, and the planning-eval state/goal blocks) instead shows the RAW GRID that the
learned perception would otherwise have processed.

Keeping the ablation inside the perception slot -- rather than routing around it -- is
what makes it a clean single-variable ablation: the windows, the scorers, the decoys,
the prompts and the artifact layout are byte-identical to the learned-P run; only P's
content and the component selector change.

Contract (same as any learned P): perceive(observation_history) -> str, never raises.
"""


def perceive(observation_history: list[str]) -> str:
    obs = observation_history[-1] if observation_history else ""
    return obs if isinstance(obs, str) else str(obs)
