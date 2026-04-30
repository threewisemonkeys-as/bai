# stepwise_eb_learn regression: apr28 (0e3569d) vs apr29 (HEAD + uncommitted)

Date: 2026-04-29
Author: investigation log

## TL;DR

apr29 runs of `stepwise_eb_learn.py` on `MiniHack-Quest-Easy-v0` solve **0/16 episodes** across 3 repeats. apr28 runs on commit `0e3569d` solve **4/10 episodes** across 4 different runs. Same command, same model (`google/gemini-2.5-flash`), same env config.

The regression is caused by **uncommitted local edits** to the improvement loop architecture (multi-turn conversational → single-shot per outer turn) plus sampling/image changes. The single-shot perception/QA loop produces self-contradictory perception modules that misclassify `}` (lava) as `open_doors` and `-`/`|` (walls) as `hazards`, causing the agent to walk east into lava and die in ~5 steps every episode.

## Runs analyzed

| Run | Code | Episodes | Solved | Solve rate |
|---|---|---|---|---|
| apr28-v1_minihack-150830 | 0e3569d | 2 | 2 | 100% |
| apr28-v1_minihack-162052 | 0e3569d | 5 | 1 | 20% |
| apr28-v1_minihack_w_policy | 0e3569d | 2 | 1 | 50% |
| apr28-v1_minihack_w_qtrimming | 0e3569d | 1 | 0 | 0% |
| apr29-rep000 | HEAD + local edits | 10 | 0 | 0% |
| apr29-rep001 | HEAD + local edits | 3 | 0 | 0% |
| apr29-rep002 | HEAD + local edits | 3 | 0 | 0% |
| **apr28 total** | | **10** | **4** | **40%** |
| **apr29 total** | | **16** | **0** | **0%** |

apr28 is variable but lands wins. apr29 is uniformly zero across 3 independent repeats.

## What changed between apr28 and apr29

`git diff 0e3569d HEAD` only shows the temp-handling refactor (commit `1953b1c`). The behaviorally significant changes are in the **working tree** (uncommitted) on `stepwise_eb_learn.py` and `stepwise_b_learn_improve.py`.

Confirmed via `diff <(git show 0e3569d:stepwise_eb_learn.py) stepwise_eb_learn.py`:

### 1. Track 1b perception loop: multi-turn → single-shot

OLD (`0e3569d`):
```python
perception_conv_1b: list[dict] = []
for turn in range(max_perception_iters):
    message = perception_from_analysis_prompt if turn == 0 else build_perception_followup_message(...)
    _, perception, turn_cost, perception_conv_1b, response_text, _ = asyncio.run(
        _improve_with_perception_validation_conversational(
            ...,
            conversation_history=perception_conv_1b,   # ← accumulating
            ...
        )
    )
```

NEW (HEAD + local edits):
```python
for turn in range(max_perception_iters):
    message = build_perception_iter_prompt(   # ← new builder
        beliefs=..., default_knowledge=...,
        original_perception=original_perception_1b,
        original_obs_section=original_obs_section_1b,
        perception_analysis=original_perception_analysis_1b,
        current_perception=perception,
        current_obs_section=current_obs_section_1b,
        current_turn=turn + 1, max_turns=max_perception_iters,
        ...
    )
    # Single-shot per outer turn: pass empty history every time.
    _, perception, turn_cost, _, response_text, _ = asyncio.run(
        _improve_with_perception_validation_conversational(
            ...,
            conversation_history=[],   # ← empty
            ...
        )
    )
```

The old prompt was a 2-message protocol per turn (assistant proposes → user followup). The new prompt is a self-contained snapshot showing `ORIGINAL` and `CURRENT` perception module + I/O, with explicit "turn N/M" counter.

### 2. Track 2 QA loop: multi-turn → single-shot

Same architectural change. `qa_conversation` accumulator removed; each call passes `conversation_history=[]` and uses `build_qa_iter_prompt` which presents `ORIGINAL BELIEFS` + `ORIGINAL QA FEEDBACK` + `CURRENT BELIEFS` + `CURRENT QA FEEDBACK` blocks side-by-side.

### 3. Sampling: even-spaced (≤3) → middle+latest (≤2)

`_sample_observations_from_buffer` (even-spaced across the trajectory, up to `num_sample_obs=3`) was replaced by `_middle_last_observations_from_buffer` (just the middle and the latest, max 2). This halves the perception evidence per improve step.

### 4. `post_image_only=True` for critical transitions

When `critical_transitions_enabled` is true (it is in this config), the perception loop now attaches **only the after-action image** for each sampled critical transition. The model loses the before-action visual context.

### 5. `_strip_raw_state_text` → `_strip_raw_pre_state_text`

The post-state text-stripping branch was removed; placeholder for hidden raw text changed from "(see attached image)" to "(raw pre-state observation hidden)". Mostly cosmetic but changes the prompt content the model sees.

### 6. Temp handling refactor (committed in `1953b1c`)

`override_temperature(config, eb_config.explore_temp)` context manager removed; `set_meta_temperature(eb_config.explore_temp)` global setter added. The agent's BALROG client now reads `config.client.generate_kwargs.temperature` directly; meta calls (`_llm_call`, `_llm_call_conversational`) read `_META_TEMPERATURE`.

**Effective temperatures are unchanged**: both runs use 1.0 for both agent rollout and meta improve calls.
- apr28: `client.generate_kwargs.temperature=0.0` in saved config, but `override_temperature(config, 1.0)` overrode it during episodes for both rollout and meta.
- apr29: `client.generate_kwargs.temperature=1.0` in saved config (user updated), `set_meta_temperature(1.0)` for meta.

This commit is **not** the cause of the regression — the temperature plumbing is functionally equivalent.

## Mechanism of the regression: how the agent collapses

### Evidence: perception modules diverge at episode 0

**apr28-150830 ep0 final `perception.py`** (this run solved both episodes):
```python
adjacent_tiles = {... '}': 'item', ...}
impassable_or_background = ['-', '|', ' ', '#', '.']
# detects "Burned by molten lava." via regex
```
`}` is "item" (loosely wrong but not actively misleading). `-`/`|` are correctly walls. The agent later identifies wand-of-cold and uses it to freeze lava.

**apr29-rep000 ep0 final `perception.py`**:
```python
'open_doors': [],
'hazards': [],
elif char == '}': # Re-add processing for open doors.
    map_entities['open_doors'].append((x, y))
    map_entities['hazards'].append((x, y))   # ALSO hazard — self-contradictory
elif char == '-':
    map_entities['hazards'].append((x, y))   # walls labeled as hazards
```
`}` is dual-tagged (open door AND hazard). Walls are tagged as hazards.

**apr29-rep000 ep4 input `perception.py`** (after 4 episodes of improvement):
```python
elif char == '}': # Re-classify '}' as an open door based on QA feedback n=5
    map_entities['open_doors'].append(current_coord)   # hazard tag REMOVED
elif char == '-': # walls/hazards
elif char == '|': # walls/hazards
```
The single-shot QA loop has resolved the dual-tag — and resolved it the wrong way. `}` is now ONLY open door. Walls remain tagged as hazards.

The companion `input_beliefs.txt` for apr29 ep4 says:
> "`-`, `|`, and `---` represent lethal environmental hazards or impassable walls. All confirmed lethal hazards (`-`, `---`) cause immediate death upon movement onto their square (e.g., 'Burned by molten lava.')."

This is internally inconsistent: walls labeled as lava-hazards, while the actual lava `}` is labeled as a safe open door.

### Result

From episode 4 onward, every apr29-rep000 episode terminates after exactly 5 steps with `action=east` repeatedly and `episode_return=0`. The agent reads "open_doors at (28, …)" from perception, plans to walk east through the door, and dies in lava.

## Why single-shot is qualitatively worse than multi-turn here

Multi-turn `_llm_call_conversational` accumulates the model's reasoning across iterations: "I tried X, the I/O changed Y, so now I will Z." This reasoning is implicit context that constrains future edits.

Single-shot strips that. Each outer turn the model sees only ORIGINAL + CURRENT + a static `perception_analysis` blob and is asked to rewrite from scratch with "compare current to original, keep what helped, fix what regressed." Failure modes:
- **Re-derivation drift**: the model picks a different mental model each turn, can't remember why a prior decision was made.
- **Over-reaction to single QA feedback**: when QA says "is `}` an open door? YES" (probably from a transient experiment after the wand of cold froze the lava and the agent traversed it once), the model rewrites the perception to make `}` an open door — and drops the hazard tag in the process. Multi-turn would have surfaced earlier turns' "but it killed me before" reasoning.
- **No "defensive" memory**: the dual-tag (`}` is open_door AND hazard) was a sane in-progress state; single-shot collapses it to one classification per turn.

## Cost analysis

Per-improve-step cost is similar between modes:

| Run | Improve steps | Avg cost / improve step |
|---|---|---|
| apr28-150830 | 19 | $0.179 |
| apr28-162052 | (varies) | $0.282 |
| apr28-w_policy | (varies) | $0.130 |
| apr28-w_qtrimming | (varies) | $0.142 |
| apr29-rep000 | 39 | $0.176 |
| apr29-rep001 | (varies) | $0.225 |
| apr29-rep002 | (varies) | $0.270 |

Single-shot is **not** meaningfully cheaper despite skipping conversation history — the bigger prompt (ORIGINAL + CURRENT blocks) offsets the savings.

apr29 total cost is higher ($6.7–7.8 vs $3.7–5.2) only because more episodes terminate quickly and trigger more improve cycles. Per-episode cost is comparable.

## Recommendation

Bisect the regression by reverting changes in suspicion order:

1. **Restore multi-turn `conversation_history` for Tracks 1b and 2** in `stepwise_eb_learn.py` (lines ~1165 and ~1337). Highest suspicion: all apr28 wins came from multi-turn.
2. **Restore `_sample_observations_from_buffer` (even-spaced)** as the perception sampler. Middle+latest halves the evidence per turn.
3. **Disable `post_image_only`** so perception sees both pre/post images for critical transitions.

If reverting (1) alone restores apr28-level solve rates, the perception/QA loop should be reconsidered. A "structured multi-turn" pattern (single-shot proposal + one followup re-eval) might be a useful middle ground if you want bounded turns without losing cross-turn reasoning.

If solve rates remain zero after (1), proceed through (2) and (3).

## File evidence pointers

- Code diff: `diff <(git show 0e3569d:stepwise_eb_learn.py) stepwise_eb_learn.py` — 420 differing lines
- New prompt builder definition: `stepwise_b_learn_improve.py:228` (`build_perception_iter_prompt`), `:591` (`build_qa_iter_prompt`)
- Old prompt builders (still present, no longer called from eb path): `build_perception_followup_message`, `build_perception_with_analysis_prompt`, `build_qa_followup_message`
- apr28 winning perception: `logs/results/apr28/gemini/v1_minihack/20260428-150830/.../episode_0/perception.py:117`
- apr29 broken perception: `logs/dev/apr29/20260429-194038/eb_learn__minihack__gemini-2p5-flash__repeat_000/.../episode_0/perception.py:86-89`, `episode_4/input_perception.py:115`
- apr29 broken belief: `episode_4/input_beliefs.txt` line 4
