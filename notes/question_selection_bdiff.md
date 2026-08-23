# Question Selection With B-Difference Scoring

We are working on `stepwise_eb_learn.py`. The current pipeline keeps the Q&A
set bounded by running an LLM trim prompt after question generation and before
experiment formulation. The goal of this improvement is to make unanswered
question pruning and experiment selection more principled by scoring questions
according to how much answering them would change the agent's implied beliefs
about other unanswered questions.

## Motivation

The current trim step asks an LLM to directly choose which questions are useful.
That can work, but it gives us little structure for comparing questions or
debugging why a question was kept. B-difference scoring estimates the value of
an unanswered question by asking: if this question were answered YES versus NO,
how different would the agent's downstream predictions become?

A high-scoring question is one whose answer changes many other predictions. This
is a useful proxy for structural information gain.

## Proposed Scoring Method

Given:

1. Current beliefs `B`
2. Current Q&A list `Q`
3. Unanswered candidate question `q_i`
4. Remaining unanswered questions `Q_rest`

For each candidate `q_i`:

1. Hypothesize that `q_i` has answer `YES`.
2. Ask the LLM to update current beliefs into `B_pos`.
3. Ask the LLM to answer `Q_rest` using `B_pos`, producing answer vector
   `A_pos`.
4. Hypothesize that `q_i` has answer `NO`.
5. Ask the LLM to update current beliefs into `B_neg`.
6. Ask the LLM to answer `Q_rest` using `B_neg`, producing answer vector
   `A_neg`.
7. Score `q_i` by the distance between `A_pos` and `A_neg`.

The simplest distance metric is disagreement rate:

```text
score(q_i) = count_j(A_pos[j] != A_neg[j]) / len(Q_rest)
```

`UNKNOWN` or unparsable answers should either be ignored in the denominator or
counted with a smaller weight. Start with ignoring positions where either branch
is unknown, and log the effective denominator.

## Integration Design

Do not implement this as another special case directly inside
`stepwise_eb_learn.py`. Add a question selection layer with multiple
implementations.

Create a new module:

```text
stepwise_eb_question_selection.py
```

Core interface:

```python
async def select_qa_pairs(
    config,
    beliefs: str,
    qa_pairs: list[EBQAPair],
    default_knowledge: str,
    max_answered_qa_pairs: int,
    max_unanswered_qa_pairs: int,
    mode: str,
) -> tuple[list[EBQAPair], float, dict]:
    ...
```

Supported modes:

```text
llm_trim       Current behavior using trim_qa_pairs.
bdiff          Score unanswered questions with B-difference and keep top-k.
bdiff_then_llm Use B-diff to rank unanswered questions, then use an LLM for
               final tie-breaking or answered-question pruning.
none           No pruning; useful for ablations.
```

Add a score record:

```python
@dataclass
class QuestionScore:
    question_index: int
    question: str
    score: float
    pos_beliefs: str | None
    neg_beliefs: str | None
    pos_answers: list[bool | None]
    neg_answers: list[bool | None]
    rationale: str | None
    cost: float
```

Add scoring function:

```python
async def score_unanswered_questions_bdiff(
    config,
    beliefs: str,
    qa_pairs: list[EBQAPair],
    default_knowledge: str,
    max_questions_to_score: int | None = None,
    max_downstream_questions: int | None = None,
) -> tuple[list[QuestionScore], float, dict]:
    ...
```

## Changes In `stepwise_eb_learn.py`

The current pipeline has this shape:

```text
generate_questions_from_steps
trim_qa_pairs
formulate_experiment_from_question
```

Replace the direct trim call with:

```python
qa_pairs, selection_cost, question_selection_log = asyncio.run(
    select_qa_pairs(
        config=config,
        beliefs=beliefs,
        qa_pairs=qa_pairs,
        default_knowledge=default_knowledge,
        max_answered_qa_pairs=eb_config.max_answered_qa_pairs,
        max_unanswered_qa_pairs=eb_config.max_unanswered_qa_pairs,
        mode=eb_config.question_selection_mode,
    )
)
```

Persist:

```text
question_selection_log.json
```

For compatibility with the existing viewer, the selection log can also be
written to `trim_log.json` or include a compatibility subset with the same keys
as the old trim log.

Add config fields:

```yaml
question_selection_mode: llm_trim
question_scoring:
  max_questions_to_score: 10
  max_downstream_questions: 20
  distance_metric: disagreement_rate
  keep_score_logs: true
```

`llm_trim` should remain the default until offline and live experiments show
that B-difference scoring is better.

## Cost Control

The naive implementation costs up to four LLM calls per candidate:

1. `B_pos`
2. `A_pos`
3. `B_neg`
4. `A_neg`

With ten unanswered questions, that can be forty calls per selection step. Start
with a cheaper one-call-per-candidate prompt that returns both branches:

```text
candidate q_i
YES branch: B_pos and A_pos
NO branch: B_neg and A_neg
score rationale
```

Then optionally test the stricter multi-call version if the cheap variant looks
promising.

Initial cost controls:

1. Cap candidates with `max_questions_to_score`.
2. Cap downstream questions with `max_downstream_questions`.
3. Cache scores by hash of `(beliefs, qa_pairs, candidate_question)`.
4. Run candidate scoring concurrently with `asyncio.gather`, bounded by a
   semaphore.
5. Log total selection cost separately from question generation and experiment
   formulation cost.

## Offline Evaluation On Past Runs

Use the existing run:

```text
logs/dev/apr20/2026-04-20_22-54-50_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn
```

Add a script:

```text
scripts/score_stepwise_eb_questions.py
```

Example command:

```bash
uv run python scripts/score_stepwise_eb_questions.py \
  logs/dev/apr20/2026-04-20_22-54-50_robust_cot_google_gemini-2.5-flash_stepwise_eb_learn \
  --mode bdiff \
  --max-steps 50 \
  --output analysis/question_scores_apr20.jsonl
```

For each `episode_*/step_*/` directory:

1. Read `beliefs.txt`.
2. Read `qa_pairs.json`.
3. Extract unanswered questions.
4. Run B-difference scoring.
5. Save score records, ranking, raw responses, parse status, and cost.
6. If `experiment_log.json` has `selected_question_index`, compare the
   historical selected question to the B-diff ranking.
7. If `trim_log.json` exists, compare the old retained/dropped questions against
   B-diff top-k.

Write one JSONL row per step:

```json
{
  "episode": 0,
  "step": 4,
  "global_step": 4,
  "num_qa_pairs": 19,
  "num_unanswered": 9,
  "historical_selected_question_index": 15,
  "scores": [
    {
      "question_index": 15,
      "question": "...",
      "score": 0.72,
      "effective_denominator": 7,
      "parse_ok": true,
      "cost": 0.0042
    }
  ],
  "historical_selected_rank": 2,
  "top_k_indices": [15, 8, 12, 4, 6],
  "cost": 0.031
}
```

## Offline Metrics

Do not claim policy improvement from offline scoring alone. Use offline scoring
as a diagnostic for whether B-difference ranking has useful signal.

Recommended metrics:

1. Historical selected-question rank under B-diff.
2. Overlap between B-diff top-k and the old LLM trim retained set.
3. Future answerability: among unanswered questions at step `t`, which ones
   become answered by step `t + k`?
4. Mean B-diff score of questions that later become answered versus those that
   remain unanswered.
5. Mean B-diff score of historically selected questions versus non-selected
   questions.
6. Diversity of top-k questions; check whether top questions collapse to
   paraphrases of the same concept.
7. Cost per selection step compared with the current trim prompt.
8. Ranking stability across reruns at temperature 0.

The most compelling offline result would be:

```text
Questions ranked high by B-diff are more likely to become answered later, and
B-diff keeps questions the old trim dropped that later matter.
```

## Counterfactual Experiments

Run these before a live A/B run:

1. Replay scoring only.
   Score unanswered questions at each step. Do not modify the run.

2. Counterfactual trim simulation.
   At each step, simulate keeping top `max_unanswered_qa_pairs` by B-diff.
   Compare against actual `qa_pairs.json` and historical `trim_log.json`.

3. Counterfactual experiment selection.
   Feed the experiment selector the B-diff-pruned Q&A list and ask what
   experiment it would choose. Compare with the historical experiment plan.

4. Live A/B run.
   Compare `question_selection_mode=llm_trim` against
   `question_selection_mode=bdiff` under the same task/model/seed budget.

## Risks And Follow-Ups

B-difference can reward broad abstract questions that affect many predictions
but are not testable from the current state. If this happens, combine it with a
feasibility score:

```text
final_score = bdiff_score * feasibility_score
```

The feasibility score should estimate whether the agent can plausibly answer
the question from the current state in the next few actions. Do not add this in
the first implementation. First test whether B-difference itself has signal.

## Recommended First Patch

1. Add `stepwise_eb_question_selection.py`.
2. Move the existing LLM trim behind `select_qa_pairs(..., mode="llm_trim")`.
3. Add `score_unanswered_questions_bdiff`.
4. Add `select_qa_pairs(..., mode="bdiff")`.
5. Add `scripts/score_stepwise_eb_questions.py`.
6. Add config plumbing in `StepwiseEBLearnConfig`.
7. Replace the direct `trim_qa_pairs` call in `stepwise_eb_learn.py` with the
   selector abstraction.
8. Keep `llm_trim` as the default.
9. Run the offline Apr 20 scoring experiment before enabling B-diff in live
   runs.
