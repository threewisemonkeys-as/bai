# Mixed Explore Algorithm

## Objects

- **Beliefs** `B`: A text document split into `<world_knowledge>` (environment mechanics, rules) and `<policy>` (action strategies for specific situations). Capped at ~8 bullet points per section.

- **Perception** `P`: A Python function `perceive(observation_text) -> str` that processes raw environment observations into a structured summary. Optional.

- **Q&A Pairs** `Q = {(question, answer, evidence, source_step)}`: A cumulative set of yes/no questions about world mechanics, each grounded in trajectory evidence. Questions are about general properties ("Does X cause Y?"), not episode-specific events.

- **Critical Moments** `M = {(state, goal, action, evidence, raw_observation, source_step)}`: A cumulative set of state-action examples where the correct action is known from trajectory evidence. Each `action` is a single command. `raw_observation` stores the full environment output at that step (optional, for scoring fidelity).

- **Experiments** `E = [e_1, ..., e_k]`: Short (1-3 sentence) hypotheses to test in the next round of rollouts. Each is a conditional strategy ("If you see X, try Y").

- **Trajectories** `T`: CSV logs of agent episodes. Each row has (Step, Action, Reasoning, Observation, Reward, Done).

- **Episode Summaries** `S`: LLM-generated natural language summaries of trajectories, used as context for belief/perception improvement.

- **Default Knowledge** `K`: Static instructions extracted from the agent's base prompt. Read-only context passed to all LLM calls.

## Parameters

| Symbol | Config key | Description |
|--------|-----------|-------------|
| `N` | `num_steps` | Number of outer loop iterations |
| `N_init` | `num_initial_trajectories` | Rollouts in step 1 (no experiments) |
| `N_traj` | `num_trajectories_per_iteration` | Rollouts per step (steps 2+) |
| `N_inner` | `num_belief_improvement` | Max scoring/improvement iterations per step |
| `N_exp` | `num_experiments` | Experiments to generate per step |
| `tau` | `explore_temp` | Temperature for experiment rollouts |

## Toggles

| Flag | Effect when off |
|------|----------------|
| `use_qa_pairs` | Skip Q&A extraction and scoring |
| `use_moments` | Skip moment extraction and scoring |
| `use_raw_observations` | Use summary state descriptions instead of full raw observations in moment scoring/improvement |
| `improve_mode` | `"both"` = beliefs + perception, `"beliefs"` = beliefs only, `"perception"` = perception only |

When both `use_qa_pairs` and `use_moments` are off, the inner loop falls back to simple belief improvement directly from episode summaries (no scoring step).

## Algorithm

```
Initialize:
  B <- load from file or empty
  P <- load from file or empty
  Q <- {}
  M <- {}
  E <- []

For step t = 1, ..., N:

  ── PHASE 1: COLLECT ──────────────────────────────────────

  If t == 1 (initial):
    Collect N_init baseline rollouts using (B, P, E=[])
  Else:
    Collect N_traj rollouts using (B, P, E) at temperature tau

  Result: trajectories T_t

  ── PHASE 2: LEARN ────────────────────────────────────────

  2a. Process trajectories (in parallel):
      For each trajectory in T_t, concurrently:
        - Summarize -> episode summary s_i
        - Extract Q&A pairs and critical moments from raw CSV
          (attach raw observations to moments by step number)
      S_t <- combine all s_i

  2b. Consolidate structured knowledge:
      Review existing Q, M against new evidence (batched, parallel):
        - Remove items contradicted by new trajectories
      Deduplicate new extractions against surviving Q, M:
        - Filter poorly grounded or speculative items
      Q <- Q \ {contradicted} + {new consolidated Q&A}
      M <- M \ {contradicted} + {new consolidated moments}

  2c. Inner loop — score and improve beliefs:
      prev_scores <- (-1, -1)
      For i = 1, ..., N_inner:

        Score:
          Present (B, K) to LLM with:
            - Q&A pairs: predict YES/NO for each question
            - Moments: predict action given state
              (uses raw observation + P output if use_raw_observations)
          qa_score  <- fraction of correct Q&A predictions
          mom_score <- fraction of correct action predictions
            (normalized substring match)

        Early stop:
          If (qa_score, mom_score) == prev_scores: break
          prev_scores <- (qa_score, mom_score)

        Improve beliefs:
          Present (B, K, S_t, score details) to LLM
          For incorrect moments: show raw observation + P output
          B <- LLM-revised beliefs

  2d. Improve perception (if enabled):
      Sample observation/perception examples from T_t
      Present (P, B, S_t, examples, K) to LLM
      P_new <- LLM-generated Python function
      Validate P_new (compile, exec, runtime test)
        Up to 3 retries; fall back to P on failure
      P <- P_new

  2e. Generate experiments:
      Present (B, Q, M, score gaps, S_t, K) to LLM
      E <- N_exp new experiment hypotheses targeting knowledge gaps

  ── Save step artifacts ───────────────────────────────────
  Persist: B, P, Q, M, E, S_t, scoring history, per-trajectory extractions
```

## Key Design Choices

**Structured knowledge as a scoring mechanism.** Q&A pairs and critical moments serve as an explicit test set for beliefs. Rather than only improving beliefs from narrative summaries (which is subjective), the inner loop scores beliefs against concrete predictions and iterates. This provides a measurable signal for whether beliefs are improving.

**Cumulative knowledge with garbage collection.** Q and M grow across steps, but the consolidation step reviews all existing items each step and removes those contradicted by new evidence. This prevents stale or incorrect knowledge from persisting indefinitely. Review is batched (batch size 20) and parallelized to handle large sets.

**Raw observations for scoring fidelity.** When scoring moments, the LLM sees the same observation the agent would see at runtime (plus perception output), not a summary. This tests whether beliefs produce correct actions under realistic conditions.

**Perception as code.** The perception module is executable Python, validated before acceptance. This keeps it testable and deterministic, unlike natural language processing instructions.

**Experiments as exploration strategy.** Instead of random rollouts, the algorithm generates targeted experiments from identified knowledge gaps. This focuses data collection on areas where beliefs are weakest.
