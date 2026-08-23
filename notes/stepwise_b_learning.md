# Algorithm sketch for B Learning

We want to learn a useful policy of behaviour from interactions, with a learning step after each environment step. The trajectory buffer and past experiments carry across episodes.

maintain:
1. a perception module made up of python code (P)
2. beliefs (made up of <world_knowledge> (d) and <policy> (pi))
3. trajectory buffer — sequence of step dicts {raw obs, result raw obs, perception output, reasoning, action, reward}, with episode boundary markers between episodes
4. a set of binary questions and answers about how the world works (Q)
5. a set of critical moments made up of the state (raw observation), a goal description, a set of desired actions at that state, a set of undesired actions at that state (C = {(s, g, a+, a-)})
6. a list of experiments (active + candidates) and a list of past experiments already tried


outer loop (while global_steps_used < n_environment_steps):
    run an episode (capped at remaining steps). on death/respawn, prepend message to first observation of next episode.

    inner loop (for each step in the episode):
    1. experiment generation (every experiment_interval steps, before agent acts):
        - prompt the llm with latest B (P, d, pi), trajectory buffer history, current state (s), P(s), current experiment, past experiments, known Q and C
        - llm returns either "null" (keep current experiment) or N new experiments (1-3 sentence actionable hypotheses)
        - first experiment becomes active; inject into agent as experiment_goal

    2. perform a step in the environment (agent conditioned on active experiment as the goal)
        - save raw obs BEFORE applying perception (perception modifies obs in-place)
        - append step to trajectory buffer
        - if done, also append terminal state entry (action=None)

    3. artifact update step (every artifact_update_interval steps, and on episode end):
        1. format trajectory buffer as steps context (trimmed to max_steps_context_chars)
        2. extract new Q-A pairs and critical moments from steps context via LLM. questions should be specific yes/no, answers based on direct evidence. do not extract when data is ambiguous.
        3. attach raw observations from buffer to new moments
        4. consolidate new + existing Q and C via LLM (identify duplicates and inaccurate entries to remove)
        5. cap totals (drop oldest if over max_total_qa_pairs / max_total_moments)

    4. improve loop (every improve_interval steps, and on episode end):
        dispatches to conversational or independent mode.

        conversational mode (beliefs/perception flow sequentially through tracks):

            track 1b — steps-based beliefs improvement (1 turn):
                - prompt LLM with default knowledge, current beliefs, step sequence
                - LLM returns updated <world_knowledge> and <policy>, plus a <perception_analysis> of how the perception module could be improved

            track 1c — perception improvement from analysis (up to max_perception_iterations turns):
                - initial prompt: beliefs, current P code, sampled perception input/output examples (K evenly spaced from buffer), perception_analysis from track 1b
                - LLM returns updated P code + <status>CONTINUE or SUBMIT</status>
                - on CONTINUE: re-run updated P on same sample observations, show before/after comparison, iterate
                - on SUBMIT or max iterations: stop

            track 2 — question based improvement (up to max_qa_iterations turns):
                1. initial evaluation:
                    - for q in Q: get predicted answer a_hat by prompting LLM with d and question part of q
                    - for each prediction: prompt model to get feedback (r) — verdict is CORRECT / INCORRECT / INCONCLUSIVE
                2. if any INCORRECT:
                    - present all feedback to LLM along with current B, P execution report, steps context
                    - B_t+1 = improve(B_t, R) -> updated beliefs and perception + CONTINUE/SUBMIT
                    - on CONTINUE: re-evaluate all QA pairs with updated beliefs, show delta and remaining failures, iterate
                    - on SUBMIT or max iterations: stop

            track 3 — critical moment based improvement (up to max_moments_iterations turns):
                1. initial evaluation:
                    - for c in C: get predicted action by passing through forward pipeline: a_hat = f(B_t, c)
                    - prompt model to get feedback (r) on a_hat given the context of c (verdict: CORRECT / INCORRECT / INCONCLUSIVE)
                2. if any INCORRECT:
                    - present all feedback to LLM along with current B, P execution report, steps context
                    - B_t+1 = improve(B_t, R) -> updated beliefs and perception + CONTINUE/SUBMIT
                    - on CONTINUE: re-evaluate all moments with updated beliefs/perception, show delta and remaining failures, iterate
                    - on SUBMIT or max iterations: stop

        independent mode (legacy — max_steps_iterations outer iterations, each runs all 3 tracks as single-turn LLM calls):
            1. improve_from_steps(B, steps_context, sample_obs) -> updated B
            2. qa_forward_pass -> qa_get_feedback -> if incorrect: improve_from_qa_feedback_with_steps -> updated B
            3. forward_pass -> get_feedback -> if incorrect: improve_from_feedback_with_steps -> updated B
            after each track, if P changed, rebuild steps_context

    5. post-improve:
        - reload perception function from updated code
        - re-apply updated perception to current obs for agent's next step
        - if perception changed, rebuild all buffered observations with new perception
        - inject updated beliefs into agent's instruction prompt


Forward model:
- This is similar to the model in explore.py. Run the perception code on the current raw observation, pass the output of this, the raw perception itself, the policy and world knowledge in a prompt to ask the model for reasoning and an action.

Things to keep in mind:
- Wherever we refer to sequence of steps so far, we present a sequence of (state s, auxiliary obs, perception output P(s), reasoning + action taken by agent a, reward). This sequence is trimmed to the longest suffix that fits into max_steps_context_chars. The last action step also includes the resulting state s_t.
- The interface uses Hydra config (config.eval.evolve), executing experiments and storing logs for analysis later.
- None of the prompts should ever reference that we are playing a game or the specific game we are playing. They only refer to interacting with an environment.
- All hyperparameters are in StepwiseBLearnConfig, populated from config.eval.evolve: n_environment_steps, max_steps_iterations, max_perception_iterations, max_qa_iterations, max_moments_iterations, max_moments_per_forward, max_qa_per_forward, max_total_moments, max_total_qa_pairs, num_experiments, num_sample_obs, explore_temp, artifact_update_interval, improve_interval, experiment_interval, max_steps_context_chars, conversational_improve.
- Perception validation: all improve functions execute proposed perception code on sample observations to verify it runs without errors before accepting it.
- Per-step progress is flushed to disk (step_log.json with phase markers: started/acting/extracting/improving/complete) so a live visualizer can track progress.
- Resume support: finds last completed episode, restores beliefs, perception, QA, moments, experiments, and global step count.
