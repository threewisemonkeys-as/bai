# Algorithm sketch for B Learning

We want to learn a useful policy of behaviour from interactions. This will be similar to b_learning.md but having a learning step after each environment step

maitain:
1. a perception module made up of python code (P)
2. beliefs (made up of <world_knowledge> (d) and <policy> (pi))
3. sequence of states and actions observed so far
4. a set of binary questions and answers about how the world works (Q)
5. a set of critical moments made up of the state (raw observation), a goal description, a set of desired actions at that state, a set of undesired actions at that state (C = {(s, g, a+, a-)})


loop (for n_environment_steps):
1. perform a step in the environment (where agent is conditioned on some experiment as the goal)
2. artifact update step:
    1. given the sequence of steps so far, update Q by coming up with new q-a pairs to add to question set and identifying inaccurate q-a pairs to remove by prompting the llm. the questions should be specific and the answers should be yes/no so that they can be easily checked. the answers should be based on direct evidence observed in the trajectories. do no come up with questions and answers when data is ambigious
    2. given the sequence of steps so far, update C by comning up new critical moments, and identifying inaccurate critical moments to remove by prompting the llm.
3. improve loop (for num_improve_iteration):
    1. current B_t = (P_t, d_t, pi_t) (where we start at t=0 with B from last main iteration)
    2. steps so far based improvement:
        1. B_t+1 = improve(B_t) where improve is a prompt to the llm that presents it with the following and asks it to come up with improved perception code (P_t+1), policy (pi_t+1) and world knowledge (d_t+1)
            1. P_t: code of perception module
            2. s, P_t(s) for s in uniformly sampled K states (s) from sequence of steps so far
            2. pi: policy
            3. d: world knowledge
            4. t: sequence of steps so far
    3. question based improvement:
        1. intialise set of feedback R = {}
        2. for q in Q:
            1. get predicted answer a_hat for the question by prompting LLM with d_t and question part of q
            2. prompt model to get feedback (r) on the answer given the question and add feedback (r) to R
        3. B_t+1 = improve(B_t, r) where improve is a prompt to the llm that presents it with the following and asks it to come up with improved perception code (P_t+1), policy (pi_t+1) and world knowledge (d_t+1)
            1. P_t: code of perception module
            2. pi: policy
            3. d: world knowledge
            4. for each (q, r) pair in Q.R (set of pairs of questions and their results)
                1. q: question
                2. a: answer
                3. a_hat: predicted action (with original reasoning) on that state
                4. r: feedback on correctness of a_hat
    4. critical moment based improvement: 
        1. intialise set of feedback R = {}
        2. for c in C:
            1. get predicted action by passing through forward pipeline: a_hat = f(B_t, c)
            2. prompt model to get feedback (r) on a_hat given the context of c (i.e. was the action correct / incorrect / inconclusive and why not)
            3. if feedback (r) is not inconclusive, then add r to R
        3. B_t+1 = improve(B_t, r) where improve is a prompt to the llm that presents it with the following and asks it to come up with improved perception code (P_t+1), policy (pi_t+1) and world knowledge (d_t+1)
            1. P_t: code of perception module
            2. pi: policy
            3. d: world knowledge
            4. for each (c, r) pair in C.R (set of pairs of critical moments and their results)
                1. s: raw observation
                2. P_t(s): output of perception module on that raw observation
                3. g: goal string
                4. a_hat: predicted action (with original reasoning) on that state
                5. r: feedback on correctness of a_hat
4. generate new experiment: prompt the llm with latest B (P, B, pi), current state (s), output of perception module on current state (P(s)) and current goal to come up with a new experiment (which can be the same as the current goal) to be used as the goal for the next step

Forward model:
- This is similar to the model in explore.py. Run the perception code on the current raw observation, pass the output of this, the raw perception itself, the policy and world knowledge in a prompt to ask the model for reasoning and an action.

Things to keep in mind:
- Whereever we refer to sequence of steps so far, we want to present a sequence of (state s, perception output P(s) and reaosning + action taken by agent a). this sequence can be trimmed to only include the longest suffix that fits into the context
- The interface should be similar to b_learn.py, using a config.yaml file, executing experiments and storing logs for analysis later.
- none of the prompts should ever reference that we are playing a game or the specific game we are playing. they just only refer to interacting with an environment.
- make sure all hyperparameters required are correctly present in the config
