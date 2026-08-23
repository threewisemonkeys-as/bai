# Sketch for simple agent baseline

Our current proposed approach is stepwise_eb_learn.py. This incorporates maintaining questions about how the world works, and optimising our beliefs to answer those questions.

We want to compare this approach with a simple llm agent. this agent is provided the default knowledge of the envrionment (that we use the current setup). the agent the proceeds in an conversational approach, where past messages to the llm and its responses are preserved in memory. then in each agent step it is provided with the raw state of the environment and asked to output an action. this action is parsed and taken in the environment and the observation is used to prompt the agent for the next action.

We an episode is complete, we show the completion observation to the agent, reset the environment and start another epsiode, showing the llm the first state of this new episode and asking it for the next action in the same prompt as showing it the completion scrren. note that we do not reset the agent's conversational memory across episode boundaries.

Things to keep in mind:
- The interface uses Hydra config (config.eval.evolve), executing experiments and storing logs for analysis later.
- None of the prompts should ever reference that we are playing a game or the specific game we are playing. They only refer to interacting with an environment.
- Per-step progress is flushed to disk (step_log.json with phase markers: started/acting/extracting/improving/complete) so a live visualizer (viz/ module) can track progress. make changes to the viz module to support this mode too
- Resume support: finds last completed episode, restores beliefs, perception, QA, moments, experiments, and global step count.
