# Specs for Mixed Explore


The interface should be similar to explore.py and offline.py, using a config.yaml file, executing experiments and storing logs for analysis later.

We want to implement an algorithm similar to two step explore. A sketch is as follows - \

Maintain:
1. beliefs (made up of <world_knowledge> and <policy>)
4. set of cumulative trajectories (experience) collected so far
2. a set of questions and answers about how the world works
3. a set of critical moments (states) and the action that should be taken in this moment to achive some specific goal.

Initial:
1. Collect experience with empty beliefs

Loop:
1. Learn beliefs
2. Collect experience (using experiments)

Learn beliefs:
1. From newly collected trajectory come up with questions and moments:
    i. come up with new q-a pairs add to question set by prompting the llm. the questions should be specific and the answers should be yes/no so that they can be easily checked. the answers should be based on direct evidence observed in the trajectories. do no come up with questions and answers when data is ambigious.
    ii. come up with new critical moments to add to the set. record the state corresponding to the moment, a goal and an action that is required to achieve that goal. and make sure that the prescribed action is definately required for the goal. do no come up with moments when data is ambigious.
2. search for beliefs that score well on questions and moments:
    i. loop for num_belief_improvement number of steps.
        a. score the current beliefs on the set of questions and critical momements
        b. prompt the llm to improve beliefs based on results
3. based on waht we know and dont know so far, come up with a set of experiments to perform for the next experience collection step (similar to 2step explore)


Make sure to have hyperparameters to control 
1. number of initial trajectories to collect
2. number of trajectories to colelct in each iteraction
3. num_belief_improvement.

