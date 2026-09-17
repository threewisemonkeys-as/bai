# Probing Analysis Plan

## Forward Prediction Evaluation

We want to evaluate the learned dynamics and perception module from NLWM in different ways. 
One way we can do this is by probing their ability for forward prediction. 
Attempting this requires making decision about what representation to evaluate the forward prediction in.
There are possible paths as forward prediction can be performed either in the raw state X space (predicting X_t+1 from P(X_t-l:t)) or the abstracted P(X) space (predicting P(X_t+1) from P(X_t-k:t)).
We will evaluate on both representation.
We can generalise our evaluation to multi-step prediction where given X_t-k:t, we evaluation predicting either X_t+1:t+n or X_t+n directly. 
Given a particular (D, P), we can evaluate it in these regimes by prompting the LM with the appropriate information and checking the prediction.
For each environment and learning iteration we can compute statistics such as the mean forward prediction performance (using exact match if predicting P(X) or match accuracy if predicting in X) at n lookahead.
For each environment we can plot how this perforance varies with n.
We can also plot one line each for each N of how this FD-N (y-axis) varies with FD-1 (x-axis) (or maybe this could be seperate point charts for each N)
For data, we an use the held out validation transitions from the human data that we already have (Q1: would we need to modify these for this evaluation?)


## Reconstrcution Probe

To evaluate the amount of information captured by the perception module, we can observe how well we are able to recreate the raw state X from the encoded state P(X). 
We can do this by prompting the LM with P(X) and asking it to generate X given incontext examples. We can compute statistics such as mean over the accuracy of the prediction across states.
We can then present these statistics for the P that was best at different points in the learning process, resulting in plots showing show how information content changes across the learning process and across different environments.
