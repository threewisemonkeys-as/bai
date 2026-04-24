we are working on stepwise_eb_learn.py. one of the parts of the pipeline we want to work on is question selection. currently we do this by trimming available questions through an llm prompt. we want to try different metrics for scoring questions. this will help us prune unasnwered questions as well as select them for experiment formation. 

approach 1: b difference scoring. use this in place of the current trim step. after computeing the new qa pairs, score the unanswered questions as follows, then drop the low scoring ones. in the next step. scoring procedure:
given:
    1. current belief B (world knowledge, policy)
    2. list of unanswered questions Q = [q_i]
for each q_i, compute a score as follows:
    1. prompt llm to come up with updated B (called B_pos) given set of answered questions which includes q_i with answer yes. prompt llm with B_pos for answers to remaining unanswered questions to get a vector of answers A_pos.
    2. do the same as above but with answer to q_i as no, producing B_neg, and subsequently A_neg
    3. assign the distance between A_pos and A_neg (|A_pos - A_neg|) as the score for q_i

