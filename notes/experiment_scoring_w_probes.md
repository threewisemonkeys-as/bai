we are working on stepwise_eb_learn.py. one of the parts of the pipeline we want to work on is question generation, scoring and selection.

currently we trim the maintained set of questions at each step. we want to modify this so that we maintain all generated question. instead of the trim step, we have a much simpler de-duplication prompt that tries to ensure that questions are duplicated. all it shows the llm is a list of all currently acculumated questions and asks it which ones to drop.

then in step 1 (question generation and experiment selection), after the  the de-duplication prompt, we want to use the current trim prompt, but instead of trimming the maintained set, we use this prompt to select top K question from this large set. we will use this top-k set for experiment selection, but the large set of questions will keep being maintained for the next step (without any trimming)

on these top-k, we then want to use the scoring mechanism we have come up with (b difference scoring). when computing a_pos, and_neg from b_pos and b_neg, we want to use all the questions generated in the large set (not just top k). the idea is this should give us more coverage and accurate estimate of the distance between a_pos and a_neg.

