# Manual audit validation sample

This directory contains a manually reviewed, equal-stratified random sample of
360 student responses from the primary experiment.

## Sampling

The sample includes 10 responses from each of the 36 experiment response JSON
files (3 experimental runs × 2 students × 6 conditions), using random seed
20260915.

## Labels

Each record contains:
- the question, reference answer, and student response;
- all three independent GPT-5.6 Luna correctness-audit labels; and
- a manual correctness label.

## Audit agreement

Across 1,080 manual-versus-audit label comparisons (360 records × 3 audit
replicates), 1,018 labels agreed (94.26%) and 62 disagreed (5.74%).

Agreement breakdown:
- both CORRECT: 606
- both INCORRECT: 412
- both UNCERTAIN: 1
- manual INCORRECT / audit UNCERTAIN: 1
- manual UNCERTAIN / audit INCORRECT: 1
To remain consistent with evaluation protocol, uncertain judgements were considered incorrect for these.

Disagreement breakdown:
- manual CORRECT / audit INCORRECT: 34
- manual INCORRECT / audit CORRECT: 25
