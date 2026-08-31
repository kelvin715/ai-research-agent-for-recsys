# Devpost story draft

## One-line description

An AI research agent that explores several model branches and improves the final combined recommender, not only the strongest model measured on its own.

## The problem

Many machine-learning agents follow a single path: change the current highest-scoring model and keep the change if that model improves. Our task exposed a weakness in this approach. LambdaRank scored only `0.601058` on its own, the lowest of our three models, but removing it caused the largest drop in the final combination: `0.000752`.

The model that looked weakest in isolation supplied the most different information. The real question was therefore not simply “Can an agent find a strong model?” It was “Can an agent discover which models are useful together?”

## What we built

We built a GPT-5.4 research agent for the KuaiRand-Pure recommendation task. It keeps a graph of measured experiments rather than one chain of edits. It may return to an earlier branch, revisit any model in the current combination, or join ideas from different branches.

A read-only live view makes that graph visible while the agent works. It shows the active phase,
execution branches, ideas reused across branches, individual and combined measurements, decisions,
and the models in the current best combination. The view is generated from the same saved records
used for audit; it does not change experiments or read labels.

Every candidate is measured both on its own and after insertion into the current model combination. The agent also receives aggregate clues about where the model helps: removal loss, ranking similarity, and performance by `tab`, user history, and item popularity. This turns the final combination into part of the research objective instead of a last-minute averaging step.

Published work and earlier experiments are stored as evidence linked to an exact model parent, code mechanism, expected result, and disproof condition. This prevents a failed implementation from incorrectly ruling out an entire idea.

The final recommendation combines three models:

- a pairwise model that learns which videos each user prefers;
- DeepFM, which learns sparse ID interactions and uses four extra engagement labels during training only;
- a LambdaRank tree model that learns different item and viewing patterns.

For each user, model scores are converted to ranks before they are combined. A small fixed table changes the three weights according to `tab`, an input field available when predictions are made.

## Why it is different

**A graph instead of one model chain.** Successful, uncertain, and failed branches remain recorded. A weak individual model can still be expanded when it contributes different errors to the final result.

**Two measures of usefulness.** The agent records both individual score change and direct change to the current model combination. Either can provide a valid reason to keep exploring a branch.

**Three different decisions.** A promising result can remain available for exploration without automatically becoming the next safe parent or replacing the deployed combination.

**Evidence connected to code.** Every research claim must map to an executable data path and a measurable outcome. Unused settings and exact repeats are rejected before a full training experiment.

**A bounded, task-specific final selector.** The system searches only small model subsets and simple weights, keeps complementary members in consideration, and uses the same combination logic for validation and submission inference.

Label separation, isolated training, hashes, repeated seeds, and complete logs provide the engineering foundation for these decisions. We treat them as safeguards, not as the core research novelty.

**Worse changes are rejected.** The final proposed combination scored `0.6060940663`, just below the saved best result of `0.6061277486`. The agent kept the better result.

## How the warm start was created

Development and demonstration were intentionally separated.

During the discovery stage, the agent continued searching across repeated sessions instead of applying the final three-small-gains early stop to every session. Verified strong solutions, negative results, and exact experiment settings were saved as prior evidence.

For the final reproducible run, GPT-5.4 received all verified prior evidence. It first retrained all three warm-start models with three seeds, checked their hashes and scores, and reconstructed the expected `0.6061277486` result. Only then could it propose new changes. This shows that the prior is executable knowledge, not a copied leaderboard number.

The challenge permits at most 50 iterations and requires convergence after three consecutive improvements of no more than `0.002`. Because the score was already near saturation, the final run reached this condition after three post-warm-start experiments. The short run reflects the required stopping rule, not a three-iteration limit in the agent. We plan a separate 50-round pilot with the early stop disabled to study longer-term search; its result will be reported as supplementary evidence.

## Earlier work and our extension

- [AIDE](https://arxiv.org/abs/2502.13138) inspired executable solution-tree search; we added branch selection based on contribution to the final model combination.
- [MLE-STAR](https://research.google/pubs/mle-star-machine-learning-engineering-agent-via-search-and-targeted-refinement/) inspired prior-based initialization, targeted refinement, and final ensembling; we feed combination value back into every research round.
- [RD-Agent](https://github.com/microsoft/RD-Agent) inspired the hypothesis-to-experiment loop; we tie each memory entry to an exact parent, code mechanism, and individual and combined measurements.
- [MLE-bench](https://github.com/openai/mle-bench) inspired separated, reproducible evaluation; we adapted it to time-split recommendation and one-time hidden-test reporting.
- [Self-Evolving Recommendation System](https://arxiv.org/abs/2602.10226) inspired optimizer, architecture, and reward-focused roles; we ground all three in one shared experiment graph and recommendation-specific diagnostics.

## Results

The official primary score is the average of GAUC and nDCG@5.

| Data split | Official FM baseline | Our solution | Gain |
|---|---:|---:|---:|
| Validation | 0.6016 | **0.606128** | **+0.004528** |
| Hidden test | 0.5946 | **0.5991** | **+0.0045** |

The hidden test was scored once, after all model choices and weights had been fixed. Test labels were never used by the agent or training code.

## Engineering work

- fixed data and code hashes detect unexpected changes;
- model training has no network access and has time and memory limits;
- library versions are recorded exactly;
- training is repeated with seeds `0`, `1`, and `2`;
- unclear gains are recorded without being presented as proven improvements;
- a live, read-only graph makes branch choices and model-combination changes inspectable;
- the experiment stops automatically when progress remains small;
- the final CSV is checked for columns, row count, numeric values, and SHA-256.

The recorded GPT-5.4 experiment used 17 LLM calls, 115,645 tokens, about 21 minutes, no GPU, and no human changes after it started.

## Challenges

The hardest part was correcting the search objective. If the agent expanded only the strongest individual model, it would underexplore LambdaRank—the weakest individual model but the largest contributor to the final combination.

A second challenge was preventing small validation changes from being overstated. The `tab` weights improved all three seed comparisons, but the measured gain was small. We report both the gain and its uncertainty.

## What we learned

- The best branch to explore is not always the current highest-scoring model.
- Model-combination value should guide research, not appear only after research is finished.
- Prior knowledge is most useful when it can be reproduced and traced to executable code.
- Repeated experiments are necessary because one training result can be lucky or unlucky.
- Rejecting a nearly equal but worse result is as important as finding a better one.

## What is next

The next steps are to run the supplementary 50-round search, repeat weight selection on another time period, choose one set of weights across both periods, and train a smaller single model to reproduce the three-model output at lower serving cost.

## Built with

Python, GPT-5.4, NumPy, SciPy, scikit-learn, LightGBM, PyTorch CPU, RecBole, TorchRec, and `bubblewrap`.
