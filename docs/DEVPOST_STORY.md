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

## What the experiment graph taught the agent

The submitted graph contains one baseline, three reproduced portfolio branches, and three new
post-warm-start branches. It records more than a final score:

| Branch | What happened | What the agent learned |
|---|---|---|
| Pairwise ranker (`w001`) | `0.604913`; accepted | Strongest verified standalone base. |
| DeepFM MTL (`w002`) | `0.604752`; accepted | A different neural interaction path adds portfolio value. |
| Item LambdaRank (`w003`) | only `0.601058` alone; standalone rollback | Do not discard it from the portfolio: removing it causes the largest loss, `0.000752`. |
| DeepFM + temporal context (`n002`) | best new standalone score, `0.605280`; portfolio `−0.000253` | A locally better member can still duplicate the ensemble's information. |
| Pairwise `tab` calibration (`n003`) | `+0.000101` alone; portfolio `−0.000034` | A connected implementation is not enough; the intended weak slices must actually improve. |
| LambdaRank `tab` residual (`n004`) | `+0.000015` alone; did not enter the best challenger | Portfolio-relative slice weakness does not directly provide a reliable correction constant. |

This is the advantage of a graph over an overwrite-and-retry loop. A rollback preserves the exact
code, measurements, and lesson. A weak standalone branch can keep a portfolio role, while a stronger
standalone child can be withheld from deployment. “Success” means improving the complete recommender
with repeatable evidence, not merely producing the newest high point estimate.

## How the warm start was created

Development and demonstration were intentionally separated.

During the discovery stage, the agent continued searching across repeated sessions instead of applying the final three-small-gains early stop to every session. Verified strong solutions, negative results, and exact experiment settings were saved as prior evidence.

For the final reproducible run, GPT-5.4 received all verified prior evidence. It first retrained all three warm-start models with three seeds, checked their hashes and scores, and reconstructed the expected `0.6061277486` result. Only then could it propose new changes. This shows that the prior is executable knowledge, not a copied leaderboard number.

The challenge permits at most 50 iterations and requires convergence after three consecutive best-result gains of no more than `0.002` (`N=3`, `ε=0.002`). This is a stopping trigger, not a minimum useful effect. Since primary is the mean of GAUC and nDCG@5, resetting the counter requires their combined change to exceed `0.004`. The `0.002` threshold is also about 47% of the final system's complete `0.004250` gain over our reproduced baseline.

At the strong warm-start operating point, the realistic marginal changes were much smaller. Some
creative ideas improved a member by `1e-5` to `5e-4`, but none improved the verified portfolio. The
deployed best history therefore stayed at `0.606128` for three rounds and the run stopped. We call
this local convergence under the required rule—not proof that the metric has no remaining
mathematical headroom, and not a three-iteration limitation of the agent.

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

A third challenge was the scale mismatch created by `N=3`, `ε=0.002`. Once a heterogeneous warm
start had captured most easy gains, plausible refinements were an order of magnitude smaller than
the reset threshold. The agent still tested the most credible sub-threshold ideas, but promotion
depended on repeated standalone and portfolio evidence. This prevented a creative but unstable
change from surviving merely because its single fitted score looked higher.

The strongest example is the final challenger router. Its raw point estimate reached `0.606411`,
but the three matched-seed changes were `+0.000072`, `+0.000068`, and `−0.000017`; the confidence
interval crossed zero. The router failed its predeclared gate. The verified challenger without it
scored `0.606094`, so the system retained `0.606128`.

## What we learned

- The best branch to explore is not always the current highest-scoring model.
- Model-combination value should guide research, not appear only after research is finished.
- Prior knowledge is most useful when it can be reproduced and traced to executable code.
- Repeated experiments are necessary because one training result can be lucky or unlucky.
- A standalone gain and a portfolio gain answer different questions; both must be recorded.
- Failed evidence should narrow the next hypothesis, not erase the branch that produced it.
- Rejecting a nearly equal but worse result is as important as finding a better one.

## Team and contributions

| Member | Contact | Contribution |
|---|---|---|
| **Zhihan Yang** | zhihan.yang@u.nus.edu | Idea, system design, core implementation, agent runtime, model portfolio, experiments, and final integration. |
| **Li Haixin** | e1113229@u.nus.edu | Evaluation contracts, quality assurance, tests, preflight checks, robustness review, and result reproduction. |
| **Dong Yicheng** | DONG0195@e.ntu.edu.sg | Experiment analysis, graph and dashboard review, decision-trace verification, and result visualization. |
| **Minxi Chen** | chen1997@e.ntu.edu.sg | README and Devpost review, judge-facing explanation, experiment narrative, and submission checklist. |
| **Qian Nuowen** | qian_nuowen@u.nus.edu | Prior-art and public-solution research, baseline comparison, method-lineage review, and diagnostic-design feedback. |

## Built with

Python, GPT-5.4, NumPy, SciPy, scikit-learn, LightGBM, PyTorch CPU, RecBole, TorchRec, and `bubblewrap`.
