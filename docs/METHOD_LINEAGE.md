# Method lineage: what we adopted and what we added

This project does not claim to invent experiment graphs, agent memory, isolated evaluation, or model combination. Its main contribution is a recommendation-focused research loop that connects these ideas around the right objective: improving the final combined recommender, not only the strongest model measured on its own.

## The central design choice

A linear model-search agent usually selects the current highest score as its next starting point. Our measurements show why that objective is incomplete:

| Model | Validation score on its own | Drop when the model is removed | Rank similarity with the combination |
|---|---:|---:|---:|
| Pairwise ranking | 0.604913372 | 0.000647425 | 0.965965 |
| DeepFM | 0.604751755 | 0.000626835 | 0.967909 |
| LambdaRank | **0.601057584** | **0.000752450** | **0.903338** |

LambdaRank is weakest on its own but contributes the most to the combination. Its rankings are also the least similar to the combined ranking. This is why our agent measures and remembers two kinds of value:

- how well a model performs on its own;
- how much that model improves the current combination.

Both measurements affect which experiment branch the agent explores next.

## Prior evidence is accumulated agent work

The warm start is not a hand-written answer added only for the demonstration. During the discovery stage, the agent explored branches across repeated sessions. Verified strong solutions, failed mechanisms, exact settings, and supporting measurements were saved as prior evidence.

The final recorded run received all verified prior evidence so the best earlier discovery would be available. It still had to retrain each warm-start model, check the code and prediction hashes, and reconstruct the expected combined score before starting new search. In this way, prior evidence is executable state carried between agent runs rather than a list of suggestions or an old leaderboard number.

The official stopping rule ends a run after three consecutive best-result gains no larger than
`0.002`, with a maximum of 50 iterations. The threshold is a convergence trigger, not a minimum
useful effect: the agent is explicitly instructed to prefer a credible smaller gain over a
speculative threshold-crossing claim. Because the reproduced portfolio was already strong, three
new candidates produced only small standalone changes and no deployed gain, so the recorded run
stopped.

## AIDE: from code-tree search to combination-aware branch search

[AIDE](https://arxiv.org/abs/2502.13138) treats machine-learning engineering as search over a tree of executable solutions. We adopted the persistent experiment structure, multiple starting drafts, parent re-selection, repair of failed code, and a journal of outcomes.

Our extension changes what makes a branch valuable. A branch may be expanded because it improves an individual score, because it improves the current combination, or because it supplies information that distinguishes competing ideas. The agent also rotates through all models already used in the final combination. Therefore, a weaker specialist such as LambdaRank is not starved of research attention.

## MLE-STAR: from final ensembling to combination feedback during search

[MLE-STAR](https://research.google/pubs/mle-star-machine-learning-engineering-agent-via-search-and-targeted-refinement/) uses external search to form an initial solution, ablation to find important code blocks, targeted refinement, and ensembling of strong final solutions. We adopted prior-based initialization, focused changes to one code block, and a separate bounded model-combination step.

Our extension moves combination value into the main research loop. After each candidate is measured, the system asks whether it improves the current combination and feeds the answer into later proposals. It also protects existing complementary members during bounded subset search, so a lower individual score does not remove a useful specialist before its combined value is tested.

## RD-Agent: from research memory to exact, reusable experiment evidence

[RD-Agent](https://github.com/microsoft/RD-Agent) connects research ideas, experiment design, implementation, execution, and feedback. We adopted structured hypotheses, recorded observations, and memory of positive and negative outcomes.

Our memory is deliberately narrow. Each result records the exact parent branch, code mechanism, changed block, evidence source, individual-score change, combined-result change, and acceptance reason. A negative result applies to that implementation and parent context; it does not automatically declare the broader idea invalid. Exact repeats and unused configuration changes are rejected before consuming a full experiment.

## MLE-bench: from held-out grading to a recommendation-specific evidence boundary

[MLE-bench](https://github.com/openai/mle-bench) demonstrates reproducible evaluation of machine-learning agents with prepared held-out data and graders. We adopted the separation between candidate execution and trusted evaluation, immutable scoring code, and complete resource and result records.

We applied that pattern to time-split recommendation. Candidate code sees training labels only. The scorer owns validation labels and returns metrics plus aggregate diagnostics. Hidden-test scoring is a one-time final report. Output columns, row count, numeric values, and file hash are checked before submission.

These are essential engineering safeguards, not the central algorithmic novelty.

## Self-Evolving Recommendation System: specialized research roles with shared evidence

[Self-Evolving Recommendation System](https://arxiv.org/abs/2602.10226) uses specialized roles that improve the optimizer, model architecture, and reward. We adopted those three research viewpoints.

In our system, the roles do not brainstorm independently from evidence. They share the same experiment graph and receive the same aggregate recommendation diagnostics: removal loss for each model, similarity and diversity between model rankings, and performance by `tab`, user-history level, and item popularity. Every proposal states whether it is refining one member, creating a complementary member, joining branches, or replacing a member in the current combination.

## MLEvolve: a budget-aware form of cross-branch search

[MLEvolve](https://github.com/InternScience/MLEvolve) motivates graph search with global memory, stagnation detection, and joining ideas from different branches. We adopted those principles without copying a long, parallel search procedure.

Our version is bounded by the challenge experiment budget. It rotates through current combination members, keeps uncertain branches available for exploration, and asks for a structural or cross-branch idea after progress stalls. This concentrates the search on branches with demonstrated individual or combined value.

## What is specific to this solution

The following pieces are our task-specific contribution as a complete system:

1. **A combination-aware experiment graph.** Branch selection and acceptance use both individual score and direct improvement to the current combined recommender.
2. **Separate exploration, safe continuation, and deployment decisions.** A promising point estimate can remain explorable without becoming the default parent or replacing the deployed combination.
3. **Recommendation-specific direction.** Aggregate removal, similarity, and slice measurements turn a scalar score into actionable feedback about which model and user context need work.
4. **An evidence-to-code contract.** Published ideas and past experiments must map to an exact parent, executable data path, expected measurement, and disproof condition; saved strong solutions must reproduce before use.
5. **A bounded final selector coupled to research.** The same combination logic is used during validation feedback and submission inference, with simple fixed `tab` weights chosen only after the component models are fixed.
6. **A complementary three-model method.** Pairwise ranking, multi-task DeepFM, and LambdaRank learn different patterns; within-user rank conversion makes their outputs comparable before weighting.

Label isolation, process limits, hashes, repeated seeds, logs, and output checks make the evidence trustworthy. They support the method, but they are not presented as innovations by themselves.
