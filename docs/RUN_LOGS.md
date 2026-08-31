# Experiment log guide

Agent model: GPT-5.4  
Result: stopped after progress became small; final file and data check passed

## Why this run starts from prior evidence

Before this recorded run, the agent explored model, feature, objective, and combination branches across repeated development sessions. Verified strong solutions, negative results, and exact settings were saved as prior evidence.

The recorded run received that evidence so the best discovery would be reproducible. It did not accept the saved score on trust: round 1 retrained the three warm-start models with three seeds, checked their hashes and scores, and reconstructed the expected combination before any new proposal was allowed.

## Four experiment rounds

| Round | What was tested | Score when used by itself | Decision |
|---:|---|---:|---|
| 1 | Reproduce the warm start: three models and fixed `tab` weights tested earlier | best individual model: 0.604913 | Reproduced the expected combined score of **0.606128** and saved it as the best result |
| 2 | Add hour and time-since-previous-action inputs to DeepFM | **0.605280** | Good individual model, but no new combination beat 0.606128 |
| 3 | Adjust the pairwise model's output according to `tab` | 0.605015 | The gain was too small and unclear, so the best result was not changed |
| 4 | Train another LambdaRank tree model on item and viewing information | 0.601073 | The model ran correctly but did not improve the result; the best result was kept |

The challenge allows at most 50 rounds and requires stopping after three consecutive rounds improve the best combined score by no more than `0.002`. Because the warm start was already strong, rounds 2–4 all fell below this threshold and the run stopped automatically. This is the required stopping behavior, not a three-round limit in the agent. A separate 50-round pilot with this early stop disabled is planned as supplementary evidence.

## Files saved for each round

Each `artifacts/experiment-records/iter-NNN/` directory contains:

- `proposal.json`: the idea, why it might work, the planned code change, and the required improvement;
- `attempt-0/pipeline.diff`: the exact code change;
- `attempt-0/static_gate.json`: checks for unsafe imports, label access, and invalid code patterns;
- `attempt-0/smoke.json`: a short check that the code starts and creates predictions;
- `attempt-0/seed_results.json`: training results for random seeds 0, 1, and 2;
- `metrics.json`: validation GAUC, nDCG@5, primary score, and variation across users and seeds;
- model error comparison: whether this model makes different errors from the other models (stored in each round's diagnostics JSON file);
- `ensemble-selection.json`: scores for tested model combinations;
- `reflection.json`: the decision to keep or discard the change and what to try next.

The complete time-ordered log is `artifacts/experiment-records/journal.jsonl`. A shorter summary is `artifacts/experiment-records/summary.json`. The final model choice is `artifacts/experiment-records/ensemble/selection.json`.

## Errors and recovery

The successful experiment had no out-of-memory error, timeout, invalid numeric value, or output-format error. Round 4 was not a software failure: its code ran correctly, but its score was not better. The program saved the result and kept the previous best solution.

One earlier launch stopped before model training because the host system did not allow `bubblewrap` to create the required `/proc` view. It produced no score and is not included as an experiment result. The successful experiment used the same settings on a host where the isolated process could start correctly.
