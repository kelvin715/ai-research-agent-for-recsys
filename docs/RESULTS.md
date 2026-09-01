# Results and evaluation rules

## Official score

The Track 2 primary score is:

```text
primary score = 0.5 × GAUC + 0.5 × nDCG@5
```

GAUC measures the overall ordering for each user. nDCG@5 measures the quality of the first five results for each user.

## Validation results used to choose the solution

| Method | GAUC | nDCG@5 | Primary score |
|---|---:|---:|---:|
| Official FM baseline | 0.6674 | 0.5357 | 0.6016 |
| Baseline repeated by our experiment | — | — | 0.6018779688793552 |
| Best new model when used by itself | — | — | 0.6052800193085559 |
| Fixed three-model combination | **0.6733099778955616** | **0.5389455192561522** | **0.6061277485758569** |

The fixed three-model combination improved by `0.004249779696501754` over the baseline repeated in the same experiment. Compared with the rounded official baseline of `0.6016`, the gain was `0.0045277485758569`.

It also beat the strongest saved individual model by `0.0012143770231538` and the best new individual model by `0.0008477292673010`.

## Hidden-test result

We used the official starter-kit scorer once on the hidden test. This happened only after model choice was complete and the final CSV could no longer be changed.

| Method | GAUC | nDCG@5 | Primary score |
|---|---:|---:|---:|
| Official FM baseline | 0.6610 | 0.5282 | 0.5946 |
| Fixed final submission | **0.6666** | **0.5317** | **0.5991** |
| Gain | **+0.0056** | **+0.0035** | **+0.0045** |

The starter kit reports hidden-test scores to four decimal places, so we do not report more digits.

### How labels were separated

- Model training code never received validation or test labels.
- Validation labels stayed inside the separate scoring program. Only validation scores and error summaries were used to choose models.
- Test labels were not used to choose models, choose weights, write prompts, update the agent's history, or recover from errors.
- The one-time hidden-test score was recorded after selection and was not returned to the agent.

## Check of the `tab` weights

The fixed `tab` weights scored `0.0002996689060545954` higher than using `0.4 / 0.4 / 0.2` for every row. We repeated this comparison for three random seeds. The gains were:

```text
seed 0: +0.0002421071942060
seed 1: +0.0001501077513975
seed 2: +0.0002110082560215
```

All three gains were positive. However, the 95% uncertainty range was `[-0.0001286678662765, 0.0007331298910159]`, which includes zero. We therefore describe this as a consistent but small validation gain, not as conclusive proof.

## Effect size, convergence, and promotion are different tests

The official run uses `N=3` and `ε=0.002`: the convergence counter resets only when the best
deployed primary score increases by more than `0.002`. This threshold is not the minimum useful
candidate gain and is not used to rank proposals.

Its scale is important when interpreting the short run:

- primary is a mean, so a reset requires `ΔGAUC + ΔnDCG@5 > 0.004`;
- `0.002` is 47.1% of the final `0.0042497797` gain over the same-run baseline;
- after the warm start, the largest new standalone delta was `+0.0005282638`, while its portfolio
  delta was `−0.0002527446`;
- the next two standalone deltas were only `+0.0001011505` and `+0.0000153753`, both with paired
  intervals crossing zero; the former entered a worse challenger and the latter did not enter the
  best challenger.

Consequently, the best deployed history was:

```text
0.6018779689 -> 0.6061277486 -> 0.6061277486 -> 0.6061277486 -> 0.6061277486
 baseline          warm start        round 2        round 3        round 4
```

This shows local convergence under the required rule. It does not show that all useful effects
below `0.002` are worthless or that the metric is globally saturated.

### Two final non-promotions

The final challenger illustrates why a raw best score is not automatically deployed:

1. A newly fitted contextual router reported `0.6064106167`. Against the same challenger's global
   weights, its matched-seed changes were `+0.0000723809`, `+0.0000683296`, and
   `−0.0000172679` (mean `+0.0000411475`), with CI
   `[-0.0002225923, 0.0008765179]`. It failed the predeclared contextual promotion gate.
2. Without that unverified router, the challenger scored `0.6060940663`, which was
   `−0.0000336823` below the incumbent `0.6061277486`.

The higher raw router score remains in the record for audit, but neither it nor the slightly worse
verified challenger changed the final submission.

## Time and API use

| Item | Recorded value |
|---|---:|
| Experiment rounds | 4 |
| New ideas tested after reproducing the warm start | 3 |
| Maximum allowed rounds | 50 |
| Required early stop | 3 consecutive gains no larger than 0.002 |
| LLM calls | 17 |
| Input tokens | 101,754 |
| Output tokens | 13,891 |
| Total tokens | 115,645 |
| Time waiting for LLM responses | 243.274 seconds |
| Total experiment time | 1,266.963 seconds |
| GPU use | 0 hours |
| Human changes after start | 0 |

The high warm-start score meant that the required early stop fired after the three new ideas. This
short recorded run should not be read as a three-round search limit; it is the official submitted
result under the required convergence rule.

The full machine-readable record is `artifacts/experiment-records/summary.json`. The shorter final result file is `artifacts/final/results.json`.
