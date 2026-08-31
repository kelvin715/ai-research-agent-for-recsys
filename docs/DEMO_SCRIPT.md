# Three-minute demo script

The aim is to show one clear story: the best model to improve next is not always the model with the highest individual score.

## 0:00–0:20 — The problem

**Show:** README title and the three-model contribution table.

**Say:**

> Most machine-learning agents keep modifying the current highest-scoring model. That is incomplete for our task. LambdaRank was our weakest individual model at 0.601058, but removing it hurt the final combination most. We built an agent that searches model branches by how much they improve the complete system.

## 0:20–0:40 — The result

**Show:** the README result table. Highlight the two “Our solution” rows.

**Say:**

> Our final primary score is 0.606128 on validation and 0.5991 on the hidden test. That is a gain of about 0.0045 over the official baseline on both splits. The hidden test was evaluated once, only after the solution was fixed.

## 0:40–1:10 — Overall flow

**Show:** the live Experiment Graph dashboard. Start on the fitted graph, point to the three warm-start branches, and then point to the best-combination node on the right.

**Say:**

> This is the graph the agent is maintaining while it works, not a diagram drawn after the run. Solid lines show which model was changed, dashed lines show ideas reused from another branch, and the green lines show the current model combination. The page updates as the agent plans, checks code, trains, and measures each candidate.

## 1:10–1:35 — Discovery and reproducible run

**Show:** the README “From open discovery to a reproducible demonstration” section, then `artifacts/experiment-records/warmstart/verification.json`.

**Say:**

> In the discovery stage, the agent searched across repeated sessions and saved verified positive and negative results as prior evidence. For this reproducible run, GPT-5.4 received all of that evidence but still had to retrain the three warm-start models and reconstruct 0.606128 before proposing anything new.

## 1:35–1:58 — Show one real experiment round

**Show:** select the round 2 temporal-context node in the dashboard. Keep the hypothesis, two score cards, confidence interval, result, and evidence-file links visible in the right panel.

**Say:**

> Here is one complete round. The agent added time information to DeepFM and trained it three times. The model improved on its own, but the combination change was negative, so it did not replace the saved result. We can inspect the hypothesis, code change, metrics, and lesson directly from this node.

## 1:58–2:22 — Final model

**Show:** the three-model table, followed by `artifacts/final/weights-by-tab.json`.

**Say:**

> The final output combines three models that learn different patterns: pairwise ranking, DeepFM, and LambdaRank. Scores are converted to ranks within each user. A small fixed table changes model weights according to tab, which is available at prediction time and is not a label.

## 2:22–2:42 — Why the recorded run stopped early

**Show:** the recorded experiment table and its 50-round cap and stopping-rule rows.

**Say:**

> The task caps runs at 50 rounds and stops after three consecutive gains no larger than 0.002. At this saturated score level, the recorded run reached that rule after three new experiments. This is required convergence behavior, not a three-round agent limit. We will report a separate 50-round pilot to study longer search.

## 2:42–2:52 — Engineering checks

**Show:** run `python3 scripts/preflight.py`, then show the passing output.

**Say:**

> Before submission, this check verifies required files, Python and JSON syntax, absence of secrets and generated caches, and the exact row count and SHA-256 of the final CSV.

## 2:52–3:00 — Close

**Show:** the fitted live graph and final combination score together.

**Say:**

> Our contribution is a combination-aware research agent: it searches several experiment branches, learns which models complement one another, and turns prior evidence into reproducible code. The engineering checks make every decision inspectable.

## Recording notes

- Keep the video between 2:40 and 2:55 to leave room for upload trimming.
- Use large text and zoom in on only the lines being discussed.
- Do not scroll through long JSON files. Use the graph node detail panel and its short evidence links.
- Keep one score format throughout: six decimals for validation, four decimals for hidden test.
- Capture one clean, fitted dashboard frame for the Devpost gallery; keep the right detail panel on round 2.
- End on the result and contribution, not on installation commands.
