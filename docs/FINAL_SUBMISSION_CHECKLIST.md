# Final submission checklist

The code and result files are ready. Complete the publication details immediately before submission.

- [x] Core source code is included and organized.
- [x] README explains the method, setup, reproduction, results, limits, and resource use.
- [x] README includes an architecture diagram.
- [x] README distinguishes borrowed methods, our extensions, and engineering safeguards.
- [x] README explains the agent-driven discovery stage, verified prior evidence, and required early stopping.
- [x] Each experiment round includes its idea, code change, scores, decision, and error record.
- [x] Final `submission.csv` is included and passes the official format reader.
- [x] Datasets, generated data copies, label caches, virtual environments, temporary predictions, and secrets are excluded.
- [x] Hidden test was evaluated once, after the final model and weights were fixed.
- [x] Devpost story draft and three-minute demo script are included.
- [ ] Confirm the Devpost team list and update the contribution section in `README.md` if needed.
- [ ] Create the public GitHub repository and confirm that all links and the diagram display correctly.
- [ ] Attach the required report or the recommended three-minute demo video on Devpost.
- [ ] Run and publish the planned 50-round supplementary pilot if time permits; label it separately from the official recorded run.
- [ ] Run `python3 scripts/preflight.py` on the exact directory that will be uploaded.
- [ ] Confirm that `.env` is absent and that the repository is public before the deadline.
