# Live experiment graph

The dashboard shows the real graph saved by the agent. It is not a manually drawn reconstruction.
It can follow a running experiment or replay the submitted experiment records.

## Start the viewer

For a running experiment:

```bash
python3 scripts/serve_experiment_graph.py \
  --run-dir runs/<experiment-id> --open
```

For the experiment included in this submission:

```bash
python3 scripts/serve_experiment_graph.py \
  --run-dir artifacts/experiment-records --open
```

The server listens on `127.0.0.1:8765` by default. Use `--port 0` to choose any available local
port, or `--port <number>` to choose a specific port. If the machine has no desktop browser, omit
`--open` and open the printed URL from a browser that can reach the machine.

No extra web framework or JavaScript package is required. The local server uses the Python
standard library, and all page assets are included in `visualization/experiment-graph/`.

## What the graph means

- A **solid line** means the new code was built from that parent model.
- A **dashed purple line** means the proposal reused an idea from another experiment branch.
- A **dotted green line** connects a model to the best verified model combination.
- **Green nodes** were kept.
- **Amber nodes** were promising but did not have strong enough evidence.
- **Rose nodes** were rolled back after measurement.
- **Gray nodes** stopped during planning or implementation.
- **Blue nodes** are currently being planned, checked, trained, or scored.

Selecting a node opens a compact evidence panel containing:

1. the hypothesis;
2. why the agent selected that branch;
3. standalone score and change from the parent;
4. the 95% paired interval;
5. change to the current model combination, only when that candidate actually entered the
   combination comparison;
6. the measured result and saved lesson;
7. links to the exact proposal, code change, metrics, implementation check, and reflection.

The three link filters can hide execution, idea-reference, or model-combination edges. The graph
can be dragged, zoomed with the mouse wheel or buttons, and fitted to the window.

## How live updates work

The experiment controller remains the only writer of experiment results. It atomically saves
`frontier.json` whenever it chooses a parent, adds a measured node, or updates a decision.

It also writes a small `live-status.json` heartbeat every three seconds with plain status such as:

- researching and comparing ideas;
- writing and checking a code change;
- training the candidate with three fixed seeds;
- scoring the model and comparing combinations.

The heartbeat contains no features, labels, predictions, or scores. The dashboard polls its
read-only state endpoint every two seconds. If the heartbeat stops, the page changes from “Live
now” to “Paused or interrupted” instead of pretending the run is still active.

The view combines several saved records without changing them:

- `frontier.json` supplies measured nodes and execution parents;
- each `proposal.json` supplies cross-branch idea references and the hypothesis;
- each `metrics.json` supplies the standalone score and paired interval;
- each `ensemble-selection.json` supplies genuine combination entry and change;
- each `reflection.json` supplies the measured conclusion and next lesson;
- `planning.jsonl` supplies proposals that stopped before model training;
- the latest final or per-round selection supplies current combination members and weights.

## Safety boundary

The viewer is deliberately separate from the research loop:

- it does not import or call the trusted evaluator;
- it does not load NumPy prediction arrays;
- it does not write into the experiment directory;
- it cannot accept or reject a model;
- it serves only a small allowlist of text evidence files through the local server;
- path traversal outside the selected experiment directory is rejected.

Closing the viewer has no effect on the agent. Stopping the agent leaves the completed graph
available for replay.
