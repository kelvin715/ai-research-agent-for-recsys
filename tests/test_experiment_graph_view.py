"""Tests for the read-only live experiment graph state."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "orchestrator"))

import graph_view  # noqa: E402
from live_status import LiveRunStatus  # noqa: E402


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ExperimentGraphViewTests(unittest.TestCase):
    def build_run(self, root: Path):
        write_json(root / "config.json", {
            "run_id": "graph-test", "model": "gpt-test", "role": "pilot",
        })
        write_json(root / "frontier.json", {
            "schema_version": "frontier-1.0",
            "nodes": [
                {
                    "node_id": "n000", "parent_node_id": None,
                    "decision": "BASELINE", "status": "COMPLETE",
                    "selection_primary": 0.60, "mechanism": "baseline",
                    "metrics_path": "baseline/metrics.json",
                    "pipeline_path": "incumbents/iter-000.py",
                    "prediction_paths": [], "times_selected_as_parent": 0,
                },
                {
                    "node_id": "w001", "parent_node_id": "n000",
                    "decision": "ACCEPT", "status": "COMPLETE",
                    "selection_primary": 0.61, "mechanism": "warm model",
                    "metrics_path": "warmstart/members/member-001/metrics.json",
                    "pipeline_path": "warmstart/member.py",
                    "prediction_paths": [], "times_selected_as_parent": 2,
                },
                {
                    "node_id": "n002", "parent_node_id": "w001",
                    "decision": "UNCERTAIN", "status": "COMPLETE",
                    "selection_primary": 0.612, "mechanism": "new context model",
                    "metrics_path": "iter-002/metrics.json",
                    "pipeline_path": "iter-002/attempt-0/pipeline.py",
                    "prediction_paths": [], "times_selected_as_parent": 0,
                },
            ],
            "parent_selections": [], "policy": {},
        })
        write_json(root / "baseline/metrics.json", {"selection_primary": 0.60})
        write_json(root / "warmstart/members/member-001/metrics.json", {
            "selection_primary": 0.61,
        })
        write_json(root / "warmstart/members/member-001/proposal.json", {
            "hypothesis": "Reproduce the saved model.",
        })
        write_json(root / "iter-002/proposal.json", {
            "mechanism": "new context model",
            "hypothesis": "Context improves ranking.",
            "justification": "The feature changes within-user ordering.",
            "parent_references": ["w001", "n000"],
            "execution_mode": "custom_patch", "patch_scope": ["features"],
        })
        write_json(root / "iter-002/metrics.json", {
            "selection_primary": 0.612,
            "paired_vs_incumbent": {
                "delta_primary": 0.002,
                "paired_ci95": [0.001, 0.003],
            },
        })
        write_json(root / "iter-002/reflection.json", {
            "analysis": "The feature changed predictions.",
            "next_lesson": "Keep the useful context.",
        })
        write_json(root / "iter-002/ensemble-selection.json", {
            "status": "SELECTED", "selection_primary": 0.62,
            "delta_vs_single_best": 0.008,
            "members": [
                {"node_id": "w001", "weight": 0.6, "standalone_primary": 0.61},
                {"node_id": "n002", "weight": 0.4, "standalone_primary": 0.612},
            ],
            "incumbent_comparison": {
                "candidate_node_id": "n002", "candidate_entered": True,
                "delta_primary": 0.001, "paired_ci95": [0.0002, 0.0018],
            },
        })
        (root / "iter-002/attempt-0").mkdir(parents=True)
        (root / "iter-002/attempt-0/pipeline.py").write_text("# model\n", encoding="utf-8")
        (root / "iter-002/attempt-0/pipeline.diff").write_text("+ feature\n", encoding="utf-8")
        planning_event = {
            "iter": 3, "status": "PLANNING_ERROR", "parent_node_id": "w001",
            "outcome": {"decision": "PLANNING_ERROR"},
            "error": {"message": "evidence needs a number"},
            "usage": {"total_tokens": 100}, "wall_s": 4.0,
        }
        (root / "planning.jsonl").write_text(
            json.dumps(planning_event) + "\n", encoding="utf-8")
        journal_event = {
            "iter": 2, "status": "COMPLETE",
            "proposal": {"mechanism": "new context model"},
            "outcome": {"decision": "UNCERTAIN", "candidate_metrics": {
                "selection_primary": 0.612}},
            "usage": {"total_tokens": 500}, "wall_s": 12.0,
        }
        (root / "journal.jsonl").write_text(
            json.dumps(journal_event) + "\n", encoding="utf-8")
        write_json(root / "iter-004/search-directive.json", {
            "selected_parent": {"node_id": "n002"},
        })
        write_json(root / "iter-004/proposal.json", {
            "mechanism": "active idea", "hypothesis": "Still training.",
            "parent_references": ["n002", "w001"],
        })

    def test_build_state_combines_execution_reference_and_portfolio_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_run(root)
            write_json(root / "live-status.json", {
                "status": "running", "iteration": 4,
                "phase": "Training the candidate with three fixed seeds",
                "updated_unix_s": time.time(),
            })
            state = graph_view.build_graph_state(root)

            nodes = {node["id"]: node for node in state["nodes"]}
            self.assertEqual(state["run"]["status"], "live")
            self.assertEqual(nodes["a004"]["stage"],
                             "Training the candidate with three fixed seeds")
            self.assertEqual(nodes["n002"]["combination_delta"], 0.001)
            self.assertEqual(nodes["n002"]["score_ci95"], [0.001, 0.003])
            self.assertIn("code change", nodes["n002"]["artifacts"])
            self.assertIn("p003", nodes)
            self.assertIn("portfolio-best", nodes)
            edge_types = {(edge["source"], edge["target"], edge["type"])
                          for edge in state["edges"]}
            self.assertIn(("w001", "n002", "execution"), edge_types)
            self.assertIn(("n000", "n002", "reference"), edge_types)
            self.assertIn(("w001", "a004", "reference"), edge_types)
            self.assertIn(("n002", "portfolio-best", "portfolio"), edge_types)
            self.assertEqual(state["stats"]["best_combination"], 0.62)
            self.assertEqual(state["stats"]["total_tokens"], 500)
            self.assertEqual(state["warnings"], [])

    def test_stale_heartbeat_marks_incomplete_node_paused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_run(root)
            write_json(root / "live-status.json", {
                "status": "running", "iteration": 4,
                "phase": "Training", "updated_unix_s": time.time() - 60,
            })
            state = graph_view.build_graph_state(root)
            nodes = {node["id"]: node for node in state["nodes"]}
            self.assertEqual(state["run"]["status"], "paused")
            self.assertEqual(nodes["a004"]["status_key"], "paused")

    def test_live_status_writes_heartbeat_without_touching_experiment_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            status = LiveRunStatus(temporary, "heartbeat-test", interval_s=0.02)
            status.start("Planning")
            status.update("Training", iteration=2)
            time.sleep(0.04)
            status.stop("complete", "done")
            payload = json.loads(
                (Path(temporary) / "live-status.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "heartbeat-test")
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["detail"], "done")
            self.assertIn("updated_unix_s", payload)
            self.assertEqual(os.listdir(temporary), ["live-status.json"])


if __name__ == "__main__":
    unittest.main()
