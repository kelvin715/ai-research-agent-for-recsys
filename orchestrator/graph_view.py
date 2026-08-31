"""Read-only state builder and HTTP server for the live experiment graph.

The experiment controller remains the sole writer of run artifacts.  This module only reads
those artifacts and turns them into a small, presentation-friendly JSON document.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import time
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ITER_RE = re.compile(r"iter-(\d{3})$")
NODE_ITER_RE = re.compile(r"n(\d{3})$")
TEXT_ARTIFACT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".diff", ".py"}
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024


def _read_json(path: Path, warnings: list[str], default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {path.name}: {type(exc).__name__}: {exc}")
        return default


def _read_jsonl(path: Path, warnings: list[str]) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    try:
        for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(
                    f"Could not read {path.name} line {line_number}: {exc.msg}")
                continue
            if isinstance(value, dict):
                rows.append(value)
    except (OSError, UnicodeDecodeError) as exc:
        warnings.append(f"Could not read {path.name}: {type(exc).__name__}: {exc}")
    return rows


def _relative(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except (OSError, ValueError):
        return None


def _iteration_from_node(node_id: str) -> int | None:
    match = NODE_ITER_RE.fullmatch(node_id)
    return int(match.group(1)) if match else None


def _iteration_directories(run_dir: Path) -> dict[int, Path]:
    result = {}
    for path in run_dir.glob("iter-???"):
        if not path.is_dir():
            continue
        match = ITER_RE.fullmatch(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def _artifact_paths(run_dir: Path, iteration: int | None, node: dict) -> dict[str, str]:
    paths: dict[str, str] = {}

    def add(name: str, path: Path | None):
        if path is not None and path.exists():
            relative = _relative(run_dir, path)
            if relative:
                paths[name] = relative

    metrics_path = node.get("metrics_path")
    add("metrics", run_dir / metrics_path if metrics_path else None)
    pipeline_path = node.get("pipeline_path")
    add("pipeline", run_dir / pipeline_path if pipeline_path else None)
    if iteration is not None:
        iteration_dir = run_dir / f"iter-{iteration:03d}"
        add("proposal", iteration_dir / "proposal.json")
        add("selection", iteration_dir / "selection-trace.json")
        add("combination", iteration_dir / "ensemble-selection.json")
        add("reflection", iteration_dir / "reflection.json")
        attempts = sorted(iteration_dir.glob("attempt-*"))
        if attempts:
            add("code change", attempts[-1] / "pipeline.diff")
            add("implementation check", attempts[-1] / "implementation-audit.json")
    return paths


def _status_key(decision: str | None, status: str | None, kind: str) -> str:
    decision = (decision or "").upper()
    status = (status or "").upper()
    if kind == "portfolio":
        return "portfolio"
    if kind == "active":
        return "running"
    if kind == "planning":
        return "planning_failed"
    if decision == "BASELINE":
        return "baseline"
    if decision == "ACCEPT":
        return "accepted"
    if decision == "UNCERTAIN":
        return "uncertain"
    if decision in {"ROLLBACK", "REJECT"}:
        return "rollback" if decision == "ROLLBACK" else "failed"
    if status in {"FAILED", "ORCHESTRATOR_ERROR", "PLANNING_ERROR"}:
        return "failed"
    return "recorded"


def _status_label(status_key: str) -> str:
    return {
        "baseline": "Baseline",
        "accepted": "Kept",
        "uncertain": "Promising, not proven",
        "rollback": "Rolled back",
        "failed": "Implementation failed",
        "planning_failed": "Planning stopped",
        "running": "In progress",
        "paused": "Paused",
        "recorded": "Recorded",
        "portfolio": "Best combination",
    }.get(status_key, status_key.replace("_", " ").title())


def _latest_selection(run_dir: Path, iteration_dirs: dict[int, Path],
                      warnings: list[str]) -> tuple[dict | None, str | None]:
    final_path = run_dir / "ensemble" / "selection.json"
    final = _read_json(final_path, warnings)
    if isinstance(final, dict) and final.get("members"):
        return final, _relative(run_dir, final_path)
    for iteration in sorted(iteration_dirs, reverse=True):
        path = iteration_dirs[iteration] / "ensemble-selection.json"
        selection = _read_json(path, warnings)
        if isinstance(selection, dict) and selection.get("members"):
            return selection, _relative(run_dir, path)
    warm_path = run_dir / "warmstart" / "selection.json"
    warm = _read_json(warm_path, warnings)
    if isinstance(warm, dict) and warm.get("members"):
        return warm, _relative(run_dir, warm_path)
    return None, None


def _active_stage(iteration_dir: Path) -> str:
    if not (iteration_dir / "proposal.json").exists():
        if (iteration_dir / "research" / "research.json").exists():
            return "Researching and comparing ideas"
        return "Planning the next experiment"
    if (iteration_dir / "metrics.json").exists():
        return "Comparing the model and the current combination"
    attempts = sorted(iteration_dir.glob("attempt-*"))
    if not attempts:
        return "Writing the code change"
    attempt = attempts[-1]
    if (attempt / "seed_results.json").exists():
        return "Scoring the completed training runs"
    if any(attempt.glob("s?/ws/pred.npy")) or any(attempt.glob("s?/logs")):
        return "Training three fixed random seeds"
    if (attempt / "smoke.json").exists():
        return "Starting full training"
    return "Checking and repairing the code change"


def _last_modified(paths: list[Path]) -> float:
    mtimes = []
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes, default=0.0)


def build_graph_state(run_dir: str | os.PathLike[str]) -> dict:
    """Build one dashboard snapshot from an experiment directory."""
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {root}")

    warnings: list[str] = []
    frontier = _read_json(root / "frontier.json", warnings, default={}) or {}
    config = _read_json(root / "config.json", warnings, default={}) or {}
    summary = _read_json(root / "summary.json", warnings)
    live_status = _read_json(root / "live-status.json", warnings, default={}) or {}
    journal = _read_jsonl(root / "journal.jsonl", warnings)
    planning = _read_jsonl(root / "planning.jsonl", warnings)
    iteration_dirs = _iteration_directories(root)
    selection, selection_path = _latest_selection(root, iteration_dirs, warnings)

    journal_by_iteration = {
        int(row["iter"]): row for row in journal if isinstance(row.get("iter"), int)
    }
    planning_by_iteration = {
        int(row["iter"]): row for row in planning if isinstance(row.get("iter"), int)
    }
    frontier_nodes = frontier.get("nodes") if isinstance(frontier, dict) else []
    if not isinstance(frontier_nodes, list):
        frontier_nodes = []

    selected_members: dict[str, dict] = {}
    if isinstance(selection, dict):
        for member in selection.get("members") or []:
            if isinstance(member, dict) and member.get("node_id"):
                selected_members[str(member["node_id"])] = member

    nodes: list[dict] = []
    edges: list[dict] = []
    known_node_ids = {
        str(node.get("node_id")) for node in frontier_nodes if node.get("node_id")
    }
    proposal_by_node: dict[str, dict] = {}

    for raw_node in frontier_nodes:
        if not isinstance(raw_node, dict) or not raw_node.get("node_id"):
            continue
        node_id = str(raw_node["node_id"])
        iteration = _iteration_from_node(node_id)
        if node_id.startswith("w"):
            try:
                member_number = int(node_id[1:])
            except ValueError:
                member_number = 0
            proposal_path = (
                root / "warmstart" / "members" / f"member-{member_number:03d}" /
                "proposal.json")
            reflection_path = proposal_path.with_name("reflection.json")
            kind = "warmstart"
        elif node_id == "n000":
            proposal_path = None
            reflection_path = None
            kind = "baseline"
        else:
            proposal_path = (
                root / f"iter-{iteration:03d}" / "proposal.json"
                if iteration is not None else None)
            reflection_path = (
                root / f"iter-{iteration:03d}" / "reflection.json"
                if iteration is not None else None)
            kind = "experiment"

        proposal = _read_json(proposal_path, warnings, default={}) if proposal_path else {}
        reflection = (
            _read_json(reflection_path, warnings, default={}) if reflection_path else {})
        proposal = proposal if isinstance(proposal, dict) else {}
        reflection = reflection if isinstance(reflection, dict) else {}
        proposal_by_node[node_id] = proposal

        metrics_path_text = raw_node.get("metrics_path")
        metrics = _read_json(
            root / metrics_path_text, warnings, default={}) if metrics_path_text else {}
        metrics = metrics if isinstance(metrics, dict) else {}
        paired = metrics.get("paired_vs_incumbent") or {}

        combination_delta = None
        combination_ci = None
        candidate_entered = None
        if iteration is not None:
            combination_record = _read_json(
                root / f"iter-{iteration:03d}" / "ensemble-selection.json",
                warnings, default={}) or {}
            comparison = combination_record.get("incumbent_comparison") or {}
            if comparison.get("candidate_node_id") == node_id:
                candidate_entered = bool(comparison.get("candidate_entered"))
                if candidate_entered:
                    combination_delta = comparison.get("delta_primary")
                    combination_ci = comparison.get("paired_ci95")

        decision = raw_node.get("decision")
        status = raw_node.get("status")
        status_key = _status_key(decision, status, kind)
        member = selected_members.get(node_id)
        node = {
            "id": node_id,
            "kind": kind,
            "iteration": iteration,
            "parent_id": raw_node.get("parent_node_id"),
            "label": raw_node.get("mechanism") or proposal.get("mechanism") or node_id,
            "mechanism": raw_node.get("mechanism") or proposal.get("mechanism"),
            "decision": decision,
            "status": status,
            "status_key": status_key,
            "status_label": _status_label(status_key),
            "score": raw_node.get("selection_primary"),
            "score_delta": paired.get("delta_primary"),
            "score_ci95": paired.get("paired_ci95"),
            "combination_delta": combination_delta,
            "combination_ci95": combination_ci,
            "candidate_entered_combination": candidate_entered,
            "hypothesis": proposal.get("hypothesis"),
            "why": proposal.get("justification"),
            "expected": proposal.get("expected_observation"),
            "result": reflection.get("analysis"),
            "lesson": reflection.get("next_lesson"),
            "execution_mode": proposal.get("execution_mode") or raw_node.get("execution_mode"),
            "operator_id": proposal.get("operator_id") or raw_node.get("operator_id"),
            "patch_scope": proposal.get("patch_scope") or raw_node.get("logical_patch_scope"),
            "parent_references": proposal.get("parent_references") or [],
            "times_selected_as_parent": raw_node.get("times_selected_as_parent", 0),
            "is_portfolio_member": member is not None,
            "portfolio_weight": member.get("weight") if member else None,
            "artifacts": _artifact_paths(root, iteration, raw_node),
        }
        nodes.append(node)
        parent_id = raw_node.get("parent_node_id")
        if parent_id:
            edges.append({
                "source": str(parent_id), "target": node_id,
                "type": "execution", "label": "built from",
            })

    for node in nodes:
        if node["kind"] != "experiment":
            continue
        for reference in node.get("parent_references") or []:
            reference = str(reference)
            if (reference in known_node_ids and reference != node.get("parent_id")
                    and reference != node["id"]):
                edges.append({
                    "source": reference, "target": node["id"],
                    "type": "reference", "label": "idea used",
                })

    represented_iterations = {
        node["iteration"] for node in nodes if node.get("iteration") is not None
    }
    for iteration, event in sorted(planning_by_iteration.items()):
        if iteration in represented_iterations:
            continue
        parent_id = (
            event.get("parent_node_id") or
            (event.get("frontier_parent_selection") or {}).get("node_id"))
        error = event.get("error") or {}
        node_id = f"p{iteration:03d}"
        nodes.append({
            "id": node_id,
            "kind": "planning",
            "iteration": iteration,
            "parent_id": parent_id,
            "label": "No valid experiment was produced",
            "mechanism": None,
            "decision": "PLANNING_ERROR",
            "status": event.get("status"),
            "status_key": "planning_failed",
            "status_label": _status_label("planning_failed"),
            "score": None,
            "score_delta": None,
            "score_ci95": None,
            "combination_delta": None,
            "combination_ci95": None,
            "candidate_entered_combination": None,
            "hypothesis": None,
            "why": "The proposed experiment did not pass the planning checks.",
            "expected": None,
            "result": error.get("message") or (event.get("outcome") or {}).get("reason"),
            "lesson": "Revise the proposal evidence or structure before training.",
            "execution_mode": None,
            "operator_id": None,
            "patch_scope": [],
            "parent_references": [],
            "times_selected_as_parent": 0,
            "is_portfolio_member": False,
            "portfolio_weight": None,
            "artifacts": {},
        })
        known_node_ids.add(node_id)
        if parent_id in known_node_ids:
            edges.append({
                "source": str(parent_id), "target": node_id,
                "type": "execution", "label": "planned from",
            })

    completed_iterations = set(journal_by_iteration) | set(planning_by_iteration)
    incomplete_iterations = [
        iteration for iteration in iteration_dirs if iteration not in completed_iterations
    ]
    active_iteration = max(incomplete_iterations, default=None)
    active_node_id = None
    if active_iteration is not None:
        iteration_dir = iteration_dirs[active_iteration]
        proposal = _read_json(iteration_dir / "proposal.json", warnings, default={}) or {}
        directive = _read_json(
            iteration_dir / "search-directive.json", warnings, default={}) or {}
        parent_references = proposal.get("parent_references") or []
        parent_id = ((directive.get("selected_parent") or {}).get("node_id")
                     or (parent_references[0] if parent_references else None))
        active_node_id = f"a{active_iteration:03d}"
        live_iteration = live_status.get("iteration")
        stage = (
            live_status.get("phase")
            if live_iteration == active_iteration and live_status.get("phase")
            else _active_stage(iteration_dir))
        nodes.append({
            "id": active_node_id,
            "kind": "active",
            "iteration": active_iteration,
            "parent_id": parent_id,
            "label": proposal.get("mechanism") or stage,
            "mechanism": proposal.get("mechanism"),
            "decision": "IN_PROGRESS",
            "status": "IN_PROGRESS",
            "status_key": "running",
            "status_label": _status_label("running"),
            "stage": stage,
            "score": None,
            "score_delta": None,
            "score_ci95": None,
            "combination_delta": None,
            "combination_ci95": None,
            "candidate_entered_combination": None,
            "hypothesis": proposal.get("hypothesis"),
            "why": proposal.get("justification"),
            "expected": proposal.get("expected_observation"),
            "result": None,
            "lesson": None,
            "execution_mode": proposal.get("execution_mode"),
            "operator_id": proposal.get("operator_id"),
            "patch_scope": proposal.get("patch_scope") or [],
            "parent_references": proposal.get("parent_references") or [],
            "times_selected_as_parent": 0,
            "is_portfolio_member": False,
            "portfolio_weight": None,
            "artifacts": _artifact_paths(
                root, active_iteration, {"metrics_path": None, "pipeline_path": None}),
        })
        if parent_id in known_node_ids:
            edges.append({
                "source": str(parent_id), "target": active_node_id,
                "type": "execution", "label": "working from",
            })
        for reference in proposal.get("parent_references") or []:
            reference = str(reference)
            if reference in known_node_ids and reference != parent_id:
                edges.append({
                    "source": reference, "target": active_node_id,
                    "type": "reference", "label": "idea used",
                })

    portfolio_node_id = None
    if isinstance(selection, dict) and selection.get("members"):
        portfolio_node_id = "portfolio-best"
        portfolio_status = selection.get("status") or "SELECTED"
        nodes.append({
            "id": portfolio_node_id,
            "kind": "portfolio",
            "iteration": None,
            "parent_id": None,
            "label": "Best model combination",
            "mechanism": selection.get("combination"),
            "decision": portfolio_status,
            "status": portfolio_status,
            "status_key": "portfolio",
            "status_label": _status_label("portfolio"),
            "score": selection.get("selection_primary"),
            "score_delta": selection.get("delta_vs_single_best"),
            "score_ci95": None,
            "combination_delta": None,
            "combination_ci95": None,
            "candidate_entered_combination": None,
            "hypothesis": "Combine models that make different ranking errors.",
            "why": "The combination is kept only when its validation evidence is stronger than the alternatives.",
            "expected": None,
            "result": selection.get("promotion_reason") or selection.get("status"),
            "lesson": None,
            "execution_mode": "combination",
            "operator_id": None,
            "patch_scope": [],
            "parent_references": [],
            "times_selected_as_parent": 0,
            "is_portfolio_member": False,
            "portfolio_weight": None,
            "artifacts": {"combination": selection_path} if selection_path else {},
        })
        for member_id, member in selected_members.items():
            if member_id in known_node_ids:
                edges.append({
                    "source": member_id, "target": portfolio_node_id,
                    "type": "portfolio", "label": "member",
                    "weight": member.get("weight"),
                })

    event_files = [root / "frontier.json", root / "journal.jsonl", root / "live-status.json"]
    event_files.extend(path for path in iteration_dirs.values())
    last_modified = _last_modified(event_files)
    age_seconds = max(0.0, time.time() - last_modified) if last_modified else None
    heartbeat_age_s = None
    if isinstance(live_status.get("updated_unix_s"), (int, float)):
        heartbeat_age_s = max(0.0, time.time() - float(live_status["updated_unix_s"]))
    heartbeat_state = str(live_status.get("status") or "").lower()
    if summary or heartbeat_state == "complete":
        run_status = "complete"
        run_status_label = "Complete"
    elif heartbeat_state == "running" and heartbeat_age_s is not None and heartbeat_age_s <= 12:
        run_status = "live"
        run_status_label = "Live now"
    elif live_status and heartbeat_state in {"running", "starting"}:
        run_status = "paused"
        run_status_label = "Paused or interrupted"
    elif active_iteration is not None and age_seconds is not None and age_seconds <= 45:
        # Backward-compatible fallback for runs created before heartbeat support.
        run_status = "live"
        run_status_label = "Live now"
    elif active_iteration is not None:
        run_status = "paused"
        run_status_label = "Paused or interrupted"
    else:
        run_status = "waiting"
        run_status_label = "Waiting for the next round"

    if active_node_id and run_status != "live":
        for node in nodes:
            if node["id"] == active_node_id:
                node["status_key"] = "paused"
                node["status_label"] = _status_label("paused")
                node["status"] = run_status.upper()
                break

    measured_nodes = [
        node for node in nodes if node["kind"] in {"experiment", "warmstart"}
        and node.get("score") is not None
    ]
    experiment_nodes = [node for node in nodes if node["kind"] == "experiment"]
    best_standalone_node = max(
        (node for node in nodes
         if node["kind"] in {"baseline", "warmstart", "experiment"}
         and isinstance(node.get("score"), (int, float))),
        key=lambda item: item["score"], default=None)
    total_tokens = sum(
        int((row.get("usage") or {}).get("total_tokens", 0)) for row in journal)
    total_wall_s = sum(float(row.get("wall_s") or 0.0) for row in journal)

    timeline = []
    for row in sorted(journal + planning, key=lambda item: item.get("iter", 0)):
        iteration = row.get("iter")
        outcome = row.get("outcome") or {}
        proposal = row.get("proposal") or {}
        timeline.append({
            "iteration": iteration,
            "decision": outcome.get("decision"),
            "status": row.get("status"),
            "label": proposal.get("mechanism") or proposal.get("hypothesis") or
                     ("Warm start" if row.get("empirical_portfolio_warmstart") else
                      "Planning did not produce an experiment"),
            "score": (outcome.get("candidate_metrics") or {}).get("selection_primary")
                     or outcome.get("selection_primary"),
            "tokens": int((row.get("usage") or {}).get("total_tokens", 0)),
            "wall_s": row.get("wall_s"),
        })
    if active_iteration is not None:
        timeline.append({
            "iteration": active_iteration, "decision": "IN_PROGRESS",
            "status": run_status.upper(),
            "label": next(
                (node.get("stage") or node["label"] for node in nodes
                 if node["id"] == active_node_id), "In progress"),
            "score": None, "tokens": None, "wall_s": None,
        })

    stats = {
        "graph_nodes": len([
            node for node in nodes if node["kind"] not in {"portfolio", "active"}
        ]),
        "measured_models": len(measured_nodes),
        "new_experiments": len(experiment_nodes),
        "completed_rounds": len(journal_by_iteration),
        "accepted": sum(node["status_key"] == "accepted" for node in experiment_nodes),
        "uncertain": sum(node["status_key"] == "uncertain" for node in experiment_nodes),
        "rolled_back": sum(node["status_key"] == "rollback" for node in experiment_nodes),
        "failed": sum(node["status_key"] == "failed" for node in experiment_nodes),
        "planning_stops": len(planning_by_iteration),
        "best_standalone": best_standalone_node.get("score") if best_standalone_node else None,
        "best_standalone_node": best_standalone_node.get("id") if best_standalone_node else None,
        "best_combination": selection.get("selection_primary") if selection else None,
        "total_tokens": total_tokens,
        "wall_s": total_wall_s,
    }
    for node in nodes:
        node["is_best_standalone"] = node["id"] == stats["best_standalone_node"]

    return {
        "schema_version": "experiment-graph-view-1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "id": config.get("run_id") or (summary or {}).get("run_id") or root.name,
            "directory": str(root),
            "model": config.get("model"),
            "role": config.get("role") or (summary or {}).get("role"),
            "status": run_status,
            "status_label": run_status_label,
            "current_iteration": active_iteration,
            "last_completed_iteration": max(completed_iterations, default=None),
            "stop_reason": (summary or {}).get("stop_reason"),
            "last_update_age_s": round(age_seconds, 1) if age_seconds is not None else None,
            "heartbeat_age_s": (
                round(heartbeat_age_s, 1) if heartbeat_age_s is not None else None),
            "phase": live_status.get("phase"),
        },
        "stats": stats,
        "nodes": nodes,
        "edges": edges,
        "timeline": timeline,
        "selection": {
            "node_id": portfolio_node_id,
            "status": selection.get("status") if selection else None,
            "score": selection.get("selection_primary") if selection else None,
            "members": list(selected_members.values()),
            "context_router": selection.get("context_router") if selection else None,
            "artifact": selection_path,
        },
        "warnings": warnings,
    }


class ExperimentGraphHandler(BaseHTTPRequestHandler):
    """Serve dashboard assets and fresh snapshots from one experiment directory."""

    server_version = "ExperimentGraph/1.0"

    def log_message(self, format_string, *args):  # noqa: A002 - BaseHTTPRequestHandler API
        if getattr(self.server, "quiet", False):
            return
        super().log_message(format_string, *args)

    def _send_bytes(self, payload: bytes, content_type: str,
                    status: HTTPStatus = HTTPStatus.OK):
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, value, status: HTTPStatus = HTTPStatus.OK):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8", status)

    def _serve_static(self, relative_path: str):
        static_root: Path = self.server.static_dir
        target = (static_root / relative_path).resolve()
        try:
            target.relative_to(static_root.resolve())
        except ValueError:
            self._send_json({"error": "invalid asset path"}, HTTPStatus.BAD_REQUEST)
            return
        if not target.is_file():
            self._send_json({"error": "asset not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
                "application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self._send_bytes(target.read_bytes(), content_type)

    def _serve_artifact(self, encoded_path: str):
        relative_path = urllib.parse.unquote(encoded_path).lstrip("/")
        target = (self.server.run_dir / relative_path).resolve()
        try:
            target.relative_to(self.server.run_dir.resolve())
        except ValueError:
            self._send_json({"error": "invalid artifact path"}, HTTPStatus.BAD_REQUEST)
            return
        if not target.is_file() or target.suffix.lower() not in TEXT_ARTIFACT_SUFFIXES:
            self._send_json({"error": "artifact is not a supported text file"},
                            HTTPStatus.NOT_FOUND)
            return
        try:
            if target.stat().st_size > MAX_ARTIFACT_BYTES:
                self._send_json({"error": "artifact is too large to display"},
                                HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            payload = target.read_bytes()
        except OSError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "text/plain"
        self._send_bytes(payload, f"{content_type}; charset=utf-8")

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/state":
            try:
                self._send_json(build_graph_state(self.server.run_dir))
            except Exception as exc:  # Keep the viewer alive while files change.
                self._send_json(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path.startswith("/artifact/"):
            self._serve_artifact(parsed.path[len("/artifact/"):])
            return
        if parsed.path in {"/", "/index.html"}:
            self._serve_static("index.html")
            return
        if parsed.path.startswith("/assets/"):
            self._serve_static(parsed.path.lstrip("/"))
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


class ExperimentGraphServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, run_dir: Path, static_dir: Path, quiet: bool = False):
        super().__init__(address, ExperimentGraphHandler)
        self.run_dir = run_dir.resolve()
        self.static_dir = static_dir.resolve()
        self.quiet = quiet


def default_static_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "visualization" / "experiment-graph"


def serve(run_dir: str | os.PathLike[str], host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = False, quiet: bool = False,
          static_dir: str | os.PathLike[str] | None = None):
    root = Path(run_dir).expanduser().resolve()
    assets = Path(static_dir).expanduser().resolve() if static_dir else default_static_dir()
    if not (assets / "index.html").is_file():
        raise FileNotFoundError(f"dashboard assets are missing: {assets}")
    server = ExperimentGraphServer((host, port), root, assets, quiet=quiet)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Experiment graph: {url}", flush=True)
    print(f"Reading: {root}", flush=True)
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Show a live, read-only visualization of one experiment graph.")
    parser.add_argument("--run-dir", required=True,
                        help="Experiment directory containing frontier.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765,
                        help="HTTP port; use 0 to select a free port")
    parser.add_argument("--open", action="store_true", dest="open_browser",
                        help="Open the dashboard in the default browser")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    serve(args.run_dir, host=args.host, port=args.port,
          open_browser=args.open_browser, quiet=args.quiet)


if __name__ == "__main__":
    main()
