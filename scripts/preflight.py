#!/usr/bin/env python3
"""Fast, dependency-free checks for the public Track 2 submission package."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SUBMISSION_ROWS = 170_588
EXPECTED_SUBMISSION_SHA256 = (
    "eb1db430f8fdce35bcc3dd0c8eacc68565d3bc75dd667f841bc560816bc4aa50"
)
REQUIRED = (
    "README.md",
    ".env.example",
    "candidate/pipeline.py",
    "orchestrator/agent.py",
    "orchestrator/graph_view.py",
    "orchestrator/live_status.py",
    "scripts/serve_experiment_graph.py",
    "visualization/experiment-graph/index.html",
    "visualization/experiment-graph/assets/styles.css",
    "visualization/experiment-graph/assets/app.js",
    "trusted/evaluator.py",
    "task_spec/evaluate.py",
    "empirical_priors/track2-solutions-valid-only-v1.json",
    "env.lock.json",
    "manifest.json",
    "artifacts/final/weights-by-tab.json",
    "artifacts/final/results.json",
    "artifacts/final/submission.csv",
    "artifacts/experiment-records/config.json",
    "artifacts/experiment-records/summary.json",
    "artifacts/experiment-records/journal.jsonl",
)
EXCLUDED_DIR_NAMES = {
    ".orchestrator-venv",
    ".venv",
    "__pycache__",
    "runs",
    "trusted_cache",
    "venv",
    "views",
}
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}"),
    re.compile(r"(?m)^OPENAI_API_KEY=.+$"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"FAIL  {message}")


def check_json(errors: list[str]) -> None:
    for path in sorted(ROOT.rglob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # report every malformed artifact in one pass
            fail(f"invalid JSON: {path.relative_to(ROOT)} ({exc})", errors)

    for path in sorted(ROOT.rglob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception as exc:
                fail(
                    f"invalid JSONL: {path.relative_to(ROOT)}:{line_number} ({exc})",
                    errors,
                )


def check_python(errors: list[str]) -> None:
    for path in sorted(ROOT.rglob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except Exception as exc:
            fail(f"invalid Python: {path.relative_to(ROOT)} ({exc})", errors)


def check_dashboard(errors: list[str]) -> None:
    index_path = ROOT / "visualization/experiment-graph/index.html"
    script_path = ROOT / "visualization/experiment-graph/assets/app.js"
    style_path = ROOT / "visualization/experiment-graph/assets/styles.css"
    if not all(path.is_file() for path in (index_path, script_path, style_path)):
        return
    index = index_path.read_text(encoding="utf-8")
    script = script_path.read_text(encoding="utf-8")
    styles = style_path.read_text(encoding="utf-8")
    required_ids = {
        "graph-svg", "graph-scene", "edge-layer", "node-layer", "detail-content",
        "timeline-list", "combination-content", "live-pill",
    }
    observed_ids = set(re.findall(r'\bid="([^"]+)"', index))
    missing_ids = sorted(required_ids - observed_ids)
    if missing_ids:
        fail(f"dashboard HTML is missing element IDs: {missing_ids}", errors)
    referenced_ids = set(re.findall(r'\bel\("([^"]+)"\)', script))
    dangling_ids = sorted(referenced_ids - observed_ids)
    if dangling_ids:
        fail(f"dashboard JavaScript references missing IDs: {dangling_ids}", errors)
    if "fetch(`/api/state" not in script or "setInterval" not in script:
        fail("dashboard is missing live state refresh", errors)
    if "--page:" not in styles or "color-scheme" not in index:
        fail("dashboard light-theme markers are missing", errors)
    external_assets = re.findall(r'(?:src|href)="(https?://[^"]+)"', index)
    if external_assets:
        fail(f"dashboard must work offline; external assets found: {external_assets}", errors)


def check_submission(errors: list[str]) -> None:
    path = ROOT / "artifacts/final/submission.csv"
    if not path.is_file():
        return

    observed_hash = sha256(path)
    if observed_hash != EXPECTED_SUBMISSION_SHA256:
        fail(f"submission SHA-256 changed: {observed_hash}", errors)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["row_id", "user_id", "video_id", "score"]:
            fail(f"unexpected submission header: {header}", errors)
        rows = sum(1 for _ in reader)

    if rows != EXPECTED_SUBMISSION_ROWS:
        fail(f"expected {EXPECTED_SUBMISSION_ROWS} submission rows, found {rows}", errors)


def check_package_hygiene(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_dir() and path.name in EXCLUDED_DIR_NAMES:
            fail(f"generated directory should not be submitted: {relative}", errors)
        if path.is_file() and path.name == ".env":
            fail(".env contains local credentials and must not be submitted", errors)

        if not path.is_file() or path.suffix.lower() in {".csv", ".npy", ".png", ".jpg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                fail(f"possible secret in {relative}", errors)


def main() -> int:
    errors: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            fail(f"missing required file: {relative}", errors)

    check_json(errors)
    check_python(errors)
    check_dashboard(errors)
    check_submission(errors)
    check_package_hygiene(errors)

    if errors:
        print(f"\nPreflight failed with {len(errors)} issue(s).")
        return 1

    print("PASS  required files present")
    print("PASS  JSON, JSONL, and Python syntax")
    print("PASS  no generated environments, caches, or obvious secrets")
    print(
        "PASS  final submission: "
        f"{EXPECTED_SUBMISSION_ROWS:,} rows, SHA-256 {EXPECTED_SUBMISSION_SHA256}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
