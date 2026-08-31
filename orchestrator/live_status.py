"""Small heartbeat file for read-only live experiment viewers."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone


class LiveRunStatus:
    """Publish phase and liveness without affecting experiment decisions."""

    def __init__(self, run_dir, run_id, interval_s=3.0):
        self.path = os.path.join(os.path.abspath(run_dir), "live-status.json")
        self.run_id = str(run_id)
        self.interval_s = float(interval_s)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._state = {
            "schema_version": "live-run-status-1.0",
            "run_id": self.run_id,
            "pid": os.getpid(),
            "status": "starting",
            "phase": "Starting the experiment",
            "iteration": None,
            "detail": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def _write(self):
        with self._lock:
            payload = dict(self._state)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["updated_unix_s"] = time.time()
        temporary = self.path + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(temporary, self.path)
        except OSError:
            # A presentation aid must never stop or change an experiment.
            try:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            except OSError:
                pass

    def _heartbeat(self):
        while not self._stop_event.wait(self.interval_s):
            self._write()

    def start(self, phase="Starting the experiment"):
        with self._lock:
            self._state.update(status="running", phase=str(phase))
        self._write()
        self._thread = threading.Thread(
            target=self._heartbeat, name="experiment-graph-heartbeat", daemon=True)
        self._thread.start()
        return self

    def update(self, phase, iteration=None, detail=None):
        with self._lock:
            self._state.update(
                status="running", phase=str(phase), iteration=iteration, detail=detail)
        self._write()

    def stop(self, status="complete", detail=None):
        with self._lock:
            self._state.update(status=str(status), phase="Experiment finished", detail=detail)
        self._stop_event.set()
        self._write()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.interval_s + 0.5))

