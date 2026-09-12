"""Phase 5 historical replay and frozen advisory service."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .gnn_adapter import FrozenGNNAdapter, PHASE5


class Phase5Service:
    def __init__(self):
        self.fixture_path = PHASE5 / "fixture" / "phase5_fixture.json"
        self.fixture = json.loads(self.fixture_path.read_text()) if self.fixture_path.exists() else None
        self.gnn = FrozenGNNAdapter()

    def metadata(self):
        if not self.fixture:
            return {"status": "unavailable", "reason": "Historical fixture not installed"}
        d = self.fixture["decision"]
        return {"status": "ok", "schema_version": self.fixture["schema_version"],
                "race": self.fixture["race"], "selection": self.fixture["selection"],
                "timestamp_semantics": self.fixture["timestamp_semantics"],
                "decision": {k: d[k] for k in ("window_id", "session_time_sec", "replay_time_sec", "lap", "attacker_driver", "target_driver")},
                "gnn": self.gnn.manifest_status()}

    def replay(self):
        return self.fixture or {"status": "unavailable", "reason": "Historical fixture not installed"}

    def state(self, replay_time: float):
        if not self.fixture:
            return {"status": "unavailable", "reason": "Historical fixture not installed"}
        if isinstance(replay_time, bool) or not isinstance(replay_time, (int, float)) or not math.isfinite(replay_time):
            raise ValueError("replay time must be finite")
        frames = self.fixture["replay"]["frames"]
        t = max(0.0, min(float(replay_time), float(self.fixture["replay"]["duration_sec"])))
        frame = max((f for f in frames if f["t"] <= t), key=lambda f: f["t"], default=frames[0])
        decision = self.fixture["decision"]
        near_decision = abs(float(frame["session_time_sec"]) - float(decision["session_time_sec"])) <= 1.0
        return {"status": "ok", "source": "observed historical telemetry reference",
                "frame": frame, "fixed_pair": {"attacker": decision["attacker_driver"], "target": decision["target_driver"]},
                "model": self.gnn.predict(decision["window_id"]) if near_decision else {
                    "status": "not_evaluated", "score_label": "experimental boundary position-swap score",
                    "reason": "Frozen graph is defined at the selected decision cadence; no second-by-second claim"},
                "simulated_variables": ["battery energy", "branch positions", "branch gaps"]}

    def inference(self, window_id=None):
        return self.gnn.predict(window_id)
