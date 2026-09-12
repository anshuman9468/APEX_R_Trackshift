#!/usr/bin/env python3
"""Create the small, label-free example request for /api/model/predict."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "models" / "frozen" / "gnn_proxy_v1"
SOURCE = FROZEN / "approved_smoke_fixture.json"
OUTPUT = ROOT / "examples" / "gnn_predict_request.json"


def main() -> None:
    fixture = json.loads(SOURCE.read_text(encoding="utf-8"))
    graph = fixture["inference_graph"]
    metadata = graph["metadata"]
    request = {
        "request_id": "gnn-demo-0001",
        "decision_timestamp_session_sec": metadata["decision_session_time_sec"],
        "attacker_driver": metadata["attacker_driver"],
        "target_driver_fixed": metadata["target_driver_fixed"],
        "input_freshness": {
            "join_policy": "backward-only",
            "interpolation": False,
            "asof_tolerance_sec": metadata["asof_tolerance_sec"],
            "lookback_sec": metadata["lookback_sec"],
            "source_timestamp_semantics": "session-relative; emitted by approved canonical graph builder",
            "source": "approved local validation graph fixture",
        },
        "graph": {
            "race_id": metadata["race_id"],
            "drivers": metadata["drivers"],
            "attacker_idx": metadata["attacker_idx"],
            "target_idx": metadata["target_idx"],
            "node_raw": graph["node_raw"],
            "edge_index": graph["edge_index"],
            "edge_raw": graph["edge_raw"],
            "pair_raw": graph["pair_raw"],
            "asof_tolerance_sec": metadata["asof_tolerance_sec"],
            "lookback_sec": metadata["lookback_sec"],
        },
    }
    if "11353" in json.dumps(request, sort_keys=True):
        raise RuntimeError("protected session identifier found in example request")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
