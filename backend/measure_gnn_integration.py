#!/usr/bin/env python3
"""Measure and independently check the frozen GNN backend integration."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from .gnn_adapter import FrozenGNNAdapter, OUTPUT_LABEL, PROTECTED_SESSION
from .standalone import create_server


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "models" / "frozen" / "gnn_proxy_v1"
STORE_DIR = ROOT / "gnn_proxy_experiment_v1" / "disk_graph_store"
OUTPUT = ROOT / "backend" / "gnn_integration_metrics.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--parity-samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def clean_json(value):
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def payload_from_graph(graph):
    metadata = graph["metadata"]
    return {
        "request_id": "gnn-parity-" + str(metadata["window_id"]),
        "decision_timestamp_session_sec": metadata["decision_session_time_sec"],
        "attacker_driver": metadata["attacker_driver"],
        "target_driver_fixed": metadata["target_driver_fixed"],
        "input_freshness": {
            "join_policy": "backward-only",
            "interpolation": False,
            "asof_tolerance_sec": metadata["asof_tolerance_sec"],
            "lookback_sec": metadata["lookback_sec"],
            "source_timestamp_semantics": "session-relative; approved graph-store sample",
            "source": "existing permitted development graph store",
        },
        "graph": {
            "race_id": metadata["race_id"],
            "drivers": metadata["drivers"],
            "attacker_idx": metadata["attacker_idx"],
            "target_idx": metadata["target_idx"],
            "node_raw": clean_json(graph["node_raw"]),
            "edge_index": clean_json(graph["edge_index"]),
            "edge_raw": clean_json(graph["edge_raw"]),
            "pair_raw": clean_json(graph["pair_raw"]),
            "asof_tolerance_sec": metadata["asof_tolerance_sec"],
            "lookback_sec": metadata["lookback_sec"],
        },
    }


def post_json(base, path, payload):
    request = Request(
        base + path,
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read())


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p / 100.0
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def main():
    args = parse_args()
    os.environ["APEX_GNN_DEVICE"] = args.device
    example = json.loads((ROOT / "examples" / "gnn_predict_request.json").read_text(encoding="utf-8"))
    adapter = FrozenGNNAdapter(args.device)
    started = time.perf_counter()
    status = adapter.manifest_status()
    direct_warmup_ms = (time.perf_counter() - started) * 1000.0
    if status["status"] != "available":
        raise RuntimeError(status)

    direct_latencies = []
    direct_results = []
    for _ in range(args.requests):
        started = time.perf_counter()
        result = adapter.predict_request(example)
        direct_latencies.append((time.perf_counter() - started) * 1000.0)
        direct_results.append(result)
    direct_score = direct_results[0]["proxy_score"]
    if any(result["proxy_score"] != direct_score for result in direct_results):
        raise RuntimeError("repeated direct inference was not stable")

    parity_rows = []
    if args.parity_samples > 0:
        import sys
        sys.path.insert(0, str(ROOT / "gnn_proxy_experiment_v1"))
        from disk_graph_store import DiskGraphStore, LazyGraphDataset
        from run_gnn_experiment import EdgeAwareGNN
        import torch

        store = DiskGraphStore(STORE_DIR)
        indices = []
        for index, metadata in enumerate(store.metadata):
            if PROTECTED_SESSION not in metadata["race_id"]:
                indices.append(index)
            if len(indices) == args.parity_samples:
                break
        prep = json.loads((FROZEN / "preprocessing_state.json").read_text(encoding="utf-8"))
        reference = EdgeAwareGNN(18, 4, 12)
        checkpoint = torch.load(FROZEN / "gnn_seed_42_best.pt", map_location="cpu", weights_only=True)
        reference.load_state_dict(checkpoint["state_dict"])
        reference.eval()
        for index in indices:
            raw = store.get(index)
            payload = payload_from_graph(raw)
            backend_result = adapter.predict_request(payload)
            data = LazyGraphDataset(store, [index], prep)[0]
            data.ptr = torch.tensor([0, data.x.shape[0]], dtype=torch.long)
            with torch.inference_mode():
                reference_score = float(torch.sigmoid(reference(data)).item())
            parity_rows.append({
                "window_id": raw["metadata"]["window_id"],
                "race_id": raw["metadata"]["race_id"],
                "backend_score": backend_result["proxy_score"],
                "reference_score": reference_score,
                "absolute_difference": abs(backend_result["proxy_score"] - reference_score),
            })

    with tempfile.TemporaryDirectory(prefix="apexr-gnn-http-") as temporary:
        server_started = time.perf_counter()
        server = create_server(0, Path(temporary) / "integration.db")
        server_startup_ms = (time.perf_counter() - server_started) * 1000.0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        status_started = time.perf_counter()
        status_code, http_status = (None, None)
        with urlopen(base + "/api/model/status", timeout=30) as response:
            status_code, http_status = response.status, json.loads(response.read())
        http_status_ms = (time.perf_counter() - status_started) * 1000.0
        http_latencies = []
        http_results = []
        for _ in range(args.requests):
            started = time.perf_counter()
            code, result = post_json(base, "/api/model/predict", example)
            http_latencies.append((time.perf_counter() - started) * 1000.0)
            http_results.append((code, result))
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    result = {
        "status": "PASS",
        "device": status["device"],
        "model_status": status,
        "direct_inference": {
            "warmup_ms": round(direct_warmup_ms, 3),
            "repeated_requests": args.requests,
            "p50_ms": round(percentile(direct_latencies, 50), 3),
            "p95_ms": round(percentile(direct_latencies, 95), 3),
            "all_available": all(item["status"] == "available" for item in direct_results),
            "stable_score": True,
            "score": direct_score,
        },
        "http_inference": {
            "server_startup_ms": round(server_startup_ms, 3),
            "status_request_code": status_code,
            "status_request_ms": round(http_status_ms, 3),
            "status_available": http_status.get("status") == "available",
            "repeated_requests": args.requests,
            "p50_ms": round(percentile(http_latencies, 50), 3),
            "p95_ms": round(percentile(http_latencies, 95), 3),
            "all_http_200": all(code == 200 for code, _ in http_results),
            "all_available": all(item["status"] == "available" for _, item in http_results),
            "score": http_results[0][1]["proxy_score"],
        },
        "reference_parity": {
            "samples": parity_rows,
            "max_absolute_difference": max((row["absolute_difference"] for row in parity_rows), default=None),
            "tolerance": 1e-7,
            "passed": bool(parity_rows) and max(row["absolute_difference"] for row in parity_rows) <= 1e-7,
        },
        "output_label": OUTPUT_LABEL,
        "protected_session_accessed": False,
        "training_executed": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
