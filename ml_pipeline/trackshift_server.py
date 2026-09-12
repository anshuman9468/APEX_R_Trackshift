#!/usr/bin/env python3
"""Serve the Trackshift frontend with the frozen hybrid model only.

This adapter deliberately does not import or execute the cloned repository's
backend, XGBoost artifact, frozen GNN proxy, or phase-5 model.  It loads the
local Phase 3 test bundle and exposes only the model status and real inference
preview endpoints needed by the copied frontend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch

try:
    from ml_pipeline.train_hybrid import (
        GraphPartition,
        HybridGNN,
        Normalizer,
        choose_device,
        collate,
        move_batch,
    )
except ModuleNotFoundError:  # direct execution: python ml_pipeline/trackshift_server.py
    from train_hybrid import GraphPartition, HybridGNN, Normalizer, choose_device, collate, move_batch


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "dist"
BUNDLE_PATH = ROOT / "deployment_packages" / "hybrid_gnn_epoch51" / "model_test_bundle.pt"


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value)))))


def json_scalar(value):
    """Convert NumPy scalars and evaluation NaNs into JSON-safe values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class BundlePartition:
    """Expose the bundled graph arrays through the existing model sampler."""

    def __init__(self, bundle: dict):
        data = bundle["graph_dataset"]
        self.path = ROOT / "gnn_sequences" / "2026" / "validation"
        self.season = int(bundle["test_definition"]["year"])
        self.split = bundle["test_definition"]["split"]
        self.x = data["x"]
        self.edge_index = data["edge_index"]
        self.edge_attr = data["edge_attr"]
        self.snapshot_ptr = data["snapshot_ptr"]
        self.edge_ptr = data["edge_ptr"]
        self.sequence_snapshot_ids = data["sequence_snapshot_ids"]
        self.target_ptr = data["target_ptr"]
        self.target_node_indices = data["target_node_indices"]
        self.target_time = data["target_lap_time_s"]
        self.target_delta = data["target_lap_time_delta_s"]
        self.target_wear = data["target_tyre_wear_estimate_fraction"]
        self.target_valid = data["target_valid"]
        self.target_high_tyre = data["target_high_tyre_degradation"]
        self.target_pit = data["target_pit_stop_observed"]
        self.target_safety = data["target_safety_constraint"]
        self.target_class_valid = data["target_classification_valid"]
        self.node_driver_index = data["node_driver_index"]
        self.snapshots = bundle["snapshots"]

    def __len__(self) -> int:
        return len(self.sequence_snapshot_ids)

    def sample(self, sequence_id: int) -> dict:
        # Reuse the audited production sampler from GraphPartition without
        # reading a second on-disk dataset or any future target values.
        sampler = object.__new__(GraphPartition)
        sampler.path = self.path
        sampler.season = self.season
        sampler.split = self.split
        sampler.x = self.x
        sampler.edge_index = self.edge_index
        sampler.edge_attr = self.edge_attr
        sampler.snapshot_ptr = self.snapshot_ptr
        sampler.edge_ptr = self.edge_ptr
        sampler.sequence_snapshot_ids = self.sequence_snapshot_ids
        sampler.target_ptr = self.target_ptr
        sampler.target_node_indices = self.target_node_indices
        sampler.target_time = self.target_time
        sampler.target_delta = self.target_delta
        sampler.target_wear = self.target_wear
        sampler.target_valid = self.target_valid
        sampler.target_high_tyre = self.target_high_tyre
        sampler.target_pit = self.target_pit
        sampler.target_safety = self.target_safety
        sampler.target_class_valid = self.target_class_valid
        sampler.node_driver_index = self.node_driver_index
        sampler.snapshots = self.snapshots
        return sampler.sample(sequence_id)


class HybridRuntime:
    def __init__(self, bundle_path: Path):
        self.bundle_path = bundle_path
        self.bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        config = self.bundle["config"]
        normalizer_config = config["normalizer"]
        self.normalizer = Normalizer(
            np.asarray(normalizer_config["x_mean"], dtype=np.float32),
            np.asarray(normalizer_config["x_std"], dtype=np.float32),
            np.asarray(normalizer_config["edge_mean"], dtype=np.float32),
            np.asarray(normalizer_config["edge_std"], dtype=np.float32),
        )
        requested_device = os.environ.get("APEX_TRACKSHIFT_DEVICE", "auto")
        self.device = choose_device(requested_device)
        self.model = HybridGNN(
            hidden=int(config.get("hidden", 128)),
            layers=int(config.get("layers", 3)),
            heads=int(config.get("heads", 4)),
            dropout=float(config.get("dropout", 0.15)),
        ).to(self.device)
        self.model.load_state_dict(self.bundle["model"], strict=True)
        self.model.eval()
        self.partition = BundlePartition(self.bundle)
        self.rows = self.bundle["predictions_and_labels"]
        self.calibration = self.bundle["calibration"]
        self.driver_vocabulary = self.bundle["graph_manifest"].get("driver_vocabulary", [])
        self.report = self.bundle["replay_report"]

    def status(self) -> dict:
        first_run = self.report["replay_runs"][0]
        test_definition = dict(self.bundle["test_definition"])
        test_definition["prediction_file_source"] = "embedded in model_test_bundle.pt"
        test_definition["graph_directory_source"] = "embedded graph_dataset in model_test_bundle.pt"
        return {
            "status": "available",
            "model_version": "hybrid_gnn_physics_epoch51",
            "model_type": "GNN + GRU + physics consistency",
            "epoch": self.bundle["best_epoch"],
            "device": str(self.device),
            "test": test_definition,
            "metrics": first_run["metrics"],
            "replay_decision": self.report["deployment_decision"],
            "bundle": self.bundle_path.name,
        }

    def _calibrated(self, logit: float, head: str) -> float:
        calibration = self.calibration[head]
        return sigmoid(float(calibration["platt_intercept"]) + float(calibration["platt_coefficient"]) * logit)

    def _driver_name(self, index: int) -> str:
        if 0 <= index < len(self.driver_vocabulary):
            return str(self.driver_vocabulary[index])
        return f"DRIVER_{index}"

    @torch.no_grad()
    def predict_sequence(self, sequence_id: int) -> dict:
        if not 0 <= sequence_id < len(self.partition):
            raise ValueError(f"sequence must be between 0 and {len(self.partition) - 1}")
        sample = self.partition.sample(sequence_id)
        batch = collate([sample], self.normalizer)
        outputs = self.model(move_batch(batch, self.device))
        arrays = {key: value.detach().cpu().numpy() for key, value in outputs.items()}
        target_start = int(self.partition.target_ptr[sequence_id])
        target_end = int(self.partition.target_ptr[sequence_id + 1])
        target_nodes = self.partition.target_node_indices[target_start:target_end]
        threshold = {
            head: float(self.calibration[head]["thresholds"]["f1_threshold"]["threshold"])
            for head in ("tyre", "pit", "safety")
        }
        predictions = []
        for index in range(target_end - target_start):
            global_row = target_start + index
            tyre_probability = self._calibrated(float(arrays["tyre_logit"][index]), "tyre")
            pit_probability = self._calibrated(float(arrays["pit_logit"][index]), "pit")
            safety_probability = self._calibrated(float(arrays["safety_logit"][index]), "safety")
            target_row = {key: json_scalar(self.rows[key][global_row]) for key in self.rows}
            predictions.append({
                "row_index": global_row,
                "driver": self._driver_name(int(self.partition.node_driver_index[target_nodes[index]])),
                "next_lap_time_s": float(arrays["time"][index]),
                "next_lap_time_delta_s": float(arrays["delta"][index]),
                "next_tyre_wear_estimate_fraction": float(arrays["wear"][index]),
                "tyre_degradation_probability": tyre_probability,
                "pit_stop_probability": pit_probability,
                "safety_constraint_probability": safety_probability,
                "tyre_degradation_alert": tyre_probability >= threshold["tyre"],
                "pit_stop_alert": pit_probability >= threshold["pit"],
                "safety_constraint_alert": safety_probability >= threshold["safety"],
                "actual_test_labels": {
                    "next_lap_time_s": target_row.get("target_next_lap_time_s"),
                    "high_tyre_degradation": target_row.get("target_high_tyre_degradation"),
                    "pit_stop_observed": target_row.get("target_pit_stop_observed"),
                    "safety_constraint": target_row.get("target_safety_constraint"),
                },
                "quality": {
                    "physics_confidence": target_row.get("physics_confidence"),
                    "weather_confidence": target_row.get("weather_confidence"),
                    "traffic_constraint_factor": target_row.get("traffic_constraint_factor"),
                },
            })
        return {
            "status": "available",
            "model_version": "hybrid_gnn_physics_epoch51",
            "source": "model_test_bundle.pt graph sequence",
            "test_context": {
                "year": self.bundle["test_definition"]["year"],
                "split": self.bundle["test_definition"]["split"],
                "event": sample["event"],
                "sequence": sequence_id,
            },
            "thresholds": threshold,
            "predictions": predictions,
        }


def json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, runtime: HybridRuntime, **kwargs):
        self.runtime = runtime
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        route = urlsplit(self.path)
        try:
            if route.path == "/api/health":
                return json_response(self, {"status": "ok", "frontend": "Trackshift dist only", "hybrid_model": self.runtime.status()})
            if route.path in {"/api/model/status", "/api/hybrid/status"}:
                return json_response(self, self.runtime.status())
            if route.path in {"/api/model/metrics", "/api/hybrid/metrics"}:
                return json_response(self, self.runtime.status()["metrics"])
            if route.path in {"/api/model/predict", "/api/hybrid/predict"}:
                raw = parse_qs(route.query).get("sequence", ["0"])[0]
                try:
                    sequence_id = int(raw)
                except ValueError as error:
                    raise ValueError("sequence must be an integer") from error
                return json_response(self, self.runtime.predict_sequence(sequence_id))
            if route.path.startswith("/api/"):
                return json_response(self, {"status": "unavailable", "reason": "Only hybrid model endpoints are enabled"}, 404)
            return super().do_GET()
        except (ValueError, TypeError) as error:
            return json_response(self, {"status": "error", "detail": str(error)}, 422)
        except Exception as error:  # explicit local service failure
            return json_response(self, {"status": "unavailable", "reason": str(error)}, 503)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Trackshift frontend with the frozen hybrid model")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if not FRONTEND.is_dir():
        parser.error(f"frontend directory not found: {FRONTEND}")
    if not BUNDLE_PATH.is_file():
        parser.error(f"model test bundle not found: {BUNDLE_PATH}")
    try:
        runtime = HybridRuntime(BUNDLE_PATH)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, runtime=runtime))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Cannot start Trackshift hybrid app: {error}")
        return 1
    print(f"Trackshift frontend: http://127.0.0.1:{args.port}/", flush=True)
    print(json.dumps(runtime.status(), indent=2), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
