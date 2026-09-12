"""Inference-only adapter for the frozen APEX-R GNN proxy model.

This module deliberately contains no training imports, optimiser, loss or
backward pass. The HTTP contract accepts the raw graph emitted by the audited
canonical graph builder; it does not accept a simplified speed/gap shortcut.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "models" / "frozen" / "gnn_proxy_v1"
PHASE5 = ROOT / "phase5_dashboard"
OUTPUT_LABEL = "experimental boundary position-swap proxy signal"
PROTECTED_SESSION = "11353"
EXPECTED_MANIFEST_VERSION = "gnn_proxy_v1"
EXPECTED_INPUT_SCHEMA = "gnn-proxy-graph-cache-v1"
ASOF_TOLERANCE_SEC = 1.0
LOOKBACK_SEC = 10.0
NODE_DIM = 18
EDGE_DIM = 4
PAIR_DIM = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _safe_json_text(value: Any) -> str:
    return json.dumps(value, allow_nan=True, separators=(",", ":"), default=str)


def _contains_protected(value: Any) -> bool:
    if isinstance(value, str):
        return PROTECTED_SESSION in value
    if isinstance(value, dict):
        return any(_contains_protected(k) or _contains_protected(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_protected(v) for v in value)
    return False


def _transform(raw: Any, mean: Any, std: Any):
    import numpy as np

    values = np.asarray(raw, dtype=float)
    mean_values = np.asarray(mean, dtype=float)
    std_values = np.asarray(std, dtype=float)
    return (np.where(np.isfinite(values), values, mean_values) - mean_values) / std_values


def _coerce_matrix(value: Any, rows: int, cols: int, name: str):
    import numpy as np

    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} must have shape ({rows}, {cols})")
    output = np.empty((rows, cols), dtype=float)
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"{name} must have shape ({rows}, {cols})")
        for col_index, item in enumerate(row):
            if item is None:
                output[row_index, col_index] = float("nan")
            elif _finite_number(item):
                output[row_index, col_index] = float(item)
            else:
                raise ValueError(f"{name}[{row_index}][{col_index}] must be finite or null")
    return output


def _coerce_vector(value: Any, length: int, name: str):
    import numpy as np

    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must have length {length}")
    output = np.empty(length, dtype=float)
    for index, item in enumerate(value):
        if item is None:
            output[index] = float("nan")
        elif _finite_number(item):
            output[index] = float(item)
        else:
            raise ValueError(f"{name}[{index}] must be finite or null")
    return output


def _normalise_edge_index(value: Any, node_count: int):
    import numpy as np

    if not isinstance(value, list):
        raise ValueError("edge_index must be a list shaped (2,E) or (E,2)")
    if not value:
        return np.empty((2, 0), dtype=np.int64)
    if len(value) == 2 and all(isinstance(row, list) for row in value):
        if len(value[0]) != len(value[1]):
            raise ValueError("edge_index rows must have equal length")
        pairs = list(zip(value[0], value[1]))
    elif all(isinstance(row, list) and len(row) == 2 for row in value):
        pairs = value
    else:
        raise ValueError("edge_index must be a list shaped (2,E) or (E,2)")
    normalised = []
    for index, pair in enumerate(pairs):
        if any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
            raise ValueError(f"edge_index[{index}] must contain integer node indices")
        if any(item < 0 or item >= node_count for item in pair):
            raise ValueError(f"edge_index[{index}] is outside the node range")
        normalised.append((int(pair[0]), int(pair[1])))
    return np.asarray(normalised, dtype=np.int64).T if normalised else np.empty((2, 0), dtype=np.int64)


def _reject_outcome_fields(value: Any, path: str = "graph") -> None:
    forbidden = {"label", "y", "proxy_label", "outcome_status", "outcome", "target_outcome"}
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in forbidden:
                raise ValueError(f"{path}.{key} is not accepted by the inference contract")
            _reject_outcome_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_outcome_fields(child, f"{path}[{index}]")


class EdgeAwareGNN:
    """Exact inference architecture used by the selected checkpoint."""

    @staticmethod
    def build(node_dim: int, edge_dim: int, pair_dim: int):
        import torch
        from torch import nn
        from torch_geometric.nn import GINEConv

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = 32
                self.node_encoder = nn.Linear(node_dim, hidden)
                self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU())
                mlp1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                mlp2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                self.conv1 = GINEConv(mlp1, edge_dim=hidden)
                self.conv2 = GINEConv(mlp2, edge_dim=hidden)
                self.dropout = nn.Dropout(p=0.2)
                self.decoder = nn.Sequential(
                    nn.Linear(hidden * 2 + pair_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(hidden, 1),
                )

            def forward(self, data):
                x = self.node_encoder(data.x)
                edge_attr = self.edge_encoder(data.edge_attr)
                x = self.dropout(torch.relu(self.conv1(x, data.edge_index, edge_attr)))
                x = self.dropout(torch.relu(self.conv2(x, data.edge_index, edge_attr)))
                ptr = data.ptr[:-1]
                attacker = data.att_idx.view(-1).long() + ptr
                target = data.target_idx.view(-1).long() + ptr
                pair = data.pair_x.view(-1, data.pair_x.shape[-1])
                return self.decoder(torch.cat([x[attacker], x[target], pair], dim=1)).squeeze(1)

        return Model()


class FrozenGNNAdapter:
    """Load the frozen model once and expose safe, graph-level inference."""

    def __init__(self, device_preference: str = "auto"):
        self.device_preference = device_preference.lower()
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: str | None = None
        self._torch = None
        self._model = None
        self._prep = None
        self._device = "unavailable"
        self._manifest: dict[str, Any] | None = None
        self._fixture: dict[str, Any] | None = None
        self._sample_payload: dict[str, Any] | None = None
        self._sample_score: float | None = None
        self.model_version = EXPECTED_MANIFEST_VERSION

    @property
    def manifest_path(self) -> Path:
        return FROZEN / "FROZEN_MODEL_MANIFEST.json"

    def _verify_artifacts(self) -> tuple[dict[str, Any], Path]:
        if not self.manifest_path.is_file():
            raise RuntimeError(f"Frozen model manifest not found: {self.manifest_path}")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_version") != EXPECTED_MANIFEST_VERSION:
            raise RuntimeError("Frozen model version mismatch")
        if manifest.get("selected_seed") != 42 or manifest.get("selected_epoch") != 28:
            raise RuntimeError("Frozen model selection metadata mismatch")
        checksum_path = FROZEN / "SHA256SUMS.txt"
        if not checksum_path.is_file():
            raise RuntimeError("Frozen model checksum file is missing")
        checked = 0
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            candidate = FROZEN / name
            if not candidate.is_file() or _sha256(candidate) != expected:
                raise RuntimeError(f"Frozen artifact checksum failed: {name}")
            checked += 1
        if checked == 0:
            raise RuntimeError("Frozen model checksum file is empty")
        checkpoint_path = ROOT / manifest["checkpoint_path"]
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Frozen checkpoint not found: {checkpoint_path}")
        expected_checkpoint = manifest.get("checkpoint_sha256")
        if expected_checkpoint != _sha256(checkpoint_path):
            raise RuntimeError("Frozen checkpoint checksum does not match manifest")
        return manifest, checkpoint_path

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                manifest, checkpoint_path = self._verify_artifacts()
                import numpy as np
                import torch
                from torch_geometric.data import Data

                if self.device_preference not in {"auto", "cpu", "cuda"}:
                    raise RuntimeError("APEX_GNN_DEVICE must be auto, cpu, or cuda")
                if self.device_preference == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
                device = torch.device(
                    "cuda" if self.device_preference == "cuda" or (self.device_preference == "auto" and torch.cuda.is_available()) else "cpu"
                )
                state = torch.load(checkpoint_path, map_location=device, weights_only=True)
                config = state.get("config", {})
                expected_config = {
                    "node_dim": NODE_DIM, "edge_dim": EDGE_DIM, "pair_dim": PAIR_DIM,
                    "hidden": 32, "dropout": 0.2, "optimizer": "Adam", "lr": 0.001,
                    "weight_decay": 0.0001, "batch_size": 32,
                    "loss": "unweighted BCEWithLogitsLoss", "seed": 42, "device": "cuda",
                }
                if config != expected_config:
                    raise RuntimeError("Checkpoint training configuration does not match frozen configuration")
                if state.get("model_class") != "EdgeAwareGNN" or state.get("best_epoch") != 28:
                    raise RuntimeError("Checkpoint model metadata does not match frozen selection")
                model = EdgeAwareGNN.build(NODE_DIM, EDGE_DIM, PAIR_DIM).to(device)
                model.load_state_dict(state["state_dict"])
                model.eval()
                preprocessing = json.loads((FROZEN / "preprocessing_state.json").read_text(encoding="utf-8"))
                fixture_path = FROZEN / "approved_smoke_fixture.json"
                fixture_text = fixture_path.read_text(encoding="utf-8")
                if PROTECTED_SESSION in fixture_text:
                    raise RuntimeError("Protected session identifier found in approved fixture")
                fixture = json.loads(fixture_text)
                graph = fixture["inference_graph"]
                decision = fixture["decision"]
                sample_payload = {
                    "request_id": "frozen-sample-0001",
                    "decision_timestamp_session_sec": decision["session_time_sec"],
                    "attacker_driver": decision["attacker_driver"],
                    "target_driver_fixed": decision["target_driver"],
                    "input_freshness": {
                        "join_policy": "backward-only",
                        "interpolation": False,
                        "asof_tolerance_sec": graph["metadata"]["asof_tolerance_sec"],
                        "lookback_sec": graph["metadata"]["lookback_sec"],
                        "source": "approved local canonical graph builder fixture",
                    },
                    "graph": {
                        "race_id": graph["metadata"]["race_id"],
                        "drivers": graph["metadata"]["drivers"],
                        "attacker_idx": graph["metadata"]["attacker_idx"],
                        "target_idx": graph["metadata"]["target_idx"],
                        "node_raw": graph["node_raw"],
                        "edge_index": graph["edge_index"],
                        "edge_raw": graph["edge_raw"],
                        "pair_raw": graph["pair_raw"],
                        "asof_tolerance_sec": graph["metadata"]["asof_tolerance_sec"],
                        "lookback_sec": graph["metadata"]["lookback_sec"],
                    },
                }
                # Set the runtime handles before validating/building the
                # approved sample through the same request path used by HTTP.
                self._torch = torch
                self._device = str(device)
                data, _quality = self._data_from_payload(sample_payload, preprocessing, Data, np)
                with torch.inference_mode():
                    sample_score = float(torch.sigmoid(model(data)).detach().cpu().item())
                if not math.isfinite(sample_score) or not 0.0 <= sample_score <= 1.0:
                    raise RuntimeError("Frozen model smoke score is not finite and bounded")
                self._manifest = manifest
                self._model = model
                self._prep = preprocessing
                self._fixture = fixture
                self._sample_payload = sample_payload
                self._sample_score = sample_score
            except Exception as error:  # The API reports this as unavailable; no fabricated score is returned.
                self._load_error = f"{type(error).__name__}: {error}"

    def _data_from_payload(self, payload: dict[str, Any], preprocessing: dict[str, Any], Data=None, np=None):
        if Data is None:
            import numpy as np
            from torch_geometric.data import Data
        graph = payload.get("graph")
        if not isinstance(graph, dict):
            raise ValueError("graph must be an object containing the canonical raw graph fields")
        _reject_outcome_fields(graph)
        required = {"race_id", "drivers", "attacker_idx", "target_idx", "node_raw", "edge_index", "edge_raw", "pair_raw", "asof_tolerance_sec", "lookback_sec"}
        missing = sorted(required - set(graph))
        if missing:
            raise ValueError("graph is missing required fields: " + ", ".join(missing))
        race_id = graph["race_id"]
        if not isinstance(race_id, str) or not race_id.strip():
            raise ValueError("graph.race_id must be a non-empty string")
        if _contains_protected(payload):
            raise ValueError("Protected session 11353 is rejected before graph/sample access")
        drivers = graph["drivers"]
        if not isinstance(drivers, list) or not drivers or any(not isinstance(driver, str) or not driver for driver in drivers):
            raise ValueError("graph.drivers must be a non-empty list of driver identifiers")
        if len(set(drivers)) != len(drivers) or drivers != sorted(drivers):
            raise ValueError("graph.drivers must be unique and in the trained sorted identifier order")
        node_count = len(drivers)
        node_raw = _coerce_matrix(graph["node_raw"], node_count, NODE_DIM, "graph.node_raw")
        edge_raw = graph["edge_raw"]
        if not isinstance(edge_raw, list):
            raise ValueError("graph.edge_raw must be a list")
        edge_matrix = []
        for index, row in enumerate(edge_raw):
            if not isinstance(row, list) or len(row) != EDGE_DIM:
                raise ValueError(f"graph.edge_raw[{index}] must have length {EDGE_DIM}")
            edge_matrix.append(row)
        edge_matrix = _coerce_matrix(edge_matrix, len(edge_matrix), EDGE_DIM, "graph.edge_raw") if edge_matrix else np.empty((0, EDGE_DIM), dtype=float)
        edge_index = _normalise_edge_index(graph["edge_index"], node_count)
        if edge_index.shape[1] != edge_matrix.shape[0]:
            raise ValueError("graph.edge_index edge count must equal graph.edge_raw row count")
        pair_raw = _coerce_vector(graph["pair_raw"], PAIR_DIM, "graph.pair_raw")
        attacker = payload.get("attacker_driver")
        target = payload.get("target_driver_fixed")
        if not isinstance(attacker, str) or not isinstance(target, str) or attacker == target:
            raise ValueError("attacker_driver and target_driver_fixed must be distinct strings")
        if attacker not in drivers or target not in drivers:
            raise ValueError("attacker/target identities must be present in graph.drivers")
        attacker_idx, target_idx = graph["attacker_idx"], graph["target_idx"]
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (attacker_idx, target_idx)):
            raise ValueError("graph attacker_idx and target_idx must be integers")
        if not (0 <= attacker_idx < node_count and 0 <= target_idx < node_count):
            raise ValueError("graph attacker_idx/target_idx are outside graph.drivers")
        if drivers[attacker_idx] != attacker or drivers[target_idx] != target:
            raise ValueError("attacker/target identity does not match graph indices")
        timestamp = payload.get("decision_timestamp_session_sec")
        if not _finite_number(timestamp):
            raise ValueError("decision_timestamp_session_sec must be finite")
        for key, expected in (("asof_tolerance_sec", ASOF_TOLERANCE_SEC), ("lookback_sec", LOOKBACK_SEC)):
            value = graph[key]
            if not _finite_number(value) or abs(float(value) - expected) > 1e-9:
                raise ValueError(f"graph.{key} must remain the frozen value {expected}")
        freshness = payload.get("input_freshness")
        if not isinstance(freshness, dict):
            raise ValueError("input_freshness must be an object")
        if freshness.get("join_policy") != "backward-only" or freshness.get("interpolation") is not False:
            raise ValueError("input_freshness must declare backward-only matching with interpolation=false")
        if not _finite_number(freshness.get("asof_tolerance_sec")) or abs(float(freshness["asof_tolerance_sec"]) - ASOF_TOLERANCE_SEC) > 1e-9:
            raise ValueError("input_freshness.asof_tolerance_sec must be 1.0")
        if not _finite_number(freshness.get("lookback_sec")) or abs(float(freshness["lookback_sec"]) - LOOKBACK_SEC) > 1e-9:
            raise ValueError("input_freshness.lookback_sec must be 10.0")
        # Columns 6 and 10 are current car/position sample ages; columns 13–15
        # are the trained explicit missingness masks.
        for index, name in ((attacker_idx, attacker), (target_idx, target)):
            for column, column_name in ((6, "car_sample_age_sec"), (10, "position_sample_age_sec")):
                age = node_raw[index, column]
                if not math.isfinite(float(age)) or age < 0.0 or age > ASOF_TOLERANCE_SEC + 1e-9:
                    raise ValueError(f"{name} {column_name} is missing or stale beyond the 1-second as-of tolerance")
        pair_position_available = pair_raw[11]
        if not math.isfinite(float(pair_position_available)) or float(pair_position_available) < 0.5:
            raise ValueError("fixed attacker/target position relationship is unavailable")
        # Preserve missingness as NaN for the frozen transform; do not replace
        # missing observations with zero. Missing nonessential cars become warnings.
        warnings = []
        nonessential_missing = int(np.isnan(node_raw[:, 0]).sum()) + int(np.isnan(node_raw[:, 7:10]).all(axis=1).sum())
        if nonessential_missing:
            warnings.append(f"{nonessential_missing} nonessential node observation groups are missing and remain masked")
        x = _transform(node_raw, preprocessing["node_mean"], preprocessing["node_std"])
        e = _transform(edge_matrix, preprocessing["edge_mean"], preprocessing["edge_std"])
        p = _transform(pair_raw, preprocessing["pair_mean"], preprocessing["pair_std"])
        edge_index_tensor = self._torch.tensor(edge_index, dtype=self._torch.long, device=self._device)
        data = Data(
            x=self._torch.tensor(x, dtype=self._torch.float32, device=self._device),
            edge_index=edge_index_tensor,
            edge_attr=self._torch.tensor(e, dtype=self._torch.float32, device=self._device),
            pair_x=self._torch.tensor(p, dtype=self._torch.float32, device=self._device).reshape(1, -1),
            att_idx=self._torch.tensor([attacker_idx], dtype=self._torch.long, device=self._device),
            target_idx=self._torch.tensor([target_idx], dtype=self._torch.long, device=self._device),
            ptr=self._torch.tensor([0, node_count], dtype=self._torch.long, device=self._device),
        )
        return data, {
            "status": "valid",
            "warnings": warnings,
            "node_count": node_count,
            "edge_count": int(edge_matrix.shape[0]),
            "attacker_car_age_sec": float(node_raw[attacker_idx, 6]),
            "target_car_age_sec": float(node_raw[target_idx, 6]),
            "attacker_position_age_sec": float(node_raw[attacker_idx, 10]),
            "target_position_age_sec": float(node_raw[target_idx, 10]),
            "missing_node_groups": nonessential_missing,
        }

    def manifest_status(self) -> dict[str, Any]:
        self._load()
        manifest = self._manifest or {}
        return {
            "status": "available" if self._sample_score is not None else "unavailable",
            "model_version": manifest.get("model_version", self.model_version),
            "task": manifest.get("task", "fixed-pair next-lap-boundary classified-order position-swap proxy"),
            "checkpoint": Path(manifest.get("checkpoint_path", "gnn_seed_42_best.pt")).name,
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "selected_seed": manifest.get("selected_seed", 42),
            "selected_epoch": manifest.get("selected_epoch", 28),
            "input_schema_version": EXPECTED_INPUT_SCHEMA,
            "device": self._device,
            "loaded": self._sample_score is not None,
            "failure_reason": self._load_error,
            "output_label": OUTPUT_LABEL,
            "action_influence": "zero; advisory only",
        }

    def predict_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_start = time.perf_counter()
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        if not isinstance(request_id, str) or not request_id:
            request_id = "gnn-" + uuid.uuid4().hex
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        # This guard runs before graph conversion or any sample lookup.
        if _contains_protected(payload):
            raise ValueError("Protected session 11353 is rejected before graph/sample access")
        self._load()
        if self._sample_score is None:
            return {
                "status": "unavailable",
                "request_id": request_id,
                "model_version": self.model_version,
                "score_label": OUTPUT_LABEL,
                "proxy_score": None,
                "score": None,
                "inference_latency_ms": round((time.perf_counter() - request_start) * 1000.0, 3),
                "failure_reason": self._load_error or "Frozen model is unavailable",
                "reason": self._load_error or "Frozen model is unavailable",
            }
        _reject_outcome_fields(payload.get("graph", {}))
        data, quality = self._data_from_payload(payload, self._prep)
        with self._torch.inference_mode():
            score = float(self._torch.sigmoid(self._model(data)).detach().cpu().item())
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            return {
                "status": "unavailable", "request_id": request_id,
                "model_version": self.model_version, "score_label": OUTPUT_LABEL,
                "proxy_score": None,
                "score": None,
                "inference_latency_ms": round((time.perf_counter() - request_start) * 1000.0, 3),
                "failure_reason": "Model returned a non-finite or out-of-range score",
                "reason": "Model returned a non-finite or out-of-range score",
            }
        freshness = payload["input_freshness"]
        return {
            "status": "available",
            "request_id": request_id,
            "decision_timestamp_session_sec": float(payload["decision_timestamp_session_sec"]),
            "attacker_driver": payload["attacker_driver"],
            "target_driver_fixed": payload["target_driver_fixed"],
            "model_version": self.model_version,
            "checkpoint_sha256": self.manifest_status()["checkpoint_sha256"],
            "score_label": OUTPUT_LABEL,
            "proxy_score": score,
            # Backward-compatible read-only alias for the existing Phase 5
            # display; both fields carry the same raw proxy score.
            "score": score,
            "input_quality": quality,
            "input_freshness": {
                "join_policy": freshness["join_policy"],
                "interpolation": False,
                "asof_tolerance_sec": ASOF_TOLERANCE_SEC,
                "lookback_sec": LOOKBACK_SEC,
                "source_timestamp_semantics": freshness.get("source_timestamp_semantics", "session-relative; supplied by canonical graph builder"),
            },
            "inference_latency_ms": round((time.perf_counter() - request_start) * 1000.0, 3),
            "not_a_calibrated_overtake_probability": True,
            "not_an_attack_benefit_probability": True,
            "strategy_action": None,
        }

    def predict(self, window_id: str | None = None) -> dict[str, Any]:
        """Compatibility path for the existing historical Phase 5 route."""
        self._load()
        if self._sample_payload is None:
            status = self.manifest_status()
            return {
                "status": "unavailable", "model_version": self.model_version,
                "score_label": OUTPUT_LABEL, "proxy_score": None,
                "score": None,
                "reason": status.get("failure_reason") or "No approved graph is available",
            }
        expected_window = self._sample_payload["graph"].get("window_id")
        if window_id not in (None, expected_window, "ee38ba6e51b8d9dbba9b"):
            return {
                "status": "unavailable", "model_version": self.model_version,
                "score_label": OUTPUT_LABEL, "proxy_score": None,
                "score": None,
                "reason": "No frozen graph is available for this replay decision window",
            }
        return self.predict_request(self._sample_payload)

    def parity(self) -> dict[str, Any]:
        self._load()
        status = self.manifest_status()
        return {
            "status": status["status"],
            "score": self._sample_score,
            "device": self._device,
            "comparison": "Frozen package transform and forward architecture are used for the approved graph fixture.",
        }
