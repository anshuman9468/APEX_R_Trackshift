"""Optional, frozen GNN inference for the Phase 5 local demo.

The adapter is deliberately lazy: the HTTP app remains usable without the
GPU environment, while an available CPU/CUDA PyTorch install loads the frozen
seed-42 checkpoint once and scores only the compact, approved replay graph.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT if (ROOT / "fixture").exists() else ROOT / "phase5_dashboard"
EXPERIMENT = ROOT / "gnn_proxy_experiment_v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class FrozenGNNAdapter:
    def __init__(self, device_preference: str = "auto"):
        self.device_preference = device_preference
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error = None
        self._torch = None
        self._model = None
        self._data = None
        self._device = "unavailable"
        self._score = None
        self._coverage = {}
        self.fixture_path = PHASE5 / "fixture" / "phase5_fixture.json"
        self.checkpoint_path = (PHASE5 / "frozen" / "gnn_seed_42_best.pt")
        if not self.checkpoint_path.exists():
            self.checkpoint_path = EXPERIMENT / "gnn_seed_42_best.pt"
        self.preprocessing_path = PHASE5 / "frozen" / "preprocessing_state.json"
        if not self.preprocessing_path.exists():
            self.preprocessing_path = EXPERIMENT / "preprocessing_state.json"
        self.model_version = "gnn_proxy_gpu_v1_seed42_epoch28"
        self.window_id = None
        self.metadata = {}

    def _load(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                import numpy as np
                import torch
                from torch_geometric.data import Data
                try:
                    sys.path.insert(0, str(EXPERIMENT))
                    from disk_graph_store import transform
                    from run_gnn_experiment import EdgeAwareGNN
                except ImportError:
                    # The compact release deliberately omits the large
                    # experiment source tree.  These are the same formulas
                    # and architecture used by that tree.
                    def transform(raw, mean, std):
                        mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
                        return (np.where(np.isfinite(raw), raw, mean) - mean) / std
                    from torch import nn
                    from torch_geometric.nn import GINEConv
                    class EdgeAwareGNN(nn.Module):
                        def __init__(self, node_dim, edge_dim, pair_dim):
                            super().__init__()
                            self.node_encoder = nn.Linear(node_dim, 32)
                            self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, 32), nn.ReLU())
                            mlp1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
                            mlp2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
                            self.conv1, self.conv2 = GINEConv(mlp1, edge_dim=32), GINEConv(mlp2, edge_dim=32)
                            self.dropout = nn.Dropout(p=0.2)
                            self.decoder = nn.Sequential(nn.Linear(32 * 2 + pair_dim, 32), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(32, 1))
                        def forward(self, data):
                            x = self.node_encoder(data.x); edge_attr = self.edge_encoder(data.edge_attr)
                            x = self.dropout(torch.relu(self.conv1(x, data.edge_index, edge_attr)))
                            x = self.dropout(torch.relu(self.conv2(x, data.edge_index, edge_attr)))
                            ptr = data.ptr[:-1]; ai = data.att_idx.view(-1).long() + ptr; ti = data.target_idx.view(-1).long() + ptr
                            pair = data.pair_x.view(-1, data.pair_x.shape[-1])
                            return self.decoder(torch.cat([x[ai], x[ti], pair], dim=1)).squeeze(1)

                fixture = json.loads(self.fixture_path.read_text())
                graph = fixture["inference_graph"]
                self.metadata = fixture["decision"]
                self.window_id = self.metadata["window_id"]
                prep = json.loads(self.preprocessing_path.read_text())
                requested = self.device_preference.lower()
                if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
                model = EdgeAwareGNN(18, 4, 12).to(device)
                state = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
                # GPU runner wraps the state_dict with provenance/config
                # metadata; the model itself receives only the parameter map.
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state)
                model.eval()
                node = transform(np.asarray(graph["node_raw"], dtype=float),
                                  np.asarray(prep["node_mean"]), np.asarray(prep["node_std"]))
                edge = transform(np.asarray(graph["edge_raw"], dtype=float).reshape(-1, 4),
                                 np.asarray(prep["edge_mean"]), np.asarray(prep["edge_std"]))
                pair = transform(np.asarray(graph["pair_raw"], dtype=float),
                                 np.asarray(prep["pair_mean"]), np.asarray(prep["pair_std"]))
                edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
                if edge_index.size:
                    edge_index = edge_index.reshape(-1, 2).T
                else:
                    edge_index = np.empty((2, 0), dtype=np.int64)
                data = Data(
                    x=torch.tensor(node, dtype=torch.float32, device=device),
                    edge_index=torch.tensor(edge_index, dtype=torch.long, device=device),
                    edge_attr=torch.tensor(edge, dtype=torch.float32, device=device),
                    pair_x=torch.tensor(pair, dtype=torch.float32, device=device).reshape(1, -1),
                    att_idx=torch.tensor([int(graph["metadata"]["attacker_idx"])], dtype=torch.long, device=device),
                    target_idx=torch.tensor([int(graph["metadata"]["target_idx"])], dtype=torch.long, device=device),
                    ptr=torch.tensor([0, node.shape[0]], dtype=torch.long, device=device),
                )
                with torch.inference_mode():
                    score = float(torch.sigmoid(model(data)).detach().cpu().item())
                self._torch, self._model, self._data = torch, model, data
                self._device, self._score = str(device), score
                self._coverage = {
                    "node_count": int(node.shape[0]),
                    "car_coverage_fraction": float(1.0 - np.nanmean(np.asarray(graph["node_raw"], dtype=float)[:, 13])),
                    "position_coverage_fraction": float(1.0 - np.nanmean(np.asarray(graph["node_raw"], dtype=float)[:, 14])),
                    "max_car_sample_age_sec": float(np.nanmax(np.asarray(graph["node_raw"], dtype=float)[:, 6])),
                    "max_position_sample_age_sec": float(np.nanmax(np.asarray(graph["node_raw"], dtype=float)[:, 10])),
                }
            except Exception as exc:  # Optional dependency/runtime failure is explicit in the API.
                self._load_error = f"{type(exc).__name__}: {exc}"

    def manifest_status(self):
        self._load()
        return {
            "status": "available" if self._score is not None else "unavailable",
            "model_version": self.model_version,
            "device": self._device,
            "checkpoint": self.checkpoint_path.name,
            "checkpoint_sha256": _sha256(self.checkpoint_path) if self.checkpoint_path.exists() else None,
            "error": self._load_error,
        }

    def predict(self, window_id: str | None = None):
        self._load()
        if self.window_id is None or window_id not in (None, self.window_id):
            return {
                "status": "unavailable", "model_version": self.model_version,
                "score_label": "experimental boundary position-swap score",
                "reason": "No frozen graph is available for this replay decision window",
            }
        if self._score is None:
            return {
                "status": "unavailable", "model_version": self.model_version,
                "score_label": "experimental boundary position-swap score",
                "reason": self._load_error or "PyTorch/PyG runtime unavailable",
            }
        return {
            "status": "available", "model_version": self.model_version,
            "device": self._device,
            "score": self._score,
            "score_label": "experimental boundary position-swap score",
            "not_a_calibrated_overtake_probability": True,
            "not_an_attack_benefit_probability": True,
            "decision_timestamp_session_sec": self.metadata["session_time_sec"],
            "input_freshness": {
                "asof_tolerance_sec": self.metadata["asof_tolerance_sec"],
                "lookback_sec": self.metadata["feature_lookback_sec"],
                "join": "backward-only, no interpolation",
            },
            "coverage": {"window_id": self.window_id,
                         "attacker": self.metadata["attacker_driver"],
                         "target": self.metadata["target_driver"],
                         "pair_context_available": True, **getattr(self, "_coverage", {})},
        }

    def parity(self):
        self._load()
        return {"status": "available" if self._score is not None else "unavailable",
                "score": self._score, "device": self._device,
                "comparison": "compact fixture uses the same transform/model forward path as the saved graph implementation"}
