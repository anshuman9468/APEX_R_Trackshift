#!/usr/bin/env python3
"""Standalone inference-only smoke test for the frozen APEX-R GNN proxy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


OUTPUT_LABEL = "experimental boundary position-swap proxy signal"
PROTECTED_SESSION = "11353"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(__file__).resolve().parent

    import numpy as np
    import torch
    from torch import nn
    from torch_geometric.data import Data
    from torch_geometric.nn import GINEConv

    checkpoint_path = base / "gnn_seed_42_best.pt"
    preprocessing_path = base / "preprocessing_state.json"
    fixture_path = base / "approved_smoke_fixture.json"

    # This fixture is a copied, already-approved validation graph. The check is
    # deliberately local and does not discover, open, or query any session data.
    fixture_text = fixture_path.read_text(encoding="utf-8")
    if PROTECTED_SESSION in fixture_text:
        raise RuntimeError("protected-session identifier found in smoke fixture")
    fixture = json.loads(fixture_text)
    graph = fixture["inference_graph"]
    metadata = graph["metadata"]
    if fixture["selection"]["proposed_split"] != "development_validation":
        raise RuntimeError("smoke fixture is not from the approved validation partition")
    if not fixture["selection"].get("not_selected_by_outcome", False):
        raise RuntimeError("smoke fixture selection provenance is incomplete")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    required = {
        "node_dim": 18,
        "edge_dim": 4,
        "pair_dim": 12,
        "hidden": 32,
        "dropout": 0.2,
        "seed": 42,
    }
    mismatches = {k: (config.get(k), v) for k, v in required.items() if config.get(k) != v}
    if mismatches:
        raise RuntimeError(f"checkpoint configuration mismatch: {mismatches}")
    if checkpoint.get("model_class") != "EdgeAwareGNN":
        raise RuntimeError("unexpected checkpoint model class")
    if checkpoint.get("best_epoch") != 28:
        raise RuntimeError("checkpoint is not the selected epoch-28 state")

    class EdgeAwareGNN(nn.Module):
        def __init__(self, node_dim: int, edge_dim: int, pair_dim: int):
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

        def forward(self, data: Data):
            x = self.node_encoder(data.x)
            edge_attr = self.edge_encoder(data.edge_attr)
            x = self.dropout(torch.relu(self.conv1(x, data.edge_index, edge_attr)))
            x = self.dropout(torch.relu(self.conv2(x, data.edge_index, edge_attr)))
            ptr = data.ptr[:-1]
            attacker = data.att_idx.view(-1).long() + ptr
            target = data.target_idx.view(-1).long() + ptr
            pair = data.pair_x.view(-1, data.pair_x.shape[-1])
            return self.decoder(torch.cat([x[attacker], x[target], pair], dim=1)).squeeze(1)

    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")

    model = EdgeAwareGNN(config["node_dim"], config["edge_dim"], config["pair_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))

    def transform(raw, mean, std):
        raw_array = np.asarray(raw, dtype=float)
        mean_array = np.asarray(mean, dtype=float)
        std_array = np.asarray(std, dtype=float)
        return (np.where(np.isfinite(raw_array), raw_array, mean_array) - mean_array) / std_array

    node = transform(graph["node_raw"], preprocessing["node_mean"], preprocessing["node_std"])
    edge = transform(
        np.asarray(graph["edge_raw"], dtype=float).reshape(-1, config["edge_dim"]),
        preprocessing["edge_mean"],
        preprocessing["edge_std"],
    )
    pair = transform(graph["pair_raw"], preprocessing["pair_mean"], preprocessing["pair_std"])
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    edge_index = edge_index.reshape(-1, 2).T if edge_index.size else np.empty((2, 0), dtype=np.int64)

    data = Data(
        x=torch.tensor(node, dtype=torch.float32, device=device),
        edge_index=torch.tensor(edge_index, dtype=torch.long, device=device),
        edge_attr=torch.tensor(edge, dtype=torch.float32, device=device),
        pair_x=torch.tensor(pair, dtype=torch.float32, device=device).reshape(1, -1),
        att_idx=torch.tensor([int(metadata["attacker_idx"])], dtype=torch.long, device=device),
        target_idx=torch.tensor([int(metadata["target_idx"])], dtype=torch.long, device=device),
        ptr=torch.tensor([0, node.shape[0]], dtype=torch.long, device=device),
    )

    # No optimizer, loss, backward pass, parameter mutation, or training call is
    # present in this program.
    with torch.inference_mode():
        score = float(torch.sigmoid(model(data)).detach().cpu().item())

    finite = math.isfinite(score)
    bounded = 0.0 <= score <= 1.0
    result = {
        "status": "PASS" if finite and bounded else "FAIL",
        "model_version": "gnn_proxy_v1",
        "model_output_label": OUTPUT_LABEL,
        "checkpoint": checkpoint_path.name,
        "selected_seed": config["seed"],
        "selected_epoch": checkpoint["best_epoch"],
        "device": str(device),
        "approved_sample_split": fixture["selection"]["proposed_split"],
        "approved_sample_race": fixture["selection"]["race_id"],
        "window_id": metadata["window_id"],
        "score": score,
        "score_is_finite": finite,
        "score_is_between_zero_and_one": bounded,
        "model_in_eval_mode": not model.training,
        "inference_mode_used": True,
        "training_code_executed": False,
        "protected_session_accessed": False,
        "note": "Raw sigmoid proxy signal; not a calibrated overtake probability or an ATTACK-benefit probability.",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
