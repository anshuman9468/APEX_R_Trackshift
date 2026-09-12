#!/usr/bin/env python3
"""Compare the frozen adapter with the original saved-graph forward path."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "INFERENCE_PARITY.json"

def main():
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "gnn_proxy_experiment_v1"))
    from disk_graph_store import DiskGraphStore, LazyGraphDataset
    from run_gnn_experiment import EdgeAwareGNN
    from torch_geometric.loader import DataLoader
    from backend.gnn_adapter import FrozenGNNAdapter
    adapter = FrozenGNNAdapter(device_preference="cpu")
    compact = adapter.predict()
    store = DiskGraphStore(ROOT / "gnn_proxy_experiment_v1" / "disk_graph_store")
    idx = next(i for i, m in enumerate(store.metadata) if m["window_id"] == compact["coverage"]["window_id"])
    prep = json.loads((ROOT / "gnn_proxy_experiment_v1" / "preprocessing_state.json").read_text())
    data = next(iter(DataLoader(LazyGraphDataset(store, [idx], prep), batch_size=1, shuffle=False)))
    model = EdgeAwareGNN(18, 4, 12)
    state = torch.load(ROOT / "gnn_proxy_experiment_gpu_v1" / "gnn_seed_42_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state); model.eval()
    with torch.inference_mode():
        saved = float(torch.sigmoid(model(data)).item())
    delta = abs(saved - float(compact["score"]))
    result = {"status": "PASS" if delta <= 1e-7 else "FAIL", "window_id": compact["coverage"]["window_id"],
              "adapter_cpu_score": compact["score"], "saved_graph_cpu_score": saved,
              "absolute_difference": delta, "tolerance": 1e-7,
              "note": "Same checkpoint, training-only preprocessing, graph topology and forward architecture; only the device adapter path differs."}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
