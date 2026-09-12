#!/usr/bin/env python3
"""Measure the repaired path without starting full training."""

from __future__ import annotations

import gc
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from disk_graph_store import (
    EDGE_DIM,
    NODE_DIM,
    PAIR_DIM,
    DiskGraphStore,
    compute_train_preprocessing,
    make_lazy_loader,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
STORE = OUT / "disk_graph_store"


def current_rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def peak_rss_bytes() -> int:
    # Linux ru_maxrss is KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def snap(stage: str) -> dict[str, int | str]:
    return {"stage": stage, "current_rss_bytes": current_rss_bytes(), "peak_rss_bytes": peak_rss_bytes()}


def reference_stats(arrays: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    """Bounded train-only reference implementing the legacy nanmean/nanstd formula."""
    if not arrays:
        return np.zeros(width), np.ones(width)
    joined = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1, width) for a in arrays], axis=0)
    mean = np.nanmean(joined, axis=0)
    std = np.nanstd(joined, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return mean, std


def compare(name: str, new_mean: list[float], new_std: list[float], ref_mean: np.ndarray, ref_std: np.ndarray) -> dict[str, object]:
    nm, ns = np.asarray(new_mean), np.asarray(new_std)
    mean_diff = np.abs(nm - ref_mean); std_diff = np.abs(ns - ref_std)
    return {"name": name, "mean_max_abs_difference": float(mean_diff.max(initial=0.0)),
            "std_max_abs_difference": float(std_diff.max(initial=0.0)),
            "mean_nonzero_difference_count": int(np.count_nonzero(mean_diff)),
            "std_nonzero_difference_count": int(np.count_nonzero(std_diff)),
            "differences_are_finite": bool(np.isfinite(mean_diff).all() and np.isfinite(std_diff).all())}


def main() -> None:
    started = time.time()
    snapshots = [snap("start")]
    store = DiskGraphStore(STORE)
    layout = store.validate_layout()
    if not layout["passed"]:
        raise RuntimeError(f"Disk store layout failed: {layout}")
    snapshots.append(snap("after_lazy_store_open"))
    train = store.split_indices("development_train")
    val = store.split_indices("development_validation")
    test = store.split_indices("development_test")
    prep = compute_train_preprocessing(store, train)
    (OUT / "ram_safe_preprocessing_state.json").write_text(json.dumps(prep, indent=2, sort_keys=True) + "\n")
    snapshots.append(snap("after_streaming_train_preprocessing"))

    # Validate the streaming formulas against bounded concatenations of only the
    # training arrays. This is a verification reference, not the production path.
    node_arrays = [store.get(i)["node_raw"] for i in train]
    # Build the edge list in a simple bounded pass using metadata offsets.
    edge_arrays = [store.edge_values[int(store.metadata[i]["edge_start"]):int(store.metadata[i]["edge_end"])]
                   for i in train if int(store.metadata[i]["edge_end"]) > int(store.metadata[i]["edge_start"])]
    pair_arrays = [store.get(i)["pair_raw"] for i in train]
    tab_arrays = []
    for i in train:
        g = store.get(i); m = g["metadata"]
        tab_arrays.append(np.concatenate([g["node_raw"][int(m["attacker_idx"])],
                                          g["node_raw"][int(m["target_idx"])], g["pair_raw"]]).reshape(1, -1))
    ref_node_mean, ref_node_std = reference_stats(node_arrays, NODE_DIM)
    ref_edge_mean, ref_edge_std = reference_stats(edge_arrays, EDGE_DIM)
    ref_pair_mean, ref_pair_std = reference_stats(pair_arrays, PAIR_DIM)
    ref_tab_mean, ref_tab_std = reference_stats(tab_arrays, NODE_DIM * 2 + PAIR_DIM)
    formula_comparison = [
        compare("node", prep["node_mean"], prep["node_std"], ref_node_mean, ref_node_std),
        compare("edge", prep["edge_mean"], prep["edge_std"], ref_edge_mean, ref_edge_std),
        compare("pair", prep["pair_mean"], prep["pair_std"], ref_pair_mean, ref_pair_std),
        compare("tabular", prep["tab_mean"], prep["tab_std"], ref_tab_mean, ref_tab_std),
    ]
    del node_arrays, edge_arrays, pair_arrays, tab_arrays
    gc.collect()
    snapshots.append(snap("after_bounded_formula_comparison"))

    # One forward/backward batch only; no optimizer step and no full training.
    from run_gnn_experiment import EdgeAwareGNN
    loader = make_lazy_loader(store, train, prep, batch_size=32, shuffle=False, seed=17)
    batch = next(iter(loader))
    snapshots.append(snap("after_one_lazy_training_batch_loaded"))
    model = EdgeAwareGNN(NODE_DIM, EDGE_DIM, PAIR_DIM)
    model.train()
    logits = model(batch)
    loss = nn.BCEWithLogitsLoss()(logits, batch.y.view(-1))
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite one-batch diagnostic loss")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite one-batch diagnostic gradient")
    snapshots.append(snap("after_one_forward_backward_batch"))
    result = {
        "status": "PASS" if all(x["differences_are_finite"] and x["mean_max_abs_difference"] <= 1e-9 and x["std_max_abs_difference"] <= 1e-9 for x in formula_comparison) else "REVIEW_FLOATING_POINT_DIFFERENCES",
        "legacy_torch_load_used": False,
        "legacy_graphs_eagerly_loaded": False,
        "store_layout": layout,
        "split_counts": {"development_train": len(train), "development_validation": len(val), "development_test": len(test)},
        "preprocessing_fit_scope": "development_train only",
        "formula_comparison": formula_comparison,
        "one_batch": {"batch_size": int(batch.y.numel()), "node_tensor_dtype": str(batch.x.dtype),
                      "edge_tensor_dtype": str(batch.edge_attr.dtype), "pair_tensor_dtype": str(batch.pair_x.dtype),
                      "loss": float(loss.detach()), "all_logits_finite": bool(torch.isfinite(logits).all()),
                      "all_gradients_finite": True},
        "snapshots": snapshots,
        "peak_rss_bytes": peak_rss_bytes(), "peak_rss_mib": peak_rss_bytes() / (1024 * 1024),
        "elapsed_seconds": time.time() - started,
        "memory_reference_note": "reference_stats concatenates training-only numeric slices for equivalence checking; production stats are streaming",
    }
    (OUT / "ram_safe_measurement.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
