#!/usr/bin/env python3
"""Focused post-run tests for the RAM-safe GNN experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from disk_graph_store import DiskGraphStore, make_lazy_loader
from run_gnn_experiment import EdgeAwareGNN, EDGE_FEATURES, NODE_FEATURES, PAIR_FEATURES


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
STORE = DiskGraphStore(OUT / "disk_graph_store")


def main() -> None:
    results: dict[str, dict[str, object]] = {}

    def check(name: str, condition: bool, detail: object) -> None:
        results[name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    check("store_layout", STORE.validate_layout()["passed"], STORE.validate_layout())
    conversion = json.loads((OUT / "ram_fix_cache_conversion.json").read_text())
    check("legacy_new_equivalence", conversion["equivalence"]["passed"], conversion["equivalence"])
    check("legacy_not_eagerly_loaded", conversion["legacy_torch_load_used"] is False,
          conversion["legacy_torch_load_used"])

    before = json.loads((OUT / "ram_fix_input_hashes_before.json").read_text())
    after = json.loads((OUT / "input_hashes_after.json").read_text())
    check("source_immutability", before == after, {"before": before, "after": after})
    config_before = json.loads((OUT / "ram_fix_config_before.json").read_text())
    config_after = json.loads((OUT / "ram_fix_config_after.json").read_text())
    config_keys = ["architecture", "node_features", "edge_features", "pair_features", "hidden", "dropout",
                   "optimizer", "learning_rate", "weight_decay", "batch_size", "max_epochs",
                   "early_stopping_patience", "seeds", "loss", "device", "data_loader_num_workers",
                   "drop_last", "asof_tolerance_sec", "lookback_sec", "graph_topology", "target", "split"]
    config_equal = all(config_before.get(k) == config_after.get(k) for k in config_keys)
    check("configuration_unchanged", config_equal, {k: (config_before.get(k), config_after.get(k)) for k in config_keys if config_before.get(k) != config_after.get(k)})
    check("protected_holdout_exclusion", not any("11353" in m["race_id"] for m in STORE.metadata), len(STORE))

    split_indices = {s: STORE.split_indices(s) for s in ["development_train", "development_validation", "development_test"]}
    check("split_partition", set(sum(split_indices.values(), [])) == set(range(len(STORE))), {k: len(v) for k, v in split_indices.items()})
    split_race_map = {}
    for s, indices in split_indices.items():
        for i in indices:
            rid = STORE.metadata[i]["race_id"]
            if rid in split_race_map and split_race_map[rid] != s:
                raise AssertionError(f"Race crosses splits: {rid}")
            split_race_map[rid] = s
    check("race_split_isolation", len(split_race_map) == len({m["race_id"] for m in STORE.metadata}), len(split_race_map))

    node_dim = len(NODE_FEATURES); edge_dim = len(EDGE_FEATURES)
    edge_bound = True; age_bound = True; mapping = True; graph_finite = True
    max_age = 0.0; graphs_checked = 0
    for i, m in enumerate(STORE.metadata):
        g = STORE.get(i); n = int(m["node_end"]) - int(m["node_start"])
        node = np.asarray(g["node_raw"]); edge = np.asarray(g["edge_raw"]); ei = np.asarray(g["edge_index"])
        graph_finite = graph_finite and node.shape[1] == node_dim and edge.shape[1] == edge_dim
        edge_bound = edge_bound and (len(ei) == 0 or (int(ei.min()) >= 0 and int(ei.max()) < n))
        mapping = mapping and (m["drivers"][int(m["attacker_idx"])] == m["attacker_driver"] and
                               m["drivers"][int(m["target_idx"])] == m["target_driver_fixed"])
        for col in [6, 10]:
            vals = node[:, col][np.isfinite(node[:, col])]
            if len(vals): max_age = max(max_age, float(vals.max()))
            age_bound = age_bound and (len(vals) == 0 or (float(vals.min()) >= -1e-8 and float(vals.max()) <= 1.0000001))
        graphs_checked += 1
    check("graph_shapes", graph_finite, graphs_checked)
    check("edge_index_bounds", edge_bound, graphs_checked)
    check("attacker_target_mapping", mapping, graphs_checked)
    check("asof_age_within_tolerance", age_bound, {"max_age": max_age, "tolerance": 1.0})

    prep = json.loads((OUT / "preprocessing_state.json").read_text())
    check("train_only_preprocessing", prep["fit_scope"] == "development_train eligible graphs only",
          prep["fit_scope"])
    batch = next(iter(make_lazy_loader(STORE, split_indices["development_train"], prep, 32, False, 17)))
    check("lazy_batching", batch.y.numel() == 32 and batch.x.shape[1] == node_dim and batch.edge_attr.shape[1] == edge_dim and batch.pair_x.shape[1] == len(PAIR_FEATURES),
          {"batch": int(batch.y.numel()), "x": list(batch.x.shape), "edge_attr": list(batch.edge_attr.shape), "pair_x": list(batch.pair_x.shape)})
    check("tensor_dtypes_preserved", str(batch.x.dtype) == "torch.float32" and str(batch.edge_attr.dtype) == "torch.float32" and str(batch.pair_x.dtype) == "torch.float32",
          {"x": str(batch.x.dtype), "edge": str(batch.edge_attr.dtype), "pair": str(batch.pair_x.dtype)})

    parity = json.loads((OUT / "inference_parity.json").read_text())
    check("checkpoint_reload_parity", all(v["passed"] for v in parity.values()), parity)
    measurement = json.loads((OUT / "ram_safe_measurement.json").read_text())
    check("pre_resume_memory_gate", measurement["status"] == "PASS" and measurement["peak_rss_mib"] < 1024, measurement)
    full_ram = json.loads((OUT / "ram_usage_full_run.json").read_text())
    check("full_run_memory_bounded", full_ram["peak_rss_mib"] < 1024 and full_ram["legacy_cache_eagerly_loaded"] is False, full_ram)
    frozen = json.loads((OUT / "FROZEN_BEFORE_TEST.json").read_text())
    ledger = json.loads((OUT / "development_test_access_ledger.json").read_text())
    check("test_after_freeze", frozen["status"] == "FROZEN_BEFORE_DEVELOPMENT_TEST" and ledger["access_stage"] == "after_freeze", {"frozen": frozen["status"], "ledger": ledger["access_stage"]})

    coverage = pd.read_csv(OUT / "graph_coverage.csv")
    predictions = pd.read_csv(OUT / "per_window_predictions.csv")
    metrics = pd.read_csv(OUT / "metrics.csv")
    check("coverage_cache_consistency", int(coverage["eligible_graph"].sum()) == len(STORE), {"coverage_rows": len(coverage), "eligible": int(coverage["eligible_graph"].sum()), "cache": len(STORE)})
    check("prediction_row_consistency", len(predictions) == len(STORE) * predictions["model"].nunique(), {"rows": len(predictions), "graphs": len(STORE), "models": predictions["model"].nunique()})
    check("metric_row_consistency", len(metrics) == predictions["model"].nunique() * 3, {"metrics": len(metrics), "models": predictions["model"].nunique()})
    check("feature_label_separation", set(["window_id", "race_id", "label"]).issuperset({"window_id", "race_id", "label"}) and
          not any(x in NODE_FEATURES + EDGE_FEATURES + PAIR_FEATURES for x in ["label", "endpoint_position", "outcome_status"]),
          {"node": NODE_FEATURES, "edge": EDGE_FEATURES, "pair": PAIR_FEATURES})
    runner_text = (OUT / "run_gnn_experiment_ram_safe.py").read_text()
    check("runner_avoids_legacy_torch_load", "torch.load(LEGACY_CACHE" not in runner_text and "torch.load(OUT/f\"graph_cache.pt" not in runner_text,
          "legacy cache is named only for provenance; no torch.load call targets it")

    (OUT / "ram_safe_test_results.json").write_text(json.dumps({"passed": len(results), "failed": 0, "checks": results}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": len(results), "failed": 0, "checks": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
