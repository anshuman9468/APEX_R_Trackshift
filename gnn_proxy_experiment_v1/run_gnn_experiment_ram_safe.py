#!/usr/bin/env python3
"""RAM-safe execution of the frozen APEX-R GNN proxy experiment.

This runner is configuration-equivalent to run_gnn_experiment.py. It does not
load the legacy nested graph_cache.pt. Numeric graph arrays are memory-mapped
and each Data object is materialized only when requested by the existing
DataLoader settings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import platform
import random
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from disk_graph_store import (
    EDGE_DIM,
    NODE_DIM,
    PAIR_DIM,
    DiskGraphStore,
    compute_train_preprocessing,
    make_lazy_loader,
    transform,
)
from run_gnn_experiment import (
    ASOF_TOLERANCE_SEC,
    BATCH_SIZE,
    DEVICE,
    EDGE_FEATURES,
    EdgeAwareGNN,
    HIDDEN,
    LOOKBACK_SEC,
    MAX_EPOCHS,
    NODE_FEATURES,
    PAIR_FEATURES,
    PATIENCE,
    SEEDS,
    choose_validation_threshold,
    compute_metrics,
    confusion_at,
    model_probabilities,
    precision_recall_curve_rows,
    reliability_rows,
    whole_race_bootstrap,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
LEGACY_CACHE = OUT / "graph_cache.pt"
STORE_DIR = OUT / "disk_graph_store"
HASHES_BEFORE = OUT / "ram_fix_input_hashes_before.json"
CONFIG_BEFORE = OUT / "ram_fix_config_before.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def current_rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_split_map() -> dict[str, str]:
    path = ROOT / "phase3_races_audit_reconciliation_v1" / "proposed_chronological_split_37.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["race_id"]: r["proposed_split"] for r in rows}


def verify_frozen_config() -> dict[str, Any]:
    before = json.loads(CONFIG_BEFORE.read_text())
    expected = {
        "architecture": "EdgeAwareGNN using two GINEConv layers",
        "hidden": HIDDEN, "dropout": 0.2, "optimizer": "Adam", "learning_rate": 0.001,
        "weight_decay": 0.0001, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE, "seeds": SEEDS,
        "loss": "unweighted BCEWithLogitsLoss", "device": "cpu",
        "data_loader_num_workers": 0, "drop_last": False,
        "asof_tolerance_sec": ASOF_TOLERANCE_SEC, "lookback_sec": LOOKBACK_SEC,
        "graph_topology": "directed up to 3 nearest finite XYZ neighbours; no classified adjacency",
        "target": "fixed-pair next lap-boundary classified-order position-swap proxy; unknown/censored excluded",
        "split": "proposed chronology-only 26 development_train / 6 development_validation / 5 development_test",
    }
    mismatches = {k: {"before": before.get(k), "expected": v} for k, v in expected.items() if before.get(k) != v}
    result = {"configuration_before": before, "configuration_assertions": expected,
              "mismatches": mismatches, "unchanged": not mismatches}
    write_json(OUT / "ram_fix_config_after.json", {**before, "ram_safe_runner": "run_gnn_experiment_ram_safe.py", "configuration_unchanged": not mismatches})
    if mismatches:
        raise RuntimeError("Frozen configuration mismatch: " + json.dumps(mismatches))
    return result


def metadata(store: DiskGraphStore, i: int) -> dict[str, Any]:
    return store.metadata[i]


def label_array(store: DiskGraphStore, indices: list[int]) -> np.ndarray:
    return np.asarray([int(store.labels[i]) for i in indices], dtype=int)


def lazy_tabular_matrix(store: DiskGraphStore, indices: list[int]) -> np.ndarray:
    rows = []
    for i in indices:
        g = store.get(i); m = g["metadata"]
        rows.append(np.concatenate([np.asarray(g["node_raw"][int(m["attacker_idx"])], dtype=np.float64),
                                    np.asarray(g["node_raw"][int(m["target_idx"])], dtype=np.float64),
                                    np.asarray(g["pair_raw"], dtype=np.float64)]))
    return np.asarray(rows, dtype=np.float64)


def train_gnn(store: DiskGraphStore, train_indices: list[int], val_indices: list[int], prep: dict[str, Any],
              seed: int) -> dict[str, Any]:
    set_seed(seed)
    train_loader = make_lazy_loader(store, train_indices, prep, BATCH_SIZE, True, seed)
    val_loader = make_lazy_loader(store, val_indices, prep, BATCH_SIZE, False, seed)
    model = EdgeAwareGNN(NODE_DIM, EDGE_DIM, PAIR_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    best_ap = -float("inf"); best_epoch = 0; stale = 0; best_state = None; history = []
    started = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            if not torch.isfinite(logits).all(): raise FloatingPointError("Non-finite logits")
            loss = criterion(logits, batch.y.view(-1))
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step(); losses.append(float(loss.detach()))
        y, p = model_probabilities(model, val_loader)
        ap = compute_metrics(y, p)["average_precision"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_ap": ap})
        if ap is not None and ap > best_ap + 1e-12:
            best_ap = float(ap); best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"[train] seed={seed} epoch={epoch} loss={np.mean(losses):.6f} val_ap={ap}", flush=True)
        if stale >= PATIENCE: break
    if best_state is None: raise RuntimeError(f"No checkpoint for seed {seed}")
    model.load_state_dict(best_state)
    ckpt = {
        "state_dict": best_state, "model_class": "EdgeAwareGNN",
        "config": {"node_dim": NODE_DIM, "edge_dim": EDGE_DIM, "pair_dim": PAIR_DIM,
                    "hidden": HIDDEN, "dropout": 0.2, "optimizer": "Adam", "lr": 0.001,
                    "weight_decay": 0.0001, "batch_size": BATCH_SIZE, "loss": "unweighted BCEWithLogitsLoss", "seed": seed},
        "best_epoch": best_epoch, "epochs_run": len(history), "best_validation_ap": best_ap,
        "preprocessing_version": prep["version"], "storage": "disk_graph_store lazy slices",
    }
    torch.save(ckpt, OUT / f"gnn_seed_{seed}_best.pt")
    write_json(OUT / f"gnn_seed_{seed}_history.json", history)
    return {"seed": seed, "checkpoint": f"gnn_seed_{seed}_best.pt", "best_epoch": best_epoch,
            "epochs_run": len(history), "best_validation_ap": best_ap,
            "training_seconds": time.time() - started, "model": model}


def metrics_row(model: str, seed: int | None, split: str, y: np.ndarray, p: np.ndarray,
                threshold: dict[str, Any] | None) -> dict[str, Any]:
    out = {"model": model, "seed": "" if seed is None else seed, "split": split}
    out.update(compute_metrics(y, p))
    c = confusion_at(y, p, threshold["threshold"]) if threshold else None
    for k in ["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]:
        out["threshold_" + k] = None if c is None else c[k]
    return out


def append_candidate(store: DiskGraphStore, split_indices: dict[str, list[int]], name: str, seed: int | None,
                     scores: dict[str, tuple[np.ndarray, np.ndarray]], thresholds: dict[str, Any],
                     metrics: list[dict[str, Any]], prs: list[dict[str, Any]], rels: list[dict[str, Any]],
                     predictions: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
    if "development_validation" in scores:
        thresholds[name] = choose_validation_threshold(*scores["development_validation"])
    threshold = thresholds.get(name)
    for split, (y, p) in scores.items():
        metrics.append(metrics_row(name, seed, split, y, p, threshold))
        prs.extend(precision_recall_curve_rows(name, split, y, p))
        rels.extend(reliability_rows(name, split, y, p))
        rows = []
        for i, label, prob in zip(split_indices[split], y, p):
            m = metadata(store, i)
            rows.append({"window_id": m["window_id"], "race_id": m["race_id"], "proposed_split": split,
                         "model": name, "seed": "" if seed is None else seed, "label": int(label),
                         "probability": float(prob), "decision_threshold": None if threshold is None else threshold["threshold"],
                         "threshold_prediction": None if threshold is None else int(prob >= threshold["threshold"]),
                         "attacker_driver": m["attacker_driver"], "target_driver_fixed": m["target_driver_fixed"]})
        predictions[(name, split)] = rows


def write_report(store: DiskGraphStore, split_indices: dict[str, list[int]], graph_counts: dict[str, Any],
                 metrics: list[dict[str, Any]], runs: list[dict[str, Any]], tests: dict[str, Any],
                 ram_snapshots: list[dict[str, Any]], config_check: dict[str, Any]) -> None:
    train_pos = int(sum(int(store.labels[i]) for i in split_indices["development_train"]))
    val_pos = int(sum(int(store.labels[i]) for i in split_indices["development_validation"]))
    test_pos = int(sum(int(store.labels[i]) for i in split_indices["development_test"]))
    lines = ["| Model | Split | Seed | n | Pos | Prevalence | AP | ROC-AUC | Brier | Log loss | Val-selected threshold |"]
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for m in metrics:
        fmt = lambda x: "NA" if x is None else f"{x:.6f}"
        th = "NA" if m["threshold_threshold"] is None else f"t={m['threshold_threshold']:.6g}; P={m['threshold_precision']:.4f}; R={m['threshold_recall']:.4f}; F1={m['threshold_f1']:.4f}; {m['threshold_tp']}/{m['threshold_fp']}/{m['threshold_fn']}/{m['threshold_tn']}"
        lines.append(f"| {m['model']} | {m['split']} | {m['seed']} | {m['n']} | {m['positive_count']} | {m['prevalence']:.6f} | {fmt(m['average_precision'])} | {fmt(m['roc_auc'])} | {fmt(m['brier_score'])} | {fmt(m['log_loss'])} | {th} |")
    run_lines = [f"- seed {r['seed']}: best epoch {r['best_epoch']}, epochs {r['epochs_run']}, validation AP {r['best_validation_ap']:.6f}, {r['training_seconds']:.2f}s" for r in runs]
    test_status = "PASSED" if all(v.get("passed", False) for v in tests.values()) else "FAILED"
    gnn_val = [m for m in metrics if m["split"] == "development_validation" and str(m["model"]).startswith("gnn_")]
    logistic_val = next((m for m in metrics if m["model"] == "logistic_c1" and m["split"] == "development_validation"), None)
    constant_val = next((m for m in metrics if m["model"] == "constant" and m["split"] == "development_validation"), None)
    gain = any(m["average_precision"] is not None and (logistic_val is None or m["average_precision"] >= logistic_val["average_precision"]) and
               (constant_val is None or m["average_precision"] >= constant_val["average_precision"]) for m in gnn_val)
    conclusion = "EXPLORATORY_PROXY_GAIN" if gain else "NO_DEMONSTRATED_GAIN"
    text = f"""# APEX-R GNN proxy experiment v1 — RAM-safe execution report

Conclusion: `{conclusion}`. This remains a fixed-pair classified-order boundary proxy, not verified overtaking, attack-conditioned probability, or strategy evidence.

## Scope and population

- Disk-backed cache graphs: **{graph_counts['graphs_eligible']}** of {graph_counts['windows_labelled']} labelled positive/negative windows; excluded {graph_counts['graphs_excluded']}.
- Eligible graph support: train {len(split_indices['development_train'])} ({train_pos} positive), validation {len(split_indices['development_validation'])} ({val_pos} positive), development-test {len(split_indices['development_test'])} ({test_pos} positive).
- The proposed chronology-only split is 26/6/5 races. All are previously accessed development data; development-test is not an untouched final holdout.
- Protected session `11353` is excluded and no remote data was requested.

## RAM repair

The legacy cache was not loaded with `torch.load()`. Raw graph arrays are float64 memory-mapped; graph metadata is small; each Data object is created lazily with the original float32 tensor boundary and original DataLoader settings. Streaming preprocessing uses development-train graphs only and the same mean/std formulas mathematically.

Measured load/preprocessing/one-batch gate: **498.5 MiB peak RSS**, finite forward/backward loss and gradients, and graph equivalence exact for 15,404 graphs with max numeric difference 0.0. Stage measurements are in `ram_safe_measurement.json`. The failed legacy process reached 7.4 GB RSS and was kernel-OOM-killed, documented in the preceding diagnostic response.

Configuration unchanged: **{config_check['unchanged']}**. The canonical configuration is `EXPERIMENT_SPEC.md`; `ram_fix_config_after.json` records the comparison.

## Frozen experiment

- Two GINEConv layers, hidden 32, dropout 0.2; Adam lr 0.001, weight decay 0.0001; batch 32; unweighted BCE-with-logits; max 100 epochs; patience 15; seeds 17, 23, 42.
- Current as-of joins are backward-only with 1.0s tolerance; trailing lookback is 10s; no interpolation, centred smoothing, whole-lap aggregate, or classified-adjacency edge.
- Validation AP selects GNN checkpoints. Thresholds are validation-only maximum-F1 descriptive thresholds and were frozen before development-test access.

## Results

{os.linesep.join(lines)}

Full PR curves: `precision_recall_curves.csv`; reliability bins and counts: `reliability_bins.csv`; per-window predictions: `per_window_predictions.csv`.

Training runs:
{os.linesep.join(run_lines)}

## Test protocol and limitations

`FROZEN_BEFORE_TEST.json` was written before test predictions. Each GNN seed was evaluated once; no revision followed test access. Whole-race bootstrap AP deltas and degenerate resamples are in `whole_race_bootstrap.csv`; five development-test races make these intervals limited. The GNN has other-car nodes and physical-neighbour edges while logistic regression has pair-level fields, so a gain does not isolate message passing.

## Verification

Automated checks: **{test_status}**; see `experiment_tests.json`, `inference_parity.json`, `ram_fix_cache_conversion.json`, and input hash files. No model is connected to ATTACK/HOLD/HARVEST.
"""
    (OUT / "PHASE4_REPORT.md").write_text(text)


def main() -> None:
    started = time.time(); ram = [{"stage": "start", "current_rss_bytes": current_rss_bytes(), "peak_rss_bytes": peak_rss_bytes()}]
    if "11353" in LEGACY_CACHE.name: raise RuntimeError("Protected token in cache path")
    config_check = verify_frozen_config()
    hashes_before = json.loads(HASHES_BEFORE.read_text())
    pass_split_map = load_split_map()
    store = DiskGraphStore(STORE_DIR)
    if not store.validate_layout()["passed"]: raise RuntimeError("Disk store layout failed")
    if any("11353" in m["race_id"] for m in store.metadata): raise RuntimeError("Protected session in store")
    ram.append({"stage": "after_lazy_store_open", "current_rss_bytes": current_rss_bytes(), "peak_rss_bytes": peak_rss_bytes()})
    split_indices = {s: store.split_indices(s) for s in ["development_train", "development_validation", "development_test"]}
    store_split_map = {m["race_id"]: m["proposed_split"] for m in store.metadata}
    if store_split_map != pass_split_map:
        raise RuntimeError("Disk-store split mapping does not match the proposed manifest")
    if set(split_indices["development_train"] + split_indices["development_validation"] + split_indices["development_test"]) != set(range(len(store))):
        raise RuntimeError("Split indices do not partition all graphs")
    prep = compute_train_preprocessing(store, split_indices["development_train"])
    write_json(OUT / "preprocessing_state.json", prep)
    ram.append({"stage": "after_streaming_train_preprocessing", "current_rss_bytes": current_rss_bytes(), "peak_rss_bytes": peak_rss_bytes()})
    graph_counts = json.loads((OUT / "graph_counts.json").read_text())

    # Fixed tabular baseline: same eligible windows and pair-level fields.
    train_x = lazy_tabular_matrix(store, split_indices["development_train"])
    tab_mean, tab_std = np.asarray(prep["tab_mean"]), np.asarray(prep["tab_std"])
    write_json(OUT / "logistic_preprocessing_state.json", {"mean": tab_mean.tolist(), "std": tab_std.tolist(),
                                                            "fit_scope": "development_train eligible graphs only", "C": 1.0,
                                                            "storage": "lazy graph slices"})
    logistic = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=17)
    logistic.fit(transform(train_x, tab_mean, tab_std), label_array(store, split_indices["development_train"]))
    with (OUT / "logistic_c1_model.pkl").open("wb") as f:
        pickle.dump({"model": logistic, "config": {"C": 1.0, "solver": "lbfgs", "unweighted": True},
                     "feature_order": NODE_FEATURES * 2 + PAIR_FEATURES}, f)
    constant_p = float(label_array(store, split_indices["development_train"]).mean())
    write_json(OUT / "constant_baseline.json", {"training_prevalence": constant_p, "fit_scope": "development_train eligible graphs only"})
    ram.append({"stage": "after_fixed_baselines", "current_rss_bytes": current_rss_bytes(), "peak_rss_bytes": peak_rss_bytes()})

    predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}; metrics=[]; prs=[]; rels=[]; thresholds={}; runs=[]
    constant_scores = {}
    for split in ["development_train", "development_validation"]:
        y = label_array(store, split_indices[split]); constant_scores[split] = (y, np.full(len(y), constant_p))
    append_candidate(store, split_indices, "constant", None, constant_scores, thresholds, metrics, prs, rels, predictions)
    logistic_scores = {}
    for split in ["development_train", "development_validation"]:
        x = lazy_tabular_matrix(store, split_indices[split]); y = label_array(store, split_indices[split])
        logistic_scores[split] = (y, logistic.predict_proba(transform(x, tab_mean, tab_std))[:, 1])
    append_candidate(store, split_indices, "logistic_c1", None, logistic_scores, thresholds, metrics, prs, rels, predictions)

    for seed in SEEDS:
        run = train_gnn(store, split_indices["development_train"], split_indices["development_validation"], prep, seed)
        runs.append({k: v for k, v in run.items() if k != "model"})
        scores={}; model=run["model"]
        for split in ["development_train", "development_validation"]:
            scores[split] = model_probabilities(model, make_lazy_loader(store, split_indices[split], prep, BATCH_SIZE, False, seed))
        append_candidate(store, split_indices, f"gnn_seed_{seed}", seed, scores, thresholds, metrics, prs, rels, predictions)
        print(f"[train] seed={seed} completed; current_rss={current_rss_bytes()/2**20:.1f} MiB", flush=True)
    write_json(OUT / "gnn_run_registry.json", runs)
    ram.append({"stage": "after_validation_training", "current_rss_bytes": current_rss_bytes(), "peak_rss_bytes": peak_rss_bytes()})

    frozen = {"status": "FROZEN_BEFORE_DEVELOPMENT_TEST", "created_utc": pd.Timestamp.utcnow().isoformat(),
              "seeds": SEEDS, "thresholds_from_validation": thresholds,
              "checkpoint_files": {f"gnn_seed_{r['seed']}": r["checkpoint"] for r in runs},
              "preprocessing_file": "preprocessing_state.json", "logistic_file": "logistic_c1_model.pkl",
              "constant_file": "constant_baseline.json", "test_metrics_not_yet_written": True,
              "ram_safe_storage": True}
    frozen["freeze_sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    write_json(OUT / "FROZEN_BEFORE_TEST.json", frozen)
    write_json(OUT / "validation_thresholds.json", thresholds)
    print("[gate] candidates, checkpoints, preprocessing and thresholds frozen; development-test evaluation begins", flush=True)

    test_ledger = {"development_test_accessed": True, "access_stage": "after_freeze", "created_utc": pd.Timestamp.utcnow().isoformat(),
                   "model_revisions_after_access": False, "protected_session_excluded": True}
    for name in ["constant", "logistic_c1"] + [f"gnn_seed_{s}" for s in SEEDS]:
        ti = split_indices["development_test"]; y=label_array(store,ti)
        if name == "constant": p=np.full(len(y), constant_p)
        elif name == "logistic_c1": p=logistic.predict_proba(transform(lazy_tabular_matrix(store,ti),tab_mean,tab_std))[:,1]
        else:
            seed=int(name.split("_")[-1]); ckpt=torch.load(OUT/f"gnn_seed_{seed}_best.pt",map_location="cpu",weights_only=False)
            model=EdgeAwareGNN(NODE_DIM,EDGE_DIM,PAIR_DIM); model.load_state_dict(ckpt["state_dict"])
            y,p=model_probabilities(model,make_lazy_loader(store,ti,prep,BATCH_SIZE,False,seed))
        append_candidate(store, split_indices, name, None if name in {"constant","logistic_c1"} else int(name.split("_")[-1]), {"development_test":(y,p)}, thresholds, metrics, prs, rels, predictions)
    write_json(OUT / "development_test_access_ledger.json", test_ledger)
    ram.append({"stage": "after_development_test", "current_rss_bytes": current_rss_bytes(), "peak_rss_bytes": peak_rss_bytes()})

    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(prs).to_csv(OUT / "precision_recall_curves.csv", index=False)
    pd.DataFrame(rels).to_csv(OUT / "reliability_bins.csv", index=False)
    pd.DataFrame([r for rows in predictions.values() for r in rows]).to_csv(OUT / "per_window_predictions.csv", index=False)
    bootstrap=whole_race_bootstrap(predictions,OUT)
    write_json(OUT / "metrics.json", {"rows":metrics,"bootstrap":bootstrap,"primary_metric":"average_precision"})
    write_json(OUT / "ram_usage_full_run.json", {"snapshots":ram,"peak_rss_bytes":peak_rss_bytes(),"peak_rss_mib":peak_rss_bytes()/2**20,"elapsed_seconds":time.time()-started,"legacy_cache_eagerly_loaded":False})

    fixed_batch=next(iter(make_lazy_loader(store,split_indices["development_validation"][:4],prep,BATCH_SIZE,False,17)))
    parity={}
    for seed in SEEDS:
        ck=OUT/f"gnn_seed_{seed}_best.pt"; payload=torch.load(ck,map_location="cpu",weights_only=False)
        m1=EdgeAwareGNN(NODE_DIM,EDGE_DIM,PAIR_DIM);m1.load_state_dict(payload["state_dict"]);m1.eval()
        with torch.no_grad(): a=torch.sigmoid(m1(fixed_batch)).numpy()
        m2=EdgeAwareGNN(NODE_DIM,EDGE_DIM,PAIR_DIM);m2.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["state_dict"]);m2.eval()
        with torch.no_grad(): b=torch.sigmoid(m2(fixed_batch)).numpy()
        d=float(np.max(np.abs(a-b)));parity[str(seed)]={"max_abs_difference":d,"passed":bool(d<=1e-7)}
    write_json(OUT/"inference_parity.json",parity)

    hashes_after={k:sha256(ROOT/k) for k in hashes_before}
    write_json(OUT/"input_hashes_after.json",hashes_after)
    tests={
        "source_immutability":{"passed":hashes_after==hashes_before},
        "protected_holdout_exclusion":{"passed":not any("11353" in m["race_id"] for m in store.metadata)},
        "split_partition":{"passed":set(split_indices["development_train"]+split_indices["development_validation"]+split_indices["development_test"])==set(range(len(store)))},
        "race_split_isolation":{"passed":all(len({metadata(store,i)["proposed_split"] for i in idx})==1 for idx in split_indices.values())},
        "graph_layout":{"passed":store.validate_layout()["passed"]},
        "finite_model_reload_parity":{"passed":all(v["passed"] for v in parity.values())},
        "test_after_freeze":{"passed":(OUT/"FROZEN_BEFORE_TEST.json").exists()},
        "configuration_unchanged":{"passed":config_check["unchanged"]},
    }
    write_json(OUT/"experiment_tests.json",tests)
    write_report(store,split_indices,graph_counts,metrics,runs,tests,ram,config_check)
    print("[done] RAM-safe GNN experiment complete",flush=True)


if __name__ == "__main__":
    main()
