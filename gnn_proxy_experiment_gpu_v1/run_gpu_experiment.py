#!/usr/bin/env python3
"""Run the frozen APEX-R GNN proxy experiment on the verified CUDA device.

This is a separate execution artifact. It reads the existing RAM-safe,
disk-backed graph store and writes only to this directory; the CPU v1
experiment and its model files are never overwritten. The only intentional
configuration difference is execution device: CPU -> CUDA. Architecture,
features, graph topology, loss, optimizer, seeds, splits and stopping rules
remain frozen.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_OUT = ROOT / "gnn_proxy_experiment_v1"
SOURCE_STORE = SOURCE_OUT / "disk_graph_store"
OUT = HERE

sys.path.insert(0, str(SOURCE_OUT))
import run_gnn_experiment_ram_safe as ram  # noqa: E402


DEVICE = torch.device("cuda")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_experiment_spec() -> None:
    """Write the frozen contract before any fit starts."""
    spec = f"""# APEX-R GNN proxy experiment — CUDA execution variant

This is a separate run under gnn_proxy_experiment_gpu_v1/. The canonical
CPU/RAM-safe experiment under gnn_proxy_experiment_v1/ is preserved.

## Task

Fixed attacker/target next-lap-boundary classified-order position-swap proxy.
1 means the fixed pair's endpoint order reverses under the existing Phase 3
contract. This is not a verified on-track-overtake label, attack-conditioned
probability, energy-strategy result, or deployment claim. Unknown/censored
windows are excluded exactly as in the audited graph cache.

## Data and split

- Source graph store: {SOURCE_STORE}
- Graph cache version: gnn-proxy-graph-cache-v1
- Existing graph construction: backward-only as-of joins, 1.0 second
  tolerance, 10 second trailing lookback, no interpolation/backfill/future
  joins, and the existing directed up-to-three-nearest-finite-XYZ topology.
- Whole-race proposed chronology split: 26 development-train / 6
  development-validation / 5 development-test races.
- The five development-test races are previously accessed development data,
  not an untouched final holdout. Protected session 11353 is excluded.

## Features actually available in this audited graph cache

- Speed and gap/proximity: speed_kmh, distance_m, relative_speed_kmh,
  attacker/target speed and distance deltas.
- Positional/timing: x_m, y_m, z_m, session-time-derived sample ages,
  and trailing 10-second speed delta/trend.
- Raw telemetry context: throttle, brake, gear, RPM and DRS state, plus
  explicit missingness masks and attacker/target role indicators.

## Requested signals unavailable and excluded

Measured tyre wear/age, ERS/energy delta, energy/power telemetry, battery
state-of-health, and battery/ES temperature do not exist in the approved
continuous telemetry schema. Values from the APEX-R simulation engine are
modelled state, not historical measured inputs, so they are not used as
training features. GPS latitude/longitude is also not present; the approved
positional fields are FastF1-derived track-frame x_m/y_m/z_m.

## Frozen model configuration

- Model: two-layer edge-aware GINEConv, hidden width 32, dropout 0.2.
- Optimizer: Adam, learning rate 0.001, weight decay 0.0001.
- Loss: unweighted BCEWithLogitsLoss.
- Batch size 32, maximum 100 epochs, early stopping patience 15 on
  validation average precision.
- Seeds: 17, 23, 42. No broad hyperparameter search, resampling, SMOTE or
  automatic calibration.
- Preprocessing: training-only mean/std using the existing formulas;
  missing values are replaced by training means at transform time and the
  explicit masks remain. Raw graph arrays remain float64 on disk and the
  existing Data boundary remains float32.
- Baselines: constant training prevalence and regularized logistic regression
  with C=1.0 on the same eligible pair-level inputs.
- Primary metric: validation average precision; ROC-AUC, Brier, log loss,
  PR curves, reliability bins and whole-race bootstrap are also reported.

## Execution-device exception

All locked experiment parameters above are unchanged. The explicit execution
change is device: cpu -> cuda on {torch.cuda.get_device_name(0)} using
PyTorch {torch.__version__}. GPU execution can differ numerically from CPU
execution even with the same seeds; bit-for-bit equality is not promised.
"""
    (OUT / "EXPERIMENT_SPEC.md").write_text(spec)

    availability = {
        "available_and_used": {
            "speed_gap": [
                "speed_kmh", "distance_m", "relative_speed_kmh",
                "attacker_target_speed_delta_kmh", "attacker_target_distance_m",
            ],
            "gps_positional_timing": [
                "x_m", "y_m", "z_m", "car_sample_age_sec",
                "position_sample_age_sec", "session_time_sec-derived ages",
            ],
            "raw_power_related_proxies": [
                "throttle_pct", "brake", "rpm", "n_gear", "drs_binary",
            ],
        },
        "unavailable_and_excluded": {
            "energy_delta": "No measured ERS/battery telemetry in audited source.",
            "energy_power": "No measured power channel in audited source; engine power is simulated.",
            "tyre_wear": "No causally safe tyre-wear stream in this graph cache.",
            "state_of_health": "No battery state-of-health channel in audited source.",
            "battery_es_temperature": "No battery/energy-storage temperature channel in audited source.",
            "gps_latitude_longitude": "Only track-frame x/y/z are available.",
        },
    }
    write_json(OUT / "FEATURE_AVAILABILITY.json", availability)


def prepare_metadata() -> None:
    """Copy only small audit metadata; never duplicate raw telemetry/cache."""
    allowed = {"run_gpu_experiment.py"}
    if OUT.exists() and any(p.name not in allowed for p in OUT.iterdir()):
        existing = sorted(p.name for p in OUT.iterdir() if p.name not in allowed)
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {existing[:10]}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in [
        "graph_counts.json", "graph_build_manifest.json", "graph_coverage.csv",
        "graph_exclusion_summary.csv", "graph_positive_lineage.csv",
        "labelled_window_context_audit.csv", "frozen_split_manifest.json",
        "GRAPH_CACHE_SCHEMA.json",
    ]:
        src = SOURCE_OUT / name
        if src.exists():
            shutil.copy2(src, OUT / name)
    write_experiment_spec()


def gpu_model_probabilities(model: nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    model.to(DEVICE)
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite CUDA inference logits")
            probs = torch.sigmoid(logits)
            if not torch.isfinite(probs).all():
                raise FloatingPointError("Non-finite CUDA inference probabilities")
            ys.append(batch.y.view(-1).detach().cpu().numpy())
            ps.append(probs.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def gpu_train_gnn(store, train_indices, val_indices, prep, seed: int) -> dict[str, Any]:
    """Exact frozen training loop with explicit batch/model CUDA transfer."""
    ram.set_seed(seed)
    train_loader = ram.make_lazy_loader(store, train_indices, prep, ram.BATCH_SIZE, True, seed)
    val_loader = ram.make_lazy_loader(store, val_indices, prep, ram.BATCH_SIZE, False, seed)
    model = ram.EdgeAwareGNN(ram.NODE_DIM, ram.EDGE_DIM, ram.PAIR_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    best_ap = -float("inf")
    best_epoch = 0
    stale = 0
    best_state = None
    history = []
    started = ram.time.time()
    for epoch in range(1, ram.MAX_EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite CUDA logits")
            loss = criterion(logits, batch.y.view(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CUDA loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite CUDA gradient")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        y, p = gpu_model_probabilities(model, val_loader)
        ap = ram.compute_metrics(y, p)["average_precision"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_ap": ap})
        if ap is not None and ap > best_ap + 1e-12:
            best_ap = float(ap)
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"[cuda-train] seed={seed} epoch={epoch} loss={np.mean(losses):.6f} val_ap={ap}", flush=True)
        if stale >= ram.PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"No CUDA checkpoint for seed {seed}")
    model.load_state_dict(best_state)
    checkpoint = {
        "state_dict": best_state,
        "model_class": "EdgeAwareGNN",
        "config": {
            "node_dim": ram.NODE_DIM, "edge_dim": ram.EDGE_DIM, "pair_dim": ram.PAIR_DIM,
            "hidden": ram.HIDDEN, "dropout": 0.2, "optimizer": "Adam", "lr": 0.001,
            "weight_decay": 0.0001, "batch_size": ram.BATCH_SIZE,
            "loss": "unweighted BCEWithLogitsLoss", "seed": seed, "device": "cuda",
        },
        "best_epoch": best_epoch, "epochs_run": len(history), "best_validation_ap": best_ap,
        "preprocessing_version": prep["version"], "storage": "disk_graph_store lazy slices",
        "execution_device": torch.cuda.get_device_name(0),
    }
    torch.save(checkpoint, OUT / f"gnn_seed_{seed}_best.pt")
    write_json(OUT / f"gnn_seed_{seed}_history.json", history)
    return {
        "seed": seed, "checkpoint": f"gnn_seed_{seed}_best.pt", "best_epoch": best_epoch,
        "epochs_run": len(history), "best_validation_ap": best_ap,
        "training_seconds": ram.time.time() - started, "model": model,
    }


def gpu_verify_frozen_config() -> dict[str, Any]:
    before = json.loads(ram.CONFIG_BEFORE.read_text())
    expected = {
        "architecture": "EdgeAwareGNN using two GINEConv layers",
        "hidden": ram.HIDDEN, "dropout": 0.2, "optimizer": "Adam", "learning_rate": 0.001,
        "weight_decay": 0.0001, "batch_size": ram.BATCH_SIZE, "max_epochs": ram.MAX_EPOCHS,
        "early_stopping_patience": ram.PATIENCE, "seeds": ram.SEEDS,
        "loss": "unweighted BCEWithLogitsLoss", "device": "cuda",
        "data_loader_num_workers": 0, "drop_last": False,
        "asof_tolerance_sec": ram.ASOF_TOLERANCE_SEC, "lookback_sec": ram.LOOKBACK_SEC,
        "graph_topology": "directed up to 3 nearest finite XYZ neighbours; no classified adjacency",
        "target": "fixed-pair next lap-boundary classified-order position-swap proxy; unknown/censored excluded",
        "split": "proposed chronology-only 26 development_train / 6 development_validation / 5 development_test",
    }
    mismatches = {
        k: {"before": before.get(k), "expected": v}
        for k, v in expected.items()
        if k != "device" and before.get(k) != v
    }
    if mismatches:
        raise RuntimeError("Locked configuration mismatch: " + json.dumps(mismatches))
    result = {
        "configuration_before": before,
        "configuration_assertions": expected,
        "locked_configuration_unchanged": True,
        "execution_device_change": {"from": before.get("device"), "to": "cuda"},
        "mismatches": mismatches,
        "unchanged": True,
    }
    write_json(OUT / "gpu_config_after.json", result)
    write_json(OUT / "ram_fix_config_after.json", {
        **before, "device": "cuda", "execution_device_change": "cpu -> cuda",
        "configuration_unchanged_except_execution_device": True,
    })
    return result


def write_gpu_summary() -> None:
    metrics = json.loads((OUT / "metrics.json").read_text())
    runs = json.loads((OUT / "gnn_run_registry.json").read_text())
    tests = json.loads((OUT / "experiment_tests.json").read_text())
    graph_counts = json.loads((OUT / "graph_counts.json").read_text())
    lines = [
        "# APEX-R GNN proxy experiment — CUDA run summary",
        "",
        "This report summarizes the executed GPU variant. The target remains the",
        "fixed-pair next-lap-boundary classified-order proxy, not verified overtaking",
        "or energy-strategy effectiveness.",
        "",
        f"- GPU: {torch.cuda.get_device_name(0)}",
        f"- PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}",
        f"- Eligible graphs: {graph_counts['graphs_eligible']} / {graph_counts['windows_labelled']} labelled windows",
        f"- Graph exclusions: {graph_counts['graphs_excluded']}",
        "- Data use: all eligible known-label graphs from the audited 37-race store;",
        "  unknown/censored windows are not valid supervised targets and remain excluded.",
        "- Development-test was evaluated once after FROZEN_BEFORE_TEST.json; it is",
        "  previously accessed development data, not an untouched final holdout.",
        "",
        "## Requested signal availability",
        "",
        "Used: speed, physical gap/proximity, relative speed, XYZ track position,",
        "session timing/sample ages, speed trends, throttle, brake, gear, RPM and DRS",
        "with explicit missingness masks. Not available in the audited source and not",
        "invented: measured energy delta, power, tyre wear, battery state-of-health,",
        "battery/ES temperature, or GPS latitude/longitude.",
        "",
        "## Training runs",
        "",
        "The complete run registry is in gnn_run_registry.json.",
        json.dumps(runs, indent=2, sort_keys=True),
        "",
        "## Metrics",
        "",
        "See metrics.csv for every baseline/seed and split, precision_recall_curves.csv",
        "for full curves, reliability_bins.csv for bin counts, and whole_race_bootstrap.csv",
        "for race-level uncertainty.",
        json.dumps(metrics, indent=2, sort_keys=True),
        "",
        "## Verification",
        "",
        json.dumps(tests, indent=2, sort_keys=True),
        "",
        "GPU execution is not promised bit-for-bit identical to the prior CPU run",
        "because CUDA kernels can differ numerically even with unchanged seeds.",
    ]
    (OUT / "GPU_RUN_REPORT.md").write_text("\n".join(lines) + "\n")


def write_manifest() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append({
                "path": str(path.relative_to(OUT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    write_json(OUT / "GPU_ARTIFACT_MANIFEST.json", {"files": rows, "file_count": len(rows)})
    sums = "".join(f"{r['sha256']}  {r['path']}\n" for r in rows)
    (OUT / "SHA256SUMS.txt").write_text(sums)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the selected GPU environment")
    prepare_metadata()
    ram.OUT = OUT
    ram.STORE_DIR = SOURCE_STORE
    ram.LEGACY_CACHE = SOURCE_OUT / "graph_cache.pt"
    ram.HASHES_BEFORE = SOURCE_OUT / "ram_fix_input_hashes_before.json"
    ram.CONFIG_BEFORE = SOURCE_OUT / "ram_fix_config_before.json"
    ram.DEVICE = DEVICE
    ram.verify_frozen_config = gpu_verify_frozen_config
    ram.train_gnn = gpu_train_gnn
    ram.model_probabilities = gpu_model_probabilities
    print(f"[cuda] using {torch.cuda.get_device_name(0)}", flush=True)
    ram.main()
    write_gpu_summary()
    write_manifest()
    print(f"[done] CUDA experiment written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
