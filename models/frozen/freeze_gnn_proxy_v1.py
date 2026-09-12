#!/usr/bin/env python3
"""Build the frozen APEX-R GNN proxy v1 bundle without training or mutation."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "gnn_proxy_experiment_gpu_v1"
TARGET = ROOT / "models" / "frozen" / "gnn_proxy_v1"
FIXTURE_SOURCE = ROOT / "phase5_dashboard" / "fixture" / "phase5_fixture.json"
DATASET_ZIP = ROOT / "phase3_passed_races" / "APEX-R_37_Passed_Races.zip"
EXPECTED_CHECKPOINT_SHA256 = "d5ce7258fa2ec62f953d516816497f1cdbd04b1b839e362ef447f79c4e618d14"
EXPECTED_DATASET_ZIP_SHA256 = "410922cd2226b54961d9d25601a90dc21b2fe5906f0bbb4badb23d4cd96ffd56"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_metric(metrics: dict, split: str) -> dict:
    matches = [
        row for row in metrics["rows"]
        if row.get("model") == "gnn_seed_42" and row.get("seed") == 42 and row.get("split") == split
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one seed-42 metric row for {split}, found {len(matches)}")
    return matches[0]


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    required_static = [TARGET / "README.md", TARGET / "MODEL_CARD.md", TARGET / "smoke_test.py"]
    missing_static = [str(path) for path in required_static if not path.is_file()]
    if missing_static:
        raise RuntimeError(f"missing static frozen-bundle files: {missing_static}")

    source_map = {
        SOURCE / "gnn_seed_42_best.pt": TARGET / "gnn_seed_42_best.pt",
        SOURCE / "EXPERIMENT_SPEC.md": TARGET / "EXPERIMENT_SPEC.md",
        SOURCE / "FEATURE_AVAILABILITY.json": TARGET / "FEATURE_SCHEMA.json",
        SOURCE / "GRAPH_CACHE_SCHEMA.json": TARGET / "GRAPH_SCHEMA.json",
        SOURCE / "preprocessing_state.json": TARGET / "preprocessing_state.json",
        SOURCE / "frozen_split_manifest.json": TARGET / "frozen_split_manifest.json",
        SOURCE / "metrics.json": TARGET / "metrics.json",
        SOURCE / "gnn_run_registry.json": TARGET / "gnn_run_registry.json",
        SOURCE / "graph_counts.json": TARGET / "graph_counts.json",
        SOURCE / "gpu_config_after.json": TARGET / "gpu_config_after.json",
        SOURCE / "GPU_RUN_REPORT.md": TARGET / "EXPERIMENT_REPORT.md",
    }
    for source in [*source_map, FIXTURE_SOURCE, DATASET_ZIP]:
        if not source.is_file():
            raise FileNotFoundError(source)

    source_hashes_before = {str(path.relative_to(ROOT)): sha256(path) for path in [*source_map, FIXTURE_SOURCE, DATASET_ZIP]}
    if source_hashes_before[str((SOURCE / "gnn_seed_42_best.pt").relative_to(ROOT))] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("selected checkpoint hash differs from the verified source hash")
    if source_hashes_before[str(DATASET_ZIP.relative_to(ROOT))] != EXPECTED_DATASET_ZIP_SHA256:
        raise RuntimeError("37-race dataset archive hash differs from its audited hash")

    import torch

    checkpoint = torch.load(SOURCE / "gnn_seed_42_best.pt", map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    required_checkpoint = {
        "model_class": "EdgeAwareGNN",
        "best_epoch": 28,
        "best_validation_ap": 0.12760248880353492,
    }
    for key, expected in required_checkpoint.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"checkpoint {key} mismatch: {checkpoint.get(key)!r} != {expected!r}")
    required_config = {
        "node_dim": 18,
        "edge_dim": 4,
        "pair_dim": 12,
        "hidden": 32,
        "dropout": 0.2,
        "optimizer": "Adam",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 32,
        "loss": "unweighted BCEWithLogitsLoss",
        "seed": 42,
    }
    if config != {**required_config, "device": "cuda"}:
        raise RuntimeError(f"checkpoint configuration mismatch: {config!r}")

    registry = json.loads((SOURCE / "gnn_run_registry.json").read_text(encoding="utf-8"))
    selected_runs = [run for run in registry if run.get("seed") == 42]
    if len(selected_runs) != 1 or selected_runs[0].get("best_epoch") != 28:
        raise RuntimeError("run registry does not uniquely identify seed 42 / epoch 28")

    graph_counts = json.loads((SOURCE / "graph_counts.json").read_text(encoding="utf-8"))
    expected_counts = {
        "graphs_eligible": 15404,
        "eligible_positive": 822,
        "eligible_negative": 14582,
        "graphs_excluded": 19,
    }
    actual_counts = {key: graph_counts.get(key) for key in expected_counts}
    if actual_counts != expected_counts:
        raise RuntimeError(f"graph-count mismatch: {actual_counts!r}")
    split_races = sum(part["races"] for part in graph_counts["by_split"].values())
    if split_races != 37:
        raise RuntimeError(f"expected 37 split races, found {split_races}")

    metrics = json.loads((SOURCE / "metrics.json").read_text(encoding="utf-8"))
    validation = selected_metric(metrics, "development_validation")
    development_test = selected_metric(metrics, "development_test")

    for source, destination in source_map.items():
        shutil.copy2(source, destination)

    fixture = json.loads(FIXTURE_SOURCE.read_text(encoding="utf-8"))
    smoke_fixture = {
        "schema_version": fixture["schema_version"],
        "selection": fixture["selection"],
        "timestamp_semantics": fixture["timestamp_semantics"],
        "decision": fixture["decision"],
        "inference_graph": fixture["inference_graph"],
        "label_and_outcome_included": False,
        "purpose": "Already-approved non-protected graph for local inference-only smoke testing",
    }
    rendered_fixture = json.dumps(smoke_fixture, sort_keys=True)
    if "11353" in rendered_fixture:
        raise RuntimeError("protected-session identifier found in approved smoke fixture")
    write_json(TARGET / "approved_smoke_fixture.json", smoke_fixture)

    model_config = {
        "model_version": "gnn_proxy_v1",
        "model_class": checkpoint["model_class"],
        "checkpoint_config": config,
        "best_epoch": checkpoint["best_epoch"],
        "epochs_run": checkpoint["epochs_run"],
        "best_validation_ap": checkpoint["best_validation_ap"],
        "preprocessing_version": checkpoint["preprocessing_version"],
        "storage": checkpoint["storage"],
        "architecture": {
            "message_passing": "two GINEConv layers",
            "hidden_width": 32,
            "dropout": 0.2,
            "decoder": "attacker embedding + target embedding + pair features",
        },
        "output_label": "experimental boundary position-swap proxy signal",
    }
    write_json(TARGET / "model_config.json", model_config)

    metrics_summary = {
        "model_version": "gnn_proxy_v1",
        "selected_seed": 42,
        "selected_epoch": 28,
        "task": "fixed-pair next-lap-boundary classified-order position-swap proxy",
        "counts": {**expected_counts, "races": split_races},
        "development_validation": validation,
        "development_test": development_test,
        "selection_policy": "highest validation average precision among predeclared seeds; development-test performance not used for checkpoint selection",
        "warning": "Metrics are for a boundary position-swap proxy, not verified on-track overtakes.",
    }
    write_json(TARGET / "metrics_summary.json", metrics_summary)

    source_hashes_after = {str(path.relative_to(ROOT)): sha256(path) for path in [*source_map, FIXTURE_SOURCE, DATASET_ZIP]}
    unchanged = source_hashes_before == source_hashes_after
    source_integrity = {
        "status": "PASS" if unchanged else "FAIL",
        "source_artifacts_unchanged": unchanged,
        "before_sha256": source_hashes_before,
        "after_sha256": source_hashes_after,
        "note": "The freeze process only read source artifacts and copied selected files into the versioned frozen directory.",
    }
    write_json(TARGET / "SOURCE_IMMUTABILITY.json", source_integrity)
    if not unchanged:
        raise RuntimeError("one or more source artifacts changed during freeze")

    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(TARGET.iterdir())
        if path.is_file() and path.name not in {"FROZEN_MODEL_MANIFEST.json", "SHA256SUMS.txt"}
    }
    manifest = {
        "model_version": "gnn_proxy_v1",
        "display_name": "APEX-R GNN Proxy v1",
        "output_label": "experimental boundary position-swap proxy signal",
        "task": "fixed-pair next-lap-boundary classified-order position-swap proxy",
        "selected_seed": 42,
        "selected_epoch": 28,
        "checkpoint_path": "models/frozen/gnn_proxy_v1/gnn_seed_42_best.pt",
        "checkpoint_sha256": artifact_hashes["gnn_seed_42_best.pt"],
        "artifact_sha256": artifact_hashes,
        "dataset": {
            "description": "37 telemetry-coverage PASS races",
            "archive_path": str(DATASET_ZIP),
            "archive_bytes": DATASET_ZIP.stat().st_size,
            "archive_sha256": source_hashes_after[str(DATASET_ZIP.relative_to(ROOT))],
            "supervised_graphs": 15404,
            "positive_proxy_windows": 822,
            "negative_proxy_windows": 14582,
            "excluded_missing_attacker_target_telemetry": 19,
        },
        "experiment_report_path": str(SOURCE / "GPU_RUN_REPORT.md"),
        "packaged_experiment_report": "EXPERIMENT_REPORT.md",
        "metrics": {
            "development_validation": {
                "average_precision": validation["average_precision"],
                "roc_auc": validation["roc_auc"],
                "brier_score": validation["brier_score"],
                "log_loss": validation["log_loss"],
            },
            "development_test": {
                "average_precision": development_test["average_precision"],
                "roc_auc": development_test["roc_auc"],
                "brier_score": development_test["brier_score"],
                "log_loss": development_test["log_loss"],
            },
        },
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protected_holdout_exclusion": "Protected session 11353 and every declared sealed holdout were excluded and were not accessed during freezing or smoke testing.",
        "selection_statement": "Seed 42, epoch 28 was selected by validation average precision before development-test evaluation; no retraining or weight change occurred during freezing.",
        "usage": "Advisory proxy signal only; the APEX-R rule/energy simulator makes the final recommendation.",
    }
    write_json(TARGET / "FROZEN_MODEL_MANIFEST.json", manifest)

    checksum_entries = []
    for path in sorted(TARGET.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksum_entries.append(f"{sha256(path)}  {path.name}")
    (TARGET / "SHA256SUMS.txt").write_text("\n".join(checksum_entries) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "target": str(TARGET),
        "checkpoint_sha256": artifact_hashes["gnn_seed_42_best.pt"],
        "dataset_zip_sha256": EXPECTED_DATASET_ZIP_SHA256,
        "source_artifacts_unchanged": unchanged,
        "files": len([path for path in TARGET.iterdir() if path.is_file()]),
    }, indent=2))


if __name__ == "__main__":
    main()
