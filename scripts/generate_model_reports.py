#!/usr/bin/env python3
"""Generate consolidated and per-model APEX-R model cards from saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
METRIC_KEYS = ("roc_auc", "average_precision", "brier_score", "accuracy_at_0_5")


def load(name: str) -> dict | None:
    path = MODELS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def metadata(artifact: dict) -> dict:
    return artifact.get("metadata", artifact)


def metric_row(name: str, values: dict) -> str:
    return "| " + name + " | " + " | ".join(str(values.get(key, "--")) for key in METRIC_KEYS) + " |"


def individual_report(name: str, artifact: dict, comparison_group: str, notes: list[str]) -> str:
    info = metadata(artifact)
    metrics = info.get("metrics", {})
    train_sessions = info.get("train_sessions", [])
    validation_sessions = info.get("validation_sessions", info.get("test_sessions", []))
    lines = [
        f"# APEX-R model card: {name}", "",
        f"Artifact: `{artifact.get('version', '--')}`", "",
        "## Training", "",
        f"- Algorithm: {info.get('algorithm', '--')}",
        f"- Train sessions: {', '.join(map(str, train_sessions)) or '--'}",
        f"- Evaluation/validation sessions: {', '.join(map(str, validation_sessions)) or '--'}",
        f"- Train rows / positives: {info.get('train_rows', '--')} / {info.get('train_positive', '--')}",
        f"- Target: {info.get('target', '--')}",
        f"- Comparison group: {comparison_group}", "",
        "## Evaluation", "",
        "| ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |",
        "| ---: | ---: | ---: | ---: |",
        "| " + " | ".join(str(metrics.get(key, "--")) for key in METRIC_KEYS) + " |", "",
        "## Features", "", ", ".join(artifact.get("features", [])) or "--", "",
        "## Hyperparameters", "", "```json", json.dumps(info.get("hyperparameters", {}), indent=2), "```", "",
        "## Interpretation", "",
    ]
    lines.extend("- " + note for note in notes)
    lines.append("")
    return "\n".join(lines)


def write_reports(models_dir: Path = MODELS) -> None:
    global MODELS
    MODELS = models_dir
    logistic = load("overtake_model.json")
    legacy_xgb = load("xgboost_overtake_model.json")
    clean_base = load("xgboost_clean_label_base_model.json")
    enriched = load("xgboost_enriched_overtake_model.json")
    comparison = load("enriched_model_comparison.json") or {}
    final_test = load("final_holdout_test.json")
    active = comparison.get("active_model", "--")

    files = [
        ("LOGISTIC_REGRESSION_REPORT.md", "Logistic Regression baseline", logistic, "Legacy-label hold-out comparison", ["Uses the original observed-overtake label and is directly comparable only with the legacy XGBoost report.", "This model is retained as the explainable baseline."]),
        ("LEGACY_XGBOOST_REPORT.md", "Legacy XGBoost", legacy_xgb, "Legacy-label hold-out comparison", ["Uses the original observed-overtake label and is directly comparable only with Logistic Regression.", "It is not used to select the clean-label active model."]),
        ("CLEAN_LABEL_BASE_XGBOOST_REPORT.md", "Clean-label base-feature XGBoost", clean_base, "Clean-label validation comparison", ["Uses the cleaned label and the original eight interval/race-state features.", "This is the active browser model because it won the like-for-like ROC-AUC comparison."]),
        ("ENRICHED_MODEL_REPORT.md", "Car-data enriched XGBoost", enriched, "Clean-label validation comparison", ["Adds rolling car telemetry features: speed, throttle, brake, RPM, gear and DRS.", "Location is cached/audited for replay provenance, but excluded from prediction because coordinates are local to each circuit."]),
    ]
    for filename, title, artifact, group, notes in files:
        if artifact:
            (MODELS / filename).write_text(individual_report(title, artifact, group, notes), encoding="utf-8")

    legacy_rows = []
    if logistic:
        legacy_rows.append(metric_row("Logistic Regression", metadata(logistic).get("metrics", {})))
    if legacy_xgb:
        legacy_rows.append(metric_row("Legacy XGBoost", metadata(legacy_xgb).get("metrics", {})))
    clean_rows = []
    if clean_base:
        clean_rows.append(metric_row("Clean-label base-feature XGBoost", metadata(clean_base).get("metrics", {})))
    if enriched:
        clean_rows.append(metric_row("Car-data enriched XGBoost", metadata(enriched).get("metrics", {})))
    all_lines = [
        "# APEX-R: all machine-learning models", "",
        f"Active browser model: **{active}**.", "",
        "## How to read this report", "",
        "There are two separate evaluation groups. Do not compare their scores directly: the clean-label group removes pit/neutralisation-adjacent examples and therefore has a different target distribution.", "",
        "## Legacy-label hold-out comparison", "",
        "Train: 7953, 7779, 7787. Held-out session: 9070. Features: interval, position, lap, stint and weather context.", "",
        "| Model | ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *legacy_rows, "",
        "## Clean-label XGBoost comparison", "",
        "Train: 7953, 7779, 7787. Validation session: 9070. Both models use the exact same cleaned labels and split; this is the valid selection comparison.", "",
        "| Model | ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *clean_rows, "",
        "## Data and label flow", "",
        "- Base features: `intervals`, `position`, `laps`, `stints`, `weather`.",
        "- Label source: `overtakes`; samples near `pit` and non-green/safety-car `race_control` records are removed in the clean-label group.",
        "- Enriched feature source: `car_data`; `location` is retained for replay/audit provenance, not for the model.",
        "- All API payloads are cached under `data/openf1-cache/` for reproducibility.", "",
        "## Individual model cards", "",
        "- `LOGISTIC_REGRESSION_REPORT.md`", "- `LEGACY_XGBOOST_REPORT.md`", "- `CLEAN_LABEL_BASE_XGBOOST_REPORT.md`", "- `ENRICHED_MODEL_REPORT.md`", "",
    ]
    if final_test:
        final_metrics = final_test.get("metrics", {})
        all_lines.extend([
            "## Locked final hold-out test", "",
            f"Session: {final_test.get('test_session', {}).get('session_key', '--')}; model: `{final_test.get('model', {}).get('version', '--')}`.",
            f"ROC-AUC: {final_metrics.get('roc_auc', '--')}; average precision: {final_metrics.get('average_precision', '--')}; Brier score: {final_metrics.get('brier_score', '--')}; F1 at the validation-selected threshold: {final_metrics.get('f1_at_selected_threshold', '--')}.",
            "See `FINAL_HOLDOUT_TEST_REPORT.md` for the locked-test protocol and confusion matrix.", "",
        ])
    all_lines.extend([
        "## Important limitation", "",
        "These models estimate overtake opportunity, not an ATTACK/HOLD/HARVEST/DEFEND label. APEX-R's strategy engine makes the final constrained action decision. Public OpenF1 data does not contain private ERS or battery telemetry.", "",
    ])
    (MODELS / "ALL_MODELS_REPORT.md").write_text("\n".join(all_lines), encoding="utf-8")
    (MODELS / "MODEL_REPORT.md").write_text("\n".join(all_lines), encoding="utf-8")
    (MODELS / "README.md").write_text(
        "# APEX-R model artifacts\n\n"
        "Start with `ALL_MODELS_REPORT.md` for the complete comparison. Individual reports are stored beside it.\n\n"
        "- `LOGISTIC_REGRESSION_REPORT.md`\n- `LEGACY_XGBOOST_REPORT.md`\n- `CLEAN_LABEL_BASE_XGBOOST_REPORT.md`\n- `ENRICHED_MODEL_REPORT.md`\n- `FINAL_HOLDOUT_TEST_REPORT.md`\n\n"
        "Artifacts: `overtake_model.json`, `xgboost_overtake_model.json`, `xgboost_clean_label_base_model.json`, and `xgboost_enriched_overtake_model.json`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_reports()
