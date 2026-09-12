#!/usr/bin/env python3
"""Score the selected APEX-R XGBoost model on one untouched OpenF1 race.

This script never fits or alters a model.  It recreates the clean-label base
features for the supplied race, evaluates the saved browser model, and writes
a final hold-out report.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import train_model as base
from train_xgboost_enriched import CONTROL_BUFFER_SECONDS, HORIZON_SECONDS, PIT_BUFFER_SECONDS, control_times, nearest_in_window, optional_api_get
from train_xgboost import sigmoid, tree_margin


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=int, default=11353, help="Untouched completed race session to score (default: 2026 Dutch GP).")
    parser.add_argument("--model", type=Path, default=MODELS / "xgboost_clean_label_base_model.json")
    parser.add_argument("--cache-dir", type=Path, default=base.CACHE)
    return parser.parse_args()


def clean_base_rows(session_key: int, cache_dir: Path) -> tuple[list[dict], dict]:
    """Recreate the same cleaned labels used by the selected base model."""
    raw_rows, base_manifest = base.session_rows(session_key, cache_dir)
    pit, pit_url, pit_available = optional_api_get("pit", {"session_key": session_key}, cache_dir)
    race_control, control_url, control_available = optional_api_get("race_control", {"session_key": session_key}, cache_dir)
    overtakes, overtakes_url = base.api_get("overtakes", {"session_key": session_key}, cache_dir)
    pits: dict[int, list[float]] = defaultdict(list)
    for row in pit:
        t, driver = base.timestamp(row.get("date")), row.get("driver_number")
        if t is not None and driver is not None:
            pits[int(driver)].append(t)
    for values in pits.values():
        values.sort()
    controls = control_times(race_control)
    events: dict[int, list[float]] = defaultdict(list)
    excluded_events = 0
    for row in overtakes:
        t, attacking, defending = base.timestamp(row.get("date")), row.get("overtaking_driver_number"), row.get("overtaken_driver_number")
        if t is None or attacking is None:
            continue
        involved = [int(attacking)] + ([int(defending)] if defending is not None else [])
        pit_adjacent = any(nearest_in_window(pits.get(driver, []), t - PIT_BUFFER_SECONDS, t + PIT_BUFFER_SECONDS) for driver in involved)
        neutralised = nearest_in_window(controls, t - CONTROL_BUFFER_SECONDS, t + CONTROL_BUFFER_SECONDS)
        if pit_adjacent or neutralised:
            excluded_events += 1
        else:
            events[int(attacking)].append(t)
    for values in events.values():
        values.sort()
    rows, excluded_windows = [], 0
    for row in raw_rows:
        t, driver = base.timestamp(row["date"]), int(row["driver_number"])
        if t is None:
            continue
        pit_adjacent = nearest_in_window(pits.get(driver, []), t - PIT_BUFFER_SECONDS, t + HORIZON_SECONDS + PIT_BUFFER_SECONDS)
        neutralised = nearest_in_window(controls, t - CONTROL_BUFFER_SECONDS, t + HORIZON_SECONDS + CONTROL_BUFFER_SECONDS)
        if pit_adjacent or neutralised:
            excluded_windows += 1
            continue
        driver_events = events.get(driver, [])
        index = bisect.bisect_right(driver_events, t)
        label = int(index < len(driver_events) and driver_events[index] <= t + HORIZON_SECONDS)
        rows.append({"features": row["features"], "label": label})
    manifest = {**base_manifest, "session_key": session_key, "rows_before_cleaning": len(raw_rows), "rows_after_cleaning": len(rows), "positive_after_cleaning": sum(row["label"] for row in rows), "raw_overtakes": len(overtakes), "excluded_overtakes": excluded_events, "excluded_windows": excluded_windows, "pit_endpoint_available": pit_available, "race_control_endpoint_available": control_available, "urls": {**base_manifest["urls"], "pit": pit_url, "race_control": control_url, "overtakes_clean_label": overtakes_url}}
    return rows, manifest


def main() -> int:
    args = parse_args()
    artifact = json.loads(args.model.read_text(encoding="utf-8"))
    if artifact.get("model_type") != "xgboost_binary_logistic":
        raise SystemExit("This evaluator currently supports the saved XGBoost browser artifacts.")
    training_sessions = set(artifact["metadata"].get("train_sessions", [])) | set(artifact["metadata"].get("validation_sessions", [])) | set(artifact["metadata"].get("test_sessions", []))
    if args.session in training_sessions:
        raise SystemExit(f"Session {args.session} is part of this artifact's train/validation history and cannot be a final hold-out test.")
    session_info, session_url = base.api_get("sessions", {"session_key": args.session}, args.cache_dir)
    rows, manifest = clean_base_rows(args.session, args.cache_dir)
    if not rows or not any(row["label"] for row in rows):
        raise SystemExit("The selected session has no usable clean positive labels.")
    probabilities = np.asarray([sigmoid(artifact["base_margin"] + sum(tree_margin(tree, row["features"]) for tree in artifact["trees"])) for row in rows])
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    threshold_info = artifact["metadata"].get("classification_threshold", {"value": 0.5, "selection_split": "not available", "selection_objective": "default"})
    threshold = float(threshold_info["value"])
    predictions = (probabilities >= threshold).astype(int)
    default_predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    default_tn, default_fp, default_fn, default_tp = confusion_matrix(labels, default_predictions, labels=[0, 1]).ravel()
    metrics = {
        "rows": int(len(labels)), "positive": int(labels.sum()), "positive_rate": round(float(labels.mean()), 6),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6), "average_precision": round(float(average_precision_score(labels, probabilities)), 6),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6), "classification_threshold": threshold_info,
        "accuracy_at_selected_threshold": round(float((labels == predictions).mean()), 6), "precision_at_selected_threshold": round(float(precision_score(labels, predictions, zero_division=0)), 6), "recall_at_selected_threshold": round(float(recall_score(labels, predictions, zero_division=0)), 6), "f1_at_selected_threshold": round(float(f1_score(labels, predictions, zero_division=0)), 6),
        "confusion_matrix_at_selected_threshold": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
        "confusion_matrix_at_0_5": {"true_negative": int(default_tn), "false_positive": int(default_fp), "false_negative": int(default_fn), "true_positive": int(default_tp)},
    }
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "purpose": "locked final hold-out evaluation; no fitting or threshold tuning performed", "model": {"path": str(args.model), "version": artifact["version"], "features": artifact["features"], "best_trees": len(artifact["trees"])}, "test_session": session_info[0] if session_info else {"session_key": args.session}, "session_url": session_url, "manifest": manifest, "metrics": metrics}
    (MODELS / "final_holdout_test.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# APEX-R final locked hold-out model test", "", f"Generated: {result['generated_at']}", "",
        "## Protocol", "", f"- Model: `{artifact['version']}` ({len(artifact['trees'])} trees).", f"- Test race: OpenF1 session {args.session}; this session was not used for training, validation, early stopping, feature selection, or threshold tuning.", "- The saved model was scored as-is. Its operating threshold was selected earlier on validation only; no refitting or threshold tuning occurred on this final test race.", "- Labels use the same clean-label procedure: overtake events are filtered around available pit and neutralisation/race-control records.", "",
        "## Results", "", "| ROC-AUC | Average precision | Brier score | Accuracy | Precision | Recall | F1 |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |", "| " + " | ".join(str(metrics[key]) for key in ("roc_auc", "average_precision", "brier_score", "accuracy_at_selected_threshold", "precision_at_selected_threshold", "recall_at_selected_threshold", "f1_at_selected_threshold")) + " |", "",
        "## Operating threshold", "", f"- Threshold: {threshold:.8f}; selected on the earlier validation race only to maximise F1 ({threshold_info.get('selection_split')}).", f"- Confusion matrix at selected threshold: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}.", f"- For transparency, at the arbitrary 0.50 threshold: TN={default_tn:,}, FP={default_fp:,}, FN={default_fn:,}, TP={default_tp:,}.", "",
        "## Test-set size", "", f"- Rows: {metrics['rows']:,}; positive overtake windows: {metrics['positive']:,} ({metrics['positive_rate']:.2%}).", "",
        "## Scope", "", "This is a final race-level generalisation check for overtake opportunity only. It does not validate private ERS/battery state, real team strategy, or APEX-R's simulated race outcomes.", "",
    ]
    (MODELS / "FINAL_HOLDOUT_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
