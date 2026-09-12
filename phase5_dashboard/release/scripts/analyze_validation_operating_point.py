#!/usr/bin/env python3
"""Analyse thresholds and raw-score calibration on validation only.

This script does not fit a model and deliberately never loads the frozen
session 11353.  It reads the selected artifact and evaluates only session 9070,
which was the original validation/early-stopping session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, precision_score, recall_score, roc_auc_score

from test_final_holdout import clean_base_rows
from train_xgboost import sigmoid, tree_margin


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "xgboost_clean_label_base_model.json"
VALIDATION_SESSION = 9070


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    positives = int(predicted.sum())
    return {
        "threshold": round(float(threshold), 8), "predicted_positive": positives,
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 6),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 6),
        "f1": round(float(2 * precision_score(labels, predicted, zero_division=0) * recall_score(labels, predicted, zero_division=0) / max(precision_score(labels, predicted, zero_division=0) + recall_score(labels, predicted, zero_division=0), 1e-12)), 6),
    }


def calibration_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> tuple[list[dict], float]:
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, bins + 1)))
    rows = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        mask = (probabilities >= low) & ((probabilities <= high) if index == len(edges) - 2 else (probabilities < high))
        if not mask.any():
            continue
        rows.append({"score_from": round(float(low), 6), "score_to": round(float(high), 6), "count": int(mask.sum()), "mean_predicted_probability": round(float(probabilities[mask].mean()), 6), "observed_positive_rate": round(float(labels[mask].mean()), 6)})
    ece = sum(row["count"] / len(labels) * abs(row["mean_predicted_probability"] - row["observed_positive_rate"]) for row in rows)
    return rows, round(float(ece), 6)


def main() -> int:
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    rows, manifest = clean_base_rows(VALIDATION_SESSION, ROOT / "data" / "openf1-cache")
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    probabilities = np.asarray([sigmoid(artifact["base_margin"] + sum(tree_margin(tree, row["features"]) for tree in artifact["trees"])) for row in rows])
    selected = float(artifact["metadata"]["classification_threshold"]["value"])
    candidates = sorted(set([0.03, 0.05, 0.07, selected, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]))
    curve = [threshold_metrics(labels, probabilities, threshold) for threshold in candidates]
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    eligible = [(float(thresholds[i]), float(precision[i]), float(recall[i])) for i in range(len(thresholds)) if recall[i] >= 0.70]
    best_precision_at_70_recall = max(eligible, key=lambda item: (item[1], item[0])) if eligible else None
    bins, ece = calibration_bins(labels, probabilities)
    result = {
        "scope": "validation session 9070 only; frozen final hold-out 11353 was not loaded", "model_version": artifact["version"],
        "validation_rows": int(len(labels)), "positive": int(labels.sum()), "positive_rate": round(float(labels.mean()), 6),
        "ranking_metrics": {"roc_auc": round(float(roc_auc_score(labels, probabilities)), 6), "average_precision": round(float(average_precision_score(labels, probabilities)), 6), "brier_score": round(float(brier_score_loss(labels, probabilities)), 6)},
        "selected_threshold": artifact["metadata"]["classification_threshold"], "threshold_curve": curve,
        "best_precision_with_recall_at_least_70_percent": None if best_precision_at_70_recall is None else {"threshold": round(best_precision_at_70_recall[0], 8), "precision": round(best_precision_at_70_recall[1], 6), "recall": round(best_precision_at_70_recall[2], 6)},
        "calibration": {"method": "raw XGBoost probabilities; decile-style quantile reliability bins", "expected_calibration_error": ece, "bins": bins},
        "class_imbalance": {"scale_pos_weight": artifact["metadata"]["hyperparameters"].get("scale_pos_weight", 1), "resampling": "not used", "focal_loss": "not used"}, "manifest": manifest,
    }
    (MODELS / "validation_operating_point_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
