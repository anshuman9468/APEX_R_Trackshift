#!/usr/bin/env python3
"""Train and compare an XGBoost overtake-opportunity model for APEX-R.

The final race session is held out. The script preserves the Logistic
Regression baseline, writes an XGBoost artifact, and activates the model with
the higher held-out ROC-AUC in the browser bundle.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

import train_model as base


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DIST = ROOT / "dist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", default="7953,7779,7787,9070")
    parser.add_argument("--cache-dir", type=Path, default=base.CACHE)
    parser.add_argument("--output", type=Path, default=MODELS / "xgboost_overtake_model.json")
    parser.add_argument("--force-activate", action="store_true", help="Activate XGBoost even if the held-out ROC-AUC is lower.")
    return parser.parse_args()


def safe_metric(function, y_true, y_score):
    try:
        return round(float(function(y_true, y_score)), 6)
    except ValueError:
        return None


def tree_margin(node: dict, values: dict[str, float]) -> float:
    while "leaf" not in node:
        value = values.get(node["split"])
        child_id = node["missing"] if value is None or not math.isfinite(value) else (node["yes"] if value < node["split_condition"] else node["no"])
        node = next(child for child in node["children"] if child["nodeid"] == child_id)
    return float(node["leaf"])


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def browser_source(artifact: dict) -> str:
    payload = json.dumps(artifact, separators=(",", ":"))
    return f"""(function(root) {{
  'use strict';
  const artifact = {payload};
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const sigmoid = x => 1 / (1 + Math.exp(-clamp(x, -30, 30)));
  function leaf(tree, features) {{
    let node = tree;
    while (!Object.prototype.hasOwnProperty.call(node, 'leaf')) {{
      const value = Number(features[node.split]);
      const childId = !Number.isFinite(value) ? node.missing : (value < node.split_condition ? node.yes : node.no);
      node = node.children.find(child => child.nodeid === childId);
    }}
    return node.leaf;
  }}
  function predict(features) {{
    const margin = artifact.base_margin + artifact.trees.reduce((sum, tree) => sum + leaf(tree, features), 0);
    return {{ probability: sigmoid(margin), modelVersion: artifact.version }};
  }}
  const api = {{ metadata: artifact, predict }};
  root.ApexModel = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
}})(typeof globalThis !== 'undefined' ? globalThis : this);\n"""


def write_report(comparison: dict, selected: str) -> None:
    logistic = comparison["models"].get("logistic_regression")
    xgboost = comparison["models"]["xgboost"]
    lines = [
        "# APEX-R machine-learning model report",
        "",
        f"Generated: {comparison['generated_at']}",
        "",
        "## Decision",
        "",
        f"Active browser model: **{selected}**. Selection uses held-out ROC-AUC, unless explicitly overridden.",
        "",
        "## Dataset and labels",
        "",
        f"- Source: OpenF1 historical API; exact request URLs are stored in each model artifact.",
        f"- Sessions: {', '.join(map(str, comparison['sessions']))}; final session {comparison['test_session']} is held out.",
        f"- Training rows: {comparison['train_rows']:,}; test rows: {comparison['test_rows']:,}.",
        f"- Target: an observed overtake by the driver within the next {comparison['horizon_seconds']:.0f} seconds.",
        f"- Samples are downsampled to one interval record every {comparison['sample_interval_seconds']:.0f} seconds per driver.",
        f"- Features: {', '.join(comparison['features'])}.",
        "",
        "## Held-out comparison",
        "",
        "| Model | ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, artifact in [("Logistic Regression", logistic), ("XGBoost", xgboost)]:
        metrics = artifact.get("metadata", {}).get("metrics", {}) if artifact else {}
        lines.append("| " + name + " | " + " | ".join(str(metrics.get(key, "--")) for key in ["roc_auc", "average_precision", "brier_score", "accuracy_at_0_5"]) + " |")
    lines.extend([
        "",
        "## Algorithms and hyperparameters",
        "",
        "### Logistic Regression",
        "",
        "```json",
        json.dumps((logistic or {}).get("metadata", {}).get("hyperparameters", {}), indent=2),
        "```",
        "",
        "### XGBoost",
        "",
        "```json",
        json.dumps(xgboost["metadata"]["hyperparameters"], indent=2),
        "```",
        "",
        "## Interpretation and limitations",
        "",
        "- Accuracy is not the main selection metric because overtake events are uncommon; an all-negative classifier can appear accurate.",
        "- The active model estimates overtake opportunity. The APEX-R optimiser still chooses ATTACK, HOLD, HARVEST, or DEFEND and enforces energy constraints.",
        "- Public OpenF1 records do not provide private team ERS/battery telemetry. Energy and rival-response outcomes remain modelled.",
        "- Retrain and compare on more held-out race sessions before claiming broad real-world generalisation.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 scripts/train_model.py",
        "python3 scripts/train_xgboost.py",
        "```",
        "",
    ])
    (MODELS / "MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (MODELS / "README.md").write_text(
        "# APEX-R model artifacts\n\n"
        "- `overtake_model.json`: Logistic Regression baseline.\n"
        "- `xgboost_overtake_model.json`: XGBoost challenger.\n"
        "- `model_comparison.json`: Held-out comparison and active-model decision.\n"
        "- `MODEL_REPORT.md`: Human-readable model card, inputs, metrics, hyperparameters and limitations.\n"
        "- `training_report.json`: Logistic Regression training report.\n\n"
        "Retrain with `python3 scripts/train_model.py` and `python3 scripts/train_xgboost.py`.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    sessions = [int(value.strip()) for value in args.sessions.split(",") if value.strip()]
    if len(sessions) < 2:
        raise SystemExit("Provide at least two race sessions.")
    rows = []
    manifests = []
    for session in sessions:
        print(f"Reading OpenF1 session {session}...", flush=True)
        session_rows, manifest = base.session_rows(session, args.cache_dir)
        rows.extend(session_rows)
        manifests.append(manifest)
        print(f"  {manifest['rows']} samples / {manifest['positive']} positive labels", flush=True)

    train_sessions = set(sessions[:-1])
    test_session = sessions[-1]
    train_rows = [row for row in rows if row["session_key"] in train_sessions]
    test_rows = [row for row in rows if row["session_key"] == test_session]
    x_train = pd.DataFrame([[row["features"][name] for name in base.FEATURES] for row in train_rows], columns=base.FEATURES)
    y_train = np.array([row["overtake_next_60s"] for row in train_rows], dtype=int)
    x_test = pd.DataFrame([[row["features"][name] for name in base.FEATURES] for row in test_rows], columns=base.FEATURES)
    y_test = np.array([row["overtake_next_60s"] for row in test_rows], dtype=int)
    if not y_train.any() or not y_test.any():
        raise SystemExit("Both training and held-out sessions need positive labels.")

    parameters = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.04,
        "min_child_weight": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.05,
        "reg_lambda": 5.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 2026,
        "n_jobs": 4,
    }
    model = XGBClassifier(**parameters)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "test_rows": int(len(test_rows)),
        "test_positive": int(y_test.sum()),
        "test_positive_rate": round(float(y_test.mean()), 6),
        "accuracy_at_0_5": round(float((prediction == y_test).mean()), 6),
        "roc_auc": safe_metric(roc_auc_score, y_test, probability),
        "average_precision": safe_metric(average_precision_score, y_test, probability),
        "brier_score": round(float(brier_score_loss(y_test, probability)), 6),
    }
    booster = model.get_booster()
    trees = [json.loads(tree) for tree in booster.get_dump(dump_format="json")]
    config = json.loads(booster.save_config())
    raw_base_score = config["learner"]["learner_model_param"]["base_score"].strip("[]")
    base_probability = float(raw_base_score)
    base_margin = math.log(base_probability / (1 - base_probability))
    # Verify that the portable JavaScript tree evaluator produces the same
    # probabilities as XGBoost before publishing it to the browser.
    portable = []
    for values, expected in zip(x_test.head(50).to_dict("records"), probability[:50]):
        portable.append(sigmoid(base_margin + sum(tree_margin(tree, values) for tree in trees)))
        if abs(portable[-1] - float(expected)) > 1e-6:
            raise RuntimeError("Portable XGBoost export did not match the trained model.")

    importance = booster.get_score(importance_type="gain")
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "OpenF1 historical API",
        "data_source_note": "OpenF1 is an unofficial public data service; the data does not contain private team ERS/battery telemetry.",
        "target": "overtake_next_60s",
        "horizon_seconds": base.HORIZON_SECONDS,
        "sample_interval_seconds": base.SAMPLE_SECONDS,
        "train_sessions": sorted(train_sessions),
        "test_sessions": [test_session],
        "session_manifests": manifests,
        "train_rows": int(len(train_rows)),
        "train_positive": int(y_train.sum()),
        "train_positive_rate": round(float(y_train.mean()), 6),
        "algorithm": "XGBoost binary classifier",
        "hyperparameters": parameters,
        "feature_importance_gain": {name: round(float(importance.get(name, 0)), 6) for name in base.FEATURES},
        "metrics": metrics,
        "limitations": [
            "Predicts observed overtake events, not private team strategy or ERS state.",
            "The held-out session is evidence for this split, not proof of race-wide generalisation.",
            "The optimiser retains final action choice and energy constraints.",
        ],
    }
    artifact = {
        "version": "1.0.0-xgboost-openf1",
        "model_type": "xgboost_binary_logistic",
        "features": base.FEATURES,
        "base_margin": base_margin,
        "trees": trees,
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    logistic_path = MODELS / "overtake_model.json"
    logistic = json.loads(logistic_path.read_text(encoding="utf-8")) if logistic_path.exists() else None
    logistic_auc = ((logistic or {}).get("metadata", {}).get("metrics", {}).get("roc_auc"))
    activate = args.force_activate or logistic_auc is None or metrics["roc_auc"] > logistic_auc
    selected = "XGBoost" if activate else "Logistic Regression"
    if activate:
        (DIST / "apex-model.js").write_text(browser_source(artifact), encoding="utf-8")

    comparison = {
        "generated_at": metadata["trained_at"],
        "selection_metric": "held_out_roc_auc",
        "active_model": selected,
        "sessions": sessions,
        "test_session": test_session,
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "horizon_seconds": base.HORIZON_SECONDS,
        "sample_interval_seconds": base.SAMPLE_SECONDS,
        "features": base.FEATURES,
        "models": {"logistic_regression": logistic, "xgboost": artifact},
    }
    (MODELS / "model_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    write_report(comparison, selected)
    print(f"Saved XGBoost artifact: {args.output}")
    print(f"Held-out metrics: {json.dumps(metrics, sort_keys=True)}")
    print(f"Active browser model: {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
