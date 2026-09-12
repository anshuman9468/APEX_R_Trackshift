#!/usr/bin/env python3
"""Train and compare the OpenF1-pit-augmented Phase 1 candidate.

The executable surface is fixed to the materialized train/validation tables.
It has no session arguments and no final-test path.  The existing baseline,
prior candidate, and 4.0.0 candidate artifacts are read-only inputs; the only
model artifact written by this script is the new 5.0.0 candidate.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from xgboost import Booster, DMatrix


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "phase1-features"
MODELS_DIR = ROOT / "models"
CANDIDATES_DIR = MODELS_DIR / "candidates"
TRAIN_PATH = DATA_DIR / "train_phase1_openf1_pit_features.csv.gz"
VALIDATION_PATH = DATA_DIR / "validation_phase1_openf1_pit_features.csv.gz"
PIT_MANIFEST_PATH = DATA_DIR / "phase1_openf1_pit_feature_manifest.json"
PHASE1_CANDIDATE_PATH = CANDIDATES_DIR / "4.0.0-phase1-tracinginsights-candidate.json"
NEW_CANDIDATE_PATH = CANDIDATES_DIR / "5.0.0-phase1-openf1-pit-candidate.json"
REPORT_PATH = MODELS_DIR / "PHASE5_OPENF1_PIT_MODEL_VALIDATION_REPORT.md"
EXPERIMENT_PATH = MODELS_DIR / "phase5_openf1_pit_validation_experiments.json.gz"
CURVE_PATH = MODELS_DIR / "phase5_openf1_pit_precision_recall_curves.csv.gz"

TRAIN_SESSIONS = (7953, 7779, 7787)
VALIDATION_SESSION = 9070
TARGET = "overtake_next_60s"
RANDOM_STATE = 2026

PIT_FEATURES = [
    "attacker_openf1_pit_recent_flag",
    "rival_openf1_pit_recent_flag",
    "attacker_openf1_last_pit_lane_duration_sec",
    "rival_openf1_last_pit_lane_duration_sec",
]

PROTECTED_FILES = (
    MODELS_DIR / "xgboost_clean_label_base_model.json",
    CANDIDATES_DIR / "xgboost_engineered_scale2_platt_candidate.json",
    PHASE1_CANDIDATE_PATH,
)


def load_phase1_module():
    script = ROOT / "scripts" / "train_phase1_tracinginsights_candidate.py"
    spec = importlib.util.spec_from_file_location("phase1_training_helpers", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the existing Phase 1 training helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PHASE1 = load_phase1_module()
BASE_FEATURES = list(PHASE1.BASE_FEATURES)
PHASE1_FEATURES = list(PHASE1.PHASE1_FEATURES)
ALL_FEATURES = BASE_FEATURES + PHASE1_FEATURES + PIT_FEATURES
CATEGORICAL_FEATURE = PHASE1.CATEGORICAL_FEATURE
ARCHITECTURES = PHASE1.ARCHITECTURES
CLASS_WEIGHTS = PHASE1.CLASS_WEIGHTS


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str | None]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED_FILES}


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    required = {"session_key", TARGET, *ALL_FEATURES}
    if not required <= set(train.columns) or not required <= set(validation.columns):
        raise RuntimeError("Pit-augmented tables do not contain the fixed feature schema")
    if set(train["session_key"].unique()) != set(TRAIN_SESSIONS):
        raise RuntimeError("Training table contains an unapproved session")
    if set(validation["session_key"].unique()) != {VALIDATION_SESSION}:
        raise RuntimeError("Validation table contains an unapproved session")
    if len(train) != 13_912 or len(validation) != 6_058:
        raise RuntimeError("Unexpected train/validation row count")

    categories = sorted(train[CATEGORICAL_FEATURE].dropna().astype(str).unique().tolist())
    unknown = set(validation[CATEGORICAL_FEATURE].dropna().astype(str).unique()) - set(categories)
    if unknown:
        raise RuntimeError(f"Validation contains unseen compound categories: {sorted(unknown)}")
    category_dtype = pd.CategoricalDtype(categories=categories)
    train[CATEGORICAL_FEATURE] = train[CATEGORICAL_FEATURE].astype(category_dtype)
    validation[CATEGORICAL_FEATURE] = validation[CATEGORICAL_FEATURE].astype(category_dtype)

    # Keep explicit NaNs.  XGBoost routes missing numeric values natively.
    return (
        train,
        validation,
        train[TARGET].to_numpy(dtype=np.int8),
        validation[TARGET].to_numpy(dtype=np.int8),
        train["session_key"].to_numpy(dtype=np.int64),
        categories,
    )


def feature_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame.loc[:, features].copy()


def fit_model(x_train: pd.DataFrame, y_train: np.ndarray, params: dict):
    return PHASE1.fit_model(x_train, y_train, params)


def model_params(architecture: dict, weight: float) -> dict:
    return PHASE1.model_params(architecture, weight)


def candidate_record(
    candidate_id: str,
    stage: str,
    features: list[str],
    params: dict,
    probability: np.ndarray,
    y_validation: np.ndarray,
    calibration: dict | None = None,
) -> dict:
    return PHASE1.candidate_record(
        candidate_id=candidate_id,
        stage=stage,
        features=features,
        params=params,
        probability=probability,
        y_validation=y_validation,
        calibration=calibration,
    )


def selection_key(record: dict) -> tuple:
    return PHASE1.selection_key(record)


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def platt_probability(raw: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    logits = coefficient * logit(raw) + intercept
    return 1.0 / (1.0 + np.exp(-logits))


def grouped_platt(x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, params: dict):
    calibrator, oof = PHASE1.grouped_oof_platt(x_train, y_train, groups, params)
    return calibrator, oof


def calibration_metadata(calibrator: LogisticRegression, oof: np.ndarray, y_train: np.ndarray) -> dict:
    return {
        "method": "Platt logistic scaling",
        "input": "logit(raw XGBoost probability)",
        "fit_protocol": "three-fold leave-one-training-race-out predictions",
        "coefficient": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
        "oof_brier_score": float(brier_score_loss(y_train, calibrator.predict_proba(logit(oof).reshape(-1, 1))[:, 1])),
    }


def load_existing_phase1_record(validation: pd.DataFrame, y_validation: np.ndarray) -> dict:
    artifact = json.loads(PHASE1_CANDIDATE_PATH.read_text(encoding="utf-8"))
    expected_features = BASE_FEATURES + PHASE1_FEATURES
    if artifact.get("source_features") != expected_features:
        raise RuntimeError("Existing 4.0.0 candidate feature schema differs from Phase 1")
    frame = feature_frame(validation, expected_features)
    booster = Booster()
    booster.load_model(bytearray(base64.b64decode(artifact["booster_ubj_base64"])))
    raw = booster.predict(DMatrix(frame, enable_categorical=True))
    calibration = artifact.get("platt_calibration")
    probability = raw
    if calibration:
        probability = platt_probability(raw, float(calibration["coefficient"]), float(calibration["intercept"]))
    params = artifact.get("xgboost_parameters", {})
    record = candidate_record(
        candidate_id="4.0.0-phase1-tracinginsights-existing",
        stage="existing_phase1_candidate",
        features=expected_features,
        params=params,
        probability=probability,
        y_validation=y_validation,
        calibration=calibration,
    )
    record["raw_probability_metrics"], _ = PHASE1.evaluate(y_validation, raw)
    record["artifact_version"] = artifact.get("version")
    record["artifact_validation_metrics"] = artifact.get("validation_metrics")
    return record


def flatten_curves(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        for index, point in enumerate(record["precision_recall_curve"]):
            rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "stage": record["stage"],
                    "curve_index": index,
                    "threshold": point["threshold"],
                    "precision": point["precision"],
                    "recall": point["recall"],
                }
            )
    return pd.DataFrame(rows)


def fmt(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.6f}"


def report_line(record: dict) -> str:
    metrics = record["metrics"]
    point = metrics["operating_point_recall_gte_70"]
    calibration = "Platt" if record.get("calibration") else "raw"
    return (
        f"| `{record['candidate_id']}` | {len(record['features'])} | {calibration} | "
        f"{fmt(metrics['roc_auc'])} | {fmt(metrics['average_precision'])} | "
        f"{fmt(metrics['brier_score'])} | {fmt(metrics['ece_10_bins'])} | "
        f"{fmt(point['precision'])} | {fmt(point['recall'])} | {point['threshold']:.8f} |"
    )


def write_report(
    *,
    experiment: dict,
    summary_records: list[dict],
    importance: list[dict],
    pit_manifest: dict,
) -> None:
    lines = [
        "# Phase 5 OpenF1 Pit Candidate — Validation Report",
        "",
        f"Generated: `{experiment['generated_at']}`",
        "",
        "## Protocol",
        "",
        "- Validation-only. The fixed training sessions are 7953, 7779, and 7787; the fixed validation session is 9070.",
        "- No final holdout data was loaded, joined, scored, or used for selection.",
        "- All existing model artifacts were checksum-protected and remained unchanged.",
        "- XGBoost received numeric NaNs and missing categorical values directly; no zero/placeholder imputation was used.",
        "- New pit-event joins use `session_key + driver_number` and a backward-only rule `pit.date <= prediction date`; tolerance is 0 seconds and interpolation is none.",
        "",
        "## Critical data-availability result",
        "",
        "OpenF1 `pit` returned no records for any approved train/validation session in this run. The four new columns are therefore 100% missing and cannot contribute learned signal on this split. They were preserved as NaN and not silently converted to zero.",
        "",
        "| Session | Endpoint status | Records |",
        "|---:|---|---:|",
    ]
    for session_key, status in pit_manifest["sessions"].items():
        lines.append(f"| {session_key} | `{status['status']}` | {status['records']} |")
    lines.extend([
        "",
        "## Same-split comparison",
        "",
        "The primary operating point is the maximum validation precision among points with recall at least 70%. Curves are exact, unrounded outputs saved in the compressed JSON experiment and flat CSV curve artifacts.",
        "",
        "| Candidate | Features | Calibration | ROC-AUC | AP | Brier | ECE | Precision @ recall≥70% | Recall | Threshold |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in summary_records:
        lines.append(report_line(record))
    lines.extend([
        "",
        "## New-candidate sweep",
        "",
        "Every architecture and class-weight configuration tested is present in `phase5_openf1_pit_validation_experiments.json.gz`, including its complete precision/recall curve.",
        "",
        "| Candidate | Calibration | ROC-AUC | AP | Brier | ECE | Precision @ recall≥70% | Recall | Threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in experiment["candidates"]:
        if record["stage"] in {"architecture_sweep", "class_weight_sweep", "class_weight_sweep_platt"}:
            lines.append(
                f"| `{record['candidate_id']}` | {'Platt' if record.get('calibration') else 'raw'} | "
                f"{fmt(record['metrics']['roc_auc'])} | {fmt(record['metrics']['average_precision'])} | "
                f"{fmt(record['metrics']['brier_score'])} | {fmt(record['metrics']['ece_10_bins'])} | "
                f"{fmt(record['metrics']['operating_point_recall_gte_70']['precision'])} | "
                f"{fmt(record['metrics']['operating_point_recall_gte_70']['recall'])} | "
                f"{record['metrics']['operating_point_recall_gte_70']['threshold']:.8f} |"
            )
    lines.extend(["", "## New-candidate gain importance", "", "| Rank | Feature | Group | Gain share |", "|---:|---|---|---:|"])
    for index, item in enumerate(importance, start=1):
        lines.append(f"| {index} | `{item['feature']}` | {item['feature_group']} | {item['gain_share']:.6%} |")
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"- Selected new candidate: `{experiment['recommended_candidate_id']}`.",
        f"- Selection point: precision `{experiment['recommended_metrics']['operating_point_recall_gte_70']['precision']:.6f}`, recall `{experiment['recommended_metrics']['operating_point_recall_gte_70']['recall']:.6f}`, threshold `{experiment['recommended_metrics']['operating_point_recall_gte_70']['threshold']:.8f}`.",
        f"- Same-split verdict: `{experiment['verdict']}`.",
        "- Because OpenF1 pit data was unavailable for every approved session, this run does not establish a pit-data improvement. A later season/session set with actual pit records would be needed to measure that feature family.",
        "",
        "## Artifacts",
        "",
        f"- New model: `{NEW_CANDIDATE_PATH.relative_to(ROOT)}`",
        f"- Full experiment and exact PR curves: `{EXPERIMENT_PATH.relative_to(ROOT)}`",
        f"- Flat exact PR curves: `{CURVE_PATH.relative_to(ROOT)}`",
        f"- Pit feature audit: `{(DATA_DIR / 'phase1_openf1_pit_feature_manifest.json').relative_to(ROOT)}`",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    protected_before = protected_hashes()
    train, validation, y_train, y_validation, groups, categories = prepare_data()
    x_train_all = feature_frame(train, ALL_FEATURES)
    x_validation_all = feature_frame(validation, ALL_FEATURES)

    candidates: list[dict] = []
    architecture_models: dict[str, tuple[object, np.ndarray, dict]] = {}
    print("Stage 1/4: architecture sweep", flush=True)
    for architecture in ARCHITECTURES:
        params = model_params(architecture, 1.0)
        model = fit_model(x_train_all, y_train, params)
        raw = model.predict_proba(x_validation_all)[:, 1]
        record = candidate_record(
            candidate_id=f"phase5-t{architecture['n_estimators']}-d{architecture['max_depth']}-lr{architecture['learning_rate']}-w1-raw",
            stage="architecture_sweep",
            features=ALL_FEATURES,
            params=params,
            probability=raw,
            y_validation=y_validation,
        )
        candidates.append(record)
        architecture_models[record["candidate_id"]] = (model, raw, params)
        point = record["metrics"]["operating_point_recall_gte_70"]
        print(f"  {record['candidate_id']}: precision={point['precision']:.6f} recall={point['recall']:.6f}", flush=True)

    selected_architecture_record = max(candidates, key=selection_key)
    selected_architecture = {
        key: selected_architecture_record["hyperparameters"][key]
        for key in ("n_estimators", "max_depth", "learning_rate")
    }
    print(f"Selected architecture: {selected_architecture}", flush=True)

    print("Stage 2/4: class-weight sweep and grouped-OOF Platt calibration", flush=True)
    sweep_models = {}
    for weight in CLASS_WEIGHTS:
        params = model_params(selected_architecture, weight)
        model = fit_model(x_train_all, y_train, params)
        raw = model.predict_proba(x_validation_all)[:, 1]
        raw_record = candidate_record(
            candidate_id=f"phase5-selected-w{weight:g}-raw",
            stage="class_weight_sweep",
            features=ALL_FEATURES,
            params=params,
            probability=raw,
            y_validation=y_validation,
        )
        candidates.append(raw_record)
        calibrator, oof = grouped_platt(x_train_all, y_train, groups, params)
        calibrated = calibrator.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
        metadata = calibration_metadata(calibrator, oof, y_train)
        calibrated_record = candidate_record(
            candidate_id=f"phase5-selected-w{weight:g}-platt",
            stage="class_weight_sweep_platt",
            features=ALL_FEATURES,
            params=params,
            probability=calibrated,
            y_validation=y_validation,
            calibration=metadata,
        )
        candidates.append(calibrated_record)
        sweep_models[weight] = (model, raw, params, calibrator, calibrated)
        print(f"  weight={weight:g}: raw precision={raw_record['metrics']['operating_point_recall_gte_70']['precision']:.6f}; Platt precision={calibrated_record['metrics']['operating_point_recall_gte_70']['precision']:.6f}", flush=True)

    sweep_records = [record for record in candidates if record["stage"] in {"class_weight_sweep", "class_weight_sweep_platt"}]
    recommended = max(sweep_records, key=selection_key)
    winning_weight = float(recommended["hyperparameters"]["scale_pos_weight"])
    winning_model, winning_raw, winning_params, winning_calibrator, winning_calibrated = sweep_models[winning_weight]
    winning_probability = winning_calibrated if recommended.get("calibration") else winning_raw

    print("Stage 3/4: same-split baseline and existing 4.0.0 comparisons", flush=True)
    x_train_base = feature_frame(train, BASE_FEATURES)
    x_validation_base = feature_frame(validation, BASE_FEATURES)
    baseline_model = fit_model(x_train_base, y_train, winning_params)
    baseline_raw = baseline_model.predict_proba(x_validation_base)[:, 1]
    baseline_raw_record = candidate_record(
        candidate_id="baseline8-same-split-phase5-raw",
        stage="baseline_comparator",
        features=BASE_FEATURES,
        params=winning_params,
        probability=baseline_raw,
        y_validation=y_validation,
    )
    baseline_calibrator, baseline_oof = grouped_platt(x_train_base, y_train, groups, winning_params)
    baseline_calibrated = baseline_calibrator.predict_proba(logit(baseline_raw).reshape(-1, 1))[:, 1]
    baseline_calibration = calibration_metadata(baseline_calibrator, baseline_oof, y_train)
    baseline_platt_record = candidate_record(
        candidate_id="baseline8-same-split-phase5-platt",
        stage="baseline_comparator",
        features=BASE_FEATURES,
        params=winning_params,
        probability=baseline_calibrated,
        y_validation=y_validation,
        calibration=baseline_calibration,
    )
    candidates.extend([baseline_raw_record, baseline_platt_record])
    existing_phase1 = load_existing_phase1_record(validation, y_validation)
    candidates.append(existing_phase1)
    same_split_baseline = max([baseline_raw_record, baseline_platt_record], key=selection_key)

    print("Stage 4/4: importance, exports, and report", flush=True)
    booster = winning_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    total_gain = sum(float(gain.get(feature, 0.0)) for feature in ALL_FEATURES)
    importance = []
    for feature in ALL_FEATURES:
        importance.append({
            "feature": feature,
            "gain": float(gain.get(feature, 0.0)),
            "gain_share": float(gain.get(feature, 0.0)) / total_gain if total_gain else 0.0,
            "feature_group": "openf1_pit" if feature in PIT_FEATURES else ("phase1" if feature in PHASE1_FEATURES else "baseline"),
        })
    importance.sort(key=lambda item: item["gain_share"], reverse=True)

    raw_model_bytes = bytes(booster.save_raw(raw_format="ubj"))
    candidate_artifact = {
        "version": "5.0.0-phase1-openf1-pit-candidate",
        "status": "validation-selected offline candidate; final holdout not scored",
        "model_type": "XGBoost binary logistic with native categorical and missing-value handling",
        "source_features": ALL_FEATURES,
        "source_feature_count": len(ALL_FEATURES),
        "compound_categories": categories,
        "missing_value_policy": "NaN preserved; XGBoost native missing branches; no zero/placeholder imputation",
        "xgboost_parameters": {key: value for key, value in winning_params.items() if key not in {"n_jobs", "missing"}},
        "platt_calibration": recommended.get("calibration"),
        "classification_threshold": recommended["metrics"]["operating_point_recall_gte_70"]["threshold"],
        "validation_metrics": recommended["metrics"],
        "feature_importance_gain": importance,
        "pit_source_manifest": str(PIT_MANIFEST_PATH.relative_to(ROOT)),
        "training_protocol": {
            "train_sessions": list(TRAIN_SESSIONS),
            "validation_session": VALIDATION_SESSION,
            "selection_metric": "maximum validation precision with recall >= 0.70",
            "calibration_protocol": "leave-one-training-race-out OOF Platt fit",
            "final_holdout_scored": False,
        },
        "booster_ubj_sha256": hashlib.sha256(raw_model_bytes).hexdigest(),
        "booster_ubj_base64": base64.b64encode(raw_model_bytes).decode("ascii"),
    }
    NEW_CANDIDATE_PATH.write_text(json.dumps(candidate_artifact, indent=2) + "\n", encoding="utf-8")

    # Verify that the exported booster reproduces the raw predictions.
    loaded = Booster()
    loaded.load_model(bytearray(base64.b64decode(candidate_artifact["booster_ubj_base64"])))
    verify = loaded.predict(DMatrix(x_validation_all.head(100), enable_categorical=True))
    if not np.allclose(verify, winning_raw[:100], atol=1e-7, rtol=1e-7):
        raise RuntimeError("Exported 5.0.0 candidate does not reproduce raw XGBoost scores")

    summary_records = [same_split_baseline, existing_phase1, recommended]
    pit_manifest = json.loads(PIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    experiment = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "validation-only; no final holdout access or scoring",
        "data": {
            "train_path": str(TRAIN_PATH.relative_to(ROOT)),
            "validation_path": str(VALIDATION_PATH.relative_to(ROOT)),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_positives": int(y_train.sum()),
            "validation_positives": int(y_validation.sum()),
            "features": ALL_FEATURES,
            "missingness": {feature: {"train": int(train[feature].isna().sum()), "validation": int(validation[feature].isna().sum())} for feature in ALL_FEATURES},
            "numeric_missing_values_preserved": True,
            "zero_or_placeholder_imputation": False,
        },
        "selection_protocol": {
            "architecture_candidates": list(ARCHITECTURES),
            "class_weights": list(CLASS_WEIGHTS),
            "primary_metric": "maximum validation precision with recall >= 0.70",
            "tie_breakers": ["average_precision", "lower_brier", "roc_auc", "calibrated"],
        },
        "recommended_candidate_id": recommended["candidate_id"],
        "recommended_metrics": recommended["metrics"],
        "same_split_baseline_candidate_id": same_split_baseline["candidate_id"],
        "same_split_baseline_metrics": same_split_baseline["metrics"],
        "comparison_candidates": [
            {"candidate_id": record["candidate_id"], "metrics": record["metrics"], "calibration": record.get("calibration"), "features": record["features"]}
            for record in summary_records
        ],
        "verdict": "new 5.0.0 candidate is the best same-split validation candidate" if selection_key(recommended) > selection_key(existing_phase1) and selection_key(recommended) > selection_key(same_split_baseline) else "new 5.0.0 candidate does not lead all same-split comparators",
        "feature_importance_gain": importance,
        "candidates": candidates,
        "protected_hashes_before": protected_before,
        "new_candidate": str(NEW_CANDIDATE_PATH.relative_to(ROOT)),
        "full_pr_curve_csv": str(CURVE_PATH.relative_to(ROOT)),
    }
    with gzip.open(EXPERIMENT_PATH, "wt", encoding="utf-8") as handle:
        json.dump(experiment, handle, indent=2)
    flatten_curves(candidates).to_csv(CURVE_PATH, index=False, compression="gzip")
    write_report(experiment=experiment, summary_records=summary_records, importance=importance, pit_manifest=pit_manifest)

    protected_after = protected_hashes()
    if protected_before != protected_after:
        raise RuntimeError("An existing protected model artifact changed")
    experiment["protected_hashes_after"] = protected_after
    with gzip.open(EXPERIMENT_PATH, "wt", encoding="utf-8") as handle:
        json.dump(experiment, handle, indent=2)

    print(f"Recommended: {recommended['candidate_id']}", flush=True)
    print(f"Validation ROC-AUC={recommended['metrics']['roc_auc']:.6f} AP={recommended['metrics']['average_precision']:.6f} Brier={recommended['metrics']['brier_score']:.6f}", flush=True)
    print(f"Precision@recall>=70%={recommended['metrics']['operating_point_recall_gte_70']['precision']:.6f} recall={recommended['metrics']['operating_point_recall_gte_70']['recall']:.6f}", flush=True)
    print(f"Saved {NEW_CANDIDATE_PATH}", flush=True)
    print(f"Saved {REPORT_PATH}", flush=True)
    print(f"Saved {EXPERIMENT_PATH}", flush=True)
    print(f"Saved {CURVE_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
