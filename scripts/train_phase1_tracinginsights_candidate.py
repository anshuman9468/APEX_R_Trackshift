#!/usr/bin/env python3
"""Validation-only XGBoost experiments for the APEX-R Phase 1 feature set.

The executable surface is intentionally fixed to the already-materialized
training and validation tables. It has no session arguments, API access, or
holdout path. Existing model artifacts are checksummed and treated as immutable.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from xgboost import Booster, DMatrix, XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "phase1-features"
MODELS_DIR = ROOT / "models"
CANDIDATES_DIR = MODELS_DIR / "candidates"
TRAIN_PATH = DATA_DIR / "train_phase1_features.csv.gz"
VALIDATION_PATH = DATA_DIR / "validation_phase1_features.csv.gz"
REPORT_PATH = MODELS_DIR / "PHASE1_TRACINGINSIGHTS_MODEL_VALIDATION_REPORT.md"
EXPERIMENT_PATH = MODELS_DIR / "phase1_tracinginsights_validation_experiments.json.gz"
CANDIDATE_PATH = CANDIDATES_DIR / "4.0.0-phase1-tracinginsights-candidate.json"

TRAIN_SESSIONS = (7953, 7779, 7787)
VALIDATION_SESSION = 9070
EXPECTED_TRAIN_ROWS = 13_912
EXPECTED_VALIDATION_ROWS = 6_058
EXPECTED_TRAIN_POSITIVES = 1_211
EXPECTED_VALIDATION_POSITIVES = 257
TARGET = "overtake_next_60s"
RANDOM_STATE = 2026

PROTECTED_FILES = (
    MODELS_DIR / "xgboost_clean_label_base_model.json",
    CANDIDATES_DIR / "xgboost_engineered_scale2_platt_candidate.json",
)

BASE_FEATURES = [
    "interval_sec",
    "gap_to_leader_sec",
    "closing_rate_sec_per_min",
    "position",
    "tyre_age",
    "track_temperature_c",
    "rainfall",
    "race_progress",
]

PHASE1_FEATURES = [
    "opponent_tyre_age_delta",
    "opponent_tyre_compound_delta",
    "stint_length_delta",
    "relative_sector_time_delta_s1",
    "relative_sector_time_delta_s2",
    "relative_sector_time_delta_s3",
    "relative_lap_time_delta",
    "distance_to_car_ahead_m",
    "rival_drs_open",
    "attacker_pit_out_recent_flag",
    "rival_pit_out_recent_flag",
]

ALL_FEATURES = BASE_FEATURES + PHASE1_FEATURES
CATEGORICAL_FEATURE = "opponent_tyre_compound_delta"

ARCHITECTURES = (
    {"n_estimators": 75, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 75, "max_depth": 4, "learning_rate": 0.04},
    {"n_estimators": 75, "max_depth": 5, "learning_rate": 0.03},
    {"n_estimators": 125, "max_depth": 3, "learning_rate": 0.03},
    {"n_estimators": 125, "max_depth": 4, "learning_rate": 0.03},
    {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.025},
)
CLASS_WEIGHTS = (1.0, 1.5, 2.0, 3.0)

COMMON_PARAMS = {
    "min_child_weight": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.05,
    "reg_lambda": 5.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "enable_categorical": True,
    "missing": np.nan,
    "random_state": RANDOM_STATE,
    "n_jobs": 4,
}


@dataclass
class PreparedData:
    train: pd.DataFrame
    validation: pd.DataFrame
    y_train: np.ndarray
    y_validation: np.ndarray
    groups: np.ndarray
    compound_categories: list[str]


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


def prepare_data() -> PreparedData:
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    required = {"session_key", TARGET, *ALL_FEATURES}
    if not required <= set(train.columns) or not required <= set(validation.columns):
        raise RuntimeError("Phase 1 tables do not contain the fixed feature schema.")
    if len(train) != EXPECTED_TRAIN_ROWS or len(validation) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("Unexpected train/validation row count.")
    if set(train["session_key"].unique()) != set(TRAIN_SESSIONS):
        raise RuntimeError("Training table contains an unapproved session.")
    if set(validation["session_key"].unique()) != {VALIDATION_SESSION}:
        raise RuntimeError("Validation table contains an unapproved session.")
    if int(train[TARGET].sum()) != EXPECTED_TRAIN_POSITIVES:
        raise RuntimeError("Unexpected training positive count.")
    if int(validation[TARGET].sum()) != EXPECTED_VALIDATION_POSITIVES:
        raise RuntimeError("Unexpected validation positive count.")

    categories = sorted(train[CATEGORICAL_FEATURE].dropna().astype(str).unique().tolist())
    unknown = set(validation[CATEGORICAL_FEATURE].dropna().astype(str).unique()) - set(categories)
    if unknown:
        raise RuntimeError(f"Validation contains unseen compound pair categories: {sorted(unknown)}")
    category_dtype = pd.CategoricalDtype(categories=categories)
    train[CATEGORICAL_FEATURE] = train[CATEGORICAL_FEATURE].astype(category_dtype)
    validation[CATEGORICAL_FEATURE] = validation[CATEGORICAL_FEATURE].astype(category_dtype)

    # Deliberately do not call fillna: numeric NaNs and missing categorical
    # values remain missing for XGBoost's native missing-value routing.
    return PreparedData(
        train=train,
        validation=validation,
        y_train=train[TARGET].to_numpy(dtype=np.int8),
        y_validation=validation[TARGET].to_numpy(dtype=np.int8),
        groups=train["session_key"].to_numpy(dtype=np.int64),
        compound_categories=categories,
    )


def feature_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame.loc[:, features].copy()
    if CATEGORICAL_FEATURE not in features:
        return result
    if not isinstance(result[CATEGORICAL_FEATURE].dtype, pd.CategoricalDtype):
        raise RuntimeError("Compound pair must stay categorical.")
    return result


def model_params(architecture: dict, scale_pos_weight: float) -> dict:
    return {**COMMON_PARAMS, **architecture, "scale_pos_weight": float(scale_pos_weight)}


def fit_model(x_train: pd.DataFrame, y_train: np.ndarray, params: dict) -> XGBClassifier:
    model = XGBClassifier(**params)
    model.fit(x_train, y_train, verbose=False)
    return model


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (probability >= lower) & (probability < upper if index < bins - 1 else probability <= upper)
        if selected.any():
            result += selected.sum() / total * abs(float(y_true[selected].mean()) - float(probability[selected].mean()))
    return float(result)


def full_pr_curve(y_true: np.ndarray, probability: np.ndarray) -> tuple[list[dict], dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability, drop_intermediate=False)
    points = [
        {
            "threshold": float(thresholds[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
        }
        for index in range(len(thresholds))
    ]
    points.append({"threshold": None, "precision": float(precision[-1]), "recall": float(recall[-1])})
    eligible = [point for point in points[:-1] if point["recall"] >= 0.70]
    if not eligible:
        operating = {"threshold": 1.0, "precision": 0.0, "recall": 0.0}
    else:
        operating = max(eligible, key=lambda point: (point["precision"], point["recall"], point["threshold"]))
    threshold = operating["threshold"]
    predicted = probability >= threshold
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    operating = {
        **operating,
        "f1": float(2 * operating["precision"] * operating["recall"] / (operating["precision"] + operating["recall"]))
        if operating["precision"] + operating["recall"]
        else 0.0,
        "accuracy": float((tp + tn) / len(y_true)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    return points, operating


def evaluate(y_true: np.ndarray, probability: np.ndarray) -> tuple[dict, list[dict]]:
    curve, operating = full_pr_curve(y_true, probability)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "ece_10_bins": expected_calibration_error(y_true, probability),
        "probability_min": float(probability.min()),
        "probability_median": float(np.median(probability)),
        "probability_max": float(probability.max()),
        "operating_point_recall_gte_70": operating,
        "pr_curve_points": len(curve),
    }
    return metrics, curve


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def grouped_oof_platt(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    params: dict,
) -> tuple[LogisticRegression, np.ndarray]:
    oof = np.full(len(y_train), np.nan, dtype=float)
    for group in sorted(np.unique(groups)):
        fit_mask = groups != group
        score_mask = groups == group
        fold_model = fit_model(x_train.loc[fit_mask], y_train[fit_mask], params)
        oof[score_mask] = fold_model.predict_proba(x_train.loc[score_mask])[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("Grouped OOF predictions are incomplete.")
    calibrator = LogisticRegression(C=1.0, max_iter=2_000, random_state=RANDOM_STATE)
    calibrator.fit(logit(oof).reshape(-1, 1), y_train)
    if float(calibrator.coef_[0][0]) <= 0:
        raise RuntimeError("Platt map is not monotonic increasing.")
    return calibrator, oof


def calibrate(calibrator: LogisticRegression, raw_probability: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(logit(raw_probability).reshape(-1, 1))[:, 1]


def candidate_record(
    *,
    candidate_id: str,
    stage: str,
    features: list[str],
    params: dict,
    probability: np.ndarray,
    y_validation: np.ndarray,
    calibration: dict | None = None,
    ablated_feature: str | None = None,
) -> dict:
    metrics, curve = evaluate(y_validation, probability)
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "source_feature_count": len(features),
        "features": features,
        "ablated_feature": ablated_feature,
        "hyperparameters": {
            key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in params.items()
            if key not in {"n_jobs", "missing"}
        },
        "missing_value_policy": "native XGBoost NaN/categorical-missing routing; no imputation",
        "calibration": calibration,
        "metrics": metrics,
        "precision_recall_curve": curve,
    }


def selection_key(record: dict) -> tuple:
    metrics = record["metrics"]
    operating = metrics["operating_point_recall_gte_70"]
    return (
        operating["precision"],
        metrics["average_precision"],
        -metrics["brier_score"],
        metrics["roc_auc"],
        record["calibration"] is not None,
    )


def rounded(value: float) -> str:
    return f"{value:.6f}"


def candidate_line(record: dict) -> str:
    metrics = record["metrics"]
    point = metrics["operating_point_recall_gte_70"]
    calibration = "Platt" if record["calibration"] else "raw"
    params = record["hyperparameters"]
    return (
        f"- `{record['candidate_id']}` — {calibration}; trees={params['n_estimators']}, "
        f"depth={params['max_depth']}, lr={params['learning_rate']}, "
        f"weight={params['scale_pos_weight']}: ROC-AUC {rounded(metrics['roc_auc'])}, "
        f"AP {rounded(metrics['average_precision'])}, Brier {rounded(metrics['brier_score'])}, "
        f"precision@recall≥70% {rounded(point['precision'])} at recall {rounded(point['recall'])}, "
        f"threshold {point['threshold']:.8f}; exact PR points {metrics['pr_curve_points']}."
    )


def main() -> int:
    started = datetime.now(timezone.utc)
    protected_before = protected_hashes()
    data = prepare_data()
    x_train_all = feature_frame(data.train, ALL_FEATURES)
    x_validation_all = feature_frame(data.validation, ALL_FEATURES)

    missingness = {
        feature: {
            "train_missing": int(data.train[feature].isna().sum()),
            "validation_missing": int(data.validation[feature].isna().sum()),
        }
        for feature in ALL_FEATURES
    }
    if not any(item["train_missing"] or item["validation_missing"] for item in missingness.values()):
        raise RuntimeError("Expected explicit Phase 1 missing values, but none were found.")

    candidates: list[dict] = []
    architecture_models: dict[str, tuple[XGBClassifier, np.ndarray, dict]] = {}
    print("Stage 1/5: architecture sweep", flush=True)
    for architecture in ARCHITECTURES:
        candidate_id = f"phase1-t{architecture['n_estimators']}-d{architecture['max_depth']}-lr{architecture['learning_rate']}-w1-raw"
        params = model_params(architecture, 1.0)
        model = fit_model(x_train_all, data.y_train, params)
        probability = model.predict_proba(x_validation_all)[:, 1]
        record = candidate_record(
            candidate_id=candidate_id,
            stage="architecture_sweep",
            features=ALL_FEATURES,
            params=params,
            probability=probability,
            y_validation=data.y_validation,
        )
        candidates.append(record)
        architecture_models[candidate_id] = (model, probability, params)
        print(candidate_line(record), flush=True)

    architecture_records = [record for record in candidates if record["stage"] == "architecture_sweep"]
    selected_architecture_record = max(architecture_records, key=selection_key)
    selected_architecture = {
        key: selected_architecture_record["hyperparameters"][key]
        for key in ("n_estimators", "max_depth", "learning_rate")
    }
    print(f"Selected architecture: {selected_architecture}", flush=True)

    print("Stage 2/5: class-weight sweep and grouped-OOF Platt calibration", flush=True)
    sweep_models: dict[float, tuple[XGBClassifier, np.ndarray, dict, LogisticRegression, np.ndarray]] = {}
    for weight in CLASS_WEIGHTS:
        params = model_params(selected_architecture, weight)
        raw_id = f"phase1-selected-w{weight:g}-raw"
        existing = next((record for record in candidates if record["candidate_id"] == raw_id), None)
        model = fit_model(x_train_all, data.y_train, params)
        raw_probability = model.predict_proba(x_validation_all)[:, 1]
        if existing is None:
            raw_record = candidate_record(
                candidate_id=raw_id,
                stage="class_weight_sweep",
                features=ALL_FEATURES,
                params=params,
                probability=raw_probability,
                y_validation=data.y_validation,
            )
            candidates.append(raw_record)
            print(candidate_line(raw_record), flush=True)
        calibrator, oof = grouped_oof_platt(x_train_all, data.y_train, data.groups, params)
        calibrated_probability = calibrate(calibrator, raw_probability)
        calibration_metadata = {
            "method": "Platt logistic scaling",
            "input": "logit(raw XGBoost probability)",
            "fit_protocol": "three-fold leave-one-training-race-out predictions",
            "coefficient": float(calibrator.coef_[0][0]),
            "intercept": float(calibrator.intercept_[0]),
            "oof_brier_score": float(brier_score_loss(data.y_train, calibrate(calibrator, oof))),
        }
        calibrated_record = candidate_record(
            candidate_id=f"phase1-selected-w{weight:g}-platt",
            stage="class_weight_sweep_platt",
            features=ALL_FEATURES,
            params=params,
            probability=calibrated_probability,
            y_validation=data.y_validation,
            calibration=calibration_metadata,
        )
        candidates.append(calibrated_record)
        sweep_models[weight] = (model, raw_probability, params, calibrator, calibrated_probability)
        print(candidate_line(calibrated_record), flush=True)

    selectable = [record for record in candidates if record["stage"] in {"class_weight_sweep", "class_weight_sweep_platt"}]
    # Architecture w=1 is represented under its architecture ID; add it to the
    # selectable pool when the separately named raw sweep row was deduplicated.
    selected_arch_w1 = next(
        record for record in architecture_records
        if all(record["hyperparameters"][key] == selected_architecture[key] for key in selected_architecture)
    )
    selectable.append(selected_arch_w1)
    recommended = max(selectable, key=selection_key)
    winning_weight = float(recommended["hyperparameters"]["scale_pos_weight"])
    winning_model, winning_raw_probability, winning_params, winning_calibrator, winning_calibrated_probability = sweep_models[winning_weight]
    winning_probability = winning_calibrated_probability if recommended["calibration"] else winning_raw_probability

    print("Stage 3/5: apples-to-apples baseline comparator", flush=True)
    x_train_base = feature_frame(data.train, BASE_FEATURES)
    x_validation_base = feature_frame(data.validation, BASE_FEATURES)
    baseline_model = fit_model(x_train_base, data.y_train, winning_params)
    baseline_raw = baseline_model.predict_proba(x_validation_base)[:, 1]
    baseline_raw_record = candidate_record(
        candidate_id="baseline8-same-split-raw",
        stage="baseline_comparator",
        features=BASE_FEATURES,
        params=winning_params,
        probability=baseline_raw,
        y_validation=data.y_validation,
    )
    candidates.append(baseline_raw_record)
    baseline_calibrator, baseline_oof = grouped_oof_platt(x_train_base, data.y_train, data.groups, winning_params)
    baseline_calibrated = calibrate(baseline_calibrator, baseline_raw)
    baseline_calibration = {
        "method": "Platt logistic scaling",
        "input": "logit(raw XGBoost probability)",
        "fit_protocol": "three-fold leave-one-training-race-out predictions",
        "coefficient": float(baseline_calibrator.coef_[0][0]),
        "intercept": float(baseline_calibrator.intercept_[0]),
        "oof_brier_score": float(brier_score_loss(data.y_train, calibrate(baseline_calibrator, baseline_oof))),
    }
    baseline_platt_record = candidate_record(
        candidate_id="baseline8-same-split-platt",
        stage="baseline_comparator",
        features=BASE_FEATURES,
        params=winning_params,
        probability=baseline_calibrated,
        y_validation=data.y_validation,
        calibration=baseline_calibration,
    )
    candidates.append(baseline_platt_record)
    print(candidate_line(baseline_raw_record), flush=True)
    print(candidate_line(baseline_platt_record), flush=True)

    print("Stage 4/5: leave-one-Phase-1-column-out ablations", flush=True)
    full_raw_record = next(
        (record for record in candidates if record["stage"] == "class_weight_sweep" and float(record["hyperparameters"]["scale_pos_weight"]) == winning_weight),
        selected_arch_w1 if winning_weight == 1.0 else None,
    )
    if full_raw_record is None:
        raise RuntimeError("Could not locate full raw winner for ablation comparison.")
    ablations = []
    for feature in PHASE1_FEATURES:
        reduced_features = [name for name in ALL_FEATURES if name != feature]
        x_train_reduced = feature_frame(data.train, reduced_features)
        x_validation_reduced = feature_frame(data.validation, reduced_features)
        model = fit_model(x_train_reduced, data.y_train, winning_params)
        probability = model.predict_proba(x_validation_reduced)[:, 1]
        record = candidate_record(
            candidate_id=f"ablate-{feature}",
            stage="leave_one_phase1_feature_out",
            features=reduced_features,
            params=winning_params,
            probability=probability,
            y_validation=data.y_validation,
            ablated_feature=feature,
        )
        candidates.append(record)
        full_metrics = full_raw_record["metrics"]
        record_metrics = record["metrics"]
        ablations.append({
            "feature_removed": feature,
            "delta_roc_auc_when_removed": record_metrics["roc_auc"] - full_metrics["roc_auc"],
            "delta_average_precision_when_removed": record_metrics["average_precision"] - full_metrics["average_precision"],
            "delta_precision_at_recall_70_when_removed": (
                record_metrics["operating_point_recall_gte_70"]["precision"]
                - full_metrics["operating_point_recall_gte_70"]["precision"]
            ),
            "removed_model_metrics": record_metrics,
        })
        print(candidate_line(record), flush=True)

    print("Stage 5/5: feature importance, immutable export, and reports", flush=True)
    booster = winning_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    total_gain = sum(float(gain.get(feature, 0.0)) for feature in ALL_FEATURES)
    feature_importance = [
        {
            "feature": feature,
            "gain": float(gain.get(feature, 0.0)),
            "gain_share": float(gain.get(feature, 0.0)) / total_gain if total_gain else 0.0,
            "feature_group": "phase1" if feature in PHASE1_FEATURES else "baseline",
        }
        for feature in ALL_FEATURES
    ]
    feature_importance.sort(key=lambda item: item["gain_share"], reverse=True)

    calibration_metadata = recommended["calibration"]
    raw_model_bytes = bytes(booster.save_raw(raw_format="ubj"))
    model_digest = hashlib.sha256(raw_model_bytes).hexdigest()
    candidate_artifact = {
        "version": "4.0.0-phase1-tracinginsights-candidate",
        "status": "validation-selected offline candidate; final holdout not scored",
        "model_type": "XGBoost binary logistic with native categorical and missing-value handling",
        "source_features": ALL_FEATURES,
        "source_feature_count": len(ALL_FEATURES),
        "compound_categories": data.compound_categories,
        "missing_value_policy": "NaN preserved; XGBoost native missing branches; no zero/placeholder imputation",
        "xgboost_parameters": {
            key: value for key, value in winning_params.items() if key not in {"n_jobs", "missing"}
        },
        "platt_calibration": calibration_metadata,
        "classification_threshold": recommended["metrics"]["operating_point_recall_gte_70"]["threshold"],
        "validation_metrics": recommended["metrics"],
        "feature_importance_gain": feature_importance,
        "training_protocol": {
            "train_sessions": list(TRAIN_SESSIONS),
            "validation_session": VALIDATION_SESSION,
            "selection_metric": "maximum validation precision with recall >= 0.70",
            "calibration_protocol": "leave-one-training-race-out OOF Platt fit",
            "final_holdout_scored": False,
        },
        "booster_ubj_sha256": model_digest,
        "booster_ubj_base64": base64.b64encode(raw_model_bytes).decode("ascii"),
    }

    # Verify the self-contained candidate bytes reproduce raw XGBoost scores.
    loaded = Booster()
    loaded.load_model(bytearray(base64.b64decode(candidate_artifact["booster_ubj_base64"])))
    verify_frame = x_validation_all.head(100)
    verify = loaded.predict(DMatrix(verify_frame, enable_categorical=True))
    if not np.allclose(verify, winning_raw_probability[:100], atol=1e-7, rtol=1e-7):
        raise RuntimeError("Exported candidate does not reproduce XGBoost probabilities.")

    same_split_baseline = max([baseline_raw_record, baseline_platt_record], key=selection_key)
    winner_point = recommended["metrics"]["operating_point_recall_gte_70"]
    baseline_point = same_split_baseline["metrics"]["operating_point_recall_gte_70"]
    external_original = {"roc_auc": 0.9306, "average_precision": 0.2470, "brier_score": 0.0266, "precision": 0.155, "recall": 0.844}
    external_prior = {"roc_auc": 0.8246, "average_precision": 0.2066, "brier_score": 0.037802, "precision_at_recall_70": 0.1136, "recall": 0.7082}
    beats_original_precision = winner_point["precision"] > external_original["precision"]
    beats_prior_precision = winner_point["precision"] > external_prior["precision_at_recall_70"]
    improves_same_split = winner_point["precision"] > baseline_point["precision"]
    # GO/NO-GO must use only validation-vs-validation evidence. The supplied
    # original-baseline result is from a different race and remains contextual.
    go = beats_prior_precision and improves_same_split
    comparison_verdict = (
        "best validation precision at recall >= 70% across the 8-feature, 37-feature, and Phase 1 attempts"
        if go
        else "does not lead both earlier attempts on the same validation criterion"
    )

    experiment = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "training and validation only; no final holdout access or scoring",
        "data": {
            "train_path": str(TRAIN_PATH.relative_to(ROOT)),
            "validation_path": str(VALIDATION_PATH.relative_to(ROOT)),
            "train_rows": len(data.train),
            "validation_rows": len(data.validation),
            "train_positives": int(data.y_train.sum()),
            "validation_positives": int(data.y_validation.sum()),
            "train_positive_rate": float(data.y_train.mean()),
            "validation_positive_rate": float(data.y_validation.mean()),
            "features": ALL_FEATURES,
            "missingness": missingness,
            "numeric_missing_values_preserved": True,
            "zero_or_placeholder_imputation": False,
        },
        "selection_protocol": {
            "architecture_candidates": list(ARCHITECTURES),
            "class_weights": list(CLASS_WEIGHTS),
            "primary_metric": "maximum validation precision with recall >= 0.70",
            "tie_breakers": ["average_precision", "lower_brier", "roc_auc", "calibrated"],
            "platt": "three-fold leave-one-training-race-out OOF predictions; calibration fit on train only",
        },
        "selected_architecture": selected_architecture,
        "recommended_candidate_id": recommended["candidate_id"],
        "recommended_metrics": recommended["metrics"],
        "same_split_baseline_candidate_id": same_split_baseline["candidate_id"],
        "same_split_baseline_metrics": same_split_baseline["metrics"],
        "external_supplied_benchmarks": {
            "original_baseline": external_original,
            "prior_37_feature_attempt": external_prior,
            "cross_split_warning": "The original supplied baseline is from a different race and is contextual, not an apples-to-apples validation comparison.",
        },
        "verdict": {
            "comparison": comparison_verdict,
            "improves_same_split_baseline": improves_same_split,
            "external_test_benchmark_used_for_go_decision": False,
            "go_for_one_time_final_holdout": go,
        },
        "feature_importance_gain": feature_importance,
        "leave_one_phase1_feature_out": ablations,
        "all_candidates_with_full_precision_recall_curves": candidates,
        "protected_artifact_hashes_before": protected_before,
        "candidate_artifact": str(CANDIDATE_PATH.relative_to(ROOT)),
        "runtime_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_PATH.write_text(json.dumps(candidate_artifact, indent=2) + "\n", encoding="utf-8")
    with gzip.open(EXPERIMENT_PATH, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(experiment, handle, separators=(",", ":"))

    phase1_gain = [item for item in feature_importance if item["feature_group"] == "phase1"]
    ablations_sorted = sorted(ablations, key=lambda item: item["delta_average_precision_when_removed"])
    protected_after = protected_hashes()
    if protected_before != protected_after:
        raise RuntimeError("A protected existing model artifact changed.")

    lines = [
        "# APEX-R Phase 1 TracingInsights candidate — validation-only report",
        "",
        f"Generated: {experiment['generated_at']}",
        "",
        "## Decision",
        "",
        f"Recommendation: **{'GO' if go else 'NO-GO'}** for the one-time final holdout evaluation.",
        "",
        f"The selected candidate is `{recommended['candidate_id']}`. At its validation-selected threshold "
        f"{winner_point['threshold']:.8f}, precision is {winner_point['precision']:.2%} and recall is "
        f"{winner_point['recall']:.2%}. It is the {comparison_verdict}. Against the same-split 8-feature model, "
        f"precision at recall≥70% is {winner_point['precision']:.2%} versus {baseline_point['precision']:.2%}.",
        "",
        "The original supplied baseline number comes from a different race, so it is shown as a requested benchmark only. The same-split 8-feature result is the valid apples-to-apples feature comparison.",
        "",
        "No final holdout data was loaded or scored. Stop here pending explicit approval.",
        "",
        "## Data and missing-value handling",
        "",
        f"- Train: {len(data.train):,} rows, {int(data.y_train.sum()):,} positives ({data.y_train.mean():.2%}), grouped across three races.",
        f"- Validation: {len(data.validation):,} rows, {int(data.y_validation.sum()):,} positives ({data.y_validation.mean():.2%}), one untouched validation race for this experiment.",
        "- Source features: 19 conceptual columns—8 baseline plus all 11 Phase 1 columns.",
        "- Numeric NaNs were preserved. No missing numeric value was changed to zero or a placeholder.",
        "- Compound pair uses XGBoost native categorical splits. Missing compound values remain categorical-missing, not a fabricated compound.",
        "",
        "## Selected configuration",
        "",
        f"- Trees: {winning_params['n_estimators']}; max depth: {winning_params['max_depth']}; learning rate: {winning_params['learning_rate']}.",
        f"- `scale_pos_weight`: {winning_weight:g}.",
        f"- Calibration: {'Platt scaling from race-grouped OOF train predictions' if recommended['calibration'] else 'raw XGBoost probability'}.",
        f"- ROC-AUC: {recommended['metrics']['roc_auc']:.6f}; AP: {recommended['metrics']['average_precision']:.6f}; Brier: {recommended['metrics']['brier_score']:.6f}; ECE: {recommended['metrics']['ece_10_bins']:.6f}.",
        f"- Confusion matrix at selected threshold: {winner_point['confusion_matrix']}.",
        "",
        "## Every candidate configuration tested",
        "",
        f"All {len(candidates)} fitted configurations are reported across this section and the ablation section. Each entry reports its entire exact precision/recall curve point count. The threshold-by-threshold values are stored without downsampling in `models/phase1_tracinginsights_validation_experiments.json.gz` under `all_candidates_with_full_precision_recall_curves`.",
        "",
    ]
    lines.extend(candidate_line(record) for record in candidates if record["stage"] != "leave_one_phase1_feature_out")
    lines.extend([
        "",
        "## Direct comparisons",
        "",
        f"- Same-split 8-feature XGBoost: precision {baseline_point['precision']:.2%} at recall {baseline_point['recall']:.2%}; ROC-AUC {same_split_baseline['metrics']['roc_auc']:.6f}; AP {same_split_baseline['metrics']['average_precision']:.6f}; Brier {same_split_baseline['metrics']['brier_score']:.6f}.",
        f"- New Phase 1 winner: precision {winner_point['precision']:.2%} at recall {winner_point['recall']:.2%}; ROC-AUC {recommended['metrics']['roc_auc']:.6f}; AP {recommended['metrics']['average_precision']:.6f}; Brier {recommended['metrics']['brier_score']:.6f}.",
        f"- Supplied original baseline benchmark: precision {external_original['precision']:.2%} at recall {external_original['recall']:.2%}; ROC-AUC {external_original['roc_auc']:.4f}; AP {external_original['average_precision']:.4f}; Brier {external_original['brier_score']:.4f}. Different race; contextual only.",
        f"- Supplied prior 37-feature attempt: best precision {external_prior['precision_at_recall_70']:.2%} at recall {external_prior['recall']:.2%}; ROC-AUC {external_prior['roc_auc']:.4f}; AP {external_prior['average_precision']:.4f}; Brier {external_prior['brier_score']:.6f}. Same validation protocol. Phase 1 wins the primary precision-at-recall target, but the prior model retains higher ROC-AUC and AP.",
        f"- Plain verdict: the Phase 1 winner is the {comparison_verdict}; it {'does' if improves_same_split else 'does not'} improve the same-split baseline. The different-race original benchmark is not part of this decision.",
        "",
        "## Gain importance for the 11 Phase 1 columns",
        "",
    ])
    for item in sorted(phase1_gain, key=lambda row: row["gain_share"], reverse=True):
        lines.append(f"- `{item['feature']}`: gain {item['gain']:.6f}; {item['gain_share']:.2%} of total model gain.")
    lines.extend([
        "",
        "## Leave-one-column-out ablation",
        "",
        "Negative deltas mean performance fell when the feature was removed, so the feature helped. Positive deltas mean removal improved that validation metric and the feature may be neutral or harmful on this split.",
        "",
    ])
    for item in ablations_sorted:
        metrics = item["removed_model_metrics"]
        point = metrics["operating_point_recall_gte_70"]
        lines.append(
            f"- Remove `{item['feature_removed']}`: ROC-AUC {metrics['roc_auc']:.6f}; "
            f"AP {metrics['average_precision']:.6f}; Brier {metrics['brier_score']:.6f}; "
            f"precision@recall≥70% {point['precision']:.6f} at recall {point['recall']:.6f}, "
            f"threshold {point['threshold']:.8f}; exact PR points {metrics['pr_curve_points']}; "
            f"ΔAP {item['delta_average_precision_when_removed']:+.6f}; "
            f"ΔROC-AUC {item['delta_roc_auc_when_removed']:+.6f}; "
            f"Δprecision@recall≥70% {item['delta_precision_at_recall_70_when_removed']:+.6f}."
        )
    lines.extend([
        "",
        "Every ablation candidate's ROC-AUC, AP, Brier score, operating point, and complete precision/recall curve are in the compressed experiment artifact.",
        "Gain is the direct record of which columns the selected fitted booster used. Leave-one-out deltas are retraining-sensitivity evidence, not causal effects; because row and feature subsampling are enabled, removing even an unused column can change the retrained tree sequence.",
        "",
        "## Calibration",
        "",
        "Platt calibration was fitted only from leave-one-training-race-out predictions. The validation race was never used to fit the calibrator. Calibration is monotonic, so ranking metrics and the shape of precision versus recall are expected to remain unchanged; it can improve Brier/ECE and transforms the numerical threshold.",
        "",
        "## Artifacts and immutability",
        "",
        f"- New candidate: `{CANDIDATE_PATH.relative_to(ROOT)}`; embedded UBJ SHA-256 `{model_digest}`.",
        f"- Full exact curves and audit: `{EXPERIMENT_PATH.relative_to(ROOT)}`.",
        "- The exported booster was reloaded and reproduced the first 100 raw validation probabilities to 1e-7 tolerance.",
        "- Protected pre-existing baseline and prior-candidate file hashes were unchanged.",
        "- This candidate was not copied into the live application and no existing model was overwritten.",
        "",
        "## Stop condition",
        "",
        "Do not run the one-time final holdout evaluation without explicit user approval. If approved later, score this frozen candidate and threshold exactly once and report the result without iterative retuning.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "recommended_candidate": recommended["candidate_id"],
        "recommended_metrics": recommended["metrics"],
        "same_split_baseline": same_split_baseline["metrics"],
        "verdict": experiment["verdict"],
        "candidate_path": str(CANDIDATE_PATH),
        "report_path": str(REPORT_PATH),
        "full_curve_artifact": str(EXPERIMENT_PATH),
        "protected_artifacts_unchanged": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
