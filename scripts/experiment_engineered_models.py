#!/usr/bin/env python3
"""Train/validation-only APEX-R feature, calibration and model experiments.

Safety: this script has a fixed allow-list for the three training races and
session 9070 validation.  It never accepts or requests session 11353.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import train_model as base
from train_xgboost_enriched import CarLookup, fetch_session
from train_xgboost import sigmoid, tree_margin


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TRAIN_SESSIONS = (7953, 7779, 7787)
VALIDATION_SESSION = 9070
FORBIDDEN_HOLDOUT = 11353


def finite(value, default=0.0):
    result = base.numeric(value, default)
    return default if result is None else float(result)


def index_rows(rows, date_field="date"):
    grouped = defaultdict(list)
    for row in rows:
        driver = row.get("driver_number")
        t = base.timestamp(row.get(date_field) or row.get("date_start"))
        if driver is not None and t is not None:
            grouped[int(driver)].append((t, row))
    return {driver: sorted(values) for driver, values in grouped.items()}


def latest(values, t):
    if not values:
        return None
    index = bisect.bisect_right([item[0] for item in values], t) - 1
    return values[index][1] if index >= 0 else None


def last_completed_lap(values, t):
    complete = []
    for start, row in values:
        duration = finite(row.get("lap_duration"), -1)
        if duration > 0 and start + duration <= t:
            complete.append((start + duration, row))
    return latest(complete, t)


def stint_for(stints, driver, lap):
    selected = None
    for row in stints.get(driver, []):
        if finite(row.get("lap_start"), 0) <= lap:
            selected = row
        else:
            break
    return selected or {}


def car_latest(car, field, t, default):
    if not car or not car.times:
        return default
    index = bisect.bisect_right(car.times, t) - 1
    return car.values[field][index] if index >= 0 else default


def car_mean(car, field, start, end, default):
    return car.mean(field, start, end, default) if car and car.times else default


def interval_values(values, start, end):
    return [finite(row.get("interval"), np.nan) for t, row in values if start <= t <= end and base.numeric(row.get("interval")) is not None]


def interval_slope(values, t, horizon):
    now = latest(values, t)
    before = latest(values, t - horizon)
    if not now or not before:
        return 0.0
    current, prior = base.numeric(now.get("interval")), base.numeric(before.get("interval"))
    if current is None or prior is None:
        return 0.0
    return max(-30.0, min(30.0, (prior - current) / horizon * 60.0))


def opponent_at(position_rows, drivers, driver, t, own_position):
    target_position = int(round(own_position)) - 1
    if target_position < 1:
        return None
    for candidate in drivers:
        if candidate == driver:
            continue
        row = latest(position_rows.get(candidate, []), t)
        if row and int(round(finite(row.get("position"), 99))) == target_position:
            return candidate
    return None


def session_features(session_key: int):
    # fetch_session contains the approved clean-label construction and only
    # receives sessions from the allow-list in main().
    rows, manifest = fetch_session(session_key, base.CACHE)
    intervals, _ = base.api_get("intervals", {"session_key": session_key}, base.CACHE)
    positions, _ = base.api_get("position", {"session_key": session_key}, base.CACHE)
    laps, _ = base.api_get("laps", {"session_key": session_key}, base.CACHE)
    stints_raw, _ = base.api_get("stints", {"session_key": session_key}, base.CACHE)
    drivers = sorted({int(row["driver_number"]) for row in rows})
    interval_by_driver = index_rows(intervals)
    position_by_driver = index_rows(positions)
    lap_by_driver = index_rows(laps, "date_start")
    stints = defaultdict(list)
    for row in stints_raw:
        if row.get("driver_number") is not None:
            stints[int(row["driver_number"])].append(row)
    for values in stints.values():
        values.sort(key=lambda row: finite(row.get("lap_start"), 0))
    cars, raw_drs_by_driver = {}, {}
    for driver in drivers:
        car_rows, _ = base.api_get("car_data", {"session_key": session_key, "driver_number": driver}, base.CACHE)
        cars[driver] = CarLookup(car_rows)
        raw_drs_by_driver[driver] = [(base.timestamp(row.get("date")), finite(row.get("drs"), 0.0)) for row in car_rows if base.timestamp(row.get("date")) is not None]

    engineered = []
    for row in rows:
        t, driver = base.timestamp(row["date"]), int(row["driver_number"])
        if t is None:
            continue
        feature = dict(row["features"])
        own_lap = latest(lap_by_driver.get(driver, []), t)
        lap_number = finite((own_lap or {}).get("lap_number"), 1)
        ahead = opponent_at(position_by_driver, drivers, driver, t, feature["position"])
        own_stint = stint_for(stints, driver, lap_number)
        ahead_lap = latest(lap_by_driver.get(ahead, []), t) if ahead else None
        ahead_lap_number = finite((ahead_lap or {}).get("lap_number"), lap_number)
        ahead_stint = stint_for(stints, ahead, ahead_lap_number) if ahead else {}
        own_car, ahead_car = cars.get(driver), cars.get(ahead)
        raw_intervals = interval_by_driver.get(driver, [])
        interval_window = interval_values(raw_intervals, t - 60, t)
        drs = car_latest(own_car, "drs_open", t, 0.0)
        # CarLookup stores open/not-open.  The raw state also records code 8,
        # interpreted by OpenF1 as eligibility in a DRS activation zone.
        drs_samples = raw_drs_by_driver.get(driver, [])
        raw_car_index = bisect.bisect_right([sample[0] for sample in drs_samples], t) - 1
        raw_drs_value = drs_samples[raw_car_index][1] if raw_car_index >= 0 else 0.0
        own_complete = last_completed_lap(lap_by_driver.get(driver, []), t)
        ahead_complete = last_completed_lap(lap_by_driver.get(ahead, []), t) if ahead else None
        pace_available = float(bool(own_complete and ahead_complete))
        own_age = finite(own_stint.get("tyre_age_at_start"), 0) + max(0, lap_number - finite(own_stint.get("lap_start"), lap_number))
        ahead_age = finite(ahead_stint.get("tyre_age_at_start"), 0) + max(0, ahead_lap_number - finite(ahead_stint.get("lap_start"), ahead_lap_number))
        speed_now = car_latest(own_car, "speed", t, 200.0)
        speed_before = car_latest(own_car, "speed", t - 5, speed_now)
        feature.update({
            "gap_slope_5s": interval_slope(raw_intervals, t, 5),
            "gap_slope_15s": interval_slope(raw_intervals, t, 15),
            "gap_slope_30s": interval_slope(raw_intervals, t, 30),
            "gap_slope_60s": interval_slope(raw_intervals, t, 60),
            "gap_volatility_60s": float(np.std(interval_window)) if len(interval_window) > 1 else 0.0,
            "drs_open": drs,
            "drs_eligible_or_open": float(raw_drs_value in {8.0, 10.0, 12.0, 14.0}),
            "rival_drs_open": car_latest(ahead_car, "drs_open", t, 0.0) if ahead else 0.0,
            "tyre_age_delta_ahead_minus_self": ahead_age - own_age if ahead else 0.0,
            "same_tyre_compound": float(bool(ahead) and own_stint.get("compound") == ahead_stint.get("compound")),
            "stint_length_delta_ahead_minus_self": (ahead_lap_number - finite(ahead_stint.get("lap_start"), ahead_lap_number)) - (lap_number - finite(own_stint.get("lap_start"), lap_number)) if ahead else 0.0,
            "relative_pace_available": pace_available,
            "lap_time_delta_ahead_minus_self": finite((ahead_complete or {}).get("lap_duration"), 0) - finite((own_complete or {}).get("lap_duration"), 0) if pace_available else 0.0,
            "sector_1_delta_ahead_minus_self": finite((ahead_complete or {}).get("duration_sector_1"), 0) - finite((own_complete or {}).get("duration_sector_1"), 0) if pace_available else 0.0,
            "sector_2_delta_ahead_minus_self": finite((ahead_complete or {}).get("duration_sector_2"), 0) - finite((own_complete or {}).get("duration_sector_2"), 0) if pace_available else 0.0,
            "sector_3_delta_ahead_minus_self": finite((ahead_complete or {}).get("duration_sector_3"), 0) - finite((own_complete or {}).get("duration_sector_3"), 0) if pace_available else 0.0,
            "new_stint_proxy": float(own_age <= 1.0),
            "speed_delta_5s": speed_now - speed_before,
            "high_throttle_fraction_5s": car_mean(own_car, "throttle", t - 5, t, 0) / 100.0,
            "brake_fraction_5s": car_mean(own_car, "brake", t - 5, t, 0) / 100.0,
            "straight_approach_proxy": float(speed_now >= 250 and car_mean(own_car, "throttle", t - 5, t, 0) >= 75),
            "braking_approach_proxy": float(car_mean(own_car, "brake", t - 5, t, 0) >= 20 or speed_now - speed_before <= -25),
        })
        engineered.append({"session_key": session_key, "features": feature, "label": row["overtake_next_60s"]})
    return engineered, manifest


def pr_summary(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    candidates = [(float(precision[i]), float(recall[i]), float(thresholds[i])) for i in range(len(thresholds)) if recall[i] >= .70]
    best = max(candidates, default=(0.0, 0.0, 1.0), key=lambda x: (x[0], x[2]))
    return {"precision_at_recall_70": round(best[0], 6), "recall_at_selected_threshold": round(best[1], 6), "threshold": round(best[2], 8), "curve": [{"threshold": round(float(thresholds[i]), 8), "precision": round(float(precision[i]), 6), "recall": round(float(recall[i]), 6)} for i in range(0, len(thresholds), max(1, len(thresholds)//40))]}


def metrics(y, p):
    return {"roc_auc": round(float(roc_auc_score(y, p)), 6), "average_precision": round(float(average_precision_score(y, p)), 6), "brier_score": round(float(brier_score_loss(y, p)), 6), **pr_summary(y, p)}


def platt_oof(x, y, groups, params):
    oof = np.zeros(len(y))
    for group in sorted(set(groups)):
        train = groups != group
        model = XGBClassifier(**params)
        model.fit(x.loc[train], y[train], verbose=False)
        oof[~train] = model.predict_proba(x.loc[~train])[:, 1]
    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=2026)
    logit = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / np.clip(1 - oof, 1e-6, 1))
    calibrator.fit(logit.reshape(-1, 1), y)
    return calibrator


def main() -> int:
    sessions = (*TRAIN_SESSIONS, VALIDATION_SESSION)
    if FORBIDDEN_HOLDOUT in sessions:
        raise RuntimeError("Locked holdout must never enter experiments.")
    rows, manifests = [], []
    for session in sessions:
        print(f"Engineering session {session} (holdout excluded)...", flush=True)
        part, manifest = session_features(session)
        rows.extend(part); manifests.append(manifest)
    feature_names = sorted({name for row in rows for name in row["features"]})
    train = [row for row in rows if row["session_key"] in TRAIN_SESSIONS]
    validation = [row for row in rows if row["session_key"] == VALIDATION_SESSION]
    x_train = pd.DataFrame([row["features"] for row in train], columns=feature_names).fillna(0.0)
    x_val = pd.DataFrame([row["features"] for row in validation], columns=feature_names).fillna(0.0)
    y_train = np.asarray([row["label"] for row in train], dtype=int)
    y_val = np.asarray([row["label"] for row in validation], dtype=int)
    groups = np.asarray([row["session_key"] for row in train])
    engineered_names = [name for name in feature_names if name not in base.FEATURES]
    mi = mutual_info_classif(x_val[engineered_names], y_val, random_state=2026, discrete_features=[name.endswith(("open", "proxy", "available", "compound")) for name in engineered_names])
    feature_audit = []
    for name, value in zip(engineered_names, mi):
        corr = float(np.corrcoef(x_val[name], y_val)[0, 1]) if x_val[name].std() else 0.0
        feature_audit.append({"feature": name, "validation_point_biserial_correlation": round(corr, 6), "validation_mutual_information": round(float(value), 6)})
    feature_audit.sort(key=lambda row: row["validation_mutual_information"], reverse=True)
    params = {"n_estimators": 700, "max_depth": 4, "learning_rate": .03, "min_child_weight": 8, "subsample": .9, "colsample_bytree": .9, "reg_alpha": .05, "reg_lambda": 6, "max_delta_step": 1, "objective": "binary:logistic", "eval_metric": "auc", "early_stopping_rounds": 80, "random_state": 2026, "n_jobs": 4}
    candidates, trained = [], {}
    for weight in (1.0, 1.5, 2.0, 3.0):
        current = {**params, "scale_pos_weight": weight}
        model = XGBClassifier(**current)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        count = int(model.best_iteration) + 1 if model.best_iteration is not None else params["n_estimators"]
        probability = model.predict_proba(x_val, iteration_range=(0, count))[:, 1]
        item = {"name": f"XGBoost engineered, scale_pos_weight={weight}", "kind": "xgboost", "scale_pos_weight": weight, "best_trees": count, "metrics": metrics(y_val, probability)}
        candidates.append(item); trained[weight] = (model, probability, count, current)
    winner = max(candidates, key=lambda item: (item["metrics"]["precision_at_recall_70"], item["metrics"]["average_precision"]))
    chosen_weight = winner["scale_pos_weight"]
    chosen_model, raw_probability, chosen_trees, chosen_params = trained[chosen_weight]
    calibrator = platt_oof(x_train, y_train, groups, {**chosen_params, "n_estimators": chosen_trees, "early_stopping_rounds": None})
    raw_logit = np.log(np.clip(raw_probability, 1e-6, 1 - 1e-6) / np.clip(1 - raw_probability, 1e-6, 1))
    calibrated_probability = calibrator.predict_proba(raw_logit.reshape(-1, 1))[:, 1]
    candidates.append({"name": f"XGBoost engineered, scale_pos_weight={chosen_weight}, Platt calibrated", "kind": "xgboost_platt", "scale_pos_weight": chosen_weight, "metrics": metrics(y_val, calibrated_probability)})
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32, 16), alpha=1e-3, batch_size=128, early_stopping=True, validation_fraction=.15, n_iter_no_change=30, max_iter=400, random_state=2026))
    weights = np.where(y_train == 1, chosen_weight, 1.0)
    mlp.fit(x_train, y_train, mlpclassifier__sample_weight=weights)
    candidates.append({"name": "Feed-forward MLP (64,32,16), class-weighted, L2 + early stopping; no dropout runtime", "kind": "mlp", "metrics": metrics(y_val, mlp.predict_proba(x_val)[:, 1])})
    interaction = x_train.copy(); interaction_val = x_val.copy()
    for frame in (interaction, interaction_val):
        frame["interaction_tyre_delta_x_gap"] = frame["tyre_age_delta_ahead_minus_self"] * frame["interval_sec"]
        frame["interaction_drs_x_gap_slope"] = frame["drs_eligible_or_open"] * frame["gap_slope_15s"]
    interaction_model = XGBClassifier(**chosen_params)
    interaction_model.fit(interaction, y_train, eval_set=[(interaction_val, y_val)], verbose=False)
    interaction_trees = int(interaction_model.best_iteration) + 1 if interaction_model.best_iteration is not None else params["n_estimators"]
    candidates.append({"name": f"XGBoost engineered interactions, scale_pos_weight={chosen_weight}", "kind": "xgboost_interactions", "scale_pos_weight": chosen_weight, "best_trees": interaction_trees, "metrics": metrics(y_val, interaction_model.predict_proba(interaction_val, iteration_range=(0, interaction_trees))[:, 1])})
    recommended = max(candidates, key=lambda item: (item["metrics"]["precision_at_recall_70"], item["metrics"]["average_precision"], -item["metrics"]["brier_score"], item["kind"] == "xgboost_platt"))
    # Freeze the selected offline candidate.  It deliberately is not copied to
    # dist/apex-model.js: the browser replay currently lacks full rival data.
    booster = chosen_model.get_booster()
    trees = [json.loads(tree) for tree in booster.get_dump(dump_format="json")][:chosen_trees]
    config = json.loads(booster.save_config())
    base_probability = float(config["learner"]["learner_model_param"]["base_score"].strip("[]"))
    base_margin = math.log(base_probability / (1 - base_probability))
    for values, expected in zip(x_val.head(50).to_dict("records"), raw_probability[:50]):
        portable = sigmoid(base_margin + sum(tree_margin(tree, values) for tree in trees))
        if abs(portable - float(expected)) > 1e-6:
            raise RuntimeError("Frozen engineered candidate export does not match XGBoost.")
    frozen_candidate = {
        "version": "3.0.0-engineered-xgboost-scale2-platt-candidate",
        "model_type": "xgboost_binary_logistic_with_platt_calibration",
        "features": feature_names, "base_margin": base_margin, "trees": trees,
        "platt_calibration": {"input": "logit(raw_xgboost_probability)", "coefficient": float(calibrator.coef_[0][0]), "intercept": float(calibrator.intercept_[0])},
        "classification_threshold": recommended["metrics"]["threshold"],
        "metadata": {"train_sessions": list(TRAIN_SESSIONS), "validation_session": VALIDATION_SESSION, "forbidden_final_holdout": FORBIDDEN_HOLDOUT, "final_holdout_scored": False, "selection_metric": "validation_precision_at_recall_70", "metrics": recommended["metrics"], "scale_pos_weight": chosen_weight, "best_trees": chosen_trees, "feature_importance_gain": {name: round(float(booster.get_score(importance_type="gain").get(name, 0)), 6) for name in feature_names}, "integration_status": "offline candidate only; browser needs full-session rival telemetry or a feature service"},
    }
    candidate_path = MODELS / "candidates" / "xgboost_engineered_scale2_platt_candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(frozen_candidate, indent=2) + "\n", encoding="utf-8")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "protocol": {"train_sessions": list(TRAIN_SESSIONS), "validation_session": VALIDATION_SESSION, "forbidden_final_holdout": FORBIDDEN_HOLDOUT, "holdout_loaded": False}, "feature_audit_validation_only": feature_audit, "candidates_validation_only": candidates, "recommended_before_final_holdout": recommended, "frozen_candidate_path": str(candidate_path), "notes": ["Pit endpoint returned no historical records for these sessions; new_stint_proxy is an auditable proxy, not a claimed pit-lane event.", "Track geometry is represented by telemetry-only straight/braking proxies; no fabricated circuit map or DRS detection coordinate was used.", "The MLP is an honest no-dropout fallback because PyTorch/TensorFlow are not installed.", "No candidate was evaluated on session 11353.", "The frozen candidate is not activated in the browser because the current one-driver replay cannot calculate rival-relative features."], "session_manifests": manifests}
    (MODELS / "engineered_model_experiments.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# APEX-R engineered feature experiments — validation only", "", "**Locked final hold-out session 11353 was not loaded.**", "", "## Feature audit", "", "| Feature | Point-biserial correlation | Mutual information |", "| --- | ---: | ---: |"]
    lines += [f"| {row['feature']} | {row['validation_point_biserial_correlation']} | {row['validation_mutual_information']} |" for row in feature_audit]
    lines += ["", "## Candidate comparison", "", "| Candidate | ROC-AUC | AP | Brier | Best precision with recall ≥70% | Recall at selected point |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| {item['name']} | {item['metrics']['roc_auc']} | {item['metrics']['average_precision']} | {item['metrics']['brier_score']} | {item['metrics']['precision_at_recall_70']} | {item['metrics']['recall_at_selected_threshold']} |" for item in candidates]
    lines += ["", "## Recommendation before frozen holdout", "", f"**{recommended['name']}** — selected by validation precision at recall ≥70%, then average precision and lower Brier score as tie-breakers.", f"Frozen offline candidate: `{candidate_path.name}`. It is intentionally not activated in the browser until full-session rival telemetry is available.", "", "Do not score session 11353 until the user explicitly approves the frozen candidate and threshold.", ""]
    (MODELS / "ENGINEERED_MODEL_EXPERIMENTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"recommended": recommended, "top_features": feature_audit[:10], "holdout_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
