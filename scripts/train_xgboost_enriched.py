#!/usr/bin/env python3
"""Train APEX-R's browser-compatible enriched OpenF1 XGBoost model.

This pipeline is deliberately stricter than the original baseline:

* ``car_data`` supplies rolling speed, throttle, brake, RPM, gear and DRS inputs.
* ``pit`` and ``race_control`` are used to remove windows where a recorded
  position change is likely to be operational rather than an on-track pass.
* ``location`` is fetched and audited for every driver/session.  It is used by
  the replay importer; it is not used as a model feature because OpenF1's
  arbitrary local coordinate system cannot be reproduced reliably from every
  browser replay.

The final supplied session is a validation session for early stopping and
model selection.  For a publishable final metric, provide one further,
untouched race and score it separately after model selection.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

import train_model as base
from train_xgboost import browser_source, safe_metric, sigmoid, tree_margin
from generate_model_reports import write_reports


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DIST = ROOT / "dist"
HORIZON_SECONDS = base.HORIZON_SECONDS
PIT_BUFFER_SECONDS = 90.0
CONTROL_BUFFER_SECONDS = 60.0
CAR_FEATURES = [
    "speed_mean_10s",
    "speed_delta_10s",
    "throttle_mean_10s",
    "brake_fraction_10s",
    "rpm_mean_10s",
    "gear_mean_10s",
    "drs_open_fraction_10s",
]
FEATURES = base.FEATURES + CAR_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", default="7953,7779,7787,9070", help="Comma-separated race session keys; final session validates early stopping.")
    parser.add_argument("--cache-dir", type=Path, default=base.CACHE)
    parser.add_argument("--output", type=Path, default=MODELS / "xgboost_enriched_overtake_model.json")
    parser.add_argument("--force-activate", action="store_true", help="Publish this model even if its validation ROC-AUC is lower than the current active XGBoost artifact.")
    return parser.parse_args()


def finite(value, default: float) -> float:
    result = base.numeric(value, default)
    return default if result is None else float(result)


def timestamps(rows: list[dict]) -> list[float]:
    return [base.timestamp(row.get("date")) for row in rows if base.timestamp(row.get("date")) is not None]


def nearest_in_window(values: list[float], start: float, end: float) -> bool:
    index = bisect.bisect_left(values, start)
    return index < len(values) and values[index] <= end


def control_times(rows: list[dict]) -> list[float]:
    """Return timing of conditions that make a pass label unreliable."""
    values = []
    for row in rows:
        flag = str(row.get("flag") or "").upper()
        category = str(row.get("category") or "").upper()
        message = str(row.get("message") or "").upper()
        abnormal = (
            (category == "FLAG" and flag not in {"", "GREEN", "CLEAR"})
            or "SAFETYCAR" in category
            or "SAFETY CAR" in message
            or "VIRTUAL SAFETY" in message
            or flag in {"RED", "YELLOW", "DOUBLE YELLOW"}
        )
        t = base.timestamp(row.get("date"))
        if abnormal and t is not None:
            values.append(t)
    return sorted(values)


def optional_api_get(endpoint: str, params: dict, cache_dir: Path) -> tuple[list, str, bool]:
    """OpenF1 returns HTTP 404 rather than an empty array for some old feeds."""
    try:
        rows, url = base.api_get(endpoint, params, cache_dir)
        return rows, url, True
    except RuntimeError as error:
        if "HTTP Error 404" not in str(error):
            raise
        query = "&".join(f"{key}={value}" for key, value in params.items())
        return [], base.API + endpoint + "?" + query, False


class CarLookup:
    """O(1) rolling telemetry aggregates after two binary searches."""

    FIELDS = ("speed", "throttle", "brake", "rpm", "n_gear", "drs_open")

    def __init__(self, rows: list[dict]):
        samples = []
        for row in rows:
            t = base.timestamp(row.get("date"))
            if t is None:
                continue
            drs = finite(row.get("drs"), 0.0)
            samples.append((t, {
                "speed": finite(row.get("speed"), 200.0),
                "throttle": finite(row.get("throttle"), 60.0),
                "brake": finite(row.get("brake"), 0.0),
                "rpm": finite(row.get("rpm"), 10000.0),
                "n_gear": finite(row.get("n_gear"), 5.0),
                "drs_open": float(drs in {10.0, 12.0, 14.0}),
            }))
        samples.sort(key=lambda sample: sample[0])
        self.times = [sample[0] for sample in samples]
        self.values = {field: [sample[1][field] for sample in samples] for field in self.FIELDS}
        self.prefix = {field: np.concatenate(([0.0], np.cumsum(values))) for field, values in self.values.items()}

    def mean(self, field: str, start: float, end: float, default: float) -> float:
        left = bisect.bisect_left(self.times, start)
        right = bisect.bisect_right(self.times, end)
        if right <= left:
            return default
        return float((self.prefix[field][right] - self.prefix[field][left]) / (right - left))

    def latest(self, field: str, t: float, default: float) -> float:
        index = bisect.bisect_right(self.times, t) - 1
        return self.values[field][index] if index >= 0 else default

    def features(self, t: float) -> dict[str, float]:
        speed_now = self.latest("speed", t, 200.0)
        speed_then = self.latest("speed", t - 10.0, speed_now)
        return {
            "speed_mean_10s": self.mean("speed", t - 10.0, t, 200.0),
            "speed_delta_10s": max(-300.0, min(300.0, speed_now - speed_then)),
            "throttle_mean_10s": self.mean("throttle", t - 10.0, t, 60.0),
            "brake_fraction_10s": self.mean("brake", t - 10.0, t, 0.0) / 100.0,
            "rpm_mean_10s": self.mean("rpm", t - 10.0, t, 10000.0),
            "gear_mean_10s": self.mean("n_gear", t - 10.0, t, 5.0),
            "drs_open_fraction_10s": self.mean("drs_open", t - 10.0, t, 0.0),
        }


def fetch_session(session_key: int, cache_dir: Path) -> tuple[list[dict], dict]:
    """Load base features plus all additional OpenF1 sources for one race."""
    base_rows, base_manifest = base.session_rows(session_key, cache_dir)
    pit, pit_url, pit_available = optional_api_get("pit", {"session_key": session_key}, cache_dir)
    race_control, control_url, control_available = optional_api_get("race_control", {"session_key": session_key}, cache_dir)
    drivers = sorted({int(row["driver_number"]) for row in base_rows})
    car_by_driver: dict[int, CarLookup] = {}
    location_counts: dict[int, int] = {}
    car_urls, location_urls = {}, {}

    def fetch_driver(driver: int) -> tuple[int, list, str, list, str]:
        car_rows, car_url = base.api_get("car_data", {"session_key": session_key, "driver_number": driver}, cache_dir)
        location_rows, location_url = base.api_get("location", {"session_key": session_key, "driver_number": driver}, cache_dir)
        return driver, car_rows, car_url, location_rows, location_url

    # A bounded pool keeps a full race download practical without issuing an
    # unbounded burst to the public API.  Cached calls simply return locally.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_driver, driver) for driver in drivers]
        for future in as_completed(futures):
            driver, car_rows, car_url, location_rows, location_url = future.result()
            car_by_driver[driver] = CarLookup(car_rows)
            location_counts[driver] = len(location_rows)
            car_urls[str(driver)] = car_url
            location_urls[str(driver)] = location_url
            print(f"  telemetry driver {driver}: {len(car_rows):,} car / {len(location_rows):,} location", flush=True)

    pit_by_driver: dict[int, list[float]] = defaultdict(list)
    for row in pit:
        t = base.timestamp(row.get("date"))
        driver = row.get("driver_number")
        if t is not None and driver is not None:
            pit_by_driver[int(driver)].append(t)
    for values in pit_by_driver.values():
        values.sort()
    abnormal_controls = control_times(race_control)

    overtakes, overtake_url = base.api_get("overtakes", {"session_key": session_key}, cache_dir)
    clean_events: dict[int, list[float]] = defaultdict(list)
    excluded_events = 0
    for row in overtakes:
        t = base.timestamp(row.get("date"))
        overtaking = row.get("overtaking_driver_number")
        overtaken = row.get("overtaken_driver_number")
        if t is None or overtaking is None:
            continue
        drivers_to_check = [int(overtaking)] + ([int(overtaken)] if overtaken is not None else [])
        near_pit = any(nearest_in_window(pit_by_driver.get(driver, []), t - PIT_BUFFER_SECONDS, t + PIT_BUFFER_SECONDS) for driver in drivers_to_check)
        near_control = nearest_in_window(abnormal_controls, t - CONTROL_BUFFER_SECONDS, t + CONTROL_BUFFER_SECONDS)
        if near_pit or near_control:
            excluded_events += 1
            continue
        clean_events[int(overtaking)].append(t)
    for values in clean_events.values():
        values.sort()

    enriched = []
    excluded_windows = 0
    for row in base_rows:
        t = base.timestamp(row["date"])
        driver = int(row["driver_number"])
        if t is None:
            continue
        # A negative during pit/neutralisation is not a meaningful failed attack.
        near_pit = nearest_in_window(pit_by_driver.get(driver, []), t - PIT_BUFFER_SECONDS, t + HORIZON_SECONDS + PIT_BUFFER_SECONDS)
        near_control = nearest_in_window(abnormal_controls, t - CONTROL_BUFFER_SECONDS, t + HORIZON_SECONDS + CONTROL_BUFFER_SECONDS)
        if near_pit or near_control:
            excluded_windows += 1
            continue
        events = clean_events.get(driver, [])
        event_index = bisect.bisect_right(events, t)
        label = int(event_index < len(events) and events[event_index] <= t + HORIZON_SECONDS)
        features = dict(row["features"])
        features.update(car_by_driver.get(driver, CarLookup([])).features(t))
        enriched.append({"session_key": session_key, "driver_number": driver, "date": row["date"], "features": features, "overtake_next_60s": label})

    manifest = {
        **base_manifest,
        "rows_before_cleaning": len(base_rows),
        "rows_after_cleaning": len(enriched),
        "positive_after_cleaning": sum(row["overtake_next_60s"] for row in enriched),
        "excluded_windows": excluded_windows,
        "raw_overtakes": len(overtakes),
        "excluded_overtakes": excluded_events,
        "clean_overtakes": sum(len(values) for values in clean_events.values()),
        "car_data_records": sum(len(lookup.times) for lookup in car_by_driver.values()),
        "location_records": sum(location_counts.values()),
        "pit_records": len(pit),
        "race_control_records": len(race_control),
        "pit_endpoint_available": pit_available,
        "race_control_endpoint_available": control_available,
        "urls": {**base_manifest["urls"], "pit": pit_url, "race_control": control_url, "overtakes_clean_label": overtake_url, "car_data_by_driver": car_urls, "location_by_driver": location_urls},
    }
    return enriched, manifest


def model_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    return {
        "rows": int(len(y_true)), "positive": int(y_true.sum()), "positive_rate": round(float(y_true.mean()), 6),
        "accuracy_at_0_5": round(float((prediction == y_true).mean()), 6),
        "roc_auc": safe_metric(roc_auc_score, y_true, probability),
        "average_precision": safe_metric(average_precision_score, y_true, probability),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 6),
    }


def f1_operating_threshold(y_true: np.ndarray, probability: np.ndarray) -> dict:
    """Choose the classification threshold on validation only, never test."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.argmax(f1))
    threshold = float(thresholds[index])
    return {"value": round(threshold, 8), "selection_split": "validation", "selection_objective": "maximum_f1", "validation_precision": round(float(precision[index]), 6), "validation_recall": round(float(recall[index]), 6), "validation_f1": round(float(f1[index]), 6)}


def report(artifact: dict, clean_baseline: dict, existing_xgb: dict | None, active: str) -> None:
    metadata = artifact["metadata"]
    baseline_metrics = clean_baseline["metrics"]
    metrics = metadata["metrics"]
    lines = [
        "# APEX-R enriched machine-learning model report", "",
        f"Generated: {metadata['trained_at']}", "",
        "## Active model", "",
        f"Browser model: **{active}**. Enriched XGBoost is activated only when its validation ROC-AUC is at least the clean-label base-feature XGBoost score (or `--force-activate` is used).", "",
        "## Dataset flow", "",
        f"- Train sessions: {', '.join(map(str, metadata['train_sessions']))}; validation session: {metadata['validation_sessions'][0]}.",
        f"- Training rows: {metadata['train_rows']:,}; training positive labels: {metadata['train_positive']:,}.",
        f"- Validation rows: {metrics['rows']:,}; validation positive labels: {metrics['positive']:,}.",
        "- Label: a clean observed overtake by the driver in the following 60 seconds.",
        "- `overtakes` is the event source. Samples around pit records and non-green/safety-car race-control records are removed; this reduces pit-stop and neutralisation position-change contamination but cannot prove every retained event is an on-track pass.",
        "- `car_data` is matched to each 10-second interval sample and aggregated over the preceding 10 seconds. `location` is fetched, cached and audited for replay provenance but omitted from prediction because its coordinates are circuit-local/arbitrary.", "",
        "## Validation comparison", "",
        "| Model | ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |", "| --- | ---: | ---: | ---: | ---: |",
        "| Clean-label base-feature XGBoost | " + " | ".join(str(baseline_metrics.get(key, "--")) for key in ["roc_auc", "average_precision", "brier_score", "accuracy_at_0_5"]) + " |",
        "| Clean-label enriched XGBoost | " + " | ".join(str(metrics.get(key, "--")) for key in ["roc_auc", "average_precision", "brier_score", "accuracy_at_0_5"]) + " |", "",
        "The two rows above use the identical cleaned labels and split; this is the selection comparison. The legacy simple XGBoost used noisier labels and is retained as a historical artifact, not as an apples-to-apples comparator.", "",
        "## Model and training", "", "```json", json.dumps(metadata["hyperparameters"], indent=2), "```", "",
        f"Requested boosting rounds: {metadata['hyperparameters']['n_estimators']}; enriched retained after validation early stopping: {metadata['best_trees']}; base-feature retained: {clean_baseline['best_trees']}.",
        "", "## Features", "", ", ".join(artifact["features"]), "",
        "## Limits", "", "- This is an overtake-opportunity classifier, not an ATTACK/HOLD/DEFEND labeler. The strategy engine still makes that constrained decision.", "- Validation is used for early stopping, so it is not a fully untouched final test. Score a fifth held-out race before making broader performance claims.", "- Public OpenF1 data has no private ERS, battery, tyre model or rival-intent telemetry.", "",
        "## Reproduce", "", "```bash", "python3 scripts/train_xgboost_enriched.py", "```", "",
    ]
    (MODELS / "ENRICHED_MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (MODELS / "MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (MODELS / "README.md").write_text(
        "# APEX-R model artifacts\n\n"
        "- `overtake_model.json`: original Logistic Regression baseline.\n"
        "- `xgboost_overtake_model.json`: simple XGBoost challenger.\n"
        "- `xgboost_clean_label_base_model.json`: clean-label base-feature XGBoost control.\n"
        "- `xgboost_enriched_overtake_model.json`: car-data enriched XGBoost model.\n"
        "- `ENRICHED_MODEL_REPORT.md`: data lineage, cleaning rules, features, hyperparameters and validation metrics.\n\n"
        "Run `python3 scripts/train_xgboost_enriched.py` to reproduce the enriched pipeline. Raw OpenF1 responses are cached in `data/openf1-cache/`.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    sessions = [int(value.strip()) for value in args.sessions.split(",") if value.strip()]
    if len(sessions) < 2:
        raise SystemExit("Provide at least two race sessions.")
    rows, manifests = [], []
    for session in sessions:
        print(f"Reading and enriching OpenF1 session {session}...", flush=True)
        session_rows, manifest = fetch_session(session, args.cache_dir)
        rows.extend(session_rows)
        manifests.append(manifest)
        print(f"  {manifest['rows_after_cleaning']:,} clean samples / {manifest['positive_after_cleaning']:,} clean positives; {manifest['car_data_records']:,} car and {manifest['location_records']:,} location records", flush=True)

    train_sessions, validation_session = set(sessions[:-1]), sessions[-1]
    train_rows = [row for row in rows if row["session_key"] in train_sessions]
    validation_rows = [row for row in rows if row["session_key"] == validation_session]
    x_train = pd.DataFrame([[row["features"][name] for name in FEATURES] for row in train_rows], columns=FEATURES)
    y_train = np.asarray([row["overtake_next_60s"] for row in train_rows], dtype=int)
    x_validation = pd.DataFrame([[row["features"][name] for name in FEATURES] for row in validation_rows], columns=FEATURES)
    y_validation = np.asarray([row["overtake_next_60s"] for row in validation_rows], dtype=int)
    if not y_train.any() or not y_validation.any():
        raise SystemExit("Both train and validation partitions need positive clean labels.")

    parameters = {
        "n_estimators": 1500, "max_depth": 4, "learning_rate": 0.025,
        "min_child_weight": 8, "subsample": 0.9, "colsample_bytree": 0.9,
        "reg_alpha": 0.05, "reg_lambda": 6.0, "max_delta_step": 1,
        "objective": "binary:logistic", "eval_metric": "auc", "early_stopping_rounds": 100,
        "random_state": 2026, "n_jobs": 4,
    }
    # Train a clean-label, base-feature control model first.  This is the fair
    # comparison for deciding whether car telemetry genuinely adds value.
    clean_base_model = XGBClassifier(**parameters)
    clean_base_model.fit(x_train[base.FEATURES], y_train, eval_set=[(x_validation[base.FEATURES], y_validation)], verbose=False)
    clean_base_trees = int(clean_base_model.best_iteration) + 1 if clean_base_model.best_iteration is not None else parameters["n_estimators"]
    clean_base_probability = clean_base_model.predict_proba(x_validation[base.FEATURES], iteration_range=(0, clean_base_trees))[:, 1]
    clean_baseline = {"features": base.FEATURES, "best_trees": clean_base_trees, "metrics": model_metrics(y_validation, clean_base_probability)}
    clean_base_threshold = f1_operating_threshold(y_validation, clean_base_probability)
    clean_base_booster = clean_base_model.get_booster()
    clean_base_tree_dump = [json.loads(tree) for tree in clean_base_booster.get_dump(dump_format="json")][:clean_base_trees]
    clean_base_config = json.loads(clean_base_booster.save_config())
    clean_base_probability_prior = float(clean_base_config["learner"]["learner_model_param"]["base_score"].strip("[]"))
    clean_base_margin = math.log(clean_base_probability_prior / (1 - clean_base_probability_prior))
    for values, expected in zip(x_validation[base.FEATURES].head(50).to_dict("records"), clean_base_probability[:50]):
        portable = sigmoid(clean_base_margin + sum(tree_margin(tree, values) for tree in clean_base_tree_dump))
        if abs(portable - float(expected)) > 1e-6:
            raise RuntimeError("Portable clean-label base XGBoost export did not match the validation prediction.")
    print(f"Clean-label base-feature validation ROC-AUC: {clean_baseline['metrics']['roc_auc']}", flush=True)

    model = XGBClassifier(**parameters)
    model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    best_trees = int(model.best_iteration) + 1 if model.best_iteration is not None else parameters["n_estimators"]
    probability = model.predict_proba(x_validation, iteration_range=(0, best_trees))[:, 1]
    metrics = model_metrics(y_validation, probability)
    booster = model.get_booster()
    trees = [json.loads(tree) for tree in booster.get_dump(dump_format="json")][:best_trees]
    config = json.loads(booster.save_config())
    base_probability = float(config["learner"]["learner_model_param"]["base_score"].strip("[]"))
    base_margin = math.log(base_probability / (1 - base_probability))
    for values, expected in zip(x_validation.head(50).to_dict("records"), probability[:50]):
        portable = sigmoid(base_margin + sum(tree_margin(tree, values) for tree in trees))
        if abs(portable - float(expected)) > 1e-6:
            raise RuntimeError("Portable XGBoost export did not match the validation prediction.")

    importance = booster.get_score(importance_type="gain")
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "data_source": "OpenF1 historical API",
        "target": "clean_overtake_next_60s", "horizon_seconds": HORIZON_SECONDS,
        "sample_interval_seconds": base.SAMPLE_SECONDS, "train_sessions": sorted(train_sessions), "validation_sessions": [validation_session],
        "train_rows": int(len(train_rows)), "train_positive": int(y_train.sum()), "train_positive_rate": round(float(y_train.mean()), 6),
        "algorithm": "XGBoost binary classifier with validation early stopping", "hyperparameters": parameters,
        "best_trees": best_trees, "metrics": metrics, "session_manifests": manifests,
        "feature_importance_gain": {name: round(float(importance.get(name, 0.0)), 6) for name in FEATURES},
        "data_lineage": {"features": {"base": ["intervals", "position", "laps", "stints", "weather"], "telemetry": ["car_data"], "audited_replay_source": ["location"], "label": ["overtakes"], "label_cleaning": ["pit", "race_control"]}},
        "limitations": ["Validation also controls early stopping; use a separate final race for final reporting.", "Location records are audited but excluded from features because their local coordinates are not portable.", "No private team ERS or battery telemetry is present."],
    }
    artifact = {"version": "2.0.0-xgboost-openf1-enriched", "model_type": "xgboost_binary_logistic", "features": FEATURES, "base_margin": base_margin, "trees": trees, "metadata": metadata}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    clean_base_artifact = {
        "version": "2.0.0-xgboost-openf1-clean-base", "model_type": "xgboost_binary_logistic",
        "features": base.FEATURES, "base_margin": clean_base_margin, "trees": clean_base_tree_dump,
        "metadata": {**metadata, "algorithm": "XGBoost binary classifier with validation early stopping (clean-label base-feature control)", "best_trees": clean_base_trees, "metrics": clean_baseline["metrics"], "classification_threshold": clean_base_threshold, "feature_importance_gain": {name: round(float(clean_base_booster.get_score(importance_type="gain").get(name, 0.0)), 6) for name in base.FEATURES}},
    }
    (MODELS / "xgboost_clean_label_base_model.json").write_text(json.dumps(clean_base_artifact, indent=2) + "\n", encoding="utf-8")

    simple_path = MODELS / "xgboost_overtake_model.json"
    existing = json.loads(simple_path.read_text(encoding="utf-8")) if simple_path.exists() else None
    baseline_auc = clean_baseline["metrics"]["roc_auc"]
    activate = args.force_activate or baseline_auc is None or metrics["roc_auc"] >= baseline_auc
    active = "Enriched XGBoost" if activate else "Clean-label base-feature XGBoost"
    selected_artifact = artifact if activate else clean_base_artifact
    (DIST / "apex-model.js").write_text(browser_source(selected_artifact), encoding="utf-8")
    comparison = {"generated_at": metadata["trained_at"], "active_model": active, "selection_metric": "validation_roc_auc_on_identical_clean_labels", "legacy_xgboost": existing, "clean_label_base_feature_xgboost": clean_baseline, "enriched_xgboost": artifact}
    (MODELS / "enriched_model_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    report(artifact, clean_baseline, existing, active)
    write_reports(MODELS)
    print(f"Saved enriched XGBoost artifact: {args.output}")
    print(f"Validation metrics: {json.dumps(metrics, sort_keys=True)}")
    print(f"Best trees after early stopping: {best_trees} / {parameters['n_estimators']}")
    print(f"Active browser model: {active}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
