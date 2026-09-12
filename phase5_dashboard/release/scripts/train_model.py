#!/usr/bin/env python3
"""Train the APEX-R overtake-opportunity model from historical OpenF1 data.

The model predicts whether a driver will complete an overtake in the next
60 seconds. It is deliberately a small, explainable logistic model. It does
not pretend that public data contains a team's private ERS/battery telemetry.

Examples:
    python3 scripts/train_model.py
    python3 scripts/train_model.py --sessions 7953,7779,7787,9070
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "openf1-cache"
MODEL_DIR = ROOT / "models"
DIST = ROOT / "dist"
API = "https://api.openf1.org/v1/"
HORIZON_SECONDS = 60.0
SAMPLE_SECONDS = 10.0
FEATURES = [
    "interval_sec",
    "gap_to_leader_sec",
    "closing_rate_sec_per_min",
    "position",
    "tyre_age",
    "track_temperature_c",
    "rainfall",
    "race_progress",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        default="7953,7779,7787,9070",
        help="Comma-separated OpenF1 race session keys; the last session is held out for testing.",
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "overtake_model.json")
    parser.add_argument("--min-samples", type=int, default=100)
    return parser.parse_args()


def api_get(endpoint: str, params: dict, cache_dir: Path) -> tuple[list, str]:
    query = urlencode(params)
    url = API + endpoint + "?" + query
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), url
    request = Request(url, headers={"User-Agent": "APEX-R-hackathon/1.0"})
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90) as response:
                payload = json.load(response)
            if not isinstance(payload, list):
                raise ValueError(f"Unexpected response from {endpoint}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            return payload, url
        except Exception as error:  # pragma: no cover - network-dependent retry path
            last_error = error
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def numeric(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def latest_value(rows_by_driver: dict, driver: int, t: float, field: str, default):
    rows = rows_by_driver.get(driver, [])
    if not rows:
        return default
    index = bisect.bisect_right([row[0] for row in rows], t) - 1
    if index < 0:
        return default
    return rows[index][1].get(field, default)


def prepare_lookup(rows: list, driver_field: str = "driver_number") -> dict[int, list[tuple[float, dict]]]:
    grouped: dict[int, list[tuple[float, dict]]] = {}
    for row in rows:
        driver = row.get(driver_field)
        t = timestamp(row.get("date") or row.get("date_start"))
        if driver is None or t is None:
            continue
        grouped.setdefault(int(driver), []).append((t, row))
    for values in grouped.values():
        values.sort(key=lambda pair: pair[0])
    return grouped


def session_rows(session_key: int, cache_dir: Path) -> tuple[list[dict], dict]:
    intervals, intervals_url = api_get("intervals", {"session_key": session_key}, cache_dir)
    positions, positions_url = api_get("position", {"session_key": session_key}, cache_dir)
    laps, laps_url = api_get("laps", {"session_key": session_key}, cache_dir)
    stints, stints_url = api_get("stints", {"session_key": session_key}, cache_dir)
    weather, weather_url = api_get("weather", {"session_key": session_key}, cache_dir)
    overtakes, overtakes_url = api_get("overtakes", {"session_key": session_key}, cache_dir)

    positions_by_driver = prepare_lookup(positions)
    laps_by_driver = prepare_lookup(laps)
    stints_by_driver: dict[int, list[dict]] = {}
    for row in stints:
        driver = row.get("driver_number")
        if driver is not None:
            stints_by_driver.setdefault(int(driver), []).append(row)
    for values in stints_by_driver.values():
        values.sort(key=lambda row: numeric(row.get("lap_start"), 0))
    weather_rows = []
    for row in weather:
        t = timestamp(row.get("date"))
        if t is not None:
            weather_rows.append((t, row))
    weather_rows.sort(key=lambda pair: pair[0])
    weather_times = [pair[0] for pair in weather_rows]

    events: dict[int, list[float]] = {}
    for row in overtakes:
        driver = row.get("overtaking_driver_number")
        t = timestamp(row.get("date"))
        if driver is not None and t is not None:
            events.setdefault(int(driver), []).append(t)
    for values in events.values():
        values.sort()

    # One row per recorded interval is already approximately a 4-second
    # sample. Downsampling to ten seconds reduces autocorrelation and keeps the
    # training set small enough to inspect and reproduce.
    interval_by_driver: dict[int, list[tuple[float, dict]]] = prepare_lookup(intervals)
    rows: list[dict] = []
    for driver, values in interval_by_driver.items():
        event_times = events.get(driver, [])
        previous_interval = None
        previous_time = None
        last_sample = -float("inf")
        for t, row in values:
            interval = numeric(row.get("interval"))
            leader_gap = numeric(row.get("gap_to_leader"))
            if interval is None or leader_gap is None or t - last_sample < SAMPLE_SECONDS:
                continue
            last_sample = t
            closing = 0.0
            if previous_interval is not None and previous_time is not None and t > previous_time:
                closing = (previous_interval - interval) / ((t - previous_time) / 60.0)
            previous_interval, previous_time = interval, t

            position = numeric(latest_value(positions_by_driver, driver, t, "position", 10), 10)
            lap_number = numeric(latest_value(laps_by_driver, driver, t, "lap_number", 1), 1)
            driver_stints = stints_by_driver.get(driver, [])
            current_stint = None
            for stint in driver_stints:
                if numeric(stint.get("lap_start"), 0) <= lap_number:
                    current_stint = stint
                else:
                    break
            tyre_age = numeric((current_stint or {}).get("tyre_age_at_start"), 0)
            stint_lap = numeric((current_stint or {}).get("lap_start"), lap_number)
            tyre_age = max(0.0, tyre_age + max(0.0, lap_number - stint_lap))

            weather = {"track_temperature": 30.0, "rainfall": 0}
            weather_index = bisect.bisect_right(weather_times, t) - 1
            if weather_index >= 0:
                weather = weather_rows[weather_index][1]
            race_start = values[0][0]
            race_end = values[-1][0]
            progress = 0.0 if race_end <= race_start else (t - race_start) / (race_end - race_start)
            event_index = bisect.bisect_right(event_times, t)
            label = int(event_index < len(event_times) and event_times[event_index] <= t + HORIZON_SECONDS)
            rows.append(
                {
                    "session_key": session_key,
                    "driver_number": driver,
                    "date": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                    "features": {
                        "interval_sec": interval,
                        "gap_to_leader_sec": leader_gap,
                        "closing_rate_sec_per_min": max(-30.0, min(30.0, closing)),
                        "position": max(1.0, min(20.0, position)),
                        "tyre_age": max(0.0, min(60.0, tyre_age)),
                        "track_temperature_c": numeric(weather.get("track_temperature"), 30.0),
                        "rainfall": float(bool(weather.get("rainfall"))),
                        "race_progress": max(0.0, min(1.0, progress)),
                    },
                    "overtake_next_60s": label,
                }
            )

    urls = {
        "intervals": intervals_url,
        "position": positions_url,
        "laps": laps_url,
        "stints": stints_url,
        "weather": weather_url,
        "overtakes": overtakes_url,
    }
    return rows, {"session_key": session_key, "rows": len(rows), "positive": sum(r["overtake_next_60s"] for r in rows), "urls": urls}


def safe_metric(function, y_true, y_score):
    try:
        return round(float(function(y_true, y_score)), 6)
    except ValueError:
        return None


def export_browser_model(model, scaler, metadata: dict, output: Path) -> None:
    coefficients = model.coef_[0] / scaler.scale_
    intercept = float(model.intercept_[0] - np.dot(model.coef_[0], scaler.mean_ / scaler.scale_))
    artifact = {
        "version": "1.0.0-real-openf1",
        "model_type": "standardized_logistic_regression",
        "features": FEATURES,
        "coefficients": [round(float(value), 10) for value in coefficients],
        "intercept": round(intercept, 10),
        "metadata": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    DIST.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, separators=(",", ":"))
    browser_js = f"""(function(root) {{
  'use strict';
  const artifact = {payload};
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const sigmoid = x => 1 / (1 + Math.exp(-clamp(x, -30, 30)));
  function predict(features) {{
    const values = artifact.features.map(name => Number.isFinite(Number(features[name])) ? Number(features[name]) : 0);
    const logit = artifact.intercept + artifact.coefficients.reduce((sum, coefficient, index) => sum + coefficient * values[index], 0);
    return {{ probability: sigmoid(logit), modelVersion: artifact.version, features: values }};
  }}
  const api = {{ metadata: artifact, predict }};
  root.ApexModel = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
}})(typeof globalThis !== 'undefined' ? globalThis : this);\n"""
    (DIST / "apex-model.js").write_text(browser_js, encoding="utf-8")


def main() -> int:
    args = parse_args()
    session_keys = [int(value.strip()) for value in args.sessions.split(",") if value.strip()]
    if len(session_keys) < 2:
        raise SystemExit("Provide at least two sessions so one can be held out for testing.")
    all_rows: list[dict] = []
    manifests = []
    for session_key in session_keys:
        print(f"Fetching/reading OpenF1 session {session_key}...", flush=True)
        rows, manifest = session_rows(session_key, args.cache_dir)
        if not rows:
            raise SystemExit(f"Session {session_key} produced no usable interval rows.")
        all_rows.extend(rows)
        manifests.append(manifest)
        print(f"  {manifest['rows']} samples / {manifest['positive']} positive labels", flush=True)

    train_sessions = set(session_keys[:-1])
    test_sessions = set(session_keys[-1:])
    train_rows = [row for row in all_rows if row["session_key"] in train_sessions]
    test_rows = [row for row in all_rows if row["session_key"] in test_sessions]
    if len(train_rows) < args.min_samples or len(test_rows) < args.min_samples:
        raise SystemExit("Not enough samples for a reliable train/test split.")
    if not any(row["overtake_next_60s"] for row in train_rows) or not any(row["overtake_next_60s"] for row in test_rows):
        raise SystemExit("Train and held-out sessions must each contain positive overtake labels.")

    X_train = np.array([[row["features"][name] for name in FEATURES] for row in train_rows], dtype=float)
    y_train = np.array([row["overtake_next_60s"] for row in train_rows], dtype=int)
    X_test = np.array([[row["features"][name] for name in FEATURES] for row in test_rows], dtype=float)
    y_test = np.array([row["overtake_next_60s"] for row in test_rows], dtype=int)

    # Keep the natural class balance so probabilities remain usable as a
    # prior. Ranking metrics are reported separately; a balanced classifier
    # would make the displayed probability look much larger than the observed
    # event rate.
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=2026))
    pipeline.fit(X_train, y_train)
    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["logisticregression"]
    test_probability = pipeline.predict_proba(X_test)[:, 1]
    predictions = (test_probability >= 0.5).astype(int)
    metrics = {
        "test_rows": int(len(test_rows)),
        "test_positive": int(y_test.sum()),
        "test_positive_rate": round(float(y_test.mean()), 6),
        "accuracy_at_0_5": round(float((predictions == y_test).mean()), 6),
        "roc_auc": safe_metric(roc_auc_score, y_test, test_probability),
        "average_precision": safe_metric(average_precision_score, y_test, test_probability),
        "brier_score": round(float(brier_score_loss(y_test, test_probability)), 6),
    }
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "OpenF1 historical API",
        "data_source_note": "OpenF1 is an unofficial public data service; imported data does not contain private ERS/battery telemetry.",
        "target": "overtake_next_60s",
        "horizon_seconds": HORIZON_SECONDS,
        "sample_interval_seconds": SAMPLE_SECONDS,
        "train_sessions": sorted(train_sessions),
        "test_sessions": sorted(test_sessions),
        "session_manifests": manifests,
        "train_rows": int(len(train_rows)),
        "train_positive": int(y_train.sum()),
        "train_positive_rate": round(float(y_train.mean()), 6),
        "algorithm": "Logistic Regression",
        "hyperparameters": {
            "solver": model.solver,
            "max_iter": int(model.max_iter),
            "actual_iterations": [int(value) for value in model.n_iter_],
            "C": float(model.C),
            "class_weight": model.class_weight,
            "random_state": int(model.random_state),
            "feature_scaling": "StandardScaler",
        },
        "metrics": metrics,
        "limitations": [
            "Predicts observed overtake events, not private team strategy or ERS state.",
            "Held-out performance is evidence for these sessions, not proof of race-wide generalisation.",
            "The optimiser remains constraint-aware and does not hand control to the model.",
        ],
    }
    export_browser_model(model, scaler, metadata, args.output)
    report = args.output.with_name("training_report.json")
    report.write_text(json.dumps({"model": str(args.output), "metadata": metadata}, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved model: {args.output}")
    print(f"Saved browser model: {DIST / 'apex-model.js'}")
    print(f"Held-out metrics: {json.dumps(metrics, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Training interrupted.", file=sys.stderr)
        raise SystemExit(130)
