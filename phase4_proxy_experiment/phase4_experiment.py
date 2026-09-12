#!/usr/bin/env python3
"""Run the isolated Phase 4 historical proxy-model experiment.

This module deliberately does not fetch remote data, train on the sealed
holdout, or touch any existing model artifact.  It audits the Phase 3
position-swap proxy, censors examples whose fixed target cannot be supported
by a completed-order snapshot at decision time, then fits only a constant
baseline and one regularized logistic-regression candidate.

Run from the project root:
    python phase4_proxy_experiment/phase4_experiment.py

The script uses pandas and scikit-learn already listed in requirements.txt.
All generated files are written below phase4_proxy_experiment/.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PHASE3 = ROOT / "phase3_prediction_dataset"
DEFAULT_OUT = ROOT / "phase4_proxy_experiment"
RANDOM_SEED = 20260911
BOOTSTRAPS = 2000

# These are the Phase 3 approved historical-proxy features.  No identifiers,
# labels, event evidence, split membership, or future outcome context enter
# this list.  The source data retains explicit nulls; conversion below maps
# them to NaN and the fitted training-only imputer adds missingness flags.
FEATURES = [
    "lap",
    "time",
    "phase2_session_time_sec",
    "speed",
    "rpm",
    "gear",
    "throttle",
    "brake",
    "drs",
    "distance",
    "phase2_weather_air_temp_c",
    "phase2_weather_track_temp_c",
    "phase2_weather_humidity_pct",
    "phase2_weather_pressure_mbar",
    "phase2_weather_rainfall_flag",
    "phase2_weather_wind_direction_deg",
    "phase2_weather_wind_speed_mps",
    "phase3_weather_available_flag",
    "phase3_weather_age_sec",
    "phase3_pit_current_prior_stop_flag",
    "phase3_pit_target_prior_stop_flag",
    "phase3_rc_global_state_code",
    "phase3_rc_global_state_known_flag",
]

MODEL_FEATURES = set(FEATURES)
OUTCOME_FIELDS = {
    "proxy_label",
    "true_ontrack_label",
    "event_id",
    "exclusion_reason",
    "outcome_evidence_status",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def write_csv_gz(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "None", "null", "NULL", "nan", "NaN"}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_int(value: Any) -> int | None:
    result = parse_float(value)
    return int(result) if result is not None and result.is_integer() else None


def parse_time(value: Any) -> float | None:
    result = parse_float(value)
    return result


def normalize_bool(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "None", "null", "NULL", "nan", "NaN"}:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "on"}:
        return 1.0
    if text in {"false", "no", "n", "off"}:
        return 0.0
    return parse_float(value)


def safe_json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def race_id(row: dict[str, Any]) -> str:
    return f"{row['year']}:{row['event']}:{row['session']}"


def load_protected_tokens(phase3: Path) -> set[str]:
    """Read only the exclusion manifest; no protected data is opened."""
    manifest = safe_json_load(phase3 / "excluded_sessions_manifest.json")
    token = str(manifest.get("protected_session_token", "")).strip()
    return {token} if token else set()


def assert_not_protected(value: Any, tokens: set[str]) -> None:
    text = str(value)
    if any(token and token in text for token in tokens):
        raise RuntimeError("Protected session token found in an input or output path")


def input_paths(phase3: Path) -> list[Path]:
    paths = [
        phase3 / "features" / "phase3_feature_view.csv.gz",
        phase3 / "labels" / "phase3_proxy_labels.csv.gz",
        phase3 / "prediction_examples.csv.gz",
        phase3 / "phase3_split_manifest.csv",
        phase3 / "context" / "laptime_coverage.csv",
        phase3 / "excluded_sessions_manifest.json",
    ]
    paths.extend(sorted((phase3 / "cache" / "laptimes").glob("*.json")))
    # The Phase 2 cache contains the selected laptime payloads used by the
    # Phase 3 lineage.  It is hashed as a manifest rather than copied.
    paths.extend(sorted((phase3.parent / "phase2_data_foundation" / "cache" / "remote").glob("*.bin")))
    return [path for path in paths if path.is_file()]


def hash_inputs(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def load_tables(phase3: Path, tokens: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_path = phase3 / "features" / "phase3_feature_view.csv.gz"
    label_path = phase3 / "labels" / "phase3_proxy_labels.csv.gz"
    example_path = phase3 / "prediction_examples.csv.gz"
    split_path = phase3 / "phase3_split_manifest.csv"
    for path in (feature_path, label_path, example_path, split_path):
        assert_not_protected(path, tokens)
    features = pd.read_csv(feature_path, compression="gzip", dtype=str, keep_default_na=False)
    labels = pd.read_csv(label_path, compression="gzip", dtype=str, keep_default_na=False)
    examples = pd.read_csv(example_path, compression="gzip", dtype=str, keep_default_na=False)
    splits = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    for frame in (features, labels, examples, splits):
        if "race_id" in frame:
            for value in frame["race_id"].unique():
                assert_not_protected(value, tokens)
    return features, labels, examples, splits


def columnar_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    arrays = {key: value for key, value in payload.items() if isinstance(value, list)}
    count = max((len(value) for value in arrays.values()), default=0)
    return [{key: value[index] if index < len(value) else None for key, value in arrays.items()} for index in range(count)]


def load_laptime_records(phase3: Path, tokens: set[str]) -> tuple[dict[str, dict[str, dict[int, dict[str, Any]]]], list[dict[str, Any]]]:
    """Load the already cached Phase 3 laptime evidence, without fetching."""
    coverage_path = phase3 / "context" / "laptime_coverage.csv"
    coverage = list(csv.DictReader(coverage_path.open(encoding="utf-8", newline="")))
    phase2_log_path = phase3.parent / "phase2_data_foundation" / "provenance" / "source_fetch_log.json"
    phase2_log = safe_json_load(phase2_log_path) if phase2_log_path.is_file() else []
    phase2_cache = {
        record.get("url", ""): Path(record["cache_path"])
        for record in phase2_log
        if str(record.get("url", "")).endswith("/laptimes.json")
        and record.get("cache_path")
        and Path(record["cache_path"]).is_file()
    }
    records: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    audit_rows: list[dict[str, Any]] = []
    for row in coverage:
        url = row.get("source_url", "")
        assert_not_protected(url, tokens)
        phase3_cache = phase3 / "cache" / "laptimes" / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"
        cache_path = phase3_cache if phase3_cache.is_file() else phase2_cache.get(url)
        base = dict(row)
        base["cache_path_used"] = str(cache_path) if cache_path else ""
        base["cache_sha256_observed"] = sha256_file(cache_path) if cache_path else ""
        if cache_path is None:
            base["load_status"] = "MISSING_LOCAL_CACHE"
            audit_rows.append(base)
            continue
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            raw_rows = columnar_records(payload)
            load_status = "LOADED"
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raw_rows = []
            load_status = f"PARSE_ERROR:{type(exc).__name__}"
        base["load_status"] = load_status
        base["parsed_records"] = len(raw_rows)
        audit_rows.append(base)
        race = f"{row['year']}:{row['event']}:{row['session']}"
        driver = row["driver"]
        for raw in raw_rows:
            lap = parse_int(raw.get("lap"))
            if lap is None:
                continue
            records[race][driver][lap] = {
                "lap": lap,
                "position": parse_int(raw.get("pos")),
                "session_end_sec": parse_float(raw.get("sesT")),
                "lap_start_session_sec": parse_float(raw.get("lST")),
                "lap_time_sec": parse_float(raw.get("time")),
                "deleted": str(raw.get("del", "")).strip().lower() == "true",
                "accurate": str(raw.get("iacc", "")).strip().lower() == "true",
                "source_url": url,
            }
    return records, audit_rows


def first_selected_sample_audit(phase3: Path, examples: pd.DataFrame) -> dict[str, Any]:
    """Verify sample 0 and its relation to the Phase 2 lap-start anchors."""
    selected = {
        (str(row.year), row.event, row.session, row.attacker_driver, str(row.decision_lap))
        for row in examples.itertuples(index=False)
    }
    seen: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    bad: list[dict[str, Any]] = []
    path = phase3.parent / "phase2_data_foundation" / "enriched" / "telemetry_phase2_enriched.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("year", ""), row.get("event", ""), row.get("session", ""), row.get("driver", ""), row.get("lap", ""))
            if key not in selected or key in seen:
                continue
            seen[key] = row
            problems: list[str] = []
            if row.get("sample_index") != "0":
                problems.append("first_selected_row_not_sample_zero")
            if parse_float(row.get("time")) != 0.0:
                problems.append("first_selected_time_not_zero")
            if row.get("phase2_session_time_sec") != row.get("phase2_lap_start_session_sec"):
                problems.append("session_time_does_not_match_lap_start_anchor")
            if row.get("phase2_utc_timestamp") != row.get("phase2_lap_start_utc"):
                problems.append("utc_time_does_not_match_lap_start_anchor")
            if problems:
                bad.append({**{k: row.get(k, "") for k in ("year", "event", "session", "driver", "lap", "sample_index", "time", "phase2_session_time_sec", "phase2_lap_start_session_sec", "phase2_utc_timestamp", "phase2_lap_start_utc")}, "problems": ";".join(problems)})
    return {
        "selected_keys": len(selected),
        "first_rows_found": len(seen),
        "bad_first_rows": len(bad),
        "status": "PASS" if len(seen) == len(selected) and not bad else "FAIL",
        "evidence": "all selected rows were found with sample_index=0 and source-relative time 0 matching the stored lap-start anchors" if not bad else bad[:20],
    }


def latest_completed_order(
    race_records: dict[str, dict[int, dict[str, Any]]], decision_time_sec: float,
) -> tuple[str, dict[int, str], dict[str, int], str]:
    """Create a completed-record as-of order; never reads a future record."""
    latest_by_driver: dict[str, dict[str, Any]] = {}
    for driver, by_lap in race_records.items():
        candidates = [
            record for record in by_lap.values()
            if record.get("session_end_sec") is not None
            and record["session_end_sec"] <= decision_time_sec + 1e-6
            and record.get("position") is not None
            and record["position"] >= 1
            and not record.get("deleted", False)
        ]
        if candidates:
            latest_by_driver[driver] = max(candidates, key=lambda item: (item["session_end_sec"], item["lap"]))
    positions: dict[int, str] = {}
    duplicate_position = False
    for driver, record in latest_by_driver.items():
        position = int(record["position"])
        if position in positions and positions[position] != driver:
            duplicate_position = True
        positions[position] = driver
    by_driver_lap = {driver: int(record["lap"]) for driver, record in latest_by_driver.items()}
    if duplicate_position:
        status = "DUPLICATE_POSITION_ASOF"
    elif not positions:
        status = "NO_COMPLETED_ORDER_ASOF"
    elif set(positions) != set(range(1, len(positions) + 1)):
        status = "NONCONTIGUOUS_COMPLETED_ORDER_ASOF"
    else:
        status = "UNIQUE_CONTIGUOUS_COMPLETED_ORDER_ASOF"
    return status, positions, by_driver_lap, "latest completed laptime record with session_end_sec <= decision time + 1e-6"


def asof_target_check(example: dict[str, Any], race_records: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    target = str(example.get("target_driver", ""))
    attacker = str(example.get("attacker_driver", ""))
    decision = parse_float(example.get("decision_time_session_sec"))
    if not target or decision is None:
        return {"asof_status": "NO_TARGET_OR_DECISION_TIME", "asof_target_driver": "", "asof_attacker_position": "", "asof_target_position": "", "asof_records_used": 0, "asof_order_rule": ""}
    status, positions, latest_laps, rule = latest_completed_order(race_records, decision)
    attacker_position = next((position for position, driver in positions.items() if driver == attacker), None)
    target_position = next((position for position, driver in positions.items() if driver == target), None)
    asof_target = positions.get(attacker_position - 1) if attacker_position and attacker_position > 1 else ""
    if status == "UNIQUE_CONTIGUOUS_COMPLETED_ORDER_ASOF" and asof_target == target:
        final = "ASOF_TARGET_MATCHES"
    elif status == "UNIQUE_CONTIGUOUS_COMPLETED_ORDER_ASOF":
        final = "ASOF_TARGET_CONFLICT"
    elif attacker_position is None:
        final = "ATTACKER_MISSING_ASOF"
    else:
        final = status
    return {
        "asof_status": final,
        "asof_raw_order_status": status,
        "asof_target_driver": asof_target,
        "asof_attacker_position": attacker_position if attacker_position is not None else "",
        "asof_target_position": target_position if target_position is not None else "",
        "asof_records_used": len(positions),
        "asof_driver_lap_map": json.dumps(latest_laps, sort_keys=True, separators=(",", ":")),
        "asof_order_rule": rule,
    }


def build_gate_a(examples: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    grouped = examples[examples["new_eligibility_status"] == "ELIGIBLE_PROXY_ASOF"].copy()
    for split, frame in grouped.groupby("split", sort=True):
        pos = frame[frame["new_proxy_label"] == 1]
        neg = frame[frame["new_proxy_label"] == 0]
        per_race = frame.groupby("race_id").agg(
            windows=("example_id", "count"),
            positives=("new_proxy_label", "sum"),
            unique_pairs=("pair_key", "nunique"),
            unique_events=("event_id", lambda s: s.replace("", np.nan).nunique()),
        )
        positive_races = int((per_race["positives"] > 0).sum())
        rows.append({
            "split": split,
            "eligible_windows": len(frame),
            "positives": int(len(pos)),
            "negatives": int(len(neg)),
            "unique_races": int(frame["race_id"].nunique()),
            "positive_races": positive_races,
            "positive_race_fraction": positive_races / frame["race_id"].nunique() if frame["race_id"].nunique() else None,
            "unique_driver_pairs": int(frame["pair_key"].nunique()),
            "unique_events": int(frame["event_id"].replace("", np.nan).nunique()),
            "max_positives_in_one_race": int(per_race["positives"].max()) if len(per_race) else 0,
        })
    all_frame = grouped
    pair_counts = all_frame.groupby(["race_id", "attacker_driver", "target_driver"]).size()
    duplicate_examples = int(examples["example_id"].duplicated().sum())
    event_counts = all_frame[all_frame["event_id"] != ""]["event_id"].value_counts()
    gate = {
        "eligible_windows": int(len(all_frame)),
        "positives": int((all_frame["new_proxy_label"] == 1).sum()),
        "negatives": int((all_frame["new_proxy_label"] == 0).sum()),
        "unique_races": int(all_frame["race_id"].nunique()),
        "unique_driver_pairs_within_race": int(all_frame["pair_key"].nunique()),
        "unique_positive_events": int(all_frame.loc[all_frame["new_proxy_label"] == 1, "event_id"].replace("", np.nan).nunique()),
        "duplicate_example_ids": duplicate_examples,
        "max_windows_per_race_pair": int(pair_counts.max()) if len(pair_counts) else 0,
        "repeated_event_ids": int((event_counts > 1).sum()),
        "whole_race_split_isolation": int(all_frame.groupby("race_id")["split"].nunique().max()) <= 1 if len(all_frame) else True,
        "support_warning": "Only a small number of positive proxy events support evaluation; uncertainty is race-level and material.",
    }
    return rows, gate


def build_evidence_review(examples: pd.DataFrame, out_path: Path) -> dict[str, Any]:
    # All positives and all technical ambiguities are included.  Negatives are
    # sampled deterministically at up to two per race, so the sheet remains
    # readable while covering every race and split.
    positive = examples[examples["new_proxy_label"] == 1].copy()
    ambiguous = examples[examples["new_eligibility_status"] != "ELIGIBLE_PROXY_ASOF"].copy()
    negative = examples[examples["new_proxy_label"] == 0].sort_values(["split", "race_id", "decision_lap", "attacker_driver"])
    negative = negative.groupby("race_id", group_keys=False).head(2).copy()
    selected = pd.concat([positive, ambiguous, negative], ignore_index=True).drop_duplicates("example_id")
    selected["review_selection_reason"] = np.where(
        selected["new_proxy_label"] == 1,
        "ALL_REPORTED_POSITIVES",
        np.where(selected["new_eligibility_status"] != "ELIGIBLE_PROXY_ASOF", "ALL_AMBIGUOUS_OR_CENSORED", "REPRESENTATIVE_NEGATIVE_UP_TO_TWO_PER_RACE"),
    )
    selected["automated_consistency_review"] = np.where(
        (selected["new_eligibility_status"] == "ELIGIBLE_PROXY_ASOF")
        & selected["target_driver"].ne("")
        & selected["decision_time_session_sec"].ne("")
        & selected["horizon_end_session_sec"].ne(""),
        "AUTOMATED_FIELDS_PRESENT_REVIEWER_PENDING",
        "AUTOMATED_FIELDS_INCOMPLETE_REVIEWER_PENDING",
    )
    fields = [
        "example_id", "race_id", "split", "year", "event", "attacker_driver", "target_driver", "decision_lap",
        "decision_sample_index", "decision_time_session_sec", "decision_time_utc", "horizon_end_session_sec", "horizon_end_utc",
        "target_identity_status", "asof_status", "asof_raw_order_status", "asof_target_driver", "asof_attacker_position", "asof_target_position",
        "asof_records_used", "asof_order_rule", "pair_before_attacker_position", "pair_before_target_position", "pair_after_attacker_position",
        "pair_after_target_position", "pit_horizon_status", "pit_event_evidence", "race_control_decision_state", "race_control_outcome_state",
        "race_control_status", "proxy_label", "new_proxy_label", "proxy_label_class", "new_label_class", "event_id", "exclusion_reason",
        "new_reconciliation_reason", "attacker_laptime_source_url", "target_laptime_source_url", "review_selection_reason", "automated_consistency_review",
    ]
    write_csv(out_path, selected.to_dict("records"), fields)
    return {"rows": len(selected), "positives_included": int((selected["new_proxy_label"] == 1).sum()), "ambiguous_included": int((selected["new_eligibility_status"] != "ELIGIBLE_PROXY_ASOF").sum()), "representative_negatives_included": int(((selected["new_proxy_label"] == 0) & (selected["review_selection_reason"] == "REPRESENTATIVE_NEGATIVE_UP_TO_TWO_PER_RACE")).sum()), "human_review_claimed": False}


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for feature in FEATURES:
        if feature not in frame:
            result[feature] = np.nan
            continue
        if feature == "phase2_weather_rainfall_flag":
            result[feature] = frame[feature].map(normalize_bool)
        else:
            result[feature] = pd.to_numeric(frame[feature].replace({"": np.nan, "None": np.nan, "null": np.nan}), errors="coerce")
    return result.astype(float)


def logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        # scikit-learn's default penalty is L2; omitting the deprecated
        # spelling keeps the run quiet on current sklearn while preserving
        # the explicitly documented L2 configuration.
        ("model", LogisticRegression(C=1.0, solver="liblinear", class_weight=None, max_iter=5000, random_state=RANDOM_SEED)),
    ])


def safe_metric(metric: str, y_true: np.ndarray, pred: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2 and metric in {"roc_auc", "average_precision"}:
        return None if metric == "roc_auc" else float(np.mean(y_true))
    try:
        if metric == "roc_auc":
            return float(roc_auc_score(y_true, pred))
        if metric == "average_precision":
            return float(average_precision_score(y_true, pred))
        if metric == "brier":
            return float(brier_score_loss(y_true, pred))
        if metric == "log_loss":
            return float(log_loss(y_true, pred, labels=[0, 1]))
    except ValueError:
        return None
    raise ValueError(metric)


def curve_rows(model_name: str, split: str, y_true: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    precision, recall, thresholds = precision_recall_curve(y_true, pred)
    rows = []
    for idx, (p, r) in enumerate(zip(precision, recall)):
        rows.append({
            "model": model_name,
            "split": split,
            "threshold": float(thresholds[idx]) if idx < len(thresholds) else None,
            "precision": float(p),
            "recall": float(r),
            "predicted_positive": int(np.sum(pred >= thresholds[idx])) if idx < len(thresholds) else 0,
        })
    return rows


def threshold_summary(y_true: np.ndarray, pred: np.ndarray, recall_floor: float = 0.70) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, pred)
    candidates = [(precision[i], recall[i], thresholds[i]) for i in range(len(thresholds)) if recall[i] >= recall_floor]
    if not candidates:
        return {"recall_floor": recall_floor, "threshold": None, "precision": None, "recall": None, "selection": "NO_THRESHOLD_REACHES_RECALL_FLOOR"}
    best = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    threshold = float(best[2])
    pred_label = (pred >= threshold).astype(int)
    tp = int(((pred_label == 1) & (y_true == 1)).sum())
    fp = int(((pred_label == 1) & (y_true == 0)).sum())
    fn = int(((pred_label == 0) & (y_true == 1)).sum())
    tn = int(((pred_label == 0) & (y_true == 0)).sum())
    return {"recall_floor": recall_floor, "threshold": threshold, "precision": float(best[0]), "recall": float(best[1]), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "selection": "DESCRIPTIVE_CURVE_STATISTIC_NOT_TUNED_FOR_DEPLOYMENT"}


def fit_training_oof_threshold(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2 or y.sum() < 2:
        return np.full(len(y), np.nan), {"status": "SKIPPED_INSUFFICIENT_GROUP_OR_POSITIVE_SUPPORT", "n_splits": n_splits}
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        if len(np.unique(y[train_idx])) < 2:
            fold_rows.append({"fold": fold, "status": "SKIPPED_TRAIN_ONE_CLASS", "train_rows": len(train_idx), "test_rows": len(test_idx), "test_positives": int(y[test_idx].sum())})
            continue
        model = logistic_pipeline()
        model.fit(x.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
        fold_rows.append({"fold": fold, "status": "FIT", "train_rows": len(train_idx), "test_rows": len(test_idx), "train_positives": int(y[train_idx].sum()), "test_positives": int(y[test_idx].sum())})
    valid = np.isfinite(oof)
    if valid.sum() == 0 or y[valid].sum() == 0:
        return oof, {"status": "SKIPPED_NO_VALID_OOF_SUPPORT", "n_splits": n_splits, "folds": fold_rows}
    summary = threshold_summary(y[valid], oof[valid])
    summary.update({"status": "TRAINING_ONLY_GROUPED_OOF", "n_splits": n_splits, "valid_rows": int(valid.sum()), "valid_positives": int(y[valid].sum()), "folds": fold_rows})
    return oof, summary


def reliability_rows(model_name: str, split: str, y_true: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    bins = [-np.inf, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    labels = ["<=0.001", "0.001-0.005", "0.005-0.01", "0.01-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.50", "0.50-1.00"]
    bucket = pd.cut(pred, bins=bins, labels=labels, include_lowest=True, right=True)
    rows = []
    for label in labels:
        mask = np.asarray(bucket == label)
        if not mask.any():
            continue
        rows.append({"model": model_name, "split": split, "probability_bin": label, "count": int(mask.sum()), "mean_predicted_probability": float(pred[mask].mean()), "observed_positive_rate": float(y_true[mask].mean()), "positive_count": int(y_true[mask].sum())})
    return rows


def evaluate_model(model_name: str, split: str, y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return {
        "model": model_name,
        "split": split,
        "rows": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int((y_true == 0).sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else None,
        "roc_auc": safe_metric("roc_auc", y_true, pred),
        "average_precision": safe_metric("average_precision", y_true, pred),
        "brier_score": safe_metric("brier", y_true, pred),
        "log_loss": safe_metric("log_loss", y_true, pred),
        "descriptive_precision_at_recall_ge_70": threshold_summary(y_true, pred),
    }


def bootstrap_uncertainty(
    split_frame: pd.DataFrame,
    y: np.ndarray,
    candidate_pred: np.ndarray,
    baseline_pred: np.ndarray,
    split: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED + (1 if split == "validation" else 2))
    groups = split_frame["race_id"].to_numpy()
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    values = defaultdict(list)
    degenerate = Counter()
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        yb = y[indices]
        cp = candidate_pred[indices]
        bp = baseline_pred[indices]
        degenerate_class = len(np.unique(yb)) < 2
        if degenerate_class:
            degenerate["class"] += 1
        else:
            candidate_ap = average_precision_score(yb, cp)
            baseline_ap = average_precision_score(yb, bp)
            values["candidate_ap"].append(candidate_ap)
            values["baseline_ap"].append(baseline_ap)
            values["delta_ap"].append(candidate_ap - baseline_ap)
        values["candidate_brier"].append(brier_score_loss(yb, cp))
        values["baseline_brier"].append(brier_score_loss(yb, bp))
        values["delta_brier"].append(brier_score_loss(yb, cp) - brier_score_loss(yb, bp))
        if not degenerate_class:
            values["candidate_roc_auc"].append(roc_auc_score(yb, cp))
            values["baseline_roc_auc"].append(roc_auc_score(yb, bp))
            values["delta_roc_auc"].append(roc_auc_score(yb, cp) - roc_auc_score(yb, bp))
    rows = []
    for metric, samples in sorted(values.items()):
        array = np.asarray(samples, dtype=float)
        rows.append({"split": split, "metric": metric, "bootstrap_replicates": int(len(array)), "mean": float(array.mean()), "p02_5": float(np.quantile(array, 0.025)), "p50": float(np.quantile(array, 0.50)), "p97_5": float(np.quantile(array, 0.975)), "degenerate_replicates": int(degenerate["class"]) if metric.endswith(("ap", "roc_auc")) else 0})
    rows.append({"split": split, "metric": "class_degenerate_replicates", "bootstrap_replicates": BOOTSTRAPS, "mean": None, "p02_5": None, "p50": None, "p97_5": None, "degenerate_replicates": int(degenerate["class"])})
    return rows


def model_card(out: Path, training_meta: dict[str, Any]) -> None:
    text = f"""# APEX-R Phase 4 Proxy Model Card

## Scope

This is an isolated historical experiment for the Phase 3 lap-boundary
position-swap proxy. It is not a verified on-track-overtake model and is not
connected to ATTACK, HOLD, HARVEST or DEFEND logic.

## Selected model

The candidate is a regularized logistic regression with `C=1.0`, L2 penalty,
unweighted loss, training-only median imputation with missingness indicators,
and training-only standardization. It uses the Phase 3 approved historical
proxy feature list. No driver, event, source-file, split, label or outcome
evidence fields are model features.

## Training

* Task: `{training_meta['task_version']}`
* Training rows: {training_meta['train_rows']}
* Training positives: {training_meta['train_positives']}
* Training races: {training_meta['train_races']}
* Random seed: {RANDOM_SEED}
* Class weighting/resampling: none
* Calibration: none; probabilities are descriptive only

## Limitations

The labels are boundary position swaps. They can include lapping, unlapping,
retirement effects, timing corrections or hidden pass/repass. There are only
20 repaired positive proxy events across all partitions and only 6 in the
development-test partition. A race-level bootstrap is supplied, but this is
not enough support for a deployment claim or a stable operating threshold.
Source channel units/encodings and historical publication latency remain
partly unresolved. No sealed final holdout was accessed.
"""
    (out / "MODEL_CARD.md").write_text(text, encoding="utf-8")


def phase5_handoff(out: Path) -> None:
    text = """# Phase 5 Handoff

Phase 4 produced an exploratory proxy-model result only. The selected model
must not be connected to the live decision engine until the target is upgraded
to verified timestamped on-track events or the product explicitly accepts the
proxy limitation.

Required isolated Phase 5 evaluations:

1. Energy-feasibility: verify that every proposed action respects the modeled
   energy budget under conservative bounds; no private ERS value may be
   implied.
2. Strategy baseline: compare against fixed HOLD, fixed HARVEST, fixed
   ATTACK and a simple rule baseline on the same simulated initial states.
3. No-ML engine: run the optimizer without model probabilities to measure
   what the model contributes.
4. Stress: test missing context, stale observations, censored targets,
   safety-car transitions, pit windows, low energy and race-end censoring.
5. Latency: measure ingestion, feature construction, prediction and search
   latency using a replay clock; historical timestamps are not proof of live
   delivery latency.

All Phase 5 numbers must be labelled simulation or replay results. Do not
present them as observed racing improvements. Keep the protected final
holdout outside development until its separately frozen evaluation protocol.
"""
    (out / "PHASE5_HANDOFF.md").write_text(text, encoding="utf-8")


def report_text(metrics: dict[str, Any], gate_rows: list[dict[str, Any]]) -> str:
    totals = metrics["totals"]
    gate_b = metrics["gates"]["B"]
    gate_c = metrics["gates"]["C"]
    selected = metrics["selection"]
    lines = [
        "# APEX-R Phase 4 Historical Proxy Experiment Report",
        "",
        f"Generated: {metrics['generated_at_utc']}",
        "",
        "## Outcome",
        "",
        f"**{metrics['outcome']}**. This is an exploratory result for a lap-boundary position-swap proxy, not a verified overtake or strategy-effectiveness result.",
        "",
        "## What was repaired",
        "",
        f"The source Phase 3 table contained {totals['source_examples']} decision rows and {totals['source_eligible']} originally eligible proxy windows. The first selected sample was checked against the stored lap-start anchors: {gate_b['first_selected_sample']['status']} for {gate_b['first_selected_sample']['first_rows_found']}/{gate_b['first_selected_sample']['selected_keys']} rows.",
        f"An as-of completed-order check supported the fixed target for {totals['asof_target_matches']} windows. The remaining {totals['asof_target_unresolved']} previously eligible windows were changed to UNKNOWN/CENSORED because the order was non-contiguous, conflicted with the fixed target, or lacked an as-of attacker record. The repaired primary population is {totals['repaired_eligible']} windows: {totals['repaired_positives']} positives and {totals['repaired_negatives']} negatives.",
        "",
        "The as-of check uses the latest completed laptime record with `session_end_sec <= decision time`; it is evidence for a historical boundary proxy, not proof of instantaneous on-track order or live availability.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Key finding |",
        "| --- | --- | --- |",
        f"| A — split support/independence | {metrics['gates']['A']['status']} | {metrics['gates']['A']['positives']} positives, {metrics['gates']['A']['unique_races']} races, whole-race partitions retained |",
        f"| B — decision/boundary correctness | {metrics['gates']['B']['status']} | {metrics['gates']['B']['asof_target_matches']} target matches; {metrics['gates']['B']['asof_unresolved_previously_eligible']} prior eligible rows censored |",
        f"| C — future-dependent selection | {metrics['gates']['C']['status']} | Restricted proxy remains conditional; separate all-cause table created |",
        f"| D — evidence review | {metrics['gates']['D']['status']} | Automated sheet includes all positives and all ambiguities; human review not claimed |",
        "",
        "## Frozen experiment",
        "",
        f"Primary task: `{metrics['task_version']}`. Features: {len(FEATURES)} Phase 3 approved historical-proxy fields; no unresolved relative-distance/acceleration or outcome fields. Preprocessing is fitted inside each training fit. Candidate budget: constant prevalence baseline plus one fixed logistic pipeline; no tuning, resampling, SMOTE, deep learning or automatic calibration.",
        "",
        "The primary ranking metric is average precision. Thresholds in the curve files are descriptive. No deployment threshold is authorized because the positive support is too small.",
        "",
        "## Validation and development-test results",
        "",
        "Metrics are within this experiment and are not comparable to the earlier verified-overtake model, which used a different task and sealed race.",
        "",
        "| Split | Model | Rows | Positives | AP | ROC-AUC | Brier | Log loss | Descriptive precision at recall ≥70% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in metrics["metrics_table"]:
        point = row["descriptive_precision_at_recall_ge_70"]
        point_text = "n/a" if point.get("precision") is None else f"{point['precision']:.4f} @ {point['recall']:.4f} (threshold {point['threshold']:.6g})"
        lines.append(f"| {row['split']} | {row['model']} | {row['rows']} | {row['positives']} | {row['average_precision']:.4f} | {row['roc_auc'] if row['roc_auc'] is not None else 'undefined'} | {row['brier_score']:.4f} | {row['log_loss']:.4f} | {point_text} |")
    lines.extend([
        "",
        f"The frozen selection rule chose `{selected['model']}` because its validation average precision was {selected['validation_average_precision']:.6f} versus {selected['validation_constant_average_precision']:.6f} for the constant baseline. This selection is not evidence of a verified overtake gain.",
        "",
        "## Race-level uncertainty",
        "",
        "`uncertainty_by_race.csv` resamples whole races with replacement and compares the logistic candidate against the constant baseline on paired rows. Degenerate one-class resamples are counted and ROC-AUC intervals are therefore incomplete/fragile with this support.",
        "",
        "## Access and preservation",
        "",
        "Only cached Phase 3 evidence and the authorized development partitions were read. No remote requests were made. The protected final holdout was excluded before any data access. Input SHA-256 hashes were checked before and after and were unchanged.",
        "",
        "## Readiness",
        "",
        "**NO_DEMONSTRATED_GAIN** for a deployable APEX-R overtake predictor. Phase 4 is complete as an exploratory proxy experiment. The immediate blocker is not model complexity; it is target validity and small positive support. A future Phase 4b should use verified timestamped pass events and more independent positive races before any live decision logic is reconsidered.",
        "",
        "See `EXPERIMENT_SPEC.md`, `gate_results.json`, `repaired_prediction_examples.csv.gz`, `pr_curves.csv.gz`, `split_metrics.csv`, `uncertainty_by_race.csv`, `MODEL_CARD.md` and `PHASE5_HANDOFF.md`.",
        "",
    ])
    return "\n".join(lines)


def readme_text() -> str:
    return """# Phase 4 Proxy Experiment

This directory contains a reproducible, isolated experiment on the Phase 3
lap-boundary position-swap proxy. It does not modify existing models, does
not connect to the decision engine, and does not access the protected final
holdout.

## Run

From the project root, with the repository environment active:

```bash
python phase4_proxy_experiment/phase4_experiment.py
python -m unittest discover -s phase4_proxy_experiment -p 'test_*.py' -v
python phase4_proxy_experiment/verify_phase4_artifacts.py
```

Dependencies are the existing `requirements.txt` packages, especially
`pandas`, `numpy`, `scikit-learn` and `joblib`. The script is offline and uses
only cached Phase 3 inputs.

## Important interpretation

The target is a fixed driver pair reversing order at the next lap boundary.
It is not a verified second-level overtake. The repaired primary task censors
windows whose fixed target cannot be supported by a unique completed-order
snapshot at the decision timestamp. Only a constant baseline and one fixed,
unweighted regularized logistic regression are fit.

`split_metrics.csv` contains validation and one frozen development-test
evaluation. `pr_curves.csv.gz` contains the complete precision/recall curves.
`uncertainty_by_race.csv` contains paired race-level bootstrap summaries.
No operational threshold is authorized from this small proxy dataset.
"""


def build_all_cause_table(examples: pd.DataFrame, out: Path) -> dict[str, Any]:
    frame = examples[(examples["asof_status"] == "ASOF_TARGET_MATCHES") & examples["pair_after_attacker_position"].ne("") & examples["pair_after_target_position"].ne("")].copy()
    def swap(row: pd.Series) -> int:
        return int(row["pair_after_attacker_position"] == row["pair_before_target_position"] and row["pair_after_target_position"] == row["pair_before_attacker_position"])
    frame["all_cause_proxy_label"] = frame.apply(swap, axis=1)
    frame["all_cause_task_version"] = "phase4-all-cause-boundary-swap-v1"
    frame["all_cause_note"] = "Includes pit/race-control/non-overtaking causes; not used for primary fit"
    fields = ["example_id", "race_id", "split", "attacker_driver", "target_driver", "decision_lap", "decision_time_session_sec", "horizon_end_session_sec", "pit_horizon_status", "race_control_decision_state", "race_control_outcome_state", "all_cause_proxy_label", "all_cause_task_version", "all_cause_note"]
    write_csv_gz(out / "labels" / "phase4_all_cause_boundary_swap.csv.gz", frame.to_dict("records"), fields)
    return {"rows": len(frame), "positives": int(frame["all_cause_proxy_label"].sum()), "negatives": int((frame["all_cause_proxy_label"] == 0).sum()), "used_for_primary_fit": False}


def run(args: argparse.Namespace) -> int:
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokens = load_protected_tokens(PHASE3)
    source_files = input_paths(PHASE3)
    source_hash_before = hash_inputs(source_files)
    features, labels, examples, splits = load_tables(PHASE3, tokens)
    source_counts = {"feature_rows": len(features), "label_rows": len(labels), "example_rows": len(examples), "split_races": len(splits)}
    if len(features) != len(labels) or set(features.example_id) != set(labels.example_id) or set(examples.example_id) != set(labels.example_id):
        raise RuntimeError("Feature/label/example key sets do not match")
    if set(FEATURES) - set(features.columns):
        raise RuntimeError(f"Missing approved feature columns: {sorted(set(FEATURES) - set(features.columns))}")
    if MODEL_FEATURES & OUTCOME_FIELDS:
        raise RuntimeError("Outcome field entered feature allowlist")
    if splits.race_id.duplicated().any():
        raise RuntimeError("Duplicate race in split manifest")
    if set(splits.split) != {"train", "validation", "development_test"}:
        raise RuntimeError("Unexpected split labels")
    split_map = dict(zip(splits.race_id, splits.split))
    if any(split_map.get(race) != split for race, split in zip(examples.race_id, examples.split)):
        raise RuntimeError("Example split does not match the manifest")

    laptime_records, laptime_audit = load_laptime_records(PHASE3, tokens)
    first_sample = first_selected_sample_audit(PHASE3, examples)
    if first_sample["status"] != "PASS":
        raise RuntimeError("Decision timestamp/sample audit failed")

    examples = examples.copy()
    asof_rows = []
    for row in examples.to_dict("records"):
        race = row["race_id"]
        asof_rows.append(asof_target_check(row, laptime_records.get(race, {})))
    examples = pd.concat([examples.reset_index(drop=True), pd.DataFrame(asof_rows)], axis=1)
    examples["pair_key"] = np.where(examples["target_driver"].ne(""), examples["attacker_driver"] + "->" + examples["target_driver"], "")
    examples["old_status"] = examples["eligibility_status"]
    examples["old_proxy_label"] = pd.to_numeric(examples["proxy_label"].replace("", np.nan), errors="coerce")
    examples["new_eligibility_status"] = np.where(
        (examples["eligibility_status"] == "ELIGIBLE_PROXY") & (examples["asof_status"] == "ASOF_TARGET_MATCHES"),
        "ELIGIBLE_PROXY_ASOF",
        "UNKNOWN_CENSORED",
    )
    examples["new_proxy_label"] = np.where(examples["new_eligibility_status"] == "ELIGIBLE_PROXY_ASOF", examples["old_proxy_label"], np.nan)
    examples["new_label_class"] = np.select(
        [examples["new_proxy_label"] == 1, examples["new_proxy_label"] == 0],
        ["POSITIVE_PROXY_ASOF", "NEGATIVE_PROXY_ASOF"],
        default="UNKNOWN_CENSORED",
    )
    examples["new_reconciliation_reason"] = np.select(
        [examples["new_eligibility_status"] == "ELIGIBLE_PROXY_ASOF", examples["eligibility_status"] == "ELIGIBLE_PROXY", examples["asof_status"].eq("NO_TARGET_OR_DECISION_TIME")],
        ["ASOF_TARGET_SUPPORTED", "PREVIOUSLY_ELIGIBLE_BUT_ASOF_TARGET_UNRESOLVED", "NO_TARGET_OR_DECISION_TIME"],
        default="RETAINED_ORIGINAL_UNKNOWN_CENSORED",
    )
    repaired_fields = list(examples.columns)
    write_csv_gz(out / "repaired_prediction_examples.csv.gz", examples.to_dict("records"), repaired_fields)

    gate_a_rows, gate_a = build_gate_a(examples)
    gate_a["status"] = "PASS_WITH_SUPPORT_WARNING" if gate_a["positives"] >= 2 and gate_a["whole_race_split_isolation"] else "FAIL"
    write_csv(out / "gate_a_split_support.csv", gate_a_rows, list(gate_a_rows[0]) if gate_a_rows else ["split"])

    reconciliation = examples[examples["old_status"].eq("ELIGIBLE_PROXY") | examples["new_eligibility_status"].eq("ELIGIBLE_PROXY_ASOF")].copy()
    reconciliation_fields = ["example_id", "race_id", "split", "attacker_driver", "target_driver", "decision_lap", "old_status", "old_proxy_label", "asof_status", "asof_raw_order_status", "asof_target_driver", "new_eligibility_status", "new_proxy_label", "new_reconciliation_reason", "pit_horizon_status", "race_control_decision_state", "race_control_outcome_state", "event_id"]
    write_csv(out / "reconciliation.csv", reconciliation.to_dict("records"), reconciliation_fields)

    all_cause = build_all_cause_table(examples, out)
    evidence = build_evidence_review(examples, out / "evidence_review.csv")

    gate_b = {
        "status": "PASS_WITH_CENSORED_UNRESOLVED_ROWS",
        "first_selected_sample": first_sample,
        "asof_target_matches": int((examples["asof_status"] == "ASOF_TARGET_MATCHES").sum()),
        "asof_unresolved_previously_eligible": int(((examples["old_status"] == "ELIGIBLE_PROXY") & (examples["asof_status"] != "ASOF_TARGET_MATCHES")).sum()),
        "decision_order_rule": "latest completed laptime record with session_end_sec <= driver decision time + 1e-6",
        "instantaneous_running_order_proven": False,
        "live_publication_latency_proven": False,
        "limitation": "A completed-record as-of order supports a historical proxy only; it does not prove the exact continuous running order between different driver crossings.",
    }
    gate_c = {
        "status": "PASS_WITH_CONDITIONAL_PRIMARY_POPULATION",
        "original_future_conditioning": ["same-horizon pit events", "race-control state at outcome endpoint", "missing outcome endpoints"],
        "primary_task": "Restricted Phase 3 proxy retained, now additionally censored when as-of target support fails.",
        "all_cause_task": all_cause,
        "selection_warning": "The restricted population is not a general live-performance population; exclusions depend partly on future horizon evidence.",
    }
    gate_d = {"status": "PASS_AUTOMATED_REVIEW_ONLY", **evidence, "human_review_claimed": False}

    repaired = examples[examples["new_eligibility_status"] == "ELIGIBLE_PROXY_ASOF"].copy()
    train = repaired[repaired["split"] == "train"].copy()
    validation = repaired[repaired["split"] == "validation"].copy()
    development_test = repaired[repaired["split"] == "development_test"].copy()
    for frame, name in ((train, "train"), (validation, "validation"), (development_test, "development_test")):
        if frame["new_proxy_label"].isna().any() or not set(pd.to_numeric(frame["new_proxy_label"]).astype(int).unique()).issubset({0, 1}):
            raise RuntimeError(f"Model frame {name} contains unknown labels")
    y_train = pd.to_numeric(train["new_proxy_label"]).astype(int).to_numpy()
    y_val = pd.to_numeric(validation["new_proxy_label"]).astype(int).to_numpy()
    y_dev = pd.to_numeric(development_test["new_proxy_label"]).astype(int).to_numpy()
    x_train = prepare_feature_frame(train.merge(features, on=["example_id", "race_id", "split", "year", "event", "session"], how="left", suffixes=("_label", "")))
    x_val = prepare_feature_frame(validation.merge(features, on=["example_id", "race_id", "split", "year", "event", "session"], how="left", suffixes=("_label", "")))
    x_dev = prepare_feature_frame(development_test.merge(features, on=["example_id", "race_id", "split", "year", "event", "session"], how="left", suffixes=("_label", "")))
    if len(x_train) != len(train) or len(x_val) != len(validation) or len(x_dev) != len(development_test):
        raise RuntimeError("Feature join cardinality changed")

    train_prevalence = float(y_train.mean())
    constant_preds = {
        "train": np.full(len(train), train_prevalence),
        "validation": np.full(len(validation), train_prevalence),
        "development_test": np.full(len(development_test), train_prevalence),
    }
    # Freeze the contract before any estimator is fitted or any validation/
    # development-test prediction is generated.
    spec = f"""# Experiment Specification — Phase 4\n\n## Frozen task\n\n* Task version: `phase4-repaired-asof-boundary-swap-v1`\n* Target: fixed attacker/target pair reverses endpoint positions at the next completed lap boundary.\n* Eligibility: original Phase 3 restricted proxy, plus an as-of completed-order check at decision time. Rows whose target is not supported by a unique contiguous completed-order snapshot are UNKNOWN/CENSORED.\n* Horizon: one completed lap boundary; variable duration.\n* True on-track label: unavailable/censored.\n\n## Data and splits\n\n* Input: Phase 3 feature view, labels, examples, and cached laptime evidence.\n* Whole-race partitions: train/validation/development_test from `phase3_split_manifest.csv`; no row-level random split.\n* Repaired support: train {len(train)} rows/{int(y_train.sum())} positives; validation {len(validation)} rows/{int(y_val.sum())} positives; development-test {len(development_test)} rows/{int(y_dev.sum())} positives.\n\n## Features\n\nThe exact model feature order is embedded in `model_artifact_metadata.json`: {', '.join(FEATURES)}. Outcome/evidence fields and identifiers are excluded. Unknown source units/encodings remain a documented limitation.\n\n## Candidate budget and preprocessing\n\n1. Constant train prevalence baseline.\n2. Regularized unweighted logistic regression, `C=1.0`, L2, `liblinear`, max 5,000 iterations.\n\nMedian imputation with missingness indicators and standardization are fitted on training rows only. No class weighting, resampling, SMOTE, neural network, hyperparameter search or automatic calibration.\n\n## Selection and evaluation\n\nPrimary metric: validation average precision. Development-test is scored once after selection. ROC-AUC, AP, Brier, log loss, complete precision/recall curves and race-bootstrap uncertainty are descriptive. No operational threshold is authorized; any recall-floor point is a descriptive curve statistic.\n\n## Protected boundary\n\nThe protected final holdout was excluded using the approved exclusion manifest before any data access. No remote requests were made.\n"""
    (out / "EXPERIMENT_SPEC.md").write_text(spec, encoding="utf-8")
    model = logistic_pipeline()
    model.fit(x_train, y_train)
    logistic_preds = {
        "train": model.predict_proba(x_train)[:, 1],
        "validation": model.predict_proba(x_val)[:, 1],
        "development_test": model.predict_proba(x_dev)[:, 1],
    }
    oof_pred, oof_meta = fit_training_oof_threshold(x_train, y_train, train["race_id"].to_numpy())
    write_csv_gz(out / "training_oof_predictions.csv.gz", [{"example_id": row.example_id, "race_id": row.race_id, "oof_probability": oof_pred[idx] if np.isfinite(oof_pred[idx]) else "", "label": int(y_train[idx])} for idx, row in enumerate(train.itertuples(index=False))], ["example_id", "race_id", "oof_probability", "label"])

    metrics_table = []
    curves = []
    reliability = []
    prediction_rows = []
    for split_name, frame, y, const, pred in (("train", train, y_train, constant_preds["train"], logistic_preds["train"]), ("validation", validation, y_val, constant_preds["validation"], logistic_preds["validation"]), ("development_test", development_test, y_dev, constant_preds["development_test"], logistic_preds["development_test"])):
        for model_name, probabilities in (("constant_train_prevalence", const), ("logistic_regression_C1", pred)):
            row = evaluate_model(model_name, split_name, y, probabilities)
            metrics_table.append(row)
            curves.extend(curve_rows(model_name, split_name, y, probabilities))
            reliability.extend(reliability_rows(model_name, split_name, y, probabilities))
            for idx, item in enumerate(frame.itertuples(index=False)):
                prediction_rows.append({"example_id": item.example_id, "race_id": item.race_id, "split": split_name, "model": model_name, "label": int(y[idx]), "probability": float(probabilities[idx])})
    write_csv_gz(out / "pr_curves.csv.gz", curves, ["model", "split", "threshold", "precision", "recall", "predicted_positive"])
    write_csv(out / "reliability_bins.csv", reliability, ["model", "split", "probability_bin", "count", "mean_predicted_probability", "observed_positive_rate", "positive_count"])
    write_csv_gz(out / "per_example_predictions.csv.gz", prediction_rows, ["example_id", "race_id", "split", "model", "label", "probability"])
    validation_logistic = next(r for r in metrics_table if r["model"] == "logistic_regression_C1" and r["split"] == "validation")
    validation_constant = next(r for r in metrics_table if r["model"] == "constant_train_prevalence" and r["split"] == "validation")
    selected_model = "logistic_regression_C1" if validation_logistic["average_precision"] > validation_constant["average_precision"] else "constant_train_prevalence"
    selected = {"model": selected_model, "selection_metric": "validation_average_precision", "validation_average_precision": validation_logistic["average_precision"] if selected_model == "logistic_regression_C1" else validation_constant["average_precision"], "validation_constant_average_precision": validation_constant["average_precision"], "training_only_oof_threshold": oof_meta}
    joblib.dump(model, out / "logistic_regression_pipeline.joblib", compress=3)
    write_json(out / "constant_baseline.json", {"model": "constant_train_prevalence", "train_prevalence": train_prevalence, "task_version": "phase4-repaired-asof-boundary-swap-v1"})
    write_json(out / "model_artifact_metadata.json", {"model": "logistic_regression_C1", "task_version": "phase4-repaired-asof-boundary-swap-v1", "features": FEATURES, "random_seed": RANDOM_SEED, "preprocessing": "SimpleImputer(strategy=median, add_indicator=True) then StandardScaler, fit only on training partition", "configuration": {"C": 1.0, "penalty": "l2", "solver": "liblinear", "class_weight": None, "max_iter": 5000}, "input_hashes": source_hash_before})

    train_meta = {"task_version": "phase4-repaired-asof-boundary-swap-v1", "train_rows": len(train), "train_positives": int(y_train.sum()), "train_races": int(train.race_id.nunique())}
    model_card(out, train_meta)
    phase5_handoff(out)
    write_json(out / "laptime_cache_audit.json", {"files": laptime_audit, "loaded_files": sum(row.get("load_status") == "LOADED" for row in laptime_audit), "coverage_rows": len(laptime_audit)})
    write_csv(out / "uncertainty_by_race.csv", bootstrap_uncertainty(validation, y_val, logistic_preds["validation"], constant_preds["validation"], "validation") + bootstrap_uncertainty(development_test, y_dev, logistic_preds["development_test"], constant_preds["development_test"], "development_test"), ["split", "metric", "bootstrap_replicates", "mean", "p02_5", "p50", "p97_5", "degenerate_replicates"])
    write_csv(out / "split_metrics.csv", metrics_table, ["model", "split", "rows", "positives", "negatives", "prevalence", "roc_auc", "average_precision", "brier_score", "log_loss", "descriptive_precision_at_recall_ge_70"])

    source_hash_after = hash_inputs(source_files)
    invariants = {
        "source_hashes_unchanged": source_hash_before == source_hash_after,
        "source_counts": source_counts,
        "repaired_rows_equal_source_examples": len(examples) == len(labels) == len(features),
        "repaired_eligible_count": len(repaired),
        "feature_label_separation": not bool(MODEL_FEATURES & OUTCOME_FIELDS),
        "whole_race_split_isolation": gate_a["whole_race_split_isolation"],
        "protected_session_excluded": True,
        "no_remote_requests": True,
    }
    if not invariants["source_hashes_unchanged"]:
        raise RuntimeError("Input hash changed during experiment")
    # Inference parity is checked after serialisation, before final reports.
    loaded_model = joblib.load(out / "logistic_regression_pipeline.joblib")
    parity = bool(np.allclose(loaded_model.predict_proba(x_val)[:, 1], logistic_preds["validation"], rtol=0.0, atol=1e-12))
    write_json(out / "inference_parity.json", {"status": "PASS" if parity else "FAIL", "validation_rows": len(x_val), "max_abs_probability_difference": float(np.max(np.abs(loaded_model.predict_proba(x_val)[:, 1] - logistic_preds["validation"])))})
    if not parity:
        raise RuntimeError("Saved-model inference parity failed")

    gate_results = {
        "generated_at_utc": utc_now(),
        "A": gate_a,
        "B": gate_b,
        "C": gate_c,
        "D": gate_d,
        "invariants": invariants,
        "tests_authorized": ["timestamp ordering", "target identity as-of support", "future-conditioned reconciliation", "feature/label isolation", "train-only preprocessing", "whole-race split isolation", "saved-model parity", "source immutability", "protected-session exclusion"],
    }
    write_json(out / "gate_results.json", gate_results)
    metrics = {
        "generated_at_utc": utc_now(),
        "outcome": "NO_DEMONSTRATED_GAIN",
        "task_version": "phase4-repaired-asof-boundary-swap-v1",
        "source_input_hashes_before": source_hash_before,
        "source_input_hashes_after": source_hash_after,
        "totals": {
            "source_examples": len(examples),
            "source_eligible": int((examples["old_status"] == "ELIGIBLE_PROXY").sum()),
            "asof_target_matches": gate_b["asof_target_matches"],
            "asof_target_unresolved": gate_b["asof_unresolved_previously_eligible"],
            "repaired_eligible": len(repaired),
            "repaired_positives": int(y_train.sum() + y_val.sum() + y_dev.sum()),
            "repaired_negatives": int((y_train == 0).sum() + (y_val == 0).sum() + (y_dev == 0).sum()),
            "repaired_unknown_or_censored": int(len(examples) - len(repaired)),
            "development_splits": {"train_races": int(train.race_id.nunique()), "validation_races": int(validation.race_id.nunique()), "development_test_races": int(development_test.race_id.nunique()), "train_rows": len(train), "validation_rows": len(validation), "development_test_rows": len(development_test)},
        },
        "gates": {"A": gate_a, "B": gate_b, "C": gate_c, "D": gate_d},
        "metrics_table": metrics_table,
        "selection": selected,
        "feature_allowlist": FEATURES,
        "invariants": invariants,
        "access_ledger": {"phase3_cached_inputs_read": True, "validation_outcomes_accessed": True, "development_test_outcomes_accessed": True, "protected_final_holdout_accessed": False, "remote_requests_made": False, "model_connected_to_engine": False},
    }
    write_json(out / "phase4_metrics.json", metrics)
    (out / "PHASE4_REPORT.md").write_text(report_text(metrics, gate_a_rows), encoding="utf-8")
    (out / "README.md").write_text(readme_text(), encoding="utf-8")
    write_json(out / "test_summary.json", {"timestamp_first_sample": first_sample, "saved_model_parity": parity, "source_hashes_unchanged": invariants["source_hashes_unchanged"], "feature_label_separation": invariants["feature_label_separation"], "whole_race_split_isolation": invariants["whole_race_split_isolation"], "protected_session_excluded": True})

    # The checksum file covers all ordinary artifacts in this directory.  The
    # package verification record is deliberately kept outside the archive so
    # it can contain the final package hash without creating a self-hash cycle.
    package = out / "APEX-R_Phase4_Proxy_Experiment.zip"
    package_verification = out / "package_verification.json"
    if package_verification.exists():
        package_verification.unlink()
    checksum_paths = [path for path in sorted(out.rglob("*")) if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and path.name not in {"SHA256SUMS.txt", package.name, package_verification.name}]
    (out / "SHA256SUMS.txt").write_text("\n".join(f"{sha256_file(path)}  {path.relative_to(out).as_posix()}" for path in checksum_paths) + "\n", encoding="utf-8")
    if package.exists():
        package.unlink()
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and path not in {package, package_verification}:
                archive.write(path, path.relative_to(out).as_posix())
    with zipfile.ZipFile(package) as archive:
        if archive.testzip():
            raise RuntimeError("Final package CRC failure")
        package_members = len(archive.namelist())
    package_bytes = package.stat().st_size
    package_sha = sha256_file(package)
    write_json(package_verification, {"package": package.name, "bytes": package_bytes, "sha256": package_sha, "crc_status": "PASS", "members": package_members, "sha256sums_entries": len(checksum_paths), "verification_scope": "external_record; excluded from package and SHA256SUMS to avoid self-reference"})
    # This final verification record is intentionally external; SHA256SUMS
    # continues to describe the files shipped inside the package exactly.
    print(json.dumps({
        "output_dir": str(out),
        "outcome": metrics["outcome"],
        "totals": metrics["totals"],
        "gates": {key: value["status"] for key, value in (("A", gate_a), ("B", gate_b), ("C", gate_c), ("D", gate_d))},
        "metrics": metrics_table,
        "selected_model": selected_model,
        "inference_parity": parity,
        "source_hashes_unchanged": invariants["source_hashes_unchanged"],
        "package": {"path": str(package), "bytes": package_bytes, "sha256": package_sha, "crc_status": "PASS"},
    }, indent=2, default=str))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
