#!/usr/bin/env python3
"""Augment the frozen Phase 1 tables with causal OpenF1 pit features.

This script deliberately has a fixed train/validation surface.  It never
accepts an arbitrary session key and never reads a model or a sealed test
table.  The existing eleven TracingInsights-derived columns are copied
unchanged; four additional columns are derived from OpenF1's ``pit`` event
records when the endpoint has data for an approved session.

The OpenF1 ``pit.date`` field is treated as a pit-event timestamp, not silently
as a pit-exit timestamp.  Consequently these columns are named ``pit_recent``
and ``last_pit_lane_duration`` rather than ``pit_out``.  The existing
TracingInsights pit-out columns remain the source for the out-lap flag.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "phase1-features"
OUTPUT_DIR = ROOT / "data" / "phase1-features"
CACHE_DIR = ROOT / "data" / "openf1-cache"
OPENF1_API = "https://api.openf1.org/v1/"

TRAIN_SESSIONS = (7953, 7779, 7787)
VALIDATION_SESSION = 9070
APPROVED_SESSIONS = (*TRAIN_SESSIONS, VALIDATION_SESSION)
PIT_RECENT_SECONDS = 90.0

TRAIN_INPUT = INPUT_DIR / "train_phase1_features.csv.gz"
VALIDATION_INPUT = INPUT_DIR / "validation_phase1_features.csv.gz"
TRAIN_OUTPUT = OUTPUT_DIR / "train_phase1_openf1_pit_features.csv.gz"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation_phase1_openf1_pit_features.csv.gz"
MANIFEST_OUTPUT = OUTPUT_DIR / "phase1_openf1_pit_feature_manifest.json"
REPORT_OUTPUT = ROOT / "models" / "PHASE1_OPENF1_PIT_FEATURE_REPORT.md"

PIT_FEATURES = [
    "attacker_openf1_pit_recent_flag",
    "rival_openf1_pit_recent_flag",
    "attacker_openf1_last_pit_lane_duration_sec",
    "rival_openf1_last_pit_lane_duration_sec",
]


def timestamp(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def numeric(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def normalize_driver(value) -> int | None:
    parsed = numeric(value)
    return int(parsed) if parsed is not None and parsed >= 0 else None


def pit_cache_path(session_key: int) -> Path:
    url = OPENF1_API + "pit?" + urlencode({"session_key": session_key})
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{digest}.json"


def load_pit(session_key: int) -> tuple[list[dict], dict]:
    """Load one approved session's pit events and record availability."""
    if session_key not in APPROVED_SESSIONS:
        raise ValueError("Session is outside the fixed train/validation allow-list")

    path = pit_cache_path(session_key)
    url = OPENF1_API + "pit?" + urlencode({"session_key": session_key})
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected cached pit payload: {path}")
        return payload, {"available": True, "status": "cached", "url": url, "cache_path": str(path.relative_to(ROOT))}

    try:
        request = Request(url, headers={"User-Agent": "APEX-R-phase1-openf1-pit/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected OpenF1 pit response for session {session_key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return payload, {"available": True, "status": "fetched", "url": url, "cache_path": str(path.relative_to(ROOT))}
    except HTTPError as error:
        if error.code == 404:
            return [], {"available": False, "status": "http_404_no_records", "url": url, "cache_path": None}
        raise RuntimeError(f"OpenF1 pit request failed for session {session_key}: HTTP {error.code}") from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"OpenF1 pit request failed for session {session_key}: {error}") from error


def prepare_events(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for source in rows:
        driver = normalize_driver(source.get("driver_number"))
        event_time = timestamp(source.get("date"))
        if driver is None or event_time is None:
            continue
        grouped[driver].append(
            {
                "time": event_time,
                "driver_number": driver,
                "lap_number": normalize_driver(source.get("lap_number")),
                "lane_duration_sec": numeric(source.get("lane_duration")),
                "stop_duration_sec": numeric(source.get("stop_duration")),
            }
        )
    for values in grouped.values():
        values.sort(key=lambda row: row["time"])
    return dict(grouped)


def latest_pit_event(events: list[dict], point: float) -> dict | None:
    """Return the latest event at or before point; never uses a future event."""
    times = [row["time"] for row in events]
    index = bisect.bisect_right(times, point) - 1
    return events[index] if index >= 0 else None


def pit_value(events: list[dict], point: float) -> tuple[float | None, float | None, dict | None]:
    event = latest_pit_event(events, point)
    if event is None:
        return 0.0, None, None
    age = point - event["time"]
    return float(age <= PIT_RECENT_SECONDS), event["lane_duration_sec"], event


def add_pit_features(frame: pd.DataFrame, session_key: int, pit_rows: list[dict], availability: dict) -> tuple[pd.DataFrame, dict, dict]:
    output = frame.copy()
    events_by_driver = prepare_events(pit_rows)
    missing: dict[str, Counter] = {feature: Counter() for feature in PIT_FEATURES}
    samples: dict[str, list[dict]] = {feature: [] for feature in PIT_FEATURES}

    for feature in PIT_FEATURES:
        output[feature] = pd.NA

    for index, row in output.iterrows():
        point = timestamp(row["date"])
        attacker = normalize_driver(row["driver_number"])
        rival = normalize_driver(row.get("car_ahead_driver_number"))
        if point is None or attacker is None:
            for feature in PIT_FEATURES:
                missing[feature]["prediction_identity_unavailable"] += 1
            continue

        for side, driver in (("attacker", attacker), ("rival", rival)):
            flag = f"{side}_openf1_pit_recent_flag"
            duration = f"{side}_openf1_last_pit_lane_duration_sec"
            if side == "rival" and driver is None:
                missing[flag]["no_car_ahead_identity"] += 1
                missing[duration]["no_car_ahead_identity"] += 1
                continue
            if not availability["available"]:
                reason = f"pit_endpoint_unavailable_{availability['status']}"
                missing[flag][reason] += 1
                missing[duration][reason] += 1
                continue

            event = latest_pit_event(events_by_driver.get(driver, []), point)
            flag_value, lane_duration, event = pit_value(events_by_driver.get(driver, []), point)
            output.at[index, flag] = flag_value
            if event is None:
                missing[duration]["no_prior_pit_event"] += 1
            elif lane_duration is None:
                missing[duration]["lane_duration_missing"] += 1
            else:
                output.at[index, duration] = lane_duration

            if len(samples[flag]) < 3:
                samples[flag].append(
                    {
                        "session_key": session_key,
                        "prediction_time": row["date"],
                        "driver_number": driver,
                        "feature_value": None if pd.isna(output.at[index, flag]) else float(output.at[index, flag]),
                        "raw_inputs": {
                            "pit_endpoint_status": availability["status"],
                            "latest_prior_pit_event": event,
                            "recent_window_seconds": PIT_RECENT_SECONDS,
                        },
                    }
                )
            if len(samples[duration]) < 3:
                samples[duration].append(
                    {
                        "session_key": session_key,
                        "prediction_time": row["date"],
                        "driver_number": driver,
                        "feature_value": None if pd.isna(output.at[index, duration]) else float(output.at[index, duration]),
                        "raw_inputs": {
                            "pit_endpoint_status": availability["status"],
                            "latest_prior_pit_event": event,
                        },
                    }
                )

    return output, {feature: dict(counter) for feature, counter in missing.items()}, samples


def feature_summary(frame: pd.DataFrame, feature: str) -> dict:
    missing_count = int(frame[feature].isna().sum())
    values = pd.to_numeric(frame[feature], errors="coerce").dropna()
    result = {
        "rows": int(len(frame)),
        "missing_count": missing_count,
        "missing_rate": float(missing_count / len(frame)) if len(frame) else None,
        "available_count": int(len(frame) - missing_count),
    }
    if len(values):
        result["statistics"] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "median": float(values.median()),
            "max": float(values.max()),
        }
    return result


def write_report(manifest: dict) -> None:
    lines = [
        "# OpenF1 Pit Feature Augmentation Report",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "## Scope",
        "",
        "Validation-only augmentation of the existing Phase 1 train/validation tables. Existing Phase 1 columns were copied unchanged. No model was trained or modified by this builder.",
        "",
        "## New columns",
        "",
        "- `attacker_openf1_pit_recent_flag` and `rival_openf1_pit_recent_flag`: 1 when the latest OpenF1 pit event for that driver is at or before the prediction time and no older than 90 seconds; 0 when the source is available but no event is recent.",
        "- `attacker_openf1_last_pit_lane_duration_sec` and `rival_openf1_last_pit_lane_duration_sec`: the `lane_duration` from the latest prior OpenF1 pit event.",
        "- Join: `session_key + driver_number`, then backward event-time lookup against the Phase 1 row's UTC `date`. No nearest-neighbour tolerance or interpolation is used for discrete pit events.",
        "- Causality: only `pit.date <= prediction date` is eligible. The endpoint timestamp is not relabeled as pit exit; the existing TracingInsights `pout` columns remain the pit-out source.",
        "",
        "## Endpoint availability",
        "",
    ]
    for session_key, info in manifest["sessions"].items():
        lines.append(f"- Session `{session_key}`: `{info['status']}`, records `{info['records']}`.")
    lines.extend(["", "## Missingness", ""])
    for split in ("train", "validation"):
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| Feature | Missing | Rows | Rate | Main reason |")
        lines.append("|---|---:|---:|---:|---|")
        for feature in PIT_FEATURES:
            summary = manifest["feature_summary"][split][feature]
            reasons = manifest["missing_reasons"][split][feature]
            main_reason = max(reasons, key=reasons.get) if reasons else "none"
            lines.append(f"| `{feature}` | {summary['missing_count']:,} | {summary['rows']:,} | {summary['missing_rate']:.2%} | `{main_reason}` |")
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "The approved 2023 train/validation sessions returned no OpenF1 `pit` records during this build, so these four new columns are explicitly missing rather than filled with zero. They cannot contribute signal to a model trained on this split. This is a data-availability limitation, not evidence that pit data is unhelpful on other sessions.",
        "",
        "## Outputs",
        "",
        f"- Train: `{manifest['outputs']['train']}`",
        f"- Validation: `{manifest['outputs']['validation']}`",
        f"- Manifest: `{MANIFEST_OUTPUT.relative_to(ROOT)}`",
    ])
    REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    train = pd.read_csv(TRAIN_INPUT)
    validation = pd.read_csv(VALIDATION_INPUT)
    if set(train["session_key"].unique()) != set(TRAIN_SESSIONS):
        raise RuntimeError("Training input contains an unapproved session")
    if set(validation["session_key"].unique()) != {VALIDATION_SESSION}:
        raise RuntimeError("Validation input contains an unapproved session")

    combined = pd.concat([train, validation], ignore_index=True)
    session_manifests = {}
    missing_reasons_by_split: dict[str, dict[str, dict[str, int]]] = {
        "train": {feature: {} for feature in PIT_FEATURES},
        "validation": {feature: {} for feature in PIT_FEATURES},
    }
    augmented_parts = []
    samples = {feature: [] for feature in PIT_FEATURES}

    for session_key in APPROVED_SESSIONS:
        part = combined[combined["session_key"] == session_key].copy()
        pit_rows, availability = load_pit(session_key)
        part, reasons, part_samples = add_pit_features(part, session_key, pit_rows, availability)
        augmented_parts.append(part)
        session_manifests[str(session_key)] = {**availability, "records": len(pit_rows), "rows": len(part)}
        split = "validation" if session_key == VALIDATION_SESSION else "train"
        for feature in PIT_FEATURES:
            aggregate = Counter(missing_reasons_by_split[split][feature])
            aggregate.update(reasons[feature])
            missing_reasons_by_split[split][feature] = dict(aggregate)
            samples[feature].extend(part_samples[feature])

    output = pd.concat(augmented_parts, ignore_index=True)
    output = output.sort_values(["split", "session_key", "driver_number", "date"], kind="stable").reset_index(drop=True)
    train_output = output[output["split"] == "train"].copy()
    validation_output = output[output["split"] == "validation"].copy()
    for frame in (train_output, validation_output):
        for feature in PIT_FEATURES:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    train_output.to_csv(TRAIN_OUTPUT, index=False, compression="gzip")
    validation_output.to_csv(VALIDATION_OUTPUT, index=False, compression="gzip")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "OpenF1 pit augmentation only; no model training, scoring, holdout access, or decision-engine changes",
        "approved_train_sessions": list(TRAIN_SESSIONS),
        "approved_validation_session": VALIDATION_SESSION,
        "pit_recent_window_seconds": PIT_RECENT_SECONDS,
        "join_policy": {
            "keys": ["session_key", "driver_number"],
            "prediction_timestamp": "existing Phase 1 UTC date",
            "event_timestamp": "OpenF1 pit.date",
            "timestamp_rule": "backward-only event_time <= prediction_time",
            "timestamp_tolerance_seconds": 0.0,
            "interpolation": "none",
            "pit_exit_semantics": "not assumed; OpenF1 pit.date is retained as an event timestamp",
        },
        "new_features": PIT_FEATURES,
        "source_fields": {
            "pit_recent_flags": ["pit.session_key", "pit.driver_number", "pit.date"],
            "last_pit_lane_duration": ["pit.session_key", "pit.driver_number", "pit.date", "pit.lane_duration"],
        },
        "sessions": session_manifests,
        "missing_reasons": missing_reasons_by_split,
        "feature_summary": {
            "train": {feature: feature_summary(train_output, feature) for feature in PIT_FEATURES},
            "validation": {feature: feature_summary(validation_output, feature) for feature in PIT_FEATURES},
        },
        "real_samples": samples,
        "causal_safety": {
            "backward_only": True,
            "future_events_rejected": True,
            "no_interpolation": True,
            "no_zero_or_placeholder_imputation": True,
            "pit_date_not_treated_as_exit": True,
        },
        "outputs": {
            "train": str(TRAIN_OUTPUT.relative_to(ROOT)),
            "validation": str(VALIDATION_OUTPUT.relative_to(ROOT)),
            "columns": list(output.columns),
            "train_rows": len(train_output),
            "validation_rows": len(validation_output),
        },
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(manifest)
    print(f"Saved {TRAIN_OUTPUT} ({len(train_output):,} rows)")
    print(f"Saved {VALIDATION_OUTPUT} ({len(validation_output):,} rows)")
    print(f"Saved {MANIFEST_OUTPUT}")
    print(f"Saved {REPORT_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
