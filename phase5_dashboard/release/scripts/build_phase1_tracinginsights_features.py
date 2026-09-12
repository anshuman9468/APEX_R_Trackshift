#!/usr/bin/env python3
"""Build causal Phase 1 TracingInsights features for approved train/validation races.

This script is intentionally feature-only:

* it reads the existing OpenF1 cache without modifying it;
* it downloads TracingInsights telemetry one race at a time and keeps raw lap
  files in memory only;
* it writes compressed train/validation tables and an audit manifest;
* it does not import, train, score, or write any model artifact.

The executable surface is fixed to the approved train and validation sessions.
There is no command-line session override.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OPENF1_CACHE = ROOT / "data" / "openf1-cache"
OUTPUT_DIR = ROOT / "data" / "phase1-features"
OPENF1_API = "https://api.openf1.org/v1/"
TRACING_RAW = "https://raw.githubusercontent.com/TracingInsights/{year}/main/{path}"

TRAIN_SESSIONS = (7953, 7779, 7787)
VALIDATION_SESSION = 9070
SESSION_CONFIG = {
    7953: {"year": 2023, "event": "Bahrain Grand Prix", "session": "Race", "split": "train"},
    7779: {"year": 2023, "event": "Saudi Arabian Grand Prix", "session": "Race", "split": "train"},
    7787: {"year": 2023, "event": "Australian Grand Prix", "session": "Race", "split": "train"},
    9070: {"year": 2023, "event": "Azerbaijan Grand Prix", "session": "Race", "split": "validation"},
}
EXPECTED_CLEAN_COUNTS = {
    7953: (4917, 626),
    7779: (5128, 398),
    7787: (3867, 187),
    9070: (6058, 257),
}

SAMPLE_SECONDS = 10.0
HORIZON_SECONDS = 60.0
CONTROL_BUFFER_SECONDS = 60.0
TELEMETRY_BACKWARD_TOLERANCE_SECONDS = 0.500
PIT_OUT_RECENT_SECONDS = 90.0
MAX_DOWNLOAD_WORKERS = 12

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
PHASE1_FEATURE_GROUPS = {
    "opponent_tyre_age_delta": ["opponent_tyre_age_delta"],
    "opponent_tyre_compound_delta": ["opponent_tyre_compound_delta"],
    "stint_length_delta": ["stint_length_delta"],
    "relative_sector_time_delta": [
        "relative_sector_time_delta_s1",
        "relative_sector_time_delta_s2",
        "relative_sector_time_delta_s3",
    ],
    "relative_lap_time_delta": ["relative_lap_time_delta"],
    "distance_to_car_ahead_m": ["distance_to_car_ahead_m"],
    "rival_drs_open": ["rival_drs_open"],
    "pit_out_recent_flag": [
        "attacker_pit_out_recent_flag",
        "rival_pit_out_recent_flag",
    ],
}
PHASE1_COLUMNS = [column for columns in PHASE1_FEATURE_GROUPS.values() for column in columns]
IDENTIFIER_COLUMNS = [
    "session_key",
    "split",
    "driver_number",
    "car_ahead_driver_number",
    "date",
]
LABEL_COLUMN = "overtake_next_60s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=MAX_DOWNLOAD_WORKERS)
    return parser.parse_args()


def numeric(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def timestamp(value) -> float | None:
    if not value or value == "None":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def iso_time(value: float | None) -> str | None:
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


def normalize_driver_number(value) -> int | None:
    number = numeric(value)
    return int(number) if number is not None and number >= 0 else None


def normalize_text(value) -> str | None:
    if value is None or str(value).strip() in {"", "None", "UNKNOWN", "nan"}:
        return None
    return str(value).strip().upper()


def records(columnar) -> list[dict]:
    if isinstance(columnar, list):
        return [row for row in columnar if isinstance(row, dict)]
    if not isinstance(columnar, dict) or not columnar:
        return []
    arrays = {key: value for key, value in columnar.items() if isinstance(value, list)}
    if not arrays:
        return [columnar]
    count = max(len(value) for value in arrays.values())
    return [
        {
            key: value[index] if isinstance(value, list) and index < len(value) else value
            for key, value in columnar.items()
        }
        for index in range(count)
    ]


def openf1_cache_path(endpoint: str, session_key: int) -> Path:
    url = OPENF1_API + endpoint + "?" + urlencode({"session_key": session_key})
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return OPENF1_CACHE / f"{key}.json"


def load_openf1(endpoint: str, session_key: int) -> list[dict]:
    if session_key not in SESSION_CONFIG:
        raise ValueError("Session is outside the fixed train/validation allow-list")
    path = openf1_cache_path(endpoint, session_key)
    if not path.is_file():
        raise FileNotFoundError(f"Required existing OpenF1 cache file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected OpenF1 cache payload: {path}")
    return payload


def prepare_lookup(rows: list[dict], date_field: str | None = None) -> dict[int, list[tuple[float, dict]]]:
    grouped: dict[int, list[tuple[float, dict]]] = defaultdict(list)
    for row in rows:
        driver = normalize_driver_number(row.get("driver_number"))
        time_value = timestamp(row.get(date_field)) if date_field else timestamp(row.get("date") or row.get("date_start"))
        if driver is not None and time_value is not None:
            grouped[driver].append((time_value, row))
    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    return dict(grouped)


def latest_row(values: list[tuple[float, dict]], point: float) -> dict | None:
    if not values:
        return None
    index = bisect.bisect_right([item[0] for item in values], point) - 1
    return values[index][1] if index >= 0 else None


def nearest_in_window(values: list[float], start: float, end: float) -> bool:
    index = bisect.bisect_left(values, start)
    return index < len(values) and values[index] <= end


def abnormal_control_times(rows: list[dict]) -> list[float]:
    result = []
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
        point = timestamp(row.get("date"))
        if abnormal and point is not None:
            result.append(point)
    return sorted(result)


def build_clean_base_rows(session_key: int) -> list[dict]:
    """Reproduce the existing clean-label base rows without loading model code."""
    intervals = load_openf1("intervals", session_key)
    positions = load_openf1("position", session_key)
    laps = load_openf1("laps", session_key)
    stints = load_openf1("stints", session_key)
    weather = load_openf1("weather", session_key)
    overtakes = load_openf1("overtakes", session_key)
    race_control = load_openf1("race_control", session_key)

    positions_by_driver = prepare_lookup(positions)
    laps_by_driver = prepare_lookup(laps, "date_start")
    stints_by_driver: dict[int, list[dict]] = defaultdict(list)
    for row in stints:
        driver = normalize_driver_number(row.get("driver_number"))
        if driver is not None:
            stints_by_driver[driver].append(row)
    for values in stints_by_driver.values():
        values.sort(key=lambda row: numeric(row.get("lap_start"), 0))

    weather_rows = sorted(
        (point, row)
        for row in weather
        if (point := timestamp(row.get("date"))) is not None
    )
    weather_times = [item[0] for item in weather_rows]
    interval_by_driver = prepare_lookup(intervals)

    base_rows: list[dict] = []
    for driver, values in interval_by_driver.items():
        previous_interval = None
        previous_time = None
        last_sample = -float("inf")
        race_start = values[0][0]
        race_end = values[-1][0]
        for point, source in values:
            interval = numeric(source.get("interval"))
            leader_gap = numeric(source.get("gap_to_leader"))
            if interval is None or leader_gap is None or point - last_sample < SAMPLE_SECONDS:
                continue
            last_sample = point
            closing = 0.0
            if previous_interval is not None and previous_time is not None and point > previous_time:
                closing = (previous_interval - interval) / ((point - previous_time) / 60.0)
            previous_interval, previous_time = interval, point

            position_row = latest_row(positions_by_driver.get(driver, []), point)
            position = numeric((position_row or {}).get("position"), 10)
            lap_row = latest_row(laps_by_driver.get(driver, []), point)
            lap_number = numeric((lap_row or {}).get("lap_number"), 1)
            current_stint = None
            for stint in stints_by_driver.get(driver, []):
                if numeric(stint.get("lap_start"), 0) <= lap_number:
                    current_stint = stint
                else:
                    break
            tyre_age = numeric((current_stint or {}).get("tyre_age_at_start"), 0)
            stint_lap = numeric((current_stint or {}).get("lap_start"), lap_number)
            tyre_age = max(0.0, tyre_age + max(0.0, lap_number - stint_lap))

            weather_row = {"track_temperature": 30.0, "rainfall": 0}
            weather_index = bisect.bisect_right(weather_times, point) - 1
            if weather_index >= 0:
                weather_row = weather_rows[weather_index][1]
            progress = 0.0 if race_end <= race_start else (point - race_start) / (race_end - race_start)
            base_rows.append(
                {
                    "session_key": session_key,
                    "split": SESSION_CONFIG[session_key]["split"],
                    "driver_number": driver,
                    "date": iso_time(point),
                    "_point": point,
                    "interval_sec": interval,
                    "gap_to_leader_sec": leader_gap,
                    "closing_rate_sec_per_min": max(-30.0, min(30.0, closing)),
                    "position": max(1.0, min(20.0, position)),
                    "tyre_age": max(0.0, min(60.0, tyre_age)),
                    "track_temperature_c": numeric(weather_row.get("track_temperature"), 30.0),
                    "rainfall": float(bool(weather_row.get("rainfall"))),
                    "race_progress": max(0.0, min(1.0, progress)),
                }
            )

    abnormal = abnormal_control_times(race_control)
    clean_events: dict[int, list[float]] = defaultdict(list)
    for event in overtakes:
        point = timestamp(event.get("date"))
        overtaking = normalize_driver_number(event.get("overtaking_driver_number"))
        if point is None or overtaking is None:
            continue
        near_control = nearest_in_window(abnormal, point - CONTROL_BUFFER_SECONDS, point + CONTROL_BUFFER_SECONDS)
        if not near_control:
            clean_events[overtaking].append(point)
    for values in clean_events.values():
        values.sort()

    clean_rows = []
    for row in base_rows:
        point = row["_point"]
        near_control = nearest_in_window(
            abnormal,
            point - CONTROL_BUFFER_SECONDS,
            point + HORIZON_SECONDS + CONTROL_BUFFER_SECONDS,
        )
        if near_control:
            continue
        events = clean_events.get(row["driver_number"], [])
        event_index = bisect.bisect_right(events, point)
        row[LABEL_COLUMN] = int(event_index < len(events) and events[event_index] <= point + HORIZON_SECONDS)
        clean_rows.append(row)

    expected_rows, expected_positive = EXPECTED_CLEAN_COUNTS[session_key]
    actual = (len(clean_rows), sum(row[LABEL_COLUMN] for row in clean_rows))
    if actual != (expected_rows, expected_positive):
        raise RuntimeError(
            f"Clean-row reproduction changed for approved session {session_key}: "
            f"expected {(expected_rows, expected_positive)}, got {actual}"
        )
    return clean_rows


@dataclass
class DownloadAudit:
    requests: int = 0
    bytes: int = 0
    missing_files: int = 0

    def __post_init__(self):
        self._lock = threading.Lock()

    def add(self, byte_count: int, missing: bool = False) -> None:
        with self._lock:
            self.requests += 1
            self.bytes += byte_count
            self.missing_files += int(missing)


_thread_local = threading.local()


def http_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "APEX-R-phase1-feature-audit/1.0"})
        _thread_local.session = session
    return session


def raw_url(config: dict, relative_path: str) -> str:
    path = "/".join([config["event"], config["session"], relative_path])
    return TRACING_RAW.format(year=config["year"], path=quote(path, safe="/"))


def fetch_json(url: str, audit: DownloadAudit, allow_missing: bool = False):
    last_error = None
    for attempt in range(5):
        try:
            response = http_session().get(url, timeout=45)
            if response.status_code == 404 and allow_missing:
                audit.add(len(response.content), missing=True)
                return None
            response.raise_for_status()
            audit.add(len(response.content))
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(1.0 * (2**attempt))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def tracing_lap_end_time(row: dict, start: float) -> float | None:
    session_end = numeric(row.get("sesT"))
    session_start = numeric(row.get("lST"))
    if session_end is not None and session_start is not None and 0 < session_end - session_start < 600:
        return start + (session_end - session_start)
    duration = numeric(row.get("time"))
    return start + duration if duration is not None and 0 < duration < 600 else None


def tracing_event_time(row: dict, event_field: str, start: float) -> float | None:
    event_session_time = numeric(row.get(event_field))
    lap_session_start = numeric(row.get("lST"))
    if event_session_time is None or lap_session_start is None:
        return None
    delta = event_session_time - lap_session_start
    return start + delta if -5 <= delta < 600 else None


def timing_quality_reasons(row: dict) -> list[str]:
    """Reject timing rows that are not comparable green, non-pit pace laps."""
    reasons = []
    if str(row.get("status")) != "1":
        reasons.append("non_clear_track_status")
    if normalize_text(row.get("pin")) is not None or normalize_text(row.get("pout")) is not None:
        reasons.append("pit_in_or_out_lap")
    if bool(row.get("ff1G")):
        reasons.append("generated_lap")
    if bool(row.get("del")):
        reasons.append("deleted_lap")
    return reasons


def prepare_tracing_laps(lap_rows: list[dict]):
    by_driver: dict[int, list[dict]] = defaultdict(list)
    code_by_driver: dict[int, str] = {}
    for row in lap_rows:
        driver = normalize_driver_number(row.get("dNum"))
        code = normalize_text(row.get("drv"))
        lap = numeric(row.get("lap"))
        start = timestamp(row.get("lSD"))
        if driver is None or code is None or lap is None or start is None:
            continue
        prepared = dict(row)
        prepared["_driver"] = driver
        prepared["_code"] = code
        prepared["_lap"] = int(lap)
        prepared["_lap_start"] = start
        prepared["_lap_end"] = tracing_lap_end_time(row, start)
        prepared["_pit_out"] = tracing_event_time(row, "pout", start)
        code_by_driver[driver] = code
        by_driver[driver].append(prepared)

    state_by_driver: dict[int, list[tuple[float, dict]]] = defaultdict(list)
    pit_out_by_driver: dict[int, list[float]] = defaultdict(list)
    sector_by_driver: dict[int, dict[int, dict[int, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    lap_time_by_driver: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    telemetry_specs = []
    timing_quality = Counter()

    for driver, rows in by_driver.items():
        rows.sort(key=lambda row: (row["_lap"], row["_lap_start"]))
        stint_starts = {}
        for row in rows:
            stint = numeric(row.get("stint"))
            if stint is not None:
                stint_starts[int(stint)] = min(stint_starts.get(int(stint), row["_lap"]), row["_lap"])
        seen_laps = set()
        for row in rows:
            lap = row["_lap"]
            if lap not in seen_laps:
                telemetry_specs.append((driver, row["_code"], lap, row["_lap_start"]))
                seen_laps.add(lap)

            stint = numeric(row.get("stint"))
            activation = row["_pit_out"] if row["_pit_out"] is not None else row["_lap_start"]
            state = {
                "lap": lap,
                "compound": normalize_text(row.get("compound")),
                "tyre_age": numeric(row.get("life")),
                "stint": int(stint) if stint is not None else None,
                "stint_length": lap - stint_starts[int(stint)] if stint is not None else None,
                "activation_time": activation,
            }
            state_by_driver[driver].append((activation, state))
            if row["_pit_out"] is not None:
                pit_out_by_driver[driver].append(row["_pit_out"])

            quality_reasons = timing_quality_reasons(row)
            if quality_reasons:
                timing_quality["excluded_rows"] += 1
                for reason in quality_reasons:
                    timing_quality[reason] += 1
                continue
            timing_quality["usable_rows"] += 1
            sector_values = {}
            for sector, value_field, event_field in (
                (1, "s1", "s1T"),
                (2, "s2", "s2T"),
                (3, "s3", "s3T"),
            ):
                value = numeric(row.get(value_field))
                event_time = tracing_event_time(row, event_field, row["_lap_start"])
                sector_values[sector] = value
                if value is not None and not 0 < value <= 60:
                    timing_quality[f"sector_{sector}_outside_0_to_60_seconds"] += 1
                    continue
                if value is not None and event_time is not None:
                    sector_by_driver[driver][sector][lap] = (event_time, value)
            lap_value = numeric(row.get("time"))
            valid_sectors = all(
                sector_values.get(sector) is not None and 0 < sector_values[sector] <= 60
                for sector in (1, 2, 3)
            )
            if lap_value is not None and not 0 < lap_value <= 180:
                timing_quality["lap_time_outside_0_to_180_seconds"] += 1
            elif not valid_sectors:
                timing_quality["lap_time_excluded_due_to_missing_or_invalid_sector"] += 1
            elif lap_value is not None and row["_lap_end"] is not None:
                lap_time_by_driver[driver][lap] = (row["_lap_end"], lap_value)

        state_by_driver[driver].sort(key=lambda item: item[0])
        pit_out_by_driver[driver].sort()

    return {
        "state": dict(state_by_driver),
        "pit_out": dict(pit_out_by_driver),
        "sectors": sector_by_driver,
        "lap_times": lap_time_by_driver,
        "telemetry_specs": telemetry_specs,
        "code_by_driver": code_by_driver,
        "timing_quality": dict(timing_quality),
    }


def fetch_telemetry_spec(config: dict, spec: tuple[int, str, int, float], audit: DownloadAudit):
    driver, code, lap, lap_start = spec
    payload = fetch_json(raw_url(config, f"{code}/{lap}_tel.json"), audit, allow_missing=True)
    if not payload or not isinstance(payload.get("tel"), dict):
        return driver, code, lap, [], [], [], []
    telemetry = payload["tel"]
    offsets = telemetry.get("time", [])
    drs_values = telemetry.get("drs", [])
    ahead_values = telemetry.get("DriverAhead", [])
    distances = telemetry.get("DistanceToDriverAhead", [])
    count = min(len(offsets), len(drs_values), len(ahead_values), len(distances))
    times, drs, ahead, distance = [], [], [], []
    for index in range(count):
        offset = numeric(offsets[index])
        if offset is None or offset < -1 or offset > 600:
            continue
        times.append(lap_start + offset)
        raw_drs = numeric(drs_values[index])
        drs.append(int(raw_drs) if raw_drs in {0.0, 1.0} else None)
        ahead.append(normalize_driver_number(ahead_values[index]))
        distance.append(numeric(distances[index]))
    return driver, code, lap, times, drs, ahead, distance


def backward_sample(times: list[float], values: list, point: float, tolerance: float):
    index = bisect.bisect_right(times, point) - 1
    if index < 0:
        return None, None
    age = point - times[index]
    if age < -1e-9 or age > tolerance:
        return None, None
    return values[index], times[index]


def latest_state(values: list[tuple[float, dict]], point: float) -> dict | None:
    if not values:
        return None
    index = bisect.bisect_right([item[0] for item in values], point) - 1
    return values[index][1] if index >= 0 else None


def latest_common_completed(
    own: dict[int, tuple[float, float]],
    rival: dict[int, tuple[float, float]],
    point: float,
):
    candidates = []
    for lap in own.keys() & rival.keys():
        own_time, own_value = own[lap]
        rival_time, rival_value = rival[lap]
        if own_time <= point and rival_time <= point:
            candidates.append((max(own_time, rival_time), lap, own_value, rival_value, own_time, rival_time))
    return max(candidates, default=None)


def recent_pit_flag(values: list[float], point: float) -> tuple[int, float | None]:
    if not values:
        return 0, None
    index = bisect.bisect_right(values, point) - 1
    if index < 0:
        return 0, None
    latest = values[index]
    return int(point - latest <= PIT_OUT_RECENT_SECONDS), latest


def set_missing(row: dict, reasons: dict[str, Counter], feature: str, reason: str) -> None:
    row[feature] = None
    reasons[feature][reason] += 1


def set_value(row: dict, feature: str, value) -> None:
    row[feature] = value


def attach_own_telemetry(
    rows: list[dict],
    config: dict,
    prepared: dict,
    audit: DownloadAudit,
    workers: int,
):
    targets: dict[int, list[tuple[float, dict]]] = defaultdict(list)
    for row in rows:
        targets[row["driver_number"]].append((row["_point"], row))
    for values in targets.values():
        values.sort(key=lambda item: item[0])

    drs_by_driver: dict[int, dict[float, tuple[int, int]]] = defaultdict(dict)
    specs = prepared["telemetry_specs"]
    print(f"  TracingInsights telemetry files scheduled: {len(specs):,}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_DOWNLOAD_WORKERS))) as pool:
        futures = [pool.submit(fetch_telemetry_spec, config, spec, audit) for spec in specs]
        for completed, future in enumerate(as_completed(futures), start=1):
            driver, _code, lap, times, drs, ahead, distance = future.result()
            if times:
                for sample_time, value in zip(times, drs):
                    if value is None:
                        continue
                    previous_drs = drs_by_driver[driver].get(sample_time)
                    if previous_drs is None or lap > previous_drs[0]:
                        drs_by_driver[driver][sample_time] = (lap, value)
                driver_targets = targets.get(driver, [])
                target_times = [item[0] for item in driver_targets]
                left = bisect.bisect_left(target_times, times[0])
                right = bisect.bisect_right(target_times, times[-1] + TELEMETRY_BACKWARD_TOLERANCE_SECONDS)
                for point, row in driver_targets[left:right]:
                    sample_index = bisect.bisect_right(times, point) - 1
                    if sample_index < 0:
                        continue
                    sample_time = times[sample_index]
                    age = point - sample_time
                    if age < -1e-9 or age > TELEMETRY_BACKWARD_TOLERANCE_SECONDS:
                        continue
                    previous = row.get("_own_tel_time")
                    previous_lap = row.get("_own_tel_lap", -1)
                    if previous is None or sample_time > previous or (sample_time == previous and lap > previous_lap):
                        row["_own_tel_time"] = sample_time
                        row["_own_tel_lap"] = lap
                        row["_ahead_driver"] = ahead[sample_index]
                        row["_distance_ahead"] = distance[sample_index]
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"    processed {completed:,}/{len(futures):,} telemetry files "
                    f"({audit.bytes / 1024 / 1024:.1f} MiB transferred)",
                    flush=True,
                )

    indexed = {}
    for driver, values in drs_by_driver.items():
        ordered = sorted(values.items())
        indexed[driver] = ([item[0] for item in ordered], [item[1][1] for item in ordered])
    return indexed


def enrich_session(session_key: int, workers: int):
    config = SESSION_CONFIG[session_key]
    print(
        f"Building {config['split']} session {session_key}: "
        f"{config['year']} {config['event']} {config['session']}",
        flush=True,
    )
    rows = build_clean_base_rows(session_key)
    audit = DownloadAudit()
    drivers_payload = fetch_json(raw_url(config, "drivers.json"), audit)
    laps_payload = fetch_json(raw_url(config, "session_laptimes.json"), audit)
    if not isinstance(drivers_payload, dict) or not isinstance(laps_payload, dict):
        raise RuntimeError("TracingInsights session metadata has an unexpected shape")
    lap_rows = records(laps_payload)
    prepared = prepare_tracing_laps(lap_rows)
    drs_index = attach_own_telemetry(rows, config, prepared, audit, workers)

    missing_reasons: dict[str, Counter] = {feature: Counter() for feature in PHASE1_COLUMNS}
    for row in rows:
        point = row["_point"]
        attacker = row["driver_number"]
        ahead = row.get("_ahead_driver")
        row["car_ahead_driver_number"] = ahead
        row["_audit"] = {
            "prediction_time": row["date"],
            "attacker_driver_number": attacker,
            "car_ahead_driver_number": ahead,
            "own_telemetry_time": iso_time(row.get("_own_tel_time")),
            "own_telemetry_age_ms": round((point - row["_own_tel_time"]) * 1000, 3)
            if row.get("_own_tel_time") is not None
            else None,
        }

        if row.get("_own_tel_time") is None:
            set_missing(row, missing_reasons, "distance_to_car_ahead_m", "no_backward_telemetry_within_500ms")
        elif ahead is None:
            set_missing(row, missing_reasons, "distance_to_car_ahead_m", "telemetry_reports_no_car_ahead")
        elif row.get("_distance_ahead") is None:
            set_missing(row, missing_reasons, "distance_to_car_ahead_m", "distance_field_missing")
        else:
            set_value(row, "distance_to_car_ahead_m", row["_distance_ahead"])
        row["_audit"]["distance_to_car_ahead_m"] = row.get("distance_to_car_ahead_m")

        own_state = latest_state(prepared["state"].get(attacker, []), point)
        ahead_state = latest_state(prepared["state"].get(ahead, []), point) if ahead is not None else None
        row["_audit"]["attacker_state"] = own_state
        row["_audit"]["rival_state"] = ahead_state
        for feature in ("opponent_tyre_age_delta", "opponent_tyre_compound_delta", "stint_length_delta"):
            if ahead is None:
                set_missing(row, missing_reasons, feature, "no_car_ahead_identity")
            elif own_state is None:
                set_missing(row, missing_reasons, feature, "attacker_tyre_state_unavailable")
            elif ahead_state is None:
                set_missing(row, missing_reasons, feature, "rival_tyre_state_unavailable")

        if row.get("opponent_tyre_age_delta") is None and ahead is not None and own_state and ahead_state:
            if own_state["tyre_age"] is None or ahead_state["tyre_age"] is None:
                missing_reasons["opponent_tyre_age_delta"]["tyre_life_field_missing"] += 1
            else:
                set_value(
                    row,
                    "opponent_tyre_age_delta",
                    own_state["tyre_age"] - ahead_state["tyre_age"],
                )
        if row.get("opponent_tyre_compound_delta") is None and ahead is not None and own_state and ahead_state:
            if own_state["compound"] is None or ahead_state["compound"] is None:
                missing_reasons["opponent_tyre_compound_delta"]["compound_field_missing"] += 1
            else:
                set_value(
                    row,
                    "opponent_tyre_compound_delta",
                    f"{own_state['compound']}__VS__{ahead_state['compound']}",
                )
        if row.get("stint_length_delta") is None and ahead is not None and own_state and ahead_state:
            if own_state["stint_length"] is None or ahead_state["stint_length"] is None:
                missing_reasons["stint_length_delta"]["stint_field_missing"] += 1
            else:
                set_value(
                    row,
                    "stint_length_delta",
                    own_state["stint_length"] - ahead_state["stint_length"],
                )

        if ahead is None:
            set_missing(row, missing_reasons, "rival_drs_open", "no_car_ahead_identity")
        elif ahead not in drs_index:
            set_missing(row, missing_reasons, "rival_drs_open", "rival_telemetry_unavailable")
        else:
            rival_times, rival_values = drs_index[ahead]
            value, source_time = backward_sample(
                rival_times,
                rival_values,
                point,
                TELEMETRY_BACKWARD_TOLERANCE_SECONDS,
            )
            if source_time is None:
                set_missing(row, missing_reasons, "rival_drs_open", "no_backward_rival_sample_within_500ms")
            else:
                set_value(row, "rival_drs_open", value)
                row["_audit"]["rival_drs_time"] = iso_time(source_time)
                row["_audit"]["rival_drs_age_ms"] = round((point - source_time) * 1000, 3)
                row["_audit"]["rival_drs_raw_binary"] = value

        for sector in (1, 2, 3):
            feature = f"relative_sector_time_delta_s{sector}"
            if ahead is None:
                set_missing(row, missing_reasons, feature, "no_car_ahead_identity")
                continue
            common = latest_common_completed(
                prepared["sectors"].get(attacker, {}).get(sector, {}),
                prepared["sectors"].get(ahead, {}).get(sector, {}),
                point,
            )
            if common is None:
                set_missing(row, missing_reasons, feature, "no_common_sector_completed_by_prediction_time")
            else:
                _, lap, own_value, ahead_value, own_time, ahead_time = common
                set_value(row, feature, own_value - ahead_value)
                row["_audit"][feature] = {
                    "source_lap": lap,
                    "attacker_seconds": own_value,
                    "rival_seconds": ahead_value,
                    "attacker_completed_at": iso_time(own_time),
                    "rival_completed_at": iso_time(ahead_time),
                }

        if ahead is None:
            set_missing(row, missing_reasons, "relative_lap_time_delta", "no_car_ahead_identity")
        else:
            common_lap = latest_common_completed(
                prepared["lap_times"].get(attacker, {}),
                prepared["lap_times"].get(ahead, {}),
                point,
            )
            if common_lap is None:
                set_missing(row, missing_reasons, "relative_lap_time_delta", "no_common_lap_completed_by_prediction_time")
            else:
                _, lap, own_value, ahead_value, own_time, ahead_time = common_lap
                set_value(row, "relative_lap_time_delta", own_value - ahead_value)
                row["_audit"]["relative_lap_time_delta"] = {
                    "source_lap": lap,
                    "attacker_seconds": own_value,
                    "rival_seconds": ahead_value,
                    "attacker_completed_at": iso_time(own_time),
                    "rival_completed_at": iso_time(ahead_time),
                }

        if own_state is None:
            set_missing(row, missing_reasons, "attacker_pit_out_recent_flag", "attacker_timing_state_unavailable")
        else:
            own_flag, own_pit_time = recent_pit_flag(prepared["pit_out"].get(attacker, []), point)
            set_value(row, "attacker_pit_out_recent_flag", own_flag)
            row["_audit"]["attacker_latest_pit_out"] = iso_time(own_pit_time)
        if ahead is None:
            set_missing(row, missing_reasons, "rival_pit_out_recent_flag", "no_car_ahead_identity")
        elif ahead_state is None:
            set_missing(row, missing_reasons, "rival_pit_out_recent_flag", "rival_timing_state_unavailable")
        else:
            ahead_flag, ahead_pit_time = recent_pit_flag(prepared["pit_out"].get(ahead, []), point)
            set_value(row, "rival_pit_out_recent_flag", ahead_flag)
            row["_audit"]["rival_latest_pit_out"] = iso_time(ahead_pit_time)

        # Every source timestamp used above is backward-only. These assertions
        # make accidental future joins fail the build rather than leak silently.
        if row.get("_own_tel_time") is not None and row["_own_tel_time"] > point + 1e-9:
            raise AssertionError("Future attacker telemetry selected")
        for audit_value in row["_audit"].values():
            if not isinstance(audit_value, dict):
                continue
            for key in ("attacker_completed_at", "rival_completed_at"):
                completed = timestamp(audit_value.get(key))
                if completed is not None and completed > point + 1e-9:
                    raise AssertionError("Future completed timing selected")

    manifest = {
        "session_key": session_key,
        "split": config["split"],
        "year": config["year"],
        "event": config["event"],
        "rows": len(rows),
        "positive_labels": sum(row[LABEL_COLUMN] for row in rows),
        "tracinginsights_requests": audit.requests,
        "tracinginsights_bytes": audit.bytes,
        "tracinginsights_missing_telemetry_files": audit.missing_files,
        "timing_quality": prepared["timing_quality"],
        "missing_reasons": {feature: dict(counter) for feature, counter in missing_reasons.items()},
    }
    return rows, manifest


def feature_summary(frame: pd.DataFrame, feature: str) -> dict:
    missing = int(frame[feature].isna().sum())
    result = {
        "rows": int(len(frame)),
        "missing_count": missing,
        "missing_rate": round(missing / len(frame), 6) if len(frame) else None,
        "available_count": int(len(frame) - missing),
    }
    values = frame[feature].dropna()
    if feature == "opponent_tyre_compound_delta":
        result["kind"] = "categorical_nominal"
        result["value_counts"] = {str(key): int(value) for key, value in values.value_counts().items()}
    else:
        result["kind"] = "numeric"
        if len(values):
            numeric_values = pd.to_numeric(values)
            result["statistics"] = {
                "mean": round(float(numeric_values.mean()), 6),
                "std": round(float(numeric_values.std()), 6),
                "min": round(float(numeric_values.min()), 6),
                "p25": round(float(numeric_values.quantile(0.25)), 6),
                "median": round(float(numeric_values.median()), 6),
                "p75": round(float(numeric_values.quantile(0.75)), 6),
                "max": round(float(numeric_values.max()), 6),
            }
    return result


def select_audit_samples(rows: list[dict]) -> dict[str, list[dict]]:
    samples = {feature: [] for feature in PHASE1_COLUMNS}
    for feature in PHASE1_COLUMNS:
        candidates = [row for row in rows if row.get(feature) is not None]
        if not candidates:
            continue
        selected = [candidates[0], candidates[len(candidates) // 2], candidates[-1]]
        if feature in {"rival_drs_open", "attacker_pit_out_recent_flag", "rival_pit_out_recent_flag"}:
            positive = next((row for row in candidates if row.get(feature) == 1), None)
            zero = next((row for row in candidates if row.get(feature) == 0), None)
            selected = [row for row in (positive, zero, candidates[-1]) if row is not None]
        unique = []
        seen = set()
        for row in selected:
            key = (row["session_key"], row["driver_number"], row["date"])
            if key not in seen:
                unique.append(row)
                seen.add(key)
        for row in unique:
            audit = row["_audit"]
            if feature in {"opponent_tyre_age_delta", "opponent_tyre_compound_delta", "stint_length_delta"}:
                def serial_state(state):
                    if not state:
                        return state
                    return {
                        **state,
                        "activation_time": iso_time(state.get("activation_time")),
                    }
                raw = {
                    "attacker_state": serial_state(audit.get("attacker_state")),
                    "rival_state": serial_state(audit.get("rival_state")),
                }
            elif feature == "distance_to_car_ahead_m":
                raw = {
                    "own_telemetry_time": audit.get("own_telemetry_time"),
                    "own_telemetry_age_ms": audit.get("own_telemetry_age_ms"),
                    "DriverAhead": audit.get("car_ahead_driver_number"),
                    "DistanceToDriverAhead": audit.get("distance_to_car_ahead_m"),
                }
            elif feature == "rival_drs_open":
                raw = {
                    "rival_drs_time": audit.get("rival_drs_time"),
                    "rival_drs_age_ms": audit.get("rival_drs_age_ms"),
                    "rival_drs_raw_binary": audit.get("rival_drs_raw_binary"),
                }
            elif feature.startswith("relative_sector") or feature == "relative_lap_time_delta":
                raw = audit.get(feature)
            elif feature == "attacker_pit_out_recent_flag":
                raw = {"latest_pit_out": audit.get("attacker_latest_pit_out"), "window_seconds": PIT_OUT_RECENT_SECONDS}
            else:
                raw = {"latest_pit_out": audit.get("rival_latest_pit_out"), "window_seconds": PIT_OUT_RECENT_SECONDS}
            samples[feature].append(
                {
                    "session_key": row["session_key"],
                    "prediction_time": row["date"],
                    "attacker_driver_number": row["driver_number"],
                    "car_ahead_driver_number": row.get("car_ahead_driver_number"),
                    "feature_value": row[feature],
                    "raw_inputs": raw,
                }
            )
    return samples


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.workers > MAX_DOWNLOAD_WORKERS:
        raise SystemExit(f"--workers must be between 1 and {MAX_DOWNLOAD_WORKERS}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    session_manifests = []
    for session_key in (*TRAIN_SESSIONS, VALIDATION_SESSION):
        rows, manifest = enrich_session(session_key, args.workers)
        all_rows.extend(rows)
        session_manifests.append(manifest)
        print(
            f"  completed {manifest['rows']:,} rows; "
            f"downloaded {manifest['tracinginsights_bytes'] / 1024 / 1024:.1f} MiB",
            flush=True,
        )

    output_columns = IDENTIFIER_COLUMNS + BASE_FEATURES + PHASE1_COLUMNS + [LABEL_COLUMN]
    frame = pd.DataFrame([{column: row.get(column) for column in output_columns} for row in all_rows])
    frame["opponent_tyre_compound_delta"] = frame["opponent_tyre_compound_delta"].astype("string")
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()

    train_path = args.output_dir / "train_phase1_features.csv.gz"
    validation_path = args.output_dir / "validation_phase1_features.csv.gz"
    train.to_csv(train_path, index=False, compression="gzip")
    validation.to_csv(validation_path, index=False, compression="gzip")

    combined_reasons: dict[str, Counter] = {feature: Counter() for feature in PHASE1_COLUMNS}
    for session in session_manifests:
        for feature, values in session["missing_reasons"].items():
            combined_reasons[feature].update(values)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "feature engineering only; no model training, fitting, scoring, or artifact modification",
        "protocol": {
            "train_sessions": list(TRAIN_SESSIONS),
            "validation_session": VALIDATION_SESSION,
            "session_override_supported": False,
            "openf1_cache_mode": "read_only",
            "tracinginsights_raw_persistence": "none; each telemetry response was processed in memory",
        },
        "join_policy": {
            "session": "fixed year + event + Race mapping, verified from OpenF1 session metadata",
            "driver": "OpenF1 driver_number equals TracingInsights dNum / DriverAhead",
            "lap": "TracingInsights lap; current state activates at lSD, except a new stint activates at pout",
            "telemetry_timestamp": "latest TracingInsights sample at or before the OpenF1 interval timestamp",
            "telemetry_backward_tolerance_ms": int(TELEMETRY_BACKWARD_TOLERANCE_SECONDS * 1000),
            "interpolation": "none",
            "pit_recent_window_seconds": PIT_OUT_RECENT_SECONDS,
            "relative_timing": "latest common lap/sector completed by both cars at or before prediction time",
        },
        "phase1_feature_groups": PHASE1_FEATURE_GROUPS,
        "model_feature_columns": BASE_FEATURES + PHASE1_COLUMNS,
        "compound_encoding": {
            "storage": "nominal string ATTACKER__VS__RIVAL",
            "training_requirement": "one-hot encode or use explicit native categorical handling; never treat category text/IDs as ordinal",
        },
        "outputs": {
            "train": str(train_path.relative_to(ROOT)),
            "validation": str(validation_path.relative_to(ROOT)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "columns": output_columns,
        },
        "session_manifests": session_manifests,
        "feature_summary": {
            "train": {feature: feature_summary(train, feature) for feature in PHASE1_COLUMNS},
            "validation": {feature: feature_summary(validation, feature) for feature in PHASE1_COLUMNS},
            "combined": {feature: feature_summary(frame, feature) for feature in PHASE1_COLUMNS},
        },
        "missing_reasons_combined": {
            feature: dict(counter) for feature, counter in combined_reasons.items()
        },
        "real_samples": select_audit_samples(all_rows),
        "causal_safety": {
            "telemetry": "backward-only; future samples are never selected",
            "tyres_and_stints": "state activates no earlier than lap start and, for a new stint, no earlier than pit-out",
            "sectors": "sector value is used only after both cars' sector-completion timestamps",
            "lap_time": "lap duration is used only after both cars completed the same lap",
            "pit_flags": "only pit-out events at or before prediction time are considered",
            "runtime_assertions": "the build fails if any selected telemetry or timing completion timestamp is in the future",
        },
    }
    manifest_path = args.output_dir / "phase1_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Saved train dataset: {train_path} ({len(train):,} rows)")
    print(f"Saved validation dataset: {validation_path} ({len(validation):,} rows)")
    print(f"Saved audit manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
