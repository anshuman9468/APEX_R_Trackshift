#!/usr/bin/env python3
"""Serve the APEX-R frontend with the frozen hybrid model and live bridge.

This adapter deliberately does not import or execute the cloned repository's
backend, XGBoost artifact, frozen GNN proxy, or phase-5 model. It loads the
local Phase 3 test bundle and polls MultiViewer's local Live Timing API while
keeping observed telemetry separate from the simulator's modelled branches.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import json
import math
import os
import re
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import torch

try:
    import pyarrow.parquet as parquet
except ModuleNotFoundError:  # the live bridge can still run without local replay tooling
    parquet = None

try:
    from ml_pipeline.train_hybrid import (
        GraphPartition,
        HybridGNN,
        Normalizer,
        choose_device,
        collate,
        move_batch,
    )
except ModuleNotFoundError:  # direct execution: python ml_pipeline/trackshift_server.py
    from train_hybrid import GraphPartition, HybridGNN, Normalizer, choose_device, collate, move_batch


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "dist"
BUNDLE_PATH = ROOT / "deployment_packages" / "hybrid_gnn_epoch51" / "model_test_bundle.pt"


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value)))))


def json_scalar(value):
    """Convert NumPy scalars and evaluation NaNs into JSON-safe values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


MULTIVIEWER_ENDPOINTS = (
    "Heartbeat",
    "SessionInfo",
    "DriverList",
    "TimingData",
    "CarData",
    "TimingAppData",
    "TimingStats",
    "RaceControlMessages",
    "LapCount",
    "TrackStatus",
    "ExtrapolatedClock",
    "SessionData",
    "WeatherData",
)


def _field(value, *names, default=None):
    """Read a field case-insensitively from one of MultiViewer's payload shapes."""
    if not isinstance(value, dict):
        return default
    for name in names:
        for key, candidate in value.items():
            if str(key).lower() == str(name).lower():
                return candidate
    return default


def _number(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("Value", "value", "Seconds", "seconds", "Time", "time"):
            candidate = _field(value, key)
            if candidate is not None:
                return _number(candidate)
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            try:
                parsed = float(match.group(0))
                return parsed if math.isfinite(parsed) else None
            except ValueError:
                return None
    return None


def _duration_seconds(value):
    """Parse MultiViewer duration values such as ``01:23.456`` safely."""
    if isinstance(value, dict):
        for key in ("Value", "value", "Time", "time", "Seconds", "seconds"):
            candidate = _field(value, key)
            if candidate is not None:
                return _duration_seconds(candidate)
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (float(part) for part in parts)
            return hours * 3600.0 + minutes * 60.0 + seconds
        if len(parts) == 2:
            minutes, seconds = (float(part) for part in parts)
            return minutes * 60.0 + seconds
        parsed = float(text.replace(",", ""))
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return _number(text)


def _integer(value):
    parsed = _number(value)
    return int(round(parsed)) if parsed is not None else None


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "open", "active", "rain"}
    return None


def _records(payload, collection_names=()):
    """Turn list, map, and wrapper responses into records while retaining map keys."""
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if not isinstance(payload, dict):
        return []
    for name in collection_names:
        value = _field(payload, name)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
        if isinstance(value, dict):
            records = []
            for map_key, record in value.items():
                if isinstance(record, dict):
                    item = dict(record)
                    item["_map_key"] = str(map_key)
                    records.append(item)
            if records:
                return records
    if payload and all(isinstance(candidate, dict) for candidate in payload.values()):
        return [dict(record, _map_key=str(map_key)) for map_key, record in payload.items()]
    return [payload]


def _endpoint_records(payload):
    records = _records(payload, ("Cars", "cars", "Lines", "lines", "Entries", "entries", "Data", "data", "Items", "items"))
    expanded = []
    for record in records:
        nested = _records(record, ("Cars", "cars"))
        if nested != [record]:
            expanded.extend(nested)
        else:
            expanded.append(record)
    return expanded


def _driver_maps(payload):
    by_key = {}
    for record in _endpoint_records(payload):
        code = _field(record, "TLA", "Tla", "Abbreviation", "Driver", "DriverCode", "ShortName", "Name")
        number = _field(record, "RacingNumber", "Number", "CarNumber", "_map_key")
        if isinstance(code, dict):
            code = _field(code, "TLA", "Tla", "Abbreviation", "Code", "Name")
        if code is None:
            continue
        code = str(code).upper()
        for key in (number, code):
            if key is not None:
                by_key[str(key)] = code
    return by_key


def _driver_details(payload):
    """Return exact MultiViewer roster metadata keyed by code and racing number."""
    details = {}
    for record in _endpoint_records(payload):
        code = _field(record, "TLA", "Tla", "Abbreviation", "Driver", "DriverCode", "ShortName")
        number = _field(record, "RacingNumber", "Number", "CarNumber", "_map_key")
        if code is None and number is None:
            continue
        code = str(code or number).upper()
        detail = {
            "driver": code,
            "driver_number": str(number) if number is not None else None,
            "name": _field(record, "FullName", "FullNameDisplay", "Name", "BroadcastName"),
            "first_name": _field(record, "FirstName", "FirstNameDisplay"),
            "last_name": _field(record, "LastName", "LastNameDisplay"),
            "team": _field(record, "TeamName", "Team"),
            "team_colour": _field(record, "TeamColour", "TeamColor"),
            "line": _integer(_field(record, "Line")),
        }
        detail = {key: value for key, value in detail.items() if value is not None}
        for key in (code, str(number) if number is not None else None):
            if key is not None:
                details[str(key).upper()] = detail
    return details


def _driver_code(record, driver_map):
    token = _field(record, "TLA", "Tla", "Abbreviation", "Driver", "DriverCode", "RacingNumber", "Number", "CarNumber", "_map_key")
    if isinstance(token, dict):
        token = _field(token, "TLA", "Tla", "Abbreviation", "Code", "Name")
    if token is None:
        return None
    token = str(token)
    return driver_map.get(token, driver_map.get(token.upper(), token.upper() if len(token) <= 4 else token))


def _stint_values(record):
    stints = _field(record, "Stints", "stints", "TyreStints", "tyreStints")
    if isinstance(stints, dict):
        stints = list(stints.values())
    if isinstance(stints, list) and stints:
        return stints[-1] if isinstance(stints[-1], dict) else {}
    return record


def _sector_rows(record):
    sectors = _field(record, "Sectors", "sectors")
    if not isinstance(sectors, list):
        return []
    rows = []
    for index, sector in enumerate(sectors, start=1):
        if not isinstance(sector, dict):
            continue
        row = {
            "sector": index,
            "time_s": _duration_seconds(_field(sector, "Value", "Time", "time")),
            "status": _field(sector, "Status", "status"),
            "overall_fastest": _boolean(_field(sector, "OverallFastest", "overallFastest")),
            "personal_fastest": _boolean(_field(sector, "PersonalFastest", "personalFastest")),
            "stopped": _boolean(_field(sector, "Stopped", "stopped")),
        }
        row = {key: value for key, value in row.items() if value is not None}
        rows.append(row)
    return rows


def _flatten_race_control(payload):
    """Flatten the Messages wrapper used by the local Live Timing API."""
    messages = []
    if isinstance(payload, dict):
        direct = _field(payload, "Messages", "messages")
        if isinstance(direct, list):
            return [record for record in direct if isinstance(record, dict)]
        for record in _endpoint_records(payload):
            nested = _field(record, "Messages", "messages")
            if isinstance(nested, list):
                messages.extend(item for item in nested if isinstance(item, dict))
    elif isinstance(payload, list):
        messages.extend(record for record in payload if isinstance(record, dict))
    return messages


def normalise_multiviewer_payload(payloads: dict, focus_driver: str | None = None) -> dict:
    """Normalize the local MultiViewer streams into the APEX-R live twin contract."""
    driver_map = _driver_maps(payloads.get("DriverList"))
    driver_details = _driver_details(payloads.get("DriverList"))
    cars = {}

    def car_for(record):
        code = _driver_code(record, driver_map)
        if not code:
            return None
        car = cars.setdefault(code, {"driver": code})
        detail = driver_details.get(code) or driver_details.get(str(_field(record, "RacingNumber", "Number", "CarNumber", "_map_key")).upper())
        if detail:
            for key, value in detail.items():
                if value is not None:
                    car.setdefault(key, value)
        return car

    for record in _endpoint_records(payloads.get("TimingData")):
        car = car_for(record)
        if not car:
            continue
        position = _integer(_field(record, "Position", "position", "Place"))
        gap = _number(_field(record, "GapToLeader", "gapToLeader", "GapToLeaderTime"))
        interval = _number(_field(record, "IntervalToPositionAhead", "intervalToPositionAhead", "Interval"))
        stats = _records(_field(record, "Stats", "stats"))
        stat = stats[0] if stats else {}
        if gap is None:
            gap = _number(_field(stat, "TimeDiffToFastest", "TimeDiffToLeader"))
        if interval is None:
            interval = _number(_field(stat, "TimeDifftoPositionAhead", "TimeDiffToPositionAhead"))
        if position is not None:
            car["position"] = position
        if gap is not None:
            car["gap_to_leader_s"] = gap
        if interval is not None:
            car["interval_to_ahead_s"] = interval
        last_lap = _duration_seconds(_field(record, "LastLapTime", "lastLapTime"))
        if last_lap is not None:
            car["last_lap_time_s"] = last_lap
        sector_rows = _sector_rows(record)
        if sector_rows:
            car["sectors"] = sector_rows
            car["sector_times_s"] = [row.get("time_s") for row in sector_rows]
        current_sector = _integer(_field(record, "CurrentSector", "currentSector", "Sector"))
        if current_sector is not None:
            car["current_sector"] = current_sector

    for record in _endpoint_records(payloads.get("CarData")):
        car = car_for(record)
        if not car:
            continue
        mappings = {
            "speed_kmh": ("Speed", "speed", "SpeedKmh"),
            "throttle_pct": ("Throttle", "throttle", "ThrottlePct"),
            "brake_pct": ("Brake", "brake", "BrakePct"),
            "gear": ("Gear", "gear", "NGear"),
            "rpm": ("RPM", "Rpm", "rpm"),
        }
        for target, names in mappings.items():
            value = _number(_field(record, *names))
            if value is not None:
                car[target] = value
        channels = _field(record, "Channels", "channels")
        if isinstance(channels, dict):
            channel_map = {"rpm": "0", "speed_kmh": "2", "gear": "3", "throttle_pct": "4", "brake_pct": "5", "drs_status": "45"}
            for target, channel in channel_map.items():
                value = _number(_field(channels, channel))
                if value is not None:
                    car[target] = value
        drs = _field(record, "DRS", "Drs", "drs")
        if drs is not None:
            car["drs"] = _boolean(drs)
        elif car.get("drs_status") is not None:
            car["drs"] = bool(car["drs_status"] in {1, 10, 12, 14})

    for record in _endpoint_records(payloads.get("TimingAppData")):
        car = car_for(record)
        if not car:
            continue
        stint = _stint_values(record)
        compound = _field(stint, "Compound", "compound", "Tyre", "tyre")
        age = _integer(_field(stint, "TotalLaps", "Laps", "Age", "tyreAge"))
        pit_stops = _integer(_field(record, "PitStops", "pitStops", "Stops"))
        if compound is not None:
            car["tyre_compound"] = str(compound).upper()
        if age is not None:
            car["tyre_age_laps"] = age
        if pit_stops is not None:
            car["pit_stops"] = pit_stops
        for target, names in {
            "in_pit": ("InPit", "inPit"),
            "pit_out": ("PitOut", "pitOut"),
            "retired": ("Retired", "retired"),
            "stopped": ("Stopped", "stopped"),
        }.items():
            value = _boolean(_field(record, *names))
            if value is not None:
                car[target] = value

    car_rows = sorted(cars.values(), key=lambda row: (row.get("position", 999), row["driver"]))
    for row in car_rows:
        position = row.get("position")
        if position is None:
            continue
        ahead_row = next((candidate for candidate in car_rows if candidate.get("position") == position - 1), None)
        behind_row = next((candidate for candidate in car_rows if candidate.get("position") == position + 1), None)
        if row.get("interval_to_ahead_s") is not None:
            row["gap_to_ahead_s"] = row["interval_to_ahead_s"]
        if behind_row and behind_row.get("interval_to_ahead_s") is not None:
            row["gap_behind_s"] = behind_row["interval_to_ahead_s"]
        if ahead_row:
            row["driver_ahead"] = ahead_row["driver"]
        if behind_row:
            row["driver_behind"] = behind_row["driver"]
    focus = None
    if focus_driver:
        focus_upper = focus_driver.upper()
        resolved_focus = driver_map.get(str(focus_driver), driver_map.get(focus_upper, focus_upper))
        focus = next((row for row in car_rows if row["driver"] in {focus_upper, resolved_focus}), None)
    if focus is None and car_rows:
        focus = car_rows[0]
    if focus:
        focus_position = focus.get("position")
        behind = next((row for row in car_rows if focus_position is not None and row.get("position") == focus_position + 1), None)
        if behind and behind.get("interval_to_ahead_s") is not None:
            focus["interval_behind_s"] = behind["interval_to_ahead_s"]

    weather_records = _endpoint_records(payloads.get("WeatherData"))
    weather_row = weather_records[0] if weather_records else {}
    rainfall = _boolean(_field(weather_row, "Rainfall", "rainfall", "RainfallActive", "IsRaining"))
    if rainfall is None:
        rainfall_value = _number(_field(weather_row, "Rainfall", "rainfall"))
        rainfall = rainfall_value if rainfall_value is not None else 0.0
    weather = {
        "air_temperature_c": _number(_field(weather_row, "AirTemp", "AirTemperature", "airTemperature")),
        "track_temperature_c": _number(_field(weather_row, "TrackTemp", "TrackTemperature", "trackTemperature")),
        "humidity_pct": _number(_field(weather_row, "Humidity", "humidity")),
        "rainfall": float(rainfall),
        "wind_speed_kmh": _number(_field(weather_row, "WindSpeed", "windSpeed")),
        "wind_direction_deg": _number(_field(weather_row, "WindDirection", "windDirection")),
    }
    weather = {key: value for key, value in weather.items() if value is not None}

    session_records = _endpoint_records(payloads.get("SessionInfo"))
    session_row = session_records[0] if session_records else {}
    meeting = _field(session_row, "Meeting") or {}
    circuit = _field(meeting, "Circuit") or {}
    country = _field(meeting, "Country") or {}
    session = {
        "name": _field(session_row, "Name", "SessionName", "MeetingName", "EventName"),
        "type": _field(session_row, "Type", "SessionType"),
        "status": _field(session_row, "Status", "SessionStatus"),
        "key": _field(session_row, "Key", "SessionKey"),
        "meeting_key": _field(meeting, "Key"),
        "meeting_name": _field(meeting, "Name", "MeetingName", "EventName"),
        "official_name": _field(meeting, "OfficialName"),
        "location": _field(meeting, "Location"),
        "country": _field(country, "Name") or _field(session_row, "Country", "CountryName"),
        "country_code": _field(country, "Code"),
        "circuit_key": _field(circuit, "Key"),
        "circuit_name": _field(circuit, "Name", "ShortName"),
        "start_utc": _field(session_row, "StartDate", "StartTime"),
        "end_utc": _field(session_row, "EndDate", "EndTime"),
        "path": _field(session_row, "Path"),
    }
    session = {key: value for key, value in session.items() if value is not None}

    lap_records = _endpoint_records(payloads.get("LapCount"))
    lap_row = lap_records[0] if lap_records else {}
    current_lap = _integer(_field(lap_row, "CurrentLap", "currentLap", "Lap", "lap"))
    total_laps = _integer(_field(lap_row, "TotalLaps", "totalLaps", "Total"))
    status_records = _endpoint_records(payloads.get("TrackStatus"))
    track_row = status_records[0] if status_records else {}
    status_code = _field(track_row, "Status", "StatusCode", "TrackStatus", "trackStatus")
    status_message = _field(track_row, "Message", "StatusMessage", "TrackStatusMessage", "trackStatusMessage")
    status_text = " ".join(str(value) for value in (status_code, status_message) if value is not None).upper()
    track_status = {
        "code": status_code,
        "message": status_message,
        "safety_car_active": "SAFETY CAR" in status_text or str(status_code) in {"4"},
        "vsc_active": "VSC" in status_text or str(status_code) in {"6", "7"},
        "caution_active": any(token in status_text for token in ("SAFETY", "VSC", "YELLOW", "RED", "DOUBLE")),
    }
    track_status = {key: value for key, value in track_status.items() if value is not None}

    race_control = _flatten_race_control(payloads.get("RaceControlMessages"))
    clock_records = _endpoint_records(payloads.get("ExtrapolatedClock"))
    clock_row = clock_records[0] if clock_records else {}
    clock = {
        "remaining_s": _duration_seconds(_field(clock_row, "Remaining", "RemainingTime", "SessionTimeRemaining")),
        "extrapolated": _boolean(_field(clock_row, "Extrapolated", "extrapolated")),
    }
    clock = {key: value for key, value in clock.items() if value is not None}
    focus_sectors = (focus or {}).get("sectors", []) if focus else []
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "lap": {key: value for key, value in {"current": current_lap, "total": total_laps}.items() if value is not None},
        "track_status": track_status,
        "weather": weather,
        "clock": clock,
        "circuit": {
            "name": session.get("circuit_name") or session.get("meeting_name") or session.get("name"),
            "meeting": session.get("meeting_name"),
            "location": session.get("location"),
            "key": session.get("circuit_key"),
            "geometry_available": False,
        },
        "sectors": {
            "count": len(focus_sectors) or 3,
            "driver": focus.get("driver") if focus else None,
            "current": focus.get("current_sector") if focus else None,
            "observed": focus_sectors,
        },
        "cars": car_rows,
        "focus_car": focus,
        "race_control_messages": race_control[-20:],
        "available_streams": sorted(name for name, value in payloads.items() if value is not None),
        "live_timing_active": bool(car_rows),
    }


class MultiViewerBridge:
    """Poll MultiViewer's local Live Timing API and retain a bounded test capture."""

    def __init__(self, base_url: str, poll_ms: int, capture_path: Path | None, focus_driver: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.poll_seconds = max(0.25, poll_ms / 1000.0)
        self.capture_path = capture_path
        self.focus_driver = focus_driver
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.history = deque(maxlen=600)
        self.latest = {"status": "waiting", "source": "multiviewer_local_api", "connected": False, "live_timing_active": False, "latest": None}
        self.endpoint_status = {name: {"ok": False, "http_status": None, "error": "not polled"} for name in MULTIVIEWER_ENDPOINTS}
        self.last_raw = {}
        self.samples = 0
        self.thread = threading.Thread(target=self._poll_loop, name="apex-multiviewer", daemon=True)
        self.thread.start()

    def _get_endpoint(self, name: str):
        request = Request(f"{self.base_url}/{name}", headers={"Accept": "application/json", "User-Agent": "APEX-R/1.0"})
        try:
            with urlopen(request, timeout=1.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return name, payload, {"ok": True, "http_status": response.status, "error": None}
        except HTTPError as error:
            return name, None, {"ok": False, "http_status": error.code, "error": f"HTTP {error.code}"}
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            return name, None, {"ok": False, "http_status": None, "error": str(error)}

    def _poll_once(self):
        payloads, endpoint_status = {}, {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._get_endpoint, name) for name in MULTIVIEWER_ENDPOINTS]
            for future in as_completed(futures):
                name, payload, endpoint = future.result()
                endpoint_status[name] = endpoint
                if endpoint["ok"]:
                    payloads[name] = payload
        normalized = normalise_multiviewer_payload(payloads, self.focus_driver)
        api_reachable = any(entry["ok"] for entry in endpoint_status.values())
        live_active = normalized["live_timing_active"]
        latest_error = None if live_active else ("MultiViewer is reachable but Live Timing has not emitted car data" if api_reachable else "MultiViewer local API is unreachable; open MultiViewer and start Live Timing")
        normalized["status"] = "available" if live_active else ("waiting" if api_reachable else "unavailable")
        normalized["source"] = "multiviewer_local_api"
        normalized["api_reachable"] = api_reachable
        normalized["connected"] = live_active
        normalized["error"] = latest_error
        normalized["base_url"] = self.base_url
        with self.lock:
            self.endpoint_status = endpoint_status
            self.last_raw = payloads
            self.latest = normalized
            if live_active:
                self.history.append({
                    "captured_at_utc": normalized["captured_at_utc"],
                    "session": normalized["session"],
                    "lap": normalized["lap"],
                    "circuit": normalized["circuit"],
                    "sectors": normalized["sectors"],
                    "weather": normalized["weather"],
                    "track_status": normalized["track_status"],
                    "focus_car": normalized["focus_car"],
                    "cars": normalized["cars"],
                })
                self.samples += 1
        if live_active and self.capture_path:
            self.capture_path.parent.mkdir(parents=True, exist_ok=True)
            with self.capture_path.open("a", encoding="utf-8") as handle:
                json.dump({"captured_at_utc": normalized["captured_at_utc"], "normalized": normalized, "raw": payloads}, handle, allow_nan=False)
                handle.write("\n")

    def _poll_loop(self):
        while not self.stop_event.is_set():
            try:
                self._poll_once()
            except Exception as error:  # keep the local app alive if one payload changes shape
                with self.lock:
                    self.latest = {"status": "unavailable", "source": "multiviewer_local_api", "connected": False, "live_timing_active": False, "error": str(error), "base_url": self.base_url, "latest": None}
            self.stop_event.wait(self.poll_seconds)

    def status(self) -> dict:
        with self.lock:
            latest = dict(self.latest)
            endpoint_status = dict(self.endpoint_status)
            samples = self.samples
            history_size = len(self.history)
        return {
            "status": latest.get("status", "waiting"),
            "source": "multiviewer_local_api",
            "base_url": self.base_url,
            "api_reachable": bool(latest.get("api_reachable")),
            "connected": bool(latest.get("connected")),
            "live_timing_active": bool(latest.get("live_timing_active")),
            "last_update_utc": latest.get("captured_at_utc"),
            "last_error": latest.get("error"),
            "samples_captured": samples,
            "rolling_history_samples": history_size,
            "capture_path": str(self.capture_path) if self.capture_path else None,
            "endpoints": endpoint_status,
        }

    def snapshot(self) -> dict:
        with self.lock:
            latest = dict(self.latest)
            history = list(self.history)
        return {**latest, "history": history}

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)


TEST_YEARS = (2024, 2025)
TEST_SPLITS = ("train", "validation", "test")


def local_test_catalog() -> dict:
    """Describe the prepared 2024/2025 event data without loading telemetry into memory."""
    seasons = []
    for year in TEST_YEARS:
        split_rows = {}
        for split in TEST_SPLITS:
            snapshot_path = ROOT / "gnn_sequences" / str(year) / split / "snapshots.jsonl"
            manifest_path = snapshot_path.parent / "manifest.json"
            events = {}
            if snapshot_path.is_file():
                for line in snapshot_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    event = str(row.get("event", "Unknown event"))
                    events.setdefault(event, set()).add(str(row.get("session", "Unknown session")))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            split_rows[split] = {
                "events": [{"name": event, "sessions": sorted(sessions)} for event, sessions in sorted(events.items())],
                "sequences": manifest.get("counts", {}).get("sequences"),
                "target_rows": manifest.get("counts", {}).get("target_rows"),
            }
        seasons.append({"year": year, "splits": split_rows})
    return {"status": "available", "source": "local prepared_data_all and gnn_sequences", "seasons": seasons}


def local_test_replay(year: int, split: str, event: str, session: str | None, driver: str | None, limit: int = 900) -> dict:
    """Return a bounded, single-driver sample from the prepared historical telemetry."""
    if parquet is None:
        raise RuntimeError("pyarrow is required for local 2024/2025 test replays")
    if year not in TEST_YEARS or split not in TEST_SPLITS:
        raise ValueError("year must be 2024 or 2025 and split must be train, validation, or test")
    limit = max(2, min(5000, int(limit)))
    path = ROOT / "prepared_data_all" / str(year) / split / "telemetry.parquet"
    if not path.is_file():
        raise ValueError(f"prepared telemetry not found for {year} {split}")
    columns = ["season", "event", "session", "driver", "time_s", "speed_kmh", "top_speed_kmh", "gap_to_driver_ahead_m"]
    selected_driver = driver.upper() if driver else None
    selected_session = session
    records = []
    available_sessions = set()
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=10000, columns=columns):
        for row in batch.to_pylist():
            if row.get("event") != event:
                continue
            row_session = str(row.get("session") or "")
            available_sessions.add(row_session)
            if selected_session is None:
                selected_session = row_session
            if row_session != selected_session:
                continue
            row_driver = str(row.get("driver") or "").upper()
            if selected_driver is None:
                selected_driver = row_driver
            if row_driver != selected_driver:
                continue
            records.append({
                "time_s": json_scalar(row.get("time_s")),
                "speed_kmh": json_scalar(row.get("speed_kmh")),
                "top_speed_kmh": json_scalar(row.get("top_speed_kmh")),
                "gap_to_driver_ahead_m": json_scalar(row.get("gap_to_driver_ahead_m")),
                "driver": row_driver,
            })
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break
    if len(records) < 2:
        detail = f"; available sessions: {sorted(available_sessions)}" if available_sessions else ""
        raise ValueError(f"no telemetry sample found for {year} {event} / {session or 'any session'} / {selected_driver or 'any driver'}{detail}")
    return {
        "status": "available",
        "source": str(path.relative_to(ROOT)),
        "year": year,
        "split": split,
        "event": event,
        "session": selected_session,
        "driver": selected_driver,
        "records": records,
        "sample_count": len(records),
        "relative_time": True,
        "note": "Recorded prepared telemetry; time_s is relative to the source session. Energy, rival branches and outcomes remain simulated.",
    }


class BundlePartition:
    """Expose the bundled graph arrays through the existing model sampler."""

    def __init__(self, bundle: dict):
        data = bundle["graph_dataset"]
        self.path = ROOT / "gnn_sequences" / "2026" / "validation"
        self.season = int(bundle["test_definition"]["year"])
        self.split = bundle["test_definition"]["split"]
        self.x = data["x"]
        self.edge_index = data["edge_index"]
        self.edge_attr = data["edge_attr"]
        self.snapshot_ptr = data["snapshot_ptr"]
        self.edge_ptr = data["edge_ptr"]
        self.sequence_snapshot_ids = data["sequence_snapshot_ids"]
        self.target_ptr = data["target_ptr"]
        self.target_node_indices = data["target_node_indices"]
        self.target_time = data["target_lap_time_s"]
        self.target_delta = data["target_lap_time_delta_s"]
        self.target_wear = data["target_tyre_wear_estimate_fraction"]
        self.target_valid = data["target_valid"]
        self.target_high_tyre = data["target_high_tyre_degradation"]
        self.target_pit = data["target_pit_stop_observed"]
        self.target_safety = data["target_safety_constraint"]
        self.target_class_valid = data["target_classification_valid"]
        self.node_driver_index = data["node_driver_index"]
        self.snapshots = bundle["snapshots"]

    def __len__(self) -> int:
        return len(self.sequence_snapshot_ids)

    def sample(self, sequence_id: int) -> dict:
        # Reuse the audited production sampler from GraphPartition without
        # reading a second on-disk dataset or any future target values.
        sampler = object.__new__(GraphPartition)
        sampler.path = self.path
        sampler.season = self.season
        sampler.split = self.split
        sampler.x = self.x
        sampler.edge_index = self.edge_index
        sampler.edge_attr = self.edge_attr
        sampler.snapshot_ptr = self.snapshot_ptr
        sampler.edge_ptr = self.edge_ptr
        sampler.sequence_snapshot_ids = self.sequence_snapshot_ids
        sampler.target_ptr = self.target_ptr
        sampler.target_node_indices = self.target_node_indices
        sampler.target_time = self.target_time
        sampler.target_delta = self.target_delta
        sampler.target_wear = self.target_wear
        sampler.target_valid = self.target_valid
        sampler.target_high_tyre = self.target_high_tyre
        sampler.target_pit = self.target_pit
        sampler.target_safety = self.target_safety
        sampler.target_class_valid = self.target_class_valid
        sampler.node_driver_index = self.node_driver_index
        sampler.snapshots = self.snapshots
        return sampler.sample(sequence_id)


class HybridRuntime:
    def __init__(self, bundle_path: Path):
        self.bundle_path = bundle_path
        self.bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        config = self.bundle["config"]
        normalizer_config = config["normalizer"]
        self.normalizer = Normalizer(
            np.asarray(normalizer_config["x_mean"], dtype=np.float32),
            np.asarray(normalizer_config["x_std"], dtype=np.float32),
            np.asarray(normalizer_config["edge_mean"], dtype=np.float32),
            np.asarray(normalizer_config["edge_std"], dtype=np.float32),
        )
        requested_device = os.environ.get("APEX_TRACKSHIFT_DEVICE", "auto")
        self.device = choose_device(requested_device)
        self.model = HybridGNN(
            hidden=int(config.get("hidden", 128)),
            layers=int(config.get("layers", 3)),
            heads=int(config.get("heads", 4)),
            dropout=float(config.get("dropout", 0.15)),
        ).to(self.device)
        self.model.load_state_dict(self.bundle["model"], strict=True)
        self.model.eval()
        self.partition = BundlePartition(self.bundle)
        self.partition_cache = {(self.partition.season, self.partition.split): self.partition}
        self.rows = self.bundle["predictions_and_labels"]
        self.calibration = self.bundle["calibration"]
        self.driver_vocabulary = self.bundle["graph_manifest"].get("driver_vocabulary", [])
        self.report = self.bundle["replay_report"]

    def status(self) -> dict:
        first_run = self.report["replay_runs"][0]
        test_definition = dict(self.bundle["test_definition"])
        test_definition["prediction_file_source"] = "embedded in model_test_bundle.pt"
        test_definition["graph_directory_source"] = "embedded graph_dataset in model_test_bundle.pt"
        test_definition.pop("prediction_file", None)
        test_definition.pop("graph_directory", None)
        return {
            "status": "available",
            "model_version": "hybrid_gnn_physics_epoch51",
            "model_type": "GNN + GRU + physics consistency",
            "epoch": self.bundle["best_epoch"],
            "device": str(self.device),
            "test": test_definition,
            "metrics": first_run["metrics"],
            "replay_decision": self.report["deployment_decision"],
            "bundle": self.bundle_path.name,
        }

    def _calibrated(self, logit: float, head: str) -> float:
        calibration = self.calibration[head]
        return sigmoid(float(calibration["platt_intercept"]) + float(calibration["platt_coefficient"]) * logit)

    def _driver_name(self, index: int) -> str:
        if 0 <= index < len(self.driver_vocabulary):
            return str(self.driver_vocabulary[index])
        return f"DRIVER_{index}"

    def _partition_for(self, year: int, split: str):
        key = (int(year), str(split))
        if key not in self.partition_cache:
            path = ROOT / "gnn_sequences" / str(year) / str(split)
            if not path.is_dir():
                raise ValueError(f"graph sequence split not found for {year} {split}")
            self.partition_cache[key] = GraphPartition(path)
        return self.partition_cache[key]

    def _sequence_for_event(self, partition, event: str) -> int:
        for sequence_id, snapshot_ids in enumerate(partition.sequence_snapshot_ids):
            if partition.snapshots[int(snapshot_ids[-1])].get("event") == event:
                return sequence_id
        raise ValueError(f"event not found in {partition.season} {partition.split}: {event}")

    @torch.no_grad()
    def predict_sequence(self, sequence_id: int, year: int | None = None, split: str | None = None, event: str | None = None) -> dict:
        partition = self._partition_for(year if year is not None else self.partition.season, split or self.partition.split)
        if event:
            sequence_id = self._sequence_for_event(partition, event)
        if not 0 <= sequence_id < len(partition):
            raise ValueError(f"sequence must be between 0 and {len(partition) - 1}")
        sample = partition.sample(sequence_id)
        batch = collate([sample], self.normalizer)
        outputs = self.model(move_batch(batch, self.device))
        arrays = {key: value.detach().cpu().numpy() for key, value in outputs.items()}
        target_start = int(partition.target_ptr[sequence_id])
        target_end = int(partition.target_ptr[sequence_id + 1])
        target_nodes = partition.target_node_indices[target_start:target_end]
        threshold = {
            head: float(self.calibration[head]["thresholds"]["f1_threshold"]["threshold"])
            for head in ("tyre", "pit", "safety")
        }
        predictions = []
        for index in range(target_end - target_start):
            global_row = target_start + index
            tyre_probability = self._calibrated(float(arrays["tyre_logit"][index]), "tyre")
            pit_probability = self._calibrated(float(arrays["pit_logit"][index]), "pit")
            safety_probability = self._calibrated(float(arrays["safety_logit"][index]), "safety")
            current_raw = partition.x[int(target_nodes[index])]
            target_row = {
                "target_next_lap_time_s": json_scalar(partition.target_time[global_row]),
                "target_next_lap_time_delta_s": json_scalar(partition.target_delta[global_row]),
                "target_next_tyre_wear_estimate_fraction": json_scalar(partition.target_wear[global_row]),
                "target_high_tyre_degradation": json_scalar(partition.target_high_tyre[global_row]),
                "target_pit_stop_observed": json_scalar(partition.target_pit[global_row]),
                "target_safety_constraint": json_scalar(partition.target_safety[global_row]),
                "physics_confidence": json_scalar(current_raw[29]),
                "weather_confidence": json_scalar(current_raw[26]),
                "traffic_constraint_factor": json_scalar(current_raw[20]),
            }
            predictions.append({
                "row_index": global_row,
                "driver": self._driver_name(int(partition.node_driver_index[target_nodes[index]])),
                "next_lap_time_s": float(arrays["time"][index]),
                "next_lap_time_delta_s": float(arrays["delta"][index]),
                "next_tyre_wear_estimate_fraction": float(arrays["wear"][index]),
                "tyre_degradation_probability": tyre_probability,
                "pit_stop_probability": pit_probability,
                "safety_constraint_probability": safety_probability,
                "tyre_degradation_alert": tyre_probability >= threshold["tyre"],
                "pit_stop_alert": pit_probability >= threshold["pit"],
                "safety_constraint_alert": safety_probability >= threshold["safety"],
                "actual_test_labels": {
                    "next_lap_time_s": target_row.get("target_next_lap_time_s"),
                    "high_tyre_degradation": target_row.get("target_high_tyre_degradation"),
                    "pit_stop_observed": target_row.get("target_pit_stop_observed"),
                    "safety_constraint": target_row.get("target_safety_constraint"),
                },
                "quality": {
                    "physics_confidence": target_row.get("physics_confidence"),
                    "weather_confidence": target_row.get("weather_confidence"),
                    "traffic_constraint_factor": target_row.get("traffic_constraint_factor"),
                },
            })
        return {
            "status": "available",
            "model_version": "hybrid_gnn_physics_epoch51",
            "source": f"gnn_sequences/{partition.season}/{partition.split}/graph sequence",
            "test_context": {
                "year": partition.season,
                "split": partition.split,
                "event": sample["event"],
                "sequence": sequence_id,
            },
            "thresholds": threshold,
            "predictions": predictions,
        }


def json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, runtime: HybridRuntime, live: MultiViewerBridge, **kwargs):
        self.runtime = runtime
        self.live = live
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        route = urlsplit(self.path)
        try:
            if route.path == "/api/health":
                return json_response(self, {"status": "ok", "frontend": "APEX-R frontend", "hybrid_model": self.runtime.status(), "multiviewer": self.live.status()})
            if route.path == "/api/live/status":
                return json_response(self, self.live.status())
            if route.path == "/api/live/snapshot":
                return json_response(self, self.live.snapshot())
            if route.path == "/api/test/catalog":
                return json_response(self, local_test_catalog())
            if route.path == "/api/test/replay":
                query = parse_qs(route.query)
                year = int(query.get("year", ["2025"])[0])
                split = query.get("split", ["test"])[0]
                event = query.get("event", [""])[0]
                session = query.get("session", [None])[0]
                driver = query.get("driver", [None])[0]
                if not event:
                    raise ValueError("event is required; choose a 2024/2025 event from /api/test/catalog")
                return json_response(self, local_test_replay(year, split, event, session, driver, int(query.get("limit", ["900"])[0])))
            if route.path in {"/api/model/status", "/api/hybrid/status"}:
                return json_response(self, self.runtime.status())
            if route.path in {"/api/model/metrics", "/api/hybrid/metrics"}:
                return json_response(self, self.runtime.status()["metrics"])
            if route.path in {"/api/model/predict", "/api/hybrid/predict"}:
                query = parse_qs(route.query)
                raw = query.get("sequence", ["0"])[0]
                try:
                    sequence_id = int(raw)
                except ValueError as error:
                    raise ValueError("sequence must be an integer") from error
                year_value = query.get("year", [None])[0]
                year = int(year_value) if year_value is not None else None
                split = query.get("split", [None])[0]
                event = query.get("event", [None])[0]
                return json_response(self, self.runtime.predict_sequence(sequence_id, year=year, split=split, event=event))
            if route.path.startswith("/api/"):
                return json_response(self, {"status": "unavailable", "reason": "Only hybrid model endpoints are enabled"}, 404)
            return super().do_GET()
        except (ValueError, TypeError) as error:
            return json_response(self, {"status": "error", "detail": str(error)}, 422)
        except Exception as error:  # explicit local service failure
            return json_response(self, {"status": "unavailable", "reason": str(error)}, 503)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Trackshift frontend with the frozen hybrid model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--multiviewer-url",
        default=os.environ.get("APEX_MULTIVIEWER_URL", "http://127.0.0.1:10101/api/v1/live-timing"),
        help="Base URL of MultiViewer's local Live Timing API",
    )
    parser.add_argument(
        "--multiviewer-poll-ms",
        type=int,
        default=int(os.environ.get("APEX_MULTIVIEWER_POLL_MS", "1000")),
        help="Polling interval for MultiViewer telemetry in milliseconds",
    )
    parser.add_argument(
        "--multiviewer-capture",
        type=Path,
        default=ROOT / "runtime" / "multiviewer_live.jsonl",
        help="Local JSONL capture path for normalized and raw live samples",
    )
    parser.add_argument("--no-multiviewer-capture", action="store_true", help="Do not write a live telemetry capture")
    parser.add_argument(
        "--multiviewer-driver",
        default=os.environ.get("APEX_MULTIVIEWER_DRIVER"),
        help="Driver code or number to use as the focus car; defaults to the leading car",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if not FRONTEND.is_dir():
        parser.error(f"frontend directory not found: {FRONTEND}")
    if not BUNDLE_PATH.is_file():
        parser.error(f"model test bundle not found: {BUNDLE_PATH}")
    if not 250 <= args.multiviewer_poll_ms <= 10000:
        parser.error("--multiviewer-poll-ms must be between 250 and 10000")
    capture_path = None if args.no_multiviewer_capture else args.multiviewer_capture.expanduser()
    live = None
    try:
        runtime = HybridRuntime(BUNDLE_PATH)
        live = MultiViewerBridge(args.multiviewer_url, args.multiviewer_poll_ms, capture_path, args.multiviewer_driver)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, runtime=runtime, live=live))
    except (OSError, RuntimeError, ValueError) as error:
        if live:
            live.close()
        print(f"Cannot start Trackshift hybrid app: {error}")
        return 1
    print(f"Trackshift frontend: http://127.0.0.1:{args.port}/", flush=True)
    print(f"MultiViewer bridge: {args.multiviewer_url} (poll {args.multiviewer_poll_ms} ms)", flush=True)
    print(f"Live capture: {capture_path if capture_path else 'disabled'}", flush=True)
    print(json.dumps(runtime.status(), indent=2), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        live.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
