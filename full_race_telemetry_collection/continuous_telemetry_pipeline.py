#!/usr/bin/env python3
"""APEX-R continuous race-telemetry inventory and collection.

The source archives supplied to APEX-R contain selected lap slices.  This
pipeline inventories those rows first, then retrieves raw FastF1 car_data and
pos_data for the approved 2018--2022 Race manifest.  It uses no OpenF1 request
and never creates labels.  The first three approved train races are validated
against participation and lap-context coverage before the remaining approved
races are attempted.

Raw FastF1 cache files are temporary and are removed after each race to keep
disk use bounded.  Download/cache metadata and SHA-256 hashes are retained;
the exported CSVs retain raw telemetry fields and source lineage.

Run inside the existing FastF1 environment:
  source ../.venv_tracinginsights/bin/activate
  python continuous_telemetry_pipeline.py --workspace .. --output .
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import logging
import math
import shutil
import tempfile
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


VERSION = "continuous-telemetry-collection-v1.0"
SESSION = "Race"
YEARS = (2018, 2019, 2020, 2021, 2022)
GAP_THRESHOLD_SEC = 5.0
CAR_FIELDS = [
    "race_id", "split", "year", "event", "session", "driver", "driver_number", "lap",
    "date", "session_time_sec", "time_sec", "rpm", "speed_kmh", "n_gear",
    "throttle_pct", "brake", "drs", "source_channel", "timestamp_semantics",
]
POSITION_FIELDS = [
    "race_id", "split", "year", "event", "session", "driver", "driver_number", "lap",
    "date", "session_time_sec", "time_sec", "status", "x_m", "y_m", "z_m",
    "source_channel", "timestamp_semantics",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
            count += 1
    return count


class GzipCsvAppender:
    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        self.count = 0
        self.first = not path.exists()
        self.handle = gzip.open(path, "at", newline="", encoding="utf-8", compresslevel=6)
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, extrasaction="ignore")
        if self.first:
            self.writer.writeheader()

    def write(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in self.fields})
        self.count += 1

    def write_frame(self, frame: pd.DataFrame) -> None:
        """Append a prepared frame without row-by-row Python serialization."""
        if frame.empty:
            return
        frame.loc[:, self.fields].to_csv(self.handle, header=False, index=False, na_rep="", lineterminator="\n")
        self.count += len(frame)

    def close(self) -> None:
        self.handle.close()


def race_id(year: Any, event: Any, session: Any = SESSION) -> str:
    return f"{year}:{event}:{session}"


def safe_num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def timedelta_sec(v: Any) -> float | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "total_seconds"):
        try:
            x = float(v.total_seconds())
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None
    return safe_num(v)


def scalar_text(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            pass
    return str(v)


def quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else None


def aggregate_intervals(times: list[float]) -> dict[str, Any]:
    if not times:
        return {"timestamp_count": 0, "positive_interval_count": 0, "duplicate_timestamp_count": 0, "nonmonotonic_count": 0, "median_interval_sec": None, "p95_interval_sec": None, "max_interval_sec": None, "gap_count_over_5s": 0, "gap_seconds_over_5s": 0.0}
    raw_deltas = np.diff(np.asarray(times, dtype=float)) if len(times) > 1 else np.asarray([], dtype=float)
    positive = [float(x) for x in raw_deltas if x > 0]
    sorted_times = sorted(set(float(x) for x in times))
    sorted_deltas = np.diff(np.asarray(sorted_times, dtype=float)) if len(sorted_times) > 1 else np.asarray([], dtype=float)
    long_gaps = [float(x) for x in sorted_deltas if x > GAP_THRESHOLD_SEC]
    return {
        "timestamp_count": len(times),
        "positive_interval_count": len(positive),
        "duplicate_timestamp_count": int(sum(x == 0 for x in raw_deltas)),
        "nonmonotonic_count": int(sum(x < 0 for x in raw_deltas)),
        "median_interval_sec": quantile(positive, 0.5),
        "p95_interval_sec": quantile(positive, 0.95),
        "max_interval_sec": max(positive) if positive else None,
        "gap_count_over_5s": len(long_gaps),
        "gap_seconds_over_5s": float(sum(long_gaps)),
    }


def cache_inventory(cache_dir: Path) -> tuple[list[dict[str, Any]], int, str]:
    rows = []
    digest = hashlib.sha256()
    total = 0
    for path in sorted(p for p in cache_dir.rglob("*") if p.is_file()):
        data = path.read_bytes()
        total += len(data)
        digest.update(path.name.encode("utf-8"))
        digest.update(data)
        rows.append({"name": str(path.relative_to(cache_dir)), "bytes": len(data), "sha256": sha256_bytes(data)})
    return rows, total, digest.hexdigest()


def clear_cache_files(cache_dir: Path) -> None:
    """Remove only this task's temporary FastF1 cache contents."""
    if not cache_dir.exists():
        return
    for path in sorted(cache_dir.iterdir(), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def build_source_archive_inventory(base: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, int], list[str]]:
    rows: list[dict[str, Any]] = []
    archive_hashes: dict[str, str] = {}
    archive_row_counts: dict[str, int] = {}
    fields_seen: list[str] = []
    for year in YEARS:
        path = base / f"APEX-R_Telemetry_{year}.zip"
        if not path.is_file():
            raise FileNotFoundError(path)
        archive_hashes[str(path)] = sha256_file(path)
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"ZIP CRC failure in {path.name}: {bad}")
            infos = zf.infolist()
            safe = all(not n.filename.startswith("/") and ".." not in n.filename.split("/") for n in infos)
            if not safe or len(infos) != 1:
                raise RuntimeError(f"unexpected/unsafe source ZIP members: {path}")
            info = infos[0]
            rows.append({"year": year, "archive": path.name, "archive_bytes": path.stat().st_size, "archive_sha256": archive_hashes[str(path)], "member": info.filename, "member_bytes": info.file_size, "member_compressed_bytes": info.compress_size, "member_crc": info.CRC, "zip_crc_status": "PASS", "file_granularity": "seasonal gzip CSV; rows keyed by race/driver/lap/sample_index", "timestamp_semantics": "archive time is lap-relative seconds and resets per selected lap; no UTC/session anchor in this archive"})
            with zf.open(info) as raw, gzip.GzipFile(fileobj=raw) as gz, io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                reader = csv.DictReader(text)
                if not fields_seen:
                    fields_seen = list(reader.fieldnames or [])
                count = 0
                for _ in reader:
                    count += 1
                archive_row_counts[str(year)] = count
                rows[-1]["parsed_rows"] = count
    return rows, archive_hashes, archive_row_counts, fields_seen


def build_supplied_inventory(base: Path, expected: Mapping[str, set[str]], split_by_race: Mapping[str, Mapping[str, str]], protected: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    archive_rows: Counter[str] = Counter()
    for year in YEARS:
        path = base / f"APEX-R_Telemetry_{year}.zip"
        with zipfile.ZipFile(path) as zf, zf.open(zf.namelist()[0]) as raw, gzip.GzipFile(fileobj=raw) as gz, io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
            reader = csv.DictReader(text)
            for row in reader:
                rid = race_id(row.get("year"), row.get("event"), row.get("session"))
                if protected in rid:
                    raise RuntimeError("protected race appeared in supplied telemetry rows")
                if rid not in split_by_race:
                    continue
                code = str(row.get("driver", ""))
                key = (str(row.get("year")), str(row.get("event")), str(row.get("session")), code)
                lap = int(float(row["lap"]))
                time = safe_num(row.get("time"))
                sample = int(float(row["sample_index"])) if row.get("sample_index") not in {None, ""} else None
                group = groups.setdefault(key, {"rows": 0, "laps": set(), "lap_rows": Counter(), "times": [], "by_lap_last_time": {}, "positive_deltas": [], "duplicate_timestamps": 0, "nonmonotonic": 0, "source_data_keys": set()})
                group["rows"] += 1
                group["laps"].add(lap)
                group["lap_rows"][lap] += 1
                if time is not None:
                    last = group["by_lap_last_time"].get(lap)
                    if last is not None:
                        delta = time - last
                        if delta > 0:
                            group["positive_deltas"].append(delta)
                        elif delta == 0:
                            group["duplicate_timestamps"] += 1
                        else:
                            group["nonmonotonic"] += 1
                    group["by_lap_last_time"][lap] = time
                    group["times"].append(time)
                group["source_data_keys"].add(str(row.get("source_data_key", "")))
                archive_rows[str(year)] += 1
    result: list[dict[str, Any]] = []
    for rid, codes in sorted(expected.items()):
        year, event, session = rid.split(":", 2)
        for driver in sorted(codes):
            group = groups.get((year, event, session, driver))
            split = split_by_race[rid]["split"]
            if group is None:
                result.append({"race_id": rid, "split": split, "year": year, "event": event, "session": session, "driver": driver, "data_origin": "PROVIDED_ARCHIVE_CACHED", "telemetry_status": "MISSING_FROM_SUPPLIED_ARCHIVE", "selected_lap_count": 0, "selected_laps": "", "supplied_rows": 0, "source_data_keys": "", "timestamp_semantics": "lap-relative seconds; no UTC/session anchor"})
                continue
            positive = group["positive_deltas"]
            result.append({"race_id": rid, "split": split, "year": year, "event": event, "session": session, "driver": driver, "data_origin": "PROVIDED_ARCHIVE_CACHED", "telemetry_status": "CACHED_SELECTED_LAP_STREAM", "selected_lap_count": len(group["laps"]), "selected_laps": ",".join(str(x) for x in sorted(group["laps"])), "supplied_rows": group["rows"], "source_data_keys": "|".join(sorted(group["source_data_keys"])), "time_min_sec": min(group["times"]) if group["times"] else None, "time_max_sec": max(group["times"]) if group["times"] else None, "median_interval_sec": quantile(positive, 0.5), "p95_interval_sec": quantile(positive, 0.95), "max_interval_sec": max(positive) if positive else None, "duplicate_timestamp_count": group["duplicate_timestamps"], "nonmonotonic_timestamp_count": group["nonmonotonic"], "timestamp_semantics": "lap-relative seconds; no UTC/session anchor"})
    stats = {"driver_race_entries_expected": sum(len(x) for x in expected.values()), "driver_race_entries_cached": sum(x["telemetry_status"] == "CACHED_SELECTED_LAP_STREAM" for x in result), "driver_race_entries_missing": sum(x["telemetry_status"] == "MISSING_FROM_SUPPLIED_ARCHIVE" for x in result), "selected_driver_lap_streams": sum(int(x["selected_lap_count"]) for x in result), "observations": sum(archive_rows.values()), "archive_rows_by_year": dict(archive_rows)}
    return result, stats


def load_scope(workspace: Path, protected_manifest: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, set[str]], str]:
    excluded = json.loads(protected_manifest.read_text(encoding="utf-8"))
    protected = str(excluded.get("protected_session_token", ""))
    if not protected:
        raise RuntimeError("protected-session exclusion manifest has no token")
    split_path = workspace / "phase3_prediction_dataset" / "phase3_split_manifest.csv"
    split_rows = list(csv.DictReader(split_path.open(encoding="utf-8", newline="")))
    split_by_race = {row["race_id"]: row for row in split_rows if row.get("session") == SESSION and int(row["year"]) in YEARS}
    if len(split_by_race) != 70:
        raise RuntimeError(f"approved split manifest expected 70 Race rows, found {len(split_by_race)}")
    for rid in split_by_race:
        if protected in rid:
            raise RuntimeError("protected race was in approved collection scope")
    identity_path = workspace / "phase2_data_foundation" / "context" / "identity_crosswalk.csv"
    expected: dict[str, set[str]] = defaultdict(set)
    for row in csv.DictReader(identity_path.open(encoding="utf-8", newline="")):
        rid = race_id(row["year"], row["event"], row["session"])
        if rid in split_by_race and row.get("driver_code"):
            expected[rid].add(row["driver_code"])
    if set(expected) != set(split_by_race) or any(len(x) == 0 for x in expected.values()):
        raise RuntimeError("identity crosswalk did not provide every approved race")
    return split_rows, split_by_race, expected, protected


def load_lap_context(workspace: Path) -> dict[tuple[str, str], dict[str, Any]]:
    coverage_path = workspace / "phase3_prediction_dataset" / "context" / "laptime_coverage.csv"
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in csv.DictReader(coverage_path.open(encoding="utf-8", newline="")):
        rid = race_id(row["year"], row["event"], row["session"])
        result[(rid, row["driver"])] = {"rows": int(row.get("lap_records") or 0), "max_lap": int(row["max_lap"]) if row.get("max_lap") else None, "min_lap": int(row["min_lap"]) if row.get("min_lap") else None, "status": "LAP_CONTEXT_PRESENT" if int(row.get("lap_records") or 0) > 0 else "LAP_CONTEXT_EMPTY"}
    return result


def build_lap_index(laps: pd.DataFrame, driver: str) -> tuple[float | None, float | None, list[int], list[float]]:
    subset = laps[laps["Driver"].astype(str) == str(driver)]
    if subset.empty:
        return None, None, [], []
    starts = [x for x in (timedelta_sec(v) for v in subset.get("LapStartTime", [])) if x is not None]
    ends_laps = []
    for _, row in subset.iterrows():
        end = timedelta_sec(row.get("Time"))
        lap = safe_num(row.get("LapNumber"))
        if end is not None and lap is not None:
            ends_laps.append((end, int(lap)))
    ends_laps.sort()
    return (min(starts) if starts else None, max((x[0] for x in ends_laps), default=None), [x[1] for x in ends_laps], [x[0] for x in ends_laps])


def assign_lap(seconds: float | None, end_times: list[float], lap_numbers: list[int]) -> int | None:
    if seconds is None or not end_times:
        return None
    idx = bisect.bisect_left(end_times, seconds)
    if idx >= len(end_times):
        idx = len(end_times) - 1
    return lap_numbers[idx]


def collect_fastf1_race(
    workspace: Path,
    output: Path,
    split_row: Mapping[str, str],
    expected_codes: set[str],
    lap_context: Mapping[tuple[str, str], Mapping[str, Any]],
    supplied_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    car_writer: GzipCsvAppender,
    pos_writer: GzipCsvAppender,
    cache_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    import fastf1

    year, event, session_name = int(split_row["year"]), split_row["event"], split_row["session"]
    rid = split_row["race_id"]
    before_files, before_bytes, before_digest = cache_inventory(cache_dir)
    cache_state = "CACHED_PREEXISTING" if before_files else "NEW_DOWNLOAD_EXPECTED"
    error = ""
    source_records: dict[str, Any] = {"race_id": rid, "year": year, "event": event, "session": session_name, "provider": "FastF1", "fastf1_version": getattr(fastf1, "__version__", "unknown"), "cache_state_before_load": cache_state, "cache_files_before": len(before_files), "cache_bytes_before": before_bytes, "cache_digest_before": before_digest, "raw_cache_retained": False, "request_scope_guard": "approved split row checked before get_session"}
    coverage_rows: list[dict[str, Any]] = []
    try:
        fastf1.Cache.enable_cache(cache_dir)
        session = fastf1.get_session(year, event, session_name)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = session.laps
        code_to_number: dict[str, str] = {}
        for _, row in laps[["Driver", "DriverNumber"]].dropna().drop_duplicates().iterrows():
            code_to_number[str(row["Driver"])] = str(row["DriverNumber"])
        participant_codes = set(code_to_number)
        all_codes = sorted(expected_codes | participant_codes)
        car_by_number = getattr(session, "car_data", {}) or {}
        pos_by_number = getattr(session, "pos_data", {}) or {}
        for code in all_codes:
            number = code_to_number.get(code, "")
            car = car_by_number.get(number)
            if car is None and number:
                car = car_by_number.get(str(int(float(number))))
            pos = pos_by_number.get(number)
            if pos is None and number:
                pos = pos_by_number.get(str(int(float(number))))
            start_sec, end_sec, lap_numbers, lap_ends = build_lap_index(laps, code)
            ctx = lap_context.get((rid, code), {})
            selected = supplied_by_key.get((rid, code), {})
            car_rows = 0
            pos_rows = 0
            car_times: list[float] = []
            pos_times: list[float] = []
            car_dates: list[str] = []
            pos_dates: list[str] = []
            car_laps: list[int] = []
            pos_laps: list[int] = []
            def seconds_series(frame: pd.DataFrame, column: str) -> pd.Series:
                values = frame[column]
                if pd.api.types.is_timedelta64_dtype(values):
                    return values.dt.total_seconds()
                return pd.to_numeric(values, errors="coerce")

            def filtered_stream(frame: Any, channel: str) -> tuple[pd.DataFrame, list[float], list[str], list[int]]:
                if not isinstance(frame, pd.DataFrame) or frame.empty or "SessionTime" not in frame:
                    return pd.DataFrame(), [], [], []
                out = frame.copy()
                out["_session_time_sec"] = seconds_series(out, "SessionTime")
                mask = out["_session_time_sec"].notna()
                if start_sec is not None:
                    mask &= out["_session_time_sec"] >= start_sec - 1e-6
                if end_sec is not None:
                    mask &= out["_session_time_sec"] <= end_sec + 1e-6
                out = out.loc[mask].copy()
                if out.empty:
                    return out, [], [], []
                if lap_ends:
                    idx = np.searchsorted(np.asarray(lap_ends, dtype=float), out["_session_time_sec"].to_numpy(dtype=float), side="left")
                    idx = np.minimum(idx, len(lap_numbers) - 1)
                    out["_lap"] = np.asarray(lap_numbers, dtype=int)[idx]
                else:
                    out["_lap"] = np.nan
                out["_date_text"] = out["Date"].map(scalar_text) if "Date" in out else ""
                times = [float(x) for x in out["_session_time_sec"].tolist()]
                dates = [str(x) for x in out["_date_text"].tolist()]
                laps_out = [int(x) if pd.notna(x) else -1 for x in out["_lap"].tolist()]
                return out, times, dates, laps_out

            car_frame, car_times, car_dates, car_laps = filtered_stream(car, "car")
            if not car_frame.empty:
                car_out = pd.DataFrame({
                    "race_id": rid, "split": split_row["split"], "year": year, "event": event, "session": session_name,
                    "driver": code, "driver_number": number, "lap": car_frame["_lap"], "date": car_frame["_date_text"],
                    "session_time_sec": car_frame["_session_time_sec"], "time_sec": seconds_series(car_frame, "Time") if "Time" in car_frame else np.nan,
                    "rpm": car_frame["RPM"] if "RPM" in car_frame else np.nan, "speed_kmh": car_frame["Speed"] if "Speed" in car_frame else np.nan,
                    "n_gear": car_frame["nGear"] if "nGear" in car_frame else np.nan, "throttle_pct": car_frame["Throttle"] if "Throttle" in car_frame else np.nan,
                    "brake": car_frame["Brake"] if "Brake" in car_frame else np.nan, "drs": car_frame["DRS"] if "DRS" in car_frame else np.nan,
                    "source_channel": "FastF1 car_data", "timestamp_semantics": "Date=FastF1 source datetime (returned timezone-naive); SessionTime=FastF1 session-relative seconds; no interpolation",
                }, index=car_frame.index)
                car_writer.write_frame(car_out)
                car_rows = len(car_out)

            pos_frame, pos_times, pos_dates, pos_laps = filtered_stream(pos, "position")
            if not pos_frame.empty:
                pos_out = pd.DataFrame({
                    "race_id": rid, "split": split_row["split"], "year": year, "event": event, "session": session_name,
                    "driver": code, "driver_number": number, "lap": pos_frame["_lap"], "date": pos_frame["_date_text"],
                    "session_time_sec": pos_frame["_session_time_sec"], "time_sec": seconds_series(pos_frame, "Time") if "Time" in pos_frame else np.nan,
                    "status": pos_frame["Status"] if "Status" in pos_frame else np.nan, "x_m": pos_frame["X"] if "X" in pos_frame else np.nan,
                    "y_m": pos_frame["Y"] if "Y" in pos_frame else np.nan, "z_m": pos_frame["Z"] if "Z" in pos_frame else np.nan,
                    "source_channel": "FastF1 pos_data", "timestamp_semantics": "Date=FastF1 source datetime (returned timezone-naive); SessionTime=FastF1 session-relative seconds; no interpolation",
                }, index=pos_frame.index)
                pos_writer.write_frame(pos_out)
                pos_rows = len(pos_out)
            car_stats = aggregate_intervals(car_times)
            pos_stats = aggregate_intervals(pos_times)
            laptime_max = ctx.get("max_lap")
            car_max_lap = max([x for x in car_laps if x >= 0], default=None)
            pos_max_lap = max([x for x in pos_laps if x >= 0], default=None)
            participant = code in participant_codes
            if participant and car_rows > 0 and pos_rows > 0 and laptime_max is not None and car_max_lap is not None and pos_max_lap is not None and car_max_lap >= laptime_max and pos_max_lap >= laptime_max:
                status = "TELEMETRY_COVERAGE_PASS"
            elif participant and car_rows > 0 and pos_rows > 0 and laptime_max is None:
                status = "TELEMETRY_PRESENT_LAP_CONTEXT_UNAVAILABLE"
            elif not participant and not ctx.get("rows"):
                status = "NO_FASTF1_LAPS_AND_NO_LAP_CONTEXT"
            else:
                status = "TELEMETRY_COVERAGE_INCOMPLETE"
            coverage_rows.append({"race_id": rid, "split": split_row["split"], "year": year, "event": event, "session": session_name, "driver": code, "driver_number": number, "expected_in_identity_crosswalk": "TRUE" if code in expected_codes else "FALSE", "fastf1_participant": "TRUE" if participant else "FALSE", "laptime_context_status": ctx.get("status", "LAP_CONTEXT_MISSING"), "laptime_rows": ctx.get("rows", 0), "laptime_max_lap": laptime_max, "supplied_archive_status": selected.get("telemetry_status", "MISSING_FROM_SUPPLIED_ARCHIVE"), "supplied_archive_rows": selected.get("supplied_rows", 0), "car_rows": car_rows, "position_rows": pos_rows, "car_date_start": min(car_dates) if car_dates else "", "car_date_end": max(car_dates) if car_dates else "", "position_date_start": min(pos_dates) if pos_dates else "", "position_date_end": max(pos_dates) if pos_dates else "", "car_session_time_start_sec": min(car_times) if car_times else None, "car_session_time_end_sec": max(car_times) if car_times else None, "position_session_time_start_sec": min(pos_times) if pos_times else None, "position_session_time_end_sec": max(pos_times) if pos_times else None, "car_driver_hours": ((max(car_times) - min(car_times)) / 3600.0) if car_times else 0.0, "position_driver_hours": ((max(pos_times) - min(pos_times)) / 3600.0) if pos_times else 0.0, "car_lap_min_assigned": min([x for x in car_laps if x >= 0], default=None), "car_lap_max_assigned": car_max_lap, "position_lap_min_assigned": min([x for x in pos_laps if x >= 0], default=None), "position_lap_max_assigned": pos_max_lap, "car_median_interval_sec": car_stats["median_interval_sec"], "car_p95_interval_sec": car_stats["p95_interval_sec"], "car_max_interval_sec": car_stats["max_interval_sec"], "car_gap_count_over_5s": car_stats["gap_count_over_5s"], "car_gap_seconds_over_5s": car_stats["gap_seconds_over_5s"], "car_duplicate_timestamp_count": car_stats["duplicate_timestamp_count"], "car_nonmonotonic_count": car_stats["nonmonotonic_count"], "position_median_interval_sec": pos_stats["median_interval_sec"], "position_p95_interval_sec": pos_stats["p95_interval_sec"], "position_max_interval_sec": pos_stats["max_interval_sec"], "position_gap_count_over_5s": pos_stats["gap_count_over_5s"], "position_gap_seconds_over_5s": pos_stats["gap_seconds_over_5s"], "position_duplicate_timestamp_count": pos_stats["duplicate_timestamp_count"], "position_nonmonotonic_count": pos_stats["nonmonotonic_count"], "coverage_status": status, "timestamp_semantics": "FastF1 raw Date plus session-relative SessionTime; timezone/latency is not inferred; lap assignment is interval lookup from FastF1 LapStartTime/Time"})
        source_records["participant_driver_count"] = len(participant_codes)
        source_records["expected_driver_count"] = len(expected_codes)
        source_records["fastf1_missing_expected_drivers"] = sorted(expected_codes - participant_codes)
        source_records["fastf1_extra_drivers"] = sorted(participant_codes - expected_codes)
        source_records["car_rows_written"] = sum(r["car_rows"] for r in coverage_rows)
        source_records["position_rows_written"] = sum(r["position_rows"] for r in coverage_rows)
        source_records["retrieval_status"] = "SUCCESS"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        source_records["retrieval_status"] = "FAILED"
        source_records["error"] = error
    after_files, after_bytes, after_digest = cache_inventory(cache_dir)
    source_records.update({"cache_files_after": len(after_files), "cache_bytes_after": after_bytes, "cache_digest_after": after_digest, "new_cache_bytes": max(0, after_bytes - before_bytes), "cache_file_metadata": after_files})
    # Keep no raw FastF1 payloads in the retained workspace.  The source
    # manifest retains the post-load digest and file-level hashes.
    clear_cache_files(cache_dir)
    source_records["cache_cleanup_status"] = "CLEANED" if not any(cache_dir.rglob("*")) else "CLEANUP_INCOMPLETE"
    pass_rows = sum(r["coverage_status"] == "TELEMETRY_COVERAGE_PASS" for r in coverage_rows if r["expected_in_identity_crosswalk"] == "TRUE")
    participant_rows = sum(r["fastf1_participant"] == "TRUE" for r in coverage_rows)
    race_status = "PASS" if source_records.get("retrieval_status") == "SUCCESS" and participant_rows == len(expected_codes) and pass_rows == len(expected_codes) else "INCOMPLETE_OR_UNVERIFIED"
    race_record = {"race_id": rid, "split": split_row["split"], "year": year, "event": event, "session": session_name, "expected_driver_count": len(expected_codes), "fastf1_participant_count": participant_rows, "car_rows": source_records.get("car_rows_written", 0), "position_rows": source_records.get("position_rows_written", 0), "driver_hours_car": sum(r["car_driver_hours"] for r in coverage_rows), "driver_hours_position": sum(r["position_driver_hours"] for r in coverage_rows), "drivers_with_telemetry_coverage_pass": pass_rows, "telemetry_coverage_status": race_status, "source_retrieval_status": source_records.get("retrieval_status"), "error": error, "fastf1_missing_expected_drivers": "|".join(source_records.get("fastf1_missing_expected_drivers", []))}
    return race_record, coverage_rows, source_records


def package_and_verify(output: Path, artifact_rows: Mapping[str, int]) -> dict[str, Any]:
    sums_path = output / "SHA256SUMS.txt"
    excluded = {"SHA256SUMS.txt", "APEX-R_Full_Race_Telemetry_Collection.zip", "package_verification.json", "collection_verification.json"}
    core = [p for p in output.iterdir() if p.is_file() and p.name not in excluded]
    sums_path.write_text("\n".join(f"{sha256_file(p)}  {p.name}" for p in sorted(core, key=lambda x: x.name)) + "\n", encoding="utf-8")
    package = output / "APEX-R_Full_Race_Telemetry_Collection.zip"
    if package.exists():
        package.unlink()
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(core + [sums_path], key=lambda x: x.name):
            zf.write(p, arcname=p.name)
    extract_dir = Path(tempfile.mkdtemp(prefix="apex_full_telemetry_extract_"))
    checks: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(package) as zf:
            bad = zf.testzip()
            checks.append({"test": "zip_crc", "passed": bad is None, "detail": "PASS" if bad is None else bad})
            zf.extractall(extract_dir)
        sums = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split("  ", 1)
                sums[name] = digest
        for name, digest in sorted(sums.items()):
            actual = sha256_file(extract_dir / name)
            checks.append({"test": "extracted_sha256:" + name, "passed": actual == digest, "detail": actual})
        for name, expected in sorted(artifact_rows.items()):
            p = extract_dir / name
            if not p.exists():
                checks.append({"test": "row_count:" + name, "passed": False, "detail": "missing"})
                continue
            if name.endswith(".csv.gz"):
                with gzip.open(p, "rt", encoding="utf-8", newline="") as f:
                    actual = max(sum(1 for _ in f) - 1, 0)
            elif name.endswith(".csv"):
                with p.open(encoding="utf-8", newline="") as f:
                    actual = max(sum(1 for _ in f) - 1, 0)
            elif name.endswith(".json"):
                value = json.loads(p.read_text(encoding="utf-8"))
                actual = len(value) if isinstance(value, list) else expected
            else:
                continue
            checks.append({"test": "row_count:" + name, "passed": int(actual) == int(expected), "detail": {"actual": actual, "expected": expected}})
        passed = all(x["passed"] for x in checks)
        member_inventory = []
        with zipfile.ZipFile(package) as zf:
            for info in sorted(zf.infolist(), key=lambda x: x.filename):
                member_inventory.append({"name": info.filename, "compressed_bytes": info.compress_size, "uncompressed_bytes": info.file_size, "crc32": f"{info.CRC:08x}", "sha256": sha256_file(extract_dir / info.filename)})
        return {"package": {"path": str(package), "bytes": package.stat().st_size, "sha256": sha256_file(package), "compression": "ZIP_DEFLATED", "member_count": len(core) + 1}, "members": member_inventory, "checks": checks, "all_passed": passed, "fresh_extraction_cleaned": False, "verification_directory": str(extract_dir)}
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def build_report(output: Path, metrics: Mapping[str, Any]) -> None:
    cov = metrics["coverage"]
    events = metrics["events"]
    split = metrics["split_coverage"]
    validation_races = ", ".join(metrics["validation_gate"]["races"])
    lines = [
        "# APEX-R continuous telemetry collection report", "",
        f"Generated: {metrics['generated_at_utc']}", "",
        "## Outcome", "",
        "Continuous car and position streams were inventoried and collected in a separate package. The source seasonal archives and the prior lap-context package were preserved. No labels, model work, replay, decision-engine change, or holdout scoring was performed.", "",
        f"The three-race validation gate was **{'PASS' if metrics['validation_gate']['all_three_passed'] else 'FAIL'}** before expansion. The validation races were: {validation_races}. They were selected from approved development races and remain identified in `validation_three_races.csv`.", "",
        "## Measured coverage", "",
        f"* Approved manifest: **{cov['approved_races']} races**, seasons 2018–2022, Race session.",
        f"* Supplied selected-lap archive: **{metrics['supplied_archive']['observations']:,} observations**, **{metrics['supplied_archive']['selected_driver_lap_streams']:,} selected driver-lap streams**, **{metrics['supplied_archive']['driver_race_entries_missing']:,} missing expected driver-race entries**.",
        f"* FastF1 exported car observations: **{cov['car_observations']:,}**; position observations: **{cov['position_observations']:,}**.",
        f"* Driver-hours covered: car **{cov['car_driver_hours']:.3f}**, position **{cov['position_driver_hours']:.3f}**.",
        f"* Race-level telemetry coverage pass: **{cov['telemetry_race_passes']}**; incomplete/unverified: **{cov['telemetry_race_incomplete']}**.",
        f"* FastF1 source retrievals: **{metrics['downloads']['new_download_races']} new**, **{metrics['downloads']['cached_races']} pre-existing-cache reuse**, **{metrics['downloads']['failed_races']} failed**. Temporary raw caches were cleaned after each race.", "",
        "## Split coverage", "",
        "The development-test races remain identified as development data and were not used for model scoring in this task.", "",
        "| Split | Approved races | Telemetry pass races | Car rows | Position rows | Car driver-hours | Position driver-hours |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in ["train", "validation", "development_test"]:
        x = split.get(s, {})
        lines.append(f"| {s} | {x.get('approved_races', 0)} | {x.get('telemetry_pass_races', 0)} | {x.get('car_rows', 0):,} | {x.get('position_rows', 0):,} | {x.get('car_driver_hours', 0.0):.3f} | {x.get('position_driver_hours', 0.0):.3f} |")
    lines += [
        "", "## Timestamp and fields", "",
        "The retained car table contains speed (km/h), throttle (%), brake, gear, RPM and DRS. The retained position table contains X/Y/Z coordinates and source status where available. FastF1 `Date` is retained as returned (timezone-naive in the observed v3.8.3 runtime); `SessionTime` is retained as session-relative seconds. These are measurement/source timestamps, not proof of live publication latency.", "",
        "No interpolation or nearest-time merge was used between car and position streams. Lap numbers are an explicit interval lookup using FastF1 `LapStartTime` and `Time`, with the method recorded in each row. Any timestamp gaps greater than five seconds are reported per driver in `telemetry_coverage.csv`; they remain gaps, not imputed samples.", "",
        "## Events and labels", "",
        f"The prior **{events['preserved_position_swap_candidates']:,}** position-swap candidates were preserved by hash and remain unresolved. This task produced **0 verified overtake events** and did not relabel candidates. FastF1 telemetry supplies measurements but not a verified historical overtake-event feed sufficient to turn every boundary swap into an overtake.", "",
        "OpenF1 was not queried for 2018–2022 outcome data. Its documented free historical coverage begins in 2023; a separate 2023+ collection should resolve session keys and exclude protected holdouts before any request, then use timestamped OpenF1 overtake/position evidence. Do not mix that future collection into this 2018–2022 package.", "",
        "## Integrity", "",
        f"Source archive immutability: **{'PASS' if metrics['source_archive_immutable'] else 'FAIL'}**. Prior lap-context package immutability: **{'PASS' if metrics['prior_package_immutable'] else 'FAIL'}**. ZIP DEFLATE, CRC, extracted checksums and row counts: **{('PASS' if metrics.get('package', {}).get('verification_passed') else 'PENDING')}**.", "",
        "## Limits", "",
        "A telemetry coverage pass means raw FastF1 car and position streams were available for the participating driver and covered the observed laptime lap span. It does not prove every physical track sample is present, prove live delivery latency, or verify an overtake event. The approved data can proceed to a separate event-evidence review, not directly to verified-overtake model training.", "",
        "Exact file bytes and ZIP SHA-256 are in `package_verification.json`; raw FastF1 cache payloads were not retained in the package.",
    ]
    (output / "PHASE_CONTINUOUS_TELEMETRY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--validation-races", default="2018:Chinese Grand Prix:Race|2019:Australian Grand Prix:Race|2020:Austrian Grand Prix:Race")
    args = ap.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / x).exists() for x in ["telemetry_car.csv.gz", "telemetry_position.csv.gz", "metrics.json"]):
        raise RuntimeError("output already contains a collection; use a fresh output directory to preserve reproducibility")
    split_rows, split_by_race, expected, protected = load_scope(workspace, workspace / "phase3_prediction_dataset" / "excluded_sessions_manifest.json")
    archive_dir = workspace.parent
    archive_inventory, archive_hashes_before, archive_row_counts, source_fields = build_source_archive_inventory(archive_dir)
    supplied, supplied_stats = build_supplied_inventory(archive_dir, expected, split_by_race, protected)
    supplied_by_key = {(r["race_id"], r["driver"]): r for r in supplied}
    lap_context = load_lap_context(workspace)
    # Preserve the prior package by hash before any new collection work.
    prior_package = workspace / "full_race_collection" / "APEX-R_Full_Race_Collection.zip"
    prior_package_hash_before = sha256_file(prior_package) if prior_package.is_file() else ""
    csv_write(output / "source_archive_inventory.csv", archive_inventory, list(archive_inventory[0].keys()))
    json_write(output / "source_archive_inventory.json", archive_inventory)
    inv_fields = ["race_id", "split", "year", "event", "session", "driver", "data_origin", "telemetry_status", "selected_lap_count", "selected_laps", "supplied_rows", "source_data_keys", "time_min_sec", "time_max_sec", "median_interval_sec", "p95_interval_sec", "max_interval_sec", "duplicate_timestamp_count", "nonmonotonic_timestamp_count", "timestamp_semantics"]
    csv_write(output / "supplied_telemetry_inventory.csv", supplied, inv_fields)
    json_write(output / "supplied_telemetry_inventory.json", supplied)

    validation_race_ids = [x for x in args.validation_races.split("|") if x]
    if len(validation_race_ids) != 3 or len(set(validation_race_ids)) != 3 or not set(validation_race_ids).issubset(split_by_race):
        raise ValueError("validation-races must contain exactly three distinct approved race IDs")
    cache_dir = output / "fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    car_writer = GzipCsvAppender(output / "telemetry_car.csv.gz", CAR_FIELDS)
    pos_writer = GzipCsvAppender(output / "telemetry_position.csv.gz", POSITION_FIELDS)
    validation_records: list[dict[str, Any]] = []
    validation_coverage: list[dict[str, Any]] = []
    validation_sources: list[dict[str, Any]] = []
    for rid in validation_race_ids:
        record, coverage_rows, source_record = collect_fastf1_race(workspace, output, split_by_race[rid], expected[rid], lap_context, supplied_by_key, car_writer, pos_writer, cache_dir)
        validation_records.append(record)
        validation_coverage.extend(coverage_rows)
        validation_sources.append(source_record)
    validation_pass = all(r["telemetry_coverage_status"] == "PASS" for r in validation_records)
    validation_gate = {"stage": "THREE_RACE_VALIDATION_BEFORE_EXPANSION", "races": validation_race_ids, "all_three_passed": validation_pass, "race_records": validation_records, "expansion_authorized": validation_pass}
    json_write(output / "validation_three_races.json", validation_gate)
    csv_write(output / "validation_three_races.csv", validation_records, list(validation_records[0].keys()))
    # Persist the per-driver gate evidence even when the gate stops expansion;
    # this makes a failed validation reproducible and auditable.
    if validation_coverage:
        csv_write(output / "validation_three_race_coverage.csv", validation_coverage, list(validation_coverage[0].keys()))
        json_write(output / "validation_three_race_coverage.json", validation_coverage)
    if not validation_pass:
        car_writer.close(); pos_writer.close()
        raise RuntimeError("three-race telemetry validation failed; expansion was not run")

    all_races: list[dict[str, Any]] = list(validation_records)
    all_coverage: list[dict[str, Any]] = list(validation_coverage)
    all_sources: list[dict[str, Any]] = list(validation_sources)
    already = set(validation_race_ids)
    for rid in sorted(split_by_race, key=lambda x: int(split_by_race[x]["chronology_rank"])):
        if rid in already:
            continue
        record, coverage_rows, source_record = collect_fastf1_race(workspace, output, split_by_race[rid], expected[rid], lap_context, supplied_by_key, car_writer, pos_writer, cache_dir)
        all_races.append(record)
        all_coverage.extend(coverage_rows)
        all_sources.append(source_record)
    car_writer.close(); pos_writer.close()
    if any(cache_dir.rglob("*")):
        clear_cache_files(cache_dir)
    csv_write(output / "race_collection_status.csv", all_races, list(all_races[0].keys()))
    json_write(output / "race_collection_status.json", all_races)
    coverage_fields = list(all_coverage[0].keys()) if all_coverage else ["race_id"]
    csv_write(output / "telemetry_coverage.csv", all_coverage, coverage_fields)
    json_write(output / "telemetry_coverage.json", all_coverage)
    csv_write(output / "fastf1_source_manifest.csv", all_sources, list(all_sources[0].keys()))
    json_write(output / "fastf1_source_manifest.json", all_sources)
    candidates_path = workspace / "full_race_collection" / "unresolved_position_swap_candidates.csv.gz"
    candidate_hash = sha256_file(candidates_path) if candidates_path.is_file() else ""
    candidate_rows = 0
    if candidates_path.is_file():
        with gzip.open(candidates_path, "rt", encoding="utf-8", newline="") as f:
            candidate_rows = max(sum(1 for _ in f) - 1, 0)
    race_pass = sum(x["telemetry_coverage_status"] == "PASS" for x in all_races)
    split_coverage: dict[str, Any] = {}
    for split in ["train", "validation", "development_test"]:
        rs = [x for x in all_races if x["split"] == split]
        split_coverage[split] = {"approved_races": len(rs), "telemetry_pass_races": sum(x["telemetry_coverage_status"] == "PASS" for x in rs), "car_rows": sum(x["car_rows"] for x in rs), "position_rows": sum(x["position_rows"] for x in rs), "car_driver_hours": sum(x["driver_hours_car"] for x in rs), "position_driver_hours": sum(x["driver_hours_position"] for x in rs)}
    downloads = Counter("cached" if x["cache_state_before_load"] == "CACHED_PREEXISTING" else "new" for x in all_sources)
    downloads["failed"] = sum(x.get("retrieval_status") != "SUCCESS" for x in all_sources)
    archive_hashes_after = {str(archive_dir / f"APEX-R_Telemetry_{y}.zip"): sha256_file(archive_dir / f"APEX-R_Telemetry_{y}.zip") for y in YEARS}
    prior_package_hash_after = sha256_file(prior_package) if prior_package.is_file() else ""
    coverage_metrics = {"approved_races": len(all_races), "telemetry_race_passes": race_pass, "telemetry_race_incomplete": len(all_races) - race_pass, "car_observations": sum(x["car_rows"] for x in all_races), "position_observations": sum(x["position_rows"] for x in all_races), "car_driver_hours": sum(x["driver_hours_car"] for x in all_races), "position_driver_hours": sum(x["driver_hours_position"] for x in all_races), "car_gap_count_over_5s": sum(int(x.get("car_gap_count_over_5s", 0)) for x in all_coverage), "car_gap_seconds_over_5s": sum(float(x.get("car_gap_seconds_over_5s", 0.0)) for x in all_coverage), "position_gap_count_over_5s": sum(int(x.get("position_gap_count_over_5s", 0)) for x in all_coverage), "position_gap_seconds_over_5s": sum(float(x.get("position_gap_seconds_over_5s", 0.0)) for x in all_coverage), "car_duplicate_timestamp_count": sum(int(x.get("car_duplicate_timestamp_count", 0)) for x in all_coverage), "position_duplicate_timestamp_count": sum(int(x.get("position_duplicate_timestamp_count", 0)) for x in all_coverage), "car_nonmonotonic_count": sum(int(x.get("car_nonmonotonic_count", 0)) for x in all_coverage), "position_nonmonotonic_count": sum(int(x.get("position_nonmonotonic_count", 0)) for x in all_coverage)}
    artifact_rows = {"source_archive_inventory.csv": len(archive_inventory), "source_archive_inventory.json": len(archive_inventory), "supplied_telemetry_inventory.csv": len(supplied), "supplied_telemetry_inventory.json": len(supplied), "validation_three_races.csv": len(validation_records), "validation_three_races.json": len(validation_records), "validation_three_race_coverage.csv": len(validation_coverage), "validation_three_race_coverage.json": len(validation_coverage), "race_collection_status.csv": len(all_races), "race_collection_status.json": len(all_races), "telemetry_coverage.csv": len(all_coverage), "telemetry_coverage.json": len(all_coverage), "fastf1_source_manifest.csv": len(all_sources), "fastf1_source_manifest.json": len(all_sources), "telemetry_car.csv.gz": car_writer.count, "telemetry_position.csv.gz": pos_writer.count}
    metrics = {"version": VERSION, "generated_at_utc": utc_now(), "scope": {"approved_races": len(split_by_race), "years": list(YEARS), "session": SESSION, "protected_exclusion_enforced_before_fetch": True, "network_provider": "FastF1", "openf1_requests": 0, "protected_holdout_accessed": False, "development_test_preserved_as_development": True, "source_archive_fields": source_fields}, "supplied_archive": supplied_stats, "validation_gate": validation_gate, "coverage": coverage_metrics, "split_coverage": split_coverage, "downloads": {"new_download_races": downloads["new"], "cached_races": downloads["cached"], "failed_races": downloads["failed"]}, "events": {"preserved_position_swap_candidates": candidate_rows, "preserved_position_swap_candidates_sha256": candidate_hash, "verified_overtake_events": 0}, "source_archive_hashes_before": archive_hashes_before, "source_archive_hashes_after": archive_hashes_after, "source_archive_immutable": archive_hashes_before == archive_hashes_after, "prior_package_sha256_before": prior_package_hash_before, "prior_package_sha256_after": prior_package_hash_after, "prior_package_immutable": prior_package_hash_before == prior_package_hash_after, "artifact_rows": artifact_rows, "notes": ["Supplied archive timestamps are lap-relative and selected-lap only.", "FastF1 raw car_data and pos_data were exported without interpolation or car-position nearest-time joins.", "Telemetry gaps over five seconds remain explicit in telemetry_coverage.csv.", "Position-swap candidates remain unresolved and are not labels.", "OpenF1 was not called because this collection scope is 2018-2022 and its free historical coverage begins in 2023."]}
    json_write(output / "metrics.json", metrics)
    build_report(output, metrics)
    # README is included in the package and is written here so the exact
    # executed command and source limitations travel with the data.
    validation_races_text = ", ".join(validation_race_ids)
    readme = f"""# APEX-R continuous telemetry collection\n\nRun from the application root:\n\n```bash\nsource .venv_tracinginsights/bin/activate\npython3 full_race_telemetry_collection/continuous_telemetry_pipeline.py --workspace /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application --output /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/full_race_telemetry_collection_final\npython3 full_race_telemetry_collection/verify_telemetry_collection.py --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/full_race_telemetry_collection_final\npython3 full_race_telemetry_collection/test_telemetry_collection.py --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/full_race_telemetry_collection_final\n```\n\nFastF1 {all_sources[0].get('fastf1_version', 'runtime version')} was used for historical 2018–2022 Race telemetry. The five supplied archive ZIPs were inventoried before retrieval. The protected exclusion manifest was enforced before any FastF1 session request. The pipeline validated these three train races before expansion: {validation_races_text}.\n\n`telemetry_car.csv.gz` contains raw car stream fields: speed, throttle, brake, gear, RPM, DRS, Date and SessionTime. `telemetry_position.csv.gz` contains raw X/Y/Z/status fields and the same timestamp fields. There is no interpolation or implicit car-position join. `telemetry_coverage.csv` records driver-hours, timestamp intervals, gaps over five seconds and lap-context comparison.\n\nThe prior position-swap candidate set is preserved by SHA-256 in `metrics.json`; it remains unresolved. This package contains no overtake labels and no model result. Raw FastF1 cache payloads are cleaned after collection; source URLs are represented by provider/session metadata and retained cache digests.\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifact_rows = metrics["artifact_rows"]
    # README was written after metrics; it is still packaged, while row-count
    # expectations remain unchanged. Rebuild report/checksum/package now.
    package_result = package_and_verify(output, artifact_rows)
    metrics["package"] = {"verification_passed": package_result["all_passed"], "compression": "ZIP_DEFLATED", "member_count": package_result["package"]["member_count"]}
    json_write(output / "metrics.json", metrics)
    build_report(output, metrics)
    package_result = package_and_verify(output, artifact_rows)
    # Final external verification is written beside, not inside, the ZIP so
    # it cannot invalidate its own checksum.
    package_result["fresh_extraction_cleaned"] = not Path(package_result["verification_directory"]).exists()
    json_write(output / "package_verification.json", package_result)
    print(json.dumps({"output": str(output), "approved_races": len(all_races), "telemetry_pass_races": race_pass, "car_observations": metrics["coverage"]["car_observations"], "position_observations": metrics["coverage"]["position_observations"], "car_driver_hours": metrics["coverage"]["car_driver_hours"], "position_driver_hours": metrics["coverage"]["position_driver_hours"], "new_fastf1_races": metrics["downloads"]["new_download_races"], "cached_fastf1_races": metrics["downloads"]["cached_races"], "failed_fastf1_races": metrics["downloads"]["failed_races"], "package_bytes": package_result["package"]["bytes"], "package_sha256": package_result["package"]["sha256"], "package_verified": package_result["all_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")
    raise SystemExit(main())
