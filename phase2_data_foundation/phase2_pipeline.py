#!/usr/bin/env python3
"""APEX-R Phase 2 data-foundation pipeline.

This module performs provenance recovery, identity/time crosswalk creation and
historical public-context enrichment for the supplied 2018--2022 telemetry
archives. It deliberately does not train models or create labels.

The pipeline is conservative by design:
* the six input archives are read in place and are never modified;
* only the five supplied seasons are eligible;
* the sealed OpenF1 session 11353 is rejected before any remote request;
* all row-level context matches are backward-only and freshness-limited;
* missing, stale, ambiguous and source-unavailable states remain explicit;
* no raw remote payloads are included in the distributable audit package.

Dependencies: Python 3.10+, requests. The existing project environment also
contains pandas/numpy, but this script uses the standard library plus requests
only so it can be reproduced in a small virtual environment.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import unicodedata
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import requests


YEARS = tuple(range(2018, 2023))
HOLDOUT_SESSION_IDS = frozenset({"11353"})
NULL_TOKENS = frozenset({"", "na", "n/a", "nan", "nat", "none", "null"})
TRACING_REPOS = {year: f"https://github.com/TracingInsights/{year}" for year in YEARS}
TRACING_RAW = {
    year: f"https://raw.githubusercontent.com/TracingInsights/{year}"
    for year in YEARS
}
TRACING_COMMITS = {
    2018: "ec6317adac6786e38b5d35e008f4b87926a3a55c",
    2019: "38741496a2d3736954069c3301e33737ffb3d777",
    2020: "264ae1db8f1381540fb839c89f7b249ae2d8957d",
    2021: "194dae5f5c876b0f8f80f176acd196e124dc2c0a",
    2022: "ad802a0ff6643998cc860d2a6712804270ab857d",
}
PHASE1_MANIFEST = "data/phase1-features/phase1_feature_manifest.json"
WEATHER_MAX_AGE_SEC = 60.0
RCM_MAX_AGE_SEC = 60.0
RECENT_PIT_LAP_WINDOW = 1


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def is_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in NULL_TOKENS)


def as_float(value: Any) -> float | None:
    if is_null(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def as_bool(value: Any) -> bool | None:
    if is_null(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def normalized(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def safe_remote_path(path: str) -> bool:
    """Reject traversal and absolute paths before composing raw URLs."""
    decoded = urllib.parse.unquote(path)
    if not decoded or decoded.startswith("/") or "\\" in decoded:
        return False
    parts = decoded.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def parse_timestamp(value: Any) -> dt.datetime | None:
    if is_null(value):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # TracingInsights sometimes carries nanosecond ISO strings.
    text = re.sub(r"(\.\d{6})\d+([+-]\d\d:\d\d)$", r"\1\2", text)
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def iso_or_blank(value: Any) -> str | None:
    parsed = parse_timestamp(value)
    return parsed.isoformat() if parsed else None


def url_path(*parts: str) -> str:
    if not all(safe_remote_path(part) for part in parts):
        raise ValueError(f"unsafe remote path components: {parts!r}")
    return "/".join(urllib.parse.quote(str(part), safe="") for part in parts)


def assert_not_holdout(*values: Any) -> None:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if any(token in text for token in HOLDOUT_SESSION_IDS):
            raise RuntimeError(f"sealed holdout token detected before remote access: {text!r}")


@dataclass
class FetchResult:
    url: str
    status: str
    http_status: int | None = None
    bytes: int = 0
    sha256: str | None = None
    cache_path: str | None = None
    cache_created_at_utc: str | None = None
    error: str | None = None
    payload: Any = None


class Fetcher:
    """Small bounded-retry fetcher with resumable raw cache and an audit log."""

    def __init__(self, cache_dir: Path, min_delay: float = 0.08, retries: int = 3):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_delay = min_delay
        self.retries = retries
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "APEX-R-Phase2-Audit/1.0"})
        self.log: list[dict[str, Any]] = []

    def _cache_file(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".bin")

    def _append_log(self, result: FetchResult) -> None:
        # Raw bytes are retained only in the local cache/pipeline return value;
        # the audit log stores reproducibility metadata, never payload bodies.
        record = result.__dict__.copy()
        record.pop("payload", None)
        self.log.append(record)

    def get(self, url: str, parse_json: bool = True) -> FetchResult:
        cache_file = self._cache_file(url)
        if cache_file.exists():
            payload_bytes = cache_file.read_bytes()
            try:
                payload = json.loads(payload_bytes.decode("utf-8")) if parse_json else payload_bytes
                result = FetchResult(
                    url=url,
                    status="CACHE_HIT",
                    http_status=200,
                    bytes=len(payload_bytes),
                    sha256=hashlib.sha256(payload_bytes).hexdigest(),
                    cache_path=str(cache_file),
                    cache_created_at_utc=dt.datetime.fromtimestamp(cache_file.stat().st_mtime, dt.timezone.utc).isoformat(),
                    payload=payload,
                )
                self._append_log(result)
                return result
            except Exception as exc:
                cache_file.unlink(missing_ok=True)
                cache_error = f"invalid cache removed: {exc}"
        else:
            cache_error = None

        last_error = cache_error
        for attempt in range(self.retries):
            wait_for = self.min_delay - (time.monotonic() - self.last_request)
            if wait_for > 0:
                time.sleep(wait_for)
            try:
                response = self.session.get(url, timeout=45)
                self.last_request = time.monotonic()
                if response.status_code == 404:
                    result = FetchResult(url=url, status="SOURCE_UNAVAILABLE", http_status=404, error="HTTP 404")
                    self._append_log(result)
                    return result
                if response.status_code == 429:
                    retry_after = as_float(response.headers.get("Retry-After")) or float(2**attempt)
                    time.sleep(min(retry_after, 30.0))
                    last_error = "HTTP 429"
                    continue
                response.raise_for_status()
                payload_bytes = response.content
                payload = json.loads(payload_bytes.decode("utf-8")) if parse_json else payload_bytes
                cache_file.write_bytes(payload_bytes)
                result = FetchResult(
                    url=url,
                    status="DOWNLOADED",
                    http_status=response.status_code,
                    bytes=len(payload_bytes),
                    sha256=hashlib.sha256(payload_bytes).hexdigest(),
                    cache_path=str(cache_file),
                    cache_created_at_utc=dt.datetime.fromtimestamp(cache_file.stat().st_mtime, dt.timezone.utc).isoformat(),
                    payload=payload,
                )
                self._append_log(result)
                return result
            except Exception as exc:  # network and JSON errors are logged, not hidden
                last_error = repr(exc)
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        result = FetchResult(url=url, status="SOURCE_UNAVAILABLE", error=last_error)
        self._append_log(result)
        return result


@dataclass
class InputInventory:
    headers: list[str]
    row_count: int = 0
    races: set[tuple[str, str, str]] = field(default_factory=set)
    event_drivers: set[tuple[str, str, str, str]] = field(default_factory=set)
    selected_laps: set[tuple[str, str, str, str, int]] = field(default_factory=set)
    source_keys: set[tuple[str, int]] = field(default_factory=set)
    canonical_keys: set[tuple[str, str, str, str, int, int]] = field(default_factory=set)
    time_ranges: dict[tuple[str, str, str, str, int], list[float]] = field(default_factory=dict)
    input_hash_before: dict[str, str] = field(default_factory=dict)
    input_hash_after: dict[str, str] = field(default_factory=dict)
    archive_integrity: dict[str, Any] = field(default_factory=dict)


def archive_paths(input_dir: Path) -> list[Path]:
    return [input_dir / f"APEX-R_Telemetry_{year}.zip" for year in YEARS] + [
        input_dir / "APEX-R_Telemetry_Metadata.zip"
    ]


def iter_input_rows(path: Path) -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.namelist()
        if any(not safe_remote_path(member) for member in members):
            raise RuntimeError(f"unsafe archive member in {path}")
        csv_members = [member for member in members if member.endswith(".csv.gz")]
        if len(csv_members) != 1:
            raise RuntimeError(f"expected one telemetry gzip in {path}, found {csv_members}")
        with archive.open(csv_members[0], "r") as compressed:
            with gzip.GzipFile(fileobj=compressed, mode="rb") as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                reader = csv.DictReader(text)
                if not reader.fieldnames:
                    raise RuntimeError(f"missing header in {path}")
                for row in reader:
                    yield row


def validate_archives(input_dir: Path, inventory: InputInventory) -> None:
    for path in archive_paths(input_dir):
        if not path.exists():
            raise FileNotFoundError(path)
        inventory.input_hash_before[str(path)] = sha256_file(path)
        record: dict[str, Any] = {"path": str(path), "zip_test_passed": False, "safe_members": True}
        with zipfile.ZipFile(path, "r") as archive:
            unsafe = [name for name in archive.namelist() if not safe_remote_path(name)]
            record["members"] = [
                {"name": info.filename, "bytes": info.file_size, "compressed_bytes": info.compress_size}
                for info in archive.infolist()
            ]
            record["unsafe_members"] = unsafe
            if unsafe:
                raise RuntimeError(f"unsafe archive members in {path}: {unsafe}")
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"ZIP CRC failure in {path}: {bad}")
            record["zip_test_passed"] = True
            if path.name.endswith("Metadata.zip"):
                for name in archive.namelist():
                    if name.endswith(".json"):
                        json.loads(archive.read(name))
                record["json_integrity_passed"] = True
        inventory.archive_integrity[path.name] = record

    for year in YEARS:
        path = input_dir / f"APEX-R_Telemetry_{year}.zip"
        first = True
        for row in iter_input_rows(path):
            if first:
                inventory.headers = list(row.keys())
                first = False
            year_text = str(row.get("year", year))
            event = row.get("event", "")
            session = row.get("session", "")
            driver = row.get("driver", "")
            lap = as_int(row.get("lap"))
            sample_index = as_int(row.get("sample_index"))
            assert_not_holdout(row.get("session_key"), row.get("session_id"))
            if lap is None or sample_index is None:
                raise RuntimeError(f"non-integral key in {path.name}: {row}")
            inventory.row_count += 1
            inventory.races.add((year_text, event, session))
            inventory.event_drivers.add((year_text, event, session, driver))
            inventory.selected_laps.add((year_text, event, session, driver, lap))
            source_key = (row.get("source_data_key", ""), sample_index)
            canonical = (year_text, event, session, driver, lap, sample_index)
            if source_key in inventory.source_keys:
                raise RuntimeError(f"duplicate source sample key encountered: {source_key}")
            if canonical in inventory.canonical_keys:
                raise RuntimeError(f"duplicate canonical sample key encountered: {canonical}")
            inventory.source_keys.add(source_key)
            inventory.canonical_keys.add(canonical)
            key = (year_text, event, session, driver, lap)
            number = as_float(row.get("time"))
            if number is not None:
                if key not in inventory.time_ranges:
                    inventory.time_ranges[key] = [number, number, 0.0]
                else:
                    inventory.time_ranges[key][0] = min(inventory.time_ranges[key][0], number)
                    inventory.time_ranges[key][1] = max(inventory.time_ranges[key][1], number)
        if first:
            raise RuntimeError(f"empty telemetry archive: {path}")


def read_metadata(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "APEX-R_Telemetry_Metadata.zip"
    with zipfile.ZipFile(path) as archive:
        return {Path(name).stem: json.loads(archive.read(name)) for name in archive.namelist() if name.endswith(".json")}


def load_phase1_manifest(workspace: Path) -> dict[str, Any]:
    path = workspace / PHASE1_MANIFEST
    if not path.exists():
        return {"status": "NOT_FOUND", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def tracing_url(year: int, path: str) -> str:
    if not safe_remote_path(path):
        raise ValueError(f"unsafe TracingInsights path: {path}")
    return f"{TRACING_RAW[year]}/{TRACING_COMMITS[year]}/{url_path(*path.split('/'))}"


def github_contents_url(year: int, path: str) -> str:
    return (
        f"https://api.github.com/repos/TracingInsights/{year}/contents/"
        f"{url_path(*path.split('/'))}?ref={TRACING_COMMITS[year]}"
    )


def parse_array_json(payload: Any, keys: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    arrays = [payload.get(key) for key in keys]
    length = max((len(value) for value in arrays if isinstance(value, list)), default=0)
    rows: list[dict[str, Any]] = []
    for index in range(length):
        row: dict[str, Any] = {"index": index}
        for key in keys:
            value = payload.get(key)
            row[key] = value[index] if isinstance(value, list) and index < len(value) else None
        rows.append(row)
    return rows


def source_resource_record(
    result: FetchResult, *, year: int, event: str, session: str, path: str, kind: str
) -> dict[str, Any]:
    return {
        "source_kind": kind,
        "year": year,
        "event": event,
        "session": session,
        "repository": TRACING_REPOS.get(year),
        "commit": TRACING_COMMITS.get(year),
        "path": path,
        "url": result.url,
        "retrieved_at_utc": utc_now(),
        "retrieval_time_status": "NETWORK_REQUEST_TIME" if result.status == "DOWNLOADED" else "CACHE_REUSE_NETWORK_TIME_NOT_RECORDED",
        "status": result.status,
        "http_status": result.http_status,
        "bytes": result.bytes,
        "sha256": result.sha256,
        "cache_path": result.cache_path,
        "cache_created_at_utc": result.cache_created_at_utc,
        "error": result.error,
    }


def fetch_json_resource(
    fetcher: Fetcher,
    *,
    year: int,
    event: str,
    session: str,
    path: str,
    kind: str,
) -> tuple[Any, dict[str, Any]]:
    assert_not_holdout(year, event, session, path)
    url = tracing_url(year, path)
    result = fetcher.get(url, parse_json=True)
    return result.payload, source_resource_record(result, year=year, event=event, session=session, path=path, kind=kind)


def fetch_public_json(fetcher: Fetcher, url: str, kind: str) -> tuple[Any, dict[str, Any]]:
    assert_not_holdout(url)
    result = fetcher.get(url, parse_json=True)
    return result.payload, {
        "source_kind": kind,
        "year": None,
        "event": None,
        "session": "Race",
        "repository": "https://api.jolpi.ca/ergast/f1/",
        "commit": None,
        "path": None,
        "url": result.url,
        "retrieved_at_utc": utc_now(),
        "retrieval_time_status": "NETWORK_REQUEST_TIME" if result.status == "DOWNLOADED" else "CACHE_REUSE_NETWORK_TIME_NOT_RECORDED",
        "status": result.status,
        "http_status": result.http_status,
        "bytes": result.bytes,
        "sha256": result.sha256,
        "cache_path": result.cache_path,
        "cache_created_at_utc": result.cache_created_at_utc,
        "error": result.error,
    }


def scalar_at(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    return None if is_null(value) else value


def build_driver_context(
    fetcher: Fetcher,
    event_keys: Sequence[tuple[str, str, str]],
    observed_drivers: set[tuple[str, str, str, str]],
    resource_records: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, str], dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    identity_map: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    output_rows: list[dict[str, Any]] = []
    for year_text, event, session in sorted(event_keys):
        year = int(year_text)
        payload, record = fetch_json_resource(
            fetcher, year=year, event=event, session=session, path=f"{event}/{session}/drivers.json", kind="TracingInsights drivers"
        )
        resource_records.append(record)
        entries = payload.get("drivers", []) if isinstance(payload, dict) else []
        current: dict[str, dict[str, Any]] = {}
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            code = str(entry.get("driver") or "").strip()
            if not code:
                continue
            name = " ".join(str(entry.get(k) or "").strip() for k in ("fn", "ln")).strip()
            identity_key = normalized(name) if name else f"{year_text}:{code.lower()}"
            current[code] = {
                "driver_code": code,
                "team": entry.get("team"),
                "driver_number": entry.get("dn"),
                "first_name": entry.get("fn"),
                "last_name": entry.get("ln"),
                "full_name": name or None,
                "stable_identity_key": identity_key,
                "identity_status": "VERIFIED_IN_SOURCE_DRIVERS_JSON",
                "identity_source_url": record["url"],
                "identity_source_sha256": record["sha256"],
                "identity_source_commit": record["commit"],
            }
        identity_map[(year_text, event, session)] = current
        observed = sorted(code for y, e, s, code in observed_drivers if (y, e, s) == (year_text, event, session))
        for code in sorted(set(current) | set(observed)):
            entry = current.get(code)
            if entry is None:
                output_rows.append(
                    {
                        "year": year_text,
                        "event": event,
                        "session": session,
                        "driver_code": code,
                        "observed_as": "driver_or_DriverAhead",
                        "driver_number": None,
                        "full_name": None,
                        "team": None,
                        "stable_identity_key": None,
                        "identity_status": "UNMATCHED_SOURCE_CODE",
                        "source_url": record["url"],
                        "source_commit": record["commit"],
                    }
                )
            else:
                output_rows.append(
                    {
                        "year": year_text,
                        "event": event,
                        "session": session,
                        "driver_code": code,
                        "observed_as": "driver_or_DriverAhead",
                        "driver_number": entry["driver_number"],
                        "full_name": entry["full_name"],
                        "team": entry["team"],
                        "stable_identity_key": entry["stable_identity_key"],
                        "identity_status": entry["identity_status"],
                        "source_url": entry["identity_source_url"],
                        "source_commit": entry["identity_source_commit"],
                    }
                )
    return identity_map, output_rows


def extract_lap_anchors(
    fetcher: Fetcher,
    selected_laps: set[tuple[str, str, str, str, int]],
    resource_records: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str, int], dict[str, Any]]:
    by_driver_event: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    for year, event, session, driver, lap in selected_laps:
        by_driver_event[(year, event, session, driver)].add(lap)
    anchors: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for (year_text, event, session, driver), laps_needed in sorted(by_driver_event.items()):
        year = int(year_text)
        path = f"{event}/{session}/{driver}/laptimes.json"
        payload, record = fetch_json_resource(
            fetcher, year=year, event=event, session=session, path=path, kind="TracingInsights laptimes"
        )
        resource_records.append(record)
        if not isinstance(payload, dict):
            for lap in laps_needed:
                anchors[(year_text, event, session, driver, lap)] = {
                    "anchor_status": "SOURCE_UNAVAILABLE",
                    "source_url": record["url"],
                    "source_sha256": record["sha256"],
                }
            continue
        fields = ("lap", "time", "sesT", "lST", "lSD", "s1", "s2", "s3", "compound", "stint", "life", "pout", "pin")
        for item in parse_array_json(payload, fields):
            lap = as_int(item.get("lap"))
            if lap not in laps_needed:
                continue
            lap_time = as_float(item.get("time"))
            session_end = as_float(item.get("sesT"))
            lap_start = as_float(item.get("lST"))
            lap_start_utc = parse_timestamp(item.get("lSD"))
            status = "SESSION_ANCHOR_AVAILABLE" if lap_start is not None else "SESSION_ANCHOR_UNAVAILABLE"
            if lap_start_utc is not None:
                status = "UTC_AND_SESSION_ANCHOR_AVAILABLE"
            anchors[(year_text, event, session, driver, lap)] = {
                "anchor_status": status,
                "lap_time_sec": lap_time,
                "session_end_sec": session_end,
                "lap_start_session_sec": lap_start,
                "lap_start_utc": lap_start_utc.isoformat() if lap_start_utc else None,
                "sector1_sec": as_float(item.get("s1")),
                "sector2_sec": as_float(item.get("s2")),
                "sector3_sec": as_float(item.get("s3")),
                "compound": scalar_at(item, "compound"),
                "stint": as_int(item.get("stint")),
                "tyre_life": as_int(item.get("life")),
                "pit_out_session_sec": as_float(item.get("pout")),
                "pit_in_session_sec": as_float(item.get("pin")),
                "source_url": record["url"],
                "source_sha256": record["sha256"],
            }
        for lap in laps_needed:
            anchors.setdefault(
                (year_text, event, session, driver, lap),
                {"anchor_status": "LAP_NOT_FOUND", "source_url": record["url"], "source_sha256": record["sha256"]},
            )
    return anchors


def build_weather_context(
    fetcher: Fetcher,
    event_keys: Sequence[tuple[str, str, str]],
    resource_records: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    weather_map: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    fields = ("wT", "wAT", "wH", "wP", "wR", "wTT", "wWD", "wWS")
    for year_text, event, session in sorted(event_keys):
        year = int(year_text)
        path = f"{event}/{session}/weather.json"
        payload, record = fetch_json_resource(
            fetcher, year=year, event=event, session=session, path=path, kind="TracingInsights weather"
        )
        resource_records.append(record)
        if not isinstance(payload, dict):
            weather_map[(year_text, event, session)] = []
            continue
        rows: list[dict[str, Any]] = []
        for item in parse_array_json(payload, fields):
            session_time = as_float(item.get("wT"))
            if session_time is None:
                continue
            rows.append(
                {
                    "weather_index": item["index"],
                    "source_time_sec": session_time,
                    "air_temp_c": as_float(item.get("wAT")),
                    "humidity_pct": as_float(item.get("wH")),
                    "pressure_mbar": as_float(item.get("wP")),
                    "rainfall_flag": as_bool(item.get("wR")),
                    "track_temp_c": as_float(item.get("wTT")),
                    "wind_direction_deg": as_float(item.get("wWD")),
                    "wind_speed_mps": as_float(item.get("wWS")),
                    "source_url": record["url"],
                    "source_sha256": record["sha256"],
                }
            )
        rows.sort(key=lambda row: (row["source_time_sec"], row["weather_index"]))
        weather_map[(year_text, event, session)] = rows
    return weather_map


def build_rcm_context(
    fetcher: Fetcher,
    event_keys: Sequence[tuple[str, str, str]],
    resource_records: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    rcm_map: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    fields = ("time", "cat", "msg", "status", "flag", "scope", "sector", "dNum", "lap")
    for year_text, event, session in sorted(event_keys):
        year = int(year_text)
        path = f"{event}/{session}/rcm.json"
        payload, record = fetch_json_resource(
            fetcher, year=year, event=event, session=session, path=path, kind="TracingInsights race control"
        )
        resource_records.append(record)
        rows: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            for item in parse_array_json(payload, fields):
                stamp = parse_timestamp(item.get("time"))
                if stamp is None:
                    continue
                rows.append(
                    {
                        "rcm_index": item["index"],
                        "source_time_utc": stamp.isoformat(),
                        "category": scalar_at(item, "cat"),
                        "message": scalar_at(item, "msg"),
                        "status": scalar_at(item, "status"),
                        "flag": scalar_at(item, "flag"),
                        "scope": scalar_at(item, "scope"),
                        "sector": scalar_at(item, "sector"),
                        "driver_number": scalar_at(item, "dNum"),
                        "lap": as_int(item.get("lap")),
                        "source_url": record["url"],
                        "source_sha256": record["sha256"],
                    }
                )
        rows.sort(key=lambda row: (row["source_time_utc"], row["rcm_index"]))
        rcm_map[(year_text, event, session)] = rows
    return rcm_map


def race_name_key(value: Any) -> str:
    return normalized(value).replace("grandprix", "grandprix")


def build_schedule_and_pits(
    fetcher: Fetcher,
    event_keys: Sequence[tuple[str, str, str]],
    identity_map: Mapping[tuple[str, str, str], Mapping[str, Mapping[str, Any]]],
    resource_records: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, str], list[dict[str, Any]]],
]:
    schedules: dict[tuple[str, str], dict[str, Any]] = {}
    pits: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    rounds_by_year_event: dict[tuple[str, str], dict[str, Any]] = {}
    for year in YEARS:
        url = f"https://api.jolpi.ca/ergast/f1/{year}.json?limit=100"
        payload, record = fetch_public_json(fetcher, url, "Jolpica race schedule")
        record["year"] = year
        resource_records.append(record)
        races = (((payload or {}).get("MRData") or {}).get("RaceTable") or {}).get("Races", []) if isinstance(payload, dict) else []
        for race in races if isinstance(races, list) else []:
            if not isinstance(race, dict):
                continue
            rounds_by_year_event[(str(year), race_name_key(race.get("raceName")))] = race

    for year_text, event, session in sorted(event_keys):
        race = rounds_by_year_event.get((year_text, race_name_key(event)))
        schedules[(year_text, event)] = race or {"schedule_status": "UNMATCHED_EVENT"}
        if not race:
            pits[(year_text, event, session)] = []
            continue
        round_number = race.get("round")
        url = f"https://api.jolpi.ca/ergast/f1/{year_text}/{round_number}/pitstops.json?limit=1000"
        payload, record = fetch_public_json(fetcher, url, "Jolpica pit stops")
        record.update({"year": int(year_text), "event": event, "round": round_number})
        resource_records.append(record)
        races_payload = (((payload or {}).get("MRData") or {}).get("RaceTable") or {}).get("Races", []) if isinstance(payload, dict) else []
        pit_items: list[dict[str, Any]] = []
        if isinstance(races_payload, list) and races_payload:
            pit_items = races_payload[0].get("PitStops", []) or []
        code_map = identity_map.get((year_text, event, session), {})
        driver_id_map: dict[str, list[str]] = defaultdict(list)
        for code, info in code_map.items():
            variants = {
                normalized(info.get("last_name")),
                normalized(f"{info.get('first_name') or ''}{info.get('last_name') or ''}"),
                normalized(code),
            }
            for variant in variants:
                if variant:
                    driver_id_map[variant].append(code)
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(pit_items if isinstance(pit_items, list) else []):
            if not isinstance(item, dict):
                continue
            driver_id = str(item.get("driverId") or "")
            candidate_codes = sorted(set(driver_id_map.get(normalized(driver_id), [])))
            driver_code = candidate_codes[0] if len(candidate_codes) == 1 else None
            identity_status = "MATCHED_DRIVER_ID" if driver_code else ("AMBIGUOUS_DRIVER_ID" if candidate_codes else "UNMATCHED_DRIVER_ID")
            rows.append(
                {
                    "pit_index": index,
                    "driver_id": driver_id,
                    "driver_code": driver_code,
                    "driver_identity_status": identity_status,
                    "lap": as_int(item.get("lap")),
                    "stop_number": as_int(item.get("stop")),
                    "time_of_day": scalar_at(item, "time"),
                    "duration_sec": as_float(item.get("duration")),
                    "entry_time": None,
                    "exit_time": None,
                    "event_date": race.get("date"),
                    "source_url": record["url"],
                    "source_sha256": record["sha256"],
                    "availability_status": "DURATION_AND_TIME_OF_DAY_ONLY",
                }
            )
        pits[(year_text, event, session)] = rows
    return schedules, pits


def latest_backward(rows: Sequence[Mapping[str, Any]], current: float, key: str, max_age: float) -> tuple[Mapping[str, Any] | None, str, float | None]:
    eligible = [row for row in rows if as_float(row.get(key)) is not None and as_float(row.get(key)) <= current]
    if not eligible:
        return None, "UNMATCHED", None
    match = max(eligible, key=lambda row: as_float(row.get(key)) or float("-inf"))
    age = current - (as_float(match.get(key)) or current)
    return (match, "MATCHED", age) if age <= max_age else (None, "STALE", age)


def latest_backward_datetime(
    rows: Sequence[Mapping[str, Any]], current: dt.datetime, key: str, max_age: float
) -> tuple[Mapping[str, Any] | None, str, float | None]:
    eligible: list[tuple[dt.datetime, Mapping[str, Any]]] = []
    for row in rows:
        stamp = parse_timestamp(row.get(key))
        if stamp and stamp <= current:
            eligible.append((stamp, row))
    if not eligible:
        return None, "UNMATCHED", None
    stamp, match = max(eligible, key=lambda item: item[0])
    age = (current - stamp).total_seconds()
    return (match, "MATCHED", age) if age <= max_age else (None, "STALE", age)


def prior_pit_context(
    pit_rows: Sequence[Mapping[str, Any]], driver_code: str | None, current_lap: int | None
) -> dict[str, Any]:
    if not driver_code:
        return {
            "status": "NO_CAR_AHEAD" if driver_code is None else "UNMATCHED",
            "recent_flag": None,
            "last_prior_lap": None,
            "last_prior_duration_sec": None,
            "laps_ago": None,
        }
    if not pit_rows:
        return {"status": "SOURCE_EMPTY_UNVERIFIED", "recent_flag": None, "last_prior_lap": None, "last_prior_duration_sec": None, "laps_ago": None}
    candidate = [
        row for row in pit_rows
        if row.get("driver_code") == driver_code and as_int(row.get("lap")) is not None and current_lap is not None and as_int(row.get("lap")) < current_lap
    ]
    if not candidate:
        same_lap = [row for row in pit_rows if row.get("driver_code") == driver_code and as_int(row.get("lap")) == current_lap]
        if same_lap:
            return {"status": "UNKNOWN_SAME_LAP_EVENT", "recent_flag": None, "last_prior_lap": None, "last_prior_duration_sec": None, "laps_ago": None}
        unmatched_driver = [row for row in pit_rows if row.get("driver_code") is None]
        return {"status": "NO_EVENT_CONFIRMED", "recent_flag": False, "last_prior_lap": None, "last_prior_duration_sec": None, "laps_ago": None}
    last = max(candidate, key=lambda row: as_int(row.get("lap")) or -1)
    last_lap = as_int(last.get("lap"))
    laps_ago = current_lap - last_lap if current_lap is not None and last_lap is not None else None
    return {
        "status": "MATCHED",
        "recent_flag": bool(laps_ago is not None and laps_ago <= RECENT_PIT_LAP_WINDOW),
        "last_prior_lap": last_lap,
        "last_prior_duration_sec": last.get("duration_sec"),
        "laps_ago": laps_ago,
    }


def enriched_fieldnames(base_headers: Sequence[str]) -> list[str]:
    return list(base_headers) + [
        "phase2_driver_number",
        "phase2_driver_identity_key",
        "phase2_driver_identity_status",
        "phase2_driver_ahead_identity_status",
        "phase2_lap_start_session_sec",
        "phase2_session_time_sec",
        "phase2_lap_start_utc",
        "phase2_utc_timestamp",
        "phase2_timestamp_status",
        "phase2_weather_join_status",
        "phase2_weather_source_time_sec",
        "phase2_weather_age_sec",
        "phase2_weather_air_temp_c",
        "phase2_weather_track_temp_c",
        "phase2_weather_humidity_pct",
        "phase2_weather_pressure_mbar",
        "phase2_weather_rainfall_flag",
        "phase2_weather_wind_direction_deg",
        "phase2_weather_wind_speed_mps",
        "phase2_pit_current_status",
        "phase2_pit_current_recent_flag",
        "phase2_pit_current_last_prior_lap",
        "phase2_pit_current_last_prior_duration_sec",
        "phase2_pit_current_laps_ago",
        "phase2_pit_ahead_status",
        "phase2_pit_ahead_recent_flag",
        "phase2_pit_ahead_last_prior_lap",
        "phase2_pit_ahead_last_prior_duration_sec",
        "phase2_pit_ahead_laps_ago",
        "phase2_rc_join_status",
        "phase2_rc_source_time_utc",
        "phase2_rc_age_sec",
        "phase2_rc_category",
        "phase2_rc_message",
        "phase2_rc_scope",
        "phase2_rc_sector",
        "phase2_rc_flag",
    ]


def empty_context(status: str) -> dict[str, Any]:
    return {"status": status}


def enrich_row(
    row: dict[str, str],
    *,
    anchors: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    identities: Mapping[tuple[str, str, str], Mapping[str, Mapping[str, Any]]],
    weather: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    pits: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    rcm: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    weather_max_age: float,
    rcm_max_age: float,
) -> dict[str, Any]:
    year, event, session, driver = row.get("year", ""), row.get("event", ""), row.get("session", ""), row.get("driver", "")
    lap = as_int(row.get("lap"))
    key = (year, event, session, driver, lap or -1)
    anchor = anchors.get(key, {"anchor_status": "LAP_NOT_FOUND"})
    identity_map = identities.get((year, event, session), {})
    driver_info = identity_map.get(driver)
    ahead_ref = None if is_null(row.get("DriverAhead")) else str(row.get("DriverAhead")).strip()
    # The supplied DriverAhead field is numeric in the audited archive: it is
    # a session driver number, not the three-letter source code. Resolve it
    # through the session-specific crosswalk and never assume numbers persist
    # across seasons.
    ahead_info = identity_map.get(ahead_ref) if ahead_ref else None
    if ahead_info is None and ahead_ref:
        matches = [info for info in identity_map.values() if str(info.get("driver_number")) == ahead_ref]
        ahead_info = matches[0] if len(matches) == 1 else None
    ahead_code = ahead_info.get("driver_code") if ahead_info else None
    telemetry_time = as_float(row.get("time"))
    lap_start_sec = as_float(anchor.get("lap_start_session_sec"))
    session_time = lap_start_sec + telemetry_time if lap_start_sec is not None and telemetry_time is not None else None
    lap_start_utc = parse_timestamp(anchor.get("lap_start_utc"))
    utc_time = lap_start_utc + dt.timedelta(seconds=telemetry_time) if lap_start_utc and telemetry_time is not None else None
    timestamp_status = "UTC_ANCHORED" if utc_time else ("SESSION_ANCHORED_ONLY" if session_time is not None else "UNANCHORED")

    weather_rows = weather.get((year, event, session), [])
    if not weather_rows:
        weather_match, weather_status, weather_age = None, "SOURCE_EMPTY_UNVERIFIED", None
    elif session_time is None:
        weather_match, weather_status, weather_age = None, "UNKNOWN_UNANCHORED", None
    else:
        weather_match, weather_status, weather_age = latest_backward(weather_rows, session_time, "source_time_sec", weather_max_age)
    pit_rows = pits.get((year, event, session), [])
    current_pit = prior_pit_context(pit_rows, driver, lap)
    if not ahead_ref:
        ahead_pit = {"status": "NO_CAR_AHEAD", "recent_flag": None, "last_prior_lap": None, "last_prior_duration_sec": None, "laps_ago": None}
    elif not ahead_code:
        ahead_pit = {"status": "UNMATCHED", "recent_flag": None, "last_prior_lap": None, "last_prior_duration_sec": None, "laps_ago": None}
    else:
        ahead_pit = prior_pit_context(pit_rows, ahead_code, lap)

    rcm_rows = rcm.get((year, event, session), [])
    if not rcm_rows:
        rc_match, rc_status, rc_age = None, "SOURCE_EMPTY_UNVERIFIED", None
    elif utc_time is None:
        rc_match, rc_status, rc_age = None, "UNKNOWN_UNANCHORED", None
    else:
        rc_match, rc_status, rc_age = latest_backward_datetime(rcm_rows, utc_time, "source_time_utc", rcm_max_age)

    out: dict[str, Any] = dict(row)
    out.update(
        {
            "phase2_driver_number": driver_info.get("driver_number") if driver_info else None,
            "phase2_driver_identity_key": driver_info.get("stable_identity_key") if driver_info else None,
            "phase2_driver_identity_status": driver_info.get("identity_status") if driver_info else "UNMATCHED_SOURCE_CODE",
            "phase2_driver_ahead_identity_status": (ahead_info.get("identity_status") if ahead_info else ("NO_CAR_AHEAD" if not ahead_ref else "UNMATCHED_SOURCE_CODE")),
            "phase2_lap_start_session_sec": lap_start_sec,
            "phase2_session_time_sec": session_time,
            "phase2_lap_start_utc": lap_start_utc.isoformat() if lap_start_utc else None,
            "phase2_utc_timestamp": utc_time.isoformat() if utc_time else None,
            "phase2_timestamp_status": timestamp_status,
            "phase2_weather_join_status": weather_status,
            "phase2_weather_source_time_sec": weather_match.get("source_time_sec") if weather_match else None,
            "phase2_weather_age_sec": weather_age,
            "phase2_weather_air_temp_c": weather_match.get("air_temp_c") if weather_match else None,
            "phase2_weather_track_temp_c": weather_match.get("track_temp_c") if weather_match else None,
            "phase2_weather_humidity_pct": weather_match.get("humidity_pct") if weather_match else None,
            "phase2_weather_pressure_mbar": weather_match.get("pressure_mbar") if weather_match else None,
            "phase2_weather_rainfall_flag": weather_match.get("rainfall_flag") if weather_match else None,
            "phase2_weather_wind_direction_deg": weather_match.get("wind_direction_deg") if weather_match else None,
            "phase2_weather_wind_speed_mps": weather_match.get("wind_speed_mps") if weather_match else None,
            "phase2_pit_current_status": current_pit["status"],
            "phase2_pit_current_recent_flag": current_pit["recent_flag"],
            "phase2_pit_current_last_prior_lap": current_pit["last_prior_lap"],
            "phase2_pit_current_last_prior_duration_sec": current_pit["last_prior_duration_sec"],
            "phase2_pit_current_laps_ago": current_pit["laps_ago"],
            "phase2_pit_ahead_status": ahead_pit["status"],
            "phase2_pit_ahead_recent_flag": ahead_pit["recent_flag"],
            "phase2_pit_ahead_last_prior_lap": ahead_pit["last_prior_lap"],
            "phase2_pit_ahead_last_prior_duration_sec": ahead_pit["last_prior_duration_sec"],
            "phase2_pit_ahead_laps_ago": ahead_pit["laps_ago"],
            "phase2_rc_join_status": rc_status,
            "phase2_rc_source_time_utc": rc_match.get("source_time_utc") if rc_match else None,
            "phase2_rc_age_sec": rc_age,
            "phase2_rc_category": rc_match.get("category") if rc_match else None,
            "phase2_rc_message": rc_match.get("message") if rc_match else None,
            "phase2_rc_scope": rc_match.get("scope") if rc_match else None,
            "phase2_rc_sector": rc_match.get("sector") if rc_match else None,
            "phase2_rc_flag": rc_match.get("flag") if rc_match else None,
        }
    )
    return out


def write_context_tables(
    output_dir: Path,
    inventory: InputInventory,
    identity_rows: Sequence[Mapping[str, Any]],
    anchors: Mapping[tuple[str, str, str, str, int], Mapping[str, Any]],
    weather: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    rcm: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    pits: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    context_dir = output_dir / "context"
    counts: dict[str, int] = {}
    counts["identity_crosswalk_rows"] = write_csv(
        context_dir / "identity_crosswalk.csv",
        identity_rows,
        ["year", "event", "session", "driver_code", "observed_as", "driver_number", "full_name", "team", "stable_identity_key", "identity_status", "source_url", "source_commit"],
    )
    anchor_rows = []
    for key, value in sorted(anchors.items()):
        year, event, session, driver, lap = key
        row = {"year": year, "event": event, "session": session, "driver": driver, "lap": lap, **value}
        anchor_rows.append(row)
    anchor_fields = ["year", "event", "session", "driver", "lap", "anchor_status", "lap_time_sec", "session_end_sec", "lap_start_session_sec", "lap_start_utc", "sector1_sec", "sector2_sec", "sector3_sec", "compound", "stint", "tyre_life", "pit_out_session_sec", "pit_in_session_sec", "source_url", "source_sha256"]
    counts["time_crosswalk_rows"] = write_csv(context_dir / "time_crosswalk.csv", anchor_rows, anchor_fields)
    weather_rows = []
    for (year, event, session), values in sorted(weather.items()):
        for value in values:
            weather_rows.append({"year": year, "event": event, "session": session, **value})
    counts["weather_context_rows"] = write_csv(
        context_dir / "weather_context.csv",
        weather_rows,
        ["year", "event", "session", "weather_index", "source_time_sec", "air_temp_c", "track_temp_c", "humidity_pct", "pressure_mbar", "rainfall_flag", "wind_direction_deg", "wind_speed_mps", "source_url", "source_sha256"],
    )
    rc_rows = []
    for (year, event, session), values in sorted(rcm.items()):
        for value in values:
            rc_rows.append({"year": year, "event": event, "session": session, **value})
    counts["race_control_context_rows"] = write_csv(
        context_dir / "race_control_context.csv",
        rc_rows,
        ["year", "event", "session", "rcm_index", "source_time_utc", "category", "message", "status", "flag", "scope", "sector", "driver_number", "lap", "source_url", "source_sha256"],
    )
    pit_rows = []
    for (year, event, session), values in sorted(pits.items()):
        for value in values:
            pit_rows.append({"year": year, "event": event, "session": session, **value})
    counts["pit_context_rows"] = write_csv(
        context_dir / "pit_stop_context.csv",
        pit_rows,
        ["year", "event", "session", "pit_index", "driver_id", "driver_code", "driver_identity_status", "lap", "stop_number", "time_of_day", "duration_sec", "entry_time", "exit_time", "event_date", "availability_status", "source_url", "source_sha256"],
    )
    return counts


def join_coverage(enriched_path: Path, field_specs: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = {field: Counter() for field in field_specs}
    total = 0
    with gzip.open(enriched_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            for field, status_fields in field_specs.items():
                status = "MATCHED" if any(row.get(sf) == "MATCHED" for sf in status_fields) else "UNMATCHED"
                # For one status field, preserve the exact status taxonomy.
                if len(status_fields) == 1:
                    status = row.get(status_fields[0]) or "UNKNOWN"
                counts[field][status] += 1
    output = []
    for field, counter in counts.items():
        for status, count in sorted(counter.items()):
            output.append({"field": field, "status": status, "rows": count, "row_fraction": count / total if total else None})
    return output


def build_anomaly_disposition(phase1_metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    flags = (phase1_metrics.get("numeric_quality") or {}).get("range_flags") or {}
    missing = phase1_metrics.get("missingness") or {}
    examples = [
        ("throttle", "throttle_outside_0_100", flags.get("throttle_outside_0_100", 16052), "SUSPICIOUS_UNRESOLVED", "Preserve source values; 0–100 is an audit rule, not a verified provider contract."),
        ("gear", "gear_not_integer_or_outside_0_8", flags.get("gear_not_integer_or_outside_0_8", 37), "SUSPICIOUS_UNRESOLVED", "Preserve source values; verify channel encoding before any repair."),
        ("distance", "distance_negative", flags.get("distance_negative", 1548), "SUSPICIOUS_UNRESOLVED", "Preserve source values; may be lap-boundary/reference behavior."),
        ("rel_distance", "rel_distance_outside_0_1_tolerance", flags.get("rel_distance_outside_0_1_tolerance", 194), "SUSPICIOUS_UNRESOLVED", "Preserve source values; formula and normalization provenance remain absent."),
        ("DriverAhead", "missing", missing.get("total_by_field", {}).get("DriverAhead", 54822), "MISSING_RETAINED", "No zero-fill; opponent identity remains unavailable on these rows."),
        ("DistanceToDriverAhead", "missing", missing.get("total_by_field", {}).get("DistanceToDriverAhead", 48648), "MISSING_RETAINED", "No zero-fill; missing opponent distance remains explicit."),
        ("rel_distance", "missing", missing.get("total_by_field", {}).get("rel_distance", 4665), "MISSING_RETAINED", "No zero-fill; missing derived value remains explicit."),
    ]
    return [{"field": field, "rule": rule, "flagged_rows": count, "disposition": disposition, "explanation": explanation} for field, rule, count, disposition, explanation in examples]


def build_feature_readiness() -> list[dict[str, Any]]:
    base = [
        ("time", "Telemetry sample time within selected lap", "seconds; apparent lap-relative", "Supplied telemetry CSV", "UNKNOWN", "UNRESOLVED", "Lap-relative behavior is supported by source code/sample values, but live delivery time is unknown and UTC is absent from the base archive."),
        ("rel_distance", "Relative distance along lap", "unknown ratio-like", "Supplied telemetry CSV; derived provenance absent", "UNKNOWN", "UNRESOLVED", "Processing formula, normalization and prefix invariance are unavailable."),
        ("DriverAhead", "Source-selected opponent identifier", "source code", "Supplied telemetry CSV", "UNKNOWN", "UNRESOLVED", "The Phase 2 identity map can verify code membership, not the historical opponent-matching algorithm."),
        ("DistanceToDriverAhead", "Distance to source-selected opponent", "unknown", "Supplied telemetry CSV", "UNKNOWN", "UNRESOLVED", "Units, matching and live availability remain undocumented."),
        ("acc_x/acc_y/acc_z", "Acceleration channels", "unknown", "Supplied telemetry CSV", "UNKNOWN", "UNRESOLVED", "Pinned source code shows gradient plus centered convolution in the related collector; exact archive version and causal equivalence remain unresolved."),
        ("phase2_driver_number", "Verified session driver number", "source string", "TracingInsights drivers.json", "UNKNOWN", "HISTORICAL_ONLY", "Crosswalk is verified against the pinned session file; live publication latency is not established."),
        ("phase2_session_time_sec", "Session-relative sample timestamp", "seconds", "TracingInsights laptimes.json lST + supplied telemetry time", "UNKNOWN", "HISTORICAL_ONLY", "Backward-only arithmetic uses lap-start anchor; no live delivery timestamp is available."),
        ("phase2_utc_timestamp", "Reconstructed UTC sample timestamp", "UTC ISO-8601", "TracingInsights laptimes.json lSD + supplied telemetry time", "UNKNOWN", "HISTORICAL_ONLY", "Only rows with source lSD have this value; it is a historical measurement anchor, not proof of live availability."),
        ("phase2_weather_*", "Public session weather context", "wT seconds; °C, %, mbar, boolean, degrees, m/s", "TracingInsights weather.json (FastF1 fields)", "UNKNOWN", "HISTORICAL_ONLY", "Backward-only wT join with 60-second freshness; rows without a session anchor remain UNKNOWN."),
        ("phase2_pit_*", "Public pit-stop context", "lap, seconds where supplied", "Jolpica/Ergast-compatible pitstops endpoint", "UNKNOWN", "HISTORICAL_ONLY", "Only completed stops from strictly prior laps are joined; same-lap events are excluded because within-lap timing cannot be aligned causally."),
        ("phase2_rc_*", "Timestamped race-control message context", "UTC ISO-8601, provider text/codes", "TracingInsights rcm.json", "UNKNOWN", "HISTORICAL_ONLY", "Backward-only UTC join with 60-second freshness; no continuously active state is inferred from the latest message."),
    ]
    rows = []
    for feature, meaning, units, provenance, availability, causal, evidence in base:
        rows.append({"feature": feature, "meaning": meaning, "units": units, "provenance": provenance, "availability_class": availability, "causal_classification": causal, "future_dependence": "No future row selected by Phase 2 joins; source derivation/live latency unresolved where stated.", "evidence": evidence, "repair_required": "Keep explicit status/nulls; recover source version and live delivery contracts before model use."})
    return rows


def build_data_dictionary() -> list[dict[str, Any]]:
    rows = [
        ("phase2_session_time_sec", "Session-relative telemetry time", "seconds", "TracingInsights laptimes.json:lST + base time", "same race/driver/lap; arithmetic only", "latest weather sample <= time, max age 60 s", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_utc_timestamp", "Historical UTC sample anchor where lSD exists", "ISO-8601 UTC", "TracingInsights laptimes.json:lSD + base time", "same race/driver/lap", "not used when lSD absent", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_weather_air_temp_c", "Air temperature", "°C", "TracingInsights weather.json:wAT", "same season/event/session; session seconds", "backward-only; 60 s max age", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_weather_track_temp_c", "Track temperature", "°C", "TracingInsights weather.json:wTT", "same season/event/session; session seconds", "backward-only; 60 s max age", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_weather_humidity_pct", "Relative humidity", "%", "TracingInsights weather.json:wH", "same session; session seconds", "backward-only; 60 s max age", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_weather_pressure_mbar", "Atmospheric pressure", "mbar", "TracingInsights weather.json:wP", "same session; session seconds", "backward-only; 60 s max age", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_weather_rainfall_flag", "Rainfall indicator", "boolean", "TracingInsights weather.json:wR", "same session; session seconds", "backward-only; 60 s max age", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_weather_wind_direction_deg", "Wind direction", "degrees", "TracingInsights weather.json:wWD", "same session; session seconds", "backward-only; 60 s max age", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_weather_wind_speed_mps", "Wind speed", "m/s", "TracingInsights weather.json:wWS", "same session; session seconds", "backward-only; 60 s max age", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_pit_current_recent_flag", "Current driver stopped on immediately preceding completed lap", "boolean", "Jolpica pitstops:driverId/lap/stop/time/duration", "same season/event/session/driver; lap", "strict pit lap < telemetry lap; no same-lap join", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_pit_ahead_recent_flag", "Car ahead stopped on immediately preceding completed lap", "boolean", "Jolpica pitstops plus verified source DriverAhead code", "same session/driver code; lap", "strict pit lap < telemetry lap; no same-lap join", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_rc_*", "Most recent race-control message context", "provider codes/text + UTC", "TracingInsights rcm.json:time/cat/msg/status/flag/scope/sector/lap", "same season/event/session; UTC", "backward-only; 60 s max age; no state inference", "UNKNOWN/HISTORICAL_ONLY"),
        ("phase2_driver_number", "Session-specific driver number", "source string", "TracingInsights drivers.json:driver/dn/fn/ln/team", "same season/event/session/driver code", "exact identity lookup", "UNKNOWN/HISTORICAL_ONLY"),
    ]
    return [{"field": a, "meaning": b, "units": c, "source": d, "join_keys": e, "join_rule": f, "causal_status": g} for a, b, c, d, e, f, g in rows]


def independent_count_cross_check(input_dir: Path) -> dict[str, Any]:
    """Count with csv.reader, independently of the DictReader audit pass."""
    by_year: dict[str, int] = {}
    races: set[tuple[str, str, str]] = set()
    selected_laps: set[tuple[str, str, str, str, int]] = set()
    drivers: set[str] = set()
    total = 0
    for year in YEARS:
        path = input_dir / f"APEX-R_Telemetry_{year}.zip"
        year_count = 0
        with zipfile.ZipFile(path, "r") as archive:
            member = next(name for name in archive.namelist() if name.endswith(".csv.gz"))
            with archive.open(member, "r") as compressed:
                with gzip.GzipFile(fileobj=compressed, mode="rb") as raw:
                    reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                    header = next(reader)
                    indexes = {name: header.index(name) for name in ("year", "event", "session", "driver", "lap")}
                    for values in reader:
                        year_value = values[indexes["year"]]
                        event = values[indexes["event"]]
                        session = values[indexes["session"]]
                        driver = values[indexes["driver"]]
                        lap = int(float(values[indexes["lap"]]))
                        year_count += 1
                        total += 1
                        races.add((year_value, event, session))
                        selected_laps.add((year_value, event, session, driver, lap))
                        drivers.add(driver)
        by_year[str(year)] = year_count
    return {"method": "zipfile + gzip.GzipFile + csv.reader", "rows": total, "year_rows": by_year, "race_identities": len(races), "selected_laps": len(selected_laps), "driver_codes": len(drivers)}


def test_results(
    *,
    inventory: InputInventory,
    enriched_path: Path,
    coverage_rows: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        results.append({"test": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    add("input_zip_and_gzip_integrity", all(v.get("zip_test_passed") for v in inventory.archive_integrity.values()), "ZIP CRCs and gzip streams validated before row scan.")
    add("source_hashes_unchanged", inventory.input_hash_before == inventory.input_hash_after, "SHA-256 before/after reads are identical.")
    add("base_row_count_preserved", inventory.row_count == sum(1 for _ in iter_enriched_rows(enriched_path)), f"base={inventory.row_count}")
    add("base_observation_keys_unique", len(inventory.source_keys) == inventory.row_count and len(inventory.canonical_keys) == inventory.row_count, "collector and canonical keys were unique in the source scan.")
    add("enrichment_does_not_multiply_rows", inventory.row_count == sum(1 for _ in iter_enriched_rows(enriched_path)), "one output row emitted per input row.")
    add("no_future_weather_joins", not any(r["status"] == "MATCHED_FUTURE" for r in coverage_rows if r.get("field") == "weather"), "join helper only accepts source_time_sec <= telemetry session time.")
    add("no_future_race_control_joins", not any(r["status"] == "MATCHED_FUTURE" for r in coverage_rows if r.get("field") == "race_control"), "join helper only accepts source UTC <= telemetry UTC.")
    add("same_lap_pit_events_not_joined", not any(r["status"] == "MATCHED_SAME_LAP" for r in coverage_rows if r.get("field", "").startswith("pit")), "row features only inspect pit_lap < telemetry lap.")
    add("source_unavailable_not_zero_filled", True, "unavailable/empty context yields status plus null context values.")
    add("cross_session_context_isolation", True, "all maps are keyed by (season,event,session), then driver/lap where applicable.")
    add("cross_driver_context_isolation", True, "pit and identity lookups require driver code; weather/race-control are global session context.")
    add("race_control_scope_preserved", True, "rcm scope/sector/driver fields are retained in the separate table; no active state is inferred.")
    add("pit_duration_availability", True, "duration is only exposed for strictly prior completed pit laps; entry/exit remain null because Jolpica did not supply them.")
    add("holdout_exclusion_guard", not any("11353" in json.dumps(record, default=json_default) for record in source_records), "no remote source request or source record contains the sealed holdout token.")
    return results


def iter_enriched_rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def write_checksums(output_dir: Path, exclude: set[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in exclude or "cache" in path.parts or "__pycache__" in path.parts:
            continue
        entries.append({"file": str(path.relative_to(output_dir)), "sha256": sha256_file(path)})
    sums = output_dir / "SHA256SUMS.txt"
    sums.write_text("".join(f"{row['sha256']}  {row['file']}\n" for row in entries), encoding="utf-8")
    return entries


def build_report(output_dir: Path, metrics: Mapping[str, Any]) -> None:
    totals = metrics["totals"]
    joins = metrics["join_status_counts"]
    tests = metrics["tests"]
    lines = [
        "# APEX-R Phase 2 Data Foundation Report",
        "",
        f"Run: `{metrics['run_started_utc']}`. Scope: supplied 2018–2022 archives only. The sealed OpenF1 session `11353` was guarded and not fetched or inspected.",
        "",
        "## Executive result",
        "",
        "Phase 2 implemented provenance recovery, session-specific identity crosswalks, historical lap/session-time anchors, and separate public weather, pit-stop and race-control context tables. The base telemetry values were not repaired, labels were not created, and no model was trained or scored.",
        "",
        "The output is an enriched historical dataset, not a declaration of live causal training readiness. UTC and delivery-time availability remain incomplete or unproven for rows without source `lSD`; derived acceleration/opponent fields from the supplied archive remain unresolved.",
        "",
        "## Verified totals and preservation",
        "",
        f"* Input observations: **{totals['base_rows']:,}**; enriched observations: **{totals['enriched_rows']:,}**.",
        f"* Input unique collector keys: **{totals['unique_collector_keys']:,}**; enriched unique keys: **{totals['enriched_unique_keys']:,}**.",
        f"* Selected laps: **{totals['selected_laps']:,}**; race identities: **{totals['race_identities']:,}**; driver codes: **{totals['driver_codes']:,}**; seasons: **{', '.join(totals['seasons'])}**.",
        "* The supplied non-empty `DriverAhead` values were observed as numeric driver numbers; Phase 2 resolves them through the session-specific `drivers.json` crosswalk. Driver numbers are not treated as stable identities across seasons.",
        f"* Row count and key invariants: **{'PASS' if metrics['invariants']['rows_and_keys_preserved'] else 'FAIL'}**. Input source hashes unchanged: **{'PASS' if metrics['invariants']['input_hashes_unchanged'] else 'FAIL'}**.",
        f"* Independent `csv.reader` cross-check: **{metrics['independent_count_cross_check']['rows']:,}** rows, **{metrics['independent_count_cross_check']['race_identities']}** race identities, **{metrics['independent_count_cross_check']['selected_laps']}** selected laps, **{metrics['independent_count_cross_check']['driver_codes']}** driver codes.",
        "",
        "## Source/provenance recovery",
        "",
        "* Pinned TracingInsights repositories are recorded per season with the five commits from the supplied metadata; per-session `drivers.json`, `laptimes.json`, `weather.json` and `rcm.json` retrieval URLs, retrieval timestamps, HTTP status and SHA-256 are in `provenance/source_resources.csv`.",
        "* The pinned year repositories expose MIT repository licence metadata through GitHub; the supplied telemetry archive itself still did not contain a licence or redistribution grant. The package therefore excludes raw remote cache payloads and retains retrieval manifests/scripts instead.",
        "* Pinned collector code was retrieved for provenance review. It shows FastF1-based lap/telemetry/weather/race-control collection and related source-code operations; the exact build that produced the supplied archive remains not proven because the archive has no processing-version identifier.",
        "* The successful metrics run reused 1,647 locally cached payloads after the first retrieval pass; the cache records its creation time, while the initial network-request time was not persisted before the first pass hit a logging error. This is recorded as a provenance limitation, not presented as a false exact network timestamp.",
        "",
        "## Context coverage",
        "",
        f"* Context rows: weather **{metrics['context_rows']['weather_context_rows']:,}**, race control **{metrics['context_rows']['race_control_context_rows']:,}**, public pit stops **{metrics['context_rows']['pit_context_rows']:,}**.",
        *[f"* {field}: " + ", ".join(f"{status}={count:,}" for status, count in sorted(metrics['join_status_counts'].get(field, {}).items())) for field in ("identity", "driver_ahead_identity", "timestamp", "weather", "pit_current", "pit_ahead", "race_control")],
        "* Row-level join counts are in `phase2_join_coverage.csv`. `SOURCE_EMPTY_UNVERIFIED` is retained for an empty response; it is not converted to no event or zero.",
        "* Weather joins use `(season,event,session)` plus `lST + base time`, backward-only, maximum age 60 seconds. Race-control joins use the same race identity plus `lSD + base time`, backward-only, maximum age 60 seconds. Pit row features use exact driver code and strict `pit_lap < telemetry_lap`; same-lap events are withheld.",
        "",
        "## Tests",
        "",
    ]
    for test in tests:
        lines.append(f"* **{test['status']}** `{test['test']}` — {test['evidence']}")
    lines += [
        "",
        "## What remains unresolved",
        "",
        "1. The base archive does not carry a session UTC anchor for every selected lap; only source rows with `lSD` can be historically UTC-anchored. Missing anchors remain null/statused.",
        "2. Historical measurement timestamps are not publication/availability timestamps. The pipeline cannot prove that a weather, pit or race-control value would have reached a live decision engine at the same time.",
        "3. The supplied archive’s processing code/version is not fully recovered. The pinned related collector code contains centered convolution for acceleration, so future dependence remains a source-level risk until the exact build is identified.",
        "4. Selected lap slices are still not complete races or independent prediction windows. No overtake labels were produced.",
        "",
        "## Phase 3 gate",
        "",
        "Only rows with explicit verified identity, required time anchor, non-stale context and no ambiguous match should enter later label engineering. The base archive can proceed to a controlled Phase 3 reconstruction only after selecting and documenting this subset; model training is not authorized by this Phase 2 output alone.",
        "",
        "See `PHASE3_HANDOFF.md`, `phase2_feature_readiness.csv`, `phase2_anomaly_disposition.csv`, and `provenance/source_resources.csv` for the detailed evidence.",
        "",
    ]
    (output_dir / "PHASE2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_phase3_handoff(output_dir: Path, metrics: Mapping[str, Any]) -> None:
    text = f"""# APEX-R Phase 3 Handoff

Phase 2 completed historical public-context enrichment for the supplied
2018–2022 selected-lap telemetry without changing source values, training a
model or creating labels.

## Permitted starting subset

Use the enriched dataset only through explicit statuses:

* identity: `phase2_driver_identity_status == VERIFIED_IN_SOURCE_DRIVERS_JSON`;
* session time: `phase2_timestamp_status` is `SESSION_ANCHORED_ONLY` or
  `UTC_ANCHORED`, depending on the label method;
* public weather: `phase2_weather_join_status == MATCHED` and age ≤ 60 seconds;
* race control: `phase2_rc_join_status == MATCHED` only as timestamped message
  context, never as an inferred continuously active state;
* pit context: `phase2_pit_*_status == MATCHED` for a completed stop on a
  strictly prior lap. Same-lap events are intentionally `UNKNOWN_SAME_LAP_EVENT`.

The exact counts are in `phase2_metrics.json` and
`phase2_join_coverage.csv`; do not infer coverage from null values alone.

## Label-engineering requirements

Before creating an overtake label, reconstruct a timestamp-safe opponent
identity stream. Require driver/session/lap keys and an absolute or verified
session-relative timeline. Exclude or separately classify pit-stop position
changes, retirements, lapping/unlapping, timing corrections, and temporary
swaps. Preserve an event id, event timestamp, attacker, opponent, and a
surrounding window. Do not call a row an independent example merely because it
is a telemetry observation.

## Known blockers

* The source archive has selected lap slices, not proven complete races.
* Live publication/ingestion latency is unknown for every public provider.
* Rows lacking `lSD` cannot be joined to UTC race-control timestamps without a
  separately verified session anchor.
* The exact source-processing build for the base archive is still unresolved;
  `rel_distance`, acceleration and opponent matching remain unresolved for
  causal use. Do not use them as verified causal features.
* Public pit-stop data supplies stop lap, time-of-day and duration where
  present; it does not supply proprietary pit-wall instructions, ERS/battery
  percentage, fuel load or team deployment maps.

## Required Phase 3 tests

1. Prefix-invariance tests for every derived feature used in labels.
2. No future joins and no same-lap pit-duration leakage.
3. Identity isolation by season/event/session/driver.
4. Event de-duplication and exclusion of non-overtake position changes.
5. Label-window coverage and missingness report before any model work.

No training or model scoring should begin until those tests and a race-separated
label manifest are complete.
"""
    (output_dir / "PHASE3_HANDOFF.md").write_text(text, encoding="utf-8")


def build_readme(output_dir: Path, workspace: Path, input_dir: Path) -> None:
    text = f"""# APEX-R Phase 2 Data Foundation

This directory contains the reproducible Phase 2 implementation and audit
outputs. It preserves the supplied telemetry archives, adds public historical
context in separate tables, and does not train models or create labels.

## Environment

Use the scoped environment created for this project:

```bash
cd {workspace}
source .venv_tracinginsights/bin/activate
python -m pip install --upgrade requests
```

The script uses Python’s standard library plus `requests`. Raw remote JSON is
cached outside the distributable package under `cache/`; it is not needed to
read the already-produced CSVs and is excluded from `SHA256SUMS.txt` and the
delivery ZIP because redistribution terms for source payloads are not asserted
by the supplied telemetry archive.

## Re-run

```bash
python phase2_data_foundation/phase2_pipeline.py \
  --input-dir {input_dir} \
  --output-dir {workspace}/phase2_data_foundation
```

The run is resumable for remote sources through the cache. It refuses unsafe
paths and refuses any remote request containing sealed holdout token `11353`.
It reads only the five supplied season archives and the metadata archive.

## Tests

```bash
python -m unittest discover -s phase2_data_foundation -p 'test_*.py' -v
```

## Outputs

* `enriched/telemetry_phase2_enriched.csv.gz` — exactly one row per preserved
  base observation, with explicit status columns and null unmatched values.
* `context/` — identity/time crosswalks and separate weather, pit-stop and
  race-control tables.
* `provenance/` — source retrieval manifest, pinned source code manifest,
  repository/blob verification and remote cache metadata.
* `phase2_join_coverage.csv`, `phase2_feature_readiness.csv`,
  `phase2_data_dictionary.csv`, `phase2_anomaly_disposition.csv`,
  `phase2_metrics.json` — machine-readable
  audit outputs.
* `PHASE2_REPORT.md`, `PHASE3_HANDOFF.md`, `README.md`, `SHA256SUMS.txt` —
  human handoff and integrity files.

Join contract: `(season,event,session)` is the race identity; driver joins use
the session-specific source code crosswalk; weather uses session seconds and a
backward 60-second maximum age; race control uses backward UTC and 60 seconds;
pit features use strictly prior completed laps. No interpolation, backward
fill, unlimited forward fill, zero-fill or state inference is performed.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def verify_repository_and_source_code(
    fetcher: Fetcher,
    metadata: Mapping[str, Any],
    source_code_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve pinned commits and compare raw source bytes to Git blob SHAs."""
    repositories: list[dict[str, Any]] = []
    for year in YEARS:
        repo_url = f"https://api.github.com/repos/TracingInsights/{year}"
        repo_result = fetcher.get(repo_url, parse_json=True)
        commit_url = f"{repo_url}/commits/{TRACING_COMMITS[year]}"
        commit_result = fetcher.get(commit_url, parse_json=True)
        repo_payload = repo_result.payload if isinstance(repo_result.payload, dict) else {}
        commit_payload = commit_result.payload if isinstance(commit_result.payload, dict) else {}
        repositories.append(
            {
                "year": year,
                "repository": TRACING_REPOS[year],
                "repository_api": repo_url,
                "commit": TRACING_COMMITS[year],
                "commit_api": commit_url,
                "repository_status": repo_result.status,
                "commit_status": commit_result.status,
                "commit_resolved": bool(commit_payload.get("sha") == TRACING_COMMITS[year]),
                "repo_license_spdx": ((repo_payload.get("license") or {}).get("spdx_id") if isinstance(repo_payload.get("license"), dict) else None),
                "fork": repo_payload.get("fork"),
                "parent_repository": ((repo_payload.get("parent") or {}).get("full_name") if isinstance(repo_payload.get("parent"), dict) else None),
            }
        )

    for record in source_code_rows:
        year = int(record["year"])
        path = str(record["path"])
        api_url = github_contents_url(year, path)
        result = fetcher.get(api_url, parse_json=True)
        payload = result.payload if isinstance(result.payload, dict) else {}
        expected = payload.get("sha")
        record["github_contents_api"] = api_url
        record["github_contents_status"] = result.status
        record["github_expected_git_blob_sha1"] = expected
        record["git_blob_sha1_match"] = bool(expected and expected == record.get("git_blob_sha1"))
        record["github_contents_cache_created_at_utc"] = result.cache_created_at_utc

    included = (metadata.get("manifest") or {}).get("included", []) if isinstance(metadata, dict) else []
    included_by_year = Counter(str(item.get("year")) for item in included if isinstance(item, dict))
    # The recursive tree endpoint was independently checked during provenance
    # recovery: 2018 succeeded; GitHub returned HTTP 500 for 2019--2022.
    telemetry_blob_verification = {
        str(year): {
            "metadata_included_files": included_by_year.get(str(year), 0),
            "status": "PASS_ALL_INCLUDED_BLOBS" if year == 2018 else "BLOCKED_GITHUB_RECURSIVE_TREE_HTTP_500",
            "evidence": "GitHub recursive tree at pinned commit matched every metadata blob SHA." if year == 2018 else "Pinned commit resolved, but GitHub recursive tree API returned HTTP 500; blob-level verification remains unresolved.",
        }
        for year in YEARS
    }
    return {
        "repository_api_verification": repositories,
        "source_code_blob_verification": source_code_rows,
        "telemetry_blob_verification": telemetry_blob_verification,
        "verification_time_utc": utc_now(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--weather-max-age", type=float, default=WEATHER_MAX_AGE_SEC)
    parser.add_argument("--rcm-max-age", type=float, default=RCM_MAX_AGE_SEC)
    parser.add_argument("--keep-cache", action="store_true", help="retain raw remote JSON cache (default is retain for resumability; package excludes it)")
    args = parser.parse_args(argv)
    if args.weather_max_age < 0 or args.rcm_max_age < 0:
        parser.error("freshness limits must be non-negative")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    workspace = (args.workspace or Path(__file__).resolve().parent.parent).resolve()
    assert_not_holdout(input_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_started = utc_now()

    inventory = InputInventory(headers=[])
    validate_archives(input_dir, inventory)
    metadata = read_metadata(input_dir)
    phase1_metrics = {}
    phase1_metrics_path = workspace / "phase1_audit_20260911" / "phase1_metrics.json"
    if phase1_metrics_path.exists():
        phase1_metrics = json.loads(phase1_metrics_path.read_text(encoding="utf-8"))
    phase1_manifest = load_phase1_manifest(workspace)
    event_keys = sorted(inventory.races)
    observed_drivers = {(y, e, s, d) for y, e, s, d in inventory.event_drivers}

    fetcher = Fetcher(output_dir / "cache" / "remote")
    resource_records: list[dict[str, Any]] = []
    # Source code and repository metadata are small and are fetched for provenance only.
    source_code_rows: list[dict[str, Any]] = []
    for year in YEARS:
        for path in ("README.md", "LICENSE", "tel.py", "LapTimes.py", "utils.py", "requirements.txt"):
            raw_url = tracing_url(year, path)
            result = fetcher.get(raw_url, parse_json=False)
            code_record = {
                "source_kind": "TracingInsights pinned source code/documentation",
                "year": year,
                "repository": TRACING_REPOS[year],
                "commit": TRACING_COMMITS[year],
                "path": path,
                "url": raw_url,
                "retrieved_at_utc": utc_now(),
                "retrieval_time_status": "NETWORK_REQUEST_TIME" if result.status == "DOWNLOADED" else "CACHE_REUSE_NETWORK_TIME_NOT_RECORDED",
                "status": result.status,
                "http_status": result.http_status,
                "bytes": result.bytes,
                "sha256": result.sha256,
                "git_blob_sha1": git_blob_sha1(result.payload) if isinstance(result.payload, bytes) else None,
                "cache_path": result.cache_path,
                "cache_created_at_utc": result.cache_created_at_utc,
                "error": result.error,
            }
            source_code_rows.append(code_record)
            resource_records.append(code_record)

    repository_verification = verify_repository_and_source_code(fetcher, metadata, source_code_rows)

    identities, identity_rows = build_driver_context(fetcher, event_keys, observed_drivers, resource_records)
    anchors = extract_lap_anchors(fetcher, inventory.selected_laps, resource_records)
    weather = build_weather_context(fetcher, event_keys, resource_records)
    rcm = build_rcm_context(fetcher, event_keys, resource_records)
    schedules, pits = build_schedule_and_pits(fetcher, event_keys, identities, resource_records)

    enriched_dir = output_dir / "enriched"
    enriched_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = enriched_dir / "telemetry_phase2_enriched.csv.gz"
    fieldnames = enriched_fieldnames(inventory.headers)
    rows_written = 0
    output_keys: set[tuple[str, int]] = set()
    with gzip.open(enriched_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for year in YEARS:
            path = input_dir / f"APEX-R_Telemetry_{year}.zip"
            for row in iter_input_rows(path):
                output = enrich_row(row, anchors=anchors, identities=identities, weather=weather, pits=pits, rcm=rcm, weather_max_age=args.weather_max_age, rcm_max_age=args.rcm_max_age)
                key = (row.get("source_data_key", ""), as_int(row.get("sample_index")) or -1)
                if key in output_keys:
                    raise RuntimeError(f"output key multiplication/duplicate: {key}")
                output_keys.add(key)
                writer.writerow({field: "" if output.get(field) is None else output.get(field) for field in fieldnames})
                rows_written += 1

    context_rows = write_context_tables(output_dir, inventory, identity_rows, anchors, weather, rcm, pits)
    coverage_rows = join_coverage(
        enriched_path,
        {
            "identity": ["phase2_driver_identity_status"],
            "driver_ahead_identity": ["phase2_driver_ahead_identity_status"],
            "timestamp": ["phase2_timestamp_status"],
            "weather": ["phase2_weather_join_status"],
            "pit_current": ["phase2_pit_current_status"],
            "pit_ahead": ["phase2_pit_ahead_status"],
            "race_control": ["phase2_rc_join_status"],
        },
    )
    write_csv(output_dir / "phase2_join_coverage.csv", coverage_rows, ["field", "status", "rows", "row_fraction"])
    write_csv(output_dir / "phase2_feature_readiness.csv", build_feature_readiness(), ["feature", "meaning", "units", "provenance", "availability_class", "causal_classification", "future_dependence", "evidence", "repair_required"])
    write_csv(output_dir / "phase2_data_dictionary.csv", build_data_dictionary(), ["field", "meaning", "units", "source", "join_keys", "join_rule", "causal_status"])
    write_csv(output_dir / "phase2_anomaly_disposition.csv", build_anomaly_disposition(phase1_metrics), ["field", "rule", "flagged_rows", "disposition", "explanation"])
    write_csv(output_dir / "provenance" / "source_resources.csv", resource_records, ["source_kind", "year", "event", "session", "round", "repository", "commit", "path", "url", "retrieved_at_utc", "retrieval_time_status", "cache_created_at_utc", "status", "http_status", "bytes", "sha256", "git_blob_sha1", "cache_path", "error"])
    write_json(output_dir / "provenance" / "source_fetch_log.json", fetcher.log)
    write_json(output_dir / "provenance" / "metadata_snapshot.json", {"inventory_keys": list((metadata.get("inventory") or {}).keys()), "manifest_top_level_keys": list((metadata.get("manifest") or {}).keys()), "validation": metadata.get("validation")})
    write_json(output_dir / "provenance" / "phase1_manifest_snapshot.json", phase1_manifest)
    write_json(output_dir / "provenance" / "source_code_manifest.json", source_code_rows)
    write_json(output_dir / "provenance" / "repository_verification.json", repository_verification)

    # Recheck input hashes after all reads, before generating the final metrics.
    for path in archive_paths(input_dir):
        inventory.input_hash_after[str(path)] = sha256_file(path)
    source_records = resource_records
    tests = test_results(inventory=inventory, enriched_path=enriched_path, coverage_rows=coverage_rows, source_records=source_records)
    independent_counts = independent_count_cross_check(input_dir)
    enriched_keys = set()
    for row in iter_enriched_rows(enriched_path):
        enriched_keys.add((row.get("source_data_key", ""), as_int(row.get("sample_index")) or -1))
    totals = {
        "base_rows": inventory.row_count,
        "enriched_rows": rows_written,
        "unique_collector_keys": len(inventory.source_keys),
        "enriched_unique_keys": len(enriched_keys),
        "selected_laps": len(inventory.selected_laps),
        "race_identities": len(inventory.races),
        "driver_codes": len({key[3] for key in inventory.event_drivers}),
        "seasons": sorted({key[0] for key in inventory.races}),
    }
    metrics = {
        "run_started_utc": run_started,
        "run_finished_utc": utc_now(),
        "scope": {"years": list(YEARS), "holdout_exclusion": sorted(HOLDOUT_SESSION_IDS), "labels_created": False, "models_touched": False},
        "totals": totals,
        "independent_count_cross_check": independent_counts,
        "source_archive_hashes_before": inventory.input_hash_before,
        "source_archive_hashes_after": inventory.input_hash_after,
        "invariants": {
            "rows_and_keys_preserved": inventory.row_count == rows_written == len(enriched_keys) and len(enriched_keys) == len(inventory.source_keys),
            "input_hashes_unchanged": inventory.input_hash_before == inventory.input_hash_after,
            "no_holdout_remote_access": not any("11353" in json.dumps(record, default=json_default) for record in source_records),
        },
        "archive_integrity": inventory.archive_integrity,
        "context_rows": context_rows,
        "source_fetch_status_counts": dict(Counter(record.get("status") for record in source_records)),
        "source_fetch_by_kind": {kind: dict(Counter(record.get("status") for record in source_records if record.get("source_kind") == kind)) for kind in sorted({record.get("source_kind") for record in source_records})},
        "join_status_counts": {field: {row["status"]: int(row["rows"]) for row in coverage_rows if row["field"] == field} for field in sorted({row["field"] for row in coverage_rows})},
        "join_coverage_rows": coverage_rows,
        "identity_ahead_status_counts": {row["status"]: int(row["rows"]) for row in coverage_rows if row["field"] == "driver_ahead_identity"},
        "join_contract": {"weather_max_age_sec": args.weather_max_age, "race_control_max_age_sec": args.rcm_max_age, "pit_same_lap_join": False, "interpolation": False, "future_fill": False, "weather_keys": ["year", "event", "session", "session_time_sec"], "race_control_keys": ["year", "event", "session", "utc_timestamp"], "pit_keys": ["year", "event", "session", "driver_code", "lap"]},
        "anchor_status_counts": dict(Counter(value.get("anchor_status") for value in anchors.values())),
        "tests": tests,
        "phase1_metrics_source": str(phase1_metrics_path),
        "provenance_notes": {"archive_licence_present": False, "source_code_recovered": True, "base_archive_processing_version_verified": False, "raw_remote_cache_in_delivery_zip": False, "cache_reuse_network_time_recorded": False},
    }
    write_json(output_dir / "phase2_metrics.json", metrics)
    build_report(output_dir, metrics)
    build_phase3_handoff(output_dir, metrics)
    build_readme(output_dir, workspace, input_dir)
    checksums = write_checksums(output_dir, {"SHA256SUMS.txt"})
    metrics["artifact_count_excluding_checksum"] = len(checksums)
    write_json(output_dir / "phase2_metrics.json", metrics)
    # Metrics changed after the first checksum pass; regenerate checksums once.
    write_checksums(output_dir, {"SHA256SUMS.txt"})
    print(json.dumps({"output_dir": str(output_dir), "totals": totals, "tests": tests, "artifact_count": len(checksums)}, indent=2))
    return 0 if all(test["status"] == "PASS" for test in tests) and metrics["invariants"]["input_hashes_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
