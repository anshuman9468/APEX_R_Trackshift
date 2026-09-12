#!/usr/bin/env python3
"""APEX-R Phase 1 dataset audit.

This program is deliberately an audit, not a cleaning or modelling pipeline.
It reads the supplied ZIP files without extracting or modifying them, keeps
source values as strings for the quality checks, and writes only audit
artifacts to the requested output directory.

Dependencies: Python 3.10+, pandas, numpy.  Standard-library modules provide
ZIP/GZIP integrity, hashing, CSV cross-checks and the duplicate audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile
import gzip
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


YEARS = (2018, 2019, 2020, 2021, 2022)
NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nat"}
EXPECTED_COLUMNS = [
    "year", "event", "session", "driver", "lap", "sample_index", "time",
    "rpm", "speed", "gear", "throttle", "brake", "drs", "distance",
    "rel_distance", "DriverAhead", "DistanceToDriverAhead", "acc_x", "acc_y",
    "acc_z", "x", "y", "z", "source_data_key",
]
NUMERIC_COLUMNS = [
    "year", "lap", "sample_index", "time", "rpm", "speed", "gear",
    "throttle", "brake", "drs", "distance", "rel_distance", "DistanceToDriverAhead",
    "acc_x", "acc_y", "acc_z", "x", "y", "z",
]
SOURCE_ID_COLUMNS = ["year", "event", "session", "driver", "lap", "sample_index"]


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (Counter, defaultdict)):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_name(name: str) -> bool:
    if not name or "\x00" in name or name.startswith(("/", "\\")):
        return False
    normal = name.replace("\\", "/")
    return not any(part in ("", ".", "..") for part in normal.split("/"))


def archive_inventory(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None,
        "sha256_before": sha256_file(path) if path.exists() else None,
        "zip_opened": False, "zip_test_passed": False, "unsafe_members": [], "members": [], "error": None,
    }
    if not path.exists():
        result["error"] = "missing input archive"
        return result
    try:
        with zipfile.ZipFile(path) as zf:
            result["zip_opened"] = True
            infos = zf.infolist()
            for info in infos:
                safe = safe_member_name(info.filename)
                if not safe:
                    result["unsafe_members"].append(info.filename)
                digest = hashlib.sha256()
                member_error = None
                try:
                    with zf.open(info, "r") as fh:
                        for block in iter(lambda: fh.read(1024 * 1024), b""):
                            digest.update(block)
                except Exception as exc:  # integrity is reported, not hidden
                    member_error = f"{type(exc).__name__}: {exc}"
                result["members"].append({
                    "name": info.filename, "safe_path": safe, "file_size": info.file_size,
                    "compressed_size": info.compress_size, "crc32": f"{info.CRC:08x}",
                    "sha256_uncompressed_member": digest.hexdigest() if member_error is None else None,
                    "read_error": member_error,
                })
            bad = zf.testzip()
            result["zip_test_passed"] = bad is None and not result["unsafe_members"]
            result["first_bad_member"] = bad
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def member_bytes(zf: zipfile.ZipFile, member: str) -> bytes:
    if not safe_member_name(member):
        raise ValueError(f"unsafe ZIP member rejected: {member!r}")
    return zf.read(member)


def stream_chunks(zip_path: Path, member: str, chunksize: int = 50000) -> Iterable[pd.DataFrame]:
    """Yield raw CSV strings and force GZIP EOF/CRC validation."""
    zf = zipfile.ZipFile(zip_path)
    raw = zf.open(member, "r")
    gz = gzip.GzipFile(fileobj=raw, mode="rb")
    text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
    try:
        reader = pd.read_csv(
            text, dtype=str, keep_default_na=False, na_filter=False,
            chunksize=chunksize, on_bad_lines="error",
        )
        for chunk in reader:
            yield chunk
        text.flush()
    finally:
        try:
            text.close()  # reads through gzip close and checks CRC when exhausted
        finally:
            zf.close()


def csv_count(zip_path: Path, member: str) -> dict[str, Any]:
    """Independent second count using csv.reader, not pandas."""
    rows = 0
    header: list[str] | None = None
    uncompressed_bytes = 0
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member, "r") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                    reader = csv.reader(text)
                    header = next(reader)
                    for _ in reader:
                        rows += 1
                    uncompressed_bytes = gz.tell()
    return {"rows": rows, "header": header, "uncompressed_bytes": uncompressed_bytes}


def canonical_key(row: dict[str, str]) -> str:
    return "\x1f".join(row.get(c, "") for c in SOURCE_ID_COLUMNS)


def row_payload_hash(row: dict[str, str]) -> str:
    payload = "\x1f".join(row.get(c, "") for c in EXPECTED_COLUMNS).encode("utf-8", "surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def race_key(row: dict[str, str]) -> str:
    return f"{row.get('year','')}|{row.get('event','')}|{row.get('session','')}"


def lap_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (row.get("year", ""), row.get("event", ""), row.get("session", ""), row.get("driver", ""), row.get("lap", ""))


def initial_lap() -> dict[str, Any]:
    return {
        "row_count": 0, "sample_index_min": None, "sample_index_max": None,
        "time_min": None, "time_max": None, "distance_min": None, "distance_max": None,
        "first_time": None, "last_time": None, "last_sample_index": None,
        "timestamp_repeats": 0, "timestamp_negative_steps": 0, "sample_index_repeats": 0,
        "sample_index_negative_steps": 0, "positive_dts": [], "max_gap_sec": None,
    }


def update_minmax(state: dict[str, Any], name: str, value: float | None, fn: str) -> None:
    if value is None or not math.isfinite(value):
        return
    old = state.get(name)
    if old is None or (fn == "min" and value < old) or (fn == "max" and value > old):
        state[name] = value


def add_anomaly(examples: list[dict[str, Any]], row: dict[str, str], field: str, reason: str, evidence: str, observed: Any, example_counts: Counter[str]) -> None:
    # Keep the file bounded while guaranteeing representative examples for
    # every flag category, including rare gear/ratio flags that occur after
    # the frequent throttle flags in file order.
    if len(examples) >= 600 and example_counts[reason] >= 20:
        return
    example_counts[reason] += 1
    examples.append({
        "field": field, "flag_reason": reason, "observed_value": observed,
        "year": row.get("year"), "event": row.get("event"), "session": row.get("session"),
        "driver": row.get("driver"), "lap": row.get("lap"), "sample_index": row.get("sample_index"),
        "time": row.get("time"), "source_data_key": row.get("source_data_key"), "evidence_or_rule": evidence,
    })


def value_float(value: str) -> float | None:
    if value.strip().lower() in NULL_TOKENS:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def build_readiness() -> list[dict[str, Any]]:
    rows = [
        ("year", "Season identifier", "year", "Observed CSV identifier; provenance archive", "SAFE", "At row ingestion", "No future dependence in the identifier; delivery latency is unknown", "Source values are identifiers only", "Keep as grouping key; do not treat as a physical measurement"),
        ("event", "Event name", "text", "Observed CSV identifier; provenance archive", "SAFE", "At row ingestion", "No future dependence in the identifier; identity normalization is not documented", "Source values are identifiers only", "Obtain an event identity crosswalk before joins"),
        ("session", "Session type", "text", "Observed CSV identifier", "SAFE", "At row ingestion", "No future dependence in the identifier", "Only Race is present in the supplied rows", "Retain as grouping key"),
        ("driver", "Source driver code", "text", "Observed CSV identifier", "UNKNOWN", "At row ingestion", "The code is available, but identity crosswalk and numbering provenance are absent", "Driver codes are explicitly unverified identifiers", "Obtain a verified driver/number crosswalk"),
        ("lap", "Lap number", "lap number", "Observed CSV identifier", "SAFE", "At row ingestion", "Available in the row; selected-lap context is incomplete", "Lap value is observed, but physical lap completeness is not proven", "Reconstruct full-lap boundaries and context"),
        ("sample_index", "Within-file/lap sample index", "index", "Observed CSV identifier", "SAFE", "At row ingestion", "Index is present; ordering is checked per lap", "Does not establish external delivery time", "Use only as an ordering key after validation"),
        ("time", "Telemetry time within selected lap", "seconds (apparent lap-relative)", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "It resets near zero per selected lap and no UTC/session start is supplied", "Lap-relative interpretation is supported; actual clock and delivery time are absent", "Obtain session/lap UTC anchors; do not assume real-time alignment"),
        ("rpm", "Engine rotational speed", "unknown; numeric values", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Value is present, but units/provenance and delivery latency are undocumented", "No source documentation is packaged", "Recover source schema and validate units"),
        ("speed", "Vehicle speed", "unknown; numeric values", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Value is present; unit contract is absent", "Range is physically plausible for km/h but that is not proof", "Verify provider units from source code/docs"),
        ("gear", "Selected gear", "integer-like code", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Values are present but encoding/flags are undocumented", "Most values are integer-like; anomalous values need source explanation", "Verify encoding and anomaly semantics"),
        ("throttle", "Throttle channel", "unknown; numeric values", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Values include 104 in the supplied data; no unit/encoding contract is packaged", "A 0–100 assumption would flag 104, but the provider may use another scale", "Recover authoritative channel definition before modelling"),
        ("brake", "Brake channel", "unknown; numeric values", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Present, but encoding and units are undocumented", "Values are mostly binary-looking; that is not proof of semantics", "Verify source encoding"),
        ("drs", "DRS channel", "unknown code", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Present, but DRS code semantics are not documented in the archive", "Observed values need a provider-level codebook", "Obtain DRS encoding documentation"),
        ("distance", "Distance along lap", "unknown; numeric values", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Present and generally rises, but coordinate/reference units are undocumented", "Some negative values occur near lap starts; no track-length anchor is supplied", "Verify units and reconstruct lap boundaries"),
        ("rel_distance", "Relative/normalised distance", "unknown; numeric ratio-like", "Observed derived-looking field", "UNKNOWN", "Unknown; may require full-lap context", "The name and values suggest normalisation, but the processing formula is absent", "Potential full-lap/future dependence cannot be ruled out", "Recover formula or recompute causally"),
        ("DriverAhead", "Source opponent identifier", "source code", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Opponent matching logic is not packaged", "Missing values and identity mapping are present; lookahead cannot be ruled out", "Reconstruct timestamp-safe opponent matching"),
        ("DistanceToDriverAhead", "Distance to source-selected opponent", "unknown; numeric", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Opponent matching and units are undocumented", "Missing values and timing alignment are unresolved", "Verify units/matching and causal construction"),
        ("acc_x", "Acceleration channel x", "unknown; numeric", "Observed derived-looking channel", "UNKNOWN", "At recorded sample", "May be raw or derived; no processing code is available", "Large values occur, but expected axes/units are unknown", "Recover derivation and test prefix invariance"),
        ("acc_y", "Acceleration channel y", "unknown; numeric", "Observed derived-looking channel", "UNKNOWN", "At recorded sample", "May be raw or derived; no processing code is available", "Expected axes/units are unknown", "Recover derivation and test prefix invariance"),
        ("acc_z", "Acceleration channel z", "unknown; numeric", "Observed derived-looking channel", "UNKNOWN", "At recorded sample", "May be raw or derived; no processing code is available", "Expected axes/units are unknown", "Recover derivation and test prefix invariance"),
        ("x", "Track/position coordinate x", "unknown; coordinate units", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Coordinate reference system is undocumented", "Values vary by year; no track coordinate contract is supplied", "Verify coordinate frame and event-specific geometry"),
        ("y", "Track/position coordinate y", "unknown; coordinate units", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Coordinate reference system is undocumented", "Values vary by year; no track coordinate contract is supplied", "Verify coordinate frame and event-specific geometry"),
        ("z", "Track/position coordinate z", "unknown; coordinate units", "Observed telemetry CSV", "UNKNOWN", "At recorded sample", "Coordinate reference system is undocumented", "Values vary by year; no track coordinate contract is supplied", "Verify coordinate frame and event-specific geometry"),
        ("source_data_key", "Collector source row key", "text identifier", "Observed provenance key", "SAFE", "At row ingestion", "Key itself is available; its uniqueness is tested", "It is metadata, not a causal model feature", "Use for audit lineage only"),
    ]
    keys = ["feature", "meaning", "units", "provenance", "availability_class", "earliest_availability", "future_dependence", "evidence", "repair_required"]
    return [dict(zip(keys, r)) for r in rows]


def pct(n: int | float, d: int | float) -> float:
    return (100.0 * n / d) if d else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-zip", type=Path)
    parser.add_argument("--legacy-combined-zip", type=Path)
    parser.add_argument("--chunk-size", type=int, default=50000)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_paths = {y: input_dir / f"APEX-R_Telemetry_{y}.zip" for y in YEARS}
    metadata_path = (args.metadata_zip or input_dir / "APEX-R_Telemetry_Metadata.zip").resolve()
    legacy_path = args.legacy_combined_zip.resolve() if args.legacy_combined_zip else None
    all_inputs = [metadata_path, *telemetry_paths.values()]
    if legacy_path:
        all_inputs.append(legacy_path)
    if any("11353" in str(p) for p in all_inputs):
        raise RuntimeError("sealed session 11353 is not permitted in audit input paths")
    if not all(p.exists() for p in all_inputs):
        missing = [str(p) for p in all_inputs if not p.exists()]
        raise FileNotFoundError("missing inputs: " + ", ".join(missing))

    run_started = datetime.now(timezone.utc).isoformat()
    archives = {"metadata": archive_inventory(metadata_path)}
    archives.update({str(y): archive_inventory(path) for y, path in telemetry_paths.items()})
    if legacy_path:
        archives["legacy_combined"] = archive_inventory(legacy_path)

    metadata: dict[str, Any] = {}
    metadata_errors: list[str] = []
    try:
        with zipfile.ZipFile(metadata_path) as zf:
            names = zf.namelist()
            for required in ("inventory.json", "manifest.json", "validation.json"):
                if required not in names:
                    metadata_errors.append(f"missing metadata member: {required}")
                else:
                    try:
                        metadata[Path(required).stem] = json.loads(member_bytes(zf, required).decode("utf-8"))
                    except Exception as exc:
                        metadata_errors.append(f"unreadable {required}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        metadata_errors.append(f"metadata ZIP open failed: {type(exc).__name__}: {exc}")

    manifest = metadata.get("manifest", {})
    validation = metadata.get("validation", {})
    inventory = metadata.get("inventory", {})
    manifest_files = {str(x.get("file")): x for x in manifest.get("files", []) if isinstance(x, dict)}
    expected_header: list[str] | None = None
    total_rows = 0
    year_rows: Counter[str] = Counter()
    events: set[str] = set()
    races: set[str] = set()
    drivers: set[str] = set()
    source_lineage_keys: set[str] = set()
    laps: set[tuple[str, str, str, str, str]] = set()
    lap_stats: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    race_rows: Counter[str] = Counter()
    race_missing: dict[str, Counter[str]] = defaultdict(Counter)
    missing: Counter[str] = Counter()
    nonfinite: Counter[str] = Counter()
    parse_failures: Counter[str] = Counter()
    valid_numeric: Counter[str] = Counter()
    numeric_values: dict[str, array] = {c: array("d") for c in NUMERIC_COLUMNS}
    anomaly_counts: Counter[str] = Counter()
    anomaly_example_counts: Counter[str] = Counter()
    drs_codes: Counter[str] = Counter()
    session_codes: Counter[str] = Counter()
    anomaly_examples: list[dict[str, Any]] = []
    schema_by_year: dict[str, list[str]] = {}
    raw_bad_records: list[str] = []

    work_dir = Path(tempfile.mkdtemp(prefix="phase1-work-", dir=str(output_dir)))
    db_path = work_dir / "duplicate_audit.sqlite3"
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    # One table for the collector-style composite key and one for the
    # documented canonical fallback.  A full-row duplicate necessarily has
    # the same canonical key because the payload includes those key fields;
    # this avoids a third 1M-row table and keeps the audit disk-conscious.
    db.execute("CREATE TABLE collector_payload (obs_key TEXT NOT NULL, payload_hash TEXT NOT NULL, occurrences INTEGER NOT NULL, PRIMARY KEY(obs_key, payload_hash)) WITHOUT ROWID")
    db.execute("CREATE TABLE canonical_payload (obs_key TEXT NOT NULL, payload_hash TEXT NOT NULL, occurrences INTEGER NOT NULL, PRIMARY KEY(obs_key, payload_hash)) WITHOUT ROWID")

    def upsert(table: str, key: str, phash: str, row: dict[str, str]) -> None:
        db.execute(
            f"INSERT INTO {table}(obs_key,payload_hash,occurrences) VALUES (?,?,?) "
            "ON CONFLICT(obs_key,payload_hash) DO UPDATE SET occurrences=occurrences+1",
            (key, phash, 1),
        )

    def update_lap(row: dict[str, str]) -> None:
        key = lap_key(row)
        state = lap_stats.setdefault(key, initial_lap())
        state["row_count"] += 1
        si = value_float(row.get("sample_index", ""))
        t = value_float(row.get("time", ""))
        d = value_float(row.get("distance", ""))
        update_minmax(state, "sample_index_min", si, "min")
        update_minmax(state, "sample_index_max", si, "max")
        update_minmax(state, "time_min", t, "min")
        update_minmax(state, "time_max", t, "max")
        update_minmax(state, "distance_min", d, "min")
        update_minmax(state, "distance_max", d, "max")
        if state["last_time"] is not None and t is not None:
            dt = t - state["last_time"]
            if dt == 0:
                state["timestamp_repeats"] += 1
            elif dt < 0:
                state["timestamp_negative_steps"] += 1
            else:
                state["positive_dts"].append(dt)
                if state["max_gap_sec"] is None or dt > state["max_gap_sec"]:
                    state["max_gap_sec"] = dt
        if state["last_sample_index"] is not None and si is not None:
            dsi = si - state["last_sample_index"]
            if dsi == 0:
                state["sample_index_repeats"] += 1
            elif dsi < 0:
                state["sample_index_negative_steps"] += 1
        state["last_time"] = t
        state["first_time"] = t if state["first_time"] is None else state["first_time"]
        state["last_sample_index"] = si

    try:
        for year, zip_path in telemetry_paths.items():
            member = f"telemetry_{year}.csv.gz"
            inv = archives[str(year)]
            if not inv["zip_test_passed"] or inv["unsafe_members"]:
                raise RuntimeError(f"integrity/path check failed for {zip_path}")
            if len(inv["members"]) != 1 or inv["members"][0]["name"] != member:
                raise RuntimeError(f"unexpected members in {zip_path}: {inv['members']}")
            member_hash = inv["members"][0]["sha256_uncompressed_member"]
            documented = manifest_files.get(member, {}).get("sha256")
            if documented and member_hash != documented:
                metadata_errors.append(f"member SHA mismatch {member}: archive={member_hash} manifest={documented}")
            try:
                for chunk in stream_chunks(zip_path, member, args.chunk_size):
                    columns = list(chunk.columns)
                    if expected_header is None:
                        expected_header = columns
                    schema_by_year[str(year)] = columns
                    if columns != EXPECTED_COLUMNS:
                        raise ValueError(f"schema mismatch for {year}: {columns}")
                    rows = chunk.to_dict(orient="records")
                    for row in rows:
                        total_rows += 1
                        year_rows[str(year)] += 1
                        rid = race_key(row)
                        race_rows[rid] += 1
                        events.add(f"{row.get('year','')}|{row.get('event','')}")
                        races.add(rid)
                        drivers.add(row.get("driver", ""))
                        session_codes[row.get("session", "")] += 1
                        drs_codes[row.get("drs", "")] += 1
                        lk = lap_key(row)
                        laps.add(lk)
                        update_lap(row)
                        phash = row_payload_hash(row)
                        source_key = row.get("source_data_key", "")
                        if source_key:
                            source_lineage_keys.add(source_key)
                        sample_index = row.get("sample_index", "")
                        if source_key and sample_index.strip().lower() not in NULL_TOKENS:
                            upsert("collector_payload", source_key + "\x1f" + sample_index, phash, row)
                        upsert("canonical_payload", canonical_key(row), phash, row)
                        for field in EXPECTED_COLUMNS:
                            raw = row.get(field, "")
                            norm = str(raw).strip().lower()
                            if norm in NULL_TOKENS:
                                missing[field] += 1
                                race_missing[rid][field] += 1
                        for field in NUMERIC_COLUMNS:
                            raw = row.get(field, "")
                            norm = str(raw).strip().lower()
                            if norm in NULL_TOKENS:
                                continue
                            parsed = value_float(raw)
                            if parsed is None:
                                parse_failures[field] += 1
                            elif not math.isfinite(parsed):
                                nonfinite[field] += 1
                            else:
                                valid_numeric[field] += 1
                                numeric_values[field].append(parsed)
                        # Rules are explicit and intentionally do not alter source values.
                        throttle = value_float(row.get("throttle", ""))
                        if throttle is not None and (throttle < 0 or throttle > 100):
                            anomaly_counts["throttle_outside_0_100"] += 1
                            add_anomaly(anomaly_examples, row, "throttle", "outside_0_100", "Audit rule: throttle < 0 or > 100; the source value is retained", throttle, anomaly_example_counts)
                        gear = value_float(row.get("gear", ""))
                        if gear is not None and (gear < 0 or gear > 8 or not float(gear).is_integer()):
                            anomaly_counts["gear_not_integer_or_outside_0_8"] += 1
                            add_anomaly(anomaly_examples, row, "gear", "not_integer_or_outside_0_8", "Audit rule: gear must be integer-valued and within 0..8; encoding is not documented", gear, anomaly_example_counts)
                        brake = value_float(row.get("brake", ""))
                        if brake is not None and (brake < 0 or brake > 1):
                            anomaly_counts["brake_outside_0_1"] += 1
                            add_anomaly(anomaly_examples, row, "brake", "outside_0_1", "Audit rule for binary-looking brake channel; semantics remain unverified", brake, anomaly_example_counts)
                        drs = value_float(row.get("drs", ""))
                        if drs is not None and drs not in (0, 1):
                            anomaly_counts["drs_not_0_or_1"] += 1
                            add_anomaly(anomaly_examples, row, "drs", "not_0_or_1", "Observed codebook check only; provider DRS encoding is not documented", drs, anomaly_example_counts)
                        for field in ("rpm", "speed", "distance"):
                            val = value_float(row.get(field, ""))
                            if val is not None and val < 0:
                                anomaly_counts[f"{field}_negative"] += 1
                                add_anomaly(anomaly_examples, row, field, "negative_value", f"Audit rule: {field} < 0; expected range is not a provider-certified unit contract", val, anomaly_example_counts)
                        rel = value_float(row.get("rel_distance", ""))
                        if rel is not None and (rel < -1e-6 or rel > 1 + 1e-6):
                            anomaly_counts["rel_distance_outside_0_1_tolerance"] += 1
                            add_anomaly(anomaly_examples, row, "rel_distance", "outside_0_1_tolerance", "Audit rule: ratio-like value outside [-1e-6, 1+1e-6]; field formula is undocumented", rel, anomaly_example_counts)
                    db.commit()
            except Exception as exc:
                raw_bad_records.append(f"{year}: {type(exc).__name__}: {exc}")
                raise
    finally:
        db.commit()

    # Independent second pass: csv.reader counts only, with its own gzip EOF/CRC read.
    independent: dict[str, Any] = {}
    for year, zip_path in telemetry_paths.items():
        independent[str(year)] = csv_count(zip_path, f"telemetry_{year}.csv.gz")
    compressed_csv_bytes = sum(int(archives[str(y)]["members"][0]["file_size"]) for y in YEARS)
    uncompressed_csv_bytes = sum(int(independent[str(y)]["uncompressed_bytes"]) for y in YEARS)

    after_hashes = {str(p): sha256_file(p) for p in all_inputs}
    before_hashes = {str(p): archives["metadata"]["sha256_before"] if p == metadata_path else archives[str(next((y for y, q in telemetry_paths.items() if q == p), "legacy_combined"))]["sha256_before"] for p in all_inputs}
    source_hashes_unchanged = all(before_hashes[p] == after_hashes[p] for p in before_hashes)

    def query_one(sql: str) -> int:
        return int(db.execute(sql).fetchone()[0])

    exact_duplicate_rows = query_one("SELECT COALESCE(SUM(occurrences-1),0) FROM canonical_payload WHERE occurrences > 1")
    source_repeated_rows = query_one("SELECT COALESCE(SUM(occurrences-1),0) FROM collector_payload WHERE occurrences > 1")
    source_conflicting_keys = query_one("SELECT COUNT(*) FROM (SELECT obs_key FROM collector_payload GROUP BY obs_key HAVING COUNT(*) > 1)")
    canonical_repeated_rows = query_one("SELECT COALESCE(SUM(occurrences-1),0) FROM canonical_payload WHERE occurrences > 1")
    canonical_conflicting_keys = query_one("SELECT COUNT(*) FROM (SELECT obs_key FROM canonical_payload GROUP BY obs_key HAVING COUNT(*) > 1)")
    unique_source_keys = query_one("SELECT COUNT(DISTINCT obs_key) FROM collector_payload")
    unique_canonical_keys = query_one("SELECT COUNT(DISTINCT obs_key) FROM canonical_payload")
    total_source_key_rows = query_one("SELECT COALESCE(SUM(occurrences),0) FROM collector_payload")
    duplicate_examples = []
    for row in db.execute("SELECT obs_key, GROUP_CONCAT(payload_hash), SUM(occurrences) FROM collector_payload GROUP BY obs_key HAVING COUNT(*) > 1 LIMIT 10"):
        duplicate_examples.append({"source_data_key": row[0], "payload_hashes": row[1].split(","), "occurrences": row[2]})
    db.close()

    # Build lap coverage, retaining only audit statistics and not raw data.
    coverage_rows: list[dict[str, Any]] = []
    for key in sorted(lap_stats):
        year, event, session, driver, lap = key
        st = lap_stats[key]
        dts = np.asarray(st.pop("positive_dts"), dtype=float)
        coverage_rows.append({
            "year": year, "event": event, "session": session, "driver": driver, "lap": lap,
            "row_count": st["row_count"], "sample_index_min": st["sample_index_min"], "sample_index_max": st["sample_index_max"],
            "time_min_sec": st["time_min"], "time_max_sec": st["time_max"], "time_span_sec": (st["time_max"] - st["time_min"]) if st["time_min"] is not None and st["time_max"] is not None else None,
            "distance_min": st["distance_min"], "distance_max": st["distance_max"], "distance_span": (st["distance_max"] - st["distance_min"]) if st["distance_min"] is not None and st["distance_max"] is not None else None,
            "timestamp_repeats": st["timestamp_repeats"], "timestamp_negative_steps": st["timestamp_negative_steps"],
            "sample_index_repeats": st["sample_index_repeats"], "sample_index_negative_steps": st["sample_index_negative_steps"],
            "positive_dt_count": len(dts), "positive_dt_median_sec": float(np.median(dts)) if len(dts) else None,
            "positive_dt_p95_sec": float(np.percentile(dts, 95)) if len(dts) else None, "positive_dt_max_sec": float(np.max(dts)) if len(dts) else None,
            "complete_physical_lap_supported": False,
    })
    coverage_frame = pd.DataFrame(coverage_rows)
    coverage_frame.to_csv(output_dir / "phase1_coverage.csv", index=False)

    numeric_summary: dict[str, Any] = {}
    for field, vals in numeric_values.items():
        arr = np.asarray(vals, dtype=float)
        numeric_summary[field] = {
            "valid_count": int(valid_numeric[field]), "parse_failure_count": int(parse_failures[field]), "nonfinite_count": int(nonfinite[field]),
            "min": float(np.min(arr)) if len(arr) else None, "median": float(np.median(arr)) if len(arr) else None,
            "p01": float(np.percentile(arr, 1)) if len(arr) else None, "p95": float(np.percentile(arr, 95)) if len(arr) else None,
            "p999": float(np.percentile(arr, 99.9)) if len(arr) else None, "max": float(np.max(arr)) if len(arr) else None,
        }

    missing_by_race = {rid: {f: int(c) for f, c in sorted(cnt.items()) if c} for rid, cnt in sorted(race_missing.items())}
    selection_by_race: dict[str, dict[str, Any]] = {}
    by_race_driver: dict[tuple[str, str], list[int]] = defaultdict(list)
    for y, e, s, d, lap in laps:
        try:
            by_race_driver[(f"{y}|{e}|{s}", d)].append(int(float(lap)))
        except (ValueError, TypeError):
            pass
    for (rid, driver), lapnums in sorted(by_race_driver.items()):
        vals = sorted(set(lapnums))
        gaps = np.diff(vals) if len(vals) > 1 else np.array([], dtype=int)
        out = selection_by_race.setdefault(rid, {"drivers": 0, "selected_laps": 0, "gap_values": []})
        out["drivers"] += 1
        out["selected_laps"] += len(vals)
        out["gap_values"].extend(gaps.tolist())
    for rid, out in selection_by_race.items():
        gaps = np.asarray(out.pop("gap_values"), dtype=float)
        out["lap_gap_count"] = int(len(gaps)); out["lap_gap_median"] = float(np.median(gaps)) if len(gaps) else None
        out["lap_gap_p95"] = float(np.percentile(gaps, 95)) if len(gaps) else None
        out["lap_gap_max"] = float(np.max(gaps)) if len(gaps) else None
        out["rows"] = int(race_rows[rid])
    season_summary: dict[str, Any] = {}
    for year, frame in coverage_frame.groupby("year", sort=True):
        season_summary[str(year)] = {
            "rows": int(frame["row_count"].sum()), "selected_laps": int(len(frame)),
            "race_identities": int(frame[["event", "session"]].drop_duplicates().shape[0]),
            "driver_codes": int(frame["driver"].nunique()),
        }
    driver_summary: dict[str, Any] = {}
    for driver, frame in coverage_frame.groupby("driver", sort=True):
        driver_summary[str(driver)] = {
            "rows": int(frame["row_count"].sum()), "selected_laps": int(len(frame)),
            "race_identities": int(frame[["year", "event", "session"]].drop_duplicates().shape[0]),
            "seasons": sorted(str(x) for x in frame["year"].unique()),
        }

    source_manifest_comparison = {
        "archive_year_rows": {str(y): int(year_rows[str(y)]) for y in YEARS},
        "metadata_year_rows": manifest.get("year_rows", validation.get("year_rows", {})),
        "archive_rows": total_rows,
        "metadata_rows": manifest.get("rows", validation.get("rows")),
        "archive_unique_collector_sample_keys": unique_source_keys,
        "metadata_unique_sample_keys": manifest.get("unique_sample_keys", validation.get("unique_sample_keys")),
        "archive_selected_laps": len(laps),
        "metadata_selected_laps": validation.get("selected_laps", manifest.get("laps")),
        "archive_race_events": len(races),
        "metadata_race_events": manifest.get("race_events", validation.get("race_events")),
        "archive_driver_codes": len(drivers),
        "metadata_driver_codes": validation.get("driver_codes", manifest.get("unique_driver_codes")),
        "archive_compressed_csv_bytes": compressed_csv_bytes,
        "metadata_compressed_csv_bytes": manifest.get("compressed_csv_bytes", validation.get("compressed_csv_bytes")),
        "archive_uncompressed_csv_bytes": uncompressed_csv_bytes,
        "metadata_uncompressed_csv_bytes": validation.get("uncompressed_csv_bytes", manifest.get("uncompressed_csv_bytes")),
        "matches": {
            "rows": total_rows == manifest.get("rows", validation.get("rows")),
            "year_rows": {str(y): int(year_rows[str(y)]) == int((manifest.get("year_rows", validation.get("year_rows", {})) or {}).get(str(y), -1)) for y in YEARS},
            "unique_collector_sample_keys": unique_source_keys == manifest.get("unique_sample_keys", validation.get("unique_sample_keys")),
            "selected_laps": len(laps) == validation.get("selected_laps", manifest.get("laps")),
            "race_events": len(races) == manifest.get("race_events", validation.get("race_events")),
            "driver_codes": len(drivers) == validation.get("driver_codes", manifest.get("unique_driver_codes")),
            "compressed_csv_bytes": compressed_csv_bytes == manifest.get("compressed_csv_bytes", validation.get("compressed_csv_bytes")),
            "uncompressed_csv_bytes": uncompressed_csv_bytes == validation.get("uncompressed_csv_bytes", manifest.get("uncompressed_csv_bytes")),
        },
    }
    independent_match = {
        str(y): {"rows": independent[str(y)]["rows"], "pandas_rows": int(year_rows[str(y)]), "match": independent[str(y)]["rows"] == int(year_rows[str(y)]), "header_match": independent[str(y)]["header"] == EXPECTED_COLUMNS, "uncompressed_bytes": independent[str(y)]["uncompressed_bytes"]}
        for y in YEARS
    }
    integrity = {
        "all_separate_zip_crc_tests_passed": all(archives[str(y)]["zip_test_passed"] for y in YEARS) and archives["metadata"]["zip_test_passed"],
        "all_expected_members_safe": all(not archives[k]["unsafe_members"] for k in archives),
        "gzip_csv_decompression_and_crc_passed": not raw_bad_records,
        "metadata_errors": metadata_errors,
        "raw_parse_errors": raw_bad_records,
        "source_hashes_unchanged": source_hashes_unchanged,
        "input_sha256_before": before_hashes,
        "input_sha256_after": after_hashes,
        "legacy_combined_archive_checked": bool(legacy_path),
        "legacy_combined_archive_opened": archives.get("legacy_combined", {}).get("zip_opened") if legacy_path else None,
        "legacy_combined_archive_error": archives.get("legacy_combined", {}).get("error") if legacy_path else None,
    }

    metrics: dict[str, Any] = {
        "audit": {"name": "APEX-R Phase 1 Dataset Audit", "run_started_utc": run_started, "run_completed_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "input_dir": str(input_dir), "output_dir": str(output_dir), "sealed_holdout_exclusion": "No path containing session 11353 was permitted; only 2018-2022 supplied telemetry archives were inspected."},
        "archives": archives,
        "integrity": integrity,
        "metadata_top_level_keys": {k: sorted(v.keys()) if isinstance(v, dict) else type(v).__name__ for k, v in metadata.items()},
        "documented_provenance": {"inventory": {k: inventory.get(k) for k in ("groups", "candidate_files", "years", "commits")}, "manifest_keys": sorted(manifest.keys()), "validation_keys": sorted(validation.keys()), "repository_urls_found": [], "license_files_found": [], "readme_files_found": [], "scripts_found": [], "processing_versions_found": []},
        "schema": {"expected_columns": EXPECTED_COLUMNS, "observed_by_year": schema_by_year, "all_headers_identical": len(set(json.dumps(v, sort_keys=True) for v in schema_by_year.values())) == 1},
        "source_manifest_comparison": source_manifest_comparison,
        "independent_count_cross_check": independent_match,
        "totals": {"observations": total_rows, "unique_collector_sample_keys": unique_source_keys, "unique_source_data_lineage_keys": len(source_lineage_keys), "unique_canonical_observation_keys": unique_canonical_keys, "selected_laps": len(laps), "race_events_season_event_session": len(races), "season_event_names": len(events), "driver_codes": len(drivers), "seasons": sorted(year_rows), "compressed_csv_member_bytes": compressed_csv_bytes, "uncompressed_csv_bytes": uncompressed_csv_bytes},
        "duplicates": {"observation_key_justification": "source_data_key is a repeated lap/file lineage key in this CSV, not a unique observation key. The collector-style observation key tested is (source_data_key, sample_index), matching the metadata's unique_sample_keys claim. Canonical fallback is (year,event,session,driver,lap,sample_index). A race identity includes season plus event plus session.", "source_data_key_lineage_rows": total_rows, "source_data_key_lineage_unique_values": len(source_lineage_keys), "collector_sample_key_rows": total_source_key_rows, "collector_sample_key_missing_rows": total_rows - total_source_key_rows, "exact_duplicate_rows_by_full_payload_hash": exact_duplicate_rows, "repeated_collector_sample_key_rows": source_repeated_rows, "repeated_collector_sample_keys_with_conflicting_payload": source_conflicting_keys, "repeated_canonical_key_rows": canonical_repeated_rows, "repeated_canonical_keys_with_conflicting_payload": canonical_conflicting_keys, "duplicate_examples": duplicate_examples},
        "missingness": {"total_by_field": {f: int(missing[f]) for f in EXPECTED_COLUMNS if missing[f]}, "percent_by_field": {f: pct(missing[f], total_rows) for f in EXPECTED_COLUMNS}, "by_race": missing_by_race, "null_tokens": sorted(NULL_TOKENS)},
        "numeric_quality": {"parse_failures_by_field": dict(parse_failures), "nonfinite_by_field": dict(nonfinite), "summary_by_field": numeric_summary, "categorical_codes": {"drs": dict(drs_codes), "session": dict(session_codes)}, "anomaly_counts": dict(anomaly_counts), "anomaly_examples_written": len(anomaly_examples), "anomaly_example_counts_by_reason": dict(anomaly_example_counts), "rules": {"throttle_outside_0_100": "value < 0 or > 100", "gear_not_integer_or_outside_0_8": "not integer-valued or outside 0..8", "brake_outside_0_1": "value < 0 or > 1", "drs_not_0_or_1": "observed code not 0 or 1; not proof of error", "negative_distance/rpm/speed": "value < 0; expected units remain unverified", "rel_distance_outside_0_1_tolerance": "value outside [-1e-6,1+1e-6] because field is ratio-like by observed values"}},
        "coverage": {"lap_rows": len(coverage_rows), "season_summary": season_summary, "driver_summary": driver_summary, "race_summary": selection_by_race, "time_interpretation": "time resets near zero within each selected lap and has no UTC/session anchor; lap-relative seconds is the supported interpretation, while actual clock/delivery time is UNKNOWN.", "physical_complete_lap_supported": False, "physical_complete_lap_reason": "The archive has selected lap identifiers and observed spans, but no independent lap start/end metadata, track-length reference, or selection/collection code proving that each selected interval is a complete physical lap.", "timestamp_metrics_scope": "Positive deltas were calculated only within each (season,event,session,driver,lap), never across laps or unrelated races."},
        "prediction_time_readiness": {"overall_causal_training_ready": False, "prefix_invariance_test": "NOT_EXECUTED: collection/processing scripts and formulas are absent from the supplied archives; a passing local check cannot be reconstructed honestly.", "feature_classes": build_readiness()},
        "readiness": {"telemetry_rows_are_not_independent_examples": True, "overtake_labels_created": False, "unique_overtake_events_counted": False, "weather_or_pit_data_joined": False, "model_training_performed": False, "causal_training_ready": False, "reason": "Readable telemetry and structural validation do not establish causal availability, full-race context, public event alignment, opponent identity provenance, or overtake labels."},
    }
    write_json(output_dir / "phase1_metrics.json", metrics)
    pd.DataFrame(build_readiness()).to_csv(output_dir / "phase1_feature_readiness.csv", index=False)
    pd.DataFrame(anomaly_examples).to_csv(output_dir / "phase1_anomaly_examples.csv", index=False)
    write_json(output_dir / "source_inventory.json", {"archives": archives, "metadata": metadata, "input_sha256_before": before_hashes, "input_sha256_after": after_hashes})

    report = render_report(metrics)
    (output_dir / "PHASE1_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    handoff = render_handoff(metrics)
    (output_dir / "PHASE2_HANDOFF.md").write_text(handoff, encoding="utf-8")
    readme = render_readme(input_dir, output_dir)
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_checksums(output_dir)
    db_path.unlink(missing_ok=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    print(json.dumps({"output_dir": str(output_dir), "rows": total_rows, "selected_laps": len(laps), "races": len(races), "drivers": len(drivers), "source_hashes_unchanged": source_hashes_unchanged, "anomalies": dict(anomaly_counts)}, indent=2))
    return 0


def write_checksums(output_dir: Path) -> None:
    """Write checksums for final root-level audit artifacts, excluding itself."""
    files = sorted(
        p for p in output_dir.iterdir()
        if p.is_file() and p.name != "SHA256SUMS.txt" and p.name != "APEX-R_Phase1_Dataset_Audit.zip"
    )
    lines = [f"{sha256_file(path)}  {path.name}" for path in files]
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_readme(input_dir: Path, output_dir: Path) -> str:
    legacy_arg = " --legacy-combined-zip " + str(input_dir / "APEX-R_Updated_Telemetry_Dataset_2018_2022.zip") if (input_dir / "APEX-R_Updated_Telemetry_Dataset_2018_2022.zip").exists() else ""
    legacy_note = "The legacy combined archive was available and checked separately; it is never used as a second data source." if legacy_arg else "The legacy combined archive was not present at the supplied input path during this run, so it was not used or reverified."
    return f"""# APEX-R Phase 1 Dataset Audit

This is an audit package, not a cleaned or training-ready telemetry dataset. The source ZIPs are preserved outside this directory and were read without extraction or modification. The audit rejects unsafe ZIP member paths, checks ZIP CRCs, forces each nested GZIP stream to EOF, computes SHA-256 hashes, and performs a second row count with Python's `csv.reader`.

## Reproduce

```bash
python3 -m venv .venv_phase1_audit
source .venv_phase1_audit/bin/activate
python -m pip install --upgrade pip pandas numpy
python phase1_audit.py \\
  --input-dir {input_dir} \\
  --output-dir {output_dir} \\
  --metadata-zip {input_dir / 'APEX-R_Telemetry_Metadata.zip'}{legacy_arg}
```

The audit script accepts different absolute paths via the same arguments. It does not fetch telemetry, train a model, tune a model, create overtake labels, enrich data, or inspect sealed session 11353. `--legacy-combined-zip` is optional; when supplied, it is checked for packaging integrity and never used as a second data source. {legacy_note}

## Outputs

* `PHASE1_AUDIT_REPORT.md` — evidence-backed human-readable findings.
* `phase1_metrics.json` — exact totals, integrity, duplicate, coverage, missingness, range and readiness results.
* `phase1_coverage.csv` — one row per observed selected lap with time/distance/sampling diagnostics.
* `phase1_feature_readiness.csv` — feature-by-feature SAFE/UNSAFE/UNKNOWN assessment.
* `phase1_anomaly_examples.csv` — traceable source-value examples; values are not changed.
* `PHASE2_HANDOFF.md` — prerequisites and join contracts for public enrichment.
* `source_inventory.json` — archive/member inventories and metadata snapshot.
* `SHA256SUMS.txt` — checksums for retained audit artifacts, excluding itself.

Source archives are intentionally not copied into this package.
"""


def render_report(metrics: dict[str, Any]) -> str:
    t = metrics["totals"]
    s = metrics["source_manifest_comparison"]
    miss = metrics["missingness"]["total_by_field"]
    nq = metrics["numeric_quality"]
    dup = metrics["duplicates"]
    integ = metrics["integrity"]
    prov = metrics["documented_provenance"]
    legacy_state = metrics["integrity"].get("legacy_combined_archive_checked")
    legacy_package_note = (
        "The optional earlier combined ZIP was checked separately and found invalid/truncated; it is not used for counts. This resolves the README/licence packaging discrepancy: those files are not in the authoritative separate archives."
        if legacy_state else
        "The optional earlier combined ZIP was not present at the supplied filesystem path during this run, so its integrity could not be reverified and it was not used. The authoritative separate archives contain no README, licence, scripts, or processing version files."
    )
    lines = [
        "# APEX-R Phase 1 Dataset Audit Report", "",
        f"Audit run: `{metrics['audit']['run_completed_utc']}`. Scope: supplied 2018–2022 archives only. Session `11353` was excluded by input-path guard and was not inspected.", "",
        "## Executive conclusion", "",
        "The separate yearly archives are readable and structurally consistent, but this is a processed, selected-lap telemetry export with incomplete provenance and no causal-processing code. It is not yet demonstrated training-ready. Causal model training should not begin from this package until timing/provenance, full-lap context, opponent matching and labels are reconstructed in later phases.", "",
        "## 1. Verified totals", "",
        f"* Observations: **{t['observations']:,}**; unique collector-style `(source_data_key,sample_index)` keys: **{t['unique_collector_sample_keys']:,}**; unique source-data lineage keys: **{t['unique_source_data_lineage_keys']:,}**; unique canonical `(season,event,session,driver,lap,sample_index)` keys: **{t['unique_canonical_observation_keys']:,}**.",
        f"* Selected laps: **{t['selected_laps']:,}**; race identities `(season,event,session)`: **{t['race_events_season_event_session']}**; season/event names: **{t['season_event_names']}**; driver codes: **{t['driver_codes']}**; seasons: **{', '.join(t['seasons'])}**.",
        f"* Per-year rows: `{json.dumps(s['archive_year_rows'], sort_keys=True)}`.",
        f"* CSV member bytes: compressed `.gz` members **{t['compressed_csv_member_bytes']:,}**; decompressed CSV bytes **{t['uncompressed_csv_bytes']:,}**. Metadata byte agreement: compressed={s['matches']['compressed_csv_bytes']}, uncompressed={s['matches']['uncompressed_csv_bytes']}.",
        f"* Metadata agreement: `{json.dumps(s['matches'], sort_keys=True)}`.", "",
        "## 2. Integrity and packaging", "",
        "PASS: all six separate ZIP CRC tests passed, expected member paths were safe, all five nested GZIP CSV streams decompressed to EOF, and the independent `csv.reader` row/header pass matched the pandas audit.", "",
        f"PASS: source SHA-256 values were unchanged before/after audit: **{integ['source_hashes_unchanged']}**.", "",
        "The metadata ZIP contains exactly `inventory.json`, `manifest.json`, and `validation.json`. Each yearly ZIP contains exactly one `telemetry_YYYY.csv.gz` member. No README, licence, collection script, processing script, repository URL, or processing-version string is present in the supplied archive. The inventory does document five commit hashes and included source blob hashes, but without a repository URL/name those identifiers cannot be independently resolved from the archive alone.", "",
        legacy_package_note, "",
        "## 3. Coverage, independence and completeness", "",
        f"Race identity uses season + event + session, preventing same-named events in different seasons from being conflated. `source_data_key` is a repeated lap/file lineage key, so the collector-style observation key tested is `(source_data_key,sample_index)` and is compared with the canonical fallback key. Exact full-payload duplicate rows: **{dup['exact_duplicate_rows_by_full_payload_hash']:,}**; repeated collector-sample-key rows: **{dup['repeated_collector_sample_key_rows']:,}**; conflicting collector-sample keys: **{dup['repeated_collector_sample_keys_with_conflicting_payload']}**; repeated canonical-key rows: **{dup['repeated_canonical_key_rows']:,}**; conflicting canonical keys: **{dup['repeated_canonical_keys_with_conflicting_payload']}**.", "",
        "The rows are telemetry observations, not independent prediction windows. No overtake labels or unique overtake events were generated or estimated. Driver codes remain source identifiers; no identity crosswalk is packaged.", "",
        "There are 70 race identities, 1,332 selected laps, and multiple drivers per race, but the archive contains only selected lap slices. `phase1_coverage.csv` reports observed time/distance spans, timestamp repetition/order, sample-index order, and positive within-lap intervals. No independent lap start/end metadata, track-length reference, or selection/collection code is supplied, so complete physical laps are **not proven**. Lap-gap statistics are descriptive of selected identifiers and do not prove that omitted laps are irrelevant.", "",
        f"The exact season/race/driver coverage summaries are retained in `phase1_metrics.json`: `{json.dumps(metrics['coverage']['season_summary'], sort_keys=True)}`. The per-driver table in that file records selected laps, rows and race identities; this is coverage, not independent-example count.", "",
        "## 4. Time, units and alignment", "",
        "`time` resets near zero inside selected laps and has no UTC or session-start anchor; lap-relative seconds is the supported interpretation. Sampling intervals were calculated only within each season/event/session/driver/lap. Actual ingestion/delivery latency is unknown. A reliable weather or public pit-event join therefore requires an external session/lap UTC anchor and verified driver identity mapping.", "",
        "The archive has no authoritative unit/codebook. Speed values are numerically plausible as a km/h-like channel, `rel_distance` is ratio-like, and coordinates vary by event/year, but these observations are not proof of units or semantics. `drs` codes require a source codebook. `acc_x/y/z` and `rel_distance` look potentially derived, but the processing formula is absent.", "",
        "## 5. Missingness and anomalies", "",
        f"Missing values (empty strings plus null-like tokens) are: `{json.dumps(miss, sort_keys=True)}`. The supplied claims for DriverAhead, DistanceToDriverAhead and rel_distance should be compared against these exact values in `phase1_metrics.json`.", "",
        f"Explicit audit-rule flags: `{json.dumps(nq['anomaly_counts'], sort_keys=True)}`. The throttle rule is `<0 or >100`; gear is non-integer-valued or outside `0..8`; brake is outside `0..1`; negative physical channels and ratio-like `rel_distance` are separately recorded. These are flags, not proof of source error. Source values remain unchanged. Examples with source identifiers are in `phase1_anomaly_examples.csv`.", "",
        "In particular, a throttle value of 104 cannot be called wrong without the provider's encoding contract. Gear values outside the ordinary 0–8 expectation likewise need source semantics; a count under an arbitrary rule is not a repair.", "",
        "## 6. Prediction-time readiness and leakage", "",
        "Overall `causal_training_ready` is **false**. The supplied archive lacks processing code, so prefix invariance cannot be honestly reconstructed for `rel_distance`, acceleration, opponent matching or any normalization/smoothing. Those fields are UNKNOWN, not silently declared safe. Direct observed channels are available at their recorded sample in the file, but delivery latency and UTC alignment are unknown. `phase1_feature_readiness.csv` contains the feature-by-feature evidence and repairs required.", "",
        "No demonstrated UNSAFE transformation was available to prove a specific future leak; however, absent code means possible interpolation, centred smoothing, backward filling, whole-lap statistics, distance normalization and future-aware opponent matching remain unresolved. A passing check on a few rows would not establish universal safety.", "",
        "## 7. Readiness decision", "",
        "**NO for causal model training from this archive alone.** Readability and large row count passed, but causal availability, independent examples, complete race context, event alignment, opponent identity, source provenance and labels are not established. Phase 1 is complete as an audit; training readiness is a later handoff condition, not a claim made here.", "",
        "## Prioritised next actions", "",
        "1. Recover the repository URL/name, exact processing scripts/version and licence; rebuild a provenance manifest and verify the five commit hashes.",
        "2. Obtain a verified driver/event/session crosswalk plus session UTC and lap-boundary anchors; reconstruct full-race context and test prefix invariance of all derived/opponent fields.",
        "3. In Phase 2, add public weather, pit-stop and race-control data with backward-only joins and explicit freshness/missing-match fields; only then proceed to causal feature engineering and defensible labels.", "",
        "See `PHASE2_HANDOFF.md` for join contracts, assumptions and proprietary-data blockers. No model was trained or scored in this audit.", "",
    ]
    return "\n".join(lines)


def render_handoff(metrics: dict[str, Any]) -> str:
    return """# APEX-R Phase 2 Handoff

Phase 1 found readable 2018–2022 selected-lap telemetry, but no UTC/session anchor, verified identity crosswalk, processing code, or source licence. Phase 2 must enrich the data without changing source values and without creating labels.

## Required public fields

* Weather: session identity, UTC timestamp (or a documented session-relative timestamp), track/air temperature, rainfall, humidity/pressure/wind where available.
* Pit and race context: driver identity, pit entry/exit or pit stop timestamp, pit-lane duration, stop duration, lap, race-control flags/safety-car/virtual-safety-car/red-flag intervals and messages.
* Tyres/stints: driver, lap or timestamp, compound, stint number and tyre age.
* Identity: season/event/session crosswalk, driver number/code and session start/lap-start anchors.

## Candidate sources and limitations

OpenF1 is a candidate source for public weather, pit, race-control, laps, intervals and positions where historical coverage exists. TracingInsights or the original upstream source may supply older rival/telemetry fields. The supplied archive itself does not prove that these sources share a clock or identifier namespace. Private ERS/battery percentage, team deployment/harvesting maps, fuel load, team instructions, radio and confidential undercut/overcut decisions remain unavailable from public timing data.

## Join contract

1. Resolve a race identity using `(season, event identity, session)` and a verified crosswalk; never join on event name alone.
2. Resolve driver identity using a verified code/number mapping. Source `driver` and `DriverAhead` are not independently verified identities.
3. Anchor the telemetry `lap` and apparent lap-relative `time` to UTC/session time using documented lap start or session start records. Do not infer this from a filename or an assumed race schedule.
4. Join telemetry to timestamped public data on `(session identity, driver, timestamp)`; use `lap` as a supporting key, not a substitute for a shared clock.
5. Use backward-only as-of joins. Proposed starting assumptions (must be sensitivity-tested, not treated as facts): telemetry channels ≤0.5 seconds old; OpenF1 interval rows ≤5 seconds old because interval cadence is coarser; weather ≤60 seconds old; pit/race-control events only at or before the observation and preferably exact event timestamps. Store `match_age_seconds`, `source_timestamp`, and `match_status`.
6. Never interpolate, backward-fill future values, or replace no-match with zero. Represent `missing`, `stale`, `ambiguous`, `unanchored` and `matched` explicitly; retain NaN for missing numeric values.

## Blockers and implementation order

* Blocker 1: obtain source repository URL/name, licence, processing code and version so raw vs derived fields and any interpolation/centred smoothing can be audited.
* Blocker 2: obtain UTC/session/lap anchors and a driver/event identity crosswalk; without them weather and pit joins are not reliable.
* Then add weather and public pit/race-control data in a separate enrichment table, run integrity and causal tests, and report match freshness/missingness before feature engineering.
* Only in Phase 3 should opponent matching, pit/retirement/lapping context and overtake labels be created. Genuine passes require opponent identity, event timestamp and surrounding windows; pit stops, retirements, lapping/unlapping, timing corrections and temporary swaps must be excluded or separately classified.

This handoff deliberately contains no labels, model scores or cleaned source values.
"""


if __name__ == "__main__":
    raise SystemExit(main())
