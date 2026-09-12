#!/usr/bin/env python3
"""APEX-R full-race context collection and evidence export.

This is a historical-data collection audit, not a model-training pipeline.
It materializes complete *lap-level* race context from the already approved,
pinned TracingInsights cache and joins public context that already exists in
Phase 2.  It deliberately does not claim continuous car telemetry or verified
overtake timing.  Lap-boundary position reversals are exported as unresolved
candidates; public pit-stop and race-control records are exported separately
as source-recorded public events.

The program runs in two gates in one invocation:
  1. validate three approved train races;
  2. expand only after all three pass the complete-source-lap-context gate.

No remote request is made by this version.  Every source payload is reused
from the Phase 2/Phase 3 pinned caches and checked against its recorded SHA256.
The protected-session exclusion manifest is read before any source access.

Standard-library only.  Example:
  python3 collection_pipeline.py \
    --workspace /path/to/APEX-R_Devsez_Full_Application \
    --output /path/to/APEX-R_Devsez_Full_Application/full_race_collection
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote


VERSION = "full-race-collection-v1.0"
SESSION = "Race"
YEARS = tuple(range(2018, 2023))
NULLS = {"", "none", "null", "nan", "nat", "na", "n/a"}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
            count += 1
    return count


def write_csv_gz(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=6) as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
            count += 1
    return count


def parse_number(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in NULLS:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_int(value: Any) -> int | None:
    x = parse_number(value)
    if x is None or not x.is_integer():
        return None
    return int(x)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def race_id(year: str, event: str, session: str = SESSION) -> str:
    return f"{year}:{event}:{session}"


def split_race_id(value: str) -> tuple[str, str, str]:
    bits = value.split(":", 2)
    if len(bits) != 3:
        raise ValueError(f"invalid race_id: {value!r}")
    return bits[0], bits[1], bits[2]


def columnar_records(payload: Any) -> list[dict[str, Any]]:
    """Convert TracingInsights columnar JSON without changing source files."""
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    arrays = {k: v for k, v in payload.items() if isinstance(v, list)}
    if not arrays:
        return [dict(payload)]
    n = max((len(v) for v in arrays.values()), default=0)
    return [
        {k: (v[i] if isinstance(v, list) and i < len(v) else v) for k, v in payload.items()}
        for i in range(n)
    ]


def source_cache_record(url: str, record: Mapping[str, Any], protected: str) -> dict[str, Any]:
    """Check a cached source reference before reading its payload."""
    if protected and protected in url:
        raise RuntimeError("protected-session token found in source URL before access")
    cache_path = Path(str(record.get("cache_path", "")))
    if not cache_path.is_file():
        raise FileNotFoundError(f"missing cached source payload: {cache_path}")
    raw = cache_path.read_bytes()
    actual = sha256_bytes(raw)
    expected = str(record.get("sha256", ""))
    return {
        "url": url,
        "cache_path": str(cache_path),
        "bytes": len(raw),
        "sha256": actual,
        "expected_sha256": expected,
        "sha256_match": bool(expected) and actual == expected,
        "status": record.get("status", "UNKNOWN"),
        "http_status": record.get("http_status"),
        "retrieved_at_utc": record.get("retrieved_at_utc", record.get("cache_created_at_utc", "")),
        "payload_bytes": raw,
    }


class Inputs:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.phase2 = self.workspace / "phase2_data_foundation"
        self.phase3 = self.workspace / "phase3_prediction_dataset"
        self.split_path = self.phase3 / "phase3_split_manifest.csv"
        self.exclusion_path = self.phase3 / "excluded_sessions_manifest.json"
        self.coverage_path = self.phase3 / "context" / "laptime_coverage.csv"
        self.provenance_path = self.phase3 / "provenance" / "phase3_provenance_manifest.json"
        self.fetch_log_path = self.phase2 / "provenance" / "source_fetch_log.json"
        self.pit_path = self.phase2 / "context" / "pit_stop_context.csv"
        self.rc_path = self.phase2 / "context" / "race_control_context.csv"
        self.weather_path = self.phase2 / "context" / "weather_context.csv"
        required = [self.split_path, self.exclusion_path, self.coverage_path, self.provenance_path, self.fetch_log_path, self.pit_path, self.rc_path, self.weather_path]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("required prior artifact(s) missing: " + ", ".join(missing))

        self.exclusion = json.loads(self.exclusion_path.read_text(encoding="utf-8"))
        self.protected = str(self.exclusion.get("protected_session_token", ""))
        if not self.protected:
            raise RuntimeError("exclusion manifest has no protected session token")

        self.split_rows = list(csv.DictReader(self.split_path.open(encoding="utf-8", newline="")))
        self.coverage_rows = list(csv.DictReader(self.coverage_path.open(encoding="utf-8", newline="")))
        self.fetch_rows = json.loads(self.fetch_log_path.read_text(encoding="utf-8"))
        prov = json.loads(self.provenance_path.read_text(encoding="utf-8"))
        self.laptime_rows = prov.get("laptime_retrievals", [])
        self.url_records: dict[str, dict[str, Any]] = {}
        for rec in self.fetch_rows:
            url = str(rec.get("url", ""))
            if url:
                self.url_records.setdefault(url, rec)
        for rec in self.laptime_rows:
            url = str(rec.get("url", ""))
            if url:
                self.url_records[url] = rec

        self.split_by_race = {r["race_id"]: r for r in self.split_rows}
        self.coverage_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.coverage_rows:
            self.coverage_by_race[race_id(row["year"], row["event"], row["session"])].append(row)
        self.pit_rows = list(csv.DictReader(self.pit_path.open(encoding="utf-8", newline="")))
        self.rc_rows = list(csv.DictReader(self.rc_path.open(encoding="utf-8", newline="")))
        self.weather_rows = list(csv.DictReader(self.weather_path.open(encoding="utf-8", newline="")))
        self.pits_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.rc_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.weather_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.pit_rows:
            self.pits_by_race[race_id(row["year"], row["event"], row["session"])].append(row)
        for row in self.rc_rows:
            self.rc_by_race[race_id(row["year"], row["event"], row["session"])].append(row)
        for row in self.weather_rows:
            self.weather_by_race[race_id(row["year"], row["event"], row["session"])].append(row)

        self.input_paths = [self.split_path, self.exclusion_path, self.coverage_path, self.provenance_path, self.fetch_log_path, self.pit_path, self.rc_path, self.weather_path]

    def hashes(self) -> dict[str, str]:
        return {str(p.relative_to(self.workspace)): sha256_file(p) for p in self.input_paths}

    def guarded_source(self, url: str) -> dict[str, Any]:
        if self.protected in url or self.protected in unquote(url):
            raise RuntimeError("protected-session token found before source access")
        rec = self.url_records.get(url)
        if rec is None:
            raise KeyError(f"no prior cached source record for {url}")
        out = source_cache_record(url, rec, self.protected)
        if not out["sha256_match"]:
            raise RuntimeError(f"source SHA256 mismatch: {url}")
        return out

    def payload(self, url: str) -> tuple[Any, dict[str, Any]]:
        source = self.guarded_source(url)
        try:
            obj = json.loads(source["payload_bytes"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cached JSON unreadable: {url}: {exc}") from exc
        source.pop("payload_bytes", None)
        return obj, source

    def drivers_url(self, year: str, event: str, session: str) -> str:
        prefix = f"https://raw.githubusercontent.com/TracingInsights/{year}/"
        needle = f"/{event.replace(' ', '%20')}/{session.replace(' ', '%20')}/drivers.json"
        candidates = [u for u in self.url_records if u.startswith(prefix) and needle in u]
        if len(candidates) != 1:
            raise KeyError(f"expected one drivers.json URL for {year}/{event}/{session}, got {len(candidates)}")
        return candidates[0]


def inspect_race(inputs: Inputs, split_row: Mapping[str, str], source_access: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    year, event, session = split_row["year"], split_row["event"], split_row["session"]
    rid = split_row["race_id"]
    driver_url = inputs.drivers_url(year, event, session)
    drivers_payload, driver_source = inputs.payload(driver_url)
    source_access.append({"race_id": rid, "source_kind": "TracingInsights drivers.json", **driver_source})
    expected = drivers_payload.get("drivers", []) if isinstance(drivers_payload, dict) else []
    expected_by_code = {str(d.get("driver")): d for d in expected if isinstance(d, dict) and d.get("driver")}
    coverage = inputs.coverage_by_race.get(rid, [])
    by_code = {r["driver"]: r for r in coverage}
    issues: list[str] = []
    if set(expected_by_code) != set(by_code):
        missing = sorted(set(expected_by_code) - set(by_code))
        extra = sorted(set(by_code) - set(expected_by_code))
        if missing:
            issues.append("missing_laptime_driver=" + ",".join(missing))
        if extra:
            issues.append("unexpected_laptime_driver=" + ",".join(extra))

    driver_rows: list[dict[str, Any]] = []
    all_source_fields: set[str] = set()
    source_files = 0
    valid_files = 0
    total_laps = 0
    field_min = None
    field_max = None
    for code in sorted(expected_by_code):
        cov = by_code.get(code)
        if cov is None:
            issues.append("no_coverage=" + code)
            continue
        url = cov["source_url"]
        try:
            payload, src = inputs.payload(url)
            source_access.append({"race_id": rid, "source_kind": "TracingInsights laptimes.json", "driver": code, **src})
        except Exception as exc:
            issues.append(f"laptime_read_error={code}:{type(exc).__name__}")
            continue
        source_files += 1
        records = columnar_records(payload)
        all_source_fields.update(payload.keys() if isinstance(payload, dict) else ())
        if not records:
            issues.append("empty_laptimes=" + code)
            continue
        valid_files += 1
        laps = [parse_int(r.get("lap")) for r in records]
        clean_laps = [x for x in laps if x is not None]
        if len(clean_laps) != len(records):
            issues.append("missing_lap_value=" + code)
        if clean_laps and clean_laps[0] != 1:
            issues.append(f"does_not_start_at_lap1={code}")
        if len(clean_laps) != len(set(clean_laps)):
            issues.append("duplicate_lap_record=" + code)
        if clean_laps:
            lo, hi = min(clean_laps), max(clean_laps)
            if set(range(lo, hi + 1)) != set(clean_laps):
                issues.append("lap_gap=" + code)
            field_min = lo if field_min is None else min(field_min, lo)
            field_max = hi if field_max is None else max(field_max, hi)
        total_laps += len(records)
        for index, row in enumerate(records):
            copied = dict(row)
            copied.update({"race_id": rid, "split": split_row["split"], "year": year, "event": event, "session": session, "driver": code, "driver_number": expected_by_code[code].get("dn", ""), "full_name": f"{expected_by_code[code].get('fn', '')} {expected_by_code[code].get('ln', '')}".strip(), "team": expected_by_code[code].get("team", ""), "source_url": url, "source_sha256": src["sha256"], "source_cache_status": src["status"], "record_index": index})
            driver_rows.append(copied)

    for kind, rows in [("pit", inputs.pits_by_race.get(rid, [])), ("race_control", inputs.rc_by_race.get(rid, [])), ("weather", inputs.weather_by_race.get(rid, []))]:
        for url in sorted({r.get("source_url", "") for r in rows if r.get("source_url")}):
            src = inputs.guarded_source(url)
            source_access.append({"race_id": rid, "source_kind": kind, **src})

    complete = not issues and source_files == len(expected_by_code) and valid_files == len(expected_by_code)
    reason = "ALL_EXPECTED_DRIVERS_VALID_CONTIGUOUS_FROM_LAP1" if complete else ";".join(issues)[:2000]
    race = {
        "race_id": rid,
        "year": year,
        "event": event,
        "session": session,
        "split": split_row["split"],
        "chronology_rank": split_row.get("chronology_rank", ""),
        "expected_driver_count": len(expected_by_code),
        "laptime_file_count": source_files,
        "valid_laptime_file_count": valid_files,
        "lap_rows": total_laps,
        "field_min_lap": field_min,
        "field_max_lap": field_max,
        "pit_event_rows": len(inputs.pits_by_race.get(rid, [])),
        "race_control_rows": len(inputs.rc_by_race.get(rid, [])),
        "weather_rows": len(inputs.weather_by_race.get(rid, [])),
        "complete_source_lap_context": "PASS" if complete else "FAIL",
        "complete_context_reason": reason,
        "physical_complete_lap_proven": "NO",
        "physical_complete_lap_note": "Source is lap-level timing context; it is not continuous sub-lap telemetry and cannot prove every physical lap segment.",
        "verified_overtake_event_count": 0,
        "unresolved_position_swap_candidate_count": 0,
    }
    return race, driver_rows, {"expected_drivers": expected_by_code, "source_fields": sorted(all_source_fields)}


def stable_event_id(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(x) for x in parts).encode("utf-8")).hexdigest()[:24]


def make_verified_events(inputs: Inputs, complete_races: set[str]) -> list[dict[str, Any]]:
    """Export source-recorded public events; none are called overtakes."""
    out: list[dict[str, Any]] = []
    for rid in sorted(complete_races):
        for row in inputs.pits_by_race.get(rid, []):
            out.append({
                "event_id": stable_event_id("pit", rid, row.get("pit_index", "")),
                "race_id": rid,
                "event_type": "PUBLIC_PIT_STOP",
                "event_status": "VERIFIED_PUBLIC_EVENT",
                "verified_overtake": "FALSE",
                "driver": row.get("driver_code", ""),
                "target_driver": "",
                "event_lap": row.get("lap", ""),
                "event_time": row.get("time_of_day", ""),
                "duration_sec": row.get("duration_sec", ""),
                "source_url": row.get("source_url", ""),
                "source_sha256": row.get("source_sha256", ""),
                "evidence": "Jolpica/Ergast-compatible public pit-stop record; timing is source-provided time-of-day and may not be an aligned session timestamp.",
            })
        for row in inputs.rc_by_race.get(rid, []):
            out.append({
                "event_id": stable_event_id("rc", rid, row.get("rcm_index", "")),
                "race_id": rid,
                "event_type": "PUBLIC_RACE_CONTROL_MESSAGE",
                "event_status": "VERIFIED_PUBLIC_EVENT",
                "verified_overtake": "FALSE",
                "driver": row.get("driver_number", ""),
                "target_driver": "",
                "event_lap": row.get("lap", ""),
                "event_time": row.get("source_time_utc", ""),
                "duration_sec": "",
                "source_url": row.get("source_url", ""),
                "source_sha256": row.get("source_sha256", ""),
                "evidence": "TracingInsights rcm.json source-recorded message; no continuous active-state or overtake meaning is inferred.",
            })
    return out


def make_candidates(complete_races: list[dict[str, Any]], lap_rows: list[dict[str, Any]], inputs: Inputs) -> list[dict[str, Any]]:
    by_race_driver_lap: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in lap_rows:
        lap = parse_int(row.get("lap"))
        pos = parse_int(row.get("pos"))
        if lap is None or pos is None or pos <= 0:
            continue
        by_race_driver_lap[(row["race_id"], row["driver"], lap)] = row
    drivers_by_race: dict[str, set[str]] = defaultdict(set)
    for row in lap_rows:
        drivers_by_race[row["race_id"]].add(row["driver"])
    candidates: list[dict[str, Any]] = []
    for race in complete_races:
        rid = race["race_id"]
        laps = sorted({parse_int(r.get("lap")) for r in lap_rows if r["race_id"] == rid and parse_int(r.get("lap")) is not None})
        pit_by_driver_lap: set[tuple[str, int]] = set()
        for pit in inputs.pits_by_race.get(rid, []):
            lap = parse_int(pit.get("lap"))
            if lap is not None and pit.get("driver_code"):
                pit_by_driver_lap.add((pit["driver_code"], lap))
        rc_by_lap = {parse_int(r.get("lap")) for r in inputs.rc_by_race.get(rid, []) if parse_int(r.get("lap")) is not None}
        for lap in laps:
            if lap is None or lap + 1 not in laps:
                continue
            drivers = sorted(drivers_by_race[rid])
            for i, attacker in enumerate(drivers):
                a0 = by_race_driver_lap.get((rid, attacker, lap))
                a1 = by_race_driver_lap.get((rid, attacker, lap + 1))
                if not a0 or not a1:
                    continue
                p_a0, p_a1 = parse_int(a0.get("pos")), parse_int(a1.get("pos"))
                if p_a0 is None or p_a1 is None:
                    continue
                for target in drivers[i + 1:]:
                    t0 = by_race_driver_lap.get((rid, target, lap))
                    t1 = by_race_driver_lap.get((rid, target, lap + 1))
                    if not t0 or not t1:
                        continue
                    p_t0, p_t1 = parse_int(t0.get("pos")), parse_int(t1.get("pos"))
                    if p_t0 is None or p_t1 is None:
                        continue
                    # Fixed pair: attacker is behind at the decision boundary
                    # and ahead at the next observed boundary.
                    # Keep the candidate tied to the car immediately ahead at
                    # the decision boundary.  Non-adjacent reversals are
                    # usually pit/retirement/order effects and are not a
                    # defensible "car-ahead" candidate.
                    if not (p_a0 == p_t0 + 1 and p_a1 < p_t1):
                        continue
                    pit_flag = (attacker, lap) in pit_by_driver_lap or (attacker, lap + 1) in pit_by_driver_lap or (target, lap) in pit_by_driver_lap or (target, lap + 1) in pit_by_driver_lap
                    rc_flag = lap in rc_by_lap or lap + 1 in rc_by_lap
                    reasons = ["LAP_BOUNDARY_POSITION_REVERSAL_NO_SUBLAP_PASS_TIMESTAMP"]
                    if pit_flag:
                        reasons.append("PIT_CONTEXT_ON_BOUNDARY_OR_ENDPOINT_LAP")
                    if rc_flag:
                        reasons.append("RACE_CONTROL_MESSAGE_ON_BOUNDARY_OR_ENDPOINT_LAP")
                    candidates.append({
                        "candidate_id": stable_event_id("candidate", rid, lap, attacker, target),
                        "race_id": rid,
                        "split": race["split"],
                        "decision_lap": lap,
                        "endpoint_lap": lap + 1,
                        "attacker": attacker,
                        "target_driver": target,
                        "decision_position_attacker": p_a0,
                        "decision_position_target": p_t0,
                        "endpoint_position_attacker": p_a1,
                        "endpoint_position_target": p_t1,
                        "decision_boundary_utc_attacker": a0.get("lSD", ""),
                        "decision_boundary_utc_target": t0.get("lSD", ""),
                        "endpoint_boundary_utc_attacker": a1.get("lSD", ""),
                        "endpoint_boundary_utc_target": t1.get("lSD", ""),
                        "pit_context_flag": "TRUE" if pit_flag else "FALSE",
                        "race_control_lap_flag": "TRUE" if rc_flag else "FALSE",
                        "candidate_status": "UNRESOLVED_CANDIDATE",
                        "verified_overtake": "FALSE",
                        "disposition_reasons": ";".join(reasons),
                        "source_urls": "|".join(sorted({a0.get("source_url", ""), a1.get("source_url", ""), t0.get("source_url", ""), t1.get("source_url", "")})),
                        "source_sha256s": "|".join(sorted({a0.get("source_sha256", ""), a1.get("source_sha256", ""), t0.get("source_sha256", ""), t1.get("source_sha256", "")})),
                    })
    return candidates


def source_manifest_rows(inputs: Inputs, source_access: list[dict[str, Any]], complete_races: set[str]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in source_access:
        rid = row.get("race_id", "")
        if rid not in complete_races:
            continue
        key = (rid, row.get("source_kind", ""), row.get("url", ""))
        clean = {k: v for k, v in row.items() if k != "payload_bytes"}
        clean.update({
            "repository_or_provider": "TracingInsights" if "TracingInsights" in str(row.get("source_kind")) else "Jolpica/Ergast-compatible",
            "verification_status": "CACHE_SHA256_VERIFIED" if row.get("sha256_match", False) else "UNVERIFIED",
            "redistribution_status": "SOURCE_PAYLOAD_NOT_REPACKAGED",
        })
        unique[key] = clean
    fields = ["race_id", "source_kind", "driver", "url", "cache_path", "bytes", "sha256", "expected_sha256", "sha256_match", "status", "http_status", "retrieved_at_utc", "repository_or_provider", "verification_status", "redistribution_status"]
    return [dict((field, row.get(field, "")) for field in fields) for _, row in sorted(unique.items())]


def read_source_access_from_rows(inputs: Inputs, source_access: list[dict[str, Any]], races: list[Mapping[str, str]]) -> None:
    """A separate source pass guarantees all non-laptime context is hashed."""
    for split_row in races:
        rid = split_row["race_id"]
        for kind, rows in [("pit", inputs.pits_by_race.get(rid, [])), ("race_control", inputs.rc_by_race.get(rid, [])), ("weather", inputs.weather_by_race.get(rid, []))]:
            for url in sorted({r.get("source_url", "") for r in rows if r.get("source_url")}):
                src = inputs.guarded_source(url)
                source_access.append({"race_id": rid, "source_kind": kind, **src})


def build_package(output: Path, core_files: list[Path], row_counts: Mapping[str, Any]) -> dict[str, Any]:
    sums_path = output / "SHA256SUMS.txt"
    lines = [f"{sha256_file(path)}  {path.name}" for path in sorted(core_files, key=lambda p: p.name)]
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    package = output / "APEX-R_Full_Race_Collection.zip"
    if package.exists():
        package.unlink()
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(core_files + [sums_path], key=lambda p: p.name):
            zf.write(path, arcname=path.name)
    return {"path": str(package), "bytes": package.stat().st_size, "sha256": sha256_file(package), "compression": "ZIP_DEFLATED", "member_count": len(core_files) + 1, "row_counts": dict(row_counts)}


def verify_package(package: Path, sums_path: Path, expected_counts: Mapping[str, Any]) -> dict[str, Any]:
    verify_dir = Path(tempfile.mkdtemp(prefix="apex_full_race_verify_"))
    checks: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(package) as zf:
            bad = zf.testzip()
            checks.append({"test": "zip_crc", "passed": bad is None, "detail": "PASS" if bad is None else bad})
            names = zf.namelist()
            zf.extractall(verify_dir)
        sum_rows = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split("  ", 1)
                sum_rows[name] = digest
        all_hashes = True
        for name, digest in sorted(sum_rows.items()):
            actual = sha256_file(verify_dir / name)
            ok = actual == digest
            all_hashes = all_hashes and ok
            checks.append({"test": "extracted_sha256:" + name, "passed": ok, "detail": actual})
        checks.append({"test": "all_extracted_sha256", "passed": all_hashes, "detail": f"files={len(sum_rows)}"})
        count_ok = True
        for name, expected in expected_counts.items():
            path = verify_dir / name
            if not path.exists():
                count_ok = False
                checks.append({"test": "row_count:" + name, "passed": False, "detail": "missing"})
                continue
            if name.endswith(".csv.gz"):
                with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
                    actual = max(sum(1 for _ in f) - 1, 0)
            elif name.endswith(".csv"):
                with path.open(encoding="utf-8", newline="") as f:
                    actual = max(sum(1 for _ in f) - 1, 0)
            elif name.endswith(".json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                actual = len(payload) if isinstance(payload, list) else expected
            else:
                continue
            ok = int(actual) == int(expected)
            count_ok = count_ok and ok
            checks.append({"test": "row_count:" + name, "passed": ok, "detail": {"actual": actual, "expected": expected}})
        checks.append({"test": "all_row_counts", "passed": count_ok, "detail": f"files={len(expected_counts)}"})
        return {"verification_directory": str(verify_dir), "checks": checks, "all_passed": all(c["passed"] for c in checks), "extracted_bytes": sum(p.stat().st_size for p in verify_dir.rglob("*") if p.is_file())}
    finally:
        # The fresh extraction is intentionally temporary; package verification
        # records the path and cleanup result without retaining another copy.
        shutil.rmtree(verify_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-races", default="2018:Chinese Grand Prix:Race|2019:Australian Grand Prix:Race|2020:Austrian Grand Prix:Race")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = Inputs(workspace)
    before = inputs.hashes()
    split_rows = [r for r in inputs.split_rows if r.get("year") in {str(y) for y in YEARS} and r.get("session") == SESSION]
    if len(split_rows) != 70:
        raise RuntimeError(f"expected 70 approved race rows, found {len(split_rows)}")
    # Guard before reading any source payload.  The manifest's token is not
    # used to derive any race; it only rejects a protected identifier.
    for row in split_rows:
        if inputs.protected in row["race_id"]:
            raise RuntimeError("protected race was present in approved collection scope")
    requested_validation = [x for x in args.validation_races.split("|") if x]
    if len(requested_validation) != 3 or len(set(requested_validation)) != 3:
        raise ValueError("exactly three distinct validation races are required")
    by_id = {r["race_id"]: r for r in split_rows}
    if not set(requested_validation).issubset(by_id):
        raise KeyError("validation race is not in approved split manifest")
    source_access: list[dict[str, Any]] = []
    validation_races: list[dict[str, Any]] = []
    validation_lap_rows: list[dict[str, Any]] = []
    for rid in requested_validation:
        race, rows, meta = inspect_race(inputs, by_id[rid], source_access)
        validation_races.append(race)
        validation_lap_rows.extend(rows)
    validation_pass = all(r["complete_source_lap_context"] == "PASS" for r in validation_races)
    validation_result = {
        "stage": "THREE_RACE_VALIDATION_GATE",
        "races": requested_validation,
        "all_three_passed": validation_pass,
        "race_results": validation_races,
        "expansion_authorized": validation_pass,
    }
    json_dump(output / "validation_three_races.json", validation_result)
    write_csv(output / "validation_three_races.csv", validation_races, list(validation_races[0].keys()))
    if not validation_pass:
        raise RuntimeError("three-race validation gate failed; expansion was not run")

    all_races: list[dict[str, Any]] = []
    all_lap_rows: list[dict[str, Any]] = []
    race_meta: dict[str, Any] = {}
    for split_row in split_rows:
        race, rows, meta = inspect_race(inputs, split_row, source_access)
        all_races.append(race)
        all_lap_rows.extend(rows)
        race_meta[race["race_id"]] = meta
    complete_race_rows = [r for r in all_races if r["complete_source_lap_context"] == "PASS"]
    complete_ids = {r["race_id"] for r in complete_race_rows}
    included_laps = [r for r in all_lap_rows if r["race_id"] in complete_ids]
    verified_events = make_verified_events(inputs, complete_ids)
    candidates = make_candidates(complete_race_rows, included_laps, inputs)
    candidate_counts = Counter(c["race_id"] for c in candidates)
    for race in all_races:
        race["unresolved_position_swap_candidate_count"] = candidate_counts.get(race["race_id"], 0)

    context_fields = list(all_races[0].keys())
    write_csv(output / "race_context.csv", all_races, context_fields)
    # Include every source field present in the selected laptime payloads.
    lineage_fields = ["race_id", "split", "year", "event", "session", "driver", "driver_number", "full_name", "team", "source_url", "source_sha256", "source_cache_status", "record_index"]
    data_fields = sorted({k for row in included_laps for k in row if k not in lineage_fields})
    lap_fields = lineage_fields + data_fields
    lap_rows_count = write_csv_gz(output / "driver_laps.csv.gz", included_laps, lap_fields)
    event_fields = ["event_id", "race_id", "event_type", "event_status", "verified_overtake", "driver", "target_driver", "event_lap", "event_time", "duration_sec", "source_url", "source_sha256", "evidence"]
    verified_count = write_csv_gz(output / "verified_public_events.csv.gz", verified_events, event_fields)
    candidate_fields = list(candidates[0].keys()) if candidates else ["candidate_id", "race_id"]
    candidate_count = write_csv_gz(output / "unresolved_position_swap_candidates.csv.gz", candidates, candidate_fields)
    source_rows = source_manifest_rows(inputs, source_access, complete_ids)
    source_fields = ["race_id", "source_kind", "driver", "url", "cache_path", "bytes", "sha256", "expected_sha256", "sha256_match", "status", "http_status", "retrieved_at_utc", "repository_or_provider", "verification_status", "redistribution_status"]
    source_count = write_csv(output / "source_manifest.csv", source_rows, source_fields)
    json_dump(output / "race_context.json", all_races)
    json_dump(output / "verified_public_events.json", verified_events)
    json_dump(output / "unresolved_position_swap_candidates.json", candidates)
    json_dump(output / "source_manifest.json", source_rows)

    split_counts: dict[str, Any] = {}
    for split in sorted({r["split"] for r in all_races}):
        rs = [r for r in complete_race_rows if r["split"] == split]
        split_counts[split] = {"complete_races": len(rs), "lap_rows": sum(int(r["lap_rows"]) for r in rs), "unresolved_candidates": sum(candidate_counts.get(r["race_id"], 0) for r in rs), "verified_public_events": sum(1 for e in verified_events if next((x["split"] for x in all_races if x["race_id"] == e["race_id"]), "") == split)}
    metrics = {
        "version": VERSION,
        "generated_at_utc": now_utc(),
        "scope": {"years": list(YEARS), "session": SESSION, "approved_manifest_races": len(split_rows), "protected_exclusion_enforced": True, "network_requests_issued": 0, "source_mode": "REUSED_PINNED_PHASE2_PHASE3_CACHES"},
        "validation_gate": validation_result,
        "coverage": {"approved_races": len(all_races), "complete_source_lap_context_races": len(complete_race_rows), "excluded_incomplete_races": len(all_races) - len(complete_race_rows), "expected_driver_entries": sum(int(r["expected_driver_count"]) for r in all_races), "included_driver_lap_rows": lap_rows_count, "included_race_lap_rows": sum(int(r["lap_rows"]) for r in complete_race_rows), "field_min_lap": min((r["field_min_lap"] for r in complete_race_rows if r["field_min_lap"] is not None), default=None), "field_max_lap": max((r["field_max_lap"] for r in complete_race_rows if r["field_max_lap"] is not None), default=None)},
        "events": {"verified_public_event_rows": verified_count, "verified_overtake_events": 0, "unresolved_position_swap_candidates": candidate_count, "candidate_races": len({c["race_id"] for c in candidates})},
        "split_coverage": split_counts,
        "artifact_rows": {"race_context.csv": len(all_races), "race_context.json": len(all_races), "driver_laps.csv.gz": lap_rows_count, "verified_public_events.csv.gz": verified_count, "verified_public_events.json": len(verified_events), "unresolved_position_swap_candidates.csv.gz": candidate_count, "unresolved_position_swap_candidates.json": len(candidates), "source_manifest.csv": source_count, "source_manifest.json": len(source_rows), "validation_three_races.csv": len(validation_races), "validation_three_races.json": len(validation_races)},
        "input_hashes_before": before,
        "notes": ["Complete means every source-listed driver has a readable laptimes.json beginning at lap 1 with contiguous observed lap identifiers; it does not prove uninterrupted physical track coverage.", "Position reversals are unresolved candidates because the source has lap-boundary positions but no verified sub-lap overtake timestamp.", "Source payloads are not repackaged; only derived exports and retrieval/hash manifests are distributed.", "No labels, model features, model training, holdout scoring or decision-engine changes were performed."],
    }
    json_dump(output / "metrics.json", metrics)

    # Hash source inputs after collection before creating the package.
    after = inputs.hashes()
    metrics["input_hashes_after"] = after
    metrics["source_immutability_passed"] = before == after
    metrics["package"] = {"compression": "ZIP_DEFLATED", "verification_passed": "PENDING", "note": "ZIP byte/hash details are recorded outside the package to avoid a self-referential manifest."}
    json_dump(output / "metrics.json", metrics)
    core_files = [p for p in output.iterdir() if p.is_file() and p.name not in {"SHA256SUMS.txt", "APEX-R_Full_Race_Collection.zip", "package_verification.json", "collection_verification.json"}]
    package_info = build_package(output, core_files, metrics["artifact_rows"])
    package_check = verify_package(output / "APEX-R_Full_Race_Collection.zip", output / "SHA256SUMS.txt", metrics["artifact_rows"])
    # Freeze the metrics before the final package build; these fields do not
    # contain package bytes/hash, so the final package is self-consistent.
    metrics["package"] = {"compression": "ZIP_DEFLATED", "verification_passed": package_check["all_passed"], "member_count": package_info["member_count"]}
    json_dump(output / "metrics.json", metrics)
    core_files = [p for p in output.iterdir() if p.is_file() and p.name not in {"SHA256SUMS.txt", "APEX-R_Full_Race_Collection.zip", "package_verification.json", "collection_verification.json"}]
    package_info = build_package(output, core_files, metrics["artifact_rows"])
    package_check = verify_package(output / "APEX-R_Full_Race_Collection.zip", output / "SHA256SUMS.txt", metrics["artifact_rows"])
    package_info = {"path": str(output / "APEX-R_Full_Race_Collection.zip"), "bytes": (output / "APEX-R_Full_Race_Collection.zip").stat().st_size, "sha256": sha256_file(output / "APEX-R_Full_Race_Collection.zip"), "compression": "ZIP_DEFLATED", "member_count": len(core_files) + 1}
    json_dump(output / "package_verification.json", {"generated_at_utc": now_utc(), "package": package_info, "zip_crc_and_extracted_sha256_and_row_counts_passed": package_check["all_passed"], "checks": package_check["checks"], "fresh_extraction_cleaned": not Path(package_check["verification_directory"]).exists(), "package_verification_note": "This file is retained outside the distributable ZIP to avoid changing the package after verification."})
    print(json.dumps({"output": str(output), "races": len(all_races), "complete_races": len(complete_race_rows), "lap_rows": lap_rows_count, "verified_public_events": verified_count, "unresolved_candidates": candidate_count, "package_bytes": package_info["bytes"], "package_sha256": package_info["sha256"], "package_verified": package_check["all_passed"], "source_immutable": metrics["source_immutability_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
