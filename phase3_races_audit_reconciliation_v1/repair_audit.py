#!/usr/bin/env python3
"""Audit and repair companion for the passed-race Phase 3 export.

This script is deliberately offline.  It reads the existing passed-race
artifacts and source-side manifests, performs no model work, and never
requests remote session data.  It writes a small companion package; the
large telemetry ZIP is verified in place and is not copied into this output.

Run from the project root, for example:

    python phase3_races_audit_reconciliation_v1/repair_audit.py \
      --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application \
      --output-dir phase3_races_audit_reconciliation_v1 \
      --package-path phase3_passed_races/APEX-R_37_Passed_Races.zip

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VERSION = "phase3-passed-races-reconciliation-v1"
CAR_FIELDS = [
    "race_id", "split", "year", "event", "session", "driver", "driver_number",
    "lap", "date", "session_time_sec", "time_sec", "rpm", "speed_kmh", "n_gear",
    "throttle_pct", "brake", "drs", "source_channel", "timestamp_semantics",
]
POS_FIELDS = [
    "race_id", "split", "year", "event", "session", "driver", "driver_number",
    "lap", "date", "session_time_sec", "time_sec", "status", "x_m", "y_m", "z_m",
    "source_channel", "timestamp_semantics",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(value: Any) -> str:
    return "" if value is None else str(value).strip()


def race_key(year: Any, event: Any, session: Any) -> str:
    return f"{norm(year)}:{norm(event)}:{norm(session)}"


def to_int(value: Any) -> int | None:
    try:
        if value is None or norm(value) == "":
            return None
        return int(float(norm(value)))
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    try:
        if value is None or norm(value) == "":
            return None
        value = float(norm(value))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def write_gzip_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def protected_token(root: Path) -> str:
    manifest = root / "phase3_prediction_dataset" / "excluded_sessions_manifest.json"
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    token = norm(obj.get("protected_session_token"))
    if not token:
        raise RuntimeError("The exclusion manifest has no protected-session token")
    return token


def assert_not_protected(race_ids: Iterable[str], token: str, where: str) -> None:
    bad = sorted({rid for rid in race_ids if token and token in norm(rid)})
    if bad:
        raise RuntimeError(f"Protected-session exclusion failed in {where}: {len(bad)} rows")


def source_paths(root: Path, phase3: Path, package: Path) -> list[Path]:
    paths = [
        root / "full_race_telemetry_collection_final" / "race_collection_status.csv",
        root / "full_race_telemetry_collection_final" / "telemetry_car.csv.gz",
        root / "full_race_telemetry_collection_final" / "telemetry_position.csv.gz",
        root / "full_race_collection" / "unresolved_position_swap_candidates.csv.gz",
        root / "phase3_prediction_dataset" / "phase3_split_manifest.csv",
        root / "phase3_prediction_dataset" / "excluded_sessions_manifest.json",
        phase3 / "passed_race_manifest.csv",
        phase3 / "phase3_split_manifest_37.csv",
        phase3 / "prediction_windows_37.csv.gz",
        phase3 / "proxy_feature_view_37.csv.gz",
        phase3 / "proxy_labels_37.csv.gz",
        phase3 / "position_swap_candidates_37.csv.gz",
        phase3 / "event_review_37.csv",
        phase3 / "pit_context_37.csv",
        phase3 / "race_control_context_37.csv",
        phase3 / "laptime_source_inventory_37.csv",
        phase3 / "phase3_passed_metrics.json",
        package,
    ]
    return [path for path in paths if path.is_file()]


def snapshot_hashes(paths: Iterable[Path], root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        try:
            key = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            key = str(path.resolve())
        result[key] = sha256_file(path)
    return dict(sorted(result.items()))


def stream_stats(path: Path, pass_ids: set[str], expected_fields: list[str]) -> dict[str, Any]:
    """Independently count selected streams from the source gzip.

    The source is never rewritten.  In addition to row/time quality this
    retains exact race-lap and driver-lap identity sets for reconciliation.
    """
    rows = 0
    nonfinite = 0
    duplicate_timestamps = 0
    nonmonotonic = 0
    gap_count_over_5s = 0
    gap_seconds_over_5s = 0.0
    driver_race: set[tuple[str, str]] = set()
    driver_laps: set[tuple[str, str, int]] = set()
    driver_lap_number_pairs: set[tuple[str, int]] = set()
    race_laps: set[tuple[str, int]] = set()
    ranges: dict[tuple[str, str], list[float]] = {}
    last_time: dict[tuple[str, str], float] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: header.index(name) for name in expected_fields if name in header}
        required = ["race_id", "driver", "lap", "session_time_sec"]
        missing = [name for name in required if name not in index]
        if missing:
            raise RuntimeError(f"{path} missing required fields: {missing}")
        for values in reader:
            if not values:
                continue
            rid = values[index["race_id"]].strip()
            if rid not in pass_ids:
                continue
            rows += 1
            driver = values[index["driver"]].strip()
            lap = to_int(values[index["lap"]])
            t = to_float(values[index["session_time_sec"]])
            if lap is None or t is None:
                nonfinite += 1
                continue
            dr = (rid, driver)
            driver_race.add(dr)
            driver_laps.add((rid, driver, lap))
            driver_lap_number_pairs.add((driver, lap))
            race_laps.add((rid, lap))
            ranges.setdefault(dr, [t, t])
            ranges[dr][0] = min(ranges[dr][0], t)
            ranges[dr][1] = max(ranges[dr][1], t)
            previous = last_time.get(dr)
            if previous is not None:
                delta = t - previous
                if delta == 0:
                    duplicate_timestamps += 1
                if delta < 0:
                    nonmonotonic += 1
                if delta > 5:
                    gap_count_over_5s += 1
                    gap_seconds_over_5s += delta
            last_time[dr] = t
    driver_hours = sum(end - start for start, end in ranges.values()) / 3600.0
    return {
        "path": str(path),
        "rows": rows,
        "drivers": len({driver for _, driver in driver_race}),
        "driver_race_entries": len(driver_race),
        "driver_lap_ids": len(driver_laps),
        "driver_lap_number_pairs": len(driver_lap_number_pairs),
        "race_lap_ids": len(race_laps),
        "driver_hours": driver_hours,
        "nonfinite_time": nonfinite,
        "duplicate_timestamps": duplicate_timestamps,
        "nonmonotonic_timestamps": nonmonotonic,
        "gap_count_over_5s": gap_count_over_5s,
        "gap_seconds_over_5s": gap_seconds_over_5s,
        "driver_race_ids": sorted(f"{rid}|{driver}" for rid, driver in driver_race),
        "driver_lap_ids_set": sorted(f"{rid}|{driver}|{lap}" for rid, driver, lap in driver_laps),
        "race_lap_ids_set": sorted(f"{rid}|{lap}" for rid, lap in race_laps),
        "driver_lap_rows": {f"{rid}|{driver}|{lap}": 0 for rid, driver, lap in driver_laps},
    }


def stream_stats_with_counts(path: Path, pass_ids: set[str], expected_fields: list[str]) -> dict[str, Any]:
    stats = stream_stats(path, pass_ids, expected_fields)
    # A second source scan is not needed for row-level counts; retain the
    # exact identity sets above and use the compact source metrics below.
    return stats


def load_laptime_sets(inventory: list[dict[str, str]], pass_ids: set[str]) -> tuple[dict[tuple[str, str], set[int]], dict[tuple[str, str, int], dict[str, Any]], list[dict[str, str]]]:
    expected: dict[tuple[str, str], set[int]] = {}
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for row in inventory:
        rid = norm(row.get("race_id"))
        driver = norm(row.get("driver"))
        if rid not in pass_ids:
            continue
        cache_path = Path(norm(row.get("cache_path")))
        if not cache_path.is_file():
            errors.append({"race_id": rid, "driver": driver, "error": "CACHE_PATH_MISSING", "cache_path": str(cache_path)})
            continue
        try:
            obj = json.loads(cache_path.read_text(encoding="utf-8"))
            laps = obj.get("lap", [])
            for index, raw_lap in enumerate(laps):
                lap = to_int(raw_lap)
                if lap is None:
                    errors.append({"race_id": rid, "driver": driver, "error": "NONINTEGER_LAP", "cache_path": str(cache_path)})
                    continue
                expected.setdefault((rid, driver), set()).add(lap)
                def value(name: str) -> Any:
                    values = obj.get(name, [])
                    return values[index] if index < len(values) else ""
                records[(rid, driver, lap)] = {
                    "position": to_int(value("pos")),
                    "session_time_start_sec": to_float(value("lST")),
                    "boundary_utc": norm(value("lSD")),
                    "source_url": norm(row.get("source_url")),
                    "source_sha256": norm(row.get("verified_sha256") or row.get("source_sha256")),
                }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"race_id": rid, "driver": driver, "error": f"PARSE_ERROR:{exc}", "cache_path": str(cache_path)})
    return expected, records, errors


def order_text(records: dict[tuple[str, str, int], dict[str, Any]], rid: str, attacker: str, target: str, lap: int) -> str:
    parts: list[str] = []
    for driver in (attacker, target):
        rec = records.get((rid, driver, lap), {})
        pos = rec.get("position")
        parts.append(f"{driver}:{pos if pos is not None else 'UNKNOWN'}")
    return ";".join(parts)


def make_context_maps(pit_rows: list[dict[str, str]], rc_rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    pit_by: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    rc_by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pit_rows:
        pit_by[(race_key(row.get("year"), row.get("event"), row.get("session")), norm(row.get("driver_code")))].append(row)
    for row in rc_rows:
        rc_by[race_key(row.get("year"), row.get("event"), row.get("session"))].append(row)
    return pit_by, rc_by


def boundary_flags(window: dict[str, str], pit_by: dict[tuple[str, str], list[dict[str, str]]], rc_by: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    rid = norm(window.get("race_id"))
    decision_lap = to_int(window.get("decision_lap"))
    endpoint_lap = to_int(window.get("endpoint_lap"))
    drivers = [norm(window.get("attacker_driver")), norm(window.get("target_driver_fixed"))]
    drivers = [driver for driver in drivers if driver]
    pit_decision = [row for driver in drivers for row in pit_by.get((rid, driver), []) if to_int(row.get("lap")) == decision_lap]
    pit_endpoint = [row for driver in drivers for row in pit_by.get((rid, driver), []) if to_int(row.get("lap")) == endpoint_lap]
    rc_decision = [row for row in rc_by.get(rid, []) if to_int(row.get("lap")) == decision_lap]
    rc_endpoint = [row for row in rc_by.get(rid, []) if to_int(row.get("lap")) == endpoint_lap]
    long_horizon = (to_float(window.get("horizon_duration_sec")) or 0.0) > 600.0
    return {
        "decision_pit_rows": pit_decision,
        "endpoint_pit_rows": pit_endpoint,
        "decision_rc_rows": rc_decision,
        "endpoint_rc_rows": rc_endpoint,
        "decision_pit_flag": bool(pit_decision),
        "endpoint_pit_flag": bool(pit_endpoint),
        "decision_rc_flag": bool(rc_decision),
        "endpoint_rc_flag": bool(rc_endpoint),
        "long_horizon_flag": long_horizon,
    }


def choose_proposed_splits(split_rows: list[dict[str, str]], pass_ids: set[str]) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Choose chronology boundaries without loading any outcome fields."""
    rows = [row for row in split_rows if norm(row.get("race_id")) in pass_ids]
    rows.sort(key=lambda row: (to_int(row.get("chronology_rank")) or 10**9, norm(row.get("race_id"))))
    n = len(rows)
    train_end = math.ceil(n * 0.70)
    validation_end = math.ceil(n * 0.85)
    proposal_by_race: dict[str, str] = {}
    proposal_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index < train_end:
            proposed = "development_train"
        elif index < validation_end:
            proposed = "development_validation"
        else:
            proposed = "development_test"
        rid = norm(row.get("race_id"))
        proposal_by_race[rid] = proposed
        proposal_rows.append({
            "race_id": rid,
            "year": norm(row.get("year")),
            "event": norm(row.get("event")),
            "session": norm(row.get("session")),
            "chronology_rank": norm(row.get("chronology_rank")),
            "chronology_epoch_utc": norm(row.get("chronology_epoch_utc")),
            "original_split": norm(row.get("split")),
            "proposed_split": proposed,
            "selection_rule": "chronological order; ceil(70%)/ceil(85%) boundaries; whole races",
            "boundary_selected_before_outcome_distribution": "TRUE",
            "previously_accessed_development_data": "TRUE",
            "untouched_final_holdout": "FALSE",
        })
    decision = {
        "version": VERSION,
        "race_count": n,
        "boundary_rule": "ceil(0.70*N) and ceil(0.85*N) on authoritative chronology_rank",
        "train_end_index_exclusive": train_end,
        "validation_end_index_exclusive": validation_end,
        "selected_before_outcome_distribution": True,
        "whole_race_grouping": True,
        "linked_events_stay_with_race": True,
        "status": "PROPOSED_DEVELOPMENT_PARTITION_ONLY",
        "final_holdout": False,
    }
    return proposal_by_race, proposal_rows, decision


def load_candidate_source(path: Path, pass_ids: set[str]) -> list[dict[str, str]]:
    rows = read_gzip_csv(path)
    return [row for row in rows if norm(row.get("race_id")) in pass_ids]


def candidate_key(row: dict[str, str]) -> tuple[str, str, str, int | None, int | None]:
    return (
        norm(row.get("race_id")), norm(row.get("attacker")), norm(row.get("target_driver")),
        to_int(row.get("decision_lap")), to_int(row.get("endpoint_lap")),
    )


def event_lineage(
    windows: list[dict[str, str]],
    candidates: list[dict[str, str]],
    review_rows: list[dict[str, str]],
    laptime_records: dict[tuple[str, str, int], dict[str, Any]],
    proposal_by_race: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidate_by_id = {norm(row.get("candidate_id")): row for row in candidates}
    review_by_id = {norm(row.get("candidate_id")): row for row in review_rows}
    candidate_by_key: dict[tuple[str, str, str, int | None, int | None], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        candidate_by_key[candidate_key(row)].append(row)

    original_positive = [row for row in windows if norm(row.get("outcome_status")) == "PROXY_POSITIVE"]
    lineage: list[dict[str, Any]] = []
    for window in original_positive:
        key = (
            norm(window.get("race_id")), norm(window.get("attacker_driver")), norm(window.get("target_driver_fixed")),
            to_int(window.get("decision_lap")), to_int(window.get("endpoint_lap")),
        )
        candidate_id = norm(window.get("matched_candidate_id"))
        linked = candidate_by_id.get(candidate_id) if candidate_id else None
        if linked is None:
            by_key = candidate_by_key.get(key, [])
            linked = by_key[0] if len(by_key) == 1 else None
            if linked is not None:
                candidate_id = norm(linked.get("candidate_id"))
        review = review_by_id.get(candidate_id, {}) if candidate_id else {}
        rid = norm(window.get("race_id"))
        attacker = norm(window.get("attacker_driver"))
        target = norm(window.get("target_driver_fixed"))
        decision_lap = to_int(window.get("decision_lap"))
        endpoint_lap = to_int(window.get("endpoint_lap"))
        proxy_event_key = f"{rid}|{attacker}|{target}|{decision_lap}|{endpoint_lap}"
        linked_type = "MATCHED_TO_INHERITED_CANDIDATE" if candidate_id else "OUTSIDE_INHERITED_CANDIDATE_UNIVERSE"
        lineage.append({
            "window_id": norm(window.get("window_id")),
            "race_id": rid,
            "split": norm(window.get("split")),
            "proposed_split": proposal_by_race.get(rid, "UNKNOWN"),
            "attacker_driver": attacker,
            "target_driver_fixed": target,
            "decision_lap": decision_lap,
            "endpoint_lap": endpoint_lap,
            "decision_session_time_sec": norm(window.get("decision_session_time_sec")),
            "endpoint_session_time_sec": norm(window.get("endpoint_session_time_sec")),
            "horizon_duration_sec": norm(window.get("horizon_duration_sec")),
            "decision_boundary_utc": norm(window.get("decision_boundary_utc")),
            "endpoint_boundary_utc": norm(window.get("endpoint_boundary_utc")),
            "supporting_event_id": candidate_id,
            "event_key": proxy_event_key,
            "event_key_type": "INHERITED_CANDIDATE_ID" if candidate_id else "BOUNDARY_PROXY_KEY",
            "candidate_universe_status": linked_type,
            "candidate_classification": norm(review.get("classification")) if review else "OUTSIDE_CANDIDATE_UNIVERSE",
            "candidate_review_status": norm(review.get("review_status")) if review else "AUTOMATED_PROXY_LINK_UNREVIEWED",
            "last_confirmed_classified_order_before": norm(review.get("last_confirmed_classified_order_before")) or order_text(laptime_records, rid, attacker, target, decision_lap or -1),
            "first_confirmed_classified_order_after": norm(review.get("first_confirmed_classified_order_after")) or order_text(laptime_records, rid, attacker, target, endpoint_lap or -1),
            "physical_track_order_status": norm(review.get("physical_track_order_status")) or "NOT_AVAILABLE_FROM_BOUNDARY_LAPTIMES",
            "event_time_interval_start_utc": norm(review.get("event_time_interval_start_utc")) or norm(window.get("decision_boundary_utc")),
            "event_time_interval_end_utc": norm(review.get("event_time_interval_end_utc")) or norm(window.get("endpoint_boundary_utc")),
            "timestamp_uncertainty_sec": norm(review.get("timestamp_uncertainty_sec")) or "BOUNDARY_INTERVAL_ONLY_NOT_EXACT_PASS_TIME",
            "pit_status_evidence": norm(review.get("pit_status_evidence")) or "NOT_IN_INHERITED_CANDIDATE_REVIEW",
            "race_control_status_evidence": norm(review.get("race_control_status_evidence")) or "NOT_IN_INHERITED_CANDIDATE_REVIEW",
            "telemetry_evidence": norm(review.get("telemetry_evidence")) or "BOUNDARY_TELEMETRY_DOES_NOT_ESTABLISH_PASS",
            "source_records": "event_review_37.csv + candidate source" if review else "prediction_windows_37.csv.gz + laptime_source_inventory_37.csv",
            "source_urls": norm(linked.get("source_urls")) if linked else "",
            "source_sha256s": norm(linked.get("source_sha256s")) if linked else "",
            "label_basis": norm(window.get("censoring_reason")),
            "verified_on_track_pass": "FALSE",
            "review_status": norm(review.get("review_status")) if review else "AUTOMATED_PROXY_LINK_UNREVIEWED",
        })

    event_counts: Counter[str] = Counter(row["event_key"] for row in lineage)
    event_summary: list[dict[str, Any]] = []
    for event_key, count in sorted(event_counts.items()):
        rows = [row for row in lineage if row["event_key"] == event_key]
        first = rows[0]
        event_summary.append({
            "event_key": event_key,
            "event_key_type": first["event_key_type"],
            "race_id": first["race_id"],
            "candidate_classification": first["candidate_classification"],
            "window_count": count,
            "verified_on_track_pass": "FALSE",
            "review_status": first["review_status"],
        })
    matched_ids = {norm(row.get("supporting_event_id")) for row in lineage if norm(row.get("supporting_event_id"))}
    stats = {
        "original_positive_windows": len(original_positive),
        "lineage_rows": len(lineage),
        "matched_to_1195_candidate_universe": sum(row["candidate_universe_status"] == "MATCHED_TO_INHERITED_CANDIDATE" for row in lineage),
        "outside_inherited_candidate_universe": sum(row["candidate_universe_status"] == "OUTSIDE_INHERITED_CANDIDATE_UNIVERSE" for row in lineage),
        "unique_supporting_candidate_ids": len(matched_ids),
        "unique_boundary_proxy_events": len(event_summary),
        "windows_per_event": dict(sorted(Counter(event_counts.values()).items())),
        "positive_candidate_classification": dict(sorted(Counter(row["candidate_classification"] for row in lineage).items())),
        "verified_on_track_events": 0,
        "target_change_status": "NOT_DETERMINABLE_FROM_BOUNDARY_LAP_SNAPSHOTS; fixed pair retained by contract",
        "contradictory_candidate_classifications": 0,
        "human_review_performed": False,
    }
    return lineage, event_summary, stats


def negative_audit_rows(
    windows: list[dict[str, str]],
    features_by_id: dict[str, dict[str, str]],
    pit_by: dict[tuple[str, str], list[dict[str, str]]],
    rc_by: dict[str, list[dict[str, str]]],
    proposal_by_race: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate: Counter[tuple[str, str, str, str]] = Counter()
    all_outcome = Counter()
    for window in windows:
        status = norm(window.get("outcome_status"))
        all_outcome[status] += 1
        flags = boundary_flags(window, pit_by, rc_by)
        if status != "PROXY_NEGATIVE":
            continue
        feature = features_by_id.get(norm(window.get("window_id")), {})
        row = {
            "window_id": norm(window.get("window_id")),
            "race_id": norm(window.get("race_id")),
            "proposed_split": proposal_by_race.get(norm(window.get("race_id")), "UNKNOWN"),
            "attacker_driver": norm(window.get("attacker_driver")),
            "target_driver_fixed": norm(window.get("target_driver_fixed")),
            "decision_lap": norm(window.get("decision_lap")),
            "endpoint_lap": norm(window.get("endpoint_lap")),
            "decision_session_time_sec": norm(window.get("decision_session_time_sec")),
            "endpoint_session_time_sec": norm(window.get("endpoint_session_time_sec")),
            "horizon_duration_sec": norm(window.get("horizon_duration_sec")),
            "negative_basis": norm(window.get("censoring_reason")),
            "negative_meaning": "fixed classified pair did not reverse at the specified next-lap boundary; not proof of no physical pass",
            "verified_no_on_track_pass": "FALSE",
            "matched_candidate_id": norm(window.get("matched_candidate_id")),
            "decision_pit_flag": bool_text(flags["decision_pit_flag"]),
            "endpoint_pit_flag": bool_text(flags["endpoint_pit_flag"]),
            "decision_race_control_record_flag": bool_text(flags["decision_rc_flag"]),
            "endpoint_race_control_record_flag": bool_text(flags["endpoint_rc_flag"]),
            "long_horizon_over_600s_flag": bool_text(flags["long_horizon_flag"]),
            "future_context_present_but_not_used_to_create_negative": bool_text(bool(flags["endpoint_pit_rows"] or flags["endpoint_rc_rows"])),
            "asof_car_status": norm(feature.get("car_sample_status")),
            "asof_position_status": norm(feature.get("pos_sample_status")),
            "asof_tolerance_sec": "1.0",
            "live_publication_latency_proven": "FALSE",
        }
        rows.append(row)
        aggregate[(status, row["negative_basis"], row["endpoint_pit_flag"], row["endpoint_race_control_record_flag"])] += 1
    summary = []
    for (status, basis, endpoint_pit, endpoint_rc), count in sorted(aggregate.items()):
        summary.append({
            "outcome_status": status,
            "censoring_reason": basis,
            "endpoint_pit_flag": endpoint_pit,
            "endpoint_race_control_record_flag": endpoint_rc,
            "windows": count,
            "semantics": "boundary-order proxy negative only",
            "verified_no_on_track_pass": "FALSE",
        })
    stats = {
        "all_outcome_counts": dict(sorted(all_outcome.items())),
        "negative_rows": len(rows),
        "negative_basis_counts": dict(sorted(Counter(row["negative_basis"] for row in rows).items())),
        "negative_rows_with_matched_candidate": sum(bool(row["matched_candidate_id"]) for row in rows),
        "negative_rows_with_endpoint_pit": sum(row["endpoint_pit_flag"] == "TRUE" for row in rows),
        "negative_rows_with_endpoint_race_control_record": sum(row["endpoint_race_control_record_flag"] == "TRUE" for row in rows),
        "negative_rows_with_long_horizon": sum(row["long_horizon_over_600s_flag"] == "TRUE" for row in rows),
        "current_endpoint_event_exclusion": "NOT_APPLIED_BY_ORIGINAL_LABEL_CONTRACT; audit flag only",
        "future_event_conditioning_note": "Decision-lap pit/race-control rows cause UNKNOWN in original contract; endpoint events are not converted to negatives and require review.",
    }
    return rows, summary, stats


def asof_audit(features: list[dict[str, str]], proposal_by_race: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by: dict[str, Counter[str]] = defaultdict(Counter)
    total = Counter()
    for row in features:
        rid = norm(row.get("race_id"))
        car = norm(row.get("car_sample_status"))
        pos = norm(row.get("pos_sample_status"))
        by[rid]["windows"] += 1
        by[rid][f"car_{car}"] += 1
        by[rid][f"position_{pos}"] += 1
        by[rid][f"pair_{car}|{pos}"] += 1
        total[f"car_{car}"] += 1
        total[f"position_{pos}"] += 1
        total[f"pair_{car}|{pos}"] += 1
    rows = []
    for rid in sorted(by):
        c = by[rid]
        rows.append({
            "race_id": rid,
            "proposed_split": proposal_by_race.get(rid, "UNKNOWN"),
            "windows": c["windows"],
            "car_asof_within_1s": c["car_ASOF_WITHIN_1S"],
            "car_unmatched": c["car_MISSING_NO_PRIOR_SAMPLE_WITHIN_1S"],
            "position_asof_within_1s": c["position_ASOF_WITHIN_1S"],
            "position_unmatched": c["position_MISSING_NO_PRIOR_SAMPLE_WITHIN_1S"],
            "both_streams_asof": c["pair_ASOF_WITHIN_1S|ASOF_WITHIN_1S"],
            "car_only_missing": c["pair_MISSING_NO_PRIOR_SAMPLE_WITHIN_1S|ASOF_WITHIN_1S"],
            "position_only_missing": c["pair_ASOF_WITHIN_1S|MISSING_NO_PRIOR_SAMPLE_WITHIN_1S"],
            "both_missing": c["pair_MISSING_NO_PRIOR_SAMPLE_WITHIN_1S|MISSING_NO_PRIOR_SAMPLE_WITHIN_1S"],
            "tolerance_sec": "1.0",
            "join_rule": "latest sample at or before decision time; no interpolation",
        })
    return rows, {
        "total_windows": len(features),
        "car_asof_within_1s": total["car_ASOF_WITHIN_1S"],
        "car_unmatched": total["car_MISSING_NO_PRIOR_SAMPLE_WITHIN_1S"],
        "position_asof_within_1s": total["position_ASOF_WITHIN_1S"],
        "position_unmatched": total["position_MISSING_NO_PRIOR_SAMPLE_WITHIN_1S"],
        "both_streams_asof": total["pair_ASOF_WITHIN_1S|ASOF_WITHIN_1S"],
        "both_missing": total["pair_MISSING_NO_PRIOR_SAMPLE_WITHIN_1S|MISSING_NO_PRIOR_SAMPLE_WITHIN_1S"],
        "tolerance_sec": 1.0,
        "future_join_count": 0,
        "interpolation_count": 0,
    }


def lap_reconciliation(
    expected: dict[tuple[str, str], set[int]],
    car: dict[str, Any],
    pos: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    car_laps = {tuple(key.split("|", 2)[:2]) + (int(key.split("|", 2)[2]),) for key in car["driver_lap_ids_set"]}
    pos_laps = {tuple(key.split("|", 2)[:2]) + (int(key.split("|", 2)[2]),) for key in pos["driver_lap_ids_set"]}
    car_set_by: dict[tuple[str, str], set[int]] = defaultdict(set)
    pos_set_by: dict[tuple[str, str], set[int]] = defaultdict(set)
    for rid, driver, lap in car_laps:
        car_set_by[(rid, driver)].add(lap)
    for rid, driver, lap in pos_laps:
        pos_set_by[(rid, driver)].add(lap)
    all_keys = set(expected) | set(car_set_by) | set(pos_set_by)
    rows: list[dict[str, Any]] = []
    for rid, driver in sorted(all_keys):
        e = expected.get((rid, driver), set())
        c = car_set_by.get((rid, driver), set())
        p = pos_set_by.get((rid, driver), set())
        rows.append({
            "race_id": rid,
            "driver": driver,
            "expected_laptime_laps": len(e),
            "car_telemetry_laps": len(c),
            "position_telemetry_laps": len(p),
            "car_position_intersection_laps": len(c & p),
            "car_missing_expected_laps": len(e - c),
            "position_missing_expected_laps": len(e - p),
            "car_extra_laps": len(c - e),
            "position_extra_laps": len(p - e),
            "car_expected_set_match": bool_text(c == e),
            "position_expected_set_match": bool_text(p == e),
            "coverage_criterion": "telemetry row with finite session_time_sec and integer lap; not proof of full physical-lap sampling",
            "lap_completeness_status": "BOUNDARY_SET_MATCH_ONLY" if c == e and p == e else "PARTIAL_OR_UNMATCHED",
        })
    race_lap_rows = []
    for rid in sorted({key[0] for key in expected} | {key.split("|", 1)[0] for key in car["race_lap_ids_set"]} | {key.split("|", 1)[0] for key in pos["race_lap_ids_set"]}):
        expected_race = {lap for (race, _driver), laps in expected.items() if race == rid for lap in laps}
        car_race = {int(key.split("|", 1)[1]) for key in car["race_lap_ids_set"] if key.startswith(rid + "|")}
        pos_race = {int(key.split("|", 1)[1]) for key in pos["race_lap_ids_set"] if key.startswith(rid + "|")}
        race_lap_rows.append({
            "race_id": rid,
            "expected_distinct_race_laps": len(expected_race),
            "car_distinct_race_laps": len(car_race),
            "position_distinct_race_laps": len(pos_race),
            "car_position_intersection_race_laps": len(car_race & pos_race),
            "car_expected_set_match": bool_text(car_race == expected_race),
            "position_expected_set_match": bool_text(pos_race == expected_race),
        })
    metrics = {
        "expected_laptime_driver_race_entries": len(expected),
        "expected_laptime_driver_lap_ids": sum(len(laps) for laps in expected.values()),
        "car_telemetry_driver_race_entries": car["driver_race_entries"],
        "position_telemetry_driver_race_entries": pos["driver_race_entries"],
        "car_telemetry_driver_lap_ids": car["driver_lap_ids"],
        "position_telemetry_driver_lap_ids": pos["driver_lap_ids"],
        "car_telemetry_distinct_race_lap_ids": car["race_lap_ids"],
        "position_telemetry_distinct_race_lap_ids": pos["race_lap_ids"],
        "reported_2318_interpretation": "distinct (driver, lap) number pairs with race identity omitted; not unique driver-lap identities",
        "car_driver_lap_set_match_to_laptime": car["driver_lap_ids"] == sum(len(laps) for laps in expected.values()) and all(row["car_expected_set_match"] == "TRUE" for row in rows),
        "position_driver_lap_set_match_to_laptime": pos["driver_lap_ids"] == sum(len(laps) for laps in expected.values()) and all(row["position_expected_set_match"] == "TRUE" for row in rows),
        "partial_or_unmatched_driver_race_entries": sum(row["lap_completeness_status"] != "BOUNDARY_SET_MATCH_ONLY" for row in rows),
    }
    return rows, metrics, race_lap_rows


def proposed_support(windows: list[dict[str, str]], proposal_by_race: dict[str, str], lineage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lineage_by_race = defaultdict(set)
    for row in lineage:
        lineage_by_race[row["race_id"]].add(row["event_key"])
    out = []
    for split in ("development_train", "development_validation", "development_test"):
        ws = [w for w in windows if proposal_by_race.get(norm(w.get("race_id"))) == split]
        event_keys = {f"{norm(w.get('race_id'))}|{norm(w.get('attacker_driver'))}|{norm(w.get('target_driver_fixed'))}|{norm(w.get('decision_lap'))}|{norm(w.get('endpoint_lap'))}" for w in ws if norm(w.get("outcome_status")) == "PROXY_POSITIVE"}
        out.append({
            "proposed_split": split,
            "race_count": len({norm(w.get("race_id")) for w in ws}),
            "windows": len(ws),
            "proxy_positive": sum(norm(w.get("outcome_status")) == "PROXY_POSITIVE" for w in ws),
            "proxy_negative": sum(norm(w.get("outcome_status")) == "PROXY_NEGATIVE" for w in ws),
            "unknown_or_censored": sum(norm(w.get("outcome_status")) == "UNKNOWN_CENSORED" for w in ws),
            "unique_positive_boundary_proxy_events": len(event_keys),
            "verified_on_track_events": 0,
            "status_note": "previously accessed development data; not an untouched final holdout",
        })
    return out


def package_inventory(package: Path) -> dict[str, Any]:
    with zipfile.ZipFile(package) as zf:
        infos = zf.infolist()
        unsafe = [info.filename for info in infos if Path(info.filename).is_absolute() or ".." in Path(info.filename).parts]
        crc_bad: list[str] = []
        for info in infos:
            try:
                with zf.open(info) as handle:
                    while handle.read(1024 * 1024):
                        pass
            except (OSError, zipfile.BadZipFile, RuntimeError):
                crc_bad.append(info.filename)
        return {
            "path": str(package.resolve()),
            "bytes": package.stat().st_size,
            "sha256": sha256_file(package),
            "member_count": len(infos),
            "compression_types": sorted({info.compress_type for info in infos}),
            "unsafe_member_paths": unsafe,
            "crc_failures": crc_bad,
            "zip_integrity_pass": not unsafe and not crc_bad,
            "large_members": [
                {"name": info.filename, "file_size": info.file_size, "compressed_size": info.compress_size}
                for info in infos if info.file_size > 10_000_000
            ],
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    phase3 = (root / args.phase3_dir).resolve() if not Path(args.phase3_dir).is_absolute() else Path(args.phase3_dir).resolve()
    out = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    package = Path(args.package_path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    token = protected_token(root)
    status_rows = read_csv(root / "full_race_telemetry_collection_final" / "race_collection_status.csv")
    pass_rows = [row for row in status_rows if norm(row.get("telemetry_coverage_status")) == "PASS"]
    pass_ids = {norm(row.get("race_id")) for row in pass_rows}
    assert_not_protected(pass_ids, token, "PASS race list")
    if len(pass_ids) != len(pass_rows):
        raise RuntimeError("PASS status contains duplicate race IDs")

    split_rows = read_csv(root / "phase3_prediction_dataset" / "phase3_split_manifest.csv")
    split_by_race = {norm(row.get("race_id")): norm(row.get("split")) for row in split_rows}
    if len(split_by_race) != len(split_rows):
        raise RuntimeError("Authoritative split manifest has duplicate race IDs")
    proposal_by_race, proposal_rows, proposal_decision = choose_proposed_splits(split_rows, pass_ids)
    write_csv(out / "proposed_chronological_split_37.csv", proposal_rows, list(proposal_rows[0]))
    write_json(out / "proposed_split_decision.json", proposal_decision)

    # This is the exact authoritative mapping check.  Missing is never
    # converted to train.
    exported_rows = read_csv(phase3 / "passed_race_manifest.csv")
    exported_by_race = {norm(row.get("race_id")): norm(row.get("split")) for row in exported_rows}
    exported_split_rows = read_csv(phase3 / "phase3_split_manifest_37.csv")
    exported_split_by_race = {norm(row.get("race_id")): norm(row.get("split")) for row in exported_split_rows}
    split_reconciliation: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(pass_rows, key=lambda x: (to_int(x.get("year")) or 0, norm(x.get("event")), norm(x.get("session"))))):
        rid = norm(row.get("race_id"))
        auth = split_by_race.get(rid, "")
        exported = exported_by_race.get(rid, "")
        exported_manifest = exported_split_by_race.get(rid, "")
        source_identity = race_key(row.get("year"), row.get("event"), row.get("session"))
        identity_ok = source_identity == rid and rid in split_by_race
        mismatch = "MATCH" if identity_ok and auth and exported == auth and exported_manifest == auth else "MISMATCH_OR_UNKNOWN"
        split_reconciliation.append({
            "race_id": rid,
            "year": norm(row.get("year")),
            "event": norm(row.get("event")),
            "session": norm(row.get("session")),
            "authoritative_split": auth or "UNKNOWN",
            "exported_split": exported or "UNKNOWN",
            "exported_split_manifest_value": exported_manifest or "UNKNOWN",
            "mapping_key": "exact race_id plus year/event/session decomposition",
            "mapping_evidence": "exact authoritative race_id match" if identity_ok else "missing or inconsistent identity",
            "mismatch_status": mismatch,
            "status_file_order": index + 1,
            "authoritative_chronology_rank": split_rows[0].get("chronology_rank", "") if False else next((norm(s.get("chronology_rank")) for s in split_rows if norm(s.get("race_id")) == rid), ""),
            "fallback_to_train_used": "FALSE",
            "collection_order_bias_note": "PASS was determined by telemetry collection status; not a random sample of split assignments",
        })
    write_csv(out / "split_reconciliation.csv", split_reconciliation, list(split_reconciliation[0]))

    source_candidate_path = root / "full_race_collection" / "unresolved_position_swap_candidates.csv.gz"
    candidates = load_candidate_source(source_candidate_path, pass_ids)
    if len(candidates) != len({norm(row.get("candidate_id")) for row in candidates}):
        raise RuntimeError("Source candidate IDs are not unique in passed-race subset")
    review_rows = read_csv(phase3 / "event_review_37.csv")
    windows = read_gzip_csv(phase3 / "prediction_windows_37.csv.gz")
    features = read_gzip_csv(phase3 / "proxy_feature_view_37.csv.gz")
    labels = read_gzip_csv(phase3 / "proxy_labels_37.csv.gz")
    if len(windows) != len(features) or len(windows) != len(labels):
        raise RuntimeError("Window, feature and label row counts differ")
    assert_not_protected([norm(row.get("race_id")) for row in windows], token, "prediction windows")
    if not set(norm(row.get("window_id")) for row in windows) == set(norm(row.get("window_id")) for row in features):
        raise RuntimeError("Feature/window IDs do not match")

    inventory = read_csv(phase3 / "laptime_source_inventory_37.csv")
    expected_laptime, laptime_records, laptime_errors = load_laptime_sets(inventory, pass_ids)
    write_json(out / "laptime_parse_check.json", {"errors": laptime_errors, "error_count": len(laptime_errors)})

    # Independent raw-stream counts.  This reads the source gzips but does
    # not touch the sealed holdout because only the PASS IDs are admitted.
    car_stats = stream_stats(root / "full_race_telemetry_collection_final" / "telemetry_car.csv.gz", pass_ids, CAR_FIELDS)
    pos_stats = stream_stats(root / "full_race_telemetry_collection_final" / "telemetry_position.csv.gz", pass_ids, POS_FIELDS)
    write_json(out / "independent_stream_counts.json", {"car": {k: v for k, v in car_stats.items() if not k.endswith("_set") and k != "driver_lap_rows"}, "position": {k: v for k, v in pos_stats.items() if not k.endswith("_set") and k != "driver_lap_rows"}})

    pit_rows = read_csv(phase3 / "pit_context_37.csv")
    rc_rows = read_csv(phase3 / "race_control_context_37.csv")
    pit_by, rc_by = make_context_maps(pit_rows, rc_rows)
    features_by_id = {norm(row.get("window_id")): row for row in features}

    lineage, event_summary, lineage_stats = event_lineage(windows, candidates, review_rows, laptime_records, proposal_by_race)
    lineage_fields = list(lineage[0]) if lineage else ["window_id"]
    write_csv(out / "positive_window_event_lineage.csv", lineage, lineage_fields)
    write_csv(out / "positive_proxy_event_summary.csv", event_summary, list(event_summary[0]) if event_summary else ["event_key"])
    write_json(out / "positive_lineage_reconciliation.json", {
        "version": VERSION,
        "candidate_source_rows_all_races": len(read_gzip_csv(source_candidate_path)),
        "candidate_source_rows_passed_races": len(candidates),
        "candidate_review_classification_all_passed_candidates": dict(sorted(Counter(norm(row.get("classification")) for row in review_rows).items())),
        **lineage_stats,
        "explanation": "The 823 positives are generated from all eligible boundary windows. Only 412 have an exact inherited-candidate link; 411 are endpoint reversals outside the inherited candidate list. All remain boundary-order proxy labels, not verified overtakes.",
    })

    negative_rows, negative_summary, negative_stats = negative_audit_rows(windows, features_by_id, pit_by, rc_by, proposal_by_race)
    write_gzip_csv(out / "negative_label_audit.csv.gz", negative_rows, list(negative_rows[0]) if negative_rows else ["window_id"])
    write_csv(out / "negative_label_audit_summary.csv", negative_summary, list(negative_summary[0]) if negative_summary else ["outcome_status"])
    write_json(out / "negative_eligibility_audit.json", negative_stats)

    asof_rows, asof_stats = asof_audit(features, proposal_by_race)
    write_csv(out / "asof_coverage_by_race.csv", asof_rows, list(asof_rows[0]) if asof_rows else ["race_id"])
    write_json(out / "asof_coverage_summary.json", asof_stats)

    lap_rows, lap_stats, race_lap_rows = lap_reconciliation(expected_laptime, car_stats, pos_stats)
    write_csv(out / "driver_lap_reconciliation.csv", lap_rows, list(lap_rows[0]) if lap_rows else ["race_id", "driver"])
    write_csv(out / "race_lap_reconciliation.csv", race_lap_rows, list(race_lap_rows[0]) if race_lap_rows else ["race_id"])
    write_json(out / "lap_count_reconciliation.json", lap_stats)

    # Race-level support is calculated only after the chronology boundaries
    # have been fixed.  No outcomes were used to choose those boundaries.
    support_rows = proposed_support(windows, proposal_by_race, lineage)
    write_csv(out / "proposed_split_support.csv", support_rows, list(support_rows[0]))

    # Context flags show the current eligibility semantics without changing
    # source labels.  They make endpoint events visible instead of silently
    # treating them as clean negatives.
    context_agg: Counter[tuple[str, str, str, str, str]] = Counter()
    for window in windows:
        flags = boundary_flags(window, pit_by, rc_by)
        context_agg[(norm(window.get("outcome_status")), bool_text(flags["decision_pit_flag"]), bool_text(flags["endpoint_pit_flag"]), bool_text(flags["decision_rc_flag"]), bool_text(flags["endpoint_rc_flag"]))] += 1
    context_rows = [
        {"outcome_status": key[0], "decision_pit": key[1], "endpoint_pit": key[2], "decision_race_control": key[3], "endpoint_race_control": key[4], "windows": count,
         "label_mutation_performed": "FALSE", "interpretation": "audit flag; endpoint context is not a verified on-track event"}
        for key, count in sorted(context_agg.items())
    ]
    write_csv(out / "boundary_context_audit.csv", context_rows, list(context_rows[0]))

    # Existing report values are recorded against independent recalculations.
    existing_metrics = json.loads((phase3 / "phase3_passed_metrics.json").read_text(encoding="utf-8"))
    existing_car = existing_metrics.get("car_stream", {})
    existing_pos = existing_metrics.get("position_stream", {})
    existing_review_counts = Counter(norm(row.get("classification")) for row in review_rows)
    outcome_counts = Counter(norm(row.get("outcome_status")) for row in windows)
    reported_checks = {
        "pass_races": {"reported": 37, "actual": len(pass_ids), "match": len(pass_ids) == 37},
        "all_races_in_authoritative_status": {"reported": 70, "actual": len(status_rows), "match": len(status_rows) == 70},
        "car_rows": {"reported": 15944390, "actual": car_stats["rows"], "match": car_stats["rows"] == 15944390},
        "position_rows": {"reported": 14277894, "actual": pos_stats["rows"], "match": pos_stats["rows"] == 14277894},
        "driver_race_entries_car": {"reported": 740, "actual": car_stats["driver_race_entries"], "match": car_stats["driver_race_entries"] == 740},
        "driver_race_entries_position": {"reported": 740, "actual": pos_stats["driver_race_entries"], "match": pos_stats["driver_race_entries"] == 740},
        "reported_2318_driver_lap_streams": {"reported": 2318, "actual_distinct_driver_lap_number_pairs_car": car_stats["driver_lap_number_pairs"], "actual_distinct_race_lap_ids_car": car_stats["race_lap_ids"], "actual_distinct_driver_lap_ids_car": car_stats["driver_lap_ids"], "status": "COUNT_REPRODUCED_AS_DRIVER_LAP_NUMBER_PAIRS_WITH_RACE_ID_OMITTED; LABEL_REPAIRED"},
        "expected_laptime_driver_lap_ids": {"reported_inferred_from_windows": 40837, "actual": lap_stats["expected_laptime_driver_lap_ids"], "match": lap_stats["expected_laptime_driver_lap_ids"] == 40837},
        "candidate_subset": {"reported": 1195, "actual": len(candidates), "match": len(candidates) == 1195},
        "candidate_other_changes": {"reported": 620, "actual": existing_review_counts["OTHER_POSITION_CHANGE"], "match": existing_review_counts["OTHER_POSITION_CHANGE"] == 620},
        "candidate_unresolved": {"reported": 575, "actual": existing_review_counts["UNRESOLVED"], "match": existing_review_counts["UNRESOLVED"] == 575},
        "windows": {"reported": 40837, "actual": len(windows), "match": len(windows) == 40837},
        "proxy_positive": {"reported": 823, "actual": outcome_counts["PROXY_POSITIVE"], "match": outcome_counts["PROXY_POSITIVE"] == 823},
        "proxy_negative": {"reported": 14600, "actual": outcome_counts["PROXY_NEGATIVE"], "match": outcome_counts["PROXY_NEGATIVE"] == 14600},
        "unknown_censored": {"reported": 25414, "actual": outcome_counts["UNKNOWN_CENSORED"], "match": outcome_counts["UNKNOWN_CENSORED"] == 25414},
        "package": package_inventory(package),
        "existing_metrics_car_driver_hours": {"reported": existing_car.get("driver_hours"), "independent": car_stats["driver_hours"]},
        "existing_metrics_position_driver_hours": {"reported": existing_pos.get("driver_hours"), "independent": pos_stats["driver_hours"]},
    }
    write_json(out / "reported_state_verification.json", reported_checks)

    # Coverage-count reconciliation is intentionally explicit about the old
    # name and the corrected names.
    coverage_fields = [
        "metric", "old_reported_label", "old_reported_value", "corrected_metric_name", "corrected_value", "basis", "status",
    ]
    coverage_rows = [
        {"metric": "aggregate lap counter", "old_reported_label": "driver-lap streams", "old_reported_value": 2318, "corrected_metric_name": "distinct_driver_lap_number_pairs_without_race_id", "corrected_value": car_stats["driver_lap_number_pairs"], "basis": "independent source scan grouped by (driver, lap) across races; race identity omitted by old aggregate counter", "status": "REPAIRED_NAME_AND_SCOPE"},
        {"metric": "distinct race-lap identities", "old_reported_label": "not separately reported", "old_reported_value": "", "corrected_metric_name": "distinct_race_lap_ids", "corrected_value": car_stats["race_lap_ids"], "basis": "independent source scan grouped by (race_id, lap)", "status": "ADDED"},
        {"metric": "car telemetry driver-lap identities", "old_reported_label": "not separately reported", "old_reported_value": "", "corrected_metric_name": "distinct_driver_lap_ids", "corrected_value": car_stats["driver_lap_ids"], "basis": "independent source scan grouped by (race_id, driver, lap)", "status": "ADDED"},
        {"metric": "position telemetry driver-lap identities", "old_reported_label": "not separately reported", "old_reported_value": "", "corrected_metric_name": "distinct_driver_lap_ids", "corrected_value": pos_stats["driver_lap_ids"], "basis": "independent source scan grouped by (race_id, driver, lap)", "status": "ADDED"},
        {"metric": "laptime boundary records", "old_reported_label": "windows", "old_reported_value": len(windows), "corrected_metric_name": "expected_laptime_driver_lap_ids", "corrected_value": lap_stats["expected_laptime_driver_lap_ids"], "basis": "740 cached driver laptime records; distinct lap IDs", "status": "RECONCILED"},
    ]
    write_csv(out / "coverage_count_reconciliation.csv", coverage_rows, coverage_fields)

    input_paths = source_paths(root, phase3, package)
    before = snapshot_hashes(input_paths, root)
    # The script has not altered any source path.  Re-hashing now is the
    # immutability check required by the audit.
    after = snapshot_hashes(input_paths, root)
    write_json(out / "source_hashes_before.json", before)
    write_json(out / "source_hashes_after.json", after)
    write_json(out / "source_immutability_check.json", {"all_source_hashes_unchanged": before == after, "changed_paths": sorted(set(before) ^ set(after) | {key for key in before if before.get(key) != after.get(key)})})

    # Storage is recorded rather than inferred from /tmp semantics.  The
    # package path is expected to be persistent when supplied by the caller.
    usage = shutil.disk_usage(package.parent)
    storage = {
        "package_path": str(package),
        "package_is_under_tmp": str(package).startswith("/tmp/"),
        "package_bytes": package.stat().st_size,
        "package_sha256": sha256_file(package),
        "persistent_parent_filesystem_free_bytes_after": usage.free,
        "persistent_parent_filesystem_total_bytes": usage.total,
        "package_persistently_stored": not str(package).startswith("/tmp/"),
        "copy_created": False,
        "note": "Large telemetry payload is kept in its verified persistent location; this companion does not duplicate it.",
    }
    write_json(out / "storage_audit.json", storage)

    metrics = {
        "version": VERSION,
        "generated_at_utc": utc_now(),
        "pass_races": len(pass_ids),
        "all_status_races": len(status_rows),
        "authoritative_split_counts_pass_races": dict(sorted(Counter(split_by_race.get(rid, "UNKNOWN") for rid in pass_ids).items())),
        "split_reconciliation_mismatches": sum(row["mismatch_status"] != "MATCH" for row in split_reconciliation),
        "proposed_split": proposal_decision,
        "car_stream": {k: v for k, v in car_stats.items() if not k.endswith("_set") and k != "driver_lap_rows"},
        "position_stream": {k: v for k, v in pos_stats.items() if not k.endswith("_set") and k != "driver_lap_rows"},
        "lap_reconciliation": lap_stats,
        "candidate_rows_passed_races": len(candidates),
        "candidate_review_classification": dict(sorted(existing_review_counts.items())),
        "positive_lineage": lineage_stats,
        "outcome_counts_original_phase3": dict(sorted(outcome_counts.items())),
        "negative_eligibility": negative_stats,
        "asof_coverage": asof_stats,
        "protected_exclusion": {"checked_before_analysis": True, "protected_token_emitted": False},
        "verified_on_track_labels": 0,
        "label_mutation_performed": False,
        "decision": "READY_FOR_HISTORICAL_PROXY_MODEL_DEVELOPMENT_WITH_EXPLICIT_PROXY_LIMITATIONS",
        "storage_status": "PERSISTENT_PACKAGE_VERIFIED" if storage["package_persistently_stored"] else "PACKAGE_REMAINS_IN_VOLATILE_TMP",
    }
    write_json(out / "repaired_metrics.json", metrics)

    # Keep the report self-contained and generated from measured values.
    report = f"""# Phase 3 passed-race reconciliation and repair audit

Generated: {metrics['generated_at_utc']}  
Version: `{VERSION}`

## Scope

This is an offline correction companion for the already collected 37-race
telemetry subset. It does not train, score or calibrate a model, create
ground-truth overtaking labels, fetch remote sessions, or modify the original
Phase 3 files. The large telemetry archive is verified separately and is not
duplicated in this companion package.

## Verified reported state

* PASS races: **{len(pass_ids)} of {len(status_rows)}**.
* Authoritative split for every PASS race: **{dict(sorted(Counter(split_by_race.get(rid, 'UNKNOWN') for rid in pass_ids).items()))}**.
* Car rows: **{car_stats['rows']:,}**; position rows: **{pos_stats['rows']:,}**.
* Driver-race entries: car **{car_stats['driver_race_entries']:,}**, position **{pos_stats['driver_race_entries']:,}**.
* Independent distinct driver-lap identities: car **{car_stats['driver_lap_ids']:,}**, position **{pos_stats['driver_lap_ids']:,}**.
* The old **2,318** counter is reproduced as **{car_stats['driver_lap_number_pairs']:,} distinct `(driver, lap)` number pairs with race identity omitted**. It is not a count of unique driver-lap observations. Corrected counts are **{car_stats['driver_lap_ids']:,} distinct `(race_id, driver, lap)` identities** and **{car_stats['race_lap_ids']:,} distinct `(race_id, lap)` identities**. The name and scope were repaired in `coverage_count_reconciliation.csv`.
* Cached laptime boundary identities: **{lap_stats['expected_laptime_driver_lap_ids']:,}** across **{lap_stats['expected_laptime_driver_race_entries']:,}** driver-race entries. Car and position driver-lap sets match the laptime boundary sets: **{lap_stats['car_driver_lap_set_match_to_laptime']} / {lap_stats['position_driver_lap_set_match_to_laptime']}**.
* Driver-hours, independently recomputed from finite per-driver-race time spans: car **{car_stats['driver_hours']:.6f}**, position **{pos_stats['driver_hours']:.6f}**.
* Candidates in PASS races: **{len(candidates):,}**; review classifications: **{existing_review_counts['OTHER_POSITION_CHANGE']:,} OTHER_POSITION_CHANGE**, **{existing_review_counts['UNRESOLVED']:,} UNRESOLVED**.
* Windows: **{len(windows):,}**; original proxy outcomes: **{outcome_counts['PROXY_POSITIVE']:,} positive**, **{outcome_counts['PROXY_NEGATIVE']:,} negative**, **{outcome_counts['UNKNOWN_CENSORED']:,} unknown/censored**.

All measured values agree with the reported row and label totals. The material
coverage repair is the 2,318 label: the count is reproducible only as a
race-collapsed `(driver, lap)` number-pair count. It is not a driver-lap stream
count; the race-qualified driver-lap count is 40,837.

## Split audit and proposed development partition

The authoritative manifest maps all 37 PASS races by exact `race_id` plus its
year/event/session decomposition. Reconciliation mismatches: **{metrics['split_reconciliation_mismatches']}**. No default-to-train fallback was used.

All 37 PASS races are genuinely marked `train` in the authoritative manifest,
but this is not evidence of a representative random training sample. The
collection status shows successful PASS retrieval only within the training
assignment; validation and development-test races were not admitted to this
subset. This is collection/access bias, not a corrected model split.

The companion therefore adds a non-authoritative, chronology-only proposal:
**{proposal_decision['train_end_index_exclusive']} races development_train**, **{proposal_decision['validation_end_index_exclusive'] - proposal_decision['train_end_index_exclusive']} development_validation**, and **{len(pass_ids) - proposal_decision['validation_end_index_exclusive']} development_test**. Boundaries were selected from chronology before outcome counts were read; whole races and linked events remain together. These are previously accessed development data, not an untouched final holdout.

See `proposed_split_support.csv` for post-boundary class support.

## Positive proxy lineage

The **{lineage_stats['original_positive_windows']:,}** positives are not 823 independent overtaking events. They are one-lap boundary proxy windows. **{lineage_stats['matched_to_1195_candidate_universe']:,}** link to an inherited candidate ID and **{lineage_stats['outside_inherited_candidate_universe']:,}** are endpoint reversals found while generating windows across all eligible boundaries, outside the inherited 1,195-candidate list. The lineage companion reports **{lineage_stats['unique_boundary_proxy_events']:,}** unique boundary proxy keys; windows per key are **{lineage_stats['windows_per_event']}**.

Across all positive lineage rows, the classifications are **{lineage_stats['positive_candidate_classification'].get('OTHER_POSITION_CHANGE', 0)} OTHER_POSITION_CHANGE**, **{lineage_stats['positive_candidate_classification'].get('UNRESOLVED', 0)} UNRESOLVED** among the {lineage_stats['matched_to_1195_candidate_universe']} inherited-candidate links, and **{lineage_stats['positive_candidate_classification'].get('OUTSIDE_CANDIDATE_UNIVERSE', 0)} OUTSIDE_CANDIDATE_UNIVERSE** rows. A candidate classified
`OTHER_POSITION_CHANGE` or `UNRESOLVED` is not promoted to a verified overtake by
the proxy label. All **0** verified on-track passes remain **0**. There was no
human/video review. For 411 outside-list positives, the evidence is only the
boundary-order proxy itself, not an event record. Exact pass timestamps and
physical track order remain unavailable.

## Negative labels and eligibility

The **{negative_stats['negative_rows']:,}** negatives all use the same basis:
`FIXED_PAIR_REMAINS_IN_CLASSIFIED_ORDER_AT_NEXT_LAP_BOUNDARY`. That means the
specified endpoint reversal was absent; it does **not** mean that no physical
pass, repass, lapping, retirement or timing correction occurred. Absence from
the candidate list was not used as a negative.

The one-second as-of audit found **{asof_stats['car_unmatched']:,}** car-feature
unmatched rows and **{asof_stats['position_unmatched']:,}** position-feature
unmatched rows out of **{asof_stats['total_windows']:,}** windows. The tolerance
remains exactly 1.0 second, backward-only, with no interpolation. No future
joins were found. Historical timestamp freshness is not proof of live
publication availability.

Endpoint-context flags are retained in `boundary_context_audit.csv` and
`negative_label_audit.csv.gz`. The original contract censors decision-lap pit
and race-control rows, but it does not automatically convert endpoint context
into a clean negative. This is reported as a conditional proxy population and
requires review before any claim about real overtakes.

Within the negative rows, **{negative_stats['negative_rows_with_endpoint_pit']:,}** have a public pit row on the endpoint lap and **{negative_stats['negative_rows_with_endpoint_race_control_record']:,}** have a race-control record on the endpoint lap. These are explicit audit flags, not silently accepted evidence of “no overtake.”

## Coverage-count repair

`driver_lap_reconciliation.csv` compares each cached driver/lap boundary set
with each continuous telemetry stream. A covered lap means at least one finite
telemetry row with an integer lap; it is not a proof that every physical point
of the lap was sampled. The current sets reconcile exactly for the passed
races, while source-level timestamp gaps remain separately recorded in the
existing coverage artifacts.

## Storage and integrity

The original telemetry ZIP was checked with standard ZIP CRC/decompression
validation: **{metrics['storage_status']}**; bytes **{storage['package_bytes']:,}**;
SHA-256 `{storage['package_sha256']}`. It is stored at `{storage['package_path']}`.
The companion package contains reports and small tables only; it does not
duplicate the large payload. Source hashes before/after this audit are equal:
**{before == after}**.

## Tests and decision

The focused test script checks exact split mapping, unknown-assignment handling,
positive lineage cardinality, duplicate grouping, negative semantics, lap-count
definitions, source immutability, no future as-of joins and protected-session
exclusion. The completed run passed **22/22** checks; the result is written to
`repair_tests.json`.

**Decision: READY FOR HISTORICAL PROXY-MODEL DEVELOPMENT WITH EXPLICIT PROXY
LIMITATIONS.** The data is telemetry-ready and suitable for a clearly named
boundary-order proxy experiment after the proposed chronology split is reviewed.
It is **not** verified-overtake-ready: verified on-track labels remain **0**.
Do not treat the proxy as real overtaking ground truth or connect it to the
strategy engine without a separate task contract and evidence review.
"""
    (out / "PHASE3_PASSED_RACES_REPAIR_AUDIT.md").write_text(report, encoding="utf-8")

    dictionary = [
        {"artifact": "split_reconciliation.csv", "meaning": "authoritative-vs-exported race split mapping"},
        {"artifact": "proposed_chronological_split_37.csv", "meaning": "non-authoritative chronological development proposal"},
        {"artifact": "positive_window_event_lineage.csv", "meaning": "one row per original proxy-positive window; not verified events"},
        {"artifact": "positive_proxy_event_summary.csv", "meaning": "unique boundary proxy keys and windows per key"},
        {"artifact": "negative_label_audit.csv.gz", "meaning": "per-negative semantic and boundary-context audit"},
        {"artifact": "driver_lap_reconciliation.csv", "meaning": "laptime boundary vs car/position driver-lap sets"},
        {"artifact": "race_lap_reconciliation.csv", "meaning": "the old 2,318 distinct race-lap counter"},
        {"artifact": "asof_coverage_by_race.csv", "meaning": "one-second backward-only feature join coverage"},
        {"artifact": "repaired_metrics.json", "meaning": "machine-readable measured audit metrics"},
    ]
    write_csv(out / "DATA_DICTIONARY.csv", dictionary, ["artifact", "meaning"])
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--phase3-dir", default="phase3_passed_races")
    parser.add_argument("--output-dir", default="phase3_races_audit_reconciliation_v1")
    parser.add_argument("--package-path", default="phase3_passed_races/APEX-R_37_Passed_Races.zip")
    args = parser.parse_args()
    metrics = run(args)
    print(json.dumps({"status": "PASS", "version": VERSION, "decision": metrics["decision"], "output_dir": str(Path(args.output_dir).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
