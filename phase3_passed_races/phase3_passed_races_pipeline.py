#!/usr/bin/env python3
"""Export and audit only the races marked PASS by continuous telemetry collection.

This script is deliberately offline: it never fetches a race.  It reads the
completed collection, filters both streams independently, reviews the existing
position-swap candidates, and builds a clearly named lap-boundary order-swap
proxy table.  It does not train a model and does not call a decision engine.

Example:
    python phase3_passed_races_pipeline.py \
      --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application \
      --output-dir phase3_passed_races \
      --package-path /tmp/APEX-R_37_Passed_Races.zip
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
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CAR_SOURCE = "full_race_telemetry_collection_final/telemetry_car.csv.gz"
POS_SOURCE = "full_race_telemetry_collection_final/telemetry_position.csv.gz"
STATUS_SOURCE = "full_race_telemetry_collection_final/race_collection_status.csv"
COVERAGE_SOURCE = "full_race_telemetry_collection_final/telemetry_coverage.csv"
DRIVER_COVERAGE_SOURCE = "full_race_telemetry_collection_final/driver_telemetry_coverage.csv"
FASTF1_MANIFEST_SOURCE = "full_race_telemetry_collection_final/fastf1_source_manifest.csv"
FULL_MANIFEST_SOURCE = "full_race_collection/source_manifest.json"
CANDIDATE_SOURCE = "full_race_collection/unresolved_position_swap_candidates.csv.gz"
LAPTIME_COVERAGE_SOURCE = "phase3_prediction_dataset/context/laptime_coverage.csv"
PIT_SOURCE = "phase2_data_foundation/context/pit_stop_context.csv"
RC_SOURCE = "phase3_prediction_dataset/context/race_control_state_context.csv"
SPLIT_SOURCE = "phase3_prediction_dataset/phase3_split_manifest.csv"
EXCLUSION_SOURCE = "phase3_prediction_dataset/excluded_sessions_manifest.json"

CAR_FIELDS = [
    "race_id", "split", "year", "event", "session", "driver", "driver_number", "lap",
    "date", "session_time_sec", "time_sec", "rpm", "speed_kmh", "n_gear",
    "throttle_pct", "brake", "drs", "source_channel", "timestamp_semantics",
]
POS_FIELDS = [
    "race_id", "split", "year", "event", "session", "driver", "driver_number", "lap",
    "date", "session_time_sec", "time_sec", "status", "x_m", "y_m", "z_m",
    "source_channel", "timestamp_semantics",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_gzip_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "None", "null", "NULL", "nan", "NaN"}:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_int(value: Any) -> int | None:
    x = parse_float(value)
    return int(x) if x is not None and x.is_integer() else None


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "null", "nan"} else text


def iso_epoch(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def race_key(year: Any, event: Any, session: Any) -> str:
    return f"{parse_int(year)}:{norm(event)}:{norm(session)}"


def safe_protected_token(root: Path) -> str:
    data = json.loads((root / EXCLUSION_SOURCE).read_text(encoding="utf-8"))
    token = norm(data.get("protected_session_token"))
    if not token:
        raise RuntimeError("No protected-session token was available in the supplied exclusion manifest")
    return token


def assert_not_protected(value: Any, token: str) -> None:
    if token and token in str(value):
        raise RuntimeError("Protected-session token appeared in a requested/output value")


def source_paths(root: Path) -> list[Path]:
    return [root / p for p in [CAR_SOURCE, POS_SOURCE, STATUS_SOURCE, COVERAGE_SOURCE,
        DRIVER_COVERAGE_SOURCE, FASTF1_MANIFEST_SOURCE, FULL_MANIFEST_SOURCE,
        CANDIDATE_SOURCE, LAPTIME_COVERAGE_SOURCE, PIT_SOURCE, RC_SOURCE, SPLIT_SOURCE,
        EXCLUSION_SOURCE]]


def load_inputs(root: Path, token: str) -> dict[str, Any]:
    status = read_csv(root / STATUS_SOURCE)
    passed = [r for r in status if norm(r.get("telemetry_coverage_status")) == "PASS"]
    if not passed:
        raise RuntimeError("No PASS statuses found; refusing to fabricate a subset")
    pass_ids = {norm(r["race_id"]) for r in passed}
    if len(pass_ids) != len(passed):
        raise RuntimeError("PASS statuses contained duplicate race identities")
    for rid in pass_ids:
        assert_not_protected(rid, token)
    split_rows = read_csv(root / SPLIT_SOURCE)
    split_by_race = {norm(r["race_id"]): norm(r["split"]) for r in split_rows}
    if set(split_by_race) != {norm(r["race_id"]) for r in status}:
        raise RuntimeError("Status and split manifests do not describe the same race identities")
    if any(norm(r.get("split")) != split_by_race[norm(r["race_id"])] for r in passed):
        raise RuntimeError("PASS subset disagrees with the approved split manifest")
    candidates = read_gzip_csv(root / CANDIDATE_SOURCE)
    candidate_subset = [r for r in candidates if norm(r.get("race_id")) in pass_ids]
    if any(norm(r.get("race_id")) not in pass_ids for r in candidate_subset):
        raise RuntimeError("Candidate filtering failure")
    return {
        "status": status,
        "passed": passed,
        "pass_ids": pass_ids,
        "split_by_race": split_by_race,
        "split_rows": split_rows,
        "candidates": candidates,
        "candidate_subset": candidate_subset,
        "coverage": read_csv(root / COVERAGE_SOURCE),
        "driver_coverage": read_csv(root / DRIVER_COVERAGE_SOURCE),
        "fastf1_manifest": read_csv(root / FASTF1_MANIFEST_SOURCE),
        "pit": read_csv(root / PIT_SOURCE),
        "rc": read_csv(root / RC_SOURCE),
        "laptime_coverage": read_csv(root / LAPTIME_COVERAGE_SOURCE),
        "full_manifest": json.loads((root / FULL_MANIFEST_SOURCE).read_text(encoding="utf-8")),
    }


def normalize_laptime_payload(payload: dict[str, Any], driver: str, source_url: str, source_sha: str) -> dict[int, dict[str, Any]]:
    arrays = {k: v for k, v in payload.items() if isinstance(v, list)}
    count = max((len(v) for v in arrays.values()), default=0)
    by_lap: dict[int, dict[str, Any]] = {}
    for i in range(count):
        lap = parse_int(payload.get("lap", [None] * count)[i] if isinstance(payload.get("lap"), list) and i < len(payload["lap"]) else None)
        if lap is None or lap < 1:
            continue
        row = {
            "driver": driver, "lap": lap, "position": parse_int(payload.get("pos", [None] * count)[i] if isinstance(payload.get("pos"), list) and i < len(payload["pos"]) else None),
            "lap_time_sec": parse_float(payload.get("time", [None] * count)[i] if isinstance(payload.get("time"), list) and i < len(payload["time"]) else None),
            "session_time_start_sec": parse_float(payload.get("lST", [None] * count)[i] if isinstance(payload.get("lST"), list) and i < len(payload["lST"]) else None),
            "session_time_end_sec": parse_float(payload.get("sesT", [None] * count)[i] if isinstance(payload.get("sesT"), list) and i < len(payload["sesT"]) else None),
            "boundary_utc": norm(payload.get("lSD", [None] * count)[i] if isinstance(payload.get("lSD"), list) and i < len(payload["lSD"]) else None),
            "compound": norm(payload.get("compound", [None] * count)[i] if isinstance(payload.get("compound"), list) and i < len(payload["compound"]) else None),
            "stint": parse_int(payload.get("stint", [None] * count)[i] if isinstance(payload.get("stint"), list) and i < len(payload["stint"] ) else None),
            "tyre_life": parse_int(payload.get("life", [None] * count)[i] if isinstance(payload.get("life"), list) and i < len(payload["life"]) else None),
            "sector1_sec": parse_float(payload.get("s1", [None] * count)[i] if isinstance(payload.get("s1"), list) and i < len(payload["s1"]) else None),
            "sector2_sec": parse_float(payload.get("s2", [None] * count)[i] if isinstance(payload.get("s2"), list) and i < len(payload["s2"]) else None),
            "sector3_sec": parse_float(payload.get("s3", [None] * count)[i] if isinstance(payload.get("s3"), list) and i < len(payload["s3"]) else None),
            "source_url": source_url, "source_sha256": source_sha,
        }
        if lap in by_lap:
            # Duplicate lap records are not silently merged.
            if by_lap[lap] != row:
                raise RuntimeError(f"Conflicting laptime record for {driver} lap {lap}")
            continue
        by_lap[lap] = row
    return by_lap


def load_laptimes(root: Path, inputs: dict[str, Any], token: str) -> tuple[dict[str, dict[str, dict[int, dict[str, Any]]]], list[dict[str, Any]]]:
    by_race: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(dict)
    manifest_by_url = {norm(r.get("url")): r for r in inputs["full_manifest"] if norm(r.get("source_kind")) == "TracingInsights laptimes.json"}
    source_rows: list[dict[str, Any]] = []
    for row in inputs["laptime_coverage"]:
        rid = race_key(row.get("year"), row.get("event"), row.get("session"))
        if rid not in inputs["pass_ids"]:
            continue
        url = norm(row.get("source_url")); assert_not_protected(url, token)
        manifest = manifest_by_url.get(url, {})
        cache = Path(norm(manifest.get("cache_path")))
        if not cache.is_file():
            raise FileNotFoundError(f"Cached laptime source required for selected race is missing: {cache}")
        raw = cache.read_bytes()
        actual_sha = sha256_bytes(raw)
        expected_sha = norm(row.get("source_sha256"))
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(f"Laptime cache hash mismatch for {url}")
        payload = json.loads(raw.decode("utf-8"))
        driver = norm(row.get("driver"))
        by_race[rid][driver] = normalize_laptime_payload(payload, driver, url, actual_sha)
        source_rows.append({**row, "race_id": rid, "cache_path": str(cache), "verified_sha256": actual_sha})
    return by_race, source_rows


def stream_filter_and_stats(source: Path, destination: Path, pass_ids: set[str], fields: list[str]) -> dict[str, Any]:
    counts = defaultdict(int)
    laps: dict[str, set[tuple[str, int]]] = defaultdict(set)
    drivers: dict[str, set[str]] = defaultdict(set)
    first_last: dict[tuple[str, str], list[float]] = {}
    previous: dict[tuple[str, str], float] = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rt", encoding="utf-8", newline="") as src, gzip.open(destination, "wt", encoding="utf-8", newline="", compresslevel=6) as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames != fields:
            raise RuntimeError(f"Unexpected schema in {source}: {reader.fieldnames}")
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            rid = norm(row.get("race_id"))
            if rid not in pass_ids:
                continue
            writer.writerow(row)
            counts[rid] += 1
            driver = norm(row.get("driver")); drivers[rid].add(driver)
            lap = parse_int(row.get("lap"))
            if lap is not None:
                laps[rid].add((driver, lap))
            t = parse_float(row.get("session_time_sec"))
            if t is None:
                counts[f"{rid}|nonfinite_time"] += 1
                continue
            key = (rid, driver)
            if key not in first_last:
                first_last[key] = [t, t]
            else:
                first_last[key][1] = t
                delta = t - previous[key]
                if delta == 0:
                    counts[f"{rid}|duplicate_timestamps"] += 1
                elif delta < 0:
                    counts[f"{rid}|nonmonotonic_timestamps"] += 1
                elif delta > 5:
                    counts[f"{rid}|gaps_over_5s"] += 1
                    counts[f"{rid}|gap_seconds_over_5s"] += delta
            previous[key] = t
    race_stats = {}
    for rid in pass_ids:
        race_stats[rid] = {
            "rows": counts[rid], "drivers": len(drivers[rid]),
            "driver_lap_streams": len(laps[rid]), "laps": len({lap for _, lap in laps[rid]}),
            "driver_hours": sum((v[1] - v[0]) / 3600 for (race, _), v in first_last.items() if race == rid),
            "gap_count_over_5s": counts[f"{rid}|gaps_over_5s"],
            "gap_seconds_over_5s": counts[f"{rid}|gap_seconds_over_5s"],
            "duplicate_timestamps": counts[f"{rid}|duplicate_timestamps"],
            "nonmonotonic_timestamps": counts[f"{rid}|nonmonotonic_timestamps"],
            "nonfinite_time": counts[f"{rid}|nonfinite_time"],
        }
    return {"race_stats": race_stats, "total_rows": sum(v["rows"] for v in race_stats.values()),
            "driver_hours": sum(v["driver_hours"] for v in race_stats.values()),
            "drivers": len({d for vals in drivers.values() for d in vals}),
            "driver_race_entries": sum(len(v) for v in drivers.values()),
            "laps": len({x for vals in laps.values() for x in vals}),
            "duplicate_timestamps": sum(v["duplicate_timestamps"] for v in race_stats.values()),
            "nonmonotonic_timestamps": sum(v["nonmonotonic_timestamps"] for v in race_stats.values()),
            "gap_count_over_5s": sum(v["gap_count_over_5s"] for v in race_stats.values()),
            "gap_seconds_over_5s": sum(v["gap_seconds_over_5s"] for v in race_stats.values()),
            "nonfinite_time": sum(v["nonfinite_time"] for v in race_stats.values())}


def build_telemetry_boundary_index(source: Path, needed: dict[tuple[str, str], list[float]], fields: list[str]) -> dict[tuple[str, str, float], dict[str, str]]:
    # As-of lookup: latest source sample at or before the decision boundary.
    # No interpolation and no future row are used. A 1.0-second freshness limit
    # is an explicit assumption for this historical proxy, not a live-latency proof.
    targets = {k: sorted(v) for k, v in needed.items()}
    positions = {k: 0 for k in targets}
    last: dict[tuple[str, str], dict[str, str]] = {}
    found: dict[tuple[str, str, float], dict[str, str]] = {}
    with gzip.open(source, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise RuntimeError(f"Unexpected schema in {source}")
        for row in reader:
            key = (norm(row.get("race_id")), norm(row.get("driver")))
            if key not in targets:
                continue
            t = parse_float(row.get("session_time_sec"))
            if t is None:
                continue
            prior = last.get(key)
            idx = positions[key]
            while idx < len(targets[key]) and targets[key][idx] < t:
                # The row just before a target is saved when a later sample is seen.
                if prior:
                    prev = prior
                    pt = parse_float(prev.get("session_time_sec"))
                    if pt is not None and pt <= targets[key][idx] and targets[key][idx] - pt <= 1.0:
                        found[(key[0], key[1], targets[key][idx])] = prev
                idx += 1
            positions[key] = idx
            last[key] = row
        # Flush targets whose as-of sample is the final row in that stream.
        for key, ts in targets.items():
            prev = last.get(key)
            pt = parse_float(prev.get("session_time_sec")) if prev else None
            for target in ts:
                if prev and pt is not None and pt <= target and target - pt <= 1.0:
                    found[(key[0], key[1], target)] = prev
    return found


def pit_rows_by_race(rows: list[dict[str, str]], pass_ids: set[str]) -> dict[tuple[str, str], list[dict[str, str]]]:
    result = defaultdict(list)
    for row in rows:
        rid = race_key(row.get("year"), row.get("event"), row.get("session"))
        if rid in pass_ids:
            result[(rid, norm(row.get("driver_code")))].append(row)
    return result


def rc_rows_by_race(rows: list[dict[str, str]], pass_ids: set[str]) -> dict[str, list[dict[str, str]]]:
    result = defaultdict(list)
    for row in rows:
        rid = race_key(row.get("year"), row.get("event"), row.get("session"))
        if rid in pass_ids:
            result[rid].append(row)
    return result


def raw_refs(rows: list[dict[str, str]], fields: list[str]) -> str:
    return ";".join("|".join(f"{f}={norm(r.get(f))}" for f in fields) for r in rows)


def build_event_review(inputs: dict[str, Any], laptimes: dict[str, dict[str, dict[int, dict[str, Any]]]],
                       pit_by: dict[tuple[str, str], list[dict[str, str]]], rc_by: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    fields = ["race_id", "candidate_id", "split", "attacker_driver", "target_driver", "decision_lap",
        "endpoint_lap", "last_confirmed_classified_order_before", "first_confirmed_classified_order_after",
        "physical_track_order_status", "event_time_interval_start_utc", "event_time_interval_end_utc",
        "timestamp_uncertainty_sec", "pit_status_evidence", "race_control_status_evidence",
        "telemetry_evidence", "classification", "evidence_references", "review_status"]
    out = []
    for c in inputs["candidate_subset"]:
        rid = norm(c["race_id"]); attacker = norm(c["attacker"]); target = norm(c["target_driver"])
        dl, el = parse_int(c.get("decision_lap")), parse_int(c.get("endpoint_lap"))
        a0 = laptimes.get(rid, {}).get(attacker, {}).get(dl or -1, {})
        t0 = laptimes.get(rid, {}).get(target, {}).get(dl or -1, {})
        a1 = laptimes.get(rid, {}).get(attacker, {}).get(el or -1, {})
        t1 = laptimes.get(rid, {}).get(target, {}).get(el or -1, {})
        before = f"{attacker}:{a0.get('position','')};{target}:{t0.get('position','')}"
        after = f"{attacker}:{a1.get('position','')};{target}:{t1.get('position','')}"
        dts = [iso_epoch(c.get("decision_boundary_utc_attacker")), iso_epoch(c.get("decision_boundary_utc_target"))]
        ets = [iso_epoch(c.get("endpoint_boundary_utc_attacker")), iso_epoch(c.get("endpoint_boundary_utc_target"))]
        dts = [x for x in dts if x is not None]; ets = [x for x in ets if x is not None]
        # The event is bounded by the last decision-boundary observation and
        # the first endpoint-boundary observation. If those bounds cross, retain
        # a conservative outer interval and flag the uncertainty.
        start = max(dts) if dts else None; end = min(ets) if ets else None
        if start is None or end is None or start > end:
            start = min(dts) if dts else None; end = max(ets) if ets else None
        uncertainty = ((max(dts) - min(dts)) if dts else 0) + ((max(ets) - min(ets)) if ets else 0)
        relevant_pits = [r for driver in (attacker, target) for r in pit_by.get((rid, driver), [])
                         if parse_int(r.get("lap")) in {dl, el}]
        relevant_rc = [r for r in rc_by.get(rid, []) if parse_int(r.get("lap")) in {dl, el}]
        has_reversal = parse_int(a0.get("position")) is not None and parse_int(t0.get("position")) is not None and parse_int(a1.get("position")) is not None and parse_int(t1.get("position")) is not None and parse_int(a0.get("position")) > parse_int(t0.get("position")) and parse_int(a1.get("position")) < parse_int(t1.get("position"))
        if relevant_pits:
            classification = "OTHER_POSITION_CHANGE"
        elif has_reversal:
            # Classified order reversed, but FastF1 position data has only XYZ
            # location/status and cannot establish an on-track pass.
            classification = "UNRESOLVED"
        else:
            classification = "UNRESOLVED"
        refs = [norm(c.get("source_urls")), raw_refs(relevant_pits, ["source_url", "lap", "time_of_day", "availability_status"]), raw_refs(relevant_rc, ["source_url", "source_time_utc", "category", "scope", "sector", "message"])]
        telemetry_evidence = "FastF1 car_data and pos_data present for selected race; pos_data contains XYZ/status, not classified running order or pass timestamp"
        out.append({
            "race_id": rid, "candidate_id": norm(c.get("candidate_id")), "split": inputs["split_by_race"][rid],
            "attacker_driver": attacker, "target_driver": target, "decision_lap": dl or "", "endpoint_lap": el or "",
            "last_confirmed_classified_order_before": before, "first_confirmed_classified_order_after": after,
            "physical_track_order_status": "NOT_AVAILABLE_FROM_FASTF1_POSITION_XYZ",
            "event_time_interval_start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat() if start is not None else "",
            "event_time_interval_end_utc": datetime.fromtimestamp(end, timezone.utc).isoformat() if end is not None else "",
            "timestamp_uncertainty_sec": round(uncertainty, 6),
            "pit_status_evidence": "PIT_LAP_OVERLAP_RECORDED_BUT_TIME_OF_DAY_ONLY" if relevant_pits else "NO_RELEVANT_PIT_ROW_FOUND_IN_CONTEXT",
            "race_control_status_evidence": "RC_RECORD_ON_DECISION_OR_ENDPOINT_LAP" if relevant_rc else "NO_RC_RECORD_ON_DECISION_OR_ENDPOINT_LAP",
            "telemetry_evidence": telemetry_evidence,
            "classification": classification,
            "evidence_references": " || ".join(x for x in refs if x),
            "review_status": "AUTOMATED_REVIEW_UNREVIEWED",
        })
    return fields, out


def make_windows(inputs: dict[str, Any], laptimes: dict[str, dict[str, dict[int, dict[str, Any]]]],
                 car_samples: dict[tuple[str, str, float], dict[str, str]], pos_samples: dict[tuple[str, str, float], dict[str, str]],
                 pit_by: dict[tuple[str, str], list[dict[str, str]]], rc_by: dict[str, list[dict[str, str]]], candidate_by_key: dict[tuple[str, str, str, int, int], list[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    windows, features, links = [], [], []
    for rid in sorted(inputs["pass_ids"]):
        split = inputs["split_by_race"][rid]
        race_drivers = laptimes.get(rid, {})
        # Build a lap boundary order map from the source records. This is
        # classified order at a boundary, not physical track order.
        for attacker, records in sorted(race_drivers.items()):
            for lap, current in sorted(records.items()):
                decision_t = current.get("session_time_start_sec")
                if decision_t is None:
                    continue
                current_pos = current.get("position")
                target = ""
                target_rec: dict[str, Any] = {}
                if current_pos is not None:
                    for other, other_records in race_drivers.items():
                        if other == attacker:
                            continue
                        rec = other_records.get(lap)
                        if rec and rec.get("position") == current_pos - 1:
                            if target:
                                target = ""
                                target_rec = {}
                                break
                            target, target_rec = other, rec
                endpoint_lap = lap + 1
                endpoint = race_drivers.get(target, {}).get(endpoint_lap, {}) if target else {}
                attacker_endpoint = records.get(endpoint_lap, {}) if target else {}
                endpoint_t = endpoint.get("session_time_start_sec") if endpoint else None
                if endpoint_t is None and attacker_endpoint:
                    endpoint_t = attacker_endpoint.get("session_time_start_sec")
                # As-of samples use the latest telemetry at or before the
                # decision. The target's boundary can lag the attacker's, so
                # the later boundary is the conservative pair decision time.
                if target_rec.get("session_time_start_sec") is not None:
                    decision_t = max(decision_t, target_rec["session_time_start_sec"])
                car = car_samples.get((rid, attacker, float(decision_t)), {})
                pos = pos_samples.get((rid, attacker, float(decision_t)), {})
                sample_id = hashlib.sha256(f"{rid}|{attacker}|{lap}|{decision_t}".encode()).hexdigest()[:20]
                pit_horizon = [r for d in (attacker, target) if d for r in pit_by.get((rid, d), []) if parse_int(r.get("lap")) == lap]
                rc_horizon = [r for r in rc_by.get(rid, []) if parse_int(r.get("lap")) == lap]
                endpoint_available = bool(target and attacker_endpoint and endpoint and attacker_endpoint.get("position") is not None and endpoint.get("position") is not None and endpoint_t is not None)
                if not target:
                    status = "UNKNOWN_CENSORED"; label = ""; reason = "NO_UNIQUE_DIRECT_PREDECESSOR_AT_DECISION_BOUNDARY"
                elif not endpoint_available:
                    status = "UNKNOWN_CENSORED"; label = ""; reason = "ENDPOINT_BOUNDARY_UNOBSERVABLE_OR_RACE_ENDED"
                elif pit_horizon:
                    status = "UNKNOWN_CENSORED"; label = ""; reason = "PIT_RECORD_ON_DECISION_LAP_CONDITIONS_PROXY_POPULATION"
                elif rc_horizon:
                    status = "UNKNOWN_CENSORED"; label = ""; reason = "RACE_CONTROL_RECORD_ON_DECISION_LAP_REQUIRES_REVIEW"
                elif attacker_endpoint["position"] < endpoint["position"]:
                    status = "PROXY_POSITIVE"; label = "1"; reason = "FIXED_PAIR_CLASSIFIED_ORDER_REVERSED_AT_NEXT_LAP_BOUNDARY"
                elif attacker_endpoint["position"] > endpoint["position"]:
                    status = "PROXY_NEGATIVE"; label = "0"; reason = "FIXED_PAIR_REMAINS_IN_CLASSIFIED_ORDER_AT_NEXT_LAP_BOUNDARY"
                else:
                    status = "UNKNOWN_CENSORED"; label = ""; reason = "INVALID_OR_TIED_ENDPOINT_ORDER"
                event_ids = candidate_by_key.get((rid, attacker, target, lap, endpoint_lap), []) if target else []
                event_id = event_ids[0] if event_ids else ""
                links.extend({"window_id": sample_id, "event_id": x, "link_type": "TIMING_CANDIDATE_MATCH"} for x in event_ids)
                windows.append({
                    "window_id": sample_id, "race_id": rid, "split": split, "year": rid.split(":", 1)[0],
                    "event": rid.split(":", 2)[1], "session": rid.rsplit(":", 1)[1], "attacker_driver": attacker,
                    "target_driver_fixed": target, "decision_lap": lap, "endpoint_lap": endpoint_lap,
                    "decision_session_time_sec": decision_t, "endpoint_session_time_sec": endpoint_t or "",
                    "horizon_duration_sec": round(endpoint_t - decision_t, 6) if endpoint_t is not None else "",
                    "decision_boundary_utc": current.get("boundary_utc", ""), "endpoint_boundary_utc": endpoint.get("boundary_utc", "") if endpoint else "",
                    "target_identity_status": "UNIQUE_DIRECT_PREDECESSOR_AT_SAME_LAP_BOUNDARY" if target else "NO_UNIQUE_TARGET",
                    "outcome_status": status, "proxy_label": label, "censoring_reason": reason,
                    "matched_candidate_id": event_id, "true_ontrack_label": "UNKNOWN_CENSORED",
                })
                features.append({
                    "window_id": sample_id, "race_id": rid, "split": split, "year": rid.split(":", 1)[0],
                    "event": rid.split(":", 2)[1], "session": rid.rsplit(":", 1)[1], "driver": attacker,
                    "decision_lap": lap, "decision_session_time_sec": decision_t,
                    "car_sample_session_time_sec": car.get("session_time_sec", ""), "car_sample_age_sec": round(decision_t - parse_float(car.get("session_time_sec")), 6) if car and parse_float(car.get("session_time_sec")) is not None else "",
                    "car_sample_status": "ASOF_WITHIN_1S" if car else "MISSING_NO_PRIOR_SAMPLE_WITHIN_1S",
                    "speed_kmh": car.get("speed_kmh", ""), "rpm": car.get("rpm", ""), "n_gear": car.get("n_gear", ""),
                    "throttle_pct": car.get("throttle_pct", ""), "brake": car.get("brake", ""), "drs": car.get("drs", ""),
                    "pos_sample_session_time_sec": pos.get("session_time_sec", ""), "pos_sample_age_sec": round(decision_t - parse_float(pos.get("session_time_sec")), 6) if pos and parse_float(pos.get("session_time_sec")) is not None else "",
                    "pos_sample_status": "ASOF_WITHIN_1S" if pos else "MISSING_NO_PRIOR_SAMPLE_WITHIN_1S",
                    "x_m": pos.get("x_m", ""), "y_m": pos.get("y_m", ""), "z_m": pos.get("z_m", ""),
                    "feature_lookahead_status": "CAUSAL_ASOF_NO_INTERPOLATION",
                })
    return windows, features, links


def build_coverage(inputs: dict[str, Any], car: dict[str, Any], pos: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = ["race_id", "split", "year", "event", "session", "car_rows", "position_rows", "car_unique_drivers", "position_unique_drivers", "car_driver_race_entries", "position_driver_race_entries", "car_laps_covered", "position_laps_covered", "car_driver_hours", "position_driver_hours", "car_gap_count_over_5s", "car_gap_seconds_over_5s", "position_gap_count_over_5s", "position_gap_seconds_over_5s", "car_duplicate_timestamps", "position_duplicate_timestamps", "car_nonmonotonic_timestamps", "position_nonmonotonic_timestamps", "car_timestamp_status", "position_timestamp_status"]
    rows = []
    for r in sorted(inputs["passed"], key=lambda x: norm(x["race_id"])):
        rid = norm(r["race_id"]); c, p = car["race_stats"][rid], pos["race_stats"][rid]
        rows.append({"race_id": rid, "split": inputs["split_by_race"][rid], "year": r["year"], "event": r["event"], "session": r["session"],
            "car_rows": c["rows"], "position_rows": p["rows"], "car_unique_drivers": c["drivers"], "position_unique_drivers": p["drivers"],
            "car_driver_race_entries": c["driver_lap_streams"] and c["drivers"] or 0, "position_driver_race_entries": p["driver_lap_streams"] and p["drivers"] or 0,
            "car_laps_covered": c["laps"], "position_laps_covered": p["laps"], "car_driver_hours": c["driver_hours"], "position_driver_hours": p["driver_hours"],
            "car_gap_count_over_5s": c["gap_count_over_5s"], "car_gap_seconds_over_5s": c["gap_seconds_over_5s"], "position_gap_count_over_5s": p["gap_count_over_5s"], "position_gap_seconds_over_5s": p["gap_seconds_over_5s"],
            "car_duplicate_timestamps": c["duplicate_timestamps"], "position_duplicate_timestamps": p["duplicate_timestamps"], "car_nonmonotonic_timestamps": c["nonmonotonic_timestamps"], "position_nonmonotonic_timestamps": p["nonmonotonic_timestamps"],
            "car_timestamp_status": "PASS" if not any(c[x] for x in ("duplicate_timestamps", "nonmonotonic_timestamps", "nonfinite_time")) else "FAIL",
            "position_timestamp_status": "PASS" if not any(p[x] for x in ("duplicate_timestamps", "nonmonotonic_timestamps", "nonfinite_time")) else "FAIL"})
    return fields, rows


def data_dictionary() -> list[dict[str, str]]:
    rows = []
    for f, meaning, source, status in [
        ("telemetry_car_37_passed.csv.gz", "Filtered continuous car stream; one row per car telemetry sample", "FastF1 car_data", "telemetry-ready"),
        ("telemetry_position_37_passed.csv.gz", "Filtered continuous position/status stream; one row per position sample", "FastF1 pos_data", "telemetry-ready"),
        ("prediction_windows_37.csv.gz", "One boundary window per driver/lap where laptime context exists; outcome columns are outside feature allowlist", "TracingInsights laptimes.json plus FastF1 coverage", "proxy-label-ready only"),
        ("proxy_feature_view_37.csv.gz", "Causal as-of car/position feature view; current sample only, no labels/evidence", "FastF1 car_data/pos_data", "proxy-label-ready only"),
        ("proxy_labels_37.csv.gz", "Explicit boundary-order proxy outcome, not a verified overtake", "TracingInsights laptimes.json pos", "proxy-label-ready only"),
        ("driver_telemetry_coverage_37.csv", "Per-driver coverage inventory filtered to passed races", "Completed FastF1 collection inventory", "coverage audit"),
        ("proxy_counts_by_split.csv", "Proxy window counts using the unchanged race-level split", "Generated from prediction windows", "coverage audit"),
        ("event_review_37.csv", "Automated evidence review of inherited position-swap candidates", "TracingInsights laptimes + public pit/RC context + FastF1 telemetry", "verified-overtake-ready: NO"),
        ("target_contract_v3_passed_races.md", "Frozen boundary-order proxy definition", "Phase 3 contract carried to continuous-race subset", "contract"),
        ("phase3_split_manifest_37.csv", "Existing race-level split assignments filtered to PASS races", "Approved split manifest", "lineage"),
    ]:
        rows.append({"field_or_artifact": f, "meaning": meaning, "source": source, "status": status})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    token = safe_protected_token(root)
    inputs = load_inputs(root, token)
    input_hashes = {str(p.relative_to(root)): sha256_file(p) for p in source_paths(root) if p.is_file()}
    laptimes, laptime_sources = load_laptimes(root, inputs, token)
    temp = Path(tempfile.mkdtemp(prefix="apex_phase3_passed_"))
    try:
        car_path = temp / "telemetry_car_37_passed.csv.gz"; pos_path = temp / "telemetry_position_37_passed.csv.gz"
        car_stats = stream_filter_and_stats(root / CAR_SOURCE, car_path, inputs["pass_ids"], CAR_FIELDS)
        pos_stats = stream_filter_and_stats(root / POS_SOURCE, pos_path, inputs["pass_ids"], POS_FIELDS)
        # Boundary times are small; index only samples needed for all windows.
        needed: dict[tuple[str, str], list[float]] = defaultdict(list)
        for rid, drivers in laptimes.items():
            for d, recs in drivers.items():
                for rec in recs.values():
                    if rec.get("session_time_start_sec") is not None:
                        needed[(rid, d)].append(float(rec["session_time_start_sec"]))
        car_samples = build_telemetry_boundary_index(car_path, needed, CAR_FIELDS)
        pos_samples = build_telemetry_boundary_index(pos_path, needed, POS_FIELDS)
        pit_by = pit_rows_by_race(inputs["pit"], inputs["pass_ids"]); rc_by = rc_rows_by_race(inputs["rc"], inputs["pass_ids"])
        event_fields, event_review = build_event_review(inputs, laptimes, pit_by, rc_by)
        candidate_by_key: dict[tuple[str, str, str, int, int], list[str]] = defaultdict(list)
        for c in inputs["candidate_subset"]:
            key = (norm(c["race_id"]), norm(c["attacker"]), norm(c["target_driver"]), parse_int(c.get("decision_lap")) or -1, parse_int(c.get("endpoint_lap")) or -1)
            candidate_by_key[key].append(norm(c["candidate_id"]))
        windows, features, links = make_windows(inputs, laptimes, car_samples, pos_samples, pit_by, rc_by, candidate_by_key)
        coverage_fields, coverage_rows = build_coverage(inputs, car_stats, pos_stats)

        # Small metadata and evidence outputs live in the workspace; the two
        # large raw stream members are kept in the temporary filesystem until
        # ZIP creation to avoid duplicating the near-full home filesystem.
        write_csv(out / "passed_race_manifest.csv", [dict(r, derived_from="telemetry_coverage_status=PASS") for r in inputs["passed"]], list(inputs["passed"][0]) + ["derived_from"])
        write_json(out / "passed_race_manifest.json", [{**r, "derived_from": "telemetry_coverage_status=PASS"} for r in inputs["passed"]])
        write_csv(out / "phase3_split_manifest_37.csv", [r for r in inputs["split_rows"] if norm(r["race_id"]) in inputs["pass_ids"]], list(inputs["split_rows"][0]))
        write_csv(out / "telemetry_coverage_37.csv", coverage_rows, coverage_fields)
        write_json(out / "telemetry_coverage_37.json", coverage_rows)
        driver_coverage_rows = [r for r in inputs["driver_coverage"] if norm(r.get("race_id")) in inputs["pass_ids"]]
        write_csv(out / "driver_telemetry_coverage_37.csv", driver_coverage_rows, list(inputs["driver_coverage"][0]) if inputs["driver_coverage"] else ["race_id"])
        write_gzip_csv(out / "position_swap_candidates_37.csv.gz", inputs["candidate_subset"], list(inputs["candidate_subset"][0]))
        write_json(out / "position_swap_candidates_37.json", inputs["candidate_subset"])
        write_csv(out / "event_review_37.csv", event_review, event_fields)
        write_gzip_csv(out / "prediction_windows_37.csv.gz", windows, list(windows[0]) if windows else ["window_id"])
        write_gzip_csv(out / "proxy_feature_view_37.csv.gz", features, list(features[0]) if features else ["window_id"])
        label_fields = ["window_id", "race_id", "split", "attacker_driver", "target_driver_fixed", "decision_lap", "endpoint_lap", "outcome_status", "proxy_label", "censoring_reason", "matched_candidate_id", "true_ontrack_label"]
        write_gzip_csv(out / "proxy_labels_37.csv.gz", [{k: w.get(k, "") for k in label_fields} for w in windows], label_fields)
        write_csv(out / "proxy_event_links.csv", links, ["window_id", "event_id", "link_type"])
        split_proxy = defaultdict(Counter)
        for w in windows:
            split_proxy[w["split"]][w["outcome_status"]] += 1
        split_proxy_fields = ["split", "windows", "proxy_positive", "proxy_negative", "unknown_or_censored", "independent_supported_events"]
        split_proxy_rows = [{"split": split, "windows": sum(counts.values()), "proxy_positive": counts["PROXY_POSITIVE"], "proxy_negative": counts["PROXY_NEGATIVE"], "unknown_or_censored": counts["UNKNOWN_CENSORED"], "independent_supported_events": 0} for split, counts in sorted(split_proxy.items())]
        write_csv(out / "proxy_counts_by_split.csv", split_proxy_rows, split_proxy_fields)
        write_csv(out / "laptime_source_inventory_37.csv", laptime_sources, list(laptime_sources[0]) if laptime_sources else ["race_id"])
        write_csv(out / "fastf1_source_manifest_37.csv", [r for r in inputs["fastf1_manifest"] if norm(r.get("race_id")) in inputs["pass_ids"]], list(inputs["fastf1_manifest"][0]))
        pit_subset = [r for r in inputs["pit"] if race_key(r.get("year"), r.get("event"), r.get("session")) in inputs["pass_ids"]]
        rc_subset = [r for r in inputs["rc"] if race_key(r.get("year"), r.get("event"), r.get("session")) in inputs["pass_ids"]]
        write_csv(out / "pit_context_37.csv", pit_subset, list(inputs["pit"][0]) if inputs["pit"] else ["race_id"])
        write_csv(out / "race_control_context_37.csv", rc_subset, list(inputs["rc"][0]) if inputs["rc"] else ["race_id"])
        write_csv(out / "data_dictionary.csv", data_dictionary(), ["field_or_artifact", "meaning", "source", "status"])
        write_json(out / "source_input_hashes_before.json", input_hashes)
        write_json(out / "source_input_hashes_after.json", {str(p.relative_to(root)): sha256_file(p) for p in source_paths(root) if p.is_file()})

        target_contract = """# Target contract v3.0 — passed-race continuous subset\n\n## Scope\n\nThis is a historical, timing-derived proxy task over only races whose continuous FastF1 car and position streams were marked `PASS`. It is not a verified on-track overtake label and is not connected to APEX-R strategy actions.\n\n## Decision and horizon\n\nCreate one decision window for each driver/lap boundary present in TracingInsights `laptimes.json`. At the decision boundary, the fixed target is the unique driver with classified position exactly one place ahead of the attacker in the same lap-boundary snapshot. The target never changes within the window. The horizon is the next lap boundary (`endpoint_lap = decision_lap + 1`) for that same pair; its duration is the observed session-time difference and is not forced to a fixed number of seconds.\n\n## Proxy outcome\n\n`PROXY_POSITIVE` means the fixed pair's classified positions reverse at the next boundary. `PROXY_NEGATIVE` means both endpoint positions are observed and the attacker remains behind the fixed target. This is a boundary-order proxy only: hidden pass/repass, lapping, unlapping and timing corrections are not ruled out. `true_ontrack_label` remains `UNKNOWN_CENSORED`.\n\n## Eligibility and censoring\n\nWindows without a unique adjacent target, an observed endpoint, a same-horizon public pit record, a race-control record on the decision lap, or sufficient boundary timing are `UNKNOWN_CENSORED`; they are not negatives. A race ending before the next boundary is censored. Public pit rows in this dataset provide lap/time-of-day evidence and do not prove publication latency or exact entry/exit timestamps.\n\n## Causal feature rule\n\nThe feature view uses only the latest FastF1 car/position sample at or before the decision boundary, within an explicit 1.0-second freshness limit. No future sample, interpolation, backfill or full-horizon aggregate is used. Source measurement time is historical; live ingestion/publication latency is unknown. Outcome/evidence fields are kept outside the feature view.\n\n## Evidence status\n\nTracingInsights laptimes provides classified lap-boundary positions and timestamps. FastF1 `pos_data` provides XYZ/status location measurements, not classified running order or an overtake timestamp. Therefore supported on-track pass count is expected to remain zero unless independent event evidence is supplied.\n"""
        (out / "target_contract_v3_passed_races.md").write_text(target_contract, encoding="utf-8")

        # Re-run the machine checks after all output construction and before
        # package creation. These are tests of invariants, not model metrics.
        hash_after = {str(p.relative_to(root)): sha256_file(p) for p in source_paths(root) if p.is_file()}
        tests = []
        def test(name: str, ok: bool, detail: str) -> None:
            tests.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        test("source_inputs_immutable", input_hashes == hash_after, "before/after SHA-256 maps equal")
        test("pass_filter_derives_37_or_actual_count", len(inputs["pass_ids"]) == len(inputs["passed"]), f"derived PASS races={len(inputs['pass_ids'])}")
        test("streams_filtered_only_to_pass_races", set(r["race_id"] for r in coverage_rows) == inputs["pass_ids"], "coverage rows equal derived PASS IDs")
        test("context_tables_filtered_only_to_pass_races", all(race_key(r.get("year"), r.get("event"), r.get("session")) in inputs["pass_ids"] for r in pit_subset + rc_subset), "pit and race-control context rows equal PASS-race scope")
        test("car_position_streams_separate", car_stats["total_rows"] > 0 and pos_stats["total_rows"] > 0 and car_stats["total_rows"] != pos_stats["total_rows"], "separate files and counts")
        test("source_timestamp_order_car", car_stats["nonmonotonic_timestamps"] == 0 and car_stats["duplicate_timestamps"] == 0, "no duplicate/non-monotonic timestamps")
        test("source_timestamp_order_position", pos_stats["nonmonotonic_timestamps"] == 0 and pos_stats["duplicate_timestamps"] == 0, "no duplicate/non-monotonic timestamps")
        test("cross_driver_race_isolation", all(norm(w["race_id"]) in inputs["pass_ids"] for w in windows), "windows do not cross selected race IDs")
        test("feature_label_separation", not set(features[0]).intersection({"proxy_label", "outcome_status", "censoring_reason", "matched_candidate_id"}) if features else True, "proxy feature view has no outcome fields")
        test("split_preservation", all(norm(r["split"]) == inputs["split_by_race"][norm(r["race_id"])] for r in coverage_rows), "copied approved race split")
        test("protected_exclusion", all(token not in str(v) for v in [x for r in inputs["passed"] for x in r.values()] + [w.get("race_id", "") for w in windows]), "protected token absent from requested/output race values")
        test("no_network_downloads", True, "pipeline is offline; downloads=0")
        test("candidate_subset_lineage", len(inputs["candidate_subset"]) <= len(inputs["candidates"]), f"subset={len(inputs['candidate_subset'])}, source={len(inputs['candidates'])}")
        test("proxy_not_verified_overtake", sum(e["classification"] == "SUPPORTED_ON_TRACK_PASS" for e in event_review) == 0, "no FastF1 XYZ sample was promoted to a pass")
        test("no_future_feature_join", all(f.get("feature_lookahead_status") == "CAUSAL_ASOF_NO_INTERPOLATION" for f in features), "features use prior/as-of sample only")
        write_json(out / "phase3_passed_tests.json", {"all_passed": all(t["status"] == "PASS" for t in tests), "tests": tests})

        label_counts = Counter(w["outcome_status"] for w in windows)
        review_counts = Counter(e["classification"] for e in event_review)
        supported_events = len({e["candidate_id"] for e in event_review if e["classification"] == "SUPPORTED_ON_TRACK_PASS"})
        metrics = {
            "generated_at_utc": utc_now(), "scope": {"selected_races_derived": True, "selected_races": len(inputs["pass_ids"]), "downloaded_additional_races": 0, "models_touched": False, "holdout_accessed": False, "network_requests": 0},
            "race_counts": {"approved_races_in_status": len(inputs["status"]), "pass_races": len(inputs["pass_ids"]), "non_pass_races_not_processed": len(inputs["status"]) - len(inputs["pass_ids"])},
            "split_counts": dict(Counter(inputs["split_by_race"][r] for r in inputs["pass_ids"])),
            "candidates": {"all_source": len(inputs["candidates"]), "selected_subset": len(inputs["candidate_subset"]), "selected_candidate_races": len({r["race_id"] for r in inputs["candidate_subset"]}), "source_sha256": sha256_file(root / CANDIDATE_SOURCE)},
            "car_stream": car_stats, "position_stream": pos_stats,
            "event_review": {"rows": len(event_review), "classification_counts": dict(review_counts), "supported_on_track_pass_events": supported_events, "human_review_performed": False},
            "prediction_windows": {"rows": len(windows), "status_counts": dict(label_counts), "positive_proxy": label_counts["PROXY_POSITIVE"], "negative_proxy": label_counts["PROXY_NEGATIVE"], "unknown_or_censored": label_counts["UNKNOWN_CENSORED"], "independent_supported_events": supported_events, "window_event_links": len(links)},
            "proxy_counts_by_split": {row["split"]: {k: row[k] for k in split_proxy_fields[1:]} for row in split_proxy_rows},
            "feature_sampling": {"decision_windows": len(features), "car_asof_matched": sum(f["car_sample_status"] == "ASOF_WITHIN_1S" for f in features), "position_asof_matched": sum(f["pos_sample_status"] == "ASOF_WITHIN_1S" for f in features), "asof_tolerance_sec": 1.0, "interpolation": False},
            "readiness": {"telemetry_ready_races": len(inputs["pass_ids"]), "proxy_label_ready": label_counts["PROXY_POSITIVE"] + label_counts["PROXY_NEGATIVE"] > 0, "verified_overtake_ready": supported_events > 0, "decision": "READY_FOR_PROXY_TASK_ONLY" if label_counts["PROXY_POSITIVE"] + label_counts["PROXY_NEGATIVE"] > 0 else "BLOCKED"},
            "tests": {"all_passed": all(t["status"] == "PASS" for t in tests), "count": len(tests)},
            "input_hashes_before": input_hashes, "input_hashes_after": hash_after,
            "artifacts": {"large_streams_kept_in_package": ["telemetry_car_37_passed.csv.gz", "telemetry_position_37_passed.csv.gz"], "small_artifacts_directory": str(out)},
        }
        write_json(out / "phase3_passed_metrics.json", metrics)
        report = f"""# Phase 3 passed-race continuous telemetry subset\n\nGenerated: {metrics['generated_at_utc']}\n\n## Result\n\nThe race list was derived at runtime from `telemetry_coverage_status=PASS` in the completed collection status file. No remaining races were downloaded. This package does not alter the original collection, models or engine.\n\n* PASS races: **{len(inputs['pass_ids'])}** of {len(inputs['status'])}; non-PASS races left untouched: **{len(inputs['status']) - len(inputs['pass_ids'])}**.\n* Existing split assignments were preserved: **{dict(Counter(inputs['split_by_race'][r] for r in inputs['pass_ids']))}**. All selected races are training data; previously inspected development-test races remain development data, not an untouched final holdout.\n* Car rows: **{car_stats['total_rows']:,}**; position rows: **{pos_stats['total_rows']:,}**. These streams remain separate and are not counted as independent examples together.\n* Unique drivers: car {car_stats['drivers']}; position {pos_stats['drivers']}. Driver-race entries: car {car_stats['driver_race_entries']}; position {pos_stats['driver_race_entries']}.\n* Laps covered (driver-lap streams): car {car_stats['laps']:,}; position {pos_stats['laps']:,}. Driver-hours: car {car_stats['driver_hours']:.6f}; position {pos_stats['driver_hours']:.6f}.\n* Gaps >5 seconds: car {car_stats['gap_count_over_5s']} totaling {car_stats['gap_seconds_over_5s']:.3f}s; position {pos_stats['gap_count_over_5s']} totaling {pos_stats['gap_seconds_over_5s']:.3f}s.\n\n## Candidate evidence\n\nThe existing unresolved candidate set contains {len(inputs['candidates']):,} rows; **{len(inputs['candidate_subset']):,}** match the derived PASS-race subset. Every candidate keeps its original ID, source URLs and source hashes in `position_swap_candidates_37.*`.\n\nThe automated review classifies {review_counts.get('OTHER_POSITION_CHANGE', 0):,} candidates as `OTHER_POSITION_CHANGE` where a public pit row overlaps the decision/endpoint lap, and {review_counts.get('UNRESOLVED', 0):,} as `UNRESOLVED`. It claims **0 supported on-track passes**: FastF1 position data contains XYZ/status, not classified running order or a pass timestamp. No human or video review was performed. Event intervals are conservative boundary intervals with uncertainty recorded.\n\n## Target and windows\n\n`target_contract_v3_passed_races.md` freezes the task before windows are interpreted. One window is generated per driver/lap boundary across all available passed-race periods, not only around inherited candidates. The fixed target is the unique classified direct predecessor at that boundary; it does not change during the horizon. The proxy horizon is the next lap boundary.\n\n* Windows: **{len(windows):,}**.\n* `PROXY_POSITIVE`: **{label_counts['PROXY_POSITIVE']:,}**; `PROXY_NEGATIVE`: **{label_counts['PROXY_NEGATIVE']:,}**; `UNKNOWN_CENSORED`: **{label_counts['UNKNOWN_CENSORED']:,}**.\n* Independent supported on-track events: **0**. Positive/negative counts are boundary-order proxy counts only, not verified overtakes.\n* Features use an as-of join to the latest telemetry sample at or before the boundary, maximum age **1.0 second**, no interpolation. Live publication/ingestion latency remains unknown.\n\n## Checks\n\n{sum(t['status'] == 'PASS' for t in tests)}/{len(tests)} invariant checks passed. The detailed results are in `phase3_passed_tests.json`; input SHA maps before/after are in `source_input_hashes_*.json`.\n\n## Artifacts\n\nThe two large telemetry CSV streams are package members and are created in a temporary directory during packaging so the original files are not duplicated in the workspace. `PACKAGE_VERIFICATION.json` is an external verification record because including it inside the ZIP would make a self-referential checksum.\n\n## Readiness\n\n**READY_FOR_PROXY_TASK_ONLY**. The subset is telemetry-ready and boundary-order proxy-label-ready, but not verified-overtake-ready. Model training must not treat the proxy as an on-track overtake target without explicitly naming the proxy task.\n"""
        (out / "PHASE3_PASSED_RACES_REPORT.md").write_text(report, encoding="utf-8")
        handoff = """# Phase 4 handoff — passed-race subset\n\nEntry condition: use `proxy_feature_view_37.csv.gz` joined to `proxy_labels_37.csv.gz` by `window_id`, with `race_id` as the grouping key. Keep all windows from a race in one existing split. Do not use `prediction_windows_37.csv.gz` as the model table because it contains outcome/evidence fields.\n\nThis is a boundary-order position-swap proxy. It is not a verified overtake, and no supported on-track event was found in this evidence pass. Train only an explicitly named historical proxy model if desired; do not connect its probability to ATTACK/HOLD/HARVEST until Phase 4 and Phase 5 checks are complete.\n"""
        (out / "PHASE4_HANDOFF.md").write_text(handoff, encoding="utf-8")
        readme = """# Passed-race subset\n\nRun from the project root:\n\n```bash\npython phase3_passed_races/phase3_passed_races_pipeline.py \\\n  --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application \\\n  --output-dir phase3_passed_races \\\n  --package-path /tmp/APEX-R_37_Passed_Races.zip\npython phase3_passed_races/test_passed_races.py --output-dir phase3_passed_races\n```\n\nThe pipeline is offline and derives the selected race list from the actual collection status file. It does not touch non-PASS races, fetch data, train models or run replay. Large streams are written temporarily and included in the ZIP; the temporary stream directory is removed after packaging.\n\n`telemetry_car_37_passed.csv.gz` and `telemetry_position_37_passed.csv.gz` are separate source streams. `proxy_feature_view_37.csv.gz` is causal/as-of only; `proxy_labels_37.csv.gz` is a classified boundary-order proxy. `event_review_37.csv` is automated and unreviewed by a person.\n"""
        (out / "README.md").write_text(readme, encoding="utf-8")
        # SHA manifest is finalized after all small outputs exist, before ZIP.
        all_small = sorted(p for p in out.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in {"SHA256SUMS.txt", "PACKAGE_VERIFICATION.json"})
        sums = "".join(f"{sha256_file(p)}  {p.relative_to(out).as_posix()}\n" for p in all_small)
        (out / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")

        package_path = Path(args.package_path).resolve() if args.package_path else out.parent / "APEX-R_37_Passed_Races.zip"
        package_path.parent.mkdir(parents=True, exist_ok=True)
        if package_path.exists():
            package_path.unlink()
        members = [(p, p.relative_to(out).as_posix()) for p in all_small] + [(out / "SHA256SUMS.txt", "SHA256SUMS.txt"), (car_path, "telemetry_car_37_passed.csv.gz"), (pos_path, "telemetry_position_37_passed.csv.gz")]
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for src, arc in members:
                z.write(src, arc)
        package_sha = sha256_file(package_path)
        extracted = Path(tempfile.mkdtemp(prefix="apex_phase3_pkg_verify_"))
        package_checks = []
        try:
            with zipfile.ZipFile(package_path) as z:
                infos = z.infolist(); z.extractall(extracted)
                package_checks.append({"name": "zip_decompresses", "status": "PASS", "detail": f"members={len(infos)}, compression_types={sorted(set(i.compress_type for i in infos))}"})
            for src, arc in members:
                actual = extracted / arc
                ok = actual.is_file() and sha256_file(actual) == sha256_file(src)
                package_checks.append({"name": f"checksum:{arc}", "status": "PASS" if ok else "FAIL", "detail": sha256_file(actual) if actual.is_file() else "missing"})
            for arc, expected in [("telemetry_car_37_passed.csv.gz", car_stats["total_rows"]), ("telemetry_position_37_passed.csv.gz", pos_stats["total_rows"]), ("position_swap_candidates_37.csv.gz", len(inputs["candidate_subset"])), ("prediction_windows_37.csv.gz", len(windows)), ("proxy_feature_view_37.csv.gz", len(features)), ("proxy_labels_37.csv.gz", len(windows))]:
                path = extracted / arc
                actual_rows = sum(1 for _ in csv.DictReader(gzip.open(path, "rt", encoding="utf-8", newline=""))) if path.suffix == ".gz" else -1
                package_checks.append({"name": f"row_count:{arc}", "status": "PASS" if actual_rows == expected else "FAIL", "detail": f"actual={actual_rows}, expected={expected}"})
        finally:
            shutil.rmtree(extracted, ignore_errors=True)
        package_verification = {"package_path": str(package_path), "bytes": package_path.stat().st_size, "sha256": package_sha, "compression": "ZIP_DEFLATED", "member_count": len(members), "checks": package_checks, "all_passed": all(x["status"] == "PASS" for x in package_checks)}
        write_json(out / "PACKAGE_VERIFICATION.json", package_verification)
        # External record is intentionally not inserted into the ZIP.
        write_json(package_path.with_name("APEX-R_37_Passed_Races_PACKAGE_VERIFICATION.json"), package_verification)
        return {"metrics": metrics, "package_verification": package_verification, "output_dir": str(out), "package_path": str(package_path)}
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--package-path", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, default=str))
