#!/usr/bin/env python3
"""Build an explicit per-driver coverage inventory and collection addendum.

This script does not fetch data, create labels, or change telemetry values. It
joins the already-exported supplied-archive inventory, FastF1 source manifest,
per-driver telemetry coverage, and approved race status using race_id+driver.
Failed retrievals are retained as explicit missing rows.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    workspace = args.workspace.resolve()

    supplied = read_csv(root / "supplied_telemetry_inventory.csv")
    coverage = read_csv(root / "telemetry_coverage.csv")
    statuses = read_csv(root / "race_collection_status.csv")
    sources = read_csv(root / "fastf1_source_manifest.csv")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    coverage_by_key = {(r["race_id"], r["driver"]): r for r in coverage}
    status_by_race = {r["race_id"]: r for r in statuses}
    source_by_race = {r["race_id"]: r for r in sources}
    fields = [
        "race_id", "split", "year", "event", "session", "driver",
        "supplied_archive_status", "supplied_archive_rows", "selected_laps",
        "fastf1_cache_state", "fastf1_retrieval_status", "retrieval_error",
        "telemetry_coverage_status", "fastf1_participant", "car_rows", "position_rows",
        "car_driver_hours", "position_driver_hours", "car_session_time_start_sec",
        "car_session_time_end_sec", "position_session_time_start_sec", "position_session_time_end_sec",
        "car_median_interval_sec", "car_p95_interval_sec", "car_max_interval_sec",
        "car_gap_count_over_5s", "car_gap_seconds_over_5s", "position_median_interval_sec",
        "position_p95_interval_sec", "position_max_interval_sec", "position_gap_count_over_5s",
        "position_gap_seconds_over_5s", "coverage_status", "coverage_scope_note",
    ]
    inventory: list[dict[str, Any]] = []
    for base in supplied:
        key = (base["race_id"], base["driver"])
        race = status_by_race[base["race_id"]]
        source = source_by_race[base["race_id"]]
        live = coverage_by_key.get(key)
        row = {field: "" for field in fields}
        row.update({
            "race_id": base["race_id"], "split": base["split"], "year": base["year"],
            "event": base["event"], "session": base["session"], "driver": base["driver"],
            "supplied_archive_status": base.get("telemetry_status", ""),
            "supplied_archive_rows": base.get("supplied_rows", "0"),
            "selected_laps": base.get("selected_laps", ""),
            "fastf1_cache_state": source.get("cache_state_before_load", ""),
            "fastf1_retrieval_status": source.get("retrieval_status", ""),
            "retrieval_error": source.get("error", race.get("error", "")),
            "coverage_status": race.get("telemetry_coverage_status", ""),
            "coverage_scope_note": "per-driver telemetry row" if live else "no per-driver telemetry rows written for this session",
        })
        if live:
            for field in fields:
                if field in live:
                    row[field] = live[field]
            row["race_id"] = base["race_id"]
            row["split"] = base["split"]
            row["year"] = base["year"]
            row["event"] = base["event"]
            row["session"] = base["session"]
            row["driver"] = base["driver"]
            row["supplied_archive_status"] = base.get("telemetry_status", "")
            row["supplied_archive_rows"] = base.get("supplied_rows", "0")
            row["selected_laps"] = base.get("selected_laps", "")
            row["fastf1_cache_state"] = source.get("cache_state_before_load", "")
            row["fastf1_retrieval_status"] = source.get("retrieval_status", "")
            row["retrieval_error"] = source.get("error", "")
            row["coverage_status"] = race.get("telemetry_coverage_status", "")
            row["coverage_scope_note"] = "per-driver telemetry row"
        inventory.append(row)

    out_csv = root / "driver_telemetry_coverage.csv"
    out_json = root / "driver_telemetry_coverage.json"
    write_csv(out_csv, inventory, fields)
    out_json.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failure_categories = Counter()
    for row in statuses:
        if row.get("source_retrieval_status") != "SUCCESS":
            failure_categories[row.get("error", "").split(":", 1)[0]] += 1
    success_count = sum(row.get("source_retrieval_status") == "SUCCESS" for row in statuses)
    pass_count = sum(row.get("telemetry_coverage_status") == "PASS" for row in statuses)
    incomplete_count = len(statuses) - pass_count
    split_summary: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "development_test"):
        group = [r for r in statuses if r["split"] == split]
        split_summary[split] = {
            "races": len(group),
            "successful_retrievals": sum(r.get("source_retrieval_status") == "SUCCESS" for r in group),
            "telemetry_pass_races": sum(r.get("telemetry_coverage_status") == "PASS" for r in group),
            "failed_retrievals": sum(r.get("source_retrieval_status") != "SUCCESS" for r in group),
            "car_rows": sum(int(float(r.get("car_rows") or 0)) for r in group),
            "position_rows": sum(int(float(r.get("position_rows") or 0)) for r in group),
        }

    package = root / "APEX-R_Full_Race_Telemetry_Collection.zip"
    package_size = package.stat().st_size if package.exists() else None
    report = [
        "# Continuous telemetry collection addendum",
        "",
        "This addendum is generated from the completed collection outputs; it does not fetch, relabel, train, or alter source telemetry.",
        "",
        "## Result",
        "",
        f"* Approved scope: {len(statuses)} Race sessions across 2018–2022; expected driver-race rows: {len(supplied):,}.",
        f"* Supplied selected-lap archive: {metrics['supplied_archive']['observations']:,} rows, {metrics['supplied_archive']['selected_driver_lap_streams']:,} selected driver-lap streams, {metrics['supplied_archive']['driver_race_entries_missing']:,} expected driver-race entries absent.",
        f"* FastF1 retrieval attempts: {len(statuses)}; successful sessions: {success_count}; failed sessions: {len(statuses) - success_count}; pre-existing cache reuse: {metrics['downloads']['cached_races']}; new successful source sessions: {success_count - metrics['downloads']['cached_races']}.",
        f"* Race-level telemetry pass: {pass_count}; incomplete/unverified: {incomplete_count}.",
        f"* Raw exported rows: car {metrics['coverage']['car_observations']:,}; position {metrics['coverage']['position_observations']:,}.",
        f"* Driver-hours: car {metrics['coverage']['car_driver_hours']:.6f}; position {metrics['coverage']['position_driver_hours']:.6f}.",
        f"* Missing intervals >5 seconds: car {metrics['coverage']['car_gap_count_over_5s']:,} totaling {metrics['coverage']['car_gap_seconds_over_5s']:.3f} seconds; position {metrics['coverage']['position_gap_count_over_5s']:,} totaling {metrics['coverage']['position_gap_seconds_over_5s']:.3f} seconds.",
        f"* Duplicate timestamps: car {metrics['coverage']['car_duplicate_timestamp_count']:,}; position {metrics['coverage']['position_duplicate_timestamp_count']:,}. Non-monotonic timestamps: car {metrics['coverage']['car_nonmonotonic_count']:,}; position {metrics['coverage']['position_nonmonotonic_count']:,}.",
        "",
        "## Three-race gate",
        "",
        f"The gate was {'PASS' if metrics['validation_gate']['all_three_passed'] else 'FAIL'} before expansion. The selected train races were: " + ", ".join(metrics["validation_gate"]["races"]) + ".",
        "",
        "| Race | car rows | position rows | expected drivers | drivers with telemetry pass | status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in metrics["validation_gate"]["race_records"]:
        report.append(f"| {r['race_id']} | {r['car_rows']:,} | {r['position_rows']:,} | {r['expected_driver_count']} | {r['drivers_with_telemetry_coverage_pass']} | {r['telemetry_coverage_status']} |")
    report += [
        "",
        "## Split and source-access status",
        "",
        "| Split | races | successful sessions | telemetry pass races | failed retrievals | car rows | position rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "validation", "development_test"):
        x = split_summary[split]
        report.append(f"| {split} | {x['races']} | {x['successful_retrievals']} | {x['telemetry_pass_races']} | {x['failed_retrievals']} | {x['car_rows']:,} | {x['position_rows']:,} |")
    report += [
        "",
        f"FastF1 failure categories: {dict(sorted(failure_categories.items()))}. The dominant blocker was the provider/API 500-calls-per-hour rate limit; these sessions are missing, not confirmed to have no telemetry. The 2018 Italian session also returned DataNotLoadedError.",
        "The development-test races remain marked as development_test in every status and inventory table; no model scoring or replay was performed.",
        "",
        "## Semantics and limits",
        "",
        "`telemetry_car.csv.gz` retains raw FastF1 car_data fields: RPM, Speed (km/h), nGear, Throttle (%), Brake, DRS, Date, Time and SessionTime. `telemetry_position.csv.gz` retains raw position-data Status, X, Y and Z plus Date, Time and SessionTime. FastF1 Date was returned timezone-naive in this runtime; SessionTime is session-relative. These are measurement timestamps, not proof of live publication latency or UTC alignment.",
        "No interpolation or nearest-time car/position merge was used. Lap numbers are interval assignments from FastF1 LapStartTime/Time and are not a substitute for a physical-lap completeness proof. Missing intervals remain missing.",
        f"The prior {metrics['events']['preserved_position_swap_candidates']:,} position-swap candidates remain unresolved. Verified overtake events remain {metrics['events']['verified_overtake_events']}; no labels were created.",
        "",
        "## Source limitation and next collection",
        "",
        "FastF1 supplied documented historical car/position access for the successful sessions, but this run cannot claim complete 2018–2022 coverage because 33 race sessions remain incomplete/unverified. The minimum repair is to resume the 30 failed requests after the provider rate-limit window (or use an equivalent documented permitted source), with the same pre-fetch protected-session guard; do not reinterpret missing requests as no data.",
        "If 2018–2022 cannot be completed, propose a separate 2023+ collection using OpenF1’s documented historical endpoints, excluding the protected session before any request. Keep that collection separate from this 2018–2022 package and use it for timestamped event evidence only where session, driver and UTC alignment are actually verified.",
        "",
        f"Final ZIP: {package.name} ({package_size:,} bytes) if present. The ZIP’s own `package_verification.json` is external to the ZIP; `driver_telemetry_coverage.*` and this addendum are post-run audit companions and are not silently represented as ZIP members.",
    ]
    (root / "TELEMETRY_COLLECTION_ADDENDUM.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"driver_inventory_rows": len(inventory), "successful_sessions": success_count, "failed_sessions": len(statuses) - success_count, "telemetry_pass_races": pass_count, "incomplete_races": incomplete_count, "output_csv": str(out_csv), "output_json": str(out_json), "addendum": str(root / 'TELEMETRY_COLLECTION_ADDENDUM.md')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
