#!/usr/bin/env python3
"""Focused, read-only checks for the Phase 3 reconciliation companion."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_package(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "companion package does not exist"
    with zipfile.ZipFile(path) as zf:
        if zf.testzip() is not None:
            return False, "ZIP CRC failure"
        if not zf.infolist() or any(info.compress_type != zipfile.ZIP_DEFLATED for info in zf.infolist()):
            return False, "not all members use ZIP_DEFLATED"
        with tempfile.TemporaryDirectory(prefix="apex_phase3_repair_test_") as tmp:
            target = Path(tmp)
            zf.extractall(target)
            sums = (target / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
            for line in sums:
                digest, name = line.split("  ", 1)
                actual = target / name
                if not actual.is_file() or sha256(actual) != digest:
                    return False, f"checksum mismatch: {name}"
    return True, f"{path.stat().st_size} bytes, {len(zipfile.ZipFile(path).infolist())} DEFLATE members"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="phase3_races_audit_reconciliation_v1")
    parser.add_argument("--package-path", default="")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    token = json.loads((root / "phase3_prediction_dataset" / "excluded_sessions_manifest.json").read_text(encoding="utf-8"))["protected_session_token"]
    tests: list[dict[str, str]] = []

    def test(name: str, condition: bool, detail: str) -> None:
        tests.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    split = read_csv(out / "split_reconciliation.csv")
    test("authoritative_split_mapping", len(split) == 37 and all(row["mismatch_status"] == "MATCH" for row in split), f"rows={len(split)}, mismatches={sum(row['mismatch_status'] != 'MATCH' for row in split)}")
    test("unknown_assignment_not_train", all(row["authoritative_split"] != "UNKNOWN" and row["exported_split"] != "UNKNOWN" and row["fallback_to_train_used"] == "FALSE" for row in split), "missing mappings are not represented as train")

    proposal = read_csv(out / "proposed_chronological_split_37.csv")
    test("proposed_whole_race_split", len(proposal) == 37 and len({row["race_id"] for row in proposal}) == 37, "one proposed assignment per race")
    test("proposed_boundary_precedes_outcomes", all(row["boundary_selected_before_outcome_distribution"] == "TRUE" and row["untouched_final_holdout"] == "FALSE" for row in proposal), "chronology proposal is development-only")
    by_split = {row["proposed_split"]: {r["race_id"] for r in proposal if r["proposed_split"] == row["proposed_split"]} for row in proposal}
    test("proposed_split_disjoint", len(set.union(*by_split.values())) == 37 and sum(len(v) for v in by_split.values()) == 37, "race IDs do not cross proposed partitions")

    lineage = read_csv(out / "positive_window_event_lineage.csv")
    event_summary = read_csv(out / "positive_proxy_event_summary.csv")
    link_stats = json.loads((out / "positive_lineage_reconciliation.json").read_text(encoding="utf-8"))
    test("positive_lineage_cardinality", len(lineage) == link_stats["original_positive_windows"] == 823, f"lineage_rows={len(lineage)}")
    test("positive_window_unique", len({row["window_id"] for row in lineage}) == len(lineage), "no duplicate positive windows")
    test("positive_event_grouping", len(event_summary) == link_stats["unique_boundary_proxy_events"] and max(int(row["window_count"]) for row in event_summary) == 1, "event keys and window multiplicity reconcile")
    test("positive_not_verified", link_stats["verified_on_track_events"] == 0 and all(row["verified_on_track_pass"] == "FALSE" for row in lineage), "proxy positives are not promoted to verified overtakes")
    test("candidate_lineage_reconciles", link_stats["matched_to_1195_candidate_universe"] + link_stats["outside_inherited_candidate_universe"] == 823, "412 inherited links plus 411 outside-list endpoint reversals")

    negative = read_gz(out / "negative_label_audit.csv.gz")
    test("negative_count", len(negative) == 14600, "reported negative row count")
    test("negative_semantics", all("not proof of no physical pass" in row["negative_meaning"] and row["verified_no_on_track_pass"] == "FALSE" for row in negative), "negative means endpoint proxy condition only")
    test("negative_not_candidate_absence", all(not row["matched_candidate_id"] for row in negative), "no negative is justified solely by absent candidate-list membership")

    laps = json.loads((out / "lap_count_reconciliation.json").read_text(encoding="utf-8"))
    test("lap_counter_repaired", laps["reported_2318_interpretation"].startswith("distinct (driver, lap)"), "old aggregate unit is explicit")
    test("driver_lap_count", laps["car_telemetry_driver_lap_ids"] == 40837 and laps["position_telemetry_driver_lap_ids"] == 40837 and laps["expected_laptime_driver_lap_ids"] == 40837, "race-qualified driver-lap identities reconcile")
    test("driver_lap_set_match", laps["car_driver_lap_set_match_to_laptime"] and laps["position_driver_lap_set_match_to_laptime"], "continuous streams match laptime boundary sets")
    test("race_lap_distinct_count", laps["car_telemetry_distinct_race_lap_ids"] == 2236 and laps["position_telemetry_distinct_race_lap_ids"] == 2236, "distinct race-lap identities")

    asof = json.loads((out / "asof_coverage_summary.json").read_text(encoding="utf-8"))
    test("asof_backward_only", asof["future_join_count"] == 0 and asof["interpolation_count"] == 0 and asof["tolerance_sec"] == 1.0, "one-second as-of contract")
    test("asof_missingness_explicit", asof["car_unmatched"] == 3179 and asof["position_unmatched"] == 3177, "unmatched fields remain counted")

    hashes = json.loads((out / "source_immutability_check.json").read_text(encoding="utf-8"))
    test("source_immutability", hashes["all_source_hashes_unchanged"] and not hashes["changed_paths"], "before/after source hashes equal")
    output_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in out.iterdir() if path.is_file() and path.suffix not in {".gz", ".zip"})
    test("protected_holdout_exclusion", token not in output_text and all(token not in row["race_id"] for row in split), "protected token absent from emitted audit content")

    if args.package_path:
        ok, detail = check_package(Path(args.package_path).resolve())
        test("companion_package", ok, detail)

    result = {"all_passed": all(row["status"] == "PASS" for row in tests), "tests": tests}
    (out / "repair_tests.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
