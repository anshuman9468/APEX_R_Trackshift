#!/usr/bin/env python3
"""Independent verification for the continuous telemetry collection package."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stream_count(path: Path) -> tuple[int, set[str]]:
    count = 0
    races: set[str] = set()
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            count += 1
            races.add(row["race_id"])
    return count, races


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"test": name, "passed": bool(passed), "detail": detail})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.root.resolve()
    workspace = args.workspace.resolve()
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    status = csv_rows(root / "race_collection_status.csv")
    coverage = csv_rows(root / "telemetry_coverage.csv")
    supplied = csv_rows(root / "supplied_telemetry_inventory.csv")
    sources = csv_rows(root / "fastf1_source_manifest.csv")
    driver_inventory = csv_rows(root / "driver_telemetry_coverage.csv") if (root / "driver_telemetry_coverage.csv").exists() else []
    add(checks, "approved_race_count", len(status) == 70, len(status))
    add(checks, "per_driver_inventory_count", len(supplied) == 1399 and len(driver_inventory) == 1399, {"supplied": len(supplied), "driver_inventory": len(driver_inventory)})
    add(checks, "source_manifest_count", len(sources) == 70, len(sources))
    add(checks, "coverage_rows_are_successful_sessions_only", len(coverage) == sum(r.get("source_retrieval_status") == "SUCCESS" for r in status) * 20, {"coverage_rows": len(coverage), "successful_sessions": sum(r.get("source_retrieval_status") == "SUCCESS" for r in status)})

    car_count, car_races = stream_count(root / "telemetry_car.csv.gz")
    pos_count, pos_races = stream_count(root / "telemetry_position.csv.gz")
    add(checks, "car_gzip_row_count", car_count == metrics["coverage"]["car_observations"], {"actual": car_count, "expected": metrics["coverage"]["car_observations"]})
    add(checks, "position_gzip_row_count", pos_count == metrics["coverage"]["position_observations"], {"actual": pos_count, "expected": metrics["coverage"]["position_observations"]})
    status_races = {r["race_id"] for r in status}
    add(checks, "car_race_scope", car_races.issubset(status_races), {"races": len(car_races), "outside_scope": sorted(car_races - status_races)})
    add(checks, "position_race_scope", pos_races.issubset(status_races), {"races": len(pos_races), "outside_scope": sorted(pos_races - status_races)})
    add(checks, "car_position_race_sets_match", car_races == pos_races, {"car": len(car_races), "position": len(pos_races)})

    token = str(json.loads((workspace / "phase3_prediction_dataset" / "excluded_sessions_manifest.json").read_text(encoding="utf-8")).get("protected_session_token", ""))
    text_files = [p for p in root.iterdir() if p.is_file() and p.suffix in {".csv", ".json", ".md", ".txt"}]
    protected_hits = [p.name for p in text_files if token and token in p.read_text(encoding="utf-8", errors="replace")]
    add(checks, "protected_token_absent_from_outputs", not protected_hits, protected_hits)
    add(checks, "protected_token_absent_from_streams", not any(token and token in rid for rid in car_races | pos_races), "stream race_ids checked against the protected-session token")
    add(checks, "no_verified_overtake_labels_created", metrics["events"]["verified_overtake_events"] == 0, metrics["events"]["verified_overtake_events"])

    archive_ok = metrics["source_archive_hashes_before"] == metrics["source_archive_hashes_after"]
    prior_ok = metrics["prior_package_sha256_before"] == metrics["prior_package_sha256_after"]
    add(checks, "source_archives_unchanged", archive_ok, {"metric": archive_ok})
    add(checks, "prior_lap_context_package_unchanged", prior_ok, {"metric": prior_ok})
    archive_base = workspace.parent
    actual_archives = {str(archive_base / f"APEX-R_Telemetry_{year}.zip"): sha256_file(archive_base / f"APEX-R_Telemetry_{year}.zip") for year in range(2018, 2023)}
    add(checks, "source_archive_hashes_match_current_files", actual_archives == metrics["source_archive_hashes_before"], actual_archives)
    prior_path = workspace / "full_race_collection" / "APEX-R_Full_Race_Collection.zip"
    actual_prior = sha256_file(prior_path) if prior_path.exists() else ""
    add(checks, "prior_package_hash_matches_current_file", actual_prior == metrics["prior_package_sha256_before"], actual_prior)
    candidate = workspace / "full_race_collection" / "unresolved_position_swap_candidates.csv.gz"
    candidate_ok = candidate.exists() and sha256_file(candidate) == metrics["events"]["preserved_position_swap_candidates_sha256"]
    add(checks, "position_swap_candidate_hash_preserved", candidate_ok, metrics["events"]["preserved_position_swap_candidates_sha256"])
    add(checks, "temporary_fastf1_cache_clean", not any((root / "fastf1_cache").rglob("*")), "fastf1_cache contains no files")

    package = root / "APEX-R_Full_Race_Telemetry_Collection.zip"
    package_verification = json.loads((root / "package_verification.json").read_text(encoding="utf-8"))
    add(checks, "package_bytes_and_hash_match_manifest", package.stat().st_size == package_verification["package"]["bytes"] and sha256_file(package) == package_verification["package"]["sha256"], {"actual_bytes": package.stat().st_size, "manifest": package_verification["package"]})
    with tempfile.TemporaryDirectory(prefix="apex_independent_verify_", dir="/tmp") as tmp:
        extract = Path(tmp)
        with zipfile.ZipFile(package) as zf:
            bad = zf.testzip()
            add(checks, "package_crc", bad is None, bad or "PASS")
            names = set(zf.namelist())
            zf.extractall(extract)
        sums = {}
        for line in (extract / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split("  ", 1)
                sums[name] = digest
        checksum_failures = [name for name, digest in sums.items() if sha256_file(extract / name) != digest]
        add(checks, "package_extracted_checksums", not checksum_failures, checksum_failures)
        artifact_failures = []
        for name, expected in metrics["artifact_rows"].items():
            p = extract / name
            if name.endswith(".csv.gz"):
                actual = max(sum(1 for _ in gzip.open(p, "rt", newline="", encoding="utf-8")) - 1, 0)
            elif name.endswith(".csv"):
                actual = max(sum(1 for _ in p.open(newline="", encoding="utf-8")) - 1, 0)
            elif name.endswith(".json"):
                value = json.loads(p.read_text(encoding="utf-8"))
                actual = len(value) if isinstance(value, list) else expected
            else:
                continue
            if actual != int(expected):
                artifact_failures.append({"name": name, "actual": actual, "expected": expected})
        add(checks, "package_row_counts", not artifact_failures, artifact_failures)
        add(checks, "package_member_set", names == {x["name"] for x in package_verification["members"]}, {"actual": len(names), "expected": len(package_verification["members"])})

    result = {"root": str(root), "checks": checks, "all_passed": all(c["passed"] for c in checks)}
    out = root / "collection_verification.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": result["all_passed"], "checks": len(checks), "failed": [c for c in checks if not c["passed"]], "output": str(out)}, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
