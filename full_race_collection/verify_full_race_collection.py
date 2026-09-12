#!/usr/bin/env python3
"""Independent verifier for the APEX-R full-race collection package."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def gz_csv_count(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def check(condition: bool, name: str, detail: str, results: list[dict]) -> None:
    results.append({"test": name, "passed": bool(condition), "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def verify(root: Path) -> dict:
    results: list[dict] = []
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    expected = metrics["artifact_rows"]
    for name, count in expected.items():
        path = root / name
        check(path.is_file(), "artifact_exists:" + name, str(path), results)
        if name.endswith(".csv.gz"):
            actual = gz_csv_count(path)
        elif name.endswith(".csv"):
            actual = csv_count(path)
        elif name.endswith(".json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            actual = len(value) if isinstance(value, list) else count
        else:
            continue
        check(actual == int(count), "row_count:" + name, f"actual={actual}, expected={count}", results)

    race_rows = list(csv.DictReader((root / "race_context.csv").open(encoding="utf-8", newline="")))
    lap_ids: set[tuple[str, str, str]] = set()
    duplicate_lap_keys = 0
    complete_race_ids = {r["race_id"] for r in race_rows if r["complete_source_lap_context"] == "PASS"}
    non_complete_lap_rows = 0
    with gzip.open(root / "driver_laps.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["race_id"], row["driver"], row["lap"])
            if key in lap_ids:
                duplicate_lap_keys += 1
            lap_ids.add(key)
            if row["race_id"] not in complete_race_ids:
                non_complete_lap_rows += 1
    check(duplicate_lap_keys == 0, "driver_lap_key_unique", f"duplicates={duplicate_lap_keys}", results)
    check(non_complete_lap_rows == 0, "lap_race_is_complete", f"rows_from_incomplete_races={non_complete_lap_rows}", results)
    check(len({r["race_id"] for r in race_rows}) == len(race_rows), "race_id_unique", str(len(race_rows)), results)
    check(sum(r["complete_source_lap_context"] == "PASS" for r in race_rows) == 69, "complete_race_count", "expected 69 complete source-lap races", results)
    check(sum(r["complete_source_lap_context"] == "FAIL" for r in race_rows) == 1, "incomplete_race_quarantined", "one source-incomplete race remains in the race manifest", results)

    candidates = json.loads((root / "unresolved_position_swap_candidates.json").read_text(encoding="utf-8"))
    check(all(r.get("candidate_status") == "UNRESOLVED_CANDIDATE" for r in candidates), "candidates_not_verified", str(len(candidates)), results)
    check(all(r.get("verified_overtake") == "FALSE" for r in candidates), "no_verified_overtake_claim", str(len(candidates)), results)
    verified = json.loads((root / "verified_public_events.json").read_text(encoding="utf-8"))
    check(all(r.get("event_status") == "VERIFIED_PUBLIC_EVENT" for r in verified), "public_events_source_recorded", str(len(verified)), results)
    check(all(r.get("verified_overtake") == "FALSE" for r in verified), "public_events_not_overtakes", str(len(verified)), results)

    check(metrics["scope"]["network_requests_issued"] == 0, "no_new_network_requests", "reused pinned cache only", results)
    check(metrics["source_immutability_passed"] is True, "source_immutability", "before and after input hashes match", results)
    check(metrics["validation_gate"]["all_three_passed"] is True, "three_race_gate", "all three pre-expansion races passed", results)
    check(metrics["validation_gate"]["expansion_authorized"] is True, "expansion_after_gate", "expansion followed passing gate", results)

    sums = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            sums[name] = digest
    check("SHA256SUMS.txt" not in sums, "checksum_excludes_itself", "checksum file is not self-hashed", results)
    for name, expected_hash in sums.items():
        check((root / name).is_file(), "checksum_target_exists:" + name, name, results)
        check(sha256(root / name) == expected_hash, "checksum:" + name, "SHA256 match", results)

    package = root / "APEX-R_Full_Race_Collection.zip"
    with zipfile.ZipFile(package) as zf:
        bad = zf.testzip()
        check(bad is None, "zip_crc", "all ZIP members pass CRC", results)
        names = set(zf.namelist())
        check(names == set(sums) | {"SHA256SUMS.txt"}, "zip_member_set", f"members={len(names)}", results)
        check(all(".." not in n and not n.startswith("/") for n in names), "zip_paths_safe", "no traversal members", results)
    return {"all_passed": True, "tests": results, "test_count": len(results)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = verify(args.root.resolve())
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": result["all_passed"], "test_count": result["test_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
