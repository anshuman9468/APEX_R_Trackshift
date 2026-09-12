#!/usr/bin/env python3
"""Package the small reconciliation companion with ZIP DEFLATE."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def row_count(path: Path) -> int | None:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _ in csv.reader(handle)) - 1)
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-path", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    package = Path(args.package_path).resolve()
    package.parent.mkdir(parents=True, exist_ok=True)
    if package.exists():
        package.unlink()
    excluded = {"SHA256SUMS.txt", "PACKAGE_VERIFICATION.json", package.name}
    members = sorted(path for path in out.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.name not in excluded)
    checksum_path = out / "SHA256SUMS.txt"
    checksum_path.write_text("".join(f"{sha256(path)}  {path.relative_to(out).as_posix()}\n" for path in members), encoding="utf-8")
    members = sorted(path for path in out.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.name not in {"PACKAGE_VERIFICATION.json", package.name})
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in members:
            zf.write(path, path.relative_to(out).as_posix())
    checks = []
    with zipfile.ZipFile(package) as zf:
        bad = zf.testzip()
        checks.append({"name": "zip_crc", "status": "PASS" if bad is None else "FAIL", "detail": bad or "all members"})
        checks.append({"name": "compression", "status": "PASS" if all(info.compress_type == zipfile.ZIP_DEFLATED for info in zf.infolist()) else "FAIL", "detail": f"members={len(zf.infolist())}"})
        with tempfile.TemporaryDirectory(prefix="apex_phase3_repair_package_") as tmp:
            extract = Path(tmp)
            zf.extractall(extract)
            for line in (extract / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", 1)
                actual = extract / name
                checks.append({"name": f"checksum:{name}", "status": "PASS" if actual.is_file() and sha256(actual) == digest else "FAIL", "detail": digest})
            row_counts = {path.relative_to(extract).as_posix(): row_count(path) for path in extract.rglob("*") if path.is_file() and path.suffix in {".csv", ".gz"}}
            expected_row_counts = {
                "DATA_DICTIONARY.csv": 9,
                "asof_coverage_by_race.csv": 37,
                "boundary_context_audit.csv": 24,
                "coverage_count_reconciliation.csv": 5,
                "driver_lap_reconciliation.csv": 740,
                "negative_label_audit.csv.gz": 14600,
                "negative_label_audit_summary.csv": 4,
                "positive_proxy_event_summary.csv": 823,
                "positive_window_event_lineage.csv": 823,
                "proposed_chronological_split_37.csv": 37,
                "proposed_split_support.csv": 3,
                "race_lap_reconciliation.csv": 37,
                "split_reconciliation.csv": 37,
            }
            for name, expected in expected_row_counts.items():
                actual = row_counts.get(name)
                checks.append({"name": f"row_count:{name}", "status": "PASS" if actual == expected else "FAIL", "detail": f"actual={actual}, expected={expected}"})
            checks.append({"name": "fresh_extraction", "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL", "detail": "checksums and expected row counts verified"})
    verification = {
        "package_path": str(package),
        "bytes": package.stat().st_size,
        "sha256": sha256(package),
        "compression": "ZIP_DEFLATED",
        "member_count": len(zipfile.ZipFile(package).infolist()),
        "checks": checks,
        "row_counts": row_counts,
        "all_passed": all(check["status"] == "PASS" for check in checks),
        "verified_from_fresh_extraction": True,
    }
    (out / "PACKAGE_VERIFICATION.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
