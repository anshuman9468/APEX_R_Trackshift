#!/usr/bin/env python3
"""Package and verify the small post-run coverage/audit companion."""
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
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path) -> int:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    with path.open(newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scripts", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    scripts = args.scripts.resolve()
    package = root / "APEX-R_Telemetry_Audit_Companion.zip"
    sums = root / "TELEMETRY_AUDIT_COMPANION_SHA256SUMS.txt"
    files: list[tuple[Path, str]] = []
    for name in [
        "TELEMETRY_COLLECTION_ADDENDUM.md", "driver_telemetry_coverage.csv",
        "driver_telemetry_coverage.json", "collection_verification.json",
        "package_verification.json", "metrics.json", "PHASE_CONTINUOUS_TELEMETRY_REPORT.md",
    ]:
        path = root / name
        if path.exists():
            files.append((path, name))
    for name in ["continuous_telemetry_pipeline.py", "summarize_telemetry_collection.py", "verify_telemetry_collection.py", "test_telemetry_collection.py", "package_telemetry_companion.py"]:
        path = scripts / name
        if path.exists():
            files.append((path, "scripts/" + name))
    sums.write_text("\n".join(f"{sha256(path)}  {arc}" for path, arc in sorted(files, key=lambda x: x[1])) + "\n", encoding="utf-8")
    if package.exists():
        package.unlink()
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path, arc in sorted(files, key=lambda x: x[1]):
            zf.write(path, arcname=arc)
        zf.write(sums, arcname=sums.name)
    checks: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="apex_telemetry_companion_", dir="/tmp") as temp:
        extract = Path(temp)
        with zipfile.ZipFile(package) as zf:
            bad = zf.testzip()
            checks.append({"test": "zip_crc", "passed": bad is None, "detail": bad or "PASS"})
            zf.extractall(extract)
        failures = []
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, arc = line.split("  ", 1)
            if sha256(extract / arc) != digest:
                failures.append(arc)
        checks.append({"test": "extracted_sha256", "passed": not failures, "detail": failures})
        for arc, expected in [("driver_telemetry_coverage.csv", 1399)]:
            actual = rows(extract / arc)
            checks.append({"test": "row_count:" + arc, "passed": actual == expected, "detail": {"actual": actual, "expected": expected}})
        value = json.loads((extract / "driver_telemetry_coverage.json").read_text(encoding="utf-8"))
        checks.append({"test": "row_count:driver_telemetry_coverage.json", "passed": len(value) == 1399, "detail": {"actual": len(value), "expected": 1399}})
    verification = {"package": {"path": str(package), "bytes": package.stat().st_size, "sha256": sha256(package), "compression": "ZIP_DEFLATED"}, "member_count": len(files) + 1, "checks": checks, "all_passed": all(c["passed"] for c in checks)}
    (root / "companion_package_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0 if verification["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
