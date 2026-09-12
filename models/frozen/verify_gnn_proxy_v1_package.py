#!/usr/bin/env python3
"""Verify the frozen GNN ZIP by fresh extraction, checksums and inference."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "models" / "frozen" / "APEX-R_Frozen_GNN_Proxy_v1.zip"
REPORT = ROOT / "models" / "frozen" / "APEX-R_Frozen_GNN_Proxy_v1.verification.json"
SHA_FILE = ROOT / "models" / "frozen" / "APEX-R_Frozen_GNN_Proxy_v1.zip.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)

    archive_hash = sha256(ARCHIVE)
    with tempfile.TemporaryDirectory(prefix="apexr-frozen-verify-") as temporary:
        extract_root = Path(temporary)
        with zipfile.ZipFile(ARCHIVE) as archive:
            members = archive.infolist()
            unsafe = [member.filename for member in members if not safe_member(member.filename)]
            if unsafe:
                raise RuntimeError(f"unsafe ZIP members: {unsafe}")
            bad_crc = archive.testzip()
            if bad_crc:
                raise RuntimeError(f"ZIP CRC failure: {bad_crc}")
            non_deflated_files = [
                member.filename for member in members
                if not member.is_dir() and member.compress_type != zipfile.ZIP_DEFLATED
            ]
            if non_deflated_files:
                raise RuntimeError(f"non-DEFLATE package members: {non_deflated_files}")
            archive.extractall(extract_root)

        package = extract_root / "gnn_proxy_v1"
        checksum_file = package / "SHA256SUMS.txt"
        checksum_failures = []
        checked = 0
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            expected, filename = line.split("  ", 1)
            candidate = package / filename
            actual = sha256(candidate)
            checked += 1
            if actual != expected:
                checksum_failures.append({"file": filename, "expected": expected, "actual": actual})
        if checksum_failures:
            raise RuntimeError(f"extracted checksum failures: {checksum_failures}")

        smoke = subprocess.run(
            [sys.executable, str(package / "smoke_test.py"), "--device", "cpu"],
            cwd=package,
            check=True,
            capture_output=True,
            text=True,
        )
        smoke_result = json.loads(smoke.stdout)
        if smoke_result.get("status") != "PASS":
            raise RuntimeError(f"smoke test failed: {smoke_result}")

        source_integrity = json.loads((package / "SOURCE_IMMUTABILITY.json").read_text(encoding="utf-8"))
        source_mismatches = []
        for relative, expected in source_integrity["after_sha256"].items():
            candidate = ROOT / relative
            actual = sha256(candidate)
            if actual != expected:
                source_mismatches.append({"file": relative, "expected": expected, "actual": actual})
        if source_mismatches:
            raise RuntimeError(f"source artifact changed after packaging: {source_mismatches}")

        graph_counts = json.loads((package / "graph_counts.json").read_text(encoding="utf-8"))
        result = {
            "status": "PASS",
            "archive": str(ARCHIVE),
            "archive_bytes": ARCHIVE.stat().st_size,
            "archive_sha256": archive_hash,
            "zip_crc": "PASS",
            "zip_file_compression": "ZIP_DEFLATED",
            "fresh_extraction": "PASS",
            "checksums_checked": checked,
            "checksum_failures": checksum_failures,
            "source_artifacts_unchanged_after_packaging": True,
            "protected_session_accessed": False,
            "smoke_test": smoke_result,
            "declared_graph_counts": {
                "eligible": graph_counts["graphs_eligible"],
                "positive_proxy": graph_counts["eligible_positive"],
                "negative_proxy": graph_counts["eligible_negative"],
                "excluded": graph_counts["graphs_excluded"],
            },
            "note": "No training, tuning, label construction, data collection or protected-session read was performed.",
        }

    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SHA_FILE.write_text(f"{archive_hash}  {ARCHIVE.name}\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
