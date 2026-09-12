#!/usr/bin/env python3
"""Package and verify RAM-safe experiment artifacts without raw-data copies."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "APEX-R_GNN_Proxy_Experiment_v1_RAM_safe.zip"
EXCLUDE_NAMES = {
    "graph_cache.pt", PACKAGE.name, "PACKAGE_VERIFICATION.json", "SHA256SUMS.txt",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def members() -> list[Path]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if p.name in EXCLUDE_NAMES:
            continue
        out.append(p)
    return out


def row_count(path: Path) -> int | None:
    if path.suffix not in {".csv", ".gz"} or path.name.endswith(".npy"):
        return None
    if path.suffix == ".gz":
        import gzip
        opener = gzip.open
    else:
        opener = open
    with opener(path, "rt", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def main() -> None:
    files = members()
    sums = {str(p.relative_to(ROOT)): digest(p) for p in files}
    (ROOT / "SHA256SUMS.txt").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))
    files = members() + [ROOT / "SHA256SUMS.txt"]
    sums = {str(p.relative_to(ROOT)): digest(p) for p in files}
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    with tempfile.TemporaryDirectory(prefix="apex_gnn_pkg_") as td:
        extract = Path(td)
        with zipfile.ZipFile(PACKAGE) as z:
            bad = [i.filename for i in z.infolist() if i.compress_type != zipfile.ZIP_DEFLATED]
            z.extractall(extract)
        verified_sums = all(digest(extract / name) == value for name, value in sums.items())
        row_counts = {name: row_count(extract / name) for name in sums if (extract / name).suffix in {".csv", ".gz"}}
    verification = {
        "package": PACKAGE.name, "package_bytes": PACKAGE.stat().st_size,
        "package_sha256": digest(PACKAGE), "member_count": len(files),
        "all_members_deflate": not bad, "fresh_extract_checksums": verified_sums,
        "row_counts_after_fresh_extract": row_counts,
        "raw_telemetry_included": any("telemetry_" in k for k in sums),
        "legacy_nested_graph_cache_included": "graph_cache.pt" in sums,
        "passed": not bad and verified_sums and not any("telemetry_" in k for k in sums) and "graph_cache.pt" not in sums,
    }
    (ROOT / "PACKAGE_VERIFICATION.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))
    if not verification["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
