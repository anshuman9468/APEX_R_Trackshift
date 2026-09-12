#!/usr/bin/env python3
"""Convert and verify the legacy APEX-R graph cache without torch.load()."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from disk_graph_store import DiskGraphStore, bounded_equivalence_check, build_disk_store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", type=Path, default=Path(__file__).resolve().parent / "graph_cache.pt")
    ap.add_argument("--store", type=Path, default=Path(__file__).resolve().parent / "disk_graph_store")
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.store.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = build_disk_store(args.legacy, args.store, force=args.force)
    store = DiskGraphStore(args.store)
    layout = store.validate_layout()
    equivalence = bounded_equivalence_check(args.legacy, store)
    report = {
        "legacy_path": str(args.legacy), "legacy_bytes": args.legacy.stat().st_size,
        "store_path": str(args.store), "store_manifest": manifest,
        "layout_validation": layout, "equivalence": equivalence,
        "bounded_conversion_seconds": time.time() - started,
        "legacy_torch_load_used": False,
        "comparison_memory_policy": "one pickle graph and one store slice at a time",
    }
    (args.output / "ram_fix_cache_conversion.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not layout.get("passed") or not equivalence.get("passed"):
        raise SystemExit("Cache conversion verification failed; see ram_fix_cache_conversion.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
