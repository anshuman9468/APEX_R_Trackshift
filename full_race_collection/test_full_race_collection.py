#!/usr/bin/env python3
"""Small deterministic regression suite for collection invariants."""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    m = json.loads((root / "metrics.json").read_text())
    assert m["scope"]["approved_manifest_races"] == 70
    assert m["scope"]["network_requests_issued"] == 0
    assert m["source_immutability_passed"] is True
    assert m["validation_gate"]["all_three_passed"] is True
    races = list(csv.DictReader((root / "race_context.csv").open(newline="")))
    assert len(races) == 70
    assert sum(r["complete_source_lap_context"] == "PASS" for r in races) == 69
    assert sum(r["complete_source_lap_context"] == "FAIL" for r in races) == 1
    candidates = json.loads((root / "unresolved_position_swap_candidates.json").read_text())
    assert candidates and all(c["candidate_status"] == "UNRESOLVED_CANDIDATE" for c in candidates)
    assert all(c["verified_overtake"] == "FALSE" for c in candidates)
    events = json.loads((root / "verified_public_events.json").read_text())
    assert events and all(e["verified_overtake"] == "FALSE" for e in events)
    with zipfile.ZipFile(root / "APEX-R_Full_Race_Collection.zip") as zf:
        assert zf.testzip() is None
    print(json.dumps({"passed": True, "test_count": 11}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
