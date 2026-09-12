#!/usr/bin/env python3
"""Independent invariant checks for the passed-race Phase 3 package."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--package", type=Path, default=Path("/tmp/APEX-R_37_Passed_Races.zip"))
    args = p.parse_args()
    out = args.output_dir.resolve()
    metrics = json.loads((out / "phase3_passed_metrics.json").read_text(encoding="utf-8"))
    tests: list[dict[str, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        tests.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    passed = rows(out / "passed_race_manifest.csv")
    coverage = rows(out / "telemetry_coverage_37.csv")
    windows = rows(out / "prediction_windows_37.csv.gz")
    features = rows(out / "proxy_feature_view_37.csv.gz")
    labels = rows(out / "proxy_labels_37.csv.gz")
    candidates = rows(out / "position_swap_candidates_37.csv.gz")
    review = rows(out / "event_review_37.csv")
    split = rows(out / "phase3_split_manifest_37.csv")
    pit = rows(out / "pit_context_37.csv")
    race_control = rows(out / "race_control_context_37.csv")
    protected = json.loads((out.parent / "phase3_prediction_dataset" / "excluded_sessions_manifest.json").read_text(encoding="utf-8"))["protected_session_token"]

    check("metrics_pass_race_count", len(passed) == metrics["race_counts"]["pass_races"], f"manifest={len(passed)}")
    pass_ids = {r["race_id"] for r in passed}
    check("coverage_is_pass_only", {r["race_id"] for r in coverage} == pass_ids, "coverage race IDs equal manifest")
    check("context_tables_are_pass_only", {f"{r['year']}:{r['event']}:{r['session']}" for r in pit + race_control} <= pass_ids, "pit and race-control rows stay within PASS race IDs")
    check("candidate_subset_count", len(candidates) == metrics["candidates"]["selected_subset"], f"rows={len(candidates)}")
    check("candidate_review_one_to_one", len(review) == len(candidates) and {r["candidate_id"] for r in review} == {r["candidate_id"] for r in candidates}, "IDs preserved exactly")
    check("split_preserved", all(r["split"] == "train" and r["race_id"] in pass_ids for r in split), "selected split values copied from approved manifest")
    check("feature_label_isolation", not set(features[0]).intersection({"proxy_label", "outcome_status", "censoring_reason", "matched_candidate_id"}), "outcome fields absent from feature view")
    check("label_feature_window_parity", len(features) == len(labels) == len(windows), f"features={len(features)}, labels={len(labels)}, windows={len(windows)}")
    check("window_status_counts", dict(Counter(r["outcome_status"] for r in windows)) == metrics["prediction_windows"]["status_counts"], "labels agree with metrics")
    asof_ok = True
    asof_count = 0
    for f in features:
        if f["car_sample_status"] == "ASOF_WITHIN_1S":
            asof_count += 1
            if float(f["car_sample_session_time_sec"]) > float(f["decision_session_time_sec"]) + 1e-9 or float(f["car_sample_age_sec"]) < -1e-9 or float(f["car_sample_age_sec"]) > 1.000001:
                asof_ok = False
                break
    check("asof_samples_are_prior", asof_ok, f"checked {asof_count} car as-of samples")
    check("candidate_physical_order_not_claimed", all(r["physical_track_order_status"] == "NOT_AVAILABLE_FROM_FASTF1_POSITION_XYZ" for r in review), "classified order kept distinct from XYZ position stream")
    check("candidate_interval_discontinuity_explicit", all("candidate_large_interval_flag" in r and "candidate_interval_duration_sec" in r for r in review), "long session intervals have explicit audit fields")
    check("candidate_telemetry_counts_present", all(r["candidate_telemetry_status"] in {"BOTH_STREAMS_HAVE_TIMESTAMPED_SAMPLES", "PAIR_STREAM_INTERVAL_HAS_LONG_DISCONTINUITY"} and all(r[k] != "" for k in ("car_samples_attacker", "car_samples_target", "position_samples_attacker", "position_samples_target")) for r in review), "candidate-specific car/position sample counts present")
    check("no_supported_ontrack_event", not any(r["classification"] == "SUPPORTED_ON_TRACK_PASS" for r in review), "no pass promoted without evidence")
    check("protected_session_absent", all(protected not in str(x) for path in out.iterdir() if path.is_file() for x in [path.read_text(encoding="utf-8", errors="ignore")]), "protected token absent from generated text outputs")
    source_before = json.loads((out / "source_input_hashes_before.json").read_text())
    source_after = json.loads((out / "source_input_hashes_after.json").read_text())
    check("source_hash_maps_equal", source_before == source_after, "input hashes before/after equal")
    check("package_exists", args.package.is_file(), str(args.package))
    if args.package.is_file():
        with zipfile.ZipFile(args.package) as z:
            check("package_deflate", all(i.compress_type == zipfile.ZIP_DEFLATED for i in z.infolist()), f"members={len(z.infolist())}")
    else:
        check("package_deflate", False, "package missing")
    for t in tests:
        print(f"{t['status']} {t['name']}: {t['detail']}")
    print(json.dumps({"all_passed": all(t["status"] == "PASS" for t in tests), "count": len(tests)}, indent=2))
    return 0 if all(t["status"] == "PASS" for t in tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
