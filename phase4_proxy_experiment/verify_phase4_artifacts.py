#!/usr/bin/env python3
"""Independent post-run checks for Phase 4 artifacts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    results = []

    def check(name: str, passed: bool, detail: str) -> None:
        results.append({"test": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    metrics = json.loads((ROOT / "phase4_metrics.json").read_text(encoding="utf-8"))
    gate = json.loads((ROOT / "gate_results.json").read_text(encoding="utf-8"))
    features = pd.read_csv(ROOT / "../phase3_prediction_dataset/features/phase3_feature_view.csv.gz", compression="gzip", dtype=str, keep_default_na=False)
    labels = pd.read_csv(ROOT / "../phase3_prediction_dataset/labels/phase3_proxy_labels.csv.gz", compression="gzip", dtype=str, keep_default_na=False)
    repaired = pd.read_csv(ROOT / "repaired_prediction_examples.csv.gz", compression="gzip", dtype=str, keep_default_na=False)
    split = pd.read_csv(ROOT / "../phase3_prediction_dataset/phase3_split_manifest.csv", dtype=str, keep_default_na=False)
    check("source_input_counts_preserved", len(features) == len(labels) == len(repaired) == metrics["totals"]["source_examples"], f"features={len(features)}, labels={len(labels)}, repaired={len(repaired)}")
    check("repaired_feature_label_keys", set(features.example_id) == set(labels.example_id) == set(repaired.example_id), "all source and repaired example IDs match")
    check("repaired_eligibility_contract", int((repaired.new_eligibility_status == "ELIGIBLE_PROXY_ASOF").sum()) == metrics["totals"]["repaired_eligible"], "only as-of-supported rows are model-eligible")
    check("asof_unresolved_censored", int(((repaired.old_status == "ELIGIBLE_PROXY") & (repaired.new_eligibility_status != "ELIGIBLE_PROXY_ASOF")).sum()) == metrics["totals"]["asof_target_unresolved"], "previously eligible unresolved target rows are censored")
    check("feature_label_isolation", not set(__import__('phase4_experiment').FEATURES).intersection(set(labels.columns)), "model feature names do not enter label table")
    check("feature_view_has_no_outcome_fields", not set(__import__('phase4_experiment').FEATURES).intersection({"proxy_label", "event_id", "exclusion_reason"}), "allowlist excludes outcome evidence")
    check("split_manifest_race_isolation", split.race_id.nunique() == len(split) and repaired.groupby("race_id").split.nunique().max() <= 1, "one manifest split per race")
    check("gate_statuses_recorded", all(gate[key]["status"].startswith("PASS") for key in ("A", "B", "C", "D")), str({key: gate[key]["status"] for key in ("A", "B", "C", "D")}))
    check("no_true_ontrack_claim", (repaired.true_ontrack_label == "UNKNOWN_CENSORED").all(), "proxy rows do not claim verified on-track labels")
    metrics_table = pd.read_csv(ROOT / "split_metrics.csv", dtype=str, keep_default_na=False)
    check("both_candidates_and_splits_present", set(metrics_table.model) == {"constant_train_prevalence", "logistic_regression_C1"} and set(metrics_table.split) == {"train", "validation", "development_test"}, str(metrics_table[["model", "split"]].to_dict("records")))
    check("curves_present", sum(1 for _ in csv.DictReader(gzip.open(ROOT / "pr_curves.csv.gz", "rt", newline=""))) > 0, "precision/recall curve rows present")
    check("uncertainty_contains_degenerate_count", "class_degenerate_replicates" in set(pd.read_csv(ROOT / "uncertainty_by_race.csv", dtype=str).metric), "race bootstrap reports one-class resamples")
    check("saved_model_exists", (ROOT / "logistic_regression_pipeline.joblib").is_file(), "serialized pipeline present")
    model = joblib.load(ROOT / "logistic_regression_pipeline.joblib")
    check("saved_model_loads", hasattr(model, "predict_proba"), "serialized sklearn pipeline loads")
    parity = json.loads((ROOT / "inference_parity.json").read_text(encoding="utf-8"))
    check("saved_model_parity", parity.get("status") == "PASS" and parity.get("max_abs_probability_difference", 1) <= 1e-12, str(parity))
    check("source_hashes_unchanged", metrics["source_input_hashes_before"] == metrics["source_input_hashes_after"] and gate["invariants"]["source_hashes_unchanged"], "before/after input hashes agree")
    check("protected_access_ledger", metrics["access_ledger"]["protected_final_holdout_accessed"] is False and metrics["access_ledger"]["remote_requests_made"] is False, "protected data not accessed; no network requests")
    package = ROOT / "APEX-R_Phase4_Proxy_Experiment.zip"
    with zipfile.ZipFile(package) as archive:
        crc_ok = archive.testzip() is None
        member_names = set(archive.namelist())
    check("package_crc", crc_ok, "ZIP CRC test passed")
    check("package_contains_core_artifacts", {"PHASE4_REPORT.md", "EXPERIMENT_SPEC.md", "split_metrics.csv", "pr_curves.csv.gz", "SHA256SUMS.txt"}.issubset(member_names), "core files are inside package")
    sums = []
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        sums.append((digest, rel))
    sum_ok = all((ROOT / rel).is_file() and sha256_file(ROOT / rel) == digest for digest, rel in sums)
    check("sha256sums_verify", sum_ok and len(sums) >= 15, f"entries={len(sums)}")
    output = {"all_passed": all(item["status"] == "PASS" for item in results), "tests": results}
    (ROOT / "phase4_artifact_verification.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
