# APEX-R Phase 1 Dataset Audit

This is an audit package, not a cleaned or training-ready telemetry dataset. The source ZIPs are preserved outside this directory and were read without extraction or modification. The audit rejects unsafe ZIP member paths, checks ZIP CRCs, forces each nested GZIP stream to EOF, computes SHA-256 hashes, and performs a second row count with Python's `csv.reader`.

## Reproduce

```bash
python3 -m venv .venv_phase1_audit
source .venv_phase1_audit/bin/activate
python -m pip install --upgrade pip pandas numpy
python phase1_audit.py \
  --input-dir /home/anshumandutta/Downloads \
  --output-dir /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/phase1_audit_20260911 \
  --metadata-zip /home/anshumandutta/Downloads/APEX-R_Telemetry_Metadata.zip
```

The audit script accepts different absolute paths via the same arguments. It does not fetch telemetry, train a model, tune a model, create overtake labels, enrich data, or inspect sealed session 11353. `--legacy-combined-zip` is optional; when supplied, it is checked for packaging integrity and never used as a second data source. The legacy combined archive was not present at the supplied input path during this run, so it was not used or reverified.

## Outputs

* `PHASE1_AUDIT_REPORT.md` — evidence-backed human-readable findings.
* `phase1_metrics.json` — exact totals, integrity, duplicate, coverage, missingness, range and readiness results.
* `phase1_coverage.csv` — one row per observed selected lap with time/distance/sampling diagnostics.
* `phase1_feature_readiness.csv` — feature-by-feature SAFE/UNSAFE/UNKNOWN assessment.
* `phase1_anomaly_examples.csv` — traceable source-value examples; values are not changed.
* `PHASE2_HANDOFF.md` — prerequisites and join contracts for public enrichment.
* `source_inventory.json` — archive/member inventories and metadata snapshot.
* `SHA256SUMS.txt` — checksums for retained audit artifacts, excluding itself.

Source archives are intentionally not copied into this package.
