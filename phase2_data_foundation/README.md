# APEX-R Phase 2 Data Foundation

This directory contains the reproducible Phase 2 implementation and audit
outputs. It preserves the supplied telemetry archives, adds public historical
context in separate tables, and does not train models or create labels.

## Environment

Use the scoped environment created for this project:

```bash
cd /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application
source .venv_tracinginsights/bin/activate
python -m pip install --upgrade requests
```

The script uses Python’s standard library plus `requests`. Raw remote JSON is
cached outside the distributable package under `cache/`; it is not needed to
read the already-produced CSVs and is excluded from `SHA256SUMS.txt` and the
delivery ZIP because redistribution terms for source payloads are not asserted
by the supplied telemetry archive.

## Re-run

```bash
python phase2_data_foundation/phase2_pipeline.py   --input-dir /home/anshumandutta/Downloads   --output-dir /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/phase2_data_foundation
```

The run is resumable for remote sources through the cache. It refuses unsafe
paths and refuses any remote request containing sealed holdout token `11353`.
It reads only the five supplied season archives and the metadata archive.

## Tests

```bash
python -m unittest discover -s phase2_data_foundation -p 'test_*.py' -v
```

## Outputs

* `enriched/telemetry_phase2_enriched.csv.gz` — exactly one row per preserved
  base observation, with explicit status columns and null unmatched values.
* `context/` — identity/time crosswalks and separate weather, pit-stop and
  race-control tables.
* `provenance/` — source retrieval manifest, pinned source code manifest,
  repository/blob verification and remote cache metadata.
* `phase2_join_coverage.csv`, `phase2_feature_readiness.csv`,
  `phase2_data_dictionary.csv`, `phase2_anomaly_disposition.csv`,
  `phase2_metrics.json` — machine-readable
  audit outputs.
* `PHASE2_REPORT.md`, `PHASE3_HANDOFF.md`, `README.md`, `SHA256SUMS.txt` —
  human handoff and integrity files.

Join contract: `(season,event,session)` is the race identity; driver joins use
the session-specific source code crosswalk; weather uses session seconds and a
backward 60-second maximum age; race control uses backward UTC and 60 seconds;
pit features use strictly prior completed laps. No interpolation, backward
fill, unlimited forward fill, zero-fill or state inference is performed.
