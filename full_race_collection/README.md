# APEX-R full-race collection

This package is the continuation of the approved 2018–2022 Race collection.
It materializes full available *lap-level* race context from the pinned
TracingInsights `drivers.json` and `laptimes.json` payloads already cached by
Phase 2/3, and reuses the cached public Jolpica pit-stop and TracingInsights
race-control context. It does not repackage raw source payloads.

## Run

From the application root:

```bash
python3 full_race_collection/collection_pipeline.py \
  --workspace /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application \
  --output /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/full_race_collection
```

The script requires only Python 3.10+ standard-library modules. It reads the
protected-session exclusion manifest before source access, makes no network
requests, verifies each reused cache payload by SHA-256, validates exactly
three train races, then expands only after that gate passes.

## Verify

```bash
python3 full_race_collection/verify_full_race_collection.py \
  --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/full_race_collection
python3 full_race_collection/test_full_race_collection.py \
  --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/full_race_collection
```

The package is built with ZIP DEFLATE. `SHA256SUMS.txt` hashes every shipped
file except itself and the ZIP. `package_verification.json` is retained beside
the ZIP because adding it after checksum verification would change the ZIP.

## Interpretation

“Complete” means every driver listed by that race’s `drivers.json` has a
readable laptimes payload beginning at lap 1 with contiguous observed lap
identifiers. It does **not** prove continuous sub-lap telemetry or a complete
physical lap. Public pit-stop and race-control rows are source-recorded public
events. Position reversals are deliberately named
`UNRESOLVED_POSITION_SWAP_CANDIDATE`; the source has lap-boundary positions but
no verified sub-lap overtake timestamp. No overtake labels or model work was
performed.
