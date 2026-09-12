# APEX-R Phase 1 Dataset Audit Report

Audit run: `2026-09-11T08:54:21.125275+00:00`. Scope: supplied 2018–2022 archives only. Session `11353` was excluded by input-path guard and was not inspected.

## Executive conclusion

The separate yearly archives are readable and structurally consistent, but this is a processed, selected-lap telemetry export with incomplete provenance and no causal-processing code. It is not yet demonstrated training-ready. Causal model training should not begin from this package until timing/provenance, full-lap context, opponent matching and labels are reconstructed in later phases.

## 1. Verified totals

* Observations: **1,027,303**; unique collector-style `(source_data_key,sample_index)` keys: **1,027,303**; unique source-data lineage keys: **1,332**; unique canonical `(season,event,session,driver,lap,sample_index)` keys: **1,027,303**.
* Selected laps: **1,332**; race identities `(season,event,session)`: **70**; season/event names: **70**; driver codes: **34**; seasons: **2018, 2019, 2020, 2021, 2022**.
* Per-year rows: `{"2018": 191251, "2019": 189884, "2020": 216990, "2021": 235252, "2022": 193926}`.
* CSV member bytes: compressed `.gz` members **92,414,695**; decompressed CSV bytes **278,538,705**. Metadata byte agreement: compressed=True, uncompressed=True.
* Metadata agreement: `{"compressed_csv_bytes": true, "driver_codes": true, "race_events": true, "rows": true, "selected_laps": true, "uncompressed_csv_bytes": true, "unique_collector_sample_keys": true, "year_rows": {"2018": true, "2019": true, "2020": true, "2021": true, "2022": true}}`.

## 2. Integrity and packaging

PASS: all six separate ZIP CRC tests passed, expected member paths were safe, all five nested GZIP CSV streams decompressed to EOF, and the independent `csv.reader` row/header pass matched the pandas audit.

PASS: source SHA-256 values were unchanged before/after audit: **True**.

The metadata ZIP contains exactly `inventory.json`, `manifest.json`, and `validation.json`. Each yearly ZIP contains exactly one `telemetry_YYYY.csv.gz` member. No README, licence, collection script, processing script, repository URL, or processing-version string is present in the supplied archive. The inventory does document five commit hashes and included source blob hashes, but without a repository URL/name those identifiers cannot be independently resolved from the archive alone.

The optional earlier combined ZIP was not present at the supplied filesystem path during this run, so its integrity could not be reverified and it was not used. The authoritative separate archives contain no README, licence, scripts, or processing version files.

## 3. Coverage, independence and completeness

Race identity uses season + event + session, preventing same-named events in different seasons from being conflated. `source_data_key` is a repeated lap/file lineage key, so the collector-style observation key tested is `(source_data_key,sample_index)` and is compared with the canonical fallback key. Exact full-payload duplicate rows: **0**; repeated collector-sample-key rows: **0**; conflicting collector-sample keys: **0**; repeated canonical-key rows: **0**; conflicting canonical keys: **0**.

The rows are telemetry observations, not independent prediction windows. No overtake labels or unique overtake events were generated or estimated. Driver codes remain source identifiers; no identity crosswalk is packaged.

There are 70 race identities, 1,332 selected laps, and multiple drivers per race, but the archive contains only selected lap slices. `phase1_coverage.csv` reports observed time/distance spans, timestamp repetition/order, sample-index order, and positive within-lap intervals. No independent lap start/end metadata, track-length reference, or selection/collection code is supplied, so complete physical laps are **not proven**. Lap-gap statistics are descriptive of selected identifiers and do not prove that omitted laps are irrelevant.

The exact season/race/driver coverage summaries are retained in `phase1_metrics.json`: `{"2018": {"driver_codes": 20, "race_identities": 14, "rows": 191251, "selected_laps": 267}, "2019": {"driver_codes": 20, "race_identities": 14, "rows": 189884, "selected_laps": 267}, "2020": {"driver_codes": 23, "race_identities": 14, "rows": 216990, "selected_laps": 266}, "2021": {"driver_codes": 21, "race_identities": 14, "rows": 235252, "selected_laps": 266}, "2022": {"driver_codes": 22, "race_identities": 14, "rows": 193926, "selected_laps": 266}}`. The per-driver table in that file records selected laps, rows and race identities; this is coverage, not independent-example count.

## 4. Time, units and alignment

`time` resets near zero inside selected laps and has no UTC or session-start anchor; lap-relative seconds is the supported interpretation. Sampling intervals were calculated only within each season/event/session/driver/lap. Actual ingestion/delivery latency is unknown. A reliable weather or public pit-event join therefore requires an external session/lap UTC anchor and verified driver identity mapping.

The archive has no authoritative unit/codebook. Speed values are numerically plausible as a km/h-like channel, `rel_distance` is ratio-like, and coordinates vary by event/year, but these observations are not proof of units or semantics. `drs` codes require a source codebook. `acc_x/y/z` and `rel_distance` look potentially derived, but the processing formula is absent.

## 5. Missingness and anomalies

Missing values (empty strings plus null-like tokens) are: `{"DistanceToDriverAhead": 48648, "DriverAhead": 54822, "rel_distance": 4665}`. The supplied claims for DriverAhead, DistanceToDriverAhead and rel_distance should be compared against these exact values in `phase1_metrics.json`.

Explicit audit-rule flags: `{"distance_negative": 1548, "gear_not_integer_or_outside_0_8": 37, "rel_distance_outside_0_1_tolerance": 194, "throttle_outside_0_100": 16052}`. The throttle rule is `<0 or >100`; gear is non-integer-valued or outside `0..8`; brake is outside `0..1`; negative physical channels and ratio-like `rel_distance` are separately recorded. These are flags, not proof of source error. Source values remain unchanged. Examples with source identifiers are in `phase1_anomaly_examples.csv`.

In particular, a throttle value of 104 cannot be called wrong without the provider's encoding contract. Gear values outside the ordinary 0–8 expectation likewise need source semantics; a count under an arbitrary rule is not a repair.

## 6. Prediction-time readiness and leakage

Overall `causal_training_ready` is **false**. The supplied archive lacks processing code, so prefix invariance cannot be honestly reconstructed for `rel_distance`, acceleration, opponent matching or any normalization/smoothing. Those fields are UNKNOWN, not silently declared safe. Direct observed channels are available at their recorded sample in the file, but delivery latency and UTC alignment are unknown. `phase1_feature_readiness.csv` contains the feature-by-feature evidence and repairs required.

No demonstrated UNSAFE transformation was available to prove a specific future leak; however, absent code means possible interpolation, centred smoothing, backward filling, whole-lap statistics, distance normalization and future-aware opponent matching remain unresolved. A passing check on a few rows would not establish universal safety.

## 7. Readiness decision

**NO for causal model training from this archive alone.** Readability and large row count passed, but causal availability, independent examples, complete race context, event alignment, opponent identity, source provenance and labels are not established. Phase 1 is complete as an audit; training readiness is a later handoff condition, not a claim made here.

## Prioritised next actions

1. Recover the repository URL/name, exact processing scripts/version and licence; rebuild a provenance manifest and verify the five commit hashes.
2. Obtain a verified driver/event/session crosswalk plus session UTC and lap-boundary anchors; reconstruct full-race context and test prefix invariance of all derived/opponent fields.
3. In Phase 2, add public weather, pit-stop and race-control data with backward-only joins and explicit freshness/missing-match fields; only then proceed to causal feature engineering and defensible labels.

See `PHASE2_HANDOFF.md` for join contracts, assumptions and proprietary-data blockers. No model was trained or scored in this audit.
