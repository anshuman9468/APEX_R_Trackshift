# APEX-R Phase 2 Data Foundation Report

Run: `2026-09-11T10:36:17.725157+00:00`. Scope: supplied 2018–2022 archives only. The sealed OpenF1 session `11353` was guarded and not fetched or inspected.

## Executive result

Phase 2 implemented provenance recovery, session-specific identity crosswalks, historical lap/session-time anchors, and separate public weather, pit-stop and race-control context tables. The base telemetry values were not repaired, labels were not created, and no model was trained or scored.

The output is an enriched historical dataset, not a declaration of live causal training readiness. UTC and delivery-time availability remain incomplete or unproven for rows without source `lSD`; derived acceleration/opponent fields from the supplied archive remain unresolved.

## Verified totals and preservation

* Input observations: **1,027,303**; enriched observations: **1,027,303**.
* Input unique collector keys: **1,027,303**; enriched unique keys: **1,027,303**.
* Selected laps: **1,332**; race identities: **70**; driver codes: **34**; seasons: **2018, 2019, 2020, 2021, 2022**.
* The supplied non-empty `DriverAhead` values were observed as numeric driver numbers; Phase 2 resolves them through the session-specific `drivers.json` crosswalk. Driver numbers are not treated as stable identities across seasons.
* Row count and key invariants: **PASS**. Input source hashes unchanged: **PASS**.
* Independent `csv.reader` cross-check: **1,027,303** rows, **70** race identities, **1332** selected laps, **34** driver codes.

## Source/provenance recovery

* Pinned TracingInsights repositories are recorded per season with the five commits from the supplied metadata; per-session `drivers.json`, `laptimes.json`, `weather.json` and `rcm.json` retrieval URLs, retrieval timestamps, HTTP status and SHA-256 are in `provenance/source_resources.csv`.
* The pinned year repositories expose MIT repository licence metadata through GitHub; the supplied telemetry archive itself still did not contain a licence or redistribution grant. The package therefore excludes raw remote cache payloads and retains retrieval manifests/scripts instead.
* Pinned collector code was retrieved for provenance review. It shows FastF1-based lap/telemetry/weather/race-control collection and related source-code operations; the exact build that produced the supplied archive remains not proven because the archive has no processing-version identifier.
* The successful metrics run reused 1,647 locally cached payloads after the first retrieval pass; the cache records its creation time, while the initial network-request time was not persisted before the first pass hit a logging error. This is recorded as a provenance limitation, not presented as a false exact network timestamp.

## Context coverage

* Context rows: weather **9,797**, race control **5,794**, public pit stops **2,295**.
* identity: VERIFIED_IN_SOURCE_DRIVERS_JSON=1,027,303
* driver_ahead_identity: NO_CAR_AHEAD=54,822, VERIFIED_IN_SOURCE_DRIVERS_JSON=972,481
* timestamp: UNANCHORED=818, UTC_ANCHORED=1,026,485
* weather: MATCHED=1,015,006, STALE=11,479, UNKNOWN_UNANCHORED=818
* pit_current: MATCHED=201,334, NO_EVENT_CONFIRMED=741,285, SOURCE_EMPTY_UNVERIFIED=39,454, UNKNOWN_SAME_LAP_EVENT=45,230
* pit_ahead: MATCHED=164,869, NO_CAR_AHEAD=54,822, NO_EVENT_CONFIRMED=704,372, SOURCE_EMPTY_UNVERIFIED=37,151, UNKNOWN_SAME_LAP_EVENT=66,089
* race_control: MATCHED=356,706, STALE=669,779, UNKNOWN_UNANCHORED=818
* Row-level join counts are in `phase2_join_coverage.csv`. `SOURCE_EMPTY_UNVERIFIED` is retained for an empty response; it is not converted to no event or zero.
* Weather joins use `(season,event,session)` plus `lST + base time`, backward-only, maximum age 60 seconds. Race-control joins use the same race identity plus `lSD + base time`, backward-only, maximum age 60 seconds. Pit row features use exact driver code and strict `pit_lap < telemetry_lap`; same-lap events are withheld.

## Tests

* **PASS** `input_zip_and_gzip_integrity` — ZIP CRCs and gzip streams validated before row scan.
* **PASS** `source_hashes_unchanged` — SHA-256 before/after reads are identical.
* **PASS** `base_row_count_preserved` — base=1027303
* **PASS** `base_observation_keys_unique` — collector and canonical keys were unique in the source scan.
* **PASS** `enrichment_does_not_multiply_rows` — one output row emitted per input row.
* **PASS** `no_future_weather_joins` — join helper only accepts source_time_sec <= telemetry session time.
* **PASS** `no_future_race_control_joins` — join helper only accepts source UTC <= telemetry UTC.
* **PASS** `same_lap_pit_events_not_joined` — row features only inspect pit_lap < telemetry lap.
* **PASS** `source_unavailable_not_zero_filled` — unavailable/empty context yields status plus null context values.
* **PASS** `cross_session_context_isolation` — all maps are keyed by (season,event,session), then driver/lap where applicable.
* **PASS** `cross_driver_context_isolation` — pit and identity lookups require driver code; weather/race-control are global session context.
* **PASS** `race_control_scope_preserved` — rcm scope/sector/driver fields are retained in the separate table; no active state is inferred.
* **PASS** `pit_duration_availability` — duration is only exposed for strictly prior completed pit laps; entry/exit remain null because Jolpica did not supply them.
* **PASS** `holdout_exclusion_guard` — no remote source request or source record contains the sealed holdout token.

## What remains unresolved

1. The base archive does not carry a session UTC anchor for every selected lap; only source rows with `lSD` can be historically UTC-anchored. Missing anchors remain null/statused.
2. Historical measurement timestamps are not publication/availability timestamps. The pipeline cannot prove that a weather, pit or race-control value would have reached a live decision engine at the same time.
3. The supplied archive’s processing code/version is not fully recovered. The pinned related collector code contains centered convolution for acceleration, so future dependence remains a source-level risk until the exact build is identified.
4. Selected lap slices are still not complete races or independent prediction windows. No overtake labels were produced.

## Phase 3 gate

Only rows with explicit verified identity, required time anchor, non-stale context and no ambiguous match should enter later label engineering. The base archive can proceed to a controlled Phase 3 reconstruction only after selecting and documenting this subset; model training is not authorized by this Phase 2 output alone.

See `PHASE3_HANDOFF.md`, `phase2_feature_readiness.csv`, `phase2_anomaly_disposition.csv`, and `provenance/source_resources.csv` for the detailed evidence.
