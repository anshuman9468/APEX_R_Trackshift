# APEX-R Phase 2 Handoff

Phase 1 found readable 2018–2022 selected-lap telemetry, but no UTC/session anchor, verified identity crosswalk, processing code, or source licence. Phase 2 must enrich the data without changing source values and without creating labels.

## Required public fields

* Weather: session identity, UTC timestamp (or a documented session-relative timestamp), track/air temperature, rainfall, humidity/pressure/wind where available.
* Pit and race context: driver identity, pit entry/exit or pit stop timestamp, pit-lane duration, stop duration, lap, race-control flags/safety-car/virtual-safety-car/red-flag intervals and messages.
* Tyres/stints: driver, lap or timestamp, compound, stint number and tyre age.
* Identity: season/event/session crosswalk, driver number/code and session start/lap-start anchors.

## Candidate sources and limitations

OpenF1 is a candidate source for public weather, pit, race-control, laps, intervals and positions where historical coverage exists. TracingInsights or the original upstream source may supply older rival/telemetry fields. The supplied archive itself does not prove that these sources share a clock or identifier namespace. Private ERS/battery percentage, team deployment/harvesting maps, fuel load, team instructions, radio and confidential undercut/overcut decisions remain unavailable from public timing data.

## Join contract

1. Resolve a race identity using `(season, event identity, session)` and a verified crosswalk; never join on event name alone.
2. Resolve driver identity using a verified code/number mapping. Source `driver` and `DriverAhead` are not independently verified identities.
3. Anchor the telemetry `lap` and apparent lap-relative `time` to UTC/session time using documented lap start or session start records. Do not infer this from a filename or an assumed race schedule.
4. Join telemetry to timestamped public data on `(session identity, driver, timestamp)`; use `lap` as a supporting key, not a substitute for a shared clock.
5. Use backward-only as-of joins. Proposed starting assumptions (must be sensitivity-tested, not treated as facts): telemetry channels ≤0.5 seconds old; OpenF1 interval rows ≤5 seconds old because interval cadence is coarser; weather ≤60 seconds old; pit/race-control events only at or before the observation and preferably exact event timestamps. Store `match_age_seconds`, `source_timestamp`, and `match_status`.
6. Never interpolate, backward-fill future values, or replace no-match with zero. Represent `missing`, `stale`, `ambiguous`, `unanchored` and `matched` explicitly; retain NaN for missing numeric values.

## Blockers and implementation order

* Blocker 1: obtain source repository URL/name, licence, processing code and version so raw vs derived fields and any interpolation/centred smoothing can be audited.
* Blocker 2: obtain UTC/session/lap anchors and a driver/event identity crosswalk; without them weather and pit joins are not reliable.
* Then add weather and public pit/race-control data in a separate enrichment table, run integrity and causal tests, and report match freshness/missingness before feature engineering.
* Only in Phase 3 should opponent matching, pit/retirement/lapping context and overtake labels be created. Genuine passes require opponent identity, event timestamp and surrounding windows; pit stops, retirements, lapping/unlapping, timing corrections and temporary swaps must be excluded or separately classified.

This handoff deliberately contains no labels, model scores or cleaned source values.
