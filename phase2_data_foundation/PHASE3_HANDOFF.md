# APEX-R Phase 3 Handoff

Phase 2 completed historical public-context enrichment for the supplied
2018–2022 selected-lap telemetry without changing source values, training a
model or creating labels.

## Permitted starting subset

Use the enriched dataset only through explicit statuses:

* identity: `phase2_driver_identity_status == VERIFIED_IN_SOURCE_DRIVERS_JSON`;
* session time: `phase2_timestamp_status` is `SESSION_ANCHORED_ONLY` or
  `UTC_ANCHORED`, depending on the label method;
* public weather: `phase2_weather_join_status == MATCHED` and age ≤ 60 seconds;
* race control: `phase2_rc_join_status == MATCHED` only as timestamped message
  context, never as an inferred continuously active state;
* pit context: `phase2_pit_*_status == MATCHED` for a completed stop on a
  strictly prior lap. Same-lap events are intentionally `UNKNOWN_SAME_LAP_EVENT`.

The exact counts are in `phase2_metrics.json` and
`phase2_join_coverage.csv`; do not infer coverage from null values alone.

## Label-engineering requirements

Before creating an overtake label, reconstruct a timestamp-safe opponent
identity stream. Require driver/session/lap keys and an absolute or verified
session-relative timeline. Exclude or separately classify pit-stop position
changes, retirements, lapping/unlapping, timing corrections, and temporary
swaps. Preserve an event id, event timestamp, attacker, opponent, and a
surrounding window. Do not call a row an independent example merely because it
is a telemetry observation.

## Known blockers

* The source archive has selected lap slices, not proven complete races.
* Live publication/ingestion latency is unknown for every public provider.
* Rows lacking `lSD` cannot be joined to UTC race-control timestamps without a
  separately verified session anchor.
* The exact source-processing build for the base archive is still unresolved;
  `rel_distance`, acceleration and opponent matching remain unresolved for
  causal use. Do not use them as verified causal features.
* Public pit-stop data supplies stop lap, time-of-day and duration where
  present; it does not supply proprietary pit-wall instructions, ERS/battery
  percentage, fuel load or team deployment maps.

## Required Phase 3 tests

1. Prefix-invariance tests for every derived feature used in labels.
2. No future joins and no same-lap pit-duration leakage.
3. Identity isolation by season/event/session/driver.
4. Event de-duplication and exclusion of non-overtake position changes.
5. Label-window coverage and missingness report before any model work.

No training or model scoring should begin until those tests and a race-separated
label manifest are complete.
