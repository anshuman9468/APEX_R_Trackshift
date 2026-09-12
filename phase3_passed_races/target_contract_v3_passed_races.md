# Target contract v3.0 — passed-race continuous subset

## Scope

This is a historical, timing-derived proxy task over only races whose continuous FastF1 car and position streams were marked `PASS`. It is not a verified on-track overtake label and is not connected to APEX-R strategy actions.

## Decision and horizon

Create one decision window for each driver/lap boundary present in TracingInsights `laptimes.json`. At the decision boundary, the fixed target is the unique driver with classified position exactly one place ahead of the attacker in the same lap-boundary snapshot. The target never changes within the window. The horizon is the next lap boundary (`endpoint_lap = decision_lap + 1`) for that same pair; its duration is the observed session-time difference and is not forced to a fixed number of seconds.

## Proxy outcome

`PROXY_POSITIVE` means the fixed pair's classified positions reverse at the next boundary. `PROXY_NEGATIVE` means both endpoint positions are observed and the attacker remains behind the fixed target. This is a boundary-order proxy only: hidden pass/repass, lapping, unlapping and timing corrections are not ruled out. `true_ontrack_label` remains `UNKNOWN_CENSORED`.

## Eligibility and censoring

Windows without a unique adjacent target, an observed endpoint, a same-horizon public pit record, a race-control record on the decision lap, or sufficient boundary timing are `UNKNOWN_CENSORED`; they are not negatives. A race ending before the next boundary is censored. Public pit rows in this dataset provide lap/time-of-day evidence and do not prove publication latency or exact entry/exit timestamps.

## Causal feature rule

The feature view uses only the latest FastF1 car/position sample at or before the decision boundary, within an explicit 1.0-second freshness limit. No future sample, interpolation, backfill or full-horizon aggregate is used. Source measurement time is historical; live ingestion/publication latency is unknown. Outcome/evidence fields are kept outside the feature view.

## Evidence status

TracingInsights laptimes provides classified lap-boundary positions and timestamps. FastF1 `pos_data` provides XYZ/status location measurements, not classified running order or an overtake timestamp. Therefore supported on-track pass count is expected to remain zero unless independent event evidence is supplied.
