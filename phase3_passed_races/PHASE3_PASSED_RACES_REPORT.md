# Phase 3 passed-race continuous telemetry subset

Generated: 2026-09-11T15:33:20.318236+00:00

## Result

The race list was derived at runtime from `telemetry_coverage_status=PASS` in the completed collection status file. No remaining races were downloaded. This package does not alter the original collection, models or engine.

* PASS races: **37** of 70; non-PASS races left untouched: **33**.
* Existing split assignments were preserved: **{'train': 37}**. All selected races are training data; previously inspected development-test races remain development data, not an untouched final holdout.
* Car rows: **15,944,390**; position rows: **14,277,894**. These streams remain separate and are not counted as independent examples together.
* Unique drivers: car 29; position 29. Driver-race entries: car 740; position 740.
* Laps covered (driver-lap streams): car 2,318; position 2,318. Driver-hours: car 1066.944765; position 1066.939239.
* Gaps >5 seconds: car 19 totaling 110.960s; position 0 totaling 0.000s.

## Candidate evidence

The existing unresolved candidate set contains 2,095 rows; **1,195** match the derived PASS-race subset. Every candidate keeps its original ID, source URLs and source hashes in `position_swap_candidates_37.*`.

The automated review classifies 620 candidates as `OTHER_POSITION_CHANGE` where a public pit row overlaps the decision/endpoint lap, and 575 as `UNRESOLVED`. It claims **0 supported on-track passes**: FastF1 position data contains XYZ/status, not classified running order or a pass timestamp. No human or video review was performed. Event intervals are conservative boundary intervals with uncertainty recorded. The candidate telemetry audit found timestamped samples for both drivers in both streams for 1,192 candidates; 3 intervals are explicitly flagged as spanning a >600-second session discontinuity (4,883.481 seconds in the largest case).

## Target and windows

`target_contract_v3_passed_races.md` freezes the task before windows are interpreted. One window is generated per driver/lap boundary across all available passed-race periods, not only around inherited candidates. The fixed target is the unique classified direct predecessor at that boundary; it does not change during the horizon. The proxy horizon is the next lap boundary.

* Windows: **40,837**.
* `PROXY_POSITIVE`: **823**; `PROXY_NEGATIVE`: **14,600**; `UNKNOWN_CENSORED`: **25,414**.
* Independent supported on-track events: **0**. Positive/negative counts are boundary-order proxy counts only, not verified overtakes.
* Features use an as-of join to the latest telemetry sample at or before the boundary, maximum age **1.0 second**, no interpolation. Live publication/ingestion latency remains unknown.

## Checks

16/16 invariant checks passed. The detailed results are in `phase3_passed_tests.json`; input SHA maps before/after are in `source_input_hashes_*.json`.

## Artifacts

The two large telemetry CSV streams are package members and are created in a temporary directory during packaging so the original files are not duplicated in the workspace. `PACKAGE_VERIFICATION.json` is an external verification record because including it inside the ZIP would make a self-referential checksum.

## Readiness

**READY_FOR_PROXY_TASK_ONLY**. The subset is telemetry-ready and boundary-order proxy-label-ready, but not verified-overtake-ready. Model training must not treat the proxy as an on-track overtake target without explicitly naming the proxy task.
