# APEX-R Phase 1 TracingInsights Feature Engineering Report

Generated: 2026-09-06

## Outcome and scope

The eight requested feature groups were built and attached to the existing OpenF1 train and validation rows. Because sector timing and pit-out state must remain separate, the eight groups produce 11 concrete columns. No model was trained, refit, calibrated, or scored.

- Train: sessions 7953, 7779, and 7787; 13,912 rows and 1,211 positive labels.
- Validation: session 9070; 6,058 rows and 257 positive labels.
- Combined engineering audit: 19,970 rows.
- The sealed final holdout is not accepted as an input by the builder and was not loaded, joined, inspected, or scored.
- OpenF1 remains the canonical source for each prediction row and UTC timestamp. TracingInsights adds rival state and telemetry.
- `interval_sec` remains in the output. `distance_to_car_ahead_m` supplements it and does not replace it.

## Files delivered

- `data/phase1-features/train_phase1_features.csv.gz`
- `data/phase1-features/validation_phase1_features.csv.gz`
- `data/phase1-features/phase1_feature_manifest.json`
- `scripts/build_phase1_tracinginsights_features.py`
- `tests/test_phase1_features.py`

The manifest contains machine-readable coverage, missingness reasons, real raw-input examples, source-race transfer totals, join policy, and causal-safety statements.

## Join and timestamp policy

Every output row starts from an existing OpenF1 interval row.

- Session key: a fixed, allowlisted mapping from OpenF1 `session_key` to TracingInsights `year / Grand Prix / Race`. There is no arbitrary session override.
- Driver key: OpenF1 `driver_number` equals TracingInsights `dNum`; the rival is TracingInsights telemetry field `DriverAhead` from the attacker's latest causal telemetry sample.
- Lap key: TracingInsights `lap`. Tyre/stint state activates at `lSD`, except a new stint is delayed until `pout` so new tyres cannot appear before pit exit.
- Timestamp key: the OpenF1 interval `date` is the prediction timestamp and canonical UTC time.
- Telemetry tolerance: latest TracingInsights sample at or before the prediction time, no more than 500 ms old. This is a backward-only as-of join. No future-nearest matching and no interpolation are allowed.
- Relative timing: latest common lap or sector for which both cars' completion timestamps are at or before the prediction timestamp. No fixed telemetry tolerance applies because these are explicitly lagged completed-event features.
- Pit window: a pit exit is “recent” from the `pout` timestamp through 90 seconds later, inclusive of the causal boundary.

The 500 ms tolerance was chosen for fast car-state telemetry: it is wide enough to bridge small sampling/clock-grid differences while remaining materially shorter than OpenF1's roughly four-second interval updates. A stale sample is left missing rather than carried forward. The builder has runtime assertions that fail if a selected telemetry or timing-completion timestamp is in the future.

## Feature 1 — opponent_tyre_age_delta

Definition: attacker `life` minus the ahead car's `life`, in laps of tyre use.

- TracingInsights source: `session_laptimes.json` fields `dNum`, `lap`, `life`, `stint`, `lSD`, and `pout`.
- OpenF1 source: interval row `session_key`, `driver_number`, and `date`.
- Join: session + driver + causal lap-state activation; rival identity comes from the attacker's backward telemetry `DriverAhead` sample.
- Time rule: tyre state must be active by the prediction time. New-stint state is not active before `pout`.
- Lookahead check: passed. A future lap state or pre-exit tyre state cannot be selected.
- Missingness: train 331/13,912 (2.38%); validation 349/6,058 (5.76%); combined 680/19,970 (3.41%). All combined missing rows had no causal car-ahead identity.
- Combined values: mean -0.155 laps; median 0; range -53 to 49.

Real examples:

- Session 7953, 15:27:03.662 UTC, attacker 1 vs rival 11: attacker `life=1`, rival `life=18`, result `-17`. The states activated at 15:26:59.976 and 15:26:45.863, both before prediction.
- Session 7779, 17:49:52.448 UTC, attacker 2 vs rival 27: `9 - 16 = -7`.

## Feature 2 — opponent_tyre_compound_delta

Definition: nominal attacker/rival compound pairing stored as `ATTACKER__VS__RIVAL`.

- TracingInsights source: `session_laptimes.json` fields `compound`, `dNum`, `lap`, `stint`, `lSD`, and `pout`.
- OpenF1 source and join: same canonical row, rival identity, state activation, and backward telemetry policy as Feature 1.
- Time rule and lookahead check: same causal state rule as tyre age; passed.
- Encoding rule: this is nominal, not ordinal. The next training step must one-hot encode it or use explicit native categorical handling. It must never be converted to ordered compound IDs.
- Missingness: train 331 (2.38%); validation 349 (5.76%); combined 680 (3.41%).
- Categories: nine pairings in train and six in validation; every validation category already exists in train.

Real examples:

- Session 7953, attacker 1 vs 11: `SOFT__VS__SOFT`.
- Session 7779, attacker 2 vs 27: `MEDIUM__VS__HARD`.

## Feature 3 — stint_length_delta

Definition: attacker's zero-based laps completed in the current race stint minus the rival's. The current stint's first lap has length zero.

- TracingInsights source: `session_laptimes.json` fields `dNum`, `lap`, `stint`, `lSD`, and `pout`.
- Computation: `lap - first_lap_for_that_driver_and_stint`, then attacker minus rival.
- Join and time rule: session + driver + active lap/stint state; rival from causal `DriverAhead`; new stint starts no earlier than pit exit.
- Lookahead check: passed. Future stint records are inaccessible.
- Missingness: train 331 (2.38%); validation 349 (5.76%); combined 680 (3.41%).
- Combined values: mean -0.187 laps; median 0; range -46 to 49.

Real examples:

- Session 7953, attacker 1 vs 11: current-stint lengths `0 - 14 = -14`.
- Session 7779, attacker 2 vs 27: `8 - 15 = -7`.

`life` and stint length intentionally differ: `life` can include prior-session use on a previously used tyre set, while stint length measures only the current race stint.

## Feature 4 — relative_sector_time_delta

Definition: attacker minus rival sector time, retained separately as `relative_sector_time_delta_s1`, `_s2`, and `_s3`. Negative means the attacker was faster in that sector.

- TracingInsights source: `session_laptimes.json`; values `s1`, `s2`, `s3`; completion markers `s1T`, `s2T`, `s3T`; lap/session timing fields `lap`, `lSD`, and `lST`.
- Join: session + attacker/rival driver pair + latest common lap with the requested sector completed by both drivers.
- Time rule: both sector completion times must be at or before the OpenF1 prediction `date`. The current, incomplete sector is never used.
- Quality rule: only clear-status, non-pit, non-generated, non-deleted timing rows are eligible; sector values must be greater than 0 and at most 60 seconds.
- Lookahead check: passed after explicit lagging. Using the current lap's final sector time before it completed would have leaked future information, so the feature uses the latest common completed sector instead.
- Combined missingness: S1 1,199 (6.00%); S2 1,292 (6.47%); S3 1,445 (7.24%). Missingness is car-ahead unavailable plus no common completed clean sector by prediction time.
- Train missingness: S1 5.92%, S2 6.05%, S3 6.91%.
- Validation missingness: S1 6.21%, S2 7.43%, S3 7.97%.
- Combined means: S1 +0.053 s, S2 +0.050 s, S3 +0.034 s.

Real example, session 7953 at 15:27:03.662 UTC, attacker 1 vs 11, common lap 13:

- S1: `31.389 - 31.570 = -0.181 s`; both completed by 15:23:59.383.
- S2: `42.907 - 43.298 = -0.391 s`; both completed by 15:24:42.681.
- S3: `24.186 - 24.468 = -0.282 s`; both completed by 15:25:07.149.

## Feature 5 — relative_lap_time_delta

Definition: attacker minus rival full-lap time for the latest common completed clean lap. Negative means the attacker was faster.

- TracingInsights source: `session_laptimes.json` fields `time`, `sesT`, `lST`, `lSD`, `lap`, and the sector fields used for quality validation.
- Join: session + attacker/rival driver pair + latest common lap completed by both.
- Time rule: both lap completion timestamps must be at or before prediction time.
- Quality rule: clear-status, non-pit, non-generated, non-deleted lap; duration in `(0, 180]` seconds; all three sector durations present and within `(0, 60]` seconds.
- Lookahead check: passed after lagging. A lap duration is not known until the lap finishes, so the in-progress lap is excluded.
- Missingness: train 1,091 (7.84%); validation 483 (7.97%); combined 1,574 (7.88%), the highest of the delivered columns but below the 15% warning threshold.
- Combined values: mean +0.140 s; median +0.110 s; range -3.867 to +3.624 s.

Real examples:

- Session 7953, common lap 13, attacker 1 vs 11: `98.482 - 99.336 = -0.854 s`; both laps completed more than two minutes before prediction.
- Validation session 9070, common lap 47, attacker 27 vs 20: `108.374 - 106.946 = +1.428 s`; both completion times precede prediction.

## Feature 6 — distance_to_car_ahead_m

Definition: attacker's real-time `DistanceToDriverAhead`, in metres. It is added beside the existing OpenF1 `interval_sec`.

- TracingInsights source: attacker `{driver}/{lap}_tel.json`, object `tel`, fields `time`, `DriverAhead`, and `DistanceToDriverAhead`.
- OpenF1 source: canonical interval `date`, session, and attacker `driver_number`.
- Join: session + attacker driver + latest telemetry timestamp at or before prediction, within 500 ms. The returned `DriverAhead` supplies the rival identity for all opponent-relative features.
- Interpolation: none.
- Lookahead check: passed. Nearest-future matching was explicitly rejected; stale or future samples become missing.
- Missingness: train 331 (2.38%); validation 349 (5.76%); combined 680 (3.41%). Combined reasons: 107 rows had no backward telemetry inside 500 ms; 573 telemetry samples reported no car ahead.
- Combined values: mean 185.68 m; median 93.55 m; range 0.059 to 2,549.655 m.

Real examples:

- Session 7953 at 15:27:03.662, attacker 1: telemetry at 15:27:03.575 was 87 ms old, `DriverAhead=11`, distance `451.450 m`.
- Session 7779 at 17:49:52.448, attacker 2: telemetry was 44 ms old, `DriverAhead=27`, distance `95.936 m`.

## Feature 7 — rival_drs_open

Definition: binary current DRS state of the identified car ahead. This is not DRS eligibility, detection-zone status, or track-zone geometry.

- TracingInsights source: ahead driver's own `{driver}/{lap}_tel.json`, object `tel`, fields `time` and `drs`; ahead identity comes from the attacker's `DriverAhead` sample.
- Join: session + ahead driver + latest rival telemetry at or before prediction, within 500 ms.
- Interpolation: none; only raw binary 0/1 states are accepted.
- Lookahead check: passed. Both attacker identity selection and rival DRS lookup are backward-only.
- Missingness: train 331 (2.38%); validation 349 (5.76%); combined 680 (3.41%). These rows had no causal car-ahead identity.
- Among available combined rows, DRS was open in 5.55% of samples.

Real examples:

- Session 7953 at 15:07:31.183, attacker 18 vs rival 77: rival sample was 92 ms old and `drs=1`.
- Session 7953 at 15:27:03.662, attacker 1 vs rival 11: rival sample was 87 ms old and `drs=0`.

## Feature 8 — pit_out_recent_flag

Two columns were built because both sides are derivable: `attacker_pit_out_recent_flag` and `rival_pit_out_recent_flag`.

- TracingInsights source: `session_laptimes.json` fields `pout`, `lST`, `lSD`, and `dNum`.
- Join: session + driver; rival identity comes from the causal attacker telemetry sample.
- Time rule: flag is 1 only when the latest translated pit-out timestamp is at or before prediction and no more than 90 seconds old.
- Lookahead check: passed. Future pit exits cannot set the flag.
- Attacker missingness: train 3 (0.02%); validation 1 (0.02%); combined 4 (0.02%), caused by unavailable attacker timing state.
- Rival missingness: train 331 (2.38%); validation 349 (5.76%); combined 680 (3.41%), caused by no causal car-ahead identity.
- Among available rows, attacker-recent is 1 in 3.18%; rival-recent is 1 in 2.65%.

Real examples:

- Session 7953 at 15:27:03.662, attacker 1: pit-out at 15:26:59.976, so attacker flag `1`.
- Session 7953 at 15:27:11.209, attacker 18 vs rival 1: rival pit-out at 15:26:59.976, so rival flag `1`.

## Missingness decision

No concrete feature exceeds 15% missingness in train, validation, or combined data. Therefore none should be dropped solely for coverage.

Recommended handling for the next, separate training task:

- Preserve numeric missing values and let XGBoost route them natively.
- Consider explicit availability indicators for lagged sector/lap features because “not yet jointly completed” is meaningful race state, not random missingness.
- Encode missing compound pairing as an explicit nominal category only if the training pipeline documents it.
- Do not zero-fill distance, relative pace, tyre deltas, or DRS state: zero has a real physical meaning and silent filling would corrupt it.

## Causal and data-quality risks found

- Future-nearest telemetry leakage: prevented with backward-only 500 ms joins.
- In-progress sector/lap leakage: prevented by requiring both cars' completion timestamps and lagging to the latest common completed item.
- New tyre appearing before pit exit: prevented by delaying new-stint activation to `pout`.
- Red-flag, pit, deleted, and generated timing rows: excluded from relative pace. This also removed physically implausible lap-time deltas discovered during validation.
- Duplicate telemetry timestamps around lap boundaries: resolved deterministically by preferring the higher lap number at an identical timestamp. This made repeated builds byte-for-byte reproducible.
- Compound encoding risk: retained as a nominal pair string and explicitly prohibited from ordinal encoding.

## Verification

- The builder recreated the expected clean row and positive-label counts for every approved session and aborts on a mismatch.
- Seven unit tests cover backward joins, stale-sample rejection, common-completion causality, bounded pit windows, the 8-to-11 column mapping, timing-row filters, and physical timing bounds.
- Full build was repeated. Decompressed output hashes matched exactly:
  - Train SHA-256: `c7e34a3fef7670285810835a73be976e7a943604ae1b60a4c3ad68f5c04f1c9e`
  - Validation SHA-256: `65bda7dd6084f20f842cfdca0736f4b188ec48ade89bdd043cd3b3a6ca140ea6`
- All 3,964 requested per-lap telemetry files were available across the four approved races.
- TracingInsights raw telemetry was processed in memory one race at a time and never persisted. Cumulative network transfer was 647,746,720 bytes (617.7 MiB); the largest one-race transfer was 190,114,381 bytes (181.3 MiB), so simultaneous downloaded race data remained well below 500 MB.

## Environment and retained footprint

- The scoped environment `.venv_phase1_features` used only `requests`, `pandas`, and their required dependencies.
- Pre-teardown venv size: 156 MB.
- Retained enriched datasets plus manifest: 1.1 MB.
- Raw downloaded race folders/files retained: 0 bytes.
- Final teardown status and post-cleanup footprint are recorded at the end of this report after final validation.

## Final teardown record

Final integrity checks passed and `.venv_phase1_features` was removed. The retained task artifacts occupy approximately 1.2 MB in total: 1.1 MB for the two enriched datasets and manifest, plus approximately 68 KB for the builder, tests, and this report. No raw TracingInsights telemetry or temporary race directory remains. The separate pre-existing `.venv_tracinginsights` environment was not modified.
