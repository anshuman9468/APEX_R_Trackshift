# Experiment Specification — Phase 4

## Frozen task

* Task version: `phase4-repaired-asof-boundary-swap-v1`
* Target: fixed attacker/target pair reverses endpoint positions at the next completed lap boundary.
* Eligibility: original Phase 3 restricted proxy, plus an as-of completed-order check at decision time. Rows whose target is not supported by a unique contiguous completed-order snapshot are UNKNOWN/CENSORED.
* Horizon: one completed lap boundary; variable duration.
* True on-track label: unavailable/censored.

## Data and splits

* Input: Phase 3 feature view, labels, examples, and cached laptime evidence.
* Whole-race partitions: train/validation/development_test from `phase3_split_manifest.csv`; no row-level random split.
* Repaired support: train 353 rows/7 positives; validation 117 rows/5 positives; development-test 93 rows/4 positives.

## Features

The exact model feature order is embedded in `model_artifact_metadata.json`: lap, time, phase2_session_time_sec, speed, rpm, gear, throttle, brake, drs, distance, phase2_weather_air_temp_c, phase2_weather_track_temp_c, phase2_weather_humidity_pct, phase2_weather_pressure_mbar, phase2_weather_rainfall_flag, phase2_weather_wind_direction_deg, phase2_weather_wind_speed_mps, phase3_weather_available_flag, phase3_weather_age_sec, phase3_pit_current_prior_stop_flag, phase3_pit_target_prior_stop_flag, phase3_rc_global_state_code, phase3_rc_global_state_known_flag. Outcome/evidence fields and identifiers are excluded. Unknown source units/encodings remain a documented limitation.

## Candidate budget and preprocessing

1. Constant train prevalence baseline.
2. Regularized unweighted logistic regression, `C=1.0`, L2, `liblinear`, max 5,000 iterations.

Median imputation with missingness indicators and standardization are fitted on training rows only. No class weighting, resampling, SMOTE, neural network, hyperparameter search or automatic calibration.

## Selection and evaluation

Primary metric: validation average precision. Development-test is scored once after selection. ROC-AUC, AP, Brier, log loss, complete precision/recall curves and race-bootstrap uncertainty are descriptive. No operational threshold is authorized; any recall-floor point is a descriptive curve statistic.

## Protected boundary

The protected final holdout was excluded using the approved exclusion manifest before any data access. No remote requests were made.
