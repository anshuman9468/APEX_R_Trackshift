# Phase 5 OpenF1 Pit Candidate — Validation Report

Generated: `2026-09-10T17:44:25.407999+00:00`

## Protocol

- Validation-only. The fixed training sessions are 7953, 7779, and 7787; the fixed validation session is 9070.
- No final holdout data was loaded, joined, scored, or used for selection.
- All existing model artifacts were checksum-protected and remained unchanged.
- XGBoost received numeric NaNs and missing categorical values directly; no zero/placeholder imputation was used.
- New pit-event joins use `session_key + driver_number` and a backward-only rule `pit.date <= prediction date`; tolerance is 0 seconds and interpolation is none.

## Critical data-availability result

OpenF1 `pit` returned no records for any approved train/validation session in this run. The four new columns are therefore 100% missing and cannot contribute learned signal on this split. They were preserved as NaN and not silently converted to zero.

| Session | Endpoint status | Records |
|---:|---|---:|
| 7953 | `http_404_no_records` | 0 |
| 7779 | `http_404_no_records` | 0 |
| 7787 | `http_404_no_records` | 0 |
| 9070 | `http_404_no_records` | 0 |

## Same-split comparison

The primary operating point is the maximum validation precision among points with recall at least 70%. Curves are exact, unrounded outputs saved in the compressed JSON experiment and flat CSV curve artifacts.

| Candidate | Features | Calibration | ROC-AUC | AP | Brier | ECE | Precision @ recall≥70% | Recall | Threshold |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline8-same-split-phase5-raw` | 8 | raw | 0.807565 | 0.154175 | 0.038285 | 0.012075 | 0.098752 | 0.708171 | 0.05596293 |
| `4.0.0-phase1-tracinginsights-existing` | 19 | Platt | 0.818115 | 0.188670 | 0.037707 | 0.020933 | 0.127298 | 0.700389 | 0.07526283 |
| `phase5-selected-w1-raw` | 23 | raw | 0.819668 | 0.186187 | 0.037604 | 0.011769 | 0.132469 | 0.715953 | 0.06664582 |

## New-candidate sweep

Every architecture and class-weight configuration tested is present in `phase5_openf1_pit_validation_experiments.json.gz`, including its complete precision/recall curve.

| Candidate | Calibration | ROC-AUC | AP | Brier | ECE | Precision @ recall≥70% | Recall | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `phase5-t75-d3-lr0.05-w1-raw` | raw | 0.819668 | 0.186187 | 0.037604 | 0.011769 | 0.132469 | 0.715953 | 0.06664582 |
| `phase5-t75-d4-lr0.04-w1-raw` | raw | 0.799882 | 0.171636 | 0.038048 | 0.012687 | 0.098901 | 0.700389 | 0.05388677 |
| `phase5-t75-d5-lr0.03-w1-raw` | raw | 0.786951 | 0.173146 | 0.037961 | 0.012121 | 0.091185 | 0.700389 | 0.05022121 |
| `phase5-t125-d3-lr0.03-w1-raw` | raw | 0.808723 | 0.180055 | 0.037814 | 0.013209 | 0.114431 | 0.700389 | 0.06350701 |
| `phase5-t125-d4-lr0.03-w1-raw` | raw | 0.785264 | 0.166257 | 0.038244 | 0.010243 | 0.092952 | 0.708171 | 0.04718189 |
| `phase5-t150-d3-lr0.025-w1-raw` | raw | 0.809777 | 0.181845 | 0.037751 | 0.013962 | 0.111043 | 0.700389 | 0.06100956 |
| `phase5-selected-w1-raw` | raw | 0.819668 | 0.186187 | 0.037604 | 0.011769 | 0.132469 | 0.715953 | 0.06664582 |
| `phase5-selected-w1-platt` | Platt | 0.819668 | 0.186187 | 0.037978 | 0.024584 | 0.132469 | 0.715953 | 0.08220490 |
| `phase5-selected-w1.5-raw` | raw | 0.821485 | 0.191151 | 0.038591 | 0.031700 | 0.126939 | 0.700389 | 0.09219541 |
| `phase5-selected-w1.5-platt` | Platt | 0.821485 | 0.191151 | 0.037859 | 0.024519 | 0.126939 | 0.700389 | 0.07882409 |
| `phase5-selected-w2-raw` | raw | 0.817830 | 0.188094 | 0.041172 | 0.050704 | 0.116883 | 0.700389 | 0.11311520 |
| `phase5-selected-w2-platt` | Platt | 0.817830 | 0.188094 | 0.038006 | 0.025697 | 0.116883 | 0.700389 | 0.07935610 |
| `phase5-selected-w3-raw` | raw | 0.812800 | 0.188761 | 0.045810 | 0.075457 | 0.107914 | 0.700389 | 0.13813557 |
| `phase5-selected-w3-platt` | Platt | 0.812800 | 0.188761 | 0.038030 | 0.023868 | 0.107914 | 0.700389 | 0.07391072 |

## New-candidate gain importance

| Rank | Feature | Group | Gain share |
|---:|---|---|---:|
| 1 | `tyre_age` | baseline | 16.705913% |
| 2 | `interval_sec` | baseline | 10.809913% |
| 3 | `race_progress` | baseline | 9.260354% |
| 4 | `gap_to_leader_sec` | baseline | 8.718647% |
| 5 | `track_temperature_c` | baseline | 8.243748% |
| 6 | `relative_lap_time_delta` | phase1 | 7.240500% |
| 7 | `position` | baseline | 5.575555% |
| 8 | `relative_sector_time_delta_s2` | phase1 | 4.955447% |
| 9 | `opponent_tyre_age_delta` | phase1 | 4.563910% |
| 10 | `stint_length_delta` | phase1 | 4.555285% |
| 11 | `relative_sector_time_delta_s1` | phase1 | 4.538199% |
| 12 | `opponent_tyre_compound_delta` | phase1 | 4.426792% |
| 13 | `distance_to_car_ahead_m` | phase1 | 3.282912% |
| 14 | `rival_pit_out_recent_flag` | phase1 | 2.938061% |
| 15 | `closing_rate_sec_per_min` | baseline | 2.586066% |
| 16 | `relative_sector_time_delta_s3` | phase1 | 1.598697% |
| 17 | `rainfall` | baseline | 0.000000% |
| 18 | `rival_drs_open` | phase1 | 0.000000% |
| 19 | `attacker_pit_out_recent_flag` | phase1 | 0.000000% |
| 20 | `attacker_openf1_pit_recent_flag` | openf1_pit | 0.000000% |
| 21 | `rival_openf1_pit_recent_flag` | openf1_pit | 0.000000% |
| 22 | `attacker_openf1_last_pit_lane_duration_sec` | openf1_pit | 0.000000% |
| 23 | `rival_openf1_last_pit_lane_duration_sec` | openf1_pit | 0.000000% |

## Recommendation

- Selected new candidate: `phase5-selected-w1-raw`.
- Selection point: precision `0.132469`, recall `0.715953`, threshold `0.06664582`.
- Same-split verdict: `new 5.0.0 candidate is the best same-split validation candidate`.
- Because OpenF1 pit data was unavailable for every approved session, this run does not establish a pit-data improvement. A later season/session set with actual pit records would be needed to measure that feature family.

## Artifacts

- New model: `models/candidates/5.0.0-phase1-openf1-pit-candidate.json`
- Full experiment and exact PR curves: `models/phase5_openf1_pit_validation_experiments.json.gz`
- Flat exact PR curves: `models/phase5_openf1_pit_precision_recall_curves.csv.gz`
- Pit feature audit: `data/phase1-features/phase1_openf1_pit_feature_manifest.json`
