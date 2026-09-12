# APEX-R Phase 4 Historical Proxy Experiment Report

Generated: 2026-09-11T12:25:13.898518+00:00

## Outcome

**NO_DEMONSTRATED_GAIN**. This is an exploratory result for a lap-boundary position-swap proxy, not a verified overtake or strategy-effectiveness result.

## What was repaired

The source Phase 3 table contained 1332 decision rows and 790 originally eligible proxy windows. The first selected sample was checked against the stored lap-start anchors: PASS for 1332/1332 rows.
An as-of completed-order check supported the fixed target for 709 windows. The remaining 227 previously eligible windows were changed to UNKNOWN/CENSORED because the order was non-contiguous, conflicted with the fixed target, or lacked an as-of attacker record. The repaired primary population is 563 windows: 16 positives and 547 negatives.

The as-of check uses the latest completed laptime record with `session_end_sec <= decision time`; it is evidence for a historical boundary proxy, not proof of instantaneous on-track order or live availability.

## Gates

| Gate | Result | Key finding |
| --- | --- | --- |
| A — split support/independence | PASS_WITH_SUPPORT_WARNING | 16 positives, 56 races, whole-race partitions retained |
| B — decision/boundary correctness | PASS_WITH_CENSORED_UNRESOLVED_ROWS | 709 target matches; 227 prior eligible rows censored |
| C — future-dependent selection | PASS_WITH_CONDITIONAL_PRIMARY_POPULATION | Restricted proxy remains conditional; separate all-cause table created |
| D — evidence review | PASS_AUTOMATED_REVIEW_ONLY | Automated sheet includes all positives and all ambiguities; human review not claimed |

## Frozen experiment

Primary task: `phase4-repaired-asof-boundary-swap-v1`. Features: 23 Phase 3 approved historical-proxy fields; no unresolved relative-distance/acceleration or outcome fields. Preprocessing is fitted inside each training fit. Candidate budget: constant prevalence baseline plus one fixed logistic pipeline; no tuning, resampling, SMOTE, deep learning or automatic calibration.

The primary ranking metric is average precision. Thresholds in the curve files are descriptive. No deployment threshold is authorized because the positive support is too small.

## Validation and development-test results

Metrics are within this experiment and are not comparable to the earlier verified-overtake model, which used a different task and sealed race.

| Split | Model | Rows | Positives | AP | ROC-AUC | Brier | Log loss | Descriptive precision at recall ≥70% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | constant_train_prevalence | 353 | 7 | 0.0198 | 0.5 | 0.0194 | 0.0974 | 0.0198 @ 1.0000 (threshold 0.01983) |
| train | logistic_regression_C1 | 353 | 7 | 0.3575 | 0.9066886870355079 | 0.0171 | 0.0767 | 0.1087 @ 0.7143 (threshold 0.0539315) |
| validation | constant_train_prevalence | 117 | 5 | 0.0427 | 0.5 | 0.0414 | 0.1867 | 0.0427 @ 1.0000 (threshold 0.01983) |
| validation | logistic_regression_C1 | 117 | 5 | 0.0582 | 0.5178571428571428 | 0.0445 | 0.2043 | 0.0463 @ 1.0000 (threshold 0.00504543) |
| development_test | constant_train_prevalence | 93 | 4 | 0.0430 | 0.5 | 0.0417 | 0.1878 | 0.0430 @ 1.0000 (threshold 0.01983) |
| development_test | logistic_regression_C1 | 93 | 4 | 0.2418 | 0.9101123595505618 | 0.0396 | 0.1504 | 0.2727 @ 0.7500 (threshold 0.184862) |

The frozen selection rule chose `logistic_regression_C1` because its validation average precision was 0.058199 versus 0.042735 for the constant baseline. This selection is not evidence of a verified overtake gain.

## Race-level uncertainty

`uncertainty_by_race.csv` resamples whole races with replacement and compares the logistic candidate against the constant baseline on paired rows. Degenerate one-class resamples are counted and ROC-AUC intervals are therefore incomplete/fragile with this support.

## Access and preservation

Only cached Phase 3 evidence and the authorized development partitions were read. No remote requests were made. The protected final holdout was excluded before any data access. Input SHA-256 hashes were checked before and after and were unchanged.

## Readiness

**NO_DEMONSTRATED_GAIN** for a deployable APEX-R overtake predictor. Phase 4 is complete as an exploratory proxy experiment. The immediate blocker is not model complexity; it is target validity and small positive support. A future Phase 4b should use verified timestamped pass events and more independent positive races before any live decision logic is reconsidered.

See `EXPERIMENT_SPEC.md`, `gate_results.json`, `repaired_prediction_examples.csv.gz`, `pr_curves.csv.gz`, `split_metrics.csv`, `uncertainty_by_race.csv`, `MODEL_CARD.md` and `PHASE5_HANDOFF.md`.
