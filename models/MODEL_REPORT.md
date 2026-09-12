# APEX-R: all machine-learning models

Active browser model: **Clean-label base-feature XGBoost**.

## How to read this report

There are two separate evaluation groups. Do not compare their scores directly: the clean-label group removes pit/neutralisation-adjacent examples and therefore has a different target distribution.

## Legacy-label hold-out comparison

Train: 7953, 7779, 7787. Held-out session: 9070. Features: interval, position, lap, stint and weather context.

| Model | ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.711676 | 0.122823 | 0.058589 | 0.937185 |
| Legacy XGBoost | 0.787855 | 0.193889 | 0.055747 | 0.933912 |

## Clean-label XGBoost comparison

Train: 7953, 7779, 7787. Validation session: 9070. Both models use the exact same cleaned labels and split; this is the valid selection comparison.

| Model | ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |
| --- | ---: | ---: | ---: | ---: |
| Clean-label base-feature XGBoost | 0.806429 | 0.167154 | 0.038128 | 0.957577 |
| Car-data enriched XGBoost | 0.794901 | 0.166549 | 0.038203 | 0.957577 |

## Data and label flow

- Base features: `intervals`, `position`, `laps`, `stints`, `weather`.
- Label source: `overtakes`; samples near `pit` and non-green/safety-car `race_control` records are removed in the clean-label group.
- Enriched feature source: `car_data`; `location` is retained for replay/audit provenance, not for the model.
- All API payloads are cached under `data/openf1-cache/` for reproducibility.

## Individual model cards

- `LOGISTIC_REGRESSION_REPORT.md`
- `LEGACY_XGBOOST_REPORT.md`
- `CLEAN_LABEL_BASE_XGBOOST_REPORT.md`
- `ENRICHED_MODEL_REPORT.md`

## Locked final hold-out test

Session: 11353; model: `2.0.0-xgboost-openf1-clean-base`.
ROC-AUC: 0.930631; average precision: 0.247008; Brier score: 0.026587; F1 at the validation-selected threshold: 0.262097.
See `FINAL_HOLDOUT_TEST_REPORT.md` for the locked-test protocol and confusion matrix.

## Important limitation

These models estimate overtake opportunity, not an ATTACK/HOLD/HARVEST/DEFEND label. APEX-R's strategy engine makes the final constrained action decision. Public OpenF1 data does not contain private ERS or battery telemetry.
