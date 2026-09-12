# APEX-R model card: Car-data enriched XGBoost

Artifact: `2.0.0-xgboost-openf1-enriched`

## Training

- Algorithm: XGBoost binary classifier with validation early stopping
- Train sessions: 7779, 7787, 7953
- Evaluation/validation sessions: 9070
- Train rows / positives: 13912 / 1211
- Target: clean_overtake_next_60s
- Comparison group: Clean-label validation comparison

## Evaluation

| ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |
| ---: | ---: | ---: | ---: |
| 0.794901 | 0.166549 | 0.038203 | 0.957577 |

## Features

interval_sec, gap_to_leader_sec, closing_rate_sec_per_min, position, tyre_age, track_temperature_c, rainfall, race_progress, speed_mean_10s, speed_delta_10s, throttle_mean_10s, brake_fraction_10s, rpm_mean_10s, gear_mean_10s, drs_open_fraction_10s

## Hyperparameters

```json
{
  "n_estimators": 1500,
  "max_depth": 4,
  "learning_rate": 0.025,
  "min_child_weight": 8,
  "subsample": 0.9,
  "colsample_bytree": 0.9,
  "reg_alpha": 0.05,
  "reg_lambda": 6.0,
  "max_delta_step": 1,
  "objective": "binary:logistic",
  "eval_metric": "auc",
  "early_stopping_rounds": 100,
  "random_state": 2026,
  "n_jobs": 4
}
```

## Interpretation

- Adds rolling car telemetry features: speed, throttle, brake, RPM, gear and DRS.
- Location is cached/audited for replay provenance, but excluded from prediction because coordinates are local to each circuit.
