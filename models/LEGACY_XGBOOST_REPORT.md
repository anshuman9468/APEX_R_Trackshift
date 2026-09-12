# APEX-R model card: Legacy XGBoost

Artifact: `1.0.0-xgboost-openf1`

## Training

- Algorithm: XGBoost binary classifier
- Train sessions: 7779, 7787, 7953
- Evaluation/validation sessions: 9070
- Train rows / positives: 21212 / 1935
- Target: overtake_next_60s
- Comparison group: Legacy-label hold-out comparison

## Evaluation

| ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |
| ---: | ---: | ---: | ---: |
| 0.787855 | 0.193889 | 0.055747 | 0.933912 |

## Features

interval_sec, gap_to_leader_sec, closing_rate_sec_per_min, position, tyre_age, track_temperature_c, rainfall, race_progress

## Hyperparameters

```json
{
  "n_estimators": 300,
  "max_depth": 4,
  "learning_rate": 0.04,
  "min_child_weight": 8,
  "subsample": 0.9,
  "colsample_bytree": 0.9,
  "reg_alpha": 0.05,
  "reg_lambda": 5.0,
  "objective": "binary:logistic",
  "eval_metric": "logloss",
  "random_state": 2026,
  "n_jobs": 4
}
```

## Interpretation

- Uses the original observed-overtake label and is directly comparable only with Logistic Regression.
- It is not used to select the clean-label active model.
