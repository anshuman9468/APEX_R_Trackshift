# APEX-R model card: Clean-label base-feature XGBoost

Artifact: `2.0.0-xgboost-openf1-clean-base`

## Training

- Algorithm: XGBoost binary classifier with validation early stopping (clean-label base-feature control)
- Train sessions: 7779, 7787, 7953
- Evaluation/validation sessions: 9070
- Train rows / positives: 13912 / 1211
- Target: clean_overtake_next_60s
- Comparison group: Clean-label validation comparison

## Evaluation

| ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |
| ---: | ---: | ---: | ---: |
| 0.806429 | 0.167154 | 0.038128 | 0.957577 |

## Features

interval_sec, gap_to_leader_sec, closing_rate_sec_per_min, position, tyre_age, track_temperature_c, rainfall, race_progress

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

- Uses the cleaned label and the original eight interval/race-state features.
- This is the active browser model because it won the like-for-like ROC-AUC comparison.
