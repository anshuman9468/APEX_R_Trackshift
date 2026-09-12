# APEX-R model card: Logistic Regression baseline

Artifact: `1.0.0-real-openf1`

## Training

- Algorithm: Logistic Regression
- Train sessions: 7779, 7787, 7953
- Evaluation/validation sessions: 9070
- Train rows / positives: 21212 / 1935
- Target: overtake_next_60s
- Comparison group: Legacy-label hold-out comparison

## Evaluation

| ROC-AUC | Average precision | Brier score | Accuracy at 0.50 |
| ---: | ---: | ---: | ---: |
| 0.711676 | 0.122823 | 0.058589 | 0.937185 |

## Features

interval_sec, gap_to_leader_sec, closing_rate_sec_per_min, position, tyre_age, track_temperature_c, rainfall, race_progress

## Hyperparameters

```json
{
  "solver": "lbfgs",
  "max_iter": 2000,
  "actual_iterations": [
    12
  ],
  "C": 1.0,
  "class_weight": null,
  "random_state": 2026,
  "feature_scaling": "StandardScaler"
}
```

## Interpretation

- Uses the original observed-overtake label and is directly comparable only with the legacy XGBoost report.
- This model is retained as the explainable baseline.
