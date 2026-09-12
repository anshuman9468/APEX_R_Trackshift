# APEX-R engineered feature experiments — validation only

**Locked final hold-out session 11353 was not loaded.**

## Feature audit

| Feature | Point-biserial correlation | Mutual information |
| --- | ---: | ---: |
| lap_time_delta_ahead_minus_self | 0.032862 | 0.096707 |
| sector_3_delta_ahead_minus_self | -0.001962 | 0.084182 |
| sector_1_delta_ahead_minus_self | 0.007903 | 0.079241 |
| sector_2_delta_ahead_minus_self | 0.085905 | 0.077599 |
| tyre_age_delta_ahead_minus_self | 0.062528 | 0.018149 |
| speed_mean_10s | -0.008838 | 0.013832 |
| gap_volatility_60s | 0.055436 | 0.011827 |
| gap_slope_60s | 0.017095 | 0.011632 |
| throttle_mean_10s | -0.023232 | 0.010953 |
| gear_mean_10s | -0.018177 | 0.010783 |
| gap_slope_15s | 0.037721 | 0.010462 |
| stint_length_delta_ahead_minus_self | 0.062448 | 0.009414 |
| gap_slope_30s | 0.022845 | 0.009127 |
| gap_slope_5s | 0.04043 | 0.006492 |
| brake_fraction_10s | -0.005472 | 0.006131 |
| same_tyre_compound | -0.119342 | 0.004511 |
| brake_fraction_5s | -0.035209 | 0.003725 |
| speed_delta_5s | 0.032821 | 0.003024 |
| rpm_mean_10s | 0.006075 | 0.002411 |
| speed_delta_10s | 0.017606 | 0.002393 |
| new_stint_proxy | 0.066343 | 0.001667 |
| braking_approach_proxy | -0.042601 | 0.000958 |
| rival_drs_open | -0.033445 | 0.000774 |
| straight_approach_proxy | 0.029039 | 0.000414 |
| relative_pace_available | 0.020331 | 0.000403 |
| high_throttle_fraction_5s | 0.003017 | 0.000337 |
| drs_open_fraction_10s | -0.018036 | 0.000215 |
| drs_open | 0.012656 | 7.3e-05 |
| drs_eligible_or_open | -0.004219 | 9e-06 |

## Candidate comparison

| Candidate | ROC-AUC | AP | Brier | Best precision with recall ≥70% | Recall at selected point |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost engineered, scale_pos_weight=1.0 | 0.813796 | 0.170905 | 0.038155 | 0.096641 | 0.727626 |
| XGBoost engineered, scale_pos_weight=1.5 | 0.820597 | 0.201444 | 0.038715 | 0.109756 | 0.700389 |
| XGBoost engineered, scale_pos_weight=2.0 | 0.824588 | 0.206596 | 0.041459 | 0.113608 | 0.708171 |
| XGBoost engineered, scale_pos_weight=3.0 | 0.823136 | 0.203604 | 0.046523 | 0.112485 | 0.708171 |
| XGBoost engineered, scale_pos_weight=2.0, Platt calibrated | 0.824588 | 0.206596 | 0.037802 | 0.113608 | 0.708171 |
| Feed-forward MLP (64,32,16), class-weighted, L2 + early stopping; no dropout runtime | 0.582879 | 0.055759 | 0.043294 | 0.053428 | 0.700389 |
| XGBoost engineered interactions, scale_pos_weight=2.0 | 0.819668 | 0.199591 | 0.041235 | 0.112353 | 0.70428 |

## Recommendation before frozen holdout

**XGBoost engineered, scale_pos_weight=2.0, Platt calibrated** — selected by validation precision at recall ≥70%, then average precision and lower Brier score as tie-breakers.
Frozen offline candidate: `xgboost_engineered_scale2_platt_candidate.json`. It is intentionally not activated in the browser until full-session rival telemetry is available.

Do not score session 11353 until the user explicitly approves the frozen candidate and threshold.
