# APEX-R model-training master report

Generated: 2026-09-12  
Scope: every completed model-training experiment currently recorded in this project, including the latest RTX 3050 GNN run.

## Executive summary

APEX-R contains three different prediction tasks. Their scores must not be compared as if they measure the same thing.

| Experiment family | Target | Evaluation status | May be called verified overtaking? |
| --- | --- | --- | --- |
| OpenF1 opportunity models | Cleaned or legacy observed overtake-in-next-60-seconds label | Session 9070 validation; the clean 8-feature model also has one locked score on session 11353 | Only as an observed OpenF1 overtake-opportunity proxy |
| Feature-engineering candidates | The same cleaned OpenF1 60-second opportunity label | Validation only; session 11353 was not used | No final-test result for these candidates |
| Continuous-telemetry proxy models | Fixed attacker/target classified-order reversal at the next lap boundary | Chronological development train/validation/development-test data | No. This is not a timestamped physical pass label |

The only model with a sealed final-holdout score is the clean-label 8-feature XGBoost model. The final holdout was session 11353 and was intentionally not reused for newer candidates.

The latest GPU GNN improved the historical boundary-swap proxy over its constant and logistic baselines, but it does not validate an overtake predictor, ATTACK/HOLD/HARVEST logic, energy strategy, or private ERS behaviour.

## Available versus unavailable F1 signals

The audited continuous-telemetry graph experiments use speed, physical proximity, relative speed, track-frame XYZ position, timing/sample ages, 10-second speed trends, throttle, brake, gear, RPM, DRS and explicit missingness masks.

The following requested signals are not measured in the approved public historical data and were never fabricated as model features: ERS/energy delta, real power deployment, battery state-of-health, battery/ES temperature and a causally safe tyre-wear stream. The simulator has modelled energy/battery variables, but those are not historical measurements and were excluded from supervised training.

## Data-preparation work that led to the experiments

1. The original 2018–2022 telemetry audit verified 1,027,303 observations, 1,332 selected laps, 70 race identities and 34 driver codes. It also concluded that the selected-lap source alone was not causally training-ready.
2. Targeted OpenF1 and TracingInsights work added causal rival-car feature columns for the small OpenF1 opportunity datasets. The resulting Phase 1 candidate used 19 model columns.
3. The OpenF1 pit integration was implemented and tested, but the approved historical sessions returned no pit records. Those fields remained missing rather than being zero-filled.
4. Continuous telemetry collection then validated 37 races with both car and position streams. It retained 15,944,390 car readings and 14,277,894 position readings.
5. The continuous-telemetry evidence audit found no independently verified timestamped physical on-track passes in the collected candidate set. It therefore produced fixed-pair boundary-order proxy labels instead of claiming ground-truth overtakes.

## 1. Original OpenF1 60-second opportunity models

Training sessions: 7953, 7779 and 7787.  
Evaluation session: 9070.  
Original training population: 21,212 rows, 1,935 positives.  
Features: interval_sec, gap_to_leader_sec, closing_rate_sec_per_min, position, tyre_age, track_temperature_c, rainfall and race_progress.

### Legacy observed-label comparison

Target: overtake_next_60s using the original observed-overturn label procedure.

| Model | Algorithm and key settings | ROC-AUC | AP | Brier | Accuracy at 0.50 | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1.0.0-real-openf1 | Logistic regression; StandardScaler; lbfgs; C=1.0; 12 actual iterations | 0.711676 | 0.122823 | 0.058589 | 0.937185 | Explainable baseline |
| 1.0.0-xgboost-openf1 | XGBoost; 300 trees; depth 4; learning rate 0.04 | 0.787855 | 0.193889 | 0.055747 | 0.933912 | Better than legacy logistic on this split |

These two are comparable only with each other because they use the same legacy label and session 9070.

### Clean-label comparison

Target: clean_overtake_next_60s. Windows around available pit and non-green/neutralisation race-control records were excluded.  
Clean training population: 13,912 rows, 1,211 positives.  
Clean validation population: 6,058 rows, 257 positives.

| Model | Features | ROC-AUC | AP | Brier | Accuracy at 0.50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2.0.0-xgboost-openf1-clean-base | Original 8 features; XGBoost with early stopping | 0.806429 | 0.167154 | 0.038128 | 0.957577 | Browser-active baseline |
| 2.0.0-xgboost-openf1-enriched | 8 base features plus rolling speed, throttle, brake, RPM, gear and DRS features | 0.794901 | 0.166549 | 0.038203 | 0.957577 | Did not beat the base model on this validation split |

The base XGBoost configuration permits up to 1,500 trees with early stopping. The saved locked-test model contains 75 trees.

### One-time locked final holdout: clean 8-feature XGBoost

Model: 2.0.0-xgboost-openf1-clean-base.  
Sealed test session: 11353. It was not used for training, validation, feature selection or threshold selection.  
Clean test population: 2,583 rows, 77 positive windows, 2.98% prevalence.

| ROC-AUC | AP | Brier | Accuracy | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.930631 | 0.247008 | 0.026587 | 0.858304 | 0.155131 | 0.844156 | 0.262097 |

The operating threshold was 0.09427804, selected earlier on validation to maximise F1. Test confusion matrix: TN 2,152; FP 354; FN 12; TP 65. At the arbitrary 0.50 threshold the model predicted no positives. This is one honest race-level result, not a universal 93% claim.

Source reports: FINAL_HOLDOUT_TEST_REPORT.md, CLEAN_LABEL_BASE_XGBOOST_REPORT.md, ENRICHED_MODEL_REPORT.md, LOGISTIC_REGRESSION_REPORT.md and LEGACY_XGBOOST_REPORT.md.

## 2. Earlier engineered 37-feature validation experiment

This validation-only experiment used the cleaned OpenF1 task and session 9070. Session 11353 was not loaded.

| Candidate | ROC-AUC | AP | Brier | Best precision with recall at least 70% | Recall at selected point |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost engineered, scale_pos_weight 1.0 | 0.813796 | 0.170905 | 0.038155 | 0.096641 | 0.727626 |
| XGBoost engineered, scale_pos_weight 1.5 | 0.820597 | 0.201444 | 0.038715 | 0.109756 | 0.700389 |
| XGBoost engineered, scale_pos_weight 2.0 | 0.824588 | 0.206596 | 0.041459 | 0.113608 | 0.708171 |
| XGBoost engineered, scale_pos_weight 3.0 | 0.823136 | 0.203604 | 0.046523 | 0.112485 | 0.708171 |
| XGBoost engineered, weight 2.0 plus Platt calibration | 0.824588 | 0.206596 | 0.037802 | 0.113608 | 0.708171 |
| Feed-forward MLP, 64/32/16, weighted loss and early stopping | 0.582879 | 0.055759 | 0.043294 | 0.053428 | 0.700389 |
| XGBoost with explicit interaction terms, weight 2.0 | 0.819668 | 0.199591 | 0.041235 | 0.112353 | 0.704280 |

Selected candidate: XGBoost engineered, scale_pos_weight 2.0 with Platt calibration. It was not promoted to the application because its strongest rival-car features were not available in the earlier live-app source.

The neural network was trained, but it was clearly weaker than XGBoost on this data and task.

Source report: ENGINEERED_MODEL_EXPERIMENTS.md.

## 3. Phase 1 targeted TracingInsights feature candidate

Task: same cleaned OpenF1 60-second opportunity label.  
Features: 8 baseline features plus 11 causal Phase 1 columns, including relative sector/lap-time deltas, opponent tyre state, distance to car ahead, rival DRS state and recent-pit flags.  
Training: 13,912 rows, 1,211 positives. Validation: 6,058 rows, 257 positives.  
No missing numeric feature was zero-filled; XGBoost handled NaN values directly.

The sweep recorded 27 fitted configurations, including class-weight and calibration comparisons. The selected candidate was:

| Candidate | Settings | ROC-AUC | AP | Brier | Precision at recall at least 70% | Recall | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4.0.0-phase1-tracinginsights-candidate | 125 trees; depth 3; learning rate 0.03; scale_pos_weight 2; Platt scaling from race-grouped OOF predictions | 0.818115 | 0.188670 | 0.037707 | 0.127298 | 0.700389 | 0.07526282 |

Fair same-validation comparison on the predeclared primary metric:

| Candidate | Precision at recall at least 70% | Recall |
| --- | ---: | ---: |
| Phase 1 targeted candidate | 12.73% | 70.04% |
| Earlier 37-feature candidate | 11.36% | 70.82% |
| Same-split 8-feature comparator | 9.75% | 76.26% |

The original 15.50% precision number is from the sealed session 11353 and is not a valid validation-ranking comparison. The Phase 1 candidate was never scored on the sealed final holdout.

Source reports: PHASE1_TRACINGINSIGHTS_MODEL_VALIDATION_REPORT.md and PHASE1_VALIDATION_RECOMMENDATION_CORRECTION.md.

## 4. Phase 5 OpenF1 pit-feature candidate

Task: same cleaned OpenF1 60-second opportunity label.  
Features: 8 baseline plus 11 Phase 1 columns plus 4 intended OpenF1 pit columns.  
Validation-only; sealed session 11353 was not loaded.

Critical result: OpenF1 pit returned zero records for every approved train/validation session in this run. All four pit columns were 100% missing, retained as missing values, and had zero gain. Therefore this experiment does not demonstrate a pit-data benefit.

The sweep recorded 17 XGBoost configurations. The selected candidate was:

| Candidate | Settings | ROC-AUC | AP | Brier | ECE | Precision at recall at least 70% | Recall | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.0.0-phase1-openf1-pit-candidate | 75 trees; depth 3; learning rate 0.05; weight 1.0; raw probabilities | 0.819668 | 0.186187 | 0.037604 | 0.011769 | 0.132469 | 0.715953 | 0.06664582 |

Same-split references:

| Candidate | ROC-AUC | AP | Brier | Precision at recall at least 70% |
| --- | ---: | ---: | ---: | ---: |
| 8-feature Phase 5 comparator | 0.807565 | 0.154175 | 0.038285 | 0.098752 |
| Existing Phase 1 candidate | 0.818115 | 0.188670 | 0.037707 | 0.127298 |
| New 5.0 candidate | 0.819668 | 0.186187 | 0.037604 | 0.132469 |

The 5.0 candidate is the best same-split validation candidate on its declared primary metric. It has not been evaluated on the sealed final holdout, and its improvement cannot be attributed to pit data because pit records were unavailable.

Source report: PHASE5_OPENF1_PIT_MODEL_VALIDATION_REPORT.md.

## 5. Small Phase 4 historical proxy logistic-regression experiment

This is a different task from the OpenF1 opportunity models.

Target: fixed-pair classified-order reversal at the next lap boundary. It is a historical proxy, not a verified physical pass.  
Repaired primary population: 563 windows, 16 positive and 547 negative.  
Training: 353 windows, 7 positives. Validation: 117 windows, 5 positives. Development test: 93 windows, 4 positives.  
Model: regularized logistic regression, C=1.0, L2 penalty, training-only median imputation, missingness indicators and training-only standardisation. No resampling or calibration.

| Split | Model | Rows | Positives | AP | ROC-AUC | Brier | Log loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation | Constant prevalence | 117 | 5 | 0.0427 | 0.5000 | 0.0414 | 0.1867 |
| Validation | Logistic regression C=1 | 117 | 5 | 0.0582 | 0.5179 | 0.0445 | 0.2043 |
| Development test | Constant prevalence | 93 | 4 | 0.0430 | 0.5000 | 0.0417 | 0.1878 |
| Development test | Logistic regression C=1 | 93 | 4 | 0.2418 | 0.9101 | 0.0396 | 0.1504 |

Conclusion: NO_DEMONSTRATED_GAIN. The positive support is too small and the endpoint can contain lapping, retirement, timing-correction or pass/repass effects. No threshold is authorised for deployment.

Source report: phase4_proxy_experiment/PHASE4_REPORT.md.

## 6. 37-race continuous-telemetry GNN experiment

This is also a different historical proxy task: fixed attacker/target classified-order reversal at the next lap boundary. It is not a verified overtake label.

Input context: 37 telemetry-coverage-passed races, 15,944,390 car readings and 14,277,894 position readings. Raw readings are context for graph construction; they are not 30 million independent labelled examples.

Graph eligibility:

| Split | Races | Graphs | Positives | Negatives |
| --- | ---: | ---: | ---: | ---: |
| Development train | 26 | 10,671 | 613 | 10,058 |
| Development validation | 6 | 2,533 | 103 | 2,430 |
| Development test | 5 | 2,200 | 106 | 2,094 |
| Total | 37 | 15,404 | 822 | 14,582 |

Nineteen labelled windows were excluded because fixed attacker/target current telemetry was missing. Unknown/censored windows were excluded, not converted to negatives. Development-test races are previously accessed development data, not a sealed final holdout.

### Frozen GNN configuration

- Two edge-aware GINEConv layers; hidden width 32; dropout 0.2.
- Adam learning rate 0.001; weight decay 0.0001.
- Unweighted BCE-with-logits; batch size 32; maximum 100 epochs; patience 15.
- Backward-only one-second as-of joins; 10-second trailing lookback.
- Three predeclared seeds: 17, 23 and 42.
- Logistic C=1.0 and constant-prevalence baselines used identical eligible windows.

### CPU RAM-safe GNN run

This run used the disk-backed graph store. Seed selection is based on validation AP, so seed 17 was the validation winner.

| Model | Validation AP | Validation ROC-AUC | Development-test AP | Development-test ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Constant | 0.040663 | 0.500000 | 0.048182 | 0.500000 |
| Logistic C=1 | 0.089998 | 0.692301 | 0.077608 | 0.631805 |
| GNN seed 17 | 0.120806 | 0.709765 | 0.113141 | 0.686706 |
| GNN seed 23 | 0.106054 | 0.661193 | 0.102951 | 0.667059 |
| GNN seed 42 | 0.117812 | 0.694195 | 0.106208 | 0.702501 |

Epochs actually run after early stopping:

| Seed | Best validation epoch | Epochs run | Best validation AP |
| ---: | ---: | ---: | ---: |
| 17 | 24 | 39 | 0.120806 |
| 23 | 21 | 36 | 0.106054 |
| 42 | 22 | 37 | 0.117812 |

### RTX 3050 CUDA GNN run

This is a separate repeat with the same frozen model/data configuration and CPU-to-CUDA as the only intentional execution change. CUDA numerical kernels need not produce bit-identical results.

| Model | Validation AP | Validation ROC-AUC | Development-test AP | Development-test ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Constant | 0.040663 | 0.500000 | 0.048182 | 0.500000 |
| Logistic C=1 | 0.089998 | 0.692301 | 0.077608 | 0.631805 |
| GNN seed 17 | 0.123615 | 0.692489 | 0.099680 | 0.667189 |
| GNN seed 23 | 0.108146 | 0.685377 | 0.088548 | 0.652745 |
| GNN seed 42 | 0.127602 | 0.689272 | 0.104338 | 0.700857 |

The GPU validation winner is seed 42. Its validation-selected threshold was 0.18466353. On development-test, frozen at that threshold: precision 0.130435, recall 0.141509, F1 0.135747, TP 15, FP 100, FN 91 and TN 1,994.

GPU epochs actually run:

| Seed | Best validation epoch | Epochs run | Best validation AP |
| ---: | ---: | ---: | ---: |
| 17 | 33 | 48 | 0.123615 |
| 23 | 16 | 31 | 0.108146 |
| 42 | 28 | 43 | 0.127602 |

Conclusion: EXPLORATORY_PROXY_GAIN only. The GNN has a stronger proxy signal than the proxy baselines, but it must not be represented as a verified overtake model, a 90% ROC model, or an energy-strategy model.

Source reports: gnn_proxy_experiment_v1/PHASE4_REPORT.md and gnn_proxy_experiment_gpu_v1/GPU_RUN_REPORT.md.

## What can be used now

| Item | Status | Reason |
| --- | --- | --- |
| Clean 8-feature XGBoost overtake-opportunity model | Existing application baseline | It is the only candidate with one locked session-11353 result |
| 4.0 targeted Phase 1 candidate | Validation-selected candidate only | Never final-tested; dependent on added rival telemetry |
| 5.0 pit-feature candidate | Validation-selected candidate only | Never final-tested; actual pit fields were entirely missing |
| Historical proxy logistic model | Research artifact only | Too few positives and proxy endpoint |
| CPU/GPU GNN models | Research artifact only | Boundary-order proxy, not verified overtaking |
| Energy/ERS strategy model | Not trained | Required measured power, battery, SOH and energy-temperature signals are absent |

## Main limitations and next step

1. Do not compare ROC-AUC or AP across the three task families. Different labels, rows, horizons and race splits make those rankings invalid.
2. Do not add unknown/censored windows as negatives merely to make the graph dataset larger.
3. The highest-value next data step is a source with verified timestamped physical pass events plus actual tyre and energy/ERS telemetry. Public OpenF1 does not provide private battery/SOH/power channels.
4. If a candidate is to be final-tested, freeze the exact model and threshold first, then score the sealed holdout once. Do not tune after observing it.

## Primary source artifacts

- ALL_MODELS_REPORT.md
- FINAL_HOLDOUT_TEST_REPORT.md
- ENGINEERED_MODEL_EXPERIMENTS.md
- PHASE1_TRACINGINSIGHTS_MODEL_VALIDATION_REPORT.md
- PHASE5_OPENF1_PIT_MODEL_VALIDATION_REPORT.md
- phase4_proxy_experiment/PHASE4_REPORT.md
- gnn_proxy_experiment_v1/PHASE4_REPORT.md
- gnn_proxy_experiment_gpu_v1/GPU_RUN_REPORT.md
