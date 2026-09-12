# APEX-R Phase 1 TracingInsights candidate — validation-only report

> **Correction (2026-09-06):** The NO-GO below was produced by requiring this validation candidate to beat a precision number from a different sealed-test race. That cross-split gate was invalid. The corrected, same-split recommendation is **GO**, documented in `PHASE1_VALIDATION_RECOMMENDATION_CORRECTION.md`. Model metrics and artifacts are unchanged.

Generated: 2026-09-06T15:43:52.310616+00:00

## Decision

Recommendation after same-split correction: **GO** for the one-time final holdout evaluation, pending explicit user approval.

The selected candidate is `phase1-selected-w2-platt`. At its validation-selected threshold 0.07526282, precision is 12.73% and recall is 70.04%. It is the best of the three same-validation-split attempts: 12.73% versus 11.36% for the prior 37-feature attempt and 9.75% for the 8-feature comparator.

The original supplied baseline number comes from a different race, so it is shown as a requested benchmark only. The same-split 8-feature result is the valid apples-to-apples feature comparison.

No final holdout data was loaded or scored. Stop here pending explicit approval.

## Data and missing-value handling

- Train: 13,912 rows, 1,211 positives (8.70%), grouped across three races.
- Validation: 6,058 rows, 257 positives (4.24%), one untouched validation race for this experiment.
- Source features: 19 conceptual columns—8 baseline plus all 11 Phase 1 columns.
- Numeric NaNs were preserved. No missing numeric value was changed to zero or a placeholder.
- Compound pair uses XGBoost native categorical splits. Missing compound values remain categorical-missing, not a fabricated compound.

## Selected configuration

- Trees: 125; max depth: 3; learning rate: 0.03.
- `scale_pos_weight`: 2.
- Calibration: Platt scaling from race-grouped OOF train predictions.
- ROC-AUC: 0.818115; AP: 0.188670; Brier: 0.037707; ECE: 0.020933.
- Confusion matrix at selected threshold: {'tn': 4567, 'fp': 1234, 'fn': 77, 'tp': 180}.

## Every candidate configuration tested

All 27 fitted configurations are reported across this section and the ablation section. Each entry reports its entire exact precision/recall curve point count. The threshold-by-threshold values are stored without downsampling in `models/phase1_tracinginsights_validation_experiments.json.gz` under `all_candidates_with_full_precision_recall_curves`.

- `phase1-t75-d3-lr0.05-w1-raw` — raw; trees=75, depth=3, lr=0.05, weight=1.0: ROC-AUC 0.805328, AP 0.185632, Brier 0.037773, precision@recall≥70% 0.107271 at recall 0.700389, threshold 0.05960904; exact PR points 3075.
- `phase1-t75-d4-lr0.04-w1-raw` — raw; trees=75, depth=4, lr=0.04, weight=1.0: ROC-AUC 0.804647, AP 0.186879, Brier 0.037504, precision@recall≥70% 0.108905 at recall 0.704280, threshold 0.05724546; exact PR points 3634.
- `phase1-t75-d5-lr0.03-w1-raw` — raw; trees=75, depth=5, lr=0.03, weight=1.0: ROC-AUC 0.790139, AP 0.179938, Brier 0.037747, precision@recall≥70% 0.099338 at recall 0.700389, threshold 0.05337877; exact PR points 3849.
- `phase1-t125-d3-lr0.03-w1-raw` — raw; trees=125, depth=3, lr=0.03, weight=1.0: ROC-AUC 0.820469, AP 0.194993, Brier 0.037439, precision@recall≥70% 0.121821 at recall 0.708171, threshold 0.06245589; exact PR points 3437.
- `phase1-t125-d4-lr0.03-w1-raw` — raw; trees=125, depth=4, lr=0.03, weight=1.0: ROC-AUC 0.798484, AP 0.180833, Brier 0.037786, precision@recall≥70% 0.098501 at recall 0.715953, threshold 0.04815182; exact PR points 4071.
- `phase1-t150-d3-lr0.025-w1-raw` — raw; trees=150, depth=3, lr=0.025, weight=1.0: ROC-AUC 0.817247, AP 0.188663, Brier 0.037592, precision@recall≥70% 0.117111 at recall 0.700389, threshold 0.06114859; exact PR points 3733.
- `phase1-selected-w1-raw` — raw; trees=125, depth=3, lr=0.03, weight=1.0: ROC-AUC 0.820469, AP 0.194993, Brier 0.037439, precision@recall≥70% 0.121821 at recall 0.708171, threshold 0.06245589; exact PR points 3437.
- `phase1-selected-w1-platt` — Platt; trees=125, depth=3, lr=0.03, weight=1.0: ROC-AUC 0.820469, AP 0.194993, Brier 0.037685, precision@recall≥70% 0.121821 at recall 0.708171, threshold 0.07454018; exact PR points 3437.
- `phase1-selected-w1.5-raw` — raw; trees=125, depth=3, lr=0.03, weight=1.5: ROC-AUC 0.824120, AP 0.193078, Brier 0.038505, precision@recall≥70% 0.116709 at recall 0.712062, threshold 0.08509353; exact PR points 3349.
- `phase1-selected-w1.5-platt` — Platt; trees=125, depth=3, lr=0.03, weight=1.5: ROC-AUC 0.824120, AP 0.193078, Brier 0.037562, precision@recall≥70% 0.116709 at recall 0.712062, threshold 0.06651814; exact PR points 3349.
- `phase1-selected-w2-raw` — raw; trees=125, depth=3, lr=0.03, weight=2.0: ROC-AUC 0.818115, AP 0.188670, Brier 0.040845, precision@recall≥70% 0.127298 at recall 0.700389, threshold 0.11580408; exact PR points 3219.
- `phase1-selected-w2-platt` — Platt; trees=125, depth=3, lr=0.03, weight=2.0: ROC-AUC 0.818115, AP 0.188670, Brier 0.037707, precision@recall≥70% 0.127298 at recall 0.700389, threshold 0.07526282; exact PR points 3218.
- `phase1-selected-w3-raw` — raw; trees=125, depth=3, lr=0.03, weight=3.0: ROC-AUC 0.813511, AP 0.185226, Brier 0.046259, precision@recall≥70% 0.115385 at recall 0.700389, threshold 0.14459835; exact PR points 3137.
- `phase1-selected-w3-platt` — Platt; trees=125, depth=3, lr=0.03, weight=3.0: ROC-AUC 0.813511, AP 0.185226, Brier 0.037900, precision@recall≥70% 0.115385 at recall 0.700389, threshold 0.07272961; exact PR points 3137.
- `baseline8-same-split-raw` — raw; trees=125, depth=3, lr=0.03, weight=2.0: ROC-AUC 0.811429, AP 0.169757, Brier 0.041320, precision@recall≥70% 0.097464 at recall 0.762646, threshold 0.09172543; exact PR points 2486.
- `baseline8-same-split-platt` — Platt; trees=125, depth=3, lr=0.03, weight=2.0: ROC-AUC 0.811429, AP 0.169757, Brier 0.038127, precision@recall≥70% 0.097464 at recall 0.762646, threshold 0.06219555; exact PR points 2486.

## Direct comparisons

- Same-split 8-feature XGBoost: precision 9.75% at recall 76.26%; ROC-AUC 0.811429; AP 0.169757; Brier 0.038127.
- New Phase 1 winner: precision 12.73% at recall 70.04%; ROC-AUC 0.818115; AP 0.188670; Brier 0.037707.
- Supplied original baseline benchmark: precision 15.50% at recall 84.40%; ROC-AUC 0.9306; AP 0.2470; Brier 0.0266. Different race; contextual only.
- Supplied prior 37-feature attempt: best precision 11.36% at recall 70.82%; ROC-AUC 0.8246; AP 0.2066; Brier 0.037802. Same validation protocol. Phase 1 wins the primary precision-at-recall target, but the prior model retains higher ROC-AUC and AP.
- Corrected plain verdict: Phase 1 is the best validation candidate on the predeclared primary metric. The different-race 15.50% result is contextual and is excluded from GO/NO-GO.

## Gain importance for the 11 Phase 1 columns

- `relative_lap_time_delta`: gain 71.949898; 7.51% of total model gain.
- `opponent_tyre_age_delta`: gain 48.829834; 5.10% of total model gain.
- `relative_sector_time_delta_s2`: gain 48.059746; 5.02% of total model gain.
- `opponent_tyre_compound_delta`: gain 39.123405; 4.08% of total model gain.
- `rival_pit_out_recent_flag`: gain 38.439297; 4.01% of total model gain.
- `relative_sector_time_delta_s1`: gain 35.878815; 3.74% of total model gain.
- `distance_to_car_ahead_m`: gain 32.489243; 3.39% of total model gain.
- `relative_sector_time_delta_s3`: gain 31.600555; 3.30% of total model gain.
- `stint_length_delta`: gain 28.687361; 2.99% of total model gain.
- `rival_drs_open`: gain 0.000000; 0.00% of total model gain.
- `attacker_pit_out_recent_flag`: gain 0.000000; 0.00% of total model gain.

## Leave-one-column-out ablation

Negative deltas mean performance fell when the feature was removed, so the feature helped. Positive deltas mean removal improved that validation metric and the feature may be neutral or harmful on this split.

- Remove `relative_sector_time_delta_s1`: ROC-AUC 0.815994; AP 0.186867; Brier 0.040884; precision@recall≥70% 0.123035 at recall 0.700389, threshold 0.11557965; exact PR points 3039; ΔAP -0.001803; ΔROC-AUC -0.002121; Δprecision@recall≥70% -0.004264.
- Remove `relative_sector_time_delta_s2`: ROC-AUC 0.809962; AP 0.187507; Brier 0.040630; precision@recall≥70% 0.125874 at recall 0.700389, threshold 0.11655608; exact PR points 3158; ΔAP -0.001163; ΔROC-AUC -0.008153; Δprecision@recall≥70% -0.001424.
- Remove `attacker_pit_out_recent_flag`: ROC-AUC 0.813313; AP 0.187966; Brier 0.041144; precision@recall≥70% 0.125348 at recall 0.700389, threshold 0.11798699; exact PR points 3237; ΔAP -0.000704; ΔROC-AUC -0.004802; Δprecision@recall≥70% -0.001950.
- Remove `rival_pit_out_recent_flag`: ROC-AUC 0.813313; AP 0.187966; Brier 0.041144; precision@recall≥70% 0.125348 at recall 0.700389, threshold 0.11798699; exact PR points 3237; ΔAP -0.000704; ΔROC-AUC -0.004802; Δprecision@recall≥70% -0.001950.
- Remove `rival_drs_open`: ROC-AUC 0.813271; AP 0.188051; Brier 0.041141; precision@recall≥70% 0.125348 at recall 0.700389, threshold 0.11799160; exact PR points 3235; ΔAP -0.000619; ΔROC-AUC -0.004844; Δprecision@recall≥70% -0.001950.
- Remove `opponent_tyre_compound_delta`: ROC-AUC 0.814945; AP 0.193831; Brier 0.041065; precision@recall≥70% 0.120000 at recall 0.700389, threshold 0.11385222; exact PR points 3152; ΔAP +0.005161; ΔROC-AUC -0.003170; Δprecision@recall≥70% -0.007298.
- Remove `distance_to_car_ahead_m`: ROC-AUC 0.817440; AP 0.194444; Brier 0.040715; precision@recall≥70% 0.122867 at recall 0.700389, threshold 0.11324838; exact PR points 2730; ΔAP +0.005774; ΔROC-AUC -0.000675; Δprecision@recall≥70% -0.004432.
- Remove `relative_sector_time_delta_s3`: ROC-AUC 0.814463; AP 0.194472; Brier 0.040886; precision@recall≥70% 0.120426 at recall 0.704280, threshold 0.11189681; exact PR points 3107; ΔAP +0.005802; ΔROC-AUC -0.003652; Δprecision@recall≥70% -0.006873.
- Remove `opponent_tyre_age_delta`: ROC-AUC 0.817491; AP 0.194749; Brier 0.040844; precision@recall≥70% 0.125523 at recall 0.700389, threshold 0.11458498; exact PR points 2934; ΔAP +0.006079; ΔROC-AUC -0.000624; Δprecision@recall≥70% -0.001775.
- Remove `relative_lap_time_delta`: ROC-AUC 0.814634; AP 0.196385; Brier 0.040773; precision@recall≥70% 0.102916 at recall 0.700389, threshold 0.09894080; exact PR points 3093; ΔAP +0.007715; ΔROC-AUC -0.003481; Δprecision@recall≥70% -0.024382.
- Remove `stint_length_delta`: ROC-AUC 0.815844; AP 0.197282; Brier 0.040566; precision@recall≥70% 0.128297 at recall 0.700389, threshold 0.11378992; exact PR points 3147; ΔAP +0.008612; ΔROC-AUC -0.002272; Δprecision@recall≥70% +0.000998.

Every ablation candidate's ROC-AUC, AP, Brier score, operating point, and complete precision/recall curve are in the compressed experiment artifact.
Gain is the direct record of which columns the selected fitted booster used. Leave-one-out deltas are retraining-sensitivity evidence, not causal effects; because row and feature subsampling are enabled, removing even an unused column can change the retrained tree sequence.

## Calibration

Platt calibration was fitted only from leave-one-training-race-out predictions. The validation race was never used to fit the calibrator. Calibration is monotonic, so ranking metrics and the shape of precision versus recall are expected to remain unchanged; it can improve Brier/ECE and transforms the numerical threshold.

## Artifacts and immutability

- New candidate: `models/candidates/4.0.0-phase1-tracinginsights-candidate.json`; embedded UBJ SHA-256 `02b2a61d802ccbe4884c96e02108beaa6afa11c2eebe0f7a4a7c17ab3a1745f9`.
- Full exact curves and audit: `models/phase1_tracinginsights_validation_experiments.json.gz`.
- The exported booster was reloaded and reproduced the first 100 raw validation probabilities to 1e-7 tolerance.
- Protected pre-existing baseline and prior-candidate file hashes were unchanged.
- This candidate was not copied into the live application and no existing model was overwritten.

## Stop condition

Do not run the one-time final holdout evaluation without explicit user approval. If approved later, score this frozen candidate and threshold exactly once and report the result without iterative retuning.

## Environment cleanup

- The scoped `.venv_phase1_model` environment occupied 12 MB before teardown and has been removed.
- No packages were installed system-wide; the scoped environment reused already-installed scientific packages read-only.
- The temporary training log was deleted.
- Retained candidate, exact-curve audit, report, and reproducibility script occupy approximately 1.9 MB.
