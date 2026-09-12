# APEX-R Phase 1 — corrected validation recommendation

Generated: 2026-09-06

## Corrected decision

**GO** on spending the one-time final holdout evaluation on the already-frozen `4.0.0-phase1-tracinginsights-candidate`—but only after explicit approval.

This corrects the previous NO-GO. No data was loaded and no model was trained or scored for this correction.

## What the 15.50% number actually is

The 15.50% precision number is from the sealed final test race, not from validation. Its source is `models/FINAL_HOLDOUT_TEST_REPORT.md`, which identifies the tested model as `2.0.0-xgboost-openf1-clean-base` with 75 trees and reports precision `0.155131`, recall `0.844156`, and test confusion matrix TN=2,152, FP=354, FN=12, TP=65.

The previous Phase 1 decision code required all three conditions: beat 15.50%, beat the prior 37-feature validation result, and improve the same-split 8-feature comparator. Because Phase 1 did not beat the different-race 15.50% result, that invalid cross-split gate forced NO-GO.

That was not a fair model-selection rule. The 15.50% result must not be used to rank validation candidates.

## Fair validation-only ranking

All three figures below use the same validation criterion: maximum precision while recall remains at least 70%.

1. Phase 1 candidate: **12.73% precision at 70.04% recall**.
2. Prior 37-feature attempt: **11.36% precision at 70.82% recall**.
3. Same-split 8-feature comparator: **9.75% precision at 76.26% recall**.

Phase 1 is therefore the best validation candidate across the three attempts on the predeclared primary metric. It improves precision by 1.37 percentage points over the prior 37-feature attempt and by 2.98 points over the same-split 8-feature comparator.

The result is not uniformly superior on every metric: the prior 37-feature attempt has higher validation ROC-AUC (`0.8246` vs `0.8181`) and AP (`0.2066` vs `0.1887`). Phase 1 wins because precision at recall≥70% was explicitly designated as the primary selection metric before comparison.

## Important model-identity nuance

The 9.75% same-split 8-feature comparator is not literally the same serialized model as the sealed-test baseline. The sealed report identifies a 75-tree model. The same-split comparator used the Phase 1-selected setup: 125 trees, depth 3, learning rate 0.03, `scale_pos_weight=2`, and Platt calibration.

Therefore the observed 9.75%-versus-15.50% difference cannot be claimed as a pure race-difficulty effect on an identical model. Both race and model configuration differ. This does not rescue the old comparison: validation and sealed-test results still cannot be used as if they came from one common evaluation set.

## What remains unknown

We do not know how Phase 1 performs on the sealed final test race because it has never been evaluated there. No test precision, recall, ROC-AUC, AP, Brier score, or false-positive rate should be estimated by transferring the baseline's cross-race change to Phase 1.

## False-positive-rate correction

Raw false-positive counts are not comparable because the datasets contain different numbers of actual negatives.

- Original baseline on the sealed test: `FP / (FP + TN) = 354 / (354 + 2,152) = 354 / 2,506 = 14.13%`.
- Phase 1 on validation: `1,234 / (1,234 + 4,567) = 1,234 / 5,801 = 21.27%`.

The observed cross-dataset false-positive rate increased by **7.15 percentage points** (approximately 50.6% relative). This is a factual rate comparison, not evidence that Phase 1 itself generalizes worse, because the races, model configurations, thresholds, and recall levels differ.

The fair same-validation comparison is more favorable: the 8-feature comparator had `1,815 / 5,801 = 31.29%` false-positive rate, while Phase 1 had `21.27%`, a decrease of **10.02 percentage points**. Phase 1 also operated at lower recall, 70.04% versus 76.26%, so part of that reduction is the intended precision/recall tradeoff.

## Actionable recommendation

Use the same-split result to select the candidate: Phase 1 ranks first on the agreed primary metric, is already frozen, and has not been exposed to the final test. That is sufficient validation evidence to justify one—and only one—final holdout evaluation.

If approved, score the frozen candidate and threshold exactly once, report the result as-is, and do not return to validation or feature selection afterward.
