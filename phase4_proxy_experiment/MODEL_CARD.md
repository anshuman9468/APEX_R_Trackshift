# APEX-R Phase 4 Proxy Model Card

## Scope

This is an isolated historical experiment for the Phase 3 lap-boundary
position-swap proxy. It is not a verified on-track-overtake model and is not
connected to ATTACK, HOLD, HARVEST or DEFEND logic.

## Selected model

The candidate is a regularized logistic regression with `C=1.0`, L2 penalty,
unweighted loss, training-only median imputation with missingness indicators,
and training-only standardization. It uses the Phase 3 approved historical
proxy feature list. No driver, event, source-file, split, label or outcome
evidence fields are model features.

## Training

* Task: `phase4-repaired-asof-boundary-swap-v1`
* Training rows: 353
* Training positives: 7
* Training races: 36
* Random seed: 20260911
* Class weighting/resampling: none
* Calibration: none; probabilities are descriptive only

## Limitations

The labels are boundary position swaps. They can include lapping, unlapping,
retirement effects, timing corrections or hidden pass/repass. There are only
20 repaired positive proxy events across all partitions and only 6 in the
development-test partition. A race-level bootstrap is supplied, but this is
not enough support for a deployment claim or a stable operating threshold.
Source channel units/encodings and historical publication latency remain
partly unresolved. No sealed final holdout was accessed.
