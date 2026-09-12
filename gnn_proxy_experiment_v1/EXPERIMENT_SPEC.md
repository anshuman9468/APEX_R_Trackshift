# APEX-R GNN proxy experiment v1 — frozen specification

Status: frozen before graph construction and fitting; historical development experiment only.

## Scope and target

This experiment uses the existing Phase 3 target contract v3.0. Each example is
one selected driver/lap boundary. The attacker and target are fixed at the
decision boundary; the label is 1 only when their classified positions reverse
at the next observed lap boundary, and 0 only when both endpoint positions are
observed and the attacker remains behind. Unknown/censored windows are not
training or evaluation negatives. This is a boundary-order position-swap proxy,
not a verified on-track overtake label, an attack-conditioned probability, or an
energy-strategy objective. Verified on-track events remain zero in the source
audit.

The decision state is formed only from measurements at or before the recorded
decision timestamp. Current samples use the latest sample at or before the
decision time and no older than 1.0 second. Trailing speed features use the
latest sample at or before `decision_time - 10 seconds`; they are missing when
that historical point is unavailable. No interpolation, backward fill,
centred smoothing, whole-lap aggregate, or future-aware matching is used.

## Races and split

Input races are derived from the continuous-telemetry collection's
`telemetry_coverage_status=PASS` records and cross-checked against the Phase 3
37-race export. The protected session `11353` and every declared exclusion are
blocked before any feature retrieval. The proposed chronology-only manifest is
used without modification:

* 26 races: `development_train`
* 6 races: `development_validation`
* 5 races: `development_test`

The authoritative original assignment is preserved separately. All of these
races are previously accessed development data; `development_test` is not an
untouched final holdout. Whole races, fixed-pair events and all windows linked
to an event remain in one partition. No row-level random split is used.

## Graph construction

One graph is built per eligible labelled window. Nodes are the drivers observed
in the race, sorted by source driver identifier. Node fields are:

`speed_kmh`, `throttle_pct`, `brake`, `n_gear`, `rpm`, `drs_binary`,
`car_sample_age_sec`, `x_m`, `y_m`, `z_m`, `position_sample_age_sec`,
`speed_delta_10s_kmh`, `speed_trend_10s_kmh_per_s`,
`car_missing_mask`, `position_missing_mask`, `drs_missing_mask`,
`role_attacker`, `role_target`.

No tyre field is included because no causally safe tyre stream is present in
the approved passed-race telemetry view. Source encodings and units remain as
documented in the Phase 3 readiness table; values are not clamped or silently
reinterpreted.

Pair decoder fields are:

`attacker_target_speed_delta_kmh`, `attacker_target_distance_m`,
`attacker_target_speed_delta_10s_kmh`, `attacker_car_age_sec`,
`target_car_age_sec`, `attacker_position_age_sec`, `target_position_age_sec`,
`attacker_car_missing`, `target_car_missing`, `attacker_position_missing`,
`target_position_missing`, `pair_position_available`.

For each node, directed edges go to up to three nearest other cars by current
FastF1 XYZ position, with reciprocal directions when both endpoints have
finite positions. Classified adjacency is never used as physical proximity.
Edge fields are `distance_m`, `relative_speed_kmh`,
`source_position_age_i_sec`, and `source_position_age_j_sec`. An edge is absent
when either position is unavailable; this is represented by the graph topology,
not a fabricated distance.

## Preprocessing and models

Raw NaNs are retained in the graph cache and explicit missing masks are kept.
For fitting, feature-wise means and standard deviations are fitted only on
`development_train` graphs. NaNs are replaced with those training means only
inside the frozen transform, never in source data or the cache; no target,
event, race, driver, filename, split, or post-decision field enters the
transform.

The GNN is a compact two-layer GINE message-passing model with hidden width 32,
dropout 0.2, an edge encoder, and a pair decoder from attacker embedding,
target embedding, and pair fields. It uses Adam (learning rate 0.001, weight
decay 0.0001), batch size 32, unweighted binary cross-entropy with logits, a
maximum of 100 epochs, validation average precision checkpoint selection, and
15-epoch patience. Exactly three predeclared seeds are run: 17, 23, and 42.
There is no broad search, class weighting, SMOTE, or evaluation resampling.

Baselines use the same eligible windows and target. The constant baseline is
the training positive prevalence. The tabular baseline is regularized logistic
regression with C=1.0, unweighted, using the attacker node fields, target node
fields, and pair fields; preprocessing is fitted on training graphs only.
The comparison does not isolate graph structure perfectly because the GNN also
uses the other cars' approved as-of node context and physical-neighbor edges,
whereas logistic regression receives only pair-level information. This is
reported explicitly.

## Evaluation and selection

Average precision is the primary ranking metric. ROC-AUC, Brier score, log loss,
prevalence, precision-recall curves, and reliability bins are also reported.
Validation AP selects each GNN checkpoint. After all checkpoints and
preprocessing states are frozen, each seed is evaluated once on the five-race
development test. The operational threshold is selected on validation only by
maximum F1, ties broken by higher precision and then higher threshold; it is
frozen before development-test scoring. Threshold claims are omitted where
the curve has no usable positive/negative support.

Uncertainty uses 1,000 whole-race bootstrap resamples for candidate-versus-
constant AP deltas. Five development-test races provide limited uncertainty;
class-degenerate resamples are counted and metrics are left undefined where
appropriate. No model or threshold is revised after development-test access.

## Reproducibility and integrity

The experiment records input SHA-256 hashes before and after construction,
environment and hardware information, seeds, actual epochs and durations,
checkpoint metadata, cache schema, split hashes, and inference parity after
checkpoint reload. The sealed session token `11353` is rejected by all source
selection and output checks.
