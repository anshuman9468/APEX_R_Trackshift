# Phase 4 handoff — passed-race subset

Entry condition: use `proxy_feature_view_37.csv.gz` joined to `proxy_labels_37.csv.gz` by `window_id`, with `race_id` as the grouping key. Keep all windows from a race in one existing split. Do not use `prediction_windows_37.csv.gz` as the model table because it contains outcome/evidence fields.

This is a boundary-order position-swap proxy. It is not a verified overtake, and no supported on-track event was found in this evidence pass. Train only an explicitly named historical proxy model if desired; do not connect its probability to ATTACK/HOLD/HARVEST until Phase 4 and Phase 5 checks are complete.
