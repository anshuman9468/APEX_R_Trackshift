# Phase 3 deployment package

Frozen hybrid GNN + GRU + physics model selected at epoch 51.

Use `model_deployment.pt` for inference. It contains the model weights and the
exact training configuration, including the graph normalizer. Use
`feature_contract.json` for feature order and graph construction, and
`calibration.json` plus `output_schema.json` for calibrated probabilities and
operating thresholds.

The exact training checkpoint, including optimizer/history state, is retained
as `model_checkpoint.pt` for reproducibility. `replay_report.json` is added
after the Phase 3 holdout replay completes. `model_test_bundle.pt` contains
the deployment weights together with the exact 2026 unseen holdout graph
arrays, labels, predictions, metrics, and feature contract.

Deployment regression output is the raw `next_lap_time_s` prediction. The
affine lap-time calibration was measured but not selected because it increased
holdout MAE. Classification probabilities use Platt calibration fitted only
on the Phase 2 calibration split.
