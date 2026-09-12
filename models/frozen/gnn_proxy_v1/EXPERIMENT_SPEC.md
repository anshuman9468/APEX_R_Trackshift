# APEX-R GNN proxy experiment — CUDA execution variant

This is a separate run under gnn_proxy_experiment_gpu_v1/. The canonical
CPU/RAM-safe experiment under gnn_proxy_experiment_v1/ is preserved.

## Task

Fixed attacker/target next-lap-boundary classified-order position-swap proxy.
1 means the fixed pair's endpoint order reverses under the existing Phase 3
contract. This is not a verified on-track-overtake label, attack-conditioned
probability, energy-strategy result, or deployment claim. Unknown/censored
windows are excluded exactly as in the audited graph cache.

## Data and split

- Source graph store: /run/media/anshumandutta/ADATA HD710M PRO/APEX-R_Devsez_Full_Application/gnn_proxy_experiment_v1/disk_graph_store
- Graph cache version: gnn-proxy-graph-cache-v1
- Existing graph construction: backward-only as-of joins, 1.0 second
  tolerance, 10 second trailing lookback, no interpolation/backfill/future
  joins, and the existing directed up-to-three-nearest-finite-XYZ topology.
- Whole-race proposed chronology split: 26 development-train / 6
  development-validation / 5 development-test races.
- The five development-test races are previously accessed development data,
  not an untouched final holdout. Protected session 11353 is excluded.

## Features actually available in this audited graph cache

- Speed and gap/proximity: speed_kmh, distance_m, relative_speed_kmh,
  attacker/target speed and distance deltas.
- Positional/timing: x_m, y_m, z_m, session-time-derived sample ages,
  and trailing 10-second speed delta/trend.
- Raw telemetry context: throttle, brake, gear, RPM and DRS state, plus
  explicit missingness masks and attacker/target role indicators.

## Requested signals unavailable and excluded

Measured tyre wear/age, ERS/energy delta, energy/power telemetry, battery
state-of-health, and battery/ES temperature do not exist in the approved
continuous telemetry schema. Values from the APEX-R simulation engine are
modelled state, not historical measured inputs, so they are not used as
training features. GPS latitude/longitude is also not present; the approved
positional fields are FastF1-derived track-frame x_m/y_m/z_m.

## Frozen model configuration

- Model: two-layer edge-aware GINEConv, hidden width 32, dropout 0.2.
- Optimizer: Adam, learning rate 0.001, weight decay 0.0001.
- Loss: unweighted BCEWithLogitsLoss.
- Batch size 32, maximum 100 epochs, early stopping patience 15 on
  validation average precision.
- Seeds: 17, 23, 42. No broad hyperparameter search, resampling, SMOTE or
  automatic calibration.
- Preprocessing: training-only mean/std using the existing formulas;
  missing values are replaced by training means at transform time and the
  explicit masks remain. Raw graph arrays remain float64 on disk and the
  existing Data boundary remains float32.
- Baselines: constant training prevalence and regularized logistic regression
  with C=1.0 on the same eligible pair-level inputs.
- Primary metric: validation average precision; ROC-AUC, Brier, log loss,
  PR curves, reliability bins and whole-race bootstrap are also reported.

## Execution-device exception

All locked experiment parameters above are unchanged. The explicit execution
change is device: cpu -> cuda on NVIDIA GeForce RTX 3050 Laptop GPU using
PyTorch 2.11.0+cu128. GPU execution can differ numerically from CPU
execution even with the same seeds; bit-for-bit equality is not promised.
