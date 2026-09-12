# Phase 4 Proxy Experiment

This directory contains a reproducible, isolated experiment on the Phase 3
lap-boundary position-swap proxy. It does not modify existing models, does
not connect to the decision engine, and does not access the protected final
holdout.

## Run

From the project root, with the repository environment active:

```bash
python phase4_proxy_experiment/phase4_experiment.py
python -m unittest discover -s phase4_proxy_experiment -p 'test_*.py' -v
python phase4_proxy_experiment/verify_phase4_artifacts.py
```

Dependencies are the existing `requirements.txt` packages, especially
`pandas`, `numpy`, `scikit-learn` and `joblib`. The script is offline and uses
only cached Phase 3 inputs.

## Important interpretation

The target is a fixed driver pair reversing order at the next lap boundary.
It is not a verified second-level overtake. The repaired primary task censors
windows whose fixed target cannot be supported by a unique completed-order
snapshot at the decision timestamp. Only a constant baseline and one fixed,
unweighted regularized logistic regression are fit.

`split_metrics.csv` contains validation and one frozen development-test
evaluation. `pr_curves.csv.gz` contains the complete precision/recall curves.
`uncertainty_by_race.csv` contains paired race-level bootstrap summaries.
No operational threshold is authorized from this small proxy dataset.
