# Hybrid GNN + physics model

This directory contains the inference server and model implementation used by
the integrated APEX-R frontend. The server loads only the frozen epoch-51
package in `deployment_packages/hybrid_gnn_epoch51/`.

## Run the integrated app

Install the Python dependencies from the repository root, then start the local
server:

```bash
python3 -m venv .venv-hybrid
source .venv-hybrid/bin/activate
python -m pip install -r requirements.txt
python ml_pipeline/trackshift_server.py --port 8000
```

Open `http://127.0.0.1:8000/`. The Pit wall displays the four model-guided
strategy actions—ATTACK, HOLD, DEFEND and HARVEST—alongside the frozen GNN
timing and risk outputs. The action policy combines those outputs with
physics-feasible action scores; the current checkpoint was not trained with a
dedicated four-class action head.

## API checks

```bash
curl -sS http://127.0.0.1:8000/api/hybrid/status
curl -sS 'http://127.0.0.1:8000/api/hybrid/predict?sequence=0'
```

`model_deployment.pt` is the lean inference artifact. `model_checkpoint.pt`
retains optimizer and training history, while `model_test_bundle.pt` retains
the model, 2026 unseen holdout graph arrays, labels, predictions and metrics.
