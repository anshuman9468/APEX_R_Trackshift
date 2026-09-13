# APEX-R Race Strategy Studio

**Team Devsez | VIPS-TC (GGSIPU)**
**TrackShift Hackathon 2025**

APEX-R is an offline-first Formula 1 race-strategy decision-support demo. It combines historical telemetry references, proxy model inference, a constrained simulated energy model, and ATTACK / HOLD / HARVEST / DEFEND strategy comparison.

> This is a research and demonstration system. Public historical data does not contain private ERS percentage, battery state of health, battery temperature, fuel load, team deployment maps, or pit-wall instructions.

## Final demo model

The final demo model presentation is **Frozen Hybrid GNN + GRU + Physics Model**. It combines graph-based race-state context, temporal sequence modelling, tyre-degradation, pit-stop, and safety-constraint views.

The reproducible frozen inference artifact behind the graph advisory is **GPU GNN seed 42, best checkpoint epoch 28**. It predicts a fixed-pair next-lap classified-order position-swap proxy, not a verified overtake and not the probability that ATTACK will be beneficial. The GNN is advisory only; the energy simulator and rule-based optimiser select the strategy action.

The Hybrid GNN figures in the report are supplied visual evidence. Their standalone checkpoint and full reproducible metric artifact are not retained in this repository, so no independent Hybrid-GNN F1 or deployment accuracy is claimed.

## Repository map

| Path | Purpose |
| --- | --- |
| `backend/` | Local API, model adapter, SQLite audit service, and replay endpoints |
| `dist/` | Browser application, shared strategy engine, telemetry view, and model bundle |
| `models/` | Frozen model manifest, model card, checkpoints, and model reports |
| `gnn_proxy_experiment_v1/` | CPU GNN experiment artifacts and disk-backed graph store |
| `gnn_proxy_experiment_gpu_v1/` | GPU GNN runs, histories, predictions, and validation reports |
| `reports/` | Structured model-training PDF, figures, and reproducible report builder |
| `scripts/` | Validation, telemetry import, model-training, and packaging helpers |
| `tests/` | Engine, API, import, persistence, and model-integration tests |
| `runtime/` | Local SQLite and benchmark output; generated at runtime |

## Quick start: judge demo

The simplest demo needs Python 3.10+ and does not require CUDA, Node packages, Docker, or an API key.

```bash
cd "/run/media/anshumandutta/ADATA HD710M PRO/APEX-R_Devsez_Full_Application"
python3 run.py --standalone --port 8001
```

Open `http://127.0.0.1:8001/`. If the port is occupied, choose another free port. Stop the server with `Ctrl+C`.

For a file-only offline preview, open `dist/index.html` in a modern desktop browser. The local server is recommended for model status, API checks, replay, and SQLite persistence.

## GPU model integration

Use the existing GPU environment when available. This does not retrain the model.

```bash
cd "/run/media/anshumandutta/ADATA HD710M PRO/APEX-R_Devsez_Full_Application"
source .venv_gnn_gpu/bin/activate
APEX_GNN_DEVICE=auto python run.py --standalone --port 8001
```

Check the frozen model:

```bash
curl -sS http://127.0.0.1:8001/api/model/status
echo
```

Run the approved label-free smoke request:

```bash
python scripts/build_gnn_predict_example.py
curl -sS -X POST http://127.0.0.1:8001/api/model/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/gnn_predict_request.json
echo
```

Expected output includes `model_version: gnn_proxy_v1`, `selected_seed: 42`, `selected_epoch: 28`, `device: cuda` when CUDA is available, and a finite `proxy_score`.

The output label is `experimental boundary position-swap proxy signal`. Complete graph inputs are required; stale or incomplete essential inputs are rejected and missing values are not silently replaced with invented telemetry.

## Application workflow

1. Start the local server.
2. Open **Pit wall** or choose **Judge demo**.
3. Use the historical reference replay to show the selected driver pair and timestamped state.
4. Inspect the GNN advisory score and freshness/status explanation.
5. Open **Strategy lab** to compare ATTACK, HOLD, HARVEST, and DEFEND.
6. Change scenario inputs or the judge action and compare branch outcomes.
7. Use **Decision log** to inspect inputs, model version, scores, constraints, and simulated results.

Observed telemetry, model output, and simulated energy are separate sources. A model score must not be presented as proof of a successful overtake or as real battery telemetry.

## Model history

Metrics belong to different tasks and splits; they are not one common benchmark.

| Model | Main configuration | Reported result / role |
| --- | --- | --- |
| Legacy Logistic Regression | Eight baseline features; StandardScaler; C=1.0; lbfgs | ROC-AUC 0.711676; AP 0.122823; linear baseline |
| Legacy XGBoost | 300 trees; depth 4; learning rate 0.04 | ROC-AUC 0.787855; AP 0.193889 |
| Cleaned Feature XGBoost | Clean labels; saved 75-tree model | Locked holdout ROC-AUC 0.930631; AP 0.247008 |
| Enriched XGBoost | Rolling speed, throttle, brake, RPM, gear, and DRS context | ROC-AUC 0.794901; AP 0.166549 |
| Engineered XGBoost | Scale-pos-weight sweep; best weight 2.0 | ROC-AUC 0.824588; AP 0.206596 |
| Phase 1 TracingInsights XGBoost | 19 features; weight 2.0; grouped Platt calibration | ROC-AUC 0.818115; AP 0.188670 |
| GPU GNN | Two GINEConv layers; hidden 32; dropout 0.2; seed 42; best epoch 28 | Validation AP 0.127602; ROC-AUC 0.689272; frozen graph advisory |
| Hybrid GNN + GRU + Physics | Supplied integrated probability and classification views | Final demo presentation model; standalone artifact metrics unavailable |

The original XGBoost and GNN tasks differ. XGBoost models use an overtake-opportunity target, while the GNN uses a fixed-pair boundary position-swap proxy. Comparisons must be read within their task and split.

## Data and causal boundaries

- The GNN experiment used 37 telemetry-coverage-passed races, 15,404 supervised graphs, 822 positive proxy windows, and 14,582 negative proxy windows.
- The frozen GNN uses a 10-second trailing lookback and backward-only one-second as-of joins.
- The target is a classified-order position swap at the next lap boundary, not verified on-track overtaking.
- Historical replay is a reference view. Branch futures are simulated and are not recorded race outcomes caused by APEX-R actions.
- Energy, battery state, opponent responses, future positions, and strategy outcomes in the simulator are modelled assumptions.
- The sealed holdout session `11353` is excluded from demo inference and development workflows.

## API endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | Server and adapter health |
| `/api/model/status` | GET | Frozen model availability, checksum, device, and contract |
| `/api/model/predict` | POST | Label-free graph advisory inference |
| `/api/scenarios` | GET | Bundled scenario definitions and configuration |
| `/api/compare` | POST | Judge-versus-optimiser simulated comparison |
| `/api/validate` | POST | Synthetic strategy benchmark |
| `/api/audit?limit=100` | GET | Recent decision records |
| `/api/telemetry?soc=42` | GET | Labelled replay samples |

FastAPI documentation, when optional dependencies are installed, is available at `http://127.0.0.1:8001/docs`.

## Verification commands

```bash
node --test tests/engine.test.cjs
python3 -m unittest discover -s tests -p 'test_*.py' -v
node scripts/validate.cjs
```

Run the frozen-model smoke test directly:

```bash
source .venv_gnn_gpu/bin/activate
python models/frozen/gnn_proxy_v1/smoke_test.py --device cuda
```

The smoke test should report `status: PASS`, a finite score, evaluation/no-grad inference, and `training_code_executed: false`.

## Reports

- [APEX-R model training report](reports/APEX-R_MODEL_TRAINING_REPORT.pdf)
- [Frozen GNN model card](models/frozen/gnn_proxy_v1/MODEL_CARD.md)
- [Frozen GNN manifest](models/frozen/gnn_proxy_v1/FROZEN_MODEL_MANIFEST.json)
- [GPU GNN experiment report](gnn_proxy_experiment_gpu_v1/GPU_RUN_REPORT.md)
- [GPU GNN Phase 4 report](gnn_proxy_experiment_gpu_v1/PHASE4_REPORT.md)
- [Master model-training report](models/APEX_R_MODEL_TRAINING_MASTER_REPORT.md)

The PDF includes the title page, model configurations, confusion matrices, ROC-AUC curves, Average Precision curves, GNN result graph, Hybrid GNN figures, UI screenshots, limitations, and future work.

## Important limitations

1. High accuracy can be misleading because positive events are rare; AP, ROC-AUC, precision, recall, F1, and confusion counts should be read together.
2. The GNN proxy is not a calibrated overtake probability and does not estimate ATTACK benefit.
3. The optimiser currently gives the GNN zero influence over action selection. It is advisory only.
4. The Hybrid GNN figures are supplied reference visuals; no standalone reproducible Hybrid checkpoint is available in this repository.
5. Simulated energy and strategy gains are not observed racing improvements.
6. Browser visual behaviour should be checked on the presentation laptop before the hackathon.

## Research-only training commands

These commands are for controlled experiments, not for the judge demo. Do not run them when you only need to demonstrate the frozen model.

```bash
python3 scripts/train_model.py
python3 scripts/train_xgboost.py
python3 scripts/train_xgboost_enriched.py
```

Training and evaluation must preserve race-separated splits, causal feature rules, explicit missingness, and protected-holdout exclusions. Do not overwrite frozen model artifacts.

## Packaging

Create a release directory and package the application without environments, caches, credentials, or unnecessary raw telemetry:

```bash
node scripts/package.cjs /absolute/path/to/release
```

Keep the report PDF and model manifest with the release. Verify the archive after extraction before distributing it.
