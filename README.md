# APEX-R Race Strategy Studio

Team Devsez / VIPS-TC (GGSIPU)

A working, offline-first race-strategy application. The pit wall, optimiser, counterfactual comparison, benchmark, telemetry import and decision log all compute or process data. There are no scripted winning outcomes.

## Quick start

**No installation:** open `dist/index.html` in a modern desktop browser. All core functionality runs locally. The separate release file `APEX-R_Devsez_Offline.html` bundles every required asset into one HTML file.

**Full local app with SQLite:** install Python 3.10+ and Node.js 18+ (no npm packages required), then run from this directory:

```bash
python3 run.py
```

Open http://127.0.0.1:8000/. If occupied, use `python3 run.py --port 8001`. Ctrl+C stops the server. No GPU, Docker, API key or Internet connection is required for the bundled demo. The server listens only on the local machine.

## FastAPI mode

FastAPI is an optional adapter over the same tested local service and engine. Install dependencies while online:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --fastapi
```

On Windows, activate with `.venv\Scripts\activate` instead. API docs are at http://127.0.0.1:8000/docs. FastAPI adds WebSocket replay at `/ws/replay`; the frontend connects automatically. If FastAPI is unavailable, `python3 run.py` uses the included dependency-free HTTP adapter with the same comparison, benchmark, telemetry and SQLite services. `--standalone` explicitly selects it.

Dependency installation was blocked in the build environment. The dependency-free HTTP adapter was runtime-tested. FastAPI and its WebSocket adapter were syntax-checked but could not be runtime-tested here; the included optional tests run once requirements are installed. No server is automatically started by opening the HTML file.

## Frozen GNN proxy API

The selected GPU-trained checkpoint is integrated as an inference-only,
advisory backend service from `models/frozen/gnn_proxy_v1/`. Its output is
labelled **experimental boundary position-swap proxy signal**. It scores the
fixed-pair boundary position-swap proxy task; it is not a calibrated overtake
probability and it never returns an ATTACK/HOLD/HARVEST decision. The existing
rule/energy simulator remains responsible for that decision.

Use the GPU environment when available so the service can select CUDA; the
`--standalone` flag keeps the local HTTP adapter dependency-light:

```bash
cd "/run/media/anshumandutta/ADATA HD710M PRO/APEX-R_Devsez_Full_Application"
source .venv_gnn_gpu/bin/activate
python run.py --standalone --port 8000
```

In another terminal, check the manifest and model status, build the label-free
approved example, and send it to the graph-level endpoint:

```bash
cd "/run/media/anshumandutta/ADATA HD710M PRO/APEX-R_Devsez_Full_Application"
curl -sS http://127.0.0.1:8000/api/model/status
python scripts/build_gnn_predict_example.py
curl -sS -X POST http://127.0.0.1:8000/api/model/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/gnn_predict_request.json
```

The prediction request must contain the complete raw multi-car graph emitted
by the canonical builder: node features, edge topology and features, fixed
attacker/target indices, pair features, and the one-second backward-only /
ten-second lookback freshness declaration. Labels, future outcomes and
simplified speed/gap-only dictionaries are rejected. The service loads and
checksums the frozen model once at startup, preserves explicit missingness,
reports warnings and latency, and returns an explicit unavailable response on
checksum, dependency or model failures. Malformed, stale or incomplete
essential-pair requests are rejected with a validation error rather than
receiving a fabricated score.

## What's implemented

| View | Behaviour |
| --- | --- |
| Pit wall | Animated telemetry replay; play, pause, seek and speed; capture a decision window; four action choices; live model recommendation |
| Counterfactual comparison | Judge first action plus HOLD versus an optimised full sequence; 96 paired trials; seeded branch events; energy timeline; JSON export |
| Strategy lab | Energy, front/rear gaps, closing rate, tyre proxy, wetness, aggression, risk and opportunity controls; feasible sequence ranking and rejection reasons |
| Validation | Synthetic scenario suite; four policies; hidden outcome perturbations; wins, ties, losses, energy, pass/loss rates and executed violations |
| Decision log | Inputs, seed, provenance, model version, scores, outcomes; restore inputs; export logs; browser storage and optional SQLite |
| Data & model | OpenF1-style JSON import, missing-field handling, optional location samples, explicit data provenance and model limitations |

The starting replay is deliberately paused. Choose **Judge demo** for the built-in 90-second flow. Recommendations recompute when scenario parameters change. The random seed changes simulated outcomes rather than secretly forcing APEX-R to win.

## Integrated hybrid GNN + physics frontend

The `dist/` frontend in this branch is wired to the frozen deployment package
under `deployment_packages/hybrid_gnn_epoch51/`. Run the model-backed local
server from the repository root with:

```bash
python3 -m venv .venv-hybrid
source .venv-hybrid/bin/activate
python -m pip install -r requirements.txt
python ml_pipeline/trackshift_server.py --port 8000
```

The Pit wall makes the model-guided strategy decision the primary output and
shows ATTACK, HOLD, DEFEND and HARVEST scores. It also reports the frozen
epoch-51 next-lap and calibrated risk outputs. The action layer combines those
GNN signals with physics-feasible strategy scores; the checkpoint itself does
not contain a separately trained four-class action head. The test bundle keeps
the actual 2026 unseen holdout labels, predictions and metrics for inspection.

The model service is evaluated against a local test-holdout replay and does
not claim measured private F1 telemetry. Live MultiViewer observations are
kept separate from modelled energy and counterfactual outcomes.

When MultiViewer is open with Live Timing active, the same local server can
also poll its local timing API. The Pit wall then shows the observed meeting,
circuit, driver roster, sectors, tyre state, weather, and race-control status.
Use `--multiviewer-driver VER` (or another driver code/number) to set the
initial focus car; the dashboard also supports selecting a target driver.
Circuit geometry is matched from `dist/track_maps.js`, while live car markers
use lap progress because the local timing API does not expose live XY points.

## Data honesty

The bundled replay and circuit remain synthetic. Energy, opponent gaps, future positions and simulator outcomes are still modelled. The integrated Pit wall uses the frozen hybrid GNN + GRU + physics package described above; the legacy XGBoost artifact remains in the repository for its original scripts and tests but is not loaded by this frontend. Neither model predicts private team battery telemetry or a complete F1 strategy. Importing public telemetry changes playback and provenance but does not create private ERS measurements or guarantee calibration for a new race.

The model searches two to five lap windows with one strategic decision per lap. Probabilities are explicit heuristics. The objective is a weighted position/energy/risk utility; position value per MJ is separately reported. See `docs/MODEL.md` for equations, units, assumptions and evaluation limits.

## Historical telemetry

In **Data & model**, choose an OpenF1 `car_data` JSON export for one driver and one session. Each record requires a timestamp and speed. `throttle`, `brake` and `n_gear` are displayed when present. A combined `{ "car_data": [...], "location": [...], "metadata": {...} }` file can include a recorded path. Files are limited to 20 MB and 100,000 car records; keep segments short.

Fetch an existing historical interval with the included helper after checking the session's actual UTC timestamps:

```bash
python3 scripts/fetch_openf1.py --help
```

Provide `--session`, `--driver`, `--start`, `--end`, and `--output` plus optional `--location`. The helper requests only a fixed public OpenF1 endpoint and limits the interval to ten minutes. It fails explicitly on unavailable data or access errors; it never substitutes invented race data.

Reference: https://openf1.org/docs/ . Live access and historical access have different requirements; this application does not promise a live F1 data connection.

## Train the real-data prediction model

The legacy enriched reproducible training pipeline uses OpenF1 historical sessions, creates a cleaned label for an overtake within the next 60 seconds, uses the final session for validation-based early stopping, and exports its browser-compatible `dist/apex-model.js` artifact. That legacy artifact is not used by the integrated hybrid frontend:

```bash
python3 scripts/train_model.py
python3 scripts/train_xgboost.py
python3 scripts/train_xgboost_enriched.py
```

The enriched script adds `car_data`, `location`, `pit` and `race_control` to the original interval/position/lap/stint/weather/overtake pipeline. Car telemetry becomes rolling model features; pit and neutralisation windows are removed from labels; location data is cached and audited for replay use rather than used as a non-portable coordinate feature. The default runs use sessions `7953,7779,7787,9070`; pass another comma-separated list with `--sessions`. Raw API responses are cached under `data/openf1-cache/`. Artifacts, validation metrics, model-selection decision and the full machine-learning report are saved in `models/`. The selected model informs overtake probability, while the existing optimiser still selects the action and enforces energy constraints.

## Architecture

`dist/engine.js` is the sole strategic implementation. It runs in the browser and in Node. Python calls `scripts/engine-cli.cjs` with structured stdin, without shell interpolation. This avoids a second Python implementation drifting from offline results.

`backend/service.py` owns the subprocess boundary and SQLite audit store. `backend/standalone.py` and `backend/app.py` expose the same services through the Python standard library or FastAPI. A server comparison writes its record before returning a result. Repeated request IDs are idempotent; conflicting input reuse is rejected.

`dist/telemetry.js` handles replay construction, data validation, sorting and interpolation. Imported data remains in browser memory and is not uploaded. Browser audit records retain its provenance, not the full telemetry. The most recent 100 decisions are shown; SQLite keeps all server decisions. File-mode browser storage depends on the browser; JSON export is the portable backup.

| Endpoint | Method | Contract |
| --- | --- | --- |
| `/api/health` | GET | Version, adapter, SQLite status and WebSocket capability |
| `/api/scenarios` | GET | Model configuration and bundled scenario definitions |
| `/api/compare` | POST | `{input, provenance, request_id}`; computes and stores a comparison |
| `/api/validate` | POST | `{count, seed}`; runs the synthetic benchmark |
| `/api/audit?limit=100` | GET | Recent decision records |
| `/api/telemetry?soc=42` | GET | Labelled synthetic replay samples |
| `/ws/replay` | WebSocket, FastAPI only | Send `{time, soc}`; receive `{source: "synthetic", frame}` |

The shared engine rejects out-of-range and non-finite inputs. Local adapters reject cross-origin browser requests. The app is intended for a trusted laptop demo, not unauthenticated Internet hosting. Runtime data is written to `runtime/apex.db`; set `APEX_DATABASE` to use a different local database file.

## Verification

```bash
node --test tests/engine.test.cjs
python3 -m unittest discover -s tests -p 'test_*.py' -v
node scripts/validate.cjs
```

The baseline build passed 13 engine/import tests and 4 HTTP/storage tests. Two FastAPI/WebSocket tests require optional dependencies and are skipped if missing. The browser/Node parity check executes the shared browser bundle in an isolated JavaScript context; it is not a browser UI test.

The recorded 60-scenario run used 11,520 strategy rollouts and had 54 utility wins, 0 ties and 6 losses against the threshold policy, with zero executed model-constraint violations. These are synthetic results, not real-race gains. Re-run the suite on your laptop; timings depend on hardware. The generated detailed report is `runtime/benchmark.json`.

Browser visual QA was not run in this environment. Before presenting, check play/pause/seek, every scenario, a low-energy rejection, imported telemetry, comparison exports, reload persistence, and the four views on the actual presentation laptop. Test a second device as the fallback.

## Team handoff and release

Read `docs/DEMO_AND_TEAM.md` for the judge flow, four-person ownership and the three-day preparation / 24-hour event schedule. Check event rules before reusing this preparation build.

Create the single-file offline app and source archive with:

```bash
node scripts/package.cjs /absolute/path/to/release
```

Source files are deliberately buildless and self-contained, so the team can inspect and change the model without a bundler or a large dependency installation.
