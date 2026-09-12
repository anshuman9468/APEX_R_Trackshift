# APEX-R Phase 5 report

## Delivered

- Frozen GNN: `gnn_proxy_gpu_v1_seed42_epoch28`, seed 42, best checkpoint epoch 28; checkpoint selection came from validation AP before development-test access.
- Historical replay: `2019:Abu Dhabi Grand Prix:Race`, an approved development-validation race selected by chronological first eligible coverage. The fixture has 77 exact one-second display frames and 20 drivers.
- Replay inputs use FastF1 `SessionTime` and backward-only source samples within 1.0 seconds. No UTC/live-latency claim is made; recorded future frames are reference-only.
- Frozen advisory score at the selected decision: `0.0052168904803693295`. It is labeled experimental boundary position-swap score and has zero influence on recommendations.
- Branching simulator: 3-lap deterministic seeded comparison, ATTACK/HOLD/HARVEST/DEFEND, battery-side MJ accounting, reserve/capacity/power checks, explicit infeasibility logging.

## Verification

- Existing energy/simulator suite: `node --test tests/engine.test.cjs` (13 passed).
- Focused Phase 5 checks: `node --test tests/phase5_engine.test.cjs` (3 passed); inference parity: `INFERENCE_PARITY.json`.
- Browser/Node shared-engine parity remains covered by the existing suite.
- Release benchmark: 24 predeclared synthetic scenarios, 4,608 paired rollouts, APEX vs threshold 22/0/2 wins/ties/losses; zero executed constraint violations in every policy row. These are simulated outcomes, not observed racing improvement.
- Measured runtime: see `RUNTIME_METRICS.json`; adapter load and warm inference are reported separately for CPU/CUDA, complete engine decision latency is measured separately, and process peak RSS is recorded.
- Protected holdouts and session 11353 were not read or evaluated by Phase 5.
- Preserved input hashes for the frozen checkpoint/preprocessing, graph/split metadata and the two audited telemetry sources are in `PHASE5_INPUT_HASHES.json`; no input values were rewritten.
- Historical outcome labels are not included in the fixture. The historical view is observed telemetry; branch positions, gaps and battery are simulated.

## Assumptions and non-claims

The simulator parameters are demo assumptions, not FIA compliance values. Actual ERS/battery percentage, SOH, battery temperature, private fuel load and pit-wall instructions are unavailable. GNN inference is CPU/CUDA portable when PyTorch/PyG is installed, but the browser-only demo uses the compact precomputed advisory reference and remains functional if the Python runtime is unavailable. No calibrated overtake probability or action-success confidence is shown.

## Judge sequence

1. Start with `python3 run.py` from the project directory and open `http://127.0.0.1:8000/`.
2. Open **Data & model → Load approved historical replay**; point out observed vs simulated labels.
3. Play at 4x and pause at `T+30.0s`; identify attacker `ALB` and target `VET`.
4. Show the frozen GNN advisory and explain zero influence; choose an action and compare futures with the same start/seed.
5. Open the validation tab, run 24 scenarios, show wins/ties/losses and constraints, then export the decision JSON.

## Startup

```bash
cd "/run/media/anshumandutta/ADATA HD710M PRO/APEX-R_Devsez_Full_Application"
source .venv_gnn_gpu/bin/activate
python run.py
```

For a portable CPU inference run, set `APEX_GNN_DEVICE=cpu`; the application otherwise selects CUDA when available. Opening `dist/index.html` directly still provides the embedded offline replay and simulator.
