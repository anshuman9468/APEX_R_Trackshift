# APEX-R Backend GNN Integration Report

## Status

**Backend GNN inference integrated and verified.**

The service is inference-only. It uses the frozen `gnn_proxy_v1` artifact and
does not retrain, recalibrate, tune, collect data, change labels, or connect
the score to strategy selection.

## Implementation

Files added or changed:

- `backend/gnn_adapter.py` — source-of-truth frozen adapter. It resolves the
  checkpoint through `FROZEN_MODEL_MANIFEST.json`, validates the frozen
  `SHA256SUMS.txt`, loads one `EdgeAwareGNN` in evaluation mode, and runs
  `torch.inference_mode()` only. It contains no training-module import,
  optimiser, loss or backward pass.
- `backend/service.py` — creates one adapter per application service and loads
  it during startup. The existing Phase 5 service shares that adapter.
- `backend/app.py` — adds FastAPI routes for model status and prediction.
- `backend/standalone.py` — adds the same routes to the dependency-light local
  HTTP server.
- `backend/phase5.py` — shares the startup adapter and uses the frozen output
  label; the existing strategy engine is unchanged.
- `backend/GNN_INTEGRATION_MANIFEST.json` — integration provenance, frozen
  artifact hashes, runtime versions and implementation hashes.
- `backend/measure_gnn_integration.py` — reproducible direct/HTTP latency and
  multi-sample reference-parity measurement.
- `tests/test_gnn_backend.py` — focused real-HTTP and validation tests.
- `scripts/build_gnn_predict_example.py` and
  `examples/gnn_predict_request.json` — label-free approved request example.
- `README.md` — startup, status and prediction commands.

## Frozen model provenance

- Version: `gnn_proxy_v1`.
- Task: fixed-pair next-lap-boundary classified-order position-swap proxy.
- Selected seed: 42.
- Selected epoch: 28.
- Checkpoint: `models/frozen/gnn_proxy_v1/gnn_seed_42_best.pt`.
- Checkpoint SHA-256:
  `d5ce7258fa2ec62f953d516816497f1cdbd04b1b839e362ef447f79c4e618d14`.
- Input schema: `gnn-proxy-graph-cache-v1`.
- Architecture: two GINEConv layers, hidden width 32, dropout 0.2, 18 node
  features, 4 edge features and 12 pair features.
- Preprocessing: frozen development-train means/stds, non-finite raw values
  mapped to those training means, explicit missingness masks retained.
- Dataset: 37 telemetry-coverage PASS races; 15,404 supervised graphs; 822
  positive proxy windows; 14,582 negative proxy windows; 19 excluded for
  missing attacker/target telemetry.
- Development-test metrics are historical development data, not an untouched
  final holdout: AP `0.10433820299403382`, ROC-AUC `0.7008568957128184`.

## Input contract

`POST /api/model/predict` accepts JSON with:

```json
{
  "request_id": "gnn-demo-0001",
  "decision_timestamp_session_sec": 2745.795,
  "attacker_driver": "ALB",
  "target_driver_fixed": "VET",
  "input_freshness": {
    "join_policy": "backward-only",
    "interpolation": false,
    "asof_tolerance_sec": 1.0,
    "lookback_sec": 10.0
  },
  "graph": {
    "race_id": "2019:Abu Dhabi Grand Prix:Race",
    "drivers": "sorted unique source driver identifiers",
    "attacker_idx": 0,
    "target_idx": 19,
    "node_raw": "N x 18",
    "edge_index": "E x 2 or 2 x E integer topology",
    "edge_raw": "E x 4",
    "pair_raw": "12 values",
    "asof_tolerance_sec": 1.0,
    "lookback_sec": 10.0
  }
}
```

The `graph` must be the raw multi-car graph emitted by the audited canonical
builder. The adapter keeps the trained node/edge/pair order, fixed attacker
and target mapping, topology, one-second backward-only as-of tolerance and
ten-second trailing lookback. `null` is the JSON representation of a supported
missing raw value; missing essential attacker/target car or position evidence
is rejected as unavailable. Non-essential missingness remains explicit and is
reported as a warning. Labels, outcome fields, future data and simplified
speed/gap-only payloads are rejected.

## Output contract

Successful responses contain:

- `status: "available"`;
- request ID, decision timestamp and fixed attacker/target identities;
- model version and checkpoint hash;
- `score_label: "experimental boundary position-swap proxy signal"`;
- raw `proxy_score` (with a read-only `score` compatibility alias for the
  existing Phase 5 display);
- input quality and freshness details; and
- measured `inference_latency_ms`.

The endpoint returns no ATTACK/HOLD/HARVEST/DEFEND decision. On checksum,
dependency or model failures, it returns HTTP 503 with `status: "unavailable"`
and a null score. Malformed, stale or incomplete essential-pair requests
return HTTP 422 and no score. Protected session `11353` is rejected before
graph conversion/sample access.

## Verification performed

| Check | Result |
| --- | --- |
| Frozen artifact checksums | PASS; 19 files checked at startup |
| Seed/epoch/config validation | PASS; seed 42, epoch 28, exact dimensions/config |
| Approved fixture CPU smoke | PASS; score `0.0052168904803693295` |
| Approved fixture CUDA load/inference | PASS; RTX 3050 CUDA device selected |
| Reference graph parity | PASS; 3 permitted graph samples, maximum absolute difference `0.0`, tolerance `1e-7` |
| Repeated inference stability | PASS; 20 direct requests, identical score |
| Focused backend tests | PASS; 5/5 |
| Existing HTTP/API tests | PASS; 4/4 |
| Invalid shape and label rejection | PASS |
| Protected-session rejection | PASS; no protected data accessed |
| Existing routes | PASS; health and Phase 5 metadata remain available |
| Source frozen-artifact immutability | PASS |

## Measured latency

The following measurements were run in the existing GPU environment with the
adapter explicitly set to CPU, so they are portable CPU figures rather than
targets:

| Path | Warm-up | Requests | p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| Direct adapter inference | 5,455.606 ms | 20 | 2.564 ms | 2.833 ms |
| Standalone HTTP prediction | server startup 174.724 ms | 20 | 5.309 ms | 5.985 ms |

The direct warm-up includes Python/PyG import, checksum verification, model
construction and the first inference. The HTTP figures include local JSON and
HTTP overhead. A separate CUDA smoke call selected `cuda` on the RTX 3050 and
returned the same finite score; its observed single-call latency was 3.809 ms.

Detailed measurements are in `backend/gnn_integration_metrics.json`.

## Commands for local testing

Start the backend from the external drive:

```bash
cd "/run/media/anshumandutta/ADATA HD710M PRO/APEX-R_Devsez_Full_Application"
source .venv_gnn_gpu/bin/activate
APEX_GNN_DEVICE=auto python run.py --standalone --port 8000
```

In a second terminal:

```bash
cd "/run/media/anshumandutta/ADATA HD710M PRO/APEX-R_Devsez_Full_Application"
source .venv_gnn_gpu/bin/activate
curl -sS http://127.0.0.1:8000/api/model/status
python scripts/build_gnn_predict_example.py
curl -sS -X POST http://127.0.0.1:8000/api/model/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/gnn_predict_request.json
```

Run focused checks without starting a persistent server:

```bash
APEX_GNN_DEVICE=cpu .venv_gnn_gpu/bin/python -m unittest tests.test_gnn_backend -v
APEX_GNN_DEVICE=cpu .venv_gnn_gpu/bin/python -m unittest tests.test_api -v
APEX_GNN_DEVICE=cpu .venv_gnn_gpu/bin/python -m backend.measure_gnn_integration
```

Use `APEX_GNN_DEVICE=cpu` if a portable CPU run is preferred, or
`APEX_GNN_DEVICE=cuda` to require CUDA. No packages or drivers are installed
by these commands.

## Limitations

The signal is not a calibrated overtake probability, confidence estimate,
ATTACK-benefit probability or verified on-track-overtake label. It does not
contain actual ERS/battery percentage, SOH, battery temperature, fuel load or
private pit-wall instructions. Historical timestamp alignment does not prove
live publication latency. The existing energy/rule simulator remains the
only source of final strategy recommendations, and its integration has not
been changed in this backend-only step.
