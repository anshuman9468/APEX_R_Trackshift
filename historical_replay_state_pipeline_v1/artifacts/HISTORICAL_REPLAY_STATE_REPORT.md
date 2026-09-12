# Historical Replay State Pipeline v1

## Contract verified before execution

- Frozen model: `gnn_proxy_v1`, seed 42, epoch 28; checkpoint SHA-256 `d5ce7258fa2ec62f953d516816497f1cdbd04b1b839e362ef447f79c4e618d14`.
- Task: fixed-pair next-lap-boundary classified-order position-swap proxy. It is **not** a rolling five-second overtake predictor.
- Target: fixed classified direct predecessor at the prior completed lap boundary; it is never re-selected inside the horizon.
- Graph timing: 10-second trailing lookback and 1.0-second backward-only as-of tolerance; no interpolation/backfill/future joins.
- Protected session 11353 was rejected before window or telemetry access.

## Validation

- Approved races validated end-to-end: `2019:Abu Dhabi Grand Prix:Race` (1 model-applicable states), `2020:Hungarian Grand Prix:Race` (1 model-applicable states), `2020:70th Anniversary Grand Prix:Race` (1 model-applicable states).
- GNN applicable states: 3/3. Each passed fresh fixed-pair car and position checks.
- Every trace keeps target classification distinct from physical proximity. The graph's XYZ Euclidean distance is metres only and is never converted into a time gap.
- The optimiser was refreshed in strict mode but correctly returned **unavailable** for every trace: the audited sources do not supply a verified timestamped front/rear time gap in seconds, and real battery state is unobserved. No action is forced and no existing optimiser weight/configuration was changed.
- `action_scores` are null with their explicit source-data blocker instead of being computed from fabricated gaps.

## Backend integration

- The local FastAPI and standalone servers expose each trace at `/api/phase5/historical-state/{window_id}`.
- The endpoint is read-only: it returns the exact frozen advisory score and strict optimiser status generated here. It does not re-select targets, infer a five-second prediction, or change an action.

## Stream checks

- `car`: 17,031,555 source rows scanned; 15,944,390 rows used; non-monotonic requested streams: 0.
- `position`: 15,484,853 source rows scanned; 14,277,894 rows used; non-monotonic requested streams: 0.

## Input hashes

- `phase3_passed_races/prediction_windows_37.csv.gz`: `03a5b634245f0431479c2297d744b905ca803b888b3dd10705c81f4b4c98e01b`
- `phase3_passed_races/laptime_source_inventory_37.csv`: `a671a57826d0810447857c70e97c5b16eecc6238d2f0071d59734954ee7d4dd8`
- `phase3_races_audit_reconciliation_v1/proposed_chronological_split_37.csv`: `f613de06f939b7527984639776dcdcc7d00d5cde05b699bb5b20a3983f302557`
- `full_race_telemetry_collection_final/telemetry_car.csv.gz`: `22b6ed4e57c3e8e72616fe545f6244aeabbc9a15ab12158fa2de1e499f8ca9ca`
- `full_race_telemetry_collection_final/telemetry_position.csv.gz`: `5e3dc5c9229ce855e0b23d79ccdf50b7159e6b49daeb77659c7a916d49d7cf0d`
- `models/frozen/gnn_proxy_v1/FROZEN_MODEL_MANIFEST.json`: `97bd462899be13c43e127c50c03cb6a40819c93097b0355687a1ab9b4f78fc2a`
- `gnn_proxy_experiment_v1/graph_build_manifest.json`: `e0fcf2f2eda5a8dea3bbc4901a9d6c53ebc526f95212c3668e0845304276b8ff`
- `gnn_proxy_experiment_v1/disk_graph_store/manifest.json`: `175011c6ae933e396ff483f7d1411bd730b7b0356f74544a5706cd49daaea42f`
- `gnn_proxy_experiment_v1/disk_graph_store/metadata.json`: `f403a19d94467be5e3a5530931a448245bd343d1db489d1396881239b326edad`
- `gnn_proxy_experiment_v1/disk_graph_store/node_values.npy`: `028ff25e0a5f88740b81ccf14c037a6b0d535cf5f13955ae963bac8a0e6ecd12`
- `gnn_proxy_experiment_v1/disk_graph_store/edge_values.npy`: `4d8bca927c3c4e11692f79a660903f4c76d78e06d127b7e91c42a3ecc97dfc19`
- `gnn_proxy_experiment_v1/disk_graph_store/edge_index.npy`: `59f33d593b0b9cc8d20012e0cffa0e1031baa19be2e5010ff1fd6c295e417b93`
- `gnn_proxy_experiment_v1/disk_graph_store/pair_values.npy`: `3d1718ac9d1a88844da73986588a2e81f6af82e40a229b17210d9fa31058419f`
- `gnn_proxy_experiment_v1/disk_graph_store/node_offsets.npy`: `df09ad35e252a4317e792ab630d55b8c6dac85837ec73835c012d8fcc6333bd0`
- `gnn_proxy_experiment_v1/disk_graph_store/edge_offsets.npy`: `92324c249c4ecb7e263351ec5fb623e86c6a204045783e0910a6f4525727320e`

Detailed decision traces are in `historical_state_traces.json` and `historical_state_trace_summary.csv`.
