# RAM-safe APEX-R GNN experiment

This directory contains the host-memory repair and the completed experiment.
The canonical runner is `run_gnn_experiment_ram_safe.py`; it uses the
memory-mapped `disk_graph_store/` and does not eagerly load `graph_cache.pt`.

## Reproduce the repair checks

```bash
python3 -m venv /tmp/apex_gnn_venv
/tmp/apex_gnn_venv/bin/pip install numpy pandas scikit-learn torch --index-url https://download.pytorch.org/whl/cpu
/tmp/apex_gnn_venv/bin/pip install torch-geometric
/tmp/apex_gnn_venv/bin/python gnn_proxy_experiment_v1/convert_legacy_cache_bounded.py
/tmp/apex_gnn_venv/bin/python gnn_proxy_experiment_v1/measure_ram_safe.py
/tmp/apex_gnn_venv/bin/python gnn_proxy_experiment_v1/test_gnn_experiment_ram_safe.py
```

The conversion is optional when `disk_graph_store/` is already present. It
preserves the legacy cache and verifies every graph one at a time. The
measurement script performs no full training; it measures lazy loading,
train-only preprocessing and one forward/backward batch. The full experiment
runner is:

```bash
/tmp/apex_gnn_venv/bin/python gnn_proxy_experiment_v1/run_gnn_experiment_ram_safe.py
```

The package intentionally excludes the legacy nested cache and raw telemetry;
their hashes are recorded in `ram_fix_input_hashes_before.json` and
`input_hashes_after.json`.
