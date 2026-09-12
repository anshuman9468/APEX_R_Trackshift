# APEX-R Phase 5 release

This compact runtime release contains the dashboard, frozen seed-42 GNN checkpoint, compact approved replay fixture, simulator configuration and verification artifacts. It intentionally does not contain raw telemetry, graph caches, environments or credentials.

Start from this directory with `source .venv_gnn_gpu/bin/activate && python run.py` when the source environment is available, then open `http://127.0.0.1:8000/`. Without PyTorch/PyG the dashboard and simulator still run and the GNN status is explicitly unavailable. Opening `dist/index.html` provides the offline simulator/replay view.

The GNN is an experimental boundary position-swap advisory and has zero influence on recommendations. See `PHASE5_REPORT.md` and `MODEL_INFERENCE_MANIFEST.json`.
