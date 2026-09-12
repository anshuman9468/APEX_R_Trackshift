# APEX-R Phase 5

Run `python3 build_fixture.py` to rebuild the compact approved replay fixture from the preserved audited sources. Run `python3 build_release.py` to regenerate the manifest, benchmark, checksums and verified release archive.

The release has no raw telemetry or virtual environment. Use `source .venv_gnn_gpu/bin/activate && python run.py` for CUDA/CPU GNN advisory loading, or `python3 run.py` for the simulator/dashboard when optional PyTorch is unavailable.

The frozen model is advisory-only: its experimental boundary position-swap score has zero influence on ATTACK/HOLD/HARVEST/DEFEND selection.
