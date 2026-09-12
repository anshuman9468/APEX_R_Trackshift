# Historical replay state pipeline

This pipeline builds three strict, approved historical replay states for the
frozen `gnn_proxy_v1` model. It does not train, tune, score a protected
holdout, generate labels, or change the optimiser.

Run it with the existing GPU environment:

```bash
cd "/run/media/anshumandutta/ADATA HD710M PRO/APEX-R_Devsez_Full_Application"
source .venv_gnn_gpu/bin/activate
python historical_replay_state_pipeline_v1/replay_state_pipeline.py
python -m unittest discover -s historical_replay_state_pipeline_v1 -p 'test_*.py'
```

The output `artifacts/historical_state_traces.json` has one timestamped state
for each approved race. `pair_raw[1]` is a graph-only XYZ Euclidean distance in
metres. It is not a race gap in seconds; with no verified front/rear time gaps
or real battery state, the optimiser intentionally returns `unavailable` and
does not issue an action or action score.

After restarting the local API, inspect an approved state through:

```bash
curl -sS http://127.0.0.1:8001/api/phase5/historical-state/ee38ba6e51b8d9dbba9b
```

The response contains the frozen GNN advisory score, target identity and
freshness. It deliberately contains `optimizer.action: null`; this is the
correct result until source-backed front/rear time gaps (seconds) and a clearly
labelled simulated battery input are provided through a future contract.
