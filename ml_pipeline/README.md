# ML data preparation

`prepare_data.py` creates a non-destructive, model-ready copy of the selected
F1 telemetry and lap metrics.

## Run

The 2023 output has already been generated at `prepared_data_2023/`:

```bash
.venv-ml/bin/python ml_pipeline/prepare_data.py \
  --years 2023 \
  --output-root prepared_data_2023
```

To process every discovered season:

```bash
.venv-ml/bin/python ml_pipeline/prepare_data.py \
  --output-root prepared_data_all
```

Use `--overwrite` only when intentionally replacing a generated output
directory.

## Output

Each season is divided into deterministic event-level `train`, `validation`,
and `test` folders. Each contains compressed Parquet files for telemetry and
lap metrics. Raw numeric columns are retained, and normalized columns use the
`*_norm` suffix.

Normalization statistics are fitted from training rows only and are recorded
in `normalization_stats.json`. `metadata.json` records row counts, dropped-row
reasons, split settings, and the cleaning rules.

`valid_lap` is false for missing, out-of-range, or deleted lap times. No
physics-derived columns are generated yet; that is the next pipeline stage.

## Live MultiViewer bridge

`trackshift_server.py` polls MultiViewer's local Live Timing API and exposes
the normalized snapshot at `/api/live/status` and `/api/live/snapshot`. Start
MultiViewer, sign in with an eligible F1 TV subscription, open the Live Timing
window, and then start APEX-R:

```bash
cd "/Users/keshavgeer/Desktop/APex R"
.venv-ml/bin/python ml_pipeline/trackshift_server.py --port 8000
```

Live samples are captured locally at `runtime/multiviewer_live.jsonl` by
default. The bridge records car timing, speed, controls, gear, RPM, DRS, tyre
stint, weather, lap, track-status, race-control, exact meeting/circuit
metadata, full driver names, racing numbers, teams, and observed S1/S2/S3
sector records when MultiViewer provides them. The live dashboard lets the
operator choose a `MY DRIVER` and `TARGET DRIVER`; the selected relation
(ahead/behind) and observed gap are fed into the physics-aware policy. A live
safety-car, VSC, yellow, or red status constrains the policy toward HOLD and
reduces simulated overtake probability. Energy accounting, action branches,
and counterfactual outcomes remain explicitly simulated.

The Data & model dialog also exposes bounded recorded test replays from the
prepared 2024 and 2025 Grand Prix test splits. Those replays are historical
test inputs, not live MultiViewer sessions.

## Race-specific circuit maps

`dist/track_maps.js` is a compact catalogue of 2024 circuit layouts generated
from the fastest valid FastF1 race-lap XY telemetry. The frontend matches the
MultiViewer meeting/circuit or selected test replay to the corresponding
layout; matching 2025 events reuse the same physical circuit geometry. The
build command is:

```bash
python ml_pipeline/build_track_maps.py --years 2024 --session R --overwrite
```

MultiViewer timing identifies the meeting and driver progress but does not
provide live XY coordinates through this bridge, so live car markers are
placed by lap progress on the matched circuit path.
