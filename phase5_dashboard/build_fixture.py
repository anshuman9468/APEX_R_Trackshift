#!/usr/bin/env python3
"""Build the compact, offline Phase 5 replay/inference fixture.

The fixture is derived from an already approved development-validation race.
It intentionally contains no labels, endpoints, event outcomes, or raw-data
copies.  Per-driver values in a replay frame are backward as-of samples only;
there is no interpolation or future fill.
"""

from __future__ import annotations

import bisect
import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "fixture"
GRAPH_STORE = ROOT / "gnn_proxy_experiment_v1" / "disk_graph_store"
GRAPH_COVERAGE = ROOT / "gnn_proxy_experiment_gpu_v1" / "graph_coverage.csv"
CAR_SOURCE = ROOT / "full_race_telemetry_collection_final" / "telemetry_car.csv.gz"
POS_SOURCE = ROOT / "full_race_telemetry_collection_final" / "telemetry_position.csv.gz"
PROTECTED = {"11353"}
TOLERANCE = 1.0
BEFORE_SEC = 30
AFTER_SEC = 45


def finite(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def boolean(value):
    if str(value).strip().lower() in {"true", "1", "yes", "y", "on"}:
        return 1
    if str(value).strip().lower() in {"false", "0", "no", "n", "off"}:
        return 0
    return None


def choose_window():
    """Choose by sorted race identity and first eligible graph, never labels."""
    import sys
    sys.path.insert(0, str(ROOT / "gnn_proxy_experiment_v1"))
    from disk_graph_store import DiskGraphStore

    coverage = list(csv.DictReader(GRAPH_COVERAGE.open(newline="")))
    races = sorted({r["race_id"] for r in coverage
                    if r.get("proposed_split") == "development_validation"
                    and r.get("eligible_graph", "").lower() == "true"})
    races = [r for r in races if not any(tok in r for tok in PROTECTED)]
    if not races:
        raise RuntimeError("No approved development-validation replay race")
    store = DiskGraphStore(GRAPH_STORE)
    for idx, meta in enumerate(store.metadata):
        if meta["race_id"] == races[0]:
            # The record is used only as an input snapshot.  Deliberately do
            # not read or copy its label/outcome fields.
            graph = store.get(idx)
            public_meta = {k: meta[k] for k in (
                "window_id", "race_id", "proposed_split", "decision_lap",
                "decision_session_time_sec", "drivers", "attacker_driver",
                "target_driver_fixed", "attacker_idx", "target_idx",
                "asof_tolerance_sec", "lookback_sec")}
            return idx, public_meta, graph
    raise RuntimeError(f"No graph found for selected race {races[0]}")


def load_stream(path, race_id, drivers, start, end, kind):
    rows = {d: [] for d in drivers}
    if kind == "car":
        usecols = ["race_id", "driver", "session_time_sec", "speed_kmh",
                   "throttle_pct", "n_gear", "rpm", "brake", "drs", "date"]
    else:
        usecols = ["race_id", "driver", "session_time_sec", "x_m", "y_m",
                   "z_m", "status", "date"]
    # pandas' C parser keeps the scan bounded to one chunk while avoiding the
    # much slower Python csv parser over the 15M-row source stream.
    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols,
                             chunksize=250_000, low_memory=False):
        subset = chunk[(chunk["race_id"] == race_id)
                       & chunk["driver"].isin(rows)
                       & (chunk["session_time_sec"] >= start - TOLERANCE)
                       & (chunk["session_time_sec"] <= end)]
        for row in subset.to_dict("records"):
            t = finite(row.get("session_time_sec"))
            if t is None:
                continue
            if kind == "car":
                values = {
                    "speed_kmh": finite(row.get("speed_kmh")),
                    "throttle_pct": finite(row.get("throttle_pct")),
                    "brake": boolean(row.get("brake")),
                    "gear": finite(row.get("n_gear")),
                    "rpm": finite(row.get("rpm")),
                    "drs": boolean(row.get("drs")),
                    "date": row.get("date"),
                }
            else:
                values = {
                    "x_m": finite(row.get("x_m")), "y_m": finite(row.get("y_m")),
                    "z_m": finite(row.get("z_m")), "status": row.get("status"),
                    "date": row.get("date"),
                }
            rows[str(row["driver"])].append((t, values))
    for d in rows:
        rows[d].sort(key=lambda x: x[0])
    return rows


def asof(rows, t):
    times = [x[0] for x in rows]
    i = bisect.bisect_right(times, t) - 1
    if i < 0 or t - times[i] > TOLERANCE + 1e-9:
        return None
    source_t, values = rows[i]
    return {**values, "source_session_time_sec": source_t,
            "age_sec": max(0.0, t - source_t)}


def norm_points(points):
    finite_points = [(p[0], p[1]) for p in points if p[0] is not None and p[1] is not None]
    if not finite_points:
        return []
    xs, ys = [p[0] for p in finite_points], [p[1] for p in finite_points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    sx = 720.0 / max(1.0, xmax - xmin)
    sy = 400.0 / max(1.0, ymax - ymin)
    scale = min(sx, sy)
    ox = 40.0 + (720.0 - (xmax - xmin) * scale) / 2.0
    oy = 20.0 + (400.0 - (ymax - ymin) * scale) / 2.0
    return [[round(ox + (x - xmin) * scale, 3),
             round(oy + (ymax - y) * scale, 3)] for x, y in finite_points]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    graph_index, meta, graph = choose_window()
    race = meta["race_id"]
    if any(tok in race for tok in PROTECTED):
        raise RuntimeError("Protected race selected")
    t0 = float(meta["decision_session_time_sec"])
    start, end = t0 - BEFORE_SEC, t0 + AFTER_SEC
    drivers = list(meta["drivers"])
    car = load_stream(CAR_SOURCE, race, drivers, start, end, "car")
    pos = load_stream(POS_SOURCE, race, drivers, start, end, "position")
    grid = [float(math.floor(start)) + i for i in range(int(math.ceil(end) - math.floor(start)) + 1)]
    frames = []
    path_points = []
    for t in grid:
        cars = []
        for d in drivers:
            c, p = asof(car[d], t), asof(pos[d], t)
            if not c and not p:
                continue
            item = {"driver": d}
            if c:
                item.update({k: c[k] for k in ("speed_kmh", "throttle_pct", "brake", "gear", "rpm", "drs")})
                item["car_source_session_time_sec"] = c["source_session_time_sec"]
                item["car_age_sec"] = c["age_sec"]
                item["observed_car"] = True
            else:
                item.update({"speed_kmh": None, "throttle_pct": None, "brake": None,
                             "gear": None, "rpm": None, "drs": None,
                             "car_source_session_time_sec": None, "car_age_sec": None,
                             "observed_car": False})
            if p:
                item.update({k: p[k] for k in ("x_m", "y_m", "z_m", "status")})
                item["position_source_session_time_sec"] = p["source_session_time_sec"]
                item["position_age_sec"] = p["age_sec"]
                item["observed_position"] = True
            else:
                item.update({"x_m": None, "y_m": None, "z_m": None, "status": None,
                             "position_source_session_time_sec": None, "position_age_sec": None,
                             "observed_position": False})
            cars.append(item)
            if d == meta["attacker_driver"] and p and p["x_m"] is not None and p["y_m"] is not None:
                path_points.append((p["x_m"], p["y_m"]))
        frames.append({"t": t - start, "session_time_sec": t, "cars": cars})

    all_xy = [(c["x_m"], c["y_m"]) for f in frames for c in f["cars"]
              if c.get("x_m") is not None and c.get("y_m") is not None]
    if all_xy:
        xs, ys = [p[0] for p in all_xy], [p[1] for p in all_xy]
        bounds = {"xmin": min(xs), "xmax": max(xs), "ymin": min(ys), "ymax": max(ys)}
    else:
        bounds = None

    # Pair context is copied from the exact selected as-of graph, not rebuilt
    # from a label or endpoint.  These are display-only model-input references.
    pair_raw = np.asarray(graph["pair_raw"], dtype=float).tolist()
    compact_graph = {
        "graph_index": graph_index,
        "metadata": meta,
        "node_raw": np.asarray(graph["node_raw"], dtype=float).tolist(),
        "edge_index": np.asarray(graph["edge_index"], dtype=int).tolist(),
        "edge_raw": np.asarray(graph["edge_raw"], dtype=float).tolist(),
        "pair_raw": pair_raw,
        "source": "exact selected graph cache record; labels intentionally omitted",
    }
    fixture = {
        "schema_version": "phase5-historical-replay-v1",
        "selection": {
            "basis": "chronologically first development-validation race with eligible graph coverage",
            "race_id": race, "proposed_split": meta["proposed_split"],
            "not_selected_by_outcome": True,
        },
        "race": {"race_id": race, "event": race.split(":", 2)[1],
                 "year": int(race.split(":", 1)[0]), "session": "Race"},
        "timestamp_semantics": "FastF1 SessionTime seconds; each car/position value is an exact source sample selected backward-only within 1.0 second; no UTC/live-latency claim",
        "replay": {"start_session_time_sec": start, "end_session_time_sec": end,
                   "duration_sec": end - start, "cadence_sec": 1.0,
                   "frames": frames, "track_points": norm_points(path_points),
                   "track_bounds": bounds},
        "decision": {
            "window_id": meta["window_id"], "session_time_sec": t0,
            "replay_time_sec": t0 - start, "lap": meta["decision_lap"],
            "attacker_driver": meta["attacker_driver"],
            "target_driver": meta["target_driver_fixed"],
            "drivers": drivers, "asof_tolerance_sec": TOLERANCE,
            "feature_lookback_sec": meta["lookback_sec"],
            "pair_context_feature_names": ["attacker_target_speed_delta_kmh", "attacker_target_distance_m", "attacker_target_speed_delta_10s_kmh"],
            "pair_context_values": pair_raw[:3],
        },
        "inference_graph": compact_graph,
    }
    # Score the exact compact graph with the frozen adapter when the GPU/CPU
    # runtime is present.  This is an advisory reference value, not a label.
    try:
        sys.path.insert(0, str(ROOT))
        from backend.gnn_adapter import FrozenGNNAdapter
        advisory = FrozenGNNAdapter(device_preference="auto").predict(meta["window_id"])
        if advisory.get("status") == "available":
            fixture["decision"]["gnn_reference_score"] = advisory["score"]
    except Exception:
        pass
    (OUT / "phase5_fixture.json").write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    # Embed a small deterministic asset so opening dist/index.html remains an
    # offline demo; the server still serves the canonical JSON fixture.
    escaped = json.dumps(fixture, separators=(",", ":"))
    (OUT / "phase5_fixture.js").write_text("window.ApexHistoricalFixture=" + escaped + ";\n")
    print(json.dumps({"race_id": race, "graph_index": graph_index,
                      "window_id": meta["window_id"], "decision_session_time_sec": t0,
                      "frames": len(frames), "fixture_bytes": (OUT / "phase5_fixture.json").stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
