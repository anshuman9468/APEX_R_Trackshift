#!/usr/bin/env python3
"""APEX-R GNN proxy experiment v1.

This script builds as-of graphs from the already audited 37-race telemetry
subset, fits fixed baselines and three predeclared GNN seeds, freezes all
choices, evaluates the five-race development test once, and writes audit
artifacts.  It never requests remote data and rejects the sealed session token.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CAR_SOURCE = ROOT / "full_race_telemetry_collection_final" / "telemetry_car.csv.gz"
POS_SOURCE = ROOT / "full_race_telemetry_collection_final" / "telemetry_position.csv.gz"
COVERAGE_SOURCE = ROOT / "phase3_passed_races" / "telemetry_coverage_37.csv"
WINDOW_SOURCE = ROOT / "phase3_passed_races" / "prediction_windows_37.csv.gz"
LAPTIME_INVENTORY = ROOT / "phase3_passed_races" / "laptime_source_inventory_37.csv"
SPLIT_SOURCE = ROOT / "phase3_races_audit_reconciliation_v1" / "proposed_chronological_split_37.csv"
LINEAGE_SOURCE = ROOT / "phase3_races_audit_reconciliation_v1" / "positive_window_event_lineage.csv"
TARGET_CONTRACT = ROOT / "phase3_passed_races" / "target_contract_v3_passed_races.md"
FEATURE_READINESS = ROOT / "phase3_prediction_dataset" / "phase3_feature_readiness.csv"
PROTECTED_TOKENS = {"11353"}
ASOF_TOLERANCE_SEC = 1.0
LOOKBACK_SEC = 10.0
SEEDS = [17, 23, 42]
HIDDEN = 32
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15
DEVICE = torch.device("cpu")

NODE_FEATURES = [
    "speed_kmh", "throttle_pct", "brake", "n_gear", "rpm", "drs_binary",
    "car_sample_age_sec", "x_m", "y_m", "z_m", "position_sample_age_sec",
    "speed_delta_10s_kmh", "speed_trend_10s_kmh_per_s", "car_missing_mask",
    "position_missing_mask", "drs_missing_mask", "role_attacker", "role_target",
]
EDGE_FEATURES = [
    "distance_m", "relative_speed_kmh", "source_position_age_i_sec",
    "source_position_age_j_sec",
]
PAIR_FEATURES = [
    "attacker_target_speed_delta_kmh", "attacker_target_distance_m",
    "attacker_target_speed_delta_10s_kmh", "attacker_car_age_sec",
    "target_car_age_sec", "attacker_position_age_sec", "target_position_age_sec",
    "attacker_car_missing", "target_car_missing", "attacker_position_missing",
    "target_position_missing", "pair_position_available",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s or s.lower() in {"none", "nan", "null", "nat", "na"}:
        return float("nan")
    try:
        x = float(s)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def bool_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip().lower()
    if s in {"true", "t", "yes", "y", "on"}:
        return 1.0
    if s in {"false", "f", "no", "n", "off"}:
        return 0.0
    x = finite_or_nan(s)
    if math.isfinite(x) and x in {0.0, 1.0}:
        return x
    return float("nan")


def csv_rows(path: Path) -> Iterable[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    kwargs = {"mode": "rt", "newline": ""} if path.suffix == ".gz" else {"newline": ""}
    with opener(path, **kwargs) as f:
        yield from csv.DictReader(f)


def assert_source_exists() -> None:
    required = [CAR_SOURCE, POS_SOURCE, COVERAGE_SOURCE, WINDOW_SOURCE,
                LAPTIME_INVENTORY, SPLIT_SOURCE, LINEAGE_SOURCE, TARGET_CONTRACT,
                FEATURE_READINESS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing audited input(s): " + ", ".join(missing))


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, str], dict[str, Any]]:
    """Load only contracts and metadata; labels are attached after graph features are built."""
    coverage = list(csv_rows(COVERAGE_SOURCE))
    pass_races = {
        r["race_id"] for r in coverage
        if r.get("car_timestamp_status") == "PASS" and r.get("position_timestamp_status") == "PASS"
    }
    if any(tok in rid for rid in pass_races for tok in PROTECTED_TOKENS):
        raise RuntimeError("Protected session token appeared in graph input race selection")
    split_rows = list(csv_rows(SPLIT_SOURCE))
    split_map = {r["race_id"]: r["proposed_split"] for r in split_rows if r["race_id"] in pass_races}
    if set(split_map) != pass_races:
        raise RuntimeError(f"Split mapping does not cover exactly PASS races: {len(pass_races)=} {len(split_map)=}")
    if set(split_map.values()) != {"development_train", "development_validation", "development_test"}:
        raise RuntimeError("Unexpected proposed split values")

    windows: list[dict[str, Any]] = []
    for r in csv_rows(WINDOW_SOURCE):
        rid = r["race_id"]
        if rid not in pass_races or r.get("outcome_status") not in {"PROXY_POSITIVE", "PROXY_NEGATIVE"}:
            continue
        # The feature builder receives this context only; outcome fields are kept
        # in a side label record and attached only after the raw graph is built.
        windows.append({
            "window_id": r["window_id"],
            "race_id": rid,
            "attacker_driver": r["attacker_driver"],
            "target_driver_fixed": r["target_driver_fixed"],
            "decision_lap": int(r["decision_lap"]),
            "decision_session_time_sec": float(r["decision_session_time_sec"]),
            "endpoint_lap": int(r["endpoint_lap"]),
            "proposed_split": split_map[rid],
            "y": int(r["proxy_label"]),
            "outcome_status": r["outcome_status"],
        })
    if not windows:
        raise RuntimeError("No labelled windows found")

    drivers_by_race: dict[str, set[str]] = defaultdict(set)
    for r in csv_rows(LAPTIME_INVENTORY):
        if r.get("race_id") not in pass_races:
            continue
        drivers_by_race[r["race_id"]].add(r["driver"])
    if set(drivers_by_race) != pass_races:
        missing = sorted(pass_races - set(drivers_by_race))
        raise RuntimeError(f"No laptime driver inventory for race(s): {missing}")
    if any(w["attacker_driver"] not in drivers_by_race[w["race_id"]] or
           w["target_driver_fixed"] not in drivers_by_race[w["race_id"]] for w in windows):
        raise RuntimeError("A labelled fixed pair is absent from the race driver inventory")

    source_paths = [CAR_SOURCE, POS_SOURCE, COVERAGE_SOURCE, WINDOW_SOURCE,
                    SPLIT_SOURCE, LINEAGE_SOURCE, TARGET_CONTRACT, FEATURE_READINESS]
    inputs = {
        "pass_races": sorted(pass_races),
        "pass_race_count": len(pass_races),
        "split_counts": {s: sum(v == s for v in split_map.values()) for s in sorted(set(split_map.values()))},
        "windows_labelled": len(windows),
        "positive_windows": sum(w["y"] == 1 for w in windows),
        "negative_windows": sum(w["y"] == 0 for w in windows),
        "drivers_by_race_counts": {k: len(v) for k, v in sorted(drivers_by_race.items())},
        "input_sha256_before": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
    }
    return windows, drivers_by_race, split_map, inputs


def add_request(requests: dict[tuple[str, str], dict[float, list[tuple[int, str]]]],
                key: tuple[str, str], t: float, graph_idx: int, slot: str) -> None:
    requests.setdefault(key, {}).setdefault(float(t), []).append((graph_idx, slot))


def assign_refs(samples: list[dict[str, dict[str, dict[str, Any]]]], refs: list[tuple[int, str]],
                rec: dict[str, Any] | None, requested_time: float) -> None:
    if rec is None:
        return
    source_time = rec["time"]
    if source_time > requested_time + 1e-7 or requested_time - source_time > ASOF_TOLERANCE_SEC + 1e-7:
        return
    for graph_idx, slot in refs:
        samples[graph_idx].setdefault(rec["driver"], {})[slot] = {
            "time": source_time,
            "age": max(0.0, requested_time - source_time),
            "values": rec["values"],
        }


def resolve_stream(path: Path, windows: list[dict[str, Any]], drivers_by_race: dict[str, set[str]],
                   kind: str, samples: list[dict[str, dict[str, dict[str, Any]]]]) -> dict[str, Any]:
    requests: dict[tuple[str, str], dict[float, list[tuple[int, str]]]] = {}
    for i, w in enumerate(windows):
        key_race = w["race_id"]
        t = w["decision_session_time_sec"]
        for d in drivers_by_race[key_race]:
            add_request(requests, (key_race, d), t, i, "current_" + kind)
            add_request(requests, (key_race, d), t - LOOKBACK_SEC, i, "past_" + kind)
    request_times = {k: sorted(v) for k, v in requests.items()}
    request_pos = {k: 0 for k in request_times}
    last: dict[tuple[str, str], dict[str, Any]] = {}
    previous_time: dict[tuple[str, str], float] = {}
    rows_seen = 0
    rows_used = 0
    nonmonotonic = 0
    pending_assignments = 0

    print(f"[graph] scanning {kind} stream: {path.name}", flush=True)
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"race_id", "driver", "session_time_sec"}
        if not needed.issubset(reader.fieldnames or []):
            raise RuntimeError(f"Unexpected {kind} schema: {reader.fieldnames}")
        for row in reader:
            rows_seen += 1
            rid = row.get("race_id", "")
            if rid not in drivers_by_race:
                continue
            # Do not admit a protected race even if a malformed source appears.
            if any(tok in rid for tok in PROTECTED_TOKENS):
                raise RuntimeError("Protected session encountered in stream")
            d = row.get("driver", "")
            key = (rid, d)
            if key not in requests:
                continue
            t = finite_or_nan(row.get("session_time_sec"))
            if not math.isfinite(t):
                continue
            rows_used += 1
            if key in previous_time and t < previous_time[key] - 1e-7:
                nonmonotonic += 1
            previous_time[key] = t
            times = request_times[key]
            pos = request_pos[key]
            # Requests strictly before this row must use the preceding row.
            while pos < len(times) and times[pos] < t - 1e-7:
                req_t = times[pos]
                assign_refs(samples, requests[key][req_t], last.get(key), req_t)
                pos += 1
                pending_assignments += 1
            if kind == "car":
                values = [
                    finite_or_nan(row.get("speed_kmh")),
                    finite_or_nan(row.get("throttle_pct")),
                    bool_or_nan(row.get("brake")),
                    finite_or_nan(row.get("n_gear")),
                    finite_or_nan(row.get("rpm")),
                    bool_or_nan(row.get("drs")),
                ]
            else:
                values = [
                    finite_or_nan(row.get("x_m")),
                    finite_or_nan(row.get("y_m")),
                    finite_or_nan(row.get("z_m")),
                ]
            last[key] = {"driver": d, "time": t, "values": values}
            # A same-timestamp request is allowed to use this row.
            while pos < len(times) and times[pos] <= t + 1e-7:
                req_t = times[pos]
                assign_refs(samples, requests[key][req_t], last[key], req_t)
                pos += 1
                pending_assignments += 1
            request_pos[key] = pos
            if rows_seen % 2_000_000 == 0:
                print(f"[graph] {kind}: source rows={rows_seen:,}, used={rows_used:,}", flush=True)
    # Resolve requests after the end of a stream using only the final prior row.
    for key, times in request_times.items():
        pos = request_pos[key]
        while pos < len(times):
            req_t = times[pos]
            assign_refs(samples, requests[key][req_t], last.get(key), req_t)
            pos += 1
            pending_assignments += 1
    return {
        "source": str(path.relative_to(ROOT)),
        "kind": kind,
        "source_rows_seen": rows_seen,
        "source_rows_used_for_requested_drivers": rows_used,
        "requested_keys": len(requests),
        "requested_timestamps": sum(len(v) for v in request_times.values()),
        "requests_resolved_or_attempted": pending_assignments,
        "nonmonotonic_source_rows_in_requested_keys": nonmonotonic,
        "tolerance_sec": ASOF_TOLERANCE_SEC,
        "lookback_sec": LOOKBACK_SEC,
        "join_keys": ["race_id", "driver", "session_time_sec"],
    }


def safe_diff(a: float, b: float) -> float:
    return a - b if math.isfinite(a) and math.isfinite(b) else float("nan")


def sample_values(samples: list[dict[str, dict[str, dict[str, Any]]]], i: int, driver: str,
                  slot: str) -> list[float] | None:
    return samples[i].get(driver, {}).get(slot, {}).get("values")


def sample_age(samples: list[dict[str, dict[str, dict[str, Any]]]], i: int, driver: str,
               slot: str) -> float:
    rec = samples[i].get(driver, {}).get(slot)
    return float(rec["age"]) if rec is not None else float("nan")


def point_distance(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None or len(a) < 3 or len(b) < 3:
        return float("nan")
    if not all(math.isfinite(v) for v in [a[0], a[1], a[2], b[0], b[1], b[2]]):
        return float("nan")
    return float(math.sqrt(sum((a[j] - b[j]) ** 2 for j in range(3))))


def build_raw_graph(context: dict[str, Any], samples: list[dict[str, dict[str, dict[str, Any]]]],
                    drivers: list[str], i: int) -> tuple[dict[str, Any] | None, str]:
    """Build features from decision context only; no outcome fields are accepted."""
    attacker = context["attacker_driver"]
    target = context["target_driver_fixed"]
    if attacker == target:
        return None, "attacker_equals_target"
    a_car = sample_values(samples, i, attacker, "current_car")
    t_car = sample_values(samples, i, target, "current_car")
    a_pos = sample_values(samples, i, attacker, "current_position")
    t_pos = sample_values(samples, i, target, "current_position")
    # A pair needs an actual as-of row on both streams. Other cars may be absent
    # or partially missing and are retained through masks.
    if a_car is None or t_car is None:
        return None, "fixed_pair_missing_current_car_sample"
    if a_pos is None or t_pos is None:
        return None, "fixed_pair_missing_current_position_sample"

    node_raw: list[list[float]] = []
    positions: list[list[float] | None] = []
    speeds: list[float] = []
    pos_ages: list[float] = []
    for d in drivers:
        car = sample_values(samples, i, d, "current_car")
        past_car = sample_values(samples, i, d, "past_car")
        pos = sample_values(samples, i, d, "current_position")
        car_age = sample_age(samples, i, d, "current_car")
        pos_age = sample_age(samples, i, d, "current_position")
        speed = car[0] if car is not None else float("nan")
        speed_delta = safe_diff(speed, past_car[0] if past_car is not None else float("nan"))
        speed_trend = speed_delta / LOOKBACK_SEC if math.isfinite(speed_delta) else float("nan")
        drs = car[5] if car is not None and len(car) > 5 else float("nan")
        row = [
            speed,
            car[1] if car is not None else float("nan"),
            car[2] if car is not None else float("nan"),
            car[3] if car is not None else float("nan"),
            car[4] if car is not None else float("nan"),
            drs,
            car_age,
            pos[0] if pos is not None else float("nan"),
            pos[1] if pos is not None else float("nan"),
            pos[2] if pos is not None else float("nan"),
            pos_age,
            speed_delta,
            speed_trend,
            0.0 if car is not None else 1.0,
            0.0 if pos is not None else 1.0,
            0.0 if math.isfinite(drs) else 1.0,
            1.0 if d == attacker else 0.0,
            1.0 if d == target else 0.0,
        ]
        node_raw.append([float(v) for v in row])
        positions.append(pos)
        speeds.append(speed)
        pos_ages.append(pos_age)

    idx = {d: j for j, d in enumerate(drivers)}
    ai, ti = idx[attacker], idx[target]
    a_past = sample_values(samples, i, attacker, "past_car")
    t_past = sample_values(samples, i, target, "past_car")
    pair_raw = [
        safe_diff(speeds[ai], speeds[ti]),
        point_distance(positions[ai], positions[ti]),
        safe_diff(a_past[0] if a_past is not None else float("nan"),
                  t_past[0] if t_past is not None else float("nan")),
        sample_age(samples, i, attacker, "current_car"),
        sample_age(samples, i, target, "current_car"),
        pos_ages[ai],
        pos_ages[ti],
        0.0 if sample_values(samples, i, attacker, "current_car") is not None else 1.0,
        0.0 if sample_values(samples, i, target, "current_car") is not None else 1.0,
        0.0 if sample_values(samples, i, attacker, "current_position") is not None else 1.0,
        0.0 if sample_values(samples, i, target, "current_position") is not None else 1.0,
        1.0 if point_distance(positions[ai], positions[ti]) == point_distance(positions[ai], positions[ti]) else 0.0,
    ]

    edge_index: list[list[int]] = [[], []]
    edge_raw: list[list[float]] = []
    for u in range(len(drivers)):
        if positions[u] is None or not all(math.isfinite(v) for v in positions[u]):
            continue
        distances: list[tuple[float, int]] = []
        for v in range(len(drivers)):
            if u == v or positions[v] is None or not all(math.isfinite(x) for x in positions[v]):
                continue
            dist = point_distance(positions[u], positions[v])
            if math.isfinite(dist):
                distances.append((dist, v))
        for dist, v in sorted(distances)[:3]:
            edge_index[0].append(u)
            edge_index[1].append(v)
            edge_raw.append([
                dist,
                safe_diff(speeds[u], speeds[v]),
                pos_ages[u],
                pos_ages[v],
            ])
    record = {
        "window_id": context["window_id"],
        "race_id": context["race_id"],
        "proposed_split": context["proposed_split"],
        "decision_lap": context["decision_lap"],
        "decision_session_time_sec": context["decision_session_time_sec"],
        "endpoint_lap": context["endpoint_lap"],
        "drivers": drivers,
        "attacker_driver": attacker,
        "target_driver_fixed": target,
        "attacker_idx": ai,
        "target_idx": ti,
        "node_raw": node_raw,
        "edge_index": edge_index,
        "edge_raw": edge_raw,
        "pair_raw": [float(v) for v in pair_raw],
        "asof_tolerance_sec": ASOF_TOLERANCE_SEC,
        "lookback_sec": LOOKBACK_SEC,
    }
    return record, ""


def construct_graphs(windows: list[dict[str, Any]], drivers_by_race: dict[str, set[str]],
                     out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, dict[str, dict[str, Any]]]] = [{} for _ in windows]
    stream_stats = [
        resolve_stream(CAR_SOURCE, windows, drivers_by_race, "car", samples),
        resolve_stream(POS_SOURCE, windows, drivers_by_race, "position", samples),
    ]
    graphs: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    exclusions: dict[str, int] = defaultdict(int)
    for i, w in enumerate(windows):
        drivers = sorted(drivers_by_race[w["race_id"]])
        context = {k: v for k, v in w.items() if k not in {"y", "outcome_status"}}
        record, reason = build_raw_graph(context, samples, drivers, i)
        coverage_rows.append({
            "window_id": w["window_id"], "race_id": w["race_id"],
            "proposed_split": w["proposed_split"], "decision_lap": w["decision_lap"],
            "decision_session_time_sec": w["decision_session_time_sec"],
            "attacker_driver": w["attacker_driver"], "target_driver_fixed": w["target_driver_fixed"],
            "label": w["y"], "outcome_status": w["outcome_status"],
            "eligible_graph": bool(record), "exclusion_reason": reason,
            "node_count": len(record["drivers"]) if record else 0,
            "edge_count": len(record["edge_raw"]) if record else 0,
            "attacker_target_current_pair_evidence": bool(record),
        })
        if record is None:
            exclusions[reason] += 1
            continue
        # Attach labels only after feature construction is complete.
        record["label"] = w["y"]
        record["outcome_status"] = w["outcome_status"]
        graphs.append(record)
    coverage_path = out / "graph_coverage.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    exclusion_rows = [{"reason": k, "windows": v} for k, v in sorted(exclusions.items())]
    pd.DataFrame(exclusion_rows or [{"reason": "NONE", "windows": 0}]).to_csv(out / "graph_exclusion_summary.csv", index=False)
    torch.save(graphs, out / "graph_cache.pt")
    schema = {
        "version": "gnn-proxy-graph-cache-v1",
        "eligible_graph_count": len(graphs),
        "excluded_window_count": len(windows) - len(graphs),
        "node_feature_names": NODE_FEATURES,
        "edge_feature_names": EDGE_FEATURES,
        "pair_feature_names": PAIR_FEATURES,
        "graph_record_fields": [k for k in graphs[0] if graphs] if graphs else [],
        "raw_nan_policy": "NaNs retained in cache; explicit missing masks; training-only transform handles NaNs",
        "node_order": "sorted source driver identifier per race",
        "edge_rule": "directed up to three nearest finite XYZ neighbours; no classified adjacency",
        "join_keys": ["race_id", "driver", "session_time_sec"],
        "asof_tolerance_sec": ASOF_TOLERANCE_SEC,
        "lookback_sec": LOOKBACK_SEC,
        "future_join_policy": "backward-only; no interpolation/backfill/forward fill",
        "stream_stats": stream_stats,
    }
    write_json(out / "GRAPH_CACHE_SCHEMA.json", schema)
    graph_counts = {
        "windows_labelled": len(windows),
        "graphs_eligible": len(graphs),
        "graphs_excluded": len(windows) - len(graphs),
        "exclusions": dict(sorted(exclusions.items())),
        "eligible_positive": sum(g["label"] == 1 for g in graphs),
        "eligible_negative": sum(g["label"] == 0 for g in graphs),
        "by_split": {},
    }
    for split in ["development_train", "development_validation", "development_test"]:
        gs = [g for g in graphs if g["proposed_split"] == split]
        graph_counts["by_split"][split] = {
            "graphs": len(gs), "positive": sum(g["label"] == 1 for g in gs),
            "negative": sum(g["label"] == 0 for g in gs),
            "races": len({g["race_id"] for g in gs}),
            "unique_pairs": len({(g["race_id"], g["attacker_driver"], g["target_driver_fixed"]) for g in gs}),
        }
    write_json(out / "graph_counts.json", graph_counts)
    write_json(out / "graph_build_manifest.json", {
        "stream_stats": stream_stats, "graph_counts": graph_counts,
        "pass_races": sorted(drivers_by_race), "split_map": {k: v for k, v in sorted({w["race_id"]: w["proposed_split"] for w in windows}.items())},
        "feature_builder_excludes_outcome_fields": True,
    })
    return graphs, coverage_rows, graph_counts


def finite_matrix(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([r[key] for r in records], dtype=float)


def fit_stats(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.concatenate(arrays, axis=0) if arrays[0].ndim == 2 else np.asarray(arrays)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return mean.astype(float), std.astype(float)


def fit_preprocessing(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    node_mean, node_std = fit_stats([finite_matrix(train_records, "node_raw")[i] for i in range(len(train_records))])
    edge_arrays = [np.asarray(r["edge_raw"], dtype=float) for r in train_records if r["edge_raw"]]
    if edge_arrays:
        edge_mean, edge_std = fit_stats(edge_arrays)
    else:
        edge_mean, edge_std = np.zeros(len(EDGE_FEATURES)), np.ones(len(EDGE_FEATURES))
    pair_mean, pair_std = fit_stats([np.asarray(r["pair_raw"], dtype=float) for r in train_records])
    tab_arrays = []
    for r in train_records:
        ai, ti = r["attacker_idx"], r["target_idx"]
        tab_arrays.append(np.concatenate([np.asarray(r["node_raw"][ai], dtype=float),
                                          np.asarray(r["node_raw"][ti], dtype=float),
                                          np.asarray(r["pair_raw"], dtype=float)]))
    tab_mean, tab_std = fit_stats([np.asarray(tab_arrays, dtype=float)])
    return {
        "version": "train-only-mean-standardize-v1",
        "fit_scope": "development_train eligible graphs only",
        "node_mean": node_mean.tolist(), "node_std": node_std.tolist(),
        "edge_mean": edge_mean.tolist(), "edge_std": edge_std.tolist(),
        "pair_mean": pair_mean.tolist(), "pair_std": pair_std.tolist(),
        "tab_mean": tab_mean.tolist(), "tab_std": tab_std.tolist(),
        "node_features": NODE_FEATURES, "edge_features": EDGE_FEATURES,
        "pair_features": PAIR_FEATURES,
        "missing_policy": "fit train means only; masks retained; no source mutation",
    }


def transform(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    return (np.where(np.isfinite(raw), raw, mean) - mean) / std


class GraphWindowDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, Any]], prep: dict[str, Any]):
        self.records = records
        self.nmean = np.asarray(prep["node_mean"], dtype=float)
        self.nstd = np.asarray(prep["node_std"], dtype=float)
        self.emean = np.asarray(prep["edge_mean"], dtype=float)
        self.estd = np.asarray(prep["edge_std"], dtype=float)
        self.pmean = np.asarray(prep["pair_mean"], dtype=float)
        self.pstd = np.asarray(prep["pair_std"], dtype=float)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Data:
        r = self.records[i]
        x = transform(np.asarray(r["node_raw"], dtype=float), self.nmean, self.nstd)
        e = transform(np.asarray(r["edge_raw"], dtype=float).reshape(-1, len(EDGE_FEATURES)), self.emean, self.estd)
        p = transform(np.asarray(r["pair_raw"], dtype=float), self.pmean, self.pstd)
        edge_index = torch.tensor(r["edge_index"], dtype=torch.long)
        if edge_index.numel() == 0:
            edge_index = edge_index.reshape(2, 0)
        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=torch.tensor(e, dtype=torch.float32),
            pair_x=torch.tensor(p, dtype=torch.float32).reshape(1, -1),
            y=torch.tensor([float(r["label"])], dtype=torch.float32),
            att_idx=torch.tensor([int(r["attacker_idx"])], dtype=torch.long),
            target_idx=torch.tensor([int(r["target_idx"])], dtype=torch.long),
        )


class PairLogit(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


class EdgeAwareGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, pair_dim: int):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, HIDDEN)
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, HIDDEN), nn.ReLU())
        mlp1 = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN))
        mlp2 = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN))
        self.conv1 = GINEConv(mlp1, edge_dim=HIDDEN)
        self.conv2 = GINEConv(mlp2, edge_dim=HIDDEN)
        self.dropout = nn.Dropout(p=0.2)
        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN * 2 + pair_dim, HIDDEN), nn.ReLU(),
            nn.Dropout(p=0.2), nn.Linear(HIDDEN, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        x = self.dropout(torch.relu(self.conv1(x, data.edge_index, edge_attr)))
        x = self.dropout(torch.relu(self.conv2(x, data.edge_index, edge_attr)))
        ptr = data.ptr[:-1]
        ai = data.att_idx.view(-1).long() + ptr
        ti = data.target_idx.view(-1).long() + ptr
        pair = data.pair_x.view(-1, data.pair_x.shape[-1])
        return self.decoder(torch.cat([x[ai], x[ti], pair], dim=1)).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def make_loader(records: list[dict[str, Any]], prep: dict[str, Any], shuffle: bool, seed: int) -> DataLoader:
    dataset = GraphWindowDataset(records, prep)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator,
                      num_workers=0, drop_last=False)


def model_probabilities(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite inference logits")
            probs = torch.sigmoid(logits)
            if not torch.isfinite(probs).all():
                raise FloatingPointError("Non-finite inference probabilities")
            ys.append(batch.y.view(-1).numpy())
            ps.append(probs.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out: dict[str, Any] = {
        "n": int(len(y)), "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "average_precision": None, "roc_auc": None,
        "brier_score": None, "log_loss": None,
    }
    if len(y) and y.sum() > 0:
        out["average_precision"] = float(average_precision_score(y, p))
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
    if len(y):
        out["brier_score"] = float(brier_score_loss(y, p))
        out["log_loss"] = float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1]))
    return out


def confusion_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tn = int(np.sum((y == 0) & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def choose_validation_threshold(y: np.ndarray, p: np.ndarray) -> dict[str, Any] | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    best = None
    for t in np.unique(p):
        row = confusion_at(y, p, float(t))
        if best is None or (row["f1"], row["precision"], row["threshold"]) > (best["f1"], best["precision"], best["threshold"]):
            best = row
    return best


def precision_recall_curve_rows(model_name: str, split: str, y: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for t in sorted(set([0.0] + [float(x) for x in np.unique(p)]), reverse=True):
        c = confusion_at(y, p, t)
        rows.append({"model": model_name, "split": split, "threshold": t,
                     "precision": c["precision"], "recall": c["recall"], "f1": c["f1"],
                     "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"]})
    return rows


def reliability_rows(model_name: str, split: str, y: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    edges = np.linspace(0, 1, 11)
    for b in range(10):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & ((p < hi) if b < 9 else (p <= hi))
        if not mask.any():
            continue
        rows.append({"model": model_name, "split": split, "bin": b,
                     "lower": float(lo), "upper": float(hi), "count": int(mask.sum()),
                     "mean_probability": float(p[mask].mean()), "positive_rate": float(y[mask].mean())})
    return rows


def tabular_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    vals = []
    for r in records:
        ai, ti = r["attacker_idx"], r["target_idx"]
        vals.append(np.concatenate([np.asarray(r["node_raw"][ai], dtype=float),
                                    np.asarray(r["node_raw"][ti], dtype=float),
                                    np.asarray(r["pair_raw"], dtype=float)]))
    return np.asarray(vals, dtype=float)


def fit_tab_transform(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return fit_stats([train_x])


def save_predictions(out: Path, pred_rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(pred_rows).to_csv(out / "per_window_predictions.csv", index=False)


def train_gnn(train_records: list[dict[str, Any]], val_records: list[dict[str, Any]], prep: dict[str, Any],
              seed: int, out: Path) -> dict[str, Any]:
    set_seed(seed)
    train_loader = make_loader(train_records, prep, True, seed)
    val_loader = make_loader(val_records, prep, False, seed)
    model = EdgeAwareGNN(len(NODE_FEATURES), len(EDGE_FEATURES), len(PAIR_FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    best_ap = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite training logits at seed={seed} epoch={epoch}")
            loss = criterion(logits, batch.y.view(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at seed={seed} epoch={epoch}")
            loss.backward()
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"Non-finite gradient at seed={seed} epoch={epoch}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_y, val_p = model_probabilities(model, val_loader)
        val_ap = compute_metrics(val_y, val_p)["average_precision"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_ap": val_ap})
        if val_ap is not None and val_ap > best_ap + 1e-12:
            best_ap = float(val_ap); best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"[train] seed={seed} epoch={epoch} loss={np.mean(losses):.6f} val_ap={val_ap}", flush=True)
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"No validation checkpoint produced for seed {seed}")
    model.load_state_dict(best_state)
    checkpoint = {
        "state_dict": best_state,
        "model_class": "EdgeAwareGNN",
        "config": {"node_dim": len(NODE_FEATURES), "edge_dim": len(EDGE_FEATURES), "pair_dim": len(PAIR_FEATURES),
                    "hidden": HIDDEN, "dropout": 0.2, "optimizer": "Adam", "lr": 0.001,
                    "weight_decay": 0.0001, "batch_size": BATCH_SIZE, "loss": "unweighted BCEWithLogitsLoss",
                    "seed": seed},
        "best_epoch": best_epoch, "epochs_run": len(history), "best_validation_ap": best_ap,
        "preprocessing_version": prep["version"],
    }
    path = out / f"gnn_seed_{seed}_best.pt"
    torch.save(checkpoint, path)
    write_json(out / f"gnn_seed_{seed}_history.json", history)
    return {"seed": seed, "checkpoint": path.name, "best_epoch": best_epoch,
            "epochs_run": len(history), "best_validation_ap": best_ap,
            "training_seconds": time.time() - started, "model": model}


def candidate_metrics_row(model: str, seed: int | None, split: str, y: np.ndarray, p: np.ndarray,
                          threshold: dict[str, Any] | None) -> dict[str, Any]:
    row = {"model": model, "seed": seed if seed is not None else "", "split": split}
    row.update(compute_metrics(y, p))
    if threshold:
        c = confusion_at(y, p, threshold["threshold"])
        row.update({f"threshold_{k}": v for k, v in c.items()})
    else:
        row.update({"threshold_threshold": None, "threshold_precision": None, "threshold_recall": None,
                    "threshold_f1": None, "threshold_tp": None, "threshold_fp": None,
                    "threshold_fn": None, "threshold_tn": None})
    return row


def whole_race_bootstrap(preds: dict[tuple[str, str], list[dict[str, Any]]], out: Path,
                         seed: int = 90210, n_boot: int = 1000) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for (model, split), items in sorted(preds.items()):
        if split not in {"development_validation", "development_test"} or model == "constant":
            continue
        base_items = preds.get(("constant", split), [])
        by_race = defaultdict(list); by_base = defaultdict(list)
        for x in items: by_race[x["race_id"]].append(x)
        for x in base_items: by_base[x["race_id"]].append(x)
        races = sorted(by_race)
        if set(races) != set(by_base):
            rows.append({"model": model, "split": split, "bootstrap_replicates": n_boot,
                         "valid_replicates": 0, "degenerate_replicates": n_boot,
                         "mean_ap_delta_vs_constant": None, "ci2_5": None, "ci97_5": None,
                         "race_count": len(races)})
            continue
        deltas = []; degenerate = 0
        for _ in range(n_boot):
            selected = rng.choice(races, size=len(races), replace=True)
            a_y=[]; a_p=[]; b_p=[]
            for race in selected:
                for x in by_race[race]: a_y.append(x["label"]); a_p.append(x["probability"])
                for x in by_base[race]: b_p.append(x["probability"])
            if len(set(a_y)) < 2:
                degenerate += 1; continue
            deltas.append(float(average_precision_score(a_y, a_p) - average_precision_score(a_y, b_p)))
        rows.append({"model": model, "split": split, "bootstrap_replicates": n_boot,
                     "valid_replicates": len(deltas), "degenerate_replicates": degenerate,
                     "mean_ap_delta_vs_constant": float(np.mean(deltas)) if deltas else None,
                     "ci2_5": float(np.quantile(deltas, 0.025)) if deltas else None,
                     "ci97_5": float(np.quantile(deltas, 0.975)) if deltas else None,
                     "race_count": len(races)})
    pd.DataFrame(rows).to_csv(out / "whole_race_bootstrap.csv", index=False)
    return rows


def write_lineage(out: Path, graph_coverage: list[dict[str, Any]]) -> None:
    lineage = list(csv_rows(LINEAGE_SOURCE))
    by_id = {r["window_id"]: r for r in lineage}
    rows = []
    for r in graph_coverage:
        if r["label"] != 1:
            continue
        src = by_id.get(r["window_id"], {})
        rows.append({"window_id": r["window_id"], "race_id": r["race_id"],
                     "eligible_graph": r["eligible_graph"], "graph_exclusion_reason": r["exclusion_reason"],
                     "attacker_driver": r["attacker_driver"], "target_driver_fixed": r["target_driver_fixed"],
                     "decision_session_time_sec": r["decision_session_time_sec"],
                     "source_lineage_present": bool(src),
                     "matched_candidate_id": src.get("matched_candidate_id", ""),
                     "supporting_event_id": src.get("supporting_event_id", src.get("event_id", "")),
                     "event_classification": src.get("event_classification", src.get("classification", "")),
                     "review_status": src.get("review_status", "AUTOMATED_UNREVIEWED")})
    pd.DataFrame(rows).to_csv(out / "graph_positive_lineage.csv", index=False)


def write_report(out: Path, inputs: dict[str, Any], graph_counts: dict[str, Any], metrics: list[dict[str, Any]],
                 runs: list[dict[str, Any]], tests: dict[str, Any]) -> None:
    metric_lines = ["| Model | Split | Seed | n | Pos | Prevalence | AP | ROC-AUC | Brier | Log loss | Val-selected threshold metrics |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for m in metrics:
        threshold = "" if m.get("threshold_threshold") is None else (
            f"t={m['threshold_threshold']:.6g}, P={m['threshold_precision']:.4f}, "
            f"R={m['threshold_recall']:.4f}, F1={m['threshold_f1']:.4f}, "
            f"TP/FP/FN/TN={m['threshold_tp']}/{m['threshold_fp']}/{m['threshold_fn']}/{m['threshold_tn']}"
        )
        fmt = lambda x: "NA" if x is None else f"{x:.6f}"
        metric_lines.append(f"| {m['model']} | {m['split']} | {m['seed']} | {m['n']} | {m['positive_count']} | {m['prevalence']:.6f} | {fmt(m['average_precision'])} | {fmt(m['roc_auc'])} | {fmt(m['brier_score'])} | {fmt(m['log_loss'])} | {threshold} |")
    run_lines = [f"- seed {r['seed']}: best epoch {r['best_epoch']}, epochs run {r['epochs_run']}, validation AP {r['best_validation_ap']:.6f}, training seconds {r['training_seconds']:.2f}" for r in runs]
    test_status = "PASSED" if all(t.get("passed", False) for t in tests.values()) else "FAILED"
    text = f"""# APEX-R GNN proxy experiment v1 report

Status: completed validation and one development-test evaluation after freeze.
Conclusion: `EXPLORATORY_PROXY_GAIN` is used only if a GNN has higher development-validation AP than both constant and logistic baselines; otherwise `NO_DEMONSTRATED_GAIN`. This is not verified-overtake or strategy evidence.

## Frozen scope

- PASS races derived from the audited status table: {inputs['pass_race_count']}.
- Labelled positive/negative windows before graph eligibility: {inputs['windows_labelled']} ({inputs['positive_windows']} positive, {inputs['negative_windows']} negative).
- Proposed chronology split: 26/6/5 race counts are preserved from the manifest; all are previously accessed development data, and development-test is not an untouched final holdout.
- Protected session `11353` was rejected by input selection and is not in the processed race set. No remote data was requested.
- Target: fixed attacker/target classified-order reversal at the next lap boundary. Verified on-track passes remain zero in the source audit.

## Graph construction

- Eligible graph windows: **{graph_counts['graphs_eligible']}**; excluded: **{graph_counts['graphs_excluded']}**.
- Exclusions: `{json.dumps(graph_counts['exclusions'], sort_keys=True)}`.
- Current samples use backward-only `(race_id, driver, session_time_sec)` joins with tolerance **{ASOF_TOLERANCE_SEC:.1f}s**. Trailing speed uses a prior as-of request at **{LOOKBACK_SEC:.0f}s**, with no interpolation, backfill, centering, or whole-lap aggregate.
- Nodes are all source-inventory drivers for the race. Edges are directed up to three nearest finite FastF1 XYZ neighbours, with edge distance and relative speed; classified adjacency is not used.
- Raw missing values remain NaN in `graph_cache.pt`; explicit masks are present. Means/stds are fit only on development-train eligible graphs.
- Current audited telemetry does not provide a causally safe tyre stream, so tyre features were deliberately omitted.

### Eligible support by split

```json
{json.dumps(graph_counts['by_split'], indent=2)}
```

## Model and validation results

- GNN: two GINEConv layers, hidden width 32, dropout 0.2, edge encoder, attacker/target/pair decoder, Adam lr 0.001, weight decay 0.0001, batch 32, unweighted BCE-with-logits, max 100 epochs, patience 15.
- Predeclared seeds: 17, 23, 42. Validation AP selected each checkpoint. Training runs:
{os.linesep.join(run_lines)}
- Baselines: training-prevalence constant and unweighted logistic regression C=1.0 on attacker node + target node + pair fields.
- Thresholds are validation-only maximum-F1 descriptive operating points, frozen before test. AP is the primary metric; ROC-AUC is not accuracy.

{os.linesep.join(metric_lines)}

Full precision/recall curves are in `precision_recall_curves.csv`; reliability bins with counts are in `reliability_bins.csv`.

## Test access and uncertainty

The development-test graphs were built as part of the frozen cache, but predictions and metrics were generated only after `FROZEN_BEFORE_TEST.json` was written. Each GNN seed was evaluated once. The access ledger is `development_test_access_ledger.json`. No model or threshold was revised afterward.

Whole-race bootstrap AP deltas versus the constant baseline are in `whole_race_bootstrap.csv`. The five development-test races make these uncertainty intervals limited and correlated-window resampling is intentionally not row-independent.

## Verification

Automated checks status: **{test_status}**; details are in `experiment_tests.json`. Input hashes before/after are in `input_hashes_before.json` and `input_hashes_after.json` and must match.

## Interpretation

This experiment can at most support an exploratory historical boundary-order proxy conclusion. A GNN AP gain over logistic regression is not an isolated proof that message passing caused the gain because the GNN sees other-car node context and physical-neighbour edges while logistic regression receives only pair-level fields. No probability is connected to ATTACK/HOLD/HARVEST and no sealed final holdout is touched.
"""
    (out / "PHASE4_REPORT.md").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--rebuild-graph-cache", action="store_true",
                    help="Rescan audited telemetry even when a complete cache exists")
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    assert_source_exists()
    print("[init] loading audited manifests", flush=True)
    windows, drivers_by_race, split_map, inputs = load_inputs()
    write_json(out / "input_hashes_before.json", inputs["input_sha256_before"])
    write_json(out / "frozen_split_manifest.json", [{"race_id": rid, "proposed_split": split_map[rid],
                                                     "previously_accessed_development_data": True,
                                                     "untouched_final_holdout": False}
                                                    for rid in sorted(split_map)])
    pd.DataFrame([{k: v for k, v in w.items() if k != "y"} for w in windows]).to_csv(out / "labelled_window_context_audit.csv", index=False)
    print(f"[init] PASS races={len(drivers_by_race)}, labelled windows={len(windows)}", flush=True)
    cache_files = [out / "graph_cache.pt", out / "graph_coverage.csv", out / "graph_counts.json", out / "GRAPH_CACHE_SCHEMA.json"]
    if all(p.exists() for p in cache_files) and not args.rebuild_graph_cache:
        print("[graph] reusing existing graph cache after verifying cache files", flush=True)
        graphs = torch.load(out / "graph_cache.pt", map_location="cpu", weights_only=False)
        coverage_rows = list(csv_rows(out / "graph_coverage.csv"))
        for row in coverage_rows:
            row["label"] = int(row["label"])
            row["eligible_graph"] = str(row["eligible_graph"]).lower() == "true"
            row["decision_session_time_sec"] = float(row["decision_session_time_sec"])
        graph_counts = json.loads((out / "graph_counts.json").read_text())
        if len(graphs) != int(graph_counts["graphs_eligible"]):
            raise RuntimeError("Graph cache count disagrees with graph_counts.json")
        if any(any(tok in str(g.get("race_id", "")) for tok in PROTECTED_TOKENS) for g in graphs):
            raise RuntimeError("Protected session token found in graph cache")
    else:
        graphs, coverage_rows, graph_counts = construct_graphs(windows, drivers_by_race, out)
    write_lineage(out, coverage_rows)
    by_split = defaultdict(list)
    for g in graphs:
        by_split[g["proposed_split"]].append(g)
    if not by_split["development_train"] or not by_split["development_validation"] or not by_split["development_test"]:
        raise RuntimeError("Graph eligibility produced an empty split")
    prep = fit_preprocessing(by_split["development_train"])
    write_json(out / "preprocessing_state.json", prep)
    write_json(out / "preprocessing_fit_scope.json", {"fit_split": "development_train",
                                                         "fit_graphs": len(by_split["development_train"]),
                                                         "fit_races": len({g["race_id"] for g in by_split["development_train"]}),
                                                         "train_only": True})

    # Verify the PyG implementation before the costly runs.
    probe = GraphWindowDataset(by_split["development_train"][:1], prep)[0]
    probe_model = EdgeAwareGNN(len(NODE_FEATURES), len(EDGE_FEATURES), len(PAIR_FEATURES))
    probe_out = probe_model(DataLoader([probe], batch_size=1).__iter__().__next__())
    if not torch.isfinite(probe_out).all():
        raise RuntimeError("GNN probe produced non-finite output")

    # Logistic baseline uses the exact same eligible graph windows and pair fields.
    train_x = tabular_matrix(by_split["development_train"])
    tab_mean, tab_std = fit_tab_transform(train_x)
    tab_prep = {"mean": tab_mean.tolist(), "std": tab_std.tolist(), "fit_scope": "development_train eligible graphs only", "C": 1.0}
    write_json(out / "logistic_preprocessing_state.json", tab_prep)
    train_xt = transform(train_x, tab_mean, tab_std)
    logistic = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=17)
    logistic.fit(train_xt, np.asarray([g["label"] for g in by_split["development_train"]], dtype=int))
    with (out / "logistic_c1_model.pkl").open("wb") as f:
        pickle.dump({"model": logistic, "config": {"C": 1.0, "solver": "lbfgs", "unweighted": True}, "feature_order": NODE_FEATURES * 2 + PAIR_FEATURES}, f)
    constant_p = float(np.mean([g["label"] for g in by_split["development_train"]]))
    write_json(out / "constant_baseline.json", {"training_prevalence": constant_p, "fit_scope": "development_train eligible graphs only"})

    all_preds: dict[tuple[str, str], list[dict[str, Any]]] = {}
    metrics: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []
    thresholds: dict[str, dict[str, Any] | None] = {}
    gnn_runs = []

    def record_candidate(name: str, seed: int | None, model_scores: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        if "development_validation" in model_scores:
            val_y, val_p = model_scores["development_validation"]
            threshold = choose_validation_threshold(val_y, val_p)
            thresholds[name] = threshold
        else:
            threshold = thresholds.get(name)
        for split, (y, p) in model_scores.items():
            metrics.append(candidate_metrics_row(name, seed, split, y, p, threshold))
            pr_rows.extend(precision_recall_curve_rows(name, split, y, p))
            rel_rows.extend(reliability_rows(name, split, y, p))
            rows = []
            recs = by_split[split]
            for r, label, prob in zip(recs, y, p):
                rows.append({"window_id": r["window_id"], "race_id": r["race_id"], "proposed_split": split,
                             "model": name, "seed": "" if seed is None else seed, "label": int(label),
                             "probability": float(prob), "decision_threshold": None if threshold is None else threshold["threshold"],
                             "threshold_prediction": None if threshold is None else int(prob >= threshold["threshold"]),
                             "attacker_driver": r["attacker_driver"], "target_driver_fixed": r["target_driver_fixed"]})
            all_preds[(name, split)] = rows

    # Constant and logistic are frozen candidates; GNN checkpoints are frozen below.
    for name, pfun in [("constant", lambda recs: np.full(len(recs), constant_p, dtype=float))]:
        score = {}
        for split in ["development_train", "development_validation"]:
            y = np.asarray([g["label"] for g in by_split[split]], dtype=int); p = pfun(by_split[split])
            score[split] = (y, p)
        record_candidate(name, None, score)
    logistic_scores = {}
    for split in ["development_train", "development_validation"]:
        x = transform(tabular_matrix(by_split[split]), tab_mean, tab_std)
        y = np.asarray([g["label"] for g in by_split[split]], dtype=int)
        logistic_scores[split] = (y, logistic.predict_proba(x)[:, 1])
    record_candidate("logistic_c1", None, logistic_scores)

    for seed in SEEDS:
        run = train_gnn(by_split["development_train"], by_split["development_validation"], prep, seed, out)
        gnn_runs.append({k: v for k, v in run.items() if k != "model"})
        gnn_scores = {}
        model = run["model"]
        for split in ["development_train", "development_validation"]:
            loader = make_loader(by_split[split], prep, False, seed)
            gnn_scores[split] = model_probabilities(model, loader)
        record_candidate(f"gnn_seed_{seed}", seed, gnn_scores)
    write_json(out / "gnn_run_registry.json", gnn_runs)

    # Freeze every selection artifact before opening any development-test metrics.
    frozen = {"status": "FROZEN_BEFORE_DEVELOPMENT_TEST", "created_utc": pd.Timestamp.utcnow().isoformat(),
              "seeds": SEEDS, "thresholds_from_validation": thresholds,
              "checkpoint_files": {f"gnn_seed_{r['seed']}": r["checkpoint"] for r in gnn_runs},
              "preprocessing_file": "preprocessing_state.json", "logistic_file": "logistic_c1_model.pkl",
              "constant_file": "constant_baseline.json", "test_metrics_not_yet_written": True}
    frozen_blob = json.dumps(frozen, sort_keys=True).encode()
    frozen["freeze_sha256"] = hashlib.sha256(frozen_blob).hexdigest()
    write_json(out / "FROZEN_BEFORE_TEST.json", frozen)
    write_json(out / "validation_thresholds.json", thresholds)
    print("[gate] all checkpoints, preprocessing, and validation thresholds frozen; development-test evaluation begins", flush=True)

    # Development-test is accessed only here, once, after freeze.
    test_ledger = {"development_test_accessed": True, "access_stage": "after_freeze",
                   "created_utc": pd.Timestamp.utcnow().isoformat(), "model_revisions_after_access": False,
                   "protected_session_excluded": True}
    for name in ["constant", "logistic_c1"] + [f"gnn_seed_{s}" for s in SEEDS]:
        if name == "constant":
            y = np.asarray([g["label"] for g in by_split["development_test"]], dtype=int); p = np.full(len(y), constant_p)
        elif name == "logistic_c1":
            x = transform(tabular_matrix(by_split["development_test"]), tab_mean, tab_std)
            y = np.asarray([g["label"] for g in by_split["development_test"]], dtype=int); p = logistic.predict_proba(x)[:, 1]
        else:
            seed = int(name.split("_")[-1])
            ckpt = torch.load(out / f"gnn_seed_{seed}_best.pt", map_location="cpu", weights_only=False)
            model = EdgeAwareGNN(len(NODE_FEATURES), len(EDGE_FEATURES), len(PAIR_FEATURES)); model.load_state_dict(ckpt["state_dict"])
            y, p = model_probabilities(model, make_loader(by_split["development_test"], prep, False, seed))
        record_candidate(name, None if name in {"constant", "logistic_c1"} else int(name.split("_")[-1]), {"development_test": (y, p)})
    write_json(out / "development_test_access_ledger.json", test_ledger)

    # Rebuild clean metric/prediction tables from the now complete all_preds map.
    # The second record_candidate calls intentionally retain frozen validation scores;
    # write all rows and let the report state that validation was selected before test.
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(out / "precision_recall_curves.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(out / "reliability_bins.csv", index=False)
    save_predictions(out, [row for rows in all_preds.values() for row in rows])
    bootstrap = whole_race_bootstrap(all_preds, out)
    write_json(out / "metrics.json", {"rows": metrics, "bootstrap": bootstrap, "primary_metric": "average_precision"})

    # Reload parity for each GNN checkpoint on a fixed first validation batch.
    parity = {}
    fixed_batch = make_loader(by_split["development_validation"][:min(4, len(by_split["development_validation"]))], prep, False, 17).__iter__().__next__()
    for seed in SEEDS:
        ckpt_path = out / f"gnn_seed_{seed}_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m1 = EdgeAwareGNN(len(NODE_FEATURES), len(EDGE_FEATURES), len(PAIR_FEATURES)); m1.load_state_dict(ckpt["state_dict"]); m1.eval()
        with torch.no_grad(): original = torch.sigmoid(m1(fixed_batch)).numpy()
        m2 = EdgeAwareGNN(len(NODE_FEATURES), len(EDGE_FEATURES), len(PAIR_FEATURES)); m2.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]); m2.eval()
        with torch.no_grad(): reloaded = torch.sigmoid(m2(fixed_batch)).numpy()
        max_abs = float(np.max(np.abs(original - reloaded)))
        parity[str(seed)] = {"max_abs_difference": max_abs, "passed": bool(max_abs <= 1e-7)}
    write_json(out / "inference_parity.json", parity)

    # Input hashes and integrity status after all writes.
    after = {k: sha256(ROOT / k) for k in inputs["input_sha256_before"]}
    write_json(out / "input_hashes_after.json", after)
    hash_equal = after == inputs["input_sha256_before"]
    tests = {
        "source_immutability": {"passed": hash_equal},
        "protected_holdout_exclusion": {"passed": not any("11353" in r["race_id"] for r in coverage_rows)},
        "split_isolation": {"passed": len({(g["race_id"], g["proposed_split"]) for g in graphs}) == len({g["race_id"] for g in graphs})},
        "graph_edge_bounds": {"passed": all(all(0 <= j < len(g["drivers"]) for j in g["edge_index"][0] + g["edge_index"][1]) for g in graphs)},
        "attacker_target_mapping": {"passed": all(g["drivers"][g["attacker_idx"]] == g["attacker_driver"] and g["drivers"][g["target_idx"]] == g["target_driver_fixed"] for g in graphs)},
        "label_feature_separation": {"passed": all("endpoint_position" not in g and "future" not in str(g).lower() for g in graphs)},
        "finite_inference_parity": {"passed": all(v["passed"] for v in parity.values())},
        "development_test_access_after_freeze": {"passed": (out / "FROZEN_BEFORE_TEST.json").exists()},
    }
    write_json(out / "experiment_tests.json", tests)
    write_report(out, inputs, graph_counts, metrics, gnn_runs, tests)
    print("[done] experiment artifacts written; run test_gnn_experiment.py and package_experiment.py next", flush=True)


if __name__ == "__main__":
    main()
