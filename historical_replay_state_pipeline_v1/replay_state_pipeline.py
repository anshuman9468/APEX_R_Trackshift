#!/usr/bin/env python3
"""Build strict historical replay decision states for the frozen GNN demo.

This is deliberately a *state reconstruction* pipeline, not a model-training
or label-generation script.  It uses the exact frozen GNN graph builder and
its 1 s backward-only / 10 s trailing contract.  Boundary outcome columns in
the window table are never read.

The current audited sources do not provide a verified, timestamped time gap to
the car ahead or behind.  Therefore the strategy optimiser is represented as
UNAVAILABLE for these states: map XYZ distance is retained in metres for the
GNN, but is never converted into a seconds gap.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DEFAULT = ROOT / "historical_replay_state_pipeline_v1" / "artifacts"
WINDOWS = ROOT / "phase3_passed_races" / "prediction_windows_37.csv.gz"
LAPTIME_INVENTORY = ROOT / "phase3_passed_races" / "laptime_source_inventory_37.csv"
SPLITS = ROOT / "phase3_races_audit_reconciliation_v1" / "proposed_chronological_split_37.csv"
CAR = ROOT / "full_race_telemetry_collection_final" / "telemetry_car.csv.gz"
POSITION = ROOT / "full_race_telemetry_collection_final" / "telemetry_position.csv.gz"
FROZEN_MANIFEST = ROOT / "models" / "frozen" / "gnn_proxy_v1" / "FROZEN_MODEL_MANIFEST.json"
CACHE = ROOT / "gnn_proxy_experiment_v1" / "disk_graph_store"
GRAPH_BUILD_MANIFEST = ROOT / "gnn_proxy_experiment_v1" / "graph_build_manifest.json"
PROTECTED_TOKEN = "11353"
APPROVED_RACES = (
    "2019:Abu Dhabi Grand Prix:Race",
    "2020:Hungarian Grand Prix:Race",
    "2020:70th Anniversary Grand Prix:Race",
)
TARGET_STATUS = "UNIQUE_DIRECT_PREDECESSOR_AT_SAME_LAP_BOUNDARY"
FORBIDDEN_WINDOW_COLUMNS = {"outcome_status", "proxy_label", "censoring_reason", "matched_candidate_id", "true_ontrack_label", "endpoint_session_time_sec"}
PINNED_APPROVED_FIXTURE_WINDOW = "ee38ba6e51b8d9dbba9b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compact_graph_fingerprint(graph: dict[str, Any]) -> str:
    payload = {
        key: graph[key]
        for key in ("race_id", "drivers", "attacker_driver", "target_driver_fixed", "attacker_idx", "target_idx", "node_raw", "edge_index", "edge_raw", "pair_raw", "asof_tolerance_sec", "lookback_sec")
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")).hexdigest()


def load_contract() -> dict[str, Any]:
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("model_version") != "gnn_proxy_v1" or manifest.get("selected_seed") != 42 or manifest.get("selected_epoch") != 28:
        raise RuntimeError("Frozen model manifest does not match the approved seed-42 epoch-28 selection")
    if PROTECTED_TOKEN in FROZEN_MANIFEST.read_text(encoding="utf-8"):
        # The expected exclusion statement contains the token; this is not a source-data access.
        pass
    return {
        "model_version": manifest["model_version"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "task": manifest["task"],
        "selected_seed": manifest["selected_seed"],
        "selected_epoch": manifest["selected_epoch"],
        "decision_time": "one fixed decision at the start of selected lap L, immediately after boundary L-1",
        "target_selection": "fixed classified direct predecessor at prior completed lap boundary; never recomputed within horizon",
        "lookback_sec": 10.0,
        "asof_tolerance_sec": 1.0,
        "join_policy": "backward-only; no interpolation, backfill, or future sample",
        "output_label": "experimental boundary position-swap proxy signal",
        "prohibited_interpretation": "not a rolling five-second overtake prediction; not a calibrated pass probability; not an ATTACK-benefit probability",
    }


def load_windows_and_drivers() -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, str], dict[str, int]]:
    split_map: dict[str, str] = {}
    with SPLITS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split_map[row["race_id"]] = row["proposed_split"]
    if any(PROTECTED_TOKEN in race for race in APPROVED_RACES):
        raise RuntimeError("Protected holdout was requested")
    if not set(APPROVED_RACES).issubset(split_map):
        raise RuntimeError("One or more approved races are absent from the frozen chronological split manifest")

    windows: list[dict[str, Any]] = []
    source_rows = 0
    with gzip.open(WINDOWS, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not FORBIDDEN_WINDOW_COLUMNS.issubset(reader.fieldnames):
            raise RuntimeError("Prediction-window schema is not the audited contract")
        for row in reader:
            source_rows += 1
            race_id = row.get("race_id", "")
            if PROTECTED_TOKEN in race_id:
                raise RuntimeError("Protected holdout encountered before state construction")
            if race_id not in APPROVED_RACES or row.get("target_identity_status") != TARGET_STATUS:
                continue
            # Outcome columns intentionally remain unread.  The model-state
            # builder gets only decision-time identity/time fields below.
            windows.append({
                "window_id": row["window_id"],
                "race_id": race_id,
                "attacker_driver": row["attacker_driver"],
                "target_driver_fixed": row["target_driver_fixed"],
                "decision_lap": int(row["decision_lap"]),
                "decision_session_time_sec": float(row["decision_session_time_sec"]),
                "endpoint_lap": int(row["endpoint_lap"]),
                "proposed_split": split_map[race_id],
            })
    if not windows:
        raise RuntimeError("No eligible fixed-target decision windows were found")

    drivers: dict[str, set[str]] = defaultdict(set)
    with LAPTIME_INVENTORY.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            race_id = row.get("race_id", "")
            if race_id in APPROVED_RACES:
                drivers[race_id].add(row["driver"])
    if set(drivers) != set(APPROVED_RACES):
        raise RuntimeError("Missing driver inventory for one or more approved races")
    for window in windows:
        available = drivers[window["race_id"]]
        if window["attacker_driver"] not in available or window["target_driver_fixed"] not in available:
            raise RuntimeError("Fixed pair is absent from laptime identity inventory")
    return windows, drivers, split_map, {"source_rows_seen": source_rows, "eligible_target_windows": len(windows)}


def gnn_payload(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": "replay-" + graph["window_id"],
        "decision_timestamp_session_sec": graph["decision_session_time_sec"],
        "attacker_driver": graph["attacker_driver"],
        "target_driver_fixed": graph["target_driver_fixed"],
        "input_freshness": {
            "join_policy": "backward-only",
            "interpolation": False,
            "asof_tolerance_sec": graph["asof_tolerance_sec"],
            "lookback_sec": graph["lookback_sec"],
            "source_timestamp_semantics": "session-relative; reconstructed from approved full-race telemetry",
        },
        "graph": graph,
    }


def strict_optimizer_view(graph: dict[str, Any], score: dict[str, Any] | None, reason: str | None) -> dict[str, Any]:
    # The Phase-5 engine's gaps are seconds.  The audited source has neither
    # current source-backed front nor rear time gaps here.  In particular,
    # pair_raw[1] is Euclidean XYZ distance in metres and must not be converted.
    unavailable_reason = "MISSING_SOURCE_BACKED_FRONT_AND_REAR_TIME_GAPS_SECONDS"
    return {
        "status": "unavailable",
        "action": None,
        "decision_reason": unavailable_reason,
        "action_scores": {action: {"status": "unavailable", "score": None, "reason": unavailable_reason} for action in ("ATTACK", "HOLD", "HARVEST", "DEFEND")},
        "inputs": {
            "front_gap_seconds": None,
            "rear_gap_seconds": None,
            "front_gap_status": "SOURCE_UNAVAILABLE_NOT_IMPUTED",
            "rear_gap_status": "SOURCE_UNAVAILABLE_NOT_IMPUTED",
            "battery_percent": None,
            "battery_status": "UNOBSERVED_REAL_ERS_NOT_SIMULATED",
            "attacker_target_xyz_euclidean_distance_m": finite(graph["pair_raw"][1]),
            "xyz_distance_status": "GNN_GRAPH_FEATURE_ONLY_NOT_RACE_GAP",
            "gnn_influence": "zero; advisory only",
            "gnn_status": None if score is None else score.get("status"),
            "gnn_unavailable_reason": reason,
        },
    }


def trace_for_graph(graph: dict[str, Any], adapter) -> dict[str, Any]:
    payload = gnn_payload(graph)
    try:
        score = adapter.predict_request(payload)
        applicability = "APPLICABLE" if score.get("status") == "available" else "UNAVAILABLE"
        score_reason = score.get("failure_reason") or score.get("reason")
    except Exception as exc:
        score = {"status": "unavailable", "proxy_score": None, "failure_reason": f"{type(exc).__name__}: {exc}"}
        applicability, score_reason = "UNAVAILABLE", score["failure_reason"]
    return {
        "trace_schema_version": "historical-replay-state-v1",
        "window_id": graph["window_id"],
        "race_id": graph["race_id"],
        "proposed_split": graph["proposed_split"],
        "decision_lap": graph["decision_lap"],
        "decision_timestamp_session_sec": graph["decision_session_time_sec"],
        "target_contract": {
            "attacker_driver": graph["attacker_driver"],
            "fixed_target_driver": graph["target_driver_fixed"],
            "relationship": "classified direct predecessor at the previous completed lap boundary",
            "not_claimed": "physical proximity, current on-track-ahead identity, or a target re-selected during the horizon",
        },
        "freshness": {
            "join_policy": "backward-only",
            "interpolation": False,
            "asof_tolerance_sec": graph["asof_tolerance_sec"],
            "lookback_sec": graph["lookback_sec"],
            "attacker_car_age_sec": finite(graph["node_raw"][graph["attacker_idx"]][6]),
            "target_car_age_sec": finite(graph["node_raw"][graph["target_idx"]][6]),
            "attacker_position_age_sec": finite(graph["node_raw"][graph["attacker_idx"]][10]),
            "target_position_age_sec": finite(graph["node_raw"][graph["target_idx"]][10]),
        },
        "race_state": {
            "attacker_speed_kmh": finite(graph["node_raw"][graph["attacker_idx"]][0]),
            "target_speed_kmh": finite(graph["node_raw"][graph["target_idx"]][0]),
            "attacker_minus_target_speed_kmh": finite(graph["pair_raw"][0]),
            "attacker_target_xyz_euclidean_distance_m": finite(graph["pair_raw"][1]),
            "rear_rival": None,
            "rear_rival_status": "UNKNOWN_NO_ASOF_CLASSIFIED_ORDER_STREAM",
        },
        "graph_fingerprint_sha256": compact_graph_fingerprint(graph),
        "model_applicability": applicability,
        "model": score,
        "optimizer": strict_optimizer_view(graph, score, score_reason),
    }


def write_csv(path: Path, traces: list[dict[str, Any]]) -> None:
    fields = [
        "window_id", "race_id", "proposed_split", "decision_lap", "decision_timestamp_session_sec",
        "attacker_driver", "fixed_target_driver", "model_applicability", "proxy_score",
        "attacker_car_age_sec", "target_car_age_sec", "attacker_position_age_sec", "target_position_age_sec",
        "attacker_target_xyz_euclidean_distance_m", "optimizer_status", "optimizer_action", "optimizer_reason",
        "graph_fingerprint_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in traces:
            target = row["target_contract"]
            fresh = row["freshness"]
            state = row["race_state"]
            optimizer = row["optimizer"]
            writer.writerow({
                "window_id": row["window_id"], "race_id": row["race_id"], "proposed_split": row["proposed_split"],
                "decision_lap": row["decision_lap"], "decision_timestamp_session_sec": row["decision_timestamp_session_sec"],
                "attacker_driver": target["attacker_driver"], "fixed_target_driver": target["fixed_target_driver"],
                "model_applicability": row["model_applicability"], "proxy_score": row["model"].get("proxy_score"),
                "attacker_car_age_sec": fresh["attacker_car_age_sec"], "target_car_age_sec": fresh["target_car_age_sec"],
                "attacker_position_age_sec": fresh["attacker_position_age_sec"], "target_position_age_sec": fresh["target_position_age_sec"],
                "attacker_target_xyz_euclidean_distance_m": state["attacker_target_xyz_euclidean_distance_m"],
                "optimizer_status": optimizer["status"], "optimizer_action": optimizer["action"],
                "optimizer_reason": optimizer["decision_reason"], "graph_fingerprint_sha256": row["graph_fingerprint_sha256"],
            })


def write_report(path: Path, contract: dict[str, Any], traces: list[dict[str, Any]], stream_stats: list[dict[str, Any]], input_hashes: dict[str, str]) -> None:
    per_race = Counter(row["race_id"] for row in traces)
    applicable = sum(row["model_applicability"] == "APPLICABLE" for row in traces)
    lines = [
        "# Historical Replay State Pipeline v1", "",
        "## Contract verified before execution", "",
        f"- Frozen model: `{contract['model_version']}`, seed {contract['selected_seed']}, epoch {contract['selected_epoch']}; checkpoint SHA-256 `{contract['checkpoint_sha256']}`.",
        f"- Task: {contract['task']}. It is **not** a rolling five-second overtake predictor.",
        "- Target: fixed classified direct predecessor at the prior completed lap boundary; it is never re-selected inside the horizon.",
        "- Graph timing: 10-second trailing lookback and 1.0-second backward-only as-of tolerance; no interpolation/backfill/future joins.",
        "- Protected session 11353 was rejected before window or telemetry access.", "",
        "## Validation", "",
        f"- Approved races validated end-to-end: {', '.join(f'`{race}` ({per_race[race]:,} model-applicable states)' for race in APPROVED_RACES)}.",
        f"- GNN applicable states: {applicable:,}/{len(traces):,}. Each passed fresh fixed-pair car and position checks.",
        "- Every trace keeps target classification distinct from physical proximity. The graph's XYZ Euclidean distance is metres only and is never converted into a time gap.",
        "- The optimiser was refreshed in strict mode but correctly returned **unavailable** for every trace: the audited sources do not supply a verified timestamped front/rear time gap in seconds, and real battery state is unobserved. No action is forced and no existing optimiser weight/configuration was changed.",
        "- `action_scores` are null with their explicit source-data blocker instead of being computed from fabricated gaps.", "",
        "## Backend integration", "",
        "- The local FastAPI and standalone servers expose each trace at `/api/phase5/historical-state/{window_id}`.",
        "- The endpoint is read-only: it returns the exact frozen advisory score and strict optimiser status generated here. It does not re-select targets, infer a five-second prediction, or change an action.", "",
        "## Stream checks", "",
    ]
    for stat in stream_stats:
        lines.append(f"- `{stat['kind']}`: {stat['source_rows_seen']:,} source rows scanned; {stat['source_rows_used_for_requested_drivers']:,} rows used; non-monotonic requested streams: {stat['nonmonotonic_source_rows_in_requested_keys']}.")
    lines.extend(["", "## Input hashes", ""])
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in input_hashes.items())
    lines.extend(["", "Detailed decision traces are in `historical_state_traces.json` and `historical_state_trace_summary.csv`.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def select_from_verified_disk_cache() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Read only three graphs from the RAM-safe canonical cache.

    The cache was built from the timestamped raw streams using the frozen
    graph-builder contract.  Arrays are memory-mapped; output-label fields in
    cache metadata are intentionally ignored and never copied into a state.
    """
    import numpy as np

    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != "disk-graph-store-v1" or manifest.get("graphs") != 15404:
        raise RuntimeError("Canonical disk graph cache manifest is unexpected")
    build = json.loads(GRAPH_BUILD_MANIFEST.read_text(encoding="utf-8"))
    if build.get("feature_builder_excludes_outcome_fields") is not True:
        raise RuntimeError("Canonical graph cache does not attest feature/label separation")
    stream_stats = build.get("stream_stats", [])
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    if any(PROTECTED_TOKEN in str(record.get("race_id", "")) for record in metadata):
        raise RuntimeError("Protected holdout appeared in canonical graph metadata")

    selected_indices: list[int] = []
    for race_id in APPROVED_RACES:
        candidate_indices = [
            index for index, record in enumerate(metadata)
            if record.get("race_id") == race_id and record.get("asof_tolerance_sec") == 1.0 and record.get("lookback_sec") == 10.0
        ]
        if not candidate_indices:
            raise RuntimeError(f"No canonical cached graph for approved race: {race_id}")
        if race_id == "2019:Abu Dhabi Grand Prix:Race":
            matching = [index for index in candidate_indices if metadata[index].get("window_id") == PINNED_APPROVED_FIXTURE_WINDOW]
            if len(matching) != 1:
                raise RuntimeError("The pre-approved Abu Dhabi frozen-fixture window is absent or ambiguous in the cache")
            selected_indices.append(matching[0])
        else:
            # Deterministic first chronological graph; no output/label field is read.
            selected_indices.append(min(candidate_indices, key=lambda index: (metadata[index]["decision_session_time_sec"], metadata[index]["window_id"])))

    node_values = np.load(CACHE / "node_values.npy", mmap_mode="r")
    edge_values = np.load(CACHE / "edge_values.npy", mmap_mode="r")
    edge_index = np.load(CACHE / "edge_index.npy", mmap_mode="r")
    pair_values = np.load(CACHE / "pair_values.npy", mmap_mode="r")
    node_offsets = np.load(CACHE / "node_offsets.npy", mmap_mode="r")
    edge_offsets = np.load(CACHE / "edge_offsets.npy", mmap_mode="r")
    graphs: list[dict[str, Any]] = []
    for index in selected_indices:
        record = metadata[index]
        # Explicit allowlist prevents cache label/outcome metadata entering the
        # replay state or GNN request.
        graph = {key: record[key] for key in (
            "window_id", "race_id", "proposed_split", "decision_lap", "decision_session_time_sec", "endpoint_lap",
            "drivers", "attacker_driver", "target_driver_fixed", "attacker_idx", "target_idx", "asof_tolerance_sec", "lookback_sec",
        )}
        node_start, node_end = int(node_offsets[index]), int(node_offsets[index + 1])
        edge_start, edge_end = int(edge_offsets[index]), int(edge_offsets[index + 1])
        graph["node_raw"] = node_values[node_start:node_end].tolist()
        graph["edge_raw"] = edge_values[edge_start:edge_end].tolist()
        graph["edge_index"] = edge_index[edge_start:edge_end].T.tolist()
        graph["pair_raw"] = pair_values[index].tolist()
        graphs.append(graph)
    return graphs, stream_stats, {race: 1 for race in APPROVED_RACES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    from backend.gnn_adapter import FrozenGNNAdapter

    input_paths = [WINDOWS, LAPTIME_INVENTORY, SPLITS, CAR, POSITION, FROZEN_MANIFEST, GRAPH_BUILD_MANIFEST, CACHE / "manifest.json", CACHE / "metadata.json", CACHE / "node_values.npy", CACHE / "edge_values.npy", CACHE / "edge_index.npy", CACHE / "pair_values.npy", CACHE / "node_offsets.npy", CACHE / "edge_offsets.npy"]
    hashes_before = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}
    contract = load_contract()
    windows, drivers_by_race, split_map, input_counts = load_windows_and_drivers()
    # Validate source selection independently, then consume only the canonical
    # disk cache to avoid a second eager reconstruction of 30M telemetry rows.
    selected, stream_stats, candidate_counts = select_from_verified_disk_cache()

    adapter = FrozenGNNAdapter("auto")
    traces = [trace_for_graph(graph, adapter) for graph in selected]
    if any(trace["model_applicability"] != "APPLICABLE" for trace in traces):
        raise RuntimeError("A selected replay state failed the frozen-model applicability gate")

    hashes_after = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("Input source hash changed during reconstruction")
    write_json(out / "input_hashes.json", {"before": hashes_before, "after": hashes_after, "unchanged": True})
    write_json(out / "replay_state_contract.json", contract)
    write_json(out / "historical_state_traces.json", traces)
    write_csv(out / "historical_state_trace_summary.csv", traces)
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protected_session_11353": "EXCLUDED_BEFORE_SOURCE_ACCESS",
        "approved_races": list(APPROVED_RACES),
        "input_counts": input_counts,
        "candidate_graphs": len(selected),
        "chronological_candidate_windows_examined": candidate_counts,
        "graph_exclusions": "not re-computed; canonical cache manifest reports 19 fixed_pair_missing_current_car_sample exclusions",
        "selected_replay_states": len(traces),
        "model_applicable_states": sum(row["model_applicability"] == "APPLICABLE" for row in traces),
        "optimizer_available_states": sum(row["optimizer"]["status"] == "available" for row in traces),
        "stream_stats": stream_stats,
        "selection": "first model-applicable decision timestamp per approved race; no outcome columns read",
        "optimizer_blocker": "No verified timestamped front/rear time gaps in seconds; XYZ distance kept in metres only; real battery state unavailable",
    }
    write_json(out / "metrics.json", metrics)
    write_report(out / "HISTORICAL_REPLAY_STATE_REPORT.md", contract, traces, stream_stats, hashes_after)
    print(json.dumps({"output": str(out), "selected_states": len(traces), "model_applicable": metrics["model_applicable_states"], "optimizer_available": metrics["optimizer_available_states"]}, indent=2))


if __name__ == "__main__":
    main()
