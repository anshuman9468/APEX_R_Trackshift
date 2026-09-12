"""Bounded-memory storage and lazy loading for the APEX-R graph cache.

The legacy cache is a torch zip whose data.pkl contains one large Python list
of nested dictionaries.  This module parses that pickle stream incrementally,
converts it to memory-mapped NumPy arrays, and exposes one-graph-at-a-time
loading.  Raw numeric values are retained as float64 because the legacy
pickle stores Python doubles.  The existing dataset boundary still creates
float32 PyTorch tensors, exactly as the original runner did.
"""

from __future__ import annotations

import json
import math
import pickletools
import time
import zipfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


NODE_DIM = 18
EDGE_DIM = 4
PAIR_DIM = 12
_MARK = object()
_GRAPH_MARKER = object()


def _data_pickle_name(names: Iterable[str]) -> str:
    candidates = [n for n in names if n.endswith("/data.pkl") or n == "data.pkl"]
    if not candidates:
        raise ValueError("No data.pkl member found in legacy torch cache")
    return candidates[0]


def stream_legacy_graphs(path: Path, callback: Callable[[dict[str, Any], int], None]) -> int:
    """Parse legacy graph records one at a time without torch.load().

    The cache was written using protocol 2 and contains only primitive pickle
    opcodes, lists and dictionaries.  The top-level list is deliberately not
    materialized.  At each graph SETITEMS boundary the completed dictionary is
    passed to the callback and immediately replaced by a marker.
    """
    count = 0
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    with zipfile.ZipFile(path) as z:
        name = _data_pickle_name(z.namelist())
        with z.open(name) as f:
            for op, arg, _pos in pickletools.genops(f):
                name = op.name
                if name in {"PROTO", "STOP"}:
                    if name == "STOP":
                        break
                    continue
                if name == "EMPTY_LIST":
                    stack.append([])
                elif name == "EMPTY_DICT":
                    stack.append({})
                elif name in {"BININT1", "BININT2", "BININT", "LONG1", "LONG4"}:
                    stack.append(int(arg))
                elif name == "BINFLOAT":
                    stack.append(float(arg))
                elif name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
                    stack.append(str(arg))
                elif name == "NONE":
                    stack.append(None)
                elif name in {"BINPUT", "LONG_BINPUT"}:
                    memo[int(arg)] = stack[-1]
                elif name in {"BINGET", "LONG_BINGET"}:
                    stack.append(memo[int(arg)])
                elif name == "MARK":
                    stack.append(_MARK)
                elif name == "APPENDS":
                    mark = max(i for i, x in enumerate(stack) if x is _MARK)
                    target = stack[mark - 1]
                    items = stack[mark + 1:]
                    if isinstance(target, list):
                        # The root list is only a container for graph records.
                        # Its items have already been consumed at SETITEMS.
                        if not (target and target[0] is _GRAPH_MARKER):
                            target.extend(items)
                    del stack[mark:]
                elif name == "SETITEMS":
                    mark = max(i for i, x in enumerate(stack) if x is _MARK)
                    target = stack[mark - 1]
                    items = stack[mark + 1:]
                    if len(items) % 2:
                        raise ValueError("Malformed SETITEMS key/value sequence")
                    for i in range(0, len(items), 2):
                        target[items[i]] = items[i + 1]
                    del stack[mark:]
                    if not isinstance(target, dict) or "window_id" not in target or "node_raw" not in target:
                        raise ValueError("Unexpected dictionary in graph pickle")
                    callback(target, count)
                    count += 1
                    # The top-level list keeps only a bounded marker instead of
                    # retaining the completed graph dictionary.
                    stack[-1] = _GRAPH_MARKER
                else:
                    raise ValueError(f"Unsupported legacy pickle opcode: {name}")
    return count


def _array_from_record(record: dict[str, Any], key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    arr = np.asarray(record[key], dtype=dtype)
    if arr.shape != shape:
        raise ValueError(f"Unexpected {key} shape {arr.shape}; expected {shape}")
    return arr


def _metadata(record: dict[str, Any], node_start: int, node_end: int,
              edge_start: int, edge_end: int) -> dict[str, Any]:
    return {
        "window_id": record["window_id"],
        "race_id": record["race_id"],
        "proposed_split": record["proposed_split"],
        "decision_lap": int(record["decision_lap"]),
        "decision_session_time_sec": float(record["decision_session_time_sec"]),
        "endpoint_lap": int(record["endpoint_lap"]),
        "drivers": list(record["drivers"]),
        "attacker_driver": record["attacker_driver"],
        "target_driver_fixed": record["target_driver_fixed"],
        "attacker_idx": int(record["attacker_idx"]),
        "target_idx": int(record["target_idx"]),
        "asof_tolerance_sec": float(record["asof_tolerance_sec"]),
        "lookback_sec": float(record["lookback_sec"]),
        "label": int(record["label"]),
        "outcome_status": record["outcome_status"],
        "node_start": node_start,
        "node_end": node_end,
        "edge_start": edge_start,
        "edge_end": edge_end,
    }


def build_disk_store(legacy_path: Path, store_dir: Path, force: bool = False) -> dict[str, Any]:
    """Convert the legacy cache using two bounded streaming passes."""
    store_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = store_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    metadata: list[dict[str, Any]] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    totals = {"nodes": 0, "edges": 0, "graphs": 0}

    def count_graph(record: dict[str, Any], index: int) -> None:
        nodes = _array_from_record(record, "node_raw", (len(record["drivers"]), NODE_DIM), np.float64)
        edge_raw = np.asarray(record["edge_raw"], dtype=np.float64).reshape(-1, EDGE_DIM)
        edge_index = np.asarray(record["edge_index"], dtype=np.int64)
        if edge_index.shape != (2, edge_raw.shape[0]):
            raise ValueError(f"Graph {index} edge index shape mismatch")
        node_start = totals["nodes"]
        edge_start = totals["edges"]
        totals["nodes"] += int(nodes.shape[0]); totals["edges"] += int(edge_raw.shape[0]); totals["graphs"] += 1
        node_counts.append(int(nodes.shape[0])); edge_counts.append(int(edge_raw.shape[0]))
        metadata.append(_metadata(record, node_start, totals["nodes"], edge_start, totals["edges"]))

    started = time.time()
    parsed = stream_legacy_graphs(legacy_path, count_graph)
    if parsed != totals["graphs"] or parsed != len(metadata):
        raise ValueError("Legacy graph count mismatch during first conversion pass")

    node_values = np.lib.format.open_memmap(store_dir / "node_values.npy", mode="w+", dtype=np.float64,
                                            shape=(totals["nodes"], NODE_DIM))
    edge_values = np.lib.format.open_memmap(store_dir / "edge_values.npy", mode="w+", dtype=np.float64,
                                            shape=(totals["edges"], EDGE_DIM))
    edge_index_values = np.lib.format.open_memmap(store_dir / "edge_index.npy", mode="w+", dtype=np.int64,
                                                  shape=(totals["edges"], 2))
    pair_values = np.lib.format.open_memmap(store_dir / "pair_values.npy", mode="w+", dtype=np.float64,
                                            shape=(totals["graphs"], PAIR_DIM))
    labels = np.lib.format.open_memmap(store_dir / "labels.npy", mode="w+", dtype=np.int8,
                                       shape=(totals["graphs"],))

    def write_graph(record: dict[str, Any], index: int) -> None:
        if index >= len(metadata):
            raise ValueError("Legacy graph ordering changed between conversion passes")
        meta = metadata[index]
        nodes = _array_from_record(record, "node_raw", (len(meta["drivers"]), NODE_DIM), np.float64)
        edge_raw = np.asarray(record["edge_raw"], dtype=np.float64).reshape(-1, EDGE_DIM)
        edge_index = np.asarray(record["edge_index"], dtype=np.int64)
        pair = _array_from_record(record, "pair_raw", (PAIR_DIM,), np.float64)
        edge_pairs = np.column_stack([edge_index[0], edge_index[1]]) if edge_index.shape[1] else np.empty((0, 2), dtype=np.int64)
        if edge_pairs.shape[0] != edge_raw.shape[0]:
            raise ValueError(f"Graph {index} edge count changed between conversion passes")
        node_values[meta["node_start"]:meta["node_end"]] = nodes
        edge_values[meta["edge_start"]:meta["edge_end"]] = edge_raw
        edge_index_values[meta["edge_start"]:meta["edge_end"]] = edge_pairs
        pair_values[index] = pair
        labels[index] = int(record["label"])
        if record["window_id"] != meta["window_id"]:
            raise ValueError(f"Graph ordering/window ID changed at index {index}")

    parsed_second = stream_legacy_graphs(legacy_path, write_graph)
    if parsed_second != parsed:
        raise ValueError("Legacy graph count mismatch during second conversion pass")
    for arr in [node_values, edge_values, edge_index_values, pair_values, labels]:
        arr.flush()
    del node_values, edge_values, edge_index_values, pair_values, labels
    np.save(store_dir / "node_offsets.npy", np.asarray([0] + list(np.cumsum(node_counts, dtype=np.int64))))
    np.save(store_dir / "edge_offsets.npy", np.asarray([0] + list(np.cumsum(edge_counts, dtype=np.int64))))
    (store_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    manifest = {
        "version": "disk-graph-store-v1",
        "legacy_source": str(legacy_path),
        "graphs": totals["graphs"], "total_nodes": totals["nodes"], "total_edges": totals["edges"],
        "node_shape": [totals["nodes"], NODE_DIM], "edge_shape": [totals["edges"], EDGE_DIM],
        "edge_index_shape": [totals["edges"], 2], "pair_shape": [totals["graphs"], PAIR_DIM],
        "raw_numeric_dtype": "float64", "index_dtype": "int64", "label_dtype": "int8",
        "node_feature_count": NODE_DIM, "edge_feature_count": EDGE_DIM, "pair_feature_count": PAIR_DIM,
        "conversion_passes": 2, "streaming_parser": True,
        "conversion_seconds": time.time() - started,
        "files": ["node_values.npy", "edge_values.npy", "edge_index.npy", "pair_values.npy", "labels.npy",
                   "node_offsets.npy", "edge_offsets.npy", "metadata.json"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class DiskGraphStore:
    """Small metadata + memory-mapped numeric arrays; no eager graph objects."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.manifest = json.loads((self.store_dir / "manifest.json").read_text())
        self.metadata: list[dict[str, Any]] = json.loads((self.store_dir / "metadata.json").read_text())
        self.node_values = np.load(self.store_dir / "node_values.npy", mmap_mode="r")
        self.edge_values = np.load(self.store_dir / "edge_values.npy", mmap_mode="r")
        self.edge_index = np.load(self.store_dir / "edge_index.npy", mmap_mode="r")
        self.pair_values = np.load(self.store_dir / "pair_values.npy", mmap_mode="r")
        self.labels = np.load(self.store_dir / "labels.npy", mmap_mode="r")
        self.node_offsets = np.load(self.store_dir / "node_offsets.npy", mmap_mode="r")
        self.edge_offsets = np.load(self.store_dir / "edge_offsets.npy", mmap_mode="r")
        if len(self.metadata) != int(self.manifest["graphs"]):
            raise ValueError("Metadata/manifest graph count mismatch")

    def __len__(self) -> int:
        return len(self.metadata)

    def get(self, index: int) -> dict[str, Any]:
        meta = self.metadata[index]
        ns, ne = int(meta["node_start"]), int(meta["node_end"])
        es, ee = int(meta["edge_start"]), int(meta["edge_end"])
        edge_index = self.edge_index[es:ee]
        return {
            "metadata": meta,
            "node_raw": self.node_values[ns:ne],
            "edge_raw": self.edge_values[es:ee],
            "edge_index": edge_index,
            "pair_raw": self.pair_values[index],
            "label": int(self.labels[index]),
        }

    def split_indices(self, split: str) -> list[int]:
        return [i for i, m in enumerate(self.metadata) if m["proposed_split"] == split]

    def validate_layout(self) -> dict[str, Any]:
        checks = {
            "graph_count": len(self.metadata) == int(self.manifest["graphs"]),
            "node_offset_count": len(self.node_offsets) == len(self.metadata) + 1,
            "edge_offset_count": len(self.edge_offsets) == len(self.metadata) + 1,
            "node_total": int(self.node_offsets[-1]) == int(self.manifest["total_nodes"]),
            "edge_total": int(self.edge_offsets[-1]) == int(self.manifest["total_edges"]),
            "finite_pair_indices": bool(np.isfinite(self.pair_values).sum() >= 0),
        }
        for i, m in enumerate(self.metadata):
            e = self.edge_index[int(m["edge_start"]):int(m["edge_end"])]
            n = int(m["node_end"]) - int(m["node_start"])
            if len(e) and (int(e.min()) < 0 or int(e.max()) >= n):
                checks["edge_bounds"] = False
                break
        else:
            checks["edge_bounds"] = True
        checks["passed"] = all(checks.values())
        return checks


def _streaming_stats(arrays: Iterable[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    count = np.zeros(width, dtype=np.int64)
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64).reshape(-1, width)
        finite = np.isfinite(a)
        count += finite.sum(axis=0, dtype=np.int64)
        clean = np.where(finite, a, 0.0)
        total += clean.sum(axis=0, dtype=np.float64)
        total_sq += np.square(clean, dtype=np.float64).sum(axis=0, dtype=np.float64)
    mean = np.divide(total, count, out=np.zeros(width), where=count > 0)
    variance = np.divide(total_sq, count, out=np.zeros(width), where=count > 0) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return mean, std


def compute_train_preprocessing(store: DiskGraphStore, train_indices: Sequence[int]) -> dict[str, Any]:
    """Compute the original nanmean/nanstd formulas with bounded arrays."""
    node_mean, node_std = _streaming_stats((store.get(i)["node_raw"] for i in train_indices), NODE_DIM)
    edge_mean, edge_std = _streaming_stats((store.get(i)["edge_raw"] for i in train_indices), EDGE_DIM)
    pair_mean, pair_std = _streaming_stats((store.get(i)["pair_raw"] for i in train_indices), PAIR_DIM)

    def tab_rows() -> Iterable[np.ndarray]:
        for i in train_indices:
            g = store.get(i); m = g["metadata"]
            ai, ti = int(m["attacker_idx"]), int(m["target_idx"])
            yield np.concatenate([np.asarray(g["node_raw"][ai], dtype=np.float64),
                                  np.asarray(g["node_raw"][ti], dtype=np.float64),
                                  np.asarray(g["pair_raw"], dtype=np.float64)]).reshape(1, -1)

    tab_mean, tab_std = _streaming_stats(tab_rows(), NODE_DIM * 2 + PAIR_DIM)
    return {
        "version": "train-only-mean-standardize-v1",
        "fit_scope": "development_train eligible graphs only",
        "node_mean": node_mean.tolist(), "node_std": node_std.tolist(),
        "edge_mean": edge_mean.tolist(), "edge_std": edge_std.tolist(),
        "pair_mean": pair_mean.tolist(), "pair_std": pair_std.tolist(),
        "tab_mean": tab_mean.tolist(), "tab_std": tab_std.tolist(),
        "missing_policy": "fit train means only; masks retained; no source mutation",
        "raw_storage_dtype": "float64",
        "torch_tensor_dtype_preserved": "float32 at Data.x, Data.edge_attr, Data.pair_x as in legacy runner",
    }


def transform(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return (np.where(np.isfinite(raw), raw, mean) - mean) / std


class LazyGraphDataset(torch.utils.data.Dataset):
    def __init__(self, store: DiskGraphStore, indices: Sequence[int], prep: dict[str, Any]):
        self.store, self.indices, self.prep = store, list(indices), prep
        self.nmean, self.nstd = np.asarray(prep["node_mean"]), np.asarray(prep["node_std"])
        self.emean, self.estd = np.asarray(prep["edge_mean"]), np.asarray(prep["edge_std"])
        self.pmean, self.pstd = np.asarray(prep["pair_mean"]), np.asarray(prep["pair_std"])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> Data:
        g = self.store.get(self.indices[local_index]); m = g["metadata"]
        x = transform(g["node_raw"], self.nmean, self.nstd)
        e = transform(g["edge_raw"], self.emean, self.estd).reshape(-1, EDGE_DIM)
        p = transform(g["pair_raw"], self.pmean, self.pstd)
        edge_index = torch.tensor(np.asarray(g["edge_index"], dtype=np.int64).T, dtype=torch.long)
        if edge_index.numel() == 0:
            edge_index = edge_index.reshape(2, 0)
        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=torch.tensor(e, dtype=torch.float32),
            pair_x=torch.tensor(p, dtype=torch.float32).reshape(1, -1),
            y=torch.tensor([float(g["label"])], dtype=torch.float32),
            att_idx=torch.tensor([int(m["attacker_idx"])], dtype=torch.long),
            target_idx=torch.tensor([int(m["target_idx"])], dtype=torch.long),
        )


def make_lazy_loader(store: DiskGraphStore, indices: Sequence[int], prep: dict[str, Any],
                     batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(LazyGraphDataset(store, indices, prep), batch_size=batch_size,
                      shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)


def bounded_equivalence_check(legacy_path: Path, store: DiskGraphStore) -> dict[str, Any]:
    """Compare every legacy record to its disk-backed slices one graph at a time."""
    report = {"graphs_compared": 0, "metadata_mismatches": 0, "node_mismatches": 0,
              "edge_mismatches": 0, "edge_index_mismatches": 0, "pair_mismatches": 0,
              "label_mismatches": 0, "max_abs_difference": 0.0, "passed": False}

    def compare(record: dict[str, Any], index: int) -> None:
        g = store.get(index); m = g["metadata"]
        scalar_fields = ["window_id", "race_id", "proposed_split", "decision_lap",
                         "decision_session_time_sec", "endpoint_lap", "drivers",
                         "attacker_driver", "target_driver_fixed", "attacker_idx", "target_idx",
                         "asof_tolerance_sec", "lookback_sec", "outcome_status"]
        for field in scalar_fields:
            if field == "decision_session_time_sec":
                equal = abs(float(record[field]) - float(m[field])) <= 0.0
            else:
                equal = record[field] == m[field]
            if not equal:
                report["metadata_mismatches"] += 1
        for key, field in [("node_raw", "node_mismatches"), ("edge_raw", "edge_mismatches"),
                           ("pair_raw", "pair_mismatches")]:
            old = np.asarray(record[key], dtype=np.float64)
            if key == "edge_raw": old = old.reshape(-1, EDGE_DIM)
            new = np.asarray(g[key], dtype=np.float64)
            if old.shape != new.shape or not np.array_equal(old, new, equal_nan=True):
                report[field] += 1
            both = np.isfinite(old) & np.isfinite(new) if old.shape == new.shape else np.zeros(1, dtype=bool)
            if both.any():
                report["max_abs_difference"] = max(report["max_abs_difference"], float(np.max(np.abs(old[both] - new[both]))))
        old_edge = np.column_stack(record["edge_index"]) if len(record["edge_index"][0]) else np.empty((0, 2), dtype=np.int64)
        if not np.array_equal(old_edge, np.asarray(g["edge_index"]), equal_nan=True):
            report["edge_index_mismatches"] += 1
        if int(record["label"]) != int(g["label"]):
            report["label_mismatches"] += 1
        report["graphs_compared"] += 1

    parsed = stream_legacy_graphs(legacy_path, compare)
    report["legacy_graphs_parsed"] = parsed
    report["passed"] = (parsed == len(store) == report["graphs_compared"] and
                         all(report[k] == 0 for k in ["metadata_mismatches", "node_mismatches", "edge_mismatches",
                                                      "edge_index_mismatches", "pair_mismatches", "label_mismatches"]) and
                         report["max_abs_difference"] == 0.0)
    return report
