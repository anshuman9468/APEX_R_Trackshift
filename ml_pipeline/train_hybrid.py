#!/usr/bin/env python3
"""Train/evaluate the multi-task hybrid GNN + GRU + physics model.

The loader consumes gnn_sequences/*/*/graph_sequences.npz.  It keeps the graph
encoder framework-independent, so PyTorch Geometric is not required: message
passing is implemented with index_add over the stored edge list.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURE_DIM = 35
HISTORY_LAPS = 5
SPEED_IDX = 0
DISTANCE_IDX = 4
CONTINUOUS_FEATURES = list(range(31))
CAT_FEATURES = {31: 1.0, 32: 5.0, 33: 5.0, 34: 3.0}
# These are current-lap proxy values derived from the same wear model used to
# construct the tyre target. They remain available to the general race model,
# but are masked from the tyre heads to avoid a shortcut classification task.
TYRE_PROXY_FEATURES = (23, 24)  # wear fraction, degradation proxy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _finite_mean(values: list[np.ndarray], index: int) -> float:
    joined = np.concatenate([v[:, index] for v in values])
    finite = joined[np.isfinite(joined)]
    return float(finite.mean()) if finite.size else 0.0


@dataclass
class Normalizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    edge_mean: np.ndarray
    edge_std: np.ndarray

    @classmethod
    def fit(cls, partitions: list["GraphPartition"]) -> "Normalizer":
        xs = [p.x for p in partitions]
        x_mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        x_std = np.ones(FEATURE_DIM, dtype=np.float32)
        for idx in CONTINUOUS_FEATURES:
            x_mean[idx] = _finite_mean(xs, idx)
            joined = np.concatenate([v[:, idx] for v in xs])
            finite = joined[np.isfinite(joined)]
            x_std[idx] = float(finite.std()) if finite.size and finite.std() > 1e-6 else 1.0
        edges = [p.edge_attr for p in partitions if len(p.edge_attr)]
        if edges:
            joined_edges = np.concatenate(edges)
            edge_mean = np.nanmean(joined_edges, axis=0).astype(np.float32)
            edge_std = np.nanstd(joined_edges, axis=0).astype(np.float32)
            edge_std[edge_std < 1e-6] = 1.0
        else:
            edge_mean = np.zeros(3, dtype=np.float32)
            edge_std = np.ones(3, dtype=np.float32)
        return cls(x_mean, x_std, edge_mean, edge_std)

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        out = x.astype(np.float32, copy=True)
        for idx in CONTINUOUS_FEATURES:
            out[:, idx] = (out[:, idx] - self.x_mean[idx]) / self.x_std[idx]
        # Driver IDs are deliberately removed as a numeric shortcut; other categorical
        # IDs are bounded codes and are scaled to approximately [0, 1].
        out[:, 31] = 0.0
        for idx, scale in CAT_FEATURES.items():
            if idx != 31:
                out[:, idx] = out[:, idx] / scale
        return out

    def transform_edges(self, edge_attr: np.ndarray) -> np.ndarray:
        if not len(edge_attr):
            return edge_attr.astype(np.float32)
        return ((edge_attr.astype(np.float32) - self.edge_mean) / self.edge_std).astype(np.float32)

    def to_json(self) -> dict:
        return {"x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(), "edge_mean": self.edge_mean.tolist(), "edge_std": self.edge_std.tolist()}


class GraphPartition:
    def __init__(self, path: Path):
        self.path = path
        self.season = int(path.parts[-2])
        self.split = path.parts[-1]
        self.z = np.load(path / "graph_sequences.npz", allow_pickle=False)
        self.x = self.z["x"]
        self.edge_index = self.z["edge_index"]
        self.edge_attr = self.z["edge_attr"]
        self.snapshot_ptr = self.z["snapshot_ptr"]
        self.edge_ptr = self.z["edge_ptr"]
        self.sequence_snapshot_ids = self.z["sequence_snapshot_ids"]
        self.target_ptr = self.z["target_ptr"]
        self.target_node_indices = self.z["target_node_indices"]
        self.target_time = self.z["target_lap_time_s"]
        self.target_delta = self.z["target_lap_time_delta_s"]
        self.target_wear = self.z["target_tyre_wear_estimate_fraction"]
        self.target_valid = self.z["target_valid"]
        self.target_high_tyre = self.z["target_high_tyre_degradation"]
        self.target_pit = self.z["target_pit_stop_observed"]
        self.target_safety = self.z["target_safety_constraint"]
        self.target_class_valid = self.z["target_classification_valid"]
        self.node_driver_index = self.z["node_driver_index"]
        self.snapshots = [json.loads(line) for line in (path / "snapshots.jsonl").read_text().splitlines()]

    def __len__(self) -> int:
        return len(self.sequence_snapshot_ids)

    def sample(self, sequence_id: int) -> dict:
        snapshot_ids = self.sequence_snapshot_ids[sequence_id].astype(np.int64)
        target_start, target_end = int(self.target_ptr[sequence_id]), int(self.target_ptr[sequence_id + 1])
        target_nodes = self.target_node_indices[target_start:target_end].astype(np.int64)
        target_driver = self.node_driver_index[target_nodes].astype(np.int64)
        x_parts: list[np.ndarray] = []
        edge_parts: list[np.ndarray] = []
        edge_attr_parts: list[np.ndarray] = []
        gather = np.full((len(target_nodes), HISTORY_LAPS), -1, dtype=np.int64)
        node_offset = 0
        for t, sid_value in enumerate(snapshot_ids):
            sid = int(sid_value)
            start, end = int(self.snapshot_ptr[sid]), int(self.snapshot_ptr[sid + 1])
            x_parts.append(self.x[start:end])
            e_start, e_end = int(self.edge_ptr[sid]), int(self.edge_ptr[sid + 1])
            if e_end > e_start:
                e = self.edge_index[:, e_start:e_end] - start + node_offset
                # Use both directions so information can flow to/from the car ahead.
                edge_parts.append(np.concatenate([e, e[::-1]], axis=1))
                edge_attr_parts.append(np.concatenate([self.edge_attr[e_start:e_end], self.edge_attr[e_start:e_end]], axis=0))
            local_driver_map = {int(driver): node_offset + i for i, driver in enumerate(self.node_driver_index[start:end])}
            for r, driver in enumerate(target_driver):
                gather[r, t] = local_driver_map.get(int(driver), -1)
            node_offset += end - start
        current_raw = self.x[target_nodes]
        current_snapshot = self.snapshots[int(snapshot_ids[-1])]
        return {
            "x": np.concatenate(x_parts, axis=0).astype(np.float32),
            "edge_index": np.concatenate(edge_parts, axis=1).astype(np.int64) if edge_parts else np.empty((2, 0), dtype=np.int64),
            "edge_attr": np.concatenate(edge_attr_parts, axis=0).astype(np.float32) if edge_attr_parts else np.empty((0, 3), dtype=np.float32),
            "gather": gather,
            "time": self.target_time[target_start:target_end].astype(np.float32),
            "delta": self.target_delta[target_start:target_end].astype(np.float32),
            "wear": self.target_wear[target_start:target_end].astype(np.float32),
            "high_tyre": self.target_high_tyre[target_start:target_end].astype(np.float32),
            "pit": self.target_pit[target_start:target_end].astype(np.float32),
            "safety": self.target_safety[target_start:target_end].astype(np.float32),
            "reg_valid": self.target_valid[target_start:target_end].astype(bool),
            "class_valid": self.target_class_valid[target_start:target_end].astype(bool),
            "physics_ref": (current_raw[:, DISTANCE_IDX] / np.maximum(current_raw[:, SPEED_IDX], 1.0)).astype(np.float32),
            "context": current_raw[:, [32, 33, 34, 20, 21, 25, 29]].astype(np.float32),
            "event": current_snapshot["event"],
            "season": self.season,
        }


class GraphSequenceDataset(Dataset):
    def __init__(self, partitions: list[GraphPartition]):
        self.partitions = partitions
        self.refs = [(p, i) for p in partitions for i in range(len(p))]

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict:
        partition, sequence_id = self.refs[index]
        return partition.sample(sequence_id)


def collate(samples: list[dict], normalizer: Normalizer) -> dict[str, torch.Tensor | list[str]]:
    all_x, all_e, all_ea = [], [], []
    gather_parts, target_parts = [], []
    offset = 0
    for sample in samples:
        all_x.append(normalizer.transform_x(sample["x"]))
        e = sample["edge_index"] + offset
        all_e.append(e)
        all_ea.append(normalizer.transform_edges(sample["edge_attr"]))
        gather = sample["gather"].copy()
        gather[gather >= 0] += offset
        gather_parts.append(gather)
        target_parts.append(sample)
        offset += len(sample["x"])
    def cat_or_empty(items, shape, dtype):
        return np.concatenate(items, axis=0) if any(len(x) for x in items) else np.empty(shape, dtype=dtype)
    gather = np.concatenate(gather_parts, axis=0)
    event = sum(([s["event"]] * len(s["time"]) for s in target_parts), [])
    return {
        "x": torch.from_numpy(np.concatenate(all_x, axis=0)),
        "edge_index": torch.from_numpy(np.concatenate(all_e, axis=1) if all_e else np.empty((2, 0), dtype=np.int64)).long(),
        "edge_attr": torch.from_numpy(cat_or_empty(all_ea, (0, 3), np.float32)).float(),
        "gather": torch.from_numpy(gather).long(),
        "gather_valid": torch.from_numpy(gather >= 0),
        "time": torch.from_numpy(np.concatenate([s["time"] for s in target_parts])).float(),
        "delta": torch.from_numpy(np.concatenate([s["delta"] for s in target_parts])).float(),
        "wear": torch.from_numpy(np.concatenate([s["wear"] for s in target_parts])).float(),
        "high_tyre": torch.from_numpy(np.concatenate([s["high_tyre"] for s in target_parts])).float(),
        "pit": torch.from_numpy(np.concatenate([s["pit"] for s in target_parts])).float(),
        "safety": torch.from_numpy(np.concatenate([s["safety"] for s in target_parts])).float(),
        "reg_valid": torch.from_numpy(np.concatenate([s["reg_valid"] for s in target_parts]) & np.isfinite(np.concatenate([s["time"] for s in target_parts]))),
        "class_valid": torch.from_numpy(np.concatenate([s["class_valid"] for s in target_parts])),
        "physics_ref": torch.from_numpy(np.concatenate([s["physics_ref"] for s in target_parts])).float(),
        "context": torch.from_numpy(np.concatenate([s["context"] for s in target_parts])).float(),
        "events": event,
    }


class HybridGNN(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, edge_dim: int = 3, hidden: int = 128, layers: int = 3, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.updates = nn.ModuleList([nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.tyre_gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.time_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
        self.delta_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
        self.wear_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
        self.tyre_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
        self.pit_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
        self.safety_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        edge_message = self.edge_encoder(edge_attr)
        for update, norm in zip(self.updates, self.norms):
            messages = h[src] + edge_message
            aggregate = torch.zeros_like(h)
            aggregate.index_add_(0, dst, messages)
            degree = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
            degree.index_add_(0, dst, torch.ones(len(dst), device=h.device, dtype=h.dtype))
            aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(1)
            h = norm(h + update(torch.cat([h, aggregate], dim=1)))
        return h

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = self.encode_graph(batch["x"], batch["edge_index"], batch["edge_attr"])
        gather = batch["gather"].clamp_min(0)
        seq = h[gather]
        seq = seq * batch["gather_valid"].unsqueeze(-1)
        encoded, _ = self.gru(seq)
        representation = encoded[:, -1]
        # Separate leakage-controlled branch for tyre outputs. Zero is the
        # training-set mean after normalization, so these proxy columns are
        # unavailable to the tyre wear/degradation heads.
        tyre_x = batch["x"].clone()
        tyre_x[:, list(TYRE_PROXY_FEATURES)] = 0.0
        tyre_h = self.encode_graph(tyre_x, batch["edge_index"], batch["edge_attr"])
        tyre_seq = tyre_h[gather] * batch["gather_valid"].unsqueeze(-1)
        tyre_encoded, _ = self.tyre_gru(tyre_seq)
        tyre_representation = tyre_encoded[:, -1]
        return {
            "time": torch.nn.functional.softplus(self.time_head(representation).squeeze(1)) + 1.0,
            "delta": self.delta_head(representation).squeeze(1),
            "wear": torch.sigmoid(self.wear_head(tyre_representation).squeeze(1)),
            "tyre_logit": self.tyre_head(tyre_representation).squeeze(1),
            "pit_logit": self.pit_head(representation).squeeze(1),
            "safety_logit": self.safety_head(representation).squeeze(1),
        }


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return nn.functional.smooth_l1_loss(pred[mask], target[mask], beta=beta)


def compute_pos_weights(partitions: list[GraphPartition]) -> torch.Tensor:
    values = []
    for name in ("target_high_tyre", "target_pit", "target_safety"):
        positives = negatives = 0
        for p in partitions:
            mask = p.target_valid if name == "target_high_tyre" else p.target_class_valid
            y = getattr(p, name)
            positives += int(y[mask].sum())
            negatives += int(mask.sum() - y[mask].sum())
        values.append(float(np.clip(negatives / max(positives, 1), 1.0, 20.0)))
    return torch.tensor(values, dtype=torch.float32)


def initialize_time_head(model: HybridGNN, partitions: list[GraphPartition]) -> float:
    values = []
    for partition in partitions:
        valid = partition.target_valid & np.isfinite(partition.target_time)
        values.append(partition.target_time[valid])
    joined = np.concatenate(values) if values else np.asarray([90.0], dtype=np.float32)
    mean_time = float(np.clip(np.mean(joined), 2.0, 240.0))
    # time = softplus(raw) + 1; initialize raw so the first prediction is near
    # the train-set mean instead of starting around one second.
    raw_bias = math.log(math.expm1(mean_time - 1.0))
    with torch.no_grad():
        model.time_head[-1].bias.fill_(raw_bias)
    return mean_time


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor, gamma: float) -> torch.Tensor:
    base = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = target * probabilities + (1.0 - target) * (1.0 - probabilities)
    return (((1.0 - p_t).clamp_min(1e-6) ** gamma) * base).mean()


def loss_fn(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], pos_weight: torch.Tensor, physics_weight: float, focal_gamma: float) -> tuple[torch.Tensor, dict[str, float]]:
    reg = batch["reg_valid"]
    class_mask = batch["class_valid"]
    tyre_mask = reg
    time_loss = _masked_smooth_l1(outputs["time"], batch["time"], reg, 5.0)
    delta_loss = _masked_smooth_l1(outputs["delta"], batch["delta"], reg & torch.isfinite(batch["delta"]), 1.0)
    wear_loss = _masked_smooth_l1(outputs["wear"], batch["wear"], tyre_mask & torch.isfinite(batch["wear"]), 0.05)
    tyre_loss = focal_bce_with_logits(outputs["tyre_logit"][tyre_mask], batch["high_tyre"][tyre_mask], pos_weight[0], focal_gamma) if bool(tyre_mask.any()) else outputs["time"].sum() * 0.0
    pit_loss = focal_bce_with_logits(outputs["pit_logit"][class_mask], batch["pit"][class_mask], pos_weight[1], focal_gamma) if bool(class_mask.any()) else outputs["time"].sum() * 0.0
    safety_loss = focal_bce_with_logits(outputs["safety_logit"][class_mask], batch["safety"][class_mask], pos_weight[2], focal_gamma) if bool(class_mask.any()) else outputs["time"].sum() * 0.0
    physics_mask = reg & torch.isfinite(batch["physics_ref"]) & (batch["physics_ref"] > 0.0)
    physics_loss = _masked_smooth_l1(outputs["time"], batch["physics_ref"], physics_mask, 5.0)
    total = time_loss + 0.30 * delta_loss + 0.20 * wear_loss + 0.50 * tyre_loss + 0.50 * pit_loss + 0.50 * safety_loss + physics_weight * physics_loss
    return total, {"time": float(time_loss.detach()), "delta": float(delta_loss.detach()), "wear": float(wear_loss.detach()), "tyre": float(tyre_loss.detach()), "pit": float(pit_loss.detach()), "safety": float(safety_loss.detach()), "physics": float(physics_loss.detach())}


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def _macro_auc(y: np.ndarray, score: np.ndarray, events: list[str]) -> float | None:
    per_event = []
    for event in sorted(set(events)):
        mask = np.asarray([e == event for e in events])
        value = _auc(y[mask], score[mask])
        if value is not None:
            per_event.append(value)
    return float(np.mean(per_event)) if per_event else None


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_outputs = {key: [] for key in ("time", "delta", "wear", "tyre_logit", "pit_logit", "safety_logit")}
    all_targets = {key: [] for key in ("time", "delta", "wear", "high_tyre", "pit", "safety", "reg_valid", "class_valid")}
    events: list[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch)
        for key in all_outputs:
            all_outputs[key].append(outputs[key].detach().cpu().numpy())
        for key in all_targets:
            all_targets[key].append(batch[key].detach().cpu().numpy())
        events.extend(batch["events"])
    out = {key: np.concatenate(value) for key, value in all_outputs.items()}
    tar = {key: np.concatenate(value) for key, value in all_targets.items()}
    reg = tar["reg_valid"].astype(bool)
    result = {
        "lap_time_mae_s": float(mean_absolute_error(tar["time"][reg], out["time"][reg])) if reg.any() else None,
        "lap_time_rmse_s": float(math.sqrt(mean_squared_error(tar["time"][reg], out["time"][reg]))) if reg.any() else None,
        "lap_time_bias_s": float(np.mean(out["time"][reg] - tar["time"][reg])) if reg.any() else None,
    }
    for name, target, score, mask in (
        ("tyre", tar["high_tyre"], out["tyre_logit"], reg),
        ("pit", tar["pit"], out["pit_logit"], tar["class_valid"].astype(bool)),
        ("safety", tar["safety"], out["safety_logit"], tar["class_valid"].astype(bool)),
    ):
        probability = 1.0 / (1.0 + np.exp(-score))
        result[f"{name}_roc_auc"] = _auc(target[mask], probability[mask])
        result[f"{name}_pr_auc"] = float(average_precision_score(target[mask], probability[mask])) if mask.any() and len(np.unique(target[mask])) > 1 else None
        result[f"{name}_macro_race_roc_auc"] = _macro_auc(target[mask], probability[mask], [e for e, keep in zip(events, mask) if keep])
    result["samples"] = int(len(events))
    result["valid_regression_targets"] = int(reg.sum())
    result["valid_classification_targets"] = int(tar["class_valid"].sum())
    return result


def load_partitions(root: Path, years: Iterable[int], split: str) -> list[GraphPartition]:
    partitions = []
    for year in sorted(set(years)):
        path = root / str(year) / split
        if not (path / "graph_sequences.npz").exists():
            raise FileNotFoundError(path / "graph_sequences.npz")
        partitions.append(GraphPartition(path))
    return partitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", default="gnn_sequences")
    parser.add_argument("--output-dir", default="training_runs/hybrid_gnn")
    parser.add_argument("--train-years", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--validation-years", nargs="+", type=int, default=[2025])
    parser.add_argument("--test-years", nargs="+", type=int, default=[2026])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4, help="reserved for attention-compatible configs")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--physics-weight", type=float, default=0.02)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="run one epoch and one batch per split")
    parser.add_argument("--resume", default=None, help="checkpoint (.pt) to resume from")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    if name == "mps" or (name == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    graph_root = Path(args.graph_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    train_parts = load_partitions(graph_root, args.train_years, args.train_split)
    val_parts = load_partitions(graph_root, args.validation_years, args.validation_split)
    test_parts = load_partitions(graph_root, args.test_years, args.test_split)
    normalizer = Normalizer.fit(train_parts)
    train_ds, val_ds, test_ds = GraphSequenceDataset(train_parts), GraphSequenceDataset(val_parts), GraphSequenceDataset(test_parts)
    collate_fn = lambda samples: collate(samples, normalizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    model = HybridGNN(hidden=args.hidden, layers=args.layers, heads=args.heads, dropout=args.dropout).to(device)
    initialized_time_mean = initialize_time_head(model, train_parts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    pos_weight = compute_pos_weights(train_parts).to(device)
    config = vars(args) | {"device_resolved": str(device), "initialized_time_mean_s": initialized_time_mean, "train_sequences": len(train_ds), "validation_sequences": len(val_ds), "test_sequences": len(test_ds), "pos_weight": pos_weight.cpu().tolist(), "normalizer": normalizer.to_json()}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best_mae = math.inf
    best_epoch = -1
    stale = 0
    history = []
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume).resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        saved_history = checkpoint.get("history")
        if saved_history is None and (output_dir / "history.json").exists():
            saved_history = json.loads((output_dir / "history.json").read_text())
        history = [record for record in (saved_history or []) if int(record.get("epoch", 0)) < start_epoch]
        best_mae = float(checkpoint.get("best_mae", min((r["validation"]["lap_time_mae_s"] for r in history if r["validation"].get("lap_time_mae_s") is not None), default=math.inf)))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        if best_epoch < 0 and history:
            best_epoch = min((r for r in history if r["validation"].get("lap_time_mae_s") is not None), key=lambda r: r["validation"]["lap_time_mae_s"])["epoch"]
        stale = int(checkpoint.get("stale", 0))
        print(json.dumps({"resume": str(resume_path), "start_epoch": start_epoch, "best_epoch": best_epoch, "best_validation_lap_time_mae_s": best_mae}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_total = 0.0
        component_totals = {key: 0.0 for key in ("time", "delta", "wear", "tyre", "pit", "safety", "physics")}
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, components = loss_fn(outputs, batch, pos_weight, args.physics_weight, args.focal_gamma)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_total += float(loss.detach())
            for key in component_totals:
                component_totals[key] += components[key]
            batches += 1
            if args.smoke and batch_index == 0:
                break
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["lap_time_mae_s"] if val_metrics["lap_time_mae_s"] is not None else loss_total / max(batches, 1))
        record = {"epoch": epoch, "train_loss": loss_total / max(batches, 1), "train_components": {k: v / max(batches, 1) for k, v in component_totals.items()}, "validation": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(json.dumps(record), flush=True)
        val_mae = val_metrics["lap_time_mae_s"]
        if val_mae is not None and val_mae < best_mae:
            best_mae, best_epoch, stale = val_mae, epoch, 0
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best_mae": best_mae, "best_epoch": best_epoch, "stale": stale, "history": history, "config": config}, output_dir / "best.pt")
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best_mae": best_mae, "best_epoch": best_epoch, "stale": stale, "history": history, "config": config}, output_dir / "last.pt")
        if args.smoke or stale >= args.patience:
            break
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False) if (output_dir / "best.pt").exists() else {"model": model.state_dict()}
    model.load_state_dict(checkpoint["model"])
    final = {"best_epoch": best_epoch, "best_validation_lap_time_mae_s": best_mae, "validation": evaluate(model, val_loader, device), "test": evaluate(model, test_loader, device)}
    (output_dir / "metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "metrics": final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
