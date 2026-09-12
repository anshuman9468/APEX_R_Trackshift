# APEX-R GNN experiment v1 — host-RAM repair report

## Repair scope

This repair addressed only the diagnosed host-memory bottleneck. The model,
features, graph topology, labels, splits, preprocessing mathematics, DataLoader
settings, CPU device, optimizer, learning rate, weight decay, batch size,
epoch limit, early stopping and seeds were preserved.

The legacy `graph_cache.pt` was an eager nested Python list. Its 101,042,629
bytes on disk expanded to a Python process RSS of about 7.4 GB and triggered
the Linux kernel OOM killer. The RAM-safe path never calls `torch.load()` on
that legacy cache.

## Replacement storage

`disk_graph_store/` contains memory-mapped NumPy arrays for node, edge, edge
index, pair and label values, plus offsets and small per-graph metadata. Raw
numeric values remain float64, matching the legacy Python doubles. The existing
PyTorch boundary remains float32 for `Data.x`, `Data.edge_attr` and
`Data.pair_x`; this is the same dtype behavior as the original runner.

The legacy pickle was parsed in two streaming passes. Every one of the 15,404
graphs was compared against its new disk-backed slices one at a time:

* graph count: 15,404 versus 15,404;
* metadata mismatches: 0;
* node-value mismatches: 0;
* edge-value mismatches: 0;
* edge-index mismatches: 0;
* pair-value mismatches: 0;
* label mismatches: 0;
* maximum absolute numeric difference: 0.0.

## Pre-resume memory gate

The required non-training gate opened the lazy store, computed training-only
statistics, compared the formulas using bounded training arrays, and loaded one
32-graph forward/backward batch without an optimizer step.

Measured peak RSS: **498.5 MiB**.

The streaming-statistic outputs differ from the bounded legacy-formula
reference only due to summation order: maximum absolute differences were
`1.33e-14` for node means, `1.21e-10` for node standard deviations,
`7.75e-11` for edge means, `1.30e-11` for edge standard deviations,
`2.27e-11` for pair standard deviations, and `1.83e-10` for tabular standard
deviations. No non-finite or material value changes were found.

## Full run

The full RAM-safe run completed with the frozen configuration:

* seed 17: best epoch 24, 39 epochs run;
* seed 23: best epoch 21, 36 epochs run;
* seed 42: best epoch 22, 37 epochs run;
* peak RSS: **594.9 MiB**;
* validation checkpoints and thresholds were frozen before development-test
  scoring;
* the five-race development-test was accessed once after the freeze;
* no model revision followed test access.

The model metrics remain those produced by the RAM-safe run and are recorded
in `metrics.csv`, `metrics.json`, `precision_recall_curves.csv`,
`reliability_bins.csv` and `PHASE4_REPORT.md`. This remains a boundary-order
proxy task, not verified overtaking or strategy evidence.

## Verification

`ram_safe_test_results.json` records **24/24 passed** checks, including store
layout, edge bounds, attacker/target mapping, split isolation, protected
holdout exclusion, train-only preprocessing, tensor dtypes, checkpoint reload
parity, source immutability, test-after-freeze and memory bounds.

The sealed session `11353` was not processed, requested or scored. The legacy
cache and all source inputs remain preserved outside the RAM-safe package.
