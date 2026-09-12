# Frozen APEX-R GNN Proxy v1

This directory is the immutable demo-integration bundle for the selected seed-42, epoch-28 GNN checkpoint. It contains no training command and no raw telemetry payload.

## Demo contract

- **Input:** a graph built from approved telemetry using `GRAPH_SCHEMA.json`, `FEATURE_SCHEMA.json` and the frozen `preprocessing_state.json`.
- **Output label:** **experimental boundary position-swap proxy signal**.
- **Meaning:** a raw sigmoid score for the fixed-pair next-boundary position-swap proxy task.
- **Usage:** advisory signal only. The APEX-R rule/energy simulator must make the final ATTACK/HOLD/HARVEST/DEFEND decision.

Do not present the score as a calibrated overtake probability, an ATTACK-success probability, or proof of energy-strategy benefit. Do not silently create required inputs. If fixed attacker/target evidence is missing or stale, return an unavailable status.

## Runtime

The verified local environment used:

- Python 3
- PyTorch 2.11.0+cu128
- torch-geometric 2.8.0.post1
- NumPy (the version installed in the project GPU environment)

The model supports CPU inference for demo portability and CUDA inference when the compatible runtime is available. CPU and CUDA scores may differ by small floating-point amounts.

## Smoke test

From this directory, using the existing external-drive environment:

```bash
../../../.venv_gnn_gpu/bin/python smoke_test.py --device cpu
```

Optional CUDA check:

```bash
../../../.venv_gnn_gpu/bin/python smoke_test.py --device cuda
```

The test loads only `approved_smoke_fixture.json`, an already-approved development-validation graph whose labels/outcomes are omitted. It runs `eval()` plus `torch.inference_mode()`, checks that the score is finite and in `[0,1]`, and executes no optimiser, loss, backward pass or training code.

## Integrity

Verify packaged files from this directory with:

```bash
sha256sum -c SHA256SUMS.txt
```

`FROZEN_MODEL_MANIFEST.json` records the selected checkpoint metadata, metrics, dataset archive hash, experiment provenance and protected-holdout exclusion. `SOURCE_IMMUTABILITY.json` records matching before/after hashes for the original source artifacts.

## Included files

- `gnn_seed_42_best.pt`: frozen model checkpoint.
- `model_config.json`: checked checkpoint and architecture configuration.
- `EXPERIMENT_SPEC.md`: frozen experiment contract.
- `FEATURE_SCHEMA.json`, `GRAPH_SCHEMA.json`: input definitions.
- `preprocessing_state.json`: train-only frozen preprocessing state.
- `frozen_split_manifest.json`: original chronology-only development split.
- `metrics.json`, `metrics_summary.json`: full and selected-run metrics.
- `EXPERIMENT_REPORT.md`: original GPU experiment report copy.
- `approved_smoke_fixture.json`, `smoke_test.py`: offline inference check.
- `MODEL_CARD.md`: intended use and limitations.

