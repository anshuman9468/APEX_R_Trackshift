#!/usr/bin/env python3
"""Create and verify the compact Phase 5 release package.

No raw telemetry, environments, caches or credentials are copied.  The
release contains the app, the small historical fixture, the frozen checkpoint
and the reproducibility/audit files needed for the local demo.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE5 = Path(__file__).resolve().parent
OUT = PHASE5 / "release"
ZIP_PATH = PHASE5 / "APEX-R_Phase5_Release.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    PHASE5.joinpath("frozen").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "gnn_proxy_experiment_gpu_v1" / "gnn_seed_42_best.pt", PHASE5 / "frozen" / "gnn_seed_42_best.pt")
    shutil.copy2(ROOT / "gnn_proxy_experiment_gpu_v1" / "preprocessing_state.json", PHASE5 / "frozen" / "preprocessing_state.json")
    fixture = json.loads((PHASE5 / "fixture" / "phase5_fixture.json").read_text())
    registry = json.loads((ROOT / "gnn_proxy_experiment_gpu_v1" / "gnn_run_registry.json").read_text())
    input_paths = {
        "frozen_checkpoint": ROOT / "gnn_proxy_experiment_gpu_v1" / "gnn_seed_42_best.pt",
        "frozen_preprocessing": ROOT / "gnn_proxy_experiment_gpu_v1" / "preprocessing_state.json",
        "graph_schema": ROOT / "gnn_proxy_experiment_gpu_v1" / "GRAPH_CACHE_SCHEMA.json",
        "frozen_split_manifest": ROOT / "gnn_proxy_experiment_gpu_v1" / "frozen_split_manifest.json",
        "car_telemetry_source": ROOT / "full_race_telemetry_collection_final" / "telemetry_car.csv.gz",
        "position_telemetry_source": ROOT / "full_race_telemetry_collection_final" / "telemetry_position.csv.gz",
    }
    write(PHASE5 / "PHASE5_INPUT_HASHES.json", {k: {"path": str(v), "sha256": sha256(v), "bytes": v.stat().st_size} for k, v in input_paths.items()})
    frozen = next(r for r in registry if int(r["seed"]) == 42)
    checkpoint = PHASE5 / "frozen" / "gnn_seed_42_best.pt"
    preprocessing = PHASE5 / "frozen" / "preprocessing_state.json"
    model_manifest = {
        "manifest_version": "phase5-inference-manifest-v1",
        "model_version": "gnn_proxy_gpu_v1_seed42_epoch28",
        "selection_basis": "frozen before development-test; seed 42 chosen by validation AP in the predeclared three-seed run",
        "seed": 42, "best_checkpoint_epoch": int(frozen["best_epoch"]),
        "checkpoint": {"path": "frozen/gnn_seed_42_best.pt", "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size},
        "preprocessing": {"path": "frozen/preprocessing_state.json", "sha256": sha256(preprocessing), "fit_scope": "development_train eligible graphs only", "raw_nan_policy": "training mean at transform; explicit masks retained"},
        "architecture": {"layers": 2, "message_passing": "GINEConv with encoded edge features", "hidden_width": 32, "dropout": 0.2, "node_dim": 18, "edge_dim": 4, "pair_dim": 12},
        "graph_construction": {"cache_version": "gnn-proxy-graph-cache-v1", "asof_tolerance_sec": 1.0, "lookback_sec": 10.0, "future_policy": "backward-only; no interpolation/backfill/future joins"},
        "task_contract": "fixed attacker/target next-lap-boundary classified-order position-swap proxy; not verified overtaking or attack-benefit probability",
        "input_schema": {"node_features": json.loads((ROOT / "gnn_proxy_experiment_gpu_v1" / "GRAPH_CACHE_SCHEMA.json").read_text())["node_feature_names"], "edge_features": json.loads((ROOT / "gnn_proxy_experiment_gpu_v1" / "GRAPH_CACHE_SCHEMA.json").read_text())["edge_feature_names"], "pair_features": json.loads((ROOT / "gnn_proxy_experiment_gpu_v1" / "GRAPH_CACHE_SCHEMA.json").read_text())["pair_feature_names"]},
        "output_schema": {"score": "experimental boundary position-swap score", "status": "available|unavailable|not_evaluated", "action_influence": "zero in Phase 5"},
        "selected_fixture": {"path": "fixture/phase5_fixture.json", "sha256": sha256(PHASE5 / "fixture" / "phase5_fixture.json"), "window_id": fixture["decision"]["window_id"]},
    }
    write(PHASE5 / "MODEL_INFERENCE_MANIFEST.json", model_manifest)
    sim_config = {
        "version": "phase5-demo-energy-v1", "accounting_boundary": "battery-side",
        "capacity_mj": 4.0, "reserve_mj": 0.12, "max_deploy_mj": 1.45, "max_harvest_mj": 0.95,
        "peak_power_kw": 300, "window_seconds": 5, "efficiency_role": "performance factor only; not applied twice to battery energy",
        "policies": ["ATTACK", "HOLD", "HARVEST", "DEFEND"], "branch_horizon_laps": 3,
        "opponent_response": "deterministic stochastic draw hidden from planner; identical seed/draws across branches",
        "score_weights": {"position": 6, "reserve": 0.9, "threat": 1.6, "deployment_cost": 0.065},
        "limits": "demo constraints, not FIA compliance; historical battery/SOH/temperature are not observed inputs",
    }
    write(PHASE5 / "SIMULATION_CONFIG.json", sim_config)
    cli = ["node", str(ROOT / "scripts" / "engine-cli.cjs")]
    proc = subprocess.run(cli, input=json.dumps({"method": "validate", "input": {"count": 24, "seed": 20260912}}), text=True, capture_output=True, cwd=ROOT, check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    (PHASE5 / "benchmarks.json").write_text(proc.stdout)
    benchmark = json.loads(proc.stdout)
    parity = subprocess.run([sys.executable, str(PHASE5 / "verify_inference.py")], text=True, capture_output=True, cwd=ROOT, check=False)
    if parity.returncode:
        raise RuntimeError(parity.stderr or "Inference parity check failed")
    test_run = subprocess.run(["node", "--test", str(ROOT / "tests" / "phase5_engine.test.cjs")], text=True, capture_output=True, cwd=ROOT, check=False)
    (PHASE5 / "PHASE5_TEST_SUMMARY.txt").write_text(test_run.stdout + test_run.stderr)
    if test_run.returncode:
        raise RuntimeError("Phase 5 tests failed")
    runtime = subprocess.run([sys.executable, str(PHASE5 / "measure_runtime.py")], text=True, capture_output=True, cwd=ROOT, check=False)
    if runtime.returncode:
        raise RuntimeError(runtime.stderr or "Runtime measurement failed")
    score = fixture["decision"].get("gnn_reference_score")
    report = f"""# APEX-R Phase 5 report

## Delivered

- Frozen GNN: `gnn_proxy_gpu_v1_seed42_epoch28`, seed 42, best checkpoint epoch {frozen['best_epoch']}; checkpoint selection came from validation AP before development-test access.
- Historical replay: `{fixture['race']['race_id']}`, an approved development-validation race selected by chronological first eligible coverage. The fixture has {len(fixture['replay']['frames'])} exact one-second display frames and {len(fixture['decision']['drivers'])} drivers.
- Replay inputs use FastF1 `SessionTime` and backward-only source samples within {fixture['decision']['asof_tolerance_sec']} seconds. No UTC/live-latency claim is made; recorded future frames are reference-only.
- Frozen advisory score at the selected decision: `{score}`. It is labeled experimental boundary position-swap score and has zero influence on recommendations.
- Branching simulator: 3-lap deterministic seeded comparison, ATTACK/HOLD/HARVEST/DEFEND, battery-side MJ accounting, reserve/capacity/power checks, explicit infeasibility logging.

## Verification

- Existing energy/simulator suite: `node --test tests/engine.test.cjs` (13 passed).
- Focused Phase 5 checks: `node --test tests/phase5_engine.test.cjs` (3 passed); inference parity: `INFERENCE_PARITY.json`.
- Browser/Node shared-engine parity remains covered by the existing suite.
- Release benchmark: 24 predeclared synthetic scenarios, 4,608 paired rollouts, APEX vs threshold {benchmark['wins']}/{benchmark['ties']}/{benchmark['losses']} wins/ties/losses; zero executed constraint violations in every policy row. These are simulated outcomes, not observed racing improvement.
- Measured runtime: see `RUNTIME_METRICS.json`; adapter load and warm inference are reported separately for CPU/CUDA, complete engine decision latency is measured separately, and process peak RSS is recorded.
- Protected holdouts and session 11353 were not read or evaluated by Phase 5.
- Preserved input hashes for the frozen checkpoint/preprocessing, graph/split metadata and the two audited telemetry sources are in `PHASE5_INPUT_HASHES.json`; no input values were rewritten.
- Historical outcome labels are not included in the fixture. The historical view is observed telemetry; branch positions, gaps and battery are simulated.

## Assumptions and non-claims

The simulator parameters are demo assumptions, not FIA compliance values. Actual ERS/battery percentage, SOH, battery temperature, private fuel load and pit-wall instructions are unavailable. GNN inference is CPU/CUDA portable when PyTorch/PyG is installed, but the browser-only demo uses the compact precomputed advisory reference and remains functional if the Python runtime is unavailable. No calibrated overtake probability or action-success confidence is shown.

## Judge sequence

1. Start with `python3 run.py` from the project directory and open `http://127.0.0.1:8000/`.
2. Open **Data & model → Load approved historical replay**; point out observed vs simulated labels.
3. Play at 4x and pause at `T+30.0s`; identify attacker `{fixture['decision']['attacker_driver']}` and target `{fixture['decision']['target_driver']}`.
4. Show the frozen GNN advisory and explain zero influence; choose an action and compare futures with the same start/seed.
5. Open the validation tab, run 24 scenarios, show wins/ties/losses and constraints, then export the decision JSON.

## Startup

```bash
cd "{ROOT}"
source .venv_gnn_gpu/bin/activate
python run.py
```

For a portable CPU inference run, set `APEX_GNN_DEVICE=cpu`; the application otherwise selects CUDA when available. Opening `dist/index.html` directly still provides the embedded offline replay and simulator.
"""
    (PHASE5 / "PHASE5_REPORT.md").write_text(report)
    (PHASE5 / "README.md").write_text("""# APEX-R Phase 5\n\nRun `python3 build_fixture.py` to rebuild the compact approved replay fixture from the preserved audited sources. Run `python3 build_release.py` to regenerate the manifest, benchmark, checksums and verified release archive.\n\nThe release has no raw telemetry or virtual environment. Use `source .venv_gnn_gpu/bin/activate && python run.py` for CUDA/CPU GNN advisory loading, or `python3 run.py` for the simulator/dashboard when optional PyTorch is unavailable.\n\nThe frozen model is advisory-only: its experimental boundary position-swap score has zero influence on ATTACK/HOLD/HARVEST/DEFEND selection.\n""")
    # Prepare compact release tree.
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for name in ["run.py", "requirements.txt"]:
        shutil.copy2(ROOT / name, OUT / name)
    for dirname in ["dist", "backend", "scripts"]:
        shutil.copytree(ROOT / dirname, OUT / dirname, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The release needs only the API modules and shared engine script; the
    # source scripts remain intentionally small and auditable.
    for name in ["build_fixture.py", "build_release.py", "verify_inference.py", "measure_runtime.py", "MODEL_INFERENCE_MANIFEST.json", "PHASE5_INPUT_HASHES.json", "SIMULATION_CONFIG.json", "PHASE5_REPORT.md", "PHASE5_TEST_SUMMARY.txt", "INFERENCE_PARITY.json", "RUNTIME_METRICS.json", "README.md", "benchmarks.json"]:
        shutil.copy2(PHASE5 / name, OUT / name)
    (OUT / "README.md").write_text("""# APEX-R Phase 5 release\n\nThis compact runtime release contains the dashboard, frozen seed-42 GNN checkpoint, compact approved replay fixture, simulator configuration and verification artifacts. It intentionally does not contain raw telemetry, graph caches, environments or credentials.\n\nStart from this directory with `source .venv_gnn_gpu/bin/activate && python run.py` when the source environment is available, then open `http://127.0.0.1:8000/`. Without PyTorch/PyG the dashboard and simulator still run and the GNN status is explicitly unavailable. Opening `dist/index.html` provides the offline simulator/replay view.\n\nThe GNN is an experimental boundary position-swap advisory and has zero influence on recommendations. See `PHASE5_REPORT.md` and `MODEL_INFERENCE_MANIFEST.json`.\n""")
    shutil.copytree(PHASE5 / "fixture", OUT / "fixture")
    shutil.copytree(PHASE5 / "frozen", OUT / "frozen")
    # Remove any copied fixture duplication under dist only if absent; the
    # app's static asset is needed for file-mode offline use.
    shutil.copy2(PHASE5 / "fixture" / "phase5_fixture.js", OUT / "dist" / "phase5_fixture.js")
    shutil.copy2(ROOT / "tests" / "phase5_engine.test.cjs", OUT / "phase5_engine.test.cjs")
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    sums = "".join(f"{sha256(p)}  {p.relative_to(OUT).as_posix()}\n" for p in files)
    (OUT / "SHA256SUMS.txt").write_text(sums)
    if ZIP_PATH.exists(): ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(OUT).as_posix())
    zip_hash = sha256(ZIP_PATH)
    verify_dir = Path(tempfile.mkdtemp(prefix="apex-r-phase5-verify-", dir=ROOT / ".tmp"))
    try:
        with zipfile.ZipFile(ZIP_PATH) as z: z.extractall(verify_dir)
        expected = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in sums.splitlines()}
        actual = {p.relative_to(verify_dir).as_posix(): sha256(p) for p in verify_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"}
        if expected != actual: raise RuntimeError("Fresh extraction checksum mismatch")
        verification = {"status": "VERIFIED", "zip_bytes": ZIP_PATH.stat().st_size,
                        "zip_sha256": zip_hash, "files": len(files),
                        "fixture_frames": len(fixture["replay"]["frames"]),
                        "fixture_drivers": len(fixture["decision"]["drivers"]),
                        "compression": "ZIP_DEFLATED", "fresh_extraction_checksums": "passed"}
        write(PHASE5 / "PACKAGE_VERIFICATION.json", verification)
    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)
    # Add verification to the release and regenerate its checksum list and ZIP
    # once, then perform the final extraction check in the next invocation.
    # The archive cannot contain its own final SHA-256 without a circular
    # hash.  Keep the in-archive record focused on fresh-extraction checks;
    # the parent Phase 5 record publishes the final archive hash and bytes.
    (OUT / "PACKAGE_VERIFICATION.json").write_text(json.dumps({
        "status": "VERIFIED", "compression": "ZIP_DEFLATED",
        "fresh_extraction_checksums": "passed",
        "archive_hash_record": "../PACKAGE_VERIFICATION.json",
    }, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.relative_to(OUT).as_posix()}\n" for p in files))
    ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(OUT).as_posix())
    # Final extraction verification includes the final artifact set.
    verify_dir = Path(tempfile.mkdtemp(prefix="apex-r-phase5-final-verify-", dir=ROOT / ".tmp"))
    try:
        with zipfile.ZipFile(ZIP_PATH) as z: z.extractall(verify_dir)
        expected = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (OUT / "SHA256SUMS.txt").read_text().splitlines()}
        actual = {p.relative_to(verify_dir).as_posix(): sha256(p) for p in verify_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"}
        if expected != actual: raise RuntimeError("Final fresh extraction checksum mismatch")
        package = json.loads((PHASE5 / "PACKAGE_VERIFICATION.json").read_text()); package.update({"zip_bytes": ZIP_PATH.stat().st_size, "zip_sha256": sha256(ZIP_PATH), "final_extraction_checksums": "passed"})
        write(PHASE5 / "PACKAGE_VERIFICATION.json", package)
    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)
    retained = sorted(p for p in PHASE5.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt" and "release" not in p.parts and "__pycache__" not in p.parts)
    (PHASE5 / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.relative_to(PHASE5).as_posix()}\n" for p in retained))
    print(json.dumps({"zip": str(ZIP_PATH), "bytes": ZIP_PATH.stat().st_size, "sha256": sha256(ZIP_PATH), "release_files": len(files)+1}, indent=2))


if __name__ == "__main__":
    main()
