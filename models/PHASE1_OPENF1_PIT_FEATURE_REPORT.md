# OpenF1 Pit Feature Augmentation Report

Generated: `2026-09-10T17:40:28.528058+00:00`

## Scope

Validation-only augmentation of the existing Phase 1 train/validation tables. Existing Phase 1 columns were copied unchanged. No model was trained or modified by this builder.

## New columns

- `attacker_openf1_pit_recent_flag` and `rival_openf1_pit_recent_flag`: 1 when the latest OpenF1 pit event for that driver is at or before the prediction time and no older than 90 seconds; 0 when the source is available but no event is recent.
- `attacker_openf1_last_pit_lane_duration_sec` and `rival_openf1_last_pit_lane_duration_sec`: the `lane_duration` from the latest prior OpenF1 pit event.
- Join: `session_key + driver_number`, then backward event-time lookup against the Phase 1 row's UTC `date`. No nearest-neighbour tolerance or interpolation is used for discrete pit events.
- Causality: only `pit.date <= prediction date` is eligible. The endpoint timestamp is not relabeled as pit exit; the existing TracingInsights `pout` columns remain the pit-out source.

## Endpoint availability

- Session `7953`: `http_404_no_records`, records `0`.
- Session `7779`: `http_404_no_records`, records `0`.
- Session `7787`: `http_404_no_records`, records `0`.
- Session `9070`: `http_404_no_records`, records `0`.

## Missingness

### train

| Feature | Missing | Rows | Rate | Main reason |
|---|---:|---:|---:|---|
| `attacker_openf1_pit_recent_flag` | 13,912 | 13,912 | 100.00% | `pit_endpoint_unavailable_http_404_no_records` |
| `rival_openf1_pit_recent_flag` | 13,912 | 13,912 | 100.00% | `pit_endpoint_unavailable_http_404_no_records` |
| `attacker_openf1_last_pit_lane_duration_sec` | 13,912 | 13,912 | 100.00% | `pit_endpoint_unavailable_http_404_no_records` |
| `rival_openf1_last_pit_lane_duration_sec` | 13,912 | 13,912 | 100.00% | `pit_endpoint_unavailable_http_404_no_records` |

### validation

| Feature | Missing | Rows | Rate | Main reason |
|---|---:|---:|---:|---|
| `attacker_openf1_pit_recent_flag` | 6,058 | 6,058 | 100.00% | `pit_endpoint_unavailable_http_404_no_records` |
| `rival_openf1_pit_recent_flag` | 6,058 | 6,058 | 100.00% | `pit_endpoint_unavailable_http_404_no_records` |
| `attacker_openf1_last_pit_lane_duration_sec` | 6,058 | 6,058 | 100.00% | `pit_endpoint_unavailable_http_404_no_records` |
| `rival_openf1_last_pit_lane_duration_sec` | 6,058 | 6,058 | 100.00% | `pit_endpoint_unavailable_http_404_no_records` |

## Interpretation

The approved 2023 train/validation sessions returned no OpenF1 `pit` records during this build, so these four new columns are explicitly missing rather than filled with zero. They cannot contribute signal to a model trained on this split. This is a data-availability limitation, not evidence that pit data is unhelpful on other sessions.

## Outputs

- Train: `data/phase1-features/train_phase1_openf1_pit_features.csv.gz`
- Validation: `data/phase1-features/validation_phase1_openf1_pit_features.csv.gz`
- Manifest: `data/phase1-features/phase1_openf1_pit_feature_manifest.json`
