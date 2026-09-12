# APEX-R TracingInsights data-availability investigation

Generated: 2026-09-06 (Asia/Kolkata)

## Decision

**Use TracingInsights as a supplement to OpenF1, not as a complete replacement.**

TracingInsights clears the bar for the engineered candidate's most valuable missing rival features: full-field tyre/stint state, rival lap and sector times, per-car sub-lap telemetry, car-ahead identity, car-ahead distance, DRS-open state, and exact pit-in/pit-out session times. It does not provide the official time interval to the car ahead or leader, preserves only binary DRS open/closed rather than eligibility, and does not contain explicit DRS detection/activation points, straight segments, or braking-zone definitions.

No model was trained, refit, calibrated, scored, or modified. The sealed final holdout remained untouched.

## Investigation protocol

- Authoritative schema: [TracingInsights/2026 `DATA_REFERENCE.md`](https://github.com/TracingInsights/2026/blob/main/DATA_REFERENCE.md).
- Actual files inspected one race at a time:
  - 2026 Australian Grand Prix, Race: 20 active drivers in `session_laptimes.json`, 1,008 lap rows, 148 weather rows, and 167 race-control rows.
  - 2024 Australian Grand Prix, Race: 19 active drivers, 998 lap rows, 144 weather rows, and 82 race-control rows.
- The 2024 race was added because the 2026 Norris sample contained the `drs` field but no open values. The 2024 Piastri sample confirmed real `drs=1` observations and a rival car number.
- No repository was cloned. Files were fetched from raw GitHub URLs; maximum temporary data for one race was about 1.1 MB, far below the 500 MB limit.
- Each temporary race directory was deleted before proceeding. Only the schema, requirements file, and small evidence samples were retained.

## Required-feature checklist

| Requirement | Result | What is actually present | Missing part and fallback |
| --- | --- | --- | --- |
| 5s/15s/30s/60s gap-ahead slopes and volatility | **PARTIAL** | `{lap}_tel.json → tel.time`, `DriverAhead`, and `DistanceToDriverAhead` provide irregular sub-lap distance-to-rival samples in metres. | No official interval in seconds and no gap-to-leader field. Use OpenF1 `intervals.interval` and `gap_to_leader` as authoritative seconds; optionally compute distance-gap trends from TracingInsights. |
| DRS eligibility/zone flag and rival DRS-open state | **PARTIAL** | `{lap}_tel.json → tel.drs` is binary open/closed for every downloaded driver; `rcm.json` contains global DRS enabled/disabled messages; rival telemetry can be loaded by car number. | Raw state `8` (detected/eligible) is discarded, and no detection/activation coordinates exist. Keep OpenF1 `car_data.drs` for raw eligibility; do not label inferred geometry as official. |
| Opponent tyre compound/age/stint delta | **CONFIRMED-PRESENT** | `session_laptimes.json → compound, life, fresh, stint, drv, dNum`; the session aggregate covers the full active field. | Join `DriverAhead` to `dNum`, then select the rival's latest valid lap/stint state at the window timestamp. |
| Relative sector/lap-time delta to the car ahead | **CONFIRMED-PRESENT** | `session_laptimes.json → time, s1, s2, s3, s1T, s2T, s3T, lST, sesT, drv, dNum`; car-ahead identity comes from telemetry. | Use an as-of session-time join, not only equal lap numbers, to handle pit cycles and lapped traffic. |
| Pit-out/recent-stop flag | **CONFIRMED-PRESENT** | `pin` and `pout` are session-relative seconds; `lST` and `lSD` anchor the event to the lap and UTC-like date. | Compute `seconds_since_pit_out = window_session_time - latest(pout)` and cap/flag the desired recent-stop window. |
| Straight/braking/DRS-point geometry | **PARTIAL** | `corners.json` gives corner X/Y/distance/angle/rotation. Telemetry gives X/Y/Z, lap distance, speed, throttle and brake. | No explicit straight, braking-zone, DRS detection, or DRS activation objects. Build documented telemetry-derived proxies for straight/braking approach; use a separate circuit map for official DRS points. |

### Bottom line by status

- **Confirmed present:** opponent-relative tyres/stints, relative lap/sector pace, pit-out/recent-stop state, full sub-lap car telemetry, and rival DRS-open state for DRS-era races.
- **Confirmed absent as direct fields:** official interval-to-ahead seconds, gap-to-leader seconds, raw DRS eligibility state, explicit DRS detection/activation points, explicit straight segments, and explicit braking zones.
- **Partial/derivable:** gap trends from distance, DRS-zone proximity from repeated historical open points, straight approach, and braking-zone approach.

## Authoritative file and field inventory

The JSON is generally **column-oriented**: most keys point to arrays, and row `i` is formed by taking index `i` from each array. Telemetry is wrapped under a top-level `tel` key. `dataKey` is a scalar shared by all samples in that telemetry file.

### `weather.json`

Session weather, roughly minute-level.

| Field | Meaning |
| --- | --- |
| `wT` | Session-relative sample time, seconds |
| `wAT` | Air temperature, °C |
| `wH` | Humidity, % |
| `wP` | Atmospheric pressure, mbar |
| `wR` | Rainfall Boolean |
| `wTT` | Track temperature, °C |
| `wWD` | Wind direction, degrees |
| `wWS` | Wind speed, m/s |

### `rcm.json`

Race-control messages.

| Field | Meaning |
| --- | --- |
| `time` | Absolute message timestamp |
| `cat` | Message category, such as `Flag`, `Drs`, or `Other` |
| `msg` | Human-readable message |
| `status` | Status value where supplied |
| `flag` | Flag type where supplied |
| `scope` | Scope such as track, sector, or driver |
| `sector` | Affected sector |
| `dNum` | Affected driver/car number |
| `lap` | Associated lap number |

### `drivers.json`

Top-level `drivers` array, one object per active driver.

| Field | Meaning |
| --- | --- |
| `driver` | Three-letter driver code |
| `team` | Team name |
| `dn` | Driver/car number; preferred cross-source driver key |
| `fn` | First name |
| `ln` | Last name |
| `tc` | Team colour hex string |
| `url` | Driver image URL |

### Per-driver `{lap}_tel.json`

Sub-lap, timestamped car telemetry is **confirmed present**. Path shape: `<event>/<session>/<driver>/<lap>_tel.json → tel`.

| Field | Meaning |
| --- | --- |
| `time` | Seconds from the start of that lap |
| `rpm` | Engine RPM |
| `speed` | Speed, km/h |
| `gear` | Selected gear |
| `throttle` | Throttle, 0–100 |
| `brake` | Binary brake state |
| `drs` | Binary DRS open/closed; raw FastF1 states are collapsed |
| `distance` | Metres from lap start |
| `rel_distance` | Normalised lap distance, 0–1 |
| `DriverAhead` | Car number directly ahead, or `None` |
| `DistanceToDriverAhead` | Distance to that car, metres |
| `acc_x` | Computed longitudinal acceleration |
| `acc_y` | Computed lateral acceleration |
| `acc_z` | Computed vertical acceleration |
| `x` | Interpolated horizontal position, metres |
| `y` | Interpolated horizontal position, metres |
| `z` | Interpolated elevation, metres |
| `dataKey` | `Year-Event-Session-Driver-Lap` identifier |

The acceleration fields are computed by the extraction script, not raw IMU measurements. Position is interpolated.

### Per-driver `laptimes.json` and session aggregate `session_laptimes.json`

`session_laptimes.json` was present in both inspected races and has the same columns as the per-driver files. It is the easiest full-field source. `qs` is session-dependent and appears only for qualifying/sprint-qualifying data.

| Group | Fields | Meaning |
| --- | --- | --- |
| Core timing | `time`, `lap`, `sesT`, `lST`, `lSD` | Lap duration/number, lap end and start in session seconds, lap-start date |
| Sectors | `s1`, `s2`, `s3`, `s1T`, `s2T`, `s3T` | Sector durations and session timestamps |
| Speed traps | `vi1`, `vi2`, `vfl`, `vst` | Intermediate 1/2, finish-line, and straight speed |
| Tyres | `compound`, `life`, `fresh`, `stint` | Compound, tyre life in laps, new/used state, stint number |
| Mini-sectors | `ms1`, `ms2`, `ms3` | Encoded mini-sector classifications; generated only where OpenF1 matching succeeds |
| Qualifying | `qs` | Optional qualifying segment (`Q1`/`Q2`/`Q3` or sprint equivalent) |
| Position/status | `pos`, `status`, `pb` | End-of-lap position, track-status code string, personal-best flag |
| Pit events | `pin`, `pout` | Pit-entry and pit-exit session seconds |
| Quality | `iacc`, `ff1G`, `del`, `delR` | Timing accuracy, generated-lap flag, deletion flag/reason |
| Driver/team | `drv`, `dNum`, `team` | Driver code, car number, team |
| Per-lap weather | `wT`, `wAT`, `wH`, `wP`, `wR`, `wTT`, `wWD`, `wWS` | Weather row matched to the lap |

### `corners.json`

| Field | Meaning |
| --- | --- |
| `CornerNumber` | Sequential corner number |
| `X`, `Y` | Corner map coordinates, metres |
| `Angle` | Marker-placement angle, degrees |
| `Distance` | Marker distance from start/finish, metres |
| `Rotation` | Circuit-map rotation, degrees or null |

### Observed extra `cor.json` in the 2026 race

This file was present in the 2026 race but not listed as an `R.py` output in `DATA_REFERENCE.md`, and it was absent from the inspected 2024 race. Treat it as optional and non-contractual.

- Top-level fields: `corners`, `marshal_lights`, `marshal_sectors`, `rotation`.
- Every item in the three arrays contained `X`, `Y`, `Number`, `Letter`, `Angle`, and `Distance`.
- It adds marshal positions but still does not identify straights or DRS detection/activation points.

## Real sample evidence

### Sub-lap car-ahead and DRS-open state

File: [2024 Australian Race, PIA lap 5 telemetry](https://github.com/TracingInsights/2024/blob/main/Australian%20Grand%20Prix/Race/PIA/5_tel.json), `tel` sample where `time=0.249`:

```json
{
  "speed": 307.0,
  "throttle": 99.0,
  "brake": 0,
  "drs": 1,
  "DriverAhead": "16",
  "DistanceToDriverAhead": 58.69083333333333,
  "distance": 21.248888888888885,
  "x": -1349.2832514159722,
  "y": -1186.1699514960042,
  "dataKey": "2024-Australian Grand Prix-Race-PIA-5"
}
```

Car `16` resolves through `drivers.json → drivers[].dn` to Charles Leclerc. The matching rival file exists; its nearest lap-relative sample was at `time=0.258`, only 0.009 seconds away.

### Rival lap, sectors and tyres

Source: 2024 Australian Race `session_laptimes.json`, lap 5.

| Driver | Car | Lap time | S1 | S2 | S3 | Compound | Life | Stint |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| PIA | 81 | 82.594 | 28.691 | 17.918 | 35.985 | MEDIUM | 5 | 1 |
| LEC (ahead) | 16 | 82.632 | 28.895 | 17.930 | 35.807 | MEDIUM | 5 | 1 |

This directly yields ahead-minus-attacker deltas: lap `+0.038 s`, S1 `+0.204 s`, S2 `+0.012 s`, S3 `-0.178 s`, tyre age `0`, stint `0`, and same-compound `true`.

### Pit-out

File: 2024 Australian Race `session_laptimes.json`:

```json
{"drv":"ALB","dNum":"23","lap":7,"pout":4040.614,"stint":2,"compound":"HARD","life":1}
```

### Race-control DRS state

File: 2024 Australian Race `rcm.json`:

```json
{"time":"2024-03-24T04:04:44.000000000","cat":"Drs","msg":"DRS ENABLED","status":"ENABLED","lap":2}
```

### Corner geometry

File: 2026 Australian Race `corners.json`, first corner:

```json
{"CornerNumber":1,"X":-3650.594482421875,"Y":1193.7841796875,"Angle":-176.6327301808908,"Distance":353.88374054700716,"Rotation":44.0}
```

### Weather

File: 2024 Australian Race `weather.json`, first row:

```json
{"wT":8.798,"wAT":19.0,"wH":49.0,"wP":1021.6,"wR":false,"wTT":37.0,"wWD":306,"wWS":0.4}
```

## Exact 37-feature candidate mapping

The feature names below come directly from `xgboost_engineered_scale2_platt_candidate.json`; they are not reconstructed from memory.

| Feature | TracingInsights availability | Recommended authority |
| --- | --- | --- |
| `brake_fraction_10s` | Directly computable from `tel.brake/time` | TracingInsights |
| `brake_fraction_5s` | Directly computable | TracingInsights |
| `braking_approach_proxy` | Derivable, but no explicit braking-zone object | TracingInsights proxy |
| `closing_rate_sec_per_min` | Only metre-distance closing is direct | OpenF1 interval seconds |
| `drs_eligible_or_open` | Open is present; eligibility was collapsed away | OpenF1 raw DRS |
| `drs_open` | Direct binary state for DRS-era races | TracingInsights or OpenF1, one canonical source |
| `drs_open_fraction_10s` | Directly computable for DRS-era races | TracingInsights |
| `gap_slope_15s` | Distance slope direct; exact seconds slope absent | OpenF1 interval seconds |
| `gap_slope_30s` | Same | OpenF1 interval seconds |
| `gap_slope_5s` | Same | OpenF1 interval seconds |
| `gap_slope_60s` | Same | OpenF1 interval seconds |
| `gap_to_leader_sec` | Absent | OpenF1 intervals |
| `gap_volatility_60s` | Distance volatility direct; exact seconds absent | OpenF1 intervals |
| `gear_mean_10s` | Directly computable from `tel.gear/time` | TracingInsights |
| `high_throttle_fraction_5s` | Directly computable | TracingInsights |
| `interval_sec` | Absent as seconds; distance to ahead is present | OpenF1 intervals |
| `lap_time_delta_ahead_minus_self` | Directly computable full-field | TracingInsights |
| `new_stint_proxy` | Direct from `stint`, `life`, `pout` | TracingInsights |
| `position` | End-of-lap `pos`; no direct high-rate absolute race position | OpenF1 position for live windows |
| `race_progress` | Directly computable from lap/session time | Either, choose one canonical clock |
| `rainfall` | Direct `wR` | TracingInsights |
| `relative_pace_available` | Direct availability check on rival timing | TracingInsights |
| `rival_drs_open` | Rival telemetry is present for DRS-era races | TracingInsights |
| `rpm_mean_10s` | Directly computable | TracingInsights |
| `same_tyre_compound` | Directly computable full-field | TracingInsights |
| `sector_1_delta_ahead_minus_self` | Directly computable | TracingInsights |
| `sector_2_delta_ahead_minus_self` | Directly computable | TracingInsights |
| `sector_3_delta_ahead_minus_self` | Directly computable | TracingInsights |
| `speed_delta_10s` | Directly computable | TracingInsights |
| `speed_delta_5s` | Directly computable | TracingInsights |
| `speed_mean_10s` | Directly computable | TracingInsights |
| `stint_length_delta_ahead_minus_self` | Directly computable full-field | TracingInsights |
| `straight_approach_proxy` | Derivable, no explicit straight object | TracingInsights proxy |
| `throttle_mean_10s` | Directly computable | TracingInsights |
| `track_temperature_c` | Direct `wTT` | TracingInsights |
| `tyre_age` | Direct `life` | TracingInsights |
| `tyre_age_delta_ahead_minus_self` | Directly computable full-field | TracingInsights |

## Merge design with OpenF1

### Join keys

1. **Session:** map TracingInsights `(year, event folder, session folder)` to OpenF1 `meeting` and `session`. Verify the mapping with session name and UTC start date; do not rely on event-name text alone.
2. **Driver:** TracingInsights `drivers[].dn` / lap `dNum` ↔ OpenF1 `driver_number`. Use three-letter codes only as a diagnostic fallback.
3. **Lap:** TracingInsights `lap` ↔ OpenF1 `lap_number` for lap-level records.
4. **Timestamp:** derive TracingInsights session time as `lST + tel.time`, or absolute time as `parse(lSD) + tel.time`. Join to OpenF1 `date` with a nearest/as-of operation.

### Sampling and timestamp mismatch

- TracingInsights documents telemetry at approximately 3.7 Hz/~270 ms, but the three retained real lap files were irregular: median spacing was 0.125–0.130 s and effective density was about 7.56–7.67 samples/s. Treat the arrays as irregular observations, not a fixed-rate signal.
- OpenF1 documents `car_data` and `location` at about 3.7 Hz, `intervals` at roughly four-second updates, and lap `date_start` as approximate.
- Use backward/nearest as-of joins with feature-specific tolerances: approximately 0.5 seconds for car telemetry/location and up to 4–5 seconds for intervals. Record the actual age of every joined value (`source_age_ms`) and reject stale interval values outside the chosen limit.
- Never join floating timestamps by exact equality. The real PIA/LEC rival samples in the retained evidence differed by 0.009 seconds.

### Preventing duplicate or conflicting values

- Build one canonical window table keyed by `(session, driver_number, window_time)`; enrich it with source columns rather than concatenating source rows.
- Choose one authority per field family:
  - OpenF1: official interval seconds, gap to leader, raw DRS code/eligibility, live high-rate race position.
  - TracingInsights: full-field tyre/stint state, lap and sector timing, pit timestamps, rival car telemetry, corner geometry.
- Preserve provenance columns such as `interval_source`, `tyre_source`, and `source_age_ms`.
- When both sources provide speed/RPM/throttle/brake/DRS, keep suffixed raw columns during validation; do not average them. Resolve disagreements by the predefined authority and log a quality flag.
- Do not ingest TracingInsights mini-sector columns as an independent second source: the schema states that `ms1/ms2/ms3` are themselves enriched from OpenF1 when matching succeeds.
- Filter invalid/generated lap records before state joins. In the 2026 race, an extra driver folder contained one `ff1G=true` row with no lap time/position and was omitted from the 20-driver session aggregate.

## Recommended next step

Build a small, race-scoped **offline merger prototype** on train/validation races only:

1. Use OpenF1 as the canonical session/window timeline.
2. Add TracingInsights opponent tyre, stint, sectors, lap pace, pit-out, rival DRS-open, and telemetry-derived geometry features by as-of joins.
3. Produce coverage/staleness/conflict diagnostics before any retraining.
4. Retrain only after coverage is acceptable, following the existing frozen protocol.

Do not replace OpenF1 intervals or raw DRS with TracingInsights proxies; that would discard two signals the engineered candidate explicitly needs.

## Environment and disk audit

- Virtual environment: `.venv_tracinginsights`, **446 MB**.
- Packages: exactly the repository's `requirements.txt` and transitive dependencies; installation used `--no-cache-dir`.
- Retained reference/schema/sample data: **1.2 MB**.
- Temporary race data after cleanup: **0 bytes**.
- Peak single-race temporary data: **1.1 MB**, below the ~500 MB cap.
- The venv was used for every Python/network inspection command and is inactive after command completion.

Retained evidence is under `data/tracinginsights-reference/`. No full repository clone or bulk season download exists.
