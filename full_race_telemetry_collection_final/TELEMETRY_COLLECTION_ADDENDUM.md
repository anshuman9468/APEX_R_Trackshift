# Continuous telemetry collection addendum

This addendum is generated from the completed collection outputs; it does not fetch, relabel, train, or alter source telemetry.

## Result

* Approved scope: 70 Race sessions across 2018–2022; expected driver-race rows: 1,399.
* Supplied selected-lap archive: 1,027,303 rows, 1,332 selected driver-lap streams, 67 expected driver-race entries absent.
* FastF1 retrieval attempts: 70; successful sessions: 40; failed sessions: 30; pre-existing cache reuse: 0; new successful source sessions: 40.
* Race-level telemetry pass: 37; incomplete/unverified: 33.
* Raw exported rows: car 17,031,555; position 15,484,853.
* Driver-hours: car 1140.457932; position 1140.452771.
* Missing intervals >5 seconds: car 39 totaling 1536.120 seconds; position 0 totaling 0.000 seconds.
* Duplicate timestamps: car 0; position 0. Non-monotonic timestamps: car 0; position 0.

## Three-race gate

The gate was PASS before expansion. The selected train races were: 2018:Chinese Grand Prix:Race, 2019:Australian Grand Prix:Race, 2020:Hungarian Grand Prix:Race.

| Race | car rows | position rows | expected drivers | drivers with telemetry pass | status |
| --- | ---: | ---: | ---: | ---: | --- |
| 2018:Chinese Grand Prix:Race | 475,304 | 387,297 | 20 | 20 | PASS |
| 2019:Australian Grand Prix:Race | 398,270 | 317,300 | 20 | 20 | PASS |
| 2020:Hungarian Grand Prix:Race | 460,025 | 509,488 | 20 | 20 | PASS |

## Split and source-access status

| Split | races | successful sessions | telemetry pass races | failed retrievals | car rows | position rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 42 | 40 | 37 | 2 | 17,031,555 | 15,484,853 |
| validation | 14 | 0 | 0 | 14 | 0 | 0 |
| development_test | 14 | 0 | 0 | 14 | 0 | 0 |

FastF1 failure categories: {'DataNotLoadedError': 1, 'RateLimitExceededError': 29}. The dominant blocker was the provider/API 500-calls-per-hour rate limit; these sessions are missing, not confirmed to have no telemetry. The 2018 Italian session also returned DataNotLoadedError.
The development-test races remain marked as development_test in every status and inventory table; no model scoring or replay was performed.

## Semantics and limits

`telemetry_car.csv.gz` retains raw FastF1 car_data fields: RPM, Speed (km/h), nGear, Throttle (%), Brake, DRS, Date, Time and SessionTime. `telemetry_position.csv.gz` retains raw position-data Status, X, Y and Z plus Date, Time and SessionTime. FastF1 Date was returned timezone-naive in this runtime; SessionTime is session-relative. These are measurement timestamps, not proof of live publication latency or UTC alignment.
No interpolation or nearest-time car/position merge was used. Lap numbers are interval assignments from FastF1 LapStartTime/Time and are not a substitute for a physical-lap completeness proof. Missing intervals remain missing.
The prior 2,095 position-swap candidates remain unresolved. Verified overtake events remain 0; no labels were created.

## Source limitation and next collection

FastF1 supplied documented historical car/position access for the successful sessions, but this run cannot claim complete 2018–2022 coverage because 33 race sessions remain incomplete/unverified. The minimum repair is to resume the 30 failed requests after the provider rate-limit window (or use an equivalent documented permitted source), with the same pre-fetch protected-session guard; do not reinterpret missing requests as no data.
If 2018–2022 cannot be completed, propose a separate 2023+ collection using OpenF1’s documented historical endpoints, excluding the protected session before any request. Keep that collection separate from this 2018–2022 package and use it for timestamped event evidence only where session, driver and UTC alignment are actually verified.

Final ZIP: APEX-R_Full_Race_Telemetry_Collection.zip (530,147,346 bytes) if present. The ZIP’s own `package_verification.json` is external to the ZIP; `driver_telemetry_coverage.*` and this addendum are post-run audit companions and are not silently represented as ZIP members.
