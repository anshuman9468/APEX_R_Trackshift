# APEX-R continuous telemetry collection report

Generated: 2026-09-11T14:24:43.048733+00:00

## Outcome

Continuous car and position streams were inventoried and collected in a separate package. The source seasonal archives and the prior lap-context package were preserved. No labels, model work, replay, decision-engine change, or holdout scoring was performed.

The three-race validation gate was **PASS** before expansion. The validation races were: 2018:Chinese Grand Prix:Race, 2019:Australian Grand Prix:Race, 2020:Hungarian Grand Prix:Race. They were selected from approved development races and remain identified in `validation_three_races.csv`.

## Measured coverage

* Approved manifest: **70 races**, seasons 2018–2022, Race session.
* Supplied selected-lap archive: **1,027,303 observations**, **1,332 selected driver-lap streams**, **67 missing expected driver-race entries**.
* FastF1 exported car observations: **17,031,555**; position observations: **15,484,853**.
* Driver-hours covered: car **1140.458**, position **1140.453**.
* Race-level telemetry coverage pass: **37**; incomplete/unverified: **33**.
* FastF1 source retrievals: **70 new**, **0 pre-existing-cache reuse**, **30 failed**. Temporary raw caches were cleaned after each race.

## Split coverage

The development-test races remain identified as development data and were not used for model scoring in this task.

| Split | Approved races | Telemetry pass races | Car rows | Position rows | Car driver-hours | Position driver-hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 42 | 37 | 17,031,555 | 15,484,853 | 1140.458 | 1140.453 |
| validation | 14 | 0 | 0 | 0 | 0.000 | 0.000 |
| development_test | 14 | 0 | 0 | 0 | 0.000 | 0.000 |

## Timestamp and fields

The retained car table contains speed (km/h), throttle (%), brake, gear, RPM and DRS. The retained position table contains X/Y/Z coordinates and source status where available. FastF1 `Date` is retained as returned (timezone-naive in the observed v3.8.3 runtime); `SessionTime` is retained as session-relative seconds. These are measurement/source timestamps, not proof of live publication latency.

No interpolation or nearest-time merge was used between car and position streams. Lap numbers are an explicit interval lookup using FastF1 `LapStartTime` and `Time`, with the method recorded in each row. Any timestamp gaps greater than five seconds are reported per driver in `telemetry_coverage.csv`; they remain gaps, not imputed samples.

## Events and labels

The prior **2,095** position-swap candidates were preserved by hash and remain unresolved. This task produced **0 verified overtake events** and did not relabel candidates. FastF1 telemetry supplies measurements but not a verified historical overtake-event feed sufficient to turn every boundary swap into an overtake.

OpenF1 was not queried for 2018–2022 outcome data. Its documented free historical coverage begins in 2023; a separate 2023+ collection should resolve session keys and exclude protected holdouts before any request, then use timestamped OpenF1 overtake/position evidence. Do not mix that future collection into this 2018–2022 package.

## Integrity

Source archive immutability: **PASS**. Prior lap-context package immutability: **PASS**. ZIP DEFLATE, CRC, extracted checksums and row counts: **PASS**.

## Limits

A telemetry coverage pass means raw FastF1 car and position streams were available for the participating driver and covered the observed laptime lap span. It does not prove every physical track sample is present, prove live delivery latency, or verify an overtake event. The approved data can proceed to a separate event-evidence review, not directly to verified-overtake model training.

Exact file bytes and ZIP SHA-256 are in `package_verification.json`; raw FastF1 cache payloads were not retained in the package.
