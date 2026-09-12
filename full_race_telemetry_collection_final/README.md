# APEX-R continuous telemetry collection

Run from the application root:

```bash
source .venv_tracinginsights/bin/activate
python3 full_race_telemetry_collection/continuous_telemetry_pipeline.py --workspace /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application --output /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/full_race_telemetry_collection_final
python3 full_race_telemetry_collection/verify_telemetry_collection.py --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/full_race_telemetry_collection_final
python3 full_race_telemetry_collection/test_telemetry_collection.py --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/full_race_telemetry_collection_final
```

FastF1 3.8.3 was used for historical 2018–2022 Race telemetry. The five supplied archive ZIPs were inventoried before retrieval. The protected exclusion manifest was enforced before any FastF1 session request. The pipeline validated these three train races before expansion: 2018:Chinese Grand Prix:Race, 2019:Australian Grand Prix:Race, 2020:Hungarian Grand Prix:Race.

`telemetry_car.csv.gz` contains raw car stream fields: speed, throttle, brake, gear, RPM, DRS, Date and SessionTime. `telemetry_position.csv.gz` contains raw X/Y/Z/status fields and the same timestamp fields. There is no interpolation or implicit car-position join. `telemetry_coverage.csv` records driver-hours, timestamp intervals, gaps over five seconds and lap-context comparison.

The prior position-swap candidate set is preserved by SHA-256 in `metrics.json`; it remains unresolved. This package contains no overtake labels and no model result. Raw FastF1 cache payloads are cleaned after collection; source URLs are represented by provider/session metadata and retained cache digests.
