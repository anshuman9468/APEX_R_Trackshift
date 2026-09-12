# Passed-race subset

Run from the project root:

```bash
python phase3_passed_races/phase3_passed_races_pipeline.py \
  --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application \
  --output-dir phase3_passed_races \
  --package-path /tmp/APEX-R_37_Passed_Races.zip
python phase3_passed_races/augment_event_telemetry.py --output-dir phase3_passed_races
python phase3_passed_races/test_passed_races.py --output-dir phase3_passed_races
```

The pipeline is offline and derives the selected race list from the actual collection status file. It does not touch non-PASS races, fetch data, train models or run replay. Large streams are written temporarily and included in the ZIP; the temporary stream directory is removed after packaging.

`telemetry_car_37_passed.csv.gz` and `telemetry_position_37_passed.csv.gz` are separate source streams. `proxy_feature_view_37.csv.gz` is causal/as-of only; `proxy_labels_37.csv.gz` is a classified boundary-order proxy. `event_review_37.csv` is automated and unreviewed by a person.
