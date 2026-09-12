# Phase 3 passed-race reconciliation companion

This directory contains an offline audit/repair companion for the existing
37-race passed-race export. It does not retrain or score a model, fetch remote
data, create verified overtake labels, or replace the original telemetry.

The original telemetry ZIP is retained separately at:

`phase3_passed_races/APEX-R_37_Passed_Races.zip`

The audit found that the old `2,318 driver-lap streams` label was wrong. The
old aggregate is 2,318 distinct `(driver, lap)` number pairs with race identity
omitted. The race-qualified continuous telemetry count is 40,837 distinct
`(race_id, driver, lap)` identities, matching the cached laptime boundary set;
there are 2,236 distinct `(race_id, lap)` identities.

## Reproduce

From the project root, using only the Python standard library:

```bash
python3 phase3_races_audit_reconciliation_v1/repair_audit.py \
  --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application \
  --phase3-dir phase3_passed_races \
  --output-dir phase3_races_audit_reconciliation_v1 \
  --package-path phase3_passed_races/APEX-R_37_Passed_Races.zip
python3 phase3_races_audit_reconciliation_v1/test_repairs.py \
  --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application \
  --output-dir phase3_races_audit_reconciliation_v1
python3 phase3_races_audit_reconciliation_v1/package_companion.py \
  --output-dir phase3_races_audit_reconciliation_v1 \
  --package-path phase3_races_audit_reconciliation_v1/APEX-R_Phase3_Passed_Races_Reconciliation_v1.zip
python3 phase3_races_audit_reconciliation_v1/test_repairs.py \
  --root /home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application \
  --output-dir phase3_races_audit_reconciliation_v1 \
  --package-path phase3_races_audit_reconciliation_v1/APEX-R_Phase3_Passed_Races_Reconciliation_v1.zip
```

The audit scans only race IDs admitted by the existing `PASS` status list.
It retains the original labels unchanged and emits per-window evidence and
context audits. The proposed chronology split is non-authoritative development
data; it is not a new untouched holdout.

`PACKAGE_VERIFICATION.json` is deliberately external to the companion ZIP so
the package checksum is not self-referential. `SHA256SUMS.txt` covers the
package members and is independently checked after fresh extraction.
