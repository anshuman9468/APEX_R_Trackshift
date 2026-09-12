# Phase 3 passed-race reconciliation and repair audit

Generated: 2026-09-11T17:24:33.249059+00:00  
Version: `phase3-passed-races-reconciliation-v1`

## Scope

This is an offline correction companion for the already collected 37-race
telemetry subset. It does not train, score or calibrate a model, create
ground-truth overtaking labels, fetch remote sessions, or modify the original
Phase 3 files. The large telemetry archive is verified separately and is not
duplicated in this companion package.

## Verified reported state

* PASS races: **37 of 70**.
* Authoritative split for every PASS race: **{'train': 37}**.
* Car rows: **15,944,390**; position rows: **14,277,894**.
* Driver-race entries: car **740**, position **740**.
* Independent distinct driver-lap identities: car **40,837**, position **40,837**.
* The old **2,318** counter is reproduced as **2,318 distinct `(driver, lap)` number pairs with race identity omitted**. It is not a count of unique driver-lap observations. Corrected counts are **40,837 distinct `(race_id, driver, lap)` identities** and **2,236 distinct `(race_id, lap)` identities**. The name and scope were repaired in `coverage_count_reconciliation.csv`.
* Cached laptime boundary identities: **40,837** across **740** driver-race entries. Car and position driver-lap sets match the laptime boundary sets: **True / True**.
* Driver-hours, independently recomputed from finite per-driver-race time spans: car **1066.944765**, position **1066.939239**.
* Candidates in PASS races: **1,195**; review classifications: **620 OTHER_POSITION_CHANGE**, **575 UNRESOLVED**.
* Windows: **40,837**; original proxy outcomes: **823 positive**, **14,600 negative**, **25,414 unknown/censored**.

All measured values agree with the reported row and label totals. The material
coverage repair is the 2,318 label: the count is reproducible only as a
race-collapsed `(driver, lap)` number-pair count. It is not a driver-lap stream
count; the race-qualified driver-lap count is 40,837.

## Split audit and proposed development partition

The authoritative manifest maps all 37 PASS races by exact `race_id` plus its
year/event/session decomposition. Reconciliation mismatches: **0**. No default-to-train fallback was used.

All 37 PASS races are genuinely marked `train` in the authoritative manifest,
but this is not evidence of a representative random training sample. The
collection status shows successful PASS retrieval only within the training
assignment; validation and development-test races were not admitted to this
subset. This is collection/access bias, not a corrected model split.

The companion therefore adds a non-authoritative, chronology-only proposal:
**26 races development_train**, **6 development_validation**, and **5 development_test**. Boundaries were selected from chronology before outcome counts were read; whole races and linked events remain together. These are previously accessed development data, not an untouched final holdout.

See `proposed_split_support.csv` for post-boundary class support.

## Positive proxy lineage

The **823** positives are not 823 independent overtaking events. They are one-lap boundary proxy windows. **412** link to an inherited candidate ID and **411** are endpoint reversals found while generating windows across all eligible boundaries, outside the inherited 1,195-candidate list. The lineage companion reports **823** unique boundary proxy keys; windows per key are **{1: 823}**.

Across all positive lineage rows, the classifications are **137 OTHER_POSITION_CHANGE**, **275 UNRESOLVED** among the 412 inherited-candidate links, and **411 OUTSIDE_CANDIDATE_UNIVERSE** rows. A candidate classified
`OTHER_POSITION_CHANGE` or `UNRESOLVED` is not promoted to a verified overtake by
the proxy label. All **0** verified on-track passes remain **0**. There was no
human/video review. For 411 outside-list positives, the evidence is only the
boundary-order proxy itself, not an event record. Exact pass timestamps and
physical track order remain unavailable.

## Negative labels and eligibility

The **14,600** negatives all use the same basis:
`FIXED_PAIR_REMAINS_IN_CLASSIFIED_ORDER_AT_NEXT_LAP_BOUNDARY`. That means the
specified endpoint reversal was absent; it does **not** mean that no physical
pass, repass, lapping, retirement or timing correction occurred. Absence from
the candidate list was not used as a negative.

The one-second as-of audit found **3,179** car-feature
unmatched rows and **3,177** position-feature
unmatched rows out of **40,837** windows. The tolerance
remains exactly 1.0 second, backward-only, with no interpolation. No future
joins were found. Historical timestamp freshness is not proof of live
publication availability.

Endpoint-context flags are retained in `boundary_context_audit.csv` and
`negative_label_audit.csv.gz`. The original contract censors decision-lap pit
and race-control rows, but it does not automatically convert endpoint context
into a clean negative. This is reported as a conditional proxy population and
requires review before any claim about real overtakes.

Within the negative rows, **575** have a public pit row on the endpoint lap and
**5,818** have a race-control record on the endpoint lap. These are explicit
audit flags, not silently accepted evidence of “no overtake.”

## Coverage-count repair

`driver_lap_reconciliation.csv` compares each cached driver/lap boundary set
with each continuous telemetry stream. A covered lap means at least one finite
telemetry row with an integer lap; it is not a proof that every physical point
of the lap was sampled. The current sets reconcile exactly for the passed
races, while source-level timestamp gaps remain separately recorded in the
existing coverage artifacts.

## Storage and integrity

The original telemetry ZIP was checked with standard ZIP CRC/decompression
validation: **PERSISTENT_PACKAGE_VERIFIED**; bytes **495,988,397**;
SHA-256 `410922cd2226b54961d9d25601a90dc21b2fe5906f0bbb4badb23d4cd96ffd56`. It is stored at `/home/anshumandutta/Downloads/APEX-R_Devsez_Full_Application/phase3_passed_races/APEX-R_37_Passed_Races.zip`.
The companion package contains reports and small tables only; it does not
duplicate the large payload. Source hashes before/after this audit are equal:
**True**.

## Tests and decision

The focused test script checks exact split mapping, unknown-assignment handling,
positive lineage cardinality, duplicate grouping, negative semantics, lap-count
definitions, source immutability, no future as-of joins and protected-session
exclusion. The completed run passed **22/22** checks; the result is written to
`repair_tests.json`.

**Decision: READY FOR HISTORICAL PROXY-MODEL DEVELOPMENT WITH EXPLICIT PROXY
LIMITATIONS.** The data is telemetry-ready and suitable for a clearly named
boundary-order proxy experiment after the proposed chronology split is reviewed.
It is **not** verified-overtake-ready: verified on-track labels remain **0**.
Do not treat the proxy as real overtaking ground truth or connect it to the
strategy engine without a separate task contract and evidence review.
