# APEX-R GNN Proxy v1 — Model Card

## Model name

**APEX-R GNN Proxy v1** (`gnn_proxy_v1`), selected run seed 42, best validation checkpoint at epoch 28.

Its output must be labelled **“experimental boundary position-swap proxy signal.”**

## Task definition

The model scores whether a fixed attacker/target driver pair reverses classified order at the next defined lap boundary under the versioned historical proxy contract. It is a pair-specific boundary position-swap proxy. It is not a verified on-track-overtake model, an ATTACK-conditioned success model, or an energy-strategy optimiser.

## Dataset summary

- Source population: 37 telemetry-coverage PASS races from the audited historical development collection.
- Eligible supervised graphs: 15,404.
- Positive proxy windows: 822.
- Negative proxy windows: 14,582.
- Excluded windows: 19, because current attacker/target telemetry was missing.
- Chronology-only development split: 26 train races / 6 validation races / 5 development-test races.
- Split graph counts: 10,671 train, 2,533 validation, and 2,200 development-test.
- The development-test races had already been accessed during the experiment and are not an untouched final holdout.
- Protected session 11353 was excluded and was not accessed during this freeze.

Multiple windows may refer to related race situations, so 15,404 graphs are not 15,404 independent overtaking events. The labels are boundary-swap proxies, not verified pass events.

## Feature summary

Each graph represents cars as nodes and a fixed attacker/target pair. Inputs use observations available at or before the decision timestamp with a one-second backward-only as-of tolerance and a ten-second trailing lookback.

- Node features (18): speed, throttle, brake, gear, RPM, DRS state, car/position sample age, track-frame X/Y/Z, trailing speed delta/trend, explicit car/position/DRS missingness masks, and attacker/target role flags.
- Edge features (4): track-frame distance, relative speed, and source-position ages for both endpoints.
- Pair features (12): attacker-target speed, distance and trailing-speed deltas; telemetry ages; explicit missingness flags; and pair-position availability.
- Topology: directed edges to up to three nearest cars with finite track-frame coordinates; classified-position adjacency is not used.

Unresolved acceleration/relative-distance fields, identifiers that encourage memorisation, centred/future-aware transformations, real ERS state, battery state-of-health and battery temperature are not inputs.

## Training configuration

- Architecture: two GINEConv message-passing layers.
- Hidden width: 32.
- Dropout: 0.2.
- Decoder: attacker embedding + target embedding + pair features.
- Optimiser: Adam; learning rate 0.001; weight decay 0.0001.
- Loss: unweighted binary cross-entropy with logits.
- Batch size: 32.
- Maximum epochs: 100; early-stopping patience: 15 using validation average precision.
- Predeclared seeds: 17, 23 and 42.
- Preprocessing: train-only mean/standard deviation; non-finite raw values replaced by training means at transform time while explicit masks remain.
- Raw graph cache dtype: float64; model-boundary tensors: float32.
- Selected run: seed 42, epoch 28; run stopped after 43 epochs.
- Selection basis: highest validation average precision among the predeclared runs, not development-test performance.

## Selected checkpoint

- File: `gnn_seed_42_best.pt`
- Model class: `EdgeAwareGNN`
- Seed: 42.
- Best epoch: 28.
- Validation AP recorded in the checkpoint: 0.12760248880353492.
- SHA-256: `d5ce7258fa2ec62f953d516816497f1cdbd04b1b839e362ef447f79c4e618d14`.

## Recorded metrics

| Partition | Rows | Positives | AP | ROC-AUC | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development validation | 2,533 | 103 | 0.127602 | 0.689272 | 0.038754 | 0.166081 |
| Development test | 2,200 | 106 | 0.104338 | 0.700857 | 0.045965 | 0.190365 |

These metrics describe the fixed-pair boundary-swap proxy only. The sigmoid score has not been established as a calibrated overtake probability.

## Approved APEX-R usage

- Advisory display beside an approved historical replay decision.
- Input to an explicitly bounded future research experiment after validity checks.
- Diagnostic comparison with the no-ML simulator while the score has zero decision authority.
- CPU or CUDA inference using the unchanged graph schema and training-only preprocessing.

The APEX-R rule/energy simulator remains responsible for the final ATTACK/HOLD/HARVEST/DEFEND recommendation. Missing essential attacker/target evidence must produce an unavailable status, not fabricated inputs.

## Limitations

- The target is a timing-derived classified-order boundary swap, not a verified physical on-track pass.
- Proxy positives can include causes other than racing overtakes.
- Class imbalance is substantial: 822 of 15,404 eligible graphs are positive.
- Development-test support is only five races and was already accessed.
- The score is not calibrated for live use, attack benefit, energy benefit or pass probability.
- Historical measurement-time availability does not establish live publication latency.
- No real ERS/battery percentage, deployment map, battery SOH, battery temperature, private fuel load or pit-wall instruction is available.
- Validity outside the audited feature distribution is unknown.

## Prohibited claims

Do not call this output:

- a verified on-track-overtake probability;
- a calibrated probability or confidence percentage;
- proof that ATTACK will work or improve race position;
- proof of real-world strategy gain;
- a source of actual ERS, battery, fuel or proprietary pit-wall data;
- an untouched final-holdout result; or
- a production/deployment-ready racing model.

