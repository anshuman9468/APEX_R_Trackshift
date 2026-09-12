# APEX-R model specification

Version 1.0.0. Team Devsez, VIPS-TC (GGSIPU).

## Scope and provenance

This is a replay-driven strategy system with a simplified ERS model and a real-data overtake-opportunity prior. It is not a calibrated vehicle digital twin. The default circuit, telemetry, opponent gaps, battery and outcomes are synthetic. The enriched training pipeline uses historical OpenF1 interval, position, lap, stint, weather, car-data, location, pit, race-control and overtake records; it predicts a cleaned observed overtake within the next 60 seconds. Historical telemetry can be imported for playback; import does not create private ERS measurements or automatically recalibrate the energy model.

One decision window represents one lap. Each horizon contains two to five windows. This is a strategic abstraction, not a time-resolved vehicle dynamics integration. Only the nearest net position gain or loss is represented, bounded to [-1, +1]. The UI calls the three fictional nearby cars DEV, RIV and CHS; these are not real driver records.

The four modes are ATTACK, HOLD, HARVEST and DEFEND. All parameters below are illustrative design assumptions, not claims about the current F1 regulations. Historical DRS fields are not treated as 2026 eligibility or used to assert regulatory compliance.

## Energy accounting

All energy quantities are in MJ. SOC is a percentage of an illustrative 4.00 MJ capacity.

E_next = E_start - E_deploy + E_recovered

Deployment must be available before recovery. A normal action must leave at least 0.12 MJ immediately after deployment. Recovery is capped at 0.95 MJ and available battery headroom. Surplus recovery is reported as spill. Deployment is capped at 1.45 MJ per model window and 300 kW across a five-second deployment burst. The rest of the lap is abstracted.

| Mode | Deployment MJ | Base recovery MJ | Gap-change proxy | Defensive coefficient |
| --- | ---: | ---: | ---: | ---: |
| ATTACK | 1.35 | 0.30 | 0.26 | 0.02 |
| HOLD | 0.36 | 0.50 | 0.00 | 0.08 |
| HARVEST | 0.08 | 0.88 | -0.22 | -0.04 |
| DEFEND | 0.90 | 0.38 | 0.07 | 0.47 |

Recovery is adjusted by the scenario's recovery factor and wetness. Invalid actions are rejected, then replaced by a feasible recovery mode. At extreme depletion, an emergency RECOVER step deploys zero energy and can still lose a position. Rejected requests and executed constraint violations are separate metrics.

## Prediction and search

Pass and loss probabilities use transparent logistic functions in `dist/engine.js`. A bundled OpenF1-trained prediction prior adjusts the pass probability using interval, leader gap, closing rate, position, tyre age, weather, race progress and ten-second rolling car telemetry (speed, throttle, brake, RPM, gear and DRS). Pit and neutralisation windows are excluded while constructing labels; location is retained for replay provenance rather than used as an arbitrary-coordinate feature. The active model is XGBoost, selected against the existing browser model by validation ROC-AUC; the learned model predicts an observed overtake within 60 seconds. The loss probability, energy accounting and action utility remain modelled components. The model is not a private team strategy model and must be retrained and evaluated when the data distribution changes.

Search exhaustively enumerates feasible action sequences, at most 4^5 = 1,024 before pruning. A deterministic expected-state surrogate ranks sequences. It approximates the nonlinear outcome distribution and can disagree with Monte Carlo evaluation.

Utility = 6 * expected_position_gain + 0.9 * remaining_charge_fraction - 1.6 * risk_aversion * accumulated_threat - 0.065 * deployed_MJ

The coefficients are manually chosen and transparent. Risk aversion ranges from 0 to 1. The optimiser is an open-loop finite-horizon planner: it does not observe outcomes and re-plan inside a single comparison. Engineers can capture another state and run it again. Do not describe it as a validated closed-loop MPC implementation.

Position value per MJ is a secondary reporting metric: mean simulated net position gain divided by mean deployed MJ. The UI uses MJ for readable numbers. It is not the optimisation objective and is not evidence of scientific novelty.

## Counterfactual evaluation

The judge's first selected action is followed by HOLD for the rest of the horizon. APEX-R selects the whole sequence. Both receive the same initial state and two random draws per window. Draw consumption is fixed even when an action is blocked, preventing accidental differences in random conditions.

The seed controls the displayed single trajectory. Summary metrics use 96 paired simulated trajectories from a separate, deterministic seed family. They report expected position, any pass, any position loss, ending energy and energy use. A position can be lost after an earlier pass, so a pass rate is not the probability of retaining that position.

The displayed 95% interval is a normal approximation using sample standard error for net position gain. It describes Monte Carlo variability within this model. It is not a confidence interval for real-world accuracy, and it can be misleading near boundaries or with unmodelled systematic errors.

## Benchmark

The suite generates 12, 24 or 60 synthetic scenarios in the UI; the API supports 5 to 100. Each samples energy, gaps, closing speed, tyre age, opponent aggression and occasional wetness. Each policy receives 48 paired trials:

- APEX-R: the feasible sequence with the largest surrogate utility.
- Attack now: ATTACK followed by HOLD, subject to the same shield.
- Conserve: HARVEST, HARVEST, HOLD.
- Threshold: energy/gap thresholds generate a sequence from expected states.

Before execution, pass and loss logits and recovery efficiency are perturbed. These offsets are hidden from the planner. Wins, ties and losses compare mean simulated utility with the threshold policy, using a 0.01 utility tolerance for a tie. Position gain is shown separately. These are stress tests within a related model family, not independent real-race validation.

## Replay coupling

The bundled 90-second replay uses a schematic closed path and synthetic speed, throttle, gear and brake samples. Its small SOC drift is illustrative; it is not a calibrated integration of the strategic per-lap model. Capture window freezes the selected SOC and derives a demo opportunity quality from synthetic speed. Gaps remain the explicit scenario inputs.

Imported OpenF1-style telemetry preserves observed speed, throttle, brake and gear when present. Optional location samples are scaled for visualisation. Missing fields are shown as unknown, not invented. Without location, animation follows the schematic circuit. Energy and rivals remain simulated. The trained overtake prior uses scenario gaps and state fields when making a strategy estimate; it is not a live team model. An imported file's origin is user-supplied and is not independently authenticated.

## Sources and further calibration

- OpenF1 schema and historical-data access: https://openf1.org/docs/
- FastF1 data and telemetry tooling: https://docs.fastf1.dev/
- FastAPI WebSocket interface: https://fastapi.tiangolo.com/advanced/websockets/

The training artifact was produced from four historical OpenF1 race sessions, with the final session held out for evaluation. Use `scripts/train_model.py` on a permitted connection to reproduce or extend it; `scripts/fetch_openf1.py` remains available for a short, user-selected playback import. OpenF1 is an unofficial public data service, not a team telemetry connection.

Further real-world validation requires clean non-pit overtake labels, held-out sessions, probability calibration and a documented mapping from power deployment to time gained. This build makes no claim to have completed those tasks.
