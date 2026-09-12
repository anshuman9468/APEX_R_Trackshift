# APEX-R handoff for Team Devsez

VIPS-TC (GGSIPU)

## Ninety-second judge flow

1. Open the pit wall. Explain that the default replay and energy are simulated. Play a few seconds and capture a window. Ask the judge to choose the first energy action.
2. Click Compare futures. Show both action sequences, their shared seed, the energy chart, and the seeded race events. Explain the difference between one trajectory and the 96-trial summary.
3. Open Strategy lab. Reduce initial energy and change the rear gap. Show the recommendation respond. Point to an action blocked by insufficient energy.
4. Open Validation and run 24 scenarios. Report losses as well as wins. Explain that this demonstrates behaviour in a simplified simulator.
5. Open Decision log and export a decision. Restore its inputs and reproduce the outcome.

Use the scenario selector to demonstrate all four modes. The open door starts with an attack, under pressure with defence, running on reserve with harvest, and the patient attack with hold under the shipped parameters. These choices are computed; changing the parameters can change them.

Suggested closing line: "APEX-R turns an energy choice into a reproducible comparison of possible race futures. The engineer sees both the recommendation and the assumptions behind it."

## Three preparation days

Confirm permitted preparation and prebuilt-code disclosure with the organisers. This is a complete preparation build; disclose it if rules require. Do not represent it as code written during a restricted competition window.

| Day | Work | Required evidence |
| --- | --- | --- |
| 1 | Each member runs both the offline HTML and local server, reads the model, and understands ownership | Everyone can explain one full decision |
| 2 | Obtain one permitted historical segment; import and check timestamps and provenance; challenge model assumptions | A source-backed replay and a list of calibration gaps |
| 3 | Rehearse integration, run the test suite, record a fallback video, agree on a feature freeze | Offline demo on two laptops and a 90-second pitch |

## Four-member ownership

The final roster was not provided in this turn. Assign these roles to the four registered members without guessing names.

| Owner | Files | Responsibility |
| --- | --- | --- |
| Strategy lead | dist/engine.js, docs/MODEL.md | Understand assumptions, inspect policy changes, explain search and energy feasibility |
| Data and validation lead | dist/telemetry.js, scripts/fetch_openf1.py, tests | Prepare permitted telemetry, preserve provenance, evaluate calibration and edge cases |
| Frontend lead | dist/index.html, dist/styles.css, dist/app.js | Operate and refine the pit wall; check desktop/mobile layouts and all interactions |
| Backend and integration lead | backend/, run.py, requirements.txt | Run HTTP and FastAPI modes, verify SQLite and WebSockets, package and rehearse startup |

The team's integration lead owns the single working release. Everyone must understand which values are observed, inferred, or simulated.

## Twenty-four-hour event priorities

| Hours | Focus |
| --- | --- |
| 0-3 | Check final PS and constraints; adjust scope and data contracts |
| 3-8 | Verify replay ingestion and improve the energy/performance model where data supports it |
| 8-13 | Evaluate strategy changes on held-out cases; keep baseline comparisons fair |
| 13-17 | Fix integration, failure handling, and reproducibility issues |
| 17-20 | Make presentation and demo readable; gather measured evidence |
| 20-22 | Freeze features, verify both laptop setups, record backup demonstration |
| 22-24 | Rehearse, submit, and preserve the tested release |

If starting prebuilt code is prohibited, use this as a disclosed learning reference only and build the permitted event version according to the rules.

## Questions to prepare

- Is this a full F1 twin? No. It is a simplified race-strategy simulator with replay and an energy state.
- Is the overtake model trained? No. It uses explicit logistic heuristics. Training needs clean labels and held-out evaluation.
- Why not a racing game? The decision variable is energy strategy; steering and detailed vehicle dynamics are outside this scope.
- Is every action FIA legal? The shield checks configured model limits only. It is not a complete sporting or technical compliance implementation.
- Does a benchmark win prove real gains? No. It is evidence inside a synthetic, perturbed model family.
- Can the judge beat it? Yes. The approximate planner can lose against baselines, and individual seeded races vary.
- Does it require Internet or a GPU? The bundled demo requires neither. Downloading historical data and installing optional packages requires connectivity.
