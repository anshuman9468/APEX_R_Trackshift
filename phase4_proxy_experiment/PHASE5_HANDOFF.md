# Phase 5 Handoff

Phase 4 produced an exploratory proxy-model result only. The selected model
must not be connected to the live decision engine until the target is upgraded
to verified timestamped on-track events or the product explicitly accepts the
proxy limitation.

Required isolated Phase 5 evaluations:

1. Energy-feasibility: verify that every proposed action respects the modeled
   energy budget under conservative bounds; no private ERS value may be
   implied.
2. Strategy baseline: compare against fixed HOLD, fixed HARVEST, fixed
   ATTACK and a simple rule baseline on the same simulated initial states.
3. No-ML engine: run the optimizer without model probabilities to measure
   what the model contributes.
4. Stress: test missing context, stale observations, censored targets,
   safety-car transitions, pit windows, low energy and race-end censoring.
5. Latency: measure ingestion, feature construction, prediction and search
   latency using a replay clock; historical timestamps are not proof of live
   delivery latency.

All Phase 5 numbers must be labelled simulation or replay results. Do not
present them as observed racing improvements. Keep the protected final
holdout outside development until its separately frozen evaluation protocol.
