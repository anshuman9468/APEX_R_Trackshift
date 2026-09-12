const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const E = require('../dist/engine.js');

test('Phase 5 energy model exposes battery-side constraints and explicit infeasibility', () => {
  const low = { soc: 1, wet: 0, gapAhead: 1, gapBehind: 2, closing: 0, tyreAge: 10, aggression: .3, position: 7, lap: 20 };
  const value = E.energy(low, 'ATTACK', 1);
  assert.equal(E.CONFIG.accountingBoundary, 'battery-side');
  assert.equal(value.legal, false);
  assert.equal(value.constraintStatus, 'INFEASIBLE_REQUEST_LOGGED');
  assert.ok(value.violations.includes('Insufficient energy before recovery'));
  assert.ok(value.endMJ >= -1e-9 && value.endMJ <= E.CONFIG.capacityMJ + 1e-9);
});

test('Phase 5 branch snapshot and GNN advisory remain zero-influence', () => {
  const input = E.normalise({ scenarioId: 'patient', horizon: 3, seed: 7,
    telemetry: { disableLegacyPrior: true }, judgeAction: 'ATTACK' });
  const a = E.compare(input), b = E.compare(input);
  assert.deepEqual(a.branchContract.snapshot, input.state);
  assert.equal(a.branchContract.gnnInfluence, 'zero; advisory-only');
  assert.deepEqual(a.apex.sample.trace, b.apex.sample.trace);
  assert.equal(a.apex.sample.trace[1].dataModelProbability, null);
});

test('historical fixture has causal frames and no target outcome labels', () => {
  const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'phase5_dashboard', 'fixture', 'phase5_fixture.json')));
  assert.equal(fixture.selection.not_selected_by_outcome, true);
  assert.ok(fixture.replay.frames.length > 10);
  for (let i = 1; i < fixture.replay.frames.length; i++) assert.ok(fixture.replay.frames[i].t > fixture.replay.frames[i - 1].t);
  assert.equal(Object.hasOwn(fixture.decision, 'label'), false);
  assert.equal(Object.hasOwn(fixture.decision, 'outcome_status'), false);
  assert.equal(Object.hasOwn(fixture.inference_graph, 'label'), false);
  assert.equal(Object.hasOwn(fixture.inference_graph, 'outcome_status'), false);
  for (const frame of fixture.replay.frames) for (const car of frame.cars) {
    if (car.observed_car) assert.ok(car.car_source_session_time_sec <= frame.session_time_sec + 1e-9);
    if (car.observed_position) assert.ok(car.position_source_session_time_sec <= frame.session_time_sec + 1e-9);
  }
});
