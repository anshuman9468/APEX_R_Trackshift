(function (root, factory) {
  const engine = factory();
  if (typeof module === 'object' && module.exports) module.exports = engine;
  else root.ApexEngine = engine;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const VERSION = '1.0.0';
  const ACTIONS = ['ATTACK', 'HOLD', 'HARVEST', 'DEFEND'];
  const CONFIG = Object.freeze({ version: 'phase5-demo-energy-v1', accountingBoundary: 'battery-side', capacityMJ: 4, reserveMJ: 0.12, maxDeployMJ: 1.45, maxHarvestMJ: 0.95, powerKW: 300, windowSeconds: 5, efficiency: 0.9, scoreWeights: { position: 6, reserve: 0.9, threat: 1.6, deploymentCost: 0.065 } });
  const MODES = Object.freeze({ ATTACK: { deploy: 1.35, recovery: 0.30, pace: 0.26, guard: 0.02 }, HOLD: { deploy: 0.36, recovery: 0.50, pace: 0, guard: 0.08 }, HARVEST: { deploy: 0.08, recovery: 0.88, pace: -0.22, guard: -0.04 }, DEFEND: { deploy: 0.90, recovery: 0.38, pace: 0.07, guard: 0.47 } });
  const SCENARIOS = [
    { id: 'patient', name: 'The patient attack', track: 'Strategy circuit', caption: 'A better window is one lap away.', state: { soc: 42, gapAhead: 0.68, gapBehind: 0.65, closing: 3, tyreAge: 18, wet: 0, aggression: 0.55, position: 7, lap: 37 }, windows: [0.20, 1.00, 0.60, 0.80, 0.50], recovery: [1.00, 1.10, 0.80, 1.00, 1.00] },
    { id: 'opportunity', name: 'The open door', track: 'Strategy circuit', caption: 'Close gap, full battery, clear straight.', state: { soc: 88, gapAhead: 0.22, gapBehind: 2.6, closing: 11, tyreAge: 6, wet: 0, aggression: 0.25, position: 5, lap: 22 }, windows: [1, 0.25, 0.30, 0.70, 0.45], recovery: [0.8, 1, 1.1, 1, 0.8] },
    { id: 'pressure', name: 'Under pressure', track: 'Strategy circuit', caption: 'A rival is close behind. Protect the position.', state: { soc: 48, gapAhead: 2.8, gapBehind: 0.16, closing: -2, tyreAge: 24, wet: 0, aggression: 0.9, position: 6, lap: 44 }, windows: [0.6, 0.45, 0.9, 0.5, 0.7], recovery: [0.9, 1.2, 0.9, 1, 1] },
    { id: 'recovery', name: 'Running on reserve', track: 'Strategy circuit', caption: 'Recover energy before spending it.', state: { soc: 9, gapAhead: 1.4, gapBehind: 2.4, closing: 1, tyreAge: 16, wet: 0, aggression: 0.4, position: 9, lap: 29 }, windows: [0.30, 0.75, 1, 0.6, 0.9], recovery: [1.05, 1, 0.8, 1, 1] },
    { id: 'rain', name: 'Changing conditions', track: 'Strategy circuit', caption: 'Reduced grip changes the value of deployment.', state: { soc: 62, gapAhead: 0.75, gapBehind: 0.85, closing: 4, tyreAge: 20, wet: 0.75, aggression: 0.6, position: 8, lap: 32 }, windows: [0.60, 0.75, 0.95, 0.55, 0.7], recovery: [0.7, 0.85, 0.9, 0.75, 0.9] }
  ];
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const sigmoid = (x) => 1 / (1 + Math.exp(-clamp(x, -30, 30)));
  const round = (x, places = 3) => Number(x.toFixed(places));
  const now = () => typeof performance === 'object' ? performance.now() : Date.now();
  function finite(value, name, low, high) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high) throw new Error(name + ' must be between ' + low + ' and ' + high);
    return value;
  }
  function normalise(input = {}) {
    if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Input must be an object');
    if (input.state != null && (typeof input.state !== 'object' || Array.isArray(input.state))) throw new Error('State must be an object');
    const scenario = SCENARIOS.find((s) => s.id === (input.scenarioId || 'patient'));
    if (!scenario) throw new Error('Unknown scenario');
    const state = { ...scenario.state, ...(input.state || {}) };
    const fields = { soc: [0, 100], gapAhead: [0, 10], gapBehind: [0, 10], closing: [-30, 30], tyreAge: [0, 60], wet: [0, 1], aggression: [0, 1], position: [2, 19], lap: [1, 80] };
    for (const [key, bounds] of Object.entries(fields)) finite(state[key], key, ...bounds);
    if (!Number.isInteger(state.position) || !Number.isInteger(state.lap)) throw new Error('Position and lap must be integers');
    const horizon = input.horizon ?? 3;
    finite(horizon, 'horizon', 2, 5);
    if (!Number.isInteger(horizon)) throw new Error('Horizon must be an integer');
    const seed = input.seed ?? 2026;
    finite(seed, 'seed', 0, 4294967295);
    if (!Number.isInteger(seed)) throw new Error('Seed must be an integer');
    const risk = finite(input.risk ?? 0.45, 'risk', 0, 1);
    const judgeAction = input.judgeAction || 'ATTACK';
    if (!ACTIONS.includes(judgeAction)) throw new Error('Unknown action');
    const windows = input.windows || scenario.windows;
    if (!Array.isArray(windows) || windows.length < horizon || windows.length > 5) throw new Error('Provide 2 to 5 window qualities covering the horizon');
    windows.forEach((w) => finite(w, 'window quality', 0, 1));
    const telemetry = input.telemetry && typeof input.telemetry === 'object' && !Array.isArray(input.telemetry) ? { ...input.telemetry } : {};
    for (const [key, bounds] of Object.entries({ speedKmh: [0, 450], throttlePct: [0, 100], brakePct: [0, 100], gear: [0, 8], intervalSec: [0, 10], gapToLeaderSec: [0, 100], closingRateSecPerMin: [-30, 30], trackTemperatureC: [-30, 70], rainfall: [0, 1], raceProgress: [0, 1], speedMean10s: [0, 450], speedDelta10s: [-300, 300], throttleMean10s: [0, 100], brakeFraction10s: [0, 1], rpmMean10s: [0, 20000], gearMean10s: [0, 8], drsOpenFraction10s: [0, 1] })) {
      if (telemetry[key] != null) finite(telemetry[key], key, ...bounds);
    }
    return { scenarioId: scenario.id, state, telemetry, horizon, seed, risk, judgeAction, windows: windows.slice(), recovery: scenario.recovery.slice() };
  }
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function energy(state, action, recoveryFactor = 1) {
    const m = MODES[action];
    if (!m) throw new Error('Unknown action');
    const start = state.soc / 100 * CONFIG.capacityMJ;
    const requestedDeploy = m.deploy;
    // Illegal requests are still represented transparently, but their
    // projected accounting is bounded at the reserve floor.  The caller
    // must inspect `legal`/`violations`; transition() rejects the request.
    const deploy = Math.min(requestedDeploy, Math.max(0, start - CONFIG.reserveMJ));
    const grossRecovery = Math.min(CONFIG.maxHarvestMJ, m.recovery * recoveryFactor * (1 - state.wet * 0.20));
    const capacityLeft = CONFIG.capacityMJ - (start - deploy);
    const acceptedRecovery = Math.max(0, Math.min(grossRecovery, capacityLeft));
    const violations = [];
    if (start - requestedDeploy < CONFIG.reserveMJ - 1e-9) violations.push('Insufficient energy before recovery');
    if (requestedDeploy > CONFIG.maxDeployMJ) violations.push('Deployment budget exceeded');
    if (requestedDeploy / CONFIG.windowSeconds * 1000 > CONFIG.powerKW) violations.push('Power ceiling exceeded');
    return { startMJ: start, requestedDeployMJ: requestedDeploy, deployMJ: deploy, harvestMJ: acceptedRecovery, spillMJ: grossRecovery - acceptedRecovery, endMJ: start - deploy + acceptedRecovery, legal: violations.length === 0, constraintStatus: violations.length ? 'INFEASIBLE_REQUEST_LOGGED' : 'FEASIBLE', violations };
  }
  function legalActions(state, recoveryFactor) { return ACTIONS.filter((a) => energy(state, a, recoveryFactor).legal); }
  function initialState(input) { return { ...input.state, gain: 0, deployed: 0, harvested: 0, threats: 0, passes: 0, losses: 0 }; }
  function chance(state, action, quality, model = {}, telemetry = {}) {
    const m = MODES[action];
    const grip = 1 - state.wet * 0.5;
    const boost = m.deploy * CONFIG.efficiency * grip;
    const attackLogit = -2.0 + quality * 2.6 + state.closing * 0.09 - state.gapAhead * 2.15 + boost * 1.8 - state.tyreAge * 0.012 - state.aggression * 0.9;
    const pPass = state.gain >= 1 ? 0 : clamp(sigmoid(attackLogit + (model.passBias || 0)), 0.002, 0.98);
    const energyAfter = Math.max(0, state.soc / 100 * CONFIG.capacityMJ - m.deploy);
    const threat = -1.8 + state.aggression * 1.5 - state.gapBehind * 1.65 + (1 - energyAfter / CONFIG.capacityMJ) * 1.45 - m.guard * 4 + state.wet * 0.7;
    const pLoss = state.gain <= -1 ? 0 : clamp(sigmoid(threat + (model.lossBias || 0)), 0.002, 0.95);
    return { pPass, pLoss, dataModelProbability: null, dataModelVersion: null };
  }
  function transition(state, action, input, step, draws, model = {}) {
    const e = energy(state, action, input.recovery[step] * (model.recoveryScale || 1));
    if (!e.legal) return { valid: false, energy: e };
    const q = input.windows[step];
    const p = chance(state, action, q, model, input.telemetry);
    let gain = state.gain, passed = false, lost = false;
    if (draws) {
      passed = draws[0] < p.pPass;
      lost = !passed && draws[1] < p.pLoss;
      gain = clamp(gain + Number(passed) - Number(lost), -1, 1);
    } else {
      gain = clamp(gain + p.pPass * (1 - gain) - (1 - p.pPass) * p.pLoss * (1 + gain), -1, 1);
    }
    const next = {
      ...state, soc: e.endMJ / CONFIG.capacityMJ * 100, gain,
      gapAhead: passed ? 3.4 : lost ? 0.55 : clamp(state.gapAhead - MODES[action].pace - state.closing * 0.007 + 0.10 * state.wet, 0.05, 10),
      gapBehind: passed ? 0.42 : lost ? 2.5 : clamp(state.gapBehind + MODES[action].guard - state.aggression * 0.12, 0.05, 10),
      tyreAge: Math.min(60, state.tyreAge + 1), lap: state.lap + 1,
      deployed: state.deployed + e.deployMJ, harvested: state.harvested + e.harvestMJ,
      threats: state.threats + (1 - p.pPass) * p.pLoss, passes: state.passes + Number(passed), losses: state.losses + Number(lost)
    };
    return { valid: true, next, energy: e, ...p, event: passed ? 'Overtake completed' : lost ? 'Position lost' : action === 'HARVEST' ? 'Energy recovered' : 'Position maintained' };
  }
  function score(state, input) {
    return state.gain * 6 + state.soc / 100 * 0.9 - input.risk * state.threats * 1.6 - state.deployed * 0.065;
  }
  function search(input) {
    const t0 = now();
    const candidates = [], rejections = [];
    let evaluated = 0;
    function visit(state, sequence, trace, step) {
      if (step === input.horizon) {
        candidates.push({ sequence, score: score(state, input), expectedGain: state.gain, endSoc: state.soc, deployedMJ: state.deployed, projectedThreat: state.threats / input.horizon, trace });
        evaluated++; return;
      }
      for (const action of ACTIONS) {
        const result = transition(state, action, input, step);
        if (!result.valid) { if (step === 0) rejections.push({ action, reasons: result.energy.violations }); continue; }
        visit(result.next, [...sequence, action], [...trace, { step: step + 1, action, soc: result.next.soc, expectedGain: result.next.gain, passProbability: result.pPass, lossProbability: result.pLoss, dataModelProbability: result.dataModelProbability, dataModelVersion: result.dataModelVersion, deployMJ: result.energy.deployMJ, harvestMJ: result.energy.harvestMJ }], step + 1);
      }
    }
    // A depleted battery can always recover, but recovery is never borrowed before deployment.
    let start = initialState(input);
    if (legalActions(start, input.recovery[0]).length === 0) {
      return { feasible: false, recommendation: 'NO FEASIBLE ACTION', candidates: [], rejections: ACTIONS.map(action => ({ action, reasons: energy(start, action).violations })), evaluated: 0, latencyMs: now() - t0 };
    }
    visit(start, [], [], 0);
    candidates.sort((a, b) => b.score - a.score || a.deployedMJ - b.deployedMJ || a.sequence.join().localeCompare(b.sequence.join()));
    const best = candidates[0];
    if (!best) return { feasible: false, recommendation: 'NO FEASIBLE ACTION', candidates: [], rejections, evaluated, latencyMs: now() - t0 };
    const byAction = ACTIONS.map(action => candidates.find(c => c.sequence[0] === action)).filter(Boolean).sort((a, b) => b.score - a.score);
    const runner = byAction.find(c => c.sequence[0] !== best.sequence[0]);
    const advantage = runner ? best.score - runner.score : 0;
    const explanation = {
      ATTACK: 'Deploy in this window. The available gap and energy make the immediate opportunity valuable over this horizon.',
      HOLD: 'Preserve flexibility in this window. The best sequence balances a later opportunity with remaining energy.',
      HARVEST: 'Recover energy first. Spending now weakens the feasible options across the next decision windows.',
      DEFEND: 'Protect the current position. Rear pressure makes defensive deployment valuable in this state.'
    }[best.sequence[0]];
    return { feasible: true, recommendation: best.sequence[0], best, alternatives: byAction, candidates: candidates.slice(0, 8), rejections, evaluated, scoreMargin: advantage, explanation, latencyMs: now() - t0 };
  }
  function fallback(state, input, step) {
    const legal = legalActions(state, input.recovery[step]);
    if (!legal.length) return null;
    return legal.includes('HARVEST') ? 'HARVEST' : legal[0];
  }
  function rollout(input, sequence, seed, model = {}) {
    const rand = rng(seed);
    let state = initialState(input), rejected = 0, violations = 0;
    const trace = [{ step: 0, soc: state.soc, position: state.position, action: 'START', event: 'Shared starting state' }];
    for (let step = 0; step < input.horizon; step++) {
      // Fixed random draw count per window gives every strategy identical latent conditions.
      const draws = [rand(), rand()];
      let action = sequence[step];
      let result = transition(state, action, input, step, draws, model);
      let blocked = null;
      if (!result.valid) {
        rejected++; blocked = action;
        action = fallback(state, input, step);
        if (action) result = transition(state, action, input, step, draws, model);
        if (!action || !result.valid) {
          // Emergency zero-deployment recovery, outside the four strategy modes.
          const recovery = Math.min(CONFIG.maxHarvestMJ, CONFIG.maxHarvestMJ * (model.recoveryScale || 1), CONFIG.capacityMJ * (1 - state.soc / 100));
      const threat = chance(state, 'HARVEST', input.windows[step], model, input.telemetry).pLoss;
          const lost = draws[1] < threat && state.gain > -1;
          state = { ...state, gain: Math.max(-1, state.gain - Number(lost)), losses: state.losses + Number(lost), threats: state.threats + threat, soc: state.soc + recovery / CONFIG.capacityMJ * 100, harvested: state.harvested + recovery, lap: state.lap + 1 };
          trace.push({ step: step + 1, action: 'RECOVER', requestedAction: blocked, soc: state.soc, position: input.state.position - state.gain, event: 'Zero-deployment recovery' + (lost ? '; position lost' : '; position maintained'), deployMJ: 0, harvestMJ: recovery });
          continue;
        }
      }
      state = result.next;
      if (state.soc < -1e-8 || state.soc > 100 + 1e-8) violations++;
      trace.push({ step: step + 1, action, requestedAction: blocked, soc: round(state.soc), position: input.state.position - state.gain, event: blocked ? blocked + ' blocked; ' + result.event.toLowerCase() : result.event, passProbability: result.pPass, lossProbability: result.pLoss, dataModelProbability: result.dataModelProbability, dataModelVersion: result.dataModelVersion, deployMJ: result.energy.deployMJ, harvestMJ: result.energy.harvestMJ });
    }
    return { finalPosition: input.state.position - state.gain, gain: state.gain, endSoc: state.soc, deployedMJ: state.deployed, harvestedMJ: state.harvested, passes: state.passes, losses: state.losses, rejected, violations, score: score(state, input), trace };
  }
  function ensemble(input, sequence, trials = 96, model = {}, salt = 100000) {
    let gain = 0, energySum = 0, deployed = 0, passed = 0, losses = 0, rejected = 0, violations = 0, sumSquare = 0, value = 0;
    const positions = {};
    for (let i = 0; i < trials; i++) {
      const r = rollout(input, sequence, (input.seed + salt + Math.imul(i + 1, 7919)) >>> 0, model);
      gain += r.gain; sumSquare += r.gain * r.gain; energySum += r.endSoc; deployed += r.deployedMJ; passed += Number(r.passes > 0); losses += Number(r.losses > 0); rejected += r.rejected; violations += r.violations; value += r.score;
      positions[r.finalPosition] = (positions[r.finalPosition] || 0) + 1;
    }
    const mean = gain / trials;
    const standardError = Math.sqrt(Math.max(0, (sumSquare - trials * mean * mean) / Math.max(1, trials - 1)) / trials);
    const meanDeploy = deployed / trials;
    return { trials, expectedGain: mean, expectedPosition: input.state.position - mean, endSoc: energySum / trials, deployedMJ: meanDeploy, passRate: passed / trials, lossRate: losses / trials, blockedActions: rejected, executedViolations: violations, score: value / trials, positionValuePerMJ: meanDeploy > 0 ? mean / meanDeploy : null, gainInterval: [Math.max(-1, mean - 1.96 * standardError), Math.min(1, mean + 1.96 * standardError)], positions };
  }
  function compare(raw = {}) {
    const start = now(), input = normalise(raw), optimisation = search(input);
    const sequence = optimisation.feasible ? optimisation.best.sequence : Array(input.horizon).fill('HARVEST');
    const judgeSequence = [input.judgeAction, ...Array(input.horizon - 1).fill('HOLD')];
    return { version: VERSION, input, optimisation, branchContract: { snapshot: { ...input.state }, horizonLaps: input.horizon, identicalSeed: input.seed, recordedFutureTelemetryIsReferenceOnly: true, gnnInfluence: 'zero; advisory-only' }, apex: { sequence, sample: rollout(input, sequence, input.seed), summary: ensemble(input, sequence) }, judge: { sequence: judgeSequence, sample: rollout(input, judgeSequence, input.seed), summary: ensemble(input, judgeSequence) }, computeMs: now() - start };
  }
  function threshold(input) {
    let state = initialState(input); const sequence = [];
    for (let t = 0; t < input.horizon; t++) {
      let action = state.soc < 30 ? 'HARVEST' : state.gapBehind < 0.5 ? 'DEFEND' : state.gapAhead < 0.8 && state.soc > 45 ? 'ATTACK' : 'HOLD';
      if (!energy(state, action, input.recovery[t]).legal) action = fallback(state, input, t) || 'HARVEST';
      sequence.push(action);
      const next = transition(state, action, input, t);
      if (next.valid) state = next.next;
    }
    return sequence;
  }
  function validate(options = {}) {
    if (!options || typeof options !== 'object' || Array.isArray(options)) throw new Error('Validation options must be an object');
    const count = options.count ?? 24, seed = options.seed ?? 2026;
    if (!Number.isInteger(count) || count < 5 || count > 100) throw new Error('Scenario count must be 5 to 100');
    finite(seed, 'seed', 0, 4294967295);
    if (!Number.isInteger(seed)) throw new Error('Seed must be an integer');
    const start = now(), rand = rng(seed), rows = [], details = [];
    const totals = { APEX: [], 'Attack now': [], Conserve: [], Threshold: [] };
    let wins = 0, ties = 0, losses = 0;
    for (let i = 0; i < count; i++) {
      const scenario = SCENARIOS[i % SCENARIOS.length];
      const raw = { scenarioId: scenario.id, horizon: 3, seed: (seed + i * 104729) >>> 0, state: { soc: 8 + rand() * 88, gapAhead: 0.15 + rand() * 2.8, gapBehind: 0.1 + rand() * 3.0, aggression: rand(), wet: rand() < 0.25 ? 0.65 : 0, closing: -4 + rand() * 16, tyreAge: Math.floor(rand() * 35) } };
      const input = normalise(raw), result = search(input);
      const strategies = { APEX: result.feasible ? result.best.sequence : Array(3).fill('HARVEST'), 'Attack now': ['ATTACK', 'HOLD', 'HOLD'], Conserve: ['HARVEST', 'HARVEST', 'HOLD'], Threshold: threshold(input) };
      // Perturb held-out outcomes; the planner never sees these environment offsets.
      const environment = { passBias: (rand() - 0.5) * 0.9, lossBias: (rand() - 0.5) * 0.8, recoveryScale: 0.85 + rand() * 0.3 };
      const summaries = {};
      for (const [name, sequence] of Object.entries(strategies)) { summaries[name] = ensemble(input, sequence, 48, environment, 700000); totals[name].push(summaries[name]); }
      const diff = summaries.APEX.score - summaries.Threshold.score;
      if (Math.abs(diff) < 0.01) ties++; else if (diff > 0) wins++; else losses++;
      details.push({ scenario: i + 1, scenarioId: scenario.id, input, environment, sequence: strategies.APEX, apexGain: summaries.APEX.expectedGain, baselineGain: summaries.Threshold.expectedGain, scoreDifference: diff });
    }
    for (const [name, values] of Object.entries(totals)) {
      const mean = (key) => values.reduce((s, r) => s + r[key], 0) / count;
      rows.push({ name, gain: mean('expectedGain'), endSoc: mean('endSoc'), passRate: mean('passRate'), lossRate: mean('lossRate'), deployedMJ: mean('deployedMJ'), score: mean('score'), blockedActions: values.reduce((s, r) => s + r.blockedActions, 0), executedViolations: values.reduce((s, r) => s + r.executedViolations, 0) });
    }
    return { version: VERSION, count, seed, trialsPerScenario: 48, comparisons: count * 48 * 4, wins, ties, losses, rows, details, elapsedMs: now() - start, methodology: 'Synthetic scenarios with a historical OpenF1 overtake prior; 48 paired draws per strategy and scenario; outcome-model perturbations hidden from the planner. Wins compare simulated utility with the threshold baseline. Not real-race validation.' };
  }
  return { VERSION, ACTIONS, CONFIG, MODES, SCENARIOS, normalise, rng, energy, legalActions, chance, transition, initialState, search, rollout, ensemble, compare, validate };
});
