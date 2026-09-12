'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
require('../dist/apex-model.js');
const E=require('../dist/engine.js');
const T=require('../dist/telemetry.js');

test('energy is conserved and never borrows recovery before deployment',()=>{
  for(const soc of [0,3,5,9,25,42,75,100])for(const action of E.ACTIONS){
    const r=E.energy({...E.SCENARIOS[0].state,soc},action);
    assert.ok(Math.abs(r.endMJ-(r.startMJ-r.deployMJ+r.harvestMJ))<1e-10);
    if(r.legal){assert.ok(r.startMJ-r.deployMJ>=E.CONFIG.reserveMJ-1e-9);assert.ok(r.endMJ<=E.CONFIG.capacityMJ+1e-9);}
  }
  assert.equal(E.energy({...E.SCENARIOS[0].state,soc:9},'ATTACK').legal,false);
});

test('capacity spill is explicit and never exceeds the battery capacity',()=>{
  const r=E.energy({...E.SCENARIOS[0].state,soc:100},'HARVEST',2);
  assert.equal(r.endMJ,E.CONFIG.capacityMJ);assert.ok(r.spillMJ>0);
});

test('all four modes emerge from different input states',()=>{
  const choices=new Set(E.SCENARIOS.map(s=>E.search(E.normalise({scenarioId:s.id})).recommendation));
  for(const action of E.ACTIONS)assert.ok(choices.has(action),action+' should be useful in at least one scenario');
});

test('optimisation depends on state and does not mutate shared input',()=>{
  const initial=E.normalise({scenarioId:'opportunity'}),before=JSON.stringify(initial);
  const full=E.search(initial),empty=E.search(E.normalise({scenarioId:'opportunity',state:{soc:9}}));
  assert.notEqual(full.recommendation,empty.recommendation);assert.equal(JSON.stringify(initial),before);
});

test('identical seeds reproduce outcomes; identical policies have identical paired outcomes',()=>{
  const input=E.normalise({scenarioId:'patient'}),seq=['HOLD','DEFEND','DEFEND'];
  assert.deepEqual(E.rollout(input,seq,2026),E.rollout(input,seq,2026));
  assert.deepEqual(E.ensemble(input,seq),E.ensemble(input,seq));
  const observed=new Set(Array.from({length:40},(_,seed)=>E.rollout(input,seq,seed).finalPosition));
  assert.ok(observed.size>1,'future outcome must respond to stochastic draws');
});

test('emergency recovery rejects unavailable modes and executes no energy violations',()=>{
  const c=E.compare({state:{soc:0},horizon:5});
  assert.equal(c.optimisation.feasible,false);assert.equal(c.apex.sample.trace[1].action,'RECOVER');
  assert.equal(c.apex.sample.trace[1].deployMJ,0);assert.ok(c.apex.sample.rejected>0);
  for(const branch of [c.apex,c.judge]){assert.equal(branch.summary.executedViolations,0);for(const step of branch.sample.trace)assert.ok(step.soc>=0&&step.soc<=100);}
});

test('chosen sequence is highest ranked and trace has feasible budgets',()=>{
  for(const scenario of E.SCENARIOS)for(const horizon of [2,3,5]){
    const input=E.normalise({scenarioId:scenario.id,horizon});const result=E.search(input);
    assert.ok(result.feasible);assert.equal(result.best.sequence.length,horizon);assert.ok(result.evaluated<=4**horizon);
    assert.ok(result.candidates.every(c=>c.score<=result.best.score+1e-10));
    assert.equal(result.alternatives[0].score,result.best.score);
    if(result.alternatives.length>1)assert.ok(Math.abs(result.scoreMargin-(result.best.score-result.alternatives[1].score))<1e-10);
    assert.ok(result.best.trace.every(t=>t.soc>=0&&t.soc<=100));
  }
});

test('model evaluation returns real wins, ties, losses and bounded probabilities',()=>{
  const result=E.validate({count:24,seed:2026});
  assert.equal(result.wins+result.ties+result.losses,24);assert.equal(result.rows.length,4);assert.equal(result.details.length,24);
  assert.ok(result.losses>0,'benchmark must preserve adverse cases');
  for(const row of result.rows){assert.ok(row.passRate>=0&&row.passRate<=1);assert.ok(row.lossRate>=0&&row.lossRate<=1);assert.equal(row.executedViolations,0);}
});

test('invalid and non-finite input is rejected at the engine boundary',()=>{
  for(const state of [{soc:NaN},{soc:101},{gapAhead:-1},{wet:2},{position:4.5}])assert.throws(()=>E.compare({state}));
  for(const raw of [null,[],{state:'bad'},{horizon:2.5},{horizon:6},{seed:-1},{seed:1.5},{windows:[1]},{judgeAction:'FLY'},{scenarioId:'missing'}])assert.throws(()=>E.compare(raw));
  assert.throws(()=>E.validate({count:200}));assert.throws(()=>E.validate({seed:2.5}));
});

test('browser and Node runtimes execute the identical shared engine',()=>{
  const context={};vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(__dirname,'../dist/apex-model.js'),'utf8'),context);vm.runInContext(fs.readFileSync(path.join(__dirname,'../dist/engine.js'),'utf8'),context);
  const payload={scenarioId:'pressure',seed:9134,horizon:4};
  const browser=context.ApexEngine.compare(payload),server=E.compare(payload);
  assert.deepEqual(JSON.parse(JSON.stringify(browser.apex)),server.apex);
  assert.deepEqual(JSON.parse(JSON.stringify(browser.judge)),server.judge);
});

test('historical importer preserves recorded fields, orders and deduplicates dates',()=>{
  const data=[{date:'2023-09-15T13:00:01Z',speed:251,throttle:90,brake:0,n_gear:6,driver_number:55},{date:'2023-09-15T13:00:00Z',speed:245,throttle:88,brake:0,n_gear:6,driver_number:55},{date:'2023-09-15T13:00:01Z',speed:251,driver_number:55}];
  const r=T.parse(data);assert.equal(r.frames.length,2);assert.equal(r.frames[0].speed,245);assert.equal(r.frames[0].soc,null);assert.equal(r.source,'imported');
  assert.ok(T.at(r,.5).speed>245&&T.at(r,.5).speed<251);
});

test('importer rejects invalid telemetry and mixed driver sessions',()=>{
  for(const raw of [null,[],[null,null],[{date:'invalid',speed:100},{date:'bad',speed:90}],{car_data:[{date:'2023-01-01T00:00:00Z',speed:100,driver_number:1},{date:'2023-01-01T00:00:01Z',speed:200,driver_number:2}]}])assert.throws(()=>T.parse(raw));
});

test('synthetic replay and imported location interpolation stay finite',()=>{
  const s=T.synthetic(42);assert.equal(s.frames[0].soc,42);
  for(const t of [0,.1,45,90]){const f=T.at(s,t);assert.ok(Number.isFinite(f.x)&&Number.isFinite(f.y)&&Number.isFinite(f.speed));}
  const r=T.parse({car_data:[{date:'2023-01-01T00:00:00Z',speed:100},{date:'2023-01-01T00:00:02Z',speed:200}],location:[{date:'2023-01-01T00:00:00Z',x:10,y:10},{date:'2023-01-01T00:00:01Z',x:20,y:40},{date:'2023-01-01T00:00:02Z',x:40,y:60}]});
  assert.ok(r.hasLocation);assert.ok(r.points.every(([x,y])=>x>=0&&x<=800&&y>=0&&y<=440));
});
