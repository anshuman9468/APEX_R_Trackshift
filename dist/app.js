(function () {
  'use strict';
  const E = window.ApexEngine, T = window.ApexTelemetry;
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fixed = (v, n=2) => Number.isFinite(v) ? v.toFixed(n) : '--';
  const pct = (v) => fixed(v * 100, 0) + '%';
  const signed = (v) => (v >= 0 ? '+' : '') + fixed(v);
  const clamp01 = (v) => Math.max(0, Math.min(1, Number.isFinite(Number(v)) ? Number(v) : 0));
  const actionColors = {ATTACK:'#f44747',HOLD:'#c0e777',HARVEST:'#61d3e6',DEFEND:'#eabe63'};
  const actionExplanations = {
    ATTACK: 'Attack now: the current opportunity and available energy support an immediate move.',
    HOLD: 'Hold: preserve flexibility while the model sees no need to commit energy immediately.',
    DEFEND: 'Defend: rear pressure and the model safety signal make position protection the priority.',
    HARVEST: 'Harvest: recover energy and reduce degradation before spending it on a later opportunity.'
  };
  const modelInfo = document.querySelector('#dataDialog .provenance-table > div:nth-child(3) p');
  if(modelInfo) modelInfo.textContent='Frozen hybrid GNN + GRU + physics model, epoch 51. Predictions are served from the tested 2026 holdout bundle; the simulator does not fabricate live telemetry.';
  const key = 'apex-r-decisions-v1';
  let scenario = E.SCENARIOS[0], state = {...scenario.state}, windows = [...scenario.windows], risk = 0.45, judgeAction = 'ATTACK';
  let practiceSession = 'FP2', practiceRunPlan = 'performance', practiceLimitPreset = 'performance';
  let practiceLimits = {...E.CONFIG};
  let result = null, comparison = null, benchmark = null, replay = T.synthetic(state.soc), replayTime = 0, playing = false, lastTick = null;
  let activeTrackPoints = T.POINTS, activeTrackMap = null;
  let historicalFixture = window.ApexHistoricalFixture || null, hybridModel = null;
  let api = false, hybridApi = false, busy = false, logs = [], toastTimer, recalcTimer, revision = 0;
  let liveSnapshot = null, liveEnabled = false, liveViewRequested = true, livePollBusy = false, livePollTimer = null, testCatalog = null;
  let selectedDriver = null, selectedTargetDriver = null;
  let socket = null, streamedFrame = null, streamSoc = null, comparisonProvenance = null;
  const fields = [
    {key:'soc',name:'Initial energy',min:0,max:100,step:1,unit:'%'},
    {key:'gapAhead',name:'Gap ahead',min:0,max:5,step:0.05,unit:' s'},
    {key:'gapBehind',name:'Gap behind',min:0,max:5,step:0.05,unit:' s'},
    {key:'closing',name:'Closing speed',min:-20,max:25,step:1,unit:' km/h'},
    {key:'tyreAge',name:'Tyre age proxy',min:0,max:60,step:1,unit:' laps'},
    {key:'aggression',name:'Opponent aggression',min:0,max:1,step:0.05,unit:''},
    {key:'wet',name:'Wetness',min:0,max:1,step:0.05,unit:''}
  ];
  const practiceLimitFields = [
    {key:'reserveMJ',name:'Reserve floor',min:0,max:1.2,step:0.01,unit:' MJ'},
    {key:'maxDeployMJ',name:'Deploy limit',min:0.05,max:2.5,step:0.05,unit:' MJ'},
    {key:'maxHarvestMJ',name:'Recovery limit',min:0,max:1.5,step:0.05,unit:' MJ'},
    {key:'powerKW',name:'Power ceiling',min:50,max:600,step:10,unit:' kW'},
    {key:'windowSeconds',name:'Decision window',min:1,max:10,step:0.5,unit:' s'}
  ];
  const practiceLimitPresets = {
    baseline: {reserveMJ:0.20,maxDeployMJ:1.15,maxHarvestMJ:0.85,powerKW:250,windowSeconds:5},
    performance: {reserveMJ:0.12,maxDeployMJ:1.45,maxHarvestMJ:0.95,powerKW:300,windowSeconds:5},
    longrun: {reserveMJ:0.30,maxDeployMJ:1.00,maxHarvestMJ:0.90,powerKW:230,windowSeconds:5},
    wet: {reserveMJ:0.35,maxDeployMJ:0.90,maxHarvestMJ:0.75,powerKW:210,windowSeconds:6}
  };
  const practicePlans = {
    aero: {label:'Aero correlation', note:'Prioritises clean laps and protects the battery for correlation runs.'},
    performance: {label:'Performance run', note:'Balances deployment, lap time and battery reserve for a push lap.'},
    long: {label:'Long-run preparation', note:'Protects the battery and tyres to make the long-run response readable.'},
    wet: {label:'Wet setup', note:'Reduces deployment risk while grip and tyre behaviour are uncertain.'}
  };
  function toast(message) { $('toast').textContent=message; $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').hidden=true,4500); }
  function error(message='') { $('errorBanner').textContent=message; $('errorBanner').hidden=!message; }
  function historicalFrameAt(t) {
    if (!historicalFixture) return {speed:null, throttle:null, brake:null, gear:null, rpm:null, drs:null, progress:0};
    const frames = historicalFixture.replay.frames;
    const f = frames.reduce((best, value) => value.t <= t && value.t >= best.t ? value : best, frames[0]);
    const d = historicalFixture.decision.attacker_driver;
    const car = f.cars.find(v => v.driver === d) || {};
    return {speed:car.speed_kmh, throttle:car.throttle_pct, brake:car.brake == null ? null : car.brake * 100, gear:car.gear, rpm:car.rpm, drs:car.drs, x:mapX(car.x_m), y:mapY(car.y_m), progress:(f.t / historicalFixture.replay.duration_sec)};
  }
  function historicalRawFrame(t) {
    const frames = historicalFixture?.replay.frames || [];
    return frames.reduce((best, value) => value.t <= t && value.t >= best.t ? value : best, frames[0] || {t:0,cars:[]});
  }
  function mapX(x) {
    const b = historicalFixture?.replay.track_bounds; if (!b || !Number.isFinite(x)) return null;
    const sx = 720 / Math.max(1, b.xmax - b.xmin), sy = 400 / Math.max(1, b.ymax - b.ymin), scale = Math.min(sx, sy);
    return 40 + (720 - (b.xmax-b.xmin)*scale)/2 + (x-b.xmin)*scale;
  }
  function mapY(y) {
    const b = historicalFixture?.replay.track_bounds; if (!b || !Number.isFinite(y)) return null;
    const sx = 720 / Math.max(1, b.xmax - b.xmin), sy = 400 / Math.max(1, b.ymax - b.ymin), scale = Math.min(sx, sy);
    return 20 + (400 - (b.ymax-b.ymin)*scale)/2 + (b.ymax-y)*scale;
  }
  function replayFrameAt(t) { return replay.source === 'historical' ? historicalFrameAt(t) : T.at(replay,t); }
  function replayRawAt(t) { return replay.source === 'historical' ? historicalRawFrame(t) : null; }
  function liveCars(snapshot=liveSnapshot) { return Array.isArray(snapshot?.cars)?snapshot.cars.slice().sort((a,b)=>(Number(a.position)||999)-(Number(b.position)||999)):[]; }
  function liveCarFor(snapshot, driver) { return liveCars(snapshot).find(car=>car.driver===driver)||null; }
  function liveFocusCar(snapshot=liveSnapshot) { return liveCarFor(snapshot, selectedDriver)||snapshot?.focus_car||liveCars(snapshot)[0]||null; }
  function liveTargetCar(snapshot=liveSnapshot, my=liveFocusCar(snapshot)) {
    const cars=liveCars(snapshot), selected=liveCarFor(snapshot, selectedTargetDriver);
    if(selected&&selected.driver!==my?.driver)return selected;
    const position=Number(my?.position);
    return cars.filter(car=>car.driver!==my?.driver&&Number.isFinite(Number(car.position))).sort((a,b)=>{
      const aPos=Number(a.position),bPos=Number(b.position);
      const aAhead=Number.isFinite(position)&&aPos<position,bAhead=Number.isFinite(position)&&bPos<position;
      if(aAhead!==bAhead)return aAhead?-1:1;
      return aAhead?bPos-aPos:aPos-bPos;
    })[0]||null;
  }
  function livePair(snapshot=liveSnapshot) {
    const my=liveFocusCar(snapshot), target=liveTargetCar(snapshot,my);
    if(!my)return {my:null,target:null,relation:'NONE',gap:null};
    const myPosition=Number(my.position), targetPosition=Number(target?.position);
    let relation='HOLD';
    if(Number.isFinite(myPosition)&&Number.isFinite(targetPosition))relation=targetPosition<myPosition?'AHEAD':targetPosition>myPosition?'BEHIND':'HOLD';
    let gap=null;
    if(target){
      if(relation==='AHEAD'&&myPosition-targetPosition===1)gap=num(my.interval_to_ahead_s);
      if(relation==='BEHIND'&&targetPosition-myPosition===1)gap=num(target.interval_to_ahead_s);
      if(!Number.isFinite(gap)&&Number.isFinite(num(my.gap_to_leader_s))&&Number.isFinite(num(target.gap_to_leader_s)))gap=Math.abs(num(my.gap_to_leader_s)-num(target.gap_to_leader_s));
      if(!Number.isFinite(gap))gap=relation==='AHEAD'?num(my.gap_to_ahead_s):num(my.gap_behind_s);
    }
    return {my,target,relation,gap:Number.isFinite(gap)?Math.max(0,gap):null};
  }
  function liveSectorSummary(car) {
    const sectors=Array.isArray(car?.sectors)?car.sectors:[];
    if(!sectors.length)return 'S1 -- · S2 -- · S3 --';
    return sectors.slice(0,3).map(sector=>'S'+sector.sector+' '+(Number.isFinite(num(sector.time_s))?fixed(num(sector.time_s),3)+'s':'--')).join(' · ');
  }
  function renderLiveDriverSelectors(snapshot=liveSnapshot) {
    const driverSelect=$('liveDriver'), targetSelect=$('targetDriver'), cars=liveCars(snapshot);
    if(!driverSelect||!targetSelect)return;
    if(!cars.length){
      driverSelect.innerHTML='<option value="">Waiting for MultiViewer</option>';targetSelect.innerHTML='<option value="">Waiting for MultiViewer</option>';
      driverSelect.disabled=true;targetSelect.disabled=true;$('livePairContext').textContent='Choose a live driver pair when MultiViewer Live Timing is active.';return;
    }
    const preferred=liveCarFor(snapshot,selectedDriver)||snapshot?.focus_car||cars[0];
    selectedDriver=preferred.driver;
    const target=liveCarFor(snapshot,selectedTargetDriver);
    const targetCandidate=target&&target.driver!==selectedDriver?target:liveTargetCar(snapshot,preferred);
    selectedTargetDriver=targetCandidate?.driver||null;
    const label=car=>((car.name||car.driver)+' · P'+(car.position??'--'));
    driverSelect.innerHTML=cars.map(car=>'<option value="'+esc(car.driver)+'">'+esc(label(car))+'</option>').join('');
    targetSelect.innerHTML=cars.filter(car=>car.driver!==selectedDriver).map(car=>'<option value="'+esc(car.driver)+'">'+esc(label(car))+'</option>').join('');
    driverSelect.value=selectedDriver;targetSelect.value=selectedTargetDriver||'';driverSelect.disabled=false;targetSelect.disabled=!targetSelect.options.length;
    const pair=livePair(snapshot), circuit=snapshot?.circuit?.name||snapshot?.session?.meeting_name||'Live circuit',meeting=snapshot?.session?.meeting_name||snapshot?.session?.name||'Live session';
    const gap=Number.isFinite(pair.gap)?' · '+fixed(pair.gap,3)+' s gap':'';
    const caution=snapshot?.track_status?.caution_active?' · '+(snapshot.track_status.safety_car_active?'SAFETY CAR':snapshot.track_status.vsc_active?'VSC':'CAUTION'):'';
    $('livePairContext').textContent=meeting+' · '+circuit+' · '+(pair.my?.name||pair.my?.driver||'--')+' → '+(pair.target?.name||pair.target?.driver||'--')+' · target '+pair.relation.toLowerCase()+gap+caution+' · '+liveSectorSummary(pair.my);
  }
  function liveFrameFrom(snapshot=liveSnapshot, carOverride=null) {
    const car=carOverride||liveFocusCar(snapshot)||{}, lap=snapshot?.lap||{}, current=Number(lap.current), total=Number(lap.total);
    const progress=Number.isFinite(current)&&Number.isFinite(total)&&total>0?clamp01(current/total):0;
    const point=T.point(progress,activeTrackPoints);
    return {speed:num(car.speed_kmh), throttle:num(car.throttle_pct), brake:num(car.brake_pct), gear:num(car.gear), rpm:num(car.rpm), drs:num(car.drs), x:point.x, y:point.y, progress, soc:state.soc, driver:car.driver||null};
  }
  function num(value, fallback=null) { if(value==null||value==='')return fallback;const parsed=Number(value);return Number.isFinite(parsed)?parsed:fallback; }
  function liveHistoryFrames() {
    const history=liveSnapshot?.history||[], latest=Date.parse(liveSnapshot?.captured_at_utc||'');
    const recent=Number.isFinite(latest)?history.filter(sample=>{const stamp=Date.parse(sample.captured_at_utc||'');return !Number.isFinite(stamp)||latest-stamp<=10000;}):history.slice(-11);
    return recent.map(sample=>{const car=(sample.cars||[]).find(row=>row.driver===selectedDriver)||sample.focus_car;return liveFrameFrom({focus_car:car,lap:sample.lap},car);}).filter(frame=>Number.isFinite(frame.speed));
  }
  function renderLiveStatus(snapshot) {
    renderLiveDriverSelectors(snapshot);
    const pair=livePair(snapshot), status=snapshot?.status, bridgeLive=Boolean(snapshot?.connected&&snapshot?.live_timing_active&&pair.my), connected=bridgeLive&&liveViewRequested, replayMode=!liveViewRequested&&bridgeLive&&replay.source==='test';
    liveEnabled=connected;
    if(replayMode){$('liveDriver').disabled=true;$('targetDriver').disabled=true;$('livePairContext').textContent='Replay mode active; MultiViewer pairing is paused until you return to live mode.';}
    const year=Number(String(snapshot?.session?.start_utc||'').slice(0,4)), mapContext={year,event:snapshot?.session?.meeting_name,meetingName:snapshot?.session?.meeting_name,circuitName:snapshot?.circuit?.name||snapshot?.session?.circuit_name,location:snapshot?.session?.location};
    activeTrackMap=connected||status==='waiting'?T.mapFor(mapContext):replayMode&&replay.source==='test'?T.mapFor({year:replay.metadata?.year,event:replay.metadata?.event,location:replay.metadata?.location}):null;
    activeTrackPoints=activeTrackMap?.points||(replay.points||T.POINTS);
    for(const id of ['trackOutline','trackPath','trackDash'])$(id).setAttribute('d',T.path(activeTrackPoints));
    $('zones').style.display=activeTrackMap?'none':'';
    const badge=$('liveStatus');
    badge.textContent=connected?'MULTIVIEWER LIVE':replayMode?'TEST REPLAY / LIVE LINKED':status==='waiting'?'LIVE TIMING WAITING':'MULTIVIEWER OFFLINE';
    badge.className='pill '+(connected?'good':replayMode||status==='waiting'?'warning':'');
    $('liveButton').textContent=connected?'Refresh MultiViewer':replayMode?'Return to MultiViewer':'Connect MultiViewer';
    $('playButton').disabled=connected;$('scrubber').disabled=connected;
    if(connected)setPlayback(false);
    $('connection').dataset.live=connected?'true':'false';
    if(connected){
      const meeting=snapshot.session?.meeting_name||snapshot.session?.name||'Live session';
      const circuit=snapshot.circuit?.name||snapshot.session?.circuit_name||'Live circuit';
      const lap=snapshot.lap?.current?(' · lap '+snapshot.lap.current):'';
      $('dataBadge').textContent='MultiViewer observed · '+meeting+' · '+circuit+lap+' / modelled energy';
      $('replayBadge').textContent='MULTIVIEWER LIVE';$('replayBadge').className='pill good';
      $('circuitNote').textContent=activeTrackMap?'FastF1 track layout matched to MultiViewer circuit / live S1–S3':'Exact MultiViewer circuit metadata + live S1–S3 / schematic fallback';
      $('scenarioCaption').textContent='Selected driver pair is feeding the APEX-R decision twin.';
      $('trackTitle').textContent=circuit;
      $('sectorLabel').textContent='LIVE · LAP '+(snapshot.lap?.current||'--')+' · '+liveSectorSummary(pair.my);
    } else if(status==='waiting') {
      $('dataBadge').textContent='MultiViewer connected / start Live Timing to emit telemetry';
      $('replayBadge').textContent='LIVE WAITING';$('replayBadge').className='pill warning';
    } else {
      $('replayBadge').textContent=replay.source==='historical'?'HISTORICAL REFERENCE':(replay.source==='test'?'2024/2025 TEST REPLAY':(replay.source==='imported'?'IMPORTED REPLAY':'SYNTHETIC'));
      $('replayBadge').className='pill';
      $('dataBadge').textContent=replay.source==='historical'?'Observed telemetry / simulated energy and branches':(replay.source==='test'?'Recorded prepared telemetry / simulated energy and branches':(replay.source==='imported'?'Imported replay / modelled race state':'Synthetic replay / modelled energy'));
    }
  }
  function applyLiveSnapshot(snapshot) {
    const pair=livePair(snapshot), car=pair.my||{}, target=pair.target, cars=liveCars(snapshot);
    if(!car.driver)return;
    const position=num(car.position), immediateAhead=num(car.interval_to_ahead_s), immediateBehind=num(car.interval_behind_s);
    const targetGap=pair.gap, ahead=pair.relation==='AHEAD'&&Number.isFinite(targetGap)?targetGap:immediateAhead, behind=pair.relation==='BEHIND'&&Number.isFinite(targetGap)?targetGap:immediateBehind;
    const currentLap=num(snapshot.lap?.current), rainfall=num(snapshot.weather?.rainfall);
    const next={...state};
    if(Number.isFinite(position))next.position=Math.max(2,Math.min(19,Math.round(position)));
    next.gapAhead=Number.isFinite(ahead)?Math.max(.05,Math.min(10,ahead)):10;
    next.gapBehind=Number.isFinite(behind)?Math.max(.05,Math.min(10,behind)):10;
    if(Number.isFinite(car.tyre_age_laps))next.tyreAge=Math.max(0,Math.min(60,Math.round(car.tyre_age_laps)));
    if(Number.isFinite(rainfall))next.wet=clamp01(rainfall);
    if(Number.isFinite(currentLap))next.lap=Math.max(1,Math.min(80,Math.round(currentLap)));
    const closing=target&&Number.isFinite(num(car.speed_kmh))&&Number.isFinite(num(target.speed_kmh))?num(car.speed_kmh)-num(target.speed_kmh):num(car.closing_speed_kmh);
    if(Number.isFinite(closing))next.closing=Math.max(-20,Math.min(25,closing));
    state=next;
    const aheadCar=cars.find(row=>Number(row.position)===Number(car.position)-1), behindCar=cars.find(row=>Number(row.position)===Number(car.position)+1);
    $('aheadDriver').textContent=aheadCar?.driver||'—';$('yourDriver').textContent=car.driver||'—';$('behindDriver').textContent=behindCar?.driver||'—';
    renderInputControls();drawFrame();invalidate();recalculate();
  }
  async function pollMultiviewer() {
    if(livePollBusy||location.protocol==='file:')return;
    livePollBusy=true;
    try { liveSnapshot=await request('/api/live/snapshot',null,3000);renderLiveStatus(liveSnapshot);if(liveEnabled)applyLiveSnapshot(liveSnapshot);updateConnection(); }
    catch(errorValue) { liveEnabled=false;liveSnapshot={status:'unavailable',error:errorValue.message};renderLiveStatus(liveSnapshot);updateConnection(); }
    finally { livePollBusy=false; }
  }
  function input() {
    if(!$('seed').value.trim()) throw new Error('Enter a random seed.');
    const frame=liveEnabled?liveFrameFrom():replayFrameAt(replayTime);
    const liveSamples=liveEnabled?liveHistoryFrames():[];
    const samples=liveSamples.length?liveSamples:[];
    if(!samples.length)for(let seconds=0;seconds<=10;seconds+=.5)samples.push(replayFrameAt(Math.max(0,replayTime-seconds)));
    const mean=(key,fallback)=>{const values=samples.map(value=>Number(value[key])).filter(Number.isFinite);return values.length?values.reduce((sum,value)=>sum+value,0)/values.length:fallback;};
    const prior=liveEnabled?(samples[0]||frame):replayFrameAt(Math.max(0,replayTime-10));
    const openDrs=value=>[10,12,14].includes(Number(value))?1:0;
    const pair=liveEnabled?livePair():{my:null,target:null,relation:'NONE',gap:null};
    return {scenarioId:E.SCENARIOS.some(s=>s.id===scenario.id)?scenario.id:'patient',state:{...state},telemetry:{speedKmh:frame.speed,throttlePct:frame.throttle,brakePct:frame.brake,gear:frame.gear,raceProgress:frame.progress,speedMean10s:mean('speed',200),speedDelta10s:Number.isFinite(frame.speed)&&Number.isFinite(prior.speed)?frame.speed-prior.speed:0,throttleMean10s:mean('throttle',60),brakeFraction10s:mean('brake',0)/100,rpmMean10s:mean('rpm',10000),gearMean10s:mean('gear',5),drsOpenFraction10s:samples.reduce((sum,value)=>sum+openDrs(value.drs),0)/samples.length,disableLegacyPrior:true},windows:[...windows],risk,horizon:Number($('horizon').value),seed:Number($('seed').value),judgeAction,limits:{...practiceLimits},driverSelection:{myDriver:pair.my?.driver||null,targetDriver:pair.target?.driver||null,relation:pair.relation,gapSeconds:pair.gap},liveContext:liveEnabled?{session:liveSnapshot?.session||null,circuit:liveSnapshot?.circuit||null,lap:liveSnapshot?.lap||null,trackStatus:liveSnapshot?.track_status||null,sectors:pair.my?.sectors||[]} : null};
  }
  function provenance(){const pair=liveEnabled?livePair():{my:null,target:null,relation:'NONE',gap:null};return {replay:liveEnabled?'multiviewer_live':replay.source,name:liveEnabled?(liveSnapshot?.session?.meeting_name||liveSnapshot?.session?.name||'MultiViewer live session'):replay.name,metadata:liveEnabled?{...((liveSnapshot||{}).session||{}),circuit:liveSnapshot?.circuit||null,lap:liveSnapshot?.lap||null,sectors:pair.my?.sectors||[],availableStreams:liveSnapshot?.available_streams||[]}:(replay.metadata||null),description:liveEnabled?'Live MultiViewer telemetry received by the local APEX-R bridge. Energy, action branches and outcomes remain simulated.':replay.description,energy:'simulated',rivalGaps:liveEnabled?'observed MultiViewer timing':'simulated',outcomes:'simulated',driverSelection:{myDriver:pair.my?.driver||null,targetDriver:pair.target?.driver||null,relation:pair.relation,gapSeconds:pair.gap},predictionModel:'hybrid_gnn_physics_epoch51',captureSeconds:liveEnabled?null:replayTime};}
  function switchView(name) {
    const target=['pitwall','lab','validation','audit'].includes(name)?name:'pitwall';
    document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!=='view-'+target);
    document.querySelectorAll('.tab').forEach(b=>{const active=b.dataset.tab===target;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
    if(target==='audit')renderAudit();
    try{history.replaceState(null,'','#'+target);}catch(_){/* File previews may restrict history. */}
  }
  function table(headers, rows) { return '<table><thead><tr>'+headers.map(h=>'<th scope="col">'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map((c,i)=>'<td'+(i?' class="number"':'')+'>'+c+'</td>').join('')+'</tr>').join('')+'</tbody></table>'; }
  function setPlayback(on){playing=on;lastTick=null;$('playButton').setAttribute('aria-label',on?'Pause replay':'Play replay');$('playIcon').setAttribute('d',on?'M6 5h4v14H6zM14 5h4v14h-4z':'M8 5v14l11-7z');}
  function installReplay(value) {
    replay=value;replayTime=0;streamedFrame=null;setPlayback(false);$('scrubber').max=String(Math.round(replay.duration*10));$('scrubber').value='0';
    if(!liveViewRequested||!liveEnabled){activeTrackMap=replay.source==='test'?T.mapFor({year:replay.metadata?.year,event:replay.metadata?.event,location:replay.metadata?.location}):null;activeTrackPoints=activeTrackMap?.points||replay.points||T.POINTS;}
    for(const id of ['trackOutline','trackPath','trackDash'])$(id).setAttribute('d',T.path(activeTrackPoints));
    $('zones').style.display=replay.hasLocation?'none':'';
    $('trackTitle').textContent=activeTrackMap?.event||(replay.source==='historical'||replay.source==='test'||replay.source==='imported'?replay.name:'Strategy circuit');
    $('replayBadge').textContent=replay.source==='historical'?'HISTORICAL REFERENCE':(replay.source==='test'?'2024/2025 TEST REPLAY':(replay.source==='imported'?'IMPORTED REPLAY':'SYNTHETIC'));
    $('dataBadge').textContent=replay.source==='historical'?'Observed telemetry / simulated energy and branches':(replay.source==='test'?'Recorded prepared telemetry / simulated energy and branches':(replay.source==='imported'?'Imported replay / modelled race state':'Synthetic replay / modelled energy'));
    $('towerLap').textContent=(replay.source==='imported'||replay.source==='test'?'MODEL L':'LAP ')+state.lap;
    $('circuitNote').textContent=replay.source==='historical'?'Observed FastF1 positions / simulated branch outcomes':(replay.source==='test'?(activeTrackMap?'FastF1 track layout / recorded 2024/2025 samples':'Recorded 2024/2025 samples / schematic fallback'):(replay.hasLocation?'Imported path / modelled rivals':'Schematic circuit / simulated car positions'));
    $('provenanceText').textContent=replay.description;
    drawFrame();
  }
  function setHistoricalOption(active) {
    const select=$('scenario');
    let option=select.querySelector('option[data-historical]');
    if(active){
      if(!option){option=document.createElement('option');option.value='historical';option.dataset.historical='true';select.appendChild(option);}
      option.textContent='Historical reference · fixed window';
      select.value='historical';
    }else{
      if(option)option.remove();
      if(E.SCENARIOS.some(s=>s.id===scenario.id))select.value=scenario.id;
    }
  }
  function installHistorical(value) {
    liveViewRequested=false;
    historicalFixture=value; const d=value.decision, pair=value.decision.pair_context_values||[];
    const initial=historicalRawFrame(d.replay_time_sec), attacker=initial.cars.find(c=>c.driver===d.attacker_driver)||{};
    const speed=Number(attacker.speed_kmh); const distance=Number(pair[1]);
    state={...state, soc:42, lap:Number(d.lap), position:7, gapAhead:Number.isFinite(distance)&&Number.isFinite(speed)&&speed>1?Math.min(5,Math.max(.15,distance/(speed/3.6))):1.2, gapBehind:1.2, closing:Number.isFinite(Number(pair[0]))?Math.max(-20,Math.min(25,-Number(pair[0]))):0, wet:0};
    scenario={...scenario,id:'historical-2019-abu-dhabi',name:value.race.event+' / 2019',caption:'One approved historical decision window; branch futures are simulated.',state:{...state}};
    setHistoricalOption(true);
    windows=[.42,.35,.31]; judgeAction='ATTACK';
    replay={source:'historical',name:value.race.event+' 2019 — approved development replay',duration:value.replay.duration_sec,frames:value.replay.frames,points:value.replay.track_points,hasLocation:true,trackBounds:value.replay.track_bounds,metadata:{race_id:value.race.race_id,session:'Race',decision_window_id:d.window_id,decision_session_time_sec:d.session_time_sec,asof_tolerance_sec:value.decision.asof_tolerance_sec},description:'Real FastF1 car and track-frame position samples from an approved development-validation race. Each value is backward as-of within 1.0 second; no live latency or outcome claim.'};
    replayTime=d.replay_time_sec; setPlayback(false); $('scenarioCaption').textContent=scenario.caption; renderInputControls(); renderActionPicker(); invalidate(); installReplay(replay); replayTime=d.replay_time_sec; drawFrame(); recalculate();
  }
  async function loadHybridModel(){
    try{hybridModel=await request('/api/hybrid/predict?sequence=0',null,12000);hybridApi=hybridModel.status==='available';}
    catch(_){hybridApi=false;hybridModel={status:'unavailable',reason:'Start the local hybrid model server to load tested predictions.'};}
    renderGnn();
    if(result)renderRecommendation();
  }
  function selectedTestSeason() { return testCatalog?.seasons?.find(season=>String(season.year)===$('testYear').value); }
  function renderTestEvents() {
    const season=selectedTestSeason(), split=season?.splits?.test, events=split?.events||[];
    $('testEvent').innerHTML=events.map(event=>'<option value="'+esc(event.name)+'">'+esc(event.name)+'</option>').join('');
    renderTestSessions();
  }
  function renderTestSessions() {
    const season=selectedTestSeason(), event=(season?.splits?.test?.events||[]).find(row=>row.name===$('testEvent').value), sessions=event?.sessions||[];
    $('testSession').innerHTML=sessions.map(session=>'<option value="'+esc(session)+'">'+esc(session)+'</option>').join('');
  }
  async function loadTestCatalog() {
    try {
      testCatalog=await request('/api/test/catalog',null,5000);
      const summary=(testCatalog.seasons||[]).map(season=>season.year+' / '+((season.splits?.test?.events||[]).length)+' test events').join(' · ');
      $('testCatalogStatus').textContent=summary+' · recorded prepared telemetry; energy and futures remain simulated.';
      renderTestEvents();
    } catch(errorValue) { $('testCatalogStatus').textContent='2024/2025 local test catalog unavailable: '+errorValue.message; }
  }
  async function loadTestReplay() {
    const year=$('testYear').value, event=$('testEvent').value, session=$('testSession').value, button=$('testReplayButton');
    if(!event||!session){$('testCatalogStatus').textContent='Choose a 2024/2025 event and session first.';return;}
    button.disabled=true;button.textContent='Loading test replay...';
    try {
      const query=new URLSearchParams({year,split:'test',event,session,limit:'900'});
      const payload=await request('/api/test/replay?'+query.toString(),null,20000), origin=Date.UTC(Number(year),0,1);
      const rows=(payload.records||[]).map(record=>({date:new Date(origin+Number(record.time_s||0)*1000).toISOString(),speed:Number(record.speed_kmh),driver_number:null,session_key:null}));
      const loaded=T.parse({car_data:rows,metadata:{source:payload.source,year:Number(year),event,session,driver:payload.driver,relative_time:true}},year+' '+event+' / '+session+' test telemetry');
      loaded.source='test';loaded.name=year+' '+event+' · '+session;loaded.description='Recorded prepared telemetry from '+payload.source+' for '+year+' '+event+' / '+session+' / '+payload.driver+'. Time is relative to the source session; energy, rival branches and outcomes remain simulated.';loaded.metadata={year:Number(year),split:'test',event,session,driver:payload.driver,source:payload.source,relative_time:true};
      liveViewRequested=false;
      installReplay(loaded);
      try {
        const modelQuery=new URLSearchParams({year,split:'test',event});
        const seasonModel=await request('/api/hybrid/predict?'+modelQuery.toString(),null,20000);
        if(seasonModel.status==='available'){hybridModel=seasonModel;hybridApi=true;}
      } catch(_) { /* keep the tested model already loaded if the seasonal graph is unavailable */ }
      invalidate();renderGnn();recalculate();$('testCatalogStatus').textContent=payload.sample_count+' recorded samples loaded from '+year+' '+event+'; epoch 51 graph inference refreshed.';toast('2024/2025 test replay loaded.');
    } catch(errorValue) { $('testCatalogStatus').textContent='Test replay failed: '+errorValue.message; }
    finally { button.disabled=false;button.textContent='Load test replay'; }
  }
  function modelGuidedPolicy(){
    const o=result?.optimisation, current=result?.input;
    if(!o||!current||!o.feasible)return null;
    const candidates=Object.fromEntries((o.alternatives||[]).map(c=>[c.sequence[0],c]));
    const legal=E.ACTIONS.filter(action=>candidates[action]);
    if(!legal.length)return null;
    const values=legal.map(action=>Number(candidates[action].score));
    const min=Math.min(...values),max=Math.max(...values),span=Math.max(1e-9,max-min);
    const physics=Object.fromEntries(E.ACTIONS.map(action=>[action,legal.includes(action)?(Number(candidates[action].score)-min)/span:0]));
    const row=hybridModel?.predictions?.[0]||{};
    const tyre=clamp01(row.tyre_degradation_probability),pit=clamp01(row.pit_stop_probability),safety=clamp01(row.safety_constraint_probability);
    const energy=clamp01(current.state.soc/100);
    const energyNeed=clamp01((0.38-energy)/0.38);
    const opportunity=clamp01(0.50*(1-clamp01(current.state.gapAhead/3))+0.25*clamp01((current.state.closing+5)/20)+0.15*energy+0.10*(1-current.state.wet));
    const rearPressure=clamp01(0.60*(1-clamp01(current.state.gapBehind/1.5))+0.40*current.state.aggression);
    const modelRisk=clamp01(0.45*safety+0.35*tyre+0.20*pit);
    const pair=liveEnabled?livePair():{my:null,target:null,relation:'NONE',gap:null};
    const targetAhead=pair.relation==='AHEAD',targetBehind=pair.relation==='BEHIND',targetGap=Number.isFinite(pair.gap)?clamp01(pair.gap/2):1;
    const caution=Boolean(liveEnabled&&liveSnapshot?.track_status?.caution_active);
    const signals={
      ATTACK:clamp01(0.55*opportunity+0.25*(1-modelRisk)+0.20*energy),
      HOLD:clamp01(0.45*(1-opportunity)+0.35*(1-modelRisk)+0.20*(1-energyNeed)),
      DEFEND:clamp01(0.55*rearPressure+0.30*safety+0.15*(1-modelRisk)),
      HARVEST:clamp01(0.55*energyNeed+0.25*tyre+0.10*pit+0.10*current.state.wet)
    };
    if(liveEnabled&&pair.target){
      signals.ATTACK=clamp01(0.35*signals.ATTACK+0.45*(targetAhead?(1-targetGap):0)+0.20*(energy>0.42?1:0));
      signals.DEFEND=clamp01(0.35*signals.DEFEND+0.45*(targetBehind?(1-targetGap):0)+0.20*(targetBehind&&current.state.gapBehind<0.8?1:0));
      signals.HOLD=clamp01(0.65*signals.HOLD+0.20*(targetAhead&&targetGap>0.5?1:0)+0.15*(targetBehind&&targetGap>0.5?1:0));
    }
    if(caution){signals.HOLD=Math.max(signals.HOLD,.92);signals.ATTACK*=.12;signals.DEFEND=Math.max(signals.DEFEND,.35);signals.HARVEST=Math.max(signals.HARVEST,.25);}
    const liveBias={ATTACK:0,HOLD:0,HARVEST:0,DEFEND:0};
    if(caution){liveBias.HOLD=.85;liveBias.ATTACK=-.70;liveBias.DEFEND=.12;liveBias.HARVEST=.08;}
    else if(targetAhead){liveBias.ATTACK=.22;liveBias.HOLD=.04;}
    else if(targetBehind){liveBias.DEFEND=.22;liveBias.HOLD=.04;}
    const raw=Object.fromEntries(E.ACTIONS.map(action=>[action,legal.includes(action)?0.55*physics[action]+0.45*signals[action]+liveBias[action]:-Infinity]));
    const top=Math.max(...legal.map(action=>raw[action]));
    const weights=Object.fromEntries(E.ACTIONS.map(action=>[action,legal.includes(action)?Math.exp((raw[action]-top)*5):0]));
    const total=legal.reduce((sum,action)=>sum+weights[action],0)||1;
    const scores=Object.fromEntries(E.ACTIONS.map(action=>[action,weights[action]/total]));
    const recommendation=legal.slice().sort((a,b)=>scores[b]-scores[a]||physics[b]-physics[a])[0];
    const targetLabel=pair.target?.name||pair.target?.driver||'the selected target';
    const reason=caution?'Safety-car/neutralisation constraint is live; hold position and preserve the battery.':targetAhead?'Target '+targetLabel+' is ahead at '+(Number.isFinite(pair.gap)?fixed(pair.gap,3)+' s':'an unreported gap')+'; the policy weighs attack opportunity against energy and tyre risk.':targetBehind?'Target '+targetLabel+' is behind at '+(Number.isFinite(pair.gap)?fixed(pair.gap,3)+' s':'an unreported gap')+'; the policy weighs defending the position against battery and tyre cost.':'The selected live pair is not currently ordered; preserve flexibility until the timing feed resolves the relationship.';
    return {recommendation,scores,legal,modelSignals:{tyre,pit,safety},confidence:scores[recommendation],usingModel:hybridApi&&hybridModel?.status==='available',observedContext:{myDriver:pair.my?.driver||null,targetDriver:pair.target?.driver||null,relation:pair.relation,gapSeconds:pair.gap,caution},reason};
  }
  function renderActionScores(policy){
    if(!policy){$('actionScores').innerHTML='<p class="meta-note">Waiting for a feasible action window.</p>';return;}
    $('actionScores').innerHTML=E.ACTIONS.map(action=>{
      const legal=policy.legal.includes(action),value=policy.scores[action]||0,selected=action===policy.recommendation;
      return '<div class="action-score '+(selected?'selected ':'')+(legal?'':'blocked')+'" style="--action-color:'+(actionColors[action]||'#afb4b4')+'"><span>'+action+'</span><div class="action-track"><i style="width:'+(value*100).toFixed(1)+'%"></i></div><strong>'+(legal?pct(value):'BLOCKED')+'</strong></div>';
    }).join('');
  }
  function renderGnn(){
    const m=hybridModel, row=m?.predictions?.[0];
    renderActionScores(modelGuidedPolicy());
    $('actionPolicyStatus').textContent=m?.status==='available'?'ACTIVE':'WAITING';
    if(!m||m.status!=='available'){$('gnnScore').textContent='--';$('gnnStatus').textContent=m?.reason||'Loading tested hybrid model...';return;}
    $('gnnScore').textContent=Number.isFinite(Number(row?.next_lap_time_s))?fixed(Number(row.next_lap_time_s),2)+' s':'--';
    $('gnnScoreLabel').textContent=liveEnabled?'Holdout estimate / live state':'Epoch 51 test estimate';
    const pair=liveEnabled?livePair():null, inputContext=liveEnabled?'live MultiViewer state'+(pair?.my?' · '+(pair.my.name||pair.my.driver)+' vs '+(pair.target?.name||pair.target?.driver||'--'):''):replay.source==='test'?(replay.metadata?.year+' '+replay.metadata?.event+' test replay'):(m.test_context?.event||'2026 holdout');
    $('gnnStatus').textContent='Input: '+inputContext+' · Epoch 51 advisory heads: tyre '+pct(Number(row?.tyre_degradation_probability||0))+' / pit '+pct(Number(row?.pit_stop_probability||0))+' / safety '+pct(Number(row?.safety_constraint_probability||0));
  }
  function drawFrame() {
    let frame=replayFrameAt(replayTime);
    if(liveEnabled&&liveSnapshot?.focus_car)frame=liveFrameFrom();
    if(replay.source==='historical'){
      $('historicalCars').innerHTML=historicalRawFrame(replayTime).cars.filter(c=>Number.isFinite(mapX(c.x_m))&&Number.isFinite(mapY(c.y_m))).map(c=>{
        const attacker=c.driver===historicalFixture.decision.attacker_driver,target=c.driver===historicalFixture.decision.target_driver;
        const color=attacker?'#f44747':target?'#61d3e6':'#afb4b4',radius=attacker?8:target?7:4;
        return '<g transform="translate('+fixed(mapX(c.x_m),2)+' '+fixed(mapY(c.y_m),2)+')"><circle r="'+radius+'" fill="'+color+'" stroke="#101112" stroke-width="2"/><text y="-10" class="car-label">'+esc(c.driver)+'</text></g>';
      }).join('');
      $('historicalCars').style.display='';['carAhead','carRear','carMain'].forEach(id=>$(id).style.display='none');
      $('aheadPosition').textContent=historicalFixture.decision.target_driver;$('yourPosition').textContent=historicalFixture.decision.attacker_driver;$('behindPosition').textContent='—';
      $('towerAhead').textContent='TARGET';$('towerBehind').textContent='FIXED PAIR';$('towerLap').textContent='LAP '+historicalFixture.decision.lap;
      $('speedValue').innerHTML=fixed(frame.speed,0)+' <small>km/h</small>';$('gearValue').textContent=frame.gear??'--';$('throttleValue').innerHTML=(frame.throttle==null?'--':fixed(frame.throttle,0))+'<small>%</small>';$('brakeValue').innerHTML=(frame.brake==null?'--':fixed(frame.brake,0))+'<small>%</small>';
      $('throttleBar').style.width=(frame.throttle||0)+'%';$('brakeBar').style.width=(frame.brake||0)+'%';const soc=state.soc;$('liveEnergy').innerHTML=fixed(soc,0)+'<small>%</small>';$('liveEnergyBar').style.width=soc+'%';
      $('sectorLabel').textContent='SESSION T+'+fixed(historicalRawFrame(replayTime).session_time_sec,1)+'s';$('replayTime').textContent='T+'+fixed(historicalRawFrame(replayTime).session_time_sec,1)+'s';$('scrubber').value=String(Math.round(replayTime*10));
      const values=[];for(let i=0;i<32;i++){const f=replayFrameAt(Math.max(0,replayTime-8+i/4));values.push((i*160/31).toFixed(1)+','+(24-(f.speed||0)/450*24).toFixed(1));}$('speedSpark').innerHTML='<polyline points="'+values.join(' ')+'" fill="none" stroke="#afb4b4" stroke-width="1.5"/>';return;
    }
    $('historicalCars').style.display='none';['carAhead','carRear','carMain'].forEach(id=>$(id).style.display='');
    if(!liveEnabled&&replay.source==='synthetic'&&streamedFrame&&streamSoc===state.soc&&Math.abs(streamedFrame.t-replayTime)<.35){frame={...frame,speed:streamedFrame.speed,gear:streamedFrame.gear,throttle:streamedFrame.throttle,brake:streamedFrame.brake,soc:streamedFrame.soc};}
    if(liveEnabled){
      const pair=livePair(), aheadMarker=pair.relation==='AHEAD'?pair.target:liveCarFor(liveSnapshot,Number(pair.my?.position)>1?pair.my?.driver_ahead:null), behindMarker=pair.relation==='BEHIND'?pair.target:liveCarFor(liveSnapshot,pair.my?.driver_behind);
      $('carMain').querySelector('text').textContent=pair.my?.driver||'--';$('carAhead').querySelector('text').textContent=aheadMarker?.driver||'--';$('carRear').querySelector('text').textContent=behindMarker?.driver||'--';
    }
    const place=(id,p)=>$(id).setAttribute('transform','translate('+fixed(p.x,2)+' '+fixed(p.y,2)+')');
    place('carMain',frame);
    if(replay.hasLocation){place('carAhead',T.at(replay,Math.min(replay.duration,replayTime+state.gapAhead)));place('carRear',T.at(replay,Math.max(0,replayTime-state.gapBehind)));}
    else{place('carAhead',T.point(frame.progress+Math.max(.018,state.gapAhead/90),activeTrackPoints));place('carRear',T.point(frame.progress-Math.max(.018,state.gapBehind/90),activeTrackPoints));}
    $('speedValue').innerHTML=fixed(frame.speed,0)+' <small>km/h</small>';
    $('gearValue').textContent=frame.gear??'--';
    $('throttleValue').innerHTML=(frame.throttle==null?'--':fixed(frame.throttle,0))+'<small>%</small>';
    $('brakeValue').innerHTML=(frame.brake==null?'--':fixed(frame.brake,0))+'<small>%</small>';
    $('throttleBar').style.width=(frame.throttle||0)+'%';$('brakeBar').style.width=(frame.brake||0)+'%';
    const soc=frame.soc??state.soc;$('liveEnergy').innerHTML=fixed(soc,0)+'<small>%</small>';$('liveEnergyBar').style.width=soc+'%';
    $('sectorLabel').textContent=liveEnabled?'LIVE · LAP '+(liveSnapshot?.lap?.current||'--'):(replay.source==='imported'?'RECORDED SAMPLES':'SECTOR '+Math.min(3,1+Math.floor(frame.progress*3)));
    $('replayTime').textContent=liveEnabled?'LIVE':String(Math.floor(replayTime/60)).padStart(2,'0')+':'+String(Math.floor(replayTime%60)).padStart(2,'0');
    if(!liveEnabled)$('scrubber').value=String(Math.round(replayTime*10));
    const liveSamples=liveHistoryFrames(), values=[];for(let i=0;i<32;i++){const f=liveEnabled?(liveSamples[Math.max(0,liveSamples.length-32+i)]||frame):T.at(replay,Math.max(0,replayTime-8+i/4));values.push((i*160/31).toFixed(1)+','+(24-(f.speed||0)/450*24).toFixed(1));}
    $('speedSpark').innerHTML='<polyline points="'+values.join(' ')+'" fill="none" stroke="#afb4b4" stroke-width="1.5"/>';
  }
  function tick(timestamp){
    if(playing&&!document.hidden){if(lastTick!=null){replayTime=Math.min(replay.duration,replayTime+Math.min(0.1,(timestamp-lastTick)/1000)*Number($('playbackSpeed').value));drawFrame();if(replayTime>=replay.duration)setPlayback(false);}lastTick=timestamp;}else lastTick=null;
    requestAnimationFrame(tick);
  }
  function renderInputControls() {
    $('stateInputs').innerHTML=fields.map(f=>'<label class="slider-field"><span>'+esc(f.name)+' <output id="value-'+f.key+'"></output></span><input id="state-'+f.key+'" data-state="'+f.key+'" type="range" min="'+f.min+'" max="'+f.max+'" step="'+f.step+'" value="'+state[f.key]+'"></label>').join('');
    $('windowInputs').innerHTML=windows.slice(0,Number($('horizon').value)).map((v,i)=>'<label class="slider-field"><span>Lap +'+(i+1)+' <output id="window-value-'+i+'">'+fixed(v)+'</output></span><input data-window="'+i+'" type="range" min="0" max="1" step="0.05" value="'+v+'"></label>').join('');
    for(const f of fields)$('value-'+f.key).textContent=fixed(state[f.key],f.step<1?2:0)+f.unit;
    $('risk').value=String(risk*100);$('riskOutput').textContent=pct(risk);
  }
  function renderPracticeLimits() {
    $('practiceLimitPreset').value=practiceLimitPreset;
    $('practiceLimits').innerHTML=practiceLimitFields.map(f=>'<label class="slider-field practice-limit-field"><span>'+esc(f.name)+' <output id="practice-value-'+f.key+'"></output></span><input id="practice-limit-'+f.key+'" data-practice-limit="'+f.key+'" type="range" min="'+f.min+'" max="'+f.max+'" step="'+f.step+'" value="'+practiceLimits[f.key]+'"></label>').join('');
    for(const f of practiceLimitFields){const value=Number(practiceLimits[f.key]);$('practice-value-'+f.key).textContent=fixed(value,f.step<1?2:0)+f.unit;}
  }
  function renderPracticeSession() {
    const session=E.PRACTICE_SESSIONS[practiceSession]||E.PRACTICE_SESSIONS.FP2,plan=practicePlans[practiceRunPlan]||practicePlans.performance;
    const model=hybridModel?.predictions?.[0],active=Boolean(model&&hybridApi&&hybridModel?.status==='available');
    $('practiceSession').value=practiceSession;$('practiceRunPlan').value=practiceRunPlan;
    $('practiceModelBadge').textContent=active?'MODEL ACTIVE':'SIMULATOR ONLY';$('practiceModelBadge').className='pill '+(active?'good':'warning');
    $('practiceModelSource').textContent=active?'Epoch 51 · '+(hybridModel.test_context?.event||'tested holdout'):'Offline physics simulator';
    $('practiceModelNote').textContent=active?'Advisory heads are loaded; what-if limits are simulated.':'Start the local hybrid model service for advisory predictions.';
    $('practiceCarStatus').textContent=session.name.toUpperCase();$('practiceUpdated').textContent=session.duration+' · '+plan.label+' · '+plan.note;
  }
  let practiceBaselineCache=null;
  function practiceBaseline(raw) {
    const key=JSON.stringify({scenarioId:raw.scenarioId,state:raw.state,windows:raw.windows,risk:raw.risk,horizon:raw.horizon,seed:raw.seed,driverSelection:raw.driverSelection,liveContext:raw.liveContext});
    if(practiceBaselineCache?.key===key)return practiceBaselineCache.value;
    const value=E.compare({...raw,limits:{...E.CONFIG}});practiceBaselineCache={key,value};return value;
  }
  function renderPracticeDashboard() {
    const raw=result?.input;
    if(!raw)return;
    renderPracticeSession();
    const current=result,base=practiceBaseline(raw),summary=current.apex.summary,baseSummary=base.apex.summary,best=current.optimisation.feasible?current.optimisation.best:null,model=hybridModel?.predictions?.[0]||null;
    const legal=E.ACTIONS.filter(action=>E.energy(raw.state,action,raw.recovery[0],raw.limits).legal);
    const positionDelta=summary.expectedGain-baseSummary.expectedGain,energyDelta=summary.endSoc-baseSummary.endSoc,passDelta=(summary.passRate-baseSummary.passRate)*100;
    const modelTime=Number(model?.next_lap_time_s),modelWear=Number(model?.next_tyre_wear_estimate_fraction),modelPit=Number(model?.pit_stop_probability),modelSafety=Number(model?.safety_constraint_probability);
    $('impactStatus').textContent=current.optimisation.feasible?'RECALCULATED':'RECOVERY REQUIRED';$('impactStatus').className='pill '+(current.optimisation.feasible?'good':'warning');
    $('impactCards').innerHTML=[
      ['MODEL NEXT LAP',Number.isFinite(modelTime)?fixed(modelTime,2)+' s':'--',Number.isFinite(Number(model?.next_lap_time_delta_s))?'Model delta '+signed(Number(model.next_lap_time_delta_s))+' s':'Awaiting hybrid prediction'],
      ['SIMULATED POSITION',best?signed(summary.expectedGain):'--',best?signed(positionDelta)+' vs factory limits':'No feasible normal sequence'],
      ['SIMULATED END BATTERY',best?fixed(summary.endSoc,1)+'%':'--',best?signed(energyDelta,)+' pp vs factory limits':'Recovery required'],
      ['LEGAL ACTIONS',legal.length+'/'+E.ACTIONS.length,legal.length?('Best first move '+current.optimisation.recommendation):'All four modes blocked']
    ].map(([label,value,detail])=>'<div class="impact-card"><span>'+esc(label)+'</span><strong>'+esc(value)+'</strong><small>'+esc(detail)+'</small></div>').join('');
    const tyreValue=Number.isFinite(modelWear)?clamp01(modelWear):clamp01((raw.state.tyreAge+raw.horizon)/60),safetyValue=Number.isFinite(modelSafety)?clamp01(modelSafety):clamp01(summary.lossRate),paceValue=clamp01(summary.passRate);
    $('carResponse').innerHTML=[['Attack potential',paceValue,'#f44747',pct(paceValue)+' simulated'],['Battery margin',clamp01(summary.endSoc/100),'#61d3e6',fixed(summary.endSoc,1)+'% after horizon'],['Tyre load',tyreValue,'#eabe63',Number.isFinite(modelWear)?pct(modelWear)+' model estimate':pct(tyreValue)+' proxy'],['Constraint risk',safetyValue,'#c0e777',Number.isFinite(modelSafety)?pct(modelSafety)+' model estimate':pct(safetyValue)+' simulated']].map(([label,value,color,detail])=>'<div class="response-row"><div><span>'+esc(label)+'</span><strong>'+esc(detail)+'</strong></div><div class="response-track"><i style="width:'+(value*100).toFixed(1)+'%;background:'+color+'"></i></div></div>').join('');
    const changes=[];
    if(Math.abs(practiceLimits.reserveMJ-E.CONFIG.reserveMJ)>.001)changes.push((practiceLimits.reserveMJ>E.CONFIG.reserveMJ?'Higher':'Lower')+' reserve '+fixed(Math.abs(practiceLimits.reserveMJ-E.CONFIG.reserveMJ),2)+' MJ '+(practiceLimits.reserveMJ>E.CONFIG.reserveMJ?'protects the battery but blocks more deployment':'opens deployment but leaves less protection'));
    if(Math.abs(practiceLimits.maxDeployMJ-E.CONFIG.maxDeployMJ)>.001)changes.push((practiceLimits.maxDeployMJ>E.CONFIG.maxDeployMJ?'More':'Less')+' deployment allowance changes the attack ceiling');
    if(Math.abs(practiceLimits.maxHarvestMJ-E.CONFIG.maxHarvestMJ)>.001)changes.push((practiceLimits.maxHarvestMJ>E.CONFIG.maxHarvestMJ?'More':'Less')+' recovery allowance changes how quickly the car can rebuild energy');
    if(Math.abs(practiceLimits.powerKW-E.CONFIG.powerKW)>.001)changes.push('The '+fixed(practiceLimits.powerKW,0)+' kW ceiling changes how much deployment fits inside the window');
    const recommendation=current.optimisation.feasible?current.optimisation.recommendation:'RECOVER';
    $('impactNarrative').textContent=(changes.length?changes.join('. ')+'. ':'Factory limits are active. ')+('Under '+(practicePlans[practiceRunPlan]?.label||'this run plan')+', the optimizer selects '+recommendation+' first. Battery and position numbers are simulated; model cards are shown separately.');
    $('practiceConfidence').textContent=model?'Hybrid model active: next-lap time, tyre load, pit-stop probability and safety probability come from the frozen epoch-51 tested bundle. Limit-change deltas and action sequences are what-if simulator estimates, so they are not claims of guaranteed real-car accuracy.':'Hybrid model unavailable: the practice dashboard is running the local physics/strategy simulator only. Start the local model service to add the epoch-51 advisory heads.';
    $('limitsStatus').textContent=practiceLimitPreset==='custom'?'Custom engineer limits applied. Every change re-runs the feasible sequence search.':(practiceLimitPreset.replace('longrun','long-run')+' practice limits applied.');
  }
  function invalidate() { revision++;comparison=null;$('comparisonResult').hidden=true;$('comparisonEmpty').hidden=false;$('comparisonStatus').textContent='READY TO COMPARE'; }
  function recalculate() {
    try{
      result=E.compare(input());error();renderRecommendation();renderLab();
      const pair=liveEnabled?livePair():null, displayPosition=Number.isFinite(num(pair?.my?.position))?num(pair.my.position):state.position, displayAhead=pair?.my?.driver_ahead||liveCars(liveSnapshot).find(car=>Number(car.position)===Number(displayPosition)-1)?.driver, displayBehind=pair?.my?.driver_behind||liveCars(liveSnapshot).find(car=>Number(car.position)===Number(displayPosition)+1)?.driver;
      $('towerAhead').textContent='+'+fixed(state.gapAhead,3);$('towerBehind').textContent='-'+fixed(state.gapBehind,3);
      $('aheadPosition').textContent=displayPosition<=1?'—':String(displayPosition-1).padStart(2,'0');$('yourPosition').textContent=String(displayPosition).padStart(2,'0');$('behindPosition').textContent=displayPosition>=20?'—':String(displayPosition+1).padStart(2,'0');$('aheadDriver').textContent=displayAhead||'—';$('yourDriver').textContent=pair?.my?.driver||$('yourDriver').textContent||'DEV';$('behindDriver').textContent=displayBehind||'—';$('towerLap').textContent=(replay.source==='imported'||replay.source==='test'?'MODEL L':'LAP ')+state.lap;
      $('compareButton').disabled=busy;
    }catch(e){result=null;error(e.message);$('compareButton').disabled=true;}
  }
  function scheduleRecalculate(){invalidate();clearTimeout(recalcTimer);recalcTimer=setTimeout(recalculate,70);}
  function renderRecommendation(){
    const o=result.optimisation,s=result.apex.summary;
    const dataModelProbability=o.best?.trace?.[0]?.dataModelProbability;
    const policy=modelGuidedPolicy(),action=policy?.recommendation||(o.feasible?o.recommendation:'RECOVER');
    $('recommendation').textContent=action;
    $('recommendation').style.color=actionColors[action]||'#eabe63';
    $('decisionIndex').textContent=policy?'MODEL OUTPUT':'NO FEASIBLE ACTION';
    $('explanation').textContent=policy?(policy.reason||actionExplanations[action]||'Model-guided strategy action selected.'):(o.explanation||'No normal action fits the available energy. Block deployment and use emergency recovery.');
    const active=policy?.usingModel;
    $('modelStatus').textContent=active?'MODEL ACTIVE':(o.feasible?'PHYSICS ONLY':'RECOVERY REQUIRED');$('modelStatus').className='pill '+(active||o.feasible?'good':'warning');
    renderGnn();
    $('decisionContext').textContent=policy?(liveEnabled?'Observed MultiViewer pair, live track status, tyre state and sectors are combined with the frozen model advisory and physics-feasible action scores.':'The highlighted action combines the frozen model risk signals with the physics-feasible action scores.'):'No normal action is currently feasible under the physics constraints.';
    $('expectedGain').textContent=signed(s.expectedGain);$('passChance').textContent=pct(s.passRate);$('endEnergy').textContent=fixed(s.endSoc,1)+'%';$('lossChance').textContent=pct(s.lossRate);
    $('searchCount').textContent=o.evaluated+' feasible sequences';$('computeTime').textContent=fixed(o.latencyMs,1)+' ms search';
  }
  function renderLab(){
    const o=result.optimisation;$('labRecommendation').textContent=o.feasible?o.recommendation:'RECOVER';
    $('rankingTable').innerHTML=o.feasible?table(['FIRST ACTION','BEST SEQUENCE','UTILITY','END ENERGY'],o.alternatives.map(c=>[esc(c.sequence[0]),'<small>'+c.sequence.map(esc).join(' / ')+'</small>',fixed(c.score),fixed(c.endSoc,1)+'%'])):'<p class="meta-note">All normal deployment modes are blocked at this energy level.</p>';
    $('constraints').innerHTML=E.ACTIONS.map(a=>{const v=E.energy(state,a,result.input.recovery[0],result.input.limits);return '<div class="constraint-row"><b>'+a+'</b><span class="pill '+(v.legal?'good':'warning')+'">'+(v.legal?'PASS':'BLOCKED')+'</span><span>'+esc(v.legal?fixed(v.deployMJ)+' MJ deploy / '+fixed(v.harvestMJ)+' MJ recovery':v.violations.join('; '))+'</span></div>';}).join('');
    renderPracticeDashboard();
  }
  function selectScenario(id, reset=true){
    if(id==='historical'&&historicalFixture){installHistorical(historicalFixture);return;}
    scenario=E.SCENARIOS.find(s=>s.id===id)||E.SCENARIOS[0];state={...scenario.state};windows=[...scenario.windows];risk=.45;judgeAction='ATTACK';if(reset){practiceLimitPreset='performance';practiceLimits={...E.CONFIG,...practiceLimitPresets.performance};practiceBaselineCache=null;renderPracticeLimits();}$('scenario').value=scenario.id;$('scenarioCaption').textContent=scenario.caption;
    setHistoricalOption(false);
    if(reset){$('seed').value='2026';$('horizon').value='3';}
    renderInputControls();renderActionPicker();invalidate();installReplay(T.synthetic(state.soc));recalculate();
  }
  function renderActionPicker(){$('actionPicker').innerHTML=E.ACTIONS.map(a=>'<button class="action-button '+(a===judgeAction?'active':'')+'" data-action="'+a+'" aria-pressed="'+(a===judgeAction)+'">'+a+'</button>').join('');}
  function renderComparison(){
    const c=comparison,j=c.judge.summary,a=c.apex.summary;$('comparisonEmpty').hidden=true;$('comparisonResult').hidden=false;$('comparisonStatus').textContent='96 PAIRED FUTURES / SEED '+c.input.seed;
    $('comparisonTable').innerHTML=table(['MODEL ESTIMATE','YOUR STRATEGY','APEX-R'],[
      ['Expected final position',fixed(j.expectedPosition),fixed(a.expectedPosition)],['Overtake in horizon',pct(j.passRate),pct(a.passRate)],['Any position loss',pct(j.lossRate),pct(a.lossRate)],['Remaining energy',fixed(j.endSoc,1)+'%',fixed(a.endSoc,1)+'%'],['Position value / MJ',fixed(j.positionValuePerMJ,3),fixed(a.positionValuePerMJ,3)],['Executed violations',String(j.executedViolations),String(a.executedViolations)]
    ]);
    const svg=$('energyChart'),h=c.input.horizon;
    const x=(i)=>46+i*(568/h),y=(v)=>185-v*1.5;
    let chart='';for(const v of [0,25,50,75,100])chart+='<path d="M46 '+y(v)+'H614" stroke="#323536"/><text x="3" y="'+(y(v)+4)+'" fill="#afb4b4" font-size="12">'+v+'%</text>';
    for(let i=0;i<=h;i++)chart+='<text text-anchor="middle" x="'+x(i)+'" y="214" fill="#afb4b4" font-size="12">'+(i?'LAP +'+i:'NOW')+'</text>';
    for(const [branch,color] of [[c.judge,'#61d3e6'],[c.apex,'#c0e777']]){const pts=branch.sample.trace.map(t=>x(t.step)+','+y(t.soc));chart+='<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+color+'" stroke-width="2.5"/>';for(const t of branch.sample.trace)chart+='<circle cx="'+x(t.step)+'" cy="'+y(t.soc)+'" r="3.5" fill="'+color+'"/>';}
    svg.innerHTML=chart;
    $('sampleNote').textContent='Seed '+c.input.seed+' / not a guaranteed race outcome';
    $('branchTimeline').innerHTML=[[c.judge,'Your strategy',''],[c.apex,'APEX-R','apex-branch']].map(([branch,name,cls])=>'<section class="branch '+cls+'"><h3>'+name+' <span class="meta-note">'+branch.sequence.map(esc).join(' / ')+'</span></h3><ol>'+branch.sample.trace.slice(1).map(t=>'<li><span>LAP +'+t.step+'</span><strong>'+esc(t.action)+'</strong><div>'+esc(t.event)+'<small>P'+t.position+' / '+fixed(t.soc,1)+'% energy</small></div></li>').join('')+'</ol></section>').join('');
    const difference=a.expectedGain-j.expectedGain;
    $('comparisonVerdict').textContent=(Math.abs(difference)<.001?'Both strategies have the same estimated position gain.':difference>0?'APEX-R gains '+fixed(difference)+' more positions on average in this model.':'Your strategy gains '+fixed(-difference)+' more positions on average in this model.')+' APEX-R gain interval: '+signed(a.gainInterval[0])+' to '+signed(a.gainInterval[1])+'. Sampling uncertainty only.';
  }
  async function request(path,body,timeout=12000){
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeout);
    try{const response=await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:undefined,body:body?JSON.stringify(body):undefined,signal:controller.signal});if(!response.ok)throw new Error('Server returned '+response.status);return await response.json();}finally{clearTimeout(timer);}
  }
  async function runComparison(){
    if(busy)return;let snapshot;try{snapshot=E.normalise(input());}catch(e){error(e.message);return;}
    const currentRevision=revision,source=provenance();busy=true;setPlayback(false);$('compareButton').disabled=true;$('compareButton').textContent='Simulating futures...';
    const id=typeof crypto.randomUUID==='function'?crypto.randomUUID():Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);
    try{
      let computed;
      if(api){try{computed=await request('/api/compare',{input:snapshot,provenance:source,request_id:id});}catch(_){api=false;updateConnection();toast('Server unavailable. Comparison completed with the offline engine.');}}
      if(!computed){await new Promise(resolve=>setTimeout(resolve,20));computed=E.compare(snapshot);}
      const entry={id,created_at:new Date().toISOString(),result:computed,provenance:source};addLog(entry);
      if(currentRevision===revision){comparison=computed;comparisonProvenance=source;renderComparison();toast('Comparison complete. Decision saved.');}else toast('Inputs changed during the run. The completed comparison is in Decision log.');
    }catch(e){error('Comparison failed: '+e.message);}finally{busy=false;$('compareButton').innerHTML='Compare futures <span aria-hidden="true">&rarr;</span>';recalculate();}
  }
  function addLog(entry){logs=[entry,...logs.filter(x=>x.id!==entry.id)].slice(0,100);persistLogs();renderAudit();}
  function persistLogs(){try{localStorage.setItem(key,JSON.stringify(logs));}catch(_){toast('Device storage is unavailable or full. Export your decision log before closing.');} $('logCount').textContent=String(logs.length);}
  function renderAudit(){
    $('logCount').textContent=String(logs.length);
    $('auditContent').innerHTML=logs.length?logs.map((l,i)=>'<details class="audit-entry"><summary><time>'+esc(new Date(l.created_at).toLocaleString())+'</time><strong>'+esc(l.result.optimisation.recommendation)+'</strong><span>'+esc(l.result.input.scenarioId)+' / SEED '+l.result.input.seed+'</span></summary><div class="audit-detail"><p class="meta-note">Model '+esc(l.result.version)+' / Replay: '+esc(l.provenance?.replay||'unspecified')+'</p><button class="button quiet restore-decision" data-index="'+i+'">Restore strategy inputs</button><pre>'+esc(JSON.stringify({input:l.result.input,provenance:l.provenance,optimisation:l.result.optimisation,apexSummary:l.result.apex.summary,judgeSummary:l.result.judge.summary},null,2))+'</pre></div></details>').join(''):'<div class="empty-state"><h2>No decisions recorded yet.</h2><p>Compare two futures in the pit wall to record the starting state, strategy, and simulated result.</p></div>';
  }
  function download(name,value){const blob=new Blob([JSON.stringify(value,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  async function localBenchmark(options){
    if(location.protocol==='file:'||typeof Worker==='undefined'){await new Promise(r=>setTimeout(r,30));return E.validate(options);}
    return new Promise((resolve,reject)=>{const worker=new Worker('./worker.js');const timer=setTimeout(()=>{worker.terminate();reject(new Error('Benchmark timed out.'));},30000);worker.onmessage=event=>{clearTimeout(timer);worker.terminate();if(event.data.error)reject(new Error(event.data.error));else resolve(event.data.result);};worker.onerror=()=>{clearTimeout(timer);worker.terminate();try{resolve(E.validate(options));}catch(e){reject(e);}};worker.postMessage({type:'validate',input:options});});
  }
  async function runBenchmark(){
    const button=$('runValidation');button.disabled=true;button.textContent='Running paired trials...';
    try{const seed=Number($('seed').value);E.normalise(input());benchmark=await localBenchmark({count:Number($('benchmarkCount').value),seed});renderBenchmark();toast('Benchmark complete. Wins and losses are included.');}catch(e){error(e.message);}finally{button.disabled=false;button.textContent='Run benchmark';}
  }
  function renderBenchmark(){
    const b=benchmark;$('validationEmpty').hidden=true;$('validationResult').hidden=false;
    $('validationMetrics').innerHTML=[['WINS / TIES / LOSSES',b.wins+' / '+b.ties+' / '+b.losses],['PAIRED ROLLOUTS',b.comparisons.toLocaleString()],['EXECUTED VIOLATIONS',String(b.rows.reduce((s,r)=>s+r.executedViolations,0))],['BENCHMARK TIME',fixed(b.elapsedMs,0)+' ms']].map(([label,value])=>'<div><strong>'+value+'</strong><span>'+label+'</span></div>').join('');
    $('validationTable').innerHTML=table(['POLICY','POSITION GAIN','END ENERGY','PASS RATE','LOSS RATE','UTILITY','BLOCKED ACTIONS'],b.rows.map(r=>[esc(r.name),signed(r.gain),fixed(r.endSoc,1)+'%',pct(r.passRate),pct(r.lossRate),fixed(r.score),String(r.blockedActions)]));
    $('validationDetails').innerHTML=table(['SCENARIO','APEX-R SEQUENCE','APEX GAIN','THRESHOLD GAIN','UTILITY DIFFERENCE'],b.details.map(r=>[String(r.scenario)+' / '+esc(r.scenarioId),'<small>'+r.sequence.map(esc).join(' / ')+'</small>',signed(r.apexGain),signed(r.baselineGain),'<span class="'+(r.scoreDifference>=0?'positive':'negative')+'">'+signed(r.scoreDifference)+'</span>']));
  }
  function updateConnection(){const live=liveEnabled,connected=api||hybridApi||live;$('connection').textContent=live?'MultiViewer live + hybrid model':(api?'Local API connected':(hybridApi?'Hybrid model connected':'Offline engine'));$('connection').className='pill '+(connected?'good':'');$('auditStorage').textContent=api?'New comparisons are saved in local SQLite and this browser. The latest 100 decisions are shown.':'Stored in this browser on this device. The latest 100 comparisons record inputs, seed, model version, scores, and outcomes.';}
  async function connect(){
    if(location.protocol==='file:')return;
    try{const status=await request('/api/health',null,1500);api=status.engine_version===E.VERSION;if(api){const saved=await request('/api/audit?limit=100');const all=[...logs,...saved.records];logs=Array.from(new Map(all.map(l=>[l.id,l])).values()).sort((a,b)=>b.created_at.localeCompare(a.created_at)).slice(0,100);persistLogs();renderAudit();if(status.websocket)connectStream();}if(status.hybrid_model?.status==='available')await loadHybridModel();}catch(_){api=false;}updateConnection();await pollMultiviewer();await loadTestCatalog();
  }
  function connectStream(){
    socket=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws/replay');
    socket.onmessage=e=>{try{const value=JSON.parse(e.data);if(value.frame){streamedFrame=value.frame;streamSoc=state.soc;}}catch(_){streamedFrame=null;}};
    socket.onerror=()=>{streamedFrame=null;};socket.onclose=()=>{streamedFrame=null;socket=null;};
  }
  setInterval(()=>{if(playing&&!document.hidden&&replay.source==='synthetic'&&socket?.readyState===WebSocket.OPEN)socket.send(JSON.stringify({time:replayTime,soc:state.soc}));},250);
  livePollTimer=setInterval(()=>{if(!document.hidden)pollMultiviewer();},1000);
  $('scenario').innerHTML=E.SCENARIOS.map(s=>'<option value="'+s.id+'">'+esc(s.name)+'</option>').join('');
  document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.tab)));
  $('scenario').addEventListener('change',e=>selectScenario(e.target.value,false));
  $('liveButton').addEventListener('click',()=>{liveViewRequested=true;if(liveSnapshot){renderLiveStatus(liveSnapshot);if(liveEnabled)applyLiveSnapshot(liveSnapshot);}else pollMultiviewer();});
  $('liveDriver').addEventListener('change',e=>{selectedDriver=e.target.value||null;selectedTargetDriver=null;renderLiveDriverSelectors(liveSnapshot);if(liveEnabled)applyLiveSnapshot(liveSnapshot);else{drawFrame();recalculate();}});
  $('targetDriver').addEventListener('change',e=>{selectedTargetDriver=e.target.value||null;renderLiveDriverSelectors(liveSnapshot);if(liveEnabled)applyLiveSnapshot(liveSnapshot);else{drawFrame();recalculate();}});
  $('testYear').addEventListener('change',renderTestEvents);
  $('testEvent').addEventListener('change',renderTestSessions);
  $('testReplayButton').addEventListener('click',loadTestReplay);
  $('resetButton').addEventListener('click',()=>selectScenario(scenario.id));
  $('horizon').addEventListener('change',()=>{renderInputControls();invalidate();recalculate();});
  $('seed').addEventListener('input',scheduleRecalculate);
  $('practiceSession').addEventListener('change',e=>{practiceSession=e.target.value;renderPracticeSession();renderPracticeDashboard();});
  $('practiceRunPlan').addEventListener('change',e=>{practiceRunPlan=e.target.value;renderPracticeSession();renderPracticeDashboard();});
  $('practiceLimitPreset').addEventListener('change',e=>{const preset=e.target.value;if(preset==='custom'){practiceLimitPreset='custom';renderPracticeSession();return;}practiceLimitPreset=preset;practiceLimits={...E.CONFIG,...practiceLimitPresets[preset]};practiceBaselineCache=null;renderPracticeLimits();invalidate();recalculate();});
  $('practiceLimits').addEventListener('input',e=>{const key=e.target.dataset.practiceLimit;if(!key)return;practiceLimits[key]=Number(e.target.value);practiceLimitPreset='custom';$('practiceLimitPreset').value='custom';$('practice-value-'+key).textContent=fixed(practiceLimits[key],practiceLimitFields.find(f=>f.key===key)?.step<1?2:0)+(practiceLimitFields.find(f=>f.key===key)?.unit||'');practiceBaselineCache=null;scheduleRecalculate();});
  $('practiceResetButton').addEventListener('click',()=>{practiceLimitPreset='performance';practiceLimits={...E.CONFIG,...practiceLimitPresets.performance};practiceBaselineCache=null;renderPracticeLimits();invalidate();recalculate();toast('Performance practice limits restored.');});
  $('stateInputs').addEventListener('input',e=>{const f=fields.find(x=>x.key===e.target.dataset.state);if(!f)return;state[f.key]=Number(e.target.value);$('value-'+f.key).textContent=fixed(state[f.key],f.step<1?2:0)+f.unit;if(f.key==='soc'&&replay.source==='synthetic')installReplay(T.synthetic(state.soc));scheduleRecalculate();});
  $('risk').addEventListener('input',e=>{risk=Number(e.target.value)/100;$('riskOutput').textContent=pct(risk);scheduleRecalculate();});
  $('windowInputs').addEventListener('input',e=>{if(e.target.dataset.window==null)return;const i=Number(e.target.dataset.window);windows[i]=Number(e.target.value);$('window-value-'+i).textContent=fixed(windows[i]);scheduleRecalculate();});
  $('actionPicker').addEventListener('click',e=>{const b=e.target.closest('[data-action]');if(!b)return;judgeAction=b.dataset.action;renderActionPicker();invalidate();recalculate();});
  $('playButton').addEventListener('click',()=>{if(replayTime>=replay.duration)replayTime=0;setPlayback(!playing);});
  $('scrubber').addEventListener('input',e=>{setPlayback(false);replayTime=Number(e.target.value)/10;drawFrame();});
  $('captureButton').addEventListener('click',()=>{setPlayback(false);const frame=replayFrameAt(replayTime);if(frame.soc!=null&&replay.source!=='historical')state.soc=Number(fixed(frame.soc,1));if(replay.source==='synthetic'){windows[0]=Math.max(.05,Math.min(1,(frame.speed-80)/250));}renderInputControls();invalidate();recalculate();toast(replay.source==='historical'?'Historical decision window captured. Branch energy remains simulated; observed future positions are reference-only.':'Window captured. Energy and opportunity remain modelled; gaps use the scenario inputs.');});
  $('compareButton').addEventListener('click',runComparison);
  $('inspectButton').addEventListener('click',()=>switchView('lab'));
  $('runValidation').addEventListener('click',runBenchmark);
  $('exportComparison').addEventListener('click',()=>{if(comparison)download('APEX-R_Decision_'+comparison.input.seed+'.json',{...comparison,provenance:comparisonProvenance});});
  $('exportValidation').addEventListener('click',()=>{if(benchmark)download('APEX-R_Benchmark_'+benchmark.seed+'.json',benchmark);});
  $('exportAudit').addEventListener('click',()=>{if(logs.length)download('APEX-R_Decision_Log.json',{version:E.VERSION,records:logs});else toast('Run a comparison before exporting the log.');});
  $('auditContent').addEventListener('click',e=>{const b=e.target.closest('.restore-decision');if(!b)return;const entry=logs[Number(b.dataset.index)],saved=E.normalise(entry.result.input);selectScenario(saved.scenarioId);state={...saved.state};windows=saved.windows;risk=saved.risk;judgeAction=saved.judgeAction;$('horizon').value=String(saved.horizon);$('seed').value=String(saved.seed);installReplay(T.synthetic(state.soc));renderInputControls();renderActionPicker();invalidate();recalculate();switchView('pitwall');toast('Strategy inputs restored. Replay uses the bundled synthetic circuit.');});
  $('dataButton').addEventListener('click',()=>$('dataDialog').showModal());$('guideButton').addEventListener('click',()=>$('guideDialog').showModal());
  document.querySelectorAll('.close-dialog').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));
  document.querySelectorAll('dialog').forEach(d=>d.addEventListener('click',e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();}}));
  $('startDemo').addEventListener('click',()=>{selectScenario('patient');switchView('pitwall');$('guideDialog').close();setPlayback(true);});
  $('telemetryFile').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;$('importStatus').textContent='Reading telemetry...';try{if(file.size>20*1024*1024)throw new Error('Keep the telemetry file below 20 MB.');const imported=T.parse(JSON.parse(await file.text()),file.name);liveViewRequested=false;installReplay(imported);invalidate();$('importStatus').textContent=imported.frames.length+' samples loaded. Rival gaps and energy are still simulated.';toast('Telemetry imported. Recorded fields are labelled.');}catch(err){$('importStatus').textContent='Import failed: '+err.message;}finally{e.target.value='';}});
  $('syntheticButton').addEventListener('click',()=>{setHistoricalOption(false);liveViewRequested=false;installReplay(T.synthetic(state.soc));invalidate();$('importStatus').textContent='Bundled synthetic replay restored.';});
  $('historicalButton').addEventListener('click',()=>{if(!historicalFixture){$('importStatus').textContent='Historical fixture is unavailable.';return;}installHistorical(historicalFixture);$('dataDialog').close();$('importStatus').textContent='Approved historical replay loaded. Observed telemetry and simulated branches are separated.';toast('Historical replay loaded.');});
  try{const saved=JSON.parse(localStorage.getItem(key)||'[]');if(Array.isArray(saved))logs=saved.filter(l=>{try{return typeof l.id==='string'&&Number.isFinite(Date.parse(l.created_at))&&l.result.version===E.VERSION&&E.normalise(l.result.input)&&l.result.optimisation&&l.result.apex.summary&&l.result.judge.summary;}catch(_){return false;}}).slice(0,100);}catch(_){logs=[];}
  selectScenario('patient');renderAudit();switchView(location.hash.slice(1));requestAnimationFrame(tick);connect();
})();
