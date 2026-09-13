(function (root, factory) { const value = factory(); if (typeof module === 'object' && module.exports) module.exports = value; else root.ApexTelemetry = value; })(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const POINTS = [[110,340],[250,340],[350,322],[510,305],[555,275],[505,238],[400,240],[368,213],[408,177],[552,172],[610,162],[638,105],[600,65],[510,60],[420,85],[290,100],[230,138],[215,203],[156,229],[96,226],[65,257],[80,300],[110,340]];
  const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
  const key = value => String(value||'').trim().toLowerCase().replace(/[^a-z0-9]+/g,' ');
  function mapFor(context={}) {
    const catalogue = (typeof window !== 'undefined' ? window.ApexTrackMaps : null)?.maps || {};
    const wanted = [context.event,context.meetingName,context.circuitName,context.location].filter(Boolean).map(key);
    const year = Number(context.year);
    const exact = Object.values(catalogue).find(record => (!Number.isFinite(year)||Number(record.season)===year) && wanted.includes(key(record.event)));
    if(exact)return exact;
    return Object.values(catalogue).find(record => wanted.includes(key(record.event))||wanted.includes(key(record.location))) || null;
  }
  function pointsFor(context={}) { return mapFor(context)?.points || POINTS; }
  function path(points = POINTS) { return points.map((p,i) => (i ? 'L' : 'M') + p[0].toFixed(2) + ' ' + p[1].toFixed(2)).join(' '); }
  function point(progress, points = POINTS) {
    const lengths = []; let total = 0;
    for (let i=1;i<points.length;i++) {const d=Math.hypot(points[i][0]-points[i-1][0],points[i][1]-points[i-1][1]);lengths.push(d);total+=d;}
    let target = ((progress % 1 + 1) % 1) * total;
    for(let i=0;i<lengths.length;i++){if(target<=lengths[i]){const ratio=lengths[i]?target/lengths[i]:0;return {x:points[i][0]+(points[i+1][0]-points[i][0])*ratio,y:points[i][1]+(points[i+1][1]-points[i][1])*ratio};}target-=lengths[i];}
    return {x:points[0][0],y:points[0][1]};
  }
  function synthetic(soc = 42) {
    const frames = [];let energy = soc;
    for(let i=0;i<=360;i++){
      const t=i/4, progress=t/90, wave=Math.sin(progress*Math.PI*10);
      const speed=clamp(235+72*Math.sin(progress*Math.PI*4+0.4)+22*wave,85,330);
      const brake=wave < -0.65 ? 100 : 0;
      const throttle=brake?0:clamp(70+30*Math.sin(progress*Math.PI*4),15,100);
      if(i)energy=clamp(energy+(brake?0.07:-0.014),0,100);
      const p=point(progress);
      frames.push({t, speed:Math.round(speed), gear:clamp(Math.round(speed/43),1,8), throttle:Math.round(throttle), brake, rpm:Math.round(speed*42), drs:throttle>92&&brake===0?10:0, x:p.x,y:p.y,soc:energy,progress});
    }
    return {name:'Bundled synthetic replay', source:'synthetic', description:'Synthetic car telemetry and schematic circuit. No historical race is represented.', frames, points:POINTS, duration:90};
  }
  function parse(raw, filename='Imported telemetry') {
    if(!raw||typeof raw!=='object')throw new Error('Telemetry must be a JSON array or object.');
    const rows=Array.isArray(raw)?raw:raw.car_data;
    if(!Array.isArray(rows)||rows.length<2||rows.length>100000)throw new Error('Provide 2 to 100,000 car_data records.');
    if(rows.some(r=>!r||typeof r!=='object'))throw new Error('Telemetry records must be objects.');
    const drivers=new Set(rows.map(r=>r.driver_number).filter(v=>v!=null));
    if(drivers.size>1)throw new Error('Import one driver at a time. Filter car_data by driver_number.');
    const sessions=new Set(rows.map(r=>r.session_key).filter(v=>v!=null));
    if(sessions.size>1)throw new Error('Import one session at a time.');
    const sorted=rows.map((r)=>{
      const timestamp=Date.parse(r.date);
      if(!Number.isFinite(timestamp)||typeof r.speed!=='number'||!Number.isFinite(r.speed)||r.speed<0||r.speed>450)throw new Error('Every record needs an ISO date and speed between 0 and 450 km/h.');
      for(const key of ['throttle','brake','n_gear','rpm','drs'])if(r[key]!=null&&(typeof r[key]!=='number'||!Number.isFinite(r[key])))throw new Error('Invalid '+key+' in telemetry.');
      if(r.throttle!=null&&(r.throttle<0||r.throttle>100))throw new Error('Throttle must be between 0 and 100.');
      if(r.brake!=null&&(r.brake<0||r.brake>100))throw new Error('Brake must be between 0 and 100.');
      if(r.n_gear!=null&&(!Number.isInteger(r.n_gear)||r.n_gear<0||r.n_gear>8))throw new Error('Gear must be an integer between 0 and 8.');
      if(r.rpm!=null&&(r.rpm<0||r.rpm>20000))throw new Error('RPM must be between 0 and 20,000.');
      if(r.drs!=null&&(r.drs<0||r.drs>20))throw new Error('DRS must be between 0 and 20.');
      return {...r,timestamp};
    }).sort((a,b)=>a.timestamp-b.timestamp);
    const unique=sorted.filter((r,i)=>!i||r.timestamp!==sorted[i-1].timestamp);
    const start=unique[0].timestamp,duration=(unique[unique.length-1].timestamp-start)/1000;
    if(duration<=0||duration>7200)throw new Error('Replay duration must be more than zero and at most two hours.');
    let locations=[];
    if(!Array.isArray(raw)&&Array.isArray(raw.location)) locations=raw.location.filter(r=>r&&Number.isFinite(r.x)&&Number.isFinite(r.y)&&Number.isFinite(Date.parse(r.date))&&Date.parse(r.date)>=start&&Date.parse(r.date)<=start+duration*1000&&(r.driver_number==null||!drivers.size||drivers.has(r.driver_number))&&(r.session_key==null||!sessions.size||sessions.has(r.session_key))).sort((a,b)=>Date.parse(a.date)-Date.parse(b.date));
    let points=POINTS,locationFrames=null;
    if(locations.length>2){
      let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
      for(const l of locations){minX=Math.min(minX,l.x);maxX=Math.max(maxX,l.x);minY=Math.min(minY,l.y);maxY=Math.max(maxY,l.y);}
      if(maxX-minX>0&&maxY-minY>0){const scale=Math.min(660/(maxX-minX),330/(maxY-minY));locationFrames=locations.map(l=>({t:(Date.parse(l.date)-start)/1000,x:400+(l.x-(minX+maxX)/2)*scale,y:220-(l.y-(minY+maxY)/2)*scale}));points=locationFrames.map(l=>[l.x,l.y]);}
    }
    let li=0;const frames=unique.map(r=>{const t=(r.timestamp-start)/1000;let p=point(t/duration);
      if(locationFrames){while(li<locationFrames.length-1&&locationFrames[li+1].t<=t)li++;p=locationFrames[li];}
      return {t,speed:r.speed,gear:r.n_gear??null,throttle:r.throttle??null,brake:r.brake??null,rpm:r.rpm??null,drs:r.drs??null,x:p.x,y:p.y,progress:t/duration,soc:null};
    });
    return {name:filename,source:'imported',description:'Imported telemetry supplied by user; provenance not independently verified. '+(locationFrames?'Imported position samples.':'Car movement uses the schematic circuit.'),frames,points,duration,hasLocation:!!locationFrames,metadata:{driver:drivers.size?[...drivers][0]:null,session:sessions.size?[...sessions][0]:null,source:raw.metadata?.source||'User-supplied file',start:new Date(start).toISOString(),count:frames.length}};
  }
  function at(replay,time){const frames=replay.frames;let lo=0,hi=frames.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].t<=time)lo=mid;else hi=mid-1;}const a=frames[lo],b=frames[Math.min(lo+1,frames.length-1)];const q=b.t===a.t?0:clamp((time-a.t)/(b.t-a.t),0,1);return {...a,t:time,x:a.x+(b.x-a.x)*q,y:a.y+(b.y-a.y)*q,speed:a.speed+(b.speed-a.speed)*q,progress:a.progress+(b.progress-a.progress)*q};}
  return {POINTS,path,point,synthetic,parse,at,mapFor,pointsFor};
});
