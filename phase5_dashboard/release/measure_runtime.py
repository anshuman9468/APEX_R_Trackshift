#!/usr/bin/env python3
"""Measure warm frozen-advisory and complete-engine decision latency."""
from __future__ import annotations
import json, resource, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'RUNTIME_METRICS.json'

def main():
    sys.path.insert(0,str(ROOT))
    from backend.gnn_adapter import FrozenGNNAdapter
    rows={}
    for name, pref in [('cpu','cpu'),('cuda','cuda')]:
        a=FrozenGNNAdapter(pref)
        first=time.perf_counter(); r=a.predict(); load_ms=(time.perf_counter()-first)*1000
        if r.get('status')=='available':
            vals=[]
            for _ in range(10):
                t=time.perf_counter(); a.predict(); vals.append((time.perf_counter()-t)*1000)
            rows[name]={'status':'available','device':r.get('device'),'load_plus_first_inference_ms':load_ms,'warm_inference_ms_mean':sum(vals)/len(vals),'warm_inference_ms_max':max(vals)}
        else: rows[name]={'status':'unavailable','reason':r.get('reason'),'load_plus_first_inference_ms':load_ms}
    payload={'method':'compare','input':{'scenarioId':'patient','horizon':3,'seed':20260912,'judgeAction':'ATTACK'}}
    vals=[]
    for _ in range(5):
        t=time.perf_counter(); p=subprocess.run(['node',str(ROOT/'scripts/engine-cli.cjs')],input=json.dumps(payload),text=True,capture_output=True,cwd=ROOT,check=True); vals.append((time.perf_counter()-t)*1000)
    rows['complete_engine_decision_ms_mean']=sum(vals)/len(vals); rows['complete_engine_decision_ms_max']=max(vals)
    rows['process_peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    OUT.write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n'); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
