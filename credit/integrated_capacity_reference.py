"""Post-hoc CREDIT-only capacity reanalysis of the inspected 13-note holdout.

This consumes frozen development scores, previously executed test sessions,
and a small supplemental set of missing exposure/seed sessions. It is not
confirmatory evidence for the expanded protocol.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json
from pathlib import Path
import numpy as np

ROOT=Path('results/credit_experiment_tables_v1/public_cv20_v1')
CAPACITY=(81,162,243,324)
SUPPLEMENTAL=Path('results/credit_integrated_capacity_reference_credit_only_v2/supplemental_test_sessions.jsonl')

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def best_bank(pool,scores,cap):
    best=(0.0,0,())
    for mask in range(1<<len(pool)):
        ids=tuple(sorted(pool[i]['uid'] for i in range(len(pool)) if mask>>i&1))
        tokens=sum(pool_by_uid[u]['tokens'] for u in ids)
        if tokens>cap: continue
        utility=sum(scores.get(u,0.0) for u in ids)
        # Maximize frozen development CREDIT score; ties prefer less storage,
        # then stable UID ordering. Empty bank is always feasible.
        cand=(utility,-tokens,tuple(reversed(ids)))
        if cand>best: best=cand
    ids=tuple(reversed(best[2]))
    return ids,sum(pool_by_uid[u]['tokens'] for u in ids),float(best[0])


def run(output:Path):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    folds={}
    for fold in (0,1):
        base=ROOT/f'fold{fold}_run'
        pool=read(base/'memory_pool.json')
        global pool_by_uid
        pool_by_uid={m['uid']:m for m in pool}
        scores=read(base/'selection_diagnostics.json')['main::Full CREDIT']['scores']
        policies=read(base/'frozen_policies.json')['main::Full CREDIT']
        sessions=read(base/'test_physical_sessions.json')
        if SUPPLEMENTAL.exists():
            sessions.extend(json.loads(line) for line in SUPPLEMENTAL.read_text().splitlines()
                            if line.strip() and
                            json.loads(line).get('label', '').startswith(f'posthoc-f{fold}-'))
        lookup={}
        for row in sessions:
            key=(row['task'],int(row['actor_seed']),tuple(sorted(row['exposure'])))
            lookup[key]=row
        folds[str(fold)]={}
        for cap in CAPACITY:
            retained,used,objective=best_bank(pool,scores,cap)
            logical=[]; missing=[]; query_stats=[]
            for task,bank in policies.items():
                restricted=tuple(sorted(set(bank)&set(retained)))
                query_stats.append({'task':task,'bank':list(restricted),
                    'text_tokens':sum(pool_by_uid[u]['tokens'] for u in restricted)})
                for seed in (4103,4127,4133,4153,4157,4177):
                    row=lookup.get((task,seed,restricted))
                    if row is None:
                        missing.append({'task':task,'seed':seed,'exposure':list(restricted)})
                    else:
                        logical.append({'task':task,'family':row['family'],'seed':seed,
                            'success':row['success'],'native_score':row['native_score'],
                            'physical_calls':row['physical_calls'],'exposure':list(restricted)})
            folds[str(fold)][str(cap)]={'retained_uids':list(retained),'retained_count':len(retained),
                'retained_tokens':used,'capacity_tokens':cap,'capacity_occupancy_pct':100*used/cap,
                'development_objective':objective,'query_stats':query_stats,'sessions':logical,
                'missing_test_exposures':missing}
    # Aggregate only complete capacity cells across the two inspected folds.
    summary=[]
    for cap in CAPACITY:
        block=[folds[str(f)][str(cap)] for f in (0,1)]
        missing=sum(len(x['missing_test_exposures']) for x in block)
        sessions=[s for x in block for s in x['sessions']]
        if missing or len(sessions)!=240:
            raise RuntimeError(f'cap {cap}: missing={missing}, sessions={len(sessions)}')
        summary.append({'cap_tokens':cap,'sessions':len(sessions),
            'successes':sum(s['success'] for s in sessions),
            'vsr_pct':100*sum(s['success'] for s in sessions)/len(sessions),
            'mean_used_tokens':float(np.mean([x['retained_tokens'] for x in block])),
            'used_capacity_pct_mean_across_folds':float(np.mean([x['capacity_occupancy_pct'] for x in block])),
            'fold0_tokens':block[0]['retained_tokens'],'fold1_tokens':block[1]['retained_tokens'],
            'fold0_retained':block[0]['retained_uids'],'fold1_retained':block[1]['retained_uids']})
    supplemental_rows = ([json.loads(line) for line in SUPPLEMENTAL.read_text().splitlines()
                          if line.strip()] if SUPPLEMENTAL.exists() else [])
    supplemental_generations = sum(len(row.get('generations', [])) for row in supplemental_rows)
    supplemental_retries = sum(sum(int(turn.get('network_attempts', 1))
                                   for turn in row.get('generations', []))
                               for row in supplemental_rows)
    protocol={'status':'post-hoc CREDIT-only reanalysis; inspected test holdout; not confirmatory',
      'source_results':str(ROOT.resolve()),'method':'Exact 13-note knapsack maximizing summed frozen development CREDIT scores; tie-break by lower storage then UID; retrieval intersects each task-specific frozen CREDIT bank with retained global bank.',
      'capacities_tokens':list(CAPACITY),'memory_pool_size':13,'memory_text_tokens':324,
      'existing_test_sessions_reused':True,
      'supplemental_test_episodes':len(supplemental_rows),
      'supplemental_model_generations':supplemental_generations,
      'supplemental_http_attempts':supplemental_retries,
      'supplemental_tool_calls':sum(int(row.get('tool_calls', 0)) for row in supplemental_rows),
      'supplemental_test_sessions':str(SUPPLEMENTAL.resolve()) if SUPPLEMENTAL.exists() else None,
      'supplemental_sessions_sha256':digest(SUPPLEMENTAL) if SUPPLEMENTAL.exists() else None,
      'test_seeds':[4103,4127,4133,4153,4157,4177],
      'source_hashes':{str(ROOT/f'fold{f}_run'/n):digest(ROOT/f'fold{f}_run'/n) for f in (0,1) for n in ('memory_pool.json','selection_diagnostics.json','frozen_policies.json','test_physical_sessions.json')}}
    write(output/'frozen_protocol.json',protocol)
    write(output/'fold_details.json',folds)
    write(output/'summary.json',{'protocol_status':protocol['status'],'capacity_results':summary})
    print(json.dumps({'protocol_status':protocol['status'],'capacity_results':summary},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
