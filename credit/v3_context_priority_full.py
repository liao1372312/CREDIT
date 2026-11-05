"""Execute complete Eq.probepriority variance, including the intercept.

This diagnostic does not change the frozen chronological experiment. It tests
the context-priority primitive under limited physical call budgets. Reference
coalitions are collected only after selections; their outcomes never rank tasks.
"""
import argparse
from dataclasses import asdict,replace
from itertools import product
import json
from pathlib import Path

import numpy as np

from .v3.experiment import digest,write
from .v3.factorial import admitted_pairs,eligible_slots,regular_fraction,characters,estimate,context_probabilities
from .v3.local import factory,ranked,stable_seed
from .v3.types import reserved_slots
from .v3_benchmark import CallLedger,stream_records
from .v3_revalidation import candidate_memories,RevalidationExecutor
from .v3_online import load_ensemble


def prototypes(models,contexts,seed):
    pool=candidate_memories();frames=[];coefficients=np.zeros((len(models),len(contexts),46));costs=[]
    for j,c in enumerate(contexts):
        eligible=[m for m in pool if any(m.eligible(n.role) for n in c.nodes)]
        k1=min(4,len(ranked(c,tuple(m.uid for m in eligible))),max(len(eligible)-1,0));k2=min(2,len(eligible)-k1)
        frame=eligible_slots(c,pool,ranked(c,tuple(m.uid for m in eligible)),k1,k2,np.random.default_rng(stable_seed(seed,'prototype',c.uid)))
        fraction=regular_fraction(len(frame),admitted_pairs(c,frame),seed=stable_seed(c.uid))
        cube=np.asarray(list(product((-1,1),repeat=len(frame))),dtype=np.int8)
        effects=fraction.effects;design=characters(cube,effects)
        items=[(c,tuple(m.uid for m,sign in zip(frame,row) if sign==1)) for row in cube]
        global_effects=[()]+[(m.uid,) for m in pool]+[(pool[a].uid,pool[b].uid) for a in range(9) for b in range(a+1,9)]
        for i,model in enumerate(models):
            beta=model.predict_many(items)@design/len(cube)
            for local,effect in enumerate(effects):
                key=tuple(sorted(frame[k].uid for k in effect));index=next(k for k,v in enumerate(global_effects) if tuple(sorted(v))==key)
                coefficients[i,j,index]=beta[local]
        frames.append((frame,fraction,cube));costs.append((fraction.runs+1)*2*len(c.nodes))
    return frames,coefficients,np.asarray(costs,dtype=float)


def collect(c,frame,fraction,ledger,seed):
    executor=RevalidationExecutor();rng=np.random.default_rng(seed)
    checkpoint=replace(c,uid=c.uid+':priority-draw',seed=stable_seed(c.seed,'priority-draw'))
    rows=fraction.draw(rng);ys=[];reference=None
    for row in rows:
        slots=reserved_slots(frame,row)
        if reference is None:
            e=ledger.execute('factorial',executor,checkpoint,slots.exposure,slots=slots);reference=e
        else:
            if ledger.spent+2*len(c.nodes)>ledger.call_cap:raise ValueError('priority replay reserve')
            e=executor.replay(checkpoint,slots.exposure,reference,slots=slots);ledger.charge('factorial',e)
        if not e.replay_valid:raise ValueError('unsafe priority draw')
        ys.append(e.outcome)
    full=ledger.execute('replay_audit',executor,checkpoint,e.exposure)
    if (full.events,full.observations,full.state_hash)!=(e.events,e.observations,e.state_hash):raise ValueError('priority replay differs')
    return estimate(fraction,[rows],[ys])['beta']


def run(source,output):
    if output.exists():raise FileExistsError('fresh diagnostic directory required')
    output.mkdir();seeds=(1501,1511,1523,1531);budgets=(512,1024,2048);summary=[]
    write(output/'protocol.json',{'seeds':seeds,'budgets':budgets,'epsilon':.1,'contexts_per_seed':64,
        'models_from':str(source.resolve()),'reference':'4 independent full-cube execution draws;collected after all choices',
        'scope':'source-version finite adapter diagnostic;separate from chronological confirming tests',
        'priority_variant':'full variance including constant,unary,pair;accuracy metric remains nonconstant attribution RMSE',
        'development_basis':'Manuscript equation completeness;fresh seeds,all outcomes retained;not a retune of the completed main experiment',
        'source_sha256':{str(Path(__file__)):digest(__file__)},'primary':'hybrid all-context coefficient RMSE,unprobed cells retain same model prior',
        'limits':'execution reference has Monte Carlo error;four seeds;no population superiority claim'})
    pool=candidate_memories()
    for seed,model_seed in zip(seeds,(1201,1213,1217,1223)):
        make=factory(seed);contexts=[make(family+'/source','priority-candidates',i) for family in ('books','travel','inventory','payments') for i in range(16)]
        models=load_ensemble(source/f'seed_{model_seed}/initial_models',contexts)
        frames,coefficients,costs=prototypes(models,contexts,seed)
        probability=context_probabilities(coefficients,costs,epsilon=.1)
        choices={};draws={}
        sd=output/f'seed_{seed}';sd.mkdir()
        write(sd/'priority.json',{'probabilities':probability.tolist(),'coefficients':coefficients.tolist(),'replay_call_bounds':costs.tolist(),
            'contexts':[asdict(c) for c in contexts],'frames':[[m.uid for m in f[0]] for f in frames]})
        for method in ('uniform','disagreement-per-call'):
            for budget in budgets:
                ledger=CallLedger({'factorial':8192,'replay_audit':128},budget)
                remaining=list(range(len(contexts)));selected=[];estimates={};rng=np.random.default_rng(stable_seed(seed,method,budget))
                while remaining:
                    feasible=[j for j in remaining if ledger.spent+costs[j]<=budget]
                    if not feasible:break
                    weights=np.ones(len(feasible)) if method=='uniform' else probability[feasible]
                    weights=weights/weights.sum();j=int(rng.choice(feasible,p=weights))
                    selected.append({'index':j,'conditional_probability':float(weights[feasible.index(j)]),'feasible':feasible})
                    estimates[j]=collect(contexts[j],frames[j][0],frames[j][1],ledger,stable_seed(seed,method,budget,j))
                    remaining.remove(j)
                name=f'{method}_{budget}';choices[name]=selected;draws[name]=(estimates,ledger.spent)
                write(sd/(name+'.json'),{'choices':selected,'beta':{str(k):v.tolist() for k,v in estimates.items()},'costs':ledger.report()})
                stream_records(sd/(name+'.jsonl.gz'),ledger.records)
        # Reference is an independent physical execution dataset, never model output.
        reference=[];reference_calls=0;reference_rows=[];executor=RevalidationExecutor()
        for j,c in enumerate(contexts):
            frame,fraction,cube=frames[j];design=characters(cube,fraction.effects);values=[]
            for repeat in range(4):
                checkpoint=replace(c,uid=c.uid+f':reference-{repeat}',seed=stable_seed(c.seed,'reference',repeat))
                outcomes=[]
                for row in cube:
                    a=tuple(m.uid for m,s in zip(frame,row) if s==1);e=executor.run(checkpoint,a)
                    outcomes.append(e.outcome);reference_calls+=e.calls
                    reference_rows.append({'context':j,'repeat':repeat,'exposure':list(a),'outcome':e.outcome,'calls':e.calls,'state_hash':e.state_hash})
                values.append(np.asarray(outcomes)@design/len(cube))
            reference.append(np.mean(values,axis=0))
        write(sd/'reference.json',{'beta':[v.tolist() for v in reference],'calls':reference_calls,'execution_rows':reference_rows})
        for name,(estimates,calls) in draws.items():
            error=[]
            for j,(frame,fraction,cube) in enumerate(frames):
                prior=[]
                for effect in fraction.effects[1:]:
                    key=tuple(sorted(frame[k].uid for k in effect))
                    effects=[()]+[(m.uid,) for m in pool]+[(pool[a].uid,pool[b].uid) for a in range(9) for b in range(a+1,9)]
                    index=next(k for k,v in enumerate(effects) if tuple(sorted(v))==key);prior.append(coefficients[:,j,index].mean())
                value=estimates[j][1:] if j in estimates else np.asarray(prior)
                error.extend(value-reference[j][1:])
            method,budget=name.rsplit('_',1)
            summary.append({'seed':seed,'method':method,'budget':int(budget),'actual_calls':calls,
                'contexts_selected':len(estimates),'rmse':float(np.sqrt(np.mean(np.square(error))))})
        print('priority diagnostic scored',seed,flush=True)
    write(output/'summary.json',summary)
    write(output/'manifest.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
