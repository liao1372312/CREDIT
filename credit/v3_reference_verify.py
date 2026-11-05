"""Reexecute every reference coalition and check oracle/model error tables."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from .v3.audit import load_context, check_close
from .v3.experiment import digest, write
from .v3.local import StatefulExecutor, stable_seed, memories
from .v3.factorial import admitted_pairs
from .v3_reference import coalitions, credits, kernel_probability, load_models


def audit(dataset,benchmark,output):
    read=lambda p:json.loads(p.read_text())
    for name,value in read(output/'artifact_manifest.json').items():
        if digest(output/name)!=value:raise ValueError('reference artifact '+name)
    protocol=read(dataset/'protocol.json');summary=read(output/'summary.json')
    if digest(dataset/'contexts.json')!=protocol['contexts_sha256'] or digest('credit/v3_reference.py')!=protocol['source_sha256']:
        raise ValueError('reference freeze')
    if digest(benchmark/'artifact_manifest.json')!=summary['benchmark_manifest_sha256']:raise ValueError('benchmark link')
    cases=[load_context(c) for c in read(dataset/'contexts.json')];pool=memories();bits,z=coalitions(8)
    exposures=[tuple(m.uid for m,b in zip(pool,row) if b) for row in bits]
    # NpzFile extracts an entire compressed member on EACH key access. Load
    # once before the inner execution loop, rather than repeatedly decompressing
    # the 16 MB state-hash member hundreds of thousands of times.
    with np.load(output/'coalitions.npz') as archive:
        data={name:archive[name] for name in archive.files}
    engine=StatefulExecutor();calls=0;episodes=0
    labels=Counter({label:0 for label in ('ok','agent','service','amb')});control_calls=0
    for i,c in enumerate(cases):
        check_close([kernel_probability(c,a) for a in exposures],data['truth'][i],'kernel oracle')
        for rep in range(protocol['replications']):
            task=replace(c,seed=stable_seed(977,'reference-noise',c.uid,rep))
            for j,a in enumerate(exposures):
                e=engine.run(task,a)
                if e.outcome!=data['outcomes'][i,rep,j] or e.calls!=data['calls'][i,rep,j] or e.state_hash.encode()!=data['state_hashes'][i,rep,j]:
                    raise ValueError('actual reference execution')
                if e.checkpoint_hash!=e.restored_hash or not e.replay_valid:raise ValueError('reference safety')
                labels.update(o.label for o in e.observations);calls+=e.calls;episodes+=1
            a=exposures[(i*31+rep*7)%256]
            es=[engine.run(task,b) for b in (a,a,a+('noop',))]
            if len({e.state_hash for e in es})!=1:raise ValueError('negative controls')
            control_calls+=sum(e.calls for e in es)
        print('reference audit',i,flush=True)
    if dict(labels)!=summary['contract_labels'] or calls!=summary['reference_api_calls'] or control_calls!=summary['negative_control_api_calls']:
        raise ValueError('reference accounting')
    torch.set_num_threads(1)
    predictions=np.load(output/'model_predictions.npz')['predictions'];errors=read(output/'model_errors.json')
    for i,sd in enumerate(sorted(benchmark.glob('seed_*'))):
        values=np.mean([m.predict_many([(c,a) for c in cases for a in exposures]) for m in load_models(sd,cases)],axis=0).reshape(data['truth'].shape)
        check_close(values,predictions[i],'saved model prediction')
        seed=int(sd.name.split('_')[1])
        for j,c in enumerate(cases):
            row=next(r for r in errors if r['seed']==seed and r['stratum']==c.stratum and r['depth']==len(c.nodes))
            effects,beta,bz,sp=credits(values[j],z);_,tb,tbz,tsp=credits(data['truth'][j],z)
            check_close(np.mean(np.abs(values[j]-data['truth'][j])),row['utility_mae'],'utility MAE')
            check_close(np.mean(np.abs(bz-tbz)),row['banzhaf_mae'],'Banzhaf MAE')
            check_close(np.mean(np.abs(sp-tsp)),row['shapley_mae'],'Shapley MAE')
            cols=[effects.index(p) for p in admitted_pairs(c,pool)]
            actual=np.abs(tb[cols])>=protocol['pair_threshold'];pred=np.abs(beta[cols])>=protocol['pair_threshold']
            denominator=int(actual.sum()+pred.sum());f1=None if denominator==0 else 2*np.sum(actual & pred)/denominator
            if f1!=row['pair_f1']:raise ValueError('interaction F1')
    report={'verified':True,'reference_episodes_reexecuted':episodes,'reference_api_calls_reexecuted':calls,
            'negative_control_episodes_reexecuted':summary['negative_control_episodes'],
            'negative_control_api_calls_reexecuted':control_calls,'oracle_coalitions_recomputed':int(data['truth'].size),
            'model_error_rows_recomputed':len(errors),'auditor_sha256':digest(__file__),
            'limits':'reference-domain regret and experimental-design summaries hashed; not independently re-derived here'}
    write(output/'execution_audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('datasets/credit_v3_reference_v1'))
    p.add_argument('--benchmark',type=Path,default=Path('results/credit_v3_benchmark_v1'))
    p.add_argument('--results',type=Path,default=Path('results/credit_v3_reference_v1'))
    a=p.parse_args();audit(a.dataset,a.benchmark,a.results)
