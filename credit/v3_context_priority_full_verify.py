"""Reexecute priority selection, fractional draws and independent references."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from .v3.experiment import digest, write
from .v3.factorial import characters, context_probabilities
from .v3.local import factory, stable_seed
from .v3_benchmark import CallLedger
from .v3_revalidation import RevalidationExecutor, candidate_memories
from .v3_online import load_ensemble
from .v3_optimization_verify import read, records, compare
from .v3_context_priority_full import prototypes, collect


def audit(output):
    torch.set_num_threads(1)
    protocol = read(output/'protocol.json')
    for name, h in protocol['source_sha256'].items():
        if digest(name) != h:
            raise ValueError('priority collector changed')
    for name, h in read(output/'manifest.json').items():
        if digest(output/name) != h:
            raise ValueError('priority artifact changed: '+name)
    source = Path(protocol['models_from'])
    summary, counts, model_hashes = [], dict(designs=0, reference_episodes=0, calls=0), {}
    pool = candidate_memories()
    effects = [()] + [(m.uid,) for m in pool] + [(pool[a].uid, pool[b].uid) for a in range(9) for b in range(a+1,9)]
    for seed, model_seed in zip(protocol['seeds'], (1201,1213,1217,1223)):
        sd = output/f'seed_{seed}'
        make = factory(seed)
        contexts = [make(f+'/source','priority-candidates',i) for f in ('books','travel','inventory','payments') for i in range(16)]
        model_dir = source/f'seed_{model_seed}/initial_models'
        for path in model_dir.glob('*.pt'):
            h = digest(path)
            if h != read(source/'completed_prefix_manifest.json')[str(path.relative_to(source))]:
                raise ValueError('priority source model changed')
            model_hashes[str(path)] = h
        models = load_ensemble(model_dir, contexts)
        frames, coefficients, costs = prototypes(models,contexts,seed)
        probability = context_probabilities(coefficients,costs,epsilon=.1)
        compare({'probabilities':probability.tolist(),'coefficients':coefficients.tolist(),
                 'replay_call_bounds':costs.tolist(),'contexts':[asdict(c) for c in contexts],
                 'frames':[[m.uid for m in f[0]] for f in frames]},read(sd/'priority.json'),'priority inputs')
        if not np.all(probability>0):
            raise ValueError('exploration probability is not positive')
        draws = {}
        for method in ('uniform','disagreement-per-call'):
            for budget in protocol['budgets']:
                name = f'{method}_{budget}'
                ledger = CallLedger({'factorial':8192,'replay_audit':128},budget)
                remaining, choices, estimates = list(range(64)), [], {}
                rng = np.random.default_rng(stable_seed(seed,method,budget))
                while remaining:
                    feasible = [j for j in remaining if ledger.spent+costs[j]<=budget]
                    if not feasible:
                        break
                    weights = np.ones(len(feasible)) if method=='uniform' else probability[feasible]
                    weights = weights/weights.sum()
                    j = int(rng.choice(feasible,p=weights))
                    choices.append({'index':j,'conditional_probability':float(weights[feasible.index(j)]),'feasible':feasible})
                    estimates[j] = collect(contexts[j],frames[j][0],frames[j][1],ledger,stable_seed(seed,method,budget,j))
                    remaining.remove(j)
                compare({'choices':choices,'beta':{str(k):v.tolist() for k,v in estimates.items()},'costs':ledger.report()},read(sd/(name+'.json')),'priority choices and estimates')
                expected = json.loads(json.dumps(ledger.records,default=lambda v: asdict(v)))
                actual = list(records(sd/(name+'.jsonl.gz')))
                compare(expected,actual,'physical fractional traces')
                counts['designs'] += len(choices)
                counts['calls'] += ledger.spent
                draws[name] = estimates,ledger.spent
        reference, reference_rows, reference_calls = [], [], 0
        executor = RevalidationExecutor()
        for j,c in enumerate(contexts):
            frame,fraction,cube = frames[j]
            design,values = characters(cube,fraction.effects),[]
            for repeat in range(4):
                checkpoint = replace(c,uid=c.uid+f':reference-{repeat}',seed=stable_seed(c.seed,'reference',repeat))
                outcomes = []
                for row in cube:
                    exposure = tuple(m.uid for m,s in zip(frame,row) if s==1)
                    e = executor.run(checkpoint,exposure)
                    outcomes.append(e.outcome)
                    reference_calls += e.calls
                    reference_rows.append({'context':j,'repeat':repeat,'exposure':list(exposure),'outcome':e.outcome,'calls':e.calls,'state_hash':e.state_hash})
                values.append(np.asarray(outcomes)@design/len(cube))
            reference.append(np.mean(values,axis=0))
        compare({'beta':[v.tolist() for v in reference],'calls':reference_calls,'execution_rows':reference_rows},read(sd/'reference.json'),'independent execution reference')
        counts['reference_episodes'] += len(reference_rows)
        counts['calls'] += reference_calls
        for name,(estimates,calls) in draws.items():
            error = []
            for j,(frame,fraction,_) in enumerate(frames):
                prior = []
                for effect in fraction.effects[1:]:
                    key = tuple(sorted(frame[k].uid for k in effect))
                    k = next(k for k,v in enumerate(effects) if tuple(sorted(v))==key)
                    prior.append(coefficients[:,j,k].mean())
                value = estimates[j][1:] if j in estimates else np.asarray(prior)
                error.extend(value-reference[j][1:])
            method,budget = name.rsplit('_',1)
            summary.append({'seed':seed,'method':method,'budget':int(budget),'actual_calls':calls,
                            'contexts_selected':len(estimates),'rmse':float(np.sqrt(np.mean(np.square(error))))})
        print('verified priority diagnostic',seed,flush=True)
    compare(summary,read(output/'summary.json'),'priority RMSE')
    write(output.with_name(output.name+'_audit.json'),{'verified':True,'counts':counts,
          'auditor_sha256':digest(__file__),'collector_sources':protocol['source_sha256'],
          'model_hashes':model_hashes,'manifest_sha256':digest(output/'manifest.json'),
          'randomization_provenance':'Recorded random sign draws and conditional probabilities reconstructed. Episode.randomized retains its legacy False default; these diagnostic episodes are never used for training.',
          'limits':'Four source snapshot seeds; reference Monte Carlo error; no optimizer retraining or general population claim.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output)
