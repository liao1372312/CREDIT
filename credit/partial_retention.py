"""Deploy reachable-path support in finite retention; fresh paired execution noise.

Missing evidence is controlled deletion, not an observed provider change.
Known-path objectives remain empirical, without uniform kernel certificates.
"""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from . import budget_grid_v2 as grid
from .budget_grid_verify import restore
from .budget_mechanisms_v2 import BankSearch,BatchForecast
from .experiments import _write_csv
from .paired_statistics import summarize
from .partial_support import PartialSupportForecast
from .policy import _worst_mixture
from .stateful import retrieve,run_episode
from .support_stress import masked
from .user_data import digest

CONDITIONS=('complete','delete_unreachable_states','missing_25pct','missing_submit_role')
EXECUTION_SEEDS=(211,223,227,229,233)


class PartialBankSearch(BankSearch):
    def evaluate(self,banks):
        pending=sorted({tuple(sorted(b)) for b in banks if self.feasible(b)}-set(self.cache))
        if not pending:return
        values,unknown=[],[]
        for t in self.contexts:
            prediction=self.forecast.predict(t,[retrieve(t,b,self.depth) for b in pending])
            values.append(prediction['known_success']);unknown.append(prediction['unknown_mass'])
        for bank,known,unidentified in zip(pending,np.asarray(values).T,np.asarray(unknown).T):
            lower=np.maximum(0,known-self.forecast.model.calibration_error)
            cost=self.token_cost*sum(self.tokens[m] for m in bank)
            self.cache[bank]={'mean':float(known.mean()),'coverage':float(np.mean(unidentified<=1e-10)),
                'unknown_mass':float(unidentified.mean()),
                'objective':_worst_mixture(lower.tolist(),[1.]*len(lower),self.radius)-cost,
                'mean_objective':_worst_mixture(known.tolist(),[1.]*len(lower),self.radius)-cost}


def build(grid_root,seed,condition):
    protocol=grid.read(grid_root/'frozen_protocol.json')
    history=Path(protocol['history_directory'])/f"seed_{protocol['historical_seed_mapping'][str(seed)]}"
    parent=grid_root/f'seed_{seed}'/'C2048_K8'
    probes,_=grid.prefix(grid.read(parent/'probes.json'),64)
    policy,api=grid.augmented(grid.read(history/'decision_observations.json'),grid.read(history/'api_observations.json'),probes)
    dump=grid.read(parent/'R64_d2'/'fitted_policy.json')
    model=restore(policy,masked(api,condition,seed),dump)
    model.calibration_error=dump['calibration_error']
    contexts=[grid.task(t) for t in protocol['future_contexts'][str(seed)]]
    tokens={m['uid']:m['tokens'] for m in grid.read(grid_root/'memories.json')}
    return model,contexts,tokens


def plan(grid_root):
    protocol=grid.read(grid_root/'frozen_protocol.json');plans=[]
    for seed,execution_seed in zip(protocol['seeds'],EXECUTION_SEEDS):
        for condition in CONDITIONS:
            model,contexts,tokens=build(grid_root,seed,condition)
            coarse=BankSearch(BatchForecast(model),contexts,tokens,2048,8)
            partial=PartialBankSearch(PartialSupportForecast(model),contexts,tokens,2048,8)
            bank_a,audit_a=coarse.joint();bank_b,audit_b=partial.joint()
            plans.append({'historical_model_seed':seed,'execution_seed':execution_seed,'condition':condition,
                'banks':{'CREDIT-coarse-support':list(bank_a),'CREDIT-reachable-support':list(bank_b)},
                'search_audit':{'coarse':audit_a,'reachable':audit_b},
                'selection_forecasts':{'coarse':coarse.cache[bank_a],'reachable':partial.cache[bank_b]}})
    return plans


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',type=Path,default=Path('results/budget_grid_v2'))
    parser.add_argument('--output',type=Path,default=Path('results/partial_retention_v1'))
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:
        protocol=grid.read(args.output/'frozen_protocol.json')
        for name,expected in protocol['source_sha256'].items():
            if digest(Path(__file__).with_name(name))!=expected:raise ValueError('source changed: '+name)
        for name,expected in protocol['input_sha256'].items():
            if digest(Path(name))!=expected:raise ValueError('input changed: '+name)
        args.grid=Path(protocol['grid_directory'])
        for name,expected in grid.read(args.output/'evaluation_freeze.json').items():
            if digest(args.output/name)!=expected:raise ValueError('evaluation freeze changed')
        rebuilt=plan(args.grid);saved=grid.read(args.output/'frozen_plans.json')
        if rebuilt!=saved:raise ValueError('retention choices differ')
        index={(p['execution_seed'],p['condition']):p for p in saved}
        contexts={s:{t.uid:t for t in grid.future_tasks(s,4)[1]} for s in EXECUTION_SEEDS}
        cache={};count=0;seen=set()
        tokens={m['uid']:m['tokens'] for m in grid.read(args.grid/'memories.json')}
        rows=grid.csv_rows(args.output/'raw_episodes.csv')
        for row in rows:
            s=int(row['seed']);t=contexts[s][row['task']]
            bank=index[s,row['condition']]['banks'][row['method']]
            exposure=retrieve(t,bank,8);key=s,t.uid,exposure
            if key not in cache:cache[key]=run_episode(t,exposure)
            trace=cache[key];paired=(s,row['condition'],row['method'],t.uid)
            if paired in seen:raise ValueError('duplicate outcome')
            seen.add(paired)
            if row['bank']!='|'.join(bank) or trace.state_hash!=row['state_sha256'] or trace.outcome!=int(row['success']):
                raise ValueError('execution/retrieval differs')
            if int(row['bank_tokens'])!=sum(tokens[m] for m in bank) or int(row['api_calls'])!=trace.calls:raise ValueError('cost differs')
            count+=1
        if count!=5*len(CONDITIONS)*2*96:raise ValueError('coverage differs')
        for condition in CONDITIONS:
            actual=[{**r,'seed':int(r['seed']),'success':int(r['success'])} for r in rows if r['condition']==condition]
            table,paired=summarize(actual,'CREDIT-reachable-support',repeats=2000,random_seed=433)
            if {'main':table,'paired':paired}!=grid.read(args.output/f'statistics_{condition}.json'):raise ValueError('statistics differ')
        grid.write(args.output/'execution_audit.json',{'verified':True,'retention_plans_recomputed':len(saved),
            'all_outcomes_reconstructed':count,'unique_counterfactuals_reexecuted':len(cache),
            'fresh_execution_noise_seeds':EXECUTION_SEEDS,'full_paper_CREDIT':False,
            'scope':'known finite kernels and controlled evidence deletion; no provider generalization',
            'verifier_source_sha256':digest(Path(__file__))})
        print('Verified partial-support retention and execution');return
    if args.output.exists():raise FileExistsError('use a new output directory')
    if not grid.read(args.grid/'execution_audit.json')['verified']:raise ValueError('grid audit required')
    args.output.mkdir();(args.output/'source').mkdir()
    names=['partial_retention.py','partial_support.py','support_stress.py','budget_grid_v2.py',
        'budget_grid_verify.py','budget_mechanisms_v2.py','stateful.py','mechanisms.py','shared_mechanisms.py','paired_statistics.py']
    source_inputs={str(p):digest(p) for p in args.grid.rglob('*.json') if p.name in
        ('frozen_protocol.json','fitted_policy.json','probes.json','execution_audit.json','memories.json')}
    source_inputs.update(grid.read(args.grid/'frozen_protocol.json')['history_sha256'])
    protocol={'scope':'finite partial-support retention correction','conditions':CONDITIONS,
        'execution_seeds':EXECUTION_SEEDS,'C':2048,'K':8,'R_history':64,'degree':2,
        'grid_directory':str(args.grid.resolve()),'input_sha256':source_inputs,
        'source_sha256':{n:digest(Path(__file__).with_name(n)) for n in names},
        'selection':'known-path empirical success minus past calibration error, robust mixture and token cost; unknown-success mass contributes zero',
        'phase':'all plans frozen before new execution noise outcomes; masks are controlled evidence loss',
        'calibration':'previous fully observed validation, no uniform certificate after masking',
        'no_new_training_calls':True,'provider_drift':False}
    grid.write(args.output/'frozen_protocol.json',protocol)
    for n in names:(args.output/'source'/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    plans=plan(args.grid);grid.write(args.output/'frozen_plans.json',plans)
    grid.write(args.output/'evaluation_freeze.json',{n:digest(args.output/n) for n in ('frozen_protocol.json','frozen_plans.json')})
    tokens={m['uid']:m['tokens'] for m in grid.read(args.grid/'memories.json')}
    rows=[]
    for p in plans:
        for method,bank in p['banks'].items():
            for t in grid.future_tasks(p['execution_seed'],4)[1]:
                exposure=retrieve(t,bank,8);trace=run_episode(t,exposure)
                rows.append({'seed':p['execution_seed'],'condition':p['condition'],'method':method,'task':t.uid,
                    'family':t.family,'epoch':t.epoch,'topology':t.template,'success':trace.outcome,
                    'bank':'|'.join(bank),'bank_tokens':sum(tokens[m] for m in bank),
                    'exposure':'|'.join(exposure),'api_calls':trace.calls,'state_sha256':trace.state_hash})
    _write_csv(args.output/'raw_episodes.csv',rows)
    summary=[]
    for condition in CONDITIONS:
        subset=[r for r in rows if r['condition']==condition]
        table,paired=summarize(subset,'CREDIT-reachable-support',repeats=2000,random_seed=433)
        grid.write(args.output/f'statistics_{condition}.json',{'main':table,'paired':paired})
        summary.extend({**r,'condition':condition} for r in table)
    grid.write(args.output/'summary.json',{'table':summary,'episodes':len(rows),
        'new_execution_tool_calls':sum(r['api_calls'] for r in rows),'full_paper_CREDIT':False,
        'confidence_certificate':False,'limits':protocol['scope']})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
