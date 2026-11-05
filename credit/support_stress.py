"""Evaluate partial-support propagation under frozen, controlled input deletions."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from . import budget_grid_v2 as grid
from .budget_grid_verify import restore
from .budget_mechanisms_v2 import BatchForecast
from .experiments import _write_csv
from .partial_support import PartialSupportForecast
from .stateful import oracle_probability,retrieve
from .user_data import digest


CONDITIONS = ('complete','delete_unreachable_states','missing_25pct','missing_50pct','missing_75pct',
    'heldout_payments_family','heldout_target_epoch','missing_submit_role')


def masked(api,condition,seed):
    rows=[]
    for row in api:
        epoch,family,role,state,action=row['key']
        u=int.from_bytes(sha256(json.dumps([seed,row['key'],'support-loss-v1']).encode()).digest()[:8],'big')/2**64
        remove = (condition=='delete_unreachable_states' and state in (2,4,5,6))
        remove |= condition=='heldout_payments_family' and family=='payments'
        remove |= condition=='heldout_target_epoch' and epoch=='target'
        remove |= condition=='missing_submit_role' and role=='submit'
        for name,rate in [('missing_25pct',.25),('missing_50pct',.5),('missing_75pct',.75)]:
            remove |= condition==name and u<rate
        if not remove:
            rows.append(deepcopy(row))
    return rows


def compute(root):
    p=grid.read(root/'frozen_protocol.json'); source=Path(p['grid_directory'])
    protocol=grid.read(source/'frozen_protocol.json'); history_root=Path(protocol['history_directory'])
    rows=[]
    for seed in protocol['seeds']:
        history=history_root/f"seed_{protocol['historical_seed_mapping'][str(seed)]}"
        parent=source/f'seed_{seed}'/'C2048_K8'
        selected,_=grid.prefix(grid.read(parent/'probes.json'),64)
        policy,api=grid.augmented(grid.read(history/'decision_observations.json'),grid.read(history/'api_observations.json'),selected)
        cell=parent/'R64_d2';dump=grid.read(cell/'fitted_policy.json')
        banks=grid.read(cell/'frozen_banks.json');contexts=[grid.task(t) for t in protocol['future_contexts'][str(seed)]]
        bank=banks['CREDIT-finite-empirical']
        full=PartialSupportForecast(restore(policy,api,dump))
        for condition in CONDITIONS:
            reduced=masked(api,condition,seed);model=restore(policy,reduced,dump)
            partial=PartialSupportForecast(model);coarse=BatchForecast(model)
            for t in contexts:
                exposure=retrieve(t,bank,8)
                result=partial.predict(t,[exposure]);_,supported=coarse.predict(t,[exposure])
                truth=oracle_probability(t,exposure)
                original=full.predict(t,[exposure])['known_success'][0]
                lower=result['known_success'][0];upper=result['conditional_upper'][0]
                if not (-1e-10 <= lower <= original+1e-10 <= upper+1e-10 <= 1+1e-10):
                    raise ValueError('partial support fails fitted-kernel envelope')
                rows.append({'seed':seed,'condition':condition,'context':t.uid,'family':t.family,'epoch':t.epoch,
                    'topology':t.template,'removed_api_input_fraction':1-len(reduced)/len(api),
                    'coarse_complete_input_coverage':bool(supported[0]),
                    'reachable_path_coverage':bool(result['complete_path_coverage'][0]),
                    'known_success_forecast':lower,'unknown_path_mass':result['unknown_mass'][0],
                    'conditional_upper':upper,'full_fitted_probability':original,'exact_evaluation_only':truth,
                    'empirical_envelope_contains_truth':lower-1e-10 <= truth <= upper+1e-10,
                    'scope':'conditional model envelope; no confidence certificate'})
    summary=[]
    for condition in CONDITIONS:
        subset=[r for r in rows if r['condition']==condition]
        summary.append({'condition':condition,'contexts':len(subset),
            'mean_removed_input_fraction':float(np.mean([r['removed_api_input_fraction'] for r in subset])),
            'coarse_transfer_coverage':float(np.mean([r['coarse_complete_input_coverage'] for r in subset])),
            'reachable_transfer_coverage':float(np.mean([r['reachable_path_coverage'] for r in subset])),
            'mean_unknown_mass':float(np.mean([r['unknown_path_mass'] for r in subset])),
            'mean_known_success_forecast':float(np.mean([r['known_success_forecast'] for r in subset])),
            'empirical_envelope_truth_coverage':float(np.mean([r['empirical_envelope_contains_truth'] for r in subset]))})
    return rows,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',type=Path,default=Path('results/budget_grid_v2'))
    parser.add_argument('--output',type=Path,default=Path('results/support_stress_v1'))
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:
        p=grid.read(args.output/'frozen_protocol.json')
        for name,expected in p['source_sha256'].items():
            if digest(Path(__file__).with_name(name))!=expected: raise ValueError('source changed')
        for name,expected in p['input_sha256'].items():
            if digest(Path(name))!=expected: raise ValueError('input changed')
        rows,summary=compute(args.output)
        actual=grid.csv_rows(args.output/'raw_diagnostics.csv')
        if len(rows)!=len(actual): raise ValueError('coverage differs')
        for expected,saved in zip(rows,actual):
            for name,value in expected.items():
                if isinstance(value,(float,np.floating)):
                    if not np.isclose(float(saved[name]),value,atol=1e-12):raise ValueError('diagnostic differs')
                elif str(value)!=saved[name]:raise ValueError('diagnostic differs')
        if grid.read(args.output/'summary.json')['table']!=summary:raise ValueError('summary differs')
        grid.write(args.output/'execution_audit.json',{'verified':True,'diagnostics_recomputed':len(rows),
            'known_kernel_envelope_checks':len(rows),'confidence_certificate':False,'source_sha256':digest(Path(__file__))})
        print('Verified partial support diagnostics');return
    if args.output.exists():raise FileExistsError('use a new output directory')
    if not grid.read(args.grid/'execution_audit.json')['verified']:raise ValueError('grid audit required')
    sources=['support_stress.py','partial_support.py','budget_grid_verify.py','budget_grid_v2.py',
        'budget_mechanisms_v2.py','stateful.py','mechanisms.py','shared_mechanisms.py']
    args.output.mkdir();(args.output/'source').mkdir()
    source_inputs={str(p):digest(p) for p in args.grid.rglob('*.json') if p.name in
        ('frozen_protocol.json','fitted_policy.json','probes.json','frozen_banks.json','execution_audit.json')}
    p={'scope':'development diagnostic: controlled missing-input evidence, fixed previously scored banks',
        'grid_directory':str(args.grid.resolve()),'conditions':CONDITIONS,'target_seeds':[101,103,107,109,113],
        'source_sha256':{n:digest(Path(__file__).with_name(n)) for n in sources},'input_sha256':source_inputs,
        'public_reachable_states':[0,1,3,7],'mask_assignment':'nested hash deletion independent of future outcomes',
        'known_path_envelope':'conditions on fitted known kernels; no estimation confidence bound',
        'no_bank_refitting':True,'no_new_service_calls':True,'provider_drift':False}
    grid.write(args.output/'frozen_protocol.json',p)
    for n in sources:(args.output/'source'/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    rows,summary=compute(args.output)
    _write_csv(args.output/'raw_diagnostics.csv',rows)
    grid.write(args.output/'summary.json',{'table':summary,'diagnostics':len(rows),'full_paper_CREDIT':False,
        'uniform_certificate':False,'scope':p['scope']})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
