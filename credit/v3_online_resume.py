"""Continue scored chronological blocks with equivalent feature computation.

Completed blocks retain their original frozen source/protocol/trace hashes.
Only subsequent refits use shared-feature neural evaluation. Source history,
method specifications, seeds, budgets and hyperparameters are unchanged.
"""
import argparse
from collections import defaultdict
import json,shutil
from pathlib import Path

import torch

from .paired_statistics import summarize
from .v3.audit import load_context
from .v3.drift import ProbeEvidence,Level,TransportGate
from .v3.experiment import digest,write
from .v3.local import stable_seed
from .v3.types import Episode,NodeObservation
from .v3_optimization_verify import records,read
from .v3_data import Corpus
from .v3_online import Store,load_ensemble
from .v3_online_fast import fit_ensemble
from .v3_online_experiment import SPECS,CAP,adaptation,selection,score_cycle,archive
from .v3_online_data import SEEDS,CYCLES,CycleCorpus


def load_episode(row):
    values=dict(row);values['context']=load_context(values['context'])
    values['observations']=tuple(NodeObservation(**v) for v in values['observations'])
    values['exposure']=tuple(values['exposure']);values['events']=tuple(values['events'])
    return Episode(**values)


def source_inputs(directory):
    groups=defaultdict(list)
    for r in records(directory/'source_history/history.jsonl.gz'):groups[r['phase']].append(load_episode(r['episode']))
    ev=[ProbeEvidence(**(p|{'level':Level(p['level'])})) for p in read(directory/'source_history/transfer_levels.json')]
    cost=read(directory/'source_history/costs.json')
    return groups['factorial'],groups['validation'],groups['calibration'],TransportGate(ev),ev,cost['total_with_fixtures']


def restore_after_block0(directory,seed):
    h=source_inputs(directory);models=load_ensemble(directory/'initial_models',[e.context for e in h[0]])
    logged=defaultdict(list)
    for r in records(directory.parent/'test_cycle_0.jsonl.gz'):
        if r['seed']==seed:logged[r['method']].append(load_episode(r['episode']))
    states={}
    for name,policy in SPECS:
        md=directory/'cycle_0'/name
        ev=[ProbeEvidence(**(p|{'level':Level(p['level'])})) for p in read(md/'adaptation/transfer_levels.json')]
        eps=[load_episode(r['episode']) for r in records(md/'adaptation/executions.jsonl.gz') if r['phase']=='target-factorial']
        state={'policy':policy,'models':() if policy=='direct' else models,'evidence':ev,
            'reports':read(md/'adaptation/factorial_estimates.json'),'episodes':eps,'logged':logged[name],
            'leases':read(md/'adaptation/leases.json'),'bank':tuple(read(md/'frozen_policy.json')['bank']),
            'deployment_policy':read(md/'frozen_policy.json'),'store':Store.restore(read(md/'store_after_test.json')),
            'posterior':read(md/'adaptation/schedule.json')['posterior'],'source_fee':0}
        states[name]=state
    return h,states


def run(prefix,output):
    if output.exists():raise FileExistsError('use fresh continuation output')
    if not (prefix/'raw_cycle_0.json').exists() or (prefix/'raw_cycle_1.json').exists():raise ValueError('exactly one scored prefix block required')
    torch.set_num_threads(1);output.mkdir(parents=True)
    original=read(prefix/'frozen_protocol.json');dataset=Path(original['dataset'])
    if original['seeds']!=list(SEEDS):raise ValueError('continuation seeds differ')
    for name,h in original['source_sha256'].items():
        if digest(name)!=h or digest(prefix/'source'/Path(name).name)!=h:raise ValueError('original source changed')
    sources=list(original['source_sha256'])+[str(Path(__file__)),str(Path('credit/v3_online_fast.py'))]
    protocol=original|{'source_sha256':{p:digest(p) for p in sources},'completed_prefix_blocks':[0],
        'prefix_protocol_sha256':digest(prefix/'frozen_protocol.json'),'prefix_directory':str(prefix.resolve()),
        'continuation':'same mathematical heads/likelihood/optimizer;shared fixed-feature evaluations;float-equivalence tested',
        'implementation_freeze':'original source before block0;equivalent-computation continuation source before block1',
        'test_outcomes_used_to_select_hyperparameters':False}
    write(output/'frozen_protocol.json',protocol);shutil.copytree(prefix/'source',output/'source')
    for p in sources:(output/'source'/Path(p).name).write_bytes(Path(p).read_bytes())
    prefix_manifest={}
    for seed in SEEDS:
        old=prefix/f'seed_{seed}';sd=output/f'seed_{seed}';sd.mkdir()
        for name in ('source_history','initial_models','cycle_0'):shutil.copytree(old/name,sd/name)
    for name in ('raw_cycle_0.json','costs_cycle_0.json','test_cycle_0.jsonl.gz','evaluation_freeze_0.json'):
        shutil.copy2(prefix/name,output/name)
    for p in output.rglob('*'):
        if p.is_file() and (p.name in ('raw_cycle_0.json','costs_cycle_0.json','test_cycle_0.jsonl.gz','evaluation_freeze_0.json') or
                          any(part in ('source_history','initial_models','cycle_0') for part in p.parts)):
            prefix_manifest[str(p.relative_to(output))]=digest(p)
    write(output/'completed_prefix_manifest.json',prefix_manifest)
    inputs={seed:(Corpus(dataset,seed),*restore_after_block0(output/f'seed_{seed}',seed)) for seed in SEEDS}
    all_rows=read(output/'raw_cycle_0.json')
    for cycle in range(1,CYCLES):
        prepared=[];ledgers={};costs=[]
        for seed,(base,h,states) in inputs.items():
            corpus=CycleCorpus(base,cycle);gates={}
            for name,policy in SPECS:
                print('cycle',cycle,seed,name,flush=True)
                sd=output/f'seed_{seed}'/f'cycle_{cycle}'/name;sd.mkdir(parents=True)
                state=states[name]
                if policy!='direct':
                    state['models']=fit_ensemble(h[0]+state['episodes'],h[1],h[2],stable_seed(seed,cycle,'online-refit')%2**31,
                                                sd/'models',logged=state['logged'])
                write(sd/'pre_adaptation.json',{'bank':state['bank'],'posterior':state['posterior'],
                    'store':state['store'].snapshot(),'logged_ids':[e.context.uid for e in state['logged']]})
                proposed,calls=adaptation(state,corpus,seed,cycle,sd/'adaptation')
                ledger,deployed=selection(state,proposed,corpus,calls,sd/'selection')
                gates[name]=deployed;ledgers[seed,name]=ledger
                write(sd/'frozen_policy.json',state['deployment_policy'])
                costs.append({'seed':seed,'cycle':cycle,'method':name,'extra_api_calls':ledger.spent,'allowance':CAP})
            prepared.append((seed,corpus,states,gates))
        frozen={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and
                f'cycle_{cycle}/' in str(p) and p.name in ('frozen_policy.json','certification_freeze.json','pre_adaptation.json')}
        write(output/f'evaluation_freeze_{cycle}.json',frozen)
        rows,episodes=score_cycle(prepared,output,cycle);all_rows.extend(rows)
        for seed,corpus,states,gates in prepared:
            for name,policy in SPECS:
                state=states[name];state['logged'].extend(episodes[seed,name]);state['store'].expose(episodes[seed,name],cycle)
                bounds=archive(state,gates[name],ledgers[seed,name],cycle);sd=output/f'seed_{seed}'/f'cycle_{cycle}'/name
                write(sd/'archive_bounds.json',bounds);write(sd/'store_after_test.json',state['store'].snapshot())
                write(sd/'evidence_ingestion.json',{'scored_before_ingestion':True,'block_ids':[e.context.uid for e in episodes[seed,name]],
                    'cumulative_logged_episodes':len(state['logged']),'randomized_training_episodes':len(state['episodes'])})
        write(output/f'costs_cycle_{cycle}.json',costs);write(output/f'raw_cycle_{cycle}.json',rows)
    write(output/'raw_test_results.json',all_rows)
    main,paired=summarize(all_rows,'CREDIT-Covered',repeats=4000,random_seed=7013)
    write(output/'statistics.json',{'main':main,'paired':paired});secondary=[]
    for a,b in [('CREDIT-VOI','CREDIT-Random'),('CREDIT-VOI','CREDIT-Covered'),('CREDIT-Covered','Direct-Covered'),('CREDIT-Covered','CREDIT-NoRevalidation')]:
        _,p=summarize([r for r in all_rows if r['method'] in (a,b)],a,repeats=4000,random_seed=7013)
        secondary.append(p[0]|{'method':a,'contrast':a+' vs '+b})
    running=0.;order=sorted(range(len(secondary)),key=lambda i:secondary[i]['seed_sign_flip_p'])
    for rank,i in enumerate(order):
        running=max(running,min(1.,secondary[i]['seed_sign_flip_p']*(len(order)-rank)));secondary[i]['secondary_family_holm_p']=running
    write(output/'secondary_statistics.json',secondary)
    write(output/'summary.json',{'test_episodes':len(all_rows),'test_api_calls':sum(r['api_calls'] for r in all_rows),
        'paper_experiments_complete':False,'chronological_cycles_complete':True,'prefix_blocks_reused':1})
    write(output/'artifact_manifest.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    print('Completed chronological continuation',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.prefix,a.output)
