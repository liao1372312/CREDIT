"""Reconstruct chronological evidence/policies and reexecute every scored task."""
import argparse,json,tempfile,time
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from .paired_statistics import summarize
from .v3.experiment import digest,write
from .v3.local import retriever,ROLE_NAMES,valid_request
from .v3.selection import bank_cost
from .v3_data import Corpus
from .v3_compact_optimization import LimitedCorpus
from .v3_optimization import history
from .v3_optimization_verify import read,records,compare,compare_traces
from .v3_revalidation import RevalidationExecutor,candidate_memories
from .v3.drift import TransportGate
from .v3_online import Store,load_ensemble
from .v3_online_fast import load_ensemble as load_fast
from .v3_online_experiment import SPECS,CAP,adaptation,selection,archive,policy_data
from .v3_online_data import SEEDS,CYCLES,CycleCorpus
from .v3_online_resume import load_episode


def wait_file(path,streaming):
    while streaming and not path.exists():time.sleep(2)
    if not path.exists():raise FileNotFoundError(path)


def guards(output,complete=False):
    p=read(output/'frozen_protocol.json')
    for name,h in p['source_sha256'].items():
        if digest(name)!=h or digest(output/'source'/Path(name).name)!=h:raise ValueError('frozen source differs: '+name)
    for name,key in [('AGENTS.md','agents_sha256'),('CREDIT_WWW2027_v3.tex','paper_sha256')]:
        if digest(name)!=p[key]:raise ValueError('protected file changed')
    data=Path(p['dataset'])
    if digest(data/'manifest.json')!=p['dataset_manifest_sha256']:raise ValueError('dataset manifest changed')
    for name,h in read(data/'manifest.json').items():
        if digest(data/name)!=h:raise ValueError('dataset changed')
    if (output/'completed_prefix_manifest.json').exists():
        for name,h in read(output/'completed_prefix_manifest.json').items():
            if digest(output/name)!=h:raise ValueError('scored prefix changed')
    if complete:
        for name,h in read(output/'artifact_manifest.json').items():
            if digest(output/name)!=h:raise ValueError('artifact changed: '+name)
        for cycle in range(CYCLES):
            for name,h in read(output/f'evaluation_freeze_{cycle}.json').items():
                if digest(output/name)!=h:raise ValueError('pre-score policy changed')
    return p,data


def check_model(directory,h,logged_count):
    paths=sorted(directory.glob('outcome_model_*.pt'))
    cls=torch.load(paths[0],map_location='cpu',weights_only=False)['model_class']
    models=(load_fast if cls=='FastOnlineModel' else load_ensemble)(directory,[e.context for e in h[0]])
    for path,cached in zip(paths,models):
        model=cached.model;data=torch.load(path,map_location='cpu',weights_only=False)
        packed,labels,recovery=model.pack_observations(h[1]);packed,counts=model.compress(packed,labels,recovery)
        with torch.no_grad():nll=float(model.compressed_loss(model.heads(*packed),counts))
        if abs(nll-model.fit_report['selected_validation_nll'])>1e-6:raise ValueError('checkpoint validation NLL')
        raw=model.predict_many([(e.context,e.exposure) for e in h[2]],calibrated=False)
        cal=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(raw,[e.outcome for e in h[2]])
        if not (np.allclose(cal.X_thresholds_,data['calibration_x'],atol=1e-7,rtol=1e-7)
                and np.allclose(cal.y_thresholds_,data['calibration_y'],atol=1e-7,rtol=1e-7)):raise ValueError('source calibration reconstruction')
        if model.fit_report['logged_baseline_episodes']!=logged_count or not model.fit_report['logged_exposure_gradients_blocked']:
            raise ValueError('logged baseline provenance')
        best=float('inf');selected=None
        for point in model.fit_report['trajectory']:
            if point['validation_nll']<best-1e-4:best,selected=point['validation_nll'],point['epoch']
        if selected!=model.fit_report['selected_epoch']:raise ValueError('validation checkpoint selection')
    return models


def audit(output,streaming=False):
    torch.set_num_threads(1);protocol,dataset=guards(output)
    proofdir=output.with_name(output.name+'_audit_blocks');proofdir.mkdir(exist_ok=True)
    deps={str(Path(__file__)):digest(__file__)}
    totals=dict(physical_episodes=0,physical_calls_with_fixtures=0,models=0,selections=0,certificates=0,
                reuse_references=0,test_episodes=0,archive_decisions=0)
    states_by_seed={};corpora={};all_rows=[]
    with tempfile.TemporaryDirectory(prefix='credit-online-audit-') as tmp:
        for seed in SEEDS:
            sd=output/f'seed_{seed}';wait_file(sd/'initial_models/training_reports.json',streaming)
            corpus=Corpus(dataset,seed);corpora[seed]=corpus;td=Path(tmp)/f'seed_{seed}';td.mkdir()
            h=history(LimitedCorpus(corpus),seed,td/'source_history')
            for name in ('factorial_estimates.json','transfer_levels.json','fixtures.json','costs.json'):
                compare(read(td/'source_history'/name),read(sd/'source_history'/name),'source '+name)
            n,c=compare_traces(td/'source_history/history.jsonl.gz',sd/'source_history/history.jsonl.gz')
            totals['physical_episodes']+=n;totals['physical_calls_with_fixtures']+=c+read(sd/'source_history/costs.json')['fixtures']
            models=check_model(sd/'initial_models',h,0);totals['models']+=2;hc=read(sd/'source_history/costs.json')
            states={}
            for name,policy in SPECS:
                states[name]={'policy':policy,'models':() if policy=='direct' else models,'evidence':list(h[4]),
                    'reports':[],'episodes':[],'logged':[],'leases':{},'bank':(),'store':Store(),'posterior':{},
                    'source_fee':hc['api_calls']['factorial']+hc['api_calls']['replay_audit']+hc['fixtures'] if policy=='direct' else h[-1],
                    'deployment_policy':policy_data(__import__('credit.v3_online',fromlist=['StoredGate']).StoredGate(h[4],{},Store()),())}
            states_by_seed[seed]=h,states
        for cycle in range(CYCLES):
            ledgers={};deployed={}
            for seed in SEEDS:
                h,states=states_by_seed[seed];corpus=CycleCorpus(corpora[seed],cycle)
                for name,policy in SPECS:
                    md=output/f'seed_{seed}'/f'cycle_{cycle}'/name;wait_file(md/'frozen_policy.json',streaming)
                    td=Path(tmp)/f'seed_{seed}'/f'cycle_{cycle}'/name;td.mkdir(parents=True)
                    state=states[name]
                    if cycle and policy!='direct':
                        state['models']=check_model(md/'models',h,len(state['logged']));totals['models']+=2
                    compare({'bank':list(state['bank']),'posterior':json.loads(json.dumps(state['posterior'])),
                             'store':state['store'].snapshot(),'logged_ids':[e.context.uid for e in state['logged']]},
                            read(md/'pre_adaptation.json'),'pre-adaptation chronology')
                    gate,fee=adaptation(state,corpus,seed,cycle,td/'adaptation')
                    for filename in ('schedule.json','fixtures.json','probe_decisions.json','transfer_levels.json',
                                     'factorial_estimates.json','admissions.json','leases.json','store_before_test.json','costs.json'):
                        compare(read(td/'adaptation'/filename),read(md/'adaptation'/filename),'adaptation/'+filename)
                    n,c=compare_traces(td/'adaptation/executions.jsonl.gz',md/'adaptation/executions.jsonl.gz')
                    totals['physical_episodes']+=n;totals['physical_calls_with_fixtures']+=c+read(md/'adaptation/costs.json')['fresh_fixture_calls']
                    ledger,actual_gate=selection(state,gate,corpus,fee+(state['source_fee'] if cycle==0 else 0),td/'selection')
                    for filename in ('screening.json','halving.json','switch_decision.json','costs.json','execution_reuse.json','certification_freeze.json'):
                        if (md/'selection'/filename).exists():compare(read(td/'selection'/filename),read(md/'selection'/filename),'selection/'+filename)
                    n,c=compare_traces(td/'selection/executions.jsonl.gz',md/'selection/executions.jsonl.gz')
                    totals['physical_episodes']+=n;totals['physical_calls_with_fixtures']+=c;totals['selections']+=1
                    totals['certificates']+=len(read(md/'selection/switch_decision.json')['certificates'])
                    if ledger.spent>CAP:raise ValueError('call budget exceeded')
                    for r in read(md/'selection/execution_reuse.json'):
                        physical=ledger.records[r['physical_record']];e=physical['episode']
                        if physical['phase']!=r['phase'] or e.context.uid!=r['context'] or list(e.exposure)!=r['exposure'] or not e.replay_valid:
                            raise ValueError('invalid exact exposure reference')
                        totals['reuse_references']+=1
                    if policy=='direct' and (md/'selection/certification_freeze.json').exists():
                        f=read(md/'selection/certification_freeze.json')
                        if f['model_report'] or any(v!=0 for row in f['pilot_weights'].values() for v in row.values()):raise ValueError('Direct borrowed model')
                    compare(state['deployment_policy'],read(md/'frozen_policy.json'),'complete deployed policy')
                    ledgers[seed,name]=ledger;deployed[seed,name]=actual_gate
                    print('audited selection',cycle,seed,name,flush=True)
            wait_file(output/f'raw_cycle_{cycle}.json',streaming)
            for name,h in read(output/f'evaluation_freeze_{cycle}.json').items():
                if digest(output/name)!=h:raise ValueError('scoring freeze differs')
            indexed={(r['seed'],r['method'],r['task']):r for r in read(output/f'raw_cycle_{cycle}.json')}
            expected={(seed,method,c.uid) for seed in SEEDS for method in [*(s[0] for s in SPECS),'NoMem']
                      for c in CycleCorpus(corpora[seed],cycle).phase('heldout-test')}
            if set(indexed)!=expected:raise ValueError('test coverage incomplete')
            seen=set();eps_by_state={key:[] for key in deployed};pool=candidate_memories();executor=RevalidationExecutor()
            contexts={seed:{c.uid:c for c in CycleCorpus(corpora[seed],cycle).phase('heldout-test')} for seed in SEEDS}
            for row in records(output/f'test_cycle_{cycle}.jsonl.gz'):
                key=row['seed'],row['method'],row['task'];seed,name,uid=key
                if key in seen:raise ValueError('duplicate test trace')
                seen.add(key);compare({k:v for k,v in row.items() if k!='episode'},indexed[key],'test table')
                state=states_by_seed[seed][1].get(name);bank=() if state is None else state['bank']
                gate=TransportGate() if state is None else deployed[seed,name];c=contexts[seed][uid]
                exposure=retriever(4)(c,tuple(m.uid for m in gate.candidates(c,[m for m in pool if m.uid in bank])))
                e=executor.run(c,exposure)
                compare(json.loads(json.dumps(__import__('dataclasses').asdict(e))),row['episode'],'actual heldout episode')
                nodes={n.uid:n for n in c.nodes}
                invalid=sum(not valid_request(ROLE_NAMES[nodes[v['node']].role.key],v['state_before'],v['request_mode'],*c.stratum.split('/')) for v in e.events)
                if (e.outcome!=row['success'] or e.calls!=row['api_calls'] or e.state_hash!=row['state_hash']
                        or list(exposure)!=row['exposure'] or bank_cost(bank,pool)!=row['bank_tokens'] or invalid!=row['contract_invalid_calls']):
                    raise ValueError('heldout endpoint mismatch')
                totals['physical_episodes']+=1;totals['physical_calls_with_fixtures']+=e.calls;totals['test_episodes']+=1
                if state is not None:eps_by_state[seed,name].append(e)
            if seen!=expected:raise ValueError('missing scored trace')
            all_rows.extend(indexed.values())
            costs=[]
            for seed in SEEDS:
                for name,policy in SPECS:
                    state=states_by_seed[seed][1][name];episodes=eps_by_state[seed,name]
                    state['logged'].extend(episodes);state['store'].expose(episodes,cycle)
                    bounds=archive(state,deployed[seed,name],ledgers[seed,name],cycle)
                    md=output/f'seed_{seed}'/f'cycle_{cycle}'/name;wait_file(md/'evidence_ingestion.json',streaming)
                    compare(bounds,read(md/'archive_bounds.json'),'paired archive evidence')
                    compare(state['store'].snapshot(),read(md/'store_after_test.json'),'storage chronology')
                    compare({'scored_before_ingestion':True,'block_ids':[e.context.uid for e in episodes],
                        'cumulative_logged_episodes':len(state['logged']),'randomized_training_episodes':len(state['episodes'])},
                        read(md/'evidence_ingestion.json'),'score before ingestion')
                    totals['archive_decisions']+=len(bounds)
                    costs.append({'seed':seed,'cycle':cycle,'method':name,'extra_api_calls':ledgers[seed,name].spent,'allowance':CAP})
            compare(costs,read(output/f'costs_cycle_{cycle}.json'),'cycle expense table')
            proof={'verified':True,'cycle':cycle,'counts':dict(totals),'source_sha256':protocol['source_sha256'],'auditor_sources':deps,
                   'block_sha256':{name:digest(output/name) for name in (f'raw_cycle_{cycle}.json',f'test_cycle_{cycle}.jsonl.gz',f'evaluation_freeze_{cycle}.json')}}
            write(proofdir/f'cycle_{cycle}.json',proof);print('complete block audit passed',cycle,flush=True)
    wait_file(output/'artifact_manifest.json',streaming);guards(output,True)
    compare(all_rows,read(output/'raw_test_results.json'),'combined results')
    main,paired=summarize(all_rows,'CREDIT-Covered',repeats=4000,random_seed=7013)
    compare({'main':main,'paired':paired},read(output/'statistics.json'),'paired statistics')
    secondary=[]
    for a,b in [('CREDIT-VOI','CREDIT-Random'),('CREDIT-VOI','CREDIT-Covered'),('CREDIT-Covered','Direct-Covered'),('CREDIT-Covered','CREDIT-NoRevalidation')]:
        _,p=summarize([r for r in all_rows if r['method'] in (a,b)],a,repeats=4000,random_seed=7013)
        secondary.append(p[0]|{'method':a,'contrast':a+' vs '+b})
    running=0.;order=sorted(range(len(secondary)),key=lambda i:secondary[i]['seed_sign_flip_p'])
    for rank,i in enumerate(order):
        running=max(running,min(1.,secondary[i]['seed_sign_flip_p']*(len(order)-rank)));secondary[i]['secondary_family_holm_p']=running
    compare(secondary,read(output/'secondary_statistics.json'),'secondary Holm contrasts')
    report={'verified':True,**totals,'benchmark_manifest_sha256':digest(output/'artifact_manifest.json'),'auditor_sources':deps,
        'limits':'same canonical policy reconstruction;frozen models checked,optimizer not retrained;finite simulator;per-cycle asymptotic certification;no real-service/LLM claims'}
    write(output.with_name(output.name+'_audit.json'),report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--streaming',action='store_true')
    a=p.parse_args();audit(a.results,a.streaming)
