"""Reexecute benchmark traces and reconstruct halving and certificates."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from .v3.audit import SavedCalibration, check_close, load_context
from .v3.drift import Level, TransportGate
from .v3.experiment import collect_designs, digest, write
from .v3.inference import certify, ppi, tune_lambda, worst_mixture
from .v3.local import StatefulExecutor, memories, differential_probes, retriever
from .v3.outcome import OutcomeModel
from .v3.selection import Ledger, bank_cost
from .v3_benchmark import CachedModel, GraphConditioned, NoPairs, make_selector
from .v3_data import Corpus
from .paired_statistics import summarize


def read(path): return json.loads(path.read_text())


def records(path):
    with gzip.open(path,'rt') as f:
        for line in f: yield json.loads(line)


def load_models(directory, train, calibration, model_class=OutcomeModel):
    models=[]
    for path in sorted(directory.glob('outcome_model_*.pt')):
        data=torch.load(path,map_location='cpu',weights_only=False)
        m=model_class(memories(),[e.context for e in train])
        if m.roles!=data['role_indices']: raise ValueError('role encoder')
        m.heads.load_state_dict(data['state_dict']);m.fit_report=data['fit_report']
        m.calibrator=SavedCalibration(data['calibration_x'],data['calibration_y'])
        values=m.predict_many([(e.context,e.exposure) for e in calibration],calibrated=False)
        refit=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(values,[e.outcome for e in calibration])
        check_close(refit.X_thresholds_,data['calibration_x'],'calibration x')
        check_close(refit.y_thresholds_,data['calibration_y'],'calibration y')
        models.append(CachedModel(m))
    if len(models)!=3: raise ValueError('ensemble incomplete')
    return models


def audit(output):
    torch.set_num_threads(1)
    protocol=read(output/'frozen_protocol.json')
    for name,value in read(output/'artifact_manifest.json').items():
        if digest(output/name)!=value: raise ValueError(f'artifact: {name}')
    for name,value in protocol['source_sha256'].items():
        if digest(name)!=value or digest(output/'source'/Path(name).name)!=value: raise ValueError(f'source: {name}')
    for name,value in read(output/'evaluation_freeze.json').items():
        if digest(output/name)!=value: raise ValueError(f'evaluation freeze: {name}')
    dataset=Path(protocol['dataset'])
    if digest(dataset/'manifest.json')!=protocol['dataset_manifest_sha256']: raise ValueError('dataset freeze')
    for name,value in read(dataset/'manifest.json').items():
        if digest(dataset/name)!=value: raise ValueError(f'dataset artifact: {name}')
    pool,engine=memories(),StatefulExecutor();seed_data={};episodes=0;calls=0;certificates=0;models_checked=0;halving_rounds=0
    for seed in protocol['seeds']:
        sd=output/f'seed_{seed}';corpus=Corpus(dataset,seed)
        hist=Ledger({'factorial':4096,'calibration':2048,'replay_audit':64})
        tc=[c for c in corpus.phase('training') if int(c.uid.rsplit(':',1)[1])<8]
        vc=[c for c in corpus.phase('calibration') if int(c.uid.rsplit(':',1)[1])<8]
        train,designs=collect_designs(tc,pool,engine,hist,phase='factorial',seed=seed,draws=2,audit=True)
        calibration,_=collect_designs(vc,pool,engine,hist,phase='calibration',seed=seed+1,draws=1)
        if json.loads(json.dumps(designs))!=read(sd/'factorial_estimates.json'): raise ValueError('randomized design reconstruction')
        saved=list(records(sd/'history.jsonl.gz'))
        if len(saved)!=len(hist.records): raise ValueError('history length')
        for r,e in zip(saved,hist.records):
            if r['phase']!=e['phase'] or r['episode']!=json.loads(json.dumps(asdict(e['episode']))):
                raise ValueError('randomized history/replay mismatch')
            episodes+=1;calls+=e['episode'].calls
        support=Counter()
        for e in train:
            ns={n.uid:n for n in e.context.nodes}
            for o in e.observations:
                if o.label in ('ok','agent'): support[ns[o.node].role.key,ns[o.node].version]+=1
        evidence,fixtures=differential_probes(seed=seed,support=support,fixtures=96)
        if fixtures!=read(sd/'fixtures.json') or json.loads(json.dumps([asdict(e) for e in evidence]))!=read(sd/'transfer_levels.json'):
            raise ValueError('probe evidence')
        fixture_calls=sum(r['api_calls'] for r in fixtures);calls+=fixture_calls
        base_gate=TransportGate(evidence)
        only_direct=set(read(sd/'frozen_banks.json'))=={'Direct-Sandbox-Only'}
        base_models=() if only_direct else load_models(sd,train,calibration)
        models_checked+=len(base_models)
        common=sum(e['episode'].calls for e in hist.records)+fixture_calls
        hc=read(sd/'history_costs.json')
        if common!=hc['total_api_calls']+hc['fixtures']: raise ValueError('shared cost')
        banks=read(sd/'frozen_banks.json');costs=read(sd/'method_costs.json');policies={}
        for method,bank in banks.items():
            cap=40 if method=='C40-K4' else 120 if method=='C120-K4' else 80
            depth=2 if method=='C80-K2' else 6 if method=='C80-K6' else 4
            if bank_cost(bank,pool)>cap or costs[method]['selection_api_calls_including_common']>protocol['call_allowance_per_strategy']:
                raise ValueError('bank/API capacity')
            gate=base_gate;models=base_models;md=sd/method
            if method=='Direct-Sandbox-Only':models=()
            if method=='Without-drift-gating': gate=TransportGate([replace(e,level=Level.V) for e in evidence])
            if method=='Heldout-payments': gate=TransportGate([replace(e,level=Level.X,support=0) if ':payments:' in e.version else e for e in evidence])
            learned=md.exists()
            policies[method]=(gate,depth) if learned or method in ('Factorial-Banzhaf','Model-MCShapley') else (None,depth)
            if not learned: continue
            if method in ('Without-interactions','Without-role-sharing','Heldout-payments','Label-error-20pct'):
                mt=NoPairs if method=='Without-interactions' else GraphConditioned if method=='Without-role-sharing' else OutcomeModel
                tr=[e for e in train if not e.context.stratum.startswith('payments/')] if method=='Heldout-payments' else train
                cal=[e for e in calibration if not e.context.stratum.startswith('payments/')] if method=='Heldout-payments' else calibration
                models=load_models(md,tr,cal,mt);models_checked+=3
            selector=make_selector(pool,models,gate,corpus,engine,Ledger({}),capacity=cap,depth=depth,
                                   radius=0 if method=='Without-robust-ranking' else .2)
            observed={};phase_counts=Counter();phase_calls=Counter();round_values=defaultdict(lambda:defaultdict(list))
            for r in records(md/'selection.jsonl.gz'):
                data=r['episode'];c=load_context(data['context']);a=tuple(data['exposure']);b=tuple(r['bank'])
                if a!=selector.exposures(c,b): raise ValueError('selection gate/retrieval')
                e=engine.run(c,a)
                if json.loads(json.dumps(asdict(e)))!=data: raise ValueError('selection actual execution')
                if e.restored_hash!=e.checkpoint_hash or not e.replay_valid: raise ValueError('selection restoration')
                observed[c.uid,b]=e.outcome;phase_counts[r['phase']]+=1;phase_calls[r['phase']]+=e.calls
                if r['phase']=='halving':
                    round_no=int(c.uid.split(':halving-')[1].split(':')[0])
                    round_values[round_no][b,c.stratum].append(e.outcome)
                episodes+=1;calls+=e.calls
            report=read(md/'costs.json')
            if dict(phase_counts)!=report['episodes'] or dict(phase_calls)!=report['api_calls']:
                raise ValueError('selection accounting')
            if sum(phase_calls.values())+common!=costs[method]['selection_api_calls_including_common']:
                raise ValueError('shared history charge')
            if any(n>report['episode_limits'][p] for p,n in phase_counts.items()): raise ValueError('phase cap')
            for rd in read(md/'halving.json')['rounds']:
                scores={}
                for key,stored in rd['scores'].items():
                    b=tuple(key.split('|')) if key else ()
                    values=[round_values[rd['round']][b,k] for k in selector.mixture]
                    if any(len(v)!=rd['episodes_per_stratum_bank'] for v in values): raise ValueError('halving samples')
                    scores[b]=worst_mixture([np.mean(v) for v in values],list(selector.mixture.values()),selector.radius)-.0001*bank_cost(b,pool)
                    check_close(scores[b],stored,'halving score')
                expected=sorted(scores,key=lambda b:(-scores[b],bank_cost(b,pool),b))[:max(2,math_ceil_half(len(scores)))]
                if [list(b) for b in expected]!=rd['survivors']: raise ValueError('halving survivor')
                halving_rounds+=1
            decision=read(md/'switch_decision.json')
            if (md/'certification_freeze.json').exists():
                frozen=read(md/'certification_freeze.json');inc=tuple(frozen['incumbent']);finalists=[tuple(b) for b in frozen['finalists']]
                preds=frozen['prediction_hash_input']
                for b in [inc]+finalists:
                    for k,pred in preds['|'.join(b)].items():
                        for phase,key,n in [('pilot','pilot',100),('certification','labeled',200),('certification-unlabeled','unlabeled',1000)]:
                            tasks=[corpus.make(k,phase,i) for i in range(n)]
                            check_close(selector.predictions(tasks,b),pred[key],'frozen model prediction')
                for b,recorded in zip(finalists,decision['certificates']):
                    es={}
                    for k in selector.mixture:
                        delta=lambda phase,n:np.asarray([observed[corpus.make(k,phase,i).uid,b]-observed[corpus.make(k,phase,i).uid,inc] for i in range(n)])
                        pred=preds['|'.join(b)][k];base=preds['|'.join(inc)][k]
                        weight=tune_lambda(delta('pilot',100),np.asarray(pred['pilot'])-base['pilot'],labeled_n=200,unlabeled_n=1000,
                                           transported=k in selector.transported_strata)
                        check_close(weight,frozen['pilot_weights']['|'.join(b)][k],'pilot weight')
                        es[k]=ppi(delta('certification',200),np.asarray(pred['labeled'])-base['labeled'],
                                  np.asarray(pred['unlabeled'])-base['unlabeled'],weight)
                        for field,value in asdict(es[k]).items(): check_close(value,recorded['estimates'][k][field],'PPI '+field)
                    expected=certify(es,selector.mixture,cost_difference=bank_cost(b,pool)-bank_cost(inc,pool),cost_penalty=.0001,finalists=len(finalists))
                    if expected['passed']!=recorded['passed']: raise ValueError('certificate decision')
                    check_close(expected['aggregate_lower'],recorded['aggregate_lower'],'aggregate lower')
                    for k in selector.mixture:check_close(expected['stratum_lower'][k],recorded['stratum_lower'][k],'guard')
                    certificates+=1
                accepted=[r for r in decision['certificates'] if r['passed']]
                best=max(accepted,key=lambda r:r['aggregate_lower']-r['cost_threshold']) if accepted else None
                if decision['bank']!=(best['bank'] if best else list(inc)): raise ValueError('certified adoption')
            elif method!='Without-certification' and decision.get('switched'): raise ValueError('uncertified adoption')
            if decision['bank']!=bank: raise ValueError('bank freeze')
        seed_data[seed]=(corpus,banks,policies,evidence)
        print('seed',seed,'history/selection/certificates audited',flush=True)
    rows=read(output/'raw_test_results.json');indexed={(r['seed'],r['method'],r['task']):r for r in rows};seen=set()
    for r in records(output/'test_traces.jsonl.gz'):
        seed,method=r['seed'],r['method'];corpus,banks,policies,evidence=seed_data[seed]
        phase_index=int(r['task'].rsplit(':',1)[1]);c=corpus.make(r['family']+'/'+r['epoch'],'heldout-test',phase_index)
        if c.uid!=r['task']:raise ValueError('held-out context')
        bank=banks[method];gate,depth=policies[method];eligible=bank
        if gate:eligible=tuple(m.uid for m in gate.candidates(c,[m for m in pool if m.uid in bank]))
        elif method=='Version-filter' and r['epoch']=='target':eligible=()
        a=retriever(depth)(c,eligible);e=engine.run(c,a)
        if list(a)!=r['exposure'] or list(e.events)!=r['events'] or e.state_hash!=r['state_hash'] or e.outcome!=r['success'] or e.calls!=r['api_calls']:
            raise ValueError('actual held-out execution')
        key=(seed,method,r['task'])
        if key in seen or indexed[key]!={k:v for k,v in r.items() if k!='events'}:raise ValueError('test row consistency')
        seen.add(key);episodes+=1;calls+=e.calls
    if seen!=set(indexed): raise ValueError('held-out coverage')
    for filename,methods in [('main_statistics.json',protocol['main_methods']),('ablation_statistics.json',['CREDIT']+protocol['variants'])]:
        if not (output/filename).exists():continue
        table,paired=summarize([r for r in rows if r['method'] in methods],'CREDIT',repeats=4000,random_seed=2991)
        if {'main':table,'paired':paired}!=read(output/filename):raise ValueError('paired statistics')
    report={'verified':True,'actual_episodes_reexecuted':episodes,'actual_api_calls_reexecuted_including_fixtures':calls,
            'certificates_reconstructed':certificates,'halving_rounds_reconstructed':halving_rounds,'model_checkpoints_checked':models_checked,
            'heldout_episodes_reexecuted':len(rows),'randomized_context_estimates_reconstructed':32*len(protocol['seeds']),
            'budgets_and_restoration_checked':True,'neural_optimizers_retrained':False,
            'limits':'screening candidate traversal not independently reconstructed; frozen source, halving and certificates checked',
            'auditor_sha256':digest(__file__)}
    write(output/'execution_audit.json',report);print(json.dumps(report,indent=2))


def math_ceil_half(n):return (n+1)//2


def audit_direct_only(corrected,original):
    """Compose unchanged source-history inputs with separately frozen real runs."""
    import tempfile
    for name,value in read(corrected/'artifact_manifest.json').items():
        if digest(corrected/name)!=value:raise ValueError('corrective artifact '+name)
    for name,value in read(corrected/'frozen_protocol.json')['source_sha256'].items():
        if digest(name)!=value:raise ValueError('corrective source '+name)
    if not read(original/'execution_audit.json')['verified']:raise ValueError('original history audit required')
    with tempfile.TemporaryDirectory(prefix='credit-v3-direct-audit-') as td:
        stage=Path(td);proto=read(original/'frozen_protocol.json')
        proto.update(main_methods=['Direct-Sandbox-Only'],variants=[])
        write(stage/'frozen_protocol.json',proto)
        (stage/'source').symlink_to((original/'source').resolve(),target_is_directory=True)
        for seed in proto['seeds']:
            sd=stage/f'seed_{seed}';sd.mkdir()
            for name in ('factorial_estimates.json','history.jsonl.gz','history_costs.json','fixtures.json','transfer_levels.json'):
                (sd/name).symlink_to((original/f'seed_{seed}'/name).resolve())
            for name in ('frozen_banks.json','method_costs.json','Direct-Sandbox-Only'):
                (sd/name).symlink_to((corrected/f'seed_{seed}'/name).resolve())
        for name in ('raw_test_results.json','test_traces.jsonl.gz','evaluation_freeze.json'):
            (stage/name).symlink_to((corrected/name).resolve())
        # The parent manifests have already verified every immutable input.
        write(stage/'artifact_manifest.json',{})
        audit(stage)
        report=read(stage/'execution_audit.json')
        report.update(model_checkpoints_checked=0,models_used=0,
                      original_manifest_sha256=digest(original/'artifact_manifest.json'),
                      corrective_manifest_sha256=digest(corrected/'artifact_manifest.json'))
        write(corrected/'execution_audit.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--direct-only',type=Path)
    args=p.parse_args()
    if args.direct_only:audit_direct_only(args.direct_only,args.results)
    else:audit(args.results)
