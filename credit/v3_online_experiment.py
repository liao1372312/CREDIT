"""Three chronological CREDIT cycles with executed VOI and coverage controls."""
import argparse
from dataclasses import asdict
import gzip,json
from pathlib import Path

import numpy as np
import torch

from .paired_statistics import summarize
from .v3.drift import Level,TransportGate,ProbeEvidence
from .v3.experiment import digest,write
from .v3.inference import worst_mixture
from .v3.local import retriever,stable_seed,ROLE_NAMES,valid_request
from .v3.selection import bank_cost
from .v3_benchmark import CallLedger,make_selector,stream_records
from .v3_compact_optimization import LimitedCorpus
from .v3_data import Corpus
from .v3_optimization import history,designs
from .v3_revalidation import candidate_memories,RevalidationExecutor,EvidenceGate,reprobe
from .v3_screening_optimization import ReuseSelector,public_profiles,distinct_banks,mixed_means
from .v3_online import Store,StoredGate,PolicySelector,fit_ensemble,queue,balanced_designs,admit
from .v3_online_data import SEEDS,CYCLES,TEST_N,CycleCorpus

SPECS=(('CREDIT-Sparse','sparse'),('CREDIT-Covered','covered'),('CREDIT-VOI','voi'),
       ('CREDIT-Random','random'),('CREDIT-NoRevalidation','none'),('Direct-Covered','direct'))
CAP=300000


def policy_data(gate,bank):
    return {'bank':list(bank),'transfer_levels':[asdict(p) for p in gate.evidence.values()],
            'leases':gate.leases,'store':gate.store.snapshot()}


def restore_policy(data):
    evidence=[ProbeEvidence(**(p|{'level':Level(p['level'])})) for p in data['transfer_levels']]
    return StoredGate(evidence,data['leases'],Store.restore(data['store']))


class BudgetExecutor(RevalidationExecutor):
    def __init__(self,ledger):super().__init__();self.ledger=ledger
    def replay(self,context,exposure,reference,*,slots=None):
        if self.ledger.spent+2*len(context.nodes)>self.ledger.call_cap:
            raise ValueError('reserve replay call bound before physical execution')
        return super().replay(context,exposure,reference,slots=slots)


def initial(seed,corpus,directory):
    directory.mkdir();h=history(LimitedCorpus(corpus),seed,directory/'source_history')
    models=fit_ensemble(*h[:3],seed,directory/'initial_models')
    hc=json.loads((directory/'source_history/costs.json').read_text())
    states={}
    for name,policy in SPECS:
        states[name]={'policy':policy,'models':() if policy=='direct' else models,'evidence':list(h[4]),
            'reports':[],'episodes':[],'logged':[],'leases':{},'bank':(),'store':Store(),'posterior':{},
            'source_fee':hc['api_calls']['factorial']+hc['api_calls']['replay_audit']+hc['fixtures'] if policy=='direct' else h[-1]}
        states[name]['deployment_policy']=policy_data(StoredGate(h[4],{},Store()),())
    return h,states


def adaptation(state,corpus,seed,cycle,directory):
    directory.mkdir();pool=candidate_memories();policy=state['policy']
    ledger=CallLedger({'target-factorial':40000,'replay_audit':2000},150000);ledger.tag='online-adaptation'
    fixtures,decisions,steps=[],[],[];probe_calls=0
    mixture={f'{f}/{e}':.125 for f in ('books','travel','inventory','payments') for e in ('source','target')}
    profiles=public_profiles(mixture)
    initial_queue=queue(state['models'],state['evidence'],state['leases'],state['bank'],state['posterior']) if state['models'] else []
    values={(r['version'].split(':')[1]+'/target',uid):r['voi'] for r in initial_queue for uid in r['unblocked']}
    state['store'].refresh(profiles,EvidenceGate(state['evidence'],state['leases'],{'new_contract'}),cycle,role_values=values)
    def differential(p):
        nonlocal probe_calls
        if ledger.spent+2048>ledger.call_cap:raise ValueError('reserve fixture calls before probing')
        updated,records,ds=reprobe([p],stable_seed(seed,'online-differential',cycle,p.version))
        state['evidence']=[updated[0] if (e.role,e.version)==(p.role,p.version) else e for e in state['evidence']]
        calls=sum(r['api_calls'] for r in records)
        fixtures.extend(records);decisions.extend(ds);probe_calls+=calls
        ledger.spent+=calls;ledger.common_calls+=calls
        return updated[0].level==Level.V
    def factorial(family,sparse=False,role=None):
        tasks=[c for c in corpus.phase('target-adaptation') if c.stratum==family+'/target']
        if sparse:
            tasks=tasks[:24]
            if ledger.spent+len(tasks)*(32*12+12)>ledger.call_cap:raise ValueError('reserve sparse batch call bound')
            episodes,reports=designs(tasks,pool,BudgetExecutor(ledger),ledger,phase='target-factorial',seed=seed+cycle,draws=1)
            for uid in sorted({e.context.uid for e in episodes}):
                e=next(e for e in reversed(episodes) if e.context.uid==uid)
                full=ledger.execute('replay_audit',RevalidationExecutor(),e.context,e.exposure)
                if (full.events,full.observations,full.state_hash)!=(e.events,e.observations,e.state_hash):raise ValueError('sparse replay')
        else:
            if ledger.spent+len(tasks)*(32*12+12)>ledger.call_cap:raise ValueError('reserve covered batch call bound')
            archived={uid for uid in (m.uid for m in pool) if state['store'].entries.get((family+'/target',uid),{}).get('state')=='archived'}
            episodes,reports=balanced_designs(tasks,pool,ledger,stable_seed(seed,cycle,family,'balanced'),archived=archived)
        before={(key,uid) for key,ids in state['leases'].items() for uid in ids}
        state['episodes'].extend(episodes);state['reports'].extend(reports)
        leases,ds=admit(state['reports'],state['episodes'],pool,state['evidence'],minimum_contexts=12 if sparse else 24)
        state['leases']=leases
        after={(key,uid) for key,ids in leases.items() for uid in ids}
        newly_admitted=after-before
        if role is not None:
            scoped={m.uid for m in pool if role in (m.scope or ())}
            newly_admitted={(key,uid) for key,uid in newly_admitted if uid in scoped}
        return bool(newly_admitted),ds
    if policy=='none':
        steps.append({'stopped':'revalidation removed;no fixture/factorial calls'})
    elif policy in ('sparse','covered','direct'):
        for p in list(state['evidence']):
            if p.level==Level.U and p.version.endswith(':target'):differential(p)
        families=('books','travel','inventory','payments') if policy=='sparse' else sorted({p.version.split(':')[1] for p in state['evidence'] if p.level!=Level.V and p.version.endswith(':target')})
        for family in families:
            for p in state['evidence']:
                if p.version.split(':')[1]==family:state['store'].fetch(p.role,p.version,cycle)
            passed,ds=factorial(family,policy=='sparse')
            steps.append({'kind':'factorial','family':family,'passed':passed,'contexts':24 if policy=='sparse' else 96,
                          'physical_calls_so_far':ledger.spent})
    else:
        used=set();rng=np.random.default_rng(stable_seed(seed,cycle,'role-exploration'))
        for turn in range(2):
            q=queue(state['models'],state['evidence'],state['leases'],state['bank'],state['posterior'])
            available=[r for r in q if (r['role'],r['version'],r['kind']) not in used]
            if not available:break
            # One exploration action per cycle, then positive-VOI exploitation.
            explore=policy=='random' or turn==0
            if not explore and available[0]['voi']<=0:
                steps.append({'turn':turn,'queue':q,'stopped':'nonpositive VOI'});break
            chosen=available[int(rng.integers(len(available)))] if explore else available[0]
            if ledger.spent+chosen['probe_call_bound']>150000:
                steps.append({'turn':turn,'queue':q,'stopped':'reserved probe bound exceeds budget'});break
            key=chosen['role'],chosen['version'],chosen['kind'];used.add(key)
            p=next(e for e in state['evidence'] if (e.role,e.version)==key[:2])
            state['store'].fetch(p.role,p.version,cycle)
            if chosen['kind']=='differential':passed=differential(p)
            else:passed,ds=factorial(p.version.split(':')[1],role=p.role)
            success,failure=state['posterior'].get(chosen['kind'],(0,0))
            state['posterior'][chosen['kind']]=(success+int(passed),failure+int(not passed))
            steps.append({'turn':turn,'queue':q,'chosen':chosen,'exploration':explore,'passed':passed,
                          'posterior_after':dict(state['posterior']),'physical_calls_so_far':ledger.spent})
    leases,admissions=admit(state['reports'],state['episodes'],pool,state['evidence'],minimum_contexts=12 if policy=='sparse' else 24)
    state['leases']=leases
    state['store'].refresh(profiles,EvidenceGate(state['evidence'],leases,{'new_contract'}),cycle,role_values=values)
    write(directory/'schedule.json',{'initial_queue':initial_queue,'steps':steps,'posterior':state['posterior'],
        'forced_exploration':'one role per cycle; remaining slot positive VOI; random control uses uniform roles'})
    write(directory/'fixtures.json',fixtures);write(directory/'probe_decisions.json',decisions)
    write(directory/'transfer_levels.json',[asdict(p) for p in state['evidence']])
    write(directory/'factorial_estimates.json',state['reports']);write(directory/'admissions.json',admissions)
    write(directory/'leases.json',leases);write(directory/'store_before_test.json',state['store'].snapshot())
    stream_records(directory/'executions.jsonl.gz',ledger.records)
    report=ledger.report()|{'fresh_fixture_calls':probe_calls,'total_with_fixtures':ledger.spent}
    write(directory/'costs.json',report)
    return StoredGate(state['evidence'],leases,state['store']),report['total_with_fixtures']


def selection(state,gate,corpus,common,directory):
    directory.mkdir();pool=candidate_memories();direct=state['policy']=='direct'
    ledger=CallLedger({'direct-screen':32768,'halving':1024,'pilot':2400,'certification':6400},CAP,common)
    ledger.tag='online-selection'
    base=make_selector(pool,() if direct else state['models'],gate,corpus,RevalidationExecutor(),ledger)
    selector=PolicySelector(pool,base.models,gate,base.retrieve,corpus.make,base.executor,ledger,
        mixture=base.mixture,capacity=80,radius=.2,cost_penalty=.0001,transported_strata=base.transported_strata)
    unlabeled,tasks=selector.contexts('screening-unlabeled',32),selector.contexts('direct-screening',16)
    candidates,feasible=distinct_banks(selector,public_profiles(selector.mixture));values={};rows=[]
    for bank in candidates:
        means=mixed_means(selector,bank,unlabeled,tasks,'direct' if direct else 'mixed')
        value=worst_mixture(means,list(selector.mixture.values()),.2)-.0001*bank_cost(bank,pool)
        values[bank]=value;rows.append({'bank':bank,'means':means,'score':value})
    leaders=sorted(values,key=lambda b:(-values[b],bank_cost(b,pool),b))[:8]
    write(directory/'screening.json',{'banks':leaders,'all_candidate_scores':rows,'feasible_banks':feasible,
        'distinct_exposures':len(candidates),'transported_strata':sorted(selector.transported_strata)})
    halving=selector.halve(leaders,gamma=1024,finalists=2)
    decision,deployed_gate=selector.switch_policy(state['bank'],restore_policy(state['deployment_policy']),halving['finalists'],pilot_n=2 if direct else 16,
        certification_n=200,unlabeled_n=2 if direct else 1000,
        freeze=lambda d:write(directory/'certification_freeze.json',d))
    state['bank']=tuple(decision['bank'])
    state['deployment_policy']=policy_data(deployed_gate,state['bank'])
    write(directory/'halving.json',halving);write(directory/'switch_decision.json',decision)
    write(directory/'costs.json',ledger.report());write(directory/'execution_reuse.json',selector.reuse_records)
    stream_records(directory/'executions.jsonl.gz',ledger.records)
    return ledger,deployed_gate


def archive(state,gate,ledger,cycle):
    """Opportunistic paired upper bounds; exact equal exposure gives delta=0.

    Equality is checked without withholding the candidate at the gate, over
    every public deployment template and every eligible stratum. Recorded
    certification episodes supply paired physical references. No absence-only
    archive is permitted; other marginals remain unresolved.
    """
    pool=candidate_memories();mixture={f'{f}/{e}':.125 for f in ('books','travel','inventory','payments') for e in ('source','target')}
    profiles=public_profiles(mixture);bounds=[]
    for m in pool:
        if m.uid in state['bank']:continue
        entries=[e for (k,uid),e in state['store'].entries.items() if uid==m.uid]
        if not entries or any(cycle-e['last_exposed']<state['store'].archive_after for e in entries):continue
        scoped=[c for c in profiles if any(m.eligible(n.role) for n in c.nodes)]
        def exposures(c):
            allowed=tuple(x.uid for x in gate.candidates(c,[x for x in pool if x.uid in state['bank']]))
            return retriever(4)(c,allowed),retriever(4)(c,tuple(sorted(set(allowed)|{m.uid})))
        if any(exposures(c)[0]!=exposures(c)[1] for c in scoped):continue
        row={'memory':m.uid,'cycle':cycle,'inactive_cycles':min(cycle-e['last_exposed'] for e in entries),
             'interval':'exact paired zero from identical ordered exposure on all finite public profiles',
             'scope':'known versions/templates/initial states only','strata':{}}
        for k in sorted({c.stratum for c in scoped}):
            refs=[];seen=set()
            for i,r in enumerate(ledger.records):
                e=r['episode'];c=e.context
                if r['phase']!='certification' or c.stratum!=k or c.uid in seen:continue
                a,b=exposures(c)
                if a!=b or e.exposure!=a or not e.replay_valid:continue
                refs.append(i);seen.add(c.uid)
            if len(refs)<12:break
            row['strata'][k]={'paired_contexts':len(refs),'physical_references':refs,
                              'upper_joint_marginal':-.0001*m.cost}
        else:
            for key,e in state['store'].entries.items():
                if key[1]==m.uid:e['state']='archived'
            state['store'].events.append({'cycle':cycle,'memory':m.uid,'action':'archive',
                'reason':'inactive Tq and nonpositive paired joint-marginal upper in every eligible stratum'})
            bounds.append(row)
    return bounds


def score_cycle(prepared,output,cycle):
    rows=[];episodes={};pool=candidate_memories();executor=RevalidationExecutor()
    with gzip.open(output/f'test_cycle_{cycle}.jsonl.gz','wt',compresslevel=6) as stream:
        for seed,corpus,states,gates in prepared:
            for name in [*(s[0] for s in SPECS),'NoMem']:
                state=states.get(name);bank=() if state is None else state['bank']
                gate=TransportGate() if state is None else gates[name];collected=[]
                for c in corpus.phase('heldout-test'):
                    exposure=retriever(4)(c,tuple(m.uid for m in gate.candidates(c,[m for m in pool if m.uid in bank])))
                    e=executor.run(c,exposure);collected.append(e);nodes={n.uid:n for n in c.nodes}
                    row={'seed':seed,'cycle':cycle,'method':name,'task':c.uid,'family':c.stratum.split('/')[0],
                        'epoch':c.stratum.split('/')[1],'success':e.outcome,'api_calls':e.calls,
                        'bank_tokens':bank_cost(bank,pool),'exposure':list(exposure),'state_hash':e.state_hash,
                        'verification_known':e.verified is not None,
                        'contract_invalid_calls':sum(not valid_request(ROLE_NAMES[nodes[v['node']].role.key],v['state_before'],v['request_mode'],*c.stratum.split('/')) for v in e.events)}
                    rows.append(row);stream.write(json.dumps(row|{'episode':asdict(e)},separators=(',',':'))+'\n')
                episodes[seed,name]=collected
            print('scored chronological block',cycle,seed,flush=True)
    return rows,episodes


def run(dataset,output):
    if output.exists():raise FileExistsError('use a fresh output directory')
    torch.set_num_threads(1);output.mkdir(parents=True);(output/'source').mkdir()
    sources=[Path(__file__),Path('credit/v3_online.py'),Path('credit/v3_online_data.py'),
        *Path('credit/v3').glob('*.py'),Path('credit/stateful.py'),Path('credit/v3_benchmark.py'),
        Path('credit/v3_optimization.py'),Path('credit/v3_optimized_model.py'),Path('credit/v3_optimized_data.py'),
        Path('credit/v3_compact_optimization.py'),Path('credit/v3_revalidation.py'),
        Path('credit/v3_screening_optimization.py'),Path('credit/v3_data.py'),Path('credit/paired_statistics.py')]
    write(output/'frozen_protocol.json',{'seeds':SEEDS,'cycles':CYCLES,'test_n':TEST_N,'specifications':SPECS,
        'dataset':str(dataset.resolve()),'dataset_manifest_sha256':digest(dataset/'manifest.json'),
        'source_sha256':{str(p):digest(p) for p in sources},'paper_sha256':digest('CREDIT_WWW2027_v3.tex'),
        'agents_sha256':digest('AGENTS.md'),'primary':'CREDIT-Covered vs CREDIT-Sparse',
        'secondary':['CREDIT-VOI vs CREDIT-Random','CREDIT-VOI vs CREDIT-Covered','CREDIT-Covered vs Direct-Covered','CREDIT-Covered vs CREDIT-NoRevalidation'],
        'budget':'same300000 extra calls/cycle,150000 adaptation cap;source history charged once;actual expenses disclosed',
        'coverage':'96 contexts/family,32/template,uniform-marginal balanced K2 permutations;minimum24 clusters;alpha.05/36',
        'sparse':'24 contexts/family,original independent K2,minimum12 clusters;alpha.05/36',
        'voi':'U uses Eq.VOI;D/X is an optimistic lease-admission proxy;single add/swap,shared robust gated objective;execute and update Beta each step',
        'exploration':'one uniformly random role each cycle;remaining one positive-VOI slot;Random two uniform slots',
        'planning':'hypothetical restored role predictions are only scheduling values;actual D source heads blocked',
        'online':'after all seeds/methods scored a block,ingest own logs,randomized target designs,refit for next block',
        'training':'two-member OnlineModel ensemble;explicit snapshot bits;source validation/calibration;observational baseline only',
        'variance':'randomized signs preserve unbiasedness;CRN covariance prevents applying the manuscript independent-row variance bound',
        'storage':'quarantine40 tokens,reversible cold storage,Tq2;archive requires exact paired marginal bounds all eligible strata',
        'certification':'independent pilot16 CREDIT/2 Direct,200 tasks/stratum/policy,alpha.10 asymptotic;prior bank/gate/store frozen;up to3 policy challengers including same-bank changes',
        'scope':'three finite chronological blocks at source/target pinned snapshots;all templates seen;no natural drift/LLM claims',
        'development':'known seed1009 only;no confirming outcomes used for parameter selection'})
    for p in sources:(output/'source'/p.name).write_bytes(p.read_bytes())
    inputs={}
    for seed in SEEDS:
        corpus=Corpus(dataset,seed);h,states=initial(seed,corpus,output/f'seed_{seed}');inputs[seed]=(corpus,h,states)
    all_rows=[]
    for cycle in range(CYCLES):
        prepared=[];ledgers={};costs=[]
        for seed,(base,h,states) in inputs.items():
            corpus=CycleCorpus(base,cycle);gates={}
            for name,policy in SPECS:
                print('cycle',cycle,seed,name,flush=True)
                sd=output/f'seed_{seed}'/f'cycle_{cycle}'/name;sd.mkdir(parents=True)
                state=states[name]
                if cycle and policy!='direct':
                    state['models']=fit_ensemble(h[0]+state['episodes'],h[1],h[2],stable_seed(seed,cycle,'online-refit')%2**31,
                                                sd/'models',logged=state['logged'])
                write(sd/'pre_adaptation.json',{'bank':state['bank'],'posterior':state['posterior'],
                                               'store':state['store'].snapshot(),'logged_ids':[e.context.uid for e in state['logged']]})
                gate,calls=adaptation(state,corpus,seed,cycle,sd/'adaptation');gates[name]=gate
                ledger,deployed_gate=selection(state,gate,corpus,calls+(state['source_fee'] if cycle==0 else 0),sd/'selection')
                gates[name]=deployed_gate
                ledgers[seed,name]=ledger
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
                bounds=archive(state,gates[name],ledgers[seed,name],cycle)
                sd=output/f'seed_{seed}'/f'cycle_{cycle}'/name
                write(sd/'archive_bounds.json',bounds);write(sd/'store_after_test.json',state['store'].snapshot())
                write(sd/'evidence_ingestion.json',{'scored_before_ingestion':True,'block_ids':[e.context.uid for e in episodes[seed,name]],
                    'cumulative_logged_episodes':len(state['logged']),'randomized_training_episodes':len(state['episodes'])})
        write(output/f'costs_cycle_{cycle}.json',costs)
        write(output/f'raw_cycle_{cycle}.json',rows)
    write(output/'raw_test_results.json',all_rows)
    main,paired=summarize(all_rows,'CREDIT-Covered',repeats=4000,random_seed=7013)
    write(output/'statistics.json',{'main':main,'paired':paired})
    secondary=[]
    for a,b in [('CREDIT-VOI','CREDIT-Random'),('CREDIT-VOI','CREDIT-Covered'),('CREDIT-Covered','Direct-Covered'),('CREDIT-Covered','CREDIT-NoRevalidation')]:
        _,p=summarize([r for r in all_rows if r['method'] in (a,b)],a,repeats=4000,random_seed=7013)
        secondary.append(p[0]|{'method':a,'contrast':a+' vs '+b})
    running=0.
    order=sorted(range(len(secondary)),key=lambda i:secondary[i]['seed_sign_flip_p'])
    for rank,i in enumerate(order):
        running=max(running,min(1.,secondary[i]['seed_sign_flip_p']*(len(order)-rank)))
        secondary[i]['secondary_family_holm_p']=running
    write(output/'secondary_statistics.json',secondary)
    write(output/'summary.json',{'test_episodes':len(all_rows),'test_api_calls':sum(r['api_calls'] for r in all_rows),
                              'paper_experiments_complete':False,'chronological_cycles_complete':True})
    write(output/'artifact_manifest.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    print('Completed chronological CREDIT experiment',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.dataset,a.output)
