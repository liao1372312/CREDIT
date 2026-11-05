"""Prospective target-version probing and direct-memory admission experiment."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from .paired_statistics import summarize
from .v3.experiment import digest,write
from .v3.inference import worst_mixture
from .v3.local import StatefulExecutor, memories, retriever, ROLE_NAMES, valid_request
from .v3.drift import Level,TransportGate
from .v3.selection import bank_cost
from .v3_benchmark import CallLedger,make_selector,stream_records
from .v3_compact_optimization import LimitedCorpus
from .v3_data import Corpus
from .v3_optimization import history,ensemble,designs
from .v3_optimized_model import StateAwareOutcomeModel,StateAwareNoPairs
from .v3_revalidation import EvidenceGate,reprobe,admit_memories,candidate_memories,RevalidationExecutor
from .v3_revalidation_data import SEEDS
from .v3_screening_optimization import ReuseSelector,public_profiles,distinct_banks,mixed_means

SPECS = (
    ('CREDIT-Frozen-K4','frozen',4,False,False),
    ('CREDIT-Probes-K4','probes',4,False,False),
    ('CREDIT-NewCandidate-K4','new-only',4,False,False),
    ('CREDIT-Revalidated-K4','leases',4,False,False),
    ('CREDIT-Revalidated-NoPairs-K4','leases',4,False,True),
    ('Direct-Revalidated-K4','leases',4,True,False),
    ('CREDIT-Revalidated-K6','leases',6,False,False),
    ('Direct-Revalidated-K6','leases',6,True,False),
)


def adaptation(corpus,seed,directory,evidence):
    directory.mkdir()
    updated,fixtures,probe_decisions = reprobe(evidence,seed)
    write(directory/'probe_decisions.json',probe_decisions)
    write(directory/'fixtures.json',fixtures)
    write(directory/'transfer_levels.json',[asdict(p) for p in updated])
    ledger = CallLedger({'target-factorial':8192,'replay_audit':128},200000)
    ledger.tag = 'target-revalidation'
    episodes,reports = designs(corpus.phase('target-revalidation'),candidate_memories(),RevalidationExecutor(),ledger,
        phase='target-factorial',seed=seed,draws=1)
    # designs audits factorial only; explicitly compare one full execution per context here.
    for uid in sorted({e.context.uid for e in episodes}):
        e = next(e for e in reversed(episodes) if e.context.uid==uid)
        full = ledger.execute('replay_audit',RevalidationExecutor(),e.context,e.exposure)
        if (full.state_hash,full.observations,full.events)!=(e.state_hash,e.observations,e.events):
            raise ValueError('target incremental replay mismatch')
    leases,decisions = admit_memories(reports,episodes,candidate_memories(),TransportGate(updated))
    write(directory/'factorial_estimates.json',reports)
    write(directory/'memory_decisions.json',decisions)
    write(directory/'leases.json',leases)
    stream_records(directory/'target_history.jsonl.gz',ledger.records)
    probe_calls = sum(r['api_calls'] for r in fixtures)
    write(directory/'costs.json',ledger.report()|{'role_probes':probe_calls,
        'total_with_probes':ledger.spent+probe_calls})
    novel = {'new_contract'}
    new_only = {k:[m for m in ids if m in novel] for k,ids in leases.items()}
    return (EvidenceGate(updated,novel_ids=novel),EvidenceGate(updated,new_only,novel),
            EvidenceGate(updated,leases,novel),updated,probe_calls,ledger.spent)


def select(name,models,gate,corpus,common,directory,*,depth,direct=False):
    directory.mkdir()
    pool = candidate_memories()
    ledger = CallLedger({'direct-screen':32768,'halving':1024,'pilot':2400,'certification':4800},200000,common)
    ledger.tag = name
    base = make_selector(pool,() if direct else models,gate,corpus,RevalidationExecutor(),ledger,depth=depth)
    selector = ReuseSelector(pool,base.models,gate,base.retrieve,corpus.make,base.executor,ledger,
        mixture=base.mixture,capacity=80,radius=.2,cost_penalty=.0001,
        transported_strata=base.transported_strata)
    unlabeled = selector.contexts('screening-unlabeled',32)
    tasks = selector.contexts('direct-screening',16)
    candidates,feasible = distinct_banks(selector,public_profiles(selector.mixture))
    started = time.perf_counter()
    scores,rows = {},[]
    for bank in candidates:
        means = mixed_means(selector,bank,unlabeled,tasks,'direct' if direct else 'mixed')
        value = worst_mixture(means,list(selector.mixture.values()),.2)-.0001*bank_cost(bank,pool)
        scores[bank] = value
        rows.append({'bank':bank,'means':dict(zip(selector.mixture,means)),'score':value})
    leaders = sorted(scores,key=lambda b:(-scores[b],bank_cost(b,pool),b))[:8]
    write(directory/'screening.json',{'banks':leaders,'scores':{'|'.join(b):scores[b] for b in leaders},
        'all_candidate_scores':rows,'feasible_banks':feasible,'distinct_exposures':len(candidates),
        'transported_strata':sorted(selector.transported_strata),'direct':direct})
    halving = selector.halve(leaders,gamma=1024,finalists=2)
    decision = selector.switch((),halving['finalists'],pilot_n=2 if direct else 16,
        certification_n=200,unlabeled_n=2 if direct else 1000,
        freeze=lambda d:write(directory/'certification_freeze.json',d))
    write(directory/'halving.json',halving)
    write(directory/'switch_decision.json',decision)
    write(directory/'costs.json',ledger.report())
    write(directory/'execution_reuse.json',selector.reuse_records)
    write(directory/'timing.json',{'selection_seconds':time.perf_counter()-started})
    stream_records(directory/'selection.jsonl.gz',ledger.records)
    return tuple(decision['bank']),ledger.report()


def prepare(seed,corpus,output):
    sd = output/f'seed_{seed}';sd.mkdir()
    h = history(LimitedCorpus(corpus),seed,sd/'compact_history')
    models = ensemble(StateAwareOutcomeModel,*h[:3],seed,sd/'models_state')
    no_pairs = ensemble(StateAwareNoPairs,*h[:3],seed,sd/'models_no_pairs')
    _,evidence,common = h[3:]
    old = EvidenceGate(evidence,novel_ids={'new_contract'})
    probed,new_only,leased,updated,probe_calls,target_calls = adaptation(corpus,seed,sd/'adaptation',evidence)
    hc = json.loads((sd/'compact_history/costs.json').read_text())
    direct_common = hc['api_calls']['factorial']+hc['api_calls']['replay_audit']+hc['fixtures']
    banks,costs,policies = {'NoMem':()},{'NoMem':{'selection_api_calls_including_common':0}},{}
    gates = {'frozen':old,'probes':probed,'new-only':new_only,'leases':leased}
    for name,policy,depth,direct,npairs in SPECS:
        calls = direct_common if direct else common
        calls += 0 if policy=='frozen' else probe_calls
        calls += target_calls if policy in ('leases','new-only') else 0
        print(seed,name,'selection',flush=True)
        banks[name],costs[name] = select(name,no_pairs if npairs else models,gates[policy],corpus,calls,
                                       sd/name,depth=depth,direct=direct)
        policies[name] = gates[policy],depth
    write(sd/'frozen_banks.json',{k:list(v) for k,v in banks.items()})
    write(sd/'method_costs.json',costs)
    write(sd/'policy_manifest.json',{name:{'gate':policy,'depth':depth} for name,policy,depth,_,_ in SPECS})
    return {'seed':seed,'corpus':corpus,'banks':banks,'policies':policies}


def score(prepared,output):
    rows = []
    pool,executor = candidate_memories(),RevalidationExecutor()
    with gzip.open(output/'test_traces.jsonl.gz','wt',compresslevel=6) as stream:
        for p in prepared:
            for method,bank in p['banks'].items():
                gate,depth = p['policies'].get(method,(TransportGate(),4))
                for c in p['corpus'].phase('heldout-test'):
                    selected = gate.candidates(c,[m for m in pool if m.uid in bank])
                    a = retriever(depth)(c,tuple(m.uid for m in selected))
                    start = time.perf_counter();e = executor.run(c,a);elapsed = time.perf_counter()-start
                    family,epoch = c.stratum.split('/')
                    nodes = {n.uid:n for n in c.nodes}
                    invalid = sum(not valid_request(ROLE_NAMES[nodes[event['node']].role.key],
                        event['state_before'],event['request_mode'],family,epoch) for event in e.events)
                    allowed = gate.allowed(c) if isinstance(gate,EvidenceGate) else frozenset()
                    row = {'seed':p['seed'],'method':method,'task':c.uid,'family':family,'epoch':epoch,
                        'nodes':len(c.nodes),'success':e.outcome,'api_calls':e.calls,'bank_tokens':bank_cost(bank,pool),
                        'exposure':list(a),'exposure_tokens':sum(m.cost for m in pool if m.uid in a),
                        'direct_evidence_exposures':sum(uid in allowed for uid in a),
                        'contract_invalid_calls':invalid,'verification_known':e.verified is not None,
                        'latency_seconds':elapsed,'state_hash':e.state_hash}
                    rows.append(row)
                    stream.write(json.dumps(row|{'events':e.events},separators=(',',':'))+'\n')
                print(p['seed'],method,'test scored',flush=True)
    write(output/'raw_test_results.json',rows)
    return rows


def run(dataset,output):
    if output.exists():raise FileExistsError('use a fresh output directory')
    torch.set_num_threads(1)
    output.mkdir(parents=True);(output/'source').mkdir()
    sources = [Path(__file__),Path('credit/v3_revalidation.py'),Path('credit/v3_revalidation_data.py'),
        Path('credit/v3_screening_optimization.py'),Path('credit/v3_optimization.py'),
        Path('credit/v3_compact_optimization.py'),Path('credit/v3_optimized_model.py'),
        Path('credit/v3_optimized_data.py'),Path('credit/v3_benchmark.py'),Path('credit/v3_data.py'),
        Path('credit/paired_statistics.py'),*Path('credit/v3').glob('*.py'),Path('credit/stateful.py')]
    write(output/'frozen_protocol.json',{'seeds':SEEDS,'dataset':str(dataset.resolve()),
        'dataset_manifest_sha256':digest(dataset/'manifest.json'),'source_sha256':{str(p):digest(p) for p in sources},
        'paper_sha256':digest('CREDIT_WWW2027_v3.tex'),'agents_sha256':digest('AGENTS.md'),'specifications':SPECS,
        'scope':'unchanged synthetic actor/API/verifier; finite pinned memory evidence leases',
        'role_probes':'only U target roles;1024 fresh fixtures,CP alpha .05/16;old fixtures excluded from new decision',
        'probe_queue':'fixed exploration queue sorted by version/role; adaptive Eq.VOI planner not claimed',
        'target_interventions':'24 balanced contexts/family,8/template,1 sign-randomized fractional draw;incremental replay audited',
        'candidate_pool':'original8 plus new_contract from the public current submit binding;no numeric kernel effects used;novel memory blocked without direct lease even on source',
        'lease_rule':'positive clustered Banzhaf Student-t lower bound,alpha .05/36,min12 contexts and12 eligible node labels;exact observed domain only',
        'lease_scope':'admission to direct evaluation only; does not upgrade D/X roles or transport their source heads;final bank certificate independent',
        'lease_limit':'approximate admission evidence on randomized reserved-slot backgrounds,not a per-memory or arbitrary-bank guarantee',
        'training':'source-only compact64/32/32;three member state/no-pairs ensembles;target labels never fit source heads',
        'costs':'same200000 API allowance,history/probes/target-interventions charged only to methods that use them;Direct excludes unused model validation/calibration',
        'selection':'enumerate finite nine-memory pool,exposure dedup,real16 unsupported screening,gamma1024,finalists2',
        'certification':'independent pilot16 CREDIT/2 Direct;200 paired tasks/stratum/bank;unsupported lambda0;asymptotic alpha sum.10',
        'primary_comparison':'CREDIT-Revalidated-K4 vs CREDIT-Probes-K4 (target admission bundle,including new candidate)',
        'secondary_comparisons':['CREDIT-Revalidated-K4 vs CREDIT-Frozen-K4','CREDIT-Revalidated-K4 vs Direct-Revalidated-K4',
            'CREDIT-Revalidated-K4 vs CREDIT-Revalidated-NoPairs-K4','CREDIT-Revalidated-K4 vs CREDIT-NewCandidate-K4',
            'CREDIT-NewCandidate-K4 vs CREDIT-Probes-K4','CREDIT-Revalidated-K6 vs Direct-Revalidated-K6'],
        'test':'all8 seeds/weights/leases/banks frozen before first test label;256 tasks/stratum;all3 templates seen in target adaptation',
        'development':'known seed1009 training/selection only;no new test outcomes used to choose parameters'})
    for p in sources:(output/'source'/p.name).write_bytes(p.read_bytes())
    prepared = [prepare(seed,Corpus(dataset,seed),output) for seed in SEEDS]
    write(output/'evaluation_freeze.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*')
        if p.is_file() and p.name in ('frozen_banks.json','method_costs.json','policy_manifest.json','leases.json',
                                    'transfer_levels.json','switch_decision.json','certification_freeze.json')})
    rows = score(prepared,output)
    t,p = summarize(rows,'CREDIT-Revalidated-K4',repeats=4000,random_seed=6011)
    write(output/'statistics.json',{'main':t,'paired':p})
    t,p = summarize([r for r in rows if r['method'] in ('CREDIT-Revalidated-K4','CREDIT-Probes-K4')],
                    'CREDIT-Revalidated-K4',repeats=4000,random_seed=6011)
    write(output/'primary_statistics.json',{'main':t,'paired':p,'predeclared_primary_comparisons':1})
    write(output/'summary.json',{'test_episodes':len(rows),'test_api_calls':sum(r['api_calls'] for r in rows),
                               'seeds':SEEDS,'paper_experiments_complete':False})
    write(output/'artifact_manifest.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    print('Completed current-version revalidation experiment',flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a = p.parse_args();run(a.dataset,a.output)
