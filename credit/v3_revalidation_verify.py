"""Reexecute the prospective revalidation study and check evidence boundaries."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import torch

from .paired_statistics import summarize
from .v3.drift import Level,ProbeEvidence,TransportGate
from .v3.experiment import digest,write
from .v3.local import retriever,ROLE_NAMES,valid_request
from .v3.selection import bank_cost
from .v3_data import Corpus
from .v3_compact_optimization import LimitedCorpus
from .v3_optimization import history
from .v3_optimization_verify import read,records,compare,compare_traces,check_models
from .v3_screening_verify import check_guards
from .v3_revalidation import EvidenceGate,candidate_memories,RevalidationExecutor
from .v3_revalidation_experiment import adaptation,select,SPECS


def restore_gates(sd):
    def evidence(path):
        return [ProbeEvidence(**(r|{'level':Level(r['level'])})) for r in read(path)]
    old = evidence(sd/'compact_history/transfer_levels.json')
    new = evidence(sd/'adaptation/transfer_levels.json')
    leases = read(sd/'adaptation/leases.json')
    novel = {'new_contract'}
    return {'frozen':EvidenceGate(old,novel_ids=novel),
            'probes':EvidenceGate(new,novel_ids=novel),
            'new-only':EvidenceGate(new,{k:[m for m in ids if m in novel] for k,ids in leases.items()},novel),
            'leases':EvidenceGate(new,leases,novel)}


def check_references(directory):
    physical = list(records(directory/'selection.jsonl.gz'))
    references = read(directory/'execution_reuse.json')
    for r in references:
        record = physical[r['physical_record']];e = record['episode']
        if (record['phase']!=r['phase'] or e['context']['uid']!=r['context']
                or e['exposure']!=r['exposure'] or not e['replay_valid']):
            raise ValueError('invalid exact execution reuse')
    return len(references)


def audit(output,selection_only=False):
    torch.set_num_threads(1)
    protocol,dataset = check_guards(output,selection_only)
    compare(protocol['specifications'],json.loads(json.dumps(SPECS)),'frozen method specifications')
    cache = output.with_name(output.name+'_selection_audit');cache.mkdir(exist_ok=True)
    dependencies = [Path(__file__),Path('credit/v3_optimization_verify.py'),Path('credit/v3_screening_verify.py')]
    deps = {str(p):digest(p) for p in dependencies}
    totals = dict(physical_episodes=0,physical_calls_with_fixtures=0,reuse_references=0,
                  models=0,selections=0,certificates=0,test_episodes=0,target_intervention_episodes=0,
                  fresh_role_probe_calls=0)
    with tempfile.TemporaryDirectory(prefix='credit-revalidation-audit-') as temporary:
        for seed in protocol['seeds']:
            sd = output/f'seed_{seed}'
            while selection_only and not (sd/'method_costs.json').exists():time.sleep(2)
            hashes = {str(p.relative_to(sd)):digest(p) for p in sd.rglob('*') if p.is_file()}
            proof = cache/f'seed_{seed}.json'
            if proof.exists():
                saved = read(proof)
                if (not saved['verified'] or saved['seed_artifacts']!=hashes or saved['auditor_sources']!=deps
                        or saved['source_sha256']!=protocol['source_sha256']):
                    raise ValueError('immutable selection audit no longer matches')
                counts = saved['counts'];print('reused selection audit',seed,flush=True)
            else:
                counts = {k:0 for k in totals if k!='test_episodes'}
                corpus = Corpus(dataset,seed);td = Path(temporary)/f'seed_{seed}';td.mkdir()
                h = history(LimitedCorpus(corpus),seed,td/'compact_history')
                for name in ('factorial_estimates.json','transfer_levels.json','fixtures.json','costs.json'):
                    compare(read(td/'compact_history'/name),read(sd/'compact_history'/name),'source history/'+name)
                n,c = compare_traces(td/'compact_history/history.jsonl.gz',sd/'compact_history/history.jsonl.gz')
                counts['physical_episodes']+=n
                counts['physical_calls_with_fixtures']+=c+read(sd/'compact_history/costs.json')['fixtures']
                state = check_models(sd/'models_state',*h[:3]);no_pairs = check_models(sd/'models_no_pairs',*h[:3])
                counts['models']=6
                _,evidence,common = h[3:]
                probed,new_only,leased,updated,probe_calls,target_calls = adaptation(corpus,seed,td/'adaptation',evidence)
                for name in ('probe_decisions.json','fixtures.json','transfer_levels.json','factorial_estimates.json',
                             'memory_decisions.json','leases.json','costs.json'):
                    compare(read(td/'adaptation'/name),read(sd/'adaptation'/name),'adaptation/'+name)
                n,c = compare_traces(td/'adaptation/target_history.jsonl.gz',sd/'adaptation/target_history.jsonl.gz')
                counts['target_intervention_episodes']=n;counts['physical_episodes']+=n
                counts['fresh_role_probe_calls']=probe_calls;counts['physical_calls_with_fixtures']+=c+probe_calls
                gates = {'frozen':EvidenceGate(evidence,novel_ids={'new_contract'}),
                         'probes':probed,'new-only':new_only,'leases':leased}
                # Direct leases may admit candidates, but cannot change source-head permission.
                for cxt in corpus.phase('target-revalidation'):
                    if leased.transported(cxt)!=probed.transported(cxt):raise ValueError('lease upgraded transport')
                    for node in cxt.nodes:
                        if leased.level(node)!=probed.level(node):raise ValueError('lease changed role class')
                hc = read(sd/'compact_history/costs.json')
                direct_common = hc['api_calls']['factorial']+hc['api_calls']['replay_audit']+hc['fixtures']
                banks,costs = read(sd/'frozen_banks.json'),read(sd/'method_costs.json')
                compare(banks['NoMem'],[],'NoMem bank')
                for name,policy,depth,direct,npairs in SPECS:
                    calls = direct_common if direct else common
                    calls += 0 if policy=='frozen' else probe_calls
                    calls += target_calls if policy in ('leases','new-only') else 0
                    bank,cost = select(name,no_pairs if npairs else state,gates[policy],corpus,calls,td/name,
                                       depth=depth,direct=direct)
                    compare(list(bank),banks[name],name+' bank');compare(cost,costs[name],name+' accounting')
                    if cost['selection_api_calls_including_common']>200000:raise ValueError('hard call cap exceeded')
                    for filename in ('screening.json','halving.json','switch_decision.json','costs.json',
                                     'certification_freeze.json','execution_reuse.json'):
                        compare(read(td/name/filename),read(sd/name/filename),name+'/'+filename)
                    n,c = compare_traces(td/name/'selection.jsonl.gz',sd/name/'selection.jsonl.gz')
                    counts['physical_episodes']+=n;counts['physical_calls_with_fixtures']+=c
                    counts['reuse_references']+=check_references(sd/name)
                    counts['selections']+=1;counts['certificates']+=len(read(sd/name/'switch_decision.json')['certificates'])
                    frozen = read(sd/name/'certification_freeze.json')
                    if direct and (frozen['model_report'] or any(v!=0 for r in frozen['pilot_weights'].values() for v in r.values())):
                        raise ValueError('Direct used a predictor')
                if hashes!={str(p.relative_to(sd)):digest(p) for p in sd.rglob('*') if p.is_file()}:
                    raise ValueError('seed changed during audit')
                write(proof,{'verified':True,'seed_artifacts':hashes,'source_sha256':protocol['source_sha256'],
                             'auditor_sources':deps,'counts':counts})
                print('selection audit passed',seed,counts,flush=True)
            for key,value in counts.items():totals[key]+=value
    if selection_only:
        print('All selection audits complete; heldout outcomes not read',flush=True);return
    rows = read(output/'raw_test_results.json')
    indexed = {(r['seed'],r['method'],r['task']):r for r in rows}
    if len(indexed)!=len(rows):raise ValueError('duplicate raw test rows')
    seeds = protocol['seeds'];methods = {'NoMem',*(s[0] for s in SPECS)}
    contexts = {seed:{c.uid:c for c in Corpus(dataset,seed).phase('heldout-test')} for seed in seeds}
    expected = {(seed,method,uid) for seed in seeds for method in methods for uid in contexts[seed]}
    if set(indexed)!=expected:raise ValueError('incomplete preregistered test coverage')
    policies = {}
    for seed in seeds:
        sd = output/f'seed_{seed}';gates = restore_gates(sd)
        compare(read(sd/'policy_manifest.json'),{name:{'gate':policy,'depth':depth} for name,policy,depth,_,_ in SPECS},'policies')
        policies[seed] = gates,read(sd/'frozen_banks.json')
    spec = {name:(policy,depth) for name,policy,depth,_,_ in SPECS}
    seen = set();executor,pool = RevalidationExecutor(),candidate_memories()
    for row in records(output/'test_traces.jsonl.gz'):
        key = row['seed'],row['method'],row['task']
        if key in seen:raise ValueError('duplicate heldout trace')
        seen.add(key);compare({k:v for k,v in row.items() if k!='events'},indexed[key],'raw test row')
        c = contexts[row['seed']][row['task']];gates,banks = policies[row['seed']]
        bank = tuple(banks[row['method']]);policy,depth = spec.get(row['method'],('none',4))
        gate = gates.get(policy,TransportGate())
        exposure = retriever(depth)(c,tuple(m.uid for m in gate.candidates(c,[m for m in pool if m.uid in bank])))
        e = executor.run(c,exposure);nodes = {n.uid:n for n in c.nodes}
        invalid = sum(not valid_request(ROLE_NAMES[nodes[event['node']].role.key],event['state_before'],
                      event['request_mode'],row['family'],row['epoch']) for event in e.events)
        allowed = gate.allowed(c) if isinstance(gate,EvidenceGate) else frozenset()
        compare(list(exposure),row['exposure'],'test exposure');compare(json.loads(json.dumps(e.events)),row['events'],'test events')
        if (e.outcome!=row['success'] or e.calls!=row['api_calls'] or e.state_hash!=row['state_hash']
                or bank_cost(bank,pool)!=row['bank_tokens'] or not e.replay_valid
                or sum(m.cost for m in pool if m.uid in exposure)!=row['exposure_tokens']
                or sum(uid in allowed for uid in exposure)!=row['direct_evidence_exposures']
                or invalid!=row['contract_invalid_calls'] or (e.verified is not None)!=row['verification_known']):
            raise ValueError('heldout execution mismatch')
        if 'new_contract' in exposure and 'new_contract' not in allowed:raise ValueError('novel memory exposed without lease')
        totals['physical_episodes']+=1;totals['physical_calls_with_fixtures']+=e.calls;totals['test_episodes']+=1
    if seen!=expected:raise ValueError('missing heldout traces')
    for filename,names in [('statistics.json',None),('primary_statistics.json',{'CREDIT-Revalidated-K4','CREDIT-Probes-K4'})]:
        subset = rows if names is None else [r for r in rows if r['method'] in names]
        t,p = summarize(subset,'CREDIT-Revalidated-K4',repeats=4000,random_seed=6011)
        stored = read(output/filename);compare(t,stored['main'],'statistics');compare(p,stored['paired'],'paired inference')
    compare(read(output/'summary.json'),{'test_episodes':len(rows),'test_api_calls':sum(r['api_calls'] for r in rows),
            'seeds':seeds,'paper_experiments_complete':False},'summary')
    check_guards(output,False)
    report = {'verified':True,**totals,'auditor_sources':deps,'dataset_manifest_sha256':digest(dataset/'manifest.json'),
              'benchmark_manifest_sha256':digest(output/'artifact_manifest.json'),
              'limits':'same canonical reconstruction; frozen models/calibration checked,optimizer not retrained;finite simulator;approximate admission and final asymptotic certificates'}
    write(output.with_name(output.name+'_audit.json'),report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--selection-only',action='store_true');a = p.parse_args();audit(a.results,a.selection_only)
