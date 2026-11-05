"""Reexecute screening study histories, physical calls, reuse and certification."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
import time

import torch

from .paired_statistics import summarize
from .v3.audit import load_context
from .v3.drift import Level, ProbeEvidence, TransportGate
from .v3.experiment import digest, write
from .v3.local import StatefulExecutor, memories, retriever
from .v3.selection import bank_cost
from .v3_data import Corpus
from .v3_compact_optimization import LimitedCorpus
from .v3_optimization import history, select as previous_select
from .v3_optimization_verify import read, records, compare, compare_traces, check_models
from .v3_screening_optimization import select, SPECS


def check_guards(output, selection_only):
    p = read(output/'frozen_protocol.json')
    for name, value in p['source_sha256'].items():
        if digest(name) != value or digest(output/'source'/Path(name).name) != value:
            raise ValueError('changed frozen source: '+name)
    for name, key in [('AGENTS.md','agents_sha256'),('CREDIT_WWW2027_v3.tex','paper_sha256')]:
        if digest(name) != p[key]:
            raise ValueError('protected file changed: '+name)
    dataset = Path(p['dataset'])
    if digest(dataset/'manifest.json') != p['dataset_manifest_sha256']:
        raise ValueError('changed dataset manifest')
    for name,value in read(dataset/'manifest.json').items():
        if digest(dataset/name) != value:
            raise ValueError('changed dataset: '+name)
    if not selection_only:
        for name,value in read(output/'artifact_manifest.json').items():
            if digest(output/name) != value:
                raise ValueError('changed experimental artifact: '+name)
        for name,value in read(output/'evaluation_freeze.json').items():
            if digest(output/name) != value:
                raise ValueError('changed evaluation freeze: '+name)
    return p, dataset


def audit(output, selection_only=False):
    torch.set_num_threads(1)
    protocol,dataset = check_guards(output,selection_only)
    cache = output.with_name(output.name+'_selection_audit')
    cache.mkdir(exist_ok=True)
    dependency_paths = [Path(__file__),Path('credit/v3_optimization_verify.py')]
    deps = {str(p):digest(p) for p in dependency_paths}
    totals = dict(physical_episodes=0, physical_calls_with_fixtures=0,
                  reuse_references=0, models=0, selections=0, certificates=0, test_episodes=0)
    with tempfile.TemporaryDirectory(prefix='credit-screening-audit-') as temporary:
        for seed in protocol['seeds']:
            sd = output/f'seed_{seed}'
            while selection_only and not (sd/'method_costs.json').exists():
                time.sleep(2)
            hashes = {str(p.relative_to(sd)):digest(p) for p in sd.rglob('*') if p.is_file()}
            proof = cache/f'seed_{seed}.json'
            if proof.exists():
                saved = read(proof)
                if not saved['verified'] or saved['seed_artifacts'] != hashes or saved['auditor_sources'] != deps:
                    raise ValueError('immutable selection proof no longer matches')
                compare(saved['source_sha256'],protocol['source_sha256'],'cached experiment sources')
                counts = saved['counts']
                print('reused immutable selection audit',seed,flush=True)
            else:
                counts = {k:0 for k in totals if k != 'test_episodes'}
                corpus = Corpus(dataset,seed)
                td = Path(temporary)/f'seed_{seed}'
                td.mkdir()
                h = history(LimitedCorpus(corpus),seed,td/'compact_history')
                for name in ('factorial_estimates.json','transfer_levels.json','fixtures.json','costs.json'):
                    compare(read(td/'compact_history'/name),read(sd/'compact_history'/name),'history/'+name)
                n,c = compare_traces(td/'compact_history/history.jsonl.gz',sd/'compact_history/history.jsonl.gz')
                counts['physical_episodes'] += n
                counts['physical_calls_with_fixtures'] += c+read(sd/'compact_history/costs.json')['fixtures']
                models = check_models(sd/'models_state',*h[:3])
                no_pairs = check_models(sd/'models_no_pairs',*h[:3])
                counts['models'] = 6
                gate,evidence,common = h[3:]
                hc = read(sd/'compact_history/costs.json')
                direct_common = hc['api_calls']['factorial']+hc['api_calls']['replay_audit']+hc['fixtures']
                banks,costs = read(sd/'frozen_banks.json'),read(sd/'method_costs.json')
                compare(banks['NoMem'],[],'NoMem bank')
                for name,mode,depth in SPECS:
                    active = no_pairs if mode == 'mixed-no-pairs' else models
                    if mode == 'previous':
                        bank,cost = previous_select(name,active,gate,corpus,common,td/name,depth=depth)
                    else:
                        bank,cost = select(name,active,gate,corpus,direct_common if mode == 'direct' else common,
                            td/name,mode='mixed' if mode == 'mixed-no-pairs' else mode,depth=depth)
                    compare(list(bank),banks[name],name+' bank')
                    compare(cost,costs[name],name+' charged calls')
                    if cost['selection_api_calls_including_common'] > 150000:
                        raise ValueError('hard API call cap exceeded')
                    names = ['screening.json','halving.json','switch_decision.json','costs.json','certification_freeze.json']
                    if mode != 'previous':
                        names.append('execution_reuse.json')
                    for filename in names:
                        if (sd/name/filename).exists():
                            compare(read(td/name/filename),read(sd/name/filename),name+'/'+filename)
                    n,c = compare_traces(td/name/'selection.jsonl.gz',sd/name/'selection.jsonl.gz')
                    counts['physical_episodes'] += n
                    counts['physical_calls_with_fixtures'] += c
                    counts['selections'] += 1
                    decision = read(sd/name/'switch_decision.json')
                    counts['certificates'] += len(decision['certificates'])
                    if mode != 'previous':
                        physical = list(records(sd/name/'selection.jsonl.gz'))
                        for reuse in read(sd/name/'execution_reuse.json'):
                            record = physical[reuse['physical_record']]
                            ep = record['episode']
                            # Every reused value must point to an identical task/exposure in the same independent phase.
                            if (record['phase'] != reuse['phase'] or ep['context']['uid'] != reuse['context']
                                    or ep['exposure'] != reuse['exposure'] or not ep['replay_valid']):
                                raise ValueError('invalid physical execution reuse reference')
                            counts['reuse_references'] += 1
                    if mode == 'direct':
                        frozen = read(sd/name/'certification_freeze.json')
                        if frozen['model_report'] or any(v != 0 for row in frozen['pilot_weights'].values() for v in row.values()):
                            raise ValueError('pure Direct used a predictor')
                write(proof,{'verified':True,'seed_artifacts':hashes,'source_sha256':protocol['source_sha256'],
                             'auditor_sources':deps,'counts':counts})
                print('selection audit passed',seed,counts,flush=True)
            for k,value in counts.items():
                totals[k] += value
    if not selection_only:
        indexed = {(r['seed'],r['method'],r['task']):r for r in read(output/'raw_test_results.json')}
        seen = set()
        corpora = {seed:Corpus(dataset,seed) for seed in protocol['seeds']}
        contexts = {seed:{c.uid:c for c in corpus.phase('heldout-test')} for seed,corpus in corpora.items()}
        policies = {}
        for seed in protocol['seeds']:
            sd = output/f'seed_{seed}'
            g = TransportGate([ProbeEvidence(**(r|{'level':Level(r['level'])}))
                               for r in read(sd/'compact_history/transfer_levels.json')])
            policies[seed] = g,read(sd/'frozen_banks.json')
        for row in records(output/'test_traces.jsonl.gz'):
            key = row['seed'],row['method'],row['task']
            if key in seen:
                raise ValueError('duplicate heldout trace')
            seen.add(key)
            compare({k:v for k,v in row.items() if k != 'events'},indexed[key],'raw test row')
            c = contexts[row['seed']][row['task']]
            gate,banks = policies[row['seed']]
            bank = tuple(banks[row['method']])
            depth = 6 if row['method'].endswith('K6') else 4
            a = retriever(depth)(c,tuple(m.uid for m in gate.candidates(c,[m for m in memories() if m.uid in bank])))
            e = StatefulExecutor().run(c,a)
            compare(list(a),row['exposure'],'heldout exposure')
            compare(json.loads(json.dumps(e.events)),row['events'],'heldout physical events')
            if (e.outcome != row['success'] or e.calls != row['api_calls'] or e.state_hash != row['state_hash']
                    or bank_cost(bank,memories()) != row['bank_tokens'] or not e.replay_valid):
                raise ValueError('heldout execution mismatch')
            totals['physical_episodes'] += 1
            totals['physical_calls_with_fixtures'] += e.calls
            totals['test_episodes'] += 1
        if seen != set(indexed):
            raise ValueError('missing heldout outcomes')
        rows = list(indexed.values())
        t,p = summarize(rows,'CREDIT-Mixed-K4',repeats=4000,random_seed=5003)
        compare({'main':t,'paired':p},read(output/'statistics.json'),'paired statistics')
        t,p = summarize([r for r in rows if r['method'] in ('CREDIT-Mixed-K4','CREDIT-Unique-K4')],
                        'CREDIT-Mixed-K4',repeats=4000,random_seed=5003)
        compare({'main':t,'paired':p,'predeclared_primary_comparisons':1},read(output/'primary_statistics.json'),'primary statistics')
        # Verify raw artifacts again after the long audit.
        check_guards(output,False)
        report = {'verified':True,**totals,'auditor_sources':deps,
            'limits':'frozen-model replay and same selector reconstruction; optimizer not retrained; simulator only'}
        write(output.with_name(output.name+'_audit.json'),report)
        print(json.dumps(report),flush=True)
    else:
        print('All preregistered selection audits complete',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--selection-only',action='store_true')
    a = p.parse_args()
    audit(a.results,a.selection_only)
