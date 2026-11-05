"""Replay the prospective optimization with frozen weights and fresh accounting."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
import torch

from .paired_statistics import summarize
from .v3.experiment import digest, write
from .v3.local import StatefulExecutor, memories, retriever
from .v3.drift import Level,ProbeEvidence,TransportGate
from .v3.selection import bank_cost
from .v3_data import Corpus
from .v3_benchmark import CachedModel
from .v3_optimization import history, select
from .v3_optimized_model import OutcomeModel, load_model
from .v3_compact_optimization import LimitedCorpus


def read(path):
    return json.loads(path.read_text())


def records(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:
            yield json.loads(line)


def compare(actual, expected, where):
    if json.loads(json.dumps(actual)) != expected:
        raise ValueError('audit mismatch: '+where)


def compare_traces(actual_path, expected_path):
    from itertools import zip_longest
    count, calls = 0, 0
    for actual, expected in zip_longest(records(actual_path),records(expected_path)):
        if actual != expected:
            raise ValueError('execution trace mismatch: '+str(expected_path))
        count += 1
        calls += actual['episode']['calls']
    return count, calls


def check_models(directory, train, validation, calibration):
    models = []
    contexts = [e.context for e in train]
    for path in sorted(directory.glob('outcome_model_*.pt')):
        model = load_model(path,memories(),contexts)
        data = torch.load(path,map_location='cpu',weights_only=False)
        raw = model.predict_many([(e.context,e.exposure) for e in calibration],calibrated=False)
        refit = IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(raw,[e.outcome for e in calibration])
        for actual, expected in ((refit.X_thresholds_,data['calibration_x']),
                                 (refit.y_thresholds_,data['calibration_y'])):
            if not np.allclose(actual,expected,rtol=1e-7,atol=1e-7):
                raise ValueError('isotonic calibration reconstruction')
        if type(model) is not OutcomeModel:
            packed,labels,recovery = model.pack_observations(validation)
            packed,counts = model.compress(packed,labels,recovery)
            with torch.no_grad():
                nll = float(model.compressed_loss(model.heads(*packed),counts))
            if abs(nll-model.fit_report['selected_validation_nll']) > 1e-6:
                raise ValueError('selected weight validation NLL')
            best, selected = float('inf'),None
            for point in model.fit_report['trajectory']:
                if point['validation_nll'] < best-1e-4:
                    best,selected = point['validation_nll'],point['epoch']
            if selected != model.fit_report['selected_epoch']:
                raise ValueError('best validation checkpoint selection')
            if model.fit_report['optimizer_stationarity_claimed']:
                raise ValueError('early stopping does not imply optimizer stationarity')
        models.append(CachedModel(model))
    if len(models) != 3:
        raise ValueError('missing ensemble member')
    return tuple(models)


def audit(output, *, selection_only=False):
    torch.set_num_threads(1)
    protocol = read(output/'frozen_protocol.json')
    compact = 'specified_before_main_test_scoring' in protocol
    if not selection_only and not (output/'artifact_manifest.json').exists():
        raise ValueError('benchmark must finish before audit')
    manifest = {} if selection_only else read(output/'artifact_manifest.json')
    for name,value in manifest.items():
        if digest(output/name) != value:
            raise ValueError('changed frozen artifact: '+name)
    for name,value in protocol['source_sha256'].items():
        if digest(name) != value or digest(output/'source'/Path(name).name) != value:
            raise ValueError('changed source: '+name)
    if digest('AGENTS.md') != protocol['agents_sha256'] or digest('CREDIT_WWW2027_v3.tex') != protocol['paper_sha256']:
        raise ValueError('protected manuscript/AGENTS changed')
    evaluation = {} if selection_only else read(output/'evaluation_freeze.json')
    for name,value in evaluation.items():
        if digest(output/name) != value:
            raise ValueError('changed pre-test decision')
    dataset = Path(protocol['dataset'])
    if digest(dataset/'manifest.json') != protocol['dataset_manifest_sha256']:
        raise ValueError('changed dataset manifest')
    for name,value in read(dataset/'manifest.json').items():
        if digest(dataset/name) != value:
            raise ValueError('changed frozen data')
    episodes, calls, certificates, weights, selections = 0,0,0,0,0
    seed_data = {}
    cache = output.with_name(output.name+'_selection_audit')
    if selection_only:
        cache.mkdir(exist_ok=True)

    def restore_seed_inputs(sd,corpus):
        history_name = 'compact_history' if compact else 'expanded_history'
        def gate_at(name):
            return TransportGate([ProbeEvidence(**(r | {'level':Level(r['level'])}))
                                  for r in read(sd/name/'transfer_levels.json')])
        gate = gate_at(history_name)
        old_gate = gate_at('legacy_history') if not compact else gate
        banks = read(sd/'frozen_banks.json')
        policies = {name:(old_gate if name == 'CREDIT-Legacy-K4' else gate,
                          6 if name.endswith('K6') else 4)
                    for name in banks if name != 'NoMem'}
        return corpus,banks,policies
    with tempfile.TemporaryDirectory(prefix='credit-v3-state-audit-') as tmp:
        root = Path(tmp)
        for seed in protocol['seeds']:
            sd = output/f'seed_{seed}'
            while selection_only and not (sd/'method_costs.json').exists():
                time.sleep(2)
            seed_hashes = {str(p.relative_to(sd)):digest(p) for p in sd.rglob('*') if p.is_file()}
            proof_path = cache/f'seed_{seed}.json'
            if proof_path.exists():
                proof = read(proof_path)
                if (proof['seed_artifacts'] != seed_hashes or proof['auditor_sha256'] != digest(__file__)
                        or proof['source_sha256'] != protocol['source_sha256'] or not proof['verified']):
                    raise ValueError('cached selection audit no longer matches frozen files')
                n = proof['counts']
                episodes += n['episodes']; calls += n['calls']; certificates += n['certificates']
                weights += n['models']; selections += n['selections']
                if not selection_only:
                    seed_data[seed] = restore_seed_inputs(sd,Corpus(dataset,seed))
                print('reused verified immutable selection audit',seed,flush=True)
                continue
            before = episodes,calls,certificates,weights,selections
            td = root/f'seed_{seed}'
            td.mkdir()
            corpus = Corpus(dataset,seed)
            if compact:
                expanded = history(LimitedCorpus(corpus),seed,td/'compact_history')
                history_names = ('compact_history',)
            else:
                legacy = history(corpus,seed,td/'legacy_history',legacy=True)
                expanded = history(corpus,seed,td/'expanded_history')
                history_names = ('legacy_history','expanded_history')
            for name in history_names:
                for filename in ('factorial_estimates.json','transfer_levels.json','fixtures.json','costs.json'):
                    compare(read(td/name/filename),read(sd/name/filename),name+'/'+filename)
                n,c = compare_traces(td/name/'history.jsonl.gz',sd/name/'history.jsonl.gz')
                episodes += n
                calls += c + read(sd/name/'costs.json')['fixtures']
            state = check_models(sd/'models_state',*expanded[:3])
            no_pairs = check_models(sd/'models_no_pairs',*expanded[:3])
            if not compact:
                old = check_models(sd/'models_legacy',*legacy[:3])
                coverage = check_models(sd/'models_coverage',*expanded[:3])
            weights += 6 if compact else 12
            gate,evidence,common = expanded[3:]
            hc = read(sd/('compact_history' if compact else 'expanded_history')/'costs.json')
            direct_common = hc['api_calls']['factorial']+hc['api_calls']['replay_audit']+hc['fixtures']
            costs = read(sd/'method_costs.json')
            banks = read(sd/'frozen_banks.json')
            policies = {}
            specifications = ([
                ('CREDIT-Compact-K4',state,gate,common,4,False),
                ('CREDIT-Compact-K6',state,gate,common,6,False),
                ('CREDIT-Compact-NoPairs-K6',no_pairs,gate,common,6,False),
                ('Direct-Compact-K4',(),gate,direct_common,4,True),
                ('Direct-Compact-K6',(),gate,direct_common,6,True)] if compact else [
                ('CREDIT-Legacy-K4',old,legacy[3],legacy[-1],4,False),
                ('CREDIT-Coverage-K4',coverage,gate,common,4,False),
                ('CREDIT-State-K4',state,gate,common,4,False),
                ('CREDIT-State-NoPairs-K4',no_pairs,gate,common,4,False),
                ('CREDIT-State-K6',state,gate,common,6,False),
                ('Direct-K4',(),gate,direct_common,4,True),
                ('Direct-K6',(),gate,direct_common,6,True),
                ('Direct-Matched-K4',(),gate,direct_common,4,True),
                ('Direct-Matched-K6',(),gate,direct_common,6,True)])
            for name,models,active_gate,history_calls,depth,direct in specifications:
                cap = costs[name]['call_allowance']
                if name.startswith('Direct-Matched'):
                    expected_cap = costs[f'CREDIT-State-K{depth}']['selection_api_calls_including_common']
                    if cap != expected_cap:
                        raise ValueError('matched hard call cap')
                bank, actual_cost = select(name,models,active_gate,corpus,history_calls,td/name,
                                            depth=depth,cap=cap,direct=direct)
                compare(list(bank),banks[name],'selected bank')
                compare(actual_cost,costs[name],'actual per-method calls')
                for filename in ('screening.json','halving.json','switch_decision.json','costs.json'):
                    compare(read(td/name/filename),read(sd/name/filename),name+'/'+filename)
                if (sd/name/'certification_freeze.json').exists():
                    compare(read(td/name/'certification_freeze.json'),read(sd/name/'certification_freeze.json'),
                            'frozen predictions/weights')
                    if direct:
                        frozen = read(sd/name/'certification_freeze.json')
                        if any(v != 0 for row in frozen['pilot_weights'].values() for v in row.values()):
                            raise ValueError('Direct used a nonzero surrogate weight')
                n,c = compare_traces(td/name/'selection.jsonl.gz',sd/name/'selection.jsonl.gz')
                episodes += n
                calls += c
                certificates += len(read(sd/name/'switch_decision.json')['certificates'])
                selections += 1
                policies[name] = (active_gate,depth)
            seed_data[seed] = corpus,banks,policies
            if selection_only:
                if seed_hashes != {str(p.relative_to(sd)):digest(p) for p in sd.rglob('*') if p.is_file()}:
                    raise ValueError('seed artifacts changed during selection audit')
                counts = dict(zip(('episodes','calls','certificates','models','selections'),
                                  (a-b for a,b in zip((episodes,calls,certificates,weights,selections),before))))
                write(proof_path,{'verified':True,'seed':seed,'seed_artifacts':seed_hashes,
                                  'counts':counts,'source_sha256':protocol['source_sha256'],
                                  'auditor_sha256':digest(__file__)})
                del seed_data[seed]
            print('audited seed',seed,'all selection/certification traces',flush=True)
    if selection_only:
        write(cache/'summary.json',{'verified':True,'seeds':protocol['seeds'],
              'episodes_reexecuted':episodes,'calls_reexecuted':calls,
              'test_outcomes_inspected':False,'auditor_sha256':digest(__file__)})
        print('All frozen selection traces audited; held-out results were not read',flush=True)
        return
    indexed = {(r['seed'],r['method'],r['task']):r for r in read(output/'raw_test_results.json')}
    if len(indexed) != read(output/'summary.json')['test_episodes']:
        raise ValueError('duplicate held-out rows')
    seen = set()
    engine,pool = StatefulExecutor(),memories()
    for r in records(output/'test_traces.jsonl.gz'):
        corpus,banks,policies = seed_data[r['seed']]
        c = corpus.make(r['family']+'/'+r['epoch'],'heldout-test',int(r['task'].rsplit(':',1)[1]))
        bank = banks[r['method']]
        if r['method'] == 'NoMem':
            eligible,depth = (),4
        else:
            gate,depth = policies[r['method']]
            eligible = tuple(m.uid for m in gate.candidates(c,[m for m in pool if m.uid in bank]))
        exposure = retriever(depth)(c,eligible)
        e = engine.run(c,exposure)
        if (r['task'] != c.uid or list(exposure) != r['exposure'] or list(e.events) != r['events']
                or e.state_hash != r['state_hash'] or e.calls != r['api_calls'] or e.outcome != r['success']
                or bank_cost(bank,pool) != r['bank_tokens']):
            raise ValueError('held-out actual execution')
        key = r['seed'],r['method'],r['task']
        if key in seen or indexed[key] != {k:v for k,v in r.items() if k != 'events'}:
            raise ValueError('held-out coverage/row integrity')
        seen.add(key)
        episodes += 1
        calls += e.calls
    if seen != set(indexed):
        raise ValueError('missing test trace')
    rows = list(indexed.values())
    stat_specs = ([('statistics.json',None,'CREDIT-Compact-K4'),
                   ('k6_statistics.json',('CREDIT-Compact-K6','CREDIT-Compact-NoPairs-K6','Direct-Compact-K6'),
                    'CREDIT-Compact-K6')] if compact else
                  [('statistics.json',None,'CREDIT-State-K4'),
                   ('primary_statistics.json',('CREDIT-State-K4','CREDIT-Legacy-K4'),'CREDIT-State-K4')])
    for filename,names,primary in stat_specs:
        subset = rows if names is None else [r for r in rows if r['method'] in names]
        table,paired = summarize(subset,primary,repeats=4000,random_seed=4099)
        stored = read(output/filename)
        compare(table,stored['main'],'summary table')
        compare(paired,stored['paired'],'paired bootstrap/sign-flip statistics')
    report = {'verified':True,'actual_episodes_reexecuted':episodes,
              'api_calls_reexecuted_including_fixtures':calls,'selections_reconstructed':selections,
              'certificates_reconstructed':certificates,'model_checkpoints_checked':weights,
              'heldout_episodes_reexecuted':len(rows),'direct_models_used':0,
              'calibration_refit':True,'selected_validation_nll_recomputed':True,
              'optimizer_retrained':False,
              'selection_audit_cache':str(cache) if cache.exists() else None,
              'limit':'reconstruction uses the same canonical selector; it verifies recorded execution and accounting, not a separate algorithm implementation',
              'auditor_sha256':digest(__file__)}
    write(output/'execution_audit.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--selection-only',action='store_true',help='audit each frozen seed while later seeds still train; never read held-out results')
    a = p.parse_args()
    audit(a.results,selection_only=a.selection_only)
