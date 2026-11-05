"""Prospective optimization benchmark: fixed simulator, fresh tasks and banks.

The corrected Direct baseline uses no model and only two pilot tasks per
stratum (its PPI weights are identically zero). In matched comparisons its
hard call cap is the observed CREDIT expense, fixed before any test labels.
Unused calls are reported; equal actual spending is never asserted.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch

from .paired_statistics import summarize
from .v3.audit import SavedCalibration
from .v3.drift import TransportGate
from .v3.experiment import collect_designs, digest, write
from .v3.factorial import admitted_pairs
from .v3.inference import worst_mixture
from .v3.local import StatefulExecutor, differential_probes, memories, ranked, stable_seed
from .v3.outcome import OutcomeModel
from .v3.selection import bank_cost
from .v3_benchmark import CachedModel, CallLedger, make_selector, stream_records, score
from .v3_data import Corpus
from .v3_optimized_data import SEEDS
from .v3_optimized_model import (ValidatedOutcomeModel, StateAwareOutcomeModel,
                                 StateAwareNoPairs)


def designs(contexts, pool, executor, ledger, *, phase, seed, draws):
    """Use all three slots for submit-only tasks; six slots for full workflows."""
    episodes, reports = [], []
    for c in contexts:
        eligible = [m for m in pool if any(m.eligible(n.role) for n in c.nodes)]
        k1 = min(4, len(ranked(c, tuple(m.uid for m in eligible))), max(len(eligible)-1, 0))
        k2 = min(2, len(eligible)-k1)
        observed, report = collect_designs([c], pool, executor, ledger, phase=phase,
                                           seed=stable_seed(seed, phase, c.uid), draws=draws,
                                           k1=k1, k2=k2, audit=phase == 'factorial')
        episodes.extend(observed)
        reports.extend(report)
    return episodes, reports


def history(corpus, seed, directory, *, legacy=False):
    directory.mkdir()
    pool, executor = memories(), StatefulExecutor()
    ledger = CallLedger({'factorial':4096, 'validation':2048, 'calibration':2048,
                         'replay_audit':128}, 160000)
    ledger.tag = 'shared-history'
    training_phase = 'legacy-training' if legacy else 'training'
    calibration_phase = 'legacy-calibration' if legacy else 'calibration'
    train, reports = designs(corpus.phase(training_phase), pool, executor, ledger,
                              phase='factorial', seed=seed, draws=2 if legacy else 1)
    validation = []
    if not legacy:
        validation, _ = designs(corpus.phase('validation'), pool, executor, ledger,
                                phase='validation', seed=seed, draws=1)
    calibration, _ = designs(corpus.phase(calibration_phase), pool, executor, ledger,
                             phase='calibration', seed=seed, draws=1)
    support = Counter()
    for e in train:
        nodes = {n.uid:n for n in e.context.nodes}
        for o in e.observations:
            if o.label in ('ok', 'agent'):
                n = nodes[o.node]
                support[n.role.key, n.version] += 1
    evidence, fixtures = differential_probes(seed=seed, support=support, fixtures=96)
    fixture_calls = sum(r['api_calls'] for r in fixtures)
    write(directory/'factorial_estimates.json', reports)
    write(directory/'transfer_levels.json', [asdict(p) for p in evidence])
    write(directory/'fixtures.json', fixtures)
    write(directory/'costs.json', ledger.report() | {'fixtures': fixture_calls,
                                                   'total_with_fixtures': ledger.spent+fixture_calls})
    stream_records(directory/'history.jsonl.gz', ledger.records)
    return train, validation, calibration, TransportGate(evidence), evidence, ledger.spent+fixture_calls


def ensemble(cls, train, validation, calibration, seed, directory):
    directory.mkdir()
    grouped = {}
    for e in train:
        grouped.setdefault(e.context.uid, []).append(e)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    fitted = []
    started = time.perf_counter()
    for i in range(3):
        sample = [e for k in rng.choice(keys, len(keys), replace=True) for e in grouped[k]]
        model = cls(memories(), [e.context for e in train], seed=seed+i)
        if cls is OutcomeModel:
            model.fit(sample, calibration, epochs=200)
        else:
            model.fit(sample, calibration, validation=validation)
        model.save(directory/f'outcome_model_{i}.pt')
        fitted.append(model)
        print(seed, cls.__name__, i, 'epochs', model.fit_report['iterations'], flush=True)
    write(directory/'training_reports.json', [m.fit_report for m in fitted])
    write(directory/'timing.json', {'training_and_calibration_seconds': time.perf_counter()-started})
    return tuple(CachedModel(m) for m in fitted)


def deterministic_screen(pool, score, *, capacity=80, bundles=(), keep=8, iterations=4, beam=3):
    """Canonical traversal, including ties and budget-limited candidate order."""
    from itertools import combinations
    ids = tuple(sorted(m.uid for m in pool))
    bundles = tuple(sorted({tuple(sorted(b)) for b in bundles}))
    values = {}

    def evaluate(bank):
        bank = tuple(sorted(set(bank)))
        if bank not in values and bank_cost(bank, pool) <= capacity:
            values[bank] = float(score(bank))

    for bank in ((), *((i,) for i in ids), *bundles):
        evaluate(bank)
    stopped = False
    for iteration in range(iterations):
        before = max(values.values())
        leaders = sorted(values, key=lambda b:(-values[b], bank_cost(b,pool), b))[:beam]
        for bank in leaders:
            selected = set(bank)
            for new in ids:
                evaluate(selected ^ {new})
            for old in bank:
                for new in ids:
                    if new not in selected:
                        evaluate((selected-{old}) | {new})
            for bundle in bundles:
                evaluate(selected | set(bundle))
                evaluate(selected-set(bundle))
                for removed in combinations(bank, min(2, len(bank))):
                    evaluate((selected-set(removed)) | set(bundle))
        if max(values.values()) <= before+1e-10:
            stopped = True
            break
    order = sorted(values, key=lambda b:(-values[b], bank_cost(b,pool), b))[:keep]
    return {'banks':order, 'scores':{'|'.join(b):values[b] for b in order},
            'evaluated':len(values), 'passes':iteration+1, 'local_stopping_met':stopped,
            'traversal':'sorted memory ids and sorted unique bundles'}


def select(method, models, gate, corpus, common, directory, *, depth=4, cap=150000,
           direct=False):
    directory.mkdir()
    pool = memories()
    ledger = CallLedger({'direct-screen':4096, 'halving':1024,
                         'pilot':2400, 'certification':4800}, cap, common)
    ledger.tag = method
    selector = make_selector(pool, () if direct else models, gate, corpus,
                             StatefulExecutor(), ledger, depth=depth)
    contexts = selector.contexts('screening-unlabeled', 32)
    bundles = {tuple(sorted((pool[i].uid, pool[j].uid)))
               for c in contexts['books/source'] for i,j in admitted_pairs(c,pool)}
    pilot_n = 2 if direct else 100
    counter = [0]
    if direct:
        tasks = selector.contexts('direct-screening', 4)
        # Reserve actual skeleton-based upper bounds for all remaining phases.
        reserve = 3 * sum(2*len(corpus.make(k, phase, i).nodes)
                          for k in selector.mixture for phase,n in
                          [('pilot',pilot_n), ('certification',200)] for i in range(n))
        # Candidate counts below eight change the number of tasks in each round.
        # Reserve the maximum over all feasible counts rather than assuming eight.
        import math
        halving_bounds = []
        for initial in range(2,9):
            remaining = initial
            rounds = math.ceil(math.log2(max(1,initial/2)))
            spent, upper_calls = 0, 0
            for rd in range(rounds):
                per_stratum = ((1024-spent)//(rounds-rd))//(remaining*8)
                upper_calls += remaining * sum(
                    2*len(corpus.make(k,f'halving-{rd}',i).nodes)
                    for k in selector.mixture for i in range(per_stratum))
                spent += per_stratum*remaining*8
                remaining = max(2,math.ceil(remaining/2))
            halving_bounds.append(upper_calls)
        reserve += max(halving_bounds)
        upper = sum(2*len(c.nodes) for group in tasks.values() for c in group)

        def objective(bank):
            if counter[0] >= 128 or ledger.spent+upper+reserve > cap:
                return -1 - .0001*bank_cost(bank,pool)
            counter[0] += 1
            observed = selector._executions('direct-screen', tasks, [bank])[bank]
            return worst_mixture([v.mean() for v in observed.values()], [.125]*8, .2) - .0001*bank_cost(bank,pool)
    else:
        objective = lambda b:selector.model_score(contexts,b)
    started = time.perf_counter()
    screen = deterministic_screen(pool, objective, bundles=bundles)
    if direct:
        # Never certify an unevaluated challenger when the call cap is restrictive.
        screen['banks'] = [b for b in screen['banks'] if screen['scores']['|'.join(b)] > -1]
        screen['real_candidates_executed'] = counter[0]
        screen['worst_case_remaining_reservation'] = reserve
    write(directory/'screening.json',screen)
    if len(screen['banks']) < 2:
        halving = {'finalists':[], 'rounds':[], 'reason':'insufficient screening allowance'}
        decision = {'bank':(), 'switched':False, 'certificates':[],
                    'reason':'no evaluated challenger within hard call cap'}
    else:
        halving = selector.halve(screen['banks'], gamma=1024, finalists=2)
        decision = selector.switch((),halving['finalists'], pilot_n=pilot_n,
                                   certification_n=200, unlabeled_n=2 if direct else 1000,
                                   freeze=lambda x:write(directory/'certification_freeze.json',x))
    write(directory/'halving.json', halving)
    write(directory/'switch_decision.json',decision)
    write(directory/'costs.json',ledger.report())
    write(directory/'timing.json',{'selection_and_certification_seconds':time.perf_counter()-started})
    stream_records(directory/'selection.jsonl.gz',ledger.records)
    return tuple(decision['bank']), ledger.report()


def prepare(seed, corpus, output):
    directory = output/f'seed_{seed}'
    directory.mkdir()
    legacy = history(corpus, seed, directory/'legacy_history', legacy=True)
    expanded = history(corpus, seed, directory/'expanded_history')
    old = ensemble(OutcomeModel,*legacy[:3],seed,directory/'models_legacy')
    coverage = ensemble(ValidatedOutcomeModel,*expanded[:3],seed,directory/'models_coverage')
    improved = ensemble(StateAwareOutcomeModel,*expanded[:3],seed,directory/'models_state')
    no_pairs = ensemble(StateAwareNoPairs,*expanded[:3],seed,directory/'models_no_pairs')
    gate, evidence, common = expanded[3:]
    history_costs = json.loads((directory/'expanded_history'/'costs.json').read_text())
    # Direct uses randomized node labels for gate support, but does not train
    # or calibrate a predictor. Do not bill it unused validation/calibration.
    direct_common = (history_costs['api_calls']['factorial']
                     + history_costs['api_calls']['replay_audit'] + history_costs['fixtures'])
    banks = {'NoMem':()}
    policies = {}
    costs = {'NoMem':{'selection_api_calls_including_common':0}}
    specs = [
        ('CREDIT-Legacy-K4', old, legacy[3], legacy[-1], 4, False, 150000),
        ('CREDIT-Coverage-K4', coverage, gate, common, 4, False, 150000),
        ('CREDIT-State-K4', improved, gate, common, 4, False, 150000),
        ('CREDIT-State-NoPairs-K4', no_pairs, gate, common, 4, False, 150000),
        ('Direct-K4', (), gate, direct_common, 4, True, 150000),
        ('CREDIT-State-K6', improved, gate, common, 6, False, 150000),
        ('Direct-K6', (), gate, direct_common, 6, True, 150000),
    ]
    for method, models, active_gate, shared, depth, direct, cap in specs:
        print(seed, method, 'selection', flush=True)
        banks[method], costs[method] = select(method, models, active_gate, corpus, shared,
                                             directory/method, depth=depth, cap=cap, direct=direct)
        policies[method] = (active_gate,depth)
    for depth in (4,6):
        method = f'Direct-Matched-K{depth}'
        cap = costs[f'CREDIT-State-K{depth}']['selection_api_calls_including_common']
        banks[method], costs[method] = select(method, (), gate, corpus, direct_common, directory/method,
                                             depth=depth, cap=cap, direct=True)
        policies[method] = (gate,depth)
        print(seed,method,'cap',cap,'actual',costs[method]['selection_api_calls_including_common'],flush=True)
    write(directory/'frozen_banks.json',{k:list(v) for k,v in banks.items()})
    write(directory/'method_costs.json', costs)
    return {'seed':seed,'corpus':corpus,'banks':banks,'policies':policies,'models':improved,
            'evidence':evidence,'directory':directory}


def run(dataset, output, seeds=SEEDS):
    if output.exists():
        raise FileExistsError('use a fresh output directory')
    if any(s in (701,709,719,727,739,907) for s in seeds):
        raise ValueError('confirmatory seeds must be fresh, distinct from historical/development seeds')
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    (output/'source').mkdir()
    sources = [Path(__file__), Path('credit/v3_optimized_model.py'), Path('credit/v3_optimized_data.py'),
               Path('credit/v3_benchmark.py'), Path('credit/v3_data.py'), Path('credit/paired_statistics.py'),
               *Path('credit/v3').glob('*.py'), Path('credit/stateful.py')]
    write(output/'frozen_protocol.json', {
        'seeds':list(seeds), 'dataset':str(dataset.resolve()),
        'dataset_manifest_sha256':digest(dataset/'manifest.json'),
        'source_sha256':{str(p):digest(p) for p in sources},
        'paper_sha256':digest('CREDIT_WWW2027_v3.tex'), 'agents_sha256':digest('AGENTS.md'),
        'scope':'unchanged synthetic actor/API/verifier, not real providers or an LLM',
        'training':'source-only randomized 128 base contexts, 1 draw; state coverage including ready-submit',
        'slots':'six for full workflows, three for submit-only; scope positivity preserved',
        'validation':'64 separate source contexts, 1 draw, no overlap with calibration/test',
        'calibration':'64 separate source contexts, 1 draw; single isotonic map',
        'optimizer':'Adam lr=.01, 1200 epoch cap, independent masked NLL patience; compressed exact likelihood',
        'state_features':'observed pre-call state in fitting; public successful-prefix postconditions in prospective prediction',
        'architecture':'unchanged masked c/r likelihood and isotonic product; state model adds features and 32-dim text/hidden layers',
        'state_scope':'successful-prefix state is determinate for this public finite adapter; not claimed for arbitrary Web services',
        'certification':'normal/asymptotic, alpha1+alpha2=.10 per cycle; 200 paired tasks per stratum/bank',
        'direct':'zero learned models; pilot_n=2, lambda=0; screening cap128 real banks; same gamma1024',
        'call_cap':150000, 'capacity':80, 'retrieval_depths':[4,6],
        'matched_rule':'Direct hard cap equals completed CREDIT actual expense at same K; unused calls reported',
        'matched_limit':'budget ceiling matched, actual call totals can differ; no unused validation/calibration billed to Direct',
        'history_cost_rule':'CREDIT pays train/validation/calibration/replay/probes; Direct pays train/replay/probes for gate support',
        'primary_comparison':'CREDIT-State-K4 vs CREDIT-Legacy-K4 on paired fresh tasks',
        'secondary_comparisons':'same-K Direct, matched call caps, coverage/training bundle and no-pairs',
        'coverage_ablation_limit':'coverage variant bundles data coverage and validation training; state variant also enlarges heads/text encoder',
        'topology_limit':'six-step test graph unseen; one/five-step families overlap fitting shapes',
        'test_per_stratum':256, 'test_inspection':'every seed/model/bank frozen before the first held-out execution'})
    for p in sources:
        (output/'source'/p.name).write_bytes(p.read_bytes())
    prepared = [prepare(seed, Corpus(dataset,seed), output) for seed in seeds]
    frozen = {str(p.relative_to(output)):digest(p) for p in output.rglob('*')
              if p.is_file() and p.name in ('frozen_banks.json','switch_decision.json','certification_freeze.json')}
    write(output/'evaluation_freeze.json',frozen)
    rows = score(prepared,output)
    table, comparisons = summarize(rows,'CREDIT-State-K4',repeats=4000,random_seed=4099)
    write(output/'statistics.json',{'main':table,'paired':comparisons})
    primary, paired = summarize([r for r in rows if r['method'] in
                                ('CREDIT-State-K4','CREDIT-Legacy-K4')],
                                'CREDIT-State-K4',repeats=4000,random_seed=4099)
    write(output/'primary_statistics.json', {'main':primary,'paired':paired,
                                            'predeclared_primary_comparisons':1})
    write(output/'summary.json',{'test_episodes':len(rows),'test_api_calls':sum(r['api_calls'] for r in rows),
                                'methods':sorted({r['method'] for r in rows}), 'seeds':list(seeds)})
    write(output/'artifact_manifest.json',{str(p.relative_to(output)):digest(p)
                                         for p in output.rglob('*') if p.is_file()})
    print('Completed prospective optimization benchmark',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    a = p.parse_args()
    run(a.dataset,a.output,a.seeds)
