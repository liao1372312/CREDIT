"""Mixed-fidelity, exposure-distinct selection for the finite CREDIT adapter.

No hidden outcome kernels or reference utilities are used. Historical modules
stay unchanged. Enumeration is limited to small pools; production-size search
needs a separate implementation. Execution reuse requires isolated, seeded,
version-pinned deterministic replay and is disabled outside that scope.
"""
import argparse
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

from .paired_statistics import summarize
from .v3.experiment import digest, write
from .v3.inference import worst_mixture
from .v3.local import StatefulExecutor, context, memories, retriever, TEMPLATES
from .v3.selection import bank_cost
from .v3_benchmark import CallLedger, TrackedSelector, make_selector, score, stream_records
from .v3_compact_optimization import LimitedCorpus
from .v3_data import Corpus
from .v3_optimization import history, ensemble, select as legacy_select
from .v3_optimized_model import StateAwareOutcomeModel, StateAwareNoPairs

SEEDS = (1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049)
SPECS = (
    ('CREDIT-Previous-K4', 'previous', 4),
    ('CREDIT-Unique-K4', 'model', 4),
    ('CREDIT-Mixed-K4', 'mixed', 4),
    ('CREDIT-Mixed-NoPairs-K4', 'mixed-no-pairs', 4),
    ('Direct-Unique-K4', 'direct', 4),
    ('CREDIT-Mixed-K6', 'mixed', 6),
    ('Direct-Unique-K6', 'direct', 6),
)


class ReuseSelector(TrackedSelector):
    """Reuse the exact same task/exposure outcome, never a nearby task's label."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_cache = {}
        self.reuse_records = []

    def _executions(self, phase, contexts, banks):
        result = {}
        for bank in banks:
            self.ledger.bank = bank
            result[bank] = {}
            for k, tasks in contexts.items():
                values = []
                for c in tasks:
                    a = self.exposures(c, bank)
                    key = c, tuple(a)
                    if key in self.execution_cache:
                        index, e = self.execution_cache[key]
                        if not e.replay_valid or e.context != c or e.exposure != tuple(a):
                            raise ValueError('unsafe or mismatched execution reuse')
                        self.reuse_records.append({'phase': phase, 'bank': list(bank),
                            'context': c.uid, 'exposure': list(a), 'physical_record': index})
                    else:
                        e = self.ledger.execute(phase, self.executor, c, a)
                        if not e.replay_valid:
                            raise ValueError('unsafe selection execution')
                        self.execution_cache[key] = len(self.ledger.records)-1, e
                    values.append(e.outcome)
                result[bank][k] = np.asarray(values, dtype=float)
        self.ledger.bank = None
        return result


def public_profiles(mixture):
    """All publicly declared deployment templates, without held-out tasks."""
    return tuple(replace(context(*k.split('/'), 'public-retrieval', i, 0, template=t),
                         uid=f'public-profile:{k}:{t}')
                 for k in sorted(mixture) for i, t in enumerate(
                     ('future', 'future_auth_repeat', 'future_ready_submit')))


def distinct_banks(selector, profiles):
    """Pick the cheapest bank for each exact retrieval signature on profiles."""
    pool = selector.memories
    if len(pool) > 16:
        raise ValueError('enumeration is restricted to pools of at most sixteen')
    ids = sorted(m.uid for m in pool)
    signatures = {}
    feasible = 0
    for n in range(len(ids)+1):
        for b in combinations(ids, n):
            cost = bank_cost(b, pool)
            if cost > selector.capacity:
                continue
            feasible += 1
            key = tuple(selector.exposures(c, b) for c in profiles)
            if key not in signatures or (cost, b) < (bank_cost(signatures[key], pool), signatures[key]):
                signatures[key] = b
    return sorted(signatures.values(), key=lambda b: (bank_cost(b, pool), b)), feasible


def mixed_means(selector, bank, unlabeled, direct_tasks, mode):
    keys = list(selector.mixture)
    real_keys = keys if mode == 'direct' else [k for k in keys if k not in selector.transported_strata]
    real = {} if mode == 'model' else selector._executions(
        'direct-screen', {k: direct_tasks[k] for k in real_keys}, [bank])[bank]
    means = []
    for k in keys:
        if k in real:
            means.append(float(real[k].mean()))
        elif k in selector.transported_strata:
            means.append(float(selector.predictions(unlabeled[k], bank).mean()))
        else:
            means.append(0.0)
    return means


def select(name, models, gate, corpus, common, directory, *, mode, depth):
    directory.mkdir()
    if mode == 'previous':
        raise ValueError('use the preserved previous selector for that control')
    direct = mode == 'direct'
    ledger = CallLedger({'direct-screen':32768, 'halving':1024,
                         'pilot':2400, 'certification':4800}, 150000, common)
    ledger.tag = name
    base = make_selector(memories(), () if direct else models, gate, corpus,
                         StatefulExecutor(), ledger, depth=depth)
    selector = ReuseSelector(base.memories, base.models, gate, base.retrieve, corpus.make,
        base.executor, ledger, mixture=base.mixture, capacity=80, radius=.2,
        cost_penalty=.0001, transported_strata=base.transported_strata)
    unlabeled = selector.contexts('screening-unlabeled', 32)
    direct_tasks = selector.contexts('direct-screening', 16) if mode != 'model' else {}
    candidates, feasible = distinct_banks(selector, public_profiles(selector.mixture))
    values, rows = {}, []
    started = time.perf_counter()
    for b in candidates:
        means = mixed_means(selector, b, unlabeled, direct_tasks, mode)
        value = worst_mixture(means, list(selector.mixture.values()), .2) - .0001*bank_cost(b, memories())
        values[b] = value
        rows.append({'bank': b, 'means': dict(zip(selector.mixture, means)), 'score': value})
    leaders = sorted(values, key=lambda b: (-values[b], bank_cost(b, memories()), b))[:8]
    screen = {'banks': leaders, 'scores': {'|'.join(b):values[b] for b in leaders},
        'all_candidate_scores': rows, 'feasible_banks':feasible, 'distinct_exposures':len(candidates),
        'transported_strata':sorted(selector.transported_strata), 'mode':mode,
        'direct_tasks_per_stratum':0 if mode == 'model' else 16,
        'dedup_scope':'all three public templates in every stratum; not arbitrary future retrieval',
        'enumeration_scope':'finite eight-memory pool, not scalable local search'}
    write(directory/'screening.json', screen)
    halving = selector.halve(leaders, gamma=1024, finalists=2)
    decision = selector.switch((), halving['finalists'], pilot_n=2 if direct else 16,
        certification_n=200, unlabeled_n=2 if direct else 1000,
        freeze=lambda d:write(directory/'certification_freeze.json',d))
    write(directory/'halving.json',halving)
    write(directory/'switch_decision.json',decision)
    write(directory/'costs.json',ledger.report())
    write(directory/'execution_reuse.json',selector.reuse_records)
    write(directory/'timing.json',{'selection_seconds':time.perf_counter()-started})
    stream_records(directory/'selection.jsonl.gz',ledger.records)
    return tuple(decision['bank']), ledger.report()


def prepare(seed, corpus, output):
    sd = output/f'seed_{seed}'
    sd.mkdir()
    h = history(LimitedCorpus(corpus), seed, sd/'compact_history')
    models = ensemble(StateAwareOutcomeModel,*h[:3],seed,sd/'models_state')
    no_pairs = ensemble(StateAwareNoPairs,*h[:3],seed,sd/'models_no_pairs')
    gate, evidence, common = h[3:]
    hc = json.loads((sd/'compact_history/costs.json').read_text())
    direct_common = hc['api_calls']['factorial']+hc['api_calls']['replay_audit']+hc['fixtures']
    banks, costs, policies = {'NoMem':()}, {'NoMem':{'selection_api_calls_including_common':0}}, {}
    for name, mode, depth in SPECS:
        active = no_pairs if mode == 'mixed-no-pairs' else models
        print(seed,name,'selection',flush=True)
        if mode == 'previous':
            b, co = legacy_select(name,active,gate,corpus,common,sd/name,depth=depth)
        else:
            b, co = select(name,active,gate,corpus,direct_common if mode == 'direct' else common,
                           sd/name,mode='mixed' if mode == 'mixed-no-pairs' else mode,depth=depth)
        banks[name],costs[name],policies[name] = b,co,(gate,depth)
    write(sd/'frozen_banks.json',{k:list(v) for k,v in banks.items()})
    write(sd/'method_costs.json',costs)
    return {'seed':seed,'corpus':corpus,'banks':banks,'policies':policies,'evidence':evidence,'directory':sd}


def run(dataset, output, seeds=SEEDS):
    if output.exists():
        raise FileExistsError('use a fresh output directory')
    if any(s not in SEEDS for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('only the preregistered fresh distinct seeds are allowed')
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    (output/'source').mkdir()
    sources = [Path(__file__), Path('credit/v3_optimization.py'),Path('credit/v3_compact_optimization.py'),
        Path('credit/v3_optimized_model.py'),Path('credit/v3_optimized_data.py'),
        Path('credit/v3_benchmark.py'),Path('credit/v3_data.py'),Path('credit/paired_statistics.py'),
        *Path('credit/v3').glob('*.py'),Path('credit/stateful.py')]
    write(output/'frozen_protocol.json',{
        'seeds':list(seeds),'dataset':str(dataset.resolve()),'dataset_manifest_sha256':digest(dataset/'manifest.json'),
        'source_sha256':{str(p):digest(p) for p in sources},
        'paper_sha256':digest('CREDIT_WWW2027_v3.tex'),'agents_sha256':digest('AGENTS.md'),
        'specifications':SPECS,'scope':'unchanged synthetic actor/API/verifier; no real-service claims',
        'training':'compact source-only 64/32/32 contexts; same state ensemble and independent validation',
        'screening':'all feasible small-pool banks, exact public-template exposure equivalence, keep8',
        'mixed_fidelity':'model only on V strata; real 16-task means on untransported strata',
        'reuse':'exact Context including seed/version/state plus ordered exposure, isolated deterministic adapter only',
        'budget':'150000 API calls per method; all used history, probes and selection charged',
        'halving':'gamma1024, two finalists; lambda0 real execution ranking',
        'certification':'200 paired tasks/stratum/bank; supported-only PPI; pilot16 CREDIT,2 Direct; alpha sum .10 asymptotic',
        'direct':'same enumerated exposure-distinct candidates, real16 per all strata, no predictor or validation/calibration billing',
        'primary_comparison':'CREDIT-Mixed-K4 vs CREDIT-Unique-K4 (isolates direct unsupported screening)',
        'secondary':'previous selector, pure Direct, no pairs, K6; Holm for all secondary comparisons',
        'limits':'unique vs previous bundles exhaustive traversal, dedup, pilot16 and exact execution reuse; not individually isolated',
        'test':'all eight seeds/banks frozen before first heldout label;256 tasks/stratum; same known distribution, fresh tasks',
        'development':'known seed811 selection diagnostics only; no new heldout results used to choose parameters'})
    for p in sources:
        (output/'source'/p.name).write_bytes(p.read_bytes())
    prepared = [prepare(seed,Corpus(dataset,seed),output) for seed in seeds]
    write(output/'evaluation_freeze.json',{str(p.relative_to(output)):digest(p)
        for p in output.rglob('*') if p.is_file() and p.name in
        ('frozen_banks.json','method_costs.json','switch_decision.json','certification_freeze.json')})
    rows = score(prepared,output)
    table, paired = summarize(rows,'CREDIT-Mixed-K4',repeats=4000,random_seed=5003)
    write(output/'statistics.json',{'main':table,'paired':paired})
    t,p = summarize([r for r in rows if r['method'] in ('CREDIT-Mixed-K4','CREDIT-Unique-K4')],
                    'CREDIT-Mixed-K4',repeats=4000,random_seed=5003)
    write(output/'primary_statistics.json',{'main':t,'paired':p,'predeclared_primary_comparisons':1})
    write(output/'summary.json',{'test_episodes':len(rows),'test_api_calls':sum(r['api_calls'] for r in rows),
        'methods':sorted({r['method'] for r in rows}),'seeds':list(seeds),'paper_experiments_complete':False})
    write(output/'artifact_manifest.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    print('Completed exposure-distinct mixed-fidelity experiment',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    run(a.dataset,a.output)
