"""Monotone recovery guard plus paired contract admission."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

from scipy.stats import binomtest

from .llm_policy import install_policy
from .stateful import NEIGHBORS
from .v3.experiment import digest, write
from .v3.local import factory, retriever
from .v3_recovery import RecoveryExecutor
from .v3_remote_expansion_diagnostic import DEV_SEEDS, CONFIRM_SEEDS, STRATA
from .v3_remote_llm_experiment import policy_guard

CYCLES = 3
TEST_N = 128
DESIGNS = (12, 24, 48)
BASE = ('repair_rule',)
TREATMENT = ('new_contract', 'repair_rule')
EXPECTED_ORACLE = {s: TREATMENT if s in ('books/target', 'inventory/target') else BASE
                   for s in STRATA}


def action(scores, key):
    row = next(row for row in scores if tuple(row['key']) == key)
    return 'ABCD'[max(range(4), key=lambda i: row['probabilities'][i])]


def monotone_guard(policy):
    """Validate that repair does not alter the frozen actor's original path."""
    scores = json.loads((policy / 'decision_scores.json').read_text())
    checks = []
    for role in ('transform', 'authorize'):
        bit = 1 << NEIGHBORS[role].index('repair_rule')
        for missing, transient in product((0, 1), repeat=2):
            before = action(scores, (role, missing, transient, 0))
            after = action(scores, (role, missing, transient, bit))
            checks.append({'role': role, 'missing': missing, 'transient': transient,
                           'without': before, 'with_repair_rule': after})
    if any(row['without'] != row['with_repair_rule'] for row in checks):
        raise ValueError('repair memory changes the original actor path; monotone guard invalid')
    return {'passed': True, 'checks': checks,
            'reason': 'actor path unchanged; RecoveryExecutor starts only after original agent failure and retains failure if compensation fails'}


def adaptation(seed, cycle, n, executor):
    make, retrieve = factory(seed), retriever(4)
    decisions, calls = {}, 0
    for stratum in STRATA:
        control, treatment = [], []
        for i in range(n):
            task = make(stratum, f'recovery-v2-{cycle}/adaptation', i)
            a = executor.run(task, retrieve(task, BASE))
            b = executor.run(task, retrieve(task, TREATMENT))
            calls += a.calls + b.calls
            control.append(bool(a.verified)); treatment.append(bool(b.verified))
        gain = sum(a and not b for a, b in zip(treatment, control))
        loss = sum(b and not a for a, b in zip(treatment, control))
        p = binomtest(gain, gain + loss, 0.5, alternative='greater').pvalue if gain + loss else 1.0
        alpha = 0.025 / len(STRATA)
        admit = bool(gain > loss and p < alpha)
        decisions[stratum] = {'gain': gain, 'loss': loss, 'ties': n - gain - loss,
                              'p_one_sided': p, 'alpha': alpha, 'admit_new_contract': admit,
                              'bank': TREATMENT if admit else BASE}
    return decisions, calls


def score(seed, cycle, decisions, executor):
    make, retrieve = factory(seed), retriever(4)
    rows = []
    for stratum in STRATA:
        focused = ('new_contract',) if stratum in ('books/target', 'inventory/target') else ()
        methods = {'NoMem': (), 'CREDIT-Focused-v1': focused,
                   'CREDIT-Recovery-v2': tuple(decisions[stratum]['bank']),
                   'Recovery-Oracle': EXPECTED_ORACLE[stratum]}
        for i in range(TEST_N):
            task = make(stratum, f'recovery-v2-{cycle}/heldout-test', i)
            cache = {}
            for method, bank in methods.items():
                exposure = retrieve(task, bank)
                if exposure not in cache:
                    cache[exposure] = executor.run(task, exposure)
                episode = cache[exposure]
                rows.append({'seed': seed, 'cycle': cycle, 'stratum': stratum,
                             'task': task.uid, 'method': method,
                             'success': bool(episode.verified), 'api_calls': episode.calls,
                             'bank': list(bank), 'exposure': list(exposure),
                             'state_hash': episode.state_hash})
    return rows


def summarize(rows, adaptation_calls):
    result = {}
    methods = ('NoMem', 'CREDIT-Focused-v1', 'CREDIT-Recovery-v2', 'Recovery-Oracle')
    for method in methods:
        selected = [row for row in rows if row['method'] == method]
        successes = sum(row['success'] for row in selected)
        result[method] = {'tasks': len(selected), 'successes': successes,
                          'vsr': successes / len(selected),
                          'test_api_calls': sum(row['api_calls'] for row in selected),
                          'extra_adaptation_calls': adaptation_calls if method == 'CREDIT-Recovery-v2' else 0}
    lookup = defaultdict(dict)
    for row in rows:
        lookup[(row['seed'], row['cycle'], row['stratum'], row['task'])][row['method']] = row['success']
    result['recovery_vs_oracle_different_tasks'] = sum(
        row['CREDIT-Recovery-v2'] != row['Recovery-Oracle'] for row in lookup.values())
    result['recovery_vs_focused_different_tasks'] = sum(
        row['CREDIT-Recovery-v2'] != row['CREDIT-Focused-v1'] for row in lookup.values())
    return result


def run(policy, output, split, design=None):
    if output.exists():
        raise FileExistsError('fresh v2 recovery experiment directory required')
    actor, _ = policy_guard(policy)
    guard = monotone_guard(policy)
    if split == 'development':
        seeds, ns = DEV_SEEDS, DESIGNS
    else:
        frozen = json.loads(design.read_text())
        if frozen['chosen_contexts'] not in DESIGNS:
            raise ValueError('invalid frozen design')
        if digest(frozen['development_summary_path']) != frozen['development_summary_sha256']:
            raise ValueError('development summary changed')
        seeds, ns = CONFIRM_SEEDS, (frozen['chosen_contexts'],)
    output.mkdir(parents=True)
    write(output / 'frozen_protocol.json', {
        'scope': 'monotone recovery guard plus paired target contract admission',
        'split': split, 'seeds': seeds, 'unused_confirmation_seeds': CONFIRM_SEEDS,
        'cycles': CYCLES, 'test_n_per_stratum': TEST_N,
        'adaptation_contexts_per_stratum': ns,
        'actor_scores_sha256': actor['decision_scores_sha256'], 'monotone_guard': guard,
        'design_sha256': digest(design) if design else None,
        'source_sha256': {name: digest(name) for name in (
            'credit/v3_remote_recovery_experiment_v2.py', 'credit/v3_recovery.py',
            'credit/llm_policy.py', 'credit/stateful.py', 'credit/v3/local.py')},
        'contract_rule': 'eight one-sided exact paired tests at .025/8',
        'test_policy': 'score after admission; no test outcome enters admission'})
    install_policy(policy / 'decision_scores.json')
    executor = RecoveryExecutor()
    summaries = {}
    for n in ns:
        schedules, rows, total_calls = [], [], 0
        for seed, cycle in product(seeds, range(CYCLES)):
            decisions, calls = adaptation(seed, cycle, n, executor)
            schedules.append({'seed': seed, 'cycle': cycle, 'decisions': decisions,
                              'adaptation_api_calls': calls})
            total_calls += calls
            rows.extend(score(seed, cycle, decisions, executor))
            print(json.dumps({'split': split, 'contexts': n, 'seed': seed, 'cycle': cycle,
                              'banks': {s: d['bank'] for s, d in decisions.items()}}), flush=True)
        write(output / f'admission_{n}.json', schedules)
        write(output / f'test_{n}.json', rows)
        summaries[str(n)] = summarize(rows, total_calls)
    write(output / 'summary.json', summaries)
    write(output / 'artifact_manifest.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})


def choose(development, output):
    if output.exists():
        raise FileExistsError('fresh v2 recovery design path required')
    manifest = json.loads((development / 'artifact_manifest.json').read_text())
    for name, expected in manifest.items():
        if digest(development / name) != expected:
            raise ValueError('development artifact changed: ' + name)
    summary = json.loads((development / 'summary.json').read_text())
    candidates = []
    for n in DESIGNS:
        schedules = json.loads((development / f'admission_{n}.json').read_text())
        correct = all(all(tuple(row['decisions'][s]['bank']) == EXPECTED_ORACLE[s] for s in STRATA)
                      for row in schedules)
        exact = summary[str(n)]['recovery_vs_oracle_different_tasks'] == 0
        if correct and exact:
            candidates.append(n)
    chosen = min(candidates) if candidates else max(DESIGNS)
    write(output, {'chosen_contexts': chosen,
                   'selection_rule': 'minimum tested n with oracle contract banks in every development block and zero heldout oracle disagreements; else 48',
                   'development_satisfied': bool(candidates),
                   'development_summary_path': str((development / 'summary.json').resolve()),
                   'development_summary_sha256': digest(development / 'summary.json'),
                   'development_manifest_sha256': digest(development / 'artifact_manifest.json'),
                   'confirmation_seeds': CONFIRM_SEEDS,
                   'generator_sha256': digest(__file__)})
    print(json.dumps({'chosen_contexts': chosen, 'development_satisfied': bool(candidates)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('development', 'confirmation', 'choose'), required=True)
    p.add_argument('--design', type=Path)
    p.add_argument('--development', type=Path)
    a = p.parse_args()
    if a.split == 'choose':
        choose(a.development, a.output)
    else:
        run(a.policy, a.output, a.split, a.design)
