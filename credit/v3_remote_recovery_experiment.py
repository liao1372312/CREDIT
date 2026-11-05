"""Two-stage paired admission for contract and recovery memories."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

from scipy.stats import binomtest

from .llm_policy import install_policy
from .v3.experiment import digest, write
from .v3.local import factory, retriever
from .v3_recovery import RecoveryExecutor
from .v3_remote_expansion_diagnostic import DEV_SEEDS, CONFIRM_SEEDS, STRATA
from .v3_remote_llm_experiment import policy_guard

CYCLES = 3
TEST_N = 128
DESIGNS = (24, 48, 96)
BANKS = ((), ('repair_rule',), ('new_contract',), ('new_contract', 'repair_rule'))
EXPECTED_ORACLE = {s: tuple(x for x in ('new_contract', 'repair_rule')
                              if x == 'repair_rule' or s in ('books/target', 'inventory/target'))
                   for s in STRATA}


def paired_counts(left, right):
    gain = sum(a and not b for a, b in zip(left, right))
    loss = sum(b and not a for a, b in zip(left, right))
    p = binomtest(gain, gain + loss, 0.5, alternative='greater').pvalue if gain + loss else 1.0
    return {'gain': gain, 'loss': loss, 'ties': len(left) - gain - loss,
            'p_one_sided': p}


def adaptation(seed, cycle, n, executor):
    make, retrieve = factory(seed), retriever(4)
    outcomes = defaultdict(lambda: defaultdict(list))
    calls = 0
    for stratum in STRATA:
        for i in range(n):
            task = make(stratum, f'recovery-{cycle}/adaptation', i)
            for bank in BANKS:
                episode = executor.run(task, retrieve(task, bank))
                outcomes[stratum][bank].append(bool(episode.verified))
                calls += episode.calls
    empty = [value for stratum in STRATA for value in outcomes[stratum][()]]
    repair = [value for stratum in STRATA for value in outcomes[stratum][('repair_rule',)]]
    repair_test = paired_counts(repair, empty)
    repair_test['alpha'] = 0.025
    repair_test['admit'] = bool(repair_test['gain'] > repair_test['loss']
                                 and repair_test['p_one_sided'] < repair_test['alpha'])
    base = ('repair_rule',) if repair_test['admit'] else ()
    decisions = {}
    for stratum in STRATA:
        treatment = ('new_contract', 'repair_rule') if base else ('new_contract',)
        test = paired_counts(outcomes[stratum][treatment], outcomes[stratum][base])
        test['alpha'] = 0.025 / len(STRATA)
        test['admit'] = bool(test['gain'] > test['loss'] and test['p_one_sided'] < test['alpha'])
        bank = tuple(x for x in ('new_contract', 'repair_rule')
                     if (x == 'repair_rule' and repair_test['admit'])
                     or (x == 'new_contract' and test['admit']))
        decisions[stratum] = {'contract_test': test, 'bank': bank}
    return {'repair_test': repair_test, 'strata': decisions}, calls


def score(seed, cycle, decision, executor):
    make, retrieve = factory(seed), retriever(4)
    rows = []
    for stratum in STRATA:
        focused = ('new_contract',) if stratum in ('books/target', 'inventory/target') else ()
        methods = {'NoMem': (), 'CREDIT-Focused-v1': focused,
                   'CREDIT-Recovery': tuple(decision['strata'][stratum]['bank']),
                   'Recovery-Oracle': EXPECTED_ORACLE[stratum]}
        for i in range(TEST_N):
            task = make(stratum, f'recovery-{cycle}/heldout-test', i)
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
    for method in ('NoMem', 'CREDIT-Focused-v1', 'CREDIT-Recovery', 'Recovery-Oracle'):
        selected = [row for row in rows if row['method'] == method]
        success = sum(row['success'] for row in selected)
        result[method] = {'tasks': len(selected), 'successes': success,
                          'vsr': success / len(selected),
                          'test_api_calls': sum(row['api_calls'] for row in selected),
                          'extra_adaptation_calls': adaptation_calls if method == 'CREDIT-Recovery' else 0}
    lookup = defaultdict(dict)
    for row in rows:
        lookup[(row['seed'], row['cycle'], row['stratum'], row['task'])][row['method']] = row['success']
    result['recovery_vs_oracle_different_tasks'] = sum(
        value['CREDIT-Recovery'] != value['Recovery-Oracle'] for value in lookup.values())
    result['recovery_vs_focused_different_tasks'] = sum(
        value['CREDIT-Recovery'] != value['CREDIT-Focused-v1'] for value in lookup.values())
    return result


def run(policy, output, split, design=None):
    if output.exists():
        raise FileExistsError('fresh recovery experiment directory required')
    actor, _ = policy_guard(policy)
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
        'scope': 'paired role-transported recovery plus per-stratum contract admission',
        'split': split, 'seeds': seeds, 'unused_confirmation_seeds': CONFIRM_SEEDS,
        'cycles': CYCLES, 'test_n_per_stratum': TEST_N,
        'adaptation_contexts_per_stratum': ns, 'candidate_banks': BANKS,
        'actor_scores_sha256': actor['decision_scores_sha256'],
        'design_sha256': digest(design) if design else None,
        'source_sha256': {name: digest(name) for name in (
            'credit/v3_remote_recovery_experiment.py', 'credit/v3_recovery.py',
            'credit/llm_policy.py', 'credit/stateful.py', 'credit/v3/local.py')},
        'repair_rule': 'pooled one-sided exact paired test at alpha .025',
        'contract_rule': 'eight one-sided exact paired tests at .025/8',
        'test_policy': 'score after admission; no test outcome enters admission'})
    install_policy(policy / 'decision_scores.json')
    executor = RecoveryExecutor()
    summaries = {}
    for n in ns:
        schedules, rows, total_calls = [], [], 0
        for seed, cycle in product(seeds, range(CYCLES)):
            decision, calls = adaptation(seed, cycle, n, executor)
            schedules.append({'seed': seed, 'cycle': cycle, 'decision': decision,
                              'adaptation_api_calls': calls})
            total_calls += calls
            rows.extend(score(seed, cycle, decision, executor))
            print(json.dumps({'split': split, 'contexts': n, 'seed': seed, 'cycle': cycle,
                              'repair_admitted': decision['repair_test']['admit'],
                              'banks': {s: d['bank'] for s, d in decision['strata'].items()}}), flush=True)
        write(output / f'admission_{n}.json', schedules)
        write(output / f'test_{n}.json', rows)
        summaries[str(n)] = summarize(rows, total_calls)
    write(output / 'summary.json', summaries)
    write(output / 'artifact_manifest.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})


def choose(development, output):
    if output.exists():
        raise FileExistsError('fresh recovery design path required')
    manifest = json.loads((development / 'artifact_manifest.json').read_text())
    for name, expected in manifest.items():
        if digest(development / name) != expected:
            raise ValueError('development artifact changed: ' + name)
    summary = json.loads((development / 'summary.json').read_text())
    candidates = []
    for n in DESIGNS:
        schedules = json.loads((development / f'admission_{n}.json').read_text())
        correct = all(row['decision']['repair_test']['admit'] and
                      all(tuple(row['decision']['strata'][s]['bank']) == EXPECTED_ORACLE[s]
                          for s in STRATA) for row in schedules)
        exact = summary[str(n)]['recovery_vs_oracle_different_tasks'] == 0
        if correct and exact:
            candidates.append(n)
    chosen = min(candidates) if candidates else max(DESIGNS)
    write(output, {'chosen_contexts': chosen,
                   'selection_rule': 'minimum tested n with oracle banks in every development block and zero heldout oracle disagreements; else 96',
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
