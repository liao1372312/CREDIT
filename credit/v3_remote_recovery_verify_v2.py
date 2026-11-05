"""Re-execute and statistically audit the v2 recovery experiment."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np

from .llm_policy import install_policy
from .v3.experiment import digest, write
from .v3_recovery import RecoveryExecutor
from .v3_remote_llm_experiment import policy_guard
from .v3_remote_recovery_experiment_v2 import adaptation, monotone_guard, score, summarize


def paired_statistics(rows, left, right):
    grouped = defaultdict(lambda: defaultdict(list))
    task_rows = defaultdict(dict)
    for row in rows:
        grouped[row['seed']][row['method']].append(int(row['success']))
        task_rows[(row['seed'], row['cycle'], row['stratum'], row['task'])][row['method']] = int(row['success'])
    seeds = sorted(grouped)
    differences = np.asarray([np.mean(grouped[seed][left]) - np.mean(grouped[seed][right])
                              for seed in seeds], dtype=float)
    signs = np.asarray(list(product((-1, 1), repeat=len(seeds))), dtype=float)
    observed = abs(float(differences.mean()))
    p = float(np.mean(np.abs(signs @ differences / len(seeds)) >= observed - 1e-12))
    rng = np.random.default_rng(2371)
    bootstrap = differences[rng.integers(len(seeds), size=(20000, len(seeds)))].mean(axis=1)
    wins = sum(value[left] > value[right] for value in task_rows.values())
    losses = sum(value[left] < value[right] for value in task_rows.values())
    return {'left': left, 'right': right, 'seeds': seeds,
            'difference_percentage_points': float(differences.mean() * 100),
            'difference_by_seed_percentage_points': [float(x * 100) for x in differences],
            'paired_seed_bootstrap_95_percentage_points': [float(x * 100) for x in np.quantile(bootstrap, [0.025, 0.975])],
            'paired_seed_exact_sign_flip_p': p,
            'task_wins': wins, 'task_losses': losses,
            'task_ties': len(task_rows) - wins - losses}


def verify(policy, source, output):
    if output.exists():
        raise FileExistsError('fresh v2 verification path required')
    actor, _ = policy_guard(policy)
    protocol = json.loads((source / 'frozen_protocol.json').read_text())
    if actor['decision_scores_sha256'] != protocol['actor_scores_sha256']:
        raise ValueError('actor changed')
    if monotone_guard(policy) != protocol['monotone_guard']:
        raise ValueError('monotone guard changed')
    for name, expected in protocol['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError('source changed: ' + name)
    inventory = json.loads((source / 'artifact_manifest.json').read_text())
    for name, expected in inventory.items():
        if digest(source / name) != expected:
            raise ValueError('artifact changed: ' + name)
    install_policy(policy / 'decision_scores.json')
    executor = RecoveryExecutor()
    rows_for_statistics = None
    for n in protocol['adaptation_contexts_per_stratum']:
        actual_schedule = json.loads((source / f'admission_{n}.json').read_text())
        actual_rows = json.loads((source / f'test_{n}.json').read_text())
        expected_schedule, expected_rows, total_calls = [], [], 0
        for seed, cycle in product(protocol['seeds'], range(protocol['cycles'])):
            decisions, calls = adaptation(seed, cycle, n, executor)
            expected_schedule.append({'seed': seed, 'cycle': cycle, 'decisions': decisions,
                                      'adaptation_api_calls': calls})
            expected_rows.extend(score(seed, cycle, decisions, executor))
            total_calls += calls
        normalized_schedule = json.loads(json.dumps(expected_schedule))
        if actual_schedule != normalized_schedule:
            raise ValueError('admission records differ from re-execution')
        if actual_rows != expected_rows:
            raise ValueError('test records differ from re-execution')
        if json.loads((source / 'summary.json').read_text())[str(n)] != summarize(expected_rows, total_calls):
            raise ValueError('summary differs from reconstructed rows')
        rows_for_statistics = expected_rows
    report = {'verified': True, 'split': protocol['split'],
              'seeds': protocol['seeds'],
              'blocks_reexecuted': len(protocol['seeds']) * protocol['cycles'],
              'designs': protocol['adaptation_contexts_per_stratum'],
              'recovery_vs_focused': paired_statistics(rows_for_statistics, 'CREDIT-Recovery-v2', 'CREDIT-Focused-v1'),
              'recovery_vs_nomem': paired_statistics(rows_for_statistics, 'CREDIT-Recovery-v2', 'NoMem'),
              'limits': 'same finite simulator and cached actor; recovery semantics are new and shared by all evaluated banks; no natural API claim'}
    write(output, report)
    print(json.dumps({'verified': True,
                      'recovery_vs_focused': report['recovery_vs_focused']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    verify(a.policy, a.source, a.output)
