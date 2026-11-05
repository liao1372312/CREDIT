"""Re-execute and audit saved focused remote actor studies."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np

from .llm_policy import install_policy
from .v3.experiment import digest, write
from .v3_revalidation import RevalidationExecutor
from .v3_remote_focused_experiment import adaptation, aggregate, score
from .v3_remote_llm_experiment import policy_guard


def verify(policy, source, output):
    if output.exists():
        raise FileExistsError('fresh verification path required')
    actor, _ = policy_guard(policy)
    protocol = json.loads((source / 'frozen_protocol.json').read_text())
    if digest('credit/v3_remote_focused_experiment.py') != protocol['generator_sha256']:
        raise ValueError('focused generator changed')
    if actor['decision_scores_sha256'] != protocol['actor_scores_sha256']:
        raise ValueError('actor changed')
    inventory = json.loads((source / 'artifact_manifest.json').read_text())
    for name, expected in inventory.items():
        if digest(source / name) != expected:
            raise ValueError('artifact changed: ' + name)
    install_policy(policy / 'decision_scores.json')
    executor = RevalidationExecutor()
    for n in protocol['adaptation_contexts']:
        actual_schedule = json.loads((source / f'admission_{n}.json').read_text())
        actual_rows = json.loads((source / f'test_{n}.json').read_text())
        if len(actual_schedule) != len(protocol['seeds']) * protocol['cycles']:
            raise ValueError('missing admission block')
        expected_schedule, expected_rows, total_calls = [], [], 0
        for seed, cycle in product(protocol['seeds'], range(protocol['cycles'])):
            decisions, calls = adaptation(seed, cycle, n, executor)
            expected_schedule.append({'seed': seed, 'cycle': cycle, 'decisions': decisions,
                                      'adaptation_api_calls': calls})
            expected_rows.extend(score(seed, cycle, decisions, executor))
            total_calls += calls
        if actual_schedule != expected_schedule:
            raise ValueError('admission records differ from re-execution')
        if actual_rows != expected_rows:
            raise ValueError('heldout records differ from re-execution')
        actual_summary = json.loads((source / 'summary.json').read_text())[str(n)]
        if actual_summary != aggregate(expected_rows, total_calls):
            raise ValueError('summary differs from rows')
    n = protocol['adaptation_contexts'][0]
    rows = json.loads((source / f'test_{n}.json').read_text())
    by_seed = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_seed[row['seed']][row['method']].append(int(row['success']))
    seeds = list(protocol['seeds'])
    paired = np.asarray([np.mean(by_seed[s]['CREDIT-Focused']) - np.mean(by_seed[s]['NoMem'])
                         for s in seeds], dtype=float)
    # Exact paired sign-flip reference for independent seed units. The three
    # cycles remain grouped within each seed.
    signs = np.asarray(list(product((-1, 1), repeat=len(seeds))), dtype=float)
    exact_p = float(np.mean(np.abs(signs @ paired / len(seeds)) >= abs(np.mean(paired)) - 1e-12))
    rng = np.random.default_rng(1847)
    boot = np.mean(paired[rng.integers(len(seeds), size=(20000, len(seeds)))], axis=1)
    report = {'verified': True, 'split': protocol['split'], 'seeds': seeds,
              'blocks_reexecuted_per_design': len(seeds) * protocol['cycles'],
              'designs': protocol['adaptation_contexts'],
              'focused_minus_nomem_percentage_points': float(np.mean(paired) * 100),
              'paired_seed_bootstrap_95_percentage_points': [float(x * 100) for x in np.quantile(boot, [0.025, 0.975])],
              'paired_seed_exact_sign_flip_p': exact_p,
              'focused_minus_nomem_by_seed_percentage_points': [float(x * 100) for x in paired],
              'limits': 'reconstruction uses the same simulator and cached actor; seeds are independent units; no general API or prompt validation'}
    write(output, report)
    print(json.dumps({'verified': True, 'split': protocol['split'],
                      'paired_p': exact_p, 'effect_pp': report['focused_minus_nomem_percentage_points']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    verify(a.policy, a.source, a.output)
