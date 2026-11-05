"""Development ceiling for contract and recovery memories under recovery semantics."""
import argparse
from collections import defaultdict
from pathlib import Path

from .llm_policy import install_policy
from .v3.experiment import digest, write
from .v3.local import factory, retriever
from .v3_recovery import RecoveryExecutor
from .v3_remote_expansion_diagnostic import DEV_SEEDS, CONFIRM_SEEDS, STRATA, TASKS_PER_STRATUM
from .v3_remote_llm_experiment import policy_guard

BANKS = ((), ('new_contract',), ('repair_rule',), ('new_contract', 'repair_rule'))


def run(policy, output):
    if output.exists():
        raise FileExistsError('fresh recovery diagnostic directory required')
    manifest, _ = policy_guard(policy)
    output.mkdir(parents=True)
    write(output / 'frozen_protocol.json', {
        'scope': 'development-only recovery semantic diagnostic',
        'development_seeds': DEV_SEEDS, 'unused_confirmation_seeds': CONFIRM_SEEDS,
        'strata': STRATA, 'tasks_per_stratum': TASKS_PER_STRATUM,
        'phase': 'remote-recovery-development-v1', 'banks': BANKS,
        'actor_scores_sha256': manifest['decision_scores_sha256'],
        'source_sha256': {name: digest(name) for name in (
            'credit/v3_remote_recovery_diagnostic.py', 'credit/v3_recovery.py',
            'credit/llm_policy.py', 'credit/stateful.py', 'credit/v3/local.py')},
        'confirmation_rule': 'freeze adaptation design before any confirmation task is executed'})
    install_policy(policy / 'decision_scores.json')
    executor, retrieve = RecoveryExecutor(), retriever(4)
    successes, calls = defaultdict(lambda: defaultdict(int)), defaultdict(int)
    for seed in DEV_SEEDS:
        make = factory(seed)
        for stratum in STRATA:
            for i in range(TASKS_PER_STRATUM):
                task = make(stratum, 'remote-recovery-development-v1', i)
                for bank in BANKS:
                    episode = executor.run(task, retrieve(task, bank))
                    successes[bank][stratum] += int(bool(episode.verified))
                    calls[bank] += episode.calls
    rows = [{'bank': bank, 'stratum_successes': dict(successes[bank]),
             'total_successes': sum(successes[bank].values()),
             'api_calls': calls[bank]} for bank in BANKS]
    best = {stratum: max(rows, key=lambda row: (row['stratum_successes'][stratum],
                                                -len(row['bank']), row['bank']))
            for stratum in STRATA}
    total = len(DEV_SEEDS) * len(STRATA) * TASKS_PER_STRATUM
    write(output / 'bank_scores.json', rows)
    write(output / 'summary.json', {'tasks_per_bank': total,
                                    'bank_vsr': {','.join(row['bank']) or 'empty': row['total_successes'] / total
                                                 for row in rows},
                                    'best_by_stratum': {s: {'bank': r['bank'],
                                                           'successes': r['stratum_successes'][s]}
                                                        for s, r in best.items()},
                                    'conditional_oracle_vsr': sum(r['stratum_successes'][s]
                                                                  for s, r in best.items()) / total})
    write(output / 'artifact_manifest.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})
    print({'bank_vsr': {','.join(row['bank']) or 'empty': row['total_successes'] / total for row in rows},
           'oracle': sum(r['stratum_successes'][s] for s, r in best.items()) / total}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.policy, a.output)
