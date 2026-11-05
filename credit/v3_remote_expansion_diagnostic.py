"""Development search over previously prompted but undeployed memories."""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from .llm_policy import install_policy, MEMORY_TEXT
from .stateful import MEMORIES, DEPENDENCIES
from .v3.experiment import digest, write
from .v3.local import ROLES, factory, retriever, stable_seed
from .v3.types import Memory
from .v3_revalidation import RevalidationExecutor
from .v3_remote_llm_experiment import policy_guard

DEV_SEEDS = (2909, 2917, 3907, 3911)
CONFIRM_SEEDS = (4109, 4111, 4127, 4133, 4139, 4153, 4157, 4159)
STRATA = tuple(f'{family}/{epoch}' for family in ('books', 'travel', 'inventory', 'payments')
               for epoch in ('source', 'target'))
TASKS_PER_STRATUM = 64


def expanded_memories():
    result = []
    for arrival, uid in enumerate(MEMORIES):
        text = MEMORY_TEXT[uid]
        scope = frozenset(ROLES[role].key for role in DEPENDENCIES[uid])
        result.append(Memory(uid, text, tuple(stable_seed(word) % 2**31 for word in text.split()),
                             scope, 'remote prompt inventory; numeric effects excluded', arrival))
    return tuple(result)


def run(policy, output):
    if output.exists():
        raise FileExistsError('fresh expansion diagnostic directory required')
    manifest, _ = policy_guard(policy)
    pool = expanded_memories()
    ids = tuple(memory.uid for memory in pool if memory.scope)
    banks = [bank for size in range(3) for bank in combinations(ids, size)]
    output.mkdir(parents=True)
    write(output / 'frozen_protocol.json', {
        'scope': 'development-only one/two-memory expansion search for the finite actor',
        'development_seeds': DEV_SEEDS, 'unused_confirmation_seeds': CONFIRM_SEEDS,
        'strata': STRATA, 'tasks_per_stratum': TASKS_PER_STRATUM,
        'phase': 'remote-expansion-development-v1', 'maximum_bank_size': 2,
        'candidate_ids': ids, 'candidate_banks': len(banks),
        'actor_scores_sha256': manifest['decision_scores_sha256'],
        'source_sha256': {name: digest(name) for name in (
            'credit/v3_remote_expansion_diagnostic.py', 'credit/llm_policy.py',
            'credit/stateful.py', 'credit/v3/local.py')},
        'selection_rule': 'per-stratum highest successes; ties prefer fewer memories then lexical bank',
        'claim_limit': 'development search; confirmation seeds frozen and untouched'})
    install_policy(policy / 'decision_scores.json')
    executor, retrieve = RevalidationExecutor(), retriever(4)
    successes = defaultdict(lambda: defaultdict(int))
    calls = defaultdict(int)
    unique = 0
    for seed in DEV_SEEDS:
        make = factory(seed)
        for stratum in STRATA:
            for i in range(TASKS_PER_STRATUM):
                task = make(stratum, 'remote-expansion-development-v1', i)
                cache = {}
                for bank in banks:
                    exposure = retrieve(task, bank)
                    if exposure not in cache:
                        episode = executor.run(task, exposure)
                        cache[exposure] = (int(bool(episode.verified)), episode.calls)
                        unique += 1
                    success, api_calls = cache[exposure]
                    successes[bank][stratum] += success
                    calls[bank] += api_calls
    rows = []
    for bank in banks:
        rows.append({'bank': bank, 'stratum_successes': dict(successes[bank]),
                     'total_successes': sum(successes[bank].values()),
                     'physical_calls_if_uncached': calls[bank]})
    rows.sort(key=lambda row: (-row['total_successes'], len(row['bank']), row['bank']))
    best = {}
    for stratum in STRATA:
        row = max(rows, key=lambda x: (x['stratum_successes'][stratum], -len(x['bank']),
                                      tuple(reversed(x['bank']))))
        best[stratum] = {'bank': row['bank'], 'successes': row['stratum_successes'][stratum],
                         'tasks': len(DEV_SEEDS) * TASKS_PER_STRATUM}
    write(output / 'bank_scores.json', rows)
    write(output / 'summary.json', {'tested_banks': len(banks), 'tasks': len(DEV_SEEDS) * len(STRATA) * TASKS_PER_STRATUM,
                                    'unique_executions': unique, 'best_by_stratum': best,
                                    'conditional_oracle_vsr': sum(x['successes'] for x in best.values()) /
                                                              (len(DEV_SEEDS) * len(STRATA) * TASKS_PER_STRATUM)})
    write(output / 'artifact_manifest.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})
    print({'banks': len(banks), 'unique_executions': unique,
           'conditional_oracle_vsr': sum(x['successes'] for x in best.values()) /
                                     (len(DEV_SEEDS) * len(STRATA) * TASKS_PER_STRATUM)}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.policy, a.output)
