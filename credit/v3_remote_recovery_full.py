"""Full chronological CREDIT protocol under versioned recovery semantics.

This wrapper keeps the original v3 sources immutable and installs one runtime
configuration consistently in history collection, randomized replay, model
fitting, screening, certification, held-out scoring, and reconstruction audit.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from .llm_policy import install_policy
from .v3.experiment import digest, write
from .v3.local import ROLES
from .v3_recovery import RecoveryExecutor
from .v3_remote_llm_experiment import policy_guard

DEV_SEEDS = (61001,)
CONFIRM_SEEDS = (62003, 62011, 62017, 62039, 62047, 62053, 62057, 62071)
CYCLES = 3


def recovery_registries():
    from .v3.local import memories as original_memories
    from .v3_revalidation import candidate_memories as original_candidates
    submit = ROLES['submit'].key
    source = tuple(replace(memory, scope=frozenset(set(memory.scope or ()) | {submit}))
                   if memory.uid == 'repair_rule' else memory
                   for memory in original_memories())
    novel = next(memory for memory in original_candidates() if memory.uid == 'new_contract')
    return source, source + (novel,)


RECOVERY_SOURCE, RECOVERY_POOL = recovery_registries()


class RecoveryBudgetExecutor(RecoveryExecutor):
    def __init__(self, ledger):
        super().__init__()
        self.ledger = ledger

    def replay(self, context, exposure, reference, *, slots=None):
        # Full paired reconstruction may add lookup/transform/authorize/submit.
        if self.ledger.spent + 2 * (len(context.nodes) + 4) > self.ledger.call_cap:
            raise ValueError('reserve recovery replay bound before physical execution')
        return super().replay(context, exposure, reference, slots=slots)


def install_runtime(seeds):
    """Install the same registry and executor at every execution boundary."""
    from . import v3_online_data as data
    from . import v3_online_experiment as experiment
    from . import v3_online_verify as verify
    from . import v3_optimization as optimization
    from . import v3_online as online
    from . import v3_online_fast as fast
    from . import v3_revalidation as revalidation
    from . import v3_benchmark as benchmark
    from . import v3_screening_optimization as screening
    from .v3_online_fast import fit_ensemble

    source = lambda: RECOVERY_SOURCE
    pool = lambda: RECOVERY_POOL
    optimization.memories = source
    benchmark.memories = source
    screening.memories = source
    revalidation.candidate_memories = pool
    online.candidate_memories = pool
    fast.candidate_memories = pool
    data.candidate_memories = pool
    experiment.candidate_memories = pool
    verify.candidate_memories = pool

    optimization.StatefulExecutor = RecoveryExecutor
    online.RevalidationExecutor = RecoveryExecutor
    experiment.RevalidationExecutor = RecoveryExecutor
    experiment.BudgetExecutor = RecoveryBudgetExecutor
    verify.RevalidationExecutor = RecoveryExecutor

    data.SEEDS = tuple(seeds)
    experiment.SEEDS = tuple(seeds)
    verify.SEEDS = tuple(seeds)
    experiment.CYCLES = CYCLES
    verify.CYCLES = CYCLES
    experiment.fit_ensemble = fit_ensemble
    return experiment, verify, data


def registry_payload():
    return [{'uid': memory.uid, 'scope': sorted(memory.scope or ()),
             'tokens': list(memory.tokens), 'arrival': memory.arrival}
            for memory in RECOVERY_POOL]


def generate_dataset(output, split):
    seeds = DEV_SEEDS if split == 'development' else CONFIRM_SEEDS
    _, _, data = install_runtime(seeds)
    data.generate(output)
    write(output / 'recovery_dataset_protocol.json', {
        'split': split, 'seeds': seeds, 'cycles': CYCLES,
        'registry': registry_payload(),
        'wrapper_sha256': digest(__file__), 'recovery_executor_sha256': digest('credit/v3_recovery.py')})
    # Bind the additional protocol into the canonical manifest.
    manifest = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'manifest.json'}
    write(output / 'manifest.json', manifest)


def fixed_protocol(policy, dataset, split):
    actor, _ = policy_guard(policy)
    seeds = DEV_SEEDS if split == 'development' else CONFIRM_SEEDS
    return {'scope': 'full chronological CREDIT and Direct-Sandbox comparison with memory-guided recovery',
            'split': split, 'seeds': seeds, 'cycles': CYCLES,
            'dataset': str(dataset.resolve()), 'dataset_manifest_sha256': digest(dataset / 'manifest.json'),
            'remote_policy': str(policy.resolve()),
            'actor_scores_sha256': actor['decision_scores_sha256'],
            'registry': registry_payload(),
            'specifications': [['CREDIT-Sparse', 'sparse'], ['CREDIT-Covered', 'covered'],
                               ['CREDIT-VOI', 'voi'], ['CREDIT-Random', 'random'],
                               ['CREDIT-NoRevalidation', 'none'], ['Direct-Covered', 'direct']],
            'cycle_call_cap': 300000,
            'matched_budget_definition': 'same hard 300000 extra-call allowance per method and cycle; actual calls reported',
            'source_sha256': {name: digest(name) for name in (
                'credit/v3_remote_recovery_full.py', 'credit/v3_recovery.py',
                'credit/v3_online_experiment.py', 'credit/v3_online.py',
                'credit/v3_online_fast.py', 'credit/v3_online_verify.py',
                'credit/v3_optimization.py', 'credit/v3_revalidation.py',
                'credit/llm_policy.py', 'credit/stateful.py')},
            'primary': 'CREDIT-Covered versus Direct-Covered under identical recovery runtime',
            'secondary': 'CREDIT-Covered versus CREDIT-Sparse and CREDIT-NoRevalidation',
            'limits': 'finite synthetic API and cached remote actor; recovery scope is versioned and not a natural-service claim'}


def run(policy, dataset, output, split):
    if output.exists():
        raise FileExistsError('fresh full recovery output required')
    protocol = fixed_protocol(policy, dataset, split)
    seeds = DEV_SEEDS if split == 'development' else CONFIRM_SEEDS
    experiment, _, _ = install_runtime(seeds)
    install_policy(policy / 'decision_scores.json')
    output.mkdir(parents=True)
    write(output / 'recovery_frozen_protocol.json', protocol)
    experiment.run(dataset, output / 'chronological')
    write(output / 'summary.json', {
        'split': split,
        'chronological_summary': json.loads((output / 'chronological/summary.json').read_text()),
        'statistics': json.loads((output / 'chronological/statistics.json').read_text()),
        'secondary_statistics': json.loads((output / 'chronological/secondary_statistics.json').read_text())})
    write(output / 'artifact_manifest.json', {
        'recovery_frozen_protocol.json': digest(output / 'recovery_frozen_protocol.json'),
        'summary.json': digest(output / 'summary.json'),
        'chronological/artifact_manifest.json': digest(output / 'chronological/artifact_manifest.json'),
        'dataset/manifest.json': digest(dataset / 'manifest.json'),
        'remote_policy/decision_scores.json': digest(policy / 'decision_scores.json')})


def audit(policy, dataset, output, split):
    expected = fixed_protocol(policy, dataset, split)
    if json.loads((output / 'recovery_frozen_protocol.json').read_text()) != expected:
        raise ValueError('outer recovery protocol changed')
    for name, expected_digest in json.loads((output / 'artifact_manifest.json').read_text()).items():
        path = (dataset / 'manifest.json' if name.startswith('dataset/') else
                policy / 'decision_scores.json' if name.startswith('remote_policy/') else output / name)
        if digest(path) != expected_digest:
            raise ValueError('outer artifact changed: ' + name)
    seeds = DEV_SEEDS if split == 'development' else CONFIRM_SEEDS
    _, verify, _ = install_runtime(seeds)
    install_policy(policy / 'decision_scores.json')
    verify.audit(output / 'chronological', streaming=False)
    inner = output / 'chronological_audit.json'
    write(output / 'recovery_execution_audit.json', {
        'verified': True, 'split': split, 'seeds': seeds,
        'outer_protocol_sha256': digest(output / 'recovery_frozen_protocol.json'),
        'inner_audit_sha256': digest(inner),
        'auditor_sha256': digest(__file__),
        'limits': expected['limits']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, default=Path('results/credit_v3_remote_llm_policy_v2'))
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--split', choices=('development', 'confirmation'), required=True)
    p.add_argument('--generate-dataset', action='store_true')
    p.add_argument('--audit', action='store_true')
    a = p.parse_args()
    if a.generate_dataset:
        generate_dataset(a.dataset, a.split)
    elif a.audit:
        audit(a.policy, a.dataset, a.output, a.split)
    else:
        run(a.policy, a.dataset, a.output, a.split)
