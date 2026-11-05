"""JSON-normalized reconstruction audit for the full recovery wrapper."""
import argparse
import json
from pathlib import Path

from .llm_policy import install_policy
from .v3.experiment import digest, write
from .v3_remote_recovery_full import (CONFIRM_SEEDS, DEV_SEEDS, fixed_protocol,
                                      install_runtime)


def normalized(value):
    return json.loads(json.dumps(value))


def audit(policy, dataset, output, split):
    expected = normalized(fixed_protocol(policy, dataset, split))
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
        'inner_audit_sha256': digest(inner), 'auditor_sha256': digest(__file__),
        'limits': expected['limits']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('development', 'confirmation'), required=True)
    a = p.parse_args()
    audit(a.policy, a.dataset, a.output, a.split)
