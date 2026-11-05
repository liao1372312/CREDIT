"""Replay explicit fault collection and reconstruct blocked feature diagnostics."""

import argparse
import csv
import json
from pathlib import Path

from .emp_argument_audit import blocked_diagnostic, environment_with_state, fault_rows
from .emp_binding_v2_verify import verify_sources
from .emp_joint_v4 import public_acceptance_possible
from .user_data import digest


def verify(output):
    protocol = verify_sources(output); source = Path(protocol['source_directory'])
    verify_sources(source)
    for name, expected in protocol['input_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('original observations changed')
    rows, examples, edges = fault_rows(source)
    for name, expected in [('controlled_rows', rows), ('examples', examples)]:
        if json.loads((output / (name + '.json')).read_text()) != expected:
            raise ValueError('fault collection differs: ' + name)
    results = blocked_diagnostic(examples, edges)
    for name, expected in results.items():
        if json.loads((output / (name + '.json')).read_text()) != expected:
            raise ValueError('blocked diagnostic differs: ' + name)
    baseline = []
    for family in ('value_fault', 'operation_fault'):
        subset = [e for e in examples if e['partition'] != 'train'
            and e['family'] in ('legal_anchor', family)]
        predictions = [public_acceptance_possible(environment_with_state(edges[e['edge_uid']], e['state']), e['request'])
            for e in subset]
        accuracy = sum(p == e['output']['accepted'] for p, e in zip(predictions, subset)) / len(subset)
        baseline.append({'version': 'public_consistency_guard', 'fault_family': family,
            'inputs': len(subset), 'legal_inputs': sum(e['output']['accepted'] for e in subset),
            'illegal_inputs': sum(not e['output']['accepted'] for e in subset),
            'accuracy': accuracy, 'balanced_accuracy': accuracy,
            'Brier': 1 - accuracy, 'binary_NLL': None,
            'always_reject_accuracy': .5, 'always_accept_accuracy': .5,
            'general_CREDIT_contract_support': False})
    (output / 'public_guard_baseline.json').write_text(json.dumps({
        'baseline': baseline, 'baseline_source_sha256': digest(Path(__file__).with_name('emp_joint_v4.py')),
        'scope': 'necessary public checks resolve these constructed faults; not an oracle for arbitrary source bindings'}, indent=2))
    table = results['summary']['table'] + baseline
    with (output / 'comparison_with_guard.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    report = {'original_sources_and_observations_unchanged': True,
        'fault_trials_replayed': len(rows), 'controlled_collection_local_calls': sum(len(r['events']) for r in rows),
        'contract_disjoint_classifier_fits_reconstructed': len(results['fits']),
        'matched_legal_and_illegal_classes_verified': True, 'public_guard_baseline_verified': True,
        'all_data_weights_used': False, 'new_LLM_calls': 0, 'live_API_calls': 0,
        'business_goal_success': None, 'verifier_sha256': digest(Path(__file__))}
    (output / 'integrity.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/emp_argument_audit_v1'))
    verify(parser.parse_args().results)
