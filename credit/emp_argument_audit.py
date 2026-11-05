"""Controlled argument faults and blocked diagnostics for public API features.

Faults are explicitly constructed local tests, not observed provider drift.
The diagnostic classifier predicts acceptance only; it does not replace CREDIT's
complete-response kernel or confer support on unseen contracts.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from .emp_binding import BindingEnvironment, get_path, set_path
from .emp_generalization_audit_v2 import contract_blocks, load, observed_inputs, physical_key
from .emp_joint_v2 import frame_key, request_features as old_features
from .emp_joint_v3 import request_features as corrected_features
from .user_data import digest


def changed_value(value, suffix):
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + suffix
    if isinstance(value, (int, float)):
        return value + (101 if suffix == '__train_value_fault' else 202)
    raise ValueError('no type-preserving identity-value fault for this scalar')


def fault_rows(source):
    rows, edges, memories = load(source)
    positives, seen = [], set()
    for example in observed_inputs(rows, edges):
        frame = frame_key(edges[example['edge_uid']])
        if example['output']['accepted'] and frame not in seen:
            positives.append(example); seen.add(frame)
    generated, examples = [], []
    for positive in positives:
        edge = edges[positive['edge_uid']]
        target = sorted(binding['target'] for binding in edge['bindings'])[0]
        examples.append({**positive, 'family': 'legal_anchor', 'partition': 'anchor'})
        for partition in ('train', 'eval'):
            for family in ('value_fault', 'operation_fault'):
                env = BindingEnvironment(edge); env.call(env.producer_request())
                request = deepcopy(positive['request'])
                if family == 'value_fault':
                    value = get_path(request['arguments'], target)
                    set_path(request['arguments'], target,
                        changed_value(value, '__' + partition + '_value_fault'))
                else:
                    request['bindings'][target]['operation'] = 'controlled_' + partition + '_wrong_operation'
                event = env.call(request)
                if event['accepted']:
                    raise ValueError('constructed fault was not rejected by the declared contract')
                key = physical_key(environment_with_state(edge, positive['state']), request)
                example = {'input_uid': key, 'edge_uid': edge['uid'], 'state': positive['state'],
                    'request': request, 'output': {k: event[k] for k in ('accepted', 'response', 'errors', 'semantic_scope')},
                    'family': family, 'partition': partition}
                examples.append(example)
                generated.append({'stage': 'api_fixture_train', 'workflow_group': edge['workflow_group'],
                    'edge_uid': edge['uid'], 'bank_uid': partition + ':' + family, 'exposure': [],
                    'forced_event_indices': [1], 'events': deepcopy(env.history), 'final': env.verify(),
                    'provenance': 'explicit controlled local argument fault; no new LLM/provider observation'})
    return generated, examples, edges


def environment_with_state(edge, state):
    env = BindingEnvironment(edge); env.responses = deepcopy(state)
    return env


def feature_vector(example, edges, feature_function):
    env = environment_with_state(edges[example['edge_uid']], example['state'])
    return feature_function(env, example['request'])


def blocked_diagnostic(examples, edges):
    blocks = contract_blocks(edges); predictions, fits = [], []
    for fold, block in enumerate(blocks):
        frames = {frame_key(edges[u]) for u in block}
        train = [e for e in examples if e['partition'] != 'eval' and frame_key(edges[e['edge_uid']]) not in frames]
        test = [e for e in examples if e['partition'] != 'train' and frame_key(edges[e['edge_uid']]) in frames]
        counts = Counter((frame_key(edges[e['edge_uid']]), e['family']) for e in train)
        n_frames = len({frame for frame, _ in counts})
        # Uniform contracts; legal anchors carry half the weight and the two
        # controlled rejection families share the other half.
        weights = np.asarray([(.5 if e['family'] == 'legal_anchor' else .25) / (
            n_frames * counts[frame_key(edges[e['edge_uid']]), e['family']]) for e in train])
        weights *= len(train)
        for version, features in [('old_13_features', old_features), ('corrected_16_features', corrected_features)]:
            x = np.asarray([feature_vector(e, edges, features) for e in train]); y = np.asarray([e['output']['accepted'] for e in train])
            scaler = StandardScaler().fit(x)
            classifier = LogisticRegression(C=1., max_iter=2000, random_state=42)
            classifier.fit(scaler.transform(x), y, sample_weight=weights)
            scores = classifier.predict_proba(scaler.transform(np.asarray([feature_vector(e, edges, features) for e in test])))[:, 1]
            fits.append({'fold': fold, 'version': version, 'heldout_frames': sorted(frames),
                'training_frames': sorted({frame_key(edges[e['edge_uid']]) for e in train}),
                'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                'coefficients': classifier.coef_.tolist(), 'intercept': classifier.intercept_.tolist()})
            for e, probability in zip(test, scores):
                predictions.append({'fold': fold, 'version': version, 'input_uid': e['input_uid'],
                    'edge_uid': e['edge_uid'], 'family': e['family'], 'accepted': e['output']['accepted'],
                    'acceptance_probability': float(probability)})
    table = []
    for version in ('old_13_features', 'corrected_16_features'):
        for family in ('value_fault', 'operation_fault'):
            subset = [p for p in predictions if p['version'] == version and p['family'] in ('legal_anchor', family)]
            y = np.asarray([p['accepted'] for p in subset]); p = np.asarray([p['acceptance_probability'] for p in subset])
            table.append({'version': version, 'fault_family': family, 'inputs': len(y),
                'legal_inputs': int(y.sum()), 'illegal_inputs': int((~y).sum()),
                'accuracy': float(np.mean((p >= .5) == y)), 'balanced_accuracy': float(balanced_accuracy_score(y, p >= .5)),
                'Brier': float(np.mean((p-y)**2)), 'binary_NLL': float(log_loss(y, p)),
                'always_reject_accuracy': .5, 'always_accept_accuracy': .5,
                'general_CREDIT_contract_support': False})
    return {'fits': fits, 'predictions': predictions, 'summary': {'table': table,
        'contract_disjoint_blocks': len(blocks), 'public_contracts': len(examples) // 5,
        'all_data_weights_used': False, 'C': 1.,
        'scope': 'retrospective controlled-fault feature diagnostic; binary classifier only, no full-response/agent transfer',
        'new_LLM_calls': 0, 'live_API_calls': 0, 'business_goal_success': None}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/emp_complete_v2'))
    parser.add_argument('--output', type=Path, default=Path('results/emp_argument_audit_v1'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    args.output.mkdir(parents=True); (args.output / 'source').mkdir()
    names = ['emp_argument_audit.py', 'emp_generalization_audit_v2.py', 'emp_joint_v2.py', 'emp_joint_v3.py',
        'emp_structural.py', 'emp_binding.py', 'emp_binding_tools_v3.py', 'emp_binding_tools_v2.py',
        'emp_binding_sdk.py', 'emp_binding_shared.py', 'emp_binding_agent.py', 'emp_binding_pilot.py', 'user_data.py']
    protocol = {'source_directory': str(args.source.resolve()),
        'input_sha256': {n: digest(args.source / n) for n in ('trials.jsonl', 'api_probes.jsonl', 'frozen_protocol.json', 'memories.json')},
        'source_sha256': {n: digest(Path(__file__).with_name(n)) for n in names},
        'experiment': 'legal anchors plus explicitly constructed value/provenance faults; whole-contract exclusion',
        'feature_checks': 'public request/source value, declared operation, fixed arguments; no private correct-source lookup',
        'binary_classifier_only': True, 'C': 1., 'max_iter': 2000}
    (args.output / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2))
    for name in names:
        (args.output / 'source' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    rows, examples, edges = fault_rows(args.source)
    (args.output / 'controlled_rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (args.output / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    result = blocked_diagnostic(examples, edges)
    for name, value in result.items():
        (args.output / (name + '.json')).write_text(json.dumps(value, indent=2))
    with (args.output / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['summary']['table'][0])); writer.writeheader()
        writer.writerows(result['summary']['table'])
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
