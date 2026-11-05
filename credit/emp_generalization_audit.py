"""Contract-disjoint, class-balanced support audit of the bounded EMP learner.

Reuses previously examined development observations. This is a retrospective
support audit, not a new blind benchmark. No fitted all-data weights are used:
each fold constructs only its training alphabets. When the held-out contract is
absent, unsupported predictions are independent of the shared weights, so no
gradient fit is needed to prove this architectural limitation.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path

from .emp_binding import BindingEnvironment
from .emp_binding_agent import parse_request
from .emp_joint_v2 import JointStructuredMechanisms, canonical_action, frame_key
from .emp_structural import StructuredMechanisms, api_input, api_observation, apply_learned, fingerprint
from .user_data import digest, jsonl


def contract_blocks(edges):
    """Keep duplicate workflows and duplicate public contracts in one fold."""
    parent = {u: u for u in edges}

    def root(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    seen = {}
    for u in sorted(edges):
        edge = edges[u]
        for key in [('workflow', edge['workflow_group']), ('contract', frame_key(edge))]:
            if key in seen:
                parent[root(u)] = root(seen[key])
            seen[key] = u
    groups = defaultdict(list)
    for u in sorted(edges):
        groups[root(u)].append(u)
    return sorted(groups.values(), key=lambda group: group[0])


def physical_key(env, request):
    value = api_input(env, request)
    value['request'].pop('sdk_target_aliases', None)
    return fingerprint(value)


def observed_inputs(rows, edges):
    """Deduplicate physical API inputs, without counting SDK aliases as data."""
    examples = {}
    for row in rows:
        env = BindingEnvironment(edges[row['edge_uid']])
        for index, event in enumerate(row['events']):
            if index and 'consumer' not in env.responses:
                key = physical_key(env, event['request'])
                label = api_observation(event)
                if key in examples and examples[key]['output'] != label:
                    raise ValueError('conflicting outputs for one deterministic physical input')
                examples.setdefault(key, {'input_uid': key, 'edge_uid': row['edge_uid'],
                    'state': deepcopy(env.responses), 'request': deepcopy(event['request']),
                    'output': label})
            apply_learned(env, event['request'], api_observation(event))
    return sorted(examples.values(), key=lambda example: example['input_uid'])


def training_alphabets(rows, edges, memories):
    literal = StructuredMechanisms().update(rows, edges, memories)
    catalog = defaultdict(set)
    for row in rows:
        env = BindingEnvironment(edges[row['edge_uid']])
        for index, event in enumerate(row['events']):
            if index and index not in row.get('forced_event_indices', []) and 'agent_output' in event:
                catalog[frame_key(env.edge)].add(canonical_action(env, event['agent_output']))
            apply_learned(env, event['request'], api_observation(event))
    return JointStructuredMechanisms(literal, {key: sorted(value) for key, value in catalog.items()})


def evaluate(rows, edges, memories):
    blocks = contract_blocks(edges)
    folds, predictions = [], []
    for number, heldout in enumerate(blocks):
        heldout = set(heldout)
        train = [row for row in rows if row['edge_uid'] not in heldout]
        test = observed_inputs([row for row in rows if row['edge_uid'] in heldout], edges)
        model = training_alphabets(train, edges, memories)
        heldout_frames = {frame_key(edges[u]) for u in heldout}
        train_frames = {frame_key(edges[row['edge_uid']]) for row in train}
        if heldout_frames & train_frames:
            raise ValueError('public contract leaked between folds')
        if heldout_frames & set(model.catalog):
            raise ValueError('heldout policy action alphabet appeared in training')
        if heldout_frames & set(model.outputs):
            raise ValueError('heldout API response alphabet appeared in training')
        train_inputs = {physical_key_for_payload(p) for p in model.literal.api_inputs.values()}
        if train_inputs & {example['input_uid'] for example in test}:
            raise ValueError('physical API input leaked between folds')

        # One legal and one illegal physical request per public contract. The
        # fixed hash order selects examples only after the fold is frozen.
        strata = defaultdict(list)
        for example in test:
            strata[frame_key(edges[example['edge_uid']]), example['output']['accepted']].append(example)
        balanced, missing = [], []
        for frame in sorted(heldout_frames):
            if not all(strata.get((frame, accepted)) for accepted in (False, True)):
                missing.append(frame)
                continue
            balanced.extend(strata[frame, accepted][0] for accepted in (False, True))
        balanced_uids = {example['input_uid'] for example in balanced}
        for example in test:
            env = BindingEnvironment(edges[example['edge_uid']]); env.responses = deepcopy(example['state'])
            distribution = model.api_distribution(env, example['request'])
            # A missing distribution is unknown, never a predicted rejection.
            if distribution:
                raise ValueError('unexpected support for a heldout contract')
            predictions.append({'fold': number, 'input_uid': example['input_uid'],
                'edge_uid': example['edge_uid'], 'accepted': example['output']['accepted'],
                'balanced_subset': example['input_uid'] in balanced_uids,
                'supported': False, 'acceptance_probability': None})
        fold_predictions = [model.predict(edges[u], []) for u in sorted(heldout)]
        if any(p['supported'] for p in fold_predictions):
            raise ValueError('unexpected policy support for a heldout contract')
        folds.append({'fold': number, 'heldout_edge_uids': sorted(heldout),
            'training_edge_uids': sorted(set(edges) - heldout),
            'heldout_contracts': len(heldout_frames), 'training_contracts': len(train_frames),
            'training_API_inputs': len(model.literal.api), 'heldout_physical_inputs': len(test),
            'accepted_inputs': sum(e['output']['accepted'] for e in test),
            'balanced_inputs': len(balanced), 'contracts_without_both_classes': missing,
            'API_prediction_coverage': 0., 'policy_prediction_coverage': 0.,
            'training_alphabets_sha256': fingerprint({'actions': model.catalog, 'api': literal_labels(model)})})
    balanced = [p for p in predictions if p['balanced_subset']]
    positives = sum(p['accepted'] for p in predictions)
    table = []
    for name, subset in [('all_physical_inputs', predictions), ('balanced_contract_inputs', balanced)]:
        accepted = sum(p['accepted'] for p in subset)
        table.append({'subset': name, 'inputs': len(subset), 'accepted': accepted,
            'rejected': len(subset) - accepted, 'API_prediction_coverage': 0.,
            'covered_accuracy': None, 'always_reject_accuracy': 1 - accepted / len(subset),
            'always_accept_accuracy': accepted / len(subset), 'always_reject_balanced_accuracy': .5,
            'always_accept_balanced_accuracy': .5})
    return {'folds': folds, 'predictions': predictions, 'summary': {'table': table,
        'contract_disjoint_folds': len(folds), 'public_contracts': sum(f['heldout_contracts'] for f in folds),
        'unique_physical_inputs': len(predictions), 'accepted_inputs': positives,
        'policy_prediction_coverage': 0., 'refitted_weights': False,
        'reason_no_refit': 'All heldout frames lack training alphabets; abstention holds for every weight vector.',
        'scope': 'retrospective contract-disjoint support audit; observed development data, not new blind trials',
        'new_LLM_calls': 0, 'live_API_calls': 0, 'business_goal_success': None}}


def physical_key_for_payload(payload):
    value = deepcopy(payload); value['request'].pop('sdk_target_aliases', None)
    return fingerprint(value)


def literal_labels(model):
    return {key: sorted(value) for key, value in model.literal.api.items()}


def load(source):
    protocol = json.loads((source / 'frozen_protocol.json').read_text())
    all_edges = {edge['uid']: edge for edge in jsonl(Path(protocol['data_directory']) / 'edges.jsonl')}
    edges = {u: all_edges[u] for u in protocol['training_edge_uids']}
    rows = [row for row in jsonl(source / 'trials.jsonl')
        if row['stage'] in ('intervention_train', 'repair_context_train')]
    rows += list(jsonl(source / 'api_probes.jsonl'))
    if any(row['edge_uid'] not in edges for row in rows):
        raise ValueError('evaluation trial entered training portfolio')
    memories = {note['uid']: note for note in json.loads((source / 'memories.json').read_text())}
    return rows, edges, memories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/emp_complete_v2'))
    parser.add_argument('--output', type=Path, default=Path('results/emp_generalization_v1'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    rows, edges, memories = load(args.source)
    args.output.mkdir(parents=True); (args.output / 'source').mkdir()
    names = ['emp_generalization_audit.py', 'emp_joint_v2.py', 'emp_structural.py', 'emp_binding.py',
        'emp_binding_tools_v3.py', 'emp_binding_tools_v2.py', 'emp_binding_sdk.py',
        'emp_binding_shared.py', 'emp_binding_agent.py', 'emp_binding_pilot.py', 'user_data.py']
    protocol = {'source_directory': str(args.source.resolve()),
        'input_sha256': {name: digest(args.source / name) for name in
            ['trials.jsonl', 'api_probes.jsonl', 'frozen_protocol.json', 'memories.json']},
        'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in names},
        'blocks': contract_blocks(edges), 'balanced_sampling': 'lowest physical input hash per contract and acceptance class',
        'all_data_weights_used': False, 'scope': 'alphabet-only retrospective support audit, not weight generalization'}
    (args.output / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2))
    for name in names:
        (args.output / 'source' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    results = evaluate(rows, edges, memories)
    for name, value in results.items():
        (args.output / (name + '.json')).write_text(json.dumps(value, indent=2))
    with (args.output / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results['summary']['table'][0]))
        writer.writeheader(); writer.writerows(results['summary']['table'])
    print(json.dumps(results['summary'], indent=2))


if __name__ == '__main__':
    main()
