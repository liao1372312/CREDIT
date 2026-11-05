"""Train the shared complete-action learner and its loss ablations."""

import argparse
import json
from math import log
from pathlib import Path
from time import perf_counter

import numpy as np

from .emp_binding import BindingEnvironment
from .emp_joint import fit_joint
from .emp_prospective import choose, coalitions
from .emp_structural import StructuredMechanisms, api_key, api_observation
from .user_data import digest, jsonl


def prepare(source, edges, memories):
    rows = [r for r in jsonl(source / 'trials.jsonl') if r['stage'] in ('intervention_train', 'repair_context_train')]
    base = StructuredMechanisms().update(rows, edges, memories)
    train, holdout, seen = [], [], set()
    for row in jsonl(source / 'api_probes.jsonl'):
        env = BindingEnvironment(edges[row['edge_uid']]); env.call(env.producer_request())
        key = api_key(env, row['events'][1]['request'])
        if key in seen:
            continue
        seen.add(key)
        # Exclude all inputs already present in actual policy/repair traces.
        withheld = key not in base.api and int(key[:8], 16) % 5 == 0
        (holdout if withheld else train).append(row)
    return rows, train, holdout


def score_api(model, holdout, edges):
    nll, mae, correct, distributions = [], [], 0, []
    for row in holdout:
        env = BindingEnvironment(edges[row['edge_uid']]); env.call(env.producer_request())
        event = row['events'][1]
        distribution = model.api_distribution(env, event['request'])
        label = json.dumps(api_observation(event), ensure_ascii=False)
        probability = distribution.get(label)
        accepted = sum(p for encoded, p in distribution.items() if json.loads(encoded)['accepted'])
        mae.append(abs(accepted - float(event['accepted'])))
        correct += (accepted >= .5) == event['accepted']
        if probability is not None:
            nll.append(-log(max(probability, 1e-12)))
        distributions.append({'edge_uid': row['edge_uid'], 'input_uid': api_key(env, event['request']),
            'accepted_probability': accepted, 'observed_accepted': event['accepted'],
            'observed_output_probability': probability, 'known_output_alphabet': probability is not None})
    return {'heldout_API_inputs': len(holdout), 'output_alphabet_covered_inputs': len(nll),
        'acceptance_MAE': float(np.mean(mae)), 'acceptance_accuracy': correct / len(holdout),
        'covered_full_output_NLL': float(np.mean(nll)) if nll else None}, distributions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/emp_complete_v1'))
    parser.add_argument('--output', type=Path, default=Path('results/emp_joint_v1'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    old = json.loads((args.source / 'frozen_protocol.json').read_text()); data = Path(old['data_directory'])
    edges = {r['uid']: r for r in jsonl(data / 'edges.jsonl')}
    pool = json.loads((args.source / 'memories.json').read_text()); memories = {n['uid']: n for n in pool}
    rows, api_train, holdout = prepare(args.source, edges, memories)
    args.output.mkdir(parents=True); (args.output / 'source').mkdir()
    names = ['emp_joint.py', 'emp_joint_experiment.py', 'emp_structural.py', 'emp_prospective.py',
        'emp_binding.py', 'emp_binding_tools_v3.py', 'emp_binding_tools_v2.py', 'emp_binding_sdk.py',
        'emp_binding_shared.py', 'emp_binding_agent.py', 'emp_binding_pilot.py', 'policy.py', 'user_data.py']
    protocol = {'source_directory': str(args.source.resolve()), 'data_directory': str(data.resolve()),
        'input_sha256': {name: digest(args.source / name) for name in ['trials.jsonl', 'api_probes.jsonl', 'memories.json', 'frozen_protocol.json']},
        'edges_sha256': digest(data / 'edges.jsonl'), 'source_sha256': {n: digest(Path(__file__).with_name(n)) for n in names},
        'scope': 'shared finite complete-action/output learner; fixed losses; local API-input holdout, not unseen workflow validation',
        'heldout_API_inputs': [r['bank_uid'] + '@' + r['edge_uid'] for r in holdout],
        'test_workflow_outcomes_used_for_fitting': False, 'test_portfolio_previously_examined': True,
        'epochs': 250, 'variants': {'full': [.2, .002], 'without_outcome': [0., .002], 'without_group_penalty': [.2, 0.]},
        'semantic_encoder': 'shared linear public typed request/route features, not learned text embeddings',
        'new_LLM_calls': 0, 'live_API_calls': 0, 'certificate': False}
    (args.output / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2))
    for name in names:
        (args.output / 'source' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    tables, predictions = [], []
    future = [edges[u] for u in old['selection_edge_uids']]; test = [edges[u] for u in old['test_edge_uids']]
    for variant, (outcome, group) in protocol['variants'].items():
        began = perf_counter()
        model, fit, literal = fit_joint(rows, api_train, edges, memories,
            outcome_weight=outcome, group_weight=group, epochs=protocol['epochs'])
        selected, scores = choose(model, future, coalitions(pool, 2))
        metrics, api_predictions = score_api(model, holdout, edges)
        for edge in test:
            predictions.append({'variant': variant, 'edge_uid': edge['uid'], **model.predict(edge, [])})
        (args.output / (variant + '.json')).write_text(json.dumps({'model': model.to_dict(), 'literal': literal.to_dict(),
            'fit': fit, 'selected_bank': [n['uid'] for n in selected], 'selection_scores': scores,
            'API_holdout_predictions': api_predictions}, ensure_ascii=False, indent=2))
        row = {'variant': variant, **metrics, 'parameters': fit['parameters'],
            'final_loss': fit['curve'][-1]['loss'], 'final_decision_NLL': fit['curve'][-1]['decision_nll'],
            'final_API_NLL': fit['curve'][-1]['API_nll'], 'final_outcome_BCE': fit['curve'][-1]['outcome_bce'],
            'group_norm': fit['curve'][-1]['group_norm'], 'elapsed_seconds': perf_counter() - began}
        tables.append(row); print(json.dumps(row), flush=True)
    (args.output / 'predictions.json').write_text(json.dumps(predictions, indent=2))
    (args.output / 'summary.json').write_text(json.dumps({'table': tables, 'new_LLM_calls': 0,
        'heldout_workflow_prediction_coverage': sum(p['supported'] for p in predictions) / len(predictions),
        'scope': protocol['scope']}, indent=2))


if __name__ == '__main__':
    main()
