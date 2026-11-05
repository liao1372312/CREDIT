"""Rebuild shared learner losses, disjoint API holdout and all predictions."""

import argparse
from pathlib import Path
import json

import numpy as np

from .emp_binding import BindingEnvironment
from .emp_binding_v2_verify import read, verify_sources
from .emp_binding_verify import require
from .emp_joint_v3 import fit_joint
from .emp_joint_experiment_v3 import prepare, score_api, controlled_training, score_controlled
from .emp_prospective import choose, coalitions
from .emp_structural import api_key, api_input, fingerprint
from copy import deepcopy
from .user_data import digest, jsonl


def verify(output):
    protocol = verify_sources(output)
    source, data = Path(protocol['source_directory']), Path(protocol['data_directory'])
    verify_sources(source)
    for name, expected in protocol['input_sha256'].items():
        require(digest(source / name) == expected, 'joint training source changed')
    require(digest(data / 'edges.jsonl') == protocol['edges_sha256'], 'edge data changed')
    faults = Path(protocol['fault_directory'])
    verify_sources(faults)
    for name, expected in protocol['fault_sha256'].items():
        require(digest(faults / name) == expected, 'controlled fault source changed')
    edges = {e['uid']: e for e in jsonl(data / 'edges.jsonl')}
    pool = read(source, 'memories.json'); memories = {n['uid']: n for n in pool}
    rows, api_train, holdout = prepare(source, edges, memories)
    api_train += controlled_training(faults)
    require([r['bank_uid'] + '@' + r['edge_uid'] for r in holdout] == protocol['heldout_API_inputs'], 'holdout assignment differs')
    require(all(r['stage'] != 'heldout_evaluation' for r in rows), 'test outcomes used for fitting')
    old = read(source, 'frozen_protocol.json')
    future = [edges[u] for u in old['selection_edge_uids']]
    test = [edges[u] for u in old['test_edge_uids']]
    summary = read(output, 'summary.json'); predictions = []
    for variant, (outcome, group) in protocol['variants'].items():
        saved = read(output, variant + '.json')
        model, fit, literal = fit_joint(rows, api_train, edges, memories,
            outcome_weight=outcome, group_weight=group, epochs=protocol['epochs'])
        require(model.to_dict() == saved['model'] and literal.to_dict() == saved['literal'], 'joint weights/alphabet reconstruction differs')
        require(fit == saved['fit'], 'joint loss curve reconstruction differs')
        def physical_key(payload):
            value = deepcopy(payload); value['request'].pop('sdk_target_aliases', None)
            return fingerprint(value)
        physical_inputs = {physical_key(p) for p in literal.api_inputs.values()}
        for row in holdout:
            env = BindingEnvironment(edges[row['edge_uid']]); env.call(env.producer_request())
            require(api_key(env, row['events'][1]['request']) not in literal.api, 'API holdout input appeared in fitting')
            require(physical_key(api_input(env, row['events'][1]['request'])) not in physical_inputs, 'physical API holdout input appeared in fitting')
        metrics, api_predictions = score_api(model, holdout, edges)
        require(api_predictions == saved['API_holdout_predictions'], 'API holdout distributions differ')
        controlled_metrics, controlled_predictions = score_controlled(model, faults, edges)
        require(controlled_metrics == saved['controlled_pair_metrics']
            and controlled_predictions == saved['controlled_predictions'], 'controlled paired predictions differ')
        known_physical = {physical_key(p) for p in literal.api_inputs.values()}
        for example in read(faults, 'examples.json'):
            if example['partition'] != 'eval':
                continue
            require(example['input_uid'] not in known_physical, 'controlled evaluation request entered fitting')
        measured = next(r for r in summary['table'] if r['variant'] == variant)
        require(all(measured[k] == v for k, v in metrics.items()), 'API metrics differ')
        selected, scores = choose(model, future, coalitions(pool, 2))
        require(saved['selected_bank'] == [n['uid'] for n in selected] and saved['selection_scores'] == scores, 'joint bank selection differs')
        predictions.extend({'variant': variant, 'edge_uid': e['uid'], **model.predict(e, [])} for e in test)
    require(predictions == read(output, 'predictions.json'), 'joint workflow predictions differ')
    report = {'sources_and_inputs_unchanged': True, 'full_shared_weights_and_loss_curves_reconstructed': True,
        'API_holdout_inputs_disjoint_from_fitting': len(holdout), 'physical_API_inputs_disjoint_ignoring_SDK_alias_metadata': True,
        'API_holdout_accepted_inputs': sum(r['events'][1]['accepted'] for r in holdout), 'ablations_reconstructed': len(protocol['variants']),
        'complete_output_distributions_and_retention_predictions_verified': True,
        'test_workflow_outcomes_used_for_fitting': False, 'test_portfolio_previously_examined': True,
        'new_LLM_calls': 0, 'live_API_calls': 0, 'certificate': False, 'verifier_sha256': digest(Path(__file__))}
    report.update(parameters=90, controlled_training_faults=len(controlled_training(faults)),
        controlled_eval_inputs_disjoint_from_training=True, legal_anchors_previously_seen=True)
    (output / 'integrity.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/emp_joint_v3'))
    verify(parser.parse_args().results)
