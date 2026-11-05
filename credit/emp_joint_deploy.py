"""Connect the shared learner, robust retention, VOI and frozen Qwen executor.

This is a paired development deployment on previously examined local tasks.
All action generations are exact recorded-prompt replays, never new evidence.
"""

import argparse
import json
from pathlib import Path

from .emp_binding_pilot import bank_name, summarize
from .emp_complete_v2 import ReplayAgent, run, write
from .emp_joint_v2 import JointStructuredMechanisms
from .emp_prospective_v2 import choose, coalitions, exposure_cell, rank_revalidation, robust_value
from .emp_structural import StructuredMechanisms
from .user_data import digest, jsonl


def load_model(directory):
    saved = json.loads((directory / 'full.json').read_text())
    literal = StructuredMechanisms.from_dict(saved['literal']); weights = saved['model']
    return JointStructuredMechanisms(literal, weights['action_catalog'], weights['policy_weights'], weights['api_weights'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/emp_complete_v2'))
    parser.add_argument('--joint', type=Path, default=Path('results/emp_joint_v2'))
    parser.add_argument('--output', type=Path, default=Path('results/emp_joint_deploy_v1'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    source_protocol = json.loads((args.source / 'frozen_protocol.json').read_text())
    data = Path(source_protocol['data_directory'])
    edges = {e['uid']: e for e in jsonl(data / 'edges.jsonl')}
    pool = json.loads((args.source / 'memories.json').read_text()); memories = {m['uid']: m for m in pool}
    model = load_model(args.joint); future = [edges[u] for u in source_protocol['selection_edge_uids']]
    feasible = coalitions(pool, 2)
    selected, scores = choose(model, future, feasible)
    train = [r for r in jsonl(args.source / 'trials.jsonl') if r['stage'] == 'intervention_train']
    observed = {exposure_cell(edges[r['edge_uid']], [memories[u] for u in r['exposure']]) for r in train}
    schedule = rank_revalidation(model, future, feasible, observed)
    if any(r['voi'] > 0 for r in schedule):
        raise ValueError('new revalidation is required; recorded-only deployment cannot substitute a hypothetical observation')
    configs = json.loads((args.source / 'frozen_banks.json').read_text())['banks']
    configs = {k: v for k, v in configs.items() if k != 'CREDIT-no-revalidation'}
    configs['CREDIT-shared-joint'] = [n['uid'] for n in selected]
    args.output.mkdir(parents=True); (args.output / 'source').mkdir()
    names = ['emp_joint_deploy.py', 'emp_joint_v2.py', 'emp_structural.py', 'emp_prospective_v2.py', 'emp_prospective.py',
        'emp_complete_v2.py', 'emp_binding.py', 'emp_binding_tools_v3.py', 'emp_binding_tools_v2.py',
        'emp_binding_sdk.py', 'emp_binding_agent.py', 'emp_binding_pilot.py', 'emp_binding_shared.py', 'policy.py', 'user_data.py']
    protocol = {'scope': __doc__, 'source_directory': str(args.source.resolve()), 'joint_directory': str(args.joint.resolve()),
        'source_protocol_sha256': digest(args.source / 'frozen_protocol.json'),
        'source_trials_sha256': digest(args.source / 'trials.jsonl'),
        'joint_model_sha256': digest(args.joint / 'full.json'), 'cache_sha256': digest(args.source / 'generations.jsonl'),
        'source_sha256': {n: digest(Path(__file__).with_name(n)) for n in names},
        'test_outcomes_used_for_bank_selection': False, 'test_portfolio_previously_examined': True,
        'test_edge_uids': source_protocol['test_edge_uids'], 'new_LLM_calls': 0, 'live_API_calls': 0,
        'certificate': False, 'business_goal_success': None}
    write(args.output, 'frozen_protocol.json', protocol)
    for name in names:
        (args.output / 'source' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write(args.output, 'frozen_banks.json', {'banks': configs, 'scores': scores})
    write(args.output, 'revalidation_schedule.json', schedule)
    predictions = [{'edge_uid': uid, **model.predict(edges[uid], [])} for uid in protocol['test_edge_uids']]
    write(args.output, 'frozen_predictions.json', predictions)
    write(args.output, 'evaluation_freeze.json', {n: digest(args.output / n) for n in
        ['frozen_protocol.json', 'frozen_banks.json', 'revalidation_schedule.json', 'frozen_predictions.json']})
    agent = ReplayAgent(args.output, args.source)
    evaluation = run(agent, [{'edge': edges[uid], 'bank': [memories[u] for u in bank], 'method': method}
        for uid in protocol['test_edge_uids'] for method, bank in configs.items()], args.output, 'heldout_evaluation')
    agent.close()
    # Post-selection diagnostic only: compare model selection to already
    # measured training-bank utility. This never changes weights or chosen bank.
    by_cell = {(r['edge_uid'], r['bank_uid']): r for r in train}
    objectives = []
    for bank in feasible:
        rows = [by_cell[e['uid'], bank_name(bank)] for e in future]
        value = robust_value([float(r['final']['binding_contract_success']) - .02 * r['consumer_calls'] for r in rows], future, bank)
        objectives.append({'bank_uid': bank_name(bank), 'measured_development_objective': value})
    best = max(r['measured_development_objective'] for r in objectives)
    chosen_value = next(r['measured_development_objective'] for r in objectives if r['bank_uid'] == bank_name(selected))
    write(args.output, 'post_selection_diagnostic.json', {'objectives': objectives,
        'selected_bank_uid': bank_name(selected), 'measured_development_regret': best - chosen_value,
        'selection_changed_after_diagnostic': False})
    write(args.output, 'summary.json', {'main': summarize(evaluation), 'trials': len(evaluation),
        'local_calls': sum(len(r['events']) for r in evaluation),
        'selected_shared_bank': configs['CREDIT-shared-joint'], 'revalidation_probes_executed': 0,
        'test_prediction_coverage': sum(p['supported'] for p in predictions) / len(predictions),
        'measured_development_selection_regret': best - chosen_value,
        'scope': 'shared finite learner integrated with selection and recorded Qwen execution; development comparison',
        'new_LLM_calls': 0, 'live_API_calls': 0, 'business_goal_success': None})
    print(json.dumps(json.loads((args.output / 'summary.json').read_text()), indent=2))


if __name__ == '__main__':
    main()
