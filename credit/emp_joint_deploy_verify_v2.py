"""Verify shared-model retention and its recorded-agent deployment."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .emp_binding import BindingEnvironment
from .emp_binding_agent import parse_request
from .emp_binding_pilot import bank_name, summarize
from .emp_binding_tools_v3 import assemble, messages, retrieve_compatible
from .emp_binding_v2_verify import read, verify_sources
from .emp_binding_verify import require
from .emp_joint_deploy_v2 import load_model
from .emp_prospective_v2 import choose, coalitions, exposure_cell, rank_revalidation, robust_value
from .emp_structural import replay_observation
from .user_data import digest, jsonl


def verify(output):
    from transformers import AutoTokenizer
    protocol = verify_sources(output); source, joint = Path(protocol['source_directory']), Path(protocol['joint_directory'])
    for path, key in [(source / 'frozen_protocol.json', 'source_protocol_sha256'), (source / 'trials.jsonl', 'source_trials_sha256'),
        (joint / 'full.json', 'joint_model_sha256'), (source / 'generations.jsonl', 'cache_sha256')]:
        require(digest(path) == protocol[key], 'deployment input changed')
    verify_sources(source); verify_sources(joint)
    for name, expected in read(output, 'evaluation_freeze.json').items():
        require(digest(output / name) == expected, 'deployment freeze changed')
    parent = read(source, 'frozen_protocol.json'); data = Path(parent['data_directory'])
    edges = {e['uid']: e for e in jsonl(data / 'edges.jsonl')}
    pool = read(source, 'memories.json'); memories = {n['uid']: n for n in pool}
    model = load_model(joint); future = [edges[u] for u in parent['selection_edge_uids']]
    bank, scores = choose(model, future, coalitions(pool, 2)); configs = read(output, 'frozen_banks.json')
    require(configs['scores'] == scores and configs['banks']['CREDIT-shared-joint'] == [n['uid'] for n in bank], 'shared retention differs')
    train = [r for r in jsonl(source / 'trials.jsonl') if r['stage'] == 'intervention_train']
    observed = {exposure_cell(edges[r['edge_uid']], [memories[u] for u in r['exposure']]) for r in train}
    require(rank_revalidation(model, future, coalitions(pool, 2), observed) == read(output, 'revalidation_schedule.json'), 'shared VOI differs')
    cache = {r['prompt_sha256']: r for r in jsonl(source / 'generations.jsonl')}
    agent = read(output, 'agent_manifest.json'); tokenizer = AutoTokenizer.from_pretrained(agent['model_path'], local_files_only=True)
    rows = list(jsonl(output / 'trials.jsonl')); seen = set()
    for row in rows:
        identity = row['edge_uid'], row['bank_uid']; require(identity not in seen, 'duplicate deployment'); seen.add(identity)
        bank = [memories[u] for u in configs['banks'][row['bank_uid']]]
        exposure = retrieve_compatible(edges[row['edge_uid']], bank)
        require([n['uid'] for n in exposure] == row['exposure'], 'deployment exposure differs')
        env = BindingEnvironment(edges[row['edge_uid']]); env.call(env.producer_request())
        require(env.checkpoint() == row['checkpoint'], 'deployment prefix differs')
        for g in row['generations']:
            chat = tokenizer.apply_chat_template(messages(env, exposure), tokenize=False, add_generation_prompt=True)
            key = sha256(chat.encode()).hexdigest(); require(key == g['prompt_sha256'] and key in cache, 'deployment prompt differs')
            record = cache[key]
            require(tokenizer.decode(record['output_token_ids'], skip_special_tokens=True) == record['agent_output'], 'deployment tokens differ')
            action, error = parse_request(record['agent_output']); request, errors = assemble(env, action)
            replay_observation(env, request, record['agent_output'], error, errors)
        require(env.history == row['events'] and env.verify() == row['final'], 'deployment state/result differs')
    require(len(rows) == len(protocol['test_edge_uids']) * len(configs['banks']), 'deployment pairing differs')
    summary = read(output, 'summary.json'); require(summarize(rows) == summary['main'], 'deployment table differs')
    diagnostic = read(output, 'post_selection_diagnostic.json'); by_cell = {(r['edge_uid'], r['bank_uid']): r for r in train}
    objectives = []
    for bank in coalitions(pool, 2):
        subset = [by_cell[e['uid'], bank_name(bank)] for e in future]
        value = robust_value([float(r['final']['binding_contract_success']) - .02 * r['consumer_calls'] for r in subset], future, bank)
        objectives.append({'bank_uid': bank_name(bank), 'measured_development_objective': value})
    require(objectives == diagnostic['objectives'] and not diagnostic['selection_changed_after_diagnostic'], 'post-selection diagnostic differs')
    selected = next(r['measured_development_objective'] for r in objectives if r['bank_uid'] == diagnostic['selected_bank_uid'])
    require(max(r['measured_development_objective'] for r in objectives) - selected == summary['measured_development_selection_regret'], 'selection regret differs')
    require(read(output, 'frozen_predictions.json') == [{'edge_uid': u, **model.predict(edges[u], [])} for u in protocol['test_edge_uids']], 'deployment predictions differ')
    report = {'frozen_weights_and_sources_verified': True, 'shared_selection_and_VOI_reconstructed': True,
        'recorded_Qwen_deployment_trials_replayed': len(rows), 'local_calls_verified': sum(len(r['events']) for r in rows),
        'post_selection_regret_verified': True, 'new_LLM_calls': 0, 'live_API_calls': 0,
        'business_goal_success': None, 'certificate': False, 'verifier_sha256': digest(Path(__file__))}
    (output / 'integrity.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/emp_joint_deploy_v2'))
    verify(parser.parse_args().results)
