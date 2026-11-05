"""Reconstruct every frozen EMP pilot prompt, SDK payload and local replay."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from .emp_binding import BindingEnvironment
from .emp_binding_agent import messages, parse_request
from .emp_binding_sdk import assemble, messages as sdk_messages
from .emp_binding_pilot import all_banks, bank_name, retrieve, split_edges, summarize
from .user_data import digest, jsonl


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(output):
    from transformers import AutoTokenizer
    protocol = json.loads((output / 'frozen_protocol.json').read_text())
    data = Path(protocol['data_directory'])
    require(digest(data / 'manifest.json') == protocol['data_manifest_sha256'], 'data manifest changed')
    manifest = json.loads((data / 'manifest.json').read_text())
    root = Path(manifest['source_root'])
    for name, expected in manifest['sources_sha256'].items():
        require(digest(root / name) == expected, 'source data changed: ' + name)
    for name, expected in manifest['canonical_sha256'].items():
        require(digest(data / name) == expected, 'canonical data changed: ' + name)
    for name, expected in protocol['source_sha256'].items():
        require(digest(Path(__file__).with_name(name)) == expected, 'execution code changed: ' + name)
        require(digest(output / 'source' / name) == expected, 'frozen source changed: ' + name)
    edges = {e['uid']: e for e in jsonl(data / 'edges.jsonl')}
    development = [e for e in edges.values() if e['eligible'] and e['partition'] == 'development']
    require(split_edges(development) == protocol['splits'], 'split reconstruction mismatch')
    pool = json.loads((output / 'memories.json').read_text())
    memories = {m['uid']: m for m in pool}
    bank_config = json.loads((output / 'frozen_banks.json').read_text())['banks']
    for uids in bank_config.values():
        require(len(uids) <= 2 and all(uid in memories for uid in uids), 'storage cap violated')
        require(sum(memories[uid]['tokens'] for uid in uids) <= 1024, 'storage token cap violated')
    agent_manifest = json.loads((output / 'agent_manifest.json').read_text())
    from .emp_binding_sdk import SYSTEM as sdk_system
    from .emp_binding_agent import SYSTEM as full_system
    require(agent_manifest['system_prompt'] == (sdk_system if protocol['binding_interface'] == 'sdk' else full_system), 'system prompt metadata mismatch')
    tokenizer = AutoTokenizer.from_pretrained(agent_manifest['model_path'], local_files_only=True)
    cache = {}
    for row in jsonl(output / 'generations.jsonl'):
        key = row['prompt_sha256']
        require(key not in cache, 'duplicate cached generation')
        require(sha256(row['messages_rendered'].encode()).hexdigest() == key, 'prompt cache digest mismatch')
        require(tokenizer.decode(row['output_token_ids'], skip_special_tokens=True) == row['agent_output'], 'saved tokens disagree with output')
        require(len(tokenizer.encode(row['messages_rendered'])) == row['input_tokens'], 'input token count mismatch')
        cache[key] = row
    trials = list(jsonl(output / 'trials.jsonl'))
    seen, prompt_hits, replayed_calls = set(), Counter(), 0
    for row in trials:
        identity = row['stage'], row['edge_uid'], row['bank_uid']
        require(identity not in seen, 'duplicate trial')
        seen.add(identity)
        edge = edges[row['edge_uid']]
        require(edge['partition'] == 'development' and edge['eligible'], 'reserved or unknown edge executed')
        partition = protocol['splits'][edge['uid']]['partition']
        permitted = {'history': 'history_train', 'intervention_train': 'history_train',
                     'validation': 'validation', 'development_evaluation': 'development_evaluation'}
        require(permitted[row['stage']] == partition, 'trial crossed partition')
        exposure = [memories[uid] for uid in row['exposure']]
        require(len(exposure) <= 2, 'retrieval cap violated')
        if row['stage'] == 'history':
            require(not exposure, 'history is not NoMem')
        else:
            banks = all_banks(pool)
            bank = [memories[uid] for uid in bank_config[row['bank_uid']]] if row['stage'] == 'development_evaluation' else next(b for b in banks if bank_name(b) == row['bank_uid'])
            require(retrieve(edge, bank) == exposure, 'exposure differs from fixed retriever')
        env = BindingEnvironment(edge)
        env.call(env.producer_request())
        require(env.history[0] == row['events'][0], 'producer replay mismatch')
        require(env.checkpoint() == row['checkpoint'] and env.state_hash() == row['prefix_sha256'], 'checkpoint mismatch')
        require(len(row['generations']) == len(row['events']) - 1 <= 2, 'call budget mismatch')
        for index, generation in enumerate(row['generations'], start=1):
            builder = sdk_messages if row.get('binding_interface', 'full_json') == 'sdk' else messages
            chat = tokenizer.apply_chat_template(builder(env, exposure), tokenize=False, add_generation_prompt=True)
            key = sha256(chat.encode()).hexdigest()
            require(key == generation['prompt_sha256'] and key in cache, 'reconstructed public prompt differs')
            record = cache[key]; prompt_hits[key] += 1
            action, error = parse_request(record['agent_output'])
            request, sdk_errors = assemble(env, action) if row.get('binding_interface') == 'sdk' else (action, [])
            require(error == generation['parse_error'], 'parse status differs')
            require(sdk_errors == generation.get('sdk_errors', []), 'SDK assembly differs')
            env.call(request)
            event = env.history[-1]
            if sdk_errors:
                event['response']['sdk_errors'] = sdk_errors
            if error:
                event['response']['parse_error'] = error
            event['agent_output'] = record['agent_output']
            event['state_after'] = env.state_hash()
            require(event == row['events'][index], 'request/response/state replay mismatch')
        require(env.verify() == row['final'], 'final verification mismatch')
        require(row['first_try_success'] == env.history[1]['accepted'], 'first-call label mismatch')
        category = 'accepted' if env.history[1]['accepted'] else 'malformed' if row['generations'][0]['parse_error'] else 'contract_error'
        require(category == row['first_category'], 'action category mismatch')
        require(row['business_goal_success'] is None, 'unknown business goal labeled')
        replayed_calls += len(env.history)
    require(set(cache) == set(prompt_hits), 'orphan or missing generation cache')
    demonstrations = list(jsonl(output / 'assisted_history.jsonl'))
    for row in demonstrations:
        require(protocol['splits'][row['edge_uid']]['partition'] == 'history_train', 'reference demo crossed partition')
        env = BindingEnvironment(edges[row['edge_uid']])
        for event in row['events']:
            require(env.call(event['request']) == event, 'reference demonstration mismatch')
        require(env.verify() == row['final'], 'reference demonstration label mismatch')
    history = trials + demonstrations
    for note in pool:
        origins = [r for r in history if r['edge_uid'] == note['source_edge'] and r['stage'] == note['source_stage'] and r['bank_uid'] == note['source_trial_bank']]
        require(len(origins) == 1 and origins[0]['final']['binding_contract_success'], 'note lacks verified training origin')
        accepted = next(e for e in reversed(origins[0]['events']) if e['accepted'])
        require(accepted['request']['bindings'] == note['verified_mapping'], 'memory mapping differs from origin')
        require(note['tokens'] == len(tokenizer.encode(note['text'], add_special_tokens=False)) <= 512, 'note token limit mismatch')
    evaluation = [r for r in trials if r['stage'] == 'development_evaluation']
    saved_summary = json.loads((output / 'summary.json').read_text())
    require(summarize(evaluation) == saved_summary['main'], 'summary disagrees with raw trials')
    expected_counts = Counter()
    n_banks = len(all_banks(pool))
    for spec in protocol['splits'].values():
        if spec['partition'] == 'history_train':
            expected_counts['history'] += 1; expected_counts['intervention_train'] += n_banks
        elif spec['partition'] == 'validation':
            expected_counts['validation'] += n_banks
        else:
            expected_counts['development_evaluation'] += len(bank_config)
    require(Counter(r['stage'] for r in trials) == expected_counts, 'paired trial coverage differs')
    # Contract reference ceiling is diagnostic only, after frozen evaluations.
    reference = []
    for edge in development:
        env = BindingEnvironment(edge); env.call(env.producer_request()); env.call(env.reference_request())
        reference.append(env.verify()['binding_contract_success'])
    report = {'sources_unchanged': True, 'frozen_execution_sources_match': True,
        'public_prompts_and_output_tokens_verified': len(cache), 'reconstructed_trials': len(trials),
        'replayed_local_calls': replayed_calls, 'training_reference_demonstrations': len(demonstrations),
        'memory_origins_and_budgets_verified': True, 'paired_coverage_verified': True,
        'reserved_workflows_executed': 0, 'local_contract_reference_ceiling': sum(reference) / len(reference),
        'business_goal_reference_ceiling': None, 'live_api_calls': 0,
        'verifier_sha256': digest(Path(__file__))}
    (output / 'integrity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/emp_binding_pilot_v5'))
    args = parser.parse_args()
    verify(args.results)


if __name__ == '__main__':
    main()
