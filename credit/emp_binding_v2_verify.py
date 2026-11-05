"""Verify SDK v2 artifacts by reconstructing prompts, contracts and predictions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from .emp_binding import BindingEnvironment
from .emp_binding_agent import parse_request
from .emp_binding_pilot import all_banks, bank_name, memories_from_history, summarize
from .emp_binding_shared import SharedBindingMechanism, distinct_trials, score_predictions, tune_shared
from .emp_binding_tools_v2 import SYSTEM, assemble, messages, retrieve_compatible
from .emp_binding_v2 import select
from .emp_binding_verify import require
from .user_data import digest, jsonl


def read(directory, name):
    return json.loads((directory / name).read_text())


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


def load_model(directory, pool):
    saved = read(directory, 'model.json')
    profiles = saved['training_profiles']
    for item in profiles:
        item['profile']['skeleton'] = tuples(item['profile']['skeleton'])
    return SharedBindingMechanism(saved['weights'], saved['mean'], saved['scale'], pool, profiles)


def verify_sources(directory):
    protocol = read(directory, 'frozen_protocol.json')
    for name, expected in protocol['source_sha256'].items():
        require(digest(Path(__file__).with_name(name)) == expected, 'execution source changed: ' + name)
        require(digest(directory / 'source' / name) == expected, 'frozen source changed: ' + name)
    return protocol


def verify_cv(directory, training, edges, pool):
    cv = read(directory, 'grouped_cv.json')
    groups = {r['workflow_group'] for r in training}
    for candidate in cv:
        require({f['heldout_group'] for f in candidate['folds']} == groups, 'CV groups differ')
        for fold in candidate['folds']:
            blocked = {m['uid'] for m in pool if edges[m['source_edge']]['workflow_group'] == fold['heldout_group']}
            require(sorted(blocked) == fold['excluded_origin_notes'], 'CV note-origin exclusion differs')
            eligible = [r for r in training if r['workflow_group'] == fold['heldout_group'] and not blocked.intersection(r['exposure'])]
            require(len(distinct_trials(eligible)[0]) == fold['rows'], 'CV heldout coverage differs')
        for source, target in [('final_nll', 'group_mean_nll'), ('brier', 'group_mean_brier')]:
            require(abs(sum(f[source] for f in candidate['folds']) / len(groups) - candidate[target]) < 1e-12, 'CV aggregate differs')
    chosen = min(cv, key=lambda c: (c['group_mean_nll'], c['group_mean_brier'], -c['ridge']))
    fit = read(directory, 'model.json')['fit']
    require((fit['ridge'], fit['outcome_weight']) == (chosen['ridge'], chosen['outcome_weight']), 'CV selection differs')
    require(fit['distinct_exposure_rows'] == len(distinct_trials(training)[0]), 'training duplicate count differs')
    model, rebuilt_fit, rebuilt_cv = tune_shared(training, edges, pool)
    require(rebuilt_cv == cv and rebuilt_fit == fit, 'train-only CV reconstruction differs')
    saved = read(directory, 'model.json')
    require(model.weights.tolist() == saved['weights'] and model.mean.tolist() == saved['mean'] and model.scale.tolist() == saved['scale'], 'train-only weights reconstruction differs')


def verify(directory):
    from transformers import AutoTokenizer
    protocol = verify_sources(directory)
    system, message_builder = SYSTEM, messages
    credit_method = 'CREDIT-binding-v2'
    if protocol['tool_version'] == 'sdk3':
        from .emp_binding_tools_v3 import SYSTEM as system, messages as message_builder
        credit_method = 'CREDIT-binding-v3'
    baseline, expanded = Path(protocol['baseline_directory']), Path(protocol['expanded_directory'])
    require(digest(baseline / 'frozen_protocol.json') == protocol['baseline_protocol_sha256'], 'baseline protocol changed')
    require(digest(expanded / 'manifest.json') == protocol['expansion_manifest_sha256'], 'expansion manifest changed')
    manifest = read(expanded, 'manifest.json')
    for name, expected in manifest['sources_sha256'].items():
        require(digest(Path(manifest['source_root']) / name) == expected, 'user data changed')
    for name, expected in manifest['canonical_sha256'].items():
        require(digest(expanded / name) == expected, 'expanded data changed')
    require(digest(Path(__file__).with_name('emp_binding_shared.py')) == manifest['field_normalization_sha256'], 'field normalization changed')
    original = {e['uid']: e for e in jsonl(Path(manifest['source_data_directory']) / 'edges.jsonl')}
    require(digest(Path(manifest['source_data_directory']) / 'edges.jsonl') == manifest['source_edges_sha256'], 'original edges changed')
    edges = {e['uid']: e for e in jsonl(expanded / 'edges.jsonl')}
    require(all(edges[k] == v for k, v in original.items() if v['partition'] != 'development'), 'reserved edges changed')
    pool = read(directory, 'memories.json'); memories = {m['uid']: m for m in pool}
    bank_file = read(directory, 'frozen_banks.json'); configs = bank_file['banks']
    banks = all_banks(pool)
    for uids in configs.values():
        require(len(uids) <= 2 and sum(memories[u]['tokens'] for u in uids) <= 1024, 'storage budget differs')
    agent = read(directory, 'agent_manifest.json')
    require(agent['system_prompt'] == system and agent['binding_interface'] == protocol['tool_version'], 'agent interface metadata differs')
    tokenizer = AutoTokenizer.from_pretrained(agent['model_path'], local_files_only=True)
    cache = {}
    for record in jsonl(directory / 'generations.jsonl'):
        key = record['prompt_sha256']
        require(key not in cache, 'duplicate generation')
        require(sha256(record['messages_rendered'].encode()).hexdigest() == key, 'prompt hash differs')
        require(tokenizer.decode(record['output_token_ids'], skip_special_tokens=True) == record['agent_output'], 'output tokens differ')
        require(len(tokenizer.encode(record['messages_rendered'])) == record['input_tokens'], 'input tokens differ')
        cache[key] = record
    trials = list(jsonl(directory / 'trials.jsonl')); seen, hits, calls = set(), Counter(), 0
    for row in trials:
        identity = row['stage'], row['edge_uid'], row['bank_uid']
        require(identity not in seen, 'duplicate trial'); seen.add(identity)
        edge = edges[row['edge_uid']]
        require(edge['partition'] == 'development' and edge['eligible'], 'reserved or ineligible edge executed')
        if row['stage'] == 'additional_development':
            require(edge['uid'] in protocol['additional_edge_uids'], 'unknown additional edge')
        else:
            permitted = {'history': 'history_train', 'intervention_train': 'history_train', 'validation': 'validation', 'development_evaluation': 'development_evaluation'}
            require(protocol['splits'][edge['uid']]['partition'] == permitted[row['stage']], 'partition differs')
        exposure = [memories[u] for u in row['exposure']]
        if row['stage'] == 'history':
            require(not exposure, 'history exposed memory')
        else:
            bank = [memories[u] for u in configs[row['bank_uid']]] if row['stage'] in ('development_evaluation', 'additional_development') else next(b for b in banks if bank_name(b) == row['bank_uid'])
            require(exposure == retrieve_compatible(edge, bank), 'compatible retrieval differs')
        env = BindingEnvironment(edge); env.call(env.producer_request())
        require(env.history[0] == row['events'][0], 'producer replay differs')
        require(env.checkpoint() == row['checkpoint'] and env.state_hash() == row['prefix_sha256'], 'checkpoint differs')
        require(len(row['generations']) == len(row['events']) - 1 <= 2, 'call budget differs')
        for index, generation in enumerate(row['generations'], 1):
            chat = tokenizer.apply_chat_template(message_builder(env, exposure), tokenize=False, add_generation_prompt=True)
            key = sha256(chat.encode()).hexdigest()
            require(key == generation['prompt_sha256'] and key in cache, 'public prompt reconstruction differs')
            record = cache[key]; hits[key] += 1
            action, error = parse_request(record['agent_output']); request, sdk_errors = assemble(env, action)
            require(error == generation['parse_error'] and sdk_errors == generation['sdk_errors'], 'parse/SDK errors differ')
            env.call(request); event = env.history[-1]
            if error:
                event['response']['parse_error'] = error
            if sdk_errors:
                event['response']['sdk_errors'] = sdk_errors
            event['agent_output'] = record['agent_output']; event['state_after'] = env.state_hash()
            require(event == row['events'][index], 'request/response/state replay differs')
        require(env.verify() == row['final'], 'final contract label differs')
        require(row['first_try_success'] == env.history[1]['accepted'], 'first success differs')
        category = 'accepted' if env.history[1]['accepted'] else 'malformed' if row['generations'][0]['parse_error'] else 'contract_error'
        require(category == row['first_category'], 'first category differs')
        require(row['consumer_calls'] == len(env.history) - 1 and row['business_goal_success'] is None, 'call count/business label differs')
        calls += len(env.history)
    require(set(cache) == set(hits), 'orphan generation')
    demos = list(jsonl(directory / 'assisted_history.jsonl'))
    for row in demos:
        require(protocol['splits'][row['edge_uid']]['partition'] == 'history_train', 'demo crossed partition')
        env = BindingEnvironment(edges[row['edge_uid']])
        for event in row['events']:
            require(env.call(event['request']) == event, 'reference demo replay differs')
        require(env.verify() == row['final'], 'demo label differs')
    history = [r for r in trials if r['stage'] == 'history']
    require(memories_from_history(history + demos, edges, tokenizer) == pool, 'memory origins/text/budgets differ')
    counts = Counter()
    for spec in protocol['splits'].values():
        part = spec['partition']
        if part == 'history_train':
            counts['history'] += 1; counts['intervention_train'] += len(banks)
        else:
            counts[part] += len(banks) if part == 'validation' else len(configs)
    counts['additional_development'] = len(protocol['additional_edge_uids']) * len(configs)
    require(Counter(r['stage'] for r in trials) == counts, 'paired coverage differs')
    training = [r for r in trials if r['stage'] == 'intervention_train']
    verify_cv(directory, training, edges, pool); model = load_model(directory, pool)
    future = [edges[k] for k, v in protocol['splits'].items() if v['partition'] == 'validation'] + [edges[k] for k in protocol['additional_edge_uids']]
    chosen, scores = select(model, future, banks)
    require(scores == bank_file['scores'] and [m['uid'] for m in chosen] == configs[credit_method], 'public-context bank selection differs')
    predictions = read(directory, 'frozen_predictions.json'); prediction_map = {}
    for row in predictions:
        bank = [memories[u] for u in configs[row['bank_uid']]] if row['stage'] != 'validation' else next(b for b in banks if bank_name(b) == row['bank_uid'])
        exposure = retrieve_compatible(edges[row['edge_uid']], bank)
        expected = {'stage': row['stage'], 'edge_uid': row['edge_uid'], 'bank_uid': row['bank_uid'], 'exposure': [m['uid'] for m in exposure], **model.predict(edges[row['edge_uid']], exposure)}
        require(expected == row, 'saved mechanism prediction differs')
        prediction_map[row['stage'], row['edge_uid'], row['bank_uid']] = row
    require(set(prediction_map) == {i for i in seen if i[0] not in ('history', 'intervention_train')}, 'prediction coverage differs')
    summary = read(directory, 'summary.json'); metrics = []
    for stage in ('validation', 'development_evaluation', 'additional_development'):
        subset = [r for r in trials if r['stage'] == stage]
        metrics.append({'stage': stage, **score_predictions(subset, lambda r: prediction_map[r['stage'], r['edge_uid'], r['bank_uid']])})
        if stage != 'validation':
            require(summarize(subset) == summary['main' if stage == 'development_evaluation' else 'additional'], 'success summary differs')
    require(metrics == summary['prediction_coverage'], 'calibration summary differs')
    require(summary['history_success'] == sum(r['final']['binding_contract_success'] for r in history), 'history summary differs')
    report = {'sources_unchanged': True, 'frozen_execution_sources_match': True, 'public_prompts_and_output_tokens_verified': len(cache),
        'reconstructed_trials': len(trials), 'replayed_local_calls': calls, 'training_reference_demonstrations': len(demos),
        'memory_origins_and_budgets_verified': True, 'paired_coverage_verified': True, 'saved_predictions_and_selection_verified': True,
        'CV_origin_exclusions_and_selection_verified': True, 'train_only_CV_and_weights_reconstructed': True, 'reserved_workflows_executed': 0, 'live_api_calls': 0,
        'business_goal_success': None, 'verifier_sha256': digest(Path(__file__))}
    (directory / 'integrity.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/emp_binding_v2_run1'))
    verify(parser.parse_args().results)
