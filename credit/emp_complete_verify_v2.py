"""Verify literal kernels, controlled prefixes, attribution and frozen evaluation."""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random

from .emp_binding import BindingEnvironment
from .emp_binding_agent import parse_request
from .emp_binding_pilot import bank_name, summarize
from .emp_binding_tools_v3 import SYSTEM, assemble, messages, retrieve_compatible
from .emp_complete_v2 import forced_action
from .emp_prospective_v2 import choose, coalitions, exact_attribution, exposure_cell, rank_revalidation
from .emp_structural import StructuredMechanisms, compare_api_kernels, replay_observation
from .emp_binding_v2_verify import read, verify_sources
from .emp_binding_verify import require
from .user_data import digest, jsonl


def verify(output):
    from transformers import AutoTokenizer
    protocol = verify_sources(output)
    data, history = Path(protocol['data_directory']), Path(protocol['history_directory'])
    require(digest(data / 'manifest.json') == protocol['data_manifest_sha256'], 'data manifest changed')
    manifest = read(data, 'manifest.json')
    for name, expected in manifest['canonical_sha256'].items():
        require(digest(data / name) == expected, 'canonical input changed')
    for name, expected in manifest['sources_sha256'].items():
        require(digest(Path(manifest['source_root']) / name) == expected, 'user source changed')
    for name, key in [('memories.json', 'history_memories_sha256'), ('generations.jsonl', 'history_generations_sha256')]:
        require(digest(history / name) == protocol[key], 'historical input changed')
    pool = read(output, 'memories.json'); memories = {n['uid']: n for n in pool}
    require(pool == read(history, 'memories.json'), 'memory provenance changed')
    edges = {r['uid']: r for r in jsonl(data / 'edges.jsonl')}
    train_ids, test_ids = set(protocol['training_edge_uids']), set(protocol['test_edge_uids'])
    require(not train_ids.intersection(test_ids), 'train/test overlap')
    require(all(edges[u]['partition'] == 'development' and edges[u]['eligible'] for u in train_ids), 'training partition differs')
    require(test_ids == {e['uid'] for e in edges.values() if e['eligible'] and e['partition'] == 'reserved_unexecuted'}, 'eligible reserved test differs')
    for name, expected in read(output, 'evaluation_freeze.json').items():
        require(digest(output / name) == expected, 'frozen evaluation input changed')
    agent = read(output, 'agent_manifest.json'); require(agent['system_prompt'] == SYSTEM, 'system metadata differs')
    tokenizer = AutoTokenizer.from_pretrained(agent['model_path'], local_files_only=True)
    require(all(n['tokens'] == len(tokenizer.encode(n['text'], add_special_tokens=False)) <= 512 for n in pool), 'memory token budget differs')
    cache_directory = Path(protocol['cache_directory'])
    require(digest(cache_directory / 'generations.jsonl') == protocol['cache_generations_sha256'], 'cache source changed')
    previous_cache = {r['prompt_sha256']: r for r in jsonl(cache_directory / 'generations.jsonl')}
    cache, reused = {}, 0
    for r in jsonl(output / 'generations.jsonl'):
        key = r['prompt_sha256']; require(key not in cache, 'duplicate cache key')
        require(sha256(r['messages_rendered'].encode()).hexdigest() == key, 'prompt hash differs')
        require(tokenizer.decode(r['output_token_ids'], skip_special_tokens=True) == r['agent_output'], 'tokens differ')
        require(len(tokenizer.encode(r['messages_rendered'])) == r['input_tokens'], 'input token count differs')
        if r.get('generation_origin'):
            require(key in previous_cache and all(r[k] == previous_cache[key][k] for k in previous_cache[key] if k != 'generation_origin'), 'reused inference differs')
            reused += 1
        cache[key] = r
    rows, api_rows = list(jsonl(output / 'trials.jsonl')), list(jsonl(output / 'api_probes.jsonl'))
    hits, seen, calls = Counter(), set(), 0
    entered_test = False
    for row in rows + api_rows:
        identity = row['stage'], row['edge_uid'], row['bank_uid']
        require(identity not in seen, 'duplicate trial identity'); seen.add(identity)
        uid = row['edge_uid']; stage = row['stage']; edge = edges[uid]
        if stage == 'heldout_evaluation':
            entered_test = True; require(uid in test_ids, 'test partition differs')
        else:
            require(uid in train_ids, 'probe used reserved outcomes')
            if stage != 'api_fixture_train':
                require(not entered_test, 'training occurred after test')
        bank = [memories[u] for u in row.get('stored_bank', [])]
        exposure = [memories[u] for u in row['exposure']]
        require(exposure == retrieve_compatible(edge, bank), 'retrieval differs')
        require(len(exposure) <= 2 and sum(n['tokens'] for n in exposure) <= 1024, 'exposure budget differs')
        if stage == 'heldout_evaluation':
            config = read(output, 'frozen_banks.json')['banks'][row['bank_uid']]
            require(row['stored_bank'] == config and len(bank) <= 2, 'test bank differs')
        env = BindingEnvironment(edge); env.call(env.producer_request())
        require(env.history[0] == row['events'][0], 'producer differs')
        require(env.checkpoint() == row['checkpoint'] and env.state_hash() == row['prefix_sha256'], 'prefix restore differs')
        generations = {g['event_index']: g for g in row.get('generations', [])}
        for i, event in enumerate(row['events'][1:], 1):
            if i in row.get('forced_event_indices', []):
                if stage == 'repair_context_train':
                    action = forced_action(edge, row['forced_mode'])
                    require(event['agent_output'] == json.dumps(action, ensure_ascii=False), 'controlled action differs')
                action, error = parse_request(event['agent_output']); request, errors = assemble(env, action)
            else:
                g = generations[i]
                chat = tokenizer.apply_chat_template(messages(env, exposure), tokenize=False, add_generation_prompt=True)
                key = sha256(chat.encode()).hexdigest()
                require(key == g['prompt_sha256'] and key in cache, 'public prompt reconstruction differs')
                hits[key] += 1; action, error = parse_request(cache[key]['agent_output'])
                request, errors = assemble(env, action)
                require(error == g['parse_error'] and errors == g['sdk_errors'], 'generation errors differ')
            require(request == event['request'], 'structured request differs')
            replay_observation(env, request, event['agent_output'], error, errors)
            require(env.history[-1] == event, 'response/state transition differs')
        require(env.verify() == row['final'], 'verifier outcome differs')
        require(len(env.history) - 1 <= 2, 'consumer budget violated')
        calls += len(env.history)
    require(set(cache) == set(hits), 'unused cache record')
    require(reused == agent['reused_generation_contexts'] and len(cache) - reused == agent['new_generation_contexts'], 'inference accounting differs')
    require(sum(r['input_tokens'] for r in cache.values()) == agent['generation_input_tokens_total'], 'total input tokens differ')
    require(sum(r['output_tokens'] for r in cache.values()) == agent['generation_output_tokens_total'], 'total output tokens differ')
    full, feasible = coalitions(pool), coalitions(pool, 2)
    train = [r for r in rows if r['stage'] == 'intervention_train']
    expected_assignments, rng = {}, random.Random(protocol['randomization_seed'])
    for uid in protocol['training_edge_uids']:
        order = list(full); rng.shuffle(order)
        for i, bank in enumerate(order):
            expected_assignments[uid, bank_name(bank)] = {'scheme': 'random permutation without replacement',
                'position': i, 'remaining_candidates': len(full) - i, 'conditional_probability': 1 / (len(full) - i)}
    require({(r['edge_uid'], r['bank_uid']): r['assignment'] for r in train} == expected_assignments, 'exposure assignment probabilities differ')
    fitting = [r for r in rows if r['stage'] in ('intervention_train', 'repair_context_train')]
    before = StructuredMechanisms().update(fitting + api_rows, edges, memories)
    require(before.to_dict() == read(output, 'model_before.json'), 'historical kernels reconstruction differs')
    # Every observed deterministic exposure must compose the literal kernels to
    # its recorded final result, without invoking the verifier during prediction.
    for row in train:
        p = before.predict(edges[row['edge_uid']], [memories[u] for u in row['exposure']])
        require(p['supported'] and p['final_success'] == float(row['final']['binding_contract_success']), 'structured training rollout differs')
    future = [edges[u] for u in protocol['selection_edge_uids']]
    initial, initial_scores = choose(before, future, feasible)
    observed = {exposure_cell(edges[r['edge_uid']], [memories[u] for u in r['exposure']]) for r in train}
    after = StructuredMechanisms.from_dict(before.to_dict()); probes = [r for r in rows if r['stage'] == 'revalidation']
    for step, round_ in enumerate(read(output, 'revalidation_schedule.json')):
        ranking = rank_revalidation(after, future, feasible, observed, seed=42 + step,
            particles=protocol['VOI']['particles'], probe_cost=protocol['VOI']['probe_cost'])
        choice = ranking[0] if ranking and ranking[0]['voi'] > 0 else None
        require(round_ == {'round': step, 'rankings': ranking, 'chosen': choice}, 'VOI schedule differs')
        if choice:
            row = next(r for r in probes if r['bank_uid'] == 'VOI:' + choice['cell_uid'])
            after.update([row], edges, memories); observed.add(choice['cell_uid'])
    require(after.to_dict() == read(output, 'model_after.json'), 'updated kernels differ')
    selected, final_scores = choose(after, future, feasible); bank_file = read(output, 'frozen_banks.json')
    require(bank_file['initial_scores'] == initial_scores and bank_file['final_scores'] == final_scores, 'robust objective differs')
    require(bank_file['banks']['CREDIT-structured'] == [n['uid'] for n in selected], 'selected bank differs')
    require(bank_file['banks']['CREDIT-no-revalidation'] == [n['uid'] for n in initial], 'initial bank differs')
    require(exact_attribution(train, pool, full) == bank_file['attribution'], 'full LOO/Shapley differs')
    predictions = read(output, 'frozen_predictions.json')
    for row in predictions:
        bank = [memories[u] for u in bank_file['banks'][row['method']]]
        exposure = retrieve_compatible(edges[row['edge_uid']], bank)
        model = before if row['method'] == 'CREDIT-no-revalidation' else after
        require(row == {'edge_uid': row['edge_uid'], 'method': row['method'], 'exposure': [n['uid'] for n in exposure], **model.predict(edges[row['edge_uid']], exposure)}, 'frozen prediction differs')
    evaluation = [r for r in rows if r['stage'] == 'heldout_evaluation']
    require(len(evaluation) == len(test_ids) * len(bank_file['banks']), 'paired test coverage differs')
    summary = read(output, 'summary.json'); require(summarize(evaluation) == summary['main'], 'main table differs')
    require(Counter(r['stage'] for r in rows) == Counter(summary['trial_counts']), 'trial counts differ')
    require(summary['api_probe_trials'] == len(api_rows), 'API probe cost differs')
    diagnostic = read(output, 'injected_drift_diagnostic.json')
    source = StructuredMechanisms().update([diagnostic['source_trial']], edges, memories)
    changed_edges = deepcopy(edges); changed_edges[diagnostic['target_trial']['edge_uid']]['fixture_version'] = diagnostic['target_fixture_version']
    target = StructuredMechanisms().update([diagnostic['target_trial']], changed_edges, memories)
    require(all(diagnostic[k] == v for k, v in compare_api_kernels(source, target).items()), 'injected drift diagnostic differs')
    report = {'frozen_sources_and_user_data_verified': True, 'rendered_prompts_and_tokens_verified': len(cache),
        'reused_contexts_verified': reused, 'new_contexts_verified': len(cache) - reused,
        'trial_and_API_probe_replays': len(rows) + len(api_rows), 'replayed_local_calls': calls,
        'controlled_prefixes_and_assignments_verified': True, 'literal_policy_API_and_state_kernels_reconstructed': True,
        'full_coalition_attribution_and_robust_selection_verified': True, 'VOI_schedule_verified': True,
        'reserved_eligible_test_tasks': len(test_ids), 'test_outcomes_used_for_fitting': False,
        'injected_drift_trace_replayed': True, 'live_api_calls': 0, 'business_goal_success': None,
        'certificate': False, 'verifier_sha256': digest(Path(__file__))}
    (output / 'integrity.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/emp_complete_v2'))
    verify(parser.parse_args().results)
