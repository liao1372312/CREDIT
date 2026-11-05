"""Unified finite EMP experiment: structured kernels, retention and revalidation.

All previously inspected development edges are training fixtures in this new
protocol. Two originally reserved eligible edges are evaluated only after the
protocol, kernels, banks and predictions freeze. Business goals remain unknown.
"""

import argparse
from collections import Counter
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import random
from time import perf_counter

from .emp_binding import BindingEnvironment, leaves, pointer
from .emp_binding_agent import parse_request
from .emp_binding_pilot import MODEL, bank_name, summarize
from .emp_binding_tools_v3 import SYSTEM, assemble, messages, retrieve_compatible
from .emp_binding_v3 import QwenToolAgent
from .emp_prospective_v2 import choose, coalitions, exact_attribution, exposure_cell, rank_revalidation
from .emp_structural import StructuredMechanisms, compare_api_kernels, environment, replay_observation
from .user_data import digest, jsonl


def write(directory, name, value):
    (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


class ReplayAgent:
    def __init__(self, output, source):
        from transformers import AutoTokenizer
        self.output, self.source = output, source
        self.manifest = json.loads((source / 'agent_manifest.json').read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(self.manifest['model_path'], local_files_only=True)
        self.previous = {r['prompt_sha256']: r for r in jsonl(source / 'generations.jsonl')}
        self.cache = {}

    def infer(self, chats):
        from hashlib import sha256
        results = []
        for chat in chats:
            rendered = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            key = sha256(rendered.encode()).hexdigest()
            if key not in self.previous:
                raise ValueError('new inference required; use a fresh Qwen protocol')
            if key not in self.cache:
                record = deepcopy(self.previous[key])
                record['generation_origin'] = 'exact previous prompt: ' + str(self.source.resolve())
                self.cache[key] = record
                with (self.output / 'generations.jsonl').open('a') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            results.append(deepcopy(self.cache[key]))
        return results

    def close(self):
        self.manifest.update(system_prompt=SYSTEM, input_tokens=0, output_tokens=0, cuda_peak_bytes=0,
            new_generation_contexts=0, reused_generation_contexts=len(self.cache), unique_generation_contexts=len(self.cache),
            generation_input_tokens_total=sum(r['input_tokens'] for r in self.cache.values()),
            generation_output_tokens_total=sum(r['output_tokens'] for r in self.cache.values()),
            cache='exact recorded-prompt replay; no new LLM inference')
        write(self.output, 'agent_manifest.json', self.manifest)


def forced_action(edge, mode):
    targets = [b['target'] for b in edge['bindings']]
    available = [pointer(p) for p, _ in leaves(edge['producer_catalog']['response_example'])]
    # No private expected source is consulted: choose a public first leaf, or an
    # explicitly invalid pointer, to collect controlled repair contexts.
    route = sorted(available)[0] if mode == 'first_public_leaf' else '/fields_to_bind/0'
    return {'operation': 'bind_consumer', 'api_id': edge['consumer_api'],
            'bindings': {target: {'operation': 'producer', 'pointer': route} for target in targets}}


def run(agent, jobs, output, stage):
    """Jobs carry optional controlled prefix; it is excluded from policy fitting."""
    active, done = [], []
    for job in jobs:
        edge, bank = job['edge'], job['bank']
        exposure = retrieve_compatible(edge, bank)
        env = BindingEnvironment(edge); env.call(env.producer_request())
        prefix, prefix_hash = env.checkpoint(), env.state_hash()
        forced_indices = []
        if job.get('forced'):
            action = forced_action(edge, job['forced']); raw = json.dumps(action, ensure_ascii=False)
            request, errors = assemble(env, action)
            replay_observation(env, request, raw, sdk_errors=errors); forced_indices.append(1)
        item = {**job, 'env': env, 'notes': exposure, 'checkpoint': prefix, 'prefix_sha256': prefix_hash,
                'forced_event_indices': forced_indices, 'generations': []}
        if env.verify()['binding_contract_success']:
            done.append(item)
        else:
            active.append(item)
    for _ in range(2):
        if not active:
            break
        generations = agent.infer([messages(j['env'], j['notes']) for j in active]); remaining = []
        for job, generation in zip(active, generations):
            action, error = parse_request(generation['agent_output']); request, sdk_errors = assemble(job['env'], action)
            event_index = len(job['env'].history)
            replay_observation(job['env'], request, generation['agent_output'], error, sdk_errors)
            job['generations'].append({'event_index': event_index, 'prompt_sha256': generation['prompt_sha256'],
                'parse_error': error, 'sdk_errors': sdk_errors, 'hit_generation_limit': generation['hit_generation_limit']})
            calls = len(job['env'].history) - 1
            if job['env'].verify()['binding_contract_success'] or calls >= 2:
                done.append(job)
            else:
                remaining.append(job)
        active = remaining
    rows = []
    for job in done:
        env = job['env']; first = env.history[1]
        row = {'stage': stage, 'edge_uid': job['edge']['uid'], 'task_uid': job['edge']['task_uid'],
            'workflow_group': job['edge']['workflow_group'], 'bank_uid': job['method'],
            'stored_bank': [n['uid'] for n in job['bank']], 'exposure': [n['uid'] for n in job['notes']],
            'checkpoint': job['checkpoint'], 'prefix_sha256': job['prefix_sha256'],
            'events': env.history, 'generations': job['generations'], 'forced_event_indices': job['forced_event_indices'],
            'forced_mode': job.get('forced'), 'assignment': job.get('assignment'),
            'first_try_success': first['accepted'], 'consumer_calls': len(env.history) - 1,
            'first_category': 'accepted' if first['accepted'] else 'contract_error',
            'binding_error_calls': sum(any('binding_' in err or 'missing_binding' in err for err in e['errors']) for e in env.history[1:]),
            'final': env.verify(), 'business_goal_success': None}
        with (output / 'trials.jsonl').open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        rows.append(row)
    return rows


def fixture_probes(edges, output):
    """Independent local API inputs: every leaf for single targets, no Qwen."""
    rows = []
    for edge in edges:
        candidates = [pointer(path) for path, _ in leaves(edge['producer_catalog']['response_example'])]
        for source in candidates + ['/fields_to_bind/0']:
            env = BindingEnvironment(edge); env.call(env.producer_request())
            checkpoint, prefix = env.checkpoint(), env.state_hash()
            action = {'operation': 'bind_consumer', 'api_id': edge['consumer_api'],
                'bindings': {b['target']: {'operation': 'producer', 'pointer': source} for b in edge['bindings']}}
            request, errors = assemble(env, action)
            replay_observation(env, request, json.dumps(action, ensure_ascii=False), sdk_errors=errors)
            rows.append({'stage': 'api_fixture_train', 'edge_uid': edge['uid'],
                'bank_uid': 'source_probe:' + source, 'exposure': [], 'events': env.history,
                'checkpoint': checkpoint, 'prefix_sha256': prefix, 'forced_event_indices': [1], 'final': env.verify()})
    with (output / 'api_probes.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/emp_binding_expansion_v2'))
    parser.add_argument('--history', type=Path, default=Path('results/emp_binding_v3_run1'))
    parser.add_argument('--output', type=Path, default=Path('results/emp_complete_v2'))
    parser.add_argument('--cache', type=Path, default=Path('results/emp_complete_v1'))
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for name, expected in manifest['canonical_sha256'].items():
        if digest(args.data / name) != expected:
            raise ValueError('data changed')
    for name, expected in manifest['sources_sha256'].items():
        if digest(Path(manifest['source_root']) / name) != expected:
            raise ValueError('user data changed')
    previous = json.loads((args.history / 'frozen_protocol.json').read_text())
    for name, expected in previous['source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('previous execution source changed')
    pool = json.loads((args.history / 'memories.json').read_text()); memories = {n['uid']: n for n in pool}
    edges = {e['uid']: e for e in jsonl(args.data / 'edges.jsonl')}
    training = sorted([e for e in edges.values() if e['eligible'] and e['partition'] == 'development'], key=lambda e: e['uid'])
    test = sorted([e for e in edges.values() if e['eligible'] and e['partition'] == 'reserved_unexecuted'], key=lambda e: e['uid'])
    # Public, previously inspected development portfolio only for bank choice.
    future = [e for e in training if e['uid'] not in previous['splits'] or previous['splits'][e['uid']]['partition'] != 'history_train']
    full, feasible = coalitions(pool), coalitions(pool, 2)
    args.output.mkdir(parents=True); (args.output / 'source').mkdir()
    names = ['emp_complete_v2.py', 'emp_structural.py', 'emp_prospective_v2.py', 'emp_prospective.py', 'emp_binding.py',
             'emp_binding_agent.py', 'emp_binding_tools_v3.py', 'emp_binding_tools_v2.py',
             'emp_binding_sdk.py', 'emp_binding_v3.py', 'emp_binding_pilot.py', 'emp_binding_shared.py',
             'policy.py', 'user_data.py']
    protocol = {'scope': 'unified finite document-binding implementation; complete business outcomes unknown',
        'data_directory': str(args.data.resolve()), 'data_manifest_sha256': digest(args.data / 'manifest.json'),
        'history_directory': str(args.history.resolve()), 'cache_directory': str(args.cache.resolve()), 'cache_generations_sha256': digest(args.cache / 'generations.jsonl'), 'history_memories_sha256': digest(args.history / 'memories.json'),
        'history_generations_sha256': digest(args.history / 'generations.jsonl'),
        'source_sha256': {n: digest(Path(__file__).with_name(n)) for n in names},
        'training_edge_uids': [e['uid'] for e in training], 'selection_edge_uids': [e['uid'] for e in future],
        'test_edge_uids': [e['uid'] for e in test], 'test_outcomes_available_before_freeze': False,
        'randomization_seed': 42, 'training_coalitions': len(full), 'storage_max_notes': 2,
        'retrieval_max_notes': 2, 'note_max_tokens': 512, 'max_consumer_calls': 2,
        'kernel': 'literal actions + API responses + observable state; categorical empirical estimates',
        'certificate': False, 'robust_radius': .2, 'call_cost': .02, 'storage_cost': .01,
        'VOI': {'particles': 128, 'probe_cost': .04, 'max_probes': 6, 'positive_only': True,
                'prior': 'Beta(1,1) on unknown outcomes; utility approximation'},
        'baselines': 'NoMem/FIFO/Recency/full-training-LOO/exact-training-Shapley under common retrieval',
        'physical_scope': 'static XLSX first-record fixture contracts; no network or business verification'}
    write(args.output, 'frozen_protocol.json', protocol)
    for name in names:
        (args.output / 'source' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write(args.output, 'memories.json', pool)
    began = perf_counter(); agent = ReplayAgent(args.output, args.cache)
    agent.binding_interface = 'sdk3'
    rng, jobs = random.Random(42), []
    for edge in training:
        order = list(full); rng.shuffle(order)
        for index, bank in enumerate(order):
            jobs.append({'edge': edge, 'bank': bank, 'method': bank_name(bank),
                'assignment': {'scheme': 'random permutation without replacement', 'position': index,
                    'remaining_candidates': len(full) - index, 'conditional_probability': 1 / (len(full) - index)}})
    train = run(agent, jobs, args.output, 'intervention_train')
    repair_jobs = [{'edge': e, 'bank': bank, 'method': mode + ':' + bank_name(bank), 'forced': mode}
        for e in training for bank in [[], pool[-2:]] for mode in ['first_public_leaf', 'invalid_source']]
    repair = run(agent, repair_jobs, args.output, 'repair_context_train')
    api_rows = fixture_probes(training, args.output)
    model = StructuredMechanisms().update(train + repair + api_rows, edges, memories)
    write(args.output, 'model_before.json', model.to_dict())
    initial, initial_scores = choose(model, future, feasible)
    observed = {exposure_cell(edges[r['edge_uid']], [memories[u] for u in r['exposure']]) for r in train}
    probe_rows, rounds = [], []
    for step in range(protocol['VOI']['max_probes']):
        rankings = rank_revalidation(model, future, feasible, observed, seed=42 + step,
            particles=protocol['VOI']['particles'], probe_cost=protocol['VOI']['probe_cost'])
        choice = rankings[0] if rankings and rankings[0]['voi'] > 0 else None
        rounds.append({'round': step, 'rankings': rankings, 'chosen': choice})
        # Decision is written before issuing the probe.
        write(args.output, 'revalidation_schedule.json', rounds)
        if choice is None:
            break
        bank = next(b for b in feasible if bank_name(b) == choice['bank_uid'])
        row = run(agent, [{'edge': edges[choice['edge_uid']], 'bank': bank,
            'method': 'VOI:' + choice['cell_uid']}], args.output, 'revalidation')
        model.update(row, edges, memories); probe_rows.extend(row); observed.add(choice['cell_uid'])
    selected, scores = choose(model, future, feasible)
    attribution = exact_attribution(train, pool, full)
    rank = lambda values: sorted(pool, key=lambda n: (-values[n['uid']], n['arrival_index']))[:2]
    chosen = {'NoMem': [], 'FIFO': pool[:2], 'Recency': pool[-2:],
        'LOO': rank(attribution['full_bank_LOO']), 'Exact-Shapley': rank(attribution['exact_bank_Shapley']),
        'CREDIT-structured': selected, 'CREDIT-no-revalidation': initial}
    write(args.output, 'model_after.json', model.to_dict())
    write(args.output, 'frozen_banks.json', {'banks': {k: [n['uid'] for n in b] for k, b in chosen.items()},
        'initial_scores': initial_scores, 'final_scores': scores, 'attribution': attribution})
    predictions = [{'edge_uid': e['uid'], 'method': method, 'exposure': [n['uid'] for n in retrieve_compatible(e, bank)],
        **(StructuredMechanisms.from_dict(json.loads((args.output / 'model_before.json').read_text())) if method == 'CREDIT-no-revalidation' else model).predict(e, retrieve_compatible(e, bank))}
        for e in test for method, bank in chosen.items()]
    write(args.output, 'frozen_predictions.json', predictions)
    # Hash the concrete inputs to evaluation before evaluating any reserved edge.
    write(args.output, 'evaluation_freeze.json', {name: digest(args.output / name) for name in
        ['frozen_protocol.json', 'model_before.json', 'model_after.json', 'frozen_banks.json', 'frozen_predictions.json', 'revalidation_schedule.json']})
    evaluation = run(agent, [{'edge': e, 'bank': bank, 'method': method} for e in test for method, bank in chosen.items()], args.output, 'heldout_evaluation')
    agent.close()
    frame = {r['edge_uid']: r for r in evaluation if r['bank_uid'] == 'NoMem'}
    paired = []
    for method in chosen:
        rows = [r for r in evaluation if r['bank_uid'] == method]
        paired.append({'method': method, 'paired_tasks': len(rows),
            'success_delta_vs_NoMem': sum(int(r['final']['binding_contract_success']) - int(frame[r['edge_uid']]['final']['binding_contract_success']) for r in rows) / len(rows),
            'local_call_delta_vs_NoMem': sum(r['final']['calls'] - frame[r['edge_uid']]['final']['calls'] for r in rows) / len(rows),
            'statistical_claim': 'two document-fixture tasks; no significance or generality claim'})
    # Synthetic diagnostic changes only fixture acceptance, with unchanged schema.
    source, target = StructuredMechanisms(), StructuredMechanisms()
    sample = next(r for r in train if r['final']['binding_contract_success'])
    source.update([sample], edges, memories)
    changed_edges = deepcopy(edges); changed_edges[sample['edge_uid']]['fixture_version'] = 'injected_reject_v2'
    # Execute the same concrete first consumer request in a test-only changed
    # validator. Both the raw injected trace and its mutation are disclosed.
    env = environment(changed_edges[sample['edge_uid']]); env.call(env.producer_request())
    checkpoint, prefix = env.checkpoint(), env.state_hash()
    env.call(sample['events'][1]['request'])
    changed = {'stage': 'api_fixture_train', 'edge_uid': sample['edge_uid'], 'bank_uid': 'injected_drift',
        'exposure': [], 'events': env.history, 'forced_event_indices': [1],
        'checkpoint': checkpoint, 'prefix_sha256': prefix, 'final': env.verify()}
    target.update([changed], changed_edges, memories)
    write(args.output, 'injected_drift_diagnostic.json', {'scope': 'declared synthetic rejection, not observed provider drift',
        'source_trial': sample, 'target_trial': changed, 'target_fixture_version': 'injected_reject_v2',
        **compare_api_kernels(source, target)})
    counts = Counter(r['stage'] for r in train + repair + probe_rows + evaluation)
    summary = {'main': summarize(evaluation), 'paired': paired, 'trial_counts': dict(counts),
        'api_probe_trials': len(api_rows), 'api_probe_calls': sum(len(r['events']) for r in api_rows),
        'local_calls_by_stage': {stage: sum(len(r['events']) for r in train + repair + probe_rows + evaluation if r['stage'] == stage) for stage in counts},
        'policy_cells': len(model.policy), 'API_cells': len(model.api),
        'test_prediction_coverage': sum(p['supported'] for p in predictions) / len(predictions),
        'revalidation_probes_executed': len(probe_rows), 'test_edges_executed': len(test),
        'reserved_workflows_with_eligible_edges_executed': len({e['workflow_group'] for e in test}),
        'reserved_ineligible_edges_not_executed': sum(not e['eligible'] and e['partition'] == 'reserved_unexecuted' for e in edges.values()),
        'elapsed_seconds': perf_counter() - began, 'business_goal_success': None, 'live_api_calls': 0,
        'scope': protocol['scope']}
    write(args.output, 'summary.json', summary); print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
