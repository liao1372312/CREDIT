"""Execute every fully specified dependency star without repeating its producer.

Only three supplied dependency workflows have contracts for every edge. Other
workflows are reported unknown, rather than silently reduced to one known edge.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .emp_binding import BindingEnvironment
from .emp_binding_agent import parse_request
from .emp_binding_tools_v3 import assemble, messages, retrieve_compatible
from .emp_structural import StructuredMechanisms, fingerprint, replay_observation
from .user_data import digest, jsonl


def compile_workflows(tasks, edges):
    by_task = defaultdict(list)
    for edge in edges:
        by_task[edge['task_uid']].append(edge)
    graphs, excluded = [], []
    for task in tasks:
        if task['metadata']['composition_relation'] != 'data_dependency':
            continue
        group = by_task[task['id']]
        expected = {(dep, step['id']) for step in task['TaskList'] for dep in step.get('depends_on', [])}
        actual = {(e['producer_step'], e['consumer_step']) for e in group}
        reason = ''
        if expected != actual or not group:
            reason = 'missing_dependency_edges'
        elif not all(e['eligible'] for e in group):
            reason = 'incomplete_document_contracts'
        else:
            producers = {e['producer_step'] for e in group}; consumers = {e['consumer_step'] for e in group}
            if len(producers) != 1 or producers.intersection(consumers) or any(sum(e['consumer_step'] == c for e in group) != 1 for c in consumers):
                reason = 'non_star_requires_additional_state_backend'
            elif producers | consumers != {s['id'] for s in task['TaskList']}:
                reason = 'uncovered_task_nodes'
            elif len({fingerprint([e['producer_api'], e['producer_catalog']]) for e in group}) != 1:
                reason = 'inconsistent_shared_producer_fixture'
        if reason:
            excluded.append({'task_uid': task['id'], 'reason': reason,
                'known_edges': sum(e['eligible'] for e in group), 'total_edges': len(group),
                'binding_workflow_success': None, 'business_goal_success': None})
        else:
            graphs.append({'task_uid': task['id'], 'edge_uids': [e['uid'] for e in sorted(group, key=lambda e: e['consumer_step'])],
                'node_count': len(task['TaskList']), 'producer_step': next(iter(producers)),
                'scope': 'all dependency-star argument contracts; business outcome unknown'})
    return graphs, excluded


class RecordedAgent:
    def __init__(self, directory):
        from transformers import AutoTokenizer
        manifest = json.loads((directory / 'agent_manifest.json').read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True)
        self.cache = {r['prompt_sha256']: r for r in jsonl(directory / 'generations.jsonl')}

    def infer(self, chat):
        rendered = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        key = sha256(rendered.encode()).hexdigest()
        if key not in self.cache:
            raise ValueError('unobserved workflow prompt: new inference is required')
        return self.cache[key]


def execute(graph, edge_lookup, bank, agent, model):
    edges = [edge_lookup[u] for u in graph['edge_uids']]
    producer = BindingEnvironment(edges[0]); producer.call(producer.producer_request())
    checkpoint = producer.checkpoint(); branches, global_events = [], [deepcopy(producer.history[0])]
    for edge in edges:
        env = BindingEnvironment(edge); env.restore(checkpoint)
        require_request = env.producer_request()
        if require_request != producer.history[0]['request']:
            raise ValueError('cannot reuse a different producer request')
        exposure = retrieve_compatible(edge, bank); generations = []
        prediction = model.predict(edge, exposure)
        for _ in range(2):
            record = agent.infer(messages(env, exposure))
            action, error = parse_request(record['agent_output']); request, sdk_errors = assemble(env, action)
            replay_observation(env, request, record['agent_output'], error, sdk_errors)
            generations.append({'prompt_sha256': record['prompt_sha256'], 'parse_error': error, 'sdk_errors': sdk_errors})
            if env.verify()['binding_contract_success']:
                break
        branches.append({'edge_uid': edge['uid'], 'exposure': [n['uid'] for n in exposure],
            'events': env.history[1:], 'generations': generations, 'prediction': prediction, 'final': env.verify()})
        global_events.extend(deepcopy(env.history[1:]))
    all_supported = all(b['prediction']['supported'] for b in branches)
    # The adapter explicitly isolates consumer state and fixes one producer
    # response. Conditional branch factorization is scoped to these fixtures.
    probability = 1.
    for branch in branches:
        if all_supported:
            probability *= branch['prediction']['final_success']
    edge_replay_calls = sum(len(b['events']) + 1 for b in branches)
    return {'task_uid': graph['task_uid'], 'producer_checkpoint': checkpoint, 'branches': branches,
        'global_events': global_events, 'nodes_executed': 1 + len(branches),
        'local_calls': len(global_events), 'independent_edge_replay_calls': edge_replay_calls,
        'saved_producer_calls': edge_replay_calls - len(global_events),
        'binding_workflow_success': all(b['final']['binding_contract_success'] for b in branches),
        'predicted_binding_workflow_success': probability if all_supported else None,
        'supported': all_supported, 'business_goal_success': None, 'provider_success': None,
        'agent_source': 'exact previously generated SDK3 prompt replay; no new LLM inference',
        'factorization': 'shared fixed producer; explicitly isolated consumer branches'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/emp_complete_v1'))
    parser.add_argument('--output', type=Path, default=Path('results/emp_workflows_v1'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    protocol = json.loads((args.results / 'frozen_protocol.json').read_text())
    data = Path(protocol['data_directory']); manifest = json.loads((data / 'manifest.json').read_text())
    tasks_path = Path(manifest['source_root']) / 'emp_data2_synthetic_compositions.jsonl'
    if digest(tasks_path) != manifest['sources_sha256']['emp_data2_synthetic_compositions.jsonl']:
        raise ValueError('user tasks changed')
    edges = {e['uid']: e for e in jsonl(data / 'edges.jsonl')}
    graphs, excluded = compile_workflows(list(jsonl(tasks_path)), list(edges.values()))
    args.output.mkdir(parents=True)
    freeze = {'scope': 'complete local dependency-star contracts; previously examined development tasks',
        'input_directory': str(args.results.resolve()), 'source_tasks_sha256': digest(tasks_path),
        'edges_sha256': digest(data / 'edges.jsonl'),
        'model_sha256': digest(args.results / 'model_after.json'), 'banks_sha256': digest(args.results / 'frozen_banks.json'),
        'generation_cache_sha256': digest(args.results / 'generations.jsonl'),
        'executor_sha256': digest(Path(__file__)), 'graphs': graphs, 'new_LLM_calls': 0, 'live_API_calls': 0}
    (args.output / 'frozen_protocol.json').write_text(json.dumps(freeze, indent=2))
    (args.output / 'executor.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'exclusions.json').write_text(json.dumps(excluded, indent=2))
    pool = json.loads((args.results / 'memories.json').read_text()); memories = {n['uid']: n for n in pool}
    banks = json.loads((args.results / 'frozen_banks.json').read_text())['banks']
    model = StructuredMechanisms.from_dict(json.loads((args.results / 'model_after.json').read_text()))
    agent = RecordedAgent(args.results); episodes = []
    for graph in graphs:
        for method, uids in banks.items():
            episodes.append({'method': method, **execute(graph, edges, [memories[u] for u in uids], agent, model)})
    with (args.output / 'episodes.jsonl').open('w') as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + '\n')
    table = []
    for method in banks:
        subset = [r for r in episodes if r['method'] == method]
        table.append({'method': method, 'workflow_contracts': len(subset),
            'successes': sum(r['binding_workflow_success'] for r in subset),
            'local_calls': sum(r['local_calls'] for r in subset),
            'independent_edge_replay_calls': sum(r['independent_edge_replay_calls'] for r in subset),
            'saved_producer_calls': sum(r['saved_producer_calls'] for r in subset),
            'business_goal_success': None})
    summary = {'table': table, 'eligible_complete_workflows': len(graphs), 'unknown_workflows': len(excluded),
        'exclusions': dict(Counter(r['reason'] for r in excluded)), 'episodes': len(episodes),
        'new_LLM_calls': 0, 'live_API_calls': 0, 'business_goal_success': None,
        'scope': freeze['scope']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
