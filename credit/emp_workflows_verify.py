"""Reconstruct full local-star episodes, including the shared producer prefix."""

import argparse
from pathlib import Path

from .emp_binding_v2_verify import read, verify_sources
from .emp_binding_verify import require
from .emp_structural import StructuredMechanisms
from .emp_workflows import RecordedAgent, compile_workflows, execute
from .user_data import digest, jsonl


def verify(output):
    import json
    protocol = read(output, 'frozen_protocol.json'); parent = Path(protocol['input_directory'])
    parent_protocol = verify_sources(parent); data = Path(parent_protocol['data_directory'])
    manifest = read(data, 'manifest.json')
    tasks_path = Path(manifest['source_root']) / 'emp_data2_synthetic_compositions.jsonl'
    for path, key in [(tasks_path, 'source_tasks_sha256'), (data / 'edges.jsonl', 'edges_sha256'),
        (parent / 'model_after.json', 'model_sha256'), (parent / 'frozen_banks.json', 'banks_sha256'),
        (parent / 'generations.jsonl', 'generation_cache_sha256')]:
        require(digest(path) == protocol[key], 'workflow input changed')
    from . import emp_workflows
    require(digest(Path(emp_workflows.__file__)) == protocol['executor_sha256'], 'workflow executor changed')
    require(digest(output / 'executor.py') == protocol['executor_sha256'], 'frozen executor changed')
    edges = {r['uid']: r for r in jsonl(data / 'edges.jsonl')}
    graphs, excluded = compile_workflows(list(jsonl(tasks_path)), list(edges.values()))
    require(graphs == protocol['graphs'] and excluded == read(output, 'exclusions.json'), 'complete/unknown classification differs')
    pool = read(parent, 'memories.json'); memories = {m['uid']: m for m in pool}
    banks = read(parent, 'frozen_banks.json')['banks']
    model = StructuredMechanisms.from_dict(read(parent, 'model_after.json')); agent = RecordedAgent(parent)
    episodes = list(jsonl(output / 'episodes.jsonl')); replayed = []
    for graph in graphs:
        for method, uids in banks.items():
            replayed.append({'method': method, **execute(graph, edges, [memories[u] for u in uids], agent, model)})
    require(replayed == episodes, 'whole-workflow replay differs')
    summary = read(output, 'summary.json')
    for row in summary['table']:
        subset = [e for e in episodes if e['method'] == row['method']]
        require(row['workflow_contracts'] == len(subset), 'workflow coverage differs')
        for field, source in [('successes', 'binding_workflow_success'), ('local_calls', 'local_calls'),
            ('independent_edge_replay_calls', 'independent_edge_replay_calls'), ('saved_producer_calls', 'saved_producer_calls')]:
            require(row[field] == sum(e[source] for e in subset), 'whole-workflow summary differs')
    report = {'sources_and_frozen_inputs_verified': True, 'complete_graphs_verified': len(graphs),
        'unknown_workflows_preserved': len(excluded), 'full_workflow_episodes_replayed': len(episodes),
        'shared_prefix_calls_verified': sum(e['saved_producer_calls'] for e in episodes),
        'consumer_request_response_state_and_predictions_verified': True,
        'new_LLM_calls': 0, 'live_API_calls': 0, 'business_goal_success': None,
        'verifier_sha256': digest(Path(__file__))}
    (output / 'integrity.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/emp_workflows_v1'))
    verify(parser.parse_args().results)
