"""Reexecute frozen local ToolSandbox tools and original task graders, without an LLM."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import tempfile

import dill
import numpy as np

from . import toolsandbox_local_v3 as adapter
from .user_data import digest


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_report(episodes, settings):
    future = [r for r in episodes if r['stage'] == 'future_test']
    methods = sorted({r['method'] for r in future})
    grouped = defaultdict(dict)
    for row in future:
        require(row['method'] not in grouped[row['task']], 'duplicate task/method')
        grouped[row['task']][row['method']] = row
    require(all(set(values) == set(methods) for values in grouped.values()), 'unpaired future rows')
    families = defaultdict(list)
    for task, values in grouped.items():
        families[values['NoMem']['family']].append(task)
    clusters = sorted(families)
    rng = np.random.default_rng(settings['bootstrap_seed'])
    # Identical cluster draws for every method and metric; variant counts vary.
    draws = rng.integers(0, len(clusters), (settings['paired_bootstrap_replicates'], len(clusters)))
    report = []
    for method in methods:
        for metric in ('strict_VSR', 'upstream_score', 'tool_calls'):
            def value(row):
                if metric == 'strict_VSR':
                    return row['success']
                if metric == 'tool_calls':
                    return row['tool_calls']
                return row['upstream_evaluation']['similarity'] if row['upstream_evaluation'] else 0.
            sums = np.array([sum(value(grouped[t][method]) for t in families[c]) for c in clusters])
            base = np.array([sum(value(grouped[t]['NoMem']) for t in families[c]) for c in clusters])
            counts = np.array([len(families[c]) for c in clusters])
            denominator = counts[draws].sum(axis=1)
            samples = sums[draws].sum(axis=1) / denominator
            differences = (sums - base)[draws].sum(axis=1) / denominator
            estimate = float(sums.sum() / counts.sum())
            baseline = float(base.sum() / counts.sum())
            report.append({'method': method, 'metric': metric, 'estimate': estimate,
                'cluster_CI95': np.quantile(samples, [.025, .975]).tolist(),
                'delta_vs_NoMem': estimate - baseline,
                'relative_change_vs_NoMem': (estimate - baseline) / baseline if baseline else None,
                'paired_delta_CI95': np.quantile(differences, [.025, .975]).tolist()})
    return {'table': report, 'episodes_per_method': len(grouped), 'task_families': len(clusters),
        'bootstrap': settings, 'p_values': None,
        'limitations': 'one deterministic actor seed, task-family interval only; no provider/general seed inference'}


def verify(directory, *, full_replay=True):
    protocol = read(directory / 'frozen_protocol.json')
    for name, expected in protocol['source_sha256'].items():
        require(digest(directory / 'source' / name) == expected, 'frozen source snapshot changed: ' + name)
        require(digest(Path(adapter.__file__).with_name(name)) == expected, 'current source changed: ' + name)
    upstream = Path(protocol['upstream_directory'])
    for name, expected in protocol['upstream_source_sha256'].items():
        require(digest(upstream / name) == expected, 'upstream source changed: ' + name)
    require(digest(directory / 'scenarios.pkl') == read(directory / 'scenario_freeze.json')['sha256'],
        'scenario snapshot changed')
    for name, expected in read(directory / 'evaluation_freeze.json').items():
        require(digest(directory / name) == expected, 'pre-evaluation artifact changed: ' + name)
    u = adapter.upstream_imports(upstream)
    adapter.install_local_clock(protocol['clock'])
    scenarios = dill.loads((directory / 'scenarios.pkl').read_bytes())
    manifest = read(directory / 'agent_manifest.json')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True)
    generations = rows(directory / 'generations.jsonl')
    cache = {r['prompt_sha256']: r for r in generations}
    require(len(cache) == len(generations), 'duplicate physical generation contexts')
    require(len(cache) == manifest['unique_generation_contexts'], 'generation count differs')
    for key, value in cache.items():
        require(sha256(value['messages_rendered'].encode()).hexdigest() == key, 'cached prompt hash differs')
        require(len(tokenizer.encode(value['messages_rendered'], add_special_tokens=False)) == value['input_tokens'],
            'generation input token count differs')
    episodes = rows(directory / 'episodes.jsonl')
    history = [r for r in episodes if r['stage'] == 'history']
    interventions = [r for r in episodes if r['stage'] == 'intervention_train']
    future = [r for r in episodes if r['stage'] == 'future_test']
    require({r['task'] for r in history} == set(protocol['history']), 'history coverage differs')
    require(not set(protocol['history']) & set(protocol['future']), 'history/future leakage')
    notes = read(directory / 'memories.json')
    reconstructed = []
    for row in sorted(history, key=lambda r: r['task']):
        note = adapter.memory_from(row, tokenizer)
        if note and not any(set(n['tools']) == set(note['tools']) for n in reconstructed):
            reconstructed.append(note)
        if len(reconstructed) == protocol['maximum_memory_pool']:
            break
    require(reconstructed == notes, 'memory provenance differs')
    memory = {n['uid']: n for n in notes}
    assignments = read(directory / 'intervention_assignment.json')
    actual = [{'task': r['task'], 'bank': sorted(r['bank'])} for r in interventions]
    require(actual == [{k: r[k] for k in ('task', 'bank')} for r in assignments[:len(actual)]],
        'interventions are not a prefix of frozen randomized schedule')
    for index, row in enumerate(assignments):
        require(row['conditional_order_propensity'] == 1 / (len(assignments) - index), 'order propensity differs')
    used = sum(r['tool_calls'] for r in interventions)
    require(used <= protocol['extra_tool_call_budget'], 'probe budget exceeded')
    banks = read(directory / 'frozen_banks.json')
    fake_agent = type('ReplayTokenizer', (), {'tokenizer': tokenizer})()
    rebuilt_banks, selection = adapter.scoring_banks(notes, interventions, protocol['future'], fake_agent, scenarios, u)
    require({m: list(adapter.bank_key(b)) for m, b in rebuilt_banks.items()} == banks, 'retention differs')
    require(selection == read(directory / 'selection.json'), 'training-only selection differs')
    require(len(future) == len(banks) * len(protocol['future']), 'future coverage incomplete')
    for method in banks:
        subset = [r for r in future if r['method'] == method]
        require({r['task'] for r in subset} == set(protocol['future']), 'method coverage differs: ' + method)
        require(all(sorted(r['bank']) == banks[method] for r in subset), 'deployed bank differs')
        require(len(banks[method]) <= protocol['deployment_bank_memory_capacity'], 'memory capacity exceeded')
        require(sum(memory[n]['tokens'] for n in banks[method]) <= protocol['storage_tokens'], 'storage exceeded')
    compared = 0
    if full_replay:
        with tempfile.TemporaryDirectory(prefix='credit-toolsandbox-verify-') as temp:
            target = Path(temp)
            for row in episodes:
                bank = [memory[n] for n in row['bank']]
                job = adapter.initial_job(row['task'], scenarios[row['task']], bank, u, tokenizer, row['stage'], row['method'])
                require(job['initial_state_sha256'] == row['initial_state_sha256'], 'initial context differs')
                require(job['exposure'] == row['exposure'], 'retrieval differs')
                for turn in row['generations']:
                    rendered = tokenizer.apply_chat_template(job['chat'], tokenize=False, add_generation_prompt=True)
                    key = sha256(rendered.encode()).hexdigest()
                    require(key == turn['prompt_sha256'] and key in cache,
                        f'public actor prompt differs: {row["stage"]}/{row["task"]}/{row["method"]}')
                    adapter.consume(job, cache[key], u, protocol['maximum_tool_calls_per_episode'])
                if not job['done']:
                    job['termination'] = 'turn_budget'
                rebuilt = adapter.finish(job, target, u)
                for key in ('calls', 'generations', 'ending_state_sha256', 'upstream_evaluation', 'success',
                        'verification_coverage', 'verification_error', 'termination'):
                    require(adapter.fingerprint(rebuilt[key]) == adapter.fingerprint(row[key]),
                        f'replay differs: {row["stage"]}/{row["task"]}/{row["method"]}/{key}')
                compared += 1
    # Reconstruct summary from raw rows, including all unknown verifications as failures.
    summary = read(directory / 'summary.json')
    require(summary['training_extra_tool_calls'] == used, 'summary probe cost differs')
    for entry in summary['table']:
        subset = [r for r in future if r['method'] == entry['method']]
        require(entry['strict_VSR'] == np.mean([r['success'] for r in subset]), 'summary VSR differs')
        require(entry['verification_coverage'] == np.mean([r['verification_coverage'] for r in subset]),
            'summary verification coverage differs')
        require(entry['upstream_mean_score'] == np.mean([r['upstream_evaluation']['similarity']
            if r['upstream_evaluation'] else 0. for r in subset]), 'summary upstream score differs')
        require(entry['mean_tool_calls'] == np.mean([r['tool_calls'] for r in subset]), 'summary cost differs')
    stats = paired_report(episodes, protocol['statistics'])
    adapter.write(directory, 'paired_statistics.json', stats)
    environment = {name: importlib.metadata.version(name) for name in ('polars', 'ccy', 'pint', 'phonenumbers',
        'holidays', 'rouge-score', 'dill', 'torch', 'transformers', 'numpy', 'openai', 'langchain-core')}
    report = {'verified': True, 'replayed_episodes': compared, 'full_replay': full_replay,
        'history_episodes': len(history), 'intervention_episodes': len(interventions), 'future_episodes': len(future),
        'total_local_tool_calls': sum(r['tool_calls'] for r in episodes), 'extra_probe_tool_calls': used,
        'upstream_source_files': len(protocol['upstream_source_sha256']), 'unique_generation_contexts': len(cache),
        'verification_errors': dict(Counter(r['verification_error'] for r in episodes if r['verification_error'])),
        'environment': environment, 'python_compatibility_override': 'ccy 1.4.0 replaces upstream 1.3.1 on Python 3.12',
        'verifier_source_sha256': digest(Path(__file__)), 'full_paper_CREDIT': False}
    report['replay_comparison_scope'] = 'all public prompts, original grader outputs and complete state; ephemeral object repr addresses normalized'
    adapter.write(directory, 'execution_audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == '__main__':
    main()
