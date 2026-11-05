"""Frozen development pilot: structured bindings, repair, and memory interventions."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from .emp_binding_agent import QwenJSONAgent, run_trials
from .emp_binding import BindingEnvironment
from .emp_binding_mechanism import fit
from .user_data import digest, jsonl

MODEL = Path('/home/lxc/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')


def split_edges(edges):
    # Merge workflows connected by an identical API-pair binding contract.
    parent = {e['workflow_group']: e['workflow_group'] for e in edges}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    seen = {}
    for e in edges:
        if e['binding_group'] in seen:
            a, b = find(e['workflow_group']), find(seen[e['binding_group']])
            parent[a] = b
        seen[e['binding_group']] = e['workflow_group']
    groups = defaultdict(list)
    for e in edges:
        groups[find(e['workflow_group'])].append(e)
    order = sorted(groups, key=lambda x: sha256(('pilot42:' + x).encode()).hexdigest())
    n_train, n_val = max(1, int(len(order) * .6)), max(1, int(len(order) * .2))
    result = {}
    for i, group in enumerate(order):
        name = 'history_train' if i < n_train else 'validation' if i < n_train + n_val else 'development_evaluation'
        for edge in groups[group]:
            result[edge['uid']] = {'partition': name, 'component': group}
    return result


def memories_from_history(rows, edges, tokenizer):
    notes, signatures = [], set()
    for row in sorted(rows, key=lambda r: r['edge_uid']):
        if not row['final']['binding_contract_success']:
            continue
        event = next(e for e in reversed(row['events']) if e['accepted'])
        mapping = event['request']['bindings']
        signature = json.dumps(mapping, sort_keys=True)
        if signature in signatures:
            continue
        signatures.add(signature)
        pairs = list(mapping.items())
        kept = []
        for target, source in pairs:
            candidate = kept + [f"{target} <- producer{source['pointer']}"]
            text = '一次验收通过的历史绑定：' + '; '.join(candidate) + '。仅在本次字段与响应结构匹配时复用路径，必须从本次响应取值，不能复用历史值。其他参数保持当前模板不变。'
            if len(tokenizer.encode(text, add_special_tokens=False)) <= 512:
                kept = candidate
        if not kept:
            continue
        text = '一次验收通过的历史绑定：' + '; '.join(kept) + '。仅在本次字段与响应结构匹配时复用路径，必须从本次响应取值，不能复用历史值。其他参数保持当前模板不变。'
        notes.append({'uid': 'memory_' + sha256(signature.encode()).hexdigest()[:16], 'text': text,
            'arrival_index': len(notes), 'source_edge': row['edge_uid'], 'source_stage': row['stage'],
            'source_trial_bank': row['bank_uid'], 'verified_mapping': mapping,
            'target_names': [target.split('/')[-1].casefold() for target, _ in pairs],
            'tokens': len(tokenizer.encode(text, add_special_tokens=False))})
        if len(notes) == 4:
            break
    return notes


def retrieve(edge, bank):
    names = {b['target'].split('/')[-1].casefold() for b in edge['bindings']}
    scored = [(len(names.intersection(note['target_names'])), note['uid'], note) for note in bank]
    # Irrelevant notes are not forced into every task; exposure changes by context.
    return [note for score, _, note in sorted(scored, key=lambda x: (-x[0], x[1]))[:2] if score > 0]


def bank_name(bank):
    return 'empty' if not bank else '+'.join(sorted(note['uid'] for note in bank))


def all_banks(pool):
    return [list(group) for size in range(min(2, len(pool)) + 1) for group in combinations(pool, size)]


def utility(row):
    return float(row['final']['binding_contract_success']) - .02 * row['consumer_calls']


def attribution_banks(rows, pool, banks):
    by_bank = {bank_name(bank): mean(utility(r) for r in rows if r['bank_uid'] == bank_name(bank)) for bank in banks}
    empty = by_bank['empty']
    loo, shapley = {}, {}
    # These are restricted, budget-feasible training-coalition scores.
    for note in pool:
        uid = note['uid']; deltas = []
        for other in pool:
            if other['uid'] != uid:
                pair = bank_name([note, other]); singleton = bank_name([other])
                deltas.append(by_bank[pair] - by_bank[singleton])
        loo[uid] = mean(deltas) if deltas else by_bank[bank_name([note])] - empty
        # Exact average of admissible size-0/1 marginal coalitions, not full-bank MC-Shapley.
        shapley[uid] = mean([by_bank[bank_name([note])] - empty] + deltas)
    rank = lambda scores: sorted(pool, key=lambda note: (-scores[note['uid']], note['arrival_index']))[:2]
    return rank(loo), rank(shapley), {'restricted_LOO': loo, 'restricted_marginal_average': shapley}


def select_joint(model, validation_edges, pool, banks, supported_fields):
    supported = [edge for edge in validation_edges if {b['target'].split('/')[-1].casefold() for b in edge['bindings']}.issubset(supported_fields)]
    if not supported:
        return [], [{'bank_uid': bank_name(bank), 'predicted_objective': None, 'retained_count': len(bank), 'reason': 'no future field-stratum supported by training; fall back to NoMem'} for bank in banks]
    scores = []
    for bank in banks:
        cells = defaultdict(list)
        for edge in supported:
            prediction = model.predict(edge, retrieve(edge, bank))
            family = tuple(sorted(b['target'].split('/')[-1].casefold() for b in edge['bindings']))
            cells[family].append(prediction['final_success'] - .02 * prediction['expected_consumer_calls'])
        means = [mean(values) for values in cells.values()]
        # Explicit conservative contamination mixture over public future strata.
        value = .8 * mean(means) + .2 * min(means) - .01 * len(bank)
        scores.append({'bank_uid': bank_name(bank), 'predicted_objective': value,
                       'retained_count': len(bank), 'scenario_count': len(cells)})
    chosen = max(scores, key=lambda row: (row['predicted_objective'], -row['retained_count'], row['bank_uid']))
    bank = next(bank for bank in banks if bank_name(bank) == chosen['bank_uid'])
    return bank, scores


def summarize(rows):
    result = []
    for method in sorted({r['bank_uid'] for r in rows}):
        subset = [r for r in rows if r['bank_uid'] == method]
        result.append({'method': method, 'binding_tasks': len(subset),
            'first_try_binding_success': mean(r['first_try_success'] for r in subset),
            'final_binding_success': mean(r['final']['binding_contract_success'] for r in subset),
            'mean_local_calls': mean(r['final']['calls'] for r in subset),
            'binding_error_calls': sum(r['binding_error_calls'] for r in subset),
            'repair_attempts': sum(r['consumer_calls'] - 1 for r in subset),
            'exposure_patterns': len({tuple(r['exposure']) for r in subset}),
            'business_goal_success': None, 'provider_success': None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/emp_binding_data_v2'))
    parser.add_argument('--output', type=Path, default=Path('results/emp_binding_pilot_v1'))
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--interface', choices=['full_json', 'sdk'], default='sdk')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new output directory')
    data_manifest = json.loads((args.data / 'manifest.json').read_text())
    for name, expected in data_manifest['canonical_sha256'].items():
        if digest(args.data / name) != expected:
            raise ValueError('input changed since adaptation')
    if digest(Path(__file__).with_name('emp_binding.py')) != data_manifest['adapter_sha256']:
        raise ValueError('adapter changed since input was frozen')
    edges = [e for e in jsonl(args.data / 'edges.jsonl') if e['eligible'] and e['partition'] == 'development']
    splits = split_edges(edges)
    partitions = {name: [e for e in edges if splits[e['uid']]['partition'] == name]
                  for name in ['history_train', 'validation', 'development_evaluation']}
    if any(not values for values in partitions.values()):
        raise ValueError('not enough independent development groups')
    args.output.mkdir(parents=True)
    sources = ['emp_binding.py', 'emp_binding_agent.py', 'emp_binding_mechanism.py', 'emp_binding_pilot.py', 'emp_binding_sdk.py']
    protocol = {'scope': 'development binding pilot, not full CREDIT or business-goal VSR',
        'data_directory': str(args.data.resolve()), 'data_manifest_sha256': digest(args.data / 'manifest.json'),
        'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in sources},
        'splits': splits, 'reserved_workflows_executed': 0, 'history_source': 'successful history_train NoMem trials and explicitly labeled document-reference demonstrations on history_train only',
        'candidate_pool_max': 4, 'storage_max_notes': 2, 'note_max_tokens': 512, 'retrieval_max_notes': 2,
        'retrieval': 'positive target-field overlap, fixed UID ties; context-dependent exposure',
        'binding_interface': args.interface,
        'agent': {'do_sample': False, 'max_consumer_calls': 2, 'max_new_tokens': 3072, 'max_prompt_tokens': 12288},
        'local_mechanism': 'shared contextual unary and pair coefficient groups; coarsened decision/repair likelihood + outcome loss + group norm',
        'selection': 'unlabeled validation contexts, .8 mean + .2 worst field-stratum; success -.02 consumer calls -.01 retained notes',
        'baseline_limits': 'LOO and marginal-average are restricted to feasible coalitions, not published independent baseline reproductions',
        'statistical_scope': 'single deterministic development portfolio; no independent-seed confidence or significance claims',
        'api_kernel': 'known document-derived binding validator; no learned provider transitions',
        'business_goal_verifier': 'unknown, never counted as pass or fail',
        'support_gate': 'unseen target-field strata excluded from bank selection; field vocabulary check is necessary only, not a transfer certificate'}
    (args.output / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2))
    (args.output / 'source').mkdir()
    for name in sources:
        (args.output / 'source' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    started = perf_counter()
    agent = QwenJSONAgent(args.model, args.output, args.batch_size)
    agent.binding_interface = args.interface
    print(json.dumps({'split_counts': {k: len(v) for k, v in partitions.items()}}), flush=True)
    history = run_trials(agent, [(edge, 'empty', []) for edge in partitions['history_train']], args.output, 'history')
    lookup = {e['uid']: e for e in edges}
    demonstrations = []
    for edge in partitions['history_train']:
        env = BindingEnvironment(edge); env.call(env.producer_request())
        checkpoint = env.checkpoint(); prefix = env.state_hash()
        env.call(env.reference_request())
        demonstrations.append({'stage': 'assisted_history', 'edge_uid': edge['uid'],
            'bank_uid': 'document_reference', 'events': env.history, 'checkpoint': checkpoint,
            'prefix_sha256': prefix, 'final': env.verify(),
            'agent_source': 'deterministic document-contract reference; not an autonomous Qwen success'})
    with (args.output / 'assisted_history.jsonl').open('w') as handle:
        for row in demonstrations:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    pool = memories_from_history(history + demonstrations, lookup, agent.tokenizer)
    if not pool:
        raise ValueError('no verified historical notes; cannot claim memory experiment completed')
    (args.output / 'memories.json').write_text(json.dumps(pool, ensure_ascii=False, indent=2))
    banks = all_banks(pool)
    train = run_trials(agent, [(edge, bank_name(bank), retrieve(edge, bank)) for edge in partitions['history_train'] for bank in banks], args.output, 'intervention_train')
    model = fit(train, lookup, pool, args.output)
    # The predictions and bank choice precede validation/evaluation outcomes.
    trained_names = {b['target'].split('/')[-1].casefold() for edge in partitions['history_train'] for b in edge['bindings']}
    joint, scores = select_joint(model, partitions['validation'], pool, banks, trained_names)
    loo, marginal, attribution = attribution_banks(train, pool, banks)
    chosen = {'NoMem': [], 'FIFO': pool[:2], 'Recency': pool[-2:],
              'Restricted-LOO': loo, 'Restricted-marginal': marginal, 'CREDIT-binding-pilot': joint}
    (args.output / 'frozen_banks.json').write_text(json.dumps({'banks': {method: [n['uid'] for n in bank] for method, bank in chosen.items()},
        'joint_candidates': scores, 'attribution_training_scores': attribution}, indent=2))
    predictions = [{'edge_uid': edge['uid'], 'bank_uid': bank_name(bank),
                    'exposure': [n['uid'] for n in retrieve(edge, bank)],
                    'field_vocabulary_supported': {b['target'].split('/')[-1].casefold() for b in edge['bindings']}.issubset(trained_names), **model.predict(edge, retrieve(edge, bank))}
                   for edge in partitions['validation'] for bank in banks]
    (args.output / 'validation_predictions.json').write_text(json.dumps(predictions, indent=2))
    validation = run_trials(agent, [(edge, bank_name(bank), retrieve(edge, bank)) for edge in partitions['validation'] for bank in banks], args.output, 'validation')
    observed = {(r['edge_uid'], r['bank_uid']): r for r in validation}
    trained_names = {b['target'].split('/')[-1].casefold() for edge in partitions['history_train'] for b in edge['bindings']}
    diagnostics = {'trained_field_names': sorted(trained_names), 'validation_unknown_field_names': sorted({b['target'].split('/')[-1].casefold() for edge in partitions['validation'] for b in edge['bindings']} - trained_names), 'validation_final_MAE': mean(abs(p['final_success'] - observed[p['edge_uid'], p['bank_uid']]['final']['binding_contract_success']) for p in predictions),
                   'validation_first_MAE': mean(abs(p['first_success'] - observed[p['edge_uid'], p['bank_uid']]['first_try_success']) for p in predictions),
                   'certificate': False}
    evaluation = run_trials(agent, [(edge, method, retrieve(edge, bank)) for edge in partitions['development_evaluation'] for method, bank in chosen.items()], args.output, 'development_evaluation')
    agent.close()
    summary = {'main': summarize(evaluation), 'validation': diagnostics,
        'history_trials': len(history), 'document_reference_demonstrations': len(demonstrations), 'intervention_train_trials': len(train), 'validation_trials': len(validation),
        'evaluation_trials': len(evaluation), 'pool_notes': len(pool), 'candidate_banks': len(banks),
        'elapsed_seconds': perf_counter() - started, 'scope': protocol['scope'],
        'all_complete_business_goals': 'unknown', 'reserved_workflows_executed': 0}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
