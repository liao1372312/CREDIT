"""Full C/K/R/d sensitivity grid for the finite controlled CREDIT reference.

Extra-call caps apply after an explicitly charged, common historical fixture
dataset. This is not a claim of learning all kernels from 16 total calls.
All configurations freeze before any new future outcome is collected.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path
from random import Random
import shutil

import numpy as np

from .budget_mechanisms_v2 import BankSearch, BatchForecast, fit_order
from .experiments import _write_csv
from .paired_statistics import summarize
from .stateful import (ACTIONS, ARRIVAL, DEPENDENCIES, FAMILIES, MEMORIES, NEIGHBORS,
    RECENCY, StateTask, decision_key, oracle_probability, replay, retrieve, run_episode)
from .user_data import digest

MODEL = '/home/lxc/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
GRAPHS = {'short_ready_submit': (('submit',), 7),
    'medium_authorize_before_transform': (('lookup', 'authorize', 'transform', 'submit'), 0),
    'long_repeated_state_mutation': (('lookup', 'transform', 'lookup', 'transform',
        'authorize', 'transform', 'authorize', 'authorize', 'submit'), 0)}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def task(value):
    return StateTask(**{**value, 'roles': tuple(value['roles'])})


def future_tasks(seed, repetitions=4):
    rng = Random(seed + 10000)
    contexts, episodes = [], []
    for family, epoch, graph in product(FAMILIES, ('source', 'target'), GRAPHS):
        roles, initial = GRAPHS[graph]
        uid = f'{family}-{epoch}-{graph}'
        context = StateTask(uid, family, roles, epoch, 0, graph, initial)
        contexts.append(context)
        for repetition in range(repetitions):
            episodes.append(StateTask(f'future-{seed}-{uid}-{repetition}', family, roles, epoch,
                rng.randrange(2**31), graph, initial))
    return contexts, episodes


def memories(policy, tokenizer):
    indexed = {tuple(r['key']): r for r in policy}
    result = []
    for memory in MEMORIES:
        observations = []
        for role in sorted(DEPENDENCIES[memory]):
            bit = 1 << NEIGHBORS[role].index(memory)
            for missing in ((0,) if role == 'lookup' else (0, 1)):
                for transient in (0, 1):
                    for mask in (0, bit):
                        key = role, missing, transient, mask
                        row = indexed[key]
                        observations.append({'role': role, 'missing': missing, 'transient': transient,
                            'exposed_candidate': mask != 0, 'action_counts': row['counts'], 'samples': row['samples']})
        text = json.dumps({'candidate': memory, 'dependent_operations': sorted(DEPENDENCIES[memory]),
            'historical_action_order': ACTIONS, 'recorded_local_fixtures': observations,
            'interpretation': 'Recorded action counts, not a causal-effect label or future service response. '
                'State transitions and operation outcomes must be checked in the current workflow.'},
            ensure_ascii=False, separators=(',', ':'))
        result.append({'uid': memory, 'text': text, 'tokens': len(tokenizer.encode(text, add_special_tokens=False)),
            'provenance': 'frozen seed-7 historical decision fixture rows; no future labels', 'roles': sorted(DEPENDENCIES[memory])})
    return result


def feasible_banks(tokens, capacity):
    names = tuple(tokens)
    return [tuple(names[i] for i in range(len(names)) if mask & (1 << i))
        for mask in range(1 << len(names))
        if sum(tokens[names[i]] for i in range(len(names)) if mask & (1 << i)) <= capacity]


def collect_probes(seed, tokens, capacity, depth, admissible, limit=256):
    rng = Random(seed + capacity * 17 + depth * 31)
    rows, used = [], 0
    while limit - used >= 16:
        bank = rng.choice(admissible)
        eligible = [m for m in MEMORIES if sum(tokens[x] for x in set(bank) ^ {m}) <= capacity]
        memory = rng.choice(eligible); following = tuple(sorted(set(bank) ^ {memory}))
        family = rng.choice(FAMILIES); epoch = rng.choice(('source', 'target'))
        replica = StateTask(f'probe-{seed}-{capacity}-{depth}-{len(rows)}', family,
            ('lookup', 'transform', 'authorize', 'submit'), epoch, rng.randrange(2**31), 'history_chain')
        control, treatment = retrieve(replica, bank, depth), retrieve(replica, following, depth)
        a, b = replay(replica, control, treatment)
        _, full = replay(replica, control, treatment, incremental=False)
        if b.state_hash != full.state_hash:
            raise ValueError('invalid replay pair, cannot use its causal effect')
        learning_calls = a.calls + b.calls
        audit_calls = a.calls + full.calls
        cost = learning_calls + audit_calls
        rows.append({'task': asdict(replica), 'bank': bank, 'following_bank': following, 'toggled_memory': memory,
            'assignment_propensity': 1 / (len(admissible) * len(eligible) * len(FAMILIES) * 2),
            'task_noise_seed_probability': 1 / 2**31, 'used_before': used, 'new_calls': cost,
            'learning_calls': learning_calls,
            'control': asdict(a), 'treatment': asdict(b), 'full_treatment_sha256': full.state_hash,
            'probe_audit_full_replay_calls': audit_calls})
        used += cost
    return rows


def prefix(rows, budget):
    selected, used = [], 0
    for row in rows:
        if budget - used < 16:
            break
        selected.append(row); used += row['new_calls']
    return selected, used


def augmented(policy, api, probes):
    policy, api = deepcopy(policy), deepcopy(api)
    decisions = {tuple(r['key']): r for r in policy}; responses = {tuple(r['key']): r for r in api}
    for row in probes:
        for side in ('control', 'treatment'):
            trace = row[side]
            # A reused prefix is one recorded observation, not another sample.
            for step in trace['steps'][trace['reused']:]:
                pkey = decision_key(step['role'], step['state_before'], step['previous_status'], trace['exposure'])
                decisions[pkey]['counts'][ACTIONS.index(step['request_mode'])] += 1
                decisions[pkey]['samples'] += 1
                key = trace['task']['epoch'], trace['task']['family'], step['role'], step['state_before'], step['request_mode']
                counts = responses[key]['counts']; label = f"{step['response_status']}:{step['state_after']}"
                counts[label] = counts.get(label, 0) + 1; responses[key]['samples'] += 1
    return policy, api


def calibration(directory):
    manifest = read(directory / 'split_manifest.json')
    tasks = {t['uid']: task(t) for t in manifest['calibration']}
    traces = []
    for row in csv_rows(directory / 'calibration_episodes.csv'):
        trace = run_episode(tasks[row['task']], row['exposure'].split('|') if row['exposure'] else ())
        if trace.state_hash != row['state_hash'] or trace.outcome != int(row['outcome']):
            raise ValueError('historical calibration differs')
        traces.append(trace)
    return traces


def choose(forecast, contexts, tokens, C, K, seed):
    search = BankSearch(forecast, contexts, tokens, C, K)
    empirical, empirical_audit = search.joint()
    mean, mean_audit = search.joint('mean_objective')
    loo, shapley, scores = search.scalar(seed)
    search.evaluate([(), *[(m,) for m in MEMORIES]])
    empty = search.cache[()]['objective']
    unary = {m: search.cache[(m,)]['objective'] - empty if (m,) in search.cache else -np.inf for m in MEMORIES}
    greedy = search.pack(sorted(MEMORIES, key=lambda m: (-unary[m], m)), positive_scores=unary)
    scoped = ['legacy_contract', 'new_contract', *[m for m in ARRIVAL if m not in ('legacy_contract', 'new_contract')]]
    banks = {'NoMem': (), 'FIFO': search.pack(ARRIVAL),
        'Recency': search.pack(sorted(MEMORIES, key=RECENCY.get, reverse=True)),
        'Version-label heuristic': search.pack(scoped), 'Model-LOO': loo, 'Model-Shapley': shapley,
        'CREDIT-finite-empirical': empirical, 'CREDIT-finite-mean': mean, 'Without joint selection': greedy}
    search.evaluate(banks.values())
    return banks, {'joint_empirical': empirical_audit, 'joint_mean': mean_audit, 'scalar': scores,
        'selected_forecasts': {m: search.cache[b] for m, b in banks.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=Path('results/stateful_v1'))
    parser.add_argument('--model', type=Path, default=Path(MODEL), help='tokenizer only; no LLM generation')
    parser.add_argument('--output', type=Path, default=Path('results/budget_grid_v2'))
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    old = read(args.history / 'frozen_protocol.json')
    for name, expected in old['source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('historical source differs: ' + name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    notes = memories(read(args.history / 'seed_7' / 'decision_observations.json'), tokenizer)
    tokens = {n['uid']: n['tokens'] for n in notes}
    seeds = [101, 103, 107, 109, 113] if not args.pilot else [101]
    mapping = dict(zip(seeds, old['executed_seeds']))
    capacities, depths, budgets, orders = ([512, 2048, 8192], [4, 8, 16], [16, 64, 256], [1, 2, 3])
    if args.pilot:
        capacities, depths, budgets, orders = [512], [4], [16], [1, 2]
    args.output.mkdir(parents=True); (args.output / 'source').mkdir()
    sources = ['budget_grid_v2.py', 'budget_mechanisms_v2.py', 'stateful.py', 'mechanisms.py', 'shared_mechanisms.py',
        'policy.py', 'paired_statistics.py', 'user_data.py', 'experiments.py']
    historical_files = ['decision_observations.json', 'api_observations.json', 'split_manifest.json',
        'calibration_episodes.csv', 'report.json']
    protocol = {'suite': 'controlled finite token/probe/order grid', 'C': capacities, 'K': depths, 'R': budgets, 'd': orders,
        'seeds': seeds, 'historical_seed_mapping': mapping, 'history_directory': str(args.history.resolve()),
        'history_sha256': {str(args.history / f'seed_{s}' / n): digest(args.history / f'seed_{s}' / n)
            for s in mapping.values() for n in historical_files},
        'source_sha256': {n: digest(Path(__file__).with_name(n)) for n in sources},
        'tokenizer_directory': str(args.model.resolve()), 'memory_pool_tokens': sum(tokens.values()),
        'fixed_hyperparameters': {'logistic_C': 1., 'max_iter': 2000, 'tol': 1e-6, 'mixture_L1_radius': .2,
            'storage_cost_per_token': .000005, 'MC_permutations': 32, 'joint_search_passes': 4},
        'budget_scope': 'R new local replay calls per history block, in addition to charged common complete fixture history',
        'nested_R': 'same randomized valid probe prefix, stopping when fewer than 16 calls remain; only that prefix enters fitting',
        'counterfactual_model_queries': 'no service calls; Model-LOO/Shapley query archived coalitions then pack within C',
        'probe_full_replay_audit_cost': 'charged inside R total calls; audit repeats excluded from fitting to avoid duplicate observations',
        'scope_limits': 'categorical finite API and polynomial policy, no semantic payloads, joint neural losses or real provider drift',
        'future_label_usage': 'all cells frozen before first future outcome; oracle only after all future scoring',
        'numerical_gate': 'every conditional fit must converge before any future scoring; v1 aborted at preparation, no new future labels inspected',
        'pilot': args.pilot, 'statistics': 'five new execution noise seeds; paired seed/family bootstrap and within-cell Holm',
        'grid_hypotheses': 'exploratory sensitivity, no cross-grid adjusted advantage claim',
        'future_contexts': {str(s): [asdict(t) for t in future_tasks(s, 1 if args.pilot else 4)[0]] for s in seeds},
        'future_tasks': {str(s): [asdict(t) for t in future_tasks(s, 1 if args.pilot else 4)[1]] for s in seeds}}
    write(args.output / 'frozen_protocol.json', protocol); write(args.output / 'memories.json', notes)
    for name in sources:
        shutil.copy2(Path(__file__).with_name(name), args.output / 'source' / name)
    admissible = {C: feasible_banks(tokens, C) for C in capacities}
    prepared = []
    for seed in seeds:
        history = args.history / f'seed_{mapping[seed]}'
        base_policy, base_api = read(history / 'decision_observations.json'), read(history / 'api_observations.json')
        traces = calibration(history); contexts, tasks = future_tasks(seed, 1 if args.pilot else 4)
        report = read(history / 'report.json')
        historical_cost = {k: report[k] for k in ('policy_observations', 'source_api_probe_calls', 'target_api_probe_calls', 'calibration_api_calls')}
        for C, K in product(capacities, depths):
            parent = args.output / f'seed_{seed}' / f'C{C}_K{K}'; parent.mkdir(parents=True)
            probes = collect_probes(seed, tokens, C, K, admissible[C], max(budgets))
            write(parent / 'probes.json', probes)
            for R in budgets:
                selected, used = prefix(probes, R)
                policy, api = augmented(base_policy, base_api, selected)
                for d in orders:
                    cell = parent / f'R{R}_d{d}'; cell.mkdir()
                    model, fitted = fit_order(policy, api, d)
                    forecast = BatchForecast(model); calibration_error = forecast.calibrate(traces)
                    banks, selection = choose(forecast, contexts, tokens, C, K, seed + 50)
                    model_dump = {'degree': d, 'policy_keys': [r['key'] for r in policy],
                        'policy_probabilities': [(model.policy[tuple(r['key'])] / sum(r['counts'])).tolist() for r in policy],
                        'fits': fitted, 'calibration_error': calibration_error}
                    write(cell / 'fitted_policy.json', model_dump)
                    write(cell / 'frozen_banks.json', {m: list(b) for m, b in banks.items()})
                    write(cell / 'selection.json', selection)
                    write(cell / 'costs.json', {'historical_collection': historical_cost, 'new_probe_calls': used,
                        'assigned_probe_cap': R, 'probe_pairs': len(selected), 'feasible_bank_count': len(admissible[C]),
                        'decision_queries_added': sum(p['learning_calls'] for p in selected), 'common_historical_API_calls':
                            sum(historical_cost[k] for k in ('source_api_probe_calls', 'target_api_probe_calls', 'calibration_api_calls'))})
                    prepared.append((seed, C, K, R, d, cell, banks, forecast, contexts, tasks))
            print(json.dumps({'prepared_seed': seed, 'C': C, 'K': K, 'models_frozen': len(budgets)*len(orders)}), flush=True)
    # Freeze the entire sensitivity grid, not merely each method before its own run.
    frozen_files = [p for p in args.output.rglob('*.json') if 'source' not in p.parts]
    write(args.output / 'evaluation_freeze.json', {str(p.relative_to(args.output)): digest(p) for p in frozen_files})
    raw_path = args.output / 'raw_episodes.csv'
    row_fields = ['seed','C','K','R','d','method','task','family','epoch','topology','observed_length','success',
        'bank','bank_tokens','exposure','exposure_tokens','exposure_count','api_calls','state_sha256','checkpoint_sha256']
    all_rows = defaultdict(list)
    with raw_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields); writer.writeheader()
        for seed, C, K, R, d, cell, banks, forecast, contexts, tasks in prepared:
            for method, bank in banks.items():
                for t in tasks:
                    exposed = retrieve(t, bank, K); trace = run_episode(t, exposed)
                    row = {'seed': seed, 'C': C, 'K': K, 'R': R, 'd': d, 'method': method, 'task': t.uid,
                        'family': t.family, 'epoch': t.epoch, 'topology': t.template, 'observed_length': len(trace.steps),
                        'success': trace.outcome, 'bank': '|'.join(bank), 'bank_tokens': sum(tokens[m] for m in bank),
                        'exposure': '|'.join(exposed), 'exposure_tokens': sum(tokens[m] for m in exposed),
                        'exposure_count': len(exposed), 'api_calls': trace.calls, 'state_sha256': trace.state_hash,
                        'checkpoint_sha256': trace.checkpoint_hash}
                    writer.writerow(row); all_rows[C,K,R,d].append(row)
            handle.flush()
    tables, comparisons = [], []
    for (C,K,R,d), rows in all_rows.items():
        main, paired = summarize(rows, 'CREDIT-finite-empirical', repeats=2000, random_seed=431)
        tables.extend({**r, 'C': C,'K': K,'R': R,'d': d} for r in main)
        comparisons.extend({**r, 'C': C,'K': K,'R': R,'d': d} for r in paired)
    _write_csv(args.output / 'comparison.csv', tables)
    _write_csv(args.output / 'paired_statistics.csv', comparisons)
    diagnostic = []
    for seed,C,K,R,d,cell,banks,forecast,contexts,tasks in prepared:
        for method, bank in banks.items():
            exposures = [retrieve(t, bank, K) for t in contexts]
            for t, exposure in zip(contexts, exposures):
                predictions, coverage = forecast.predict(t, [exposure])
                truth = oracle_probability(t, exposure)
                diagnostic.append({'seed':seed,'C':C,'K':K,'R':R,'d':d,'method':method,'context':t.uid,
                    'family':t.family,'epoch':t.epoch,'topology':t.template,'prediction':predictions[0],
                    'exact_evaluation_only':truth,'absolute_error':abs(predictions[0]-truth),'supported':bool(coverage[0])})
    _write_csv(args.output / 'forecast_diagnostics.csv', diagnostic)
    summary = {'scope': protocol['suite'], 'full_paper_CREDIT': False, 'real_provider_drift': False,
        'grid_cells': len(all_rows), 'executed_seed_cells': len(prepared),
        'future_episodes': sum(len(r) for r in all_rows.values()), 'methods': list(prepared[0][6]),
        'common_history_API_calls_per_seed': 132224, 'common_history_decision_draws_per_seed': 86016,
        'actual_future_tool_calls': sum(r['api_calls'] for rows in all_rows.values() for r in rows),
        'primary_grid_cell': {'C':2048,'K':8,'R':64,'d':2} if not args.pilot else None,
        'primary_cell_label': 'reporting anchor frozen by design, not a tuned optimum',
        'limits': protocol['scope_limits']}
    write(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
