"""Reconstruct budgets, learned kernels, retained banks and every grid outcome."""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from . import budget_grid_v2 as grid
from .budget_mechanisms_v2 import BatchForecast, design
from .paired_statistics import summarize
from .shared_mechanisms import SharedMechanisms
from .stateful import MEMORIES, ROLES, oracle_probability, retrieve, run_episode
from .user_data import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def restore(policy, api, dump):
    model = SharedMechanisms(policy, api)
    recorded = {tuple(k): np.array(p) for k, p in zip(dump['policy_keys'], dump['policy_probabilities'])}
    require(set(recorded) == set(model.policy), 'fitted key coverage differs')
    for role in ROLES:
        fit = dump['fits'][role]
        require(fit['converged'] and max(fit['iterations']) < 2000, 'unconverged conditional fit')
        rows = [r for r in policy if r['key'][0] == role]
        x, groups = design([tuple(r['key']) for r in rows], role, dump['degree'])
        require(same(groups, fit['groups']), 'interaction degree/group membership differs')
        probabilities = softmax(x @ np.array(fit['coef']).T + np.array(fit['intercept']), axis=1)
        for row, probability in zip(rows, probabilities):
            key = tuple(row['key']); ordered = np.zeros(4)
            ordered[np.array(fit['classes'])] = probability
            require(np.allclose(ordered, recorded[key], rtol=1e-10, atol=1e-12), 'conditional coefficients differ')
            require(np.isclose(ordered.sum(), 1), 'conditional kernel is not normalized')
            model.policy[key] = ordered * sum(row['counts'])
    model.certificate_valid = False
    return model


def verify(root):
    protocol = grid.read(root / 'frozen_protocol.json')
    require(not protocol['pilot'], 'full-grid audit requires the full protocol')
    for name, expected in protocol['source_sha256'].items():
        require(digest(root / 'source' / name) == expected, 'snapshot changed: ' + name)
        require(digest(Path(grid.__file__).with_name(name)) == expected, 'source changed: ' + name)
    for name, expected in protocol['history_sha256'].items():
        require(digest(Path(name)) == expected, 'history changed: ' + name)
    for name, expected in grid.read(root / 'evaluation_freeze.json').items():
        require(digest(root / name) == expected, 'pre-evaluation artifact changed: ' + name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(protocol['tokenizer_directory'], local_files_only=True)
    history_root = Path(protocol['history_directory'])
    notes = grid.memories(grid.read(history_root / 'seed_7' / 'decision_observations.json'), tokenizer)
    require(notes == grid.read(root / 'memories.json'), 'candidate provenance/token counts differ')
    tokens = {r['uid']: r['tokens'] for r in notes}
    require(sum(tokens.values()) == protocol['memory_pool_tokens'], 'pool tokens differ')
    admissible = {C: grid.feasible_banks(tokens, C) for C in protocol['C']}
    cases = {}; cost_rows = []; probe_calls = 0; audited_probes = 0; coefficient_predictions = 0
    selection_checks = 0; max_iterations = 0
    for seed in protocol['seeds']:
        history = history_root / f"seed_{protocol['historical_seed_mapping'][str(seed)]}"
        base_policy = grid.read(history / 'decision_observations.json'); base_api = grid.read(history / 'api_observations.json')
        traces = grid.calibration(history)
        contexts = [grid.task(t) for t in protocol['future_contexts'][str(seed)]]
        for C, K in product(protocol['C'], protocol['K']):
            parent = root / f'seed_{seed}' / f'C{C}_K{K}'
            probes = grid.read(parent / 'probes.json')
            rebuilt = grid.collect_probes(seed, tokens, C, K, admissible[C], max(protocol['R']))
            require(same(probes, rebuilt), 'randomized probe assignment or physical replay differs')
            probe_calls += sum(r['new_calls'] for r in probes); audited_probes += len(probes)
            for R in protocol['R']:
                selected, used = grid.prefix(probes, R)
                require(used <= R and used == sum(r['learning_calls'] + r['probe_audit_full_replay_calls'] for r in selected),
                    'probe cap does not include all training-time verification calls')
                policy, api = grid.augmented(base_policy, base_api, selected)
                added_decisions = sum(r['samples'] for r in policy) - sum(r['samples'] for r in base_policy)
                added_api = sum(r['samples'] for r in api) - sum(r['samples'] for r in base_api)
                learning_calls = sum(r['learning_calls'] for r in selected)
                require(added_decisions == added_api == learning_calls, 'reused or audit observations counted twice')
                for d in protocol['d']:
                    cell = parent / f'R{R}_d{d}'
                    dump = grid.read(cell / 'fitted_policy.json')
                    require(dump['degree'] == d, 'interaction order differs')
                    model = restore(policy, api, dump); forecast = BatchForecast(model)
                    error = forecast.calibrate(traces)
                    require(np.isclose(error, dump['calibration_error'], rtol=1e-10, atol=1e-12), 'past-only calibration differs')
                    coefficient_predictions += len(policy)
                    max_iterations = max(max_iterations, *(max(f['iterations']) for f in dump['fits'].values()))
                    banks = {m: tuple(b) for m, b in grid.read(cell / 'frozen_banks.json').items()}
                    require(all(set(b) <= set(MEMORIES) and sum(tokens[m] for m in b) <= C for b in banks.values()),
                        'storage or candidate constraint failed')
                    selection = grid.read(cell / 'selection.json')
                    # Independently rerun all nine selectors at the predeclared
                    # anchor, once for every seed; other banks get full hash,
                    # constraint, forecast and execution reconstruction.
                    if (C,K,R,d) == (2048,8,64,2):
                        rebuilt_banks, rebuilt_selection = grid.choose(forecast, contexts, tokens, C, K, seed + 50)
                        require(banks == rebuilt_banks, 'anchor retention differs')
                        require(same(selection, rebuilt_selection) or all(np.isclose(
                            selection['selected_forecasts'][m]['objective'], rebuilt_selection['selected_forecasts'][m]['objective'])
                            for m in banks), 'anchor forecast objective differs')
                        selection_checks += 1
                    cost = grid.read(cell / 'costs.json')
                    require(cost['new_probe_calls'] == used and cost['decision_queries_added'] == learning_calls,
                        'recorded incremental budget differs')
                    require(cost['common_historical_API_calls'] == 132224, 'historical API costs hidden or changed')
                    cases[seed,C,K,R,d] = (banks, forecast, contexts)
                    cost_rows.append({'seed':seed,'C':C,'K':K,'R':R,'d':d,'actual_extra_calls':used,
                        'learning_observations':learning_calls,'probe_pairs':len(selected)})
        print(json.dumps({'audit_prepared_seed':seed,'complete':True}), flush=True)
    expected_cases = set(product(protocol['seeds'], protocol['C'], protocol['K'], protocol['R'], protocol['d']))
    require(set(cases) == expected_cases, 'full grid incomplete')
    task_index = {seed: {t['uid']: grid.task(t) for t in protocol['future_tasks'][str(seed)]} for seed in protocol['seeds']}
    counterfactual_cache = {}; groups = defaultdict(list); future_calls = 0; rows_checked = 0
    tasks_seen = defaultdict(set); method_set = set(next(iter(cases.values()))[0])
    for row in grid.csv_rows(root / 'raw_episodes.csv'):
        seed,C,K,R,d = [int(row[n]) for n in ('seed','C','K','R','d')]
        case = seed,C,K,R,d; banks,forecast,contexts = cases[case]
        t = task_index[seed][row['task']]; bank = banks[row['method']]
        exposure = retrieve(t, bank, K)
        require(row['bank'] == '|'.join(bank) and row['exposure'] == '|'.join(exposure), 'future bank/retrieval differs')
        require(int(row['bank_tokens']) == sum(tokens[m] for m in bank) <= C, 'future token budget differs')
        require(int(row['exposure_count']) == len(exposure) <= K, 'future exposure cap differs')
        require(int(row['exposure_tokens']) == sum(tokens[m] for m in exposure) <= C, 'future exposure tokens differ')
        key = seed,t.uid,exposure
        if key not in counterfactual_cache:
            counterfactual_cache[key] = run_episode(t, exposure)
        trace = counterfactual_cache[key]
        require(trace.state_hash == row['state_sha256'] and trace.checkpoint_hash == row['checkpoint_sha256'], 'future replay state differs')
        require(trace.outcome == int(row['success']) and trace.calls == int(row['api_calls']), 'future verifier/cost differs')
        require(len(trace.steps) == int(row['observed_length']), 'observed chain length differs')
        match = case + (row['method'],)
        require(t.uid not in tasks_seen[match], 'duplicate future outcome')
        tasks_seen[match].add(t.uid)
        numeric = {**row,'seed':seed,'success':trace.outcome}
        groups[C,K,R,d].append(numeric); rows_checked += 1; future_calls += trace.calls
    for case in cases:
        for method in method_set:
            require(tasks_seen[case+(method,)] == set(task_index[case[0]]), 'missing paired task coverage')
    stored = {(int(r['C']),int(r['K']),int(r['R']),int(r['d']),r['method']):r for r in grid.csv_rows(root / 'comparison.csv')}
    paired = {(int(r['C']),int(r['K']),int(r['R']),int(r['d']),r['baseline']):r for r in grid.csv_rows(root / 'paired_statistics.csv')}
    for settings, rows in groups.items():
        table, comparisons = summarize(rows, 'CREDIT-finite-empirical', repeats=2000, random_seed=431)
        for value in table:
            saved = stored[settings+(value['method'],)]
            require(all(np.isclose(float(saved[n]),value[n]) for n in ('vsr','ci_low','ci_high','episodes','independent_seeds')),
                'pooled metric or cluster uncertainty differs')
        for value in comparisons:
            saved = paired[settings+(value['baseline'],)]
            require(all(np.isclose(float(saved[n]),value[n]) for n in ('delta_vsr','ci_low','ci_high','seed_sign_flip_p','holm_p')),
                'paired/Holm statistics differ')
    truth_cache = {}; diagnostics_checked = 0
    context_index = {seed: {t['uid']:grid.task(t) for t in protocol['future_contexts'][str(seed)]} for seed in protocol['seeds']}
    for row in grid.csv_rows(root / 'forecast_diagnostics.csv'):
        seed,C,K,R,d = [int(row[n]) for n in ('seed','C','K','R','d')]
        banks,forecast,contexts = cases[seed,C,K,R,d]
        t = context_index[seed][row['context']]; exposure = retrieve(t,banks[row['method']],K)
        key = t.family,t.epoch,t.roles,t.initial_state,exposure
        if key not in truth_cache:
            truth_cache[key] = oracle_probability(t,exposure)
        prediction,support = forecast.predict(t,[exposure]); truth = truth_cache[key]
        require(np.isclose(float(row['prediction']),prediction[0]) and np.isclose(float(row['exact_evaluation_only']),truth),
            'post-test forecast diagnostic differs')
        require(np.isclose(float(row['absolute_error']),abs(prediction[0]-truth)), 'diagnostic error differs')
        require((row['supported']=='True') == bool(support[0]), 'support disclosure differs')
        diagnostics_checked += 1
    summary = grid.read(root / 'summary.json')
    require(summary['future_episodes'] == rows_checked and summary['actual_future_tool_calls'] == future_calls,
        'total measured episodes/calls differ')
    from .experiments import _write_csv
    _write_csv(root / 'audited_costs.csv', cost_rows)
    report = {'verified':True,'grid_cells':len(groups),'seed_cells':len(cases),'future_rows_reconstructed':rows_checked,
        'unique_counterfactuals_reexecuted':len(counterfactual_cache),'future_calls_in_original_run':future_calls,
        'probe_pairs_reexecuted':audited_probes,'unique_max_R_collection_calls':probe_calls,
        'conditional_kernel_rows_reconstructed':coefficient_predictions,'forecast_diagnostics_recomputed':diagnostics_checked,
        'anchor_full_selector_reconstructions':selection_checks,'other_selectors':'frozen hashes, constraints, forecasts and execution checked',
        'all_conditional_fits_converged':True,'maximum_optimizer_iterations':max_iterations,
        'full_paper_CREDIT':False,'budget_scope':protocol['budget_scope'],
        'verifier_source_sha256':digest(Path(__file__))}
    grid.write(root / 'execution_audit.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    print(json.dumps(verify(parser.parse_args().results),indent=2))
