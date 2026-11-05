"""Post-selection K=8 attribution and replay diagnostics for the frozen finite sandbox.

This is an evaluation-only CausalCompose reference, not a real-provider suite.
Previously frozen local count models never consume these reference outcomes.
"""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import combinations, product
import json
from math import comb
from pathlib import Path
from random import Random

import numpy as np
from scipy.stats import spearmanr

from .experiments import _write_csv
from .mechanisms import FiniteMechanisms, without_decision_history, without_memory_interactions
from .stateful import FAMILIES, WORKFLOWS, StateTask, oracle_probability, replay, run_episode
from .user_data import digest

POOL = ('lookup_key', 'transform_rule', 'legacy_contract', 'new_contract',
        'auth_pair_a', 'auth_pair_b', 'conflict', 'noop')


def attribution_weights(k):
    """Exact subset-form Shapley weights; rows map coalition utilities to phi."""
    weights = np.zeros((k, 1 << k))
    for i in range(k):
        for mask in range(1 << k):
            if mask & (1 << i):
                continue
            coefficient = 1 / (k * comb(k - 1, mask.bit_count()))
            weights[i, mask] -= coefficient
            weights[i, mask | (1 << i)] += coefficient
    return weights


def signed_pairs(values, k, threshold):
    results = []
    for i, j in combinations(range(k), 2):
        effect = values[(1 << i) | (1 << j)] - values[1 << i] - values[1 << j] + values[0]
        sign = 0 if abs(effect) < threshold else (1 if effect > 0 else -1)
        results.append((i, j, float(effect), sign))
    return results


def pair_f1(truth, predicted):
    actual = [r[3] for r in truth]; estimate = [r[3] for r in predicted]
    tp = sum(a != 0 and a == b for a, b in zip(actual, estimate))
    fp = sum(b != 0 and b != a for a, b in zip(actual, estimate))
    fn = sum(a != 0 and a != b for a, b in zip(actual, estimate))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None


def monte_carlo_phi(values, k, permutations, seed):
    rng = Random(seed); answer = np.zeros(k)
    for _ in range(permutations):
        order = list(range(k)); rng.shuffle(order); mask = 0
        for i in order:
            following = mask | (1 << i)
            answer[i] += (values[following] - values[mask]) / permutations
            mask = following
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=Path('results/stateful_v1'))
    parser.add_argument('--output', type=Path, default=Path('results/paper_attribution_v1'))
    parser.add_argument('--repetitions', type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    if args.repetitions < 2:
        raise ValueError('at least two independent reference repetitions required')
    frozen = json.loads((args.training / 'frozen_protocol.json').read_text())
    for name, expected in frozen['source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('training source changed: ' + name)
    seeds = frozen['executed_seeds']
    inputs = {}
    for seed in seeds:
        directory = args.training / f'seed_{seed}'
        for name in ('decision_observations.json', 'api_observations.json', 'frozen_banks.json', 'raw_episodes.csv'):
            inputs[str(directory / name)] = digest(directory / name)
    tasks = [StateTask(f'{family}-{epoch}-{template}', family, WORKFLOWS[template], epoch, 423,
        template) for family, epoch, template in product(FAMILIES, ('source', 'target'), ('reordered', 'repair_path'))]
    args.output.mkdir(parents=True)
    protocol = {'scope': 'evaluation-only finite CausalCompose reference', 'pool': POOL,
        'contexts': [asdict(t) for t in tasks], 'training_directory': str(args.training.resolve()),
        'training_seeds': seeds, 'reference_seed': 424, 'reference_repetitions': args.repetitions,
        'coalitions': 256, 'paired_noise_across_coalitions': True, 'interaction_sign_threshold': .015,
        'MC_permutation_sensitivity': [16, 64, 256],
        'MC_cost_scope': 'frozen model queries only; not a matched service-call training-budget experiment',
        'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in
            ('paper_attribution.py', 'stateful.py', 'mechanisms.py')}, 'training_input_sha256': inputs,
        'no_retuning_or_reference_training': True,
        'certificates': 'reference standard errors are empirical Monte Carlo uncertainty, not uniform certificates'}
    (args.output / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2))
    exposures = [tuple(POOL[i] for i in range(8) if mask & (1 << i)) for mask in range(256)]
    weights = attribution_weights(8)
    oracle = np.zeros((len(tasks), 256))
    samples = np.zeros((len(tasks), args.repetitions, 256), dtype=np.uint8)
    reference_calls = 0
    for index, task in enumerate(tasks):
        for mask, exposure in enumerate(exposures):
            oracle[index, mask] = oracle_probability(task, exposure)
        rng = Random(protocol['reference_seed'] + index)
        for repetition in range(args.repetitions):
            replica = replace(task, seed=rng.randrange(2**31), uid=f'{task.uid}-r{repetition}')
            for mask, exposure in enumerate(exposures):
                trace = run_episode(replica, exposure)
                samples[index, repetition, mask] = trace.outcome
                reference_calls += trace.calls
        print(json.dumps({'reference_context': task.uid, 'completed': index + 1}), flush=True)
    np.savez_compressed(args.output / 'reference_outcomes.npz', oracle=oracle, outcomes=samples)
    reference_rows = []; attribution = []; pair_rows = []; sensitivity = []
    for index, task in enumerate(tasks):
        empirical = samples[index].mean(axis=0)
        standard_error = samples[index].std(axis=0, ddof=1) / np.sqrt(args.repetitions)
        for mask in range(256):
            reference_rows.append({'context': task.uid, 'mask': mask, 'exposure': '|'.join(exposures[mask]),
                'exact_probability': oracle[index, mask], 'reference_mean': empirical[mask],
                'reference_standard_error': standard_error[mask], 'reference_repetitions': args.repetitions})
    for seed in seeds:
        directory = args.training / f'seed_{seed}'
        full = FiniteMechanisms(json.loads((directory / 'decision_observations.json').read_text()),
            json.loads((directory / 'api_observations.json').read_text()), delta=frozen['config']['confidence_delta'])
        variants = {'Full finite mechanism': full, 'Without decision history': without_decision_history(full),
            'Without memory interactions': without_memory_interactions(full)}
        for method, model in variants.items():
            for index, task in enumerate(tasks):
                values = np.array([model.predict(task, exposure).mean for exposure in exposures])
                phi = weights @ values; exact_phi = weights @ oracle[index]
                empirical_phi = samples[index] @ weights.T
                phi_se = empirical_phi.std(axis=0, ddof=1) / np.sqrt(args.repetitions)
                reference_pairs = signed_pairs(oracle[index], 8, protocol['interaction_sign_threshold'])
                predicted_pairs = signed_pairs(values, 8, protocol['interaction_sign_threshold'])
                corr = float(spearmanr(exact_phi, phi).statistic)
                attribution.append({'seed': seed, 'method': method, 'context': task.uid, 'family': task.family,
                    'epoch': task.epoch, 'topology': task.template, 'Shapley_MAE': float(np.mean(abs(phi - exact_phi))),
                    'Shapley_rank_correlation': corr if np.isfinite(corr) else None,
                    'signed_pair_F1': pair_f1(reference_pairs, predicted_pairs),
                    'reference_active_pairs': sum(p[3] != 0 for p in reference_pairs),
                    'coalition_probability_MAE': float(np.mean(abs(values - oracle[index]))),
                    'reference_MC_Shapley_MAE': float(np.mean(abs(empirical_phi.mean(axis=0) - exact_phi))),
                    'mean_reference_Shapley_SE': float(phi_se.mean()),
                    'noop_exact_credit': float(exact_phi[-1]), 'noop_predicted_credit': float(phi[-1])})
                for truth, predicted in zip(reference_pairs, predicted_pairs):
                    i, j, effect, sign = truth
                    pair_rows.append({'seed': seed, 'method': method, 'context': task.uid,
                        'memory_a': POOL[i], 'memory_b': POOL[j], 'reference_effect': effect,
                        'predicted_effect': predicted[2], 'reference_sign': sign, 'predicted_sign': predicted[3]})
                if method == 'Full finite mechanism':
                    for permutations in protocol['MC_permutation_sensitivity']:
                        approximate = monte_carlo_phi(values, 8, permutations, seed + 425)
                        sensitivity.append({'seed': seed, 'context': task.uid, 'permutations': permutations,
                            'MC_error_vs_exact_model_Shapley': float(np.mean(abs(approximate - phi))),
                            'total_MAE_vs_exact_environment': float(np.mean(abs(approximate - exact_phi))),
                            'model_queries_with_duplicates': permutations * 9, 'additional_service_calls': 0})
        print(json.dumps({'trained_seed': seed, 'completed': True}), flush=True)
    replay_rows = []; controls = []; rng = Random(426)
    for index, task in enumerate(tasks):
        for trial in range(32):
            replica = replace(task, uid=f'control-{task.uid}-{trial}', seed=rng.randrange(2**31))
            exposure = exposures[rng.randrange(256)]
            # A/A, a true placebo, and a stored record explicitly excluded from exposure.
            a, aa = replay(replica, exposure, exposure)
            _, placebo = replay(replica, exposure, set(exposure) | {'noop'})
            unexposed = run_episode(replica, exposure)
            try:
                run_episode(replace(replica, epoch='target' if replica.epoch == 'source' else 'source'),
                    exposure, prefix=a, start=1)
                rejected = False
            except ValueError:
                rejected = True
            controls.append({'context': task.uid, 'trial': trial,
                'AA_same_state': int(a.state_hash == aa.state_hash), 'noop_same_state': int(a.state_hash == placebo.state_hash),
                'unexposed_stored_note': 'hidden_fix', 'unexposed_same_state': int(a.state_hash == unexposed.state_hash),
                'changed_checkpoint_rejected': int(rejected)})
            memory = rng.choice(POOL); background = set(exposure) - {memory}
            c, incremental = replay(replica, background, background | {memory})
            f, complete = replay(replica, background, background | {memory}, incremental=False)
            replay_rows.append({'context': task.uid, 'trial': trial, 'memory': memory,
                'control_state_sha256': c.state_hash, 'treatment_state_sha256': incremental.state_hash,
                'full_treatment_sha256': complete.state_hash, 'checkpoint_sha256': c.checkpoint_hash,
                'incremental_calls': c.calls + incremental.calls, 'full_calls': f.calls + complete.calls,
                'same_state': int(incremental.state_hash == complete.state_hash),
                'same_outcome': int(incremental.outcome == complete.outcome), 'reused_steps': incremental.reused})
    for name, table in [('coalition_reference.csv', reference_rows), ('attribution.csv', attribution),
            ('interactions.csv', pair_rows), ('MC_sensitivity.csv', sensitivity),
            ('negative_controls.csv', controls), ('paired_replay.csv', replay_rows)]:
        _write_csv(args.output / name, table)
    summary = []
    for method in sorted({r['method'] for r in attribution}):
        subset = [r for r in attribution if r['method'] == method]
        row = {'method': method, 'trained_seeds': len(seeds), 'contexts_per_seed': len(tasks)}
        for metric in ('Shapley_MAE', 'Shapley_rank_correlation', 'signed_pair_F1', 'coalition_probability_MAE'):
            means = [np.mean([r[metric] for r in subset if r['seed'] == seed and r[metric] is not None]) for seed in seeds]
            row[metric] = float(np.mean(means)); row[metric + '_seed_SD'] = float(np.std(means, ddof=1))
        summary.append(row)
    incremental_calls = sum(r['incremental_calls'] for r in replay_rows)
    full_calls = sum(r['full_calls'] for r in replay_rows)
    report = {'table': summary, 'reference_episodes': int(samples.size), 'reference_tool_calls': reference_calls,
        'replay_pairs': len(replay_rows), 'replay_state_mismatches': sum(not r['same_state'] for r in replay_rows),
        'replay_outcome_mismatches': sum(not r['same_outcome'] for r in replay_rows),
        'incremental_calls': incremental_calls, 'full_calls': full_calls,
        'call_savings_fraction': 1 - incremental_calls / full_calls,
        'negative_control_pairs': len(controls), 'negative_control_failures': sum(
            not all(r[k] for k in ('AA_same_state', 'noop_same_state', 'unexposed_same_state', 'changed_checkpoint_rejected'))
            for r in controls), 'full_paper_CREDIT': False, 'provider_drift': False,
        'metric_scope': 'Full finite counts; interactions only at empty background, not all higher-order causal interactions',
        'F1_missing_if_no_active_pairs': True, 'results_are_post_selection_diagnostics': True}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    (args.output / 'artifact_sha256.json').write_text(json.dumps({p.name: digest(p) for p in args.output.iterdir()
        if p.is_file()}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
