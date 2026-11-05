"""Check finite attribution tables, sampled reference replicas and every replay pair."""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from random import Random

import numpy as np
from scipy.stats import spearmanr

from . import paper_attribution as experiment
from .mechanisms import FiniteMechanisms, without_decision_history, without_memory_interactions
from .stateful import StateTask, oracle_probability, replay, run_episode
from .user_data import digest


def table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, label):
    if expected is None:
        if actual not in (None, ''):
            raise ValueError(label)
    elif not np.isclose(float(actual), expected, rtol=1e-10, atol=1e-12):
        raise ValueError(label)


def verify(directory):
    protocol = json.loads((directory / 'frozen_protocol.json').read_text())
    for name, expected in json.loads((directory / 'artifact_sha256.json').read_text()).items():
        if digest(directory / name) != expected:
            raise ValueError('artifact changed: ' + name)
    for name, expected in protocol['source_sha256'].items():
        if digest(Path(experiment.__file__).with_name(name)) != expected:
            raise ValueError('source changed: ' + name)
    for name, expected in protocol['training_input_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError('training input changed: ' + name)
    tasks = [StateTask(**{**t, 'roles': tuple(t['roles'])}) for t in protocol['contexts']]
    outcomes = np.load(directory / 'reference_outcomes.npz')
    oracle, samples = outcomes['oracle'], outcomes['outcomes']
    if samples.shape != (len(tasks), protocol['reference_repetitions'], 256) or not np.isin(samples, (0, 1)).all():
        raise ValueError('reference shape/alphabet differs')
    pool = tuple(protocol['pool'])
    exposures = [tuple(pool[i] for i in range(8) if mask & (1 << i)) for mask in range(256)]
    weights = experiment.attribution_weights(8)
    checked_replicas = 0
    reference = {(r['context'], int(r['mask'])): r for r in table(directory / 'coalition_reference.csv')}
    for index, task in enumerate(tasks):
        exact = np.array([oracle_probability(task, exposure) for exposure in exposures])
        if not np.allclose(exact, oracle[index], rtol=0, atol=1e-12):
            raise ValueError('exact oracle differs')
        rng = Random(protocol['reference_seed'] + index)
        for repetition in range(protocol['reference_repetitions']):
            seed = rng.randrange(2**31)
            if repetition not in (0, protocol['reference_repetitions'] - 1):
                continue
            replica = replace(task, seed=seed, uid=f'{task.uid}-r{repetition}')
            actual = np.array([run_episode(replica, exposure).outcome for exposure in exposures])
            if not np.array_equal(actual, samples[index, repetition]):
                raise ValueError('sampled stochastic reference differs')
            checked_replicas += len(exposures)
        for mask in range(256):
            row = reference[(task.uid, mask)]
            close(row['reference_mean'], samples[index, :, mask].mean(), 'reference mean')
            close(row['reference_standard_error'], samples[index, :, mask].std(ddof=1) /
                np.sqrt(protocol['reference_repetitions']), 'reference SE')
    attr = {(int(r['seed']), r['method'], r['context']): r for r in table(directory / 'attribution.csv')}
    pairs = {(int(r['seed']), r['method'], r['context'], r['memory_a'], r['memory_b']): r
        for r in table(directory / 'interactions.csv')}
    mc = {(int(r['seed']), r['context'], int(r['permutations'])): r for r in table(directory / 'MC_sensitivity.csv')}
    checked_predictions = 0
    for seed in protocol['training_seeds']:
        training = Path(protocol['training_directory']) / f'seed_{seed}'
        full = FiniteMechanisms(json.loads((training / 'decision_observations.json').read_text()),
            json.loads((training / 'api_observations.json').read_text()))
        models = {'Full finite mechanism': full, 'Without decision history': without_decision_history(full),
            'Without memory interactions': without_memory_interactions(full)}
        for method, model in models.items():
            for index, task in enumerate(tasks):
                values = np.array([model.predict(task, exposure).mean for exposure in exposures])
                checked_predictions += len(values)
                phi, exact_phi = weights @ values, weights @ oracle[index]
                row = attr[(seed, method, task.uid)]
                close(row['Shapley_MAE'], np.mean(abs(phi - exact_phi)), 'Shapley MAE')
                close(row['Shapley_rank_correlation'], spearmanr(exact_phi, phi).statistic, 'rank')
                close(row['coalition_probability_MAE'], np.mean(abs(values - oracle[index])), 'probability MAE')
                close(row['noop_predicted_credit'], phi[-1], 'placebo prediction')
                close(row['noop_exact_credit'], exact_phi[-1], 'placebo reference')
                actual_pairs = experiment.signed_pairs(oracle[index], 8, protocol['interaction_sign_threshold'])
                predicted = experiment.signed_pairs(values, 8, protocol['interaction_sign_threshold'])
                close(row['signed_pair_F1'], experiment.pair_f1(actual_pairs, predicted), 'pair F1')
                for a, b in zip(actual_pairs, predicted):
                    pair = pairs[(seed, method, task.uid, pool[a[0]], pool[a[1]])]
                    close(pair['reference_effect'], a[2], 'reference interaction')
                    close(pair['predicted_effect'], b[2], 'predicted interaction')
                    close(pair['reference_sign'], a[3], 'reference sign')
                    close(pair['predicted_sign'], b[3], 'predicted sign')
                if method == 'Full finite mechanism':
                    for permutations in protocol['MC_permutation_sensitivity']:
                        estimate = experiment.monte_carlo_phi(values, 8, permutations, seed + 425)
                        record = mc[(seed, task.uid, permutations)]
                        close(record['MC_error_vs_exact_model_Shapley'], np.mean(abs(estimate - phi)), 'MC approximation')
                        close(record['total_MAE_vs_exact_environment'], np.mean(abs(estimate - exact_phi)), 'MC total')
    controls = {(r['context'], int(r['trial'])): r for r in table(directory / 'negative_controls.csv')}
    replay_rows = {(r['context'], int(r['trial'])): r for r in table(directory / 'paired_replay.csv')}
    rng = Random(426)
    for task in tasks:
        for trial in range(32):
            replica = replace(task, uid=f'control-{task.uid}-{trial}', seed=rng.randrange(2**31))
            exposure = exposures[rng.randrange(256)]
            a, aa = replay(replica, exposure, exposure)
            _, noop = replay(replica, exposure, set(exposure) | {'noop'})
            unexposed = run_episode(replica, exposure)
            try:
                run_episode(replace(replica, epoch='target' if replica.epoch == 'source' else 'source'),
                    exposure, prefix=a, start=1)
                rejected = False
            except ValueError:
                rejected = True
            row = controls[(task.uid, trial)]
            for metric, expected in [('AA_same_state', a.state_hash == aa.state_hash),
                    ('noop_same_state', a.state_hash == noop.state_hash), ('unexposed_same_state', a.state_hash == unexposed.state_hash),
                    ('changed_checkpoint_rejected', rejected)]:
                close(row[metric], int(expected), 'negative control: ' + metric)
            memory = rng.choice(pool); background = set(exposure) - {memory}
            c, incremental = replay(replica, background, background | {memory})
            f, complete = replay(replica, background, background | {memory}, incremental=False)
            row = replay_rows[(task.uid, trial)]
            if row['memory'] != memory or row['treatment_state_sha256'] != incremental.state_hash or row['full_treatment_sha256'] != complete.state_hash:
                raise ValueError('paired replay state differs')
            close(row['incremental_calls'], c.calls + incremental.calls, 'incremental cost')
            close(row['full_calls'], f.calls + complete.calls, 'full cost')
    summary = json.loads((directory / 'summary.json').read_text())
    close(summary['reference_episodes'], samples.size, 'reference episode count')
    close(summary['reference_tool_calls'], sum(samples.shape[1] * 256 * len(t.roles) for t in tasks), 'reference call count')
    report = {'verified': True, 'exact_coalitions_recomputed': int(oracle.size),
        'stochastic_episodes_sampled_and_reexecuted': checked_replicas, 'reference_episodes_in_artifact': int(samples.size),
        'model_predictions_recomputed': checked_predictions, 'all_replay_and_negative_controls_reexecuted': len(controls),
        'scope': 'post-selection finite diagnostics; stochastic reference verified on first/last replicas per context',
        'verifier_source_sha256': digest(Path(__file__))}
    (directory / 'execution_audit.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().results), indent=2))
