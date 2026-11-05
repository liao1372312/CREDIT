"""Full-size fresh simulations with raw replication estimates, not original tables."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
import numpy as np
from .v3.factorial import characters, regular_fraction
from .v3.simulation import PAIRS, utility
from .v3.inference import certify, ppi, tune_lambda
from .v3.experiment import digest, write


def factorial(reps, budget, crn, seed):
    rng = np.random.default_rng(seed)
    d = regular_fraction(12, PAIRS, seed=seed)
    effects = d.effects
    full = np.asarray(list(product([-1, 1], repeat=12)))
    truth = utility(full) @ characters(full, effects) / len(full)
    estimates = np.zeros((3, reps, len(effects)))
    for start in range(0, reps, 100):
        b = min(100, reps - start)
        r = budget // d.runs
        signs = rng.choice([-1, 1], (b, r, 1, 12))
        z = d.rows[None, None] * signs
        noise = (np.broadcast_to(rng.random((b, r, 1)), (b, r, d.runs)) if crn
                 else rng.random((b, r, d.runs)))
        y = noise < utility(z.reshape(-1, 12)).reshape(b, r, d.runs)
        x = characters(z.reshape(-1, 12), effects).reshape(b, budget, -1)
        estimates[0, start:start+b] = np.einsum('bn,bnp->bp', y.reshape(b, budget), x) / budget
        for j in range(b):
            z = rng.choice([-1, 1], (budget, 12))
            noise = np.repeat(rng.random(r), d.runs) if crn else rng.random(budget)
            y = noise < utility(z)
            estimates[1, start+j] = np.linalg.lstsq(characters(z, effects), y, rcond=None)[0]
        blocks = budget // 64
        estimates[2, start:start+b, 0] = np.nan
        for column, effect in enumerate(effects[1:], 1):
            size = 2 if len(effect) == 1 else 4
            z = np.repeat(rng.choice([-1, 1], (b, blocks, 1, 12)), size, axis=2)
            if size == 2:
                z[:, :, :, effect[0]] = [1, -1]
                contrast = np.array([1, -1]) / 2
            else:
                z[:, :, :, effect[0]] = [1, 1, -1, -1]
                z[:, :, :, effect[1]] = [1, -1, 1, -1]
                contrast = np.array([1, -1, -1, 1]) / 4
            noise = np.broadcast_to(rng.random((b, blocks, 1)), (b, blocks, size)) if crn else rng.random((b, blocks, size))
            y = noise < utility(z.reshape(-1, 12)).reshape(b, blocks, size)
            estimates[2, start:start+b, column] = np.mean(y @ contrast, axis=1)
    summary = []
    for i, name in enumerate(['Randomized-fraction', 'Random-OLS', 'OAT']):
        errors = estimates[i] - truth
        summary.append({'method': name, 'budget': budget, 'noise': 'CRN' if crn else 'independent',
                        'main_rmse': float(np.sqrt(np.mean(errors[:, 1:13]**2))),
                        'pair_rmse': float(np.sqrt(np.mean(errors[:, 13:]**2))),
                        'main_max_bias': float(np.max(np.abs(errors[:, 1:13].mean(axis=0)))),
                        'replications': reps})
    return estimates, truth, summary


def cert_sample(rng, mean, gain, n, tau, bias, independent):
    p = rng.beta(mean * 20, (1 - mean) * 20, n)
    # Centered task heterogeneity prevents the null from having identical paired outcomes.
    change = rng.normal(gain, .025, n)
    q = np.clip(p + change, 0, 1)
    u = rng.random(n)
    y = ((rng.random(n) if independent else u) < q).astype(float) - (u < p).astype(float)
    f = np.clip(q - p + rng.normal(bias, tau, n), -1, 1)
    return y, f


def certification(reps, n, gain, quality, seed, *, independent=False, harmful=False):
    rng = np.random.default_rng(seed)
    weights = np.array([.2, .16, .14, .12, .1, .1, .1, .08])
    tau, bias = {'none': (0, 0), 'good': (.05, 0), 'noisy': (.2, 0), 'biased': (.05, .03)}[quality]
    raw = np.zeros((reps, 8, 3))
    passed = np.zeros(reps, dtype=bool)
    for rep in range(reps):
        estimates = {}
        for k, mean in enumerate(np.linspace(.45, .75, 8)):
            delta = -.1 if harmful and k == 7 else gain
            py, pf = cert_sample(rng, mean, delta, 100, tau, bias, independent)
            y, f = cert_sample(rng, mean, delta, n, tau, bias, independent)
            _, u = cert_sample(rng, mean, delta, 1000, tau, bias, independent)
            lam = 0 if quality == 'none' else tune_lambda(py, pf, labeled_n=n, unlabeled_n=1000)
            e = ppi(y, f, u, lam)
            estimates[str(k)] = e
            raw[rep, k] = (e.mean, e.standard_error, lam)
        result = certify(estimates, {str(k): float(v) for k, v in enumerate(weights)}, minimum_n=2)
        passed[rep] = result['passed']
    return raw, passed, {'n': n, 'gain': gain, 'quality': quality, 'independent': independent,
                         'harmful_stratum': harmful, 'switch_rate': float(passed.mean()),
                         'replications': reps, 'pilot_n': 100, 'unlabeled_n': 1000}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--factorial-reps', type=int, default=2000)
    p.add_argument('--cert-reps', type=int, default=1000)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('use a fresh directory')
    args.output.mkdir(parents=True)
    sources = [Path(__file__), Path('credit/v3/simulation.py'), Path('credit/v3/factorial.py'), Path('credit/v3/inference.py')]
    write(args.output/'frozen_protocol.json', {'seed': 811, 'K': 12, 'budgets': [256, 512, 1024],
        'factorial_reps': args.factorial_reps, 'cert_reps': args.cert_reps,
        'source_sha256': {str(f): digest(f) for f in sources},
        'source': 'fresh synthetic utility and task-noise fixtures; not original unpublished table data',
        'alpha1': .05, 'alpha2': .05, 'scope': 'statistical simulation, no actual tool calls'})
    table = []
    for budget in (256, 512, 1024):
        for crn in (False, True):
            raw, truth, rows = factorial(args.factorial_reps, budget, crn, 811 + budget + int(crn))
            np.savez_compressed(args.output/f'factorial_{budget}_{int(crn)}.npz', estimates=raw, truth=truth)
            table += rows
            print('factorial', budget, crn, flush=True)
    write(args.output/'factorial_summary.json', table)
    scenarios = [(n, g, 'none', False, False) for n in (50, 100, 200, 400) for g in (0., .01, .02, .03)]
    scenarios += [(n, .03, q, False, False) for n in (25, 50, 100, 200, 400) for q in ('good', 'noisy', 'biased')]
    scenarios += [(n, .05, q, True, False) for n in (25, 50, 100, 200, 400) for q in ('none', 'good', 'noisy', 'biased')]
    scenarios += [(n, .03, 'none', False, True) for n in (50, 100, 200, 400)]
    table = []
    for i, (n, gain, quality, independent, harmful) in enumerate(scenarios):
        raw, passed, row = certification(args.cert_reps, n, gain, quality, 1821+i,
                                         independent=independent, harmful=harmful)
        np.savez_compressed(args.output/f'certification_{i:03d}.npz', estimates=raw, passed=passed)
        table.append(row)
        print('certification', i, row['switch_rate'], flush=True)
    write(args.output/'certification_summary.json', table)
    write(args.output/'artifact_manifest.json', {f.name: digest(f) for f in args.output.iterdir() if f.is_file()})


if __name__ == '__main__': main()
