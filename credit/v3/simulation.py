"""Fresh statistical validation of v3 factorial estimation and bank guards.

The manuscript does not supply its original coefficient vector, seeds or
simulation scripts. These explicit new fixtures test its statistical methods;
their results are not reproductions of its printed numerical tables.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from .experiment import digest, write
from .factorial import characters, regular_fraction
from .inference import certify, ppi


PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11),
         (0, 2), (1, 3), (4, 6), (5, 7))
TRIPLES = ((0, 2, 4), (1, 6, 10), (3, 7, 11))


def utility(rows):
    values = np.full(len(rows), .48)
    values += rows[:, :6] @ np.asarray([.04, .03, -.04, .03, .02, -.025])
    for pair, effect in zip(PAIRS, [.06, .05, .04, -.04] + [0.] * 6):
        values += effect * np.prod(rows[:, pair], axis=1)
    for triple, effect in zip(TRIPLES, [.04, -.035, .025]):
        values += effect * np.prod(rows[:, triple], axis=1)
    return np.clip(values, .02, .98)


def factorial_validation(repetitions, budget, crn, seed):
    design = regular_fraction(12, PAIRS, seed=seed)
    if budget % design.runs or budget % 64:
        raise ValueError("budget must be a multiple of the fraction and 64-run OAT block")
    full = np.asarray(list(product([-1, 1], repeat=12)))
    truth = utility(full) @ characters(full, design.effects) / len(full)
    rng = np.random.default_rng(seed)
    estimates = {method: [] for method in ("randomized_fraction", "random_OLS", "one_at_a_time")}
    for _ in range(repetitions):
        rows = np.vstack([design.draw(rng) for _ in range(budget // design.runs)])
        noises = (np.repeat(rng.random(budget // design.runs), design.runs) if crn else rng.random(budget))
        y = (noises < utility(rows)).astype(float)
        estimates["randomized_fraction"].append(y @ characters(rows, design.effects) / budget)
        rows = rng.choice([-1, 1], (budget, 12))
        noises = (np.repeat(rng.random(budget // design.runs), design.runs) if crn else rng.random(budget))
        y = (noises < utility(rows)).astype(float)
        estimates["random_OLS"].append(np.linalg.lstsq(characters(rows, design.effects), y, rcond=None)[0])
        singles, pairs = [], []
        for _ in range(budget // 64):
            main, interaction = [], []
            for i in range(12):
                background = rng.choice([-1, 1], (1, 12))
                rows = np.repeat(background, 2, axis=0)
                rows[:, i] = (1, -1)
                u = np.repeat(rng.random(), 2) if crn else rng.random(2)
                y = (u < utility(rows)).astype(float)
                main.append(float((y[0] - y[1]) / 2))
            for i, j in PAIRS:
                rows = np.repeat(rng.choice([-1, 1], (1, 12)), 4, axis=0)
                rows[:, i], rows[:, j] = (1, 1, -1, -1), (1, -1, 1, -1)
                u = np.repeat(rng.random(), 4) if crn else rng.random(4)
                y = (u < utility(rows)).astype(float)
                interaction.append(float(y @ np.asarray([1, -1, -1, 1]) / 4))
            singles.append(main)
            pairs.append(interaction)
        estimates["one_at_a_time"].append(np.r_[np.nan, np.mean(singles, axis=0), np.mean(pairs, axis=0)])
    result = []
    for method, array in estimates.items():
        array = np.asarray(array)
        result.append({"method": method, "budget": budget, "noise": "CRN" if crn else "independent",
                       "repetitions": repetitions,
                       "main_rmse": float(np.sqrt(np.mean((array[:, 1:13] - truth[1:13])**2))),
                       "pair_rmse": float(np.sqrt(np.mean((array[:, 13:] - truth[13:])**2))),
                       "main_max_absolute_bias": float(np.max(np.abs(array[:, 1:13].mean(axis=0) - truth[1:13])))})
    return {"rows": result, "ground_truth_beta": truth.tolist(), "runs_per_draw": design.runs,
            "generator_columns": design.columns, "minimum_proven": design.minimum_proven}


def certification_validation(repetitions, n, gain, seed, *, independent=False, negative_stratum=False):
    rng = np.random.default_rng(seed)
    weights = np.asarray([.2, .16, .14, .12, .1, .1, .1, .08])
    accepted, aggregate_errors, guard_errors = 0, 0, 0
    for _ in range(repetitions):
        estimates, actual_gains = {}, []
        for k, mean in enumerate(np.linspace(.45, .75, 8)):
            baseline_probability = rng.beta(mean * 20, (1 - mean) * 20, n)
            delta = -.1 if negative_stratum and k == 7 else gain
            candidate_probability = np.clip(baseline_probability + delta, 0, 1)
            u = rng.random(n)
            incumbent = u < baseline_probability
            candidate = (rng.random(n) if independent else u) < candidate_probability
            estimates[str(k)] = ppi(candidate.astype(float) - incumbent.astype(float),
                                     np.zeros(n), np.zeros(1000))
            actual_gains.append(delta)
        test = certify(estimates, {str(k): float(w) for k, w in enumerate(weights)}, minimum_n=2)
        accepted += test["passed"]
        aggregate_errors += test["passed"] and np.dot(weights, actual_gains) <= 0
        guard_errors += test["passed"] and min(actual_gains) <= -.05
    return {"episodes_per_stratum": n, "gain": gain, "negative_stratum": negative_stratum,
            "pairing": "independent" if independent else "CRN",
            "repetitions": repetitions, "switch_rate": accepted / repetitions,
            "false_aggregate_switch_rate": aggregate_errors / repetitions,
            "false_guard_switch_rate": guard_errors / repetitions,
            "alpha1": .05, "alpha2": .05, "total_asymptotic_error_budget": .1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    if args.replications < 2:
        raise ValueError("at least two replications required")
    args.output.mkdir(parents=True)
    write(args.output / "frozen_protocol.json", {"paper": "CREDIT_WWW2027_v3.tex",
        "paper_sha256": digest("CREDIT_WWW2027_v3.tex"), "source_sha256": digest(__file__),
        "seed": args.seed, "replications": args.replications, "budgets": args.budgets,
        "K": 12, "pairs": PAIRS, "triples": TRIPLES,
        "fixture": "explicit fresh utility in simulation.py, NOT original unpublished table data",
        "CRN_variance": "cluster by draw; manuscript independent-row variance bound is not asserted for CRN"})
    factorial = [factorial_validation(args.replications, budget, crn, args.seed + budget + int(crn))
                 for budget in args.budgets for crn in (False, True)]
    cert = [certification_validation(args.replications, n, gain, args.seed + n + int(gain * 1000))
            for n in (50, 100, 200, 400) for gain in (0., .01, .02, .03)]
    cert.extend(certification_validation(args.replications, n, .03, args.seed + n + 800,
                                          negative_stratum=True) for n in (50, 100, 200, 400))
    write(args.output / "factorial.json", factorial)
    write(args.output / "certification.json", cert)
    print(f"Saved new factorial/certification simulation to {args.output}")


if __name__ == "__main__":
    main()
