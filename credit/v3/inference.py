"""Independent-pilot PPI estimates and paired bank-switch certification.

Normal intervals implement the manuscript's asymptotic rule. The optional
empirical-Bernstein rule splits error over the two independent sample means
and strata; it is more conservative and valid for bounded finite samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def _array(values, low, high):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite independent observations required")
    if np.any(values < low - 1e-9) or np.any(values > high + 1e-9):
        raise ValueError("observations outside declared bounds")
    return values


def tune_lambda(pilot_outcomes, pilot_predictions, *, labeled_n, unlabeled_n,
                transported=True):
    if not transported:
        return 0.0
    y, f = _array(pilot_outcomes, -1, 1), _array(pilot_predictions, -1, 1)
    if y.shape != f.shape or labeled_n < 2 or unlabeled_n < 2:
        raise ValueError("aligned pilot pairs and positive evaluation sizes required")
    variance = float(np.var(f, ddof=1))
    if variance < 1e-12:
        return 0.0
    covariance = float(np.cov(y, f, ddof=1)[0, 1])
    return float(np.clip(covariance / (variance * (1 + labeled_n / unlabeled_n)), 0, 1))


@dataclass(frozen=True)
class PPIEstimate:
    mean: float
    standard_error: float
    weight: float
    labeled_n: int
    unlabeled_n: int
    residual_variance: float
    prediction_variance: float


def ppi(outcomes, labeled_predictions, unlabeled_predictions, weight=0.0):
    y, f = _array(outcomes, -1, 1), _array(labeled_predictions, -1, 1)
    u = _array(unlabeled_predictions, -1, 1)
    if y.shape != f.shape or not 0 <= weight <= 1:
        raise ValueError("aligned pairs and frozen weight in [0,1] required")
    residual = y - weight * f
    residual_variance, prediction_variance = float(np.var(residual, ddof=1)), float(np.var(u, ddof=1))
    return PPIEstimate(float(weight * u.mean() + residual.mean()),
                       float(np.sqrt(residual_variance / len(y) + weight**2 * prediction_variance / len(u))),
                       float(weight), len(y), len(u), residual_variance, prediction_variance)


def bernstein_radius(variance, n, width, alpha):
    if n < 2 or not 0 < alpha < 1 or variance < 0 or width < 0:
        raise ValueError("invalid empirical-Bernstein inputs")
    logarithm = np.log(2 / alpha)
    return float(np.sqrt(2 * variance * logarithm / n) + 7 * width * logarithm / (3 * (n - 1)))


def _radius(estimate, alpha):
    # Two independent means; when lambda=0 the prediction mean is absent.
    if estimate.weight == 0:
        return bernstein_radius(estimate.residual_variance, estimate.labeled_n, 2, alpha)
    return (bernstein_radius(estimate.residual_variance, estimate.labeled_n,
                             2 + 2 * estimate.weight, alpha / 2)
            + estimate.weight * bernstein_radius(estimate.prediction_variance,
                                                 estimate.unlabeled_n, 2, alpha / 2))


def certify(estimates, mixture, *, cost_difference=0.0, cost_penalty=0.0,
            finalists=1, alpha1=0.05, alpha2=0.05, margin=0.05, method="normal",
            minimum_n=200):
    if set(estimates) != set(mixture) or not estimates:
        raise ValueError("every stratum needs certification and an independently frozen mixture")
    weights = np.asarray([mixture[k] for k in estimates], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("mixture weights must be nonnegative and sum to one")
    if finalists < 1 or minimum_n < 2 or margin < 0 or cost_penalty < 0:
        raise ValueError("invalid certification configuration")
    if not 0 < alpha1 < 1 or not 0 < alpha2 < 1 or alpha1 + alpha2 >= 1:
        raise ValueError("invalid error allocations")
    if method not in {"normal", "bernstein"}:
        raise ValueError("use normal or bernstein certification")
    enough = all(e.labeled_n >= minimum_n for e in estimates.values())
    aggregate = sum(mixture[k] * e.mean for k, e in estimates.items())
    standard_error = float(np.sqrt(sum(mixture[k]**2 * e.standard_error**2 for k, e in estimates.items())))
    if method == "normal":
        aggregate_lower = aggregate - norm.ppf(1 - alpha1 / finalists) * standard_error
        guards = {k: e.mean - norm.ppf(1 - alpha2 / (finalists * len(estimates))) * e.standard_error
                  for k, e in estimates.items()}
    else:
        aggregate_lower = sum(mixture[k] * (e.mean - _radius(e, alpha1 / (finalists * len(estimates))))
                              for k, e in estimates.items())
        guards = {k: e.mean - _radius(e, alpha2 / (finalists * len(estimates))) for k, e in estimates.items()}
    threshold = cost_penalty * cost_difference
    passed = enough and aggregate_lower > threshold and all(v > -margin for v in guards.values())
    return {"passed": bool(passed), "aggregate_difference": float(aggregate),
            "aggregate_standard_error": standard_error, "aggregate_lower": float(aggregate_lower),
            "cost_threshold": float(threshold), "stratum_lower": {k: float(v) for k, v in guards.items()},
            "enough_episodes": enough, "method": method,
            "guarantee": "asymptotic" if method == "normal" else "bounded finite-sample",
            "familywise_error_budget": alpha1 + alpha2, "finalists": finalists,
            "noninferiority_margin": margin}


def worst_mixture(values, weights, radius=0.2):
    """Prop. robust(a), mass moved from largest value to a minimum-value cell."""
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float).copy()
    if values.ndim != 1 or values.shape != weights.shape or not len(values):
        raise ValueError("aligned nonempty strata required")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("invalid robust values or weights")
    if not np.isclose(weights.sum(), 1) or not 0 <= radius <= 2:
        raise ValueError("normalized weights and L1 radius in [0,2] required")
    receiver = int(np.argmin(values))
    remaining = min(radius / 2, 1 - weights[receiver])
    for donor in np.argsort(-values):
        if donor == receiver:
            continue
        amount = min(remaining, weights[donor])
        weights[donor] -= amount
        weights[receiver] += amount
        remaining -= amount
        if remaining <= 1e-15:
            break
    return float(weights @ values)
