"""Shared future-mixture objective, bundle search, and posterior probe VOI."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .mechanisms import FiniteMechanisms
from .policy import _worst_mixture
from .stateful import MEMORIES, NEIGHBORS, StateTask, retrieve


def scenario_weights(scenarios: list[StateTask], epoch_weights: dict[str, float]) -> list[float]:
    if (set(epoch_weights) != {task.epoch for task in scenarios}
            or any(not np.isfinite(weight) or weight < 0 for weight in epoch_weights.values())
            or abs(sum(epoch_weights.values()) - 1.0) > 1e-9):
        raise ValueError("future epoch weights must cover the scenarios and sum to one")
    strata = {(task.epoch, task.family, task.template) for task in scenarios}
    return [epoch_weights[task.epoch]
            / sum(epoch == task.epoch for epoch, _, _ in strata)
            / sum((other.epoch, other.family, other.template) == (task.epoch, task.family, task.template)
                  for other in scenarios)
            for task in scenarios]


def score_values(values, scenarios: list[StateTask], bank, *,
                 epoch_weights: dict[str, float], radius=0.2, storage_cost=0.005) -> float:
    # Aggregate within strata before allowing an adversarial mixture shift;
    # duplicating probe contexts must not expand the ambiguity set.
    grouped = {}
    for task, value in zip(scenarios, values):
        grouped.setdefault((task.epoch, task.family, task.template), []).append(float(value))
    weights = scenario_weights(scenarios, epoch_weights)
    stratum_weights = {}
    for task, weight in zip(scenarios, weights):
        key = task.epoch, task.family, task.template
        stratum_weights[key] = stratum_weights.get(key, 0.0) + weight
    keys = sorted(grouped)
    means = [float(np.mean(grouped[key])) for key in keys]
    return (_worst_mixture(means, [stratum_weights[key] for key in keys], radius)
            - storage_cost * len(bank))


def objective(model: FiniteMechanisms, scenarios: list[StateTask], bank, *,
              epoch_weights, depth=4, radius=0.2, storage_cost=0.005, mode="empirical") -> float:
    field = {"empirical": "empirical_lower", "certified": "certified_lower", "mean": "mean"}[mode]
    values = [getattr(model.predict(task, retrieve(task, bank, depth)), field) for task in scenarios]
    return score_values(values, scenarios, bank, epoch_weights=epoch_weights,
                        radius=radius, storage_cost=storage_cost)


def select_bank(model: FiniteMechanisms, scenarios: list[StateTask], *,
                epoch_weights, capacity=6, depth=4, radius=0.2, storage_cost=0.005,
                mode="empirical", pool=MEMORIES, max_passes=8, joint=True) -> dict:
    """Finite addition/removal/bundle-swap search; no submodular guarantee."""
    cache = {}
    pool = tuple(pool)

    def evaluate(bank):
        bank = tuple(sorted(bank))
        if bank not in cache:
            cache[bank] = objective(model, scenarios, bank, epoch_weights=epoch_weights,
                                    depth=depth, radius=radius, storage_cost=storage_cost, mode=mode)
        return cache[bank]

    groups = {tuple(sorted(group)) for neighbors in NEIGHBORS.values()
              for size in (2, 3) for group in combinations([m for m in neighbors if m in pool], size)}
    starts = [(), *((memory,) for memory in pool)]
    if joint:
        starts.extend(group for group in groups if len(group) <= capacity)
    for bank in starts:
        evaluate(bank)
    if not joint:
        empty = evaluate(())
        scores = sorted(((evaluate((m,)) - empty, m) for m in pool), reverse=True)
        bank = tuple(sorted(m for gain, m in scores[:capacity] if gain > 0))
        return {"bank": bank, "objective": evaluate(bank), "evaluated": len(cache),
                "passes": 0, "converged": True, "candidate_values": cache}
    converged = False
    for iteration in range(max_passes):
        beam = sorted(cache, key=lambda bank: (-cache[bank], len(bank), bank))[:3]
        before = max(cache.values())
        proposals = set()
        for bank in beam:
            selected = set(bank)
            for memory in pool:
                proposals.add(tuple(sorted(selected ^ {memory})))
            for old in selected:
                for new in set(pool) - selected:
                    proposals.add(tuple(sorted((selected - {old}) | {new})))
            for group in groups:
                proposals.add(tuple(sorted(selected | set(group))))
                proposals.add(tuple(sorted(selected - set(group))))
                # Permit a complementary bundle to displace up to two entries.
                for remove in combinations(bank, min(2, len(bank))):
                    proposals.add(tuple(sorted((selected - set(remove)) | set(group))))
        for bank in sorted(proposals):
            if len(bank) <= capacity:
                evaluate(bank)
        if max(cache.values()) <= before + 1e-10:
            converged = True
            break
    best = min(cache, key=lambda bank: (-cache[bank], len(bank), bank))
    return {"bank": best, "objective": cache[best], "evaluated": len(cache),
            "passes": iteration + 1, "converged": converged, "candidate_values": cache}


def rank_api_revalidation(model: FiniteMechanisms, scenarios: list[StateTask], search: dict, *,
                          epoch_weights, seed=7, depth=4, radius=0.2, storage_cost=0.005,
                          particles=12, batch=32, probe_cost=0.001, hypothetical_runs=32) -> list[dict]:
    """Estimate Eq. VOI with the same robust objective before/after evidence.

    A finite conjugate posterior is approximated by particles. Hypothetical
    multinomial request batches update particle weights, then recompute the
    robust maximum over one fixed candidate set. Counts and held-out outcomes
    are not inspected to choose the probe. Observed API output support is a
    posterior approximation; the separate frequentist certificate uses the
    full finite output alphabet.
    """
    rng = np.random.default_rng(seed)
    models = [model.posterior_particle(rng) for _ in range(particles)]
    candidates = sorted(search["candidate_values"],
                        key=lambda bank: (-search["candidate_values"][bank], len(bank), bank))[:24]
    if () not in candidates:
        candidates.append(())
    forecasts = np.asarray([[
        [particle.predict(task, retrieve(task, bank, depth)).mean for task in scenarios]
        for bank in candidates
    ] for particle in models])

    def maximum(weights):
        means = np.tensordot(weights, forecasts, axes=1)
        lower = np.maximum(0.0, means - model.calibration_error)
        return max(score_values(values, scenarios, bank, epoch_weights=epoch_weights,
                                radius=radius, storage_cost=storage_cost)
                   for values, bank in zip(lower, candidates))

    prior = np.ones(particles) / particles
    baseline = maximum(prior)
    probe_candidates = []
    for key, counts in model.api.items():
        if key[0] != "target" or len(counts) < 2:
            continue
        labels = sorted(counts)
        probabilities = np.asarray([[particle.api[key][label] / sum(particle.api[key].values())
                                     for label in labels] for particle in models])
        variance = float(np.var(probabilities, axis=0).sum())
        probe_candidates.append((variance, key, labels, probabilities))
    rows = []
    for _, key, labels, probabilities in sorted(probe_candidates, reverse=True)[:16]:
        gains = []
        for _ in range(hypothetical_runs):
            truth = rng.integers(particles)
            counts = rng.multinomial(batch, probabilities[truth])
            log_likelihood = np.log(np.maximum(probabilities, 1e-300)) @ counts
            likelihood = np.exp(log_likelihood - log_likelihood.max())
            posterior = prior * likelihood
            posterior /= posterior.sum()
            gains.append(maximum(posterior) - baseline)
        rows.append({"epoch": key[0], "family": key[1], "role": key[2],
                     "state": key[3], "action": key[4], "batch_calls": batch,
                     "expected_gain": float(np.mean(gains)), "probe_cost": probe_cost,
                     "voi": float(np.mean(gains)) - probe_cost,
                     "mc_standard_error": float(np.std(gains, ddof=1) / np.sqrt(len(gains))),
                     "candidate_banks": len(candidates), "particles": particles})
    return sorted(rows, key=lambda row: row["voi"], reverse=True)
