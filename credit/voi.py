"""One-step particle approximation to the manuscript's revalidation VOI."""

from __future__ import annotations

from random import Random

import numpy as np

from .model import StructuralModel
from .policy import all_banks, bank_objective, retrieve
from .sandbox import MEMORIES, Task, Trace


def one_step_voi(utilities: np.ndarray, probe_success: np.ndarray,
                 probe_cost: float = 0.005) -> float:
    """Expected gain from one binary probe under equally weighted particles."""
    utility = np.asarray(utilities, dtype=float)
    success = np.asarray(probe_success, dtype=float)
    if utility.ndim != 2 or success.shape != (utility.shape[0],):
        raise ValueError("particle shapes disagree")
    prior = np.ones(len(success)) / len(success)
    baseline = float(np.max(prior @ utility))
    expected_best = 0.0
    for likelihood in (success, 1 - success):
        event_probability = float(prior @ likelihood)
        if event_probability > 0:
            posterior = prior * likelihood / event_probability
            expected_best += event_probability * float(np.max(posterior @ utility))
    return expected_best - baseline - probe_cost


def rank_revalidation(
    traces: list[Trace],
    scenarios: list[Task],
    probe_task: Task,
    current_bank: tuple[str, ...],
    *,
    seed: int,
    bank_budget: int = 512,
    depth: int = 4,
    particles: int = 5,
    version_weights: dict[str, float] | None = None,
    mixture_radius: float = 0.2,
    lambda_cost: float = 0.015,
) -> list[dict]:
    """Rank safe exposure probes without looking at future test outcomes.

    Uses bootstrap model particles and a binary probe outcome. This is a
    decision-theoretic approximation, not a calibrated Bayesian posterior.
    """
    if not traces or not scenarios:
        return []
    rng = Random(seed)
    by_task: dict[str, list[Trace]] = {}
    for trace in traces:
        by_task.setdefault(trace.task_id, []).append(trace)
    task_ids = sorted(by_task)
    models = [StructuralModel(max_order=2).fit([
        trace for task_id in (rng.choice(task_ids) for _ in task_ids)
        for trace in by_task[task_id]
    ]) for _ in range(particles)]
    banks = list(all_banks(budget=bank_budget))
    utilities = np.zeros((particles, len(banks)), dtype=float)
    for particle, model in enumerate(models):
        cache = {}
        for index, bank in enumerate(banks):
            utilities[particle, index] = bank_objective(
                model, scenarios, bank, depth=depth,
                version_weights=version_weights, mixture_radius=mixture_radius,
                lambda_cost=lambda_cost, cache=cache,
            )
    baseline_exposure = set(retrieve(probe_task, current_bank, depth=depth))
    rows = []
    for memory in MEMORIES:
        forced = tuple(sorted(baseline_exposure | {memory}))
        probabilities = np.array([
            model.predict(probe_task, forced, require_support=False).mean
            for model in models
        ])
        rows.append({
            "memory": memory,
            "voi": one_step_voi(utilities, probabilities),
            "particle_success_sd": float(np.std(probabilities, ddof=1)),
            "probe_version": probe_task.version,
        })
    return sorted(rows, key=lambda row: row["voi"], reverse=True)
