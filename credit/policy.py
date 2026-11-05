"""Fixed retriever and retrieval-aware joint bank optimization."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import isfinite
from typing import Iterable

from .model import Prediction, StructuralModel
from .sandbox import MEMORIES, MEMORY_ROLES, Task


MEMORY_TOKENS = {memory: 128 for memory in MEMORIES}
RECENCY = {
    "lookup_key": 5, "transform_rule": 4, "submit_field": 6,
    "stale_hint": 7, "version_patch": 8, "hidden_fix": 0,
    "conflict": 3, "noop": 1,
}


def retrieve(
    task: Task,
    bank: Iterable[str],
    *,
    depth: int = 4,
    drift_safeguard: bool = True,
) -> tuple[str, ...]:
    """Deterministic frozen retriever; hidden_fix is deliberately never exposed."""
    candidates = []
    for memory in bank:
        if memory == "hidden_fix":
            continue
        if drift_safeguard:
            if memory == "stale_hint" and task.version != "v1":
                continue
            if memory == "version_patch" and task.version != "v2":
                continue
        relevance = len(MEMORY_ROLES[memory] & set(task.roles))
        if relevance:
            candidates.append((10 * relevance + RECENCY[memory], memory))
    candidates.sort(reverse=True)
    return tuple(sorted(memory for _, memory in candidates[:depth]))


def bank_cost(bank: Iterable[str]) -> int:
    return sum(MEMORY_TOKENS[m] for m in bank)


def fifo_bank(budget: int = 512) -> tuple[str, ...]:
    """Retain candidates in their recorded arrival order."""
    return tuple(sorted(MEMORIES[:max(0, budget // 128)]))


def recency_bank(budget: int = 512) -> tuple[str, ...]:
    """Retain candidates with the latest recorded timestamps."""
    ordered = sorted(MEMORIES, key=lambda memory: RECENCY[memory], reverse=True)
    return tuple(sorted(ordered[:max(0, budget // 128)]))


def version_aware_bank(
    scenarios: list[Task], budget: int = 512
) -> tuple[str, ...]:
    """FIFO for cross-version memories, with one recent scoped-memory slot.

    This heuristic uses only retrieval eligibility and arrival order. It does
    not inspect outcomes, fitted effects, or the planted task mechanism.
    """
    count = budget // 128
    if count <= 0 or not scenarios:
        return ()
    versions = {task.version for task in scenarios}
    eligible = {
        version: {
            memory for memory in MEMORIES
            if any(
                memory in retrieve(task, (memory,), depth=1)
                for task in scenarios if task.version == version
            )
        }
        for version in versions
    }
    common = [
        memory for memory in MEMORIES
        if all(memory in eligible[version] for version in versions)
    ]
    scoped = [
        memory for memory in MEMORIES
        if any(memory in eligible[version] for version in versions)
        and memory not in common
    ]
    scoped_by_recency = sorted(scoped, key=lambda memory: RECENCY[memory],
                               reverse=True)
    reserve = bool(scoped and count > 1)
    selected = common[:count - reserve]
    if reserve:
        selected.append(scoped_by_recency[0])
    for memory in common + scoped_by_recency:
        if len(selected) >= count:
            break
        if memory not in selected:
            selected.append(memory)
    return tuple(sorted(selected))


def _worst_mixture(values: list[float], counts: list[float], radius: float) -> float:
    """Minimum expected value in an L1 ball about empirical stratum weights."""
    total = sum(counts)
    if not total:
        return 0.0
    weights = [n / total for n in counts]
    budget = max(0.0, min(radius, 2.0)) / 2.0
    low = sorted(range(len(values)), key=lambda i: values[i])
    high = list(reversed(low))
    i = j = 0
    while budget > 1e-12 and i < len(low) and j < len(high):
        source, sink = high[j], low[i]
        if values[source] <= values[sink] + 1e-12:
            break
        move = min(budget, weights[source], 1.0 - weights[sink])
        if move <= 1e-12:
            if weights[source] <= 1e-12:
                j += 1
            if 1.0 - weights[sink] <= 1e-12:
                i += 1
            continue
        weights[source] -= move
        weights[sink] += move
        budget -= move
        if weights[source] <= 1e-12:
            j += 1
        if 1.0 - weights[sink] <= 1e-12:
            i += 1
    return sum(w * v for w, v in zip(weights, values))


def all_banks(pool: tuple[str, ...] = MEMORIES, budget: int = 512):
    for size in range(len(pool) + 1):
        for bank in combinations(pool, size):
            if bank_cost(bank) <= budget:
                yield bank


def bank_objective(
    model: StructuralModel, scenarios: list[Task], bank: tuple[str, ...], *,
    depth: int = 4, lambda_cost: float = 0.015, mixture_radius: float = 0.2,
    drift_safeguard: bool = True, version_weights: dict[str, float] | None = None,
    cache: dict | None = None,
) -> float:
    """One shared robust objective for retention and information valuation.

    Explicit future version weights are independent of calibration/probe
    sample counts. Conditional template proportions remain empirical.
    """
    if not scenarios:
        return -lambda_cost * bank_cost(bank) / 128
    if version_weights is not None:
        if (set(version_weights) != {task.version for task in scenarios}
                or any(not isfinite(weight) or weight < 0 for weight in version_weights.values())
                or abs(sum(version_weights.values()) - 1.0) > 1e-9):
            raise ValueError("future version weights must cover scenarios and sum to one")
    cache = {} if cache is None else cache
    by_stratum: dict[tuple[str, str], list[float]] = defaultdict(list)
    for task in scenarios:
        exposed = retrieve(task, bank, depth=depth, drift_safeguard=drift_safeguard)
        key = task.roles, task.version, exposed
        if key not in cache:
            cache[key] = model.predict(task, exposed)
        by_stratum[(task.version, task.template)].append(cache[key].lower)
    strata = sorted(by_stratum)
    means = [sum(by_stratum[key]) / len(by_stratum[key]) for key in strata]
    if version_weights is None:
        weights = [float(len(by_stratum[key])) for key in strata]
    else:
        totals = {version: sum(len(rows) for (v, _), rows in by_stratum.items() if v == version)
                  for version in version_weights}
        weights = [version_weights[v] * len(by_stratum[(v, template)]) / totals[v]
                   for v, template in strata]
    return (_worst_mixture(means, weights, mixture_radius)
            - lambda_cost * bank_cost(bank) / 128)


def optimize_bank(
    model: StructuralModel,
    scenarios: list[Task],
    *,
    budget: int = 512,
    depth: int = 4,
    lambda_cost: float = 0.015,
    mixture_radius: float = 0.2,
    drift_safeguard: bool = True,
    joint: bool = True,
    version_weights: dict[str, float] | None = None,
) -> tuple[tuple[str, ...], float]:
    """Exact search for the small planted pool; no greedy guarantee is claimed."""
    if not scenarios:
        return (), 0.0
    cache: dict[tuple[tuple[str, ...], str, tuple[str, ...]], Prediction] = {}

    def objective(bank: tuple[str, ...]) -> float:
        return bank_objective(model, scenarios, bank, depth=depth,
                              lambda_cost=lambda_cost, mixture_radius=mixture_radius,
                              drift_safeguard=drift_safeguard,
                              version_weights=version_weights, cache=cache)

    if not joint:
        single = sorted(((objective((m,)) - objective(()), m) for m in MEMORIES),
                        reverse=True)
        chosen = tuple(sorted(m for gain, m in single[:budget // 128] if gain > 0))
        return chosen, objective(chosen)
    best_bank: tuple[str, ...] = ()
    best_value = objective(())
    for bank in all_banks(budget=budget):
        value = objective(bank)
        if value > best_value + 1e-12:
            best_bank, best_value = bank, value
    return best_bank, best_value


def rank_bank(scores: dict[str, float], budget: int = 512) -> tuple[str, ...]:
    count = budget // 128
    return tuple(sorted(sorted(MEMORIES, key=lambda m: scores.get(m, 0.0),
                               reverse=True)[:count]))
