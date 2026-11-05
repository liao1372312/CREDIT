"""Regular fractions, uniform sign randomization, and Walsh/Banzhaf credit.

Construction searches GF(2) generator columns. Admitted character columns
must be distinct, including the constant column. Search failure is explicit;
a search limit is never reported as proof that a larger fraction is minimal.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np

from .types import Context, Memory


def admitted_pairs(context: Context, memories: tuple[Memory, ...]):
    neighbors = {n.uid: {i for i, m in enumerate(memories) if m.eligible(n.role)}
                 for n in context.nodes}
    pairs = set()
    for values in neighbors.values():
        pairs.update(combinations(sorted(values), 2))
    for a, b in context.edges:
        pairs.update(tuple(sorted((i, j))) for i in neighbors[a] for j in neighbors[b] if i != j)
    return tuple(sorted(pairs))


def eligible_slots(context, pool, ranked_ids, k1, k2, rng):
    """Top K1 plus uniformly sampled K2 scope-eligible nonretrieved memories."""
    if k1 < 0 or k2 < 0:
        raise ValueError("slot counts must be nonnegative")
    eligible = {m.uid: m for m in pool if any(m.eligible(n.role) for n in context.nodes)}
    top = tuple(dict.fromkeys(uid for uid in ranked_ids if uid in eligible))[:k1]
    remaining = sorted(set(eligible) - set(top))
    if len(top) != k1 or len(remaining) < k2:
        raise ValueError("not enough scope-eligible memories for the reserved slots")
    sampled = tuple(rng.choice(remaining, k2, replace=False)) if k2 else ()
    return tuple(eligible[uid] for uid in top + sampled)


def characters(rows, effects):
    rows = np.asarray(rows)
    return np.column_stack([np.prod(rows[:, effect], axis=1) if effect else np.ones(len(rows))
                            for effect in effects])


@dataclass(frozen=True)
class Fraction:
    columns: tuple[int, ...]
    effects: tuple[tuple[int, ...], ...]
    rows: np.ndarray
    minimum_proven: bool

    @property
    def runs(self):
        return len(self.rows)

    def draw(self, rng):
        sign = rng.choice([-1, 1], len(self.columns))
        return (self.rows * sign)[rng.permutation(self.runs)]


def regular_fraction(k, pairs=(), *, seed=0, max_visits=250_000):
    if k < 1 or k > 24:
        raise ValueError("generator search supports 1..24 slots")
    pairs = tuple(sorted(set(tuple(sorted(p)) for p in pairs)))
    if any(len(p) != 2 or p[0] == p[1] or p[0] < 0 or p[1] >= k for p in pairs):
        raise ValueError("invalid admitted pair")
    effects = ((),) + tuple((i,) for i in range(k)) + pairs
    rng = np.random.default_rng(seed)
    lower = math.ceil(math.log2(len(effects)))
    minimum_proven = True
    for exponent in range(lower, k + 1):
        if 2**exponent > 65536:
            raise ValueError("fraction too large; reduce slots or admitted pairs")
        choices = rng.permutation(np.arange(1, 2**exponent)).tolist()
        visits = 0
        limit_hit = False

        def search(columns):
            nonlocal visits, limit_hit
            visits += 1
            if visits > max_visits:
                limit_hit = True
                return None
            if len(columns) == k:
                return tuple(columns)
            # Fixing the first nonzero column at 1 loses no designs (basis symmetry).
            for value in ([1] if not columns else choices):
                candidate = columns + [value]
                codes = [0] + candidate.copy()
                codes += [candidate[a] ^ candidate[b] for a, b in pairs if b < len(candidate)]
                if len(set(codes)) != len(codes):
                    continue
                found = search(candidate)
                if found is not None:
                    return found
                if limit_hit:
                    return None
            return None

        columns = search([])
        if columns is None:
            minimum_proven &= not limit_hit
            continue
        rows = np.asarray([[1 if (int(row & col).bit_count() % 2 == 0) else -1
                            for col in columns] for row in range(2**exponent)], dtype=np.int8)
        gram = characters(rows, effects).T @ characters(rows, effects)
        if not np.allclose(gram, len(rows) * np.eye(len(effects))):
            raise RuntimeError("generator design is not admissible")
        return Fraction(columns, effects, rows, minimum_proven)
    raise ValueError("no admissible fraction found within search limits")


def estimate(fraction: Fraction, draws, outcomes):
    """Estimate Eq. factorialest; uncertainty is clustered at randomized draw."""
    if len(draws) != len(outcomes) or not draws:
        raise ValueError("nonempty aligned randomized draws are required")
    contrasts = []
    for rows, y in zip(draws, outcomes):
        y = np.asarray(y, dtype=float)
        if np.asarray(rows).shape != fraction.rows.shape or y.shape != (fraction.runs,):
            raise ValueError("one verified outcome per fractional row is required")
        if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
            raise ValueError("outcomes must be in [0,1]")
        contrasts.append(y @ characters(rows, fraction.effects) / fraction.runs)
    contrasts = np.asarray(contrasts)
    beta = contrasts.mean(axis=0)
    se = (contrasts.std(axis=0, ddof=1) / np.sqrt(len(contrasts))
          if len(contrasts) > 1 else np.full(len(beta), np.nan))
    return {"effects": fraction.effects, "beta": beta, "draw_standard_error": se,
            "banzhaf": 2 * beta[1:1 + len(fraction.columns)],
            "assignment_probability": 2.0 ** -len(fraction.columns),
            "draws": len(draws), "executions": len(draws) * fraction.runs}


def context_probabilities(coefficient_ensemble, replay_costs, epsilon=0.1):
    values, costs = np.asarray(coefficient_ensemble), np.asarray(replay_costs, dtype=float)
    if values.ndim != 3 or values.shape[1] != len(costs) or not np.isfinite(values).all():
        raise ValueError("expected [ensemble, contexts, admitted effects]")
    if not 0 < epsilon <= 1 or np.any(costs <= 0):
        raise ValueError("positive exploration and replay costs required")
    scores = values.var(axis=0).sum(axis=1) / costs
    normalized = scores / scores.sum() if scores.sum() else np.full(len(costs), 1 / len(costs))
    return epsilon / len(costs) + (1 - epsilon) * normalized
