"""Finite conditional mechanism estimation and exact state-distribution rollout.

Only recorded actions/replies and public dependency metadata enter this model.
The finite reference uses categorical conditional MLEs rather than embeddings;
all orders of local exposure interaction are represented by the local mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import log, sqrt

import numpy as np
from sklearn.linear_model import LogisticRegression

from .stateful import ACTIONS, ROLES, STATUSES, StateTask, decision_key


@dataclass(frozen=True)
class StatePrediction:
    mean: float
    empirical_lower: float
    certified_lower: float
    supported: bool
    uniform_error: float


@lru_cache(maxsize=4096)
def tv_radius(samples: int, alphabet: int, cells: int, delta: float = 0.05) -> float:
    """Multinomial L1 concentration, divided by two to express total variation.

    P(TV > e) <= (2**K - 2) exp(-2 n e**2); union bound over cells.
    The certificate is conditional on independent per-cell fixture trials,
    the declared finite alphabet and the frozen decision mechanism.
    """
    if samples <= 0:
        return 1.0
    numerator = log(2 ** alphabet - 2) + log(max(1, cells) / delta)
    return min(1.0, sqrt(numerator / (2 * samples)))


class FiniteMechanisms:
    def __init__(self, policy_records: list[dict], api_records: list[dict], *,
                 delta: float = 0.05, min_support: int = 12):
        self.policy = {tuple(row["key"]): np.asarray(row["counts"], dtype=float)
                       for row in policy_records}
        self.api = {tuple(row["key"]): dict(row["counts"]) for row in api_records}
        self.delta, self.min_support = delta, min_support
        self.total_cells = len(self.policy) + len(self.api)
        self.calibration_error = 0.0
        self.certificate_valid = True
        self._matrices: dict[tuple, np.ndarray] = {}
        self._predictions: dict[tuple, StatePrediction] = {}

    def policy_distribution(self, key: tuple) -> np.ndarray:
        counts = self.policy.get(key, np.zeros(len(ACTIONS)))
        return counts / counts.sum() if counts.sum() else np.ones(len(ACTIONS)) / len(ACTIONS)

    def api_distribution(self, key: tuple) -> dict[tuple[str, int], float]:
        counts = self.api.get(key)
        if not counts or not sum(counts.values()):
            # No positive reuse from unsupported inputs; use a point forecast
            # solely so diagnostics can expose the failure of coverage.
            return {("reject", key[3]): 1.0}
        total = sum(counts.values())
        return {(label.split(":")[0], int(label.split(":")[1])): count / total
                for label, count in counts.items() if count}

    def _matrix(self, role: str, task: StateTask, exposure: tuple[str, ...]) -> np.ndarray:
        key = role, task.family, task.epoch, exposure
        if key in self._matrices:
            return self._matrices[key]
        matrix = np.zeros((33, 33))
        matrix[32, 32] = 1.0
        for state in range(8):
            for previous_index, previous in enumerate(STATUSES):
                source = state * 4 + previous_index
                probabilities = self.policy_distribution(decision_key(role, state, previous, exposure))
                for action, p_action in zip(ACTIONS, probabilities):
                    if not p_action:
                        continue
                    for (status, after), p_api in self.api_distribution(
                        (task.epoch, task.family, role, state, action)
                    ).items():
                        destination = 32 if after & 8 else after * 4 + STATUSES.index(status)
                        matrix[source, destination] += p_action * p_api
        if len(self._matrices) >= 4096:
            self._matrices.clear()
        self._matrices[key] = matrix
        return matrix

    def predict(self, task: StateTask, exposure) -> StatePrediction:
        exposure = tuple(sorted(exposure))
        key = task.family, task.roles, task.epoch, task.initial_state, exposure
        if key in self._predictions:
            return self._predictions[key]
        distribution = np.zeros(33)
        distribution[32 if task.initial_state & 8 else task.initial_state * 4] = 1.0
        error, supported = 0.0, True
        if any(role not in ROLES for role in task.roles):
            return StatePrediction(0.0, 0.0, 0.0, False, 1.0)
        for role in task.roles:
            decision_error = api_error = 0.0
            for index in np.flatnonzero(distribution[:32] > 0):
                state, previous_index = divmod(int(index), 4)
                pkey = decision_key(role, state, STATUSES[previous_index], exposure)
                n_policy = int(self.policy.get(pkey, np.zeros(4)).sum())
                supported &= n_policy >= self.min_support
                decision_error = max(decision_error, tv_radius(n_policy, 4, self.total_cells, self.delta))
                for action, probability in zip(ACTIONS, self.policy_distribution(pkey)):
                    if not probability:
                        continue
                    counts = self.api.get((task.epoch, task.family, role, state, action), {})
                    n_api = int(sum(counts.values()))
                    supported &= n_api >= self.min_support
                    # Three response statuses and sixteen observable bit states.
                    api_error = max(api_error, tv_radius(n_api, 48, self.total_cells, self.delta))
            error += decision_error + api_error
            distribution = distribution @ self._matrix(role, task, exposure)
        probability = float(distribution[32])
        prediction = StatePrediction(
            probability, max(0.0, probability - self.calibration_error) if supported else 0.0,
            max(0.0, probability - error) if supported and self.certificate_valid else 0.0,
            bool(supported), min(1.0, error),
        )
        self._predictions[key] = prediction
        return prediction

    def calibrate(self, traces) -> "FiniteMechanisms":
        groups = {}
        for trace in traces:
            key = trace.task.epoch, trace.task.template
            prediction = self.predict(trace.task, trace.exposure).mean
            groups.setdefault(key, []).append((trace.outcome, prediction))
        errors = [abs(np.mean([y for y, _ in rows]) - np.mean([p for _, p in rows]))
                  for rows in groups.values() if len(rows) >= 16]
        self.calibration_error = float(max(errors, default=0.0))
        self._predictions.clear()
        return self

    def posterior_particle(self, rng: np.random.Generator) -> "FiniteMechanisms":
        """Conjugate cell posterior over observed output support; empirical use."""
        particle = FiniteMechanisms([], [], delta=self.delta, min_support=self.min_support)
        particle.policy = {key: rng.dirichlet(counts + 0.5) * counts.sum()
                           for key, counts in self.policy.items()}
        for key, counts in self.api.items():
            labels = list(counts)
            total = sum(counts.values())
            values = rng.dirichlet(np.asarray([counts[label] for label in labels]) + 0.5) * total
            particle.api[key] = dict(zip(labels, values))
        particle.total_cells = self.total_cells
        particle.calibration_error = self.calibration_error
        particle.certificate_valid = False
        return particle


def mechanism_tests(source: list[dict], target: list[dict], *, delta=0.05,
                    tolerance=0.15) -> list[dict]:
    """Equivalence requires a small upper bound; failure to reject is insufficient."""
    left = {tuple(row["key"][1:]): row for row in source}
    right = {tuple(row["key"][1:]): row for row in target}
    output = []
    cells = max(1, len(set(left) | set(right))) * 2
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            output.append({"family": key[0], "role": key[1], "state": key[2],
                           "action": key[3], "status": "unsupported",
                           "tv_estimate": "", "tv_lower": 0.0, "tv_upper": 1.0})
            continue
        labels = set(a["counts"]) | set(b["counts"])
        na, nb = sum(a["counts"].values()), sum(b["counts"].values())
        distance = sum(abs(a["counts"].get(label, 0) / na - b["counts"].get(label, 0) / nb)
                       for label in labels) / 2
        margin = tv_radius(na, 48, cells, delta) + tv_radius(nb, 48, cells, delta)
        low, high = max(0.0, distance - margin), min(1.0, distance + margin)
        status = "changed" if low > tolerance else "equivalent" if high <= tolerance else "inconclusive"
        output.append({"family": key[0], "role": key[1], "state": key[2],
                       "action": key[3], "status": status, "tv_estimate": distance,
                       "tv_lower": low, "tv_upper": high})
    return output


def without_decision_history(model: FiniteMechanisms) -> FiniteMechanisms:
    result = FiniteMechanisms([], [], delta=model.delta, min_support=model.min_support)
    pooled = {}
    for (role, missing, transient, mask), counts in model.policy.items():
        pooled[(role, mask)] = pooled.get((role, mask), np.zeros(4)) + counts
    result.policy = {key: pooled[(key[0], key[3])] / pooled[(key[0], key[3])].sum() * counts.sum()
                     for key, counts in model.policy.items()}
    result.api = {key: dict(counts) for key, counts in model.api.items()}
    result.total_cells = model.total_cells
    result.certificate_valid = False
    return result


def without_memory_interactions(model: FiniteMechanisms) -> FiniteMechanisms:
    from .stateful import NEIGHBORS
    result = FiniteMechanisms([], [], delta=model.delta, min_support=model.min_support)
    grouped = {}
    for key, counts in model.policy.items():
        grouped.setdefault(key[:3], []).append((key, counts))
    for group, rows in grouped.items():
        width = len(NEIGHBORS[group[0]])
        features, labels, weights = [], [], []
        for key, counts in rows:
            x = [int(key[3] & (1 << index) != 0) for index in range(width)]
            for action, count in enumerate(counts):
                if count:
                    features.append(x)
                    labels.append(action)
                    weights.append(count)
        fit = LogisticRegression(C=1.0, max_iter=500).fit(features, labels, sample_weight=weights)
        for key, counts in rows:
            x = [[int(key[3] & (1 << index) != 0) for index in range(width)]]
            probabilities = np.zeros(4)
            probabilities[fit.classes_] = fit.predict_proba(x)[0]
            result.policy[key] = probabilities * counts.sum()
    result.api = {key: dict(counts) for key, counts in model.api.items()}
    result.total_cells = model.total_cells
    result.certificate_valid = False
    return result
