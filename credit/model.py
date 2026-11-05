"""Local decision/API mechanisms and empirical transfer calibration."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt

import numpy as np
from sklearn.linear_model import LogisticRegression

from .sandbox import MEMORIES, MEMORY_ROLES, Task, Trace


@dataclass(frozen=True)
class Prediction:
    mean: float
    lower: float
    supported: bool
    reason: str


class StructuralModel:
    """Sparse role/version policy kernels composed with API response kernels.

    The lower value is an empirical validation bound, not a finite-sample
    certificate for the uniform total-variation assumption in the paper.
    """

    def __init__(self, max_order: int = 2, min_support: int = 12):
        if max_order not in (1, 2, 3):
            raise ValueError("max_order must be 1, 2, or 3")
        self.max_order = max_order
        self.min_support = min_support
        self.groups = tuple(
            group for order in range(2, max_order + 1)
            for group in combinations(MEMORIES[:-1], order)
        )
        self.policy: dict[tuple[str, str], LogisticRegression | float] = {}
        self.counts: dict[tuple[str, str], int] = {}
        self.api: dict[tuple[str, str, int], tuple[int, int]] = {}
        self.calibration_error = 0.18
        self.fitted = False

    def features(
        self, exposure: tuple[str, ...] | frozenset[str], *, role: str | None = None,
    ) -> np.ndarray:
        exposed = set(exposure)
        if role is not None:
            # Local kernels use the instrumented dependency graph. Unrelated
            # exposures must not acquire effects through sample correlations.
            exposed &= {memory for memory in MEMORIES if role in MEMORY_ROLES[memory]}
        unary = [float(m in exposed) for m in MEMORIES]
        higher = [float(set(g) <= exposed) for g in self.groups]
        return np.array(unary + higher, dtype=float)

    def fit(self, traces: list[Trace]) -> "StructuralModel":
        rows: dict[tuple[str, str], list[np.ndarray]] = {}
        labels: dict[tuple[str, str], list[int]] = {}
        self.api.clear()
        self.policy.clear()
        self.counts.clear()
        for trace in traces:
            for step in trace.steps:
                key = (step.role, trace.version)
                x = self.features(trace.exposure, role=step.role)
                rows.setdefault(key, []).append(x)
                labels.setdefault(key, []).append(step.correct)
                api_key = (step.role, trace.version, step.correct)
                n, k = self.api.get(api_key, (0, 0))
                self.api[api_key] = (n + 1, k + step.api_ok)
        for key, xrows in rows.items():
            y = np.asarray(labels[key], dtype=int)
            self.counts[key] = len(y)
            if len(set(y)) < 2:
                self.policy[key] = float((y.sum() + 1) / (len(y) + 2))
            else:
                fit = LogisticRegression(
                    penalty="l1", C=0.6, solver="liblinear",
                    max_iter=300, random_state=0,
                )
                fit.fit(np.stack(xrows), y)
                self.policy[key] = fit
        self.fitted = True
        return self

    def calibrate(self, traces: list[Trace]) -> "StructuralModel":
        groups: dict[tuple[str, tuple[str, ...]], list[tuple[float, float]]] = {}
        for trace in traces:
            if trace.outcome is None:
                continue
            task = Task(trace.task_id, trace.roles,
                        trace.version, "calibration", "calibration", 0)
            pred = self.predict(task, trace.exposure, require_support=False)
            groups.setdefault((trace.version, trace.roles), []).append(
                (float(trace.outcome), pred.mean)
            )
        errors = [abs(np.mean([x[0] for x in values])
                      - np.mean([x[1] for x in values]))
                  for values in groups.values() if len(values) >= 8]
        if errors:
            self.calibration_error = float(max(0.04, np.quantile(errors, 0.9)))
        return self

    def _decision(self, role: str, version: str, exposure: tuple[str, ...]) -> float:
        model = self.policy.get((role, version))
        if model is None:
            return 0.5
        if isinstance(model, float):
            return model
        return float(model.predict_proba(self.features(exposure, role=role)[None, :])[0, 1])

    def _response(self, role: str, version: str, correct: int) -> float:
        n, k = self.api.get((role, version, correct), (0, 0))
        return (k + 1) / (n + 2)

    def predict(
        self,
        task: Task,
        exposure: tuple[str, ...] | frozenset[str],
        *,
        require_support: bool = True,
    ) -> Prediction:
        if not self.fitted:
            raise RuntimeError("fit model first")
        exposure = tuple(sorted(exposure))
        missing = [role for role in task.roles
                   if self.counts.get((role, task.version), 0) < self.min_support]
        supported = not missing
        success = 1.0
        for role in task.roles:
            p_decision = self._decision(role, task.version, exposure)
            success *= p_decision * self._response(role, task.version, 1)
        mean = float(np.clip(success, 0.0, 1.0))
        if require_support and not supported:
            return Prediction(mean, 0.0, False, f"missing role/version support: {missing}")
        support_n = min((self.counts.get((r, task.version), 0) for r in task.roles),
                        default=0)
        uncertainty = self.calibration_error + 0.25 / sqrt(max(1, support_n))
        return Prediction(mean, max(0.0, mean - uncertainty), supported,
                          "empirical calibration" if supported else "unsupported")

    def interaction_coefficient(
        self, role: str, version: str, group: tuple[str, ...]
    ) -> float:
        model = self.policy.get((role, version))
        if not isinstance(model, LogisticRegression) or group not in self.groups:
            return 0.0
        index = len(MEMORIES) + self.groups.index(group)
        return float(model.coef_[0, index])
