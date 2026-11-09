"""Episode-weighted final-test comparison with paper source-group bootstrap."""
from __future__ import annotations

import numpy as np

from .paper_7944_protocol import PAPER_7944


def paired_cluster_interval(incumbent, challenger, source_groups, *, workload: str,
                            seed: int) -> dict:
    """Resample source groups 10,000 times, keeping every paired episode."""
    spec = next((row for row in PAPER_7944.workloads if row.name == workload), None)
    if spec is None:
        raise ValueError("unknown registered workload")
    baseline = np.asarray(incumbent, dtype=float)
    candidate = np.asarray(challenger, dtype=float)
    groups = np.asarray(source_groups, dtype=str)
    if (baseline.shape != candidate.shape or baseline.ndim != 1 or
            len(baseline) != spec.test_episodes or groups.shape != baseline.shape or
            not np.isfinite(baseline).all() or not np.isfinite(candidate).all() or
            np.any((baseline < 0) | (baseline > 1)) or
            np.any((candidate < 0) | (candidate > 1)) or
            np.any(groups == "")):
        raise ValueError("aligned binary paired final-test outcomes and groups are required")
    if not np.all(np.isin(baseline, (0, 1))) or not np.all(np.isin(candidate, (0, 1))):
        raise ValueError("final-test success outcomes must be binary")
    labels, inverse = np.unique(groups, return_inverse=True)
    if len(labels) != spec.test_source_groups:
        raise ValueError("source-group count differs from the registered final test")
    differences = candidate - baseline
    counts = np.bincount(inverse, minlength=len(labels))
    sums = np.bincount(inverse, weights=differences, minlength=len(labels))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(10000, len(labels)))
    replicated = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(replicated, (2.5, 97.5))
    return {"paired_difference": float(differences.mean()),
            "lower_95": float(low), "upper_95": float(high),
            "source_groups": len(labels), "bootstrap_resamples": 10000,
            "episodes": len(differences)}
