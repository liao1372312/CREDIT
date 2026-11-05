"""Paired finite-portfolio summaries with common bootstrap draws per method."""

from itertools import product
import numpy as np


def summarize(rows, primary, *, repeats=2000, random_seed=2042):
    methods = sorted({row["method"] for row in rows})
    if primary not in methods:
        raise ValueError(f"primary method absent: {primary}")
    seeds = sorted({row["seed"] for row in rows})
    families = sorted({row["family"] for row in rows})
    indexed = {method: {} for method in methods}
    for row in rows:
        key = row["seed"], row["task"]
        if key in indexed[row["method"]]:
            raise ValueError(f"duplicate task for method: {key}")
        indexed[row["method"]][key] = row
    keys = set(indexed[primary])
    if any(set(indexed[method]) != keys for method in methods):
        raise ValueError("paired task coverage differs across methods")
    if any(row["family"] != indexed[primary][key]["family"]
           for method in methods for key, row in indexed[method].items()):
        raise ValueError("paired family assignment differs")
    matrices = {}
    for method in methods:
        matrix = []
        for seed in seeds:
            values = []
            for family in families:
                cell = [int(row["success"]) for row in indexed[method].values()
                        if row["seed"] == seed and row["family"] == family]
                if not cell or any(outcome not in (0, 1) for outcome in cell):
                    raise ValueError("missing stratum or invalid verification")
                values.append(np.mean(cell))
            matrix.append(values)
        matrices[method] = np.asarray(matrix)
    rng = np.random.default_rng(random_seed)
    sample_seeds = rng.integers(len(seeds), size=(repeats, len(seeds), 1))
    sample_families = rng.integers(len(families), size=(repeats, len(seeds), len(families)))
    def interval(matrix):
        return np.quantile(matrix[sample_seeds, sample_families].mean(axis=(1, 2)), [0.025, 0.975]).tolist()
    main, comparisons = [], []
    for method, matrix in matrices.items():
        low, high = interval(matrix)
        main.append({"method": method, "vsr": float(matrix.mean()), "ci_low": low,
                     "ci_high": high, "episodes": len(indexed[method]), "independent_seeds": len(seeds)})
        if method == primary:
            continue
        difference = matrices[primary] - matrix
        low, high = interval(difference)
        seed_means = difference.mean(axis=1)
        effect = float(seed_means.mean())
        null = [np.mean(np.asarray(signs) * seed_means) for signs in product((-1, 1), repeat=len(seeds))]
        comparisons.append({"baseline": method, "delta_vsr": effect, "ci_low": low, "ci_high": high,
            "seed_sign_flip_p": float(np.mean(np.abs(null) >= abs(effect) - 1e-12))})
    running = 0.0
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["seed_sign_flip_p"])
    for rank, index in enumerate(order):
        running = max(running, min(1, comparisons[index]["seed_sign_flip_p"] * (len(order) - rank)))
        comparisons[index]["holm_p"] = running
    return main, comparisons
