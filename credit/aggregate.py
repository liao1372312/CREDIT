"""Run independent seeds and aggregate paired service-family uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from .experiments import Config, _write_csv, run


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(seeds: list[int], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    reports = []
    for seed in seeds:
        directory = output / f"seed_{seed}"
        reports.append(run(Config(seed=seed), directory))
        for row in _read(directory / "raw_test_episodes.csv"):
            row["seed"] = seed
            row["success"] = int(row["success"])
            episodes.append(row)
    methods = sorted({r["method"] for r in episodes})
    rng = np.random.default_rng(2027)
    grouped: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for row in episodes:
        grouped[(row["seed"], row["family"], row["method"])].append(row["success"])
    clusters = sorted({(seed, family) for seed, family, _ in grouped})
    main = []
    for method in methods:
        values = [value for seed, family in clusters
                  for value in grouped[(seed, family, method)]]
        bootstrap = []
        for _ in range(2000):
            chosen = rng.integers(0, len(clusters), len(clusters))
            sample = [value for index in chosen
                      for value in grouped[(*clusters[index], method)]]
            bootstrap.append(mean(sample))
        main.append({
            "method": method, "n": len(values), "seeds": len(seeds),
            "clusters": len(clusters), "vsr": mean(values),
            "cluster_ci_low": float(np.quantile(bootstrap, 0.025)),
            "cluster_ci_high": float(np.quantile(bootstrap, 0.975)),
        })
    comparisons = []
    for method in methods:
        if method == "CREDIT":
            continue
        cluster_diffs = np.array([
            mean(grouped[(seed, family, "CREDIT")])
            - mean(grouped[(seed, family, method)])
            for seed, family in clusters
        ])
        observed = float(np.mean(cluster_diffs))
        if abs(observed) < 1e-12:
            observed = 0.0
        signs = rng.choice((-1, 1), size=(20000, len(clusters)))
        null = (signs * cluster_diffs).mean(axis=1)
        p = float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (len(null) + 1))
        boot = [float(np.mean(cluster_diffs[rng.integers(
            0, len(clusters), len(clusters)
        )])) for _ in range(2000)]
        comparisons.append({
            "baseline": method, "delta_credit_minus_baseline": observed,
            "cluster_ci_low": float(np.quantile(boot, 0.025)),
            "cluster_ci_high": float(np.quantile(boot, 0.975)),
            "randomization_p": p, "holm_p": 1.0,
        })
    indexed = sorted(range(len(comparisons)),
                     key=lambda i: comparisons[i]["randomization_p"])
    running = 0.0
    for rank, index in enumerate(indexed):
        raw = comparisons[index]["randomization_p"]
        running = max(running, min(1.0, raw * (len(indexed) - rank)))
        comparisons[index]["holm_p"] = running
    selected = []
    for report in reports:
        seed = report["seed"]
        baseline = report["validation_selected_baseline"]
        seed_rows = [r for r in episodes if r["seed"] == seed]
        score = mean(r["success"] for r in seed_rows if r["method"] == "CREDIT")
        control = mean(r["success"] for r in seed_rows if r["method"] == baseline)
        selected.append({
            "seed": seed, "validation_selected_baseline": baseline,
            "credit_vsr": score, "baseline_vsr": control,
            "paired_delta": score - control,
        })
    _write_csv(output / "pooled_main.csv", main)
    _write_csv(output / "pooled_comparisons.csv", comparisons)
    _write_csv(output / "selected_baseline_by_seed.csv", selected)
    selected_delta = mean(r["paired_delta"] for r in selected)
    if abs(selected_delta) < 1e-12:
        selected_delta = 0.0
    report = {
        "status": "synthetic CausalCompose only",
        "seeds": seeds, "test_episodes_per_method": len(episodes) // len(methods),
        "validation_selected_mean_delta": selected_delta,
        "inference": ("Service-family-by-seed cluster bootstrap and paired sign-flip "
                      "randomization p-values; Holm adjusted across method comparisons."),
    }
    (output / "aggregate_report.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13, 17, 19])
    parser.add_argument("--output", type=Path, default=Path("results/multiseed"))
    args = parser.parse_args()
    print(json.dumps(aggregate(args.seeds, args.output), indent=2))


if __name__ == "__main__":
    main()
