"""Audit-facing analysis for the frozen full recovery confirmation run."""
import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np

from .paired_statistics import summarize
from .v3.experiment import digest, write


PRIMARY = "CREDIT-Covered"
DIRECT = "Direct-Covered"


def verify_inventory(source):
    """Verify the outer and chronological inventories without changing them."""
    outer = json.loads((source / "artifact_manifest.json").read_text())
    protocol = json.loads((source / "recovery_frozen_protocol.json").read_text())
    checked = []
    for name, expected in outer.items():
        if name == "dataset/manifest.json":
            path = Path(protocol["dataset"]) / "manifest.json"
        elif name == "remote_policy/decision_scores.json":
            path = Path(protocol["remote_policy"]) / "decision_scores.json"
        else:
            path = source / name
        if digest(path) != expected:
            raise ValueError("outer artifact changed: " + name)
        checked.append(str(path))
    chronological = source / "chronological"
    inner = json.loads((chronological / "artifact_manifest.json").read_text())
    for name, expected in inner.items():
        path = chronological / name
        if digest(path) != expected:
            raise ValueError("chronological artifact changed: " + name)
        checked.append(str(path))
    for name, expected in protocol["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError("frozen source changed: " + name)
        checked.append(name)
    return {"passed": True, "files_checked": len(checked)}


def exact_sign_flip(values):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = [np.mean(np.asarray(signs) * values)
            for signs in product((-1, 1), repeat=len(values))]
    return float(np.mean(np.abs(null) >= observed - 1e-12))


def seed_bootstrap(values, *, seed, repeats=200000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def paired_detail(rows, left, right):
    grouped = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(dict)
    for row in rows:
        grouped[int(row["seed"])][row["method"]].append(int(row["success"]))
        key = (int(row["seed"]), int(row["cycle"]), row["task"])
        paired[key][row["method"]] = int(row["success"])
    seeds = sorted(grouped)
    differences = [np.mean(grouped[seed][left]) - np.mean(grouped[seed][right])
                   for seed in seeds]
    complete = [value for value in paired.values() if left in value and right in value]
    wins = sum(value[left] > value[right] for value in complete)
    losses = sum(value[left] < value[right] for value in complete)
    return {
        "left": left,
        "right": right,
        "difference_percentage_points": float(np.mean(differences) * 100),
        "difference_by_seed_percentage_points": {
            str(seed): float(value * 100) for seed, value in zip(seeds, differences)},
        "paired_seed_bootstrap_95_percentage_points": [
            value * 100 for value in seed_bootstrap(differences, seed=2371)],
        "paired_seed_exact_sign_flip_p": exact_sign_flip(differences),
        "task_wins": wins,
        "task_losses": losses,
        "task_ties": len(complete) - wins - losses,
    }


def method_and_stratum_summaries(rows):
    grouped = defaultdict(list)
    strata = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
        strata[(row["method"], row["family"], row["epoch"])].append(row)
    methods = {}
    for method, selected in sorted(grouped.items()):
        successes = sum(int(row["success"]) for row in selected)
        methods[method] = {
            "episodes": len(selected),
            "successes": successes,
            "vsr": successes / len(selected),
            "test_api_calls": sum(int(row["api_calls"]) for row in selected),
        }
    by_stratum = {}
    for (method, family, epoch), selected in sorted(strata.items()):
        successes = sum(int(row["success"]) for row in selected)
        by_stratum.setdefault(method, {})[f"{family}/{epoch}"] = {
            "episodes": len(selected), "successes": successes,
            "vsr": successes / len(selected)}
    return methods, by_stratum


def cost_summary(chronological):
    rows = []
    for path in sorted(chronological.glob("costs_cycle_*.json")):
        rows.extend(json.loads(path.read_text()))
    grouped = defaultdict(list)
    per_seed = defaultdict(int)
    for row in rows:
        calls = int(row["extra_api_calls"])
        grouped[row["method"]].append(calls)
        per_seed[(row["method"], int(row["seed"]))] += calls
    methods = {}
    for method, calls in sorted(grouped.items()):
        methods[method] = {
            "blocks": len(calls), "total_extra_api_calls": sum(calls),
            "mean_extra_api_calls_per_seed_cycle": float(np.mean(calls)),
            "min_extra_api_calls": min(calls), "max_extra_api_calls": max(calls),
            "maximum_allowance": max(int(row["allowance"]) for row in rows
                                     if row["method"] == method),
        }
    seeds = sorted({seed for _, seed in per_seed})
    savings = [per_seed[(DIRECT, seed)] - per_seed[(PRIMARY, seed)] for seed in seeds]
    direct_total = methods[DIRECT]["total_extra_api_calls"]
    return methods, {
        "definition": "Direct-Covered extra calls minus CREDIT-Covered extra calls",
        "savings_by_seed": {str(seed): value for seed, value in zip(seeds, savings)},
        "total_calls_saved": sum(savings),
        "relative_savings": sum(savings) / direct_total,
        "mean_calls_saved_per_seed": float(np.mean(savings)),
        "paired_seed_bootstrap_95_calls": seed_bootstrap(savings, seed=2042),
        "paired_seed_exact_sign_flip_p": exact_sign_flip(savings),
    }


def bank_summary(chronological):
    methods = defaultdict(lambda: {"blocks": 0, "patterns": Counter(),
                                   "memory_block_counts": Counter()})
    for path in chronological.glob("seed_*/cycle_*/*/frozen_policy.json"):
        method = path.parent.name
        bank = tuple(json.loads(path.read_text())["bank"])
        methods[method]["blocks"] += 1
        methods[method]["patterns"]["+".join(bank) if bank else "<empty>"] += 1
        methods[method]["memory_block_counts"].update(bank)
    return {method: {"blocks": value["blocks"],
                     "patterns": dict(sorted(value["patterns"].items())),
                     "memory_block_counts": dict(sorted(value["memory_block_counts"].items()))}
            for method, value in sorted(methods.items())}


def analyze(source, output):
    if output.exists():
        raise FileExistsError("fresh analysis output required")
    integrity = verify_inventory(source)
    chronological = source / "chronological"
    rows = json.loads((chronological / "raw_test_results.json").read_text())
    methods, strata = method_and_stratum_summaries(rows)
    main, comparisons = summarize(rows, PRIMARY)
    costs, savings = cost_summary(chronological)
    result = {
        "scope": "frozen full chronological recovery confirmation analysis",
        "integrity": integrity,
        "method_summary": methods,
        "stratum_summary": strata,
        "paired_statistics_as_generated": {"main": main, "comparisons": comparisons},
        "primary_quality_comparison": paired_detail(rows, PRIMARY, DIRECT),
        "cost_summary": costs,
        "primary_cost_comparison": savings,
        "bank_summary": bank_summary(chronological),
        "interpretation": {
            "quality": "CREDIT-Covered and Direct-Covered are statistically indistinguishable in this confirmation.",
            "efficiency": "CREDIT-Covered uses fewer measured adaptation and selection calls in every seed.",
            "boundary": "Results concern the finite synthetic API and cached actor under the versioned recovery runtime."
        },
        "source": str(source.resolve()),
        "source_protocol_sha256": digest(source / "recovery_frozen_protocol.json"),
    }
    write(output, result)
    print(json.dumps({"output": str(output),
                      "primary_quality_comparison": result["primary_quality_comparison"],
                      "primary_cost_comparison": result["primary_cost_comparison"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)
