"""Post-test exact oracle and storage-cost diagnostics for finite workflows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from .experiments import _write_csv
from .prospective import scenario_weights, score_values
from .stateful import MEMORIES, ROLES, StateTask, oracle_probability, retrieve


def _csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _task(row):
    return StateTask(**{**row, "roles": tuple(row["roles"])})


def diagnose(root):
    frozen = json.loads((root / "frozen_protocol.json").read_text())
    for name, digest in frozen["source_sha256"].items():
        if sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"source differs from frozen run: {name}")
    if "backend" in frozen:
        from .llm_policy import install_policy
        scores = Path(frozen["policy_directory"]) / "decision_scores.json"
        if sha256(scores.read_bytes()).hexdigest() != frozen["backend"]["decision_scores_sha256"]:
            raise ValueError("pinned model scores differ")
        install_policy(scores)
    config = frozen["config"]
    seeds = frozen["executed_seeds"]
    # Refuse to run oracles before test results and frozen banks are available.
    for seed in seeds:
        directory = root / f"seed_{seed}"
        banks = json.loads((directory / "frozen_banks.json").read_text())
        raw = _csv(directory / "raw_episodes.csv")
        counts = defaultdict(int)
        for row in raw:
            counts[row["method"]] += 1
            assert set(filter(None, row["bank"].split("|"))) == set(banks[row["method"]])
        if set(counts) != set(banks) or any(n != config["test_episodes_per_seed"] for n in counts.values()):
            raise ValueError("held-out scoring incomplete; oracle diagnostics are forbidden")
    manifest = json.loads((root / f"seed_{seeds[0]}" / "split_manifest.json").read_text())
    scenarios = [_task(row) for row in manifest["forecast_contexts"]]
    if any(set(task.roles) != set(ROLES) for task in scenarios):
        raise ValueError("subset-to-depth oracle shortcut requires all roles in every graph")
    options = dict(epoch_weights=config["future_epoch_weights"], radius=config["mixture_radius"],
                   storage_cost=config["storage_cost_per_memory"])
    weights = scenario_weights(scenarios, options["epoch_weights"])
    oracle_rows = []
    for size in range(min(config["bank_capacity"], config["retrieval_depth"]) + 1):
        for bank in combinations(MEMORIES, size):
            values = [oracle_probability(task, retrieve(task, bank, config["retrieval_depth"])) for task in scenarios]
            oracle_rows.append({"bank": "|".join(bank), "mean_vsr": sum(w * value for w, value in zip(weights, values)),
                                "robust_cost_value": score_values(values, scenarios, bank, **options)})
    _write_csv(root / "evaluation_only_oracle.csv", oracle_rows)
    report = {"diagnostic_only": True, "banks": len(oracle_rows),
              "mean_vsr_oracle": max(oracle_rows, key=lambda row: row["mean_vsr"]),
              "robust_cost_oracle": max(oracle_rows, key=lambda row: row["robust_cost_value"]),
              "nomem": oracle_rows[0], "diagnostic_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    report["nominal_headroom_over_nomem"] = report["mean_vsr_oracle"]["mean_vsr"] - report["nomem"]["mean_vsr"]
    bank_rows = []
    for seed in seeds:
        directory = root / f"seed_{seed}"
        manifest = json.loads((directory / "split_manifest.json").read_text())
        tasks = {row["uid"]: _task(row) for row in manifest["heldout"]}
        grouped = defaultdict(list)
        for row in _csv(directory / "raw_episodes.csv"):
            grouped[row["method"]].append(row)
        for method, rows in grouped.items():
            bank = tuple(filter(None, rows[0]["bank"].split("|")))
            contexts = [tasks[row["task"]] for row in rows]
            values = [oracle_probability(task, tuple(filter(None, row["exposure"].split("|"))))
                      for task, row in zip(contexts, rows)]
            weights = scenario_weights(contexts, options["epoch_weights"])
            bank_rows.append({"seed": seed, "method": method, "bank": "|".join(bank), "stored_memories": len(bank),
                              "observed_robust_cost": score_values([int(row["success"]) for row in rows], contexts, bank, **options),
                              "oracle_mean_vsr_diagnostic_only": sum(w * value for w, value in zip(weights, values)),
                              "oracle_robust_cost_diagnostic_only": score_values(values, contexts, bank, **options)})
    _write_csv(root / "diagnostic_bank_values.csv", bank_rows)
    (root / "oracle_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.results), indent=2))


if __name__ == "__main__":
    main()
