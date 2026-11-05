"""Audit behavioral collapse and neural kernel defects from saved experiments."""

from collections import defaultdict
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from .shared_mechanisms import RESPONSE_LABELS, SharedKernels


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def audit(root, output):
    frozen = json.loads((root / "frozen_protocol.json").read_text())
    for name, digest in frozen["source_sha256"].items():
        if sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"source mismatch: {name}")
    behavior, api_rows, penalty_rows, convergence = [], [], [], []
    for seed in frozen["executed_seeds"]:
        directory = root / f"seed_{seed}"
        episodes = read_csv(directory / "raw_episodes.csv")
        steps = defaultdict(list)
        for row in read_csv(directory / "executed_steps.csv"):
            steps[row["method"], row["task"]].append(tuple(row[key] for key in
                ("role", "state_before", "previous_status", "request_mode", "response_status", "state_after")))
        primary = {row["task"]: row for row in episodes if row["method"] == "CREDIT-empirical"}
        grouped = defaultdict(list)
        for row in episodes:
            grouped[row["method"]].append(row)
        for method, rows in grouped.items():
            behavior.append({"seed": seed, "method": method, "tasks": len(rows),
                "different_outcomes": sum(row["success"] != primary[row["task"]]["success"] for row in rows),
                "different_execution_traces": sum(steps[method, row["task"]] !=
                    steps["CREDIT-empirical", row["task"]] for row in rows),
                "different_requests": sum(tuple(s[3] for s in steps[method, row["task"]]) !=
                    tuple(s[3] for s in steps["CREDIT-empirical", row["task"]]) for row in rows),
                "unique_retrieved_exposures": len({row["exposure"] for row in rows}),
                "vsr": np.mean([int(row["success"]) for row in rows])})
        observations = json.loads((directory / "api_observations.json").read_text())
        observed = {tuple(row["key"]): row["counts"] for row in observations}
        for index in range(4):
            kernels = json.loads((directory / f"model_{index}_kernels.json").read_text())
            early, failure_changes, unobserved = [], [], []
            for row in kernels["api"]:
                key = tuple(row["key"])
                total = sum(row["counts"].values())
                p = {label: count / total for label, count in row["counts"].items()}
                if key[2] != "submit":
                    early.append(sum(prob for label, prob in p.items() if int(label.split(":")[1]) & 8))
                failure_changes.append(sum(prob for label, prob in p.items()
                    if label.split(":")[0] != "ok" and int(label.split(":")[1]) != key[3]))
                unobserved.append(sum(prob for label, prob in p.items() if not observed[key].get(label, 0)))
            api_rows.append({"seed": seed, "model": index,
                "premature_terminal_mass_mean": float(np.mean(early)),
                "premature_terminal_mass_max": float(max(early)),
                "failure_state_mutation_mass_mean": float(np.mean(failure_changes)),
                "unobserved_label_mass_mean_diagnostic": float(np.mean(unobserved))})
            curve = read_csv(directory / f"model_{index}_training.csv")
            convergence.append({"seed": seed, "model": index,
                "objective_20_epochs_before_end": float(curve[-21]["penalized_objective"]),
                "final_objective": float(curve[-1]["penalized_objective"]),
                "final_gradient_norm": float(curve[-1]["gradient_norm_before_clip"])})
        # Algebraic reparameterization: shrink group weights, expand shared head.
        records = json.loads((directory / "decision_observations.json").read_text())
        network = SharedKernels([tuple(r["key"]) for r in records], list(observed))
        network.load_state_dict(torch.load(directory / "model_0_weights.pt", weights_only=True))
        with torch.no_grad():
            before = network.decision_logits().clone()
            norm_before = float(network.group_weights.norm(dim=1).sum())
            network.group_weights.mul_(0.1)
            network.interaction[-1].weight.mul_(10)
            network.interaction[-1].bias.mul_(10)
            difference = float((before - network.decision_logits()).abs().max())
            penalty_rows.append({"seed": seed, "max_logit_change": difference,
                "group_norm_before": norm_before,
                "group_norm_after": float(network.group_weights.norm(dim=1).sum())})
    output.mkdir(parents=True, exist_ok=True)
    from .experiments import _write_csv
    for name, rows in (("behavior", behavior), ("api_support", api_rows),
                       ("penalty_scaling", penalty_rows), ("convergence", convergence)):
        _write_csv(output / f"{name}.csv", rows)
    methods = sorted({row["method"] for row in behavior})
    summary = {"source": str(root.resolve()), "all_source_hashes_match": True,
        "methods": [{"method": method,
            "tasks": sum(row["tasks"] for row in behavior if row["method"] == method),
            "different_outcomes": sum(row["different_outcomes"] for row in behavior if row["method"] == method),
            "different_requests": sum(row["different_requests"] for row in behavior if row["method"] == method),
            "different_traces": sum(row["different_execution_traces"] for row in behavior if row["method"] == method)}
            for method in methods],
        "primary_mean_premature_terminal_mass": float(np.mean([row["premature_terminal_mass_mean"]
            for row in api_rows if row["model"] == 0])),
        "primary_mean_failure_state_mutation_mass": float(np.mean([row["failure_state_mutation_mass_mean"]
            for row in api_rows if row["model"] == 0])),
        "penalty_scaling": penalty_rows,
        "limits": "Observed-label absence alone does not prove impossibility; terminal-role and failure-mutation checks use sandbox semantics."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/llm_shared_v1"))
    parser.add_argument("--output", type=Path, default=Path("results/collapse_audit_v1"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(audit(args.results, args.output), indent=2))


if __name__ == "__main__":
    main()
