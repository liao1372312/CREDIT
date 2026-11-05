"""Verify and summarize the completed correction comparison independently."""

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from .experiments import _write_csv
from .llm_policy import install_policy
from .paired_statistics import summarize
from .stateful import StateTask, run_episode, oracle_probability, retrieve
from .prospective import scenario_weights, score_values
from .supported_mechanisms import response_contract


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def diagnostic_costs(root, frozen):
    """Evaluation only, invoked after complete heldout verification."""
    config, result = frozen["config"], []
    for version in ("original", "corrected"):
        for seed in frozen["executed_seeds"]:
            directory = root / version / f"seed_{seed}"
            scenarios = [StateTask(**{**row, "roles": tuple(row["roles"])}) for row in
                json.loads((directory / "split_manifest.json").read_text())["forecast_contexts"]]
            banks = json.loads((directory / "frozen_banks.json").read_text())
            for method in ("CREDIT-empirical", "NoMem", "Categorical reference"):
                bank = tuple(banks[method])
                values = [oracle_probability(task, retrieve(task, bank, config["retrieval_depth"])) for task in scenarios]
                result.append({"version": version, "seed": seed, "method": method, "bank": "|".join(bank),
                    "stored_memories": len(bank), "mean_vsr_diagnostic_only": sum(w * value for w, value in
                        zip(scenario_weights(scenarios, config["future_epoch_weights"]), values)),
                    "robust_cost_value_diagnostic_only": score_values(values, scenarios, bank,
                        epoch_weights=config["future_epoch_weights"], radius=config["mixture_radius"],
                        storage_cost=config["storage_cost_per_memory"])})
    _write_csv(root / "cost_diagnostics.csv", result)
    return result


def report(root):
    frozen = json.loads((root / "frozen_protocol.json").read_text())
    # The original runner is preserved verbatim, even after its summary-only fix.
    # All modules affecting execution or fitting must still match current code.
    for name, digest in frozen["source_sha256"].items():
        path = root / "source" / name if name == "mechanism_correction_experiments.py" else Path(__file__).with_name(name)
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen source mismatch: {name}")
    scores = Path(frozen["policy_directory"]) / "decision_scores.json"
    if sha256(scores.read_bytes()).hexdigest() != frozen["backend"]["decision_scores_sha256"]:
        raise ValueError("policy checksum mismatch")
    install_policy(scores)
    config, rows, metrics, integrity, checkpoints = frozen["config"], [], [], [], []
    for seed in frozen["executed_seeds"]:
        for name in ("decision_observations.json", "api_observations.json", "training_traces.json", "calibration_traces.json", "split_manifest.json"):
            left, right = (root / version / f"seed_{seed}" / name for version in ("corrected", "original"))
            if left.read_bytes() != right.read_bytes():
                raise ValueError(f"comparison inputs differ: {seed}, {name}")
        for version in ("corrected", "original"):
            directory = root / version / f"seed_{seed}"
            run_report = json.loads((directory / "report.json").read_text())
            banks = json.loads((directory / "frozen_banks.json").read_text())
            manifest = json.loads((directory / "split_manifest.json").read_text())
            tasks = {r["uid"]: StateTask(**{**r, "roles": tuple(r["roles"])}) for r in manifest["heldout"]}
            steps = defaultdict(list)
            for row in read_csv(directory / "executed_steps.csv"):
                steps[row["method"], row["task"]].append(row)
            episodes = read_csv(directory / "raw_episodes.csv")
            method_counts, count_steps = defaultdict(int), 0
            for row in episodes:
                method_counts[row["method"]] += 1
                if sorted(filter(None, row["bank"].split("|"))) != sorted(banks[row["method"]]):
                    raise ValueError("bank differs from frozen selection")
                exposure = tuple(filter(None, row["exposure"].split("|")))
                trace = run_episode(tasks[row["task"]], exposure)
                if trace.state_hash != row["state_hash"] or trace.outcome != int(row["success"]):
                    raise ValueError("episode cannot be reconstructed")
                if trace.checkpoint_hash != row["checkpoint_hash"]:
                    raise ValueError("checkpoint mismatch")
                saved = steps.pop((row["method"], row["task"]), [])
                if len(saved) != len(trace.steps):
                    raise ValueError("saved step count differs")
                for index, (actual, recorded) in enumerate(zip(trace.steps, saved)):
                    if int(recorded["step"]) != index or any(str(value) != recorded[key] for key, value in asdict(actual).items()):
                        raise ValueError("saved execution step differs")
                count_steps += len(trace.steps)
                rows.append({**row, "seed": seed, "success": int(row["success"]), "method": f"{version}/{row['method']}"})
            if steps or set(method_counts) != set(banks) or any(count != config["test_episodes_per_seed"] for count in method_counts.values()):
                raise ValueError("incomplete heldout execution")
            integrity.append({"seed": seed, "version": version, "episodes": len(episodes), "steps": count_steps})
            for method in range(4):
                kernels = json.loads((directory / f"model_{method}_kernels.json").read_text())
                forbidden = []
                for cell in kernels["api"]:
                    key, counts = cell["key"], cell["counts"]
                    forbidden.append(sum(count for label, count in counts.items()
                        if label not in response_contract(key[2], key[3])) / sum(counts.values()))
                if version == "corrected" and max(forbidden) != 0:
                    raise ValueError("corrected kernel violates mutation contract")
                metrics.append({"seed": seed, "version": version, "model": method,
                    "mutation_contract_violation_mass": mean(forbidden),
                    "zero_groups": run_report["group_counts"][list(run_report["group_counts"])[method]]["zero"]})
            for index, item in enumerate(run_report.get("validation_checkpoints", [])):
                checkpoints.append({"seed": seed, "version": version, "model": index, **item})
    main, comparisons = summarize(rows, "corrected/CREDIT-empirical")
    _write_csv(root / "pooled_main.csv", main)
    _write_csv(root / "pooled_comparisons.csv", comparisons)
    _write_csv(root / "checkpoints.csv", checkpoints)
    _write_csv(root / "support_audit.csv", metrics)
    _write_csv(root / "execution_integrity.csv", integrity)
    accuracy = []
    for version in ("corrected", "original"):
        for metric, filename, field in (("prediction", "evaluation_only_prediction_error.csv", "oracle_mae_diagnostic_only"),
            ("effect", "evaluation_only_effect_error.csv", "effect_mae_diagnostic_only")):
            values = defaultdict(list)
            for seed in frozen["executed_seeds"]:
                for row in read_csv(root / version / f"seed_{seed}" / filename):
                    values[row["method"]].append(float(row[field]))
            accuracy.extend({"version": version, "method": method, "metric": metric,
                "mean_mae": mean(value)} for method, value in values.items())
    _write_csv(root / "diagnostic_accuracy.csv", accuracy)
    costs = diagnostic_costs(root, frozen)
    result = {"main": main, "diagnostic_accuracy": accuracy, "diagnostic_costs": costs,
        "comparison_inputs_byte_identical": True, "all_execution_hashes_and_steps_match": True,
        "heldout_episodes_verified": sum(row["episodes"] for row in integrity),
        "heldout_steps_verified": sum(row["steps"] for row in integrity),
        "corrected_mutation_violation_mass": 0,
        "scope": "restricted workload unchanged; improvements in forecasting do not prove VSR superiority",
        "report_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "statistics_source_sha256": sha256(Path(__file__).with_name("paired_statistics.py").read_bytes()).hexdigest(),
        "summary_note": "explicit prefixed primary; common bootstrap draws; original frozen runner snapshot preserved"}
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/mechanism_correction_v2"))
    args = parser.parse_args()
    result = report(args.results)
    print(json.dumps({key: result[key] for key in ("comparison_inputs_byte_identical",
        "heldout_episodes_verified", "heldout_steps_verified", "corrected_mutation_violation_mass")}, indent=2))


if __name__ == "__main__":
    main()
