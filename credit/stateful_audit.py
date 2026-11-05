"""Reconstruct frozen executions and diagnose fitted reference models after testing."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from .experiments import _write_csv
from .mechanisms import FiniteMechanisms, without_decision_history, without_memory_interactions
from .prospective import score_values
from .stateful import StateTask, api_probes, decision_probes, oracle_probability, replay, run_episode


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _task(data):
    return StateTask(**{**data, "roles": tuple(data["roles"])})


def audit(root: Path):
    frozen = _read(root / "frozen_protocol.json")
    # Reconstruction must use precisely the environment that generated the data.
    environment = Path(__file__).with_name("stateful.py")
    if sha256(environment.read_bytes()).hexdigest() != frozen["source_sha256"]["stateful.py"]:
        raise ValueError("environment differs from frozen execution; use the matching source")
    config = frozen["config"]
    root_summary, mechanism_rows, bank_rows, attribution_rows = [], [], [], []
    for seed in frozen["executed_seeds"]:
        directory = root / f"seed_{seed}"
        manifest = _read(directory / "split_manifest.json")
        tasks = {data["uid"]: _task(data) for name in ("calibration", "forecast_contexts", "heldout")
                 for data in manifest[name]}
        raw = _csv(directory / "raw_episodes.csv")
        steps, traces = [], {}
        for row in raw:
            exposure = tuple(filter(None, row["exposure"].split("|")))
            task = tasks[row["task"]]
            trace = run_episode(task, exposure)
            assert trace.state_hash == row["state_hash"]
            assert trace.checkpoint_hash == row["checkpoint_hash"]
            assert trace.outcome == int(row["success"])
            traces[(row["task"], exposure)] = trace
            steps.extend({"seed": seed, "method": row["method"], "task": task.uid,
                          "epoch": task.epoch, "family": task.family, "step": index,
                          **asdict(step)} for index, step in enumerate(trace.steps))
        _write_csv(directory / "executed_steps.csv", steps)
        calibration = []
        calibration_steps = []
        for row in _csv(directory / "calibration_episodes.csv"):
            exposure = tuple(filter(None, row["exposure"].split("|")))
            trace = run_episode(tasks[row["task"]], exposure)
            assert trace.state_hash == row["state_hash"]
            calibration.append(trace)
            calibration_steps.extend({"task": trace.task.uid, "step": index, **asdict(step)}
                                     for index, step in enumerate(trace.steps))
        _write_csv(directory / "calibration_steps.csv", calibration_steps)
        policy = _read(directory / "decision_observations.json")
        api = _read(directory / "api_observations.json")
        model = FiniteMechanisms(policy, api, delta=config["confidence_delta"]).calibrate(calibration)
        variants = {"Full": model, "Without decision history": without_decision_history(model),
                    "Without memory interactions": without_memory_interactions(model)}
        for label, variant in variants.items():
            errors, deviations = [], []
            for (uid, exposure), trace in traces.items():
                truth = oracle_probability(trace.task, exposure)
                prediction = variant.predict(trace.task, exposure).mean
                errors.append(abs(prediction - truth))
                deviations.append(abs(prediction - model.predict(trace.task, exposure).mean))
            mechanism_rows.append({"seed": seed, "model": label,
                                   "oracle_mae_diagnostic_only": mean(errors),
                                   "mean_prediction_change_from_full": mean(deviations),
                                   "max_prediction_change_from_full": max(deviations)})
        grouped = defaultdict(list)
        for row in raw:
            grouped[row["method"]].append(row)
        for method, rows in grouped.items():
            contexts = [tasks[row["task"]] for row in rows]
            bank = tuple(filter(None, rows[0]["bank"].split("|")))
            oracle_values = [oracle_probability(task, tuple(filter(None, row["exposure"].split("|"))))
                             for task, row in zip(contexts, rows)]
            options = dict(epoch_weights=config["future_epoch_weights"], radius=config["mixture_radius"],
                           storage_cost=config["storage_cost_per_memory"])
            bank_rows.append({"seed": seed, "method": method, "bank": "|".join(bank),
                              "observed_vsr": mean(int(row["success"]) for row in rows),
                              "observed_robust_cost_value": score_values(
                                  [int(row["success"]) for row in rows], contexts, bank, **options),
                              "oracle_mean_diagnostic_only": mean(oracle_values),
                              "oracle_robust_cost_diagnostic_only": score_values(
                                  oracle_values, contexts, bank, **options),
                              "test_api_calls": sum(int(row["api_calls"]) for row in rows)})
        # Reconstruct the independently seeded paired replay audit, including prefixes.
        from .stateful import make_tasks
        replay_tasks = {task.uid: task for task in make_tasks(seed + 9, "paired-replay", 256)}
        replay_steps = []
        pair_effects = defaultdict(list)
        for row in _csv(directory / "paired_replay.csv"):
            task = replay_tasks[row["task"]]
            background = tuple(filter(None, row["background"].split("|")))
            control, treatment = replay(task, background, (*background, row["memory"]))
            assert control.state_hash == row["control_state_hash"]
            assert treatment.state_hash == row["treatment_state_hash"]
            true_effect = (oracle_probability(task, treatment.exposure)
                           - oracle_probability(task, control.exposure))
            for label, variant in variants.items():
                estimated_effect = (variant.predict(task, treatment.exposure).mean
                                    - variant.predict(task, control.exposure).mean)
                pair_effects[label].append(estimated_effect - true_effect)
            for arm, trace in (("control", control), ("treatment", treatment)):
                replay_steps.extend({"task": task.uid, "memory": row["memory"], "arm": arm,
                                     "step": index, "reused_prefix": int(index < trace.reused),
                                     **asdict(step)} for index, step in enumerate(trace.steps))
        _write_csv(directory / "paired_replay_steps.csv", replay_steps)
        attribution_rows.extend({"seed": seed, "model": label, "pairs": len(errors),
                                 "effect_mae_diagnostic_only": mean(abs(error) for error in errors),
                                 "effect_bias_diagnostic_only": mean(errors)}
                                for label, errors in pair_effects.items())
        report = _read(directory / "report.json")
        root_summary.append({"seed": seed, "verified_episodes": len(raw), "saved_steps": len(steps),
                             "calibration_error": model.calibration_error,
                             "certified_positive_main": sum(float(row["certified_lower"]) > 0 for row in raw),
                             "mechanism_tests": report["mechanism_tests"],
                             "training_api_calls": report["source_api_probe_calls"] + report["target_api_probe_calls"]
                                                   + report["calibration_api_calls"]})
    _write_csv(root / "diagnostic_model_errors.csv", mechanism_rows)
    _write_csv(root / "diagnostic_bank_values.csv", bank_rows)
    _write_csv(root / "diagnostic_attribution.csv", attribution_rows)
    first_seed = frozen["executed_seeds"][0]
    directory = root / f"seed_{first_seed}"
    if (directory / "certification_cost.json").exists():
        n = config["certification_samples_per_cell"]
        high_policy = decision_probes(first_seed + 500, n)
        high_api = api_probes(first_seed + 501, n, "source") + api_probes(first_seed + 502, n, "target")
        (directory / "certification_observations.json").write_text(
            json.dumps({"policy": high_policy, "api": high_api}), encoding="utf-8")
        drift_counts = dict(Counter(row["status"] for row in _csv(directory / "high_coverage_mechanism_tests.csv")))
        certificates = _csv(directory / "finite_certificates.csv")
    else:
        drift_counts, certificates = {}, []
    summary = {"scope": "post-hoc diagnostic; no bank changes", "seeds": root_summary,
               "all_reconstructed_hashes_match": True,
               "high_coverage_drift": drift_counts, "high_coverage_certificates": len(certificates),
               "high_coverage_positive_bounds": sum(float(row["certified_lower"]) > 0 for row in certificates),
               "high_coverage_covered_bounds": sum(int(row["covered"]) for row in certificates),
               "audit_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / "execution_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/stateful_v1"))
    args = parser.parse_args()
    print(json.dumps(audit(args.results), indent=2))


if __name__ == "__main__":
    main()
