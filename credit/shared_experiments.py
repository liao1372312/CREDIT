"""Fresh-seed experiment for shared sparse joint mechanism learning."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import torch

from .experiments import _write_csv
from .mechanisms import FiniteMechanisms, mechanism_tests
from .prospective import select_bank
from .shared_mechanisms import augment_counts, fit_shared
from .stateful import (ARRIVAL, MEMORIES, RECENCY, api_probes, decision_probes,
                       make_tasks, oracle_probability, replay, retrieve, run_episode)
from .stateful_experiments import (_json, _replay_audit, _selection_options, aggregate, plot,
                                   scalar_baselines)


def _save_kernels(path, model):
    _json(path, {"certificate_valid": False, "calibration_error": model.calibration_error,
                 "policy": [{"key": list(key), "counts": counts.tolist()} for key, counts in model.policy.items()],
                 "api": [{"key": list(key), "counts": counts} for key, counts in model.api.items()]})


def run_seed(config, seed, output):
    began = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    training_tasks = make_tasks(seed + 1000, "joint-training", config["training_intervention_pairs"],
                                templates=("chain", "repeat"))
    calibration_tasks = make_tasks(seed + 1001, "calibration", config["calibration_episodes"],
                                   templates=("chain", "repeat"))
    scenarios = make_tasks(seed + 1002, "forecast-context", 16, templates=("reordered", "repair_path"))
    heldout = make_tasks(seed + 1003, "heldout", config["test_episodes_per_seed"],
                        templates=("reordered", "repair_path"))
    _json(output / "split_manifest.json", {"training": [asdict(t) for t in training_tasks],
        "calibration": [asdict(t) for t in calibration_tasks], "forecast_contexts": [asdict(t) for t in scenarios],
        "heldout": [asdict(t) for t in heldout]})
    policy = decision_probes(seed + 1004, config["training_policy_samples_per_cell"])
    api = api_probes(seed + 1005, config["training_api_samples_per_cell"], "source")
    api += api_probes(seed + 1006, config["target_probe_samples_per_cell"], "target")
    rng = Random(seed + 1007)
    training, assignments = [], []
    for task in training_tasks:
        memory = rng.choice(MEMORIES)
        background = {m for m in MEMORIES if m != memory and rng.random() < 0.5}
        control, treatment = replay(task, background, background | {memory})
        training.extend((control, treatment))
        assignments.append({"task": task.uid, "memory": memory, "background": "|".join(sorted(background)),
                            "propensity": 1 / (len(MEMORIES) * 2 ** (len(MEMORIES) - 1)),
                            "control_hash": control.state_hash, "treatment_hash": treatment.state_hash,
                            "checkpoint_hash": control.checkpoint_hash,
                            "extra_calls": control.calls + treatment.calls})
    policy, api = augment_counts(policy, api, training)
    _json(output / "decision_observations.json", policy)
    _json(output / "api_observations.json", api)
    _json(output / "training_traces.json", [asdict(trace) for trace in training])
    _write_csv(output / "training_assignments.csv", assignments)
    calibration = [run_episode(task, tuple(m for m in MEMORIES if rng.random() < 0.5))
                   for task in calibration_tasks]
    _json(output / "calibration_traces.json", [asdict(trace) for trace in calibration])
    categorical = FiniteMechanisms(policy, api)
    # Counts now include trajectory-dependent cell visits and coupled replay
    # arms. They do not satisfy the fixed-n independent-fixture certificate.
    categorical.certificate_valid = False
    models = {"Categorical reference": categorical.calibrate(calibration)}
    specifications = {
        "CREDIT-empirical": {},
        "Shared without outcome loss": {"outcome_weight": 0},
        "Shared without group penalty": {"sparse_weight": 0},
        "Shared without interactions": {"interactions": False},
    }
    fitting = {key: config[key] for key in ("epochs", "learning_rate", "outcome_weight", "sparse_weight", "threads")}
    fit_costs = {}
    for index, (method, overrides) in enumerate(specifications.items()):
        start = perf_counter()
        # Identical initialization seed for architectural/objective ablations.
        model = fit_shared(policy, api, training, seed=seed + 1008, **{**fitting, **overrides}).calibrate(calibration)
        models[method] = model
        name = f"model_{index}"
        _write_csv(output / f"{name}_training.csv", model.training_history)
        torch.save(model.network.state_dict(), output / f"{name}_weights.pt")
        _save_kernels(output / f"{name}_kernels.json", model)
        fit_costs[method] = perf_counter() - start
        print(json.dumps({"seed": seed, "trained": method, "seconds": fit_costs[method],
                          "final_loss": model.training_history[-1]["penalized_objective"]}), flush=True)
    options = _selection_options(config)
    searches = {method: select_bank(model, scenarios, **options) for method, model in models.items()}
    capacity = config["bank_capacity"]
    banks = {"NoMem": (), "FIFO": tuple(sorted(ARRIVAL[:capacity])),
             "Recency": tuple(sorted(sorted(MEMORIES, key=RECENCY.get, reverse=True)[:capacity]))}
    banks.update({method: result["bank"] for method, result in searches.items()})
    primary = models["CREDIT-empirical"]
    scalars, scores = scalar_baselines(primary, scenarios, config, seed + 1009)
    banks.update(scalars)
    _write_csv(output / "memory_scores.csv", scores)
    banks["Without joint selection"] = select_bank(primary, scenarios, joint=False, **options)["bank"]
    banks["Without mixture robustness"] = select_bank(primary, scenarios, **{**options, "radius": 0})["bank"]
    _json(output / "frozen_banks.json", {method: list(bank) for method, bank in banks.items()})
    # No held-out outcomes or oracle probability have been used above this line.
    episodes, steps, diagnostic = [], [], defaultdict(list)
    for method, bank in banks.items():
        fitted = models.get(method, primary)
        for task in heldout:
            exposure = retrieve(task, bank, config["retrieval_depth"])
            prediction = fitted.predict(task, exposure)
            trace = run_episode(task, exposure)
            episodes.append({"seed": seed, "method": method, "task": task.uid, "family": task.family,
                             "epoch": task.epoch, "template": task.template, "success": trace.outcome,
                             "bank": "|".join(bank), "exposure": "|".join(exposure), "api_calls": trace.calls,
                             "prediction": prediction.mean, "empirical_lower": prediction.empirical_lower,
                             "supported": int(prediction.supported), "checkpoint_hash": trace.checkpoint_hash,
                             "state_hash": trace.state_hash})
            steps.extend({"method": method, "task": task.uid, "step": index, **asdict(step)}
                         for index, step in enumerate(trace.steps))
            diagnostic[method].append(abs(prediction.mean - oracle_probability(task, exposure)))
    _write_csv(output / "raw_episodes.csv", episodes)
    _write_csv(output / "executed_steps.csv", steps)
    _write_csv(output / "evaluation_only_prediction_error.csv", [
        {"method": method, "oracle_mae_diagnostic_only": mean(errors)} for method, errors in diagnostic.items()])
    # Compare effect estimates on identical independently sampled diagnostic pairs.
    replay_summary = _replay_audit(seed + 8, output)
    diagnostics = make_tasks(seed + 1010, "effect-diagnostic", 256)
    rng = Random(seed + 1011)
    effect_errors = defaultdict(list)
    for task in diagnostics:
        memory = rng.choice(MEMORIES)
        control = {m for m in MEMORIES if m != memory and rng.random() < 0.5}
        treatment = control | {memory}
        truth = oracle_probability(task, treatment) - oracle_probability(task, control)
        for method, model in models.items():
            effect = model.predict(task, treatment).mean - model.predict(task, control).mean
            effect_errors[method].append(abs(effect - truth))
    _write_csv(output / "evaluation_only_effect_error.csv", [
        {"method": method, "effect_mae_diagnostic_only": mean(errors)} for method, errors in effect_errors.items()])
    report = {"seed": seed, "scope": "shared neural finite-state reference; no LLM agent or live API",
              "primary": "CREDIT-empirical", "banks": banks, "training_seconds": fit_costs,
              "calibration_errors": {name: model.calibration_error for name, model in models.items()},
              "epochs": config["epochs"], "training_intervention_pairs": len(assignments),
              "training_replay_api_calls": sum(row["extra_calls"] for row in assignments),
              "policy_fixture_observations": config["training_policy_samples_per_cell"] * len(policy),
              "api_fixture_samples": (config["training_api_samples_per_cell"] + config["target_probe_samples_per_cell"]) * len(api) // 2,
              "calibration_api_calls": sum(trace.calls for trace in calibration),
              "replay": replay_summary, "elapsed_seconds": perf_counter() - began,
              "certificate": "none for neural estimates; empirical calibration only",
              "group_counts": {name: {"candidate": len(model.network.groups),
                                      "zero": int((model.network.group_weights.norm(dim=1) <= 1e-8).sum())}
                               for name, model in models.items() if hasattr(model, "network")},
              "optimizer": "fixed-epoch Adam with proximal group shrinkage; no global convergence guarantee"}
    _json(output / "report.json", report)
    return episodes, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/shared_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/shared_v1"))
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    payload = args.config.read_bytes()
    config = json.loads(payload)
    seeds = args.seeds or config["seeds"]
    args.output.mkdir(parents=True, exist_ok=True)
    names = ("shared_mechanisms.py", "shared_experiments.py", "stateful.py", "mechanisms.py",
             "prospective.py", "stateful_experiments.py")
    _json(args.output / "frozen_protocol.json", {"config": config, "executed_seeds": seeds,
        "config_sha256": sha256(payload).hexdigest(), "torch_version": torch.__version__,
        "source_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}})
    rows = []
    for seed in seeds:
        episodes, report = run_seed(config, seed, args.output / f"seed_{seed}")
        rows.extend(episodes)
        print(json.dumps({"seed": seed, "complete": True, "seconds": report["elapsed_seconds"]}), flush=True)
    table = aggregate(rows, args.output)
    plot(table, args.output)
    _json(args.output / "summary.json", {"primary": "shared sparse joint CREDIT-empirical", "main": table,
                                         "scope": "finite structured actions and synthetic snapshots"})


if __name__ == "__main__":
    main()
