"""Frozen finite-state reference experiments, with raw data and honest modes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import numpy as np

from .experiments import _write_csv
from .mechanisms import (FiniteMechanisms, mechanism_tests, without_decision_history,
                         without_memory_interactions)
from .prospective import rank_api_revalidation, score_values, select_bank, scenario_weights
from .stateful import (
    ACTIONS, ARRIVAL, FAMILIES, MEMORIES, RECENCY, SCHEMA_FINGERPRINT, StateTask,
    api_probes, decision_probes, make_tasks, oracle_probability, replay,
    request_probe, retrieve, run_episode,
)


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _selection_options(config):
    return dict(epoch_weights=config["future_epoch_weights"], capacity=config["bank_capacity"],
                depth=config["retrieval_depth"], radius=config["mixture_radius"],
                storage_cost=config["storage_cost_per_memory"])


def scalar_baselines(model, scenarios, config, seed):
    weights = scenario_weights(scenarios, config["future_epoch_weights"])
    cache = {}
    def value(exposure):
        exposure = tuple(sorted(exposure))
        if exposure not in cache:
            cache[exposure] = sum(weight * model.predict(task, exposure).mean
                                  for task, weight in zip(scenarios, weights))
        return cache[exposure]
    full = value(MEMORIES)
    loo = {memory: full - value(set(MEMORIES) - {memory}) for memory in MEMORIES}
    shapley = dict.fromkeys(MEMORIES, 0.0)
    rng = Random(seed)
    for _ in range(16):
        order = list(MEMORIES)
        rng.shuffle(order)
        exposure = set()
        previous = value(exposure)
        for memory in order:
            exposure.add(memory)
            following = value(exposure)
            shapley[memory] += (following - previous) / 16
            previous = following
    capacity = config["bank_capacity"]
    return {"Model-LOO": tuple(sorted(sorted(MEMORIES, key=loo.get, reverse=True)[:capacity])),
            "Model-Shapley": tuple(sorted(sorted(MEMORIES, key=shapley.get, reverse=True)[:capacity]))}, [
                {"memory": memory, "loo": loo[memory], "mc_shapley": shapley[memory]}
                for memory in MEMORIES
            ]


def _replay_audit(seed, output):
    rng = Random(seed)
    tasks = make_tasks(seed + 1, "paired-replay", 256)
    rows = []
    for task in tasks:
        memory = rng.choice(MEMORIES)
        background = {m for m in MEMORIES if m != memory and rng.random() < 0.5}
        a, b = replay(task, background, background | {memory})
        full_a, full_b = replay(task, background, background | {memory}, incremental=False)
        rows.append({"task": task.uid, "epoch": task.epoch, "family": task.family,
                     "memory": memory, "background": "|".join(sorted(background)),
                     "propensity": 1 / (len(MEMORIES) * 2 ** (len(MEMORIES) - 1)),
                     "control_outcome": a.outcome, "treatment_outcome": b.outcome,
                     "state_matches": int(b.state_hash == full_b.state_hash),
                     "outcome_matches": int(b.outcome == full_b.outcome),
                     "checkpoint_hash": a.checkpoint_hash,
                     "control_state_hash": a.state_hash, "treatment_state_hash": b.state_hash,
                     "incremental_calls": a.calls + b.calls, "full_calls": full_a.calls + full_b.calls,
                     "reused_steps": b.reused})
    _write_csv(output / "paired_replay.csv", rows)
    return {"pairs": len(rows), "state_mismatches": sum(not r["state_matches"] for r in rows),
            "outcome_mismatches": sum(not r["outcome_matches"] for r in rows),
            "incremental_calls": sum(r["incremental_calls"] for r in rows),
            "full_calls": sum(r["full_calls"] for r in rows)}


def run_seed(config, seed: int, output: Path, include_voi=False):
    output.mkdir(parents=True, exist_ok=True)
    began = perf_counter()
    calibration_tasks = make_tasks(seed + 100, "calibration", config["calibration_episodes"],
                                   templates=("chain", "repeat"))
    scenarios = make_tasks(seed + 200, "forecast-context", 16,
                           templates=("reordered", "repair_path"))
    test_tasks = make_tasks(seed + 300, "heldout", config["test_episodes_per_seed"],
                            templates=("reordered", "repair_path"))
    # Freeze IDs and assignment rules before generating calibration or test labels.
    _json(output / "split_manifest.json", {
        "seed": seed, "calibration": [asdict(task) for task in calibration_tasks],
        "forecast_contexts": [asdict(task) for task in scenarios],
        "heldout": [asdict(task) for task in test_tasks],
        "assignment": "independent finite-context fixture trials plus randomized paired replay",
        "schema_fingerprint": SCHEMA_FINGERPRINT,
    })
    policy = decision_probes(seed + 1, config["training_policy_samples_per_cell"])
    source = api_probes(seed + 2, config["training_api_samples_per_cell"], "source")
    target = api_probes(seed + 3, config["target_probe_samples_per_cell"], "target")
    _json(output / "decision_observations.json", policy)
    _json(output / "api_observations.json", source + target)
    drift = mechanism_tests(source, target, delta=config["confidence_delta"],
                            tolerance=config["mechanism_equivalence_tolerance"])
    _write_csv(output / "mechanism_tests.csv", drift)
    rng = Random(seed + 4)
    calibration = [run_episode(task, tuple(m for m in MEMORIES if rng.random() < 0.5))
                   for task in calibration_tasks]
    _write_csv(output / "calibration_episodes.csv", [
        {"task": trace.task.uid, "family": trace.task.family, "epoch": trace.task.epoch,
         "template": trace.task.template, "exposure": "|".join(trace.exposure),
         "outcome": trace.outcome, "checkpoint_hash": trace.checkpoint_hash,
         "state_hash": trace.state_hash, "calls": trace.calls} for trace in calibration
    ])
    model = FiniteMechanisms(policy, source + target, delta=config["confidence_delta"]).calibrate(calibration)
    options = _selection_options(config)
    searches = {mode: select_bank(model, scenarios, mode=mode, **options)
                for mode in ("mean", "empirical", "certified")}
    capacity = config["bank_capacity"]
    banks = {"NoMem": (), "FIFO": tuple(sorted(ARRIVAL[:capacity])),
             "Recency": tuple(sorted(sorted(MEMORIES, key=RECENCY.get, reverse=True)[:capacity]))}
    scalars, memory_scores = scalar_baselines(model, scenarios, config, seed + 5)
    banks.update(scalars)
    banks.update({f"CREDIT-{mode}": result["bank"] for mode, result in searches.items()})
    _write_csv(output / "memory_scores.csv", memory_scores)
    variants = {
        "Without decision history": without_decision_history(model).calibrate(calibration),
        "Without memory interactions": without_memory_interactions(model).calibrate(calibration),
        "Without target probes": FiniteMechanisms(policy, source,
                                                  delta=config["confidence_delta"]).calibrate(
                                                      [trace for trace in calibration if trace.task.epoch == "source"]),
    }
    for label, variant in variants.items():
        banks[label] = select_bank(variant, scenarios, **options)["bank"]
    banks["Without mixture robustness"] = select_bank(model, scenarios, **{**options, "radius": 0})["bank"]
    banks["Without joint selection"] = select_bank(model, scenarios, joint=False, **options)["bank"]
    voi_rows, revalidation = [], []
    if include_voi:
        voi_rows = rank_api_revalidation(
            model, scenarios, searches["empirical"], epoch_weights=config["future_epoch_weights"],
            seed=seed + 6, depth=config["retrieval_depth"], radius=config["mixture_radius"],
            storage_cost=config["storage_cost_per_memory"], particles=config["voi_particles"],
            batch=config["voi_probe_batch"], probe_cost=config["voi_probe_cost"],
        )
        _write_csv(output / "voi_ranking.csv", voi_rows)
        if voi_rows and voi_rows[0]["voi"] > 0:
            selected = voi_rows[0]
            key = (selected["epoch"], selected["family"], selected["role"], selected["state"], selected["action"])
            observation = request_probe(seed + 7, config["voi_probe_batch"], key)
            revalidation = [observation]
            amended = [{**row, "counts": dict(row["counts"])} for row in source + target]
            row = next(row for row in amended if tuple(row["key"]) == key)
            for label, count in observation["counts"].items():
                row["counts"][label] = row["counts"].get(label, 0) + count
            row["samples"] += observation["samples"]
            updated = FiniteMechanisms(policy, amended, delta=config["confidence_delta"]).calibrate(calibration)
            # This empirical VOI experiment uses adaptive sample sizes. A new
            # fixed-n frequentist certificate is not asserted after selection.
            updated.certificate_valid = False
            banks["CREDIT-VOI"] = select_bank(updated, scenarios, **options)["bank"]
        else:
            banks["CREDIT-VOI"] = banks["CREDIT-empirical"]
        _json(output / "revalidation_observations.json", revalidation)
    # No oracle values or held-out outcomes have been read when banks are frozen.
    _json(output / "frozen_banks.json", {method: list(bank) for method, bank in banks.items()})
    episodes = []
    for method, bank in banks.items():
        for task in test_tasks:
            exposed = retrieve(task, bank, config["retrieval_depth"])
            if method == "Without target probes" and not variants[method].predict(task, exposed).supported:
                exposed = ()
            prediction = model.predict(task, exposed)
            trace = run_episode(task, exposed)
            episodes.append({
                "seed": seed, "method": method, "task": task.uid, "family": task.family,
                "epoch": task.epoch, "template": task.template, "success": trace.outcome,
                "bank": "|".join(bank), "bank_memories": len(bank),
                "exposure": "|".join(exposed), "api_calls": trace.calls,
                "prediction": prediction.mean, "empirical_lower": prediction.empirical_lower,
                "certified_lower": prediction.certified_lower, "supported": int(prediction.supported),
                "checkpoint_hash": trace.checkpoint_hash, "state_hash": trace.state_hash,
            })
    _write_csv(output / "raw_episodes.csv", episodes)
    replay_summary = _replay_audit(seed + 8, output)
    report = {
        "seed": seed, "status": "finite-state synthetic reference; no LLM or real provider",
        "primary_method": "CREDIT-empirical", "banks": {method: list(bank) for method, bank in banks.items()},
        "searches": {mode: {key: value for key, value in result.items() if key != "candidate_values"}
                     for mode, result in searches.items()},
        "calibration_error": model.calibration_error,
        "policy_observations": sum(row["samples"] for row in policy),
        "source_api_probe_calls": sum(row["samples"] for row in source),
        "target_api_probe_calls": sum(row["samples"] for row in target),
        "calibration_api_calls": sum(trace.calls for trace in calibration),
        "voi_additional_api_calls": sum(row["samples"] for row in revalidation),
        "mechanism_tests": dict(Counter(row["status"] for row in drift)),
        "replay": replay_summary, "elapsed_seconds": perf_counter() - began,
        "certificate_conditions": ["independent per-cell fixture trials", "known finite action/state alphabet",
                                   "frozen decision policy and complete instrumented memory dependencies"],
        "limits": ["empirical lower values are not uniform confidence certificates",
                   "categorical cell reference replaces shared neural encoders and group sparsity",
                   "bundle search and particle VOI approximate their full combinatorial objectives",
                   "fixture call counts exceed the original small replay-budget setting and are disclosed"],
    }
    _json(output / "report.json", report)
    return episodes, model, searches, scenarios, report


def diagnostics(config, model, searches, scenarios, seed, output):
    """Oracle diagnostics run after retention and held-out bank decisions freeze."""
    depth = config["retrieval_depth"]
    # All main graphs contain all four operation roles. A bank's exposure is
    # then context-invariant, and removing unexposed records improves cost.
    # Enumerating banks up to depth therefore covers the exact nominal oracle.
    candidates = [bank for size in range(min(depth, config["bank_capacity"]) + 1)
                  for bank in combinations(MEMORIES, size)]
    rows = []
    weights = scenario_weights(scenarios, config["future_epoch_weights"])
    for bank in candidates:
        values = [oracle_probability(task, retrieve(task, bank, depth)) for task in scenarios]
        rows.append({"bank": "|".join(bank), "mean_vsr": sum(w * p for w, p in zip(weights, values)),
                     "robust_cost_value": score_values(values, scenarios, bank,
                         epoch_weights=config["future_epoch_weights"], radius=config["mixture_radius"],
                         storage_cost=config["storage_cost_per_memory"])})
    _write_csv(output / "evaluation_only_oracle.csv", rows)
    oracle = max(rows, key=lambda row: row["robust_cost_value"])
    _json(output / "oracle_report.json", {"diagnostic_only": True, "banks": len(rows),
                                         "robust_oracle": oracle,
                                         "vsr_oracle": max(rows, key=lambda row: row["mean_vsr"])})
    certificate_n = config["certification_samples_per_cell"]
    cert_policy = decision_probes(seed + 500, certificate_n)
    cert_source = api_probes(seed + 501, certificate_n, "source")
    cert_target = api_probes(seed + 502, certificate_n, "target")
    certified = FiniteMechanisms(cert_policy, cert_source + cert_target, delta=config["confidence_delta"])
    certificates = []
    for family, epoch, exposure in product(FAMILIES, ("source", "target"),
                                         ((), ("legacy_contract", "submit_field"), ("new_contract", "submit_field"))):
        task = StateTask("certified-fixture", family, ("submit",), epoch, seed, "submit-fixture", initial_state=7)
        prediction = certified.predict(task, exposure)
        reference = oracle_probability(task, exposure)
        certificates.append({"family": family, "epoch": epoch, "exposure": "|".join(exposure),
                             "prediction": prediction.mean, "certified_lower": prediction.certified_lower,
                             "true_probability_diagnostic_only": reference,
                             "covered": int(prediction.certified_lower <= reference),
                             "supported": int(prediction.supported)})
    _write_csv(output / "finite_certificates.csv", certificates)
    _write_csv(output / "high_coverage_mechanism_tests.csv", mechanism_tests(
        cert_source, cert_target, delta=config["confidence_delta"],
        tolerance=config["mechanism_equivalence_tolerance"]))
    _json(output / "certification_cost.json", {
        "separate_from_main_selection": True,
        "policy_observations": sum(row["samples"] for row in cert_policy),
        "api_calls": sum(row["samples"] for row in cert_source + cert_target),
        "scope": "fully enumerated independent finite-context fixtures, not a live-Web certificate",
    })


def aggregate(rows, output):
    seeds = sorted({row["seed"] for row in rows})
    methods = sorted({row["method"] for row in rows if row["method"] != "CREDIT-VOI"})
    groups = {(seed, family, method): [row["success"] for row in rows
              if row["seed"] == seed and row["family"] == family and row["method"] == method]
              for seed in seeds for family in FAMILIES for method in methods}
    rng = np.random.default_rng(2042)
    def interval(matrix):
        sampled = []
        for _ in range(2000):
            seed_indices = rng.integers(0, len(seeds), len(seeds))
            families = rng.integers(0, len(FAMILIES), (len(seeds), len(FAMILIES)))
            sampled.append(float(np.mean(matrix[seed_indices[:, None], families])))
        return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))
    matrices = {method: np.asarray([[mean(groups[(seed, family, method)]) for family in FAMILIES]
                                   for seed in seeds]) for method in methods}
    main, comparisons = [], []
    for method, matrix in matrices.items():
        low, high = interval(matrix)
        main.append({"method": method, "vsr": float(matrix.mean()), "ci_low": low, "ci_high": high,
                     "episodes": sum(len(groups[(seed, family, method)]) for seed in seeds for family in FAMILIES),
                     "independent_seeds": len(seeds), "families_per_seed": len(FAMILIES)})
        if method != "CREDIT-empirical":
            difference = matrices["CREDIT-empirical"] - matrix
            low, high = interval(difference)
            seed_means = difference.mean(axis=1)
            observed = float(seed_means.mean())
            null = np.asarray([np.mean(np.asarray(signs) * seed_means)
                               for signs in product((-1, 1), repeat=len(seeds))])
            p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
            comparisons.append({"baseline": method, "delta_vsr": observed,
                                "ci_low": low, "ci_high": high, "seed_sign_flip_p": p})
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["seed_sign_flip_p"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, comparisons[index]["seed_sign_flip_p"] * (len(order) - rank)))
        comparisons[index]["holm_p"] = running
    _write_csv(output / "pooled_main.csv", main)
    _write_csv(output / "pooled_comparisons.csv", comparisons)
    return main


def plot(main, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = sorted(main, key=lambda row: row["vsr"])
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.barh([row["method"] for row in data], [row["vsr"] for row in data])
    ax.set_xlabel("Held-out verified success rate (finite-state synthetic reference)")
    ax.set_xlim(0, max(row["ci_high"] for row in data) * 1.15)
    for index, row in enumerate(data):
        ax.plot([row["ci_low"], row["ci_high"]], [index, index], color="black")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"stateful_results.{suffix}", dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stateful_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/stateful_v1"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--skip-voi", action="store_true")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    seeds = args.seeds or config["seeds"]
    args.output.mkdir(parents=True, exist_ok=True)
    _json(args.output / "frozen_protocol.json", {
        "config": config, "config_sha256": sha256(config_bytes).hexdigest(),
        "executed_seeds": seeds,
        "source_sha256": {name: sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("stateful.py", "mechanisms.py", "prospective.py", "stateful_experiments.py")},
    })
    rows = []
    for index, seed in enumerate(seeds):
        directory = args.output / f"seed_{seed}"
        episodes, model, searches, scenarios, report = run_seed(
            config, seed, directory, include_voi=index == 0 and not args.skip_voi)
        rows.extend(episodes)
        print(json.dumps({"seed": seed, "elapsed_seconds": report["elapsed_seconds"],
                          "bank": report["banks"]["CREDIT-empirical"], "replay": report["replay"]}), flush=True)
        if index == 0 and not args.skip_diagnostics:
            diagnostics(config, model, searches, scenarios, seed, directory)
            print(json.dumps({"seed": seed, "diagnostics": "complete"}), flush=True)
    main_rows = aggregate(rows, args.output)
    plot(main_rows, args.output)
    _json(args.output / "summary.json", {"scope": "finite-state synthetic reference only",
                                         "primary": "CREDIT-empirical", "seeds": seeds,
                                         "main": main_rows,
                                         "inference": "hierarchical seed/family bootstrap; exact sign flips over independent seeds"})
    print(json.dumps(main_rows, indent=2))


if __name__ == "__main__":
    main()
