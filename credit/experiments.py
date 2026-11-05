"""Reproducible CausalCompose experiments corresponding to manuscript RQ1--RQ5."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import combinations
from math import comb, factorial
from pathlib import Path
from random import Random
from statistics import mean

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from .model import StructuralModel
from .policy import (
    all_banks, bank_cost, fifo_bank, optimize_bank, rank_bank, recency_bank, retrieve,
    version_aware_bank,
)
from .sandbox import MEMORIES, ReplayPair, Task, Trace, make_tasks, replay_pair, run_task


@dataclass(frozen=True)
class Config:
    seed: int = 7
    train_pairs: int = 384
    drift_pairs: int = 128
    test_per_stratum: int = 80
    bank_budget: int = 512
    retrieval_depth: int = 4
    reference_repetitions: int = 64
    future_version_weights: tuple[tuple[str, float], ...] = (("v1", 0.5), ("v2", 0.5))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_pairs(
    tasks: list[Task],
    count: int,
    *,
    seed: int,
    incremental: bool = True,
    prioritize: tuple[str, ...] = (),
) -> tuple[list[ReplayPair], list[Trace]]:
    """Half random-background pairs, half full-bank leave-one-out pairs."""
    rng = Random(seed)
    pairs: list[ReplayPair] = []
    traces: list[Trace] = []
    for index in range(count):
        task = tasks[index % len(tasks)]
        if prioritize:
            focus = rng.random() < 0.8
            memory = rng.choice(prioritize if focus else MEMORIES)
            memory_probability = (
                0.2 / len(MEMORIES)
                + (0.8 / len(prioritize) if memory in prioritize else 0.0)
            )
        else:
            memory = MEMORIES[rng.randrange(len(MEMORIES))]
            memory_probability = 1 / len(MEMORIES)
        if rng.random() < 0.5:
            background = {m for m in MEMORIES if m != memory and rng.random() < 0.5}
        else:
            background = set(MEMORIES) - {memory}
        probability = memory_probability * (
            0.5 / (2 ** (len(MEMORIES) - 1))
            + (0.5 if len(background) == len(MEMORIES) - 1 else 0.0)
        )
        pair = replay_pair(task, background, background | {memory},
                           propensity=probability, incremental=incremental)
        pairs.append(pair)
        if pair.valid:
            traces.extend((pair.control, pair.treatment))
    return pairs, traces


def _validation_traces(tasks: list[Task], *, seed: int, each: int = 3) -> list[Trace]:
    rng = Random(seed)
    return [run_task(task, (m for m in MEMORIES if rng.random() < 0.5))
            for task in tasks for _ in range(each)]


def _pair_scores(pairs: list[ReplayPair], *, mode: str) -> dict[str, float]:
    totals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for pair in pairs:
        if not pair.valid or pair.control.outcome is None or pair.treatment.outcome is None:
            continue
        memory = next(iter(set(pair.treatment.exposure) - set(pair.control.exposure)))
        background_size = len(pair.control.exposure)
        if mode == "loo" and background_size != len(MEMORIES) - 1:
            continue
        weight = (1.0 / (comb(len(MEMORIES) - 1, background_size)
                         * pair.propensity)
                  if mode == "shapley" else 1.0)
        totals[memory].append((pair.treatment.outcome - pair.control.outcome, weight))
    return {m: sum(x * w for x, w in rows) / sum(w for _, w in rows)
            for m, rows in totals.items()}


def _flat_scores(traces: list[Trace], scenarios: list[Task]) -> dict[str, float]:
    """CMP-like randomized-outcome adaptation without local graph structure."""
    usable = [t for t in traces if t.outcome is not None]
    x = np.asarray([[int(m in t.exposure) for m in MEMORIES]
                    + [int(t.version == "v2")] for t in usable], dtype=float)
    y = np.asarray([t.outcome for t in usable], dtype=int)
    if len(set(y)) < 2:
        return {m: 0.0 for m in MEMORIES}
    fit = LogisticRegression(C=1.0, max_iter=300, random_state=0).fit(x, y)
    scores = {}
    for index, memory in enumerate(MEMORIES):
        differences = []
        for task in scenarios:
            row = np.zeros(len(MEMORIES) + 1)
            row[-1] = int(task.version == "v2")
            baseline = fit.predict_proba(row[None, :])[0, 1]
            row[index] = 1
            differences.append(fit.predict_proba(row[None, :])[0, 1] - baseline)
        scores[memory] = float(mean(differences))
    return scores


def _make_banks(
    model: StructuralModel,
    pairs: list[ReplayPair],
    traces: list[Trace],
    scenarios: list[Task],
    config: Config,
) -> dict[str, tuple[str, ...]]:
    budget = config.bank_budget
    credit, _ = optimize_bank(model, scenarios, budget=budget,
                              depth=config.retrieval_depth,
                              version_weights=dict(config.future_version_weights))
    return {
        "NoMem": (),
        "FIFO": fifo_bank(budget),
        "Recency": recency_bank(budget),
        "Version-aware": version_aware_bank(scenarios, budget),
        "LOO": rank_bank(_pair_scores(pairs, mode="loo"), budget),
        "MC-Shapley": rank_bank(_pair_scores(pairs, mode="shapley"), budget),
        "CMP-adapted": rank_bank(_flat_scores(traces, scenarios), budget),
        "CAR-adapted": rank_bank(_pair_scores(pairs, mode="car"), budget),
        "CREDIT": credit,
    }


def _evaluate_banks(
    banks: dict[str, tuple[str, ...]],
    tasks: list[Task],
    *,
    depth: int,
    drift_safeguard: bool = True,
) -> list[dict]:
    rows = []
    for method, bank in banks.items():
        for task in tasks:
            exposed = retrieve(task, bank, depth=depth,
                               drift_safeguard=drift_safeguard)
            trace = run_task(task, exposed)
            rows.append({
                "method": method, "task": task.uid, "family": task.family,
                "stratum": "drift" if task.version == "v2" else "new_workflow",
                "version": task.version, "template": task.template,
                "success": int(trace.outcome or 0),
                "verified": int(trace.outcome is not None),
                "stale_exposures": int("stale_hint" in exposed and task.version == "v2"),
                "exposures": len(exposed), "api_calls": trace.api_calls,
                "bank_tokens": bank_cost(bank),
            })
    return rows


def _cluster_interval(rows: list[dict], *, seed: int) -> tuple[float, float]:
    families = sorted({r["family"] for r in rows})
    if not families:
        return 0.0, 0.0
    by_family = {family: [r["success"] for r in rows if r["family"] == family]
                 for family in families}
    rng = Random(seed)
    samples = []
    for _ in range(400):
        chosen = [rng.choice(families) for _ in families]
        values = [value for family in chosen for value in by_family[family]]
        samples.append(mean(values))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _main_summary(rows: list[dict], seed: int) -> list[dict]:
    out = []
    for method in sorted({r["method"] for r in rows}):
        for stratum in ("all", "new_workflow", "drift"):
            subset = [r for r in rows if r["method"] == method
                      and (stratum == "all" or r["stratum"] == stratum)]
            low, high = _cluster_interval(subset, seed=seed + len(out))
            out.append({
                "method": method, "stratum": stratum,
                "n": len(subset), "vsr": mean(r["success"] for r in subset),
                "ci_low": low, "ci_high": high,
                "verify_coverage": mean(r["verified"] for r in subset),
                "stale_rate": (sum(r["stale_exposures"] for r in subset)
                               / max(1, sum(r["exposures"] for r in subset))),
                "mean_api_calls": mean(r["api_calls"] for r in subset),
                "bank_tokens": subset[0]["bank_tokens"],
            })
    return out


def _coalitions() -> list[frozenset[str]]:
    return [frozenset(group) for size in range(len(MEMORIES) + 1)
            for group in combinations(MEMORIES, size)]


def _shapley(values: dict[frozenset[str], float]) -> dict[str, float]:
    k = len(MEMORIES)
    result = {}
    for memory in MEMORIES:
        total = 0.0
        for coalition, baseline in values.items():
            if memory in coalition:
                continue
            size = len(coalition)
            weight = factorial(size) * factorial(k - size - 1) / factorial(k)
            total += weight * (values[coalition | {memory}] - baseline)
        result[memory] = total
    return result


def attribution_and_transfer(
    model: StructuralModel, *,
    seed: int, repetitions: int,
) -> tuple[list[dict], list[dict]]:
    coalitions = _coalitions()
    attribution = []
    transfer = []
    for version, template in (("v1", "new_order"), ("v2", "chain")):
        references = make_tasks(repetitions, split=f"reference-{seed}",
                                version=version, templates=(template,),
                                seed=seed + (900 if version == "v1" else 901))
        exemplar = references[0]
        truth = {}
        estimates = {}
        for exposure in coalitions:
            outcomes = [int(run_task(task, exposure).outcome or 0) for task in references]
            truth[exposure] = mean(outcomes)
            pred = model.predict(exemplar, exposure)
            estimates[exposure] = pred.mean
            ref_se = float(np.std(outcomes, ddof=1) / np.sqrt(len(outcomes)))
            transfer.append({
                "version": version, "template": template,
                "coalition": "|".join(sorted(exposure)),
                "reference": truth[exposure], "reference_se": ref_se,
                "prediction": pred.mean, "lower": pred.lower,
                "supported": int(pred.supported),
                "absolute_error": abs(pred.mean - truth[exposure]),
                "lower_covered_with_ref_se": int(pred.lower <= truth[exposure] + 1.96 * ref_se),
            })
        ref_phi, predicted_phi = _shapley(truth), _shapley(estimates)
        pair_truth, pair_predicted = {}, {}
        empty = frozenset()
        for a, b in combinations(MEMORIES, 2):
            pair = frozenset((a, b))
            only_a, only_b = frozenset((a,)), frozenset((b,))
            pair_truth[(a, b)] = truth[pair] - truth[only_a] - truth[only_b] + truth[empty]
            pair_predicted[(a, b)] = (
                estimates[pair] - estimates[only_a] - estimates[only_b] + estimates[empty]
            )
        threshold = 0.08
        tp = fp = fn = 0
        for pair, actual in pair_truth.items():
            pred = pair_predicted[pair]
            actual_sign = 0 if abs(actual) < threshold else (1 if actual > 0 else -1)
            predicted_sign = 0 if abs(pred) < threshold else (1 if pred > 0 else -1)
            tp += int(actual_sign != 0 and actual_sign == predicted_sign)
            fp += int(predicted_sign != 0 and predicted_sign != actual_sign)
            fn += int(actual_sign != 0 and actual_sign != predicted_sign)
        corr = spearmanr(list(ref_phi.values()), list(predicted_phi.values())).statistic
        attribution.append({
            "version": version, "template": template,
            "attr_mae": mean(abs(predicted_phi[m] - ref_phi[m]) for m in MEMORIES),
            "shapley_spearman": float(corr) if np.isfinite(corr) else 0.0,
            "signed_pair_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "reference_repetitions": repetitions,
            "reference_mean_se": mean(r["reference_se"] for r in transfer
                                      if r["version"] == version),
            "noop_reference_credit": ref_phi["noop"],
            "noop_predicted_credit": predicted_phi["noop"],
            "hidden_reference_credit": ref_phi["hidden_fix"],
            "hidden_predicted_credit": predicted_phi["hidden_fix"],
        })
    return attribution, transfer


def replay_efficiency(tasks: list[Task], *, seed: int, count: int = 100) -> dict:
    rng = Random(seed)
    incremental_calls = full_calls = matched = valid = fallback = 0
    for index in range(count):
        task = tasks[index % len(tasks)]
        memory = rng.choice(MEMORIES)
        background = {m for m in MEMORIES if m != memory and rng.random() < 0.5}
        inc = replay_pair(task, background, background | {memory},
                          propensity=1 / (len(MEMORIES) * 128), incremental=True)
        full = replay_pair(task, background, background | {memory},
                           propensity=inc.propensity, incremental=False)
        valid += int(inc.valid)
        fallback += int(inc.mode == "full")
        if inc.valid and full.valid:
            matched += int(inc.control.outcome == full.control.outcome
                           and inc.treatment.outcome == full.treatment.outcome
                           and inc.treatment.state_hash == full.treatment.state_hash)
            incremental_calls += inc.extra_calls
            full_calls += full.extra_calls
    return {
        "pairs": count, "valid": valid, "outcome_and_state_matches": matched,
        "fallback_pairs": fallback, "incremental_calls": incremental_calls,
        "full_calls": full_calls,
        "calls_saved_fraction": (1 - incremental_calls / full_calls) if full_calls else 0,
    }


def optimizer_regret(
    selected: tuple[str, ...],
    tasks: list[Task],
    *,
    budget: int, depth: int,
) -> dict:
    """Post-hoc small-pool oracle; test outcomes do not select the deployed bank."""
    def value(bank: tuple[str, ...]) -> float:
        successes = [int(run_task(task, retrieve(task, bank, depth=depth)).outcome or 0)
                     for task in tasks]
        return mean(successes) - 0.015 * bank_cost(bank) / 128
    oracle_bank, oracle_value = max(
        ((bank, value(bank)) for bank in all_banks(budget=budget)),
        key=lambda item: item[1],
    )
    chosen_value = value(selected)
    return {
        "selected_bank": "|".join(selected), "selected_value": chosen_value,
        "oracle_bank": "|".join(oracle_bank), "oracle_value": oracle_value,
        "test_oracle_gap": oracle_value - chosen_value,
        "candidate_banks": sum(1 for _ in all_banks(budget=budget)),
    }


def ablations(
    full_model: StructuralModel,
    all_traces: list[Trace],
    historical_traces: list[Trace],
    validation_traces: list[Trace],
    scenarios: list[Task],
    historical_scenarios: list[Task],
    test_tasks: list[Task],
    config: Config,
) -> list[dict]:
    variants: list[tuple[str, StructuralModel, list[Task], bool, bool, float]] = [
        ("CREDIT prototype", full_model, scenarios, True, True, 0.2),
    ]
    no_interaction = StructuralModel(max_order=1).fit(all_traces).calibrate(validation_traces)
    variants.append(("Without interaction terms", no_interaction, scenarios, True, True, 0.2))
    no_revalidation = StructuralModel(max_order=2).fit(historical_traces).calibrate(
        [t for t in validation_traces if t.version == "v1"]
    )
    variants.append(("Without drift probe data", no_revalidation,
                     historical_scenarios, True, True, 0.2))
    observational = [run_task(task, retrieve(task, MEMORIES, depth=config.retrieval_depth))
                     for task in make_tasks(len(all_traces), split="observational",
                                            version="v1", templates=("short", "chain"),
                                            seed=config.seed + 600)]
    observational_model = StructuralModel(max_order=2).fit(observational).calibrate(
        [t for t in validation_traces if t.version == "v1"]
    )
    variants.append(("Without exposure randomization", observational_model,
                     historical_scenarios, True, True, 0.2))
    variants.append(("Without joint bank selection", full_model, scenarios, False, True, 0.2))
    variants.append(("Without mixture robustness", full_model, scenarios, True, True, 0.0))
    variants.append(("Without version-scope filtering", full_model, scenarios, True, False, 0.2))
    rows = []
    for name, model, training_scenarios, joint, safeguard, radius in variants:
        bank, _ = optimize_bank(
            model, training_scenarios, budget=config.bank_budget,
            depth=config.retrieval_depth, joint=joint,
            drift_safeguard=safeguard, mixture_radius=radius,
            version_weights=(dict(config.future_version_weights)
                             if {task.version for task in training_scenarios} == {"v1", "v2"}
                             else None),
        )
        scored = _evaluate_banks({name: bank}, test_tasks,
                                 depth=config.retrieval_depth,
                                 drift_safeguard=safeguard)
        rows.append({
            "variant": name, "bank": "|".join(bank),
            "new_workflow_vsr": mean(r["success"] for r in scored
                                     if r["stratum"] == "new_workflow"),
            "drift_vsr": mean(r["success"] for r in scored if r["stratum"] == "drift"),
            "stale_rate": sum(r["stale_exposures"] for r in scored)
                          / max(1, sum(r["exposures"] for r in scored)),
            "role_support_failure_rate": mean(
                not model.predict(task, retrieve(task, bank, depth=config.retrieval_depth,
                                                drift_safeguard=safeguard)).supported
                for task in test_tasks
            ),
        })
    flat_bank = rank_bank(_flat_scores(all_traces, scenarios), config.bank_budget)
    flat_rows = _evaluate_banks({"Without structural transport": flat_bank},
                                test_tasks, depth=config.retrieval_depth)
    rows.append({
        "variant": "Flat outcome model", "bank": "|".join(flat_bank),
        "new_workflow_vsr": mean(r["success"] for r in flat_rows
                                 if r["stratum"] == "new_workflow"),
        "drift_vsr": mean(r["success"] for r in flat_rows if r["stratum"] == "drift"),
        "stale_rate": sum(r["stale_exposures"] for r in flat_rows)
                      / max(1, sum(r["exposures"] for r in flat_rows)),
        "role_support_failure_rate": None,
    })
    return rows


def _paired_delta(rows: list[dict], baseline: str, *, seed: int) -> dict:
    target = {r["task"]: r for r in rows if r["method"] == "CREDIT"}
    control = {r["task"]: r for r in rows if r["method"] == baseline}
    tasks = sorted(set(target) & set(control))
    differences = [(target[t]["family"],
                    target[t]["success"] - control[t]["success"]) for t in tasks]
    families = sorted({family for family, _ in differences})
    rng = Random(seed)
    draws = []
    for _ in range(800):
        chosen = [rng.choice(families) for _ in families]
        values = [d for family in chosen for group, d in differences if group == family]
        draws.append(mean(values))
    return {
        "baseline": baseline, "n_paired": len(tasks),
        "delta_vsr": mean(d for _, d in differences),
        "cluster_ci_low": float(np.quantile(draws, 0.025)),
        "cluster_ci_high": float(np.quantile(draws, 0.975)),
    }


def _plot_results(summary: list[dict], replay: dict, transfer: list[dict],
                  output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in summary if r["stratum"] == "all"]
    all_vsr = [r["vsr"] for r in summary if r["stratum"] == "all"]
    drift_vsr = [next(row["vsr"] for row in summary
                      if row["method"] == method and row["stratum"] == "drift")
                 for method in methods]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    y = np.arange(len(methods))
    axes[0].barh(y - 0.18, all_vsr, height=0.34, label="All")
    axes[0].barh(y + 0.18, drift_vsr, height=0.34, label="Post-drift")
    axes[0].set_yticks(y, methods, fontsize=8)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Verified success rate")
    axes[0].legend(fontsize=8)
    axes[1].bar(["Incremental", "Full"],
                [replay["incremental_calls"], replay["full_calls"]])
    axes[1].set_ylabel("Additional sandbox calls")
    axes[1].set_title("Paired replay cost")
    versions = ["v1", "v2"]
    errors = [mean(r["absolute_error"] for r in transfer if r["version"] == version)
              for version in versions]
    axes[2].bar(versions, errors, color=["#2a759b", "#9a669f"])
    axes[2].set_ylim(0, max(0.05, max(errors) * 1.2))
    axes[2].set_ylabel("Coalition utility MAE")
    axes[2].set_title("Held-out topology / drift")
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"causalcompose_dashboard.{suffix}", dpi=300)
    plt.close(fig)


def run(config: Config, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    train_tasks = make_tasks(max(48, config.train_pairs // 3), split="train",
                             version="v1", templates=("short", "chain", "double_lookup"),
                             seed=config.seed + 1, invalid_rate=0.1)
    probe_tasks = make_tasks(max(32, config.drift_pairs // 3), split="probe",
                             version="v2", templates=("short", "chain"),
                             seed=config.seed + 2, invalid_rate=0.1)
    validation_v1 = make_tasks(36, split="validation", version="v1",
                               templates=("short", "chain"), seed=config.seed + 3)
    validation_v2 = make_tasks(18, split="validation-probe", version="v2",
                               templates=("short", "chain"), seed=config.seed + 4)
    test_v1 = make_tasks(config.test_per_stratum, split="future-unseen",
                         version="v1", templates=("new_order", "long"),
                         seed=config.seed + 5)
    test_v2 = make_tasks(config.test_per_stratum, split="future-drift",
                         version="v2", templates=("new_order", "long"),
                         seed=config.seed + 6)
    test_tasks = test_v1 + test_v2
    historical_pairs, historical_traces = collect_pairs(
        train_tasks, config.train_pairs, seed=config.seed + 7
    )
    drift_pairs, drift_traces = collect_pairs(
        probe_tasks, config.drift_pairs, seed=config.seed + 8
    )
    pairs = historical_pairs + drift_pairs
    traces = historical_traces + drift_traces
    validation = _validation_traces(validation_v1 + validation_v2,
                                    seed=config.seed + 9)
    model = StructuralModel(max_order=2).fit(traces).calibrate(validation)
    scenarios = validation_v1 + validation_v2
    banks = _make_banks(model, pairs, traces, scenarios, config)
    test_rows = _evaluate_banks(banks, test_tasks, depth=config.retrieval_depth)
    main = _main_summary(test_rows, config.seed)
    attribution, transfer = attribution_and_transfer(
        model, seed=config.seed, repetitions=config.reference_repetitions
    )
    replay = replay_efficiency(train_tasks, seed=config.seed + 10)
    regret = optimizer_regret(banks["CREDIT"], test_tasks,
                              budget=config.bank_budget, depth=config.retrieval_depth)
    ablation = ablations(
        model, traces, historical_traces, validation, scenarios, validation_v1,
        test_tasks, config,
    )
    unknown = make_tasks(24, split="unsupported", version="v2",
                         templates=("unknown_role",), seed=config.seed + 11)
    abstain_rate = mean(not model.predict(task, ()).supported for task in unknown)
    validation_rows = _evaluate_banks(banks, scenarios, depth=config.retrieval_depth)
    baseline = max(
        (method for method in banks if method != "CREDIT"),
        key=lambda method: mean(r["success"] for r in validation_rows
                                if r["method"] == method),
    )
    paired = _paired_delta(test_rows, baseline, seed=config.seed + 12)
    _write_csv(output / "raw_test_episodes.csv", test_rows)
    _write_csv(output / "interventions.csv", [{
        "task": pair.control.task_id,
        "version": pair.control.version,
        "control_exposure": "|".join(pair.control.exposure),
        "treatment_exposure": "|".join(pair.treatment.exposure),
        "propensity": pair.propensity,
        "control_outcome": pair.control.outcome,
        "treatment_outcome": pair.treatment.outcome,
        "control_checkpoint_hash": pair.control.checkpoint_hash,
        "treatment_checkpoint_hash": pair.treatment.checkpoint_hash,
        "control_state_hash": pair.control.state_hash,
        "treatment_state_hash": pair.treatment.state_hash,
        "valid": int(pair.valid),
        "mode": pair.mode,
        "extra_calls": pair.extra_calls,
    } for pair in pairs])
    _write_csv(output / "main_results.csv", main)
    _write_csv(output / "attribution.csv", attribution)
    _write_csv(output / "transfer_coalitions.csv", transfer)
    _write_csv(output / "ablations.csv", ablation)
    _plot_results(main, replay, transfer, output)
    split_manifest = {
        "train": [task.uid for task in train_tasks],
        "safe_drift_probes": [task.uid for task in probe_tasks],
        "validation": [task.uid for task in scenarios],
        "future_test": [task.uid for task in test_tasks],
        "reference_coalitions": ["reference v1 and v2 tasks generated from independent seeds"],
    }
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    report = {
        "status": "measured synthetic CausalCompose prototype only",
        "seed": config.seed, "config": asdict(config),
        "training_manifest": {
            "historical_tasks": len(train_tasks),
            "safe_drift_probe_tasks": len(probe_tasks),
            "validation_tasks": len(scenarios),
            "heldout_test_tasks": len(test_tasks),
            "training_pairs": len(pairs),
            "valid_pairs": sum(p.valid for p in pairs),
            "training_extra_calls": sum(p.extra_calls for p in pairs),
            "test_templates": sorted({t.template for t in test_tasks}),
        },
        "banks": {name: list(bank) for name, bank in banks.items()},
        "validation_selected_baseline": baseline,
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"), "scipy": version("scipy"),
            "scikit-learn": version("scikit-learn"),
            "matplotlib": version("matplotlib"),
        },
        "replay": replay, "optimizer": regret,
        "unsupported_role_abstention_rate": abstain_rate,
        "paired_credit_vs_best_baseline": paired,
        "mean_transfer_mae": mean(row["absolute_error"] for row in transfer),
        "empirical_lower_coverage": mean(row["lower_covered_with_ref_se"]
                                         for row in transfer if row["supported"]),
        "limitations": [
            "No observed provider drift or ToolSandbox-M data are in this repository.",
            "The sandbox is in-process; API calls are simulated calls, not HTTP traffic.",
            "The lower bound is empirically calibrated, not formally certified.",
            "CMP-adapted and CAR-adapted are task-specific approximations, not reproductions of original code.",
            "The small-pool oracle is evaluated post hoc and excluded from deployment.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/causalcompose"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-pairs", type=int, default=384)
    parser.add_argument("--drift-pairs", type=int, default=128)
    parser.add_argument("--test-per-stratum", type=int, default=80)
    parser.add_argument("--reference-repetitions", type=int, default=64)
    args = parser.parse_args()
    config = Config(
        seed=args.seed, train_pairs=args.train_pairs, drift_pairs=args.drift_pairs,
        test_per_stratum=args.test_per_stratum,
        reference_repetitions=args.reference_repetitions,
    )
    report = run(config, args.output)
    print(json.dumps({
        "output": str(args.output), "paired_comparison": report["paired_credit_vs_best_baseline"],
        "replay": report["replay"], "unsupported_abstention":
        report["unsupported_role_abstention_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
