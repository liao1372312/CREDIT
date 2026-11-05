"""Run CREDIT-only E2 and Hoeffding E5 diagnostics for the integrated tables.

These deterministic synthetic diagnostics do not use remote APIs. They are
kept separate from the new public-workload/capacity collection.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

from . import revised_credit_only as core


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def forecast_diagnostics() -> dict:
    train = core.make_families(9001, core.TRAIN_FAMILIES, "train", allow_stale=False)
    heldout = core.make_families(9007, core.HELDOUT_FAMILIES, "heldout", allow_stale=True)
    x, y, training_calls = core.collect_training(
        train, core.BUDGET_DRAWS[core.PRIMARY_BUDGET], 9301 + int(float(core.PRIMARY_BUDGET) * 100))
    model = core.fit_model(x, y)
    pair_columns = len(core.MODELED_PAIRS)
    x_no_pairs = x[:, :-pair_columns] if pair_columns else x
    model_no_pairs = core.fit_model(x_no_pairs, y)
    flat_prediction = float(np.clip(np.mean(y), .02, .98))
    rng = np.random.default_rng(101_027)
    grouped = {name: {"new_workflow": [], "drift": []}
               for name in ("flat", "role_no_pairs", "full_role_model")}
    n_exec = 256
    for family in heldout:
        applicable = [i for i in range(core.MEMORIES) if core.applicable(family, i)]
        exposures = [e for size in range(min(2, len(applicable)) + 1)
                     for e in itertools.combinations(applicable, size)]
        # Balance the supported exposure set while keeping each target family
        # disjoint from the model-fitting families.
        for exposure in exposures:
            true_p = core.probability(family, exposure)
            features = core.feature_row(family, exposure)
            predictions = {
                "flat": flat_prediction,
                "role_no_pairs": float(np.clip(features[:-pair_columns] @ model_no_pairs, .02, .98))
                    if pair_columns else float(np.clip(features @ model_no_pairs, .02, .98)),
                "full_role_model": core.predicted(model, family, exposure),
            }
            # Independent execution references, sampled separately for each
            # held-out exposure. Use their empirical mean as the target.
            observed = rng.binomial(n_exec, true_p) / n_exec
            group = "drift" if family.stale and any(m in family.stale for m in exposure) else "new_workflow"
            for method, predicted_p in predictions.items():
                grouped[method][group].append(abs(predicted_p - observed))
    metrics = {
        method: {
            "new_workflow_mae": float(np.mean(values["new_workflow"])),
            "drift_mae": float(np.mean(values["drift"])),
            "new_workflow_exposure_cells": len(values["new_workflow"]),
            "drift_exposure_cells": len(values["drift"]),
        }
        for method, values in grouped.items()
    }
    full = metrics["full_role_model"]
    result = {
        **full,
        "predictor_comparison": metrics,
        "flat_prediction_probability": flat_prediction,
        "role_no_pairs_fit": "ridge least squares over intercept, roles, workflow depth, branching and singleton memory terms",
        "independent_executions_per_cell": n_exec,
        "training_families": len(train),
        "heldout_families": len(heldout),
        "training_calls": training_calls,
        "scope": "synthetic held-out family/version diagnostic; not ToolSandbox or tau-bench VSR",
    }
    if not result["drift_exposure_cells"]:
        raise RuntimeError("no held-out drift exposure cells")
    return result


def hoeffding_e5(n: int, reps: int = 2000, cycles: int = 50) -> dict:
    gains = {
        "null": np.zeros(8),
        "harmful_guard": np.asarray([.03] * 7 + [-.10]),
        "safe_2pp": np.full(8, .02),
        "safe_5pp": np.full(8, .05),
        "safe_10pp": np.full(8, .10),
    }
    rates = {}
    for i, (label, truth) in enumerate(gains.items()):
        switched = core.certification_stream(
            "finite", truth, cycles=cycles, margin=.05, quality="good",
            seed=230_000 + n * 17 + i, reps=reps, n=n)
        rates[label] = float(switched.mean())
    return {
        "sample_pairs_per_stratum_per_attempt": n,
        "strata": 8,
        "streams": reps,
        "attempts_per_stream": cycles,
        "alpha_star": .05,
        "margin": .05,
        "null_any_switch_error_pct": 100 * rates["null"],
        "harmful_guard_any_switch_error_pct": 100 * rates["harmful_guard"],
        "power_2pp_pct": 100 * rates["safe_2pp"],
        "power_5pp_pct": 100 * rates["safe_5pp"],
        "power_10pp_pct": 100 * rates["safe_10pp"],
        "certification_calls_per_attempt": 2 * 8 * n,
        "scope": "CREDIT Hoeffding fixed-sample certificate with alpha spending",
    }


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    output.mkdir(parents=True)
    old_anchor = core.E2_ESTIMATION_CALLS
    e2_grid, rows_grid = {}, {}
    for anchor in (512, 1024, 2048):
        core.E2_ESTIMATION_CALLS = anchor
        e2_grid[str(anchor)], rows_grid[str(anchor)] = core.e2_estimation()
    core.E2_ESTIMATION_CALLS = old_anchor
    e2, rows = e2_grid["1024"], rows_grid["1024"]
    forecast = forecast_diagnostics()
    cert = {str(n): hoeffding_e5(n) for n in (200, 5000)}
    protocol = {
        "experiment": "Integrated paper CREDIT-only diagnostics",
        "tables": ["tab:mechanism-diagnostics Factorial exposure and Full role model rows",
                   "tab:certification-new Hoeffding, spent alpha rows"],
        "E2": {"frozen_synthetic_families": 180,
               "estimation_calls_per_family_anchor": 1024,
               "retained_budget_grid": [512, 1024, 2048],
               "reference_repetitions_per_coalition": core.E2_REFERENCE_REPS,
               "pair_detection_threshold": core.E2_PAIR_THRESHOLD,
               "heldout_forecasts": forecast},
        "E5": {"streams": 2000, "attempts": 50, "strata": 8,
               "alpha_star": .05, "margin": .05,
               "sample_sizes_per_stratum_per_attempt": [200, 5000],
               "rule": "CREDIT Hoeffding finite bounds with alpha spending"},
        "no_remote_or_LLM_calls": True,
        "source_sha256": sha(Path(__file__)),
        "generator_source_sha256": sha(Path(core.__file__)),
    }
    write(output / "frozen_protocol.json", protocol)
    write(output / "E2_factorial_summary.json", e2_grid)
    write(output / "E2_forecast_summary.json", forecast)
    write(output / "E5_hoeffding_summary.json", cert)
    with (output / "E2_family_rows.jsonl").open("w") as f:
        for anchor in (512, 1024, 2048):
            for row in rows_grid[str(anchor)]:
                f.write(json.dumps({"anchor_calls": anchor, **row}, sort_keys=True) + "\n")
    summary = {"E2_factorial_primary_1024": e2,
               "E2_factorial_budget_grid": e2_grid,
               "E2_full_role_forecast": forecast,
               "E5_hoeffding": cert,
               "scope": "CREDIT-only synthetic diagnostics; does not populate new public workload or global-capacity results"}
    write(output / "summary.json", summary)
    manifest = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    write(output / "artifact_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.output)

if __name__ == "__main__":
    main()
