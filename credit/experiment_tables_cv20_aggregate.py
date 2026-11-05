"""Aggregate the two family-disjoint ToolSandbox outer folds for CV20."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/credit_experiment_tables_v1/public_cv20_v1")
FOLDS = (ROOT / "fold0_run", ROOT / "fold1_run")
OUT = ROOT / "aggregate"
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 292_021
EXPECTED_FAMILIES = 20
EXPECTED_SESSIONS_PER_CONFIG = 240


def read(path: Path):
    return json.loads(path.read_text())


def family(task: str) -> str:
    return task.removesuffix("_alt").removesuffix("_implicit")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def summarize_config(rows: list[dict], seed: int) -> dict:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[family(row["task"])].append(row)
    if len(by_family) != EXPECTED_FAMILIES:
        raise ValueError(f"expected 20 test families, got {len(by_family)}")
    if any(len(items) != 12 for items in by_family.values()):
        raise ValueError("each family must contain 2 task variants x 6 actor seeds")
    native = np.asarray([np.mean([r["native_score"] for r in by_family[k]])
                         for k in sorted(by_family)])
    success = np.asarray([np.mean([r["success"] for r in by_family[k]])
                          for k in sorted(by_family)])
    return {
        "sessions": len(rows),
        "test_families": len(by_family),
        "task_variants": len({r["task"] for r in rows}),
        "native_score": float(native.mean()),
        "native_score_family_bootstrap_95ci": bootstrap_ci(native, seed),
        "vsr": float(success.mean()),
        "vsr_family_bootstrap_95ci": bootstrap_ci(success, seed + 1),
        "online_calls": int(sum(r["online_calls"] for r in rows)),
        "family_native_scores": {k: float(native[i])
                                 for i, k in enumerate(sorted(by_family))},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict] = []
    protocols, audits = [], []
    main_adaptation = defaultdict(int)
    budget_costs = defaultdict(lambda: {"adaptation_calls": 0, "total_calls": 0})
    ablation_costs = defaultdict(lambda: {"total_calls": 0})
    for fold_index, folder in enumerate(FOLDS):
        protocols.append(read(folder / "frozen_protocol.json"))
        audits.append(read(folder / "public_audit.json"))
        logical = read(folder / "logical_results.json")
        all_rows.extend({**row, "fold": fold_index} for row in logical)
        for row in read(folder / "main_summary.json"):
            main_adaptation[row["method"]] += int(row["adaptation_calls"])
        for row in read(folder / "budget_summary.json"):
            key = (float(row["multiplier"]), row["method"])
            budget_costs[key]["adaptation_calls"] += int(row["adaptation_calls"])
            budget_costs[key]["total_calls"] += int(row["total_calls"])
        for row in read(folder / "ablation_summary.json"):
            ablation_costs[row["variant"]]["total_calls"] += int(row["total_calls"])

    test_families = sorted({f for protocol in protocols for f in protocol["test_families"]})
    if len(test_families) != EXPECTED_FAMILIES:
        raise ValueError(f"outer folds cover {len(test_families)} families, expected 20")
    for index, (protocol, audit) in enumerate(zip(protocols, audits)):
        if audit["checks"]["main_logical_sessions"] != 840:
            raise ValueError(f"fold {index} main logical session count failed")
        if not audit["checks"]["all_native_evaluations_present"]:
            raise ValueError(f"fold {index} has missing native evaluations")
        if not audit["checks"]["no_test_tasks_in_adaptation"]:
            raise ValueError(f"fold {index} test/adaptation split is contaminated")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        if row["group"] == "main":
            key = f"main::{row['method']}"
        elif row["group"] == "budget":
            key = f"budget::{float(row['multiplier']):g}::{row['method']}"
        else:
            key = f"ablation::{row['variant']}"
        grouped[key].append(row)

    summaries = {}
    for index, (key, rows) in enumerate(sorted(grouped.items())):
        if len(rows) != EXPECTED_SESSIONS_PER_CONFIG:
            raise ValueError(f"{key} has {len(rows)} sessions, expected 240")
        summaries[key] = summarize_config(rows, BOOTSTRAP_SEED + 10 * index)
        if key.startswith("main::"):
            method = key.split("::", 1)[1]
            adaptation = main_adaptation[method]
            summaries[key].update({"adaptation_calls": adaptation,
                                   "total_calls": adaptation + summaries[key]["online_calls"]})
        elif key.startswith("budget::"):
            _, multiplier, method = key.split("::")
            cost = budget_costs[(float(multiplier), method)]
            summaries[key].update(cost)
        else:
            variant = key.split("::", 1)[1]
            total = ablation_costs[variant]["total_calls"]
            summaries[key].update({"total_calls": total,
                                   "adaptation_calls": total - summaries[key]["online_calls"]})

    contrasts = {}
    for key in summaries:
        if key.startswith("main::"):
            reference = "main::Semantic"
        elif key.startswith("budget::"):
            reference = "budget::" + key.split("::")[1] + "::Semantic"
        else:
            reference = "ablation::Full CREDIT"
        if key == reference:
            delta = np.zeros(EXPECTED_FAMILIES)
        else:
            a = summaries[key]["family_native_scores"]
            b = summaries[reference]["family_native_scores"]
            delta = np.asarray([a[f] - b[f] for f in test_families])
        contrasts[key] = {
            "reference": reference,
            "mean_family_delta": float(delta.mean()),
            "family_bootstrap_95ci": bootstrap_ci(
                delta, BOOTSTRAP_SEED + 50_000 + len(contrasts)),
        }

    result = {
        "status": "completed; descriptive cross-fitted evaluation",
        "suite": "official public ToolSandbox scenarios at pinned upstream commit",
        "upstream_commit": protocols[0]["upstream_commit"],
        "design": {
            "outer_folds": 2,
            "development_certification_test_families_per_fold": [8, 2, 10],
            "unique_test_families_across_folds": len(test_families),
            "task_variants_per_family": 2,
            "actor_seeds_per_task": len(protocols[0]["actor_seeds"]["test"]),
            "sessions_per_configuration": EXPECTED_SESSIONS_PER_CONFIG,
            "bootstrap_unit": "family; 20,000 resamples",
            "hypothesis_tests": "not reported; descriptive estimates and intervals only",
        },
        "test_families": test_families,
        "configurations": summaries,
        "descriptive_contrasts": contrasts,
        "source_sha256": {str(folder / "public_audit.json"): sha(folder / "public_audit.json")
                           for folder in FOLDS},
    }
    (OUT / "cv20_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# ToolSandbox 20-family cross-fitted results", "",
             "Descriptive results only. Each configuration has 240 logical test sessions "
             "(20 families × 2 task variants × 6 actor seeds); confidence intervals resample "
             "families, not individual sessions.", "",
             "| Configuration | Sessions | Native score (family bootstrap 95% CI) | VSR (family bootstrap 95% CI) | Total calls |",
             "|---|---:|---:|---:|---:|"]
    for key, value in summaries.items():
        nci = ", ".join(f"{x:.3f}" for x in value["native_score_family_bootstrap_95ci"])
        vci = ", ".join(f"{x:.3f}" for x in value["vsr_family_bootstrap_95ci"])
        lines.append(f"| {key} | {value['sessions']} | {value['native_score']:.3f} [{nci}] | "
                     f"{value['vsr']:.3f} [{vci}] | {value.get('total_calls', 'N/A')} |")
    (OUT / "cv20_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT), "families": len(test_families),
                      "configurations": len(summaries),
                      "main_sessions_per_method": {
                          k: v["sessions"] for k, v in summaries.items()
                          if k.startswith("main::")}}, indent=2))


if __name__ == "__main__":
    main()
