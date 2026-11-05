"""Evaluation-only diagnostics for replay validity and the synthetic ceiling.

Ground-truth kernels are used only here to diagnose the simulator. They are
never passed to the learner, bank selector, or deployment policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean

from .experiments import _write_csv
from .policy import RECENCY, all_banks, bank_cost, retrieve
from .sandbox import (
    MEMORIES, Task, _decision_probability, _service_probability,
    make_tasks, replay_pair,
)


def expected_success(task: Task, bank: tuple[str, ...], depth: int = 4) -> float:
    """Analytic nominal expectation for this simulator's independent kernels."""
    exposure = frozenset(retrieve(task, bank, depth=depth))
    probability = 1.0
    for role in task.roles:
        probability *= (
            _decision_probability(role, task.version, exposure, task.drift_intensity)
            * _service_probability(role, task.version, 1, task.drift_intensity)
        )
    return probability


def audit_replay(tasks_per_version: int = 32) -> tuple[list[dict], list[dict]]:
    rows, examples = [], []
    tasks = [
        task for version in ("v1", "v2")
        for task in make_tasks(
            tasks_per_version, split="audit-replay", version=version,
            templates=("short", "double_lookup", "chain", "new_order"), seed=120,
        )
    ]
    for memory in MEMORIES:
        other = tuple(m for m in MEMORIES if m != memory)
        backgrounds = [frozenset(group) for size in range(len(other) + 1)
                       for group in combinations(other, size)]
        states = outcomes = 0
        for task in tasks:
            for background in backgrounds:
                treatment = background | {memory}
                incremental = replay_pair(task, background, treatment,
                                          propensity=1 / len(backgrounds))
                full = replay_pair(task, background, treatment,
                                   propensity=1 / len(backgrounds), incremental=False)
                state_mismatch = incremental.treatment.state_hash != full.treatment.state_hash
                outcome_mismatch = incremental.treatment.outcome != full.treatment.outcome
                states += int(state_mismatch)
                outcomes += int(outcome_mismatch)
                if state_mismatch and len(examples) < 20:
                    examples.append({
                        "task": task.uid, "memory": memory,
                        "background": "|".join(sorted(background)),
                        "mode": incremental.mode,
                        "incremental_outcome": incremental.treatment.outcome,
                        "full_outcome": full.treatment.outcome,
                    })
        rows.append({"memory": memory, "pairs": len(tasks) * len(backgrounds),
                     "state_mismatches": states, "outcome_mismatches": outcomes})
    return rows, examples


def run_audit(reports: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        task for version in ("v1", "v2")
        for task in make_tasks(2, split="audit-ceiling", version=version,
                               templates=("new_order", "long"), seed=200)
    ]
    candidates = [
        {"bank": "|".join(bank), "expected_vsr": mean(expected_success(t, bank) for t in tasks),
         "tokens": bank_cost(bank)}
        for bank in all_banks(budget=512)
    ]
    for row in candidates:
        row["expected_cost_adjusted_value"] = row["expected_vsr"] - 0.015 * row["tokens"] / 128
    oracle_vsr = max(candidates, key=lambda row: row["expected_vsr"])
    oracle_objective = max(candidates, key=lambda row: row["expected_cost_adjusted_value"])
    methods, overlap = [], []
    for path in sorted(reports.glob("seed_*/report.json")):
        report = json.loads(path.read_text())
        banks = report["banks"]
        credit = tuple(sorted(banks["CREDIT"]))
        for method, stored in banks.items():
            bank = tuple(sorted(stored))
            nominal = mean(expected_success(task, bank) for task in tasks)
            methods.append({
                "seed": report["seed"], "method": method, "bank": "|".join(bank),
                "expected_vsr": nominal,
                "expected_vsr_oracle_gap": oracle_vsr["expected_vsr"] - nominal,
                "expected_cost_adjusted_value": nominal - 0.015 * bank_cost(bank) / 128,
                "same_bank_as_credit": int(bank == credit),
            })
        for baseline in ("FIFO", "Version-aware"):
            bank = tuple(banks[baseline])
            overlap.append({
                "seed": report["seed"], "baseline": baseline,
                "same_bank": int(set(bank) == set(credit)),
                "exposure_agreement": mean(
                    retrieve(task, bank) == retrieve(task, credit) for task in tasks
                ),
            })
    replay, examples = audit_replay()
    _write_csv(output / "replay_by_memory.csv", replay)
    with (output / "replay_mismatches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "task", "memory", "background", "mode", "incremental_outcome", "full_outcome",
        ))
        writer.writeheader()
        writer.writerows(examples)
    _write_csv(output / "expected_method_values.csv", methods)
    _write_csv(output / "bank_overlap.csv", overlap)
    _write_csv(output / "expected_bank_oracle.csv", candidates)
    topology_gap = max(
        abs(expected_success(Task("a", ("lookup", "transform", "submit"),
                                  version, "chain", "audit", 1), bank)
            - expected_success(Task("b", ("transform", "lookup", "submit"),
                                    version, "new_order", "audit", 1), bank))
        for bank in all_banks(budget=512) for version in ("v1", "v2")
    )
    summary = {
        "scope": "evaluation-only diagnostics of the eight-memory synthetic simulator",
        "reports_source": str(reports),
        "replay_pairs": sum(row["pairs"] for row in replay),
        "state_mismatches": sum(row["state_mismatches"] for row in replay),
        "outcome_mismatches": sum(row["outcome_mismatches"] for row in replay),
        "exact_expected_vsr_oracle": oracle_vsr,
        "exact_expected_cost_adjusted_oracle": oracle_objective,
        "recency_metadata_top_four": sorted(MEMORIES, key=lambda m: RECENCY[m], reverse=True)[:4],
        "maximum_chain_to_new_order_expected_vsr_difference": topology_gap,
        "maximum_bank_memories": 512 // 128,
        "retrieval_depth": 4,
        "retrieval_competition_possible_at_main_budget": False,
        "expected_value_mixture": "equal v1/v2 and new_order/long strata, depth=4, budget=512",
        "limitations": [
            "Oracle values use planted kernels for diagnosis only; they are not a deployable baseline.",
            "The expected cost-adjusted oracle uses the nominal equal mixture, not the selector's robust validation mixture.",
            "These independent binary kernels do not represent a history-dependent LLM agent.",
        ],
    }
    (output / "audit_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("results/multiseed"))
    parser.add_argument("--output", type=Path, default=Path("results/audit"))
    args = parser.parse_args()
    print(json.dumps(run_audit(args.reports, args.output), indent=2))


if __name__ == "__main__":
    main()
