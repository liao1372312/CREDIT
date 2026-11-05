"""Build an all-in controlled-recovery cost and amortization report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from .user_data import digest


def break_even_horizon(treatment: dict, baseline: dict) -> float | None:
    """Tasks per adapted block where calls per verified success cross."""
    success_ratio = treatment["vsr"] / baseline["vsr"]
    numerator = (treatment["mean_offline_calls_per_seed_cycle"] -
                 success_ratio * baseline["mean_offline_calls_per_seed_cycle"])
    denominator = (treatment["vsr"] * baseline["online_calls_per_task"] /
                   baseline["vsr"] - treatment["online_calls_per_task"])
    if numerator <= 0:
        return 0.0
    if denominator <= 0:
        return None
    return numerator / denominator


def build(root: Path) -> dict:
    costs_path = root / "costs.json"
    raw_path = root / "raw_test_results.json"
    statistics_path = root / "statistics.json"
    audit_path = root / "execution_audit.json"
    costs = json.loads(costs_path.read_text())
    raw = json.loads(raw_path.read_text())
    statistics = json.loads(statistics_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit.get("verified"):
        raise ValueError("source execution audit is not verified")

    offline = defaultdict(int)
    blocks = defaultdict(set)
    for row in costs:
        offline[row["method"]] += int(row["attributed_api_calls"])
        blocks[row["method"]].add((row["seed"], row["cycle"]))
    online_calls = defaultdict(int)
    episodes = defaultdict(int)
    successes = defaultdict(int)
    for row in raw:
        method = row["method"]
        online_calls[method] += int(row["api_calls"])
        episodes[method] += 1
        successes[method] += int(row["success"])

    stat_by_method = {row["method"]: row for row in statistics["main"]}
    methods = []
    horizons = (1024, 10000, 100000, 1000000)
    for method in sorted(stat_by_method):
        n_blocks = len(blocks[method])
        if n_blocks != 24:
            raise ValueError(f"expected 24 seed-cycle blocks for {method}, got {n_blocks}")
        item = {"method": method, "vsr": stat_by_method[method]["vsr"],
            "episodes": episodes[method], "verified_successes": successes[method],
            "online_calls": online_calls[method],
            "online_calls_per_task": online_calls[method] / episodes[method],
            "offline_calls": offline[method], "seed_cycle_blocks": n_blocks,
            "mean_offline_calls_per_seed_cycle": offline[method] / n_blocks}
        item["study_all_in_calls"] = item["offline_calls"] + item["online_calls"]
        item["amortized_calls_per_task"] = {
            str(horizon): (item["online_calls_per_task"] +
                           item["mean_offline_calls_per_seed_cycle"] / horizon)
            for horizon in horizons}
        item["amortized_calls_per_verified_success"] = {
            str(horizon): (item["online_calls_per_task"] +
                           item["mean_offline_calls_per_seed_cycle"] / horizon) /
                          item["vsr"] for horizon in horizons}
        methods.append(item)

    by_method = {row["method"]: row for row in methods}
    credit = by_method["CREDIT-Covered"]
    crossovers = []
    for baseline, row in by_method.items():
        if baseline == "CREDIT-Covered":
            continue
        value = break_even_horizon(credit, row)
        crossovers.append({"baseline": baseline,
            "tasks_per_adapted_seed_cycle": None if value is None else math.ceil(value),
            "criterion": "CREDIT calls per verified success <= baseline"})
    return {"source": str(root), "source_sha256": {
            "costs.json": digest(costs_path), "raw_test_results.json": digest(raw_path),
            "statistics.json": digest(statistics_path),
            "execution_audit.json": digest(audit_path)},
        "scope": "all physical offline adaptation calls plus held-out online execution calls",
        "amortization_unit": "tasks served after one seed-cycle adaptation",
        "equal_call_value_assumption": "one offline and one online physical API call have equal unit cost",
        "methods": methods, "credit_break_even": crossovers,
        "source_audit_verified": True}


def markdown(report: dict) -> str:
    methods = report["methods"]
    lines = ["# Controlled Recovery All-In Cost Report", "",
        "This report counts every attributed pre-test adaptation call and every "
        "held-out execution call. Amortization assumes that one physical API call "
        "has the same unit cost in both phases.", "",
        "| Method | VSR | Offline calls | Online calls/task | All-in calls in study | "
        "Calls/task at 10k | Calls/task at 100k |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in methods:
        lines.append("| {method} | {vsr:.2%} | {offline_calls:,} | "
            "{online_calls_per_task:.2f} | {study_all_in_calls:,} | {ten:.2f} | {hundred:.2f} |".format(
                **row, ten=row["amortized_calls_per_task"]["10000"],
                hundred=row["amortized_calls_per_task"]["100000"]))
    lines.extend(["", "## Break-even Against CREDIT", "",
        "The threshold is the number of deployment tasks after one adaptation block "
        "at which CREDIT's total calls per verified success become no greater than "
        "the comparator's.", "", "| Comparator | Tasks per adapted block |", "|---|---:|"])
    for row in report["credit_break_even"]:
        value = row["tasks_per_adapted_seed_cycle"]
        lines.append(f"| {row['baseline']} | {'never' if value is None else f'{value:,}'} |")
    lines.extend(["", "These thresholds are descriptive engineering quantities. "
        "They do not assign a monetary value to a verified success and do not include "
        "provider-specific token prices or wall-clock opportunity cost.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
        default=Path("results/credit_v3_remote_recovery_baselines_confirm_v1"))
    parser.add_argument("--output", type=Path,
        default=Path("results/credit_v4_controlled_total_cost_v1"))
    parser.add_argument("--markdown", type=Path,
        default=Path("CONTROLLED_TOTAL_COST_RESULTS.md"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    report = build(args.source)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    args.markdown.write_text(markdown(report))
    print(json.dumps({"output": str(args.output), "methods": len(report["methods"]),
                      "report_sha256": digest(args.output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
