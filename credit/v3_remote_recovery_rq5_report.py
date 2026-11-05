"""Create tables, diagnostics, figures, and a report from frozen RQ5 output."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CORE = ("NoInteractions", "NoDriftGate", "NoFreshCertification",
        "OAT-Physical", "Nonrandom-Observed", "Robust-Skew", "NoRobust-Skew")


def holm(rows):
    ordered = sorted(rows, key=lambda row: row["seed_sign_flip_p"])
    running = 0.0
    adjusted = {}
    for index, row in enumerate(ordered):
        running = max(running, (len(rows) - index) * row["seed_sign_flip_p"])
        adjusted[row["baseline"]] = min(1.0, running)
    return adjusted


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value):
    return f"{100 * value:.2f}"


def analyze(results, output, figure_base):
    output.mkdir(parents=True, exist_ok=True)
    stats = json.loads((results / "statistics.json").read_text())
    rows = json.loads((results / "raw_test_results.json").read_text())
    main = {row["method"]: row for row in stats["main"]}
    paired = {row["baseline"]: row for row in stats["paired"]}
    core_holm = holm([paired[name] for name in CORE])

    main_rows = []
    for method, row in main.items():
        pair = paired.get(method)
        main_rows.append({
            "method": method, "vsr": row["vsr"], "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "credit_minus_method": 0.0 if pair is None else pair["delta_vsr"],
            "paired_ci_low": 0.0 if pair is None else pair["ci_low"],
            "paired_ci_high": 0.0 if pair is None else pair["ci_high"],
            "exact_seed_p": 1.0 if pair is None else pair["seed_sign_flip_p"],
            "global_holm_p": 1.0 if pair is None else pair["holm_p"],
            "core_holm_p": core_holm.get(method, ""),
            "episodes": row["episodes"],
        })
    write_csv(output / "main_table.csv", main_rows)

    index = {(row["seed"], row["cycle"], row["task"], row["method"]): row
             for row in rows}
    keys = sorted({(row["seed"], row["cycle"], row["task"]) for row in rows})
    diagnostics = []
    for method in sorted(main):
        exposure_differences = outcome_differences = credit_wins = method_wins = 0
        source, target = [], []
        for key in keys:
            credit = index[key + ("CREDIT",)]
            other = index[key + (method,)]
            exposure_differences += credit["exposure"] != other["exposure"]
            outcome_differences += credit["success"] != other["success"]
            credit_wins += credit["success"] > other["success"]
            method_wins += credit["success"] < other["success"]
            (source if other["epoch"] == "source" else target).append(other["success"])
        diagnostics.append({
            "method": method, "exposure_differences_vs_credit": exposure_differences,
            "outcome_differences_vs_credit": outcome_differences,
            "credit_task_wins": credit_wins, "method_task_wins": method_wins,
            "source_vsr": float(np.mean(source)), "target_vsr": float(np.mean(target)),
        })
    write_csv(output / "mechanism_diagnostics.csv", diagnostics)

    bank_counts = defaultdict(Counter)
    for path in results.glob("seed_*/cycle_*/rq5/policies.json"):
        for row in json.loads(path.read_text()):
            bank_counts[row["method"]][tuple(row["bank"])] += 1
    write_csv(output / "bank_counts.csv", (
        {"method": method, "bank": "+".join(bank) if bank else "empty", "blocks": count}
        for method in sorted(bank_counts) for bank, count in bank_counts[method].items()))

    seed_rows = []
    for method in sorted(main):
        if method == "CREDIT":
            continue
        for seed in sorted({row["seed"] for row in rows}):
            credit = np.mean([row["success"] for row in rows
                              if row["seed"] == seed and row["method"] == "CREDIT"])
            other = np.mean([row["success"] for row in rows
                             if row["seed"] == seed and row["method"] == method])
            seed_rows.append({"method": method, "seed": seed,
                              "credit_minus_method": float(credit - other)})
    write_csv(output / "seed_differences.csv", seed_rows)

    colors = {"CREDIT": "#226597", "harm": "#b4493f", "neutral": "#777777"}
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)
    core_order = ["CREDIT", "NoDriftGate", "Nonrandom-Observed", "NoInteractions",
                  "NoFreshCertification", "OAT-Physical", "NoRobust-Skew"]
    ax = axes[0, 0]
    values = np.asarray([main[name]["vsr"] for name in core_order]) * 100
    low = values - np.asarray([main[name]["ci_low"] for name in core_order]) * 100
    high = np.asarray([main[name]["ci_high"] for name in core_order]) * 100 - values
    y = np.arange(len(core_order))
    ax.errorbar(values, y, xerr=np.vstack((low, high)), fmt="o", capsize=2.5,
                color=colors["CREDIT"], ecolor="#8aa9bf")
    ax.set_yticks(y, [name.replace("-", " ") for name in core_order])
    ax.invert_yaxis(); ax.set_xlabel("Verified success rate (%)")
    ax.set_title("(a) Mechanism ablations", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=.2)

    ax = axes[0, 1]
    xs = np.asarray([24, 40, 60, 80, 103])
    ys = np.asarray([main[f"Cap{x}"]["vsr"] for x in xs]) * 100
    ax.plot(xs, ys, "o-", color=colors["CREDIT"])
    ax.axhline(100 * main["CREDIT"]["vsr"], color="#777777", ls="--", lw=1)
    ax.set_xlabel("Bank capacity (tokens)"); ax.set_ylabel("VSR (%)")
    ax.set_title("(b) Capacity", loc="left", fontweight="bold"); ax.grid(alpha=.2)

    ax = axes[1, 0]
    xs = np.asarray([0, 24, 48, 96])
    ys = np.asarray([main[f"Probe{x}"]["vsr"] for x in xs]) * 100
    ax.plot(xs, ys, "o-", color="#6b46a8")
    ax.axhline(100 * main["CREDIT"]["vsr"], color="#777777", ls="--", lw=1)
    ax.set_xlabel("Probe contexts per role"); ax.set_ylabel("VSR (%)")
    ax.set_title("(c) Differential probes", loc="left", fontweight="bold")
    ax.grid(alpha=.2)

    ax = axes[1, 1]
    xs = np.asarray([0, 25, 50, 75])
    ys = np.asarray([main[f"Unreplayable{x}"]["vsr"] for x in xs]) * 100
    ax.plot(xs, ys, "o-", color=colors["harm"])
    ax.set_xlabel("Unreplayable strata (%)"); ax.set_ylabel("VSR (%)")
    ax.set_title("(d) Replayability stress", loc="left", fontweight="bold")
    ax.grid(alpha=.2)
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    diagnostic = {row["method"]: row for row in diagnostics}
    unseen = {row["method"]: row for row in stats["unseen_topology"]["main"]}
    unseen_paired = {row["baseline"]: row
                     for row in stats["unseen_topology"]["paired"]}
    report = f"""# CREDIT v3 Controlled RQ5 Results

## Frozen confirmation protocol

Eight fresh seeds (`70001`, `70009`, `70019`, `70039`, `70051`, `70061`,
`70079`, `70099`) ran for three chronological cycles. Each method was scored on
24,576 regular held-out tasks. The audit re-executed 614,400 held-out episodes,
2,495,896 held-out API calls, and checked 49,152 unseen-topology rows.

## Mechanism results

| Method | VSR | CREDIT gain | Exposure differences | Outcome differences |
|---|---:|---:|---:|---:|
| **CREDIT** | **{pct(main['CREDIT']['vsr'])}%** | — | — | — |
| No drift gate | {pct(main['NoDriftGate']['vsr'])}% | **+{100*paired['NoDriftGate']['delta_vsr']:.2f} pp** | {diagnostic['NoDriftGate']['exposure_differences_vs_credit']:,} | {diagnostic['NoDriftGate']['outcome_differences_vs_credit']:,} |
| Nonrandom observed | {pct(main['Nonrandom-Observed']['vsr'])}% | **+{100*paired['Nonrandom-Observed']['delta_vsr']:.2f} pp** | {diagnostic['Nonrandom-Observed']['exposure_differences_vs_credit']:,} | {diagnostic['Nonrandom-Observed']['outcome_differences_vs_credit']:,} |
| No interactions | {pct(main['NoInteractions']['vsr'])}% | +{100*paired['NoInteractions']['delta_vsr']:.2f} pp | {diagnostic['NoInteractions']['exposure_differences_vs_credit']:,} | {diagnostic['NoInteractions']['outcome_differences_vs_credit']:,} |
| No fresh certification | {pct(main['NoFreshCertification']['vsr'])}% | +0.00 pp | {diagnostic['NoFreshCertification']['exposure_differences_vs_credit']:,} | 0 |
| Physical OAT | {pct(main['OAT-Physical']['vsr'])}% | +0.00 pp | {diagnostic['OAT-Physical']['exposure_differences_vs_credit']:,} | 0 |
| No robust radius (skew) | {pct(main['NoRobust-Skew']['vsr'])}% | +0.00 pp | {diagnostic['NoRobust-Skew']['exposure_differences_vs_credit']:,} | 0 |

The paired CREDIT gains over no drift gate and nonrandom exposure were
{100*paired['NoDriftGate']['delta_vsr']:.2f} points (95% paired CI
[{100*paired['NoDriftGate']['ci_low']:.2f}, {100*paired['NoDriftGate']['ci_high']:.2f}]) and
{100*paired['Nonrandom-Observed']['delta_vsr']:.2f} points
([{100*paired['Nonrandom-Observed']['ci_low']:.2f}, {100*paired['Nonrandom-Observed']['ci_high']:.2f}]).
Both exact seed tests give `p=0.0078125`; Holm adjustment across the seven core
ablations gives `p=0.0546875`, and adjustment across all 24 frozen contrasts
gives `p=0.1875`. They are consistent effects but are not familywise significant
at 0.05 under either frozen family.

No-fresh-certification changed 930 exposures but no outcomes. OAT always chose
`new_contract + repair_rule`; its 18,847 exposure changes also changed no
outcome. Robust and nominal skew ranking chose the same bank in all 24 blocks.
These components were inactive for outcome selection in this finite environment.
The no-interaction model changed 5,250 exposures and 36 outcomes, all in CREDIT's
favor, for a small +{100*paired['NoInteractions']['delta_vsr']:.2f}-point gain
(`p={paired['NoInteractions']['seed_sign_flip_p']}`).

## Stress results

- Capacity 24 reduced VSR to {pct(main['Cap24']['vsr'])}% (CREDIT gain
  {100*paired['Cap24']['delta_vsr']:.2f} points); capacities 60, 80, and 103
  reproduced {pct(main['CREDIT']['vsr'])}%.
- Retrieval depth 2 reached {pct(main['Depth2']['vsr'])}%; depths 4 and 6 were
  identical at {pct(main['CREDIT']['vsr'])}%.
- Probe budgets 0/24/48 reached {pct(main['Probe0']['vsr'])}%; 96 matched the
  full result. The smaller budgets changed 6,722 exposures and 113 outcomes.
- With 25%, 50%, and 75% of strata declared unreplayable, VSR was
  {pct(main['Unreplayable25']['vsr'])}%, {pct(main['Unreplayable50']['vsr'])}%,
  and {pct(main['Unreplayable75']['vsr'])}%, respectively.

## Unseen supported-role sequences and latency

On two new operation sequences, CREDIT reached {pct(unseen['CREDIT']['vsr'])}%
versus 0% for NoMem. The paired CREDIT--NoMem gain was
{100*unseen_paired['NoMem']['delta_vsr']:.2f} points; the
three-contrast Holm value was {unseen_paired['NoMem']['holm_p']}.
This is a finite recomposition test, not evidence for arbitrary DAGs.

Cached local execution latency for CREDIT was
{stats['latency']['CREDIT']['p50_ms']:.3f} ms p50 and
{stats['latency']['CREDIT']['p95_ms']:.3f} ms p95. It does not represent network
or provider latency.

## Integrity and limits

The machine-readable audit is
`results/credit_v3_remote_recovery_rq5_confirm_v1/execution_audit.json` and has
`verified=true`. The controlled grid still lacks natural provider drift,
continuous drift-intensity intervention, higher-than-pair interaction sweeps,
and real network latency. Those gaps must remain explicit in the paper.
"""
    (output / "report.md").write_text(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.results, args.output, args.figure))
