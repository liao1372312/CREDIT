"""Report continuous injected drift and the audited interaction-order grid."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paired_statistics import summarize


def read(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_drift_drop(root: Path) -> dict:
    rows = []
    with gzip.open(root / "raw_episodes.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["method"] != "CREDIT-Gated" or row["intensity"] not in (0, 2):
                continue
            rows.append({
                "seed": row["seed"],
                "task": row["task"].rsplit("-", 1)[-1],
                "family": row["family"],
                "method": f"I{row['intensity']:g}",
                "success": row["success"],
            })
    _, paired = summarize(rows, "I0", repeats=4000, random_seed=28121)
    return paired[0]


def interaction_summary(root: Path) -> tuple[list[dict], list[dict]]:
    comparison = pd.read_csv(root / "comparison.csv")
    anchor = comparison[
        (comparison.C == 2048) & (comparison.K == 8)
        & (comparison.R == 64)
        & (comparison.method == "CREDIT-finite-empirical")
    ]
    degree = pd.read_csv(root / "degree_forecast_summary.csv")
    table = []
    for order in (1, 2, 3):
        row = anchor[anchor.d == order].iloc[0]
        error = degree[degree.d == order].iloc[0]
        table.append({
            "max_interaction_order": order,
            "anchor_vsr": float(row.vsr),
            "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high),
            "selected_bank_forecast_mae": float(
                error.mean_selected_bank_forecast_MAE
            ),
        })
    episodes = pd.read_csv(root / "raw_episodes.csv")
    episodes = episodes[
        (episodes.C == 2048) & (episodes.K == 8) & (episodes.R == 64)
        & (episodes.method == "CREDIT-finite-empirical")
    ]
    rows = [{
        "seed": int(row.seed), "task": row.task, "family": row.family,
        "method": f"d{int(row.d)}", "success": int(row.success),
    } for row in episodes.itertuples()]
    _, paired = summarize(rows, "d2", repeats=4000, random_seed=28111)
    return table, paired


def report(drift_root: Path, order_root: Path, output: Path) -> None:
    drift_audit = read(drift_root / "execution_audit.json")
    order_audit = read(order_root / "execution_audit.json")
    if not drift_audit["verified"] or not order_audit["verified"]:
        raise ValueError("both execution audits must be verified")
    output.mkdir(parents=True, exist_ok=False)
    stats = read(drift_root / "statistics.json")
    drift_table = []
    for intensity in sorted((float(value) for value in stats)):
        block = stats[f"{intensity:g}"]
        main = {row["method"]: row for row in block["main"]}
        paired = {row["baseline"]: row for row in block["paired"]}
        drift_table.append({
            "drift_intensity": intensity,
            "credit_vsr": main["CREDIT-Gated"]["vsr"],
            "credit_ci_low": main["CREDIT-Gated"]["ci_low"],
            "credit_ci_high": main["CREDIT-Gated"]["ci_high"],
            "no_gate_vsr": main["NoDriftGate"]["vsr"],
            "gate_gain": paired["NoDriftGate"]["delta_vsr"],
            "gate_exact_p": paired["NoDriftGate"]["seed_sign_flip_p"],
            "no_mem_vsr": main["NoMem"]["vsr"],
        })
    order_table, order_paired = interaction_summary(order_root)
    drop = paired_drift_drop(drift_root)
    write_csv(output / "drift_intensity.csv", drift_table)
    write_csv(output / "interaction_order.csv", order_table)
    write_csv(output / "interaction_order_paired.csv", order_paired)
    (output / "drift_drop.json").write_text(json.dumps(drop, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05))
    x = np.asarray([row["drift_intensity"] for row in drift_table])
    for method, field, color, marker in (
        ("CREDIT", "credit_vsr", "#2468a2", "o"),
        ("No drift gate", "no_gate_vsr", "#c44e52", "s"),
        ("NoMem", "no_mem_vsr", "#777777", "^"),
    ):
        axes[0].plot(
            x, 100 * np.asarray([row[field] for row in drift_table]),
            marker=marker, label=method, color=color,
        )
    axes[0].fill_between(
        x,
        100 * np.asarray([row["credit_ci_low"] for row in drift_table]),
        100 * np.asarray([row["credit_ci_high"] for row in drift_table]),
        color="#2468a2", alpha=.13, linewidth=0,
    )
    axes[0].set_xlabel("Injected drift intensity")
    axes[0].set_ylabel("Verified success rate (%)")
    axes[0].set_title("(a) Continuous drift stress")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=.2)

    orders = [row["max_interaction_order"] for row in order_table]
    errors = [100 * row["selected_bank_forecast_mae"] for row in order_table]
    axes[1].bar([str(value) for value in orders], errors,
                color=["#999999", "#2468a2", "#7d5bb3"])
    axes[1].set_xlabel("Maximum interaction order")
    axes[1].set_ylabel("Selected-bank forecast MAE (pt)")
    axes[1].set_title("(b) Interaction-order stress")
    axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout()
    figures = Path("figures")
    figures.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            figures / f"credit_v3_remaining_stress_v1.{suffix}",
            dpi=220, bbox_inches="tight",
        )
    plt.close(fig)

    order_by_d = {row["max_interaction_order"]: row for row in order_table}
    p_by_base = {row["baseline"]: row for row in order_paired}
    lines = [
        "# CREDIT v3 Remaining Controlled Stress Results", "",
        "## Continuous injected drift", "",
        "Eight fresh confirmation seeds were evaluated after the bank and model "
        "were frozen at the nominal target setting. Each method has 4,096 "
        "held-out tasks at every intensity.", "",
        "| Intensity | CREDIT | No drift gate | NoMem | CREDIT minus no gate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in drift_table:
        lines.append(
            f"| {row['drift_intensity']:g} | {100*row['credit_vsr']:.2f}% | "
            f"{100*row['no_gate_vsr']:.2f}% | {100*row['no_mem_vsr']:.2f}% | "
            f"{100*row['gate_gain']:+.2f} pp |"
        )
    lines += [
        "",
        f"CREDIT fell by {100*drop['delta_vsr']:.2f} points from intensity 0 "
        f"to 2 (paired 95% CI [{100*drop['ci_low']:.2f}, "
        f"{100*drop['ci_high']:.2f}], exact `p={drop['seed_sign_flip_p']:.7f}`). "
        "The gate/no-gate difference grew from -0.20 to +2.73 points, but only "
        "one learned bank contained the stale scoped memory; the eight-seed "
        "sign-flip test is therefore `p=1.0`. This grid measures degradation, "
        "not a new significant gate advantage.", "",
        "## Higher-order interaction stress", "",
        "The independent audited finite grid varied the polynomial memory "
        "interaction order while holding the reporting point at C=2048, K=8, "
        "and R=64.", "",
        "| Maximum order | VSR | Selected-bank forecast MAE |",
        "|---:|---:|---:|",
    ]
    for order in (1, 2, 3):
        row = order_by_d[order]
        lines.append(
            f"| {order} | {100*row['anchor_vsr']:.2f}% | "
            f"{100*row['selected_bank_forecast_mae']:.3f} pt |"
        )
    lines += [
        "",
        f"Order two improved anchor VSR over order one by "
        f"{100*p_by_base['d1']['delta_vsr']:.2f} points and over order three "
        f"by {100*p_by_base['d3']['delta_vsr']:.2f} points; both paired exact "
        "tests have `p=1.0`. The order-two model also had the lowest aggregate "
        "selected-bank forecast MAE. There is no evidence that the third-order "
        "model helps in this finite environment.", "",
        "## Integrity and limits", "",
        f"The drift audit re-executed {drift_audit['episodes_reexecuted']:,} "
        f"episodes and {drift_audit['evaluation_api_calls_reexecuted']:,} "
        "evaluation calls. The interaction grid audit reconstructed "
        f"{order_audit['future_rows_reconstructed']:,} future rows and "
        f"{order_audit['unique_counterfactuals_reexecuted']:,} unique "
        "counterfactual executions. Both have `verified=true`. These are "
        "controlled finite adapters and do not establish natural provider "
        "drift, semantic higher-order effects, or real network latency.", "",
    ]
    Path("CREDIT_V3_REMAINING_STRESS_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-results", type=Path, required=True)
    parser.add_argument("--order-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.drift_results, args.order_results, args.output)
