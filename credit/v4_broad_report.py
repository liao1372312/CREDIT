"""Report the audited v4 chronological benchmark without rerunning selection."""

from __future__ import annotations

import json
from pathlib import Path

from .paired_statistics import summarize
from .v3.experiment import digest, write


BASE = Path("results/credit_v3_remote_recovery_baselines_confirm_v1")
RESULT = Path("results/credit_v4_broad_benchmark_v2")
REPORT = Path("CREDIT_V4_BROAD_BENCHMARK_RESULTS.md")


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def run(base: Path = BASE, result: Path = RESULT, report: Path = REPORT) -> None:
    audit = json.loads((result / "execution_audit.json").read_text())
    if not audit.get("verified"):
        raise ValueError("broad benchmark audit has not passed")
    baseline = json.loads((base / "raw_test_results.json").read_text())
    direct = json.loads((result / "direct_test_results.json").read_text())
    # FIFO and Recency are exactly identical on held-out retrieval/execution;
    # keep FIFO as their single predeclared table row.
    inference_rows = [row for row in baseline if row["method"] != "Recency"] + direct
    main, paired = summarize(inference_rows, "CREDIT-Covered", repeats=10_000,
                             random_seed=20271014)
    write(result / "paired_statistics.json", {"main": main, "paired": paired,
        "comparison_family": "seven rows shown against CREDIT; FIFO represents the identical FIFO/Recency pair",
        "source_sha256": digest(__file__)})
    table = json.loads((result / "table.json").read_text())
    direct_costs = json.loads((result / "direct_costs.json").read_text())
    maximum_relative_gap = max(
        row["unspent"] / row["matched_credit_calls"] for row in direct_costs)
    credit = next(row for row in table if row["method"] == "CREDIT")
    direct_row = next(row for row in table if row["method"] == "Direct-Sandbox")
    comparison = next(row for row in paired if row["baseline"] == "Direct-Sandbox")
    lines = [
        "# CREDIT v4 Chronological Benchmark Results", "",
        "## Protocol", "",
        "The benchmark uses the same eight frozen confirmation seeds, three chronological cycles,",
        "24,576 held-out episodes per method, cached remote actor, finite versioned API, capacity 80,",
        "and retrieval depth four. Source-version episodes measure unseen supported templates and",
        "topologies; target-version episodes measure changed contracts. Test outcomes are scored only",
        "after every bank is frozen.", "",
        "Direct-Sandbox was rerun on the identical tasks. In every seed-cycle block its total",
        "pre-test allowance equals CREDIT's observed evidence-and-selection calls. It uses the shared",
        "pre-test drift evidence, but no learned outcome model. Residual budget evaluates frozen",
        "finalists on complete eight-stratum CRN groups. The maximum block-level unmatched remainder",
        f"was {audit['maximum_total_call_gap']} calls ({100 * maximum_relative_gap:.3f}% of its block budget).", "",
        "## Results", "",
        "| Method | All VSR | New-WF VSR | Drift VSR | Stale exposure | Extra calls/test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append("| {method} | {all_vsr} | {new} | {drift} | {stale} | {calls:.2f} |".format(
            method=row["method"], all_vsr=percent(row["all_vsr"]),
            new=percent(row["new_workflow_vsr"]), drift=percent(row["drift_vsr"]),
            stale=percent(row["stale_exposure_rate"]),
            calls=row["extra_calls_per_test_episode"]))
    lines += ["",
        f"CREDIT achieved {percent(credit['all_vsr'])} overall and {percent(credit['drift_vsr'])}",
        f"on changed target contracts. Matched-total-call Direct-Sandbox achieved {percent(direct_row['all_vsr'])}",
        f"and {percent(direct_row['drift_vsr'])}, respectively. The paired overall gain was",
        f"{100 * comparison['delta_vsr']:.2f} points (cluster bootstrap 95% CI",
        f"[{100 * comparison['ci_low']:.2f}, {100 * comparison['ci_high']:.2f}];",
        f"exact seed sign-flip p={comparison['seed_sign_flip_p']:.7f};",
        f"seven-comparison Holm p={comparison['holm_p']:.4f}).", "",
        "Direct's physical search retained the empty bank in all 24 blocks at this matched total",
        "budget, explaining its equality with NoMem. The separate equal-selection-call curve gives",
        "both CREDIT and Direct an additional 50k--125k selection allowance after shared evidence and",
        "therefore answers a different cost question.", "",
        "## Integrity and Limits", "",
        f"The independent audit replayed {audit['direct_heldout_episodes_replayed']:,} Direct episodes",
        f"and {audit['direct_heldout_api_calls_replayed']:,} API calls, found no held-out identifiers",
        "in selection streams, and reconstructed all 40 table cells. FIFO and Recency had identical",
        "retrieved exposures and outcomes and are shown as one row.", "",
        "This is a controlled finite-API benchmark with a cached remote actor. Observed open-source",
        "release changes, public ToolSandbox sessions, semantic text interactions, and real network",
        "latency remain separate suites and are not pooled into these VSR values.", "",
        "## Reproduction", "", "```sh",
        "PYTHONHASHSEED=0 python -m credit.v4_broad_benchmark --output results/credit_v4_broad_benchmark_v2",
        "PYTHONHASHSEED=0 python -m credit.v4_broad_benchmark --audit --output results/credit_v4_broad_benchmark_v2",
        "python -m credit.v4_broad_report", "```", "",
    ]
    report.write_text("\n".join(lines))
    print(json.dumps({"report": str(report), "methods": len(table),
                      "paired_credit_direct": comparison}, indent=2))


if __name__ == "__main__":
    run()
