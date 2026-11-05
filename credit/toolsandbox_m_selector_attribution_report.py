"""Render the frozen ToolSandbox-M selector attribution results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .user_data import digest


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render(root: Path) -> str:
    summary = json.loads((root / "summary.json").read_text())
    audit = json.loads((root / "execution_audit.json").read_text())
    protocol = json.loads((root / "frozen_protocol.json").read_text())
    rows = summary["confirmation_table"]
    comparisons = summary["paired_statistics"]["comparisons"]
    lines = ["# CREDIT v4 Selector Attribution Results", "", "## Frozen Design", "",
        f"The confirmation completed {audit['confirmation_sessions']} sessions across "
        f"{summary['paired_statistics']['actors']} new actor seeds, "
        f"{summary['paired_statistics']['families']} task families, two prompts per "
        "actor-family cell, and eight registered methods. Exact requested-world-state "
        "VSR was the sole primary endpoint. The executor, tool catalogue, model, task "
        "order, and call limits were fixed within each selector comparison.", "",
        "## Results", "",
        "| Method | State VSR | 15 s VSR | Records | Requests | p95 latency | Repairs |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {method} | {state} | {budget} | {records:.2f} | "
            "{requests:.2f} | {p95:.2f} s | {repairs} |".format(
                method=row["method"], state=pct(row["requested_state_VSR"]),
                budget=pct(row["requested_state_VSR_at_15000ms"]),
                records=row["mean_selected_records"],
                requests=row["mean_remote_requests"],
                p95=row["p95_accounted_latency_ms"] / 1000,
                repairs=row["deterministic_repairs"]))
    lines.extend(["", "## Registered Primary Contrasts", "",
        "| Treatment vs baseline | State gain | Crossed 95% CI | Actor W/L/T | Raw p | Holm p |",
        "|---|---:|---:|---:|---:|---:|"])
    for row in comparisons:
        sign = row["actor_level_sign_test"]
        low, high = row["crossed_actor_family_bootstrap_CI95"]
        lines.append("| {t} vs {b} | {gain:+.2f} pp | [{low:+.2f}, {high:+.2f}] | "
            "{w}/{l}/{ties} | {p:.6f} | {holm:.6f} |".format(
                t=row["treatment"], b=row["baseline"],
                gain=100 * row["delta_requested_state_VSR"],
                low=100 * low, high=100 * high, w=sign["wins"],
                l=sign["losses"], ties=sign["ties"],
                p=sign["one_sided_exact_p"], holm=row["holm_actor_p"]))
    memory = next(row for row in comparisons
                  if row["treatment"] == "CREDIT/v3" and row["baseline"] == "NoMem/v3")
    guard = next(row for row in comparisons
                 if row["treatment"] == "NoMem/v3" and row["baseline"] == "NoMem/Base")
    contract = next(row for row in comparisons
                    if row["treatment"] == "Full/v3" and row["baseline"] == "CREDIT/v3")
    lines.extend(["", "## Attribution", "",
        f"- Historical CREDIT-Compose memory under the fixed v3 executor changed state "
        f"VSR by {100 * memory['delta_requested_state_VSR']:+.2f} points "
        f"(Holm p={memory['holm_actor_p']:.6f}).",
        f"- The argument-aware executor with an empty bank changed state VSR by "
        f"{100 * guard['delta_requested_state_VSR']:+.2f} points relative to the empty-bank "
        f"base executor (Holm p={guard['holm_actor_p']:.6f}).",
        f"- Adding the adaptive request-contract context beyond CREDIT historical memories "
        f"changed state VSR by {100 * contract['delta_requested_state_VSR']:+.2f} points "
        f"(Holm p={contract['holm_actor_p']:.6f}).", "",
        "The ToolSandbox claim must follow these registered contrasts. A component with a "
        "nonpositive or multiplicity-adjusted nonsignificant contrast is not described as "
        "the cause of the final system gain.", "", "## Integrity", "",
        f"The verifier replayed all {audit['replayed_episodes']} episodes from "
        f"{audit['live_remote_requests']} recorded remote generations. Prompt hashes had "
        "no exact overlap with the comparison sets. The wide tool catalogue intentionally "
        "matches the prior confirmation so selector and executor attribution remain paired.", "",
        f"- Frozen protocol SHA-256: `{digest(root / 'frozen_protocol.json')}`",
        f"- Session ledger SHA-256: `{audit['confirmation_sha256']}`",
        f"- Verifier SHA-256: `{audit['verifier_sha256']}`", "",
        f"Registered methods: {', '.join(protocol['methods'])}.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path,
        default=Path("CREDIT_V4_SELECTOR_ATTRIBUTION_RESULTS.md"))
    args = parser.parse_args()
    args.output.write_text(render(args.results))
    print(json.dumps({"output": str(args.output),
                      "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
