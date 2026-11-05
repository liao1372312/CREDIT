"""Verify the v4 paper's selector attribution and all-in cost disclosures."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    result_root = ROOT / "results/toolsandbox_m_selector_attribution_confirmation_v1"
    summary_path = result_root / "summary.json"
    execution_audit_path = result_root / "execution_audit.json"
    cost_path = ROOT / "results/credit_v4_controlled_total_cost_v2/report.json"
    paper_path = ROOT / "CREDIT_WWW2027_v4.tex"
    extended_path = ROOT / "CREDIT_WWW2027_v4_extended.tex"

    summary = json.loads(summary_path.read_text())
    execution_audit = json.loads(execution_audit_path.read_text())
    cost = json.loads(cost_path.read_text())
    paper = paper_path.read_text()
    extended = extended_path.read_text()

    require(execution_audit.get("verified") is True, "selector replay audit failed")
    require(execution_audit.get("confirmation_sessions") == 768,
            "selector confirmation is incomplete")
    require(execution_audit.get("replayed_episodes") == 768,
            "selector replay is incomplete")
    require(cost.get("source_audit_verified") is True,
            "controlled cost source audit failed")

    rows = {row["method"]: row for row in summary["confirmation_table"]}
    expected = {
        "NoMem/Base": 28.12,
        "CREDIT/Base": 35.42,
        "NoMem/v3": 76.04,
        "FIFO/v3": 81.25,
        "Semantic/v3": 94.79,
        "CREDIT/v3": 90.62,
        "ContractOnly/v3": 100.00,
        "Full/v3": 97.92,
    }
    require(set(rows) == set(expected), "selector method set differs")
    table = paper.split(r"\label{tab:tsmargsafe}", 1)[1].split(
        r"\end{table}", 1)[0]
    for method, value in expected.items():
        measured = 100 * rows[method]["requested_state_VSR"]
        require(round(measured, 2) == value, f"unexpected state VSR for {method}")
        require(f"{value:.2f}" in table, f"paper omits state VSR for {method}")

    comparisons = summary["paired_statistics"]["comparisons"]
    require(len(comparisons) == 5, "registered contrast family differs")
    keyed = {(row["treatment"], row["baseline"]): row for row in comparisons}
    require(round(100 * keyed[("CREDIT/v3", "NoMem/v3")]
                  ["delta_requested_state_VSR"], 2) == 14.58,
            "historical-memory effect differs")
    require(keyed[("CREDIT/v3", "NoMem/v3")]["holm_actor_p"] == 0.03125,
            "historical-memory adjusted p-value differs")
    require(round(100 * keyed[("NoMem/v3", "NoMem/Base")]
                  ["delta_requested_state_VSR"], 2) == 47.92,
            "executor effect differs")
    require(keyed[("NoMem/v3", "NoMem/Base")]["holm_actor_p"] == 0.01953125,
            "executor adjusted p-value differs")
    for token in ("14.58", "47.92", "0.0313", "0.0195", "4,511"):
        require(token in paper and token in extended,
                f"paper disclosure is missing {token}")

    credit = next(row for row in cost["methods"]
                  if row["method"] == "CREDIT-Covered")
    no_mem_break_even = next(row["tasks_per_adapted_seed_cycle"]
        for row in cost["credit_break_even"] if row["baseline"] == "NoMem")
    require(credit["study_all_in_calls"] == 1_929_129,
            "all-in CREDIT cost differs")
    require(no_mem_break_even == 47_613, "NoMem break-even differs")
    for token in ("1,929,129", "11.71", "4.85", "47,613"):
        require(token in paper, f"main paper cost disclosure is missing {token}")

    protected = {
        "AGENTS.md": "33168b848f700c1451620891dbbec34c3a2eef3c1bbc31391a4da0aec4e21cdf",
        "CREDIT_WWW2027_v3.tex": "31e5fb029a546ea0a21dd74b343804c2ea1ade6a17c501a1875454ad0a88d153",
    }
    for relative, expected_hash in protected.items():
        require(digest(ROOT / relative) == expected_hash,
                f"protected file changed: {relative}")

    report = {
        "verified": True,
        "selector_sessions": 768,
        "selector_remote_generations": 4_511,
        "selector_execution_audit_sha256": digest(execution_audit_path),
        "selector_summary_sha256": digest(summary_path),
        "controlled_cost_report_sha256": digest(cost_path),
        "paper_sha256": digest(paper_path),
        "extended_paper_sha256": digest(extended_path),
        "protected_files_sha256": protected,
        "verifier_sha256": digest(Path(__file__)),
    }
    output = ROOT / "results/credit_v4_experiment_audit_v4.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
