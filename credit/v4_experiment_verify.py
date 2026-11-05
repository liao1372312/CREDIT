"""Cross-check the completed v4 controlled table and realism experiments."""

from __future__ import annotations

import json
from pathlib import Path

from .v3.experiment import digest, write


def main() -> None:
    root = Path.cwd()
    broad = root / "results/credit_v4_broad_benchmark_v2"
    realism = json.loads((root / "results/realism_bundle_audit_v1.json").read_text())
    audit = json.loads((broad / "execution_audit.json").read_text())
    table = json.loads((broad / "table.json").read_text())
    paired = json.loads((broad / "paired_statistics.json").read_text())
    paper = (root / "CREDIT_WWW2027_v4.tex").read_text()
    if not realism["verified"] or not audit["verified"]:
        raise ValueError("component audit has not passed")
    if len(table) != 8 or audit["table_cells"] != 40:
        raise ValueError("controlled table coverage differs")
    if "\\crpending" in paper or "not yet measured" in paper:
        raise ValueError("paper still contains unmeasured table cells")
    for row in table:
        values = [100 * row["all_vsr"], 100 * row["new_workflow_vsr"],
                  100 * row["drift_vsr"], 100 * row["stale_exposure_rate"],
                  row["extra_calls_per_test_episode"]]
        if not all(f"{value:.2f}" in paper for value in values):
            raise ValueError("paper/table value differs: " + row["method"])
    direct = next(row for row in paired["paired"] if row["baseline"] == "Direct-Sandbox")
    if direct["delta_vsr"] <= 0 or direct["holm_p"] < .05:
        raise ValueError("paired interpretation differs")
    if digest(root / "CREDIT_WWW2027_v3.tex") != "31e5fb029a546ea0a21dd74b343804c2ea1ade6a17c501a1875454ad0a88d153":
        raise ValueError("frozen v3 changed")
    report = {
        "verified": True,
        "controlled_chronological": {"methods": len(table), "table_cells": 40,
            "episodes_per_method": table[0]["episodes"],
            "direct_replayed_episodes": audit["direct_heldout_episodes_replayed"],
            "maximum_total_call_gap": audit["maximum_total_call_gap"]},
        "realism_bundle": realism,
        "paper_pending_cells": 0,
        "frozen_v3_sha256": digest(root / "CREDIT_WWW2027_v3.tex"),
        "paper_v4_sha256": digest(root / "CREDIT_WWW2027_v4.tex"),
        "verifier_sha256": digest(__file__),
    }
    output = root / "results/credit_v4_experiment_audit_v1.json"
    write(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
