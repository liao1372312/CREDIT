"""Verify the v4 table after separating online and offline call costs."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    broad = ROOT / "results/credit_v4_broad_benchmark_v2"
    quality = json.loads((broad / "table.json").read_text())
    online_path = ROOT / "results/credit_v4_online_cost_v1.json"
    online_payload = json.loads(online_path.read_text())
    online = online_payload["table"]
    paper_path = ROOT / "CREDIT_WWW2027_v4.tex"
    paper = paper_path.read_text()
    main_table = paper.split("\\label{tab:exp:main}", 1)[1].split("\\end{table*}", 1)[0]

    if len(quality) != 8 or len(online) != 8:
        raise ValueError("main-table method coverage differs")
    if [row["method"] for row in quality] != [row["method"] for row in online]:
        raise ValueError("quality and online-cost method order differs")
    for qrow, crow in zip(quality, online):
        values = [100 * qrow["all_vsr"], 100 * qrow["new_workflow_vsr"],
                  100 * qrow["drift_vsr"], 100 * qrow["stale_exposure_rate"],
                  crow["online_calls_per_test"]]
        if not all(f"{value:.2f}" in main_table for value in values):
            raise ValueError("paper main-table value differs: " + qrow["method"])

    for relative, expected in online_payload["inputs_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("online-cost input changed: " + relative)
    if "Online calls" not in main_table or "74.41" in main_table:
        raise ValueError("main table has not separated online and offline costs")
    if "Offline calls" not in paper or "1,828,683" not in paper:
        raise ValueError("offline cost analysis is missing")
    if digest(ROOT / "CREDIT_WWW2027_v3.tex") != "31e5fb029a546ea0a21dd74b343804c2ea1ade6a17c501a1875454ad0a88d153":
        raise ValueError("frozen v3 changed")
    if digest(ROOT / "AGENTS.md") != "33168b848f700c1451620891dbbec34c3a2eef3c1bbc31391a4da0aec4e21cdf":
        raise ValueError("AGENTS changed")

    report = {
        "verified": True,
        "main_table_methods": 8,
        "main_table_cells": 40,
        "main_table_call_metric": online_payload["metric"],
        "offline_calls_location": "separate controlled-recovery cost table",
        "online_cost_artifact_sha256": digest(online_path),
        "paper_v4_sha256": digest(paper_path),
        "frozen_v3_sha256": digest(ROOT / "CREDIT_WWW2027_v3.tex"),
        "verifier_sha256": digest(Path(__file__)),
    }
    output = ROOT / "results/credit_v4_experiment_audit_v2.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
