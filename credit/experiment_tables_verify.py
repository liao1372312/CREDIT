"""Independent consistency checks for the populated experiment tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f"{label}: {left} != {right}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/credit_experiment_tables_v1"))
    parser.add_argument("--tex", type=Path, default=Path("CREDIT_Experiment_Tables.tex"))
    parser.add_argument("--pdf", type=Path,
                        default=Path("build/experiment_tables_v1/CREDIT_Experiment_Tables.pdf"))
    args = parser.parse_args()
    public, offline, replay = args.root / "public", args.root / "offline", args.root / "replay"
    mechanism = args.root / "mechanism_support"

    offline_audit = json.loads((offline / "audit.json").read_text())
    for name, expected in offline_audit["files"].items():
        if digest(offline / name) != expected:
            raise AssertionError("offline hash mismatch: " + name)
    if not all(offline_audit["checks"].values()):
        raise AssertionError("offline audit contains a failed check")

    public_audit = json.loads((public / "public_audit.json").read_text())
    for name, expected in public_audit["files"].items():
        if digest(public / name) != expected:
            raise AssertionError("public hash mismatch: " + name)
    checks = public_audit["checks"]
    scalar_checks = (checks["split_sizes_16_4_20"],
                     checks["all_native_evaluations_present"],
                     checks["no_test_tasks_in_adaptation"],
                     checks["no_test_outcomes_in_selection"],
                     checks["main_logical_sessions"] == 420,
                     set(checks["three_test_replications"].values()) == {3})
    if not all(scalar_checks):
        raise AssertionError("public audit contains a failed check")

    logical = json.loads((public / "logical_results.json").read_text())
    main = json.loads((public / "main_summary.json").read_text())
    main_logical = [row for row in logical if row.get("group") == "main"]
    if len(main_logical) != 420:
        raise AssertionError("main logical population is not 420")
    for summary in main:
        rows = [row for row in main_logical if row["method"] == summary["method"]]
        if len(rows) != 60:
            raise AssertionError("method does not have 60 held-out sessions")
        close(float(np.mean([row["native_score"] for row in rows])),
              summary["native_score"], summary["method"] + " native score")
        close(float(np.mean([row["success"] for row in rows])),
              summary["vsr"], summary["method"] + " VSR")

    replay_audit = json.loads((replay / "audit.json").read_text())
    if replay_audit["tasks"] != 20 or not replay_audit["all_evaluations_present"]:
        raise AssertionError("replay audit failed")
    replay_summary = json.loads((replay / "replay_summary.json").read_text())
    aa = replay_summary["comparisons"][0]
    if any(aa[key] for key in ("state_diff", "trace_diff", "outcome_diff", "decision_diff")):
        raise AssertionError("A/A replay is not deterministic")

    mechanism_audit = json.loads((mechanism / "audit.json").read_text())
    if not mechanism_audit["passed"] or not all(mechanism_audit["checks"].values()):
        raise AssertionError("mechanism-support audit failed")
    for name, expected in mechanism_audit["files"].items():
        if digest(mechanism / name) != expected:
            raise AssertionError("mechanism-support hash mismatch: " + name)
    mechanism_summary = json.loads((mechanism / "summary.json").read_text())
    if (mechanism_summary["independent_seeds"] != 32 or
            mechanism_summary["role_transport"]["total_compositions"] != 16_384 or
            mechanism_summary["drift_gate"]["divergent_opportunities"] != 32 or
            mechanism_summary["fresh_certification"]["full_accepted"] != 640):
        raise AssertionError("mechanism-support counts changed")

    tex = args.tex.read_text()
    forbidden = ("\\CRempty", "await measurement", "unmeasured")
    if any(value in tex for value in forbidden):
        raise AssertionError("populated TeX retains a placeholder")
    for required in ("0.8634", "151{,}254", "16/20", "$B_0=768$", "16{,}384", "1280/1920"):
        if required not in tex:
            raise AssertionError("populated TeX is missing " + required)
    if not args.pdf.exists() or args.pdf.stat().st_size < 10_000:
        raise AssertionError("compiled PDF is missing or implausibly small")

    report = {
        "passed": True,
        "counts": {"attribution_families": 180, "certification_streams_per_condition": 2000,
                   "public_main_logical_sessions": 420,
                   "public_physical_test_sessions": public_audit["counts"]["test_physical_sessions"],
                   "replay_tasks": replay_audit["tasks"],
                   "mechanism_support_seeds": mechanism_summary["independent_seeds"],
                   "role_transport_compositions": mechanism_summary["role_transport"]["total_compositions"],
                   "drift_opportunities": mechanism_summary["drift_gate"]["divergent_opportunities"],
                   "certification_proposals": mechanism_summary["fresh_certification"]["no_certification_accepted"]},
        "artifacts": {"tex_sha256": digest(args.tex), "pdf_sha256": digest(args.pdf),
                      "offline_audit_sha256": digest(offline / "audit.json"),
                      "public_audit_sha256": digest(public / "public_audit.json"),
                      "replay_audit_sha256": digest(replay / "audit.json"),
                      "mechanism_support_audit_sha256": digest(mechanism / "audit.json")},
    }
    (args.root / "overall_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
