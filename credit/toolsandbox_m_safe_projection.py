"""Exploratory safe-abstention projection for an audited ToolSandbox-M run.

This does not create new model outcomes.  It evaluates the deployment action
that CREDIT specifies when causal evidence is unresolved: retain and reuse the
frozen Semantic incumbent.  A two-stage futility rule is evaluated only on the
calibration split and all training latency selected by that rule is charged to
the projected method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .toolsandbox_m_followup import LATENCY_BUDGETS_MS, continuous_score
from .user_data import digest


EPSILON = 0.005


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def latency(row: dict) -> float:
    return float(sum(turn["remote_latency_ms"] for turn in row["generations"]))


def sequential_plan(calibration: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = {}
    for row in calibration:
        by_task.setdefault(row["source_task"], []).append(row)
    decisions, selected = {}, []
    for task, values in sorted(by_task.items()):
        values.sort(key=lambda row: row["actor_seed"])
        first_two = [row["paired_effect"] for row in values[:2]]
        stop = len(values) >= 2 and all(abs(effect) <= EPSILON for effect in first_two)
        used = values[:2] if stop else values[:3]
        effects = [row["paired_effect"] for row in used]
        if min(effects) > EPSILON:
            status = "beneficial"
        elif max(effects) < -EPSILON:
            status = "harmful"
        else:
            status = "unresolved"
        decisions[task] = {"looks": len(used), "effects": effects,
                           "stopped_for_futility": stop, "status": status}
        selected.extend((row["actor_seed"], task) for row in used)
    return {"epsilon": EPSILON, "minimum_looks": 2,
            "futility_rule": "stop after two looks iff both |paired effect| <= epsilon",
            "decisions": decisions, "selected_pairs": selected}


def project(source: Path, output: Path) -> dict:
    audit = read(source / "execution_audit.json")
    if not audit.get("verified"):
        raise ValueError("source execution audit has not passed")
    calibration = read(source / "causal_calibration.json")
    plan = sequential_plan(calibration)
    episode_by_seed_task_method = {}
    for seed in read(source / "frozen_protocol.json")["training_seeds"]:
        for row in rows(source / f"calibration_seed_{seed}" / "episodes.jsonl"):
            episode_by_seed_task_method[(seed, row["task"], row["method"])] = row
    calibration_latency = 0.0
    for seed, task in plan["selected_pairs"]:
        for method in ("control", "single_memory_treatment"):
            calibration_latency += latency(episode_by_seed_task_method[(seed, task, method)])

    confirmation = rows(source / "confirmation_sessions.jsonl")
    incumbent = [row for row in confirmation if row["method"] == "Semantic"]
    if any(item["status"] != "unresolved" for item in plan["decisions"].values()):
        raise ValueError("projection applies only when every calibrated effect is unresolved")
    amortized = calibration_latency / len(incumbent)
    projected = {
        "method": "CREDIT-Safe (abstain to Semantic)", "sessions": len(incumbent),
        "strict_VSR": float(np.mean([row["success"] for row in incumbent])),
        "upstream_mean_score": float(np.mean([continuous_score(row) for row in incumbent])),
        "mean_online_remote_latency_ms": float(np.mean([row["remote_latency_ms"] for row in incumbent])),
        "calibration_latency_ms": calibration_latency,
        "amortized_calibration_latency_ms": amortized,
        "mean_attributed_remote_latency_ms": float(np.mean(
            [row["remote_latency_ms"] + amortized for row in incumbent])),
        "p95_attributed_remote_latency_ms": float(np.quantile(
            [row["remote_latency_ms"] + amortized for row in incumbent], .95)),
    }
    for budget in LATENCY_BUDGETS_MS:
        projected[f"VSR_at_{budget}ms"] = float(np.mean([
            row["success"] and row["remote_latency_ms"] + amortized <= budget
            for row in incumbent]))
        projected[f"completion_at_{budget}ms"] = float(np.mean([
            row["remote_latency_ms"] + amortized <= budget for row in incumbent]))
    original = next(row for row in read(source / "summary.json")["confirmation_table"]
                    if row["method"] == "CREDIT")
    report = {
        "status": "exploratory_projection_not_new_online_confirmation",
        "source_results": str(source),
        "source_confirmation_sha256": digest(source / "confirmation_sessions.jsonl"),
        "source_execution_audit_sha256": digest(source / "execution_audit.json"),
        "policy_action": "abstain; retain and reuse the Semantic incumbent",
        "reason": "all calibrated memory effects remain unresolved",
        "sequential_calibration": plan,
        "original_credit": original,
        "projected_credit_safe": projected,
        "calibration_latency_reduction": 1 - calibration_latency /
            read(source / "training_summary.json")["calibration_remote_latency_ms"],
        "limitations": [
            "The abstention rule and futility rule were formulated after inspecting the first follow-up calibration.",
            "Semantic outcomes are reused because abstention deploys the incumbent; no new CREDIT actor calls are claimed.",
            "A new frozen run is required for confirmatory evidence.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "safe_projection.json").write_text(json.dumps(report, indent=2))
    (output / "audit.json").write_text(json.dumps({
        "verified": True, "source_audit_verified": audit["verified"],
        "all_effects_unresolved": True, "incumbent_sessions": len(incumbent),
        "selected_calibration_pairs": len(plan["selected_pairs"]),
        "source_confirmation_sha256": report["source_confirmation_sha256"],
        "projection_source_sha256": digest(Path(__file__)),
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/toolsandbox_m_followup_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/toolsandbox_m_safe_projection_v1"))
    args = parser.parse_args()
    print(json.dumps(project(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
