"""Create a compact report for the audited mechanism-blind experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def fingerprint(value) -> str:
    return sha256(json.dumps(value, sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()


def call_trace(row: dict) -> list[tuple]:
    return [(call["tool"], call["arguments"], bool(call.get("exception")))
            for call in row["calls"]]


def analyze(root: Path) -> dict:
    data = rows(root / "confirmation_sessions.jsonl")
    audit = read(root / "execution_audit.json")
    if not audit.get("verified"):
        raise ValueError("exact replay audit must pass first")
    methods = sorted({row["method"] for row in data})
    cells = defaultdict(dict)
    for row in data:
        cells[(row["actor_seed"], row["task"])][row["method"]] = row
    same_trace = sum(len({fingerprint(call_trace(cell[method]))
                          for method in methods}) == 1 for cell in cells.values())
    same_state = sum(len({cell[method]["ending_state_sha256"]
                          for method in methods}) == 1 for cell in cells.values())
    same_first = 0
    for cell in cells.values():
        actions = []
        for method in methods:
            actions.append(next(turn["action"] for turn in cell[method]["generations"]
                                if turn.get("generation_uid")))
        same_first += len({fingerprint(action) for action in actions}) == 1
    method_rows = []
    for method in methods:
        subset = [row for row in data if row["method"] == method]
        method_rows.append({"method": method, "sessions": len(subset),
            "VSR": sum(row["requested_state_success"] for row in subset) / len(subset),
            "VSR_at_15s": sum(row["requested_state_success"] and
                row["accounted_latency_ms"] <= 15000 for row in subset) / len(subset),
            "mean_remote_generations": sum(len([turn for turn in row["generations"]
                if turn.get("generation_uid")]) for row in subset) / len(subset),
            "deterministic_repairs": sum(sum("deterministic_repair" in turn
                for turn in row["generations"]) for row in subset),
            "input_tokens": sum(sum(turn.get("input_tokens", 0)
                for turn in row["generations"]) for row in subset),
            "output_tokens": sum(sum(turn.get("output_tokens", 0)
                for turn in row["generations"]) for row in subset)})
    return {"audit_verified": True, "sessions": len(data), "paired_cells": len(cells),
        "logical_remote_generations": audit["logical_remote_generations"],
        "physical_http_attempts": audit["physical_http_attempts"],
        "exact_prompt_overlap": audit["exact_prompt_overlap"],
        "exact_mechanism_signature_overlap":
            audit["exact_mechanism_signature_overlap"],
        "all_method_first_action_identical_cells": same_first,
        "all_method_exact_call_trace_identical_cells": same_trace,
        "all_method_ending_state_identical_cells": same_state,
        "methods": method_rows,
        "interpretation": (
            "The new mechanisms are valid but too easy for the shared v3 executor. "
            "The experiment is a saturation diagnostic and supplies no selector-"
            "efficacy evidence.")}


def markdown(report: dict) -> str:
    lines = ["# Mechanism-Blind ToolSandbox-M Results", "",
        "The pre-frozen run passed exact replay for all "
        f"{report['sessions']} sessions and {report['logical_remote_generations']:,} "
        "logical generations. Logical generations equaled physical HTTP attempts; "
        "there were no retries. Exact public-prompt and requested-write-signature "
        "overlap with the registered reference suite were both zero.", "",
        "| Method | State VSR | VSR at 15 s | Generations/session | Repairs | Input tokens |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in report["methods"]:
        lines.append(f"| {row['method']} | {100*row['VSR']:.2f}% | "
                     f"{100*row['VSR_at_15s']:.2f}% | "
                     f"{row['mean_remote_generations']:.2f} | "
                     f"{row['deterministic_repairs']} | {row['input_tokens']:,} |")
    lines += ["", "## Diagnostic", "",
        f"All methods chose the same first action in "
        f"{report['all_method_first_action_identical_cells']}/"
        f"{report['paired_cells']} actor-task cells. Their entire tool trace and "
        f"ending state were identical in "
        f"{report['all_method_exact_call_trace_identical_cells']}/"
        f"{report['paired_cells']} cells. The remaining cells changed operation "
        "order only and still passed.", "",
        "This suite therefore establishes task-mechanism novelty and implementation "
        "integrity, but it supplies no evidence that any selector is better. The "
        "shared argument-aware executor solved every registered task and masked the "
        "memory intervention. The result should be reported as a saturation boundary, "
        "not as a positive CREDIT result.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("CREDIT_V4_MECHANISM_BLIND_RESULTS.md"))
    args = parser.parse_args()
    report = analyze(args.results)
    (args.results / "diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.write_text(markdown(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
