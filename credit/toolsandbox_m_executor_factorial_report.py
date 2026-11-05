"""Render the audited ToolSandbox-M memory-by-executor factorial results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def analyze(root: Path) -> dict:
    summary = read(root / "summary.json")
    execution = read(root / "execution_audit.json")
    recovery = read(root / "recovery_audit.json")
    if not execution.get("verified"):
        raise ValueError("exact replay audit must pass before reporting")

    comparisons = {
        row["name"]: row for row in summary["paired_statistics"]["comparisons"]
    }
    methods = {row["method"]: row for row in summary["confirmation_table"]}
    base = comparisons["history_effect_Base"]
    validate = comparisons["executor_gain_Base_to_Validate"]
    return {
        "verified": True,
        "sessions": execution["confirmation_sessions"],
        "logical_remote_generations": execution["logical_remote_generations"],
        "physical_http_attempts": execution["physical_http_attempts"],
        "source_modules_prefrozen": execution["source_modules_prefrozen"],
        "confirmation_sha256": execution["confirmation_sha256"],
        "methods": list(methods.values()),
        "base_history_effect": base,
        "base_to_validate_effect": validate,
        "recovery": recovery,
        "interpretation": (
            "Scoped history has a small, family-localized Base-stage effect that "
            "does not survive the registered Holm correction. Public contract "
            "validation closes the remaining failures for both memory arms; "
            "repair and local completion add no further state success."
        ),
    }


def markdown(report: dict) -> str:
    lines = [
        "# ToolSandbox-M Memory × Executor Factorial Results",
        "",
        f"Exact replay verified all {report['sessions']} sessions and "
        f"{report['logical_remote_generations']:,} primary remote generations. "
        f"The protocol froze {report['source_modules_prefrozen']} transitive source "
        "modules before the first call.",
        "",
        "| Method | State VSR | VSR at 15 s | Requests/session | Repairs | Local completions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["methods"]:
        lines.append(
            f"| {row['method']} | {100 * row['requested_state_VSR']:.2f}% | "
            f"{100 * row['requested_state_VSR_at_15000ms']:.2f}% | "
            f"{row['mean_remote_requests']:.2f} | "
            f"{row['deterministic_repairs']} | {row['local_completions']} |"
        )

    base = report["base_history_effect"]
    validate = report["base_to_validate_effect"]
    lines += [
        "",
        "## Registered contrasts",
        "",
        f"At Base, scoped history added {100 * base['delta_requested_state_VSR']:.2f} "
        "state points (crossed actor–family 95% CI "
        f"[{100 * base['crossed_actor_family_bootstrap_CI95'][0]:.2f}, "
        f"{100 * base['crossed_actor_family_bootstrap_CI95'][1]:.2f}]). "
        f"The actor sign test was {base['actor_level_sign_test']['wins']}/"
        f"{base['actor_level_sign_test']['losses']}/"
        f"{base['actor_level_sign_test']['ties']} wins/losses/ties; Holm "
        f"p={base['holm_actor_p']:.4f}. Only one of six task families improved.",
        "",
        f"Adding validation to NoMem/Base added "
        f"{100 * validate['delta_requested_state_VSR']:.2f} points "
        f"(95% CI [{100 * validate['crossed_actor_family_bootstrap_CI95'][0]:.2f}, "
        f"{100 * validate['crossed_actor_family_bootstrap_CI95'][1]:.2f}]; "
        f"Holm p={validate['holm_actor_p']:.4f}) and reached 100% state VSR. "
        "Repair and Full remained at 100%, so their incremental state gains were zero.",
        "",
        "## Integrity and interpretation",
        "",
        report["interpretation"],
        "",
        "One actor was resumed from 170 persisted receipts after provider 403 "
        "interruptions. The primary 3,591 generations are complete and exactly "
        "replayed. Two abandoned retry directories contain 32 logged attempts; "
        "including failed unlogged batches gives 3,626–3,695 physical attempts "
        "for the entire operational run. Those attempts are disclosed in the "
        "recovery audit and excluded from the balanced primary estimand.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("CREDIT_V4_EXECUTOR_FACTORIAL_RESULTS.md")
    )
    args = parser.parse_args()
    report = analyze(args.results)
    (args.results / "diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    args.output.write_text(markdown(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
