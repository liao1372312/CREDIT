"""Read-only diagnostics for a completed CREDIT-Contract confirmation run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


TARGETED = {
    "latest_message_reminder_reply",
    "split_message_update",
    "replace_contact_and_remind",
    "replace_reminder_update_contact",
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def punctuation_normalized_recovery(row: dict) -> int:
    """Post-hoc diagnostic for the unquoted terminal-period prompt only."""
    if row["requested_state_success"]:
        return 1
    if row["task"] != "contract_neutral_recovery_alt":
        return 0
    if abs(float(row["requested_state_score"]) - 0.8) > 1e-9:
        return 0
    messages = row["ending_context"]["_dbs"]["MESSAGING"]
    index = max(item["sandbox_message_index"] for item in messages
                if item.get("sandbox_message_index") is not None)
    latest = [item for item in messages if item.get("sandbox_message_index") == index]
    return int(any(
        item.get("recipient_phone_number") == "+14155550181"
        and str(item.get("content", "")).rstrip(".!?") == "Access restored"
        for item in latest
    ))


def analyze(root: Path) -> dict:
    rows = read_rows(root / "confirmation_sessions.jsonl")
    methods = list(dict.fromkeys(row["method"] for row in rows))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    diagnostic = []
    for method in methods:
        subset = grouped[method]
        targeted = [row for row in subset if row["family"] in TARGETED]
        controls = [row for row in subset if row["family"] not in TARGETED]
        diagnostic.append({
            "method": method,
            "sessions": len(subset),
            "registered_requested_state_VSR": sum(
                row["requested_state_success"] for row in subset) / len(subset),
            "targeted_family_requested_state_VSR": sum(
                row["requested_state_success"] for row in targeted) / len(targeted),
            "neutral_control_requested_state_VSR": sum(
                row["requested_state_success"] for row in controls) / len(controls),
            "posthoc_punctuation_normalized_VSR": sum(
                punctuation_normalized_recovery(row) for row in subset) / len(subset),
        })

    contract = grouped["CREDIT-Contract"]
    violations = Counter(
        turn["contract_violation"]
        for row in contract for turn in row["generations"]
        if "contract_violation" in turn
    )
    terminations = Counter(row["termination"] for row in contract)
    affected = sum(any("contract_violation" in turn for turn in row["generations"])
                   for row in contract)
    report = {
        "scope": "read-only post-hoc diagnostics; registered endpoints are unchanged",
        "sessions": len(rows),
        "diagnostic_table": diagnostic,
        "contract_feedback": {
            "sessions_with_feedback": affected,
            "total_rejected_actions": sum(violations.values()),
            "reasons": dict(violations.most_common()),
            "termination_counts": dict(terminations),
        },
    }
    output = root / "posthoc_analysis.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.results), indent=2))


if __name__ == "__main__":
    main()
