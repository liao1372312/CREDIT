"""Read-only post-hoc semantic audit of the frozen blind-composition traces.

This audit repairs one synthetic golden-state timestamp and diagnoses the
case-sensitive lexical final-answer check.  It never changes or reruns an actor.
All repaired quantities are labeled post hoc and kept separate from the frozen
pre-registered endpoints.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .toolsandbox_local_v3 import install_local_clock, upstream_imports
from .toolsandbox_m_blind_composition import (
    BOOTSTRAP_SEED,
    CLOCK,
    METHODS,
    build_blind_scenarios,
)
from .user_data import digest


TARGET_REMINDER_ID = "54f20be9-eac3-5e76-8e57-724e48d3d477"


def _rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _successful_tools(row: dict) -> set[str]:
    return {call["tool"] for call in row["calls"] if not call.get("exception")}


def _latest_snapshot(rows: list[dict]) -> list[dict]:
    indices = [row["sandbox_message_index"] for row in rows
               if row["sandbox_message_index"] is not None]
    index = max(indices)
    return [row for row in rows if row["sandbox_message_index"] == index
            and any(value is not None for key, value in row.items()
                    if key != "sandbox_message_index")]


def _replacement_state(row: dict, expected_time: float) -> bool:
    final = _latest_snapshot(row["ending_context"]["_dbs"]["REMINDER"])
    old_absent = all(item["reminder_id"] != TARGET_REMINDER_ID for item in final)
    replacements = [item for item in final
                    if item["content"] == "Pack conference badge"
                    and abs(float(item["reminder_timestamp"]) - expected_time) < 1e-6]
    return old_absent and len(replacements) == 1


def _semantic_text(row: dict) -> bool:
    answer = str(row["generations"][-1]["action"].get("answer", "")).casefold()
    failure = any(phrase in answer for phrase in
                  ("could not", "unable", "failed", "please provide"))
    family = row["trajectory"]
    if family == "message_to_reminder":
        return not failure and "reminder" in answer and "good, keep me posted" in answer
    if family == "reminder_to_message":
        return not failure and ("sent" in answer or "message" in answer)
    if family == "add_contact_then_message":
        return not failure and "welcome aboard" in answer
    if family == "replace_upcoming_reminder":
        return not failure and "pack conference badge" in answer
    if family == "enable_wifi_and_cellular":
        normalized = answer.replace("-", "").replace("‑", "")
        return (not failure and "wifi" in normalized
                and ("cellular" in answer or "mobile" in answer)
                and ("enabled" in answer or "turned on" in answer or " are on" in answer))
    if family == "message_sender_to_contact":
        return not failure and "gpu seller" in answer
    raise ValueError(family)


def _paired(rows: list[dict], field: str) -> dict:
    by = defaultdict(dict)
    for row in rows:
        by[row["method"]][(row["actor_seed"], row["trajectory"],
                           row["session_index"])] = row
    clusters = sorted({(row["actor_seed"], row["trajectory"]) for row in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    out = []
    for method in METHODS:
        effects = []
        for seed, family in clusters:
            keys = [(seed, family, i) for i in (0, 1)]
            effects.append(float(np.mean([
                by["CREDIT-Intent"][key][field] - by[method][key][field]
                for key in keys])))
        boot = np.mean(np.asarray(effects)[draws], axis=1)
        wins, losses = sum(x > 0 for x in effects), sum(x < 0 for x in effects)
        p = float(binomtest(wins, wins + losses, .5,
                            alternative="greater").pvalue) if wins + losses else 1.0
        out.append({"baseline": method, "delta": float(np.mean(effects)),
            "cluster_bootstrap_CI95": np.quantile(boot, [.025, .975]).tolist(),
            "cluster_wins": wins, "cluster_losses": losses,
            "one_sided_exact_sign_p": p})
    return {"endpoint": field, "comparisons": out, "clusters": len(clusters),
            "resampling_unit": "actor seed by two-prompt task family"}


def _holm_primary(report: dict) -> list[dict]:
    selected = [row for row in report["comparisons"]
                if row["baseline"] in ("FIFO", "Semantic")]
    ordered = sorted(selected, key=lambda row: row["one_sided_exact_sign_p"])
    running = 0.0
    for rank, row in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * row["one_sided_exact_sign_p"])
        running = max(running, adjusted)
        row["holm_adjusted_p"] = running
    return sorted(ordered, key=lambda row: row["baseline"])


def audit(root: Path) -> dict:
    protocol = json.loads((root / "frozen_protocol.json").read_text())
    u = upstream_imports(Path(protocol["upstream_directory"]))
    install_local_clock(CLOCK)
    scenarios, _, _ = build_blind_scenarios(u)
    initial = scenarios["blind_replace_upcoming_reminder"].starting_context.get_database(
        u.DatabaseNamespace.REMINDER).to_dicts()
    expected_time = float(next(item["reminder_timestamp"] for item in initial
                               if item["reminder_id"] == TARGET_REMINDER_ID))
    frozen_wrong_time = 1792160294.229264
    rows = _rows(root / "confirmation_sessions.jsonl")
    for row in rows:
        replacement = row["trajectory"] == "replace_upcoming_reminder"
        repaired_state = (_replacement_state(row, expected_time) if replacement
                          else bool(row["mutation_success"]))
        tools = _successful_tools(row)
        repaired_tool_state = (repaired_state and
            {"search_reminder", "get_current_timestamp", "remove_reminder", "add_reminder"}
            <= tools) if replacement else bool(row["tool_state_success"])
        row["posthoc_requested_state_success"] = int(repaired_state)
        row["posthoc_required_tool_state_success"] = int(repaired_tool_state)
        row["posthoc_semantic_text_success"] = int(_semantic_text(row))

    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        item = {"method": method, "sessions": len(subset),
            "frozen_tool_state_VSR": float(np.mean([r["tool_state_success"] for r in subset])),
            "frozen_mutation_VSR": float(np.mean([r["mutation_success"] for r in subset])),
            "posthoc_required_tool_state_VSR": float(np.mean([
                r["posthoc_required_tool_state_success"] for r in subset])),
            "posthoc_requested_state_VSR": float(np.mean([
                r["posthoc_requested_state_success"] for r in subset])),
            "frozen_lexical_text_VSR": float(np.mean([r["final_text_success"] for r in subset])),
            "posthoc_semantic_text_VSR": float(np.mean([
                r["posthoc_semantic_text_success"] for r in subset]))}
        for budget in (5000, 10000, 20000):
            item[f"posthoc_requested_state_VSR_at_{budget}ms"] = float(np.mean([
                r["posthoc_requested_state_success"] and r["remote_latency_ms"] <= budget
                for r in subset]))
        table.append(item)
    latency_rankings = {}
    for budget in (5000, 10000, 20000):
        key = f"posthoc_requested_state_VSR_at_{budget}ms"
        latency_rankings[str(budget)] = [row["method"] for row in sorted(
            table, key=lambda row: (-row[key], -row["posthoc_requested_state_VSR"],
                                    row["method"]))]
    state_stats = _paired(rows, "posthoc_requested_state_success")
    tool_state_stats = _paired(rows, "posthoc_required_tool_state_success")
    result = {
        "post_hoc": True,
        "actor_outputs_rerun": False,
        "frozen_confirmation_sha256": digest(root / "confirmation_sessions.jsonl"),
        "repair_reason": {
            "replacement_timestamp": (
                "The frozen synthetic evaluator used a timestamp copied from a base scenario "
                "instantiated before the fixed clock. The request requires preserving the "
                "timestamp returned by the frozen scenario itself."),
            "text_match": (
                "The frozen column_contains_similarity check is case-sensitive and has no "
                "synonym normalization; semantically correct 'Reminder'/'enabled' answers "
                "were false negatives."),
        },
        "replacement_expected_timestamp_from_frozen_start_state": expected_time,
        "replacement_erroneous_golden_timestamp": frozen_wrong_time,
        "table": table,
        "posthoc_requested_state_latency_rankings": latency_rankings,
        "requested_state_statistics": state_stats,
        "required_tool_state_statistics": tool_state_stats,
        "holm_primary_requested_state": _holm_primary(state_stats),
        "holm_primary_required_tool_state": _holm_primary(tool_state_stats),
        "semantic_text_rule": "task-local casefolded entity/action check; diagnostic only",
        "audit_source_sha256": digest(Path(__file__)),
    }
    (root / "posthoc_semantic_audit.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.results), indent=2))


if __name__ == "__main__":
    main()
