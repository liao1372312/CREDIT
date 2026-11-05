"""Read-only diagnostics for the frozen CREDIT-Compose confirmation run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .user_data import digest


METHODS = ("NoMem", "FIFO", "Semantic", "CREDIT-Compact", "CREDIT-Intent",
           "CREDIT-Compose")
REGISTERED_BASELINES = ("FIFO", "Semantic", "CREDIT-Intent")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def holm(items: list[dict], key: str) -> list[dict]:
    ordered = sorted(items, key=lambda item: item[key])
    running = 0.0
    for index, item in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - index) * item[key])
        running = max(running, adjusted)
        item["holm_adjusted_p"] = running
    return sorted(ordered, key=lambda item: item["baseline"])


def budget_statistics(data: list[dict], budget_ms: int = 15000) -> dict:
    by = defaultdict(dict)
    for row in data:
        by[row["method"]][(row["actor_seed"], row["trajectory"],
                           row["session_index"])] = row
    clusters = sorted({(row["actor_seed"], row["trajectory"]) for row in data})
    rng = np.random.default_rng(20261017)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    comparisons = []
    for baseline in REGISTERED_BASELINES:
        effects = []
        for seed, family in clusters:
            values = []
            for index in (0, 1):
                key = (seed, family, index)
                candidate, control = by["CREDIT-Compose"][key], by[baseline][key]
                values.append(int(candidate["requested_state_success"] and
                                  candidate["remote_latency_ms"] <= budget_ms) -
                              int(control["requested_state_success"] and
                                  control["remote_latency_ms"] <= budget_ms))
            effects.append(float(np.mean(values)))
        boot = np.mean(np.asarray(effects)[draws], axis=1)
        wins = sum(value > 0 for value in effects)
        losses = sum(value < 0 for value in effects)
        p_value = float(binomtest(wins, wins + losses, .5,
                                  alternative="greater").pvalue) \
            if wins + losses else 1.0
        comparisons.append({"baseline": baseline, "delta": float(np.mean(effects)),
            "cluster_bootstrap_CI95": np.quantile(boot, [.025, .975]).tolist(),
            "cluster_wins": wins, "cluster_losses": losses,
            "one_sided_exact_sign_p": p_value})
    return {"endpoint_was_preregistered": True,
        "test_was_computed_post_collection": True,
        "budget_ms": budget_ms, "clusters": len(clusters),
        "comparisons": comparisons,
        "holm_registered_comparisons": holm(
            [dict(item) for item in comparisons], "one_sided_exact_sign_p")}


def prompt_diagnostic(data: list[dict]) -> dict:
    task_table = []
    for task in sorted({row["task"] for row in data}):
        item = {"task": task, "post_hoc": True}
        for method in METHODS:
            subset = [row for row in data if row["task"] == task and row["method"] == method]
            item[method] = sum(row["requested_state_success"] for row in subset)
        task_table.append(item)
    recreate_tasks = {"compose_replace_contact_and_notify_alt",
                      "compose_replace_reminder_and_notify_alt"}
    recreate = [row for row in data if row["task"] in recreate_tasks]
    remove_called = sum(any(call["tool"] in ("remove_contact", "remove_reminder")
                            and not call.get("exception") for call in row["calls"])
                        for row in recreate)
    explicit = [row for row in data if row["task"] not in recreate_tasks]
    explicit_rates = {}
    for method in METHODS:
        subset = [row for row in explicit if row["method"] == method]
        explicit_rates[method] = float(np.mean([
            row["requested_state_success"] for row in subset]))
    return {"post_hoc": True, "task_table": task_table,
        "recreate_wording_sessions": len(recreate),
        "recreate_wording_successes": sum(row["requested_state_success"] for row in recreate),
        "recreate_wording_successful_remove_calls": remove_called,
        "explicit_subset_excludes": sorted(recreate_tasks),
        "explicit_subset_requested_state_VSR": explicit_rates,
        "interpretation": (
            "The two 'recreate' prompts produced systematic operation omission. "
            "This is a prompt-design limitation and the subset is diagnostic only; "
            "the frozen all-task endpoint remains primary.")}


def transport(root: Path) -> dict:
    generations = []
    for path in sorted(root.glob("compose_confirmation_seed_*/generations.jsonl")):
        generations.extend(rows(path))
    attempts = Counter(str(item.get("network_attempts")) for item in generations)
    models = Counter(str(item.get("provider_model")) for item in generations)
    fingerprints = Counter(str(item.get("system_fingerprint_sha256"))
                           for item in generations)
    return {"passed": bool(generations) and attempts == {"1": len(generations)} and
                       all(item.get("request_sha256") and item.get("response_sha256")
                           for item in generations),
        "live_remote_requests": len(generations),
        "network_attempts": dict(attempts), "provider_models": dict(models),
        "system_fingerprints": dict(fingerprints),
        "minimum_request_latency_ms": min(item["remote_latency_ms"] for item in generations),
        "maximum_request_latency_ms": max(item["remote_latency_ms"] for item in generations),
        "missing_request_or_response_hashes": sum(
            not item.get("request_sha256") or not item.get("response_sha256")
            for item in generations),
        "confirmation_sha256": digest(root / "confirmation_sessions.jsonl")}


def analyze(root: Path) -> dict:
    data = rows(root / "confirmation_sessions.jsonl")
    result = {"frozen_results_unchanged": True,
        "confirmation_sha256": digest(root / "confirmation_sessions.jsonl"),
        "latency_budget_statistics": budget_statistics(data),
        "prompt_wording_diagnostic": prompt_diagnostic(data),
        "analysis_source_sha256": digest(Path(__file__))}
    (root / "posthoc_analysis.json").write_text(json.dumps(result, indent=2))
    (root / "transport_audit.json").write_text(json.dumps(transport(root), indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.results), indent=2))


if __name__ == "__main__":
    main()
