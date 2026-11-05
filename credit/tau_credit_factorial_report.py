"""Audit and summarize completed/failing CREDIT factorial collection cells."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .tau_credit_runner import ROOT, digest


def build(directory: Path, planned: int = 436) -> dict:
    source = directory / "episodes.jsonl"
    attempts = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    latest = {}
    for row in attempts:
        latest[row["run_id"]] = row
    rows = list(latest.values())
    done = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] != "completed"]
    groups = defaultdict(dict)
    by_domain_mask = defaultdict(list)
    for row in done:
        episode = row["episode"]
        groups[(row["workload"], row["task_id"], tuple(row["pair_uids"]))][row["mask"]] = episode["success"]
        by_domain_mask[(row["workload"], row["mask"])].append(episode)
    effects = []
    memory_context_support = defaultdict(set)
    for (domain, task_id, pair), values in groups.items():
        for uid in pair:
            memory_context_support[(domain, uid)].add(task_id)
        if set(values) != {"00", "10", "01", "11"}:
            continue
        y00, y10, y01, y11 = (float(values[k]) for k in ("00", "10", "01", "11"))
        effects.append({"workload": domain, "task_id": task_id, "memory_uids": list(pair),
                        "y00": y00, "y10": y10, "y01": y01, "y11": y11,
                        "effect_1": 0.5 * ((y10 - y00) + (y11 - y01)),
                        "effect_2": 0.5 * ((y01 - y00) + (y11 - y10)),
                        "interaction": y11 - y10 - y01 + y00})
    matched_mask_summary = {}
    for domain in ("retail", "airline"):
        matched = [row for row in effects if row["workload"] == domain]
        for mask, key in (("00", "y00"), ("10", "y10"), ("01", "y01"), ("11", "y11")):
            successes = sum(int(row[key]) for row in matched)
            n = len(matched)
            matched_mask_summary[f"{domain}/{mask}"] = {
                "successes": successes, "episodes": n,
                "vsr_percent": 100 * successes / n if n else None,
            }
    effect_summary = {}
    for domain in ("retail", "airline", "all"):
        subset = effects if domain == "all" else [row for row in effects if row["workload"] == domain]
        n = len(subset)
        effect_summary[domain] = {
            "complete_task_pairs": n,
            "mean_effect_1_pp": 100 * sum(row["effect_1"] for row in subset) / n if n else None,
            "mean_effect_2_pp": 100 * sum(row["effect_2"] for row in subset) / n if n else None,
            "mean_interaction_pp": 100 * sum(row["interaction"] for row in subset) / n if n else None,
            "nonzero_interactions": sum(row["interaction"] != 0 for row in subset),
        }
    coverage = {}
    pool_root = ROOT / "datasets/tau_original_credit_pool_v2"
    for domain in ("retail", "airline"):
        pool = json.loads((pool_root / f"{domain}_memory_pool_64.json").read_text())
        support = {m["uid"]: len(memory_context_support.get((domain, m["uid"]), set())) for m in pool}
        coverage[domain] = {
            "pool_records": len(pool),
            "records_seen_in_randomized_pairs": sum(v > 0 for v in support.values()),
            "records_without_randomized_support": sum(v == 0 for v in support.values()),
            "distinct_task_contexts_per_record": {
                "min_supported": min((v for v in support.values() if v > 0), default=0),
                "median_supported": sorted(v for v in support.values() if v > 0)[
                    (sum(v > 0 for v in support.values()) - 1) // 2
                ] if any(v > 0 for v in support.values()) else 0,
                "max_supported": max(support.values(), default=0),
            },
            "support_by_record_uid": support,
        }
    usage = {
        "model_requests": sum(len(row["episode"]["request_hashes"]) for row in done),
        "unique_request_hashes": len({h for row in done for h in row["episode"]["request_hashes"]}),
        "actor_generations": sum(row["episode"]["actor_generations"] for row in done),
        "user_simulator_generations": sum(row["episode"]["user_simulator_generations"] for row in done),
        "prompt_tokens": sum(row["episode"]["prompt_tokens"] for row in done),
        "completion_tokens": sum(row["episode"]["completion_tokens"] for row in done),
        "tool_calls": sum(row["episode"]["tool_calls"] for row in done),
        "http_attempts": sum(row["episode"]["http_attempts"] for row in done),
        "episode_seconds": sum(row["episode"]["duration_seconds"] for row in done),
    }
    mask_summary = {}
    for (domain, mask), episodes in sorted(by_domain_mask.items()):
        mask_summary[f"{domain}/{mask}"] = {
            "successes": sum(bool(e["success"]) for e in episodes),
            "episodes": len(episodes),
            "vsr_percent": 100 * sum(bool(e["success"]) for e in episodes) / len(episodes),
        }
    events_path = directory / "collection_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()] if events_path.exists() else []
    failed_reasons = Counter(row.get("error", "unspecified") for row in failed)
    result = {
        "scope": "development-only randomized CREDIT factorial; descriptive and not confirmatory",
        "planned_cells": planned,
        "completed_cells": len(done),
        "failed_cells_latest_attempt": len(failed),
        "unattempted_cells": max(0, planned - len(rows)),
        "unique_task_families_with_any_completed_cell": len({(r["workload"], r["task_id"]) for r in done}),
        "complete_paired_factorial_families": len(effects),
        "latest_attempt_failures": dict(failed_reasons),
        "unlogged_interrupted_attempts": len(events),
        "interrupted_billing_note": "Remote requests stopped while in flight may be billed; usage is unavailable and excluded from measured totals.",
        "success_by_domain_and_mask": mask_summary,
        "matched_complete_factorial_success_by_domain_and_mask": matched_mask_summary,
        "factorial_effects": effects,
        "factorial_effect_summary": effect_summary,
        "memory_candidate_coverage": coverage,
        "usage": usage,
        "attempt_ledger_rows": len(attempts),
        "episodes_sha256": digest(source.read_bytes()),
        "limitations": ["development split only", "no fresh certification", "no final test run", "no baseline comparison", "incomplete collection due provider quota" if len(rows) < planned else "limited to this declared task inventory"],
    }
    target = directory / "audit_summary.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    # Keep the conventional summary path in sync; this report may be run after
    # an operator stops a collection, before the runner can write its summary.
    (directory / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT / "results/tau_credit_factorial_pilot_remote_v1")
    parser.add_argument("--planned", type=int, default=436)
    args = parser.parse_args()
    print(json.dumps(build(args.directory, args.planned), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
