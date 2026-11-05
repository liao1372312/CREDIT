"""Collect a balanced, randomized development factorial over the full pool.

This supplements (and does not replace) the lexical-pair pilot. Every memory
record is assigned to at least one randomized pair within each workload, and
each task receives all four exposure masks. It is development evidence only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .tau_credit_runner import POOL_ROOT, ROOT, digest, run_episode


PROTOCOL = "tau-original-CREDIT-balanced-factorial-dev-v1"
SEED_OFFSET = 2_000_000
SCHEDULE_SEED = 946_031


def round_robin_pairs(uids: list[str], n_pairs: int, seed: int) -> list[tuple[str, str]]:
    """Return unique randomized pairs with near-equal record coverage."""
    if len(uids) < 2 or len(uids) % 2:
        raise ValueError("pool size must be an even number >= 2")
    if not 0 <= n_pairs <= len(uids) * (len(uids) - 1) // 2:
        raise ValueError("requested pair count exceeds unique pairs")
    # A local PRNG keeps the frozen assignment independent of global RNG state.
    import random
    rng = random.Random(seed)
    vertices = list(uids)
    rng.shuffle(vertices)
    fixed, ring = vertices[0], vertices[1:]
    rounds = []
    for _ in range(len(uids) - 1):
        arrangement = [fixed, *ring]
        pairs = [(arrangement[i], arrangement[-1 - i])
                 for i in range(len(uids) // 2)]
        rng.shuffle(pairs)
        rounds.append(pairs)
        ring = [ring[-1], *ring[:-1]]
    rng.shuffle(rounds)
    schedule = [pair for round_pairs in rounds for pair in round_pairs]
    selected = schedule[:n_pairs]
    if len({frozenset(pair) for pair in selected}) != len(selected):
        raise AssertionError("round-robin produced duplicate pairs")
    return selected


def task_seed(domain: str, task_id: int) -> int:
    value = hashlib.sha256(
        f"{PROTOCOL}|{domain}|{task_id}".encode()
    ).hexdigest()
    return 93000 + int(value[:8], 16) % 1_000_000 + SEED_OFFSET


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def summarize(rows: list[dict], planned: int) -> dict:
    latest = {}
    for row in rows:
        latest[row["run_id"]] = row
    completed = [row for row in latest.values() if row["status"] == "completed"]
    failed = [row for row in latest.values() if row["status"] != "completed"]
    by_pair: dict[tuple, dict] = {}
    mask_counts: dict[tuple[str, str], list[int]] = {}
    candidate_support: dict[str, dict[str, int]] = {"retail": {}, "airline": {}}
    for row in completed:
        key = (row["workload"], row["task_id"], tuple(row["pair_uids"]))
        by_pair.setdefault(key, {})[row["mask"]] = row["episode"]["success"]
        mask_key = (row["workload"], row["mask"])
        mask_counts.setdefault(mask_key, []).append(int(row["episode"]["success"]))
    effects = []
    for (domain, task_id, pair), values in by_pair.items():
        if set(values) != {"00", "10", "01", "11"}:
            continue
        for uid in pair:
            candidate_support[domain][uid] = candidate_support[domain].get(uid, 0) + 1
        y00, y10, y01, y11 = (float(values[key]) for key in ("00", "10", "01", "11"))
        effects.append({
            "workload": domain, "task_id": task_id, "memory_uids": list(pair),
            "y00": y00, "y10": y10, "y01": y01, "y11": y11,
            "effect_1": 0.5 * ((y10 - y00) + (y11 - y01)),
            "effect_2": 0.5 * ((y01 - y00) + (y11 - y10)),
            "interaction": y11 - y10 - y01 + y00,
        })
    coverage = {}
    for domain in ("retail", "airline"):
        assigned = candidate_support[domain]
        pool = json.loads((POOL_ROOT / f"{domain}_memory_pool_64.json").read_text())
        counts = {item["uid"]: assigned.get(item["uid"], 0) for item in pool}
        coverage[domain] = {
            "pool_records": len(pool),
            "records_with_complete_randomized_pair": sum(count > 0 for count in counts.values()),
            "records_without_support": sum(count == 0 for count in counts.values()),
            "min_pair_support": min(counts.values()),
            "max_pair_support": max(counts.values()),
            "pair_support_by_uid": counts,
        }
    usage = {key: 0 for key in (
        "actor_generations", "user_simulator_generations", "prompt_tokens",
        "completion_tokens", "tool_calls", "http_attempts",
    )}
    request_hashes = set()
    provider_models = set()
    episode_seconds = 0.0
    for row in completed:
        episode = row["episode"]
        for key in usage:
            usage[key] += int(episode.get(key, 0) or 0)
        request_hashes.update(episode.get("request_hashes", []))
        provider_models.update(episode.get("provider_models", []))
        episode_seconds += float(episode.get("duration_seconds", 0) or 0)
    usage["model_requests"] = usage["actor_generations"] + usage["user_simulator_generations"]
    usage["unique_request_hashes"] = len(request_hashes)
    usage["episode_seconds"] = episode_seconds
    return {
        "scope": "balanced randomized development factorial over full candidate pools; descriptive only",
        "protocol": PROTOCOL,
        "planned_episodes": planned,
        "completed_episodes": len(completed),
        "failed_latest_cells": len(failed),
        "unattempted_cells": planned - len(latest),
        "complete_task_pair_factorials": len(effects),
        "mask_vsr": {
            f"{domain}/{mask}": {
                "successes": sum(values), "episodes": len(values),
                "vsr_percent": 100 * sum(values) / len(values),
            }
            for (domain, mask), values in sorted(mask_counts.items())
        },
        "candidate_coverage": coverage,
        "usage": usage,
        "provider_models": sorted(provider_models),
        "factorial_effects": effects,
        "limitations": [
            "development split only",
            "single randomized pair per task; candidate-specific estimates are sparse",
            "not the full CREDIT bank-adaptation procedure",
            "no fresh certification or final-test results",
            "no baseline comparison",
        ],
    }


def audit_collection(output: Path) -> dict:
    protocol = json.loads((output / "frozen_protocol.json").read_text())
    episode_path = output / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text().splitlines() if line.strip()]
    schedule = protocol["schedule"]
    schedule_hash = digest(json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode())
    expected_ids = [row["run_id"] for row in schedule]
    latest = {}
    attempts_by_id: dict[str, int] = {}
    for row in rows:
        latest[row["run_id"]] = row
        attempts_by_id[row["run_id"]] = attempts_by_id.get(row["run_id"], 0) + 1
    completed = {key: row for key, row in latest.items() if row["status"] == "completed"}
    masks_by_task: dict[tuple, set[str]] = {}
    for row in completed.values():
        masks_by_task.setdefault((row["workload"], row["task_id"], tuple(row["pair_uids"])), set()).add(row["mask"])
    pool_coverage = {}
    for domain in ("retail", "airline"):
        pool = json.loads((POOL_ROOT / f"{domain}_memory_pool_64.json").read_text())
        complete_uids = {
            uid for (workload, _task, pair), masks in masks_by_task.items()
            if workload == domain and masks == {"00", "10", "01", "11"}
            for uid in pair
        }
        pool_coverage[domain] = {
            "pool_records": len(pool),
            "records_with_complete_factorial_support": sum(item["uid"] in complete_uids for item in pool),
            "records_without_complete_factorial_support": sum(item["uid"] not in complete_uids for item in pool),
        }
    checks = {
        "schedule_hash_matches": schedule_hash == protocol["schedule_sha256"],
        "schedule_run_ids_unique": len(expected_ids) == len(set(expected_ids)),
        "all_attempts_belong_to_frozen_schedule": all(row["run_id"] in set(expected_ids) for row in rows),
        "statuses_valid": all(row["status"] in {"completed", "failed"} for row in rows),
        "completed_episodes_have_unique_run_ids": len(completed) == sum(row["status"] == "completed" for row in rows),
        "candidate_pair_coverage_design_covers_full_pool": all(
            len({uid for spec in schedule if spec["workload"] == domain for uid in spec["pair_uids"]}) == 64
            for domain in ("retail", "airline")
        ),
    }
    return {
        "protocol": PROTOCOL,
        "audit_passed": all(checks.values()),
        "checks": checks,
        "planned_cells": len(schedule),
        "attempt_rows": len(rows),
        "latest_completed_cells": len(completed),
        "latest_failed_cells": len(latest) - len(completed),
        "unattempted_cells": len(schedule) - len(latest),
        "attempts_per_cell": {str(count): sum(value == count for value in attempts_by_id.values())
                              for count in sorted(set(attempts_by_id.values()))},
        "complete_factorial_candidate_coverage": pool_coverage,
        "frozen_protocol_sha256": digest((output / "frozen_protocol.json").read_bytes()),
        "episodes_sha256": digest(episode_path.read_bytes()),
        "runner_sha256": digest(Path(__file__).read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/tau_credit_balanced_factorial_dev_v1")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--retries", type=int, default=2,
                        help="additional attempts for failed cells on this invocation")
    args = parser.parse_args()
    if args.retries < 0:
        raise ValueError("retries must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "frozen_protocol.json"
    manifest_path = POOL_ROOT / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pool_sha = digest(manifest_path.read_bytes())
    schedule = []
    schedules_by_domain = {}
    for domain in ("retail", "airline"):
        pool = json.loads((POOL_ROOT / f"{domain}_memory_pool_64.json").read_text())
        ids = manifest["split_task_ids"][domain]["development"]
        ids = sorted(ids, key=lambda value: hashlib.sha256(
            f"{PROTOCOL}|task-order|{domain}|{value}".encode()).hexdigest())
        pairs = round_robin_pairs([item["uid"] for item in pool], len(ids),
                                  SCHEDULE_SEED + (0 if domain == "retail" else 1))
        task_pairs = list(zip(ids, pairs))
        schedules_by_domain[domain] = task_pairs
        for task_id, pair in task_pairs:
            pair = tuple(pair)
            seed = task_seed(domain, task_id)
            for mask, exposure in (
                ("00", []), ("10", [pair[0]]), ("01", [pair[1]]), ("11", list(pair))
            ):
                run_id = digest(
                    f"{PROTOCOL}|{domain}|{task_id}|{seed}|{mask}|{pair[0]}|{pair[1]}".encode()
                )
                schedule.append({
                    "run_id": run_id, "workload": domain, "task_id": task_id,
                    "family": f"{domain}:task-{task_id}", "seed": seed,
                    "stage": "development_balanced_randomized_factorial",
                    "pair_uids": list(pair), "mask": mask,
                    "exposure_uids": exposure, "pool_manifest_sha256": pool_sha,
                })
    frozen = {
        "protocol": PROTOCOL,
        "status": "frozen before collection",
        "model_requested": "inherited from tau_credit_protocol_v1",
        "actor_temperature": 0,
        "actor_top_p": "unset; provider default",
        "task_split": "development only; task IDs from frozen tau_original_credit_pool_v2 manifest",
        "pool_manifest_sha256": pool_sha,
        "pair_design": "randomized round-robin one-factorization over all 64 candidate UIDs; unique pairs within workload",
        "schedule_seed": SCHEDULE_SEED,
        "seed_offset": SEED_OFFSET,
        "planned_cells": len(schedule),
        "task_families": sum(len(v) for v in schedules_by_domain.values()),
        "mask_assignment": ["00", "10", "01", "11"],
        "leakage_boundary": "task outcomes are not used to form pair assignments",
        "claim_boundary": "development support expansion; sparse candidate-specific evidence; not confirmatory benchmark performance",
        "schedule_sha256": digest(json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()),
        "schedule": schedule,
    }
    if protocol_path.exists():
        prior = json.loads(protocol_path.read_text())
        if prior.get("schedule_sha256") != frozen["schedule_sha256"]:
            raise RuntimeError("existing frozen schedule differs from current code or pool")
    else:
        write_json(protocol_path, frozen)

    episodes_path = args.output / "episodes.jsonl"
    attempts = []
    if episodes_path.exists():
        attempts = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    latest = {}
    for row in attempts:
        latest[row["run_id"]] = row
    failed_this_call = 0
    with episodes_path.open("a", encoding="utf-8") as stream:
        for spec in schedule:
            previous = latest.get(spec["run_id"])
            if previous and previous["status"] == "completed":
                continue
            attempts_for_cell = 0
            while attempts_for_cell <= args.retries:
                base = {**spec, "attempt_number": int((previous or {}).get("attempt_number", 0)) + 1}
                pair_uids = set(spec["pair_uids"])
                pool = json.loads((POOL_ROOT / f"{spec['workload']}_memory_pool_64.json").read_text())
                memory_by_uid = {item["uid"]: item for item in pool}
                exposure = [memory_by_uid[uid] for uid in spec["exposure_uids"]]
                try:
                    episode = run_episode(spec["workload"], spec["task_id"], spec["seed"],
                                          exposure, env_path=args.env, backend="remote")
                    row = {**base, "status": "completed", "episode": episode}
                except Exception as exc:
                    row = {**base, "status": "failed", "error_type": type(exc).__name__,
                           "error": str(exc)[:300]}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                attempts.append(row)
                latest[spec["run_id"]] = row
                previous = row
                attempts_for_cell += 1
                if row["status"] == "completed":
                    break
                failed_this_call += 1
                err = row.get("error", "").lower()
                if "额度不足" in row.get("error", "") or "quota" in err or "insufficient" in err:
                    break
            done = sum(value["status"] == "completed" for value in latest.values())
            print(json.dumps({"completed": done, "planned": len(schedule),
                              "latest_failures": sum(value["status"] != "completed" for value in latest.values()),
                              "last_cell": spec["run_id"]}), flush=True)
            if previous and previous["status"] != "completed" and (
                "额度不足" in previous.get("error", "") or "quota" in previous.get("error", "").lower()
                or "insufficient" in previous.get("error", "").lower()
            ):
                break
    summary = summarize(attempts, len(schedule))
    summary["failed_attempts_this_invocation"] = failed_this_call
    summary["failed_attempt_rows_total"] = sum(row["status"] != "completed" for row in attempts)
    summary["failed_attempt_error_types"] = {
        error_type: sum(row.get("error_type") == error_type for row in attempts if row["status"] != "completed")
        for error_type in sorted({row.get("error_type", "unknown") for row in attempts if row["status"] != "completed"})
    }
    summary["failed_attempt_billing_note"] = (
        "Usage is included only for successful episode records; provider billing for failed attempts is not available and is not imputed."
    )
    summary["episodes_sha256"] = digest(episodes_path.read_bytes())
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "audit.json", audit_collection(args.output))
    print(json.dumps({key: summary[key] for key in (
        "planned_episodes", "completed_episodes", "failed_latest_cells",
        "unattempted_cells", "complete_task_pair_factorials", "candidate_coverage",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
