"""Run a small, leakage-safe randomized two-memory tau-bench CREDIT pilot.

This is a development-only pilot. It estimates conditional effects for pairs
selected without task action labels; it does not claim primary benchmark VSR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .tau_credit_runner import POOL_ROOT, ROOT, digest, install_tau_imports, run_episode


STOP = set("a an and are as at be by can could do for from get how i in is it me my of on or please the this to want with would you your".split())


def words(value: str) -> set[str]:
    parts = re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
    return {word for word in parts if len(word) > 1 and word not in STOP}


def task_metadata(domain: str, task_id: int) -> tuple[str, list[dict]]:
    get_env, _, _, _, active = install_tau_imports()
    active["client"] = object()  # User API is not called until env.reset().
    env = get_env(domain, user_strategy="llm", user_model="metadata-only",
                  user_provider="openai", task_split="test", task_index=task_id)
    return env.task.instruction, env.tools_info


def eligible_pair(domain: str, task_id: int, pool: list[dict]) -> tuple[dict, dict, dict]:
    instruction, tool_schemas = task_metadata(domain, task_id)
    query = words(instruction)
    tool_text = {
        (item.get("function") or {}).get("name", ""): words(
            (item.get("function") or {}).get("name", "") + " " +
            (item.get("function") or {}).get("description", "")
        ) for item in tool_schemas
    }
    ranked = []
    for item in pool:
        text_terms = words(item["text"])
        tool_overlap = sum(len(query & tool_text.get(name, set())) for name in item.get("tools", []))
        key_terms = words(" ".join(k for call in item.get("tool_chain", [])
                                    for k in call.get("argument_keys", [])))
        score = (4 * tool_overlap + 2 * len(query & key_terms)
                 + len(query & text_terms))
        ranked.append((score, item["uid"], item))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    if len(ranked) < 2:
        raise ValueError("fewer than two memory candidates")
    return ranked[0][2], ranked[1][2], {
        "task_instruction_sha256": digest(instruction.encode()),
        "retrieval_score_1": ranked[0][0], "retrieval_score_2": ranked[1][0],
        "retrieval_rule": "frozen lexical overlap with task instruction plus official tool descriptions; no task action/output access",
    }


def task_seed(domain: str, task_id: int) -> int:
    value = hashlib.sha256(f"tau-credit-factorial-pilot-v1|{domain}|{task_id}".encode()).hexdigest()
    return 93000 + int(value[:8], 16) % 1000000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dev-per-domain", type=int, default=4)
    p.add_argument("--output", type=Path, default=ROOT / "results/tau_credit_factorial_pilot_remote_v1")
    p.add_argument("--env", type=Path, default=ROOT / ".env")
    p.add_argument("--seed-offset", type=int, default=0)
    args = p.parse_args()
    if args.dev_per_domain < 1:
        raise ValueError("dev-per-domain must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    pool_sha = digest((POOL_ROOT / "split_manifest.json").read_bytes())
    saved_path = args.output / "episodes.jsonl"
    events_path = args.output / "collection_events.jsonl"
    interrupted = []
    if events_path.exists():
        interrupted = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    split_manifest = json.loads((POOL_ROOT / "split_manifest.json").read_text())
    planned_episodes = sum(
        min(len(split_manifest["split_task_ids"][domain]["development"]), args.dev_per_domain) * 4
        for domain in ("retail", "airline")
    )
    saved = {}
    if saved_path.exists():
        for line in saved_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                saved[row["run_id"]] = row
    results = []
    halt_collection = False
    with saved_path.open("a", encoding="utf-8") as stream:
        for domain in ("retail", "airline"):
            pool = json.loads((POOL_ROOT / f"{domain}_memory_pool_64.json").read_text())
            manifest = json.loads((POOL_ROOT / "split_manifest.json").read_text())
            ids = manifest["split_task_ids"][domain]["development"]
            ids = sorted(ids, key=lambda task_id: hashlib.sha256(
                f"tau-credit-factorial-order-v1|{domain}|{task_id}".encode()).hexdigest())
            for task_id in ids[:args.dev_per_domain]:
                first, second, retriever = eligible_pair(domain, task_id, pool)
                seed = task_seed(domain, task_id) + args.seed_offset
                masks = (("00", []), ("10", [first]), ("01", [second]), ("11", [first, second]))
                for mask, exposure in masks:
                    run_id = digest(f"{domain}|{task_id}|{seed}|{mask}|{first['uid']}|{second['uid']}".encode())
                    previous = saved.get(run_id)
                    if previous and previous["status"] == "completed":
                        results.append(saved[run_id]); continue
                    base = {
                        "run_id": run_id, "workload": domain, "task_id": task_id,
                        "family": f"{domain}:task-{task_id}", "seed": seed,
                        "stage": "development_randomized_factorial_pilot",
                        "pair_uids": [first["uid"], second["uid"]], "mask": mask,
                        "exposure_uids": [item["uid"] for item in exposure],
                        "retriever": retriever, "pool_manifest_sha256": pool_sha,
                        "attempt_number": int((previous or {}).get("attempt_number", 0)) + 1,
                    }
                    try:
                        episode = run_episode(domain, task_id, seed, exposure,
                                              env_path=args.env, backend="remote")
                        row = {**base, "status": "completed", "episode": episode}
                    except Exception as exc:
                        # Provider errors and malformed generations are retained;
                        # raw prompts are never logged and unavailable usage is
                        # not imputed.
                        row = {**base, "status": "failed", "error_type": type(exc).__name__,
                               "error": str(exc)[:300]}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    saved[run_id] = row
                    results.append(row)
                    print(json.dumps({"completed": sum(r["status"] == "completed" for r in results),
                                      "failed": sum(r["status"] == "failed" for r in results),
                                      "planned": planned_episodes}), flush=True)
                    if row["status"] == "failed" and ("额度不足" in row["error"] or "quota insufficient" in row["error"].lower()):
                        halt_collection = True
                        break
                if halt_collection:
                    break
            if halt_collection:
                break
    # `saved` keeps the latest attempt per planned cell, while JSONL preserves
    # every attempt for a full failure/retry audit.
    final_rows = list(saved.values())
    completed = [row for row in final_rows if row["status"] == "completed"]
    by_pair = {}
    for row in completed:
        by_pair.setdefault((row["workload"], row["task_id"], tuple(row["pair_uids"])), {})[row["mask"]] = row["episode"]["success"]
    effects = []
    for (domain, task_id, pair), values in by_pair.items():
        if set(values) != {"00", "10", "01", "11"}:
            continue
        y00, y10, y01, y11 = (float(values[k]) for k in ("00", "10", "01", "11"))
        effects.append({"workload": domain, "task_id": task_id, "memory_uids": list(pair),
                        "y00": y00, "y10": y10, "y01": y01, "y11": y11,
                        "effect_1": 0.5 * ((y10 - y00) + (y11 - y01)),
                        "effect_2": 0.5 * ((y01 - y00) + (y11 - y10)),
                        "interaction": y11 - y10 - y01 + y00})
    summary = {
        "scope": "remote randomized development factorial collection; not primary or confirmatory VSR",
        "protocol": "two retrieved development memories randomized over all four exposure masks; paired task and seed",
        "requested_model": json.loads((ROOT / "results/tau_credit_protocol_v1/protocol_manifest.json").read_text())["actor_and_user_requested_model"],
        "pool_manifest_sha256": pool_sha, "planned_episodes": planned_episodes,
        "completed_episodes": len(completed),
        "failed_episodes": len(final_rows) - len(completed),
        "collection_halted_for_quota": halt_collection,
        "unlogged_interrupted_attempts": len(interrupted),
        "interrupted_attempts_note": "Operator-stopped HTTP requests may have been billed; provider usage was unavailable and is excluded from completed-episode token totals.",
        "complete_factorial_pairs": len(effects),
        "factorial_effects": effects,
        "episodes_sha256": digest(saved_path.read_bytes()),
        "limitations": ["development-only", "no fresh certification", "not a full CREDIT bank update", "no baseline comparison", "only completed cells enter factorial effect calculations"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("scope", "planned_episodes", "completed_episodes", "failed_episodes", "complete_factorial_pairs", "factorial_effects")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
