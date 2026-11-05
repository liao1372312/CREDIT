"""Evaluate a development-frozen causal CREDIT bank on tau held-out tasks.

This is a CREDIT-only fixed-bank test. It does not run comparator methods or
claim the fresh update-certification step of the complete CREDIT cycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from .tau_credit_runner import (
    MAX_ACTOR_STEPS, POOL_ROOT, ROOT, digest, install_tau_imports, load_pool, make_chat,
)


PROTOCOL = "tau-original-CREDIT-causal-bank-test-v1"
SOURCE_SUMMARY = ROOT / "results/tau_credit_balanced_factorial_dev_v1/summary.json"
OUTPUT_DEFAULT = ROOT / "results/tau_credit_causal_bank_test_v1"


def wordset(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9_]{1,}", text.lower())}


def estimate_memory_values(domain: str, pool: list[dict], factorial: dict) -> dict:
    values: dict[str, list[float]] = defaultdict(list)
    for row in factorial["factorial_effects"]:
        if row["workload"] != domain:
            continue
        pair = row["memory_uids"]
        values[pair[0]].append(float(row["effect_1"]))
        values[pair[1]].append(float(row["effect_2"]))
    estimated = {}
    for item in pool:
        observations = values.get(item["uid"], [])
        mean = sum(observations) / len(observations) if observations else 0.0
        # Small-sample shrinkage was fixed before looking at test outcomes.
        score = mean * len(observations) / (len(observations) + 4)
        estimated[item["uid"]] = {
            "mean_factorial_effect": mean,
            "support": len(observations),
            "selection_score": score,
        }
    return estimated


def knapsack(pool: list[dict], scores: dict, capacity: int) -> list[dict]:
    # Exact 0/1 dynamic programming. Store immutable record-index tuples to
    # make tie-breaking deterministic and auditable.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for index, item in enumerate(pool):
        weight = int(item["tokens"])
        value = max(0.0, float(scores[item["uid"]]["selection_score"]))
        if value <= 0 or weight > capacity:
            continue
        for used, (utility, chosen) in list(states.items())[::-1]:
            new_used = used + weight
            if new_used > capacity:
                continue
            candidate = (utility + value, chosen + (index,))
            previous = states.get(new_used)
            if previous is None or candidate[0] > previous[0] + 1e-15:
                states[new_used] = candidate
    used, (_, indices) = max(
        states.items(), key=lambda pair: (pair[1][0], -pair[0], tuple(-i for i in pair[1][1]))
    )
    del used
    return [pool[index] for index in indices]


def retrieve(bank: list[dict], scores: dict, instruction: str) -> list[dict]:
    query = wordset(instruction)
    ranked = []
    for item in bank:
        memory_terms = wordset(" ".join(item.get("tools", [])) + " " + item["text"])
        overlap = len(query & memory_terms) / max(1, len(query))
        ranked.append((overlap, scores[item["uid"]]["selection_score"], item["uid"], item))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    # Retrieval uses only retained memory and is capped at two records.
    return [row[3] for row in ranked[:2] if row[0] > 0] or [ranked[0][3]] if ranked else []


def test_seed(domain: str, task_id: int) -> int:
    raw = hashlib.sha256(f"{PROTOCOL}|{domain}|{task_id}".encode()).hexdigest()
    return 7_500_000 + int(raw[:8], 16) % 1_000_000


def run_selected_episode(domain: str, task_id: int, seed: int, bank: list[dict],
                         scores: dict, env_path: Path) -> dict:
    """Run one task and retrieve only after reading its instruction."""
    get_env, Action, respond_name, _SolveResult, active_user = install_tau_imports()
    random.seed(seed)
    actor = make_chat("remote", env_path, seed)
    user = make_chat("remote", env_path, seed + 1)
    active_user["client"] = user
    env = get_env(domain, user_strategy="llm", user_model="configured-endpoint",
                  user_provider="openai", task_split="test", task_index=task_id)
    reset = env.reset(task_index=task_id)
    memory = retrieve(bank, scores, env.task.instruction)
    if len(memory) > 2:
        raise AssertionError("retriever exceeded the frozen exposure limit")
    env.wiki = env.wiki + "\n\nHistorical experience (advisory; do not copy historical values):\n" + \
        json.dumps([item["text"] for item in memory], ensure_ascii=False)
    messages = [{"role": "system", "content": env.wiki},
                {"role": "user", "content": reset.observation}]
    reward = 0.0
    info = reset.info.model_dump()
    action_count = 0
    started = perf_counter()
    for _ in range(MAX_ACTOR_STEPS):
        response = actor.complete(messages, env.tools_info)
        message = response["message"]
        calls = message.get("tool_calls") or []
        if calls:
            call = calls[0]
            fn = call.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError("model tool arguments are not a JSON object")
            step = env.step(Action(name=fn["name"], kwargs=args))
            action_count += 1
            reward = float(step.reward)
            info = {**info, **step.info.model_dump()}
            assistant_message = dict(message)
            assistant_message["tool_calls"] = [call]
            messages.extend([
                assistant_message,
                {"role": "tool", "tool_call_id": call.get("id") or "tau-call",
                 "name": fn["name"], "content": step.observation},
            ])
        else:
            step = env.step(Action(name=respond_name,
                                   kwargs={"content": message.get("content") or ""}))
            messages.append(message)
            reward = float(step.reward)
            info = {**info, **step.info.model_dump()}
            messages.append({"role": "user", "content": step.observation})
        if step.done:
            break
    task = env.task
    all_records = actor.records + user.records
    return {
        "workload": domain, "task_id": task_id, "seed": seed,
        "task_family": f"{domain}:task-{task_id}",
        "task_instruction_sha256": digest(task.instruction.encode()),
        "exposure_uids": [item["uid"] for item in memory],
        "reward": reward, "success": bool(reward >= 1.0 - 1e-9),
        "actor_generations": len(actor.records),
        "user_simulator_generations": len(user.records),
        "tool_calls": action_count,
        "prompt_tokens": sum(int((r.get("usage") or {}).get("prompt_tokens") or 0)
                             for r in all_records),
        "completion_tokens": sum(int((r.get("usage") or {}).get("completion_tokens") or 0)
                                 for r in all_records),
        "http_attempts": sum(int(r.get("network_attempts", 1)) for r in all_records),
        "request_hashes": [r["request_sha256"] for r in all_records],
        "provider_models": sorted({r["provider_model"] for r in all_records}),
        "duration_seconds": perf_counter() - started,
        "executor": "pinned original tau-bench local tool/state/evaluator",
        "backend": "remote",
        "evaluator_info_sha256": digest(json.dumps(info, ensure_ascii=False,
                                                     sort_keys=True, default=str).encode()),
    }


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict], protocol: dict) -> dict:
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] != "completed"]
    by_domain = {}
    for domain in ("retail", "airline"):
        subset = [row["episode"] for row in completed if row["workload"] == domain]
        by_domain[domain] = {
            "successes": sum(bool(row["success"]) for row in subset),
            "episodes": len(subset),
            "vsr_percent": 100 * sum(bool(row["success"]) for row in subset) / len(subset)
                if subset else None,
            "actor_generations": sum(row["actor_generations"] for row in subset),
            "user_simulator_generations": sum(row["user_simulator_generations"] for row in subset),
            "tool_calls": sum(row["tool_calls"] for row in subset),
            "prompt_tokens": sum(row["prompt_tokens"] for row in subset),
            "completion_tokens": sum(row["completion_tokens"] for row in subset),
            "duration_seconds": sum(row["duration_seconds"] for row in subset),
        }
    return {
        "protocol": PROTOCOL,
        "scope": "development-frozen CREDIT causal bank; held-out tasks; no comparator methods; no fresh certification",
        "planned_episodes": sum(len(v) for v in protocol["test_task_ids"].values()),
        "completed_episodes": len(completed),
        "failed_episodes": len(failed),
        "unattempted_episodes": sum(len(v) for v in protocol["test_task_ids"].values()) - len(rows),
        "results_by_workload": by_domain,
        "errors": [row.get("error", "unknown") for row in failed],
        "limitations": [
            "fixed initial bank; the update-cycle fresh certification step is not evaluated",
            "single actor seed per held-out task",
            "no baseline comparison was run",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    episodes_path = args.output / "episodes.jsonl"
    pools, manifests, banks, score_maps = {}, {}, {}, {}
    for domain in ("retail", "airline"):
        pools[domain], manifests[domain] = load_pool(domain)
        factorial = json.loads(SOURCE_SUMMARY.read_text())
        scores = estimate_memory_values(domain, pools[domain], factorial)
        capacity = sum(int(item["tokens"]) for item in pools[domain]) // 2
        bank = knapsack(pools[domain], scores, capacity)
        score_maps[domain], banks[domain] = scores, bank
    split_manifest = json.loads((POOL_ROOT / "split_manifest.json").read_text())
    protocol = {
        "protocol": PROTOCOL,
        "frozen_before_test": True,
        "source_factorial_sha256": digest(SOURCE_SUMMARY.read_bytes()),
        "pool_split_manifest_sha256": digest((POOL_ROOT / "split_manifest.json").read_bytes()),
        "selection": "per-record mean randomized factorial treatment effect, shrunk by n/(n+4); exact 0/1 knapsack under C=floor(L/2)",
        "retrieval": "lexical overlap between held-out task instruction and retained memory tool/text; deterministic score/UID tie-break; maximum two",
        "retrieval_outcome_access": "none; selection callback sees task instruction only after task reset",
        "model": "configured endpoint; temperature 0.0; top_p unset/provider default",
        "test_task_ids": {d: split_manifest["split_task_ids"][d]["test"] for d in ("retail", "airline")},
        "banks": {},
    }
    for domain in ("retail", "airline"):
        pool = pools[domain]
        bank = banks[domain]
        protocol["banks"][domain] = {
            "source_records": len(pool),
            "source_tokens_L": sum(int(item["tokens"]) for item in pool),
            "capacity_tokens_C": sum(int(item["tokens"]) for item in pool) // 2,
            "retained_records": len(bank),
            "retained_tokens": sum(int(item["tokens"]) for item in bank),
            "retained_uids": [item["uid"] for item in bank],
            "scores": score_maps[domain],
        }
    frozen_path = args.output / "frozen_protocol.json"
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text())
        if existing != protocol:
            raise RuntimeError("existing frozen test policy differs; use a new output directory")
    else:
        frozen_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    rows = read_rows(episodes_path)
    latest = {row["run_id"]: row for row in rows}
    if any(row["status"] != "completed" for row in latest.values()):
        raise RuntimeError("this output directory contains a failed attempt; use a new output directory after fixing the endpoint")
    schedule = []
    for domain in ("retail", "airline"):
        for task_id in protocol["test_task_ids"][domain]:
            seed = test_seed(domain, task_id)
            run_id = digest(f"{PROTOCOL}|{domain}|{task_id}|{seed}".encode())
            schedule.append((run_id, domain, task_id, seed))
    with episodes_path.open("a", encoding="utf-8") as out:
        for run_id, domain, task_id, seed in schedule:
            previous = latest.get(run_id)
            if previous and previous["status"] == "completed":
                continue
            retained = banks[domain]
            scores = score_maps[domain]
            try:
                episode = run_selected_episode(domain, task_id, seed, retained, scores, args.env)
                record = {"run_id": run_id, "workload": domain, "task_id": task_id,
                          "seed": seed, "status": "completed", "episode": episode}
            except Exception as exc:
                record = {"run_id": run_id, "workload": domain, "task_id": task_id,
                          "seed": seed, "status": "failed",
                          "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            latest[run_id] = record
            rows.append(record)
            summary = summarize(list(latest.values()), protocol)
            summary["frozen_protocol_sha256"] = digest(frozen_path.read_bytes())
            summary["episodes_sha256"] = digest(episodes_path.read_bytes())
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"workload": domain, "task_id": task_id,
                              "status": record["status"],
                              "success": record.get("episode", {}).get("success"),
                              "completed": summary["completed_episodes"],
                              "planned": summary["planned_episodes"]}, ensure_ascii=False),
                  flush=True)
            if record["status"] != "completed":
                print("Stopping after first failed task to avoid repeating a rejected API request.", flush=True)
                break
    final = summarize(list(latest.values()), protocol)
    final["frozen_protocol_sha256"] = digest(frozen_path.read_bytes())
    final["episodes_sha256"] = digest(episodes_path.read_bytes()) if episodes_path.exists() else None
    summary_path.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
