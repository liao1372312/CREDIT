"""Build leakage-controlled CREDIT memory pools from official tau-bench histories.

Only successful public historical trajectories from the pinned original
tau-bench checkout are used. Memory text contains tool names and argument
keys only; user, reservation, order, and product values are never copied.
Task IDs are the split groups so no exact benchmark task appears in more than
one of development, certification, and test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from . import toolsandbox_local_v3 as toolsandbox_adapter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/tau-bench/historical_trajectories"
SOURCE_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
EXPECTED_RECORDS = 64


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def family_split(domain: str, task_id: int) -> str:
    # Fixed before extraction; hold out whole task IDs across all source agents
    # and trials. The task ID is the operational family key for this benchmark.
    value = int(sha256(f"tau-original-v1|{domain}|{task_id}".encode())[:8], 16) % 100
    if value < 60:
        return "development"
    if value < 75:
        return "certification"
    return "test"


def trajectory_calls(row: dict[str, Any]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    calls = []
    for message in row.get("traj", []):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls", []) or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            raw_args = fn.get("arguments") or {}
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    continue
            else:
                args = raw_args
            if not isinstance(args, dict):
                continue
            calls.append((name, tuple(sorted(str(k) for k in args))))
    return tuple(calls)


def memory_text(domain: str, calls: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    chain = " -> ".join(
        name + ("(" + ",".join(keys) + ")" if keys else "()")
        for name, keys in calls
    )
    return (
        f"Verified {domain} workflow: {chain}. Resolve identifiers from current "
        "tool results, check policy and current state before mutations, and "
        "verify the requested end state. Never reuse historical literal values."
    )


def load_candidates(domain: str) -> tuple[list[dict], dict[str, Any]]:
    paths = sorted(SOURCE.glob(f"*-{domain}.json"))
    if len(paths) < 2:
        raise FileNotFoundError(f"expected official historical trajectories for {domain}: {SOURCE}")
    # Keep one representative for each abstract action-chain signature.
    unique: dict[tuple, dict] = {}
    total = successes = 0
    source_hashes = {}
    for path in paths:
        raw = path.read_bytes()
        source_hashes[path.name] = sha256(raw)
        trajectories = json.loads(raw)
        model_source = path.name.removesuffix(f"-{domain}.json")
        for row in trajectories:
            total += 1
            if float(row.get("reward", 0.0)) < 1.0 - 1e-9:
                continue
            calls = trajectory_calls(row)
            if not calls:
                continue
            successes += 1
            task_id = int(row["task_id"])
            split = family_split(domain, task_id)
            if split != "development":
                continue
            text = memory_text(domain, calls)
            # A chain is one unique behavioral record even if repeated by
            # another source model/trial. Keep deterministic earliest source.
            signature = tuple(calls)
            source_id = f"{model_source}:task{task_id}:trial{int(row.get('trial', 0))}"
            item = {
                "uid": sha256((domain + "|" + source_id + "|" + repr(signature)).encode())[:16],
                "text": text,
                "tools": sorted({name for name, _ in calls}),
                "tool_chain": [{"name": name, "argument_keys": list(keys)} for name, keys in calls],
                "tokens": None,
                "source_workload": domain,
                "source_task_id": task_id,
                "source_family_id": f"{domain}:task-{task_id}",
                "source_episode_id": source_id,
                "source_file": path.name,
                "source_model_family": model_source,
                "source_trial": int(row.get("trial", 0)),
                "source_reward": float(row["reward"]),
                "source_success_verified": True,
                "source_commit": SOURCE_COMMIT,
                "split": split,
                "abstract_chain_sha256": sha256(repr(signature).encode()),
            }
            previous = unique.get(signature)
            if previous is None or (source_id, item["uid"]) < (previous["source_episode_id"], previous["uid"]):
                unique[signature] = item
    candidates = sorted(unique.values(), key=lambda x: (
        x["source_task_id"], x["source_model_family"], x["source_trial"], x["uid"]))
    audit = {
        "source_trajectory_count": total,
        "successful_trajectory_count": successes,
        "unique_development_action_chains": len(candidates),
        "source_file_sha256": source_hashes,
    }
    return candidates, audit


def balanced_select(candidates: list[dict], count: int) -> list[dict]:
    by_task: dict[int, list[dict]] = defaultdict(list)
    for item in candidates:
        by_task[item["source_task_id"]].append(item)
    # Round-robin across development tasks before adding additional workflows
    # from a task. The candidate chains themselves are unique.
    selected = []
    task_ids = sorted(by_task)
    depth = 0
    while len(selected) < count:
        added = False
        for task_id in task_ids:
            if depth < len(by_task[task_id]):
                selected.append(by_task[task_id][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != count:
        raise ValueError(f"need {count} unique successful development chains; found {len(selected)}")
    return selected


def build(output: Path, tokenizer_path: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    workloads = {}
    for domain in ("retail", "airline"):
        candidates, audit = load_candidates(domain)
        selected = balanced_select(candidates, EXPECTED_RECORDS)
        for item in selected:
            item["tokens"] = len(tokenizer.encode(item["text"], add_special_tokens=False))
        total_tokens = sum(item["tokens"] for item in selected)
        workloads[domain] = {
            "pool": selected,
            "audit": audit,
            "token_total": total_tokens,
            "primary_capacity_tokens": total_tokens // 2,
            "candidate_count": len(candidates),
        }
        (output / f"{domain}_memory_pool_64.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    split = {}
    for domain in ("retail", "airline"):
        ids = sorted({int(row["task_id"]) for path in SOURCE.glob(f"*-{domain}.json")
                      for row in json.loads(path.read_text())})
        split[domain] = {key: [task_id for task_id in ids if family_split(domain, task_id) == key]
                         for key in ("development", "certification", "test")}
    manifest = {
        "protocol": "tau-original-historical-source-pool-v1",
        "source_repository": "https://github.com/sierra-research/tau-bench",
        "source_commit": SOURCE_COMMIT,
        "source_path": str(SOURCE.resolve()),
        "task_family_definition": "benchmark task_id; all historical agents/trials for an ID stay in one split",
        "split_rule": "sha256(tau-original-v1|domain|task_id) mod 100: 0-59 development, 60-74 certification, 75-99 test",
        "memory_records_per_workload": EXPECTED_RECORDS,
        "tokenizer": str(tokenizer_path.resolve()),
        "capacity_rule": "floor(0.5 * sum of the 64 selected memory-text token counts)",
        "private_values_policy": "memory text stores tool names and argument keys only; no tool argument values or conversation text",
        "split_task_ids": split,
        "workloads": {domain: {"audit": item["audit"],
                              "unique_candidate_records": item["candidate_count"],
                              "selected_records": len(item["pool"]),
                              "selected_text_tokens": item["token_total"],
                              "primary_capacity_tokens": item["primary_capacity_tokens"],
                              "selected_task_ids": sorted({r['source_task_id'] for r in item['pool']})}
                      for domain, item in workloads.items()},
        "limitation": "Official historical trajectories provide demonstrated source outcomes, not randomized causal effects; selector intervention and fresh certification data are still required.",
    }
    manifest["sha256"] = {
        path.name: sha256(path.read_bytes()) for path in sorted(output.glob("*_memory_pool_64.json"))
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=toolsandbox_adapter.MODEL)
    args = parser.parse_args()
    build(args.output, args.tokenizer)


if __name__ == "__main__":
    main()
