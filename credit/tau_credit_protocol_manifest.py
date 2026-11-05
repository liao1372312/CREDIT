"""Write a secret-free, verifiable protocol manifest for the 64-record tau pool."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .llm_remote_policy_v2 import load_config
from .tau_credit_runner import POOL_ROOT, ROOT


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    config = load_config(ROOT / ".env")
    factorial_summary_path = ROOT / "results/tau_credit_balanced_factorial_dev_v1/summary.json"
    factorial_audit_path = ROOT / "results/tau_credit_balanced_factorial_dev_v1/audit.json"
    factorial_summary = (json.loads(factorial_summary_path.read_text())
                         if factorial_summary_path.exists() else {})
    factorial_audit = (json.loads(factorial_audit_path.read_text())
                       if factorial_audit_path.exists() else {})
    episode_ledger = ROOT / "results/tau_credit_balanced_factorial_dev_v1/episodes.jsonl"
    observed_models = sorted({
        model
        for line in episode_ledger.read_text().splitlines() if line.strip()
        for model in json.loads(line).get("episode", {}).get("provider_models", [])
    }) if episode_ledger.exists() else factorial_summary.get("provider_models", [])
    source = json.loads((POOL_ROOT / "split_manifest.json").read_text())
    workloads = {}
    for domain in ("retail", "airline"):
        pool_path = POOL_ROOT / f"{domain}_memory_pool_64.json"
        pool = json.loads(pool_path.read_text())
        token_lengths = [int(row["tokens"]) for row in pool]
        if len(pool) != 64 or any(row["split"] != "development" for row in pool):
            raise ValueError(f"invalid development pool for {domain}")
        splits = source["split_task_ids"][domain]
        sets = {key: set(values) for key, values in splits.items()}
        if sets["development"] & sets["certification"] or sets["development"] & sets["test"] or sets["certification"] & sets["test"]:
            raise ValueError(f"task ID leakage across splits for {domain}")
        workloads[domain] = {
            "source_pool_sha256": sha256(pool_path.read_bytes()),
            "source_records": len(pool),
            "source_task_ids": sorted({row["source_task_id"] for row in pool}),
            "distinct_source_task_families": len({row["source_family_id"] for row in pool}),
            "memory_text_token_total_L": sum(token_lengths),
            "primary_capacity_C_floor_half_L": sum(token_lengths) // 2,
            "token_length_min_median_max": [
                min(token_lengths), sorted(token_lengths)[len(token_lengths) // 2], max(token_lengths)
            ],
            "task_id_counts": {key: len(values) for key, values in sets.items()},
            "task_ids_by_role": splits,
            "capacity_grid_tokens": {
                f"{ratio}%": (sum(token_lengths) * ratio) // 100
                for ratio in (25, 50, 75, 100)
            },
        }
    files = [
        "credit/tau_credit_runner.py",
        "credit/tau_historical_credit_pool.py",
        "credit/tau_credit_protocol_manifest.py",
    ]
    qwen_snapshot = Path(
        "/home/lxc/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
        "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    )
    local_smoke_path = ROOT / "results/tau_credit_smoke_local_qwen_retail_v3/smoke.json"
    local_smoke = None
    if local_smoke_path.exists():
        smoke = json.loads(local_smoke_path.read_text())
        episode = smoke["episode"]
        local_smoke = {
            "scope": "one no-memory runner smoke only; excluded from CREDIT and benchmark effectiveness estimates",
            "workload": episode["workload"], "task_id": episode["task_id"],
            "seed": episode["seed"], "reward": episode["reward"],
            "success": episode["success"], "actor_generations": episode["actor_generations"],
            "user_simulator_generations": episode["user_simulator_generations"],
            "tool_calls": episode["tool_calls"], "prompt_tokens": episode["prompt_tokens"],
            "completion_tokens": episode["completion_tokens"],
            "duration_seconds": episode["duration_seconds"],
            "episode_artifact_sha256": sha256(local_smoke_path.read_bytes()),
        }
    result = {
        "protocol": "tau-original-CREDIT-64-v1",
        "status": "development randomized factorial complete (436/436); fresh certification and final-test CREDIT collection remain pending",
        "benchmark_repository": source["source_repository"],
        "benchmark_commit": source["source_commit"],
        "history_source_sha256": source["workloads"],
        "split_rule": source["split_rule"],
        "task_family_key": "tau-bench task ID; task-level split is disjoint",
        "pool_source": "successful historical trajectories; memory texts retain tool names and argument keys only",
        "pool_tokenizer": source["tokenizer"],
        "capacity_rule": source["capacity_rule"],
        "max_exposed_records": 2,
        "actor_and_user_requested_model": observed_models or "not recorded",
        "actor_temperature": 0,
        "actor_top_p": "unset; provider default",
        "user_simulator": "same configured OpenAI-compatible endpoint and requested model; separate client and usage accounting",
        "local_second_actor": {
            "checkpoint": str(qwen_snapshot),
            "snapshot_id": qwen_snapshot.name,
            "config_sha256": sha256((qwen_snapshot / "config.json").read_bytes()),
            "tokenizer_config_sha256": sha256((qwen_snapshot / "tokenizer_config.json").read_bytes()),
            "generation_config_sha256": sha256((qwen_snapshot / "generation_config.json").read_bytes()),
            "decoding": "greedy; do_sample=false; temperature/top_p/top_k unset",
        },
        "endpoint_sha256": "8350a46cba5dae5daf04685d754664810f0dc8753cbaa51d27b359cdf06a5c37",
        "current_endpoint_sha256": config["base_url_sha256"],
        "historical_factorial_endpoint_sha256": "8350a46cba5dae5daf04685d754664810f0dc8753cbaa51d27b359cdf06a5c37",
        "current_configured_model": config["model"],
        "credential_value_recorded": False,
        "task_max_actor_steps": 30,
        "historical_collection_issue": "An earlier live user-simulator request returned HTTP 403 for insufficient token quota. The later development factorial completed; this historical failure is not evidence that the current endpoint is unavailable.",
        "workloads": workloads,
        "local_qwen_smoke": local_smoke,
        "source_code_sha256": {
            name: (factorial_audit.get("runner_sha256")
                   if name == "credit/tau_credit_runner.py" and factorial_audit
                   else sha256((ROOT / name).read_bytes()))
            for name in files
        },
        "claims_not_supported_yet": [
            "confirmatory CREDIT VSR on the new 64-record protocol",
            "capacity sensitivity VSR for the 64-record pool",
            "fresh task-level certification outcomes and adaptation cost",
        ],
        "historical_evidence_limitation": "Upstream successful trajectories establish demonstrated source workflows only; they do not identify causal memory effects or CREDIT performance.",
    }
    output = ROOT / "results/tau_credit_protocol_v1"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol_manifest.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(path), "workloads": {
        key: {k: value[k] for k in ("source_records", "memory_text_token_total_L", "primary_capacity_C_floor_half_L", "task_id_counts")}
        for key, value in workloads.items()
    }, "status": result["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
