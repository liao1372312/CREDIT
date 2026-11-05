"""Run one matched Semantic baseline on a single held-out ToolSandbox seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse


def run(source: Path, output: Path, env_path: Path, fold: int, seeds: tuple[int, ...],
        upstream: Path, tokenizer: Path | None, workers: int,
        defer_final_answer: bool) -> dict:
    os.environ["CREDIT_CV_FOLD"] = str(fold)
    from credit import experiment_tables_public as exp
    from credit.remote_json_agent import RemoteJSONAgent as BaseRemoteJSONAgent
    from credit import toolsandbox_local_v3 as adapter
    from transformers import AutoTokenizer
    from credit.llm_remote_policy_v2 import load_config

    config = load_config(env_path)
    protocol = json.loads((source / "frozen_protocol.json").read_text())
    source_manifest = json.loads(
        (source / f"test_seed_{exp.TEST_SEEDS[0]}" / "agent_manifest.json").read_text())
    if int(protocol["cross_validation"]["fold"]) != fold:
        raise ValueError("source protocol belongs to a different fold")
    if (source_manifest.get("model") != config["model"] or
            source_manifest.get("base_url_sha256") != config["base_url_sha256"]):
        raise ValueError("current API model/endpoint differs from the matched CREDIT run")
    if not seeds or not set(seeds) <= set(exp.TEST_SEEDS):
        raise ValueError(f"seeds must be a non-empty subset of {exp.TEST_SEEDS}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    output.mkdir(parents=True)
    for name in ("frozen_protocol.json", "memory_pool.json", "queries.json"):
        shutil.copy2(source / name, output / name)
    (output / "source").mkdir(exist_ok=True)
    for name in ("cv20_run_single_semantic_baseline.py", "experiment_tables_public.py",
                 "remote_json_agent.py", "toolsandbox_local_v3.py",
                 "toolsandbox_m_contract_fast_v3.py"):
        candidate = Path("credit") / name
        if candidate.exists():
            shutil.copy2(candidate, output / "source" / name)

    tokenizer = tokenizer or adapter.MODEL

    class BaselineRemoteAgent(BaseRemoteJSONAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reasoning_effort = "none" if self.config["model"] == "deepseek-flash" else None
            if self.reasoning_effort == "none":
                self.max_new_tokens = 512
                self.manifest["max_new_tokens"] = 512

        def close(self):
            super().close()
            manifest_path = self.output / "agent_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["live_api_calls"] = self.calls
            manifest["replayed_generation_records"] = 0
            manifest["replay_sources"] = []
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    exp.RemoteJSONAgent = BaselineRemoteAgent
    exp.executor.AUTO_COMPLETE_AFTER_MUTATION = not defer_final_answer
    protocol["methods"] = ["Semantic"]
    protocol["ablations"] = []
    protocol["execution_scope"] = "One Semantic baseline; fresh held-out actor seed only"
    protocol["test_seeds"] = list(seeds)
    protocol["defer_final_answer_to_model"] = defer_final_answer
    protocol["auto_complete_after_mutation"] = not defer_final_answer
    protocol["model_config"] = {
        "provider_model": config["model"],
        "endpoint_host": urlparse(config["url"]).hostname,
        "timeout_seconds": config["timeout"],
        "configured_max_output_tokens": config["max_tokens"],
        "effective_max_output_tokens": 512 if config["model"] == "deepseek-flash" else min(384, config["max_tokens"]),
        "temperature": 0.0,
        "reasoning_effort": "none" if config["model"] == "deepseek-flash" else "provider default",
        "top_p": "not sent; provider default",
        "workers": workers,
        "api_key_configured": bool(config["api_key"]),
    }
    exp.write(output / "frozen_protocol.json", protocol)

    class Args:
        pass

    args = Args()
    args.output = output
    args.env = env_path
    args.upstream = upstream
    args.memory_pool = output / "memory_pool.json"
    args.tokenizer = tokenizer
    args.workers = workers
    u, scenarios, _, _, test, pool, queries = exp.environment(args)
    if test != protocol["test_tasks"]:
        raise ValueError("frozen test task list no longer matches this fold")

    pool_by_uid = {row["uid"]: row for row in pool}
    # Semantic baseline: select the two most query-relevant eligible memories.
    policy = {
        task: list(exp.uids(exp.semantic_bank(pool, task, scenarios[task])))
        for task in test
    }
    exp.write(output / "semantic_policy.json", {
        "method": "Semantic", "selection_rule": "top-2 eligible memories by query relevance",
        "selected_policy": policy,
    })

    tokenizer_obj = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True)
    # Execute one worker/agent pool per seed. Persist each set under its matching seed folder.
    rows, links = [], []
    for seed in seeds:
        agent_dir = output / f"test_seed_{seed}"
        agent_dir.mkdir(parents=True, exist_ok=False)
        agent = BaselineRemoteAgent(agent_dir, seed, env=env_path, tokenizer_path=tokenizer,
                                    temperature=0.0, max_new_tokens=384, workers=workers,
                                    exact_prompt_cache=False)
        agent.manifest["benchmark"] = "CREDIT matched Semantic baseline"
        active_specs = [{"task": task, "bank": policy[task], "stage": "heldout_test",
                         "label": f"semantic-{task}",
                         "canonical_exposure": list(exp.canonical_exposure(
                             task, policy[task], scenarios, pool_by_uid, u, tokenizer_obj)),
                         "method": "Semantic"} for task in test]
        links.extend({"actor_seed": seed, "task": task, "method": "Semantic",
                      "canonical_exposure": spec["canonical_exposure"]}
                     for task, spec in zip(test, active_specs))
        try:
            rows.extend(exp.run_jobs(active_specs, agent, agent_dir, scenarios,
                                     queries, pool_by_uid, u))
        finally:
            agent.close()
    exp.write(output / "test_physical_sessions.json", rows)
    exp.write(output / "test_logical_links.json", links)
    by_key = {(row["actor_seed"], row["task"]): row for row in rows}
    logical = [{**link, "native_score": by_key[(link["actor_seed"], link["task"])] ["native_score"],
                "success": by_key[(link["actor_seed"], link["task"])] ["success"],
                "online_calls": by_key[(link["actor_seed"], link["task"])] ["physical_calls"]}
               for link in links]
    exp.write(output / "logical_results.json", logical)
    summary = {
        "method": "Semantic", "fold": fold, "sessions": len(logical),
        "test_seeds": list(seeds), "successes": sum(bool(row["success"]) for row in logical),
        "vsr": sum(bool(row["success"]) for row in logical) / len(logical),
        "native_score": sum(row["native_score"] for row in logical) / len(logical),
        "online_calls": sum(row["online_calls"] for row in logical),
        "adaptation_reused": False, "certification_reused": False,
        "defer_final_answer_to_model": defer_final_answer, "model_config": protocol["model_config"],
        "baseline_memory_rule": "top-2 eligible memories by query relevance",
        "matched_credit_comparison": str(source),
    }
    exp.write(output / "semantic_baseline_summary.json", summary)
    audit = {
        "scope": protocol["execution_scope"],
        "checks": {
            "logical_sessions_match_requested_seeds": len(logical) == len(test) * len(seeds),
            "all_native_evaluations_present": all(row["verification_coverage"] for row in rows),
            "requested_seeds_only": {row["actor_seed"] for row in logical} == set(seeds),
            "matched_model_and_endpoint": source_manifest.get("model") == config["model"],
        },
        "counts": {"test_physical_sessions": len(rows), "test_logical_sessions": len(logical)},
        "hashes": {},
    }
    for name in ("frozen_protocol.json", "semantic_policy.json", "test_physical_sessions.json",
                 "test_logical_links.json", "logical_results.json", "semantic_baseline_summary.json"):
        audit["hashes"][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    if not all(audit["checks"].values()):
        raise AssertionError(audit["checks"])
    exp.write(output / "public_audit.json", audit)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--defer-final-answer", action="store_true")
    parser.add_argument("--upstream", type=Path, default=Path("vendor/ToolSandbox"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with tempfile.NamedTemporaryFile(prefix="credit-env-", delete=False) as handle:
        snapshot = Path(handle.name)
    try:
        shutil.copyfile(args.env, snapshot)
        snapshot.chmod(0o600)
        run(args.source, args.output, snapshot, args.fold, tuple(args.seeds),
            args.upstream, args.tokenizer, args.workers, args.defer_final_answer)
    finally:
        snapshot.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
