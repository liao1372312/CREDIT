"""Evaluate one fresh held-out seed with an already certified Full CREDIT bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse


def run(source: Path, output: Path, env_path: Path, fold: int, seed: int,
        upstream: Path, tokenizer: Path | None, workers: int) -> dict:
    os.environ["CREDIT_CV_FOLD"] = str(fold)
    from credit import experiment_tables_public as exp
    from credit.remote_json_agent import RemoteJSONAgent as BaseRemoteJSONAgent
    from credit import toolsandbox_local_v3 as adapter
    from transformers import AutoTokenizer
    from credit.llm_remote_policy_v2 import load_config

    config = load_config(env_path)
    source_protocol_path = source / "frozen_protocol.json"
    source_policy_path = source / "frozen_policy.json"
    source_manifest_path = source / f"test_seed_{exp.TEST_SEEDS[0]}" / "agent_manifest.json"
    source_protocol = json.loads(source_protocol_path.read_text())
    source_policy = json.loads(source_policy_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    if int(source_protocol["cross_validation"]["fold"]) != fold:
        raise ValueError("source policy belongs to a different fold")
    if not source_policy.get("certified"):
        raise ValueError("source Full CREDIT policy did not pass certification")
    if (source_manifest.get("model") != config["model"] or
            source_manifest.get("base_url_sha256") != config["base_url_sha256"]):
        raise ValueError("current API model/endpoint differs from the certified policy run")
    if seed not in exp.TEST_SEEDS:
        raise ValueError(f"seed must be one of {exp.TEST_SEEDS}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    output.mkdir(parents=True)
    for name in ("frozen_protocol.json", "frozen_policy.json", "memory_pool.json", "queries.json"):
        shutil.copy2(source / name, output / name)
    if (source / "source").exists():
        shutil.copytree(source / "source", output / "source", dirs_exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), output / "source" / Path(__file__).name)

    tokenizer = tokenizer or adapter.MODEL

    class RepeatRemoteAgent(BaseRemoteJSONAgent):
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

    exp.RemoteJSONAgent = RepeatRemoteAgent
    exp.executor.AUTO_COMPLETE_AFTER_MUTATION = False

    protocol = json.loads((output / "frozen_protocol.json").read_text())
    protocol["methods"] = ["Full CREDIT"]
    protocol["ablations"] = []
    protocol["execution_scope"] = "Full CREDIT; fresh held-out actor seed only; certified adaptation policy reused"
    protocol["test_seeds"] = [seed]
    protocol["defer_final_answer_to_model"] = True
    protocol["auto_complete_after_mutation"] = False
    protocol["reused_certified_policy_from"] = str(source)
    protocol["reused_certified_policy_sha256"] = hashlib.sha256(source_policy_path.read_bytes()).hexdigest()
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
    selected = source_policy["selected_policy"]
    tokenizer_obj = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True)
    agent_dir = output / f"test_seed_{seed}"
    agent_dir.mkdir(parents=True, exist_ok=False)
    agent = RepeatRemoteAgent(agent_dir, seed, env=env_path, tokenizer_path=tokenizer,
                             temperature=0.0, max_new_tokens=384, workers=workers,
                             exact_prompt_cache=False)
    agent.manifest["benchmark"] = "CREDIT frozen-policy single-seed repeat"
    links, specs = [], []
    for task in test:
        exposure = exp.canonical_exposure(
            task, selected[task], scenarios, pool_by_uid, u, tokenizer_obj)
        specs.append({"task": task, "bank": selected[task], "stage": "heldout_test",
                      "label": f"credit-{task}", "canonical_exposure": list(exposure),
                      "method": "Full CREDIT"})
        links.append({"actor_seed": seed, "task": task, "method": "Full CREDIT",
                      "canonical_exposure": list(exposure)})
    try:
        rows = exp.run_jobs(specs, agent, agent_dir, scenarios, queries, pool_by_uid, u)
    finally:
        agent.close()
    exp.write(output / "test_physical_sessions.json", rows)
    exp.write(output / "test_logical_links.json", links)
    by_task = {row["task"]: row for row in rows}
    logical = [{**link, "native_score": by_task[link["task"]]["native_score"],
                "success": by_task[link["task"]]["success"],
                "online_calls": by_task[link["task"]]["physical_calls"]}
               for link in links]
    exp.write(output / "logical_results.json", logical)
    summary = {
        "method": "Full CREDIT", "fold": fold, "sessions": len(logical),
        "test_seeds": [seed], "successes": sum(bool(row["success"]) for row in logical),
        "vsr": sum(bool(row["success"]) for row in logical) / len(logical),
        "native_score": sum(row["native_score"] for row in logical) / len(logical),
        "online_calls": sum(row["online_calls"] for row in logical),
        "adaptation_reused": True, "certification_reused": True,
        "certification_passed": True, "defer_final_answer_to_model": True,
        "model_config": protocol["model_config"],
        "policy_source": str(source),
        "policy_source_sha256": protocol["reused_certified_policy_sha256"],
    }
    exp.write(output / "full_credit_summary.json", summary)
    audit = {
        "scope": protocol["execution_scope"],
        "checks": {
            "policy_source_certified": bool(source_policy.get("certified")),
            "model_matches_policy_source": source_manifest.get("model") == config["model"],
            "logical_sessions_match_requested_seed": len(logical) == len(test),
            "all_native_evaluations_present": all(row["verification_coverage"] for row in rows),
            "requested_seed_only": {row["actor_seed"] for row in logical} == {seed},
        },
        "counts": {"test_physical_sessions": len(rows), "test_logical_sessions": len(logical)},
        "hashes": {},
    }
    for name in ("frozen_protocol.json", "frozen_policy.json", "test_physical_sessions.json",
                 "test_logical_links.json", "logical_results.json", "full_credit_summary.json"):
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("vendor/ToolSandbox"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    # Freeze credentials/config for the full run without persisting them in results.
    with tempfile.NamedTemporaryFile(prefix="credit-env-", delete=False) as handle:
        snapshot = Path(handle.name)
    try:
        shutil.copyfile(args.env, snapshot)
        snapshot.chmod(0o600)
        run(args.source, args.output, snapshot, args.fold, args.seed,
            args.upstream, args.tokenizer, args.workers)
    finally:
        snapshot.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
