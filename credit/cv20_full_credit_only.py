"""Run only Full CREDIT for the archived 20-family ToolSandbox protocol."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil


def run_fold(fold: int, output: Path, env_path: Path, upstream: Path,
             memory_pool: Path, tokenizer: Path, workers: int,
             test_seeds: tuple[int, ...] | None = None,
             allow_replay: bool = True,
             defer_final_answer: bool = False) -> dict:
    os.environ["CREDIT_CV_FOLD"] = str(fold)
    from credit import experiment_tables_public as exp
    from credit.remote_json_agent import RemoteJSONAgent as BaseRemoteJSONAgent
    from credit import toolsandbox_local_v3 as adapter
    from transformers import AutoTokenizer
    from credit.llm_remote_policy_v2 import load_config
    from urllib.parse import urlparse

    class Args:
        pass

    args = Args()
    args.output = output
    args.env = env_path
    args.upstream = upstream
    args.memory_pool = memory_pool
    args.tokenizer = tokenizer
    args.workers = workers

    config = load_config(env_path)
    exp.executor.AUTO_COMPLETE_AFTER_MUTATION = not defer_final_answer
    test_seeds = tuple(test_seeds or exp.TEST_SEEDS)
    if not test_seeds or not set(test_seeds) <= set(exp.TEST_SEEDS):
        raise ValueError(f"test seeds must be a non-empty subset of {exp.TEST_SEEDS}")
    replay_records = {}
    replay_sources = []
    for previous in sorted(output.parent.glob(f"fold{fold}_failed_attempt*")) if allow_replay else ():
        for generation_path in previous.glob("**/generations.jsonl"):
            manifest_path = generation_path.parent / "agent_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("model") != config["model"] or
                    manifest.get("base_url_sha256") != config["base_url_sha256"]):
                continue
            seed = int(manifest["actor_seed"])
            for line in generation_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("provider_model") == config["model"]:
                    replay_records.setdefault((seed, record["prompt_sha256"]),
                                              (record, str(previous.name)))
            replay_sources.append(str(previous.name))

    class ReplayRemoteJSONAgent(BaseRemoteJSONAgent):
        def __init__(self, *agent_args, **agent_kwargs):
            super().__init__(*agent_args, **agent_kwargs)
            self.reasoning_effort = "none" if self.config["model"] == "deepseek-flash" else None
            if self.reasoning_effort == "none":
                self.max_new_tokens = 512
                self.manifest["max_new_tokens"] = 512
            self.replayed_generation_count = 0
            self.live_generation_count = 0

        def _call(self, item):
            key = (self.actor_seed, item[0])
            if key in replay_records:
                prior, source = replay_records[key]
                self.replayed_generation_count += 1
                replay = deepcopy(prior)
                replay["replayed_from_previous_attempt"] = source
                return replay
            result = super()._call(item)
            self.live_generation_count += 1
            return result

        def close(self):
            super().close()
            manifest_path = self.output / "agent_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["live_api_calls"] = self.live_generation_count
            manifest["replayed_generation_records"] = self.replayed_generation_count
            manifest["replay_sources"] = replay_sources
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    exp.RemoteJSONAgent = ReplayRemoteJSONAgent
    exp.prepare(args)
    protocol_path = output / "frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["methods"] = ["Full CREDIT"]
    protocol["ablations"] = []
    protocol["execution_scope"] = "Full CREDIT main only; no baseline, budget, or ablation runs"
    protocol["test_seeds"] = list(test_seeds)
    protocol["single_seed_pilot"] = len(test_seeds) == 1
    protocol["allow_generation_replay"] = allow_replay
    protocol["auto_complete_after_mutation"] = not defer_final_answer
    protocol["defer_final_answer_to_model"] = defer_final_answer
    protocol["model_config"] = {
        "provider_model": config["model"],
        "endpoint_host": urlparse(config["url"]).hostname,
        "timeout_seconds": config["timeout"],
        "configured_max_output_tokens": config["max_tokens"],
        "effective_max_output_tokens": (512 if config["model"] == "deepseek-flash"
                                        else min(384, config["max_tokens"])),
        "temperature": 0.0,
        "reasoning_effort": ("none" if config["model"] == "deepseek-flash" else "provider default"),
        "top_p": "not sent; provider default",
        "output_cap_adjustment_reason": (
            "raised from .env cap 256 after finish_reason=length returned empty content"
            if config["model"] == "deepseek-flash" and config["max_tokens"] < 512 else None),
        "workers": workers,
        "api_key_configured": bool(config["api_key"]),
    }
    protocol["replay_sources"] = replay_sources
    protocol["available_replay_records"] = len(replay_records)
    exp.write(protocol_path, protocol)
    source_copy = output / "source" / "cv20_full_credit_only.py"
    shutil.copy2(Path(__file__).resolve(), source_copy)

    u, scenarios, dev, cert, test, pool, queries = exp.environment(args)
    pool_by_uid = {row["uid"]: row for row in pool}
    adapt_dir = output / "adaptation"
    adapt_agent = exp.make_agent(adapt_dir, exp.ADAPT_SEED, args)
    try:
        adapt_rows = exp.run_jobs(protocol["adaptation_plan"], adapt_agent, adapt_dir,
                                  scenarios, queries, pool_by_uid, u)
    finally:
        adapt_agent.close()
    exp.write(output / "adaptation_sessions.json", adapt_rows)

    adaptation, adaptation_calls, exhausted = exp.prefix(adapt_rows, exp.B0)
    all_tasks = sorted(set(test + cert))
    candidate, diagnostics = exp.fit_policy(adaptation, pool, all_tasks,
                                             scenarios, "Full CREDIT")
    semantic, _ = exp.fit_policy(adaptation, pool, all_tasks,
                                  scenarios, "Semantic")
    cert_specs = []
    for task in cert:
        for arm, policy in (("candidate", candidate), ("semantic", semantic)):
            cert_specs.append({
                "task": task, "bank": policy[task], "stage": "fresh_certification",
                "label": f"cert-{arm}-{len(cert_specs):02d}", "arm": arm,
            })
    cert_dir = output / "certification"
    cert_agent = exp.make_agent(cert_dir, exp.CERT_SEED, args)
    try:
        cert_rows = exp.run_jobs(cert_specs, cert_agent, cert_dir, scenarios,
                                 queries, pool_by_uid, u)
    finally:
        cert_agent.close()
    exp.write(output / "certification_sessions.json", cert_rows)
    cert_calls = sum(row["physical_calls"] for row in cert_rows)
    cert_by = {(row["task"], row["arm"]): row for row in cert_rows}
    family_delta = defaultdict(list)
    for task in cert:
        family_delta[exp.scenario_family(task)].append(
            cert_by[(task, "candidate")]["native_score"] -
            cert_by[(task, "semantic")]["native_score"])
    family_delta = {key: float(sum(values) / len(values))
                    for key, values in family_delta.items()}
    cert_pass = (not exhausted and adaptation_calls + cert_calls <= exp.B0 and
                 float(sum(family_delta.values()) / len(family_delta)) >= 0 and
                 all(value >= -0.05 for value in family_delta.values()))
    selected = candidate if cert_pass else semantic
    exp.write(output / "certification_decision.json", {
        "passed": cert_pass, "family_deltas": family_delta,
        "adaptation_calls": adaptation_calls, "certification_calls": cert_calls,
        "total_within_B0": adaptation_calls + cert_calls <= exp.B0,
        "fallback": None if cert_pass else "Semantic",
    })
    exp.write(output / "frozen_policy.json", {
        "method": "Full CREDIT", "certified": cert_pass,
        "selected_policy": selected, "diagnostics": diagnostics,
    })

    tokenizer_obj = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True)
    links, test_rows = [], []
    for seed in test_seeds:
        test_dir = output / f"test_seed_{seed}"
        test_agent = exp.make_agent(test_dir, seed, args)
        specs = []
        for task in test:
            exposure = exp.canonical_exposure(
                task, selected[task], scenarios, pool_by_uid, u, tokenizer_obj)
            label = f"credit-{task}"
            specs.append({"task": task, "bank": selected[task], "stage": "heldout_test",
                          "label": label, "canonical_exposure": list(exposure),
                          "method": "Full CREDIT"})
            links.append({"actor_seed": seed, "task": task, "method": "Full CREDIT",
                          "canonical_exposure": list(exposure)})
        try:
            test_rows.extend(exp.run_jobs(specs, test_agent, test_dir, scenarios,
                                          queries, pool_by_uid, u))
        finally:
            test_agent.close()
    exp.write(output / "test_physical_sessions.json", test_rows)
    exp.write(output / "test_logical_links.json", links)
    row_by_key = {(r["actor_seed"], r["task"]): r for r in test_rows}
    logical = [{**link, "native_score": row_by_key[(link["actor_seed"], link["task"])]["native_score"],
                "success": row_by_key[(link["actor_seed"], link["task"])]["success"],
                "online_calls": row_by_key[(link["actor_seed"], link["task"])]["physical_calls"]}
               for link in links]
    exp.write(output / "logical_results.json", logical)
    online_calls = sum(row["online_calls"] for row in logical)
    summary = {
        "method": "Full CREDIT", "fold": fold, "sessions": len(logical),
        "test_seeds": list(test_seeds),
        "defer_final_answer_to_model": defer_final_answer,
        "successes": sum(bool(row["success"]) for row in logical),
        "vsr": sum(bool(row["success"]) for row in logical) / len(logical),
        "native_score": sum(row["native_score"] for row in logical) / len(logical),
        "adaptation_calls": adaptation_calls, "certification_calls": cert_calls,
        "online_calls": online_calls,
        "total_calls": adaptation_calls + cert_calls + online_calls,
        "certification_passed": cert_pass,
        "fallback": None if cert_pass else "Semantic",
        "model_config": protocol["model_config"],
    }
    exp.write(output / "full_credit_summary.json", summary)
    audit = {
        "scope": protocol["execution_scope"],
        "checks": {
            "fold_is_0_or_1": fold in (0, 1),
            "logical_test_sessions_match_requested_seeds": len(logical) == len(test) * len(test_seeds),
            "requested_seed_coverage_per_task": all(
                sum(row["task"] == task for row in logical) == len(test_seeds)
                for task in test),
            "single_seed_if_requested": len(test_seeds) != 1 or
                all(len({row["actor_seed"] for row in logical if row["task"] == task}) == 1
                    for task in test),
            "all_native_evaluations_present": all(row["verification_coverage"] for row in test_rows),
            "no_test_tasks_in_adaptation": not set(test) & {row["task"] for row in adapt_rows},
        },
        "counts": {"adaptation_episodes": len(adapt_rows),
                   "certification_episodes": len(cert_rows),
                   "test_physical_episodes": len(test_rows),
                   "test_logical_sessions": len(logical)},
        "hashes": {},
    }
    for name in ("frozen_protocol.json", "adaptation_sessions.json",
                 "certification_sessions.json", "certification_decision.json",
                 "frozen_policy.json", "test_physical_sessions.json",
                 "test_logical_links.json", "logical_results.json",
                 "full_credit_summary.json"):
        audit["hashes"][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    if not all(audit["checks"].values()):
        raise AssertionError(audit["checks"])
    exp.write(output / "public_audit.json", audit)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--upstream", type=Path, default=Path("vendor/ToolSandbox"))
    parser.add_argument("--memory-pool", type=Path, default=Path(
        "results/toolsandbox_m_selector_attribution_confirmation_v1/memory_pool.json"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--test-seed", type=int, action="append",
                        help="Run only the specified held-out actor seed; may be repeated.")
    parser.add_argument("--disable-replay", action="store_true",
                        help="Require fresh API generations; do not reuse earlier attempts.")
    parser.add_argument("--defer-final-answer", action="store_true",
                        help="After required mutations, let the model produce the final answer.")
    args = parser.parse_args()
    from credit import toolsandbox_local_v3 as adapter
    tokenizer = args.tokenizer or adapter.MODEL
    output = args.output_root / f"fold{args.fold}"
    summary = run_fold(args.fold, output, args.env, args.upstream,
                       args.memory_pool, tokenizer, args.workers,
                       tuple(args.test_seed) if args.test_seed else None,
                       not args.disable_replay, args.defer_final_answer)
    print(json.dumps({"completed_fold": args.fold, "output": str(output),
                      "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
