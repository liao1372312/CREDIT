"""Disclosed online ablation for latency-aware request-derived contracts.

This runner reuses the prior Contract confirmation suite as development data.
It records all remote attempts and local selection, validation, repair, and
tool-execution time in a common per-session latency total.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np

from . import toolsandbox_local_v3 as adapter
from .remote_json_agent import RemoteJSONAgent
from .toolsandbox_m_compose_selector import compose_select
from .toolsandbox_m_contract_confirmation import (
    CLOCK,
    build_confirmation_scenarios,
    requested_state_checks,
    semantic_answer_success,
)
from .toolsandbox_m_contract_fast import (
    consume_fast,
    consume_validate,
    fast_contract_select,
    fast_select,
    matched_recipe_select,
    request_required_mutations,
)
from .toolsandbox_m_contract_selector import contract_select
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_intent_optimization import balanced_order, select as prior_select
from .user_data import digest


METHODS = (
    "Matched-Recipe",
    "Contract-Text",
    "Contract-Validate",
    "Contract-FastRepair",
    "Contract-Adaptive",
    "Semantic",
    "CREDIT-Compose",
)
PRIMARY_BUDGET_MS = 15000
MAX_MODEL_TURNS = 12
ACTOR_SEEDS = (1401, 1403)


def select(method, pool, task, query, scenario, pair_vetoes, tokenizer):
    if method == "Matched-Recipe":
        return matched_recipe_select(pool, query, scenario, tokenizer)
    if method in {"Contract-Text", "Contract-Validate", "Contract-FastRepair"}:
        return fast_contract_select(pool, query, scenario, tokenizer)
    if method == "Contract-Adaptive":
        return fast_select(pool, query, scenario, tokenizer, adaptive=True)
    if method == "CREDIT-Compose":
        return compose_select(pool, query, scenario)
    return prior_select(method, pool, task, query, scenario, pair_vetoes)


def _consume(job, record, u, max_calls, query):
    began = perf_counter_ns()
    if job["method"] == "Contract-Validate":
        consume_validate(job, record, u, max_calls, query)
    elif job["method"] in {"Contract-FastRepair", "Contract-Adaptive"}:
        consume_fast(job, record, u, max_calls, query)
    else:
        adapter.consume(job, record, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6


def finish_direct(job, directory, u):
    """Persist the physical state without exponential milestone graph matching."""
    context = job["context"]
    value = context.to_dict(serialize_console=False)
    row = {key: job[key] for key in (
        "task", "family", "stage", "method", "bank", "exposure", "calls",
        "generations", "initial_state_sha256", "termination")}
    row.update({
        "tool_calls": len(job["calls"]),
        "latency_seconds": perf_counter() - job["began"],
        "ending_context": value,
        "ending_state_sha256": adapter.fingerprint(value),
        "allowed_tools": list(job["tools"]),
        "upstream_evaluation": None,
        "upstream_graph_evaluation_run": False,
    })
    with (directory / "episodes.jsonl").open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def annotate_direct(row, metadata):
    spec = metadata[row["task"]]
    checks = requested_state_checks(row, spec)
    required = request_required_mutations(
        spec["public_request"], set(row["allowed_tools"]))
    successful = {call["tool"] for call in row["calls"] if not call.get("exception")}
    row.update({
        "requested_state_score": float(np.mean(checks)) if checks else 0.0,
        "requested_state_success": int(all(value >= 1 - 1e-8 for value in checks)),
        "process_state_score": (len(required & successful) / len(required)
                                if required else 1.0),
        "process_state_success": int(required <= successful),
        "semantic_answer_success": semantic_answer_success(row, spec["family"]),
        "family": spec["family"],
        "exact_state_predicates_registered": True,
        "minefield_checked_by_upstream_graph": False,
    })
    return row


def run_batch(jobs, queries, agent, directory, u, *, max_calls=12):
    rows, active = [], jobs
    for _ in range(MAX_MODEL_TURNS):
        if not active:
            break
        records = agent.infer([job["chat"] for job in active])
        remaining = []
        for job, record in zip(active, records):
            _consume(job, record, u, max_calls, queries[job["task"]])
            if job["done"]:
                rows.append(finish_direct(job, directory, u))
            else:
                remaining.append(job)
        active = remaining
    for job in active:
        job["termination"] = "turn_budget"
        rows.append(finish_direct(job, directory, u))
    return rows


def _latency(row):
    remote = session_latency(row)
    local = float(row.get("selection_latency_ms", 0.0)) + sum(
        float(turn.get("local_consume_latency_ms", 0.0))
        for turn in row["generations"])
    return remote, local, remote + local


def run_actor(seed, actor_index, trajectories, root, args, scenarios, metadata,
              pool, pair_vetoes, u):
    directory = root / f"fast_development_seed_{seed}"
    directory.mkdir()
    agent = RemoteJSONAgent(directory, seed, env=args.env,
        tokenizer_path=args.tokenizer, temperature=0.0, max_new_tokens=384,
        workers=args.workers, exact_prompt_cache=False)
    agent.manifest["benchmark"] = "ToolSandbox-M fast-contract disclosed development"
    sessions, positions = [], defaultdict(lambda: [0] * len(METHODS))
    for trajectory_index, trajectory in enumerate(trajectories):
        for session_index, task in enumerate(trajectory["sessions"]):
            scenario, query = scenarios[task], metadata[task]["public_request"]
            order = balanced_order(list(METHODS), actor_index,
                                   trajectory_index, session_index)
            jobs, info = [], {}
            for position, method in enumerate(order):
                positions[method][position] += 1
                began = perf_counter_ns()
                bank = select(method, pool, task, query, scenario,
                              pair_vetoes, agent.tokenizer)
                selection_ms = (perf_counter_ns() - began) / 1e6
                job = initial_job_v2(task, scenario, bank, u, agent.tokenizer,
                                     "fast_contract_development", method)
                job["selection_latency_ms"] = selection_ms
                jobs.append(job)
                info[method] = (bank, position, selection_ms)
            for row in run_batch(jobs, {task: query}, agent, directory, u,
                                 max_calls=12):
                bank, position, selection_ms = info[row["method"]]
                row["selection_latency_ms"] = selection_ms
                remote, local, accounted = _latency(row)
                item = annotate_direct({**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"],
                    "session_index": session_index,
                    "selected_bank": [note["uid"] for note in bank],
                    "remote_latency_ms": remote,
                    "local_runtime_ms": local,
                    "accounted_latency_ms": accounted,
                    "batch_position": position,
                    "public_request_sha256": sha256(query.encode()).hexdigest()}, metadata)
                sessions.append(item)
                with (directory / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    agent.close()
    summary = {"actor_seed": seed, "sessions": len(sessions),
        "remote_requests": agent.calls,
        "method_batch_position_counts": dict(positions)}
    adapter.write(directory, "summary.json", summary)
    return sessions, summary


def summarize(rows):
    output = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        latency = [row["accounted_latency_ms"] for row in subset]
        turns = [turn for row in subset for turn in row["generations"]]
        repairs = [turn for turn in turns if "deterministic_repair" in turn]
        output.append({
            "method": method,
            "sessions": len(subset),
            "requested_state_VSR": float(np.mean([
                row["requested_state_success"] for row in subset])),
            f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms": float(np.mean([
                row["requested_state_success"] and value <= PRIMARY_BUDGET_MS
                for row, value in zip(subset, latency)])),
            f"completion_at_{PRIMARY_BUDGET_MS}ms": float(np.mean([
                value <= PRIMARY_BUDGET_MS for value in latency])),
            "process_state_VSR": float(np.mean([
                row["process_state_success"] for row in subset])),
            "mean_remote_requests": float(np.mean([
                sum("remote_latency_ms" in turn for turn in row["generations"])
                for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "mean_remote_latency_ms": float(np.mean([
                row["remote_latency_ms"] for row in subset])),
            "mean_local_runtime_ms": float(np.mean([
                row["local_runtime_ms"] for row in subset])),
            "mean_accounted_latency_ms": float(np.mean(latency)),
            "p95_accounted_latency_ms": float(np.quantile(latency, .95)),
            "validator_rejections": sum("contract_violation" in turn for turn in turns),
            "deterministic_repairs": len(repairs),
            "local_continuations": sum(
                bool(turn.get("local_continuation")) for turn in turns),
            "local_completions": sum(
                row["termination"] == "local_contract_complete" for row in subset),
            "repaired_tool_exceptions": sum(
                bool(turn.get("repair_execution_exception")) for turn in repairs),
            "premature_final_rejections": sum(
                "contract_violation" in turn and "answer" in (turn.get("action") or {})
                for turn in turns),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
        default=Path("results/toolsandbox_m_contract_confirmation_v1"))
    parser.add_argument("--output", type=Path,
        default=Path("results/toolsandbox_m_contract_fast_development_v1"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ACTOR_SEEDS))
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)

    protocol = json.loads((args.source / "frozen_protocol.json").read_text())
    u = adapter.upstream_imports(Path(protocol["upstream_directory"]))
    adapter.install_local_clock(CLOCK)
    scenarios, metadata, trajectories = build_confirmation_scenarios(u)
    if args.smoke:
        args.seeds = args.seeds[:1]
        trajectories = trajectories[:1]
        keep = {task for trajectory in trajectories for task in trajectory["sessions"]}
        scenarios = {name: value for name, value in scenarios.items() if name in keep}
        metadata = {name: value for name, value in metadata.items() if name in keep}
    manifest = next(args.source.glob("contract_confirmation_seed_*/agent_manifest.json"))
    args.tokenizer = args.tokenizer or Path(json.loads(manifest.read_text())["tokenizer_path"])
    pool = json.loads((args.source / "memory_pool.json").read_text())
    pair_vetoes = {tuple(pair) for pair in protocol["pair_vetoes"]}
    frozen = {
        "suite": "disclosed fast-contract development ablation",
        "development_source": str(args.source.resolve()),
        "development_source_sha256": digest(args.source / "confirmation_sessions.jsonl"),
        "methods": list(METHODS), "actor_seeds": list(args.seeds),
        "trajectories": trajectories,
        "latency_accounting": "sum of every remote attempt plus measured local selection, validation, repair, and tool execution wall time",
        "development_endpoint": "registered exact predicates over the final physical-world snapshot; upstream milestone graph matching omitted from development selection",
        "primary_budget_ms": PRIMARY_BUDGET_MS,
        "maximum_tool_calls_per_session": 12,
        "maximum_model_turns_per_session": MAX_MODEL_TURNS,
        "exact_prompt_cache": False,
        "temperature": 0.0,
        "source_sha256": {
            name: digest(Path(__file__).with_name(name)) for name in (
                "toolsandbox_m_contract_fast_development.py",
                "toolsandbox_m_contract_fast.py",
                "toolsandbox_m_contract_selector.py",
                "toolsandbox_m_contract_confirmation.py",
                "toolsandbox_local_v3.py", "remote_json_agent.py")},
        "credentials_persisted": False,
    }
    adapter.write(args.output, "protocol.json", frozen)

    rows, actors = [], []
    for index, seed in enumerate(args.seeds):
        found, actor = run_actor(seed, index, trajectories, args.output, args,
                                 scenarios, metadata, pool, pair_vetoes, u)
        rows.extend(found); actors.append(actor)
    with (args.output / "sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = summarize(rows)
    summary = {"table": table, "actor_summaries": actors,
        "ranking_at_15s": sorted(table, key=lambda row: (
            -row[f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms"],
            -row["requested_state_VSR"], row["mean_accounted_latency_ms"], row["method"]))}
    adapter.write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    adapter.write(args.output, "audit.json", {
        "sessions": len(rows),
        "expected_sessions": len(args.seeds) * len(metadata) * len(METHODS),
        "credentials_persisted": False,
        "live_remote_requests": sum(actor["remote_requests"] for actor in actors),
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
