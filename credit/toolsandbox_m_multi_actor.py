"""Multi-actor ToolSandbox-M with measured remote-model network latency."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil

import dill
import numpy as np
from scipy.stats import binomtest

from .remote_json_agent import RemoteJSONAgent
from .toolsandbox_local_v3 import (MODEL, UPSTREAM, initial_job, install_local_clock,
    local_scenarios, memory_from, run_batch, upstream_imports, write)
from .toolsandbox_m import METHODS, continuous_score, make_trajectories, select_bank
from .user_data import digest


LATENCY_BUDGETS_MS = (2500, 5000, 10000, 20000)


def session_latency(row: dict) -> float:
    return float(sum(turn.get("remote_latency_ms", 0.0) for turn in row["generations"]))


def run_actor(seed: int, root: Path, env: Path, tokenizer_path: Path, workers: int,
              scenarios: dict, history: list[str], trajectories: list[dict], u,
              source_hashes: dict[str, str], method_order: list[str]) -> tuple[list[dict], dict]:
    output = root / f"actor_seed_{seed}"
    output.mkdir()
    agent = RemoteJSONAgent(output, seed, env=env, tokenizer_path=tokenizer_path,
                            temperature=0.2, max_new_tokens=384, workers=workers,
                            exact_prompt_cache=False)
    agent.binding_interface = "tool_json"
    agent.manifest.update({"benchmark": "ToolSandbox-M multi-actor remote-latency suite"})
    history_rows = []
    for offset in range(0, len(history), 16):
        history_rows += run_batch([initial_job(name, scenarios[name], [], u, agent.tokenizer,
                                               "history", "NoMem")
                                   for name in history[offset:offset + 16]], agent, output, u)
    initial_pool = [note for row in sorted(history_rows, key=lambda row: history.index(row["task"]))
                    if (note := memory_from(row, agent.tokenizer)) is not None]
    write(output, "initial_memories.json", initial_pool)

    controls = {row["task"]: row for row in history_rows}
    calibration_rows = run_batch([
        initial_job(note["source_task"], scenarios[note["source_task"]], [note], u,
                    agent.tokenizer, "calibration", "single_memory_probe")
        for note in initial_pool], agent, output, u) if initial_pool else []
    calibration_by_task = {row["task"]: row for row in calibration_rows}
    causal_scores = {
        note["uid"]: continuous_score(calibration_by_task[note["source_task"]])
                     - continuous_score(controls[note["source_task"]])
        for note in initial_pool}
    calibration = [{"uid": note["uid"], "source_task": note["source_task"],
        "control_score": continuous_score(controls[note["source_task"]]),
        "treatment_score": continuous_score(calibration_by_task[note["source_task"]]),
        "paired_effect": causal_scores[note["uid"]],
        "treatment_strict_success": calibration_by_task[note["source_task"]]["success"]}
        for note in initial_pool]
    write(output, "causal_calibration.json", calibration)
    write(output, "evaluation_freeze.json", {
        "initial_memories.json": digest(output / "initial_memories.json"),
        "causal_calibration.json": digest(output / "causal_calibration.json")})

    sessions = []
    all_memories = {note["uid"]: note for note in initial_pool}
    for method in method_order:
        pools = {trajectory["trajectory"]: [dict(note) for note in initial_pool]
                 for trajectory in trajectories}
        for session_index in (0, 1):
            jobs, metadata = [], {}
            for trajectory in trajectories:
                task = trajectory["sessions"][session_index]
                pool = pools[trajectory["trajectory"]]
                bank = select_bank(method, pool, task, scenarios[task], causal_scores)
                jobs.append(initial_job(task, scenarios[task], bank, u, agent.tokenizer,
                                        "multi_session_test", method))
                metadata[task] = (trajectory, pool, bank, [note["uid"] for note in pool])
            rows = run_batch(jobs, agent, output, u)
            for row in rows:
                trajectory, pool, bank, before = metadata[row["task"]]
                note = memory_from(row, agent.tokenizer)
                inserted = None
                if note is not None and not any(existing["uid"] == note["uid"] for existing in pool):
                    pool.append(note)
                    all_memories[note["uid"]] = note
                    inserted = note["uid"]
                session = {**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"], "session_index": session_index,
                    "available_memory_before": before,
                    "selected_bank": [note["uid"] for note in bank],
                    "inserted_after_scoring": inserted,
                    "cross_session_exposure": (session_index == 1 and any(
                        memory["source_task"] == trajectory["sessions"][0] for memory in bank)),
                    "remote_latency_ms": session_latency(row)}
                sessions.append(session)
                with (output / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(session, ensure_ascii=False) + "\n")
        print(json.dumps({"actor_seed": seed, "method": method,
                          "sessions": 2 * len(trajectories), "completed": True}), flush=True)
    write(output, "all_memories.json", list(all_memories.values()))
    agent.close()
    actor_summary = {"actor_seed": seed, "history_sessions": len(history_rows),
        "initial_memory_records": len(initial_pool), "calibration_sessions": len(calibration_rows),
        "calibration_tool_calls": sum(row["tool_calls"] for row in calibration_rows),
        "calibration_remote_latency_ms": sum(session_latency(row) for row in calibration_rows),
        "test_sessions": len(sessions), "remote_requests": agent.calls,
        "method_execution_order": method_order,
        "source_sha256": source_hashes}
    write(output, "summary.json", actor_summary)
    return sessions, actor_summary


def paired_statistics(rows: list[dict], seed: int = 17127) -> dict:
    by_method = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][(row["actor_seed"], row["trajectory"], row["session_index"])] = row
    clusters = sorted({(row["actor_seed"], row["trajectory"]) for row in rows})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    report = []
    credit = by_method["CREDIT"]
    for method in METHODS:
        effects = []
        for actor_seed, trajectory in clusters:
            keys = [(actor_seed, trajectory, index) for index in (0, 1)]
            effects.append(float(np.mean([credit[key]["success"] - by_method[method][key]["success"]
                                          for key in keys])))
        boot = np.mean(np.asarray(effects)[draws], axis=1)
        wins, losses = sum(x > 0 for x in effects), sum(x < 0 for x in effects)
        p = float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue) if wins + losses else 1.0
        report.append({"baseline": method, "CREDIT_delta_VSR": float(np.mean(effects)),
            "cluster_bootstrap_CI95": np.quantile(boot, [.025, .975]).tolist(),
            "cluster_wins": wins, "cluster_losses": losses, "one_sided_exact_sign_p": p})
    return {"comparisons": report, "resampling_unit": "actor-seed by two-session trajectory",
            "clusters": len(clusters), "bootstrap_replicates": 10000, "bootstrap_seed": seed}


def summarize(rows: list[dict], actor_summaries: list[dict]) -> dict:
    calibration = {row["actor_seed"]: row["calibration_remote_latency_ms"]
                   for row in actor_summaries}
    tests_per_method_actor = len(rows) // (len(METHODS) * len(actor_summaries))
    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        attributed = [row["remote_latency_ms"] + (calibration[row["actor_seed"]]
                      / tests_per_method_actor if method == "CREDIT" else 0.0) for row in subset]
        entry = {"method": method, "sessions": len(subset),
            "strict_VSR": float(np.mean([row["success"] for row in subset])),
            "upstream_mean_score": float(np.mean([continuous_score(row) for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "mean_remote_requests": float(np.mean([len(row["generations"]) for row in subset])),
            "mean_online_remote_latency_ms": float(np.mean([row["remote_latency_ms"] for row in subset])),
            "mean_attributed_remote_latency_ms": float(np.mean(attributed)),
            "median_attributed_remote_latency_ms": float(np.median(attributed)),
            "p95_attributed_remote_latency_ms": float(np.quantile(attributed, .95))}
        for budget in LATENCY_BUDGETS_MS:
            entry[f"VSR_at_{budget}ms"] = float(np.mean([
                row["success"] and latency <= budget for row, latency in zip(subset, attributed)]))
            entry[f"completion_at_{budget}ms"] = float(np.mean([
                latency <= budget for latency in attributed]))
            entry[f"online_VSR_at_{budget}ms"] = float(np.mean([
                row["success"] and row["remote_latency_ms"] <= budget for row in subset]))
        table.append(entry)
    primary = 10000
    ranking = sorted(({"rank": 0, "method": row["method"],
                       "VSR_at_10000ms": row["VSR_at_10000ms"],
                       "strict_VSR": row["strict_VSR"],
                       "mean_attributed_remote_latency_ms": row["mean_attributed_remote_latency_ms"]}
                      for row in table),
                     key=lambda x: (-x["VSR_at_10000ms"], -x["strict_VSR"],
                                    x["mean_attributed_remote_latency_ms"], x["method"]))
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    return {"table": table, "latency_ranking": ranking,
        "primary_latency_budget_ms": primary, "latency_budgets_ms": list(LATENCY_BUDGETS_MS),
        "latency_definition": "sum of measured complete HTTPS chat-completion latency, including retries and backoff",
        "cost_attribution": "common history excluded; CREDIT calibration latency amortized over its 40 test sessions per actor",
        "budget_rule": "post-hoc completed-response censoring; a verified success counts only when attributed remote latency is within budget",
        "actor_summaries": actor_summaries, "paired_statistics": paired_statistics(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--tokenizer", type=Path, default=MODEL)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("results/toolsandbox_m_multi_actor_v1"))
    parser.add_argument("--actor-seeds", type=int, nargs="+", default=[101, 203, 307])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-trajectories", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    (args.output / "source").mkdir()
    u = upstream_imports(args.upstream)
    clock = "2026-10-14T12:00:00"
    install_local_clock(clock)
    scenarios, exclusions = local_scenarios(u)
    history = sorted(name for name, scenario in scenarios.items()
                     if u.ScenarioCategories.SINGLE_TOOL_CALL in scenario.categories)
    future = sorted(set(scenarios) - set(history))
    trajectories = make_trajectories(future)
    if args.limit_trajectories is not None:
        trajectories = trajectories[:args.limit_trajectories]
        future = [task for trajectory in trajectories for task in trajectory["sessions"]]
    source_names = ["toolsandbox_m_multi_actor.py", "remote_json_agent.py",
                    "toolsandbox_m_multi_actor_verify.py", "toolsandbox_m.py",
                    "toolsandbox_local_v3.py"]
    source_hashes = {name: digest(Path(__file__).with_name(name)) for name in source_names}
    for name in source_names:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    upstream_hashes = {str(path.relative_to(args.upstream)): digest(path)
                       for path in args.upstream.rglob("*.py")}
    method_orders = {}
    for index, seed in enumerate(args.actor_seeds):
        shift = (2 * index) % len(METHODS)
        method_orders[str(seed)] = list(METHODS[shift:] + METHODS[:shift])
    protocol = {"suite": "ToolSandbox-M multi-actor closed-provider evaluation",
        "upstream_directory": str(args.upstream.resolve()),
        "upstream_commit": "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1",
        "upstream_source_sha256": upstream_hashes, "source_sha256": source_hashes,
        "history": history, "future": future, "trajectories": trajectories,
        "actor_seeds": args.actor_seeds, "temperature": 0.2,
        "method_execution_orders": method_orders,
        "memory_capacity": 2, "maximum_tool_calls_per_session": 8,
        "maximum_actor_turns": 16, "clock": clock, "methods": list(METHODS),
        "latency_budgets_ms": list(LATENCY_BUDGETS_MS), "primary_latency_budget_ms": 10000,
        "latency_scope": "measured closed-provider HTTPS chat-completion latency; ToolSandbox tools remain local",
        "latency_cost_attribution": "exclude common history; amortize CREDIT-only calibration latency across its 40 test sessions per actor",
        "ranking_rule": "VSR within primary latency budget, then uncensored VSR, then mean latency",
        "request_parallelism": args.workers,
        "network_measurement": "every actor turn issues a live request; no cross-method response or latency cache",
        "test_frozen_before_generation": True,
        "provider_credentials_persisted": False, "exclusions": exclusions,
        "pilot_limit_trajectories": args.limit_trajectories}
    write(args.output, "frozen_protocol.json", protocol)
    (args.output / "scenarios.pkl").write_bytes(dill.dumps(scenarios))
    write(args.output, "scenario_freeze.json", {"sha256": digest(args.output / "scenarios.pkl")})

    all_sessions, actor_summaries = [], []
    for seed in args.actor_seeds:
        sessions, actor_summary = run_actor(seed, args.output, args.env, args.tokenizer,
            args.workers, scenarios, history, trajectories, u, source_hashes,
            method_orders[str(seed)])
        all_sessions.extend(sessions)
        actor_summaries.append(actor_summary)
    with (args.output / "sessions.jsonl").open("w") as stream:
        for row in all_sessions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(all_sessions, actor_summaries)
    write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["table"][0]))
        writer.writeheader(); writer.writerows(summary["table"])
    audit = {"passed": len(all_sessions) == len(args.actor_seeds) * len(METHODS) * len(future),
        "actor_seeds": len(args.actor_seeds), "trajectories_per_actor": len(trajectories),
        "test_sessions": len(all_sessions),
        "unique_session_keys": len({(row["actor_seed"], row["method"], row["task"])
                                     for row in all_sessions}),
        "all_transport_latencies_positive": all(row["remote_latency_ms"] > 0 for row in all_sessions),
        "all_verification_covered": all(row["verification_coverage"] for row in all_sessions),
        "sessions_sha256": digest(args.output / "sessions.jsonl"),
        "protocol_sha256": digest(args.output / "frozen_protocol.json")}
    audit["passed"] = audit["passed"] and audit["unique_session_keys"] == len(all_sessions) \
        and audit["all_transport_latencies_positive"] and audit["all_verification_covered"]
    write(args.output, "audit.json", audit)
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
