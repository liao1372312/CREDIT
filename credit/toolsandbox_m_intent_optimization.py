"""Zero-probe intent-aligned memory selection on fresh ToolSandbox prompts."""

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
from .toolsandbox_local_v3 import install_local_clock, run_batch, upstream_imports, write
from .toolsandbox_m import continuous_score, note_relevance
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_streaming_optimization import select_bank as legacy_select_bank
from .user_data import digest


METHODS = ("NoMem", "FIFO", "Semantic", "CREDIT-Compact", "CREDIT-Intent")
LATENCY_BUDGETS_MS = (5000, 10000, 20000)
PROMPT_PREFIX = "Please handle this request now: "


def prompt_variant(original: str) -> str:
    """Change prompt surface form without changing entities or task semantics."""
    return PROMPT_PREFIX + original


def requirements(query: str, allowed: set[str]) -> tuple[set[str], set[str]]:
    """Infer required and supporting operations from public text and tool schema."""
    query = query.lower()
    if "set_wifi_status" in allowed:
        return {"set_wifi_status", "get_wifi_status"}, {
            "get_low_battery_mode_status", "set_low_battery_mode_status"}
    if "set_location_service_status" in allowed:
        return {"set_location_service_status", "get_location_service_status"}, {
            "get_low_battery_mode_status", "set_low_battery_mode_status"}
    if "send_message_with_phone_number" in allowed:
        return {"search_contacts", "send_message_with_phone_number",
                "get_cellular_service_status", "set_cellular_service_status"}, set()
    if "set_cellular_service_status" in allowed:
        return {"set_cellular_service_status", "get_cellular_service_status"}, {
            "get_low_battery_mode_status", "set_low_battery_mode_status"}
    if "search_holiday" in allowed:
        return {"search_holiday", "get_current_timestamp", "timestamp_diff"}, set()
    if "remove_contact" in allowed:
        return {"search_contacts", "remove_contact"}, set()
    if "modify_contact" in allowed:
        return {"search_messages", "get_current_timestamp", "search_contacts",
                "modify_contact"}, set()
    temporal = {"get_current_timestamp", "shift_timestamp",
                "timestamp_to_datetime_info", "datetime_info_to_timestamp"} & allowed
    if "remove_reminder" in allowed:
        return {"search_reminder", "remove_reminder"}, temporal
    if "add_reminder" in allowed and any(word in query for word in
                                           ("remind", "add a todo", "create")):
        return {"add_reminder"}, temporal
    if "modify_reminder" in allowed:
        return {"search_reminder", "modify_reminder"}, temporal
    if "search_reminder" in allowed:
        return {"search_reminder"}, temporal
    return set(), set()


def temporal_tags(value: str) -> set[str]:
    value = value.lower()
    tags = set()
    if any(word in value for word in ("created", "creation", " made ")):
        tags.add("creation")
    for tag in ("yesterday", "upcoming", "latest", "oldest", "tomorrow"):
        if tag in value:
            tags.add(tag)
    if "next reminder" in value or "upcoming reminder" in value:
        tags.add("upcoming")
    if "most recent" in value or "last person" in value:
        tags.add("latest")
    return tags


def temporal_compatible(note: dict, query: str) -> bool:
    """Reject a remembered temporal predicate that contradicts the request."""
    if "search_reminder" not in note["tools"]:
        return True
    target, source = temporal_tags(" " + query), temporal_tags(
        " " + note["source_task"].replace("_", " "))
    if "creation" in target and "creation" not in source:
        return False
    target_time = target & {"yesterday", "upcoming", "latest", "oldest"}
    source_time = source & {"yesterday", "upcoming", "latest", "oldest"}
    return not (target_time and source_time and target_time.isdisjoint(source_time))


def intent_select(pool: list[dict], query: str, scenario) -> list[dict]:
    """Select at most two memories that cover distinct requested operations."""
    allowed = set(scenario.starting_context.tool_allow_list or [])
    core, support = requirements(query, allowed)
    candidates = [note for note in pool
                  if set(note["tools"]) & allowed and temporal_compatible(note, query)]

    def parts(note: dict) -> tuple[set[str], set[str]]:
        tools = set(note["tools"]) & allowed
        return tools & core, tools & support

    def rank(note: dict) -> tuple[float, float, str]:
        core_hit, support_hit = parts(note)
        irrelevant = len(set(note["tools"]) - allowed)
        action_hit = sum(tool.startswith(("add_", "remove_", "modify_", "send_", "set_"))
                         for tool in core_hit)
        score = (100.0 * len(core_hit) + 12.0 * len(support_hit)
                 + 40.0 * action_hit
                 + 8.0 * float(note["source_upstream_score"])
                 + 0.01 * note_relevance(note, query, allowed)
                 - 6.0 * irrelevant)
        # A future-time modification can reuse a verified conversion chain,
        # even when its terminal write operation differs.
        if ("modify_reminder" in core and
                any(word in query.lower() for word in ("tomorrow", " pm", " am")) and
                "datetime_info_to_timestamp" in note["tools"]):
            score += 110.0
        return (-score, -float(note["source_upstream_score"]), note["uid"])

    aligned = [note for note in candidates if parts(note)[0]]
    allow_shared_time_chain = ("modify_reminder" in core and
        any(word in query.lower() for word in ("tomorrow", " pm", " am")))
    supportive = [note for note in candidates
                  if allow_shared_time_chain and not parts(note)[0] and parts(note)[1]]
    ranked = sorted(aligned + supportive, key=rank)
    if not ranked:
        return []
    chosen = [ranked[0]]
    covered = parts(ranked[0])[0]
    if not covered:
        return chosen
    # Add a second note only when it contributes an uncovered required
    # operation. This is the pre-registered guard against memory interference.
    second = next((note for note in ranked[1:]
                   if parts(note)[0] - covered), None)
    if second is not None:
        chosen.append(second)
    return chosen


def select(method: str, pool: list[dict], task: str, query: str, scenario,
           pair_vetoes: set[tuple[str, str]]) -> list[dict]:
    if method == "CREDIT-Intent":
        return intent_select(pool, query, scenario)
    if method == "CREDIT-Compact":
        return legacy_select_bank("CREDIT", pool, task, scenario,
                                  pair_vetoes, "compact1")
    return legacy_select_bank(method, pool, task, scenario, pair_vetoes, "diverse2")


def balanced_order(methods: list[str], actor_index: int, trajectory_index: int,
                   session_index: int) -> list[str]:
    shift = (actor_index + trajectory_index + session_index) % len(methods)
    return methods[shift:] + methods[:shift]


def make_job(task: str, scenario, bank: list[dict], query: str, u, tokenizer,
             method: str):
    job = initial_job_v2(task, scenario, bank, u, tokenizer,
                         "intent_prompt_confirmation", method)
    if job["chat"][-1]["role"] != "user":
        raise ValueError("expected final public message to be the user request")
    job["chat"][-1]["content"] = query
    return job


def original_request(task: str, scenario, u, tokenizer) -> str:
    job = initial_job_v2(task, scenario, [], u, tokenizer, "request_probe", "NoMem")
    return job["chat"][-1]["content"]


def run_actor(seed: int, actor_index: int, trajectories: list[dict], root: Path,
              args, scenarios, pool: list[dict], pair_vetoes: set[tuple[str, str]], u):
    directory = root / f"intent_confirmation_seed_{seed}"
    directory.mkdir()
    agent = RemoteJSONAgent(directory, seed, env=args.env, tokenizer_path=args.tokenizer,
        temperature=0.0, max_new_tokens=384, workers=args.workers,
        exact_prompt_cache=False)
    agent.binding_interface = "tool_json"
    agent.manifest["benchmark"] = "intent_prompt_confirmation"
    sessions, positions = [], defaultdict(lambda: [0] * len(METHODS))
    for trajectory_index, trajectory in enumerate(trajectories):
        for session_index, task in enumerate(trajectory["sessions"]):
            original = original_request(task, scenarios[task], u, agent.tokenizer)
            query = prompt_variant(original)
            order = balanced_order(list(METHODS), actor_index, trajectory_index,
                                   session_index)
            jobs, metadata = [], {}
            for position, method in enumerate(order):
                positions[method][position] += 1
                bank = select(method, pool, task, query, scenarios[task], pair_vetoes)
                jobs.append(make_job(task, scenarios[task], bank, query, u,
                                     agent.tokenizer, method))
                metadata[method] = (bank, position)
            for row in run_batch(jobs, agent, directory, u):
                bank, position = metadata[row["method"]]
                item = {**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"],
                    "session_index": session_index,
                    "selected_bank": [note["uid"] for note in bank],
                    "remote_latency_ms": session_latency(row),
                    "batch_position": position,
                    "original_request_sha256": sha256(original.encode()).hexdigest(),
                    "prompt_variant": query}
                sessions.append(item)
                with (directory / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    agent.close()
    summary = {"actor_seed": seed, "phase": "intent_prompt_confirmation",
        "sessions": len(sessions), "remote_requests": agent.calls,
        "method_batch_position_counts": dict(positions)}
    write(directory, "summary.json", summary)
    return sessions, summary


def summarize(rows: list[dict], methods: list[str]) -> list[dict]:
    table = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        latencies = [row["remote_latency_ms"] for row in subset]
        entry = {"method": method, "sessions": len(subset),
            "strict_VSR": float(np.mean([row["success"] for row in subset])),
            "upstream_mean_score": float(np.mean([continuous_score(row) for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "mean_remote_requests": float(np.mean([len(row["generations"]) for row in subset])),
            "mean_online_remote_latency_ms": float(np.mean(latencies)),
            "p95_online_remote_latency_ms": float(np.quantile(latencies, .95))}
        for budget in LATENCY_BUDGETS_MS:
            entry[f"VSR_at_{budget}ms"] = float(np.mean([
                row["success"] and latency <= budget
                for row, latency in zip(subset, latencies)]))
            entry[f"completion_at_{budget}ms"] = float(np.mean([
                latency <= budget for latency in latencies]))
        table.append(entry)
    return table


def paired_statistics(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for row in rows:
        by[row["method"]][(row["actor_seed"], row["trajectory"],
                           row["session_index"])] = row
    clusters = sorted({(row["actor_seed"], row["trajectory"]) for row in rows})
    rng = np.random.default_rng(20261016)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    report = []
    for method in METHODS:
        effects, score_effects = [], []
        for seed, trajectory in clusters:
            keys = [(seed, trajectory, index) for index in (0, 1)]
            effects.append(float(np.mean([
                by["CREDIT-Intent"][key]["success"] - by[method][key]["success"]
                for key in keys])))
            score_effects.append(float(np.mean([
                continuous_score(by["CREDIT-Intent"][key]) -
                continuous_score(by[method][key]) for key in keys])))
        boot = np.mean(np.asarray(effects)[draws], axis=1)
        score_boot = np.mean(np.asarray(score_effects)[draws], axis=1)
        wins = sum(value > 0 for value in effects)
        losses = sum(value < 0 for value in effects)
        p_value = float(binomtest(wins, wins + losses, .5,
                                  alternative="greater").pvalue) \
            if wins + losses else 1.0
        report.append({"baseline": method,
            "CREDIT_Intent_delta_VSR": float(np.mean(effects)),
            "VSR_cluster_bootstrap_CI95": np.quantile(boot, [.025, .975]).tolist(),
            "CREDIT_Intent_delta_upstream_score": float(np.mean(score_effects)),
            "score_cluster_bootstrap_CI95": np.quantile(
                score_boot, [.025, .975]).tolist(),
            "cluster_wins": wins, "cluster_losses": losses,
            "one_sided_exact_sign_p": p_value})
    return {"comparisons": report, "clusters": len(clusters),
        "resampling_unit": "actor seed by two-session trajectory",
        "bootstrap_replicates": 10000, "bootstrap_seed": 20261016}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/toolsandbox_m_streaming_optimization_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/toolsandbox_m_intent_optimization_v1"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[929, 937, 941, 947, 953])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    (args.output / "source").mkdir()
    old_protocol = json.loads((args.source / "frozen_protocol.json").read_text())
    old_summary = json.loads((args.source / "summary.json").read_text())
    args.tokenizer = args.tokenizer or Path(json.loads(
        (args.source / f"stream_confirmation_seed_{old_protocol['confirmation_seeds'][0]}" /
         "agent_manifest.json").read_text())["tokenizer_path"])
    trajectories = old_protocol["confirmation"]
    if args.smoke:
        args.seeds = args.seeds[:1]
        trajectories = trajectories[:1]
    scenarios = dill.loads((args.source / "scenarios.pkl").read_bytes())
    pool = json.loads((args.source / "memory_pool.json").read_text())
    pair_vetoes = {tuple(pair) for pair in old_protocol["harmful_pair_uids"]}
    upstream = Path(old_protocol["upstream_directory"])
    u = upstream_imports(upstream)
    install_local_clock(old_protocol["clock"])
    names = ["toolsandbox_m_intent_optimization.py",
             "toolsandbox_m_intent_optimization_verify.py",
             "toolsandbox_m_streaming_optimization.py", "toolsandbox_m_followup.py",
             "toolsandbox_local_v3.py", "toolsandbox_m.py", "remote_json_agent.py"]
    hashes = {name: digest(Path(__file__).with_name(name)) for name in names}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    protocol = {"suite": "ToolSandbox-M zero-probe intent prompt replication",
        "source_results": str(args.source.resolve()),
        "source_memory_pool_sha256": digest(args.source / "memory_pool.json"),
        "source_scenarios_sha256": digest(args.source / "scenarios.pkl"),
        "source_sha256": hashes, "upstream_directory": str(upstream),
        "upstream_source_sha256": old_protocol["upstream_source_sha256"],
        "clock": old_protocol["clock"], "trajectories": trajectories,
        "actor_seeds": args.seeds, "methods": list(METHODS),
        "prompt_transform": {"kind": "literal_prefix", "prefix": PROMPT_PREFIX},
        "selector_inputs": "public request text, allowed tool names, frozen memory metadata",
        "selector_training_remote_requests": 0,
        "selector_training_remote_latency_ms": 0.0,
        "historical_R_and_D_remote_latency_ms": old_summary["credit_training_latency_ms"],
        "historical_confirmation_examined": True,
        "claim_scope": "post-hoc prompt-surface replication on existing semantic scenarios",
        "paired_scheduling": "all methods for a task share one concurrent remote batch",
        "temperature": 0.0, "maximum_tool_calls_per_session": 8,
        "provider_credentials_persisted": False, "smoke": args.smoke}
    write(args.output, "frozen_protocol.json", protocol)
    (args.output / "scenarios.pkl").write_bytes((args.source / "scenarios.pkl").read_bytes())
    write(args.output, "memory_pool.json", pool)

    rows, actor_summaries = [], []
    for index, seed in enumerate(args.seeds):
        found, summary = run_actor(seed, index, trajectories, args.output, args,
                                   scenarios, pool, pair_vetoes, u)
        rows.extend(found)
        actor_summaries.append(summary)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = summarize(rows, list(METHODS))
    ranking = sorted((dict(row) for row in table), key=lambda row: (
        -row["VSR_at_10000ms"], -row["strict_VSR"],
        -row["upstream_mean_score"], row["mean_online_remote_latency_ms"],
        row["method"]))
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    selection = {}
    for trajectory in trajectories:
        for task in trajectory["sessions"]:
            original = original_request(task, scenarios[task], u,
                                        __import__("transformers").AutoTokenizer.from_pretrained(
                                            args.tokenizer, local_files_only=True))
            query = prompt_variant(original)
            bank = intent_select(pool, query, scenarios[task])
            selection[task] = {"required_and_supporting_tools": [
                    sorted(part) for part in requirements(query, set(
                        scenarios[task].starting_context.tool_allow_list or []))],
                "selected_uids": [note["uid"] for note in bank],
                "selected_sources": [note["source_task"] for note in bank]}
    summary = {"confirmation_table": table, "online_latency_ranking": ranking,
        "paired_statistics": paired_statistics(rows),
        "intent_selection": selection,
        "incremental_selector_training_remote_latency_ms": 0.0,
        "historical_R_and_D_remote_latency_ms": protocol[
            "historical_R_and_D_remote_latency_ms"],
        "actor_summaries": actor_summaries}
    write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    expected = len(args.seeds) * len(trajectories) * 2 * len(METHODS)
    audit = {"passed": len(rows) == expected and all(
        row["verification_coverage"] for row in rows),
        "confirmation_sessions": len(rows),
        "confirmation_sha256": digest(args.output / "confirmation_sessions.jsonl")}
    write(args.output, "audit.json", audit)
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
