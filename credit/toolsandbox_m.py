"""Chronological multi-session memory evaluation on pinned ToolSandbox.

Each trajectory contains two related but independently restored ToolSandbox
sessions.  The second session may use a verified textual memory written only
after the first session was scored.  Original tools, contexts, milestones, and
minefields are preserved; no drift is claimed for this suite.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import re
from pathlib import Path
from random import Random

import dill
import numpy as np
from scipy.stats import binomtest

from .emp_binding_agent import QwenJSONAgent
from .toolsandbox_local_v3 import (
    MODEL, UPSTREAM, family, fingerprint, initial_job, install_local_clock,
    local_scenarios, memory_from, run_batch, upstream_imports, write,
)
from .user_data import digest


METHODS = ("NoMem", "FIFO", "Recency", "Semantic", "CREDIT")


def words(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z]+", value.lower())
            if token not in {"with", "and", "the", "by", "latest", "implicit", "alt"}}


def note_relevance(note: dict, task: str, allowed: set[str]) -> float:
    tool_overlap = len(set(note["tools"]) & allowed)
    lexical = len(words(note["source_task"]) & words(task))
    same_family = family(note["source_task"]) == family(task)
    return 100.0 * same_family + 10.0 * tool_overlap + 2.0 * lexical


def select_bank(method: str, pool: list[dict], task: str, scenario,
                causal_scores: dict[str, float], capacity: int = 2) -> list[dict]:
    allowed = set(scenario.starting_context.tool_allow_list or [])
    supported = [note for note in pool if set(note["tools"]) & allowed]
    if method == "NoMem":
        return []
    if method == "FIFO":
        return supported[:capacity]
    if method == "Recency":
        return list(reversed(supported))[:capacity]
    if method == "Semantic":
        return sorted(supported, key=lambda note: (-note_relevance(note, task, allowed),
                                                    note["arrival"], note["uid"]))[:capacity]
    if method == "CREDIT":
        def score(note: dict) -> tuple[float, float, str]:
            effect = causal_scores.get(note["uid"], 0.0)
            value = note_relevance(note, task, allowed) + 20.0 * effect
            # A causally harmful historical note is retained in the archive but
            # excluded from exposure.  New online notes have a neutral prior.
            if effect < -0.05:
                value -= 1000.0
            return (-value, -float(note["source_upstream_score"]), note["uid"])
        chosen = sorted(supported, key=score)
        return [note for note in chosen if score(note)[0] < 900.0][:capacity]
    raise ValueError("unknown method: " + method)


def continuous_score(row: dict) -> float:
    evaluation = row.get("upstream_evaluation")
    return float(evaluation["similarity"]) if evaluation else 0.0


def make_trajectories(future: list[str]) -> list[dict]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for task in future:
        grouped[family(task)].append(task)
    trajectories = []
    for name, members in sorted(grouped.items()):
        members.sort(key=lambda task: (task.endswith(("_alt", "_implicit")), task))
        if len(members) != 2:
            raise ValueError(f"expected two sessions for {name}, found {members}")
        trajectories.append({"trajectory": name, "sessions": members})
    return trajectories


def paired_statistics(rows: list[dict], seed: int = 7127) -> dict:
    by_method = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][(row["trajectory"], row["session_index"])] = row
    trajectories = sorted({row["trajectory"] for row in rows})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(trajectories), (10000, len(trajectories)))
    report = []
    credit = by_method["CREDIT"]
    for method in METHODS:
        values = []
        for trajectory in trajectories:
            tasks = [(trajectory, 0), (trajectory, 1)]
            values.append(float(np.mean([credit[key]["success"] - by_method[method][key]["success"]
                                         for key in tasks])))
        samples = np.mean(np.asarray(values)[draws], axis=1)
        wins = sum(value > 0 for value in values)
        losses = sum(value < 0 for value in values)
        p_value = float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue) if wins + losses else 1.0
        report.append({"baseline": method, "CREDIT_delta_VSR": float(np.mean(values)),
                       "trajectory_bootstrap_CI95": np.quantile(samples, [.025, .975]).tolist(),
                       "trajectory_wins": wins, "trajectory_losses": losses,
                       "one_sided_exact_sign_p": p_value})
    return {"comparisons": report, "resampling_unit": "two-session trajectory",
            "trajectories": len(trajectories), "bootstrap_replicates": 10000,
            "bootstrap_seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=Path("results/toolsandbox_m_v1"))
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
    upstream_sources = {str(path.relative_to(args.upstream)): digest(path)
                        for path in args.upstream.rglob("*.py")}
    source_names = ["toolsandbox_m.py", "toolsandbox_local_v3.py", "emp_binding_agent.py"]
    protocol = {"suite": "ToolSandbox-M chronological multi-session subset",
        "upstream_directory": str(args.upstream.resolve()),
        "upstream_commit": "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1",
        "upstream_url": "https://github.com/apple-aiml-research/ToolSandbox",
        "upstream_source_sha256": upstream_sources,
        "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in source_names},
        "history": history, "future": future, "trajectories": trajectories,
        "clock": clock, "methods": list(METHODS), "memory_capacity": 2,
        "maximum_tool_calls_per_session": 8, "maximum_actor_turns": 16,
        "session_state": "original scenario state restored independently for every session",
        "memory_state": "persistent within each two-session trajectory; update only after scoring",
        "CREDIT_mode": "ToolSandbox adapter: causal-weighted semantic support and online post-score insertion",
        "causal_calibration": "one paired source-task exposure per verified history memory",
        "provider_drift": False, "external_network_tool_calls": 0,
        "actor_seeds": 1, "temperature": 0,
        "latency_scope": "cache-assisted execution; no latency ranking",
        "unknown_verification_counts_as_failure": True,
        "exclusions": exclusions, "source_task_overlap_with_test": False}
    write(args.output, "frozen_protocol.json", protocol)
    for name in source_names:
        (args.output / "source" / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (args.output / "scenarios.pkl").write_bytes(dill.dumps(scenarios))
    write(args.output, "scenario_freeze.json", {"sha256": digest(args.output / "scenarios.pkl")})
    write(args.output, "session_schedule.json", trajectories)

    agent = QwenJSONAgent(args.model, args.output, batch_size=8, max_new_tokens=384)
    agent.binding_interface = "tool_json"
    agent.manifest.update({"benchmark": protocol["suite"]})
    history_rows = []
    for offset in range(0, len(history), 8):
        history_rows += run_batch([initial_job(name, scenarios[name], [], u, agent.tokenizer,
                                               "history", "NoMem")
                                   for name in history[offset:offset + 8]], agent, args.output, u)
    initial_pool = [note for row in sorted(history_rows, key=lambda row: history.index(row["task"]))
                    if (note := memory_from(row, agent.tokenizer)) is not None]
    write(args.output, "initial_memories.json", initial_pool)

    controls = {row["task"]: row for row in history_rows}
    calibration_rows = []
    for note in initial_pool:
        calibration_rows += run_batch([initial_job(note["source_task"], scenarios[note["source_task"]],
                                                   [note], u, agent.tokenizer, "calibration",
                                                   "single_memory_probe")],
                                      agent, args.output, u)
    causal_scores = {note["uid"]: continuous_score(row) - continuous_score(controls[note["source_task"]])
                     for note, row in zip(initial_pool, calibration_rows)}
    calibration = [{"uid": note["uid"], "source_task": note["source_task"],
                    "control_score": continuous_score(controls[note["source_task"]]),
                    "treatment_score": continuous_score(row),
                    "paired_effect": causal_scores[note["uid"]],
                    "treatment_strict_success": row["success"]}
                   for note, row in zip(initial_pool, calibration_rows)]
    write(args.output, "causal_calibration.json", calibration)
    write(args.output, "evaluation_freeze.json", {name: digest(args.output / name) for name in
        ("frozen_protocol.json", "scenario_freeze.json", "session_schedule.json",
         "initial_memories.json", "causal_calibration.json")})

    sessions = []
    all_memories = {note["uid"]: note for note in initial_pool}
    for method in METHODS:
        for trajectory in trajectories:
            pool = [dict(note) for note in initial_pool]
            for session_index, task in enumerate(trajectory["sessions"]):
                bank = select_bank(method, pool, task, scenarios[task], causal_scores)
                before = [note["uid"] for note in pool]
                row = run_batch([initial_job(task, scenarios[task], bank, u, agent.tokenizer,
                                             "multi_session_test", method)],
                                agent, args.output, u)[0]
                note = memory_from(row, agent.tokenizer)
                inserted = None
                if note is not None and not any(existing["uid"] == note["uid"] for existing in pool):
                    pool.append(note); all_memories[note["uid"]] = note; inserted = note["uid"]
                session = {**row, "trajectory": trajectory["trajectory"],
                           "session_index": session_index, "available_memory_before": before,
                           "selected_bank": [note["uid"] for note in bank],
                           "inserted_after_scoring": inserted,
                           "cross_session_exposure": any(memory["source_task"] == trajectory["sessions"][0]
                               for memory in bank) if session_index == 1 else False}
                sessions.append(session)
                with (args.output / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(session, ensure_ascii=False) + "\n")
        print(json.dumps({"method": method, "sessions": len(future), "completed": True}), flush=True)
    write(args.output, "all_memories.json", list(all_memories.values()))
    agent.close()

    table = []
    for method in METHODS:
        subset = [row for row in sessions if row["method"] == method]
        second = [row for row in subset if row["session_index"] == 1]
        table.append({"method": method, "sessions": len(subset),
            "strict_VSR": float(np.mean([row["success"] for row in subset])),
            "session_1_VSR": float(np.mean([row["success"] for row in subset if row["session_index"] == 0])),
            "session_2_VSR": float(np.mean([row["success"] for row in second])),
            "verification_coverage": float(np.mean([row["verification_coverage"] for row in subset])),
            "upstream_mean_score": float(np.mean([continuous_score(row) for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "exposure_rate": float(np.mean([bool(row["exposure"]) for row in subset])),
            "cross_session_exposure_rate_session_2": float(np.mean([row["cross_session_exposure"] for row in second])),
            "calibration_extra_tool_calls": sum(row["tool_calls"] for row in calibration_rows)
                if method == "CREDIT" else 0})
    statistics = paired_statistics(sessions)
    write(args.output, "paired_statistics.json", statistics)
    write(args.output, "summary.json", {"table": table, "history_sessions": len(history_rows),
        "initial_memory_records": len(initial_pool), "trajectories": len(trajectories),
        "sessions_per_method": len(future), "test_sessions": len(sessions),
        "calibration_sessions": len(calibration_rows),
        "calibration_extra_tool_calls": sum(row["tool_calls"] for row in calibration_rows),
        "provider_drift": False, "external_network_tool_calls": 0,
        "full_paper_CREDIT": "ToolSandbox-specific adapter with causal semantic retention"})
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print(json.dumps({"table": table, "paired_statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
