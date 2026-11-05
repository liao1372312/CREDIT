"""Resume the registered actor from its persisted provider receipts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns, sleep

from . import toolsandbox_local_v3 as adapter
from .remote_json_agent import RemoteJSONAgent
from . import toolsandbox_m_executor_factorial as experiment
from .toolsandbox_m_followup import initial_job_v2
from .toolsandbox_m_intent_optimization import balanced_order
from .user_data import digest


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _infer_one(agent, chat, waits: list[dict], method: str, task: str):
    for attempt in range(8):
        try:
            return agent.infer([chat])[0]
        except RuntimeError as error:
            if "HTTP status 403" not in str(error) or attempt == 7:
                raise
            seconds = min(120, 15 * (2 ** attempt))
            waits.append({"task": task, "method": method,
                          "attempt": attempt + 1, "wait_seconds": seconds})
            print(json.dumps({"recovery_403": True, "task": task,
                              "method": method, "wait_seconds": seconds}),
                  flush=True)
            sleep(seconds)
    raise AssertionError("unreachable")


def _jobs(task, scenario, query, pool, u, tokenizer, actor_index,
          trajectory_index, session_index, positions):
    order = balanced_order(list(experiment.METHODS), actor_index,
                           trajectory_index, session_index)
    jobs, info = [], {}
    for position, method in enumerate(order):
        positions[method][position] += 1
        began = perf_counter_ns()
        bank = experiment.select(method, pool, query, scenario)
        selection_ms = (perf_counter_ns() - began) / 1e6
        job = initial_job_v2(
            task, scenario, bank, u, tokenizer,
            "executor_factorial_confirmation", method)
        job["selection_latency_ms"] = selection_ms
        jobs.append(job)
        info[method] = (bank, position)
    return jobs, info


def _finish(found, info, seed, trajectory, session_index, metadata):
    output = []
    for row in found:
        bank, position = info[row["method"]]
        remote, local, accounted = experiment._latency(row)
        memory, stage = experiment.method_parts(row["method"])
        query = metadata[row["task"]]["public_request"]
        output.append(experiment.source_suite.annotate({
            **row,
            "actor_seed": seed,
            "trajectory": trajectory,
            "session_index": session_index,
            "selected_bank": [note["uid"] for note in bank],
            "selected_memory_schemas": [
                note.get("memory_schema") for note in bank],
            "memory_arm": memory,
            "executor_stage": stage,
            "remote_latency_ms": remote,
            "local_runtime_ms": local,
            "accounted_latency_ms": accounted,
            "batch_position": position,
            "public_request_sha256": sha256(query.encode()).hexdigest(),
        }, metadata))
    return output


def _complete_jobs(jobs, queries, agent, directory, u, waits,
                   rounds_already: int = 0):
    found, active = [], jobs
    for _ in range(rounds_already, experiment.MAX_MODEL_TURNS):
        if not active:
            break
        records = [
            _infer_one(agent, job["chat"], waits, job["method"], job["task"])
            for job in active]
        remaining = []
        for job, record in zip(active, records):
            experiment.consume(job["method"], job, record, u,
                               queries[job["task"]])
            if job["done"]:
                found.append(experiment.scenario_support.prior.finish_direct(
                    job, directory, u))
            else:
                remaining.append(job)
        active = remaining
    for job in active:
        job["termination"] = "turn_budget"
        found.append(experiment.scenario_support.prior.finish_direct(
            job, directory, u))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    args = parser.parse_args()
    protocol = read(args.results / "frozen_protocol.json")
    if args.seed not in protocol["actor_seeds"]:
        raise ValueError("seed is not registered")
    for name, expected in protocol["source_sha256"].items():
        if digest(Path(experiment.__file__).with_name(name)) != expected:
            raise ValueError("frozen source changed: " + name)

    expected_dir = args.results / f"executor_factorial_seed_{args.seed}"
    retry2 = args.results / f"executor_factorial_seed_{args.seed}_aborted_403_retry2"
    if retry2.exists():
        raise FileExistsError(retry2)
    expected_dir.rename(retry2)
    original = args.results / f"executor_factorial_seed_{args.seed}_aborted_403"
    original.rename(expected_dir)

    upstream = Path(protocol["upstream_directory"])
    u = adapter.upstream_imports(upstream)
    adapter.install_local_clock(protocol["clock"])
    scenarios, metadata, trajectories = \
        experiment.source_suite.build_scenarios(u)
    pool = read(args.results / "memory_pool.json")
    reference = read(args.results / "executor_factorial_seed_3461" /
                     "agent_manifest.json")
    agent = RemoteJSONAgent(
        expected_dir, args.seed, env=args.env,
        tokenizer_path=Path(reference["tokenizer_path"]), temperature=0.0,
        max_new_tokens=384, workers=1, exact_prompt_cache=False)
    ledger = rows(expected_dir / "generations.jsonl")
    agent.calls = len(ledger)
    agent.tokens_in = sum(row["input_tokens"] for row in ledger)
    agent.tokens_out = sum(row["output_tokens"] for row in ledger)
    agent.prompt_keys = {row["prompt_sha256"] for row in ledger}
    agent.manifest["benchmark"] = "ToolSandbox-M executor component factorial"
    agent.manifest["receipt_resume"] = True

    sessions = rows(expected_dir / "sessions.jsonl")
    used = {turn["generation_uid"] for row in sessions
            for turn in row["generations"] if turn.get("generation_uid")}
    orphan = [row for row in ledger if row["generation_uid"] not in used]
    if len(orphan) != len(experiment.METHODS):
        raise ValueError("expected one persisted incomplete batch")
    complete_tasks = list(dict.fromkeys(row["task"] for row in sessions))
    sequence = [(ti, si, trajectory, task)
        for ti, trajectory in enumerate(trajectories)
        for si, task in enumerate(trajectory["sessions"])]
    if [task for _, _, _, task in sequence[:len(complete_tasks)]] != complete_tasks:
        raise ValueError("completed task prefix differs")

    positions = defaultdict(lambda: [0] * len(experiment.METHODS))
    actor_index = protocol["actor_seeds"].index(args.seed)
    for ti, si, _, _ in sequence[:len(complete_tasks)]:
        for position, method in enumerate(balanced_order(
                list(experiment.METHODS), actor_index, ti, si)):
            positions[method][position] += 1
    waits = []
    start = len(complete_tasks)
    for offset, (ti, si, trajectory, task) in enumerate(sequence[start:]):
        scenario = scenarios[task]
        query = metadata[task]["public_request"]
        jobs, info = _jobs(task, scenario, query, pool, u, agent.tokenizer,
                           actor_index, ti, si, positions)
        rounds_already = 0
        if offset == 0:
            for job, receipt in zip(jobs, orphan):
                rendered = agent.tokenizer.apply_chat_template(
                    job["chat"], tokenize=False, add_generation_prompt=True)
                if sha256(rendered.encode()).hexdigest() != \
                        receipt["prompt_sha256"]:
                    raise ValueError("persisted incomplete prompt differs")
                experiment.consume(job["method"], job, receipt, u, query)
            rounds_already = 1
        found = _complete_jobs(
            jobs, {task: query}, agent, expected_dir, u, waits,
            rounds_already=rounds_already)
        annotated = _finish(found, info, args.seed,
                            trajectory["trajectory"], si, metadata)
        sessions.extend(annotated)
        with (expected_dir / "sessions.jsonl").open("a") as stream:
            for row in annotated:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    agent.close()
    actor_summary = {
        "actor_seed": args.seed,
        "sessions": len(sessions),
        "logical_remote_generations": agent.calls,
        "physical_http_attempts": sum(
            turn.get("network_attempts", 1)
            for row in sessions for turn in row["generations"]
            if turn.get("generation_uid")),
        "method_batch_position_counts": dict(positions),
        "receipt_resume": True,
        "recovery_403_waits": waits,
    }
    adapter.write(expected_dir, "summary.json", actor_summary)

    combined, actors = [], []
    for seed in protocol["actor_seeds"]:
        directory = args.results / f"executor_factorial_seed_{seed}"
        combined.extend(rows(directory / "sessions.jsonl"))
        actors.append(read(directory / "summary.json"))
    expected = (len(protocol["actor_seeds"]) * len(metadata) *
                len(experiment.METHODS))
    if len(combined) != expected:
        raise ValueError("combined session count differs")
    with (args.results / "confirmation_sessions.jsonl").open("w") as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = experiment.summarize(combined)
    statistics = experiment.paired_statistics(combined)
    breakdown = experiment.task_breakdown(combined)
    adapter.write(args.results, "summary.json", {
        "confirmation_table": table,
        "paired_statistics": statistics,
        "task_breakdown": breakdown,
        "actor_summaries": actors,
        "frozen_selection": read(args.results / "frozen_selection.json"),
    })
    with (args.results / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (args.results / "task_family_breakdown.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
        writer.writeheader()
        writer.writerows(breakdown)

    abandoned = sorted(args.results.glob(
        f"executor_factorial_seed_{args.seed}_aborted_403_retry*"))
    abandoned_ledgers = [rows(path / "generations.jsonl")
                         for path in abandoned]
    abandoned_logged = sum(row.get("network_attempts", 1)
        for ledger_rows in abandoned_ledgers for row in ledger_rows)
    primary_logical = sum(actor["logical_remote_generations"] for actor in actors)
    primary_physical = sum(actor["physical_http_attempts"] for actor in actors)
    failed_batches = 3
    recovery = {
        "reason": "three HTTP 403 interrupted batches",
        "resumed_from_persisted_receipts": True,
        "persisted_prefix_generations": len(ledger),
        "persisted_prefix_complete_sessions": 40,
        "abandoned_retry_directories": [path.name for path in abandoned],
        "abandoned_retry_logged_attempts": abandoned_logged,
        "failed_batches_without_receipts": failed_batches,
        "unlogged_attempts_lower_bound": failed_batches,
        "unlogged_attempts_upper_bound": failed_batches * 8 * 3,
        "recovery_waits": waits,
        "recovery_script_sha256": digest(Path(__file__)),
        "frozen_experiment_source_unchanged": True,
    }
    adapter.write(args.results, "recovery_audit.json", recovery)
    adapter.write(args.results, "audit.json", {
        "sessions": len(combined), "expected_sessions": expected,
        "logical_remote_generations": primary_logical,
        "physical_http_attempts": primary_physical,
        "abandoned_retry_logged_attempts": abandoned_logged,
        "all_physical_attempts_lower_bound": primary_physical +
            abandoned_logged + failed_batches,
        "all_physical_attempts_upper_bound": primary_physical +
            abandoned_logged + failed_batches * 8 * 3,
        "credentials_persisted": False,
        "all_tasks_share_identical_tools": True,
        "all_registered_sessions_present": True,
        "recovery_audit": "recovery_audit.json",
    })
    print(json.dumps({
        "resumed": True, "sessions": len(combined),
        "logical_remote_generations": primary_logical,
        "physical_http_attempts": primary_physical,
        "recovery": recovery,
        "confirmation_sha256": digest(
            args.results / "confirmation_sessions.jsonl"),
    }, indent=2))


if __name__ == "__main__":
    main()
