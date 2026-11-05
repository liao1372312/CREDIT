"""Randomized multi-tool memory calibration for the ToolSandbox-M follow-up."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import dill
import numpy as np

from .remote_json_agent import RemoteJSONAgent
from .toolsandbox_local_v3 import install_local_clock, run_batch, upstream_imports, write
from .toolsandbox_m import continuous_score, note_relevance
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .user_data import digest


def placebo(note: dict, slot: int, tokenizer) -> dict:
    """Build a masked null note with exactly the candidate's token count."""
    target = int(note["tokens"])
    text = "No relevant prior experience is available."
    while len(tokenizer.encode(text, add_special_tokens=False)) < target:
        # A leading-space English token adds one token in the frozen actor
        # tokenizer, which lets us match every source note exactly.
        text += " neutral"
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    if actual != target:
        raise ValueError(f"could not construct {target}-token placebo (got {actual})")
    return {
        # Keep the candidate UID and tool tags so ordering, eligibility, and
        # exposure metadata are invariant across treatment assignments.
        "uid": note["uid"], "text": text,
        "tools": list(note["tools"]), "tokens": actual,
        "source_task": f"placebo_slot_{slot}", "source_ending_sha256": "none",
        "source_verifier_success": False, "source_exact_milestone_verified": False,
        "source_upstream_score": 0.0, "arrival": f"placebo_slot_{slot}",
        "memory_schema": "role_matched_null_v1",
    }


def candidates(pool: list[dict], task: str, scenario, k: int = 2) -> list[dict]:
    allowed = set(scenario.starting_context.tool_allow_list or [])
    supported = [note for note in pool if set(note["tools"]) & allowed]
    return sorted(supported, key=lambda note: (-note_relevance(note, task, allowed),
                                                note["arrival"], note["uid"]))[:k]


def coefficients(rows: list[dict]) -> dict:
    report = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["actor_seed"], row["task"])].append(row)
    for (seed, task), values in sorted(grouped.items()):
        uids = values[0]["candidate_uids"]
        if len(values) != 2 ** len(uids):
            raise ValueError(f"incomplete factorial for {seed}/{task}")
        result = {"actor_seed": seed, "task": task, "candidate_uids": uids}
        for metric, getter in (("continuous", continuous_score),
                               ("strict", lambda row: float(row["success"]))):
            y = np.asarray([getter(row) for row in values], dtype=float)
            z = np.asarray([row["assignment"] for row in values], dtype=float)
            estimate = {"mean": float(np.mean(y))}
            for index in range(len(uids)):
                estimate[f"effect_{index}"] = float(2 * np.mean(y * z[:, index]))
            if len(uids) == 2:
                estimate["interaction_01"] = float(2 * np.mean(y * z[:, 0] * z[:, 1]))
            result[metric] = estimate
        report.append(result)
    return {"context_coefficients": report}


def aggregate(contexts: list[dict], pool: list[dict]) -> dict:
    by_uid = defaultdict(list)
    by_pair = defaultdict(list)
    uid_task = {note["uid"]: note["source_task"] for note in pool}
    for row in contexts:
        for index, uid in enumerate(row["candidate_uids"]):
            by_uid[uid].append(row["continuous"][f"effect_{index}"])
        if len(row["candidate_uids"]) == 2:
            a, b = row["candidate_uids"]
            by_pair[tuple(sorted((a, b)))].append(row["continuous"]["interaction_01"])
    memories = {}
    for uid, values in sorted(by_uid.items()):
        array = np.asarray(values, dtype=float)
        se = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else float("inf")
        mean = float(np.mean(array))
        lower = mean - 1.96 * se if np.isfinite(se) else float("-inf")
        upper = mean + 1.96 * se if np.isfinite(se) else float("inf")
        status = "beneficial" if lower > 0.005 else ("harmful" if upper < -0.005 else "unresolved")
        memories[uid] = {"source_task": uid_task[uid], "contexts": len(values),
            "mean_effect": mean, "se": se, "ci95": [lower, upper], "status": status,
            "effects": values}
    pairs = {}
    for key, values in sorted(by_pair.items()):
        array = np.asarray(values, dtype=float)
        se = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else float("inf")
        mean = float(np.mean(array))
        pairs["|".join(key)] = {"contexts": len(values), "mean_interaction": mean,
            "se": se, "ci95": [mean - 1.96 * se, mean + 1.96 * se]
            if np.isfinite(se) else [float("-inf"), float("inf")], "values": values}
    return {"memories": memories, "pairs": pairs,
            "decision_rule": "beneficial iff normal lower 95% bound > .005; harmful iff upper < -.005"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/toolsandbox_m_followup_v1"))
    parser.add_argument("--output", type=Path, default=Path("results/toolsandbox_m_factorial_training_v1"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--actor-seeds", type=int, nargs="+", default=[827, 829])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-contexts", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True); (args.output / "source").mkdir()
    source_protocol = json.loads((args.source / "frozen_protocol.json").read_text())
    tokenizer = args.tokenizer or Path(json.loads(
        (args.source / "training_pool_seed_701" / "agent_manifest.json").read_text())["tokenizer_path"])
    scenarios = dill.loads((args.source / "scenarios.pkl").read_bytes())
    pool = json.loads((args.source / "memory_pool.json").read_text())
    development = source_protocol["development"]
    contexts = [{"trajectory": row["trajectory"], "session_index": session_index,
                 "task": task}
                for row in development
                for session_index, task in enumerate(row["sessions"])]
    if args.limit_contexts is not None:
        contexts = contexts[:args.limit_contexts]
    for context in contexts:
        context["candidate_uids"] = [note["uid"] for note in
            candidates(pool, context["task"], scenarios[context["task"]])]
    skipped_contexts = [row for row in contexts if not row["candidate_uids"]]
    contexts = [row for row in contexts if row["candidate_uids"]]
    upstream = Path(source_protocol["upstream_directory"])
    u = upstream_imports(upstream); install_local_clock(source_protocol["clock"])
    names = ["toolsandbox_m_factorial_training.py", "toolsandbox_m_factorial_training_verify.py",
             "toolsandbox_m_followup.py", "toolsandbox_local_v3.py", "remote_json_agent.py"]
    hashes = {name: digest(Path(__file__).with_name(name)) for name in names}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    protocol = {"suite": "ToolSandbox-M multi-tool randomized memory calibration",
        "source_results": str(args.source.resolve()),
        "source_protocol_sha256": digest(args.source / "frozen_protocol.json"),
        "source_memory_pool_sha256": digest(args.source / "memory_pool.json"),
        "source_scenarios_sha256": digest(args.source / "scenarios.pkl"),
        "source_sha256": hashes, "upstream_directory": str(upstream),
        "upstream_source_sha256": source_protocol["upstream_source_sha256"],
        "clock": source_protocol["clock"], "development": development,
        "selected_contexts": contexts,
        "actor_seeds": args.actor_seeds, "temperature": 0.0,
        "skipped_contexts": skipped_contexts,
        "candidate_memories_per_context": "one or two compatible memories",
        "assignment_order_by_size": {"1": [[-1], [1]],
            "2": [[-1, -1], [1, -1], [-1, 1], [1, 1]]},
        "placebo": "role-matched null text with exact candidate token count; both reserved slots always occupied",
        "ordering": "context and coalition order independently shuffled from each actor seed",
        "outcomes": ["official continuous similarity", "exact strict VSR"],
        "confirmation_outcomes_used": False, "maximum_tool_calls_per_session": 8,
        "provider_credentials_persisted": False}
    write(args.output, "frozen_protocol.json", protocol)
    (args.output / "scenarios.pkl").write_bytes((args.source / "scenarios.pkl").read_bytes())
    write(args.output, "memory_pool.json", pool)

    all_rows = []
    for seed in args.actor_seeds:
        directory = args.output / f"actor_seed_{seed}"; directory.mkdir()
        agent = RemoteJSONAgent(directory, seed, env=args.env, tokenizer_path=tokenizer,
            temperature=0.0, max_new_tokens=384, workers=args.workers,
            exact_prompt_cache=False)
        agent.binding_interface = "tool_json"
        agent.manifest["benchmark"] = protocol["suite"]
        actor_rows = []
        rng = np.random.default_rng(seed)
        context_order = list(range(len(contexts))); rng.shuffle(context_order)
        for context_index in context_order:
                context = contexts[context_index]
                task = context["task"]
                selected = candidates(pool, task, scenarios[task])
                if not selected or len(selected) > 2:
                    raise ValueError(f"invalid compatible candidate count for {task}")
                if context["candidate_uids"] != [note["uid"] for note in selected]:
                    raise ValueError(f"candidate set changed for {task}")
                nulls = [placebo(note, slot, agent.tokenizer)
                         for slot, note in enumerate(selected)]
                assignments = protocol["assignment_order_by_size"][str(len(selected))]
                coalition_order = list(range(len(assignments))); rng.shuffle(coalition_order)
                jobs = []
                assignment_by_method = {}
                for assignment_index in coalition_order:
                    assignment = assignments[assignment_index]
                    bank = [selected[index] if assignment[index] == 1 else nulls[index]
                            for index in range(len(selected))]
                    method = "factorial_" + "".join(
                        "1" if value == 1 else "0" for value in assignment)
                    assignment_by_method[method] = assignment
                    jobs.append(initial_job_v2(task, scenarios[task], bank, u, agent.tokenizer,
                        "multi_tool_factorial_development", method))
                # run_batch emits completed jobs in completion order, so join
                # treatments by their immutable method label rather than index.
                for row in run_batch(jobs, agent, directory, u):
                    assignment = assignment_by_method[row["method"]]
                    item = {**row, "actor_seed": seed,
                        "trajectory": context["trajectory"],
                        "session_index": context["session_index"],
                        "context_index": context_index,
                        "candidate_uids": [note["uid"] for note in selected],
                        "assignment": assignment, "remote_latency_ms": session_latency(row)}
                    actor_rows.append(item); all_rows.append(item)
                    with (directory / "sessions.jsonl").open("a") as stream:
                        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        agent.close()
        write(directory, "summary.json", {"actor_seed": seed, "sessions": len(actor_rows),
            "remote_requests": agent.calls})
    with (args.output / "sessions.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    coefficient_report = coefficients(all_rows)
    coefficient_report["aggregate"] = aggregate(coefficient_report["context_coefficients"], pool)
    write(args.output, "effects.json", coefficient_report)
    expected_sessions = len(args.actor_seeds) * sum(2 ** len(row["candidate_uids"])
                                                     for row in contexts)
    summary = {"actors": len(args.actor_seeds), "contexts": len(contexts),
        "skipped_contexts": len(skipped_contexts),
        "sessions": len(all_rows), "remote_requests": sum(json.loads(
            (args.output / f"actor_seed_{seed}" / "summary.json").read_text())["remote_requests"]
            for seed in args.actor_seeds),
        "beneficial_memories": sum(row["status"] == "beneficial" for row in
            coefficient_report["aggregate"]["memories"].values()),
        "harmful_memories": sum(row["status"] == "harmful" for row in
            coefficient_report["aggregate"]["memories"].values()),
        "unresolved_memories": sum(row["status"] == "unresolved" for row in
            coefficient_report["aggregate"]["memories"].values()),
        "mean_remote_latency_ms": float(np.mean([row["remote_latency_ms"] for row in all_rows])),
        "calibration_remote_latency_ms": float(sum(row["remote_latency_ms"] for row in all_rows))}
    write(args.output, "summary.json", summary)
    write(args.output, "audit.json", {"passed": len(all_rows) == expected_sessions,
        "expected_sessions": expected_sessions, "all_verification_covered": all(
        row["verification_coverage"] for row in all_rows),
        "sessions_sha256": digest(args.output / "sessions.jsonl")})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
