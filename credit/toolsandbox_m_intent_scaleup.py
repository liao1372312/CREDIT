"""Prospective actor-seed scale-up of the frozen ToolSandbox-M intent selector.

The selector and semantic scenarios are inherited without modification from the
post-hoc intent experiment.  This run increases the number of independent
actor-by-trajectory clusters; it does not claim new task-family coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import dill

from .toolsandbox_local_v3 import install_local_clock, upstream_imports, write
from .toolsandbox_m_intent_optimization import (
    LATENCY_BUDGETS_MS,
    METHODS,
    PROMPT_PREFIX,
    intent_select,
    original_request,
    paired_statistics,
    prompt_variant,
    requirements,
    run_actor,
    summarize,
)
from .user_data import digest


DEFAULT_SEEDS = [
    76001, 76003, 76031, 76039, 76079,
    76081, 76091, 76099, 76103, 76123,
    76129, 76147, 76157, 76159, 76163,
    76207, 76213, 76231, 76243, 76249,
]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def actor_complete(directory: Path, expected_sessions: int) -> bool:
    paths = [directory / "summary.json", directory / "sessions.jsonl",
             directory / "generations.jsonl", directory / "agent_manifest.json"]
    if not all(path.exists() for path in paths):
        return False
    summary = json.loads((directory / "summary.json").read_text())
    return (summary.get("sessions") == expected_sessions and
            len(load_rows(directory / "sessions.jsonl")) == expected_sessions)


def initialize(args, old_protocol: dict, old_summary: dict,
               trajectories: list[dict]) -> None:
    args.output.mkdir(parents=True)
    (args.output / "source").mkdir()
    names = [
        "toolsandbox_m_intent_scaleup.py",
        "toolsandbox_m_intent_optimization.py",
        "toolsandbox_m_intent_optimization_verify.py",
        "toolsandbox_m_streaming_optimization.py",
        "toolsandbox_m_followup.py",
        "toolsandbox_local_v3.py",
        "toolsandbox_m.py",
        "remote_json_agent.py",
    ]
    hashes = {name: digest(Path(__file__).with_name(name)) for name in names}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    protocol = {
        "suite": "ToolSandbox-M frozen intent actor-seed scale-up",
        "design_status": "frozen before any scale-up remote request",
        "source_results": str(args.source.resolve()),
        "prior_intent_results": str(args.prior_intent.resolve()),
        "source_memory_pool_sha256": digest(args.source / "memory_pool.json"),
        "source_scenarios_sha256": digest(args.source / "scenarios.pkl"),
        "source_sha256": hashes,
        "upstream_directory": old_protocol["upstream_directory"],
        "upstream_source_sha256": old_protocol["upstream_source_sha256"],
        "clock": old_protocol["clock"],
        "trajectories": trajectories,
        "actor_seeds": args.seeds,
        "reserve_actor_seeds": [76253, 76259, 76261, 76283, 76289],
        "methods": list(METHODS),
        "primary_endpoint": "strict verified success rate",
        "primary_comparisons": ["CREDIT-Intent vs FIFO",
                                "CREDIT-Intent vs Semantic"],
        "multiplicity": "Holm correction across the two primary comparisons",
        "resampling_unit": "actor seed by two-session trajectory",
        "sample_size": {
            "actors": len(args.seeds),
            "trajectories_per_actor": len(trajectories),
            "sessions_per_trajectory": 2,
            "sessions_per_method": len(args.seeds) * len(trajectories) * 2,
            "paired_clusters": len(args.seeds) * len(trajectories),
        },
        "prompt_transform": {"kind": "literal_prefix", "prefix": PROMPT_PREFIX},
        "selector_frozen_from_prior_posthoc_study": True,
        "prior_results_examined": True,
        "claim_scope": (
            "prospective actor-seed scale-up of a frozen post-hoc selector on "
            "the same semantic scenario families; no new task-family claim"
        ),
        "paired_scheduling": (
            "all methods for a task share one concurrent remote batch"
        ),
        "temperature": 0.0,
        "maximum_tool_calls_per_session": 8,
        "latency_budgets_ms": list(LATENCY_BUDGETS_MS),
        "selector_training_remote_requests": 0,
        "selector_training_remote_latency_ms": 0.0,
        "historical_R_and_D_remote_latency_ms": old_summary[
            "credit_training_latency_ms"],
        "failure_rule": (
            "An actor is included only if all method-task cells complete. "
            "Any incomplete actor directory is retained and the run stops; "
            "a reserve seed requires an explicit exclusion record."
        ),
        "provider_credentials_persisted": False,
    }
    write(args.output, "frozen_protocol.json", protocol)
    (args.output / "scenarios.pkl").write_bytes(
        (args.source / "scenarios.pkl").read_bytes())
    shutil.copy2(args.source / "memory_pool.json", args.output / "memory_pool.json")


def finalize(args, protocol: dict, scenarios, pool: list[dict],
             pair_vetoes: set[tuple[str, str]], u) -> dict:
    rows, actor_summaries = [], []
    for seed in protocol["actor_seeds"]:
        directory = args.output / f"intent_confirmation_seed_{seed}"
        rows.extend(load_rows(directory / "sessions.jsonl"))
        actor_summaries.append(json.loads((directory / "summary.json").read_text()))
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

    tokenizer_path = json.loads((args.output /
        f"intent_confirmation_seed_{protocol['actor_seeds'][0]}" /
        "agent_manifest.json").read_text())["tokenizer_path"]
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True)
    selection = {}
    for trajectory in protocol["trajectories"]:
        for task in trajectory["sessions"]:
            query = prompt_variant(original_request(
                task, scenarios[task], u, tokenizer))
            bank = intent_select(pool, query, scenarios[task])
            selection[task] = {
                "required_and_supporting_tools": [sorted(part) for part in
                    requirements(query, set(scenarios[task].starting_context.
                                            tool_allow_list or []))],
                "selected_uids": [note["uid"] for note in bank],
                "selected_sources": [note["source_task"] for note in bank],
            }
    summary = {
        "confirmation_table": table,
        "online_latency_ranking": ranking,
        "paired_statistics": paired_statistics(rows),
        "intent_selection": selection,
        "incremental_selector_training_remote_latency_ms": 0.0,
        "historical_R_and_D_remote_latency_ms": protocol[
            "historical_R_and_D_remote_latency_ms"],
        "actor_summaries": actor_summaries,
    }
    write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    expected = (len(protocol["actor_seeds"]) * len(protocol["trajectories"])
                * 2 * len(METHODS))
    audit = {
        "passed": (len(rows) == expected and
                   all(row["verification_coverage"] for row in rows)),
        "confirmation_sessions": len(rows),
        "confirmation_sha256": digest(
            args.output / "confirmation_sessions.jsonl"),
    }
    write(args.output, "audit.json", audit)
    return {"summary": summary, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
        default=Path("results/toolsandbox_m_streaming_optimization_v1"))
    parser.add_argument("--prior-intent", type=Path,
        default=Path("results/toolsandbox_m_intent_optimization_v1"))
    parser.add_argument("--output", type=Path,
        default=Path("results/toolsandbox_m_intent_scaleup_v1"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    old_protocol = json.loads((args.source / "frozen_protocol.json").read_text())
    old_summary = json.loads((args.source / "summary.json").read_text())
    trajectories = old_protocol["confirmation"]
    args.tokenizer = args.tokenizer or Path(json.loads((args.source /
        f"stream_confirmation_seed_{old_protocol['confirmation_seeds'][0]}" /
        "agent_manifest.json").read_text())["tokenizer_path"])

    if args.output.exists():
        if not args.resume:
            raise FileExistsError("output exists; pass --resume after inspection")
        protocol = json.loads((args.output / "frozen_protocol.json").read_text())
        if protocol["actor_seeds"] != args.seeds:
            raise ValueError("resume seed list differs from frozen protocol")
    else:
        initialize(args, old_protocol, old_summary, trajectories)
        protocol = json.loads((args.output / "frozen_protocol.json").read_text())

    u = upstream_imports(Path(old_protocol["upstream_directory"]))
    install_local_clock(old_protocol["clock"])
    scenarios = dill.loads((args.source / "scenarios.pkl").read_bytes())
    pool = json.loads((args.source / "memory_pool.json").read_text())
    pair_vetoes = {tuple(pair) for pair in old_protocol["harmful_pair_uids"]}
    expected_actor_sessions = len(trajectories) * 2 * len(METHODS)
    for index, seed in enumerate(args.seeds):
        directory = args.output / f"intent_confirmation_seed_{seed}"
        if directory.exists():
            if actor_complete(directory, expected_actor_sessions):
                print(json.dumps({"actor_seed": seed, "status": "already_complete"}),
                      flush=True)
                continue
            raise RuntimeError(
                f"incomplete actor retained at {directory}; record its exclusion "
                "and use a frozen reserve seed")
        print(json.dumps({"actor_seed": seed, "status": "starting",
                          "actor_index": index}), flush=True)
        run_actor(seed, index, trajectories, args.output, args, scenarios,
                  pool, pair_vetoes, u)
        print(json.dumps({"actor_seed": seed, "status": "complete"}), flush=True)

    report = finalize(args, protocol, scenarios, pool, pair_vetoes, u)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
