"""Recover an interrupted actor in a frozen executor-factorial run.

The interrupted directory is retained as an aborted transport audit. The actor
is rerun from its first registered cell with the original frozen source and
seed; completed actors are never rerun. Aborted requests are reported in total
physical cost but excluded from the balanced primary estimand.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from . import toolsandbox_local_v3 as adapter
from . import toolsandbox_m_executor_factorial as experiment
from .user_data import digest


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    protocol = read(args.results / "frozen_protocol.json")
    if args.seed not in protocol["actor_seeds"]:
        raise ValueError("seed is not registered")
    for name, expected in protocol["source_sha256"].items():
        current = Path(experiment.__file__).with_name(name)
        if digest(current) != expected:
            raise ValueError("frozen source changed: " + name)

    actor = args.results / f"executor_factorial_seed_{args.seed}"
    base_name = f"executor_factorial_seed_{args.seed}_aborted_403"
    aborted = args.results / base_name
    suffix = 1
    while aborted.exists():
        aborted = args.results / f"{base_name}_retry{suffix}"
        suffix += 1
    if not actor.exists():
        raise FileNotFoundError(actor)
    actor.rename(aborted)

    u = adapter.upstream_imports(Path(protocol["upstream_directory"]))
    adapter.install_local_clock(protocol["clock"])
    scenarios, metadata, trajectories = experiment.source_suite.build_scenarios(u)
    pool = read(args.results / "memory_pool.json")
    old_manifest = read(aborted / "agent_manifest.json") \
        if (aborted / "agent_manifest.json").exists() else None
    if old_manifest is None:
        # close() is not reached on an interrupted actor; use the preceding
        # completed actor's immutable tokenizer identity.
        preceding = protocol["actor_seeds"][protocol["actor_seeds"].index(
            args.seed) - 1]
        old_manifest = read(args.results /
            f"executor_factorial_seed_{preceding}" / "agent_manifest.json")
    run_args = SimpleNamespace(
        env=args.env,
        tokenizer=Path(old_manifest["tokenizer_path"]),
        workers=args.workers,
    )
    actor_index = protocol["actor_seeds"].index(args.seed)
    experiment.run_actor(
        args.seed, actor_index, trajectories, args.results, run_args,
        scenarios, metadata, pool, u)

    combined, actor_summaries = [], []
    for seed in protocol["actor_seeds"]:
        directory = args.results / f"executor_factorial_seed_{seed}"
        found = rows(directory / "sessions.jsonl")
        if len(found) != len(metadata) * len(experiment.METHODS):
            raise ValueError(f"incomplete recovered actor: {seed}")
        combined.extend(found)
        actor_summaries.append(read(directory / "summary.json"))
    with (args.results / "confirmation_sessions.jsonl").open("w") as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    table = experiment.summarize(combined)
    statistics = experiment.paired_statistics(combined)
    breakdown = experiment.task_breakdown(combined)
    summary = {
        "confirmation_table": table,
        "paired_statistics": statistics,
        "task_breakdown": breakdown,
        "actor_summaries": actor_summaries,
        "frozen_selection": read(args.results / "frozen_selection.json"),
    }
    adapter.write(args.results, "summary.json", summary)
    with (args.results / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (args.results / "task_family_breakdown.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
        writer.writeheader()
        writer.writerows(breakdown)

    logical = sum(item["logical_remote_generations"]
                  for item in actor_summaries)
    physical = sum(item["physical_http_attempts"]
                   for item in actor_summaries)
    aborted_directories = sorted(args.results.glob(base_name + "*"))
    aborted_ledgers = [rows(path / "generations.jsonl")
                       for path in aborted_directories]
    logged_aborted_attempts = sum(
        row.get("network_attempts", 1)
        for ledger in aborted_ledgers for row in ledger)
    # The failing batch launched eight jobs. The adapter persists a batch only
    # if every job returns, so only a conservative attempt range is knowable.
    failed_batch_jobs = 8 * len(aborted_directories)
    recovery = {
        "reason": "remote provider returned HTTP 403 during actor seed 3517",
        "primary_actor_rerun_from_first_cell": True,
        "aborted_actor_seed": args.seed,
        "aborted_directories": [path.name for path in aborted_directories],
        "aborted_logged_generations": sum(map(len, aborted_ledgers)),
        "aborted_logged_physical_attempts": logged_aborted_attempts,
        "failed_batch_jobs_without_persisted_receipts": failed_batch_jobs,
        "aborted_physical_attempts_lower_bound":
            logged_aborted_attempts + len(aborted_directories),
        "aborted_physical_attempts_upper_bound":
            logged_aborted_attempts + 3 * failed_batch_jobs,
        "aborted_generations_sha256": {
            path.name: digest(path / "generations.jsonl")
            for path in aborted_directories},
        "recovery_script_sha256": digest(Path(__file__)),
        "frozen_experiment_source_unchanged": True,
        "aborted_rows_excluded_from_primary_estimand": True,
    }
    adapter.write(args.results, "recovery_audit.json", recovery)
    adapter.write(args.results, "audit.json", {
        "sessions": len(combined),
        "expected_sessions": (len(protocol["actor_seeds"]) * len(metadata) *
                              len(experiment.METHODS)),
        "logical_remote_generations": logical,
        "physical_http_attempts": physical,
        "aborted_logged_physical_attempts": logged_aborted_attempts,
        "all_physical_attempts_lower_bound": physical +
            recovery["aborted_physical_attempts_lower_bound"],
        "all_physical_attempts_upper_bound": physical +
            recovery["aborted_physical_attempts_upper_bound"],
        "credentials_persisted": False,
        "all_tasks_share_identical_tools": True,
        "all_registered_sessions_present": len(combined) ==
            len(protocol["actor_seeds"]) * len(metadata) *
            len(experiment.METHODS),
        "recovery_audit": "recovery_audit.json",
    })
    print(json.dumps({
        "recovered": True,
        "sessions": len(combined),
        "logical_remote_generations": logical,
        "physical_http_attempts_primary": physical,
        "recovery": recovery,
        "confirmation_sha256": digest(
            args.results / "confirmation_sessions.jsonl"),
    }, indent=2))


if __name__ == "__main__":
    main()
