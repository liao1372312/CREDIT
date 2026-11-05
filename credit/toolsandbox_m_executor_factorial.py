"""Registered memory-by-executor factorial on ToolSandbox-M.

This suite reuses the frozen mechanism-blind tasks to attribute their observed
saturation. It crosses two memory arms with four nested executor stages while
holding the actor, prompt, tool catalogue, state, and limits fixed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
from time import perf_counter_ns

import numpy as np
from scipy.stats import binomtest

from . import toolsandbox_local_v3 as adapter
from .remote_json_agent import RemoteJSONAgent
from . import toolsandbox_m_contract_fast_v3_confirmation as scenario_support
from . import toolsandbox_m_mechanism_blind as source_suite
from .toolsandbox_m_executor_stages import STAGES, consume_stage
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_intent_optimization import balanced_order
from .toolsandbox_m_scoped_compose_selector import scoped_compose_select
from .toolsandbox_m_selector_dependency_audit import closure
from .user_data import digest


CLOCK = source_suite.CLOCK
ACTOR_SEEDS = (3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517)
MEMORY_ARMS = ("NoMem", "ScopedCompose")
METHODS = tuple(f"{memory}/{stage}" for stage in STAGES
                for memory in MEMORY_ARMS)
LATENCY_BUDGETS_MS = (10000, 15000, 20000, 30000)
MAX_MODEL_TURNS = 12
MAX_TOOL_CALLS = 12
BOOTSTRAP_SEED = 20261103
BOOTSTRAP_REPLICATES = 20000
PLAN = Path("CREDIT_V4_EXECUTOR_FACTORIAL_PLAN.md")


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def method_parts(method: str) -> tuple[str, str]:
    memory, stage = method.split("/", 1)
    if memory not in MEMORY_ARMS or stage not in STAGES:
        raise ValueError("unknown factorial method: " + method)
    return memory, stage


def select(method: str, pool: list[dict], query: str, scenario) -> list[dict]:
    memory, _ = method_parts(method)
    if memory == "NoMem":
        return []
    return scoped_compose_select(pool, query, scenario)


def consume(method: str, job: dict, generation: dict, u,
            query: str) -> None:
    _, stage = method_parts(method)
    consume_stage(stage, job, generation, u, MAX_TOOL_CALLS, query)


def run_batch(jobs, queries, agent, directory, u):
    rows, active = [], jobs
    for _ in range(MAX_MODEL_TURNS):
        if not active:
            break
        records = agent.infer([job["chat"] for job in active])
        remaining = []
        for job, record in zip(active, records):
            consume(job["method"], job, record, u, queries[job["task"]])
            if job["done"]:
                rows.append(scenario_support.prior.finish_direct(
                    job, directory, u))
            else:
                remaining.append(job)
        active = remaining
    for job in active:
        job["termination"] = "turn_budget"
        rows.append(scenario_support.prior.finish_direct(job, directory, u))
    return rows


def _latency(row: dict) -> tuple[float, float, float]:
    remote = session_latency(row)
    local = float(row.get("selection_latency_ms", 0.0)) + sum(
        float(turn.get("local_consume_latency_ms", 0.0))
        for turn in row["generations"])
    return remote, local, remote + local


def run_actor(seed: int, actor_index: int, trajectories: list[dict],
              root: Path, args, scenarios, metadata, pool, u):
    directory = root / f"executor_factorial_seed_{seed}"
    directory.mkdir()
    agent = RemoteJSONAgent(
        directory, seed, env=args.env, tokenizer_path=args.tokenizer,
        temperature=0.0, max_new_tokens=384, workers=args.workers,
        exact_prompt_cache=False)
    agent.manifest["benchmark"] = "ToolSandbox-M executor component factorial"
    sessions, positions = [], defaultdict(lambda: [0] * len(METHODS))
    for trajectory_index, trajectory in enumerate(trajectories):
        for session_index, task in enumerate(trajectory["sessions"]):
            scenario = scenarios[task]
            query = metadata[task]["public_request"]
            order = balanced_order(list(METHODS), actor_index,
                                   trajectory_index, session_index)
            jobs, info = [], {}
            for position, method in enumerate(order):
                positions[method][position] += 1
                began = perf_counter_ns()
                bank = select(method, pool, query, scenario)
                selection_ms = (perf_counter_ns() - began) / 1e6
                job = initial_job_v2(
                    task, scenario, bank, u, agent.tokenizer,
                    "executor_factorial_confirmation", method)
                job["selection_latency_ms"] = selection_ms
                jobs.append(job)
                info[method] = (bank, position)
            for row in run_batch(jobs, {task: query}, agent, directory, u):
                bank, position = info[row["method"]]
                remote, local, accounted = _latency(row)
                memory, stage = method_parts(row["method"])
                item = source_suite.annotate({
                    **row,
                    "actor_seed": seed,
                    "trajectory": trajectory["trajectory"],
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
                    "public_request_sha256": _sha(query),
                }, metadata)
                sessions.append(item)
                with (directory / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    agent.close()
    summary = {
        "actor_seed": seed,
        "sessions": len(sessions),
        "logical_remote_generations": agent.calls,
        "physical_http_attempts": sum(
            turn.get("network_attempts", 1)
            for row in sessions for turn in row["generations"]
            if turn.get("generation_uid")),
        "method_batch_position_counts": dict(positions),
    }
    adapter.write(directory, "summary.json", summary)
    return sessions, summary


def summarize(rows: list[dict]) -> list[dict]:
    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        latency = np.asarray([row["accounted_latency_ms"] for row in subset])
        turns = [turn for row in subset for turn in row["generations"]]
        item = {
            "method": method,
            "sessions": len(subset),
            "requested_state_VSR": float(np.mean([
                row["requested_state_success"] for row in subset])),
            "process_state_VSR": float(np.mean([
                row["process_state_success"] for row in subset])),
            "mean_requested_state_score": float(np.mean([
                row["requested_state_score"] for row in subset])),
            "mean_selected_records": float(np.mean([
                len(row["selected_bank"]) for row in subset])),
            "mean_tool_calls": float(np.mean([
                row["tool_calls"] for row in subset])),
            "mean_remote_requests": float(np.mean([sum(
                "remote_latency_ms" in turn for turn in row["generations"])
                for row in subset])),
            "mean_accounted_latency_ms": float(np.mean(latency)),
            "p95_accounted_latency_ms": float(np.quantile(latency, .95)),
            "tool_call_exceptions": sum(bool(call.get("exception"))
                for row in subset for call in row["calls"]),
            "validator_rejections": sum(
                "contract_violation" in turn for turn in turns),
            "deterministic_repairs": sum(
                "deterministic_repair" in turn for turn in turns),
            "local_completions": sum(
                row["termination"] == "local_contract_complete"
                for row in subset),
        }
        for budget in LATENCY_BUDGETS_MS:
            item[f"requested_state_VSR_at_{budget}ms"] = float(np.mean([
                row["requested_state_success"] and value <= budget
                for row, value in zip(subset, latency)]))
        table.append(item)
    return table


def _sign(values: np.ndarray) -> dict:
    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    return {
        "wins": wins,
        "losses": losses,
        "ties": int(len(values) - wins - losses),
        "two_sided_exact_p": float(binomtest(
            wins, wins + losses, .5, alternative="two-sided").pvalue)
            if wins + losses else 1.0,
    }


def _crossed_bootstrap(matrix: np.ndarray, rng) -> list[float]:
    actors, families = matrix.shape
    values = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        actor_draw = rng.integers(0, actors, actors)
        family_draw = rng.integers(0, families, families)
        values[index] = float(np.mean(
            matrix[np.ix_(actor_draw, family_draw)]))
    return np.quantile(values, [.025, .975]).tolist()


def _effect_matrix(by, treatment: str, baseline: str, actors, families):
    matrix = np.empty((len(actors), len(families)))
    for ai, actor in enumerate(actors):
        for fi, family in enumerate(families):
            keys = [(actor, family, session) for session in (0, 1)]
            matrix[ai, fi] = np.mean([
                by[treatment][key]["requested_state_success"] -
                by[baseline][key]["requested_state_success"]
                for key in keys])
    return matrix


def paired_statistics(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for row in rows:
        key = (row["actor_seed"], row["trajectory"], row["session_index"])
        by[row["method"]][key] = row
    actors = sorted({row["actor_seed"] for row in rows})
    families = sorted({row["trajectory"] for row in rows})
    specifications = []
    for stage in STAGES:
        specifications.append({
            "name": f"history_effect_{stage}",
            "treatment": f"ScopedCompose/{stage}",
            "baseline": f"NoMem/{stage}",
            "estimand": f"scoped history effect under {stage}",
        })
    for lower, upper in zip(STAGES, STAGES[1:]):
        specifications.append({
            "name": f"executor_gain_{lower}_to_{upper}",
            "treatment": f"NoMem/{upper}",
            "baseline": f"NoMem/{lower}",
            "estimand": f"incremental executor effect of {upper} after {lower}",
        })
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = []
    matrices = {}
    for spec in specifications:
        matrix = _effect_matrix(
            by, spec["treatment"], spec["baseline"], actors, families)
        matrices[spec["name"]] = matrix
        actor_effects = np.mean(matrix, axis=1)
        family_effects = np.mean(matrix, axis=0)
        comparisons.append({
            **spec,
            "delta_requested_state_VSR": float(np.mean(matrix)),
            "crossed_actor_family_bootstrap_CI95":
                _crossed_bootstrap(matrix, rng),
            "actor_level_sign_test": _sign(actor_effects),
            "family_level_sign_sensitivity": _sign(family_effects),
            "actor_effects": actor_effects.tolist(),
            "family_effects": dict(zip(families, family_effects.tolist())),
        })
    interaction = (matrices["history_effect_Full"] -
                   matrices["history_effect_Base"])
    actor_effects = np.mean(interaction, axis=1)
    family_effects = np.mean(interaction, axis=0)
    comparisons.append({
        "name": "history_by_executor_interaction_Full_minus_Base",
        "treatment": "(ScopedCompose-NoMem)/Full",
        "baseline": "(ScopedCompose-NoMem)/Base",
        "estimand": "change in the history effect from Base to Full",
        "delta_requested_state_VSR": float(np.mean(interaction)),
        "crossed_actor_family_bootstrap_CI95":
            _crossed_bootstrap(interaction, rng),
        "actor_level_sign_test": _sign(actor_effects),
        "family_level_sign_sensitivity": _sign(family_effects),
        "actor_effects": actor_effects.tolist(),
        "family_effects": dict(zip(families, family_effects.tolist())),
    })
    order = sorted(range(len(comparisons)), key=lambda index:
        comparisons[index]["actor_level_sign_test"]["two_sided_exact_p"])
    running = 0.0
    for rank, index in enumerate(order):
        raw = comparisons[index]["actor_level_sign_test"]["two_sided_exact_p"]
        running = max(running, min(1.0, (len(order) - rank) * raw))
        comparisons[index]["holm_actor_p"] = running
    return {
        "comparisons": comparisons,
        "actors": len(actors),
        "families": len(families),
        "prompts_per_actor_family": 2,
        "primary_resampling": "crossed actor-by-family bootstrap",
        "primary_test_unit": "actor-level paired mean",
        "tests": "two-sided exact sign tests",
        "holm_family_size": len(comparisons),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def task_breakdown(rows: list[dict]) -> list[dict]:
    output = []
    for family in sorted({row["trajectory"] for row in rows}):
        for method in METHODS:
            subset = [row for row in rows if row["trajectory"] == family
                      and row["method"] == method]
            output.append({
                "family": family,
                "method": method,
                "sessions": len(subset),
                "requested_state_VSR": float(np.mean([
                    row["requested_state_success"] for row in subset])),
            })
    return output


def task_identity_preflight(metadata: dict, upstream_protocol: dict) -> dict:
    frozen = {row["name"]: row for row in upstream_protocol["tasks"]}
    current = {
        name: {
            "family": item["family"],
            "public_request_sha256": _sha(item["public_request"]),
            "required_mutations": item["required_mutations"],
            "mechanism_signature": item["mechanism_signature"],
        }
        for name, item in metadata.items()
    }
    expected = {
        name: {key: row[key] for key in current[name]}
        for name, row in frozen.items()
    }
    return {
        "passed": current == expected,
        "tasks": len(current),
        "source_suite": upstream_protocol["suite"],
    }


def freeze(args, metadata, trajectories, preflight, upstream_protocol) -> dict:
    modules = closure({"toolsandbox_m_executor_factorial",
                       "toolsandbox_m_executor_factorial_verify"})
    source_dir = args.output / "source"
    source_dir.mkdir()
    source_sha = {}
    for module in sorted(modules):
        name = module + ".py"
        source = Path(__file__).with_name(name)
        shutil.copy2(source, source_dir / name)
        source_sha[name] = digest(source)
    protocol = {
        "suite": "ToolSandbox-M memory-by-executor component factorial",
        "frozen_before_remote_calls": True,
        "clock": CLOCK,
        "actor_seeds": list(args.seeds),
        "memory_arms": list(MEMORY_ARMS),
        "executor_stages": list(STAGES),
        "methods": list(METHODS),
        "primary_endpoint": "exact requested-world-state VSR",
        "registered_comparisons": (
            "four stage-specific history effects, three adjacent NoMem executor "
            "gains, and Full-minus-Base history-effect interaction"),
        "multiplicity": "Holm over eight two-sided actor-level sign tests",
        "uncertainty": "crossed actor-by-family bootstrap",
        "maximum_model_turns_per_session": MAX_MODEL_TURNS,
        "maximum_tool_calls_per_session": MAX_TOOL_CALLS,
        "exact_prompt_cache": False,
        "provider_credentials_persisted": False,
        "tasks_reused_for_component_attribution": True,
        "external_validity_claim": False,
        "tasks": [{
            "name": name,
            "family": metadata[name]["family"],
            "public_request_sha256": _sha(metadata[name]["public_request"]),
            "required_mutations": metadata[name]["required_mutations"],
            "mechanism_signature": metadata[name]["mechanism_signature"],
        } for name in sorted(metadata)],
        "trajectories": trajectories,
        "memory_pool_source": str(args.protocol_source),
        "memory_pool_source_sha256": digest(
            args.protocol_source / "memory_pool.json"),
        "source_closure_complete": True,
        "source_sha256": source_sha,
        "plan_sha256": digest(PLAN),
        "upstream_protocol_sha256": digest(
            args.protocol_source / "frozen_protocol.json"),
        "upstream_directory": upstream_protocol["upstream_directory"],
        "upstream_source_sha256": upstream_protocol["upstream_source_sha256"],
        "preflight": preflight,
    }
    adapter.write(args.output, "frozen_protocol.json", protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(ACTOR_SEEDS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--protocol-source", type=Path, default=Path(
        "results/toolsandbox_m_mechanism_blind_v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    upstream_protocol = json.loads(
        (args.protocol_source / "frozen_protocol.json").read_text())
    u = adapter.upstream_imports(Path(upstream_protocol["upstream_directory"]))
    adapter.install_local_clock(CLOCK)
    scenarios, metadata, trajectories = source_suite.build_scenarios(u)
    pool = json.loads((args.protocol_source / "memory_pool.json").read_text())
    shutil.copy2(args.protocol_source / "memory_pool.json",
                 args.output / "memory_pool.json")
    preflight = {
        "task_identity": task_identity_preflight(metadata, upstream_protocol),
        "golden_evaluator": source_suite.golden_self_check(
            u, scenarios, metadata),
        "argument_extraction": source_suite.argument_extraction_self_check(
            metadata),
    }
    if not all(item.get("passed") for item in preflight.values()):
        raise ValueError("factorial preflight failed")
    adapter.write(args.output, "preflight.json", preflight)
    freeze(args, metadata, trajectories, preflight, upstream_protocol)

    first_manifest = next(args.protocol_source.glob(
        "mechanism_blind_seed_*/agent_manifest.json"))
    manifest = json.loads(first_manifest.read_text())
    args.tokenizer = args.tokenizer or Path(manifest["tokenizer_path"])
    selection = {}
    for task, scenario in scenarios.items():
        query = metadata[task]["public_request"]
        selection[task] = {}
        for method in METHODS:
            bank = select(method, pool, query, scenario)
            selection[task][method] = {
                "selected_uids": [note["uid"] for note in bank],
                "memory_schemas": [note.get("memory_schema") for note in bank],
                "tokens": sum(note["tokens"] for note in bank),
            }
    adapter.write(args.output, "frozen_selection.json", selection)
    if args.prepare_only:
        print(json.dumps({
            "prepared": True,
            "tasks": len(metadata),
            "methods": len(METHODS),
            "source_modules": len(json.loads(
                (args.output / "frozen_protocol.json").read_text())[
                    "source_sha256"]),
            "preflight": preflight,
        }, indent=2))
        return

    rows, actors = [], []
    for index, seed in enumerate(args.seeds):
        found, actor = run_actor(seed, index, trajectories, args.output, args,
                                 scenarios, metadata, pool, u)
        rows.extend(found)
        actors.append(actor)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = summarize(rows)
    statistics = paired_statistics(rows)
    breakdown = task_breakdown(rows)
    summary = {
        "confirmation_table": table,
        "paired_statistics": statistics,
        "task_breakdown": breakdown,
        "actor_summaries": actors,
        "frozen_selection": selection,
    }
    adapter.write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (args.output / "task_family_breakdown.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
        writer.writeheader()
        writer.writerows(breakdown)
    adapter.write(args.output, "audit.json", {
        "sessions": len(rows),
        "expected_sessions": len(args.seeds) * len(metadata) * len(METHODS),
        "logical_remote_generations": sum(
            actor["logical_remote_generations"] for actor in actors),
        "physical_http_attempts": sum(
            actor["physical_http_attempts"] for actor in actors),
        "credentials_persisted": False,
        "all_tasks_share_identical_tools": True,
        "all_registered_sessions_present": len(rows) ==
            len(args.seeds) * len(metadata) * len(METHODS),
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
