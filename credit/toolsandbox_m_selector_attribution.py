"""Frozen selector-by-executor attribution on new ToolSandbox-M prompts.

This suite separates historical-memory selection from the argument-aware v3
executor.  It must be run once on a fresh output directory after its source and
protocol have been frozen.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
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
from . import toolsandbox_m_contract_fast as v1
from . import toolsandbox_m_contract_fast_v3 as v3
from . import toolsandbox_m_contract_fast_v3_confirmation as prior
from .toolsandbox_m_compose_selector import compose_select
from .toolsandbox_m_contract_confirmation import novelty_audit
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_intent_optimization import balanced_order, select as legacy_select
from .user_data import digest


CLOCK = prior.CLOCK
ACTOR_SEEDS = (3109, 3119, 3121, 3137, 3163, 3167, 3181, 3187)
METHODS = (
    "NoMem/Base", "CREDIT/Base", "NoMem/v3", "FIFO/v3",
    "Semantic/v3", "CREDIT/v3", "ContractOnly/v3", "Full/v3",
)
LATENCY_BUDGETS_MS = (10000, 15000, 20000, 30000)
PRIMARY_BUDGET_MS = 15000
MAX_MODEL_TURNS = 12
BOOTSTRAP_SEED = 20261023
BOOTSTRAP_REPLICATES = 20000
PRIMARY_CONTRASTS = (
    ("CREDIT/v3", "NoMem/v3", "historical memory under fixed v3 executor"),
    ("CREDIT/v3", "FIFO/v3", "CREDIT versus FIFO under fixed v3 executor"),
    ("CREDIT/v3", "Semantic/v3", "CREDIT versus semantic under fixed v3 executor"),
    ("NoMem/v3", "NoMem/Base", "argument-aware executor with empty memory"),
    ("Full/v3", "CREDIT/v3", "request-derived contract context beyond historical memory"),
)


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def build_confirmation_scenarios(u):
    """Retarget the frozen v3 states with twelve previously unused prompts."""
    old_scenarios, old_metadata, _ = prior.build_confirmation_scenarios(u)
    prompts = {
        "literal_wifi": (
            "Create an archivist contact named 'Rebuild Ledger' at +14155550311; "
            "text it exactly 'replace most recent note'; disable Wi-Fi.",
            "Record +14155550311 under 'Rebuild Ledger' with relationship archivist; "
            "send the exact literal 'replace most recent note'; switch Wi-Fi off.",
        ),
        "literal_cellular": (
            "Create an inspector contact named 'Latest Substitute' at +14155550312; "
            "text it exactly 'recreate upcoming reminder'; disable mobile data.",
            "Record +14155550312 under 'Latest Substitute' with relationship inspector; "
            "send the exact literal 'recreate upcoming reminder'; switch cellular off.",
        ),
        "literal_location": (
            "Create a dispatcher contact named 'Replace Chronicle' at +14155550313; "
            "text it exactly 'swap newest memo'; disable location services.",
            "Record +14155550313 under 'Replace Chronicle' with relationship dispatcher; "
            "send the literal 'swap newest memo'; switch location service off.",
        ),
        "contact_replace_message": (
            "Replace the John Petrucci contact with a contact named 'JP Archive' as "
            "collaborator while preserving its phone; text that number 'Profile moved'; "
            "create a reminder titled 'Review JP record' for tomorrow at 14:10.",
            "Substitute the John Petrucci contact with a contact called 'JP Archive' at "
            "the retained number as collaborator; send 'Profile moved' there; add a "
            "reminder named 'Review JP record' for 2:10 PM tomorrow.",
        ),
        "reminder_replace_boss": (
            "Recreate the upcoming reminder as 'Confirm venue' at the same timestamp; "
            "mark my boss as director; text that contact 'Reminder updated'.",
            "Swap out the next reminder for 'Confirm venue' without changing its time; "
            "set my boss relationship to director; send the boss 'Reminder updated'.",
        ),
        "latest_branch_contact": (
            "Use the latest message unchanged to create a reminder for tomorrow at 07:30; "
            "mark my boss as manager; save a coordinator contact named 'Ops Desk' at "
            "+14155550182; send the boss 'Digest: ' followed by the unchanged message.",
            "Create a 7:30 AM tomorrow reminder from the newest text verbatim; update my "
            "boss relationship to manager; record +14155550182 under 'Ops Desk' with "
            "relationship coordinator; reply to the boss with Digest: plus that same text.",
        ),
    }
    scenarios, metadata = {}, {}
    for family, variants in prompts.items():
        old_names = sorted(name for name, item in old_metadata.items()
                           if item["family"] == family)
        if len(old_names) != len(variants):
            raise ValueError(f"expected two source scenarios for {family}")
        for index, (old_name, public) in enumerate(zip(old_names, variants)):
            new_name = "attr_" + family + ("" if index == 0 else "_alt")
            scenarios[new_name] = prior._retarget(old_scenarios[old_name], public, u)
            item = deepcopy(old_metadata[old_name])
            item.update({"family": family, "public_request": public,
                         "operation_graph": list(prior.WIDE_TOOLS)})
            metadata[new_name] = item
    order = ("literal_wifi", "literal_cellular", "literal_location",
             "contact_replace_message", "reminder_replace_boss",
             "latest_branch_contact")
    trajectories = [{"trajectory": family, "sessions": sorted(
        name for name, item in metadata.items() if item["family"] == family)}
        for family in order]
    return scenarios, metadata, trajectories


def _supported(pool: list[dict], scenario) -> list[dict]:
    allowed = set(scenario.starting_context.tool_allow_list or [])
    return [note for note in pool if set(note["tools"]) & allowed]


def select(method: str, pool: list[dict], task: str, query: str, scenario,
           pair_vetoes: set[tuple[str, str]], tokenizer) -> list[dict]:
    """Select a bank while keeping the executor assignment independent."""
    if method in {"NoMem/Base", "NoMem/v3"}:
        return []
    if method == "FIFO/v3":
        return _supported(pool, scenario)[:2]
    if method == "Semantic/v3":
        return legacy_select("Semantic", pool, task, query, scenario, pair_vetoes)
    if method in {"CREDIT/Base", "CREDIT/v3"}:
        return compose_select(pool, query, scenario)
    if method == "ContractOnly/v3":
        return v1.fast_contract_select(pool, query, scenario, tokenizer)[:1]
    if method == "Full/v3":
        return v1.fast_select(pool, query, scenario, tokenizer, adaptive=True)
    raise ValueError("unknown method: " + method)


def _consume(job: dict, record: dict, u, max_calls: int, query: str) -> None:
    began = perf_counter_ns()
    if job["method"].endswith("/Base"):
        adapter.consume(job, record, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
    else:
        v3.consume_fast(job, record, u, max_calls, query)


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
                rows.append(prior.prior.finish_direct(job, directory, u))
            else:
                remaining.append(job)
        active = remaining
    for job in active:
        job["termination"] = "turn_budget"
        rows.append(prior.prior.finish_direct(job, directory, u))
    return rows


def _latency(row: dict) -> tuple[float, float, float]:
    remote = session_latency(row)
    local = float(row.get("selection_latency_ms", 0.0)) + sum(
        float(turn.get("local_consume_latency_ms", 0.0))
        for turn in row["generations"])
    return remote, local, remote + local


def run_actor(seed: int, actor_index: int, trajectories: list[dict], root: Path,
              args, scenarios, metadata, pool, pair_vetoes, u):
    directory = root / f"selector_attribution_seed_{seed}"
    directory.mkdir()
    agent = RemoteJSONAgent(directory, seed, env=args.env,
        tokenizer_path=args.tokenizer, temperature=0.0, max_new_tokens=384,
        workers=args.workers, exact_prompt_cache=False)
    agent.manifest["benchmark"] = "ToolSandbox-M selector attribution"
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
                                     "selector_attribution_confirmation", method)
                job["selection_latency_ms"] = selection_ms
                jobs.append(job)
                info[method] = (bank, position)
            for row in run_batch(jobs, {task: query}, agent, directory, u):
                bank, position = info[row["method"]]
                remote, local, accounted = _latency(row)
                item = prior.annotate_direct({**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"],
                    "session_index": session_index,
                    "selected_bank": [note["uid"] for note in bank],
                    "selected_memory_schemas": [note.get("memory_schema") for note in bank],
                    "remote_latency_ms": remote, "local_runtime_ms": local,
                    "accounted_latency_ms": accounted, "batch_position": position,
                    "public_request_sha256": _sha(query)}, metadata)
                sessions.append(item)
                with (directory / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    agent.close()
    summary = {"actor_seed": seed, "sessions": len(sessions),
        "remote_requests": agent.calls,
        "method_batch_position_counts": dict(positions)}
    adapter.write(directory, "summary.json", summary)
    return sessions, summary


def summarize(rows: list[dict]) -> list[dict]:
    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        latency = np.asarray([row["accounted_latency_ms"] for row in subset])
        turns = [turn for row in subset for turn in row["generations"]]
        item = {"method": method, "sessions": len(subset),
            "requested_state_VSR": float(np.mean([
                row["requested_state_success"] for row in subset])),
            "process_state_VSR": float(np.mean([
                row["process_state_success"] for row in subset])),
            "mean_requested_state_score": float(np.mean([
                row["requested_state_score"] for row in subset])),
            "mean_selected_records": float(np.mean([
                len(row["selected_bank"]) for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "mean_remote_requests": float(np.mean([sum(
                "remote_latency_ms" in turn for turn in row["generations"])
                for row in subset])),
            "mean_accounted_latency_ms": float(np.mean(latency)),
            "p95_accounted_latency_ms": float(np.quantile(latency, .95)),
            "tool_call_exceptions": sum(bool(call.get("exception"))
                for row in subset for call in row["calls"]),
            "validator_rejections": sum("contract_violation" in turn for turn in turns),
            "deterministic_repairs": sum("deterministic_repair" in turn for turn in turns),
            "local_completions": sum(row["termination"] == "local_contract_complete"
                                     for row in subset)}
        for budget in LATENCY_BUDGETS_MS:
            item[f"requested_state_VSR_at_{budget}ms"] = float(np.mean([
                row["requested_state_success"] and value <= budget
                for row, value in zip(subset, latency)]))
        table.append(item)
    return table


def _sign(values: np.ndarray) -> dict:
    wins = int(np.sum(values > 0)); losses = int(np.sum(values < 0))
    p = float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue) \
        if wins + losses else 1.0
    return {"wins": wins, "losses": losses,
            "ties": int(len(values) - wins - losses), "one_sided_exact_p": p}


def _crossed_bootstrap(matrix: np.ndarray, rng) -> list[float]:
    actors, families = matrix.shape
    output = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        actor_draw = rng.integers(0, actors, actors)
        family_draw = rng.integers(0, families, families)
        output[index] = float(np.mean(matrix[np.ix_(actor_draw, family_draw)]))
    return np.quantile(output, [.025, .975]).tolist()


def paired_statistics(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for row in rows:
        key = (row["actor_seed"], row["trajectory"], row["session_index"])
        by[row["method"]][key] = row
    actors = sorted({row["actor_seed"] for row in rows})
    families = sorted({row["trajectory"] for row in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = []
    for treatment, baseline, estimand in PRIMARY_CONTRASTS:
        matrix = np.empty((len(actors), len(families)))
        budget_matrix = np.empty_like(matrix)
        for ai, actor in enumerate(actors):
            for fi, family in enumerate(families):
                keys = [(actor, family, session) for session in (0, 1)]
                matrix[ai, fi] = np.mean([
                    by[treatment][key]["requested_state_success"] -
                    by[baseline][key]["requested_state_success"] for key in keys])
                budget_matrix[ai, fi] = np.mean([
                    (by[treatment][key]["requested_state_success"] and
                     by[treatment][key]["accounted_latency_ms"] <= PRIMARY_BUDGET_MS) -
                    (by[baseline][key]["requested_state_success"] and
                     by[baseline][key]["accounted_latency_ms"] <= PRIMARY_BUDGET_MS)
                    for key in keys])
        actor_effects = np.mean(matrix, axis=1)
        family_effects = np.mean(matrix, axis=0)
        comparisons.append({"treatment": treatment, "baseline": baseline,
            "estimand": estimand, "delta_requested_state_VSR": float(np.mean(matrix)),
            "crossed_actor_family_bootstrap_CI95": _crossed_bootstrap(matrix, rng),
            "actor_level_sign_test": _sign(actor_effects),
            "family_level_sign_sensitivity": _sign(family_effects),
            "actor_effects": actor_effects.tolist(),
            "family_effects": dict(zip(families, family_effects.tolist())),
            "delta_state_VSR_at_15000ms": float(np.mean(budget_matrix)),
            "budget_crossed_bootstrap_CI95": _crossed_bootstrap(budget_matrix, rng)})
    order = sorted(range(len(comparisons)), key=lambda i:
        comparisons[i]["actor_level_sign_test"]["one_sided_exact_p"])
    running = 0.0
    for rank, index in enumerate(order):
        raw = comparisons[index]["actor_level_sign_test"]["one_sided_exact_p"]
        running = max(running, min(1.0, (len(order) - rank) * raw))
        comparisons[index]["holm_actor_p"] = running
    return {"comparisons": comparisons, "actors": len(actors),
        "families": len(families), "prompts_per_actor_family": 2,
        "primary_resampling": "crossed actor-by-family bootstrap",
        "primary_test_unit": "actor-level paired mean",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED}


def task_breakdown(rows: list[dict]) -> list[dict]:
    output = []
    for family in sorted({row["trajectory"] for row in rows}):
        for method in METHODS:
            subset = [row for row in rows if row["trajectory"] == family
                      and row["method"] == method]
            output.append({"family": family, "method": method,
                "sessions": len(subset), "requested_state_VSR": float(np.mean([
                    row["requested_state_success"] for row in subset])),
                "requested_state_VSR_at_15000ms": float(np.mean([
                    row["requested_state_success"] and
                    row["accounted_latency_ms"] <= PRIMARY_BUDGET_MS
                    for row in subset]))})
    return output


def freeze(args, metadata, trajectories, pair_vetoes, preflight) -> dict:
    names = ["toolsandbox_m_selector_attribution.py",
             "toolsandbox_m_contract_fast_v3.py",
             "toolsandbox_m_contract_fast_v3_confirmation.py",
             "toolsandbox_m_contract_fast.py", "toolsandbox_m_contract_selector.py",
             "toolsandbox_m_compose_selector.py", "toolsandbox_m_intent_optimization.py",
             "toolsandbox_m_followup.py", "toolsandbox_local_v3.py",
             "remote_json_agent.py"]
    source_dir = args.output / "source"; source_dir.mkdir()
    source_sha = {}
    for name in names:
        source = Path(__file__).with_name(name)
        shutil.copy2(source, source_dir / name)
        source_sha[name] = digest(source)
    previous = json.loads((args.protocol_source / "frozen_protocol.json").read_text())
    protocol = {"suite": "ToolSandbox-M selector-by-executor attribution",
        "frozen_before_remote_calls": True, "clock": CLOCK,
        "actor_seeds": list(ACTOR_SEEDS), "methods": list(METHODS),
        "primary_endpoint": "exact requested-world-state VSR",
        "secondary_latency_endpoint_ms": PRIMARY_BUDGET_MS,
        "primary_contrasts": [list(row) for row in PRIMARY_CONTRASTS],
        "multiplicity": "Holm over five registered actor-level sign tests",
        "uncertainty": "crossed actor-by-family bootstrap",
        "latency_endpoints_ms": list(LATENCY_BUDGETS_MS),
        "maximum_model_turns_per_session": MAX_MODEL_TURNS,
        "maximum_tool_calls_per_session": 12, "exact_prompt_cache": False,
        "provider_credentials_persisted": False,
        "wide_identical_tool_catalogue": list(prior.WIDE_TOOLS),
        "tasks": [{"name": name, "family": metadata[name]["family"],
                   "public_request_sha256": _sha(metadata[name]["public_request"])}
                  for name in sorted(metadata)],
        "trajectories": trajectories, "pair_vetoes": [list(x) for x in pair_vetoes],
        "memory_pool_source": str(args.protocol_source),
        "memory_pool_source_sha256": digest(args.protocol_source / "memory_pool.json"),
        "novelty_sources": [str(path) for path in args.novelty_sources],
        "source_sha256": source_sha,
        "plan_sha256": digest(Path("CREDIT_V4_SELECTOR_ATTRIBUTION_PLAN.md")),
        "upstream_directory": previous["upstream_directory"],
        "upstream_source_sha256": previous["upstream_source_sha256"],
        "preflight": preflight}
    adapter.write(args.output, "frozen_protocol.json", protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--protocol-source", type=Path,
        default=Path("results/toolsandbox_m_contract_fast_v3_confirmation_v1"))
    parser.add_argument("--novelty-sources", type=Path, nargs="+", default=[
        Path("results/toolsandbox_m_contract_fast_v2_confirmation_v1"),
        Path("results/toolsandbox_m_contract_fast_v3_development_v1"),
        Path("results/toolsandbox_m_contract_fast_v3_development_v2"),
        Path("results/toolsandbox_m_contract_fast_v3_confirmation_v1")])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    previous = json.loads((args.protocol_source / "frozen_protocol.json").read_text())
    u = adapter.upstream_imports(Path(previous["upstream_directory"]))
    adapter.install_local_clock(CLOCK)
    scenarios, metadata, trajectories = build_confirmation_scenarios(u)
    pool = json.loads((args.protocol_source / "memory_pool.json").read_text())
    shutil.copy2(args.protocol_source / "memory_pool.json", args.output / "memory_pool.json")
    pair_vetoes = {tuple(x) for x in previous["pair_vetoes"]}
    preflight = {"golden_evaluator": prior.golden_self_check(u, scenarios, metadata),
        "argument_extraction": prior.argument_extraction_self_check(metadata),
        "novelty": novelty_audit(u, scenarios, metadata, args.novelty_sources)}
    adapter.write(args.output, "preflight.json", preflight)
    freeze(args, metadata, trajectories, pair_vetoes, preflight)
    manifest = json.loads((args.protocol_source /
        "fast_v3_confirmation_seed_2801/agent_manifest.json").read_text())
    args.tokenizer = args.tokenizer or Path(manifest["tokenizer_path"])
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    selection = {}
    for task, scenario in scenarios.items():
        selection[task] = {}
        for method in METHODS:
            bank = select(method, pool, task, metadata[task]["public_request"],
                          scenario, pair_vetoes, tokenizer)
            selection[task][method] = {"selected_uids": [note["uid"] for note in bank],
                "memory_schemas": [note.get("memory_schema") for note in bank]}
    adapter.write(args.output, "frozen_selection.json", selection)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "tasks": len(metadata),
                          "methods": len(METHODS), "preflight": preflight}, indent=2))
        return

    rows, actors = [], []
    for index, seed in enumerate(ACTOR_SEEDS):
        found, actor = run_actor(seed, index, trajectories, args.output, args,
                                 scenarios, metadata, pool, pair_vetoes, u)
        rows.extend(found); actors.append(actor)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table, statistics = summarize(rows), paired_statistics(rows)
    ranking = sorted((dict(row) for row in table), key=lambda row: (
        -row["requested_state_VSR"],
        -row[f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms"],
        row["mean_accounted_latency_ms"], row["method"]))
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    breakdown = task_breakdown(rows)
    summary = {"confirmation_table": table, "ranking": ranking,
        "paired_statistics": statistics, "task_breakdown": breakdown,
        "actor_summaries": actors, "frozen_selection": selection}
    adapter.write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    with (args.output / "task_family_breakdown.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
        writer.writeheader(); writer.writerows(breakdown)
    adapter.write(args.output, "audit.json", {"sessions": len(rows),
        "expected_sessions": len(ACTOR_SEEDS) * len(metadata) * len(METHODS),
        "live_remote_requests": sum(actor["remote_requests"] for actor in actors),
        "credentials_persisted": False, "all_tasks_share_identical_tools": True,
        "all_registered_sessions_present": len(rows) ==
            len(ACTOR_SEEDS) * len(metadata) * len(METHODS)})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
