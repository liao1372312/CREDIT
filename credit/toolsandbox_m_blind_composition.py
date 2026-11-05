"""Pre-registered blind composition evaluation for the frozen CREDIT-Intent policy.

The scenarios combine ToolSandbox operations in task families that were absent
from development and confirmation.  They retain ToolSandbox's real databases,
tools, and milestone evaluator.  Tool/state completion and final-text matching
are reported separately.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

import attrs
import numpy as np
from scipy.stats import binomtest

from .remote_json_agent import RemoteJSONAgent
from .toolsandbox_local_v3 import (
    fingerprint,
    consume,
    finish,
    initial_job,
    install_local_clock,
    run_batch,
    upstream_imports,
    write,
)
from .toolsandbox_m import continuous_score
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_intent_optimization import (
    METHODS,
    balanced_order,
    intent_select,
    select,
)
from .user_data import digest


CLOCK = "2026-10-14T12:00:00"
ACTOR_SEEDS = (967, 971, 977, 983, 991)
LATENCY_BUDGETS_MS = (5000, 10000, 20000)
BOOTSTRAP_SEED = 20261016


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _imports():
    """Import upstream types after its directory has been put on sys.path."""
    import polars as pl
    from tool_sandbox.common.evaluation import (
        Milestone,
        SnapshotConstraint,
        addition_similarity,
        column_contains_similarity,
        removal_similarity,
        snapshot_similarity,
    )
    from tool_sandbox.common.execution_context import (
        DatabaseNamespace,
        RoleType,
        ScenarioCategories,
    )
    from tool_sandbox.common.scenario import ScenarioExtension
    from tool_sandbox.scenarios.base_scenarios import named_base_scenarios
    from tool_sandbox.scenarios.user_simulator_few_shot_examples import USER_INSTRUCTION
    return locals()


def build_blind_scenarios(u):
    """Build six unseen operation graphs, each with two prompt variants."""
    x = _imports()
    pl, M, C = x["pl"], x["Milestone"], x["SnapshotConstraint"]
    DB, R, Cat = x["DatabaseNamespace"], x["RoleType"], x["ScenarioCategories"]
    add, remove, snap = (x["addition_similarity"], x["removal_similarity"],
                         x["snapshot_similarity"])
    contains = x["column_contains_similarity"]
    Extension = x["ScenarioExtension"]
    bases = x["named_base_scenarios"](u.ToolBackend.DEFAULT)
    user_instruction = x["USER_INSTRUCTION"]

    tomorrow_9 = datetime.datetime(2026, 10, 15, 9).timestamp()

    def messages(private: str, public: str):
        return [
            {"sender": R.SYSTEM, "recipient": R.USER,
             "content": user_instruction + private + " You do not have any more information."},
            {"sender": R.USER, "recipient": R.AGENT, "content": public},
        ]

    def trace(tool: str, arguments: dict | None = None):
        return M(snapshot_constraints=[C(
            database_namespace=DB.SANDBOX,
            snapshot_constraint=snap,
            target_dataframe=pl.DataFrame({
                "sender": R.EXECUTION_ENVIRONMENT,
                "recipient": R.AGENT,
                "tool_trace": json.dumps({"tool_name": tool, "arguments": arguments or {}},
                                          ensure_ascii=False),
            }),
        )])

    def final_text(target: str):
        return M(snapshot_constraints=[C(
            database_namespace=DB.SANDBOX,
            snapshot_constraint=snap,
            target_dataframe=pl.DataFrame({
                "sender": R.AGENT, "recipient": R.USER, "content": target,
            }),
            column_similarity_measure={"content": contains},
        )])

    def extension(name, family, public, private, base, tools, milestones, edges,
                  mutation_indices):
        scenario = Extension(
            name=name,
            base_scenario=bases[base],
            messages=messages(private, public),
            tool_allow_list=tools,
            milestones=milestones,
            milestone_edge_list=edges,
            categories=[Cat.SINGLE_USER_TURN, Cat.CANONICALIZATION],
        ).get_extended_scenario()[name]
        metadata[name] = {
            "family": family,
            "public_request": public,
            "tool_state_indices": list(range(len(milestones) - 1)),
            "mutation_indices": mutation_indices,
            "text_index": len(milestones) - 1,
            "operation_graph": tools,
        }
        return scenario

    scenarios, metadata = {}, {}

    # New graph 1: message lookup -> dated reminder creation.
    for suffix, public in (
        ("", "Make my latest message the text of a reminder for tomorrow at 9 AM."),
        ("_alt", "At 9 tomorrow morning, remind me about whatever my newest text says."),
    ):
        name = "blind_message_to_reminder" + suffix
        ms = [
            trace("search_messages"),
            trace("get_current_timestamp"),
            M(snapshot_constraints=[C(
                database_namespace=DB.REMINDER, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({
                    "content": "Good, keep me posted", "reminder_timestamp": tomorrow_9,
                }), reference_milestone_node_index=1,
            )]),
            final_text("reminder"),
        ]
        scenarios[name] = extension(name, "message_to_reminder", public,
            "Create a reminder whose content is the most recent message and whose time is tomorrow at 9 AM.",
            "base", ["search_messages", "get_current_timestamp", "timestamp_to_datetime_info",
                     "datetime_info_to_timestamp", "add_reminder"], ms,
            [(0, 2), (1, 2), (2, 3)], [2])

    # New graph 2: reminder lookup -> contact lookup -> message creation.
    for suffix, public in (
        ("", "Text my boss the exact content of my upcoming reminder."),
        ("_alt", "Send the words from my next reminder to my boss."),
    ):
        name = "blind_reminder_to_message" + suffix
        ms = [
            trace("search_reminder"),
            trace("get_current_timestamp"),
            trace("search_contacts", {"relationship": "boss"}),
            M(snapshot_constraints=[C(
                database_namespace=DB.MESSAGING, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({
                    "recipient_phone_number": "+10000000000",
                    "content": "Buy a nice rich navy bathing dress",
                }), reference_milestone_node_index=2,
            )]),
            final_text("sent"),
        ]
        scenarios[name] = extension(name, "reminder_to_message", public,
            "Find the upcoming reminder, find the boss contact, and send the reminder content to the boss.",
            "base", ["search_reminder", "get_current_timestamp", "search_contacts",
                     "send_message_with_phone_number"], ms,
            [(0, 3), (1, 3), (2, 3), (3, 4)], [3])

    # New graph 3: duplicate check -> contact creation -> message creation.
    for suffix, public in (
        ("", "Add Casey Lin at +14155550173 as my designer, then text them 'Welcome aboard'."),
        ("_alt", "Save +14155550173 under Casey Lin (designer) and send that contact: Welcome aboard."),
    ):
        name = "blind_add_contact_then_message" + suffix
        ms = [
            trace("search_contacts", {"phone_number": "+14155550173"}),
            M(snapshot_constraints=[C(
                database_namespace=DB.CONTACT, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({
                    "name": "Casey Lin", "phone_number": "+14155550173",
                    "relationship": "designer", "is_self": False,
                }), reference_milestone_node_index=0,
            )]),
            M(snapshot_constraints=[C(
                database_namespace=DB.MESSAGING, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({
                    "recipient_phone_number": "+14155550173", "content": "Welcome aboard",
                }), reference_milestone_node_index=1,
            )]),
            final_text("Welcome aboard"),
        ]
        scenarios[name] = extension(name, "add_contact_then_message", public,
            "Check the phone is absent, add Casey Lin as designer, then send Welcome aboard.",
            "base", ["search_contacts", "add_contact", "send_message_with_phone_number"],
            ms, None, [1, 2])

    # New graph 4: reminder lookup -> removal -> replacement at the original time.
    for suffix, public in (
        ("", "Replace my upcoming reminder with 'Pack conference badge', keeping its original time."),
        ("_alt", "Delete the next reminder and create Pack conference badge for the same time."),
    ):
        name = "blind_replace_upcoming_reminder" + suffix
        ms = [
            trace("search_reminder"),
            trace("get_current_timestamp"),
            M(snapshot_constraints=[C(
                database_namespace=DB.REMINDER, snapshot_constraint=remove,
                target_dataframe=pl.DataFrame({
                    "reminder_id": "54f20be9-eac3-5e76-8e57-724e48d3d477",
                }), reference_milestone_node_index=0,
            )]),
            M(snapshot_constraints=[C(
                database_namespace=DB.REMINDER, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({
                    "content": "Pack conference badge",
                    "reminder_timestamp": 1792160294.229264,
                }), reference_milestone_node_index=2,
            )]),
            final_text("Pack conference badge"),
        ]
        scenarios[name] = extension(name, "replace_upcoming_reminder", public,
            "Remove the upcoming reminder and add Pack conference badge at the removed reminder's time.",
            "base", ["search_reminder", "get_current_timestamp", "remove_reminder",
                     "add_reminder"], ms,
            [(0, 2), (1, 2), (2, 3), (3, 4)], [2, 3])

    # New graph 5: low-battery repair -> two setting changes.
    for suffix, public in (
        ("", "Turn on both Wi-Fi and cellular data."),
        ("_alt", "Enable my two network connections: Wi-Fi and mobile service."),
    ):
        name = "blind_enable_wifi_and_cellular" + suffix
        ms = [
            M(snapshot_constraints=[C(database_namespace=DB.SETTING,
                snapshot_constraint=snap, target_dataframe=pl.DataFrame({"low_battery_mode": False}))]),
            M(snapshot_constraints=[C(database_namespace=DB.SETTING,
                snapshot_constraint=snap, target_dataframe=pl.DataFrame({"wifi": True}))]),
            M(snapshot_constraints=[C(database_namespace=DB.SETTING,
                snapshot_constraint=snap, target_dataframe=pl.DataFrame({"cellular": True}))]),
            final_text("on"),
        ]
        scenarios[name] = extension(name, "enable_wifi_and_cellular", public,
            "Disable low-battery mode if necessary and enable both wifi and cellular.",
            "base_low_battery_mode_on_wifi_off_location_service_off_cellular_off",
            ["get_low_battery_mode_status", "set_low_battery_mode_status", "get_wifi_status",
             "set_wifi_status", "get_cellular_service_status", "set_cellular_service_status"],
            ms, [(0, 1), (0, 2), (1, 3), (2, 3)], [0, 1, 2])

    # New graph 6: unknown-message lookup -> contact creation from returned phone.
    for suffix, public in (
        ("", "Find who sent 'Hey kid, you want some GPU?' and save that number as GPU Seller, relationship vendor."),
        ("_alt", "Add the sender of the GPU offer to contacts as GPU Seller and mark them as a vendor."),
    ):
        name = "blind_message_sender_to_contact" + suffix
        ms = [
            trace("search_messages", {"content": "Hey kid, you want some GPU?"}),
            trace("search_contacts", {"phone_number": "+18307976530"}),
            M(snapshot_constraints=[C(
                database_namespace=DB.CONTACT, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({
                    "name": "GPU Seller", "phone_number": "+18307976530",
                    "relationship": "vendor", "is_self": False,
                }), reference_milestone_node_index=1,
            )]),
            final_text("GPU Seller"),
        ]
        scenarios[name] = extension(name, "message_sender_to_contact", public,
            "Find the GPU-offer message sender and add that phone as GPU Seller with vendor relationship.",
            "base", ["search_messages", "search_contacts", "add_contact"], ms, None, [2])

    trajectories = []
    for family in (
        "message_to_reminder", "reminder_to_message", "add_contact_then_message",
        "replace_upcoming_reminder", "enable_wifi_and_cellular",
        "message_sender_to_contact",
    ):
        names = [name for name, item in metadata.items() if item["family"] == family]
        trajectories.append({"trajectory": family, "sessions": names})
    return scenarios, metadata, trajectories


def annotate(row: dict, metadata: dict) -> dict:
    """Attach pre-registered state/tool and final-text endpoint scores."""
    spec = metadata[row["task"]]
    evaluation = row.get("upstream_evaluation")
    if not evaluation:
        scores = {}
        minefield_clear = False
    else:
        scores = evaluation["milestone_mapping"]
        minefield_clear = evaluation["minefield_similarity"] == 0

    def score(index: int) -> float:
        pair = scores.get(index, scores.get(str(index), (-1, 0.0)))
        return float(pair[1])

    tool_state = [score(i) for i in spec["tool_state_indices"]]
    mutation = [score(i) for i in spec["mutation_indices"]]
    text_score = score(spec["text_index"])
    row.update({
        "tool_state_score": float(np.mean(tool_state)) if tool_state else 0.0,
        "tool_state_success": int(minefield_clear and all(x >= 1 - 1e-8 for x in tool_state)),
        "mutation_score": float(np.mean(mutation)) if mutation else 0.0,
        "mutation_success": int(minefield_clear and all(x >= 1 - 1e-8 for x in mutation)),
        "final_text_score": text_score,
        "final_text_success": int(minefield_clear and text_score >= 1 - 1e-8),
    })
    return row


def golden_self_check(u, scenarios, metadata) -> dict:
    """Execute one hand-checked successful trajectory through every evaluator."""
    tomorrow_9 = datetime.datetime(2026, 10, 15, 9).timestamp()
    actions = {
        "message_to_reminder": [
            {"tool": "search_messages", "arguments": {"content": "Good, keep me posted"}},
            {"tool": "get_current_timestamp", "arguments": {}},
            {"tool": "add_reminder", "arguments": {
                "content": "Good, keep me posted", "reminder_timestamp": tomorrow_9}},
            {"answer": "The reminder was created."},
        ],
        "reminder_to_message": [
            {"tool": "search_reminder", "arguments": {
                "content": "Buy a nice rich navy bathing dress"}},
            {"tool": "get_current_timestamp", "arguments": {}},
            {"tool": "search_contacts", "arguments": {"relationship": "boss"}},
            {"tool": "send_message_with_phone_number", "arguments": {
                "phone_number": "+10000000000",
                "content": "Buy a nice rich navy bathing dress"}},
            {"answer": "The reminder text was sent."},
        ],
        "add_contact_then_message": [
            {"tool": "search_contacts", "arguments": {"phone_number": "+14155550173"}},
            {"tool": "add_contact", "arguments": {"name": "Casey Lin",
                "phone_number": "+14155550173", "relationship": "designer"}},
            {"tool": "send_message_with_phone_number", "arguments": {
                "phone_number": "+14155550173", "content": "Welcome aboard"}},
            {"answer": "Saved Casey Lin and sent Welcome aboard."},
        ],
        "replace_upcoming_reminder": [
            {"tool": "search_reminder", "arguments": {
                "content": "Buy a nice rich navy bathing dress"}},
            {"tool": "get_current_timestamp", "arguments": {}},
            {"tool": "remove_reminder", "arguments": {
                "reminder_id": "54f20be9-eac3-5e76-8e57-724e48d3d477"}},
            {"tool": "add_reminder", "arguments": {"content": "Pack conference badge",
                "reminder_timestamp": 1792160294.229264}},
            {"answer": "Pack conference badge replaced the old reminder."},
        ],
        "enable_wifi_and_cellular": [
            {"tool": "set_low_battery_mode_status", "arguments": {"on": False}},
            {"tool": "set_wifi_status", "arguments": {"on": True}},
            {"tool": "set_cellular_service_status", "arguments": {"on": True}},
            {"answer": "Both connections are on."},
        ],
        "message_sender_to_contact": [
            {"tool": "search_messages", "arguments": {
                "content": "Hey kid, you want some GPU?"}},
            {"tool": "search_contacts", "arguments": {"phone_number": "+18307976530"}},
            {"tool": "add_contact", "arguments": {"name": "GPU Seller",
                "phone_number": "+18307976530", "relationship": "vendor"}},
            {"answer": "GPU Seller was added."},
        ],
    }
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        for name, scenario in scenarios.items():
            job = initial_job(name, scenario, [], u, None, "golden_preflight", "gold")
            for index, action in enumerate(actions[metadata[name]["family"]]):
                consume(job, {"agent_output": json.dumps(action),
                    "prompt_sha256": f"gold-{index}", "hit_generation_limit": False}, u, 8)
            row = annotate(finish(job, Path(temporary), u), metadata)
            checks.append({"task": name, "full_success": row["success"],
                "tool_state_success": row["tool_state_success"],
                "mutation_success": row["mutation_success"],
                "text_success": row["final_text_success"]})
    passed = all(all(item[key] == 1 for key in (
        "full_success", "tool_state_success", "mutation_success", "text_success"))
        for item in checks)
    if not passed:
        raise RuntimeError("golden evaluator preflight failed")
    return {"passed": passed, "tasks": len(checks), "checks": checks}


def novelty_audit(u, scenarios, metadata, source: Path) -> dict:
    """Check exact prompt and tool-allow-list novelty before observing outcomes."""
    official = {**u.named_single_tool_call_scenarios(u.ToolBackend.DEFAULT),
                **u.named_multiple_tool_call_scenarios(u.ToolBackend.DEFAULT)}
    official_sets = defaultdict(list)
    for name, scenario in official.items():
        official_sets[tuple(sorted(scenario.starting_context.tool_allow_list or []))].append(name)
    tool_set_overlap = {}
    for name, scenario in scenarios.items():
        key = tuple(sorted(scenario.starting_context.tool_allow_list or []))
        if key in official_sets:
            tool_set_overlap[name] = official_sets[key]

    prior_hashes = set()
    prior_path = source / "confirmation_sessions.jsonl"
    if prior_path.exists():
        with prior_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                for field in ("prompt_variant", "public_request"):
                    if isinstance(row.get(field), str):
                        prior_hashes.add(_sha(row[field]))
    current = {_sha(item["public_request"]) for item in metadata.values()}
    exact_prompt_overlap = sorted(current & prior_hashes)
    passed = not tool_set_overlap and not exact_prompt_overlap and len(current) == len(metadata)
    if not passed:
        raise RuntimeError("blind-family novelty preflight failed")
    return {"passed": passed, "tasks": len(metadata),
        "unique_public_request_hashes": len(current),
        "prior_public_prompt_hashes_checked": len(prior_hashes),
        "exact_prompt_overlap": exact_prompt_overlap,
        "exact_tool_allow_list_overlap": tool_set_overlap,
        "novelty_definition": "unique public text and no exact allowed-tool set in official single/multi-tool suites"}


def _agent(directory: Path, seed: int, args) -> RemoteJSONAgent:
    directory.mkdir(parents=True)
    agent = RemoteJSONAgent(directory, seed, env=args.env, tokenizer_path=args.tokenizer,
        temperature=0.0, max_new_tokens=384, workers=args.workers,
        exact_prompt_cache=False)
    agent.binding_interface = "tool_json"
    agent.manifest["benchmark"] = "ToolSandbox-M blind composition v1"
    return agent


def run_actor(seed, actor_index, trajectories, root, args, scenarios, metadata,
              pool, pair_vetoes, u):
    directory = root / f"blind_composition_seed_{seed}"
    agent = _agent(directory, seed, args)
    sessions, positions = [], defaultdict(lambda: [0] * len(METHODS))
    for trajectory_index, trajectory in enumerate(trajectories):
        for session_index, task in enumerate(trajectory["sessions"]):
            scenario = scenarios[task]
            query = metadata[task]["public_request"]
            order = balanced_order(list(METHODS), actor_index, trajectory_index, session_index)
            jobs, job_meta = [], {}
            for position, method in enumerate(order):
                positions[method][position] += 1
                bank = select(method, pool, task, query, scenario, pair_vetoes)
                job = initial_job_v2(task, scenario, bank, u, agent.tokenizer,
                                     "blind_composition", method)
                jobs.append(job)
                job_meta[method] = (bank, position)
            for row in run_batch(jobs, agent, directory, u):
                bank, position = job_meta[row["method"]]
                item = annotate({**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"], "session_index": session_index,
                    "selected_bank": [note["uid"] for note in bank],
                    "remote_latency_ms": session_latency(row), "batch_position": position,
                    "public_request_sha256": _sha(query)}, metadata)
                sessions.append(item)
                with (directory / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    agent.close()
    summary = {"actor_seed": seed, "sessions": len(sessions),
        "remote_requests": agent.calls, "method_batch_position_counts": dict(positions)}
    write(directory, "summary.json", summary)
    return sessions, summary


def summarize(rows):
    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        latency = [row["remote_latency_ms"] for row in subset]
        item = {"method": method, "sessions": len(subset),
            "tool_state_VSR": float(np.mean([r["tool_state_success"] for r in subset])),
            "mutation_VSR": float(np.mean([r["mutation_success"] for r in subset])),
            "final_text_VSR": float(np.mean([r["final_text_success"] for r in subset])),
            "full_strict_VSR": float(np.mean([r["success"] for r in subset])),
            "mean_tool_state_score": float(np.mean([r["tool_state_score"] for r in subset])),
            "upstream_mean_score": float(np.mean([continuous_score(r) for r in subset])),
            "mean_tool_calls": float(np.mean([r["tool_calls"] for r in subset])),
            "mean_remote_requests": float(np.mean([len(r["generations"]) for r in subset])),
            "mean_online_remote_latency_ms": float(np.mean(latency)),
            "p95_online_remote_latency_ms": float(np.quantile(latency, .95))}
        for budget in LATENCY_BUDGETS_MS:
            item[f"tool_state_VSR_at_{budget}ms"] = float(np.mean([
                r["tool_state_success"] and t <= budget for r, t in zip(subset, latency)]))
            item[f"completion_at_{budget}ms"] = float(np.mean([t <= budget for t in latency]))
        table.append(item)
    return table


def paired_statistics(rows):
    by = defaultdict(dict)
    for row in rows:
        by[row["method"]][(row["actor_seed"], row["trajectory"], row["session_index"])] = row
    clusters = sorted({(r["actor_seed"], r["trajectory"]) for r in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    report = []
    for method in METHODS:
        state_effects, full_effects = [], []
        for seed, trajectory in clusters:
            keys = [(seed, trajectory, i) for i in (0, 1)]
            state_effects.append(float(np.mean([
                by["CREDIT-Intent"][k]["tool_state_success"] - by[method][k]["tool_state_success"]
                for k in keys])))
            full_effects.append(float(np.mean([
                by["CREDIT-Intent"][k]["success"] - by[method][k]["success"] for k in keys])))
        state_boot = np.mean(np.asarray(state_effects)[draws], axis=1)
        full_boot = np.mean(np.asarray(full_effects)[draws], axis=1)
        wins, losses = sum(x > 0 for x in state_effects), sum(x < 0 for x in state_effects)
        p = float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue) \
            if wins + losses else 1.0
        report.append({"baseline": method,
            "delta_tool_state_VSR": float(np.mean(state_effects)),
            "tool_state_cluster_bootstrap_CI95": np.quantile(state_boot, [.025, .975]).tolist(),
            "delta_full_strict_VSR": float(np.mean(full_effects)),
            "full_cluster_bootstrap_CI95": np.quantile(full_boot, [.025, .975]).tolist(),
            "cluster_wins": wins, "cluster_losses": losses,
            "one_sided_exact_sign_p": p})
    return {"comparisons": report, "clusters": len(clusters),
        "resampling_unit": "actor seed by two-prompt task family",
        "bootstrap_replicates": 10000, "bootstrap_seed": BOOTSTRAP_SEED}


def task_breakdown(rows):
    out = []
    for trajectory in sorted({r["trajectory"] for r in rows}):
        for method in METHODS:
            subset = [r for r in rows if r["trajectory"] == trajectory and r["method"] == method]
            out.append({"trajectory": trajectory, "method": method, "n": len(subset),
                "tool_state_VSR": float(np.mean([r["tool_state_success"] for r in subset])),
                "mutation_VSR": float(np.mean([r["mutation_success"] for r in subset])),
                "text_VSR": float(np.mean([r["final_text_success"] for r in subset])),
                "full_VSR": float(np.mean([r["success"] for r in subset]))})
    return out


def freeze(args, u, scenarios, metadata, trajectories, pool, pair_vetoes,
           source_protocol, preflight):
    names = ["toolsandbox_m_blind_composition.py", "toolsandbox_m_blind_composition_verify.py",
             "toolsandbox_m_intent_optimization.py", "toolsandbox_m_streaming_optimization.py",
             "toolsandbox_m_followup.py", "toolsandbox_local_v3.py", "toolsandbox_m.py",
             "remote_json_agent.py"]
    source_hashes = {name: digest(Path(__file__).with_name(name)) for name in names
                     if Path(__file__).with_name(name).exists()}
    task_records = [{"name": name, **metadata[name],
        "public_request_sha256": _sha(metadata[name]["public_request"])}
        for name in sorted(metadata)]
    protocol = {
        "suite": "ToolSandbox-M pre-registered blind composition v1",
        "frozen_before_remote_calls": True,
        "source_results": str(args.source.resolve()),
        "source_memory_pool_sha256": digest(args.source / "memory_pool.json"),
        "source_sha256": source_hashes,
        "upstream_directory": str(Path(source_protocol["upstream_directory"]).resolve()),
        "upstream_source_sha256": source_protocol["upstream_source_sha256"],
        "clock": CLOCK, "tasks": task_records, "trajectories": trajectories,
        "actor_seeds": list(args.seeds), "methods": list(METHODS),
        "primary_endpoint": "tool_state_VSR: every non-text milestone exact and no minefield",
        "secondary_endpoints": ["mutation_VSR", "final_text_VSR", "full_strict_VSR",
                                "upstream_mean_score", "tool_state_VSR_at_10000ms"],
        "primary_comparisons": ["CREDIT-Intent vs FIFO", "CREDIT-Intent vs Semantic"],
        "inference": "paired actor-seed by task-family cluster bootstrap; exact sign test",
        "no_outcome_adaptation": "task generator, frozen selector, endpoints, and seeds fixed before calls",
        "preflight": preflight,
        "selector_training_remote_requests": 0,
        "historical_confirmation_examined": True,
        "claim_scope": "new synthetic task-family composition using official ToolSandbox state and evaluator",
        "temperature": 0.0, "maximum_tool_calls_per_session": 8,
        "paired_scheduling": "all five methods for one task in one concurrent remote batch",
        "exact_prompt_cache": False, "provider_credentials_persisted": False,
        "pair_vetoes": sorted([list(x) for x in pair_vetoes]), "smoke": args.smoke,
    }
    write(args.output, "frozen_protocol.json", protocol)
    write(args.output, "scenario_metadata.json", metadata)
    write(args.output, "memory_pool.json", pool)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/toolsandbox_m_intent_optimization_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/toolsandbox_m_blind_composition_v1"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ACTOR_SEEDS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    (args.output / "source").mkdir()

    source_protocol = json.loads((args.source / "frozen_protocol.json").read_text())
    upstream = Path(source_protocol["upstream_directory"])
    u = upstream_imports(upstream)
    install_local_clock(CLOCK)
    scenarios, metadata, trajectories = build_blind_scenarios(u)
    pool = json.loads((args.source / "memory_pool.json").read_text())
    parent_protocol = json.loads((Path(source_protocol["source_results"]) /
                                  "frozen_protocol.json").read_text())
    pair_vetoes = {tuple(x) for x in parent_protocol["harmful_pair_uids"]}
    if args.smoke:
        args.seeds = args.seeds[:1]
        trajectories = trajectories[:1]
        keep = {name for t in trajectories for name in t["sessions"]}
        scenarios = {k: v for k, v in scenarios.items() if k in keep}
        metadata = {k: v for k, v in metadata.items() if k in keep}
    manifest_path = args.source / f"intent_confirmation_seed_{source_protocol['actor_seeds'][0]}" / "agent_manifest.json"
    args.tokenizer = args.tokenizer or Path(json.loads(manifest_path.read_text())["tokenizer_path"])

    preflight = {"golden_evaluator": golden_self_check(u, scenarios, metadata),
                 "novelty": novelty_audit(u, scenarios, metadata, args.source)}
    write(args.output, "preflight.json", preflight)
    protocol = freeze(args, u, scenarios, metadata, trajectories, pool, pair_vetoes,
                      source_protocol, preflight)
    for name in protocol["source_sha256"]:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "tasks": len(scenarios),
                          "protocol": str(args.output / "frozen_protocol.json")}, indent=2))
        return

    rows, actor_summaries = [], []
    for index, seed in enumerate(args.seeds):
        found, summary = run_actor(seed, index, trajectories, args.output, args,
            scenarios, metadata, pool, pair_vetoes, u)
        rows.extend(found)
        actor_summaries.append(summary)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = summarize(rows)
    breakdown = task_breakdown(rows)
    ranking = sorted((dict(x) for x in table), key=lambda x: (
        -x["tool_state_VSR_at_10000ms"], -x["tool_state_VSR"],
        -x["full_strict_VSR"], -x["upstream_mean_score"],
        x["mean_online_remote_latency_ms"], x["method"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    selection = {}
    for task, scenario in scenarios.items():
        bank = intent_select(pool, metadata[task]["public_request"], scenario)
        selection[task] = {"selected_uids": [x["uid"] for x in bank],
                           "selected_sources": [x["source_task"] for x in bank]}
    summary = {"confirmation_table": table, "ranking": ranking,
        "paired_statistics": paired_statistics(rows), "task_breakdown": breakdown,
        "intent_selection": selection, "actor_summaries": actor_summaries}
    write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    with (args.output / "task_family_breakdown.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
        writer.writeheader(); writer.writerows(breakdown)
    expected = len(args.seeds) * len(trajectories) * 2 * len(METHODS)
    audit = {"passed": len(rows) == expected and all(r["verification_coverage"] for r in rows),
        "sessions": len(rows), "expected_sessions": expected,
        "confirmation_sha256": digest(args.output / "confirmation_sessions.jsonl")}
    write(args.output, "audit.json", audit)
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
