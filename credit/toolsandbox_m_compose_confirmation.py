"""Independent ToolSandbox-M confirmation of CREDIT-Compose.

The prior blind-composition suite is development evidence.  This script first
turns its requested-state-successful episodes into provenance-linked natural
recipes, freezes a generic schema coverage selector, and then evaluates it on
new operation graphs and new actor seeds.  Requested world state is primary;
process compliance and normalized final-answer semantics are secondary.
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

import numpy as np
from scipy.stats import binomtest

from .remote_json_agent import RemoteJSONAgent
from .toolsandbox_local_v3 import (
    consume,
    finish,
    initial_job,
    install_local_clock,
    run_batch,
    upstream_imports,
    write,
)
from .toolsandbox_m import continuous_score
from .toolsandbox_m_compose_selector import (
    build_composition_pool,
    compose_select,
    schema_requirements,
)
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_intent_optimization import balanced_order, select as prior_select
from .user_data import digest


CLOCK = "2026-10-14T12:00:00"
ACTOR_SEEDS = (997, 1009, 1013, 1019, 1021)
METHODS = ("NoMem", "FIFO", "Semantic", "CREDIT-Compact", "CREDIT-Intent",
           "CREDIT-Compose")
LATENCY_BUDGETS_MS = (10000, 15000, 20000, 30000)
PRIMARY_BUDGET_MS = 15000
BOOTSTRAP_SEED = 20261017


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _imports():
    import polars as pl
    from tool_sandbox.common.evaluation import (
        Milestone,
        SnapshotConstraint,
        addition_similarity,
        removal_similarity,
        snapshot_similarity,
        update_similarity,
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


def build_confirmation_scenarios(u):
    """Create six task families with graphs absent from prior evaluation."""
    x = _imports()
    pl, M, C = x["pl"], x["Milestone"], x["SnapshotConstraint"]
    DB, R, Cat = x["DatabaseNamespace"], x["RoleType"], x["ScenarioCategories"]
    add, remove, snap, update = (x["addition_similarity"], x["removal_similarity"],
                                 x["snapshot_similarity"], x["update_similarity"])
    Extension = x["ScenarioExtension"]
    bases = x["named_base_scenarios"](u.ToolBackend.DEFAULT)
    user_instruction = x["USER_INSTRUCTION"]
    tomorrow_10 = datetime.datetime(2026, 10, 15, 10).timestamp()

    fredrik_id = "9e137f06-916a-5310-8174-cf0b7e9f7054"
    fredrik_phone = "+12453344098"
    boss_phone = "+10000000000"
    upcoming_id = "54f20be9-eac3-5e76-8e57-724e48d3d477"
    upcoming_time = float(next(row["reminder_timestamp"] for row in
        bases["base"].starting_context.get_database(DB.REMINDER).to_dicts()
        if row["reminder_id"] == upcoming_id))
    gpu_phone = "+18307976530"

    def messages(private: str, public: str):
        return [
            {"sender": R.SYSTEM, "recipient": R.USER,
             "content": user_instruction + private + " You do not have any more information."},
            {"sender": R.USER, "recipient": R.AGENT, "content": public},
        ]

    def trace(tool: str):
        return M(snapshot_constraints=[C(
            database_namespace=DB.SANDBOX,
            snapshot_constraint=snap,
            target_dataframe=pl.DataFrame({
                "sender": R.EXECUTION_ENVIRONMENT,
                "recipient": R.AGENT,
                "tool_trace": json.dumps({"tool_name": tool, "arguments": {}},
                                          ensure_ascii=False),
            }),
        )])

    scenarios: dict = {}
    metadata: dict = {}

    def extension(name, family, public, private, base, tools, milestones, edges,
                  state_indices, essential_trace_indices):
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
            "state_indices": state_indices,
            "essential_trace_indices": essential_trace_indices,
            "process_state_indices": sorted(set(state_indices + essential_trace_indices)),
            "operation_graph": tools,
        }
        return scenario

    # 1. Lookup an unknown sender, save the number, then reply to it.
    for suffix, public in (
        ("", "Find the sender of the GPU offer, save them as GPU Vendor (vendor), and reply 'No thanks'."),
        ("_alt", "Add whoever texted the GPU pitch as GPU Vendor with relationship vendor, then send No thanks."),
    ):
        name = "compose_save_sender_and_reply" + suffix
        ms = [
            trace("search_messages"),
            M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "name": "GPU Vendor", "phone_number": gpu_phone,
                "relationship": "vendor", "is_self": False,
            }), reference_milestone_node_index=0)]),
            M(snapshot_constraints=[C(database_namespace=DB.MESSAGING,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "recipient_phone_number": gpu_phone, "content": "No thanks",
            }), reference_milestone_node_index=1)]),
        ]
        scenarios[name] = extension(name, "save_sender_and_reply", public,
            "Find the GPU-offer message sender, add that number as GPU Vendor and vendor, then reply No thanks.",
            "base", ["search_messages", "add_contact", "send_message_with_phone_number"],
            ms, [(0, 1), (1, 2)], [1, 2], [0])

    # 2. Resolve a contact, update it, and notify the same phone.
    for suffix, public in (
        ("", "Change Fredrik Thordendal's relationship to collaborator, then text him 'Profile updated'."),
        ("_alt", "Mark Fredrik Thordendal as my collaborator and message him: Profile updated."),
    ):
        name = "compose_update_contact_and_notify" + suffix
        ms = [
            trace("search_contacts"),
            M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=update, target_dataframe=pl.DataFrame({
                "person_id": fredrik_id, "relationship": "collaborator",
            }), reference_milestone_node_index=0)]),
            M(snapshot_constraints=[C(database_namespace=DB.MESSAGING,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "recipient_phone_number": fredrik_phone, "content": "Profile updated",
            }), reference_milestone_node_index=1)]),
        ]
        scenarios[name] = extension(name, "update_contact_and_notify", public,
            "Find Fredrik Thordendal, change the relationship to collaborator, and send Profile updated to that phone.",
            "base", ["search_contacts", "modify_contact", "send_message_with_phone_number"],
            ms, [(0, 1), (1, 2)], [1, 2], [0])

    # 3. Remove and recreate a resolved contact, then send to the preserved phone.
    for suffix, public in (
        ("", "Replace Fredrik Thordendal's contact with Fredrik T at the same number as a collaborator, then text 'Team record updated'."),
        ("_alt", "Recreate Fredrik Thordendal as Fredrik T, keep his phone, set collaborator, and send Team record updated."),
    ):
        name = "compose_replace_contact_and_notify" + suffix
        ms = [
            trace("search_contacts"),
            M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=remove, target_dataframe=pl.DataFrame({
                "person_id": fredrik_id,
            }), reference_milestone_node_index=0)]),
            M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "name": "Fredrik T", "phone_number": fredrik_phone,
                "relationship": "collaborator", "is_self": False,
            }), reference_milestone_node_index=1)]),
            M(snapshot_constraints=[C(database_namespace=DB.MESSAGING,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "recipient_phone_number": fredrik_phone, "content": "Team record updated",
            }), reference_milestone_node_index=2)]),
        ]
        scenarios[name] = extension(name, "replace_contact_and_notify", public,
            "Find and remove Fredrik Thordendal, add Fredrik T with the returned phone and collaborator relationship, then message Team record updated.",
            "base", ["search_contacts", "remove_contact", "add_contact",
                     "send_message_with_phone_number"],
            ms, [(0, 1), (1, 2), (2, 3)], [1, 2, 3], [0])

    # 4. Use one message value in both a dated reminder and a message to boss.
    for suffix, public in (
        ("", "Use my newest message as a reminder for tomorrow at 10 AM and send it to my boss prefixed 'Latest text: '."),
        ("_alt", "At 10 tomorrow, remind me about my latest text; also send my boss that text with Latest text: before it."),
    ):
        name = "compose_message_to_reminder_and_boss" + suffix
        ms = [
            trace("search_messages"),
            trace("get_current_timestamp"),
            M(snapshot_constraints=[C(database_namespace=DB.REMINDER,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "content": "Good, keep me posted", "reminder_timestamp": tomorrow_10,
            }), reference_milestone_node_index=1)]),
            trace("search_contacts"),
            M(snapshot_constraints=[C(database_namespace=DB.MESSAGING,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "recipient_phone_number": boss_phone,
                "content": "Latest text: Good, keep me posted",
            }), reference_milestone_node_index=3)]),
        ]
        scenarios[name] = extension(name, "message_to_reminder_and_boss", public,
            "Find the newest message, create a reminder with that content for tomorrow 10 AM, find the boss, and send the content prefixed Latest text:.",
            "base", ["search_messages", "get_current_timestamp",
                     "timestamp_to_datetime_info", "datetime_info_to_timestamp",
                     "add_reminder", "search_contacts", "send_message_with_phone_number"],
            ms, [(0, 2), (1, 2), (0, 4), (2, 4), (3, 4)], [2, 4], [0, 1, 3])

    # 5. Replace an identified reminder at its returned time and notify boss.
    for suffix, public in (
        ("", "Replace my upcoming reminder with 'Submit camera-ready' at the same time, then text my boss 'Reminder replaced'."),
        ("_alt", "Recreate the next reminder as Submit camera-ready without changing its time, and message my boss: Reminder replaced."),
    ):
        name = "compose_replace_reminder_and_notify" + suffix
        ms = [
            trace("search_reminder"),
            trace("get_current_timestamp"),
            M(snapshot_constraints=[C(database_namespace=DB.REMINDER,
                snapshot_constraint=remove, target_dataframe=pl.DataFrame({
                "reminder_id": upcoming_id,
            }), reference_milestone_node_index=0)]),
            M(snapshot_constraints=[C(database_namespace=DB.REMINDER,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "content": "Submit camera-ready", "reminder_timestamp": upcoming_time,
            }), reference_milestone_node_index=2)]),
            trace("search_contacts"),
            M(snapshot_constraints=[C(database_namespace=DB.MESSAGING,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "recipient_phone_number": boss_phone, "content": "Reminder replaced",
            }), reference_milestone_node_index=4)]),
        ]
        scenarios[name] = extension(name, "replace_reminder_and_notify", public,
            "Find and remove the upcoming reminder, add Submit camera-ready at its returned time, find the boss, and send Reminder replaced.",
            "base", ["search_reminder", "get_current_timestamp", "remove_reminder",
                     "add_reminder", "search_contacts", "send_message_with_phone_number"],
            ms, [(0, 2), (1, 2), (2, 3), (3, 5), (4, 5)], [2, 3, 5], [0, 1, 4])

    # 6. Repair a device prerequisite, add a contact, and send a message.
    for suffix, public in (
        ("", "Restore cellular service, add Casey Lin at +14155550173 as designer, and text 'Connected and welcome'."),
        ("_alt", "Get mobile service working, save Casey Lin (+14155550173, designer), then send Connected and welcome."),
    ):
        name = "compose_recover_add_and_send" + suffix
        ms = [
            M(snapshot_constraints=[C(database_namespace=DB.SETTING,
                snapshot_constraint=snap,
                target_dataframe=pl.DataFrame({"low_battery_mode": False}))]),
            M(snapshot_constraints=[C(database_namespace=DB.SETTING,
                snapshot_constraint=snap,
                target_dataframe=pl.DataFrame({"cellular": True}))]),
            M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "name": "Casey Lin", "phone_number": "+14155550173",
                "relationship": "designer", "is_self": False,
            }), reference_milestone_node_index=1)]),
            M(snapshot_constraints=[C(database_namespace=DB.MESSAGING,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                "recipient_phone_number": "+14155550173",
                "content": "Connected and welcome",
            }), reference_milestone_node_index=2)]),
        ]
        scenarios[name] = extension(name, "recover_add_and_send", public,
            "Disable low-battery mode if needed, enable cellular, add Casey Lin as designer, and send Connected and welcome.",
            "base_low_battery_mode_on_wifi_off_location_service_off_cellular_off",
            ["get_low_battery_mode_status", "set_low_battery_mode_status",
             "get_cellular_service_status", "set_cellular_service_status",
             "add_contact", "send_message_with_phone_number"],
            ms, [(0, 1), (1, 2), (2, 3)], [0, 1, 2, 3], [])

    trajectories = []
    for family in (
        "save_sender_and_reply", "update_contact_and_notify",
        "replace_contact_and_notify", "message_to_reminder_and_boss",
        "replace_reminder_and_notify", "recover_add_and_send",
    ):
        names = [name for name, spec in metadata.items() if spec["family"] == family]
        trajectories.append({"trajectory": family, "sessions": names})
    return scenarios, metadata, trajectories


def _score(evaluation: dict | None, index: int) -> float:
    if not evaluation:
        return 0.0
    values = evaluation["milestone_mapping"]
    pair = values.get(index, values.get(str(index), (-1, 0.0)))
    return float(pair[1])


def semantic_answer_success(row: dict, family: str) -> int:
    answer = ""
    if row.get("generations"):
        answer = str(row["generations"][-1].get("action", {}).get("answer", ""))
    text = " ".join(answer.casefold().replace("-", " ").split())
    if any(phrase in text for phrase in ("could not", "unable", "failed", "please provide")):
        return 0
    rules = {
        "save_sender_and_reply": (("no thanks",), ("gpu vendor", "saved", "contact")),
        "update_contact_and_notify": (("collaborator",), ("profile updated", "sent", "message")),
        "replace_contact_and_notify": (("fredrik t",), ("team record updated", "sent", "message")),
        "message_to_reminder_and_boss": (("reminder",), ("sent", "boss", "message")),
        "replace_reminder_and_notify": (("reminder",), ("replaced", "sent", "boss")),
        "recover_add_and_send": (("casey", "welcome"), ("cellular", "mobile", "connected", "sent")),
    }
    required, alternatives = rules[family]
    return int(bool(text) and all(term in text for term in required)
               and any(term in text for term in alternatives))


def annotate(row: dict, metadata: dict) -> dict:
    spec = metadata[row["task"]]
    evaluation = row.get("upstream_evaluation")
    minefield_clear = bool(evaluation and evaluation["minefield_similarity"] == 0)
    state = [_score(evaluation, index) for index in spec["state_indices"]]
    process = [_score(evaluation, index) for index in spec["process_state_indices"]]
    traces = [_score(evaluation, index) for index in spec["essential_trace_indices"]]
    row.update({
        "requested_state_score": float(np.mean(state)) if state else 0.0,
        "requested_state_success": int(minefield_clear and
                                       all(value >= 1 - 1e-8 for value in state)),
        "process_state_score": float(np.mean(process)) if process else 0.0,
        "process_state_success": int(minefield_clear and
                                     all(value >= 1 - 1e-8 for value in process)),
        "essential_trace_score": float(np.mean(traces)) if traces else 1.0,
        "semantic_answer_success": semantic_answer_success(row, spec["family"]),
        "family": spec["family"],
    })
    return row


def golden_self_check(u, scenarios, metadata) -> dict:
    tomorrow_10 = datetime.datetime(2026, 10, 15, 10).timestamp()
    replacement = scenarios.get("compose_replace_reminder_and_notify")
    replacement_time = 0.0 if replacement is None else float(next(
        row["reminder_timestamp"] for row in
        replacement.starting_context.get_database(u.DatabaseNamespace.REMINDER).to_dicts()
        if row["reminder_id"] == "54f20be9-eac3-5e76-8e57-724e48d3d477"))
    actions = {
        "save_sender_and_reply": [
            {"tool": "search_messages", "arguments": {"content": "Hey kid, you want some GPU?"}},
            {"tool": "add_contact", "arguments": {"name": "GPU Vendor", "phone_number": "+18307976530", "relationship": "vendor"}},
            {"tool": "send_message_with_phone_number", "arguments": {"phone_number": "+18307976530", "content": "No thanks"}},
            {"answer": "Saved GPU Vendor and sent No thanks."},
        ],
        "update_contact_and_notify": [
            {"tool": "search_contacts", "arguments": {"name": "Fredrik Thordendal"}},
            {"tool": "modify_contact", "arguments": {"person_id": "9e137f06-916a-5310-8174-cf0b7e9f7054", "relationship": "collaborator"}},
            {"tool": "send_message_with_phone_number", "arguments": {"phone_number": "+12453344098", "content": "Profile updated"}},
            {"answer": "Fredrik is now a collaborator and Profile updated was sent."},
        ],
        "replace_contact_and_notify": [
            {"tool": "search_contacts", "arguments": {"name": "Fredrik Thordendal"}},
            {"tool": "remove_contact", "arguments": {"person_id": "9e137f06-916a-5310-8174-cf0b7e9f7054"}},
            {"tool": "add_contact", "arguments": {"name": "Fredrik T", "phone_number": "+12453344098", "relationship": "collaborator"}},
            {"tool": "send_message_with_phone_number", "arguments": {"phone_number": "+12453344098", "content": "Team record updated"}},
            {"answer": "Fredrik T was saved and Team record updated was sent."},
        ],
        "message_to_reminder_and_boss": [
            {"tool": "search_messages", "arguments": {"content": "Good, keep me posted"}},
            {"tool": "get_current_timestamp", "arguments": {}},
            {"tool": "add_reminder", "arguments": {"content": "Good, keep me posted", "reminder_timestamp": tomorrow_10}},
            {"tool": "search_contacts", "arguments": {"relationship": "boss"}},
            {"tool": "send_message_with_phone_number", "arguments": {"phone_number": "+10000000000", "content": "Latest text: Good, keep me posted"}},
            {"answer": "The reminder was created and the message was sent to my boss."},
        ],
        "replace_reminder_and_notify": [
            {"tool": "search_reminder", "arguments": {"content": "Buy a nice rich navy bathing dress"}},
            {"tool": "get_current_timestamp", "arguments": {}},
            {"tool": "remove_reminder", "arguments": {"reminder_id": "54f20be9-eac3-5e76-8e57-724e48d3d477"}},
            {"tool": "add_reminder", "arguments": {"content": "Submit camera-ready", "reminder_timestamp": replacement_time}},
            {"tool": "search_contacts", "arguments": {"relationship": "boss"}},
            {"tool": "send_message_with_phone_number", "arguments": {"phone_number": "+10000000000", "content": "Reminder replaced"}},
            {"answer": "The reminder was replaced and my boss was sent the update."},
        ],
        "recover_add_and_send": [
            {"tool": "set_low_battery_mode_status", "arguments": {"on": False}},
            {"tool": "set_cellular_service_status", "arguments": {"on": True}},
            {"tool": "add_contact", "arguments": {"name": "Casey Lin", "phone_number": "+14155550173", "relationship": "designer"}},
            {"tool": "send_message_with_phone_number", "arguments": {"phone_number": "+14155550173", "content": "Connected and welcome"}},
            {"answer": "Cellular is connected, Casey was saved, and welcome was sent."},
        ],
    }
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        for name, scenario in scenarios.items():
            job = initial_job(name, scenario, [], u, None, "golden_preflight", "gold")
            for index, action in enumerate(actions[metadata[name]["family"]]):
                consume(job, {"agent_output": json.dumps(action),
                    "prompt_sha256": f"gold-{index}", "hit_generation_limit": False}, u, 10)
            row = annotate(finish(job, Path(temporary), u), metadata)
            checks.append({"task": name, "upstream_success": row["success"],
                "requested_state_success": row["requested_state_success"],
                "process_state_success": row["process_state_success"],
                "semantic_answer_success": row["semantic_answer_success"]})
    passed = all(all(item[key] == 1 for key in (
        "upstream_success", "requested_state_success", "process_state_success",
        "semantic_answer_success")) for item in checks)
    if not passed:
        raise RuntimeError("golden evaluator preflight failed: " + json.dumps(checks))
    return {"passed": passed, "tasks": len(checks), "checks": checks}


def novelty_audit(u, scenarios, metadata, source: Path) -> dict:
    def normalized(tools) -> tuple[str, ...]:
        return tuple(sorted(set(tools or []) - {"end_conversation"}))

    official = {**u.named_single_tool_call_scenarios(u.ToolBackend.DEFAULT),
                **u.named_multiple_tool_call_scenarios(u.ToolBackend.DEFAULT)}
    previous_sets = defaultdict(list)
    for name, scenario in official.items():
        previous_sets[normalized(scenario.starting_context.tool_allow_list)].append(
            "official:" + name)
    previous_meta = json.loads((source / "scenario_metadata.json").read_text())
    for name, item in previous_meta.items():
        previous_sets[normalized(item["operation_graph"])].append("blind_v1:" + name)
    overlap = {}
    for name, scenario in scenarios.items():
        key = normalized(scenario.starting_context.tool_allow_list)
        if key in previous_sets:
            overlap[name] = previous_sets[key]
    old_hashes = set()
    for line in (source / "confirmation_sessions.jsonl").read_text().splitlines():
        row = json.loads(line)
        if isinstance(row.get("public_request_sha256"), str):
            old_hashes.add(row["public_request_sha256"])
    current = {_sha(item["public_request"]) for item in metadata.values()}
    prompt_overlap = sorted(current & old_hashes)
    passed = not overlap and not prompt_overlap and len(current) == len(metadata)
    if not passed:
        raise RuntimeError("confirmation novelty preflight failed")
    return {"passed": passed, "tasks": len(metadata),
        "unique_public_request_hashes": len(current),
        "exact_prompt_overlap": prompt_overlap,
        "exact_allowed_tool_set_overlap": overlap,
        "comparison_sets": "official ToolSandbox plus blind composition v1"}


def select(method: str, pool: list[dict], task: str, query: str, scenario,
           pair_vetoes: set[tuple[str, str]]) -> list[dict]:
    if method == "CREDIT-Compose":
        return compose_select(pool, query, scenario)
    return prior_select(method, pool, task, query, scenario, pair_vetoes)


def run_actor(seed, actor_index, trajectories, root, args, scenarios, metadata,
              pool, pair_vetoes, u):
    directory = root / f"compose_confirmation_seed_{seed}"
    directory.mkdir(parents=True)
    agent = RemoteJSONAgent(directory, seed, env=args.env, tokenizer_path=args.tokenizer,
        temperature=0.0, max_new_tokens=384, workers=args.workers,
        exact_prompt_cache=False)
    agent.binding_interface = "tool_json"
    agent.manifest["benchmark"] = "ToolSandbox-M CREDIT-Compose confirmation"
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
                jobs.append(initial_job_v2(task, scenario, bank, u, agent.tokenizer,
                                           "compose_confirmation", method))
                job_meta[method] = (bank, position)
            for row in run_batch(jobs, agent, directory, u, max_calls=10):
                bank, position = job_meta[row["method"]]
                item = annotate({**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"],
                    "session_index": session_index,
                    "selected_bank": [note["uid"] for note in bank],
                    "remote_latency_ms": session_latency(row),
                    "batch_position": position,
                    "public_request_sha256": _sha(query)}, metadata)
                sessions.append(item)
                with (directory / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    agent.close()
    summary = {"actor_seed": seed, "sessions": len(sessions),
        "remote_requests": agent.calls, "method_batch_position_counts": dict(positions)}
    write(directory, "summary.json", summary)
    return sessions, summary


def summarize(rows: list[dict]) -> list[dict]:
    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        latency = [row["remote_latency_ms"] for row in subset]
        item = {"method": method, "sessions": len(subset),
            "requested_state_VSR": float(np.mean([r["requested_state_success"] for r in subset])),
            "process_state_VSR": float(np.mean([r["process_state_success"] for r in subset])),
            "semantic_answer_VSR": float(np.mean([r["semantic_answer_success"] for r in subset])),
            "upstream_strict_VSR": float(np.mean([r["success"] for r in subset])),
            "mean_requested_state_score": float(np.mean([r["requested_state_score"] for r in subset])),
            "upstream_mean_score": float(np.mean([continuous_score(r) for r in subset])),
            "mean_tool_calls": float(np.mean([r["tool_calls"] for r in subset])),
            "mean_remote_requests": float(np.mean([len(r["generations"]) for r in subset])),
            "mean_online_remote_latency_ms": float(np.mean(latency)),
            "p95_online_remote_latency_ms": float(np.quantile(latency, .95))}
        for budget in LATENCY_BUDGETS_MS:
            item[f"requested_state_VSR_at_{budget}ms"] = float(np.mean([
                row["requested_state_success"] and value <= budget
                for row, value in zip(subset, latency)]))
            item[f"completion_at_{budget}ms"] = float(np.mean([
                value <= budget for value in latency]))
        table.append(item)
    return table


def paired_statistics(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for row in rows:
        by[row["method"]][(row["actor_seed"], row["trajectory"],
                           row["session_index"])] = row
    clusters = sorted({(row["actor_seed"], row["trajectory"]) for row in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    report = []
    for method in METHODS:
        state_effects, budget_effects = [], []
        for seed, trajectory in clusters:
            keys = [(seed, trajectory, index) for index in (0, 1)]
            state_effects.append(float(np.mean([
                by["CREDIT-Compose"][key]["requested_state_success"] -
                by[method][key]["requested_state_success"] for key in keys])))
            budget_effects.append(float(np.mean([
                (by["CREDIT-Compose"][key]["requested_state_success"] and
                 by["CREDIT-Compose"][key]["remote_latency_ms"] <= PRIMARY_BUDGET_MS) -
                (by[method][key]["requested_state_success"] and
                 by[method][key]["remote_latency_ms"] <= PRIMARY_BUDGET_MS)
                for key in keys])))
        state_boot = np.mean(np.asarray(state_effects)[draws], axis=1)
        budget_boot = np.mean(np.asarray(budget_effects)[draws], axis=1)
        wins = sum(value > 0 for value in state_effects)
        losses = sum(value < 0 for value in state_effects)
        p_value = float(binomtest(wins, wins + losses, .5,
                                  alternative="greater").pvalue) \
            if wins + losses else 1.0
        report.append({"baseline": method,
            "delta_requested_state_VSR": float(np.mean(state_effects)),
            "state_cluster_bootstrap_CI95": np.quantile(
                state_boot, [.025, .975]).tolist(),
            f"delta_state_VSR_at_{PRIMARY_BUDGET_MS}ms": float(np.mean(budget_effects)),
            "budget_cluster_bootstrap_CI95": np.quantile(
                budget_boot, [.025, .975]).tolist(),
            "cluster_wins": wins, "cluster_losses": losses,
            "one_sided_exact_sign_p": p_value})
    return {"comparisons": report, "clusters": len(clusters),
        "resampling_unit": "actor seed by two-prompt task family",
        "bootstrap_replicates": 10000, "bootstrap_seed": BOOTSTRAP_SEED}


def holm_primary(report: dict) -> list[dict]:
    selected = [dict(row) for row in report["comparisons"]
                if row["baseline"] in ("FIFO", "Semantic", "CREDIT-Intent")]
    ordered = sorted(selected, key=lambda row: row["one_sided_exact_sign_p"])
    running = 0.0
    for index, row in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - index) * row["one_sided_exact_sign_p"])
        running = max(running, adjusted)
        row["holm_adjusted_p"] = running
    return sorted(ordered, key=lambda row: row["baseline"])


def task_breakdown(rows: list[dict]) -> list[dict]:
    table = []
    for family in sorted({row["trajectory"] for row in rows}):
        for method in METHODS:
            subset = [row for row in rows
                      if row["trajectory"] == family and row["method"] == method]
            table.append({"family": family, "method": method, "sessions": len(subset),
                "requested_state_VSR": float(np.mean([
                    row["requested_state_success"] for row in subset])),
                "process_state_VSR": float(np.mean([
                    row["process_state_success"] for row in subset])),
                "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
                "mean_remote_latency_ms": float(np.mean([
                    row["remote_latency_ms"] for row in subset]))})
    return table


def freeze(args, source_protocol, scenarios, metadata, trajectories, pool, added,
           provenance, pair_vetoes, preflight) -> dict:
    names = ["toolsandbox_m_compose_confirmation.py",
             "toolsandbox_m_compose_confirmation_verify.py",
             "toolsandbox_m_compose_selector.py", "toolsandbox_m_intent_optimization.py",
             "toolsandbox_m_streaming_optimization.py", "toolsandbox_m_followup.py",
             "toolsandbox_local_v3.py", "toolsandbox_m.py", "remote_json_agent.py"]
    hashes = {name: digest(Path(__file__).with_name(name)) for name in names
              if Path(__file__).with_name(name).exists()}
    records = [{"name": name, **metadata[name],
        "public_request_sha256": _sha(metadata[name]["public_request"])}
        for name in sorted(metadata)]
    protocol = {
        "suite": "ToolSandbox-M CREDIT-Compose independent confirmation",
        "frozen_before_remote_calls": True,
        "development_source": str(args.source.resolve()),
        "development_confirmation_sha256": digest(args.source / "confirmation_sessions.jsonl"),
        "development_posthoc_audit_sha256": digest(args.source / "posthoc_semantic_audit.json"),
        "source_sha256": hashes,
        "upstream_directory": source_protocol["upstream_directory"],
        "upstream_source_sha256": source_protocol["upstream_source_sha256"],
        "clock": CLOCK, "tasks": records, "trajectories": trajectories,
        "actor_seeds": list(args.seeds), "methods": list(METHODS),
        "primary_endpoint": "requested_state_VSR: all requested world-state constraints exact and no minefield",
        "primary_latency_endpoint": f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms",
        "secondary_endpoints": ["process_state_VSR", "semantic_answer_VSR",
                                "upstream_strict_VSR", "mean_tool_calls"],
        "primary_comparisons": ["CREDIT-Compose vs FIFO", "CREDIT-Compose vs Semantic",
                                "CREDIT-Compose vs CREDIT-Intent"],
        "multiplicity": "Holm adjustment over three registered primary comparisons",
        "inference": "paired actor-seed by task-family cluster bootstrap and exact sign test",
        "selector": "weighted allowed-tool schema set cover, depth two, no target outcomes",
        "schema_weights": schema_requirements(set().union(*[
            set(s.starting_context.tool_allow_list or []) for s in scenarios.values()])),
        "memory_pool_size": len(pool), "derived_memory_count": len(added),
        "derived_memory_uids": [note["uid"] for note in added],
        "memory_provenance": provenance,
        "preflight": preflight,
        "historical_confirmation_examined": True,
        "claim_scope": "new synthetic task graphs, new prompts, and new actor seeds after prior suite became development data",
        "maximum_tool_calls_per_session": 10,
        "paired_scheduling": "all six methods for a task share one concurrent remote batch",
        "temperature": 0.0, "exact_prompt_cache": False,
        "provider_credentials_persisted": False,
        "pair_vetoes": sorted([list(pair) for pair in pair_vetoes]),
        "smoke": args.smoke,
    }
    write(args.output, "frozen_protocol.json", protocol)
    write(args.output, "scenario_metadata.json", metadata)
    write(args.output, "memory_pool.json", pool)
    write(args.output, "derived_memories.json", added)
    write(args.output, "memory_provenance.json", provenance)
    for name in hashes:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/toolsandbox_m_blind_composition_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/toolsandbox_m_compose_confirmation_v1"))
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
    scenarios, metadata, trajectories = build_confirmation_scenarios(u)
    if args.smoke:
        args.seeds = args.seeds[:1]
        trajectories = trajectories[:1]
        keep = {name for trajectory in trajectories for name in trajectory["sessions"]}
        scenarios = {name: value for name, value in scenarios.items() if name in keep}
        metadata = {name: value for name, value in metadata.items() if name in keep}
    manifest = args.source / f"blind_composition_seed_{source_protocol['actor_seeds'][0]}" / "agent_manifest.json"
    args.tokenizer = args.tokenizer or Path(json.loads(manifest.read_text())["tokenizer_path"])
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    posthoc = json.loads((args.source / "posthoc_semantic_audit.json").read_text())
    pool, added, provenance = build_composition_pool(
        args.source, tokenizer,
        float(posthoc["replacement_expected_timestamp_from_frozen_start_state"]))
    pair_vetoes = {tuple(pair) for pair in source_protocol["pair_vetoes"]}
    preflight = {"golden_evaluator": golden_self_check(u, scenarios, metadata),
                 "novelty": novelty_audit(u, scenarios, metadata, args.source)}
    write(args.output, "preflight.json", preflight)
    protocol = freeze(args, source_protocol, scenarios, metadata, trajectories,
                      pool, added, provenance, pair_vetoes, preflight)
    selection = {}
    for task, scenario in scenarios.items():
        query = metadata[task]["public_request"]
        bank = compose_select(pool, query, scenario)
        selection[task] = {"selected_uids": [note["uid"] for note in bank],
            "selected_sources": [note["source_task"] for note in bank],
            "covered_tools": sorted(set().union(*[
                set(note["tools"]) for note in bank]) &
                (set(scenario.starting_context.tool_allow_list or []) - {"end_conversation"})),
            "allowed_tools": sorted(set(scenario.starting_context.tool_allow_list or []) -
                                    {"end_conversation"})}
    write(args.output, "frozen_selection.json", selection)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "tasks": len(scenarios),
                          "derived_memories": len(added), "selection": selection}, indent=2))
        return

    rows, actor_summaries = [], []
    for index, seed in enumerate(args.seeds):
        found, actor_summary = run_actor(seed, index, trajectories, args.output, args,
            scenarios, metadata, pool, pair_vetoes, u)
        rows.extend(found)
        actor_summaries.append(actor_summary)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = summarize(rows)
    statistics = paired_statistics(rows)
    breakdown = task_breakdown(rows)
    ranking = sorted((dict(row) for row in table), key=lambda row: (
        -row[f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms"],
        -row["requested_state_VSR"], -row["process_state_VSR"],
        row["mean_online_remote_latency_ms"], row["method"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    summary = {"confirmation_table": table, "ranking": ranking,
        "paired_statistics": statistics, "holm_primary": holm_primary(statistics),
        "task_breakdown": breakdown, "frozen_selection": selection,
        "actor_summaries": actor_summaries}
    write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    with (args.output / "task_family_breakdown.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
        writer.writeheader(); writer.writerows(breakdown)
    expected = len(args.seeds) * len(trajectories) * 2 * len(METHODS)
    audit = {"passed": len(rows) == expected and all(
        row["verification_coverage"] for row in rows),
        "sessions": len(rows), "expected_sessions": expected,
        "confirmation_sha256": digest(args.output / "confirmation_sessions.jsonl")}
    write(args.output, "audit.json", audit)
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
