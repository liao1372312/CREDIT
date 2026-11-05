"""Frozen held-out confirmation for latency-aware execution contracts.

Every task exposes the same wide tool catalogue.  Contracts, validation, and
repairs use only the public request, public schemas, successful tool results,
and the proposed action.  Exact final physical state is the primary endpoint.
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

from . import toolsandbox_local_v3 as adapter
from .remote_json_agent import RemoteJSONAgent
from .toolsandbox_m_compose_confirmation import _imports
from .toolsandbox_m_compose_selector import compose_select
from .toolsandbox_m_contract_fast import (
    consume_fast, consume_validate, fast_contract_select, fast_select,
    matched_recipe_select, request_required_mutations,
)
from .toolsandbox_m_contract_fast_development import finish_direct
from .toolsandbox_m_contract_confirmation import novelty_audit
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_intent_optimization import balanced_order, select as prior_select
from .user_data import digest


CLOCK = "2026-10-18T12:00:00"
ACTOR_SEEDS = (1501, 1503, 1511, 1519, 1523)
METHODS = ("Matched-Recipe", "Contract-Text", "Contract-Validate",
           "Contract-FastRepair", "Contract-Adaptive", "Semantic",
           "CREDIT-Compose")
LATENCY_BUDGETS_MS = (10000, 15000, 20000, 30000)
PRIMARY_BUDGET_MS = 15000
MAX_MODEL_TURNS = 12
BOOTSTRAP_SEED = 20261019

WIDE_TOOLS = (
    "get_current_timestamp", "shift_timestamp", "timestamp_to_datetime_info",
    "datetime_info_to_timestamp", "search_messages",
    "send_message_with_phone_number", "search_contacts", "add_contact",
    "remove_contact", "modify_contact", "search_reminder", "add_reminder",
    "remove_reminder", "get_wifi_status", "set_wifi_status",
    "get_location_service_status", "set_location_service_status",
    "get_cellular_service_status", "set_cellular_service_status",
    "get_low_battery_mode_status", "set_low_battery_mode_status",
)


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def build_confirmation_scenarios(u):
    """Build four risk combinations and two hard neutral controls."""
    x = _imports()
    pl, M, C = x["pl"], x["Milestone"], x["SnapshotConstraint"]
    DB, R, Cat = x["DatabaseNamespace"], x["RoleType"], x["ScenarioCategories"]
    add, remove, snap, update = (x["addition_similarity"], x["removal_similarity"],
                                 x["snapshot_similarity"], x["update_similarity"])
    Extension = x["ScenarioExtension"]
    bases = x["named_base_scenarios"](u.ToolBackend.DEFAULT)
    user_instruction = x["USER_INSTRUCTION"]

    day2_1020 = datetime.datetime(2026, 10, 20, 10, 20).timestamp()
    tomorrow_1410 = datetime.datetime(2026, 10, 19, 14, 10).timestamp()
    tomorrow_0730 = datetime.datetime(2026, 10, 19, 7, 30).timestamp()
    boss_id, boss_phone = "e3570ab6-0819-5032-be1e-2b366390c8ef", "+10000000000"
    fredrik_id = "9e137f06-916a-5310-8174-cf0b7e9f7054"
    john_id, john_phone = "a22e1984-6c6c-530c-8831-c3ea3b5138e7", "+1234560987"
    next_id = "54f20be9-eac3-5e76-8e57-724e48d3d477"
    next_time = float(next(row["reminder_timestamp"] for row in
        bases["base"].starting_context.get_database(DB.REMINDER).to_dicts()
        if row["reminder_id"] == next_id))

    def messages(private, public):
        return [
            {"sender": R.SYSTEM, "recipient": R.USER,
             "content": user_instruction + private + " You do not have any more information."},
            {"sender": R.USER, "recipient": R.AGENT, "content": public},
        ]

    def trace(tool):
        return M(snapshot_constraints=[C(
            database_namespace=DB.SANDBOX, snapshot_constraint=snap,
            target_dataframe=pl.DataFrame({"sender": R.EXECUTION_ENVIRONMENT,
                "recipient": R.AGENT, "tool_trace": json.dumps(
                    {"tool_name": tool, "arguments": {}}, ensure_ascii=False)})
        )])

    scenarios, metadata = {}, {}

    def extension(name, family, public, private, milestones, edges, expected):
        scenario = Extension(
            name=name, base_scenario=bases["base"], messages=messages(private, public),
            tool_allow_list=list(WIDE_TOOLS), milestones=milestones,
            milestone_edge_list=edges,
            categories=[Cat.SINGLE_USER_TURN, Cat.CANONICALIZATION],
        ).get_extended_scenario()[name]
        scenarios[name] = scenario
        metadata[name] = {"family": family, "public_request": public,
            "operation_graph": list(WIDE_TOOLS), "expected": expected}

    # Risk 1: global-latest data flow, relative time, reply, and a setting write.
    for suffix, public in (
        ("", "Use my most recent message unchanged for a reminder at 10:20 AM the day after tomorrow, reply 'Filed' to its sender, and turn cellular off."),
        ("_alt", "For the day after tomorrow at 10:20, remind me with the raw content of my newest text; send Filed back to whoever sent it and disable cellular."),
    ):
        ms = [trace("get_current_timestamp"), trace("search_messages"),
              M(snapshot_constraints=[C(database_namespace=DB.REMINDER, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "content": "Good, keep me posted", "reminder_timestamp": day2_1020}))]),
              M(snapshot_constraints=[C(database_namespace=DB.MESSAGING, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "recipient_phone_number": boss_phone, "content": "Filed"}))]),
              M(snapshot_constraints=[C(database_namespace=DB.SETTING, snapshot_constraint=snap, target_dataframe=pl.DataFrame({"cellular": False}))])]
        extension("fast_latest_reply_setting" + suffix, "latest_reply_setting", public,
            "Find the global most recent message, preserve its raw content in a day-after-tomorrow 10:20 reminder, reply Filed to its sender, and disable cellular.",
            ms, [(0,1),(1,2),(1,3),(3,4)],
            {"reminder_time": day2_1020, "boss_phone": boss_phone})

    # Risk 2: contact replacement feeds a message plus an independent reminder.
    for suffix, public in (
        ("", "Replace the John Petrucci contact with JP Archive, keeping his number and marking collaborator; text that number 'Profile moved' and remind me tomorrow at 2:10 PM to Review JP record."),
        ("_alt", "Recreate John Petrucci as JP Archive with the same phone and relationship collaborator, send Profile moved to the preserved phone, and add a Review JP record reminder for 14:10 tomorrow."),
    ):
        ms = [trace("search_contacts"), trace("get_current_timestamp"),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT, snapshot_constraint=remove, target_dataframe=pl.DataFrame({"person_id": john_id}))]),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "name": "JP Archive", "phone_number": john_phone,
                  "relationship": "collaborator", "is_self": False}))]),
              M(snapshot_constraints=[C(database_namespace=DB.MESSAGING, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "recipient_phone_number": john_phone, "content": "Profile moved"}))]),
              M(snapshot_constraints=[C(database_namespace=DB.REMINDER, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "content": "Review JP record", "reminder_timestamp": tomorrow_1410}))])]
        extension("fast_contact_replace_message" + suffix,
            "contact_replace_message", public,
            "Replace John Petrucci with JP Archive at the preserved phone as collaborator, send Profile moved there, and create Review JP record tomorrow at 14:10.",
            ms, [(0,2),(2,3),(3,4),(1,5)],
            {"reminder_time": tomorrow_1410, "john_id": john_id,
             "john_phone": john_phone})

    # Risk 3: replace the upcoming reminder, then update and message a role contact.
    for suffix, public in (
        ("", "Replace my upcoming reminder with 'Confirm venue' at its existing time, set my boss relationship to director, and text the boss 'Reminder updated'."),
        ("_alt", "Rebuild the next reminder as Confirm venue without changing its time; mark my boss as director and send Reminder updated to that contact."),
    ):
        ms = [trace("get_current_timestamp"), trace("search_reminder"),
              trace("search_contacts"),
              M(snapshot_constraints=[C(database_namespace=DB.REMINDER, snapshot_constraint=remove, target_dataframe=pl.DataFrame({"reminder_id": next_id}))]),
              M(snapshot_constraints=[C(database_namespace=DB.REMINDER, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "content": "Confirm venue", "reminder_timestamp": next_time}))]),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT, snapshot_constraint=update, target_dataframe=pl.DataFrame({
                  "person_id": boss_id, "relationship": "director"}))]),
              M(snapshot_constraints=[C(database_namespace=DB.MESSAGING, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "recipient_phone_number": boss_phone, "content": "Reminder updated"}))])]
        extension("fast_reminder_replace_boss" + suffix,
            "reminder_replace_boss", public,
            "Replace the next reminder with Confirm venue at its preserved time, set the boss contact to director, and send Reminder updated to its phone.",
            ms, [(0,1),(1,3),(3,4),(2,5),(5,6)],
            {"next_id": next_id, "next_time": next_time,
             "boss_id": boss_id, "boss_phone": boss_phone})

    # Risk 4: one latest message branches into raw and transformed outputs.
    for suffix, public in (
        ("", "Tomorrow at 7:30 AM, make my latest message a reminder exactly as written; set my boss to manager, add Ops Desk at +14155550182 as coordinator, and send the boss 'Digest: ' followed by that message."),
        ("_alt", "Use the raw newest text for a 07:30 reminder tomorrow, mark my boss as manager, save Ops Desk (+14155550182) as coordinator, then text the boss Digest: plus the raw text."),
    ):
        ms = [trace("get_current_timestamp"), trace("search_messages"), trace("search_contacts"),
              M(snapshot_constraints=[C(database_namespace=DB.REMINDER, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "content": "Good, keep me posted", "reminder_timestamp": tomorrow_0730}))]),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT, snapshot_constraint=update, target_dataframe=pl.DataFrame({
                  "person_id": boss_id, "relationship": "manager"}))]),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "name": "Ops Desk", "phone_number": "+14155550182",
                  "relationship": "coordinator", "is_self": False}))]),
              M(snapshot_constraints=[C(database_namespace=DB.MESSAGING, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "recipient_phone_number": boss_phone,
                  "content": "Digest: Good, keep me posted"}))])]
        extension("fast_latest_branch_contact" + suffix,
            "latest_branch_contact", public,
            "Use the global latest message raw in tomorrow's 07:30 reminder, set boss to manager, add Ops Desk as coordinator, and send boss Digest: plus the raw message.",
            ms, [(0,1),(1,3),(2,4),(4,5),(5,6)],
            {"reminder_time": tomorrow_0730, "boss_id": boss_id,
             "boss_phone": boss_phone})

    # Neutral 1: risk words occur only inside literal payload/name strings.
    for suffix, public in (
        ("", "Add a contact named 'Latest Replace' at +14155550183 as reviewer, send that contact the exact text 'replace latest draft', and turn Wi-Fi off."),
        ("_alt", "Save a contact at +14155550183 as reviewer under the name 'Latest Replace'; text it exactly 'replace latest draft' and disable wifi."),
    ):
        ms = [M(snapshot_constraints=[C(database_namespace=DB.CONTACT, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "name": "Latest Replace", "phone_number": "+14155550183",
                  "relationship": "reviewer", "is_self": False}))]),
              M(snapshot_constraints=[C(database_namespace=DB.MESSAGING, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "recipient_phone_number": "+14155550183",
                  "content": "replace latest draft"}))]),
              M(snapshot_constraints=[C(database_namespace=DB.SETTING, snapshot_constraint=snap, target_dataframe=pl.DataFrame({"wifi": False}))])]
        extension("fast_neutral_quoted_words" + suffix,
            "neutral_quoted_words", public,
            "Add Latest Replace at +14155550183 as reviewer, send the exact literal replace latest draft, and disable wifi.",
            ms, [(0,1),(1,2)], {"phone": "+14155550183"})

    # Neutral 2: multi-source writes with similar tool surface but no risk pattern.
    for suffix, public in (
        ("", "Mark Fredrik Thordendal as teammate, add QA Desk at +14155550184 as auditor, send QA Desk 'Status clean', and turn location services off."),
        ("_alt", "Set Fredrik's relationship to teammate; save QA Desk (+14155550184) as auditor, text it Status clean, and disable location service."),
    ):
        ms = [trace("search_contacts"),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT, snapshot_constraint=update, target_dataframe=pl.DataFrame({
                  "person_id": fredrik_id, "relationship": "teammate"}))]),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "name": "QA Desk", "phone_number": "+14155550184",
                  "relationship": "auditor", "is_self": False}))]),
              M(snapshot_constraints=[C(database_namespace=DB.MESSAGING, snapshot_constraint=add, target_dataframe=pl.DataFrame({
                  "recipient_phone_number": "+14155550184", "content": "Status clean"}))]),
              M(snapshot_constraints=[C(database_namespace=DB.SETTING, snapshot_constraint=snap, target_dataframe=pl.DataFrame({
                  "location_service": False}))])]
        extension("fast_neutral_multisource" + suffix,
            "neutral_multisource", public,
            "Set Fredrik to teammate, add QA Desk at +14155550184 as auditor, send Status clean there, and disable location service.",
            ms, [(0,1),(1,2),(2,3),(3,4)],
            {"fredrik_id": fredrik_id, "phone": "+14155550184"})

    order = ("latest_reply_setting", "contact_replace_message",
             "reminder_replace_boss", "latest_branch_contact",
             "neutral_quoted_words", "neutral_multisource")
    trajectories = [{"trajectory": family, "sessions": sorted(
        name for name, item in metadata.items() if item["family"] == family)}
        for family in order]
    return scenarios, metadata, trajectories


def _latest_rows(context, namespace):
    rows = context["_dbs"][namespace]
    indices = [row["sandbox_message_index"] for row in rows
               if row.get("sandbox_message_index") is not None]
    index = max(indices)
    return [row for row in rows if row.get("sandbox_message_index") == index]


def requested_state_checks(row, spec):
    context, family, expected = row["ending_context"], spec["family"], spec["expected"]
    contacts = _latest_rows(context, "CONTACT")
    reminders = _latest_rows(context, "REMINDER")
    messages = _latest_rows(context, "MESSAGING")
    settings = _latest_rows(context, "SETTING")
    at = lambda value, target: abs(float(value) - float(target)) < 1e-6
    contact = lambda **want: any(all(item.get(k) == v for k, v in want.items())
                                 for item in contacts)
    reminder = lambda content, timestamp: any(
        item.get("content") == content and at(item.get("reminder_timestamp"), timestamp)
        for item in reminders)
    message = lambda phone, content: any(
        item.get("recipient_phone_number") == phone and item.get("content") == content
        for item in messages)
    setting = next(item for item in settings if item.get("device_id") is not None)
    if family == "latest_reply_setting":
        checks = [reminder("Good, keep me posted", expected["reminder_time"]),
                  message(expected["boss_phone"], "Filed"), setting.get("cellular") is False]
    elif family == "contact_replace_message":
        checks = [not contact(person_id=expected["john_id"]),
                  contact(name="JP Archive", phone_number=expected["john_phone"],
                          relationship="collaborator", is_self=False),
                  message(expected["john_phone"], "Profile moved"),
                  reminder("Review JP record", expected["reminder_time"])]
    elif family == "reminder_replace_boss":
        checks = [not any(item.get("reminder_id") == expected["next_id"] for item in reminders),
                  reminder("Confirm venue", expected["next_time"]),
                  contact(person_id=expected["boss_id"], relationship="director"),
                  message(expected["boss_phone"], "Reminder updated")]
    elif family == "latest_branch_contact":
        checks = [reminder("Good, keep me posted", expected["reminder_time"]),
                  contact(person_id=expected["boss_id"], relationship="manager"),
                  contact(name="Ops Desk", phone_number="+14155550182",
                          relationship="coordinator", is_self=False),
                  message(expected["boss_phone"], "Digest: Good, keep me posted")]
    elif family == "neutral_quoted_words":
        checks = [contact(name="Latest Replace", phone_number=expected["phone"],
                          relationship="reviewer", is_self=False),
                  message(expected["phone"], "replace latest draft"),
                  setting.get("wifi") is False]
    elif family == "neutral_multisource":
        checks = [contact(person_id=expected["fredrik_id"], relationship="teammate"),
                  contact(name="QA Desk", phone_number=expected["phone"],
                          relationship="auditor", is_self=False),
                  message(expected["phone"], "Status clean"),
                  setting.get("location_service") is False]
    else:
        raise ValueError(f"unknown family: {family}")
    return [float(value) for value in checks]


def annotate_direct(row, metadata):
    spec = metadata[row["task"]]
    checks = requested_state_checks(row, spec)
    required = request_required_mutations(spec["public_request"], set(row["allowed_tools"]))
    successful = {call["tool"] for call in row["calls"] if not call.get("exception")}
    row.update({"requested_state_score": float(np.mean(checks)),
        "requested_state_success": int(all(value >= 1 - 1e-8 for value in checks)),
        "process_state_score": len(required & successful) / len(required),
        "process_state_success": int(required <= successful),
        "family": spec["family"], "exact_state_predicates_registered": True})
    return row


def golden_self_check(u, scenarios, metadata):
    now = datetime.datetime(2026, 10, 18, 12).timestamp()
    day2 = datetime.datetime(2026, 10, 20, 10, 20).timestamp()
    tomorrow_1410 = datetime.datetime(2026, 10, 19, 14, 10).timestamp()
    tomorrow_0730 = datetime.datetime(2026, 10, 19, 7, 30).timestamp()
    actions = {
        "latest_reply_setting": [
            {"tool":"get_current_timestamp","arguments":{}},
            {"tool":"search_messages","arguments":{"creation_timestamp_upperbound":now}},
            {"tool":"add_reminder","arguments":{"content":"Good, keep me posted","reminder_timestamp":day2}},
            {"tool":"send_message_with_phone_number","arguments":{"phone_number":"+10000000000","content":"Filed"}},
            {"tool":"set_cellular_service_status","arguments":{"on":False}}],
        "contact_replace_message": [
            {"tool":"search_contacts","arguments":{"name":"John Petrucci"}},
            {"tool":"get_current_timestamp","arguments":{}},
            {"tool":"remove_contact","arguments":{"person_id":"a22e1984-6c6c-530c-8831-c3ea3b5138e7"}},
            {"tool":"add_contact","arguments":{"name":"JP Archive","phone_number":"+1234560987","relationship":"collaborator"}},
            {"tool":"send_message_with_phone_number","arguments":{"phone_number":"+1234560987","content":"Profile moved"}},
            {"tool":"add_reminder","arguments":{"content":"Review JP record","reminder_timestamp":tomorrow_1410}}],
        "reminder_replace_boss": [
            {"tool":"get_current_timestamp","arguments":{}},
            {"tool":"search_reminder","arguments":{"reminder_timestamp_lowerbound":now}},
            {"tool":"search_contacts","arguments":{"relationship":"boss"}},
            {"tool":"remove_reminder","arguments":{"reminder_id":"54f20be9-eac3-5e76-8e57-724e48d3d477"}},
            {"tool":"add_reminder","arguments":{"content":"Confirm venue","reminder_timestamp":1792329542.015291}},
            {"tool":"modify_contact","arguments":{"person_id":"e3570ab6-0819-5032-be1e-2b366390c8ef","relationship":"director"}},
            {"tool":"send_message_with_phone_number","arguments":{"phone_number":"+10000000000","content":"Reminder updated"}}],
        "latest_branch_contact": [
            {"tool":"get_current_timestamp","arguments":{}},
            {"tool":"search_messages","arguments":{"creation_timestamp_upperbound":now}},
            {"tool":"search_contacts","arguments":{"relationship":"boss"}},
            {"tool":"add_reminder","arguments":{"content":"Good, keep me posted","reminder_timestamp":tomorrow_0730}},
            {"tool":"modify_contact","arguments":{"person_id":"e3570ab6-0819-5032-be1e-2b366390c8ef","relationship":"manager"}},
            {"tool":"add_contact","arguments":{"name":"Ops Desk","phone_number":"+14155550182","relationship":"coordinator"}},
            {"tool":"send_message_with_phone_number","arguments":{"phone_number":"+10000000000","content":"Digest: Good, keep me posted"}}],
        "neutral_quoted_words": [
            {"tool":"add_contact","arguments":{"name":"Latest Replace","phone_number":"+14155550183","relationship":"reviewer"}},
            {"tool":"send_message_with_phone_number","arguments":{"phone_number":"+14155550183","content":"replace latest draft"}},
            {"tool":"set_wifi_status","arguments":{"on":False}}],
        "neutral_multisource": [
            {"tool":"search_contacts","arguments":{"name":"Fredrik Thordendal"}},
            {"tool":"modify_contact","arguments":{"person_id":"9e137f06-916a-5310-8174-cf0b7e9f7054","relationship":"teammate"}},
            {"tool":"add_contact","arguments":{"name":"QA Desk","phone_number":"+14155550184","relationship":"auditor"}},
            {"tool":"send_message_with_phone_number","arguments":{"phone_number":"+14155550184","content":"Status clean"}},
            {"tool":"set_location_service_status","arguments":{"on":False}}],
    }
    next_times = [item["expected"]["next_time"] for item in metadata.values()
                  if item["family"] == "reminder_replace_boss"]
    if next_times:
        actions["reminder_replace_boss"][4]["arguments"]["reminder_timestamp"] = next_times[0]
    checks = []
    with tempfile.TemporaryDirectory(prefix="fast-golden-") as temporary:
        for name, scenario in scenarios.items():
            job = adapter.initial_job(name, scenario, [], u, None,
                                      "golden_preflight", "gold")
            for index, action in enumerate(actions[metadata[name]["family"]]):
                adapter.consume(job, {"agent_output": json.dumps(action),
                    "prompt_sha256": f"gold-{index}",
                    "hit_generation_limit": False}, u, 12)
            adapter.consume(job, {"agent_output": json.dumps({"answer":"Completed."}),
                "prompt_sha256":"gold-final", "hit_generation_limit":False}, u, 12)
            row = finish_direct(job, Path(temporary), u)
            row = annotate_direct(row, metadata)
            checks.append({"task": name,
                "requested_state_success": row["requested_state_success"],
                "process_state_success": row["process_state_success"]})
    passed = all(item["requested_state_success"] ==
                 item["process_state_success"] == 1 for item in checks)
    if not passed:
        raise RuntimeError("golden evaluator preflight failed: " + json.dumps(checks))
    return {"passed": True, "tasks": len(checks), "checks": checks}


def select(method, pool, task, query, scenario, pair_vetoes, tokenizer):
    if method == "Matched-Recipe":
        return matched_recipe_select(pool, query, scenario, tokenizer)
    if method in {"Contract-Text", "Contract-Validate", "Contract-FastRepair"}:
        return fast_contract_select(pool, query, scenario, tokenizer)
    if method == "Contract-Adaptive":
        return fast_select(pool, query, scenario, tokenizer, adaptive=True)
    if method == "CREDIT-Compose":
        return compose_select(pool, query, scenario)
    return prior_select(method, pool, task, query, scenario, pair_vetoes)


def _consume(job, record, u, max_calls, query):
    from time import perf_counter_ns
    began = perf_counter_ns()
    if job["method"] == "Contract-Validate":
        consume_validate(job, record, u, max_calls, query)
    elif job["method"] in {"Contract-FastRepair", "Contract-Adaptive"}:
        consume_fast(job, record, u, max_calls, query)
    else:
        adapter.consume(job, record, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6


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
                rows.append(finish_direct(job, directory, u))
            else:
                remaining.append(job)
        active = remaining
    for job in active:
        job["termination"] = "turn_budget"
        rows.append(finish_direct(job, directory, u))
    return rows


def _latency(row):
    remote = session_latency(row)
    local = float(row.get("selection_latency_ms", 0.0)) + sum(
        float(turn.get("local_consume_latency_ms", 0.0))
        for turn in row["generations"])
    return remote, local, remote + local


def run_actor(seed, actor_index, trajectories, root, args, scenarios, metadata,
              pool, pair_vetoes, u):
    from time import perf_counter_ns
    directory = root / f"fast_confirmation_seed_{seed}"
    directory.mkdir()
    agent = RemoteJSONAgent(directory, seed, env=args.env,
        tokenizer_path=args.tokenizer, temperature=0.0, max_new_tokens=384,
        workers=args.workers, exact_prompt_cache=False)
    agent.manifest["benchmark"] = "ToolSandbox-M fast-contract held-out confirmation"
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
                                     "fast_contract_confirmation", method)
                job["selection_latency_ms"] = selection_ms
                jobs.append(job); info[method] = (bank, position, selection_ms)
            for row in run_batch(jobs, {task: query}, agent, directory, u):
                bank, position, selection_ms = info[row["method"]]
                row["selection_latency_ms"] = selection_ms
                remote, local, accounted = _latency(row)
                item = annotate_direct({**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"],
                    "session_index": session_index,
                    "selected_bank": [note["uid"] for note in bank],
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


def summarize(rows):
    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        latency = [row["accounted_latency_ms"] for row in subset]
        turns = [turn for row in subset for turn in row["generations"]]
        item = {"method": method, "sessions": len(subset),
            "requested_state_VSR": float(np.mean([
                row["requested_state_success"] for row in subset])),
            "process_state_VSR": float(np.mean([
                row["process_state_success"] for row in subset])),
            "mean_requested_state_score": float(np.mean([
                row["requested_state_score"] for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "mean_remote_requests": float(np.mean([
                sum("remote_latency_ms" in turn for turn in row["generations"])
                for row in subset])),
            "mean_remote_latency_ms": float(np.mean([
                row["remote_latency_ms"] for row in subset])),
            "mean_local_runtime_ms": float(np.mean([
                row["local_runtime_ms"] for row in subset])),
            "mean_accounted_latency_ms": float(np.mean(latency)),
            "p95_accounted_latency_ms": float(np.quantile(latency, .95)),
            "validator_rejections": sum("contract_violation" in turn for turn in turns),
            "deterministic_repairs": sum("deterministic_repair" in turn for turn in turns),
            "local_continuations": sum(bool(turn.get("local_continuation")) for turn in turns),
            "local_completions": sum(row["termination"] == "local_contract_complete"
                                     for row in subset),
            "repaired_tool_exceptions": sum(bool(turn.get("repair_execution_exception"))
                                            for turn in turns
                                            if "deterministic_repair" in turn)}
        for budget in LATENCY_BUDGETS_MS:
            item[f"requested_state_VSR_at_{budget}ms"] = float(np.mean([
                row["requested_state_success"] and value <= budget
                for row, value in zip(subset, latency)]))
            item[f"completion_at_{budget}ms"] = float(np.mean([
                value <= budget for value in latency]))
        table.append(item)
    return table


def paired_statistics(rows):
    by = defaultdict(dict)
    for row in rows:
        by[row["method"]][(row["actor_seed"], row["trajectory"],
                           row["session_index"])] = row
    clusters = sorted({(row["actor_seed"], row["trajectory"]) for row in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    report = []
    for baseline in METHODS:
        if baseline == "Contract-Adaptive":
            continue
        state_effects, budget_effects = [], []
        for seed, trajectory in clusters:
            keys = [(seed, trajectory, index) for index in (0, 1)]
            state_effects.append(float(np.mean([
                by["Contract-Adaptive"][key]["requested_state_success"] -
                by[baseline][key]["requested_state_success"] for key in keys])))
            budget_effects.append(float(np.mean([
                (by["Contract-Adaptive"][key]["requested_state_success"] and
                 by["Contract-Adaptive"][key]["accounted_latency_ms"] <= PRIMARY_BUDGET_MS) -
                (by[baseline][key]["requested_state_success"] and
                 by[baseline][key]["accounted_latency_ms"] <= PRIMARY_BUDGET_MS)
                for key in keys])))
        state_boot = np.mean(np.asarray(state_effects)[draws], axis=1)
        budget_boot = np.mean(np.asarray(budget_effects)[draws], axis=1)
        wins = sum(value > 0 for value in budget_effects)
        losses = sum(value < 0 for value in budget_effects)
        p_value = (float(binomtest(wins, wins + losses, .5,
                         alternative="greater").pvalue) if wins + losses else 1.0)
        report.append({"baseline": baseline,
            "delta_requested_state_VSR": float(np.mean(state_effects)),
            "state_cluster_bootstrap_CI95": np.quantile(
                state_boot, [.025, .975]).tolist(),
            f"delta_state_VSR_at_{PRIMARY_BUDGET_MS}ms": float(np.mean(budget_effects)),
            "budget_cluster_bootstrap_CI95": np.quantile(
                budget_boot, [.025, .975]).tolist(),
            "budget_cluster_wins": wins, "budget_cluster_losses": losses,
            "budget_one_sided_exact_sign_p": p_value})
    return {"comparisons": report, "clusters": len(clusters),
        "resampling_unit": "actor seed by two-prompt task family",
        "bootstrap_replicates": 10000, "bootstrap_seed": BOOTSTRAP_SEED}


def holm_primary(report):
    rows = [dict(row) for row in report["comparisons"]
            if row["baseline"] in {"Contract-Validate", "Semantic"}]
    ordered = sorted(rows, key=lambda row: row["budget_one_sided_exact_sign_p"])
    running = 0.0
    for index, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) *
                      row["budget_one_sided_exact_sign_p"]))
        row["holm_adjusted_p"] = running
    return sorted(ordered, key=lambda row: row["baseline"])


def task_breakdown(rows):
    output = []
    for family in sorted({row["trajectory"] for row in rows}):
        for method in METHODS:
            subset = [row for row in rows
                      if row["trajectory"] == family and row["method"] == method]
            output.append({"family": family, "method": method,
                "sessions": len(subset),
                "requested_state_VSR": float(np.mean([
                    row["requested_state_success"] for row in subset])),
                f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms": float(np.mean([
                    row["requested_state_success"] and
                    row["accounted_latency_ms"] <= PRIMARY_BUDGET_MS
                    for row in subset])),
                "mean_remote_requests": float(np.mean([
                    sum("remote_latency_ms" in turn for turn in row["generations"])
                    for row in subset])),
                "mean_accounted_latency_ms": float(np.mean([
                    row["accounted_latency_ms"] for row in subset]))})
    return output


def freeze(args, source_protocol, scenarios, metadata, trajectories, pool,
           pair_vetoes, preflight):
    names = ["toolsandbox_m_contract_fast_confirmation.py",
             "toolsandbox_m_contract_fast_confirmation_verify.py",
             "toolsandbox_m_contract_fast.py",
             "toolsandbox_m_contract_fast_development.py",
             "toolsandbox_m_contract_selector.py",
             "toolsandbox_m_compose_selector.py",
             "toolsandbox_m_compose_confirmation.py",
             "toolsandbox_m_intent_optimization.py", "toolsandbox_m_followup.py",
             "toolsandbox_local_v3.py", "toolsandbox_m.py", "remote_json_agent.py"]
    hashes = {name: digest(Path(__file__).with_name(name)) for name in names}
    development_hashes = {}
    for path in args.development_sources:
        filename = ("sessions.jsonl" if (path / "sessions.jsonl").exists()
                    else "confirmation_sessions.jsonl")
        if (path / filename).exists():
            development_hashes[str(path.resolve())] = {
                "filename": filename, "sha256": digest(path / filename)}
    records = [{"name": name, **metadata[name],
        "public_request_sha256": _sha(metadata[name]["public_request"])}
        for name in sorted(metadata)]
    protocol = {"suite": "ToolSandbox-M fast-contract held-out confirmation",
        "frozen_before_remote_calls": True,
        "development_sources": [str(path.resolve()) for path in args.development_sources],
        "development_source_sha256": development_hashes,
        "memory_pool_source": str(args.source.resolve()),
        "memory_pool_source_sha256": digest(args.source / "memory_pool.json"),
        "source_sha256": hashes,
        "upstream_directory": source_protocol["upstream_directory"],
        "upstream_source_sha256": source_protocol["upstream_source_sha256"],
        "clock": CLOCK, "tasks": records, "trajectories": trajectories,
        "actor_seeds": list(args.seeds), "methods": list(METHODS),
        "primary_endpoint": "exact requested final physical-state VSR",
        "primary_latency_endpoint": f"exact state VSR at {PRIMARY_BUDGET_MS} ms",
        "latency_accounting": "all remote attempts plus local selection, validation, repair, and tool execution",
        "primary_comparisons": ["Contract-Adaptive vs Contract-Validate",
                                "Contract-Adaptive vs Semantic"],
        "multiplicity": "Holm adjustment over two registered primary comparisons",
        "inference": "paired actor-seed by task-family cluster bootstrap and exact sign test",
        "wide_identical_tool_catalogue": list(WIDE_TOOLS),
        "request_scoped_required_mutations": True,
        "repair_information_boundary": ["public_request", "allowed_tool_schema",
            "successful_tool_results", "proposed_action"],
        "maximum_tool_calls_per_session": 12,
        "maximum_model_turns_per_session": MAX_MODEL_TURNS,
        "paired_scheduling": "all seven methods share one concurrent task batch",
        "temperature": 0.0, "exact_prompt_cache": False,
        "provider_credentials_persisted": False,
        "pair_vetoes": sorted([list(pair) for pair in pair_vetoes]),
        "preflight": preflight, "smoke": args.smoke}
    adapter.write(args.output, "frozen_protocol.json", protocol)
    adapter.write(args.output, "scenario_metadata.json", metadata)
    adapter.write(args.output, "memory_pool.json", pool)
    (args.output / "source").mkdir()
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
        default=Path("results/toolsandbox_m_contract_confirmation_v1"))
    parser.add_argument("--development-sources", type=Path, nargs="+", default=[
        Path("results/toolsandbox_m_contract_confirmation_v1"),
        Path("results/toolsandbox_m_contract_fast_development_v1"),
        Path("results/toolsandbox_m_contract_fast_v2_targeted_v1")])
    parser.add_argument("--output", type=Path,
        default=Path("results/toolsandbox_m_contract_fast_confirmation_v1"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ACTOR_SEEDS))
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)

    source_protocol = json.loads((args.source / "frozen_protocol.json").read_text())
    u = adapter.upstream_imports(Path(source_protocol["upstream_directory"]))
    adapter.install_local_clock(CLOCK)
    scenarios, metadata, trajectories = build_confirmation_scenarios(u)
    if args.smoke:
        args.seeds = args.seeds[:1]
        trajectories = trajectories[:1]
        keep = {task for trajectory in trajectories for task in trajectory["sessions"]}
        scenarios = {name:value for name,value in scenarios.items() if name in keep}
        metadata = {name:value for name,value in metadata.items() if name in keep}
    manifest = next(args.source.glob("contract_confirmation_seed_*/agent_manifest.json"))
    args.tokenizer = args.tokenizer or Path(json.loads(manifest.read_text())["tokenizer_path"])
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    pool = json.loads((args.source / "memory_pool.json").read_text())
    pair_vetoes = {tuple(pair) for pair in source_protocol["pair_vetoes"]}
    preflight = {"golden_evaluator": golden_self_check(u, scenarios, metadata),
        "novelty": novelty_audit(u, scenarios, metadata, args.development_sources)}
    adapter.write(args.output, "preflight.json", preflight)
    protocol = freeze(args, source_protocol, scenarios, metadata, trajectories,
                      pool, pair_vetoes, preflight)
    selection = {}
    for task, scenario in scenarios.items():
        query = metadata[task]["public_request"]
        bank = fast_contract_select(pool, query, scenario, tokenizer)
        selection[task] = {"selected_uids": [note["uid"] for note in bank],
            "memory_schemas": [note["memory_schema"] for note in bank],
            "contract_text": bank[0]["text"],
            "contract_tools": bank[0]["tools"],
            "exposed_tools": list(WIDE_TOOLS)}
    adapter.write(args.output, "frozen_selection.json", selection)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "tasks": len(scenarios),
                          "preflight": preflight}, indent=2))
        return

    rows, actors = [], []
    for index, seed in enumerate(args.seeds):
        found, actor = run_actor(seed, index, trajectories, args.output, args,
                                 scenarios, metadata, pool, pair_vetoes, u)
        rows.extend(found); actors.append(actor)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = summarize(rows)
    statistics = paired_statistics(rows)
    ranking = sorted((dict(row) for row in table), key=lambda row: (
        -row[f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms"],
        -row["requested_state_VSR"], row["mean_accounted_latency_ms"], row["method"]))
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    summary = {"confirmation_table": table, "ranking": ranking,
        "paired_statistics": statistics, "holm_primary": holm_primary(statistics),
        "task_breakdown": task_breakdown(rows), "actor_summaries": actors,
        "frozen_selection": selection}
    adapter.write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    breakdown = summary["task_breakdown"]
    with (args.output / "task_family_breakdown.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
        writer.writeheader(); writer.writerows(breakdown)
    adapter.write(args.output, "audit.json", {"sessions": len(rows),
        "expected_sessions": len(args.seeds) * len(metadata) * len(METHODS),
        "live_remote_requests": sum(actor["remote_requests"] for actor in actors),
        "credentials_persisted": False,
        "all_tasks_share_identical_tools": len({tuple(row["operation_graph"])
            for row in metadata.values()}) == 1})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
