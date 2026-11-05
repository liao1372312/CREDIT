"""Frozen held-out confirmation of request-derived execution contracts.

The disclosed Compose confirmation and contract development runs are treated
as development data.  This suite uses new prompts, operation graphs, actor
seeds, and a frozen protocol.  Exact requested world state is primary.
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
from . import toolsandbox_local_v3 as adapter
from .toolsandbox_local_v3 import (
    consume, finish, initial_job, install_local_clock, run_batch,
    upstream_imports, write,
)
from .toolsandbox_m import continuous_score
from .toolsandbox_m_compose_confirmation import _imports, _score
from .toolsandbox_m_compose_selector import compose_select
from .toolsandbox_m_contract_selector import contract_select, contract_violation
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_intent_optimization import balanced_order, select as prior_select
from .user_data import digest


CLOCK = "2026-10-18T12:00:00"
ACTOR_SEEDS = (1301, 1303, 1307, 1319, 1321)
METHODS = ("NoMem", "FIFO", "Semantic", "CREDIT-Compact", "CREDIT-Intent",
           "CREDIT-Compose", "CREDIT-Contract")
LATENCY_BUDGETS_MS = (10000, 15000, 20000, 30000)
PRIMARY_BUDGET_MS = 15000
BOOTSTRAP_SEED = 20261018


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def build_confirmation_scenarios(u):
    """Build four mechanism stress families and two neutral controls."""
    x = _imports()
    pl, M, C = x["pl"], x["Milestone"], x["SnapshotConstraint"]
    DB, R, Cat = x["DatabaseNamespace"], x["RoleType"], x["ScenarioCategories"]
    add, remove, snap, update = (x["addition_similarity"], x["removal_similarity"],
                                 x["snapshot_similarity"], x["update_similarity"])
    Extension = x["ScenarioExtension"]
    bases = x["named_base_scenarios"](u.ToolBackend.DEFAULT)
    user_instruction = x["USER_INSTRUCTION"]

    tomorrow_0915 = datetime.datetime(2026, 10, 19, 9, 15).timestamp()
    tomorrow_0845 = datetime.datetime(2026, 10, 19, 8, 45).timestamp()
    tomorrow_1100 = datetime.datetime(2026, 10, 19, 11, 0).timestamp()
    boss_id, boss_phone = "e3570ab6-0819-5032-be1e-2b366390c8ef", "+10000000000"
    fredrik_id = "9e137f06-916a-5310-8174-cf0b7e9f7054"
    john_id, john_phone = "a22e1984-6c6c-530c-8831-c3ea3b5138e7", "+1234560987"
    gpu_phone = "+18307976530"
    next_id = "54f20be9-eac3-5e76-8e57-724e48d3d477"
    next_row = next(row for row in
        bases["base"].starting_context.get_database(DB.REMINDER).to_dicts()
        if row["reminder_id"] == next_id)
    next_time = float(next_row["reminder_timestamp"])

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

    scenarios, metadata = {}, {}

    def extension(name, family, public, private, base, tools, milestones, edges,
                  state_indices, essential_trace_indices):
        scenario = Extension(
            name=name, base_scenario=bases[base], messages=messages(private, public),
            tool_allow_list=tools, milestones=milestones, milestone_edge_list=edges,
            categories=[Cat.SINGLE_USER_TURN, Cat.CANONICALIZATION],
        ).get_extended_scenario()[name]
        metadata[name] = {
            "family": family, "public_request": public,
            "state_indices": state_indices,
            "essential_trace_indices": essential_trace_indices,
            "process_state_indices": sorted(set(state_indices + essential_trace_indices)),
            "operation_graph": tools,
        }
        scenarios[name] = scenario

    # 1. A current-time anchored latest-message lookup feeds two mutations.
    for suffix, public in (
        ("", "Make my newest message a reminder for tomorrow at 9:15 AM, then reply to its sender with 'Scheduled'."),
        ("_alt", "At 9:15 tomorrow, remind me of my latest text and send Scheduled back to whoever sent it."),
    ):
        name = "contract_latest_message_reminder_reply" + suffix
        ms = [trace("get_current_timestamp"), trace("search_messages"),
              M(snapshot_constraints=[C(database_namespace=DB.REMINDER,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "content": "Good, keep me posted",
                    "reminder_timestamp": tomorrow_0915,
                }), reference_milestone_node_index=1)]),
              M(snapshot_constraints=[C(database_namespace=DB.MESSAGING,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "recipient_phone_number": boss_phone, "content": "Scheduled",
                }), reference_milestone_node_index=2)])]
        extension(name, "latest_message_reminder_reply", public,
            "Find the newest message at or before now, add its unmodified content as a reminder tomorrow at 09:15, and reply Scheduled to its sender.",
            "base", ["get_current_timestamp", "search_messages",
            "shift_timestamp", "timestamp_to_datetime_info", "datetime_info_to_timestamp",
            "add_reminder", "send_message_with_phone_number"], ms,
            [(0, 1), (0, 2), (1, 2), (2, 3)], [2, 3], [0, 1])
        metadata[name]["expected"] = {"reminder_time": tomorrow_0915}

    # 2. The same raw message and a prefixed copy feed different outputs.
    for suffix, public in (
        ("", "Set my boss to manager, remind me of my newest message tomorrow at 8:45 AM, and text the boss that message prefixed 'Update: '."),
        ("_alt", "Tomorrow at 8:45, use my latest text unchanged for a reminder; change my boss relationship to manager and send them Update: followed by that text."),
    ):
        name = "contract_split_message_update" + suffix
        ms = [trace("get_current_timestamp"), trace("search_messages"),
              trace("search_contacts"),
              M(snapshot_constraints=[C(database_namespace=DB.REMINDER,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "content": "Good, keep me posted",
                    "reminder_timestamp": tomorrow_0845,
                }), reference_milestone_node_index=1)]),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=update, target_dataframe=pl.DataFrame({
                    "person_id": boss_id, "relationship": "manager",
                }), reference_milestone_node_index=2)]),
              M(snapshot_constraints=[C(database_namespace=DB.MESSAGING,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "recipient_phone_number": boss_phone,
                    "content": "Update: Good, keep me posted",
                }), reference_milestone_node_index=4)])]
        extension(name, "split_message_update", public,
            "Find the newest message at or before now. Add its raw content as a reminder tomorrow at 08:45, change the boss contact to manager, and send that phone Update: plus the raw message.",
            "base", ["get_current_timestamp", "search_messages",
            "shift_timestamp", "timestamp_to_datetime_info", "datetime_info_to_timestamp",
            "add_reminder", "search_contacts", "modify_contact",
            "send_message_with_phone_number"], ms,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            [3, 4, 5], [0, 1, 2])
        metadata[name]["expected"] = {"reminder_time": tomorrow_0845}

    # 3. Replace a contact and independently create a relative-date reminder.
    for suffix, public in (
        ("", "Replace John Petrucci with John P at the same number as a collaborator, and remind me tomorrow at 11 AM to Call John P."),
        ("_alt", "Recreate John Petrucci as John P, preserving his phone and setting collaborator; add a Call John P reminder for 11 tomorrow."),
    ):
        name = "contract_replace_contact_and_remind" + suffix
        ms = [trace("search_contacts"), trace("get_current_timestamp"),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=remove, target_dataframe=pl.DataFrame({
                    "person_id": john_id,
                }), reference_milestone_node_index=0)]),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "name": "John P", "phone_number": john_phone,
                    "relationship": "collaborator", "is_self": False,
                }), reference_milestone_node_index=2)]),
              M(snapshot_constraints=[C(database_namespace=DB.REMINDER,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "content": "Call John P", "reminder_timestamp": tomorrow_1100,
                }), reference_milestone_node_index=1)])]
        extension(name, "replace_contact_and_remind", public,
            "Search John Petrucci, remove that returned contact, add John P with the preserved phone as collaborator, then add Call John P tomorrow at 11:00.",
            "base", ["search_contacts", "remove_contact", "add_contact",
            "get_current_timestamp", "shift_timestamp", "timestamp_to_datetime_info",
            "datetime_info_to_timestamp", "add_reminder"], ms,
            [(0, 2), (2, 3), (1, 4), (3, 4)], [2, 3, 4], [0, 1])
        metadata[name]["expected"] = {"reminder_time": tomorrow_1100}

    # 4. Replace the next reminder while updating a separately resolved record.
    for suffix, public in (
        ("", "Replace my next reminder with 'Finalize slides' at the same time, and change my boss relationship to manager."),
        ("_alt", "Rebuild the upcoming reminder as Finalize slides without changing when it fires; mark my boss as manager too."),
    ):
        name = "contract_replace_reminder_update_contact" + suffix
        ms = [trace("get_current_timestamp"), trace("search_reminder"),
              trace("search_contacts"),
              M(snapshot_constraints=[C(database_namespace=DB.REMINDER,
                snapshot_constraint=remove, target_dataframe=pl.DataFrame({
                    "reminder_id": next_id,
                }), reference_milestone_node_index=1)]),
              M(snapshot_constraints=[C(database_namespace=DB.REMINDER,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "content": "Finalize slides", "reminder_timestamp": next_time,
                }), reference_milestone_node_index=3)]),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=update, target_dataframe=pl.DataFrame({
                    "person_id": boss_id, "relationship": "manager",
                }), reference_milestone_node_index=2)])]
        extension(name, "replace_reminder_update_contact", public,
            "Find the first reminder after now, preserve its time, remove it, add Finalize slides at that time, and change the boss contact to manager.",
            "base", ["get_current_timestamp", "search_reminder",
            "remove_reminder", "add_reminder", "search_contacts", "modify_contact"],
            ms, [(0, 1), (1, 3), (3, 4), (2, 5), (4, 5)], [3, 4, 5], [0, 1, 2])
        metadata[name]["expected"] = {"reminder_time": next_time}

    # 5. Neutral recovery control without replacement or relative time.
    for suffix, public in (
        ("", "Turn off low-battery mode, enable cellular, mark Fredrik Thordendal as teammate, add Morgan Yu as analyst at +14155550181, and send Morgan 'Access restored'."),
        ("_alt", "Restore mobile service by disabling low battery and enabling cellular; set Fredrik's relationship to teammate, save Morgan Yu (+14155550181) as analyst, and text Morgan Access restored."),
    ):
        name = "contract_neutral_recovery" + suffix
        ms = [M(snapshot_constraints=[C(database_namespace=DB.SETTING,
                snapshot_constraint=snap, target_dataframe=pl.DataFrame({
                    "low_battery_mode": False}))]),
              M(snapshot_constraints=[C(database_namespace=DB.SETTING,
                snapshot_constraint=snap, target_dataframe=pl.DataFrame({
                    "cellular": True}))]),
              trace("search_contacts"),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "name": "Morgan Yu", "phone_number": "+14155550181",
                    "relationship": "analyst", "is_self": False,
                }), reference_milestone_node_index=2)]),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=update, target_dataframe=pl.DataFrame({
                    "person_id": fredrik_id, "relationship": "teammate",
                }), reference_milestone_node_index=3)]),
              M(snapshot_constraints=[C(database_namespace=DB.MESSAGING,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "recipient_phone_number": "+14155550181",
                    "content": "Access restored",
                }), reference_milestone_node_index=4)])]
        extension(name, "neutral_recovery", public,
            "Disable low-battery mode, enable cellular, find Fredrik and set teammate, add Morgan Yu at +14155550181 as analyst, and send Morgan Access restored.",
            "base_low_battery_mode_on_wifi_off_location_service_off_cellular_off",
            ["get_low_battery_mode_status", "set_low_battery_mode_status",
             "get_cellular_service_status", "set_cellular_service_status",
             "search_contacts", "modify_contact", "add_contact",
             "send_message_with_phone_number"], ms,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            [0, 1, 3, 4, 5], [2])
        metadata[name]["expected"] = {}

    # 6. Neutral multi-source dataflow control.
    for suffix, public in (
        ("", "Save the GPU-offer sender as Hardware Lead (vendor), mark Fredrik Thordendal as bandmate, and text the Hardware Lead 'Let's talk'."),
        ("_alt", "Find who sent the GPU pitch and add that number as Hardware Lead, vendor; set Fredrik's relationship to bandmate, then send Let's talk to the new lead."),
    ):
        name = "contract_neutral_dual_lookup" + suffix
        ms = [trace("search_messages"), trace("search_contacts"),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "name": "Hardware Lead", "phone_number": gpu_phone,
                    "relationship": "vendor", "is_self": False,
                }), reference_milestone_node_index=0)]),
              M(snapshot_constraints=[C(database_namespace=DB.CONTACT,
                snapshot_constraint=update, target_dataframe=pl.DataFrame({
                    "person_id": fredrik_id, "relationship": "bandmate",
                }), reference_milestone_node_index=2)]),
              M(snapshot_constraints=[C(database_namespace=DB.MESSAGING,
                snapshot_constraint=add, target_dataframe=pl.DataFrame({
                    "recipient_phone_number": gpu_phone, "content": "Let's talk",
                }), reference_milestone_node_index=3)])]
        extension(name, "neutral_dual_lookup", public,
            "Find the GPU-offer sender and Fredrik. Add the sender as Hardware Lead/vendor, modify Fredrik to bandmate, and send Let's talk to the lead phone.",
            "base", ["search_messages", "search_contacts", "add_contact",
            "modify_contact", "send_message_with_phone_number"], ms,
            [(0, 2), (1, 2), (2, 3), (3, 4)], [2, 3, 4], [0, 1])
        metadata[name]["expected"] = {}

    order = ("latest_message_reminder_reply", "split_message_update",
             "replace_contact_and_remind", "replace_reminder_update_contact",
             "neutral_recovery", "neutral_dual_lookup")
    trajectories = [{"trajectory": family, "sessions": sorted(
        name for name, item in metadata.items() if item["family"] == family)}
        for family in order]
    return scenarios, metadata, trajectories


def semantic_answer_success(row: dict, family: str) -> int:
    answer = ""
    if row.get("generations"):
        answer = str(row["generations"][-1].get("action", {}).get("answer", ""))
    text = " ".join(answer.casefold().replace("-", " ").split())
    if any(term in text for term in ("could not", "unable", "failed", "provide")):
        return 0
    rules = {
        "latest_message_reminder_reply": (("reminder",), ("scheduled", "replied", "sent")),
        "split_message_update": (("reminder", "manager"), ("update", "sent")),
        "replace_contact_and_remind": (("john p", "reminder"), ("replaced", "collaborator")),
        "replace_reminder_update_contact": (("reminder", "manager"), ("replaced", "rebuilt")),
        "neutral_recovery": (("morgan", "fredrik"), ("restored", "sent", "cellular")),
        "neutral_dual_lookup": (("hardware lead", "fredrik"), ("sent", "talk", "bandmate")),
    }
    required, alternatives = rules[family]
    return int(bool(text) and all(term in text for term in required)
               and any(term in text for term in alternatives))


def annotate(row: dict, metadata: dict) -> dict:
    spec = metadata[row["task"]]
    evaluation = row.get("upstream_evaluation")
    clear = bool(evaluation and evaluation["minefield_similarity"] == 0)
    state = requested_state_checks(row, spec)
    process = [_score(evaluation, i) for i in spec["process_state_indices"]]
    row.update({
        "requested_state_score": float(np.mean(state)) if state else 0.0,
        "requested_state_success": int(clear and all(v >= 1 - 1e-8 for v in state)),
        "process_state_score": float(np.mean(process)) if process else 0.0,
        "process_state_success": int(clear and all(v >= 1 - 1e-8 for v in process)),
        "semantic_answer_success": semantic_answer_success(row, spec["family"]),
        "family": spec["family"],
    })
    return row


def _latest_rows(context: dict, namespace: str) -> list[dict]:
    rows = context["_dbs"][namespace]
    indices = [r["sandbox_message_index"] for r in rows
               if r.get("sandbox_message_index") is not None]
    index = max(indices)
    return [r for r in rows if r.get("sandbox_message_index") == index]


def requested_state_checks(row: dict, spec: dict) -> list[float]:
    """Check final requested state independently of milestone ordering."""
    context, family = row["ending_context"], spec["family"]
    contacts = _latest_rows(context, "CONTACT")
    reminders = _latest_rows(context, "REMINDER")
    messages = _latest_rows(context, "MESSAGING")
    settings = _latest_rows(context, "SETTING")
    at = lambda value, target: abs(float(value) - float(target)) < 1e-6
    has_contact = lambda **want: any(all(r.get(k) == v for k, v in want.items())
                                     for r in contacts)
    has_reminder = lambda content, timestamp: any(
        r.get("content") == content and at(r.get("reminder_timestamp"), timestamp)
        for r in reminders)
    has_message = lambda phone, content: any(
        r.get("recipient_phone_number") == phone and r.get("content") == content
        for r in messages)
    expected = spec["expected"]
    if family == "latest_message_reminder_reply":
        checks = [has_reminder("Good, keep me posted", expected["reminder_time"]),
                  has_message("+10000000000", "Scheduled")]
    elif family == "split_message_update":
        checks = [has_reminder("Good, keep me posted", expected["reminder_time"]),
                  has_contact(person_id="e3570ab6-0819-5032-be1e-2b366390c8ef",
                              relationship="manager"),
                  has_message("+10000000000", "Update: Good, keep me posted")]
    elif family == "replace_contact_and_remind":
        checks = [not has_contact(person_id="a22e1984-6c6c-530c-8831-c3ea3b5138e7"),
                  has_contact(name="John P", phone_number="+1234560987",
                              relationship="collaborator", is_self=False),
                  has_reminder("Call John P", expected["reminder_time"])]
    elif family == "replace_reminder_update_contact":
        checks = [not any(r.get("reminder_id") ==
                          "54f20be9-eac3-5e76-8e57-724e48d3d477" for r in reminders),
                  has_reminder("Finalize slides", expected["reminder_time"]),
                  has_contact(person_id="e3570ab6-0819-5032-be1e-2b366390c8ef",
                              relationship="manager")]
    elif family == "neutral_recovery":
        setting = next(r for r in settings if r.get("device_id") is not None)
        checks = [setting.get("low_battery_mode") is False,
                  setting.get("cellular") is True,
                  has_contact(person_id="9e137f06-916a-5310-8174-cf0b7e9f7054",
                              relationship="teammate"),
                  has_contact(name="Morgan Yu", phone_number="+14155550181",
                              relationship="analyst", is_self=False),
                  has_message("+14155550181", "Access restored")]
    elif family == "neutral_dual_lookup":
        checks = [has_contact(name="Hardware Lead", phone_number="+18307976530",
                              relationship="vendor", is_self=False),
                  has_contact(person_id="9e137f06-916a-5310-8174-cf0b7e9f7054",
                              relationship="bandmate"),
                  has_message("+18307976530", "Let's talk")]
    else:
        raise ValueError("unknown family: " + family)
    return [float(value) for value in checks]


def golden_self_check(u, scenarios, metadata) -> dict:
    tomorrow = {
        "latest_message_reminder_reply": datetime.datetime(2026, 10, 19, 9, 15).timestamp(),
        "split_message_update": datetime.datetime(2026, 10, 19, 8, 45).timestamp(),
        "replace_contact_and_remind": datetime.datetime(2026, 10, 19, 11).timestamp(),
    }
    next_scenario = scenarios.get("contract_replace_reminder_update_contact")
    next_time = 0.0 if next_scenario is None else float(next(
        row["reminder_timestamp"] for row in
        next_scenario.starting_context.get_database(u.DatabaseNamespace.REMINDER).to_dicts()
        if row["reminder_id"] == "54f20be9-eac3-5e76-8e57-724e48d3d477"))
    actions = {
        "latest_message_reminder_reply": [
            {"tool":"get_current_timestamp","arguments":{}},
            {"tool":"search_messages","arguments":{"creation_timestamp_upperbound":datetime.datetime(2026,10,18,12).timestamp()}},
            {"tool":"add_reminder","arguments":{"content":"Good, keep me posted","reminder_timestamp":tomorrow["latest_message_reminder_reply"]}},
            {"tool":"send_message_with_phone_number","arguments":{"phone_number":"+10000000000","content":"Scheduled"}},
            {"answer":"Reminder created and Scheduled was sent."}],
        "split_message_update": [
            {"tool":"get_current_timestamp","arguments":{}},
            {"tool":"search_messages","arguments":{"creation_timestamp_upperbound":datetime.datetime(2026,10,18,12).timestamp()}},
            {"tool":"search_contacts","arguments":{"relationship":"boss"}},
            {"tool":"add_reminder","arguments":{"content":"Good, keep me posted","reminder_timestamp":tomorrow["split_message_update"]}},
            {"tool":"modify_contact","arguments":{"person_id":"e3570ab6-0819-5032-be1e-2b366390c8ef","relationship":"manager"}},
            {"tool":"send_message_with_phone_number","arguments":{"phone_number":"+10000000000","content":"Update: Good, keep me posted"}},
            {"answer":"Reminder created, boss set to manager, and Update sent."}],
        "replace_contact_and_remind": [
            {"tool":"search_contacts","arguments":{"name":"John Petrucci"}},
            {"tool":"get_current_timestamp","arguments":{}},
            {"tool":"remove_contact","arguments":{"person_id":"a22e1984-6c6c-530c-8831-c3ea3b5138e7"}},
            {"tool":"add_contact","arguments":{"name":"John P","phone_number":"+1234560987","relationship":"collaborator"}},
            {"tool":"add_reminder","arguments":{"content":"Call John P","reminder_timestamp":tomorrow["replace_contact_and_remind"]}},
            {"answer":"John P replaced the contact and the reminder was created."}],
        "replace_reminder_update_contact": [
            {"tool":"get_current_timestamp","arguments":{}},
            {"tool":"search_reminder","arguments":{"reminder_timestamp_lowerbound":datetime.datetime(2026,10,18,12).timestamp()}},
            {"tool":"search_contacts","arguments":{"relationship":"boss"}},
            {"tool":"remove_reminder","arguments":{"reminder_id":"54f20be9-eac3-5e76-8e57-724e48d3d477"}},
            {"tool":"add_reminder","arguments":{"content":"Finalize slides","reminder_timestamp":next_time}},
            {"tool":"modify_contact","arguments":{"person_id":"e3570ab6-0819-5032-be1e-2b366390c8ef","relationship":"manager"}},
            {"answer":"Reminder replaced and boss set to manager."}],
        "neutral_recovery": [
            {"tool":"set_low_battery_mode_status","arguments":{"on":False}},
            {"tool":"set_cellular_service_status","arguments":{"on":True}},
            {"tool":"search_contacts","arguments":{"name":"Fredrik Thordendal"}},
            {"tool":"add_contact","arguments":{"name":"Morgan Yu","phone_number":"+14155550181","relationship":"analyst"}},
            {"tool":"modify_contact","arguments":{"person_id":"9e137f06-916a-5310-8174-cf0b7e9f7054","relationship":"teammate"}},
            {"tool":"send_message_with_phone_number","arguments":{"phone_number":"+14155550181","content":"Access restored"}},
            {"answer":"Morgan was saved, Fredrik updated, and Access restored was sent."}],
        "neutral_dual_lookup": [
            {"tool":"search_messages","arguments":{"content":"GPU"}},
            {"tool":"search_contacts","arguments":{"name":"Fredrik Thordendal"}},
            {"tool":"add_contact","arguments":{"name":"Hardware Lead","phone_number":"+18307976530","relationship":"vendor"}},
            {"tool":"modify_contact","arguments":{"person_id":"9e137f06-916a-5310-8174-cf0b7e9f7054","relationship":"bandmate"}},
            {"tool":"send_message_with_phone_number","arguments":{"phone_number":"+18307976530","content":"Let's talk"}},
            {"answer":"Hardware Lead saved, Fredrik updated, and Let's talk sent."}],
    }
    checks = []
    with tempfile.TemporaryDirectory() as temp:
        for name, scenario in scenarios.items():
            job = initial_job(name, scenario, [], u, None, "golden_preflight", "gold")
            for index, action in enumerate(actions[metadata[name]["family"]]):
                consume(job, {"agent_output": json.dumps(action),
                    "prompt_sha256": f"gold-{index}", "hit_generation_limit": False}, u, 12)
            row = annotate(finish(job, Path(temp), u), metadata)
            checks.append({"task": name, "upstream_success": row["success"],
                "requested_state_success": row["requested_state_success"],
                "process_state_success": row["process_state_success"],
                "semantic_answer_success": row["semantic_answer_success"]})
    passed = all(all(item[k] == 1 for k in ("upstream_success",
        "requested_state_success", "process_state_success", "semantic_answer_success"))
        for item in checks)
    if not passed:
        raise RuntimeError("golden evaluator preflight failed: " + json.dumps(checks))
    return {"passed": True, "tasks": len(checks), "checks": checks}


def novelty_audit(u, scenarios, metadata, sources: list[Path]) -> dict:
    def norm(tools):
        return tuple(sorted(set(tools or []) - {"end_conversation"}))
    previous = defaultdict(list)
    official = {**u.named_single_tool_call_scenarios(u.ToolBackend.DEFAULT),
                **u.named_multiple_tool_call_scenarios(u.ToolBackend.DEFAULT)}
    for name, scenario in official.items():
        previous[norm(scenario.starting_context.tool_allow_list)].append("official:" + name)
    old_prompts = set()
    for source in sources:
        meta_path = source / "scenario_metadata.json"
        if meta_path.exists():
            for name, item in json.loads(meta_path.read_text()).items():
                previous[norm(item["operation_graph"])].append(source.name + ":" + name)
                old_prompts.add(_sha(item["public_request"]))
    overlap = {name: previous[norm(s.starting_context.tool_allow_list)]
               for name, s in scenarios.items()
               if norm(s.starting_context.tool_allow_list) in previous}
    current = {_sha(item["public_request"]) for item in metadata.values()}
    prompt_overlap = sorted(current & old_prompts)
    passed = not overlap and not prompt_overlap and len(current) == len(metadata)
    if not passed:
        raise RuntimeError("confirmation novelty preflight failed: " +
                           json.dumps({"tools": overlap, "prompts": prompt_overlap}))
    return {"passed": True, "tasks": len(metadata),
        "unique_public_request_hashes": len(current),
        "exact_prompt_overlap": prompt_overlap,
        "exact_allowed_tool_set_overlap": overlap,
        "comparison_sets": ["official", *[p.name for p in sources]]}


def select(method, pool, task, query, scenario, pair_vetoes, tokenizer):
    if method == "CREDIT-Compose":
        return compose_select(pool, query, scenario)
    if method == "CREDIT-Contract":
        return contract_select(pool, query, scenario, tokenizer)
    return prior_select(method, pool, task, query, scenario, pair_vetoes)


def consume_contract(job, generation, u, max_calls, query):
    """Execute an action or return deterministic contract feedback."""
    action, parse_error = adapter.parse_request(generation["agent_output"])
    violation = None if parse_error else contract_violation(
        action, job["calls"], query, set(job["tools"]))
    if violation is None:
        consume(job, generation, u, max_calls)
        return
    turn = {"prompt_sha256": generation["prompt_sha256"], "action": action,
        "parse_error": parse_error, "hit_generation_limit": generation["hit_generation_limit"],
        "contract_violation": violation}
    for key in ("remote_latency_ms", "network_attempts", "provider_model",
                "system_fingerprint_sha256", "request_sha256", "response_sha256",
                "generation_uid", "successful_attempt_latency_ms",
                "rendered_input_tokens", "input_tokens", "output_tokens"):
        if key in generation:
            turn[key] = generation[key]
    job["generations"].append(turn)
    job["chat"].append({"role": "assistant", "content": generation["agent_output"]})
    job["chat"].append({"role": "user", "content": json.dumps({
        "error": "execution_contract_violation", "instruction": violation},
        ensure_ascii=False)})


def run_batch_with_contract(jobs, queries, agent, directory, u, *, max_calls=12):
    rows, active = [], jobs
    for _ in range(20):
        if not active:
            break
        records = agent.infer([job["chat"] for job in active])
        remaining = []
        for job, record in zip(active, records):
            if job["method"] == "CREDIT-Contract":
                consume_contract(job, record, u, max_calls, queries[job["task"]])
            else:
                consume(job, record, u, max_calls)
            if job["done"]:
                rows.append(finish(job, directory, u))
            else:
                remaining.append(job)
        active = remaining
    for job in active:
        job["termination"] = "turn_budget"
        rows.append(finish(job, directory, u))
    return rows


def run_actor(seed, actor_index, trajectories, root, args, scenarios, metadata,
              pool, pair_vetoes, u):
    directory = root / f"contract_confirmation_seed_{seed}"
    directory.mkdir()
    agent = RemoteJSONAgent(directory, seed, env=args.env,
        tokenizer_path=args.tokenizer, temperature=0.0, max_new_tokens=384,
        workers=args.workers, exact_prompt_cache=False)
    agent.binding_interface = "tool_json"
    agent.manifest["benchmark"] = "ToolSandbox-M CREDIT-Contract confirmation"
    sessions, positions = [], defaultdict(lambda: [0] * len(METHODS))
    for trajectory_index, trajectory in enumerate(trajectories):
        for session_index, task in enumerate(trajectory["sessions"]):
            scenario, query = scenarios[task], metadata[task]["public_request"]
            order = balanced_order(list(METHODS), actor_index, trajectory_index, session_index)
            jobs, info = [], {}
            for position, method in enumerate(order):
                positions[method][position] += 1
                bank = select(method, pool, task, query, scenario,
                              pair_vetoes, agent.tokenizer)
                jobs.append(initial_job_v2(task, scenario, bank, u, agent.tokenizer,
                                           "contract_confirmation", method))
                info[method] = (bank, position)
            queries = {task: query}
            for row in run_batch_with_contract(jobs, queries, agent, directory, u,
                                               max_calls=12):
                bank, position = info[row["method"]]
                item = annotate({**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"],
                    "session_index": session_index,
                    "selected_bank": [n["uid"] for n in bank],
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


def summarize(rows):
    table = []
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        latency = [r["remote_latency_ms"] for r in subset]
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
                r["requested_state_success"] and l <= budget for r, l in zip(subset, latency)]))
            item[f"completion_at_{budget}ms"] = float(np.mean([l <= budget for l in latency]))
        table.append(item)
    return table


def paired_statistics(rows):
    by = defaultdict(dict)
    for row in rows:
        by[row["method"]][(row["actor_seed"], row["trajectory"],
                           row["session_index"])] = row
    clusters = sorted({(r["actor_seed"], r["trajectory"]) for r in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    report = []
    for method in METHODS:
        if method == "CREDIT-Contract":
            continue
        state_effects, budget_effects = [], []
        for seed, trajectory in clusters:
            keys = [(seed, trajectory, i) for i in (0, 1)]
            state_effects.append(float(np.mean([
                by["CREDIT-Contract"][k]["requested_state_success"] -
                by[method][k]["requested_state_success"] for k in keys])))
            budget_effects.append(float(np.mean([
                (by["CREDIT-Contract"][k]["requested_state_success"] and
                 by["CREDIT-Contract"][k]["remote_latency_ms"] <= PRIMARY_BUDGET_MS) -
                (by[method][k]["requested_state_success"] and
                 by[method][k]["remote_latency_ms"] <= PRIMARY_BUDGET_MS)
                for k in keys])))
        state_boot = np.mean(np.asarray(state_effects)[draws], axis=1)
        budget_boot = np.mean(np.asarray(budget_effects)[draws], axis=1)
        wins, losses = sum(v > 0 for v in state_effects), sum(v < 0 for v in state_effects)
        p = float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue) \
            if wins + losses else 1.0
        report.append({"baseline": method,
            "delta_requested_state_VSR": float(np.mean(state_effects)),
            "state_cluster_bootstrap_CI95": np.quantile(state_boot, [.025,.975]).tolist(),
            f"delta_state_VSR_at_{PRIMARY_BUDGET_MS}ms": float(np.mean(budget_effects)),
            "budget_cluster_bootstrap_CI95": np.quantile(budget_boot, [.025,.975]).tolist(),
            "cluster_wins": wins, "cluster_losses": losses,
            "one_sided_exact_sign_p": p})
    return {"comparisons": report, "clusters": len(clusters),
        "resampling_unit": "actor seed by two-prompt task family",
        "bootstrap_replicates": 10000, "bootstrap_seed": BOOTSTRAP_SEED}


def holm_primary(report):
    chosen = [dict(r) for r in report["comparisons"] if r["baseline"] in
              ("FIFO", "Semantic", "CREDIT-Compose")]
    ordered = sorted(chosen, key=lambda r: r["one_sided_exact_sign_p"])
    running = 0.0
    for index, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered)-index)*row["one_sided_exact_sign_p"]))
        row["holm_adjusted_p"] = running
    return sorted(ordered, key=lambda r: r["baseline"])


def task_breakdown(rows):
    out = []
    for family in sorted({r["trajectory"] for r in rows}):
        for method in METHODS:
            subset = [r for r in rows if r["trajectory"] == family and r["method"] == method]
            out.append({"family": family, "method": method, "sessions": len(subset),
                "requested_state_VSR": float(np.mean([r["requested_state_success"] for r in subset])),
                "mean_tool_calls": float(np.mean([r["tool_calls"] for r in subset])),
                "mean_remote_latency_ms": float(np.mean([r["remote_latency_ms"] for r in subset]))})
    return out


def freeze(args, source_protocol, scenarios, metadata, trajectories, pool,
           pair_vetoes, preflight):
    names = ["toolsandbox_m_contract_confirmation.py",
             "toolsandbox_m_contract_confirmation_verify.py",
             "toolsandbox_m_contract_selector.py", "toolsandbox_m_compose_selector.py",
             "toolsandbox_m_compose_confirmation.py", "toolsandbox_m_intent_optimization.py",
             "toolsandbox_m_streaming_optimization.py", "toolsandbox_m_followup.py",
             "toolsandbox_local_v3.py", "toolsandbox_m.py", "remote_json_agent.py"]
    hashes = {n: digest(Path(__file__).with_name(n)) for n in names
              if Path(__file__).with_name(n).exists()}
    records = [{"name": n, **metadata[n],
        "public_request_sha256": _sha(metadata[n]["public_request"])} for n in sorted(metadata)]
    protocol = {
        "suite": "ToolSandbox-M CREDIT-Contract held-out confirmation",
        "frozen_before_remote_calls": True,
        "development_sources": [str(p.resolve()) for p in args.development_sources],
        "development_source_sha256": {
            str(p.resolve()): digest(p / "confirmation_sessions.jsonl")
            for p in args.development_sources if (p / "confirmation_sessions.jsonl").exists()},
        "source_sha256": hashes,
        "upstream_directory": source_protocol["upstream_directory"],
        "upstream_source_sha256": source_protocol["upstream_source_sha256"],
        "clock": CLOCK, "tasks": records, "trajectories": trajectories,
        "actor_seeds": list(args.seeds), "methods": list(METHODS),
        "primary_endpoint": "requested_state_VSR: all requested world-state constraints exact and no minefield",
        "requested_state_verifier": "registered family-specific exact predicates over the final physical-world snapshot, independent of action order",
        "primary_latency_endpoint": f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms",
        "secondary_endpoints": ["process_state_VSR", "semantic_answer_VSR",
                                "upstream_strict_VSR", "mean_tool_calls"],
        "primary_comparisons": ["CREDIT-Contract vs FIFO",
            "CREDIT-Contract vs Semantic", "CREDIT-Contract vs CREDIT-Compose"],
        "multiplicity": "Holm adjustment over three registered primary comparisons",
        "inference": "paired actor-seed by task-family cluster bootstrap and exact sign test",
        "selector": "public-request execution contract plus one schema-selected verified recipe",
        "memory_pool_size": len(pool), "preflight": preflight,
        "historical_confirmation_examined": True,
        "claim_scope": "mechanism-targeted held-out synthetic task graphs with two neutral controls",
        "maximum_tool_calls_per_session": 12,
        "maximum_model_turns_per_session": 20,
        "paired_scheduling": "all seven methods for a task share one concurrent remote batch",
        "temperature": 0.0, "exact_prompt_cache": False,
        "provider_credentials_persisted": False,
        "pair_vetoes": sorted([list(p) for p in pair_vetoes]), "smoke": args.smoke,
    }
    write(args.output, "frozen_protocol.json", protocol)
    write(args.output, "scenario_metadata.json", metadata)
    write(args.output, "memory_pool.json", pool)
    for name in hashes:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/toolsandbox_m_compose_confirmation_v1"))
    parser.add_argument("--development-sources", type=Path, nargs="+", default=[
        Path("results/toolsandbox_m_blind_composition_v1"),
        Path("results/toolsandbox_m_compose_confirmation_v1")])
    parser.add_argument("--output", type=Path,
                        default=Path("results/toolsandbox_m_contract_confirmation_v1"))
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
    (args.output / "source").mkdir()

    source_protocol = json.loads((args.source / "frozen_protocol.json").read_text())
    u = upstream_imports(Path(source_protocol["upstream_directory"]))
    install_local_clock(CLOCK)
    scenarios, metadata, trajectories = build_confirmation_scenarios(u)
    if args.smoke:
        args.seeds = args.seeds[:1]
        trajectories = trajectories[:1]
        keep = {n for t in trajectories for n in t["sessions"]}
        scenarios = {n:s for n,s in scenarios.items() if n in keep}
        metadata = {n:m for n,m in metadata.items() if n in keep}
    manifest = next(args.source.glob("compose_confirmation_seed_*/agent_manifest.json"))
    args.tokenizer = args.tokenizer or Path(json.loads(manifest.read_text())["tokenizer_path"])
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    pool = json.loads((args.source / "memory_pool.json").read_text())
    pair_vetoes = {tuple(p) for p in source_protocol["pair_vetoes"]}
    preflight = {"golden_evaluator": golden_self_check(u, scenarios, metadata),
        "novelty": novelty_audit(u, scenarios, metadata, args.development_sources)}
    write(args.output, "preflight.json", preflight)
    freeze(args, source_protocol, scenarios, metadata, trajectories, pool,
           pair_vetoes, preflight)
    selection = {}
    for task, scenario in scenarios.items():
        query = metadata[task]["public_request"]
        bank = contract_select(pool, query, scenario, tokenizer)
        selection[task] = {"selected_uids": [n["uid"] for n in bank],
            "memory_schemas": [n["memory_schema"] for n in bank],
            "contract_text": bank[0]["text"],
            "allowed_tools": sorted(set(scenario.starting_context.tool_allow_list or []) - {"end_conversation"})}
    write(args.output, "frozen_selection.json", selection)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "tasks": len(scenarios),
                          "selection": selection}, indent=2))
        return

    rows, actor_summaries = [], []
    for index, seed in enumerate(args.seeds):
        found, actor = run_actor(seed, index, trajectories, args.output, args,
            scenarios, metadata, pool, pair_vetoes, u)
        rows.extend(found); actor_summaries.append(actor)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table, statistics = summarize(rows), paired_statistics(rows)
    ranking = sorted((dict(r) for r in table), key=lambda r: (
        -r["requested_state_VSR"], -r[f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms"],
        -r["process_state_VSR"], r["mean_online_remote_latency_ms"], r["method"]))
    for rank, row in enumerate(ranking, 1): row["rank"] = rank
    summary = {"confirmation_table": table, "ranking": ranking,
        "paired_statistics": statistics, "holm_primary": holm_primary(statistics),
        "task_breakdown": task_breakdown(rows),
        "frozen_selection": selection, "actor_summaries": actor_summaries}
    write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    with (args.output / "task_family_breakdown.csv").open("w") as stream:
        breakdown = summary["task_breakdown"]
        writer = csv.DictWriter(stream, fieldnames=list(breakdown[0])); writer.writeheader(); writer.writerows(breakdown)
    write(args.output, "audit.json", {
        "sessions": len(rows), "expected_sessions": len(args.seeds)*len(metadata)*len(METHODS),
        "unique_initial_states_per_task": {task: len({r["initial_state_sha256"] for r in rows if r["task"] == task}) for task in metadata},
        "credentials_persisted": False,
        "network_attempt_counts": {str(n): sum(g.get("network_attempts") == n for r in rows for g in r["generations"]) for n in sorted({g.get("network_attempts") for r in rows for g in r["generations"]})},
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
