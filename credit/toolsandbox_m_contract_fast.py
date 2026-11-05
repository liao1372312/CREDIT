"""Latency-aware deterministic repairs for request-derived tool contracts.

Repairs use only the public request, allowed tool names, prior tool results,
and the model's proposed action.  The original proposal and repair reason are
retained in the episode trace; evaluator targets and hidden state are never
available to this module.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
from time import perf_counter_ns
from types import SimpleNamespace

from . import toolsandbox_local_v3 as adapter
from .toolsandbox_m_compose_selector import compose_select
from .toolsandbox_m_contract_selector import (
    REPLACEMENT_WORDS,
    _has_phrase,
    _last_success,
    _parsed_response,
    _relative_days,
    contract_note,
    contract_select,
    contract_violation,
)


LATEST_PHRASES = (
    "latest message", "latest text", "newest message", "newest text",
    "most recent message", "most recent text",
)


def _instruction_text(query: str) -> str:
    """Normalize the instruction while excluding quoted payload literals."""
    unquoted = re.sub(
        r'"(?:\\.|[^"\\])*"|(?<!\w)\'(?:\\.|[^\'\\])*\'(?!\w)',
        " ", query)
    return " ".join(unquoted.casefold().replace("-", " ").split())


def _replacement_requested(query: str) -> bool:
    return _has_phrase(_instruction_text(query), REPLACEMENT_WORDS)


def _replacement_targets(query: str, allowed_tools: set[str]) -> tuple[bool, bool]:
    """Identify whether replacement language governs a contact or reminder."""
    text = _instruction_text(query)
    if not _replacement_requested(query):
        return False, False
    chunks = re.split(r"\b(?:and|then)\b|[;,]", text)
    replacement_chunks = [chunk for chunk in chunks
                          if _has_phrase(chunk, REPLACEMENT_WORDS)]
    reminder = any(re.search(r"\breminders?\b", chunk)
                   for chunk in replacement_chunks)
    contact = any(
        re.search(r"\b(contact|phone|number|relationship)\b", chunk) or
        (not re.search(r"\breminders?\b", chunk) and
         re.search(r"\b(?:replace|recreate|re build|rebuild|swap out|substitute)\b"
                   r".{1,80}\b(?:with|as)\b", chunk))
        for chunk in replacement_chunks)
    allowed = set(allowed_tools)
    return (contact and bool(allowed & {"add_contact", "remove_contact"}),
            reminder and bool(allowed & {"add_reminder", "remove_reminder"}))


def request_required_mutations(query: str, allowed_tools: set[str]) -> set[str]:
    """Infer requested writes from public language, independent of the allowlist shape."""
    allowed = set(allowed_tools) - {"end_conversation"}
    text = _instruction_text(query)
    required: set[str] = set()
    contact_replacement, reminder_replacement = _replacement_targets(query, allowed)

    contact = contact_replacement or bool(re.search(
        r"\b(contact|person|phonebook|phone|number|relationship)\b", text)) or (
        "modify_contact" in allowed and _target_relationship(query) is not None)
    reminder = reminder_replacement or bool(re.search(
        r"\b(remind|reminder|reminders)\b", text))
    if "add_contact" in allowed and contact and (
            contact_replacement or re.search(r"\b(add|save|create|record)\b", text)):
        required.add("add_contact")
    if "remove_contact" in allowed and contact and (
            contact_replacement or re.search(r"\b(remove|delete|erase)\b", text)):
        required.add("remove_contact")
    if "modify_contact" in allowed and contact and re.search(
            r"\b(modify|change|update|mark|set)\b", text):
        required.add("modify_contact")

    if "add_reminder" in allowed and reminder and (
            reminder_replacement or re.search(
                r"\b(add|create|set|schedule|remind|make|use)\b", text)):
        required.add("add_reminder")
    if "remove_reminder" in allowed and reminder and (
            reminder_replacement or re.search(r"\b(remove|delete|cancel|erase)\b", text)):
        required.add("remove_reminder")

    if "send_message_with_phone_number" in allowed and (
            re.search(r"\b(send|reply|text)\b", text) or
            re.search(r"(?:^|[.!?]\s*)message\b", text)):
        required.add("send_message_with_phone_number")

    setting_words = {
        "set_low_battery_mode_status": ("low battery", "battery saver"),
        "set_cellular_service_status": ("cellular", "mobile data"),
        "set_wifi_status": ("wifi", "wi-fi", "wi fi"),
        "set_location_service_status": ("location service", "location services"),
    }
    setting_verb = re.search(
        r"\b(turn|switch|enable|disable|activate|deactivate|set)\b", text)
    if setting_verb:
        for tool, phrases in setting_words.items():
            if tool in allowed and any(phrase in text for phrase in phrases):
                required.add(tool)
    return required


def request_scoped_tools(query: str, allowed_tools: set[str]) -> set[str]:
    """Return the public-request-relevant schema subset for validation."""
    allowed = set(allowed_tools) - {"end_conversation"}
    text = _instruction_text(query)
    scoped = request_required_mutations(query, allowed)
    if scoped & {"add_contact", "remove_contact", "modify_contact"}:
        scoped.add("search_contacts")
    if scoped & {"add_reminder", "remove_reminder"}:
        scoped.add("search_reminder")
    if "send_message_with_phone_number" in scoped:
        if any(phrase in text for phrase in LATEST_PHRASES) or re.search(
                r"\b(sender|whoever sent)\b", text):
            scoped.add("search_messages")
        else:
            scoped.add("search_contacts")
    if _relative_days(query) is not None:
        scoped.update({"get_current_timestamp", "shift_timestamp",
                       "timestamp_to_datetime_info", "datetime_info_to_timestamp"})
    if any(phrase in text for phrase in LATEST_PHRASES):
        scoped.update({"get_current_timestamp", "search_messages"})
    if any(phrase in text for phrase in ("next reminder", "upcoming reminder")):
        scoped.update({"get_current_timestamp", "search_reminder"})
    return scoped & allowed


def fast_contract_violation(action: dict, calls: list[dict], query: str,
                            allowed_tools: set[str]) -> str | None:
    """Validate against operations requested in text, not every exposed write tool."""
    scoped = request_scoped_tools(query, allowed_tools)
    text = _instruction_text(query)
    tool = action.get("tool")
    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    global_latest = any(phrase in text for phrase in LATEST_PHRASES) and not re.search(
        r"\b(?:latest|newest|most recent)\s+(?:message|text)\s+(?:from|to)\b", text)
    if global_latest and tool == "search_messages" and (
            set(arguments) - {"creation_timestamp_upperbound"}):
        return "global latest-message search must not add sender, recipient, content, or lower-bound filters"

    keyword = _source_message_keyword(query)
    source = _source_message(calls, query) if keyword else None
    if keyword and tool in {"add_contact", "send_message_with_phone_number"}:
        if source is None:
            return "search_messages for the requested sender source before using its phone number"
        if arguments.get("phone_number") != source.get("sender_phone_number"):
            return "use the sender_phone_number from the requested source message"
    # Preserve the original ordering, temporal, and data-flow checks without
    # invoking its allowlist-wide final-answer rule.
    probe = action if "answer" not in action else {"tool": "__final_check__", "arguments": {}}
    violation = contract_violation(probe, calls, query, scoped)
    if violation is not None:
        return violation
    if "answer" in action:
        missing = sorted(request_required_mutations(query, allowed_tools) -
                         _successful_tools(calls))
        if missing:
            return "do not finish before these requested mutations succeed: " + ", ".join(missing)
    return None


def _successful_tools(calls: list[dict]) -> set[str]:
    return {call["tool"] for call in calls if not call.get("exception")}


def _result_rows(calls: list[dict], tool: str) -> list[dict]:
    """Return all nonempty public rows observed from a search tool."""
    rows: list[dict] = []
    for call in calls:
        if call.get("tool") != tool or call.get("exception"):
            continue
        value = _parsed_response(call)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    return rows


def _latest_message(calls: list[dict]) -> dict | None:
    rows = _result_rows(calls, "search_messages")
    candidates = [row for row in rows if row.get("creation_timestamp") is not None]
    return max(candidates, key=lambda row: row["creation_timestamp"]) if candidates else None


def _source_message_keyword(query: str) -> str | None:
    """Extract a public keyword for requests that identify a message sender."""
    text = _instruction_text(query)
    if not re.search(r"\b(sender|who sent|whoever sent)\b", text):
        return None
    match = re.search(r"\b([a-z0-9]+)\s+(?:offer|pitch|message|text)\b", text)
    return match.group(1) if match else None


def _source_message(calls: list[dict], query: str) -> dict | None:
    keyword = _source_message_keyword(query)
    rows = _result_rows(calls, "search_messages")
    if keyword:
        matches = [row for row in rows
                   if keyword in str(row.get("content", "")).casefold()]
        rows = matches or rows
    candidates = [row for row in rows if row.get("sender_phone_number")]
    if len(candidates) == 1:
        return candidates[0]
    stamped = [row for row in candidates if row.get("creation_timestamp") is not None]
    return max(stamped, key=lambda row: row["creation_timestamp"]) if stamped else None


def _target_relationship(query: str) -> str | None:
    normalized = " ".join(query.casefold().split())
    patterns = (
        r"relationship\s+(?:to|as)\s+([a-z][a-z_-]*)",
        r"(?:mark|set)\b[^.;,]*?\b(?:as|to)\s+([a-z][a-z_-]*)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)
    return None


def _contact_for_public_role(calls: list[dict], query: str) -> dict | None:
    normalized = " ".join(query.casefold().split())
    unique = {}
    for row in _result_rows(calls, "search_contacts"):
        person_id = row.get("person_id")
        if person_id is not None:
            unique[person_id] = row
    rows = list(unique.values())
    role_matches = [row for row in rows if row.get("relationship") and
                    re.search(r"\b" + re.escape(str(row["relationship"]).casefold()) +
                              r"\b", normalized)]
    if len(role_matches) == 1:
        return role_matches[0]
    name_matches = [row for row in rows if row.get("name") and
                    str(row["name"]).casefold() in normalized]
    if len(name_matches) == 1:
        return name_matches[0]
    return rows[0] if len(rows) == 1 else None


def _replacement_row(calls: list[dict], tool: str, id_key: str,
                     query: str) -> dict | None:
    rows = _result_rows(calls, tool)
    unique = {row[id_key]: row for row in rows if row.get(id_key) is not None}
    candidates = list(unique.values())
    normalized = query.casefold()
    named = [row for row in candidates if row.get("name") and
             str(row["name"]).casefold() in normalized]
    if len(named) == 1:
        return named[0]
    return candidates[0] if len(candidates) == 1 else None


def _risk_activated(query: str, allowed_tools: set[str]) -> bool:
    normalized = _instruction_text(query)
    replacement = _replacement_requested(query)
    relative = _relative_days(query) is not None
    latest = "search_messages" in allowed_tools and any(
        phrase in normalized for phrase in LATEST_PHRASES)
    upcoming = "search_reminder" in allowed_tools and any(
        phrase in normalized for phrase in ("next reminder", "upcoming reminder"))
    return replacement or relative or latest or upcoming


def fast_select(pool: list[dict], query: str, scenario, tokenizer,
                *, adaptive: bool = True) -> list[dict]:
    """Select the contract only for requests with a recognized risk pattern."""
    allowed = set(scenario.starting_context.tool_allow_list or []) - {"end_conversation"}
    if not adaptive or _risk_activated(query, allowed):
        return fast_contract_select(pool, query, scenario, tokenizer)
    return compose_select(pool, query, scenario)[:1]


def fast_contract_select(pool: list[dict], query: str, scenario, tokenizer) -> list[dict]:
    """Pair a request-scoped contract with the same full-schema natural recipe."""
    scoped = sorted(request_scoped_tools(
        query, set(scenario.starting_context.tool_allow_list or [])))
    proxy = SimpleNamespace(starting_context=SimpleNamespace(tool_allow_list=scoped))
    recipes = compose_select(pool, query, proxy)
    contract = contract_note(query, proxy, tokenizer)
    selected = [contract, *recipes[:1]]
    return selected if sum(note["tokens"] for note in selected) <= 512 else [contract]


def propose_fast_repair(action: dict, calls: list[dict], query: str,
                        allowed_tools: set[str]) -> tuple[dict, str] | None:
    """Return one safe executable repair, or ``None`` for feedback fallback."""
    allowed = set(allowed_tools) - {"end_conversation"}
    normalized = _instruction_text(query)
    successful = _successful_tools(calls)
    tool = action.get("tool")
    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    violation = fast_contract_violation(action, calls, query, allowed)

    # Public role words such as "my boss" belong in the relationship field.
    if tool == "search_contacts" and set(arguments) == {"name"}:
        value = str(arguments.get("name", "")).casefold()
        if value and f"my {value}" in normalized and _target_relationship(query):
            return ({"tool": "search_contacts", "arguments": {"relationship": value}},
                    "map public role phrase to the relationship search field")

    keyword = _source_message_keyword(query)
    if keyword and tool in {"add_contact", "send_message_with_phone_number"}:
        source = _source_message(calls, query)
        if source is None and "search_messages" in allowed:
            return ({"tool": "search_messages", "arguments": {"content": keyword}},
                    "resolve the requested public message sender before using its phone")
        if source is not None and arguments.get("phone_number") != source.get(
                "sender_phone_number"):
            corrected = deepcopy(arguments)
            corrected["phone_number"] = source["sender_phone_number"]
            return ({"tool": tool, "arguments": corrected},
                    "repair the phone number from the requested source-message sender")

    if violation is None:
        return None

    days = _relative_days(query)
    current_call = _last_success(calls, "get_current_timestamp")
    current = _parsed_response(current_call) if current_call else None
    shift_call = _last_success(calls, "shift_timestamp")
    shifted = _parsed_response(shift_call) if shift_call else None
    info_call = _last_success(calls, "timestamp_to_datetime_info")
    date_info = _parsed_response(info_call) if info_call else None
    rebuilt_call = _last_success(calls, "datetime_info_to_timestamp")
    rebuilt = _parsed_response(rebuilt_call) if rebuilt_call else None

    latest = "search_messages" in allowed and any(
        phrase in normalized for phrase in LATEST_PHRASES)
    if latest and tool == "search_messages":
        if current is None and "get_current_timestamp" in allowed:
            return ({"tool": "get_current_timestamp", "arguments": {}},
                    "obtain the public current-time anchor before latest-message search")
        global_latest = not re.search(
            r"\b(?:latest|newest|most recent)\s+(?:message|text)\s+(?:from|to)\b",
            normalized)
        corrected = {} if global_latest else deepcopy(arguments)
        corrected.pop("creation_timestamp_lowerbound", None)
        corrected["creation_timestamp_upperbound"] = current
        return ({"tool": "search_messages", "arguments": corrected},
                "repair latest-message search to use the current-time upper bound")

    if days is not None and "shift_timestamp" in allowed:
        if current is None:
            return ({"tool": "get_current_timestamp", "arguments": {}},
                    "obtain the public current-time anchor before relative-date conversion")
        if tool == "shift_timestamp":
            return ({"tool": "shift_timestamp",
                     "arguments": {"timestamp": current, "days": days}},
                    "repair the relative-date shift arguments")
        if shifted is None and tool in {
                "timestamp_to_datetime_info", "datetime_info_to_timestamp"}:
            return ({"tool": "shift_timestamp",
                     "arguments": {"timestamp": current, "days": days}},
                    "insert the missing relative-date shift")
        if tool == "timestamp_to_datetime_info" and shifted is not None:
            return ({"tool": "timestamp_to_datetime_info",
                     "arguments": {"timestamp": shifted}},
                    "repair datetime conversion to use the shifted timestamp")
        if tool == "datetime_info_to_timestamp":
            if shifted is not None and not isinstance(date_info, dict):
                return ({"tool": "timestamp_to_datetime_info",
                         "arguments": {"timestamp": shifted}},
                        "insert the missing shifted-timestamp conversion")
            if isinstance(date_info, dict):
                corrected = deepcopy(arguments)
                for key in ("year", "month", "day"):
                    corrected[key] = date_info[key]
                return ({"tool": "datetime_info_to_timestamp", "arguments": corrected},
                        "repair the calendar date while preserving the proposed clock time")

    contact_replacement, reminder_replacement = _replacement_targets(query, allowed)
    if contact_replacement and tool == "add_contact" and "remove_contact" not in successful:
        row = _replacement_row(calls, "search_contacts", "person_id", query)
        if row:
            return ({"tool": "remove_contact",
                     "arguments": {"person_id": row["person_id"]}},
                    "execute the required removal before contact recreation")
    if reminder_replacement and tool == "add_reminder" and "remove_reminder" in allowed \
            and "remove_reminder" not in successful:
        row = _replacement_row(calls, "search_reminder", "reminder_id", query)
        if row:
            return ({"tool": "remove_reminder",
                     "arguments": {"reminder_id": row["reminder_id"]}},
                    "execute the required removal before reminder recreation")

    if tool == "add_reminder":
        corrected = deepcopy(arguments)
        changed = False
        if days is not None and rebuilt is not None and corrected.get(
                "reminder_timestamp") != rebuilt:
            corrected["reminder_timestamp"] = rebuilt
            changed = True
        message = _latest_message(calls) if latest else None
        if message and corrected.get("content") != message.get("content"):
            corrected["content"] = message.get("content")
            changed = True
        if changed:
            return ({"tool": "add_reminder", "arguments": corrected},
                    "repair reminder arguments from observed public tool results")

    if tool == "send_message_with_phone_number" and latest:
        message = _latest_message(calls)
        if message and any(phrase in normalized for phrase in
                           ("reply", "send back", "whoever sent", "sender")):
            corrected = deepcopy(arguments)
            corrected["phone_number"] = message.get("sender_phone_number")
            return ({"tool": tool, "arguments": corrected},
                    "repair reply destination from the selected message sender")

    if "answer" in action:
        missing = request_required_mutations(query, allowed) - successful
        if missing == {"modify_contact"}:
            row = _contact_for_public_role(calls, query)
            relationship = _target_relationship(query)
            if row and relationship:
                return ({"tool": "modify_contact", "arguments": {
                    "person_id": row["person_id"], "relationship": relationship}},
                    "complete the uniquely resolved contact mutation before answering")
        if missing == {"add_reminder"} and rebuilt is not None:
            message = _latest_message(calls) if latest else None
            if message:
                return ({"tool": "add_reminder", "arguments": {
                    "content": message["content"], "reminder_timestamp": rebuilt}},
                    "complete the fully resolved reminder mutation before answering")
    return None


def matched_recipe_select(pool: list[dict], query: str, scenario, tokenizer) -> list[dict]:
    """Expose the exact natural recipe paired with a contract, without the contract."""
    return [note for note in fast_contract_select(pool, query, scenario, tokenizer)
            if note.get("memory_schema") != "request_derived_execution_contract_v1"][:1]


def _transport_turn(generation: dict, action: dict, parse_error: str | None) -> dict:
    turn = {"prompt_sha256": generation["prompt_sha256"], "action": action,
            "parse_error": parse_error,
            "hit_generation_limit": generation["hit_generation_limit"]}
    for key in ("remote_latency_ms", "network_attempts", "provider_model",
                "system_fingerprint_sha256", "request_sha256", "response_sha256",
                "generation_uid", "successful_attempt_latency_ms",
                "rendered_input_tokens", "input_tokens", "output_tokens"):
        if key in generation:
            turn[key] = generation[key]
    return turn


def consume_validate(job: dict, generation: dict, u, max_calls: int,
                     query: str) -> None:
    """Execute valid actions and return request-scoped feedback otherwise."""
    began = perf_counter_ns()
    action, parse_error = adapter.parse_request(generation["agent_output"])
    violation = None if parse_error else fast_contract_violation(
        action, job["calls"], query, set(job["tools"]))
    if violation is None:
        adapter.consume(job, generation, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
        return
    turn = _transport_turn(generation, action, parse_error)
    turn["contract_violation"] = violation
    turn["local_consume_latency_ms"] = (perf_counter_ns() - began) / 1e6
    job["generations"].append(turn)
    job["chat"].append({"role": "assistant", "content": generation["agent_output"]})
    job["chat"].append({"role": "user", "content": json.dumps({
        "error": "execution_contract_violation", "instruction": violation},
        ensure_ascii=False)})


def _changed_fields(proposed: dict, executed: dict) -> list[str]:
    keys = set(proposed) | set(executed)
    changed = []
    for key in sorted(keys):
        if proposed.get(key) != executed.get(key):
            changed.append(key)
    if proposed.get("arguments") != executed.get("arguments"):
        before = proposed.get("arguments") if isinstance(proposed.get("arguments"), dict) else {}
        after = executed.get("arguments") if isinstance(executed.get("arguments"), dict) else {}
        changed.extend("arguments." + key for key in sorted(set(before) | set(after))
                       if before.get(key) != after.get(key))
    return changed


def _execute_action(job: dict, generation: dict, action: dict, u,
                    max_calls: int) -> tuple[dict, bool]:
    """Execute one action and return its trace turn and tool-success flag."""
    began = perf_counter_ns()
    patched = dict(generation)
    patched["agent_output"] = json.dumps(action, ensure_ascii=False,
                                           separators=(",", ":"))
    calls_before = len(job["calls"])
    adapter.consume(job, patched, u, max_calls)
    turn = job["generations"][-1]
    success = (len(job["calls"]) > calls_before and
               not job["calls"][-1].get("exception"))
    if len(job["calls"]) > calls_before:
        turn["repair_execution_exception"] = job["calls"][-1].get("exception")
    turn["local_consume_latency_ms"] = (perf_counter_ns() - began) / 1e6
    return turn, success


def _repair_metadata(turn: dict, proposed: dict, executed: dict, reason: str,
                     violation: str | None) -> None:
    turn["proposed_action"] = proposed
    turn["deterministic_repair"] = reason
    turn["repair_provenance"] = {
        "sources": ["public_request", "allowed_tool_schema",
                    "successful_tool_results", "proposed_action"],
        "changed_fields": _changed_fields(proposed, executed),
    }
    if violation is not None:
        turn["contract_violation"] = violation


def _maybe_local_complete(job: dict, u, query: str) -> bool:
    """Close a session locally once every request-derived write has succeeded."""
    if job["done"]:
        return False
    required = request_required_mutations(query, set(job["tools"]))
    if not required or not required <= _successful_tools(job["calls"]):
        return False
    u.set_current_context(job["context"])
    u.BaseRole.add_messages([u.Message(
        sender=u.RoleType.AGENT, recipient=u.RoleType.USER,
        content="Completed the requested actions.", conversation_active=False)])
    job["context"] = u.get_current_context()
    job["done"] = True
    job["termination"] = "local_contract_complete"
    job["generations"][-1]["local_completion"] = {
        "required_mutations": sorted(required),
        "message": "Completed the requested actions.",
    }
    return True


def consume_fast(job: dict, generation: dict, u, max_calls: int,
                 query: str) -> None:
    """Repair, safely resume a blocked proposal, and close completed contracts."""
    began = perf_counter_ns()
    action, parse_error = adapter.parse_request(generation["agent_output"])
    if parse_error:
        adapter.consume(job, generation, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
        return
    violation = fast_contract_violation(action, job["calls"], query, set(job["tools"]))
    repair = propose_fast_repair(action, job["calls"], query, set(job["tools"]))
    if repair is None:
        if violation is None:
            adapter.consume(job, generation, u, max_calls)
            _maybe_local_complete(job, u, query)
        else:
            consume_validate(job, generation, u, max_calls, query)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
        return

    executed, reason = repair
    turn, success = _execute_action(job, generation, executed, u, max_calls)
    _repair_metadata(turn, action, executed, reason, violation)

    # If the repair inserted a prerequisite, resume the original proposal
    # locally once each intervening public-result dependency is unambiguous.
    pending = (action if success and executed.get("tool") != action.get("tool")
               and action.get("tool") else None)
    root_uid = generation.get("generation_uid")
    for index in range(4):
        if pending is None or job["done"]:
            break
        next_violation = fast_contract_violation(
            pending, job["calls"], query, set(job["tools"]))
        next_repair = propose_fast_repair(
            pending, job["calls"], query, set(job["tools"]))
        if next_repair is None:
            if next_violation is not None:
                break
            next_action = pending
            next_reason = "resume the now-valid proposed action after its prerequisite"
        else:
            next_action, next_reason = next_repair
        local_generation = {
            "prompt_sha256": generation["prompt_sha256"] + f":local-{index + 1}",
            "agent_output": json.dumps(next_action, ensure_ascii=False,
                                       separators=(",", ":")),
            "hit_generation_limit": False,
        }
        local_turn, success = _execute_action(
            job, local_generation, next_action, u, max_calls)
        local_turn["local_continuation"] = True
        local_turn["root_remote_generation_uid"] = root_uid
        _repair_metadata(local_turn, pending, next_action,
                         next_reason, next_violation)
        if not success:
            break
        if next_action.get("tool") == pending.get("tool"):
            pending = None

    _maybe_local_complete(job, u, query)

