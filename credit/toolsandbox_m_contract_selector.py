"""Request-derived execution contracts for composed ToolSandbox memories.

The contract is generated only from the public request and exposed tool names.
It carries no target answer, identifier, timestamp, or evaluator state.  A
single provenance-linked natural recipe is retained beside the contract.
"""

from __future__ import annotations

import ast
from hashlib import sha256
import re

from .toolsandbox_m_compose_selector import compose_select


REPLACEMENT_WORDS = (
    "replace", "recreate", "re-create", "rebuild", "swap out", "substitute",
)


def _has_phrase(query: str, phrases: tuple[str, ...]) -> bool:
    value = " ".join(query.casefold().split())
    return any(phrase in value for phrase in phrases)


def _relative_days(query: str) -> int | None:
    value = " ".join(query.casefold().split())
    if "day after tomorrow" in value:
        return 2
    if "tomorrow" in value or "next day" in value:
        return 1
    match = re.search(r"\bin\s+(\d+)\s+days?\b", value)
    return int(match.group(1)) if match else None


def _parsed_response(call: dict):
    try:
        return ast.literal_eval(call.get("response", ""))
    except (ValueError, SyntaxError):
        return None


def _last_success(calls: list[dict], tool: str) -> dict | None:
    return next((call for call in reversed(calls)
                 if call.get("tool") == tool and not call.get("exception")), None)


def contract_violation(action: dict, calls: list[dict], query: str,
                       allowed_tools: set[str]) -> str | None:
    """Return corrective feedback when a proposed action violates the contract.

    Validation uses only the public request, exposed schema, and observed tool
    history.  It does not inspect scenario milestones or target state.
    """
    allowed = set(allowed_tools) - {"end_conversation"}
    normalized = " ".join(query.casefold().split())
    tool = action.get("tool")
    arguments = action.get("arguments", {})
    successful = {call["tool"] for call in calls if not call.get("exception")}
    replacement = _has_phrase(query, REPLACEMENT_WORDS)

    if replacement and tool == "add_contact" and "remove_contact" not in successful:
        return "contact replacement requires a successful remove_contact before add_contact"
    if replacement and tool == "add_reminder" and "remove_reminder" in allowed \
            and "remove_reminder" not in successful:
        return "reminder replacement requires a successful remove_reminder before add_reminder"

    days = _relative_days(query)
    shifted = days is not None and "shift_timestamp" in allowed
    if shifted and tool == "shift_timestamp":
        current = _last_success(calls, "get_current_timestamp")
        current_value = _parsed_response(current) if current else None
        if current_value is None or arguments.get("timestamp") != current_value:
            return "shift_timestamp must use the exact get_current_timestamp result"
        if arguments.get("days") != days:
            return f"shift_timestamp must apply days={days} exactly once"
    if shifted and tool == "timestamp_to_datetime_info":
        shift = _last_success(calls, "shift_timestamp")
        shifted_value = _parsed_response(shift) if shift else None
        if shifted_value is None or arguments.get("timestamp") != shifted_value:
            return "timestamp_to_datetime_info must use the exact shifted timestamp"
    if shifted and tool == "datetime_info_to_timestamp":
        converted = _last_success(calls, "timestamp_to_datetime_info")
        info = _parsed_response(converted) if converted else None
        if not isinstance(info, dict):
            return "convert the shifted timestamp to datetime info before rebuilding the target time"
        if any(arguments.get(key) != info.get(key) for key in ("year", "month", "day")):
            return "keep year, month, and day from the shifted datetime result unchanged"
    if days is not None and tool == "add_reminder":
        converted = _last_success(calls, "datetime_info_to_timestamp")
        timestamp = _parsed_response(converted) if converted else None
        if timestamp is None or arguments.get("reminder_timestamp") != timestamp:
            return "add_reminder must use the exact datetime_info_to_timestamp result"

    latest_phrases = ("latest message", "latest text", "newest message", "newest text",
                      "most recent message", "most recent text")
    latest = "search_messages" in allowed and any(p in normalized for p in latest_phrases)
    if latest and tool == "search_messages":
        current = _last_success(calls, "get_current_timestamp")
        current_value = _parsed_response(current) if current else None
        if current_value is None:
            return "get the current timestamp before searching for the latest message"
        if arguments.get("creation_timestamp_upperbound") != current_value:
            return "latest-message search must use current time as creation_timestamp_upperbound"
        if "creation_timestamp_lowerbound" in arguments:
            return "latest-message search must not use a lower bound equal to now"
    message_call = _last_success(calls, "search_messages") if latest else None
    rows = _parsed_response(message_call) if message_call else None
    latest_row = (max(rows, key=lambda row: row["creation_timestamp"])
                  if isinstance(rows, list) and rows else None)
    if latest_row and tool == "add_reminder" and arguments.get("content") != latest_row.get("content"):
        return "the reminder must keep the selected latest message content unchanged"
    if latest_row and tool == "send_message_with_phone_number" and any(
            phrase in normalized for phrase in ("reply", "send back", "whoever sent", "sender")):
        if arguments.get("phone_number") != latest_row.get("sender_phone_number"):
            return "reply to the selected message's sender_phone_number"

    if "answer" in action:
        required = {name for name in allowed
                    if name.startswith(("add_", "remove_", "modify_", "send_", "set_"))}
        missing = sorted(required - successful)
        if missing:
            return "do not finish before these requested mutations succeed: " + ", ".join(missing)
    return None


def derive_execution_contract(query: str, allowed_tools: set[str]) -> list[str]:
    """Derive operation invariants without benchmark names or hidden state."""
    allowed = set(allowed_tools) - {"end_conversation"}
    rules = [
        "Use search results as the source of current IDs and values; preserve "
        "needed fields before any deletion.",
    ]

    replacement = _has_phrase(query, REPLACEMENT_WORDS)
    normalized_query = " ".join(query.casefold().split())
    if replacement and {"search_contacts", "remove_contact", "add_contact"} <= allowed:
        rules.append(
            "A contact replacement is incomplete until search_contacts, "
            "remove_contact on the returned person_id, and add_contact have "
            "all succeeded in that order; adding a second record alone is not "
            "a replacement."
        )
    if replacement and {"search_reminder", "remove_reminder", "add_reminder"} <= allowed:
        rules.append(
            "A reminder replacement is incomplete until search_reminder, "
            "remove_reminder on the returned reminder_id, and add_reminder "
            "have all succeeded in that order; preserve the requested old "
            "fields before removal."
        )

    days = _relative_days(query)
    clock_chain = {
        "get_current_timestamp", "timestamp_to_datetime_info",
        "datetime_info_to_timestamp",
    }
    shifted_clock_chain = clock_chain | {"shift_timestamp"}
    if days is not None and shifted_clock_chain <= allowed:
        rules.append(
            "Anchor the relative date to get_current_timestamp: pass that exact "
            f"result to one shift_timestamp call with days={days}, pass the "
            "shifted result unchanged to timestamp_to_datetime_info, keep its "
            "returned year/month/day unchanged, set only the requested clock "
            "time, and call datetime_info_to_timestamp once. Do not increment "
            "the calendar date again and do not anchor to a message or reminder time."
        )
    elif days is not None and clock_chain <= allowed:
        rules.append(
            "Anchor the relative date to get_current_timestamp: pass that exact "
            "returned timestamp unchanged to timestamp_to_datetime_info, move "
            f"the returned calendar date forward exactly {days} day(s), set the "
            "requested clock time, and call datetime_info_to_timestamp once. "
            "Do not pre-shift the timestamp and then advance the calendar date "
            "again, and do not anchor the date to a message or reminder time."
        )
    elif days is not None and {"get_current_timestamp", "shift_timestamp"} <= allowed:
        rules.append(
            "Anchor the relative date to the exact get_current_timestamp result "
            f"and apply exactly one shift_timestamp call with days={days}; do "
            "not apply the same day offset a second time."
        )

    if ("search_messages" in allowed and
            any(phrase in normalized_query for phrase in
                ("latest message", "latest text", "newest message", "newest text",
                 "most recent message", "most recent text"))):
        rules.append(
            "For the latest/newest message, search with "
            "creation_timestamp_upperbound equal to the current timestamp and "
            "select the returned row with the greatest creation_timestamp; a "
            "lower bound equal to now incorrectly excludes existing messages."
        )
    if ("search_reminder" in allowed and
            any(phrase in normalized_query for phrase in
                ("next reminder", "upcoming reminder"))):
        rules.append(
            "For the next/upcoming reminder, search with "
            "reminder_timestamp_lowerbound equal to the current timestamp and "
            "select the returned row with the smallest reminder_timestamp; use "
            "that result directly instead of searching for the requested new content."
        )
    if ({"search_messages", "add_reminder", "send_message_with_phone_number"} <= allowed
            and any(phrase in normalized_query for phrase in
                    ("latest message", "latest text", "newest message", "newest text",
                     "most recent message", "most recent text"))):
        rules.append(
            "When one source message feeds multiple outputs, keep its raw content "
            "for outputs that request the message itself and apply any prefix, "
            "suffix, or replacement only to the output explicitly described as "
            "transformed."
        )

    if "send_message_with_phone_number" in allowed:
        if any(phrase in normalized_query for phrase in
               ("reply", "send back", "whoever sent", "sender")):
            rules.append(
                "For a reply to a found message, send to that returned row's "
                "sender_phone_number, never its recipient_phone_number."
            )
        rules.append(
            "Resolve or preserve the destination phone number before sending, "
            "and use the requested message text exactly."
        )
    rules.append(
        "Before answering, check that every requested mutation in this contract "
        "has a successful tool result."
    )
    return rules


def contract_note(query: str, scenario, tokenizer) -> dict:
    """Render a bounded, auditable note from the request and tool schema."""
    allowed = set(scenario.starting_context.tool_allow_list or []) - {"end_conversation"}
    rules = derive_execution_contract(query, allowed)
    text = "Derived execution contract: " + " ".join(
        f"{index + 1}) {rule}" for index, rule in enumerate(rules)
    )
    tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if tokens > 300:
        raise ValueError(f"execution contract exceeds 300 tokens: {tokens}")
    uid = "contract-" + sha256((query + "\n" + "\n".join(sorted(allowed)) +
                                 "\n" + text).encode()).hexdigest()[:16]
    return {
        "uid": uid,
        "text": text,
        "tools": sorted(allowed),
        "tokens": tokens,
        "source_task": "public_request_and_tool_schema",
        "source_ending_sha256": None,
        "source_verifier_success": False,
        "source_exact_milestone_verified": False,
        "source_upstream_score": 0.0,
        "arrival": uid,
        "memory_schema": "request_derived_execution_contract_v1",
        "origin_phase": "online_request_derivation",
    }


def contract_select(pool: list[dict], query: str, scenario, tokenizer) -> list[dict]:
    """Return one request-derived contract and one verified natural recipe."""
    contract = contract_note(query, scenario, tokenizer)
    recipes = compose_select(pool, query, scenario)
    allowed = set(scenario.starting_context.tool_allow_list or [])
    # A recipe that demonstrates manual calendar arithmetic conflicts with a
    # contract that requires the safer shift tool.  Compatibility is decided
    # from tool schemas alone, before any target outcome is observed.
    if _relative_days(query) is not None and "shift_timestamp" in allowed:
        recipes = [note for note in recipes if "shift_timestamp" in note["tools"]]
    # One recipe leaves room for the higher-priority contract in the two-note
    # exposure limit and prevents two related recipes from diluting it.
    selected = [contract, *recipes[:1]]
    if sum(note["tokens"] for note in selected) > 512:
        selected = [contract]
    return selected
