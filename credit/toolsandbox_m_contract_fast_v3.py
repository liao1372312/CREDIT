"""Argument-aware fast contracts built on the frozen literal-safe v2 method.

The v2 contract verifies which requested mutations ran, but a successful tool
call can still omit an explicit public argument.  This module derives the
unambiguous contact tuple from the public request, repairs a proposed
``add_contact`` call before execution, and only completes locally after the
tuple itself has been written.  It never reads evaluator targets or hidden
world state.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
from time import perf_counter_ns

from . import toolsandbox_local_v3 as adapter
from . import toolsandbox_m_contract_fast as base
from . import toolsandbox_m_contract_fast_v2 as v2

# Reproduction runs keep the original low-call behavior by default. Accuracy
# pilots can defer completion to the model so it can produce a final response
# after a required mutation succeeds.
AUTO_COMPLETE_AFTER_MUTATION = True


fast_select = v2.fast_select
fast_contract_select = v2.fast_contract_select
matched_recipe_select = v2.matched_recipe_select
request_required_mutations = v2.request_required_mutations
request_scoped_tools = v2.request_scoped_tools


_PHONE = r"\+[0-9]{8,15}"
_WORD = r"[A-Za-z][A-Za-z0-9_-]*"
_RELATIONSHIP = r"(?!a\b|an\b|the\b)" + _WORD


def _contact_clause(query: str, phone: str) -> str:
    """Return the punctuation-delimited public clause containing ``phone``."""
    chunks = re.split(r"[;,]|\bthen\b|\band\b", query, flags=re.IGNORECASE)
    return next((chunk.strip() for chunk in chunks if phone in chunk), query)


def explicit_add_contact_arguments(query: str) -> dict[str, str] | None:
    """Extract a unique, explicit contact name, phone, and relationship.

    The extractor intentionally abstains unless the request contains exactly
    one phone number and states all three fields in the same contact clause.
    """
    phones = list(dict.fromkeys(re.findall(_PHONE, query)))
    if len(phones) != 1:
        return None
    phone = phones[0]
    clause = _contact_clause(query, phone)

    name = None
    patterns = (
        r"\b(?:named|called|under)\s+(['\"])(.+?)\1",
        r"\b(?:named|called)\s+(.+?)(?=\s+(?:at|with\s+(?:the\s+)?number)\s+" +
        re.escape(phone) + r"\b)",
        r"\bsave\s+(.+?)\s*\(\s*" + re.escape(phone) + r"\s*\)",
    )
    for pattern in patterns:
        match = re.search(pattern, clause, flags=re.IGNORECASE)
        if match:
            name = match.group(2 if match.lastindex == 2 else 1).strip(" \t'\"")
            break

    relationship = None
    relationship_patterns = (
        r"\bwith\s+(?:the\s+)?relationship\s+(" + _RELATIONSHIP + r")\b",
        r"\brelationship\s+(?:to|as)?\s*(" + _RELATIONSHIP + r")\b",
        r"\bas\s+(?:an?\s+)?(" + _RELATIONSHIP + r")\s+contact\b",
        r"\b(?:create|add|save)\s+(?:an?\s+)?(" + _RELATIONSHIP +
        r")\s+contact\b",
        re.escape(phone) + r"\s+as\s+(?:an?\s+)?(" + _RELATIONSHIP + r")\b",
    )
    for pattern in relationship_patterns:
        match = re.search(pattern, clause, flags=re.IGNORECASE)
        if match:
            relationship = match.group(1).casefold()
            break

    if not name or not relationship:
        return None
    return {"name": name, "phone_number": phone,
            "relationship": relationship}


def _successful_contact(calls: list[dict], expected: dict[str, str]) -> bool:
    return any(
        call.get("tool") == "add_contact" and not call.get("exception") and
        all(call.get("arguments", {}).get(key) == value
            for key, value in expected.items())
        for call in calls
    )


def _argument_violation(action: dict, calls: list[dict], query: str) -> str | None:
    expected = explicit_add_contact_arguments(query)
    if expected is None or action.get("tool") != "add_contact":
        return None
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, dict) else {}
    mismatched = [key for key, value in expected.items()
                  if arguments.get(key) != value]
    if mismatched:
        return ("add_contact must preserve these explicit public-request fields: " +
                ", ".join(mismatched))
    if _successful_contact(calls, expected):
        return "the explicitly requested contact has already been added"
    return None


def fast_contract_violation(action: dict, calls: list[dict], query: str,
                            allowed_tools: set[str]) -> str | None:
    """Apply v2 ordering checks plus explicit contact argument checks."""
    return (_argument_violation(action, calls, query) or
            v2.fast_contract_violation(action, calls, query, allowed_tools))


def _role_word(query: str) -> str | None:
    text = base._instruction_text(query)
    target = base._target_relationship(query)
    if target:
        match = re.search(
            r"\b(?:mark|set|change|update)\s+my\s+(" + _WORD +
            r")\b[^.;,]{0,60}\b(?:as|to)\s+" + re.escape(target) + r"\b",
            text)
        if match:
            return match.group(1)
    candidates = re.findall(r"\bmy\s+(" + _WORD + r")\b", text)
    ignored = {"latest", "most", "newest", "next", "recent"}
    return next((value for value in reversed(candidates)
                 if value not in ignored), None)


def propose_fast_repair(action: dict, calls: list[dict], query: str,
                        allowed_tools: set[str]) -> tuple[dict, str] | None:
    """Repair explicit contact arguments and mistaken role creation locally."""
    expected = explicit_add_contact_arguments(query)
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, dict) else {}
    if expected is not None and action.get("tool") == "add_contact":
        mismatched = any(arguments.get(key) != value
                         for key, value in expected.items())
        if mismatched:
            required = request_required_mutations(query, set(allowed_tools))
            target_relationship = base._target_relationship(query)
            wrong_target = arguments.get("phone_number") != expected["phone_number"]
            modified = "modify_contact" in base._successful_tools(calls)
            if wrong_target and "modify_contact" in required and not modified:
                row = base._contact_for_public_role(calls, query)
                if row is not None and target_relationship:
                    return ({"tool": "modify_contact", "arguments": {
                        "person_id": row["person_id"],
                        "relationship": target_relationship}},
                        "map mistaken role creation to the requested existing-contact update")
                role = _role_word(query)
                if role and "search_contacts" in allowed_tools:
                    return ({"tool": "search_contacts",
                             "arguments": {"relationship": role}},
                            "resolve the requested existing role contact before updating it")
            corrected = deepcopy(arguments)
            corrected.update(expected)
            return ({"tool": "add_contact", "arguments": corrected},
                    "restore explicit contact fields from the public request")
    if fast_contract_violation(action, calls, query, allowed_tools) is None:
        return None
    return v2.propose_fast_repair(action, calls, query, allowed_tools)


def _maybe_local_complete(job: dict, u, query: str) -> bool:
    """Finish only after required tools and explicit contact fields succeeded."""
    if job["done"]:
        return False
    required = request_required_mutations(query, set(job["tools"]))
    if not required or not required <= base._successful_tools(job["calls"]):
        return False
    expected = explicit_add_contact_arguments(query)
    if expected is not None and "add_contact" in required and not _successful_contact(
            job["calls"], expected):
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
        "explicit_contact_arguments": expected,
        "message": "Completed the requested actions.",
    }
    return True


def consume_validate(job: dict, generation: dict, u, max_calls: int,
                     query: str) -> None:
    """Execute actions accepted by the argument-aware request contract."""
    began = perf_counter_ns()
    action, parse_error = adapter.parse_request(generation["agent_output"])
    violation = None if parse_error else fast_contract_violation(
        action, job["calls"], query, set(job["tools"]))
    if violation is None:
        adapter.consume(job, generation, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
        return
    turn = base._transport_turn(generation, action, parse_error)
    turn["contract_violation"] = violation
    turn["local_consume_latency_ms"] = (perf_counter_ns() - began) / 1e6
    job["generations"].append(turn)
    job["chat"].append({"role": "assistant", "content": generation["agent_output"]})
    job["chat"].append({"role": "user", "content": json.dumps({
        "error": "execution_contract_violation", "instruction": violation},
        ensure_ascii=False)})


def consume_fast(job: dict, generation: dict, u, max_calls: int,
                 query: str) -> None:
    """Repair arguments, safely continue prerequisites, and finish locally."""
    began = perf_counter_ns()
    action, parse_error = adapter.parse_request(generation["agent_output"])
    if parse_error:
        adapter.consume(job, generation, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
        return

    allowed = set(job["tools"])
    violation = fast_contract_violation(action, job["calls"], query, allowed)
    repair = propose_fast_repair(action, job["calls"], query, allowed)
    if repair is None:
        if violation is None:
            adapter.consume(job, generation, u, max_calls)
            if AUTO_COMPLETE_AFTER_MUTATION:
                _maybe_local_complete(job, u, query)
        else:
            consume_validate(job, generation, u, max_calls, query)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
        return

    executed, reason = repair
    turn, success = base._execute_action(job, generation, executed, u, max_calls)
    base._repair_metadata(turn, action, executed, reason, violation)
    pending = (action if success and executed.get("tool") != action.get("tool")
               and action.get("tool") else None)
    root_uid = generation.get("generation_uid")
    for index in range(4):
        if pending is None or job["done"]:
            break
        next_violation = fast_contract_violation(
            pending, job["calls"], query, allowed)
        next_repair = propose_fast_repair(
            pending, job["calls"], query, allowed)
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
        local_turn, success = base._execute_action(
            job, local_generation, next_action, u, max_calls)
        local_turn["local_continuation"] = True
        local_turn["root_remote_generation_uid"] = root_uid
        base._repair_metadata(local_turn, pending, next_action,
                              next_reason, next_violation)
        if not success:
            break
        if next_action.get("tool") == pending.get("tool"):
            pending = None
    if AUTO_COMPLETE_AFTER_MUTATION:
        _maybe_local_complete(job, u, query)
