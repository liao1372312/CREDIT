"""Second-generation fast contract execution with literal-safe validation.

The frozen v1 implementation correctly excluded quoted literals when it
derived requested mutations, but passed the original request to the legacy
ordering validator.  Words such as ``replace`` inside a contact name or exact
message payload could therefore activate a replacement constraint.  This
module preserves the frozen implementation and narrows validation to the
literal-free instruction surface.
"""

from __future__ import annotations

import json
from time import perf_counter_ns

from . import toolsandbox_local_v3 as adapter
from . import toolsandbox_m_contract_fast as v1


# Selection and contract derivation are unchanged from the frozen version.
fast_select = v1.fast_select
fast_contract_select = v1.fast_contract_select
matched_recipe_select = v1.matched_recipe_select
request_required_mutations = v1.request_required_mutations
request_scoped_tools = v1.request_scoped_tools


def fast_contract_violation(action: dict, calls: list[dict], query: str,
                            allowed_tools: set[str]) -> str | None:
    """Validate intent while excluding quoted names and payload literals."""
    return v1.fast_contract_violation(
        action, calls, v1._instruction_text(query), allowed_tools)


def propose_fast_repair(action: dict, calls: list[dict], query: str,
                        allowed_tools: set[str]) -> tuple[dict, str] | None:
    """Use v1 repairs only when the literal-safe validator finds a violation."""
    if fast_contract_violation(action, calls, query, allowed_tools) is None:
        return None
    return v1.propose_fast_repair(action, calls, query, allowed_tools)


def consume_validate(job: dict, generation: dict, u, max_calls: int,
                     query: str) -> None:
    """Execute actions accepted by the literal-safe request-scoped contract."""
    began = perf_counter_ns()
    action, parse_error = adapter.parse_request(generation["agent_output"])
    violation = None if parse_error else fast_contract_violation(
        action, job["calls"], query, set(job["tools"]))
    if violation is None:
        adapter.consume(job, generation, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
        return
    turn = v1._transport_turn(generation, action, parse_error)
    turn["contract_violation"] = violation
    turn["local_consume_latency_ms"] = (perf_counter_ns() - began) / 1e6
    job["generations"].append(turn)
    job["chat"].append({"role": "assistant", "content": generation["agent_output"]})
    job["chat"].append({"role": "user", "content": json.dumps({
        "error": "execution_contract_violation", "instruction": violation},
        ensure_ascii=False)})


def consume_fast(job: dict, generation: dict, u, max_calls: int,
                 query: str) -> None:
    """Apply literal-safe validation, deterministic repair, and local finish."""
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
            v1._maybe_local_complete(job, u, query)
        else:
            consume_validate(job, generation, u, max_calls, query)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
        return

    executed, reason = repair
    turn, success = v1._execute_action(job, generation, executed, u, max_calls)
    v1._repair_metadata(turn, action, executed, reason, violation)

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
        local_turn, success = v1._execute_action(
            job, local_generation, next_action, u, max_calls)
        local_turn["local_continuation"] = True
        local_turn["root_remote_generation_uid"] = root_uid
        v1._repair_metadata(local_turn, pending, next_action,
                            next_reason, next_violation)
        if not success:
            break
        if next_action.get("tool") == pending.get("tool"):
            pending = None

    v1._maybe_local_complete(job, u, query)

