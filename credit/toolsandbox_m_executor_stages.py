"""Executor stages for the registered ToolSandbox-M component factorial.

The stages are nested by construction:

``Base`` executes the model proposal, ``Validate`` adds request-contract
feedback, ``Repair`` additionally applies deterministic public-data repairs,
and ``Full`` also closes a request locally after all requested writes succeed.
This module implements only the Repair stage; the other three stages call the
already frozen implementations directly.
"""

from __future__ import annotations

import json
from time import perf_counter_ns

from . import toolsandbox_local_v3 as adapter
from . import toolsandbox_m_contract_fast as base
from . import toolsandbox_m_contract_fast_v3 as v3


STAGES = ("Base", "Validate", "Repair", "Full")


def consume_repair(job: dict, generation: dict, u, max_calls: int,
                   query: str) -> None:
    """Apply v3 validation and repair without local session completion."""
    began = perf_counter_ns()
    action, parse_error = adapter.parse_request(generation["agent_output"])
    if parse_error:
        adapter.consume(job, generation, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
        return

    allowed = set(job["tools"])
    violation = v3.fast_contract_violation(
        action, job["calls"], query, allowed)
    repair = v3.propose_fast_repair(action, job["calls"], query, allowed)
    if repair is None:
        if violation is None:
            adapter.consume(job, generation, u, max_calls)
        else:
            v3.consume_validate(job, generation, u, max_calls, query)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
        return

    executed, reason = repair
    turn, success = base._execute_action(
        job, generation, executed, u, max_calls)
    base._repair_metadata(turn, action, executed, reason, violation)

    pending = (action if success and
               executed.get("tool") != action.get("tool") and
               action.get("tool") else None)
    root_uid = generation.get("generation_uid")
    for index in range(4):
        if pending is None or job["done"]:
            break
        next_violation = v3.fast_contract_violation(
            pending, job["calls"], query, allowed)
        next_repair = v3.propose_fast_repair(
            pending, job["calls"], query, allowed)
        if next_repair is None:
            if next_violation is not None:
                break
            next_action = pending
            next_reason = (
                "resume the now-valid proposed action after its prerequisite")
        else:
            next_action, next_reason = next_repair
        local_generation = {
            "prompt_sha256": generation["prompt_sha256"] +
            f":local-{index + 1}",
            "agent_output": json.dumps(
                next_action, ensure_ascii=False, separators=(",", ":")),
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


def consume_stage(stage: str, job: dict, generation: dict, u,
                  max_calls: int, query: str) -> None:
    """Dispatch one generation to the registered nested executor stage."""
    if stage == "Base":
        began = perf_counter_ns()
        adapter.consume(job, generation, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6
    elif stage == "Validate":
        v3.consume_validate(job, generation, u, max_calls, query)
    elif stage == "Repair":
        consume_repair(job, generation, u, max_calls, query)
    elif stage == "Full":
        v3.consume_fast(job, generation, u, max_calls, query)
    else:
        raise ValueError("unknown executor stage: " + stage)
