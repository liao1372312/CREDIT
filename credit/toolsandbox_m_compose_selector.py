"""Provenance-preserving composition memory and schema coverage selection.

The selector uses only the public request, allowed tool names, and verified
memory metadata.  It never reads evaluation outcomes from the target suite.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .toolsandbox_m import note_relevance
from .toolsandbox_m_followup import memory_from_v2
from .toolsandbox_m_intent_optimization import temporal_compatible


ACTION_PREFIXES = ("add_", "remove_", "modify_", "send_", "set_")
READ_PREFIXES = ("search_",)
CONVERSION_TOOLS = {
    "get_current_timestamp",
    "timestamp_to_datetime_info",
    "datetime_info_to_timestamp",
    "shift_timestamp",
    "timestamp_diff",
}
STATUS_GETTERS = {
    "get_wifi_status",
    "get_cellular_service_status",
    "get_location_service_status",
    "get_low_battery_mode_status",
}


def schema_requirements(allowed: set[str]) -> dict[str, float]:
    """Assign operation weights without recognizing benchmark task names.

    Write operations are weighted most, data-producing searches next, time
    conversion after that, and status probes last.  The experiment scenarios
    expose only tools that can contribute to the request, so the allow list is
    a compact task schema rather than a catalogue of unrelated tools.
    """
    weights: dict[str, float] = {}
    for tool in sorted(allowed):
        if tool == "end_conversation":
            continue
        if tool.startswith(ACTION_PREFIXES):
            weights[tool] = 5.0
        elif tool.startswith(READ_PREFIXES):
            weights[tool] = 3.0
        elif tool in CONVERSION_TOOLS:
            weights[tool] = 2.0
        elif tool in STATUS_GETTERS:
            weights[tool] = 1.0
        else:
            weights[tool] = 1.0
    return weights


def compose_select(pool: list[dict], query: str, scenario) -> list[dict]:
    """Greedily cover distinct schema operations with at most two memories."""
    allowed = set(scenario.starting_context.tool_allow_list or []) - {"end_conversation"}
    weights = schema_requirements(allowed)
    candidates = [
        note for note in pool
        if set(note["tools"]) & allowed and temporal_compatible(note, query)
    ]
    chosen: list[dict] = []
    covered: set[str] = set()

    def value(note: dict) -> tuple[float, float, float, str]:
        overlap = set(note["tools"]) & allowed
        marginal = overlap - covered
        action_count = sum(tool.startswith(ACTION_PREFIXES) for tool in marginal)
        composition = max(0, len(overlap) - 1)
        foreign = len(set(note["tools"]) - allowed)
        score = (
            sum(weights[tool] for tool in marginal)
            + 1.5 * action_count
            + 0.75 * composition
            + 0.25 * float(note.get("source_requested_state_verified", False))
            + 0.15 * float(note.get("source_upstream_score", 0.0))
            + 0.001 * note_relevance(note, query, allowed)
            - 0.35 * foreign
        )
        return (score, float(note.get("source_upstream_score", 0.0)),
                -float(note.get("tokens", 0)), note["uid"])

    for _ in range(2):
        remaining = [note for note in candidates if note not in chosen]
        if not remaining:
            break
        best = max(remaining, key=value)
        marginal = (set(best["tools"]) & allowed) - covered
        # A second memory must add an operation with real schema value.
        if chosen and sum(weights[tool] for tool in marginal) < 2.0:
            break
        if not marginal:
            break
        chosen.append(best)
        covered.update(set(best["tools"]) & allowed)
    return chosen


def _latest_snapshot(rows: list[dict]) -> list[dict]:
    indices = [row["sandbox_message_index"] for row in rows
               if row["sandbox_message_index"] is not None]
    index = max(indices)
    return [row for row in rows if row["sandbox_message_index"] == index
            and any(value is not None for key, value in row.items()
                    if key != "sandbox_message_index")]


def _corrected_replacement_success(row: dict, expected_time: float) -> bool:
    final = _latest_snapshot(row["ending_context"]["_dbs"]["REMINDER"])
    old_id = "54f20be9-eac3-5e76-8e57-724e48d3d477"
    old_absent = all(item["reminder_id"] != old_id for item in final)
    replacement = [item for item in final
                   if item["content"] == "Pack conference badge"
                   and abs(float(item["reminder_timestamp"]) - expected_time) < 1e-6]
    return old_absent and len(replacement) == 1


def build_composition_pool(source: Path, tokenizer, expected_replacement_time: float
                           ) -> tuple[list[dict], list[dict], list[dict]]:
    """Append one shortest verified natural recipe per prior task family.

    The prior blind suite is explicitly treated as development data.  The
    returned provenance table states which immutable episode created each note.
    """
    initial = json.loads((source / "memory_pool.json").read_text())
    rows = [json.loads(line) for line in
            (source / "confirmation_sessions.jsonl").read_text().splitlines()
            if line.strip()]
    families = sorted({row["trajectory"] for row in rows})
    added: list[dict] = []
    provenance: list[dict] = []
    for family in families:
        eligible = []
        for row in rows:
            if row["trajectory"] != family:
                continue
            state_ok = (_corrected_replacement_success(row, expected_replacement_time)
                        if family == "replace_upcoming_reminder"
                        else bool(row["mutation_success"]))
            if not state_ok:
                continue
            note = memory_from_v2(row, tokenizer)
            if note is None:
                continue
            errors = sum(bool(call.get("exception")) for call in row["calls"])
            rank = (errors, len(row["calls"]), row["remote_latency_ms"],
                    row["actor_seed"], row["method"], row["task"])
            eligible.append((rank, row, note))
        if not eligible:
            raise RuntimeError(f"no requested-state-verified source for {family}")
        _, row, note = min(eligible, key=lambda item: item[0])
        note.update({
            "origin_phase": "blind_composition_v1_development",
            "source_family": family,
            "source_actor_seed": row["actor_seed"],
            "source_method": row["method"],
            "source_requested_state_verified": True,
            "source_episode_sha256": sha256(json.dumps(
                row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "memory_schema": "verified_composition_recipe_v1",
        })
        added.append(note)
        provenance.append({
            "uid": note["uid"],
            "source_family": family,
            "source_task": row["task"],
            "source_actor_seed": row["actor_seed"],
            "source_method": row["method"],
            "source_requested_state_verified": True,
            "source_upstream_full_success": bool(row["success"]),
            "source_ending_state_sha256": row["ending_state_sha256"],
            "source_episode_sha256": note["source_episode_sha256"],
            "successful_tools": note["tools"],
            "exception_count": sum(bool(call.get("exception")) for call in row["calls"]),
        })
    return [*initial, *added], added, provenance
