"""Replay audit over the held-out public ToolSandbox trajectories.

The audit reexecutes stored actions against freshly restored official scenario
states.  ``full`` replays every recorded tool action; ``scoped`` replays only
state-changing actions and preserves the recorded final answer.  The matched
budget comparator truncates the full trace to the scoped tool-call count.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from transformers import AutoTokenizer

from . import toolsandbox_local_v3 as adapter
from .toolsandbox_m_followup import initial_job_v2


MUTATIONS = {
    "add_contact", "remove_contact", "modify_contact",
    "add_reminder", "remove_reminder", "modify_reminder",
    "send_message_with_phone_number", "set_wifi_status",
    "set_location_service_status", "set_cellular_service_status",
    "set_low_battery_mode_status",
}


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def physical_state(ending: dict) -> dict:
    value = json.loads(json.dumps(ending))
    value.get("_dbs", {}).pop("sandbox", None)
    return value


def stable(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode()).hexdigest()


def normalized_trace(row: dict) -> list[dict]:
    return [{"tool": call["tool"], "arguments": call["arguments"],
             "response": adapter.stable_exception_text(call.get("response")),
             "exception": adapter.stable_exception_text(call.get("exception"))}
            for call in row["calls"]]


def terminal_answer(source: dict) -> str:
    for turn in reversed(source["generations"]):
        action = turn.get("action") or {}
        if isinstance(action.get("answer"), str):
            return action["answer"]
    return "Completed the requested actions."


def replay(source: dict, actions: list[dict], name: str, scenario, bank, tokenizer,
           directory: Path, u) -> dict:
    began = perf_counter()
    job = initial_job_v2(source["task"], scenario, bank, u, tokenizer,
                         "replay_audit", name)
    for index, call in enumerate(actions):
        generation = {"prompt_sha256": f"replay-{name}-{index}",
                      "agent_output": json.dumps({"tool": call["tool"],
                                                   "arguments": call["arguments"]},
                                                  ensure_ascii=False),
                      "hit_generation_limit": False}
        adapter.consume(job, generation, u, max(12, len(actions)))
        if job["done"]:
            break
    if not job["done"]:
        generation = {"prompt_sha256": f"replay-{name}-answer",
                      "agent_output": json.dumps({"answer": terminal_answer(source)},
                                                 ensure_ascii=False),
                      "hit_generation_limit": False}
        adapter.consume(job, generation, u, max(12, len(actions)))
    row = adapter.finish(job, directory, u)
    row["measured_replay_seconds"] = perf_counter() - began
    row["native_score"] = (float(row["upstream_evaluation"]["similarity"])
                           if row["upstream_evaluation"] else 0.0)
    row["relevant_state_sha256"] = stable(physical_state(row["ending_context"]))
    row["normalized_trace_sha256"] = stable(normalized_trace(row))
    row["unsafe"] = int(bool(row["upstream_evaluation"] and
                             row["upstream_evaluation"]["minefield_similarity"] > 0))
    return row


def comparison(left: list[dict], right: list[dict], condition: str) -> dict:
    if len(left) != len(right):
        raise ValueError("unpaired replay rows")
    return {
        "comparison": f"{left[0]['method'].split(':')[0]} / {right[0]['method'].split(':')[0]}",
        "condition": condition, "N": len(left),
        "state_diff": sum(a["relevant_state_sha256"] != b["relevant_state_sha256"]
                          for a, b in zip(left, right)),
        "trace_diff": sum(a["normalized_trace_sha256"] != b["normalized_trace_sha256"]
                          for a, b in zip(left, right)),
        "outcome_diff": sum(abs(a["native_score"] - b["native_score"]) > 1e-12
                            for a, b in zip(left, right)),
        "decision_diff": sum(a["success"] != b["success"] for a, b in zip(left, right)),
        "calls_a": sum(a["tool_calls"] for a in left),
        "calls_b": sum(b["tool_calls"] for b in right),
    }


def policy(rows: list[dict], label: str) -> dict:
    accepted = [row for row in rows if row["native_score"] >= 0.90]
    unsafe = sum(row["unsafe"] for row in accepted)
    return {"policy": label, "primary_score": float(np.mean([r["native_score"] for r in rows])),
            "accepted_updates": len(accepted), "unsafe_accepted": unsafe,
            "accepted_denominator": len(accepted),
            "total_calls": sum(row["tool_calls"] for row in rows),
            "measured_p95_seconds": float(np.quantile(
                [row["measured_replay_seconds"] for row in rows], .95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path,
                        default=Path("results/credit_experiment_tables_v1/public"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/credit_experiment_tables_v1/replay"))
    parser.add_argument("--upstream", type=Path, default=Path("vendor/ToolSandbox"))
    parser.add_argument("--tokenizer", type=Path, default=adapter.MODEL)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh replay output directory")
    args.output.mkdir(parents=True)
    protocol = {"status": "frozen before replay", "source_actor_seed": 4103,
        "population": "20 held-out official public ToolSandbox scenarios",
        "full": "all stored tool actions plus stored final answer",
        "scoped": "state-changing stored actions plus stored final answer",
        "matched_budget": "full trace truncated to the scoped tool-call count",
        "state_hash": "all physical databases except the conversation/sandbox database",
        "trace_normalization": "tool, arguments, stable response, stable exception; timing removed",
        "acceptance": "official native similarity >= 0.90",
        "unsafe": "official minefield similarity > 0 among accepted replays"}
    write(args.output / "frozen_protocol.json", protocol)

    u = adapter.upstream_imports(args.upstream); adapter.install_local_clock("2026-10-21T12:00:00")
    scenarios, _ = adapter.local_scenarios(u)
    pool = json.loads((args.public / "memory_pool.json").read_text())
    pool_by_uid = {row["uid"]: row for row in pool}
    physical = json.loads((args.public / "test_physical_sessions.json").read_text())
    logical = json.loads((args.public / "logical_results.json").read_text())
    links = [row for row in logical if row.get("group") == "main" and
             row["method"] == "Full CREDIT" and row["actor_seed"] == 4103]
    by_label = {row["label"]: row for row in physical if row["actor_seed"] == 4103}
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    full_a, full_b, scoped, truncated = [], [], [], []
    detail = []
    for link in links:
        source = by_label[link["physical_row_label"]]
        bank = [pool_by_uid[uid] for uid in source["selected_bank"]]
        scope_actions = [call for call in source["calls"] if call["tool"] in MUTATIONS]
        variants = {
            "Full:A": source["calls"], "Full:B": source["calls"],
            "Scoped:logical": scope_actions,
            "Full:matched": source["calls"][:len(scope_actions)],
        }
        produced = {}
        for name, actions in variants.items():
            produced[name] = replay(source, actions, name, scenarios[source["task"]],
                                    bank, tokenizer, args.output, u)
        full_a.append(produced["Full:A"]); full_b.append(produced["Full:B"])
        scoped.append(produced["Scoped:logical"]); truncated.append(produced["Full:matched"])
        detail.append({"task": source["task"], "source_label": source["label"],
            "source_calls": len(source["calls"]), "scoped_calls": len(scope_actions),
            "rows": {key: {field: value for field, value in row.items() if field in {
                "native_score", "success", "unsafe", "tool_calls", "relevant_state_sha256",
                "normalized_trace_sha256", "measured_replay_seconds"}}
                     for key, row in produced.items()}})
    comparisons = [comparison(full_a, full_b, "A/A repeat"),
                   comparison(scoped, full_a, "Same logical replay"),
                   comparison(scoped, truncated, "Same physical budget")]
    policies = [policy(scoped, "Scoped replay"), policy(truncated, "Full replay")]
    write(args.output / "replay_records.json", detail)
    write(args.output / "replay_summary.json", {"comparisons": comparisons, "policies": policies})
    audit = {"tasks": len(detail), "full_aa_rows": len(full_a) + len(full_b),
             "all_evaluations_present": all(row["verification_coverage"] for rows in
                 (full_a, full_b, scoped, truncated) for row in rows),
             "source_public_audit_sha256": sha256((args.public / "public_audit.json").read_bytes()).hexdigest()}
    if audit["tasks"] != 20 or not audit["all_evaluations_present"]:
        raise AssertionError(audit)
    write(args.output / "audit.json", audit)
    print(json.dumps({"comparisons": comparisons, "policies": policies, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
