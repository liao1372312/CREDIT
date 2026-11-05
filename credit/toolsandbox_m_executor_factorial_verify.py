"""Exact replay and integrity audit for the executor component factorial."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from . import toolsandbox_local_v3 as adapter
from . import toolsandbox_m_contract_fast_v3_confirmation as scenario_support
from . import toolsandbox_m_mechanism_blind as source_suite
from .toolsandbox_m_executor_factorial import (
    METHODS, PLAN, consume, method_parts, paired_statistics, select, summarize,
    task_breakdown, task_identity_preflight,
)
from .toolsandbox_m_followup import initial_job_v2
from .user_data import digest


def read(path: Path):
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalized_generations(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        item.pop("local_consume_latency_ms", None)
        output.append(item)
    return output


def _verify_actor(directory: Path, combined: list[dict], trajectories,
                  scenarios, metadata, pool, protocol, target: Path, u):
    manifest = read(directory / "agent_manifest.json")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        manifest["tokenizer_path"], local_files_only=True)
    ledger = read_rows(directory / "generations.jsonl")
    receipts = {row["generation_uid"]: row for row in ledger}
    require(len(receipts) == len(ledger) == manifest["live_api_calls"],
            "generation ledger count differs")
    saved_rows = read_rows(directory / "sessions.jsonl")
    seed = read(directory / "summary.json")["actor_seed"]
    expected = [row for row in combined if row["actor_seed"] == seed]
    require(adapter.fingerprint(saved_rows) == adapter.fingerprint(expected),
            "per-actor and combined rows differ")
    indexed = {(row["method"], row["task"]): row for row in saved_rows}
    replayed = logical = physical = 0
    for trajectory in trajectories:
        for task in trajectory["sessions"]:
            scenario = scenarios[task]
            query = metadata[task]["public_request"]
            for method in METHODS:
                saved = indexed[(method, task)]
                bank = select(method, pool, query, scenario)
                require(saved["selected_bank"] == saved["bank"] ==
                        [note["uid"] for note in bank],
                        "selected memory differs")
                memory, stage = method_parts(method)
                require(saved["memory_arm"] == memory and
                        saved["executor_stage"] == stage,
                        "factor labels differ")
                job = initial_job_v2(
                    task, scenario, bank, u, tokenizer,
                    "executor_factorial_confirmation", method)
                remote_turns = [turn for turn in saved["generations"]
                                if turn.get("generation_uid")]
                for turn in remote_turns:
                    rendered = tokenizer.apply_chat_template(
                        job["chat"], tokenize=False,
                        add_generation_prompt=True)
                    receipt = receipts[turn["generation_uid"]]
                    require(sha256(rendered.encode()).hexdigest() ==
                            receipt["prompt_sha256"],
                            "rendered prompt differs")
                    body = json.dumps({
                        "model": manifest["model"],
                        "messages": job["chat"],
                        "temperature": manifest["temperature"],
                        "seed": seed,
                        "max_tokens": manifest["max_new_tokens"],
                    }, ensure_ascii=False, separators=(",", ":")).encode()
                    require(sha256(body).hexdigest() ==
                            receipt["request_sha256"],
                            "request body differs")
                    consume(method, job, receipt, u, query)
                    logical += 1
                    physical += receipt.get("network_attempts", 1)
                if not job["done"]:
                    job["termination"] = "turn_budget"
                rebuilt = scenario_support.prior.finish_direct(job, target, u)
                rebuilt = source_suite.annotate(rebuilt, metadata)
                for field in (
                        "calls", "ending_state_sha256", "termination",
                        "requested_state_score", "requested_state_success",
                        "process_state_score", "process_state_success",
                        "tool_calls", "mechanism_signature"):
                    require(adapter.fingerprint(rebuilt[field]) ==
                            adapter.fingerprint(saved[field]),
                            "exact replay differs: " + field)
                require(adapter.fingerprint(_normalized_generations(
                            rebuilt["generations"])) ==
                        adapter.fingerprint(_normalized_generations(
                            saved["generations"])),
                        "exact replay differs: generations")
                replayed += 1
    return replayed, logical, physical


def verify(root: Path) -> dict:
    protocol = read(root / "frozen_protocol.json")
    require(protocol["frozen_before_remote_calls"] is True,
            "protocol was not frozen")
    require(protocol["source_closure_complete"] is True,
            "source closure was not frozen")
    require(protocol["tasks_reused_for_component_attribution"] is True,
            "task-reuse scope was not disclosed")
    for item in protocol["preflight"].values():
        require(item["passed"], "stored preflight failed")
    require(read(root / "preflight.json") == protocol["preflight"],
            "preflight file differs")
    source_root = Path(__file__).parent
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "source" / name) == expected,
                "frozen source changed: " + name)
        require(digest(source_root / name) == expected,
                "current source changed: " + name)
    require(digest(PLAN) == protocol["plan_sha256"],
            "registered plan changed")
    upstream_root = Path(protocol["memory_pool_source"])
    require(digest(upstream_root / "frozen_protocol.json") ==
            protocol["upstream_protocol_sha256"],
            "upstream protocol changed")
    upstream_protocol = read(upstream_root / "frozen_protocol.json")
    for name, expected in protocol["upstream_source_sha256"].items():
        require(digest(Path(protocol["upstream_directory"]) / name) == expected,
                "upstream source changed: " + name)

    u = adapter.upstream_imports(Path(protocol["upstream_directory"]))
    adapter.install_local_clock(protocol["clock"])
    scenarios, metadata, trajectories = source_suite.build_scenarios(u)
    require(trajectories == protocol["trajectories"],
            "trajectory definition differs")
    require(task_identity_preflight(metadata, upstream_protocol) ==
            protocol["preflight"]["task_identity"],
            "task identity preflight differs")
    require(source_suite.golden_self_check(u, scenarios, metadata) ==
            protocol["preflight"]["golden_evaluator"],
            "golden evaluator preflight differs")
    require(source_suite.argument_extraction_self_check(metadata) ==
            protocol["preflight"]["argument_extraction"],
            "argument extraction preflight differs")

    source_pool = upstream_root / "memory_pool.json"
    require(digest(source_pool) == protocol["memory_pool_source_sha256"],
            "source memory pool changed")
    pool = read(root / "memory_pool.json")
    require(adapter.fingerprint(pool) == adapter.fingerprint(read(source_pool)),
            "memory pool differs")
    selection = read(root / "frozen_selection.json")
    for task, scenario in scenarios.items():
        query = metadata[task]["public_request"]
        for method in METHODS:
            bank = select(method, pool, query, scenario)
            require(selection[task][method]["selected_uids"] ==
                    [note["uid"] for note in bank],
                    "frozen selection differs")

    combined = read_rows(root / "confirmation_sessions.jsonl")
    expected_count = (len(protocol["actor_seeds"]) * len(metadata) *
                      len(METHODS))
    require(len(combined) == expected_count, "session count differs")
    keys = {(row["actor_seed"], row["task"], row["method"])
            for row in combined}
    require(len(keys) == expected_count,
            "session cells duplicated or missing")
    replayed = logical = physical = 0
    with tempfile.TemporaryDirectory(
            prefix="executor-factorial-replay-") as temporary:
        target = Path(temporary)
        for seed in protocol["actor_seeds"]:
            found, calls, attempts = _verify_actor(
                root / f"executor_factorial_seed_{seed}", combined,
                trajectories, scenarios, metadata, pool, protocol, target, u)
            replayed += found
            logical += calls
            physical += attempts

    summary = read(root / "summary.json")
    require(adapter.fingerprint(summarize(combined)) ==
            adapter.fingerprint(summary["confirmation_table"]),
            "summary differs")
    require(adapter.fingerprint(paired_statistics(combined)) ==
            adapter.fingerprint(summary["paired_statistics"]),
            "statistics differ")
    require(adapter.fingerprint(task_breakdown(combined)) ==
            adapter.fingerprint(summary["task_breakdown"]),
            "breakdown differs")
    run_audit = read(root / "audit.json")
    require(run_audit["sessions"] == run_audit["expected_sessions"] ==
            expected_count, "run audit failed")
    require(run_audit["logical_remote_generations"] == logical,
            "logical generation count differs")
    require(run_audit["physical_http_attempts"] == physical,
            "physical attempt count differs")
    report = {
        "verified": True,
        "confirmation_sessions": len(combined),
        "replayed_episodes": replayed,
        "logical_remote_generations": logical,
        "physical_http_attempts": physical,
        "source_modules_prefrozen": len(protocol["source_sha256"]),
        "confirmation_sha256": digest(
            root / "confirmation_sessions.jsonl"),
        "verifier_sha256": digest(Path(__file__)),
        "credentials_persisted": False,
    }
    adapter.write(root, "execution_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__":
    main()
