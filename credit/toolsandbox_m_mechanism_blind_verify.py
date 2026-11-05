"""Exact replay and integrity audit for the mechanism-blind confirmation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from . import toolsandbox_local_v3 as adapter
from . import toolsandbox_m_contract_fast_v3 as v3
from . import toolsandbox_m_contract_fast_v3_confirmation as scenario_support
from .toolsandbox_m_contract_confirmation import novelty_audit
from .toolsandbox_m_followup import initial_job_v2
from .toolsandbox_m_mechanism_blind import (
    METHODS, PLAN, annotate, argument_extraction_self_check, build_scenarios,
    golden_self_check, mechanism_novelty_audit, paired_statistics, select,
    summarize, task_breakdown,
)
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


def verify_actor(directory: Path, combined: list[dict], trajectories,
                 scenarios, metadata, pool, pair_vetoes, u, protocol,
                 target: Path) -> tuple[int, int, int]:
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
            scenario, query = scenarios[task], metadata[task]["public_request"]
            for method in METHODS:
                saved = indexed[(method, task)]
                bank = select(method, pool, task, query, scenario,
                              pair_vetoes, tokenizer)
                require(saved["selected_bank"] == saved["bank"] ==
                        [note["uid"] for note in bank], "selected memory differs")
                require(saved["selected_memory_schemas"] ==
                        [note.get("memory_schema") for note in bank],
                        "selected memory schemas differ")
                job = initial_job_v2(task, scenario, bank, u, tokenizer,
                                     "mechanism_blind_confirmation", method)
                remote_turns = [turn for turn in saved["generations"]
                                if turn.get("generation_uid")]
                for turn in remote_turns:
                    rendered = tokenizer.apply_chat_template(
                        job["chat"], tokenize=False, add_generation_prompt=True)
                    receipt = receipts[turn["generation_uid"]]
                    require(sha256(rendered.encode()).hexdigest() ==
                            receipt["prompt_sha256"], "rendered prompt differs")
                    body = json.dumps({"model": manifest["model"],
                        "messages": job["chat"], "temperature": manifest["temperature"],
                        "seed": seed, "max_tokens": manifest["max_new_tokens"]},
                        ensure_ascii=False, separators=(",", ":")).encode()
                    require(sha256(body).hexdigest() == receipt["request_sha256"],
                            "request body differs")
                    v3.consume_fast(job, receipt, u,
                                    protocol["maximum_tool_calls_per_session"], query)
                    logical += 1
                    physical += receipt.get("network_attempts", 1)
                if not job["done"]:
                    job["termination"] = "turn_budget"
                rebuilt = scenario_support.prior.finish_direct(job, target, u)
                rebuilt = annotate(rebuilt, metadata)
                for field in ("calls", "ending_state_sha256", "termination",
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
    for key in ("golden_evaluator", "argument_extraction",
                "exact_prompt_novelty", "mechanism_novelty"):
        require(protocol["preflight"][key]["passed"], key + " preflight failed")
    require(read(root / "preflight.json") == protocol["preflight"],
            "preflight file differs")
    source_root = Path(__file__).parent
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "source" / name) == expected,
                "frozen source changed: " + name)
        require(digest(source_root / name) == expected,
                "current source changed: " + name)
    require(digest(PLAN) == protocol["plan_sha256"], "registered plan changed")
    upstream = Path(protocol["upstream_directory"])
    for name, expected in protocol["upstream_source_sha256"].items():
        require(digest(upstream / name) == expected,
                "upstream source changed: " + name)

    u = adapter.upstream_imports(upstream)
    adapter.install_local_clock(protocol["clock"])
    scenarios, metadata, trajectories = build_scenarios(u)
    require(trajectories == protocol["trajectories"],
            "trajectory definition differs")
    frozen_tasks = {row["name"]: row for row in protocol["tasks"]}
    require(set(frozen_tasks) == set(metadata), "task keys differ")
    for name, item in metadata.items():
        frozen = frozen_tasks[name]
        require(frozen["public_request_sha256"] ==
                sha256(item["public_request"].encode()).hexdigest(),
                "public request differs: " + name)
        require(frozen["required_mutations"] == item["required_mutations"],
                "mechanism differs: " + name)
        require(frozen["mechanism_signature"] == item["mechanism_signature"],
                "mechanism hash differs: " + name)
    require(golden_self_check(u, scenarios, metadata) ==
            protocol["preflight"]["golden_evaluator"],
            "recomputed golden preflight differs")
    require(argument_extraction_self_check(metadata) ==
            protocol["preflight"]["argument_extraction"],
            "recomputed argument extraction differs")
    require(novelty_audit(u, scenarios, metadata,
            [Path(path) for path in protocol["novelty_sources"]]) ==
            protocol["preflight"]["exact_prompt_novelty"],
            "recomputed prompt novelty differs")
    require(mechanism_novelty_audit(u, metadata) ==
            protocol["preflight"]["mechanism_novelty"],
            "recomputed mechanism novelty differs")

    source_pool = Path(protocol["memory_pool_source"]) / "memory_pool.json"
    require(digest(source_pool) == protocol["memory_pool_source_sha256"],
            "source memory pool changed")
    pool = read(root / "memory_pool.json")
    require(adapter.fingerprint(pool) == adapter.fingerprint(read(source_pool)),
            "memory pool differs")
    pair_vetoes = {tuple(pair) for pair in protocol["pair_vetoes"]}
    selection = read(root / "frozen_selection.json")
    first = (root / f"mechanism_blind_seed_{protocol['actor_seeds'][0]}" /
             "agent_manifest.json")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        read(first)["tokenizer_path"], local_files_only=True)
    for task, scenario in scenarios.items():
        for method in METHODS:
            bank = select(method, pool, task, metadata[task]["public_request"],
                          scenario, pair_vetoes, tokenizer)
            require(selection[task][method]["selected_uids"] ==
                    [note["uid"] for note in bank], "frozen selection differs")
            require(selection[task][method]["tokens"] ==
                    sum(note["tokens"] for note in bank),
                    "frozen selection token count differs")

    combined = read_rows(root / "confirmation_sessions.jsonl")
    expected_count = (len(protocol["actor_seeds"]) * len(metadata) * len(METHODS))
    require(len(combined) == expected_count, "session count differs")
    keys = {(row["actor_seed"], row["task"], row["method"]) for row in combined}
    require(len(keys) == expected_count, "session cells duplicated or missing")
    replayed = logical = physical = 0
    with tempfile.TemporaryDirectory(prefix="mechanism-blind-replay-") as temporary:
        target = Path(temporary)
        for seed in protocol["actor_seeds"]:
            found, calls, attempts = verify_actor(
                root / f"mechanism_blind_seed_{seed}", combined,
                trajectories, scenarios, metadata, pool, pair_vetoes, u,
                protocol, target)
            replayed += found
            logical += calls
            physical += attempts

    summary = read(root / "summary.json")
    require(adapter.fingerprint(summarize(combined)) ==
            adapter.fingerprint(summary["confirmation_table"]), "summary differs")
    require(adapter.fingerprint(paired_statistics(combined)) ==
            adapter.fingerprint(summary["paired_statistics"]), "statistics differ")
    require(adapter.fingerprint(task_breakdown(combined)) ==
            adapter.fingerprint(summary["task_breakdown"]), "breakdown differs")
    run_audit = read(root / "audit.json")
    require(run_audit["sessions"] == run_audit["expected_sessions"] == expected_count,
            "run audit failed")
    require(run_audit["logical_remote_generations"] == logical,
            "logical generation count differs")
    require(run_audit["physical_http_attempts"] == physical,
            "physical attempt count differs")
    report = {"verified": True, "confirmation_sessions": len(combined),
        "replayed_episodes": replayed, "logical_remote_generations": logical,
        "physical_http_attempts": physical,
        "golden_evaluator_tasks": protocol["preflight"]["golden_evaluator"]["tasks"],
        "exact_prompt_overlap": protocol["preflight"]["exact_prompt_novelty"]["exact_prompt_overlap"],
        "exact_mechanism_signature_overlap": protocol["preflight"]["mechanism_novelty"]["exact_signature_overlap"],
        "source_modules_prefrozen": len(protocol["source_sha256"]),
        "confirmation_sha256": digest(root / "confirmation_sessions.jsonl"),
        "verifier_sha256": digest(Path(__file__)), "credentials_persisted": False}
    adapter.write(root, "execution_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__":
    main()
