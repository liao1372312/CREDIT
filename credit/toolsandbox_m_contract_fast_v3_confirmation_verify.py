"""Exact replay and integrity audit for the argument-aware confirmation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from . import toolsandbox_local_v3 as adapter
from .toolsandbox_m_contract_fast_v3_confirmation import (
    METHODS, _consume, annotate_direct, argument_extraction_self_check,
    build_confirmation_scenarios,
    golden_self_check, holm_primary, paired_statistics, select, summarize,
    task_breakdown,
)
from .toolsandbox_m_contract_fast_development import finish_direct
from .toolsandbox_m_followup import initial_job_v2
from .user_data import digest


def read(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _normalized_generations(rows):
    output = []
    for row in rows:
        item = dict(row)
        item.pop("local_consume_latency_ms", None)
        output.append(item)
    return output


def verify_actor(directory, combined, trajectories, scenarios, metadata, pool,
                 pair_vetoes, u, protocol, target):
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
    replayed = remote_requests = 0
    for trajectory in trajectories:
        for task in trajectory["sessions"]:
            scenario, query = scenarios[task], metadata[task]["public_request"]
            for method in METHODS:
                saved = indexed[(method, task)]
                bank = select(method, pool, task, query, scenario,
                              pair_vetoes, tokenizer)
                require(saved["selected_bank"] == saved["bank"] ==
                        [note["uid"] for note in bank], "selected memory differs")
                job = initial_job_v2(task, scenario, bank, u, tokenizer,
                                     "fast_v3_contract_confirmation", method)
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
                    _consume(job, receipt, u,
                             protocol["maximum_tool_calls_per_session"], query)
                    remote_requests += 1
                if not job["done"]:
                    job["termination"] = "turn_budget"
                rebuilt = finish_direct(job, target, u)
                rebuilt = annotate_direct(rebuilt, metadata)
                for field in ("calls", "ending_state_sha256", "termination",
                              "requested_state_score", "requested_state_success",
                              "process_state_score", "process_state_success",
                              "tool_calls"):
                    require(adapter.fingerprint(rebuilt[field]) ==
                            adapter.fingerprint(saved[field]),
                            "exact replay differs: " + field)
                require(adapter.fingerprint(_normalized_generations(
                            rebuilt["generations"])) ==
                        adapter.fingerprint(_normalized_generations(
                            saved["generations"])),
                        "exact replay differs: generations")
                replayed += 1
    return replayed, remote_requests


def verify(root):
    protocol = read(root / "frozen_protocol.json")
    require(protocol["frozen_before_remote_calls"] is True,
            "protocol was not frozen")
    require(protocol["preflight"]["golden_evaluator"]["passed"],
            "golden preflight failed")
    require(protocol["preflight"]["novelty"]["passed"],
            "novelty preflight failed")
    require(read(root / "preflight.json") == protocol["preflight"],
            "preflight file differs")
    source_root = Path(__file__).parent
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "source" / name) == expected,
                "frozen source changed: " + name)
        require(digest(source_root / name) == expected,
                "current source changed: " + name)
    for source, record in protocol["development_source_sha256"].items():
        require(digest(Path(source) / record["filename"]) == record["sha256"],
                "development episodes changed: " + source)
    upstream = Path(protocol["upstream_directory"])
    for name, expected in protocol["upstream_source_sha256"].items():
        require(digest(upstream / name) == expected,
                "upstream source changed: " + name)

    u = adapter.upstream_imports(upstream)
    adapter.install_local_clock(protocol["clock"])
    scenarios, metadata, trajectories = build_confirmation_scenarios(u)
    require(trajectories == protocol["trajectories"],
            "trajectory definition differs")
    frozen_tasks = {row["name"]: row for row in protocol["tasks"]}
    require(set(frozen_tasks) == set(metadata), "task keys differ")
    for name, item in metadata.items():
        require(frozen_tasks[name]["public_request_sha256"] ==
                sha256(item["public_request"].encode()).hexdigest(),
                "public request differs: " + name)
    require(golden_self_check(u, scenarios, metadata)["passed"],
            "recomputed golden preflight failed")
    require(argument_extraction_self_check(metadata)["passed"],
            "recomputed argument extraction preflight failed")

    pool = read(root / "memory_pool.json")
    source_pool = read(Path(protocol["memory_pool_source"]) / "memory_pool.json")
    require(digest(Path(protocol["memory_pool_source"]) / "memory_pool.json") ==
            protocol["memory_pool_source_sha256"], "source memory pool changed")
    require(adapter.fingerprint(pool) == adapter.fingerprint(source_pool),
            "memory pool differs")
    pair_vetoes = {tuple(pair) for pair in protocol["pair_vetoes"]}
    selection = read(root / "frozen_selection.json")
    first = (root / f"fast_v3_confirmation_seed_{protocol['actor_seeds'][0]}" /
             "agent_manifest.json")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        read(first)["tokenizer_path"], local_files_only=True)
    for task, scenario in scenarios.items():
        bank = select("Contract-Text", pool, task,
                      metadata[task]["public_request"], scenario,
                      pair_vetoes, tokenizer)
        require(selection[task]["selected_uids"] == [note["uid"] for note in bank],
                "frozen contract selection differs")

    combined = read_rows(root / "confirmation_sessions.jsonl")
    expected_count = len(protocol["actor_seeds"]) * len(metadata) * len(METHODS)
    require(len(combined) == expected_count, "session count differs")
    keys = {(row["actor_seed"], row["task"], row["method"]) for row in combined}
    require(len(keys) == expected_count, "session cells duplicated or missing")
    replayed = requests = 0
    with tempfile.TemporaryDirectory(prefix="fast-v3-replay-") as temporary:
        target = Path(temporary)
        for seed in protocol["actor_seeds"]:
            a, b = verify_actor(root / f"fast_v3_confirmation_seed_{seed}",
                combined, trajectories, scenarios, metadata, pool, pair_vetoes,
                u, protocol, target)
            replayed += a; requests += b

    summary = read(root / "summary.json")
    require(adapter.fingerprint(summarize(combined)) ==
            adapter.fingerprint(summary["confirmation_table"]), "summary differs")
    statistics = paired_statistics(combined)
    require(adapter.fingerprint(statistics) ==
            adapter.fingerprint(summary["paired_statistics"]), "statistics differ")
    require(adapter.fingerprint(holm_primary(statistics)) ==
            adapter.fingerprint(summary["holm_primary"]), "Holm report differs")
    require(adapter.fingerprint(task_breakdown(combined)) ==
            adapter.fingerprint(summary["task_breakdown"]), "breakdown differs")
    audit = read(root / "audit.json")
    require(audit["sessions"] == audit["expected_sessions"] == expected_count,
            "run audit failed")
    report = {"verified": True, "confirmation_sessions": len(combined),
        "replayed_episodes": replayed, "live_remote_requests": requests,
        "golden_evaluator_tasks": protocol["preflight"]["golden_evaluator"]["tasks"],
        "exact_prompt_overlap": protocol["preflight"]["novelty"]["exact_prompt_overlap"],
        "exact_allowed_tool_set_overlap":
            protocol["preflight"]["novelty"]["exact_allowed_tool_set_overlap"],
        "confirmation_sha256": digest(root / "confirmation_sessions.jsonl"),
        "credentials_persisted": False}
    adapter.write(root, "execution_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__":
    main()
