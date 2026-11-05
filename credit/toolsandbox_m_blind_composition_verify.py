"""Exact replay and integrity audit for the blind ToolSandbox composition run."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from . import toolsandbox_local_v3 as adapter
from .toolsandbox_m_blind_composition import (
    METHODS,
    annotate,
    build_blind_scenarios,
    paired_statistics,
    select,
    summarize,
    task_breakdown,
)
from .toolsandbox_m_followup import initial_job_v2
from .user_data import digest


def read(path: Path):
    return json.loads(path.read_text())


def read_rows(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def verify_actor(directory, combined, trajectories, scenarios, metadata, pool,
                 pair_vetoes, u, protocol, target):
    manifest = read(directory / "agent_manifest.json")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        manifest["tokenizer_path"], local_files_only=True)
    receipts_list = read_rows(directory / "generations.jsonl")
    receipts = {row["generation_uid"]: row for row in receipts_list}
    require(len(receipts) == len(receipts_list) == manifest["live_api_calls"],
            "generation ledger count differs")
    saved_rows = read_rows(directory / "sessions.jsonl")
    seed = read(directory / "summary.json")["actor_seed"]
    expected = [row for row in combined if row["actor_seed"] == seed]
    require(adapter.fingerprint(saved_rows) == adapter.fingerprint(expected),
            "per-actor and combined rows differ")
    indexed = {(row["method"], row["task"]): row for row in saved_rows}
    replayed = 0
    for trajectory in trajectories:
        for task in trajectory["sessions"]:
            scenario, query = scenarios[task], metadata[task]["public_request"]
            for method in METHODS:
                saved = indexed[(method, task)]
                bank = select(method, pool, task, query, scenario, pair_vetoes)
                require(saved["selected_bank"] == saved["bank"] ==
                        [note["uid"] for note in bank], "selected memory differs")
                job = initial_job_v2(task, scenario, bank, u, tokenizer,
                                     "blind_composition", method)
                for turn in saved["generations"]:
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
                    adapter.consume(job, receipt, u,
                                    protocol["maximum_tool_calls_per_session"])
                if not job["done"]:
                    job["termination"] = "turn_budget"
                rebuilt = annotate(adapter.finish(job, target, u), metadata)
                for field in ("calls", "generations", "ending_state_sha256",
                              "upstream_evaluation", "success", "verification_coverage",
                              "termination", "tool_state_score", "tool_state_success",
                              "mutation_score", "mutation_success", "final_text_score",
                              "final_text_success"):
                    require(adapter.fingerprint(rebuilt[field]) ==
                            adapter.fingerprint(saved[field]),
                            "exact replay differs: " + field)
                replayed += 1
    return replayed, len(receipts_list)


def verify(root: Path) -> dict:
    protocol = read(root / "frozen_protocol.json")
    require(protocol["frozen_before_remote_calls"] is True, "protocol was not frozen")
    require(protocol["preflight"]["golden_evaluator"]["passed"],
            "golden evaluator preflight failed")
    require(protocol["preflight"]["novelty"]["passed"], "novelty preflight failed")
    require(read(root / "preflight.json") == protocol["preflight"],
            "preflight file differs from frozen protocol")
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "source" / name) == expected,
                "frozen source changed: " + name)
        require(digest(Path(adapter.__file__).with_name(name)) == expected,
                "current source changed: " + name)
    upstream = Path(protocol["upstream_directory"])
    for name, expected in protocol["upstream_source_sha256"].items():
        require(digest(upstream / name) == expected, "upstream source changed: " + name)
    source = Path(protocol["source_results"])
    require(digest(source / "memory_pool.json") == protocol["source_memory_pool_sha256"],
            "source memory pool changed")
    require(digest(root / "memory_pool.json") == protocol["source_memory_pool_sha256"],
            "copied memory pool differs")

    u = adapter.upstream_imports(upstream)
    adapter.install_local_clock(protocol["clock"])
    scenarios, metadata, trajectories = build_blind_scenarios(u)
    require(trajectories == protocol["trajectories"], "trajectory definition differs")
    frozen_tasks = {row["name"]: row for row in protocol["tasks"]}
    require(set(frozen_tasks) == set(metadata), "task definition keys differ")
    for name, item in metadata.items():
        require(frozen_tasks[name]["public_request_sha256"] ==
                sha256(item["public_request"].encode()).hexdigest(),
                "public request differs: " + name)
    pool = read(root / "memory_pool.json")
    pair_vetoes = {tuple(pair) for pair in protocol["pair_vetoes"]}
    combined = read_rows(root / "confirmation_sessions.jsonl")
    expected_count = len(protocol["actor_seeds"]) * len(metadata) * len(METHODS)
    require(len(combined) == expected_count, "session count differs")
    keys = {(r["actor_seed"], r["task"], r["method"]) for r in combined}
    require(len(keys) == expected_count, "session cells are duplicated or missing")
    require(all(r["verification_coverage"] for r in combined),
            "an episode lacks evaluator coverage")

    replayed = requests = 0
    with tempfile.TemporaryDirectory(prefix="credit-blind-audit-") as temporary:
        for seed in protocol["actor_seeds"]:
            a, b = verify_actor(root / f"blind_composition_seed_{seed}", combined,
                trajectories, scenarios, metadata, pool, pair_vetoes, u, protocol,
                Path(temporary))
            replayed += a; requests += b
    summary = read(root / "summary.json")
    require(adapter.fingerprint(summarize(combined)) ==
            adapter.fingerprint(summary["confirmation_table"]), "summary differs")
    require(adapter.fingerprint(paired_statistics(combined)) ==
            adapter.fingerprint(summary["paired_statistics"]), "statistics differ")
    require(adapter.fingerprint(task_breakdown(combined)) ==
            adapter.fingerprint(summary["task_breakdown"]), "task breakdown differs")
    audit = read(root / "audit.json")
    require(audit["passed"] and audit["sessions"] == expected_count, "run audit failed")
    require(digest(root / "confirmation_sessions.jsonl") == audit["confirmation_sha256"],
            "combined result hash differs")
    report = {"verified": True, "confirmation_sessions": len(combined),
        "replayed_episodes": replayed, "live_remote_requests": requests,
        "golden_evaluator_tasks": protocol["preflight"]["golden_evaluator"]["tasks"],
        "exact_prompt_overlap": protocol["preflight"]["novelty"]["exact_prompt_overlap"],
        "exact_tool_allow_list_overlap": protocol["preflight"]["novelty"]["exact_tool_allow_list_overlap"],
        "confirmation_sha256": digest(root / "confirmation_sessions.jsonl"),
        "verifier_source_sha256": digest(Path(__file__)), "credentials_persisted": False}
    (root / "execution_audit.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__":
    main()
