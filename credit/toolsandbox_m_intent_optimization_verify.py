"""Exact replay audit for zero-probe intent-aligned ToolSandbox selection."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import dill

from . import toolsandbox_local_v3 as adapter
from .toolsandbox_m_intent_optimization import (
    METHODS, intent_select, make_job, original_request, paired_statistics,
    prompt_variant, select, summarize,
)
from .user_data import digest


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def verify_actor(directory: Path, combined: list[dict], trajectories: list[dict],
                 pool: list[dict], pair_vetoes: set[tuple[str, str]], scenarios,
                 u, protocol: dict, target: Path):
    manifest = read(directory / "agent_manifest.json")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        manifest["tokenizer_path"], local_files_only=True)
    receipt_rows = rows(directory / "generations.jsonl")
    receipts = {row["generation_uid"]: row for row in receipt_rows}
    require(len(receipts) == len(receipt_rows) == manifest["live_api_calls"],
            "generation count differs")
    saved_rows = rows(directory / "sessions.jsonl")
    seed = read(directory / "summary.json")["actor_seed"]
    require(adapter.fingerprint(saved_rows) == adapter.fingerprint(
        [row for row in combined if row["actor_seed"] == seed]),
        "combined rows differ")
    indexed = {(row["method"], row["task"]): row for row in saved_rows}
    replayed = 0
    for trajectory in trajectories:
        for task in trajectory["sessions"]:
            original = original_request(task, scenarios[task], u, tokenizer)
            query = prompt_variant(original)
            for method in METHODS:
                saved = indexed[(method, task)]
                bank = select(method, pool, task, query, scenarios[task], pair_vetoes)
                require(saved["selected_bank"] == saved["bank"] ==
                        [note["uid"] for note in bank], "bank differs")
                require(saved["prompt_variant"] == query, "prompt variant differs")
                job = make_job(task, scenarios[task], bank, query, u, tokenizer, method)
                for turn in saved["generations"]:
                    rendered = tokenizer.apply_chat_template(
                        job["chat"], tokenize=False, add_generation_prompt=True)
                    receipt = receipts[turn["generation_uid"]]
                    require(sha256(rendered.encode()).hexdigest() ==
                            receipt["prompt_sha256"], "prompt differs")
                    body = json.dumps({"model": manifest["model"],
                        "messages": job["chat"],
                        "temperature": manifest["temperature"], "seed": seed,
                        "max_tokens": manifest["max_new_tokens"]},
                        ensure_ascii=False, separators=(",", ":")).encode()
                    require(sha256(body).hexdigest() == receipt["request_sha256"],
                            "request differs")
                    adapter.consume(job, receipt, u,
                                    protocol["maximum_tool_calls_per_session"])
                if not job["done"]:
                    job["termination"] = "turn_budget"
                rebuilt = adapter.finish(job, target, u)
                for field in ("calls", "generations", "ending_state_sha256",
                              "upstream_evaluation", "success",
                              "verification_coverage", "termination"):
                    require(adapter.fingerprint(rebuilt[field]) ==
                            adapter.fingerprint(saved[field]),
                            "replay differs: " + field)
                replayed += 1
    return replayed, len(receipt_rows)


def verify(root: Path) -> dict:
    protocol = read(root / "frozen_protocol.json")
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "source" / name) == expected,
                "frozen source changed: " + name)
        require(digest(Path(adapter.__file__).with_name(name)) == expected,
                "current source changed: " + name)
    upstream = Path(protocol["upstream_directory"])
    for name, expected in protocol["upstream_source_sha256"].items():
        require(digest(upstream / name) == expected,
                "upstream source changed: " + name)
    source = Path(protocol["source_results"])
    require(digest(source / "memory_pool.json") ==
            protocol["source_memory_pool_sha256"], "source memory pool changed")
    require(digest(source / "scenarios.pkl") ==
            protocol["source_scenarios_sha256"], "source scenarios changed")
    scenarios = dill.loads((root / "scenarios.pkl").read_bytes())
    pool = read(root / "memory_pool.json")
    source_protocol = read(source / "frozen_protocol.json")
    pair_vetoes = {tuple(pair) for pair in source_protocol["harmful_pair_uids"]}
    u = adapter.upstream_imports(upstream)
    adapter.install_local_clock(protocol["clock"])
    confirmation = rows(root / "confirmation_sessions.jsonl")
    replayed = requests = 0
    with tempfile.TemporaryDirectory(prefix="credit-intent-audit-") as temporary:
        target = Path(temporary)
        for seed in protocol["actor_seeds"]:
            a, b = verify_actor(root / f"intent_confirmation_seed_{seed}",
                confirmation, protocol["trajectories"], pool, pair_vetoes,
                scenarios, u, protocol, target)
            replayed += a
            requests += b
    summary = read(root / "summary.json")
    require(adapter.fingerprint(summarize(confirmation, list(METHODS))) ==
            adapter.fingerprint(summary["confirmation_table"]),
            "summary differs")
    require(adapter.fingerprint(paired_statistics(confirmation)) ==
            adapter.fingerprint(summary["paired_statistics"]),
            "statistics differ")
    recomputed_selection = {}
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        read(root / f"intent_confirmation_seed_{protocol['actor_seeds'][0]}" /
             "agent_manifest.json")["tokenizer_path"], local_files_only=True)
    for trajectory in protocol["trajectories"]:
        for task in trajectory["sessions"]:
            query = prompt_variant(original_request(task, scenarios[task], u, tokenizer))
            recomputed_selection[task] = [note["uid"] for note in
                                          intent_select(pool, query, scenarios[task])]
            require(recomputed_selection[task] ==
                    summary["intent_selection"][task]["selected_uids"],
                    "intent selection summary differs")
    report = {"verified": True, "confirmation_sessions": len(confirmation),
        "replayed_episodes": replayed, "live_remote_requests": requests,
        "credentials_persisted": False,
        "confirmation_sha256": digest(root / "confirmation_sessions.jsonl"),
        "verifier_source_sha256": digest(Path(__file__))}
    (root / "execution_audit.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__":
    main()
