"""Independent reconstruction checks for the ToolSandbox-M follow-up."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import dill

from . import toolsandbox_local_v3 as adapter
from .toolsandbox_m_followup import (
    CONFIRM_METHODS, DEV_VARIANTS, initial_job_v2, memory_from_v2,
    paired_statistics, select_bank_v2, summarize,
)
from .user_data import digest


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_actor(directory: Path, scenarios, u, protocol: dict, initial: list[dict],
                  causal_model: dict, trajectories: list[dict], methods: list[str],
                  variants: dict[str, str], combined: list[dict], temp: Path) -> tuple[int, int]:
    manifest = read(directory / "agent_manifest.json")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        manifest["tokenizer_path"], local_files_only=True)
    generations = rows(directory / "generations.jsonl")
    receipts = {row["generation_uid"]: row for row in generations}
    require(len(receipts) == len(generations) == manifest["live_api_calls"],
            "generation cardinality differs: " + directory.name)
    for generation in generations:
        require(sha256(generation["messages_rendered"].encode()).hexdigest() ==
                generation["prompt_sha256"], "rendered prompt digest differs")
        require(generation["remote_latency_ms"] > 0, "invalid transport latency")
    saved_sessions = rows(directory / "sessions.jsonl")
    actor_seed = read(directory / "summary.json")["actor_seed"]
    actor_combined = [row for row in combined if row["actor_seed"] == actor_seed]
    require(adapter.fingerprint(saved_sessions) == adapter.fingerprint(actor_combined),
            "root and actor sessions differ")
    annotated = {(row["method"], row["task"]): row for row in saved_sessions}
    memories = {note["uid"]: note for note in read(directory / "all_memories.json")}
    for method in methods:
        for trajectory in trajectories:
            pool = [dict(note) for note in initial]
            for session_index, task in enumerate(trajectory["sessions"]):
                row = annotated[(method, task)]
                require(row["available_memory_before"] == [note["uid"] for note in pool],
                        "memory chronology differs")
                selector = "CREDIT" if method.startswith("CREDIT-") else method
                bank = select_bank_v2(selector, pool, task, scenarios[task], causal_model,
                    variant=variants.get(method, "coverage2"))
                require(row["selected_bank"] == [note["uid"] for note in bank] == row["bank"],
                        "bank selection differs")
                note = memory_from_v2(row, tokenizer)
                inserted = None
                if note is not None and not any(old["uid"] == note["uid"] for old in pool):
                    pool.append(note); inserted = note["uid"]
                require(row["inserted_after_scoring"] == inserted,
                        "post-score insertion differs")
    replayed = 0
    for saved in rows(directory / "episodes.jsonl"):
        bank = [memories[uid] for uid in saved["bank"]]
        job = initial_job_v2(saved["task"], scenarios[saved["task"]], bank, u,
            tokenizer, saved["stage"], saved["method"])
        require(job["initial_state_sha256"] == saved["initial_state_sha256"],
                "initial state differs")
        for turn in saved["generations"]:
            rendered = tokenizer.apply_chat_template(job["chat"], tokenize=False,
                                                      add_generation_prompt=True)
            generation = receipts[turn["generation_uid"]]
            require(sha256(rendered.encode()).hexdigest() == turn["prompt_sha256"] ==
                    generation["prompt_sha256"], "prompt reconstruction differs")
            body = json.dumps({"model": manifest["model"], "messages": job["chat"],
                "temperature": manifest["temperature"], "seed": actor_seed,
                "max_tokens": manifest["max_new_tokens"]}, ensure_ascii=False,
                separators=(",", ":")).encode()
            require(sha256(body).hexdigest() == generation["request_sha256"],
                    "request body differs")
            adapter.consume(job, generation, u, protocol["maximum_tool_calls_per_session"])
        if not job["done"]:
            job["termination"] = "turn_budget"
        rebuilt = adapter.finish(job, temp, u)
        for field in ("calls", "generations", "ending_state_sha256", "upstream_evaluation",
                      "success", "verification_coverage", "verification_error", "termination"):
            require(adapter.fingerprint(rebuilt[field]) == adapter.fingerprint(saved[field]),
                    f"episode replay differs: {directory.name}/{saved['task']}/{field}")
        replayed += 1
    return replayed, len(generations)


def verify(root: Path) -> dict:
    protocol = read(root / "frozen_protocol.json")
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "source" / name) == expected, "frozen source changed: " + name)
        require(digest(Path(adapter.__file__).with_name(name)) == expected,
                "current source changed: " + name)
    upstream = Path(protocol["upstream_directory"])
    for name, expected in protocol["upstream_source_sha256"].items():
        require(digest(upstream / name) == expected, "upstream source changed: " + name)
    require(digest(root / "scenarios.pkl") == read(root / "scenario_freeze.json")["sha256"],
            "scenario snapshot changed")
    scenarios = dill.loads((root / "scenarios.pkl").read_bytes())
    u = adapter.upstream_imports(upstream)
    adapter.install_local_clock(protocol["clock"])
    initial = read(root / "memory_pool.json")
    causal_model = read(root / "causal_model.json")
    development = rows(root / "development_sessions.jsonl")
    confirmation = rows(root / "confirmation_sessions.jsonl")
    chosen = read(root / "development_selection.json")["chosen_variant"]
    replayed = requests = 0
    verification_errors = Counter()
    with tempfile.TemporaryDirectory(prefix="credit-followup-audit-") as temporary:
        temp = Path(temporary)
        for seed in protocol["development_seeds"]:
            n, calls = _verify_actor(root / f"development_seed_{seed}", scenarios, u, protocol,
                initial, causal_model, protocol["development"],
                ["CREDIT-" + variant for variant in DEV_VARIANTS],
                {"CREDIT-" + variant: variant for variant in DEV_VARIANTS}, development, temp)
            replayed += n; requests += calls
        for seed in protocol["confirmation_seeds"]:
            n, calls = _verify_actor(root / f"confirmation_seed_{seed}", scenarios, u, protocol,
                initial, causal_model, protocol["confirmation"], list(CONFIRM_METHODS),
                {"CREDIT": chosen}, confirmation, temp)
            replayed += n; requests += calls
    summary = read(root / "summary.json")
    training = read(root / "training_summary.json")
    expected_table = summarize(confirmation, list(CONFIRM_METHODS),
                               calibration_ms=training["calibration_remote_latency_ms"])
    require(adapter.fingerprint(expected_table) == adapter.fingerprint(summary["confirmation_table"]),
            "confirmation summary differs")
    require(adapter.fingerprint(paired_statistics(confirmation)) ==
            adapter.fingerprint(summary["paired_statistics"]), "paired statistics differ")
    for row in confirmation:
        if row["verification_error"]:
            verification_errors[row["verification_error"]] += 1
    report = {
        "verified": True, "development_sessions": len(development),
        "confirmation_sessions": len(confirmation), "replayed_test_episodes": replayed,
        "test_remote_requests": requests,
        "calibration_pairs": training["calibration_pairs"],
        "verification_errors": dict(verification_errors),
        "credentials_persisted": False,
        "confirmation_sha256": digest(root / "confirmation_sessions.jsonl"),
        "verifier_source_sha256": digest(Path(__file__)),
    }
    write = root / "execution_audit.json"
    write.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__":
    main()
