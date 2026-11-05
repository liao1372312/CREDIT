"""Independent replay audit for the multi-actor ToolSandbox-M experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import dill

from . import toolsandbox_local_v3 as adapter
from .toolsandbox_m import METHODS, continuous_score, select_bank
from .toolsandbox_m_multi_actor import summarize
from .user_data import digest


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(directory: Path) -> dict:
    protocol = read(directory / "frozen_protocol.json")
    for name, expected in protocol["source_sha256"].items():
        require(digest(directory / "source" / name) == expected, "frozen source changed: " + name)
        require(digest(Path(adapter.__file__).with_name(name)) == expected, "current source changed: " + name)
    upstream = Path(protocol["upstream_directory"])
    for name, expected in protocol["upstream_source_sha256"].items():
        require(digest(upstream / name) == expected, "upstream source changed: " + name)
    require(digest(directory / "scenarios.pkl") == read(directory / "scenario_freeze.json")["sha256"],
            "scenario snapshot changed")
    scenarios = dill.loads((directory / "scenarios.pkl").read_bytes())
    u = adapter.upstream_imports(upstream)
    adapter.install_local_clock(protocol["clock"])
    combined = rows(directory / "sessions.jsonl")
    replayed = generation_contexts = remote_requests = 0
    verification_errors = Counter()
    actor_summaries = []
    for actor_seed in protocol["actor_seeds"]:
        actor = directory / f"actor_seed_{actor_seed}"
        manifest = read(actor / "agent_manifest.json")
        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
            manifest["tokenizer_path"], local_files_only=True)
        generations = rows(actor / "generations.jsonl")
        cache = {row["generation_uid"]: row for row in generations}
        require(len(cache) == len(generations) == manifest["generation_records"]
                == manifest["live_api_calls"], "generation record cardinality differs")
        require(len({row["prompt_sha256"] for row in generations})
                == manifest["unique_generation_contexts"], "unique prompt count differs")
        require(not manifest["exact_prompt_cache"], "latency run unexpectedly used a prompt cache")
        for generation in generations:
            require(sha256(generation["messages_rendered"].encode()).hexdigest()
                    == generation["prompt_sha256"],
                    "rendered prompt digest differs")
            require(len(tokenizer.encode(generation["messages_rendered"], add_special_tokens=False))
                    == generation["rendered_input_tokens"], "rendered token count differs")
            require(generation["remote_latency_ms"] > 0 and generation["network_attempts"] in (1, 2, 3),
                    "invalid transport timing metadata")
            require(bool(generation["provider_model"]), "provider model snapshot missing")
        episodes = rows(actor / "episodes.jsonl")
        sessions = rows(actor / "sessions.jsonl")
        actor_combined = [row for row in combined if row["actor_seed"] == actor_seed]
        require(adapter.fingerprint(sessions) == adapter.fingerprint(actor_combined),
                "root and actor session rows differ")
        require(len(sessions) == len(METHODS) * len(protocol["future"]), "test session count differs")
        initial = read(actor / "initial_memories.json")
        calibration = read(actor / "causal_calibration.json")
        causal_scores = {row["uid"]: row["paired_effect"] for row in calibration}
        annotated = {(row["method"], row["task"]): row for row in sessions}
        require(len(annotated) == len(sessions), "duplicate method/task/actor session")
        for method in METHODS:
            for trajectory in protocol["trajectories"]:
                pool = [dict(note) for note in initial]
                for session_index, task in enumerate(trajectory["sessions"]):
                    row = annotated[(method, task)]
                    require(row["session_index"] == session_index
                            and row["trajectory"] == trajectory["trajectory"],
                            "trajectory annotation differs")
                    require(row["available_memory_before"] == [note["uid"] for note in pool],
                            "memory chronology differs")
                    bank = select_bank(method, pool, task, scenarios[task], causal_scores)
                    require(row["selected_bank"] == [note["uid"] for note in bank] == row["bank"],
                            "memory selection differs")
                    note = adapter.memory_from(row, tokenizer)
                    inserted = None
                    if note is not None and not any(old["uid"] == note["uid"] for old in pool):
                        pool.append(note); inserted = note["uid"]
                    require(row["inserted_after_scoring"] == inserted, "post-score insertion differs")
        all_memory = {note["uid"]: note for note in read(actor / "all_memories.json")}
        with tempfile.TemporaryDirectory(prefix="credit-multi-actor-replay-") as temp:
            target = Path(temp)
            for saved in episodes:
                bank = [all_memory[uid] for uid in saved["bank"]]
                job = adapter.initial_job(saved["task"], scenarios[saved["task"]], bank, u,
                    tokenizer, saved["stage"], saved["method"])
                require(job["initial_state_sha256"] == saved["initial_state_sha256"],
                        "initial state differs")
                for turn in saved["generations"]:
                    rendered = tokenizer.apply_chat_template(job["chat"], tokenize=False,
                                                              add_generation_prompt=True)
                    key = sha256(rendered.encode()).hexdigest()
                    require(turn["generation_uid"] in cache, "generation receipt missing")
                    generation = cache[turn["generation_uid"]]
                    require(key == turn["prompt_sha256"] == generation["prompt_sha256"],
                            "public prompt differs")
                    body = json.dumps({"model": manifest["model"], "messages": job["chat"],
                        "temperature": manifest["temperature"], "seed": actor_seed,
                        "max_tokens": manifest["max_new_tokens"]}, ensure_ascii=False,
                        separators=(",", ":")).encode()
                    require(sha256(body).hexdigest() == generation["request_sha256"],
                            "remote request body differs")
                    adapter.consume(job, generation, u, protocol["maximum_tool_calls_per_session"])
                if not job["done"]:
                    job["termination"] = "turn_budget"
                rebuilt = adapter.finish(job, target, u)
                for field in ("calls", "generations", "ending_state_sha256", "upstream_evaluation",
                              "success", "verification_coverage", "verification_error", "termination"):
                    require(adapter.fingerprint(rebuilt[field]) == adapter.fingerprint(saved[field]),
                            f"replay differs: {actor_seed}/{saved['task']}/{field}")
                replayed += 1
                if saved["verification_error"]:
                    verification_errors[saved["verification_error"]] += 1
        actor_summaries.append(read(actor / "summary.json"))
        generation_contexts += len({row["prompt_sha256"] for row in generations})
        remote_requests += manifest["live_api_calls"]
    expected_summary = summarize(combined, actor_summaries)
    require(adapter.fingerprint(expected_summary) == adapter.fingerprint(read(directory / "summary.json")),
            "aggregate summary or latency ranking differs")
    report = {"verified": True, "actors": len(protocol["actor_seeds"]),
        "replayed_episodes": replayed, "test_sessions": len(combined),
        "trajectories_per_actor": len(protocol["trajectories"]),
        "unique_generation_contexts": generation_contexts, "live_remote_requests": remote_requests,
        "transport_latency_rows": sum(len(row["generations"]) for row in combined),
        "verification_errors": dict(verification_errors), "credentials_persisted": False,
        "sessions_sha256": digest(directory / "sessions.jsonl"),
        "verifier_source_sha256": digest(Path(__file__))}
    (directory / "execution_audit.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__":
    main()
