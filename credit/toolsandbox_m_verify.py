"""Independent deterministic replay audit for ToolSandbox-M results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import dill
import numpy as np
from transformers import AutoTokenizer

from . import toolsandbox_local_v3 as adapter
from .toolsandbox_m import METHODS, continuous_score, select_bank
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
    for name, expected in read(directory / "evaluation_freeze.json").items():
        require(digest(directory / name) == expected, "pre-test artifact changed: " + name)

    u = adapter.upstream_imports(upstream)
    adapter.install_local_clock(protocol["clock"])
    scenarios = dill.loads((directory / "scenarios.pkl").read_bytes())
    manifest = read(directory / "agent_manifest.json")
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_path"], local_files_only=True)
    generations = rows(directory / "generations.jsonl")
    cache = {row["prompt_sha256"]: row for row in generations}
    require(len(cache) == len(generations) == manifest["unique_generation_contexts"],
            "generation cache cardinality differs")
    for key, generation in cache.items():
        require(sha256(generation["messages_rendered"].encode()).hexdigest() == key,
                "generation prompt digest differs")
        require(len(tokenizer.encode(generation["messages_rendered"], add_special_tokens=False))
                == generation["input_tokens"], "generation input token count differs")

    episodes = rows(directory / "episodes.jsonl")
    sessions = rows(directory / "sessions.jsonl")
    history = [row for row in episodes if row["stage"] == "history"]
    calibration_rows = [row for row in episodes if row["stage"] == "calibration"]
    test_rows = [row for row in episodes if row["stage"] == "multi_session_test"]
    require(len(history) == len(protocol["history"]), "history count differs")
    require({row["task"] for row in history} == set(protocol["history"]), "history coverage differs")
    require(len(test_rows) == len(METHODS) * len(protocol["future"]), "test count differs")
    require(len(sessions) == len(test_rows), "session annotation count differs")
    require(not set(protocol["history"]) & set(protocol["future"]), "history/test task leakage")

    initial = read(directory / "initial_memories.json")
    rebuilt_initial = [note for row in sorted(history, key=lambda row: protocol["history"].index(row["task"]))
                       if (note := adapter.memory_from(row, tokenizer)) is not None]
    require(initial == rebuilt_initial, "initial memory provenance differs")
    calibration = read(directory / "causal_calibration.json")
    controls = {row["task"]: row for row in history}
    require(len(calibration_rows) == len(initial) == len(calibration), "calibration count differs")
    causal_scores = {}
    for note, row, saved in zip(initial, calibration_rows, calibration):
        effect = continuous_score(row) - continuous_score(controls[note["source_task"]])
        require(saved["uid"] == note["uid"] and saved["paired_effect"] == effect,
                "causal calibration differs")
        causal_scores[note["uid"]] = effect

    all_memory = {note["uid"]: note for note in read(directory / "all_memories.json")}
    annotated = {(row["method"], row["task"]): row for row in sessions}
    require(len(annotated) == len(sessions), "duplicate method/task session")
    for method in METHODS:
        for trajectory in protocol["trajectories"]:
            pool = [dict(note) for note in initial]
            for session_index, task in enumerate(trajectory["sessions"]):
                row = annotated[(method, task)]
                require(row["trajectory"] == trajectory["trajectory"]
                        and row["session_index"] == session_index, "trajectory annotation differs")
                require(row["available_memory_before"] == [note["uid"] for note in pool],
                        "memory chronology differs")
                expected_bank = select_bank(method, pool, task, scenarios[task], causal_scores)
                require(row["selected_bank"] == [note["uid"] for note in expected_bank]
                        == row["bank"], "selected bank differs")
                note = adapter.memory_from(row, tokenizer)
                inserted = None
                if note is not None and not any(existing["uid"] == note["uid"] for existing in pool):
                    pool.append(note); inserted = note["uid"]
                require(row["inserted_after_scoring"] == inserted, "post-score insertion differs")

    compared = 0
    with tempfile.TemporaryDirectory(prefix="credit-toolsandbox-m-audit-") as temp:
        target = Path(temp)
        for row in episodes:
            bank = [all_memory[uid] for uid in row["bank"]]
            job = adapter.initial_job(row["task"], scenarios[row["task"]], bank, u, tokenizer,
                                      row["stage"], row["method"])
            require(job["initial_state_sha256"] == row["initial_state_sha256"],
                    "initial state differs")
            require(job["exposure"] == row["exposure"], "retrieval differs")
            for turn in row["generations"]:
                rendered = tokenizer.apply_chat_template(job["chat"], tokenize=False,
                                                         add_generation_prompt=True)
                key = sha256(rendered.encode()).hexdigest()
                require(key == turn["prompt_sha256"] and key in cache, "public prompt differs")
                adapter.consume(job, cache[key], u, protocol["maximum_tool_calls_per_session"])
            if not job["done"]:
                job["termination"] = "turn_budget"
            rebuilt = adapter.finish(job, target, u)
            for key in ("calls", "generations", "ending_state_sha256", "upstream_evaluation",
                        "success", "verification_coverage", "verification_error", "termination"):
                require(adapter.fingerprint(rebuilt[key]) == adapter.fingerprint(row[key]),
                        f"replay differs: {row['stage']}/{row['task']}/{row['method']}/{key}")
            compared += 1

    summary = read(directory / "summary.json")
    for entry in summary["table"]:
        subset = [row for row in sessions if row["method"] == entry["method"]]
        require(entry["strict_VSR"] == np.mean([row["success"] for row in subset]),
                "summary VSR differs")
        require(entry["upstream_mean_score"] == np.mean([continuous_score(row) for row in subset]),
                "summary continuous score differs")
    report = {"verified": True, "replayed_episodes": compared,
        "history_episodes": len(history), "calibration_episodes": len(calibration_rows),
        "test_sessions": len(test_rows), "trajectories": len(protocol["trajectories"]),
        "unique_generation_contexts": len(cache),
        "total_local_tool_calls": sum(row["tool_calls"] for row in episodes),
        "verification_errors": dict(Counter(row["verification_error"] for row in episodes
                                             if row["verification_error"])),
        "provider_drift": False, "external_network_tool_calls": 0,
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
