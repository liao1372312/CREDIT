"""Independent audit for the supplied-dataset textual-memory factorial."""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .dataset_text_memory_interactions import (coalition_key, equalize, matched_chats,
    select_tasks, verify as verify_answer)
from .emp_binding_agent import parse_request
from .semantic_text_interactions import cross_validated_errors, factorial_summary
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
        current = Path(__file__).with_name(name)
        require(digest(directory / "source" / name) == expected, "frozen source changed: " + name)
        require(digest(current) == expected, "current source changed: " + name)
    data = Path(protocol["data_directory"])
    for name, expected in protocol["data_sha256"].items():
        require(digest(data / name) == expected, "source data changed: " + name)
    manifest = read(directory / "agent_manifest.json")
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer_path"], local_files_only=True)
    raw, inventory = select_tasks(data, protocol["selection_seed"], protocol["development_tasks"],
                                  protocol["confirmation_tasks"])
    tasks = [equalize(task, tokenizer) for task in raw]
    require(tasks == read(directory / "tasks.json"), "frozen task construction differs")
    require(inventory == read(directory / "selection_inventory.json"), "selection inventory differs")
    coalitions = [tuple(c) for size in range(4) for c in combinations("ABC", size)]
    jobs = []
    for task in tasks:
        chats = matched_chats(task, coalitions, tokenizer)
        jobs.extend((task, coalition, chats[coalition]) for coalition in coalitions)
    episodes = rows(directory / "episodes.jsonl")
    generations = rows(directory / "generations.jsonl")
    require(len(jobs) == len(episodes) == len(generations), "episode/generation cardinality differs")
    require(manifest["generation_records"] == len(generations)
            and manifest["unique_generation_contexts"] == len(generations),
            "generation manifest differs")
    rebuilt = []
    for (task, coalition, messages), saved, generation in zip(jobs, episodes, generations):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_hash = sha256(rendered.encode()).hexdigest()
        require(prompt_hash == generation["prompt_sha256"] == saved["prompt_sha256"],
                "prompt digest differs")
        require(rendered == generation["messages_rendered"], "rendered prompt text differs")
        require(len(tokenizer.encode(rendered, add_special_tokens=False))
                == generation["rendered_input_tokens"] == saved["prompt_tokens"],
                "rendered prompt token count differs")
        require("positive_endpoint_id" not in rendered, "gold-label field leaked into actor prompt")
        action, parse_error = parse_request(generation["agent_output"])
        outcome = verify_answer(task, action, parse_error)
        row = {"task_id": task["task_id"], "split": task["split"],
            "coalition": coalition_key(coalition), "exposed_roles": list(coalition),
            "prompt_tokens": saved["prompt_tokens"], "prompt_sha256": prompt_hash,
            "output_sha256": sha256(generation["agent_output"].encode()).hexdigest(),
            "remote_latency_ms": generation["remote_latency_ms"], "action": action, **outcome}
        require(row == saved, "episode score or annotation differs")
        rebuilt.append(row)
    development = factorial_summary(rebuilt, "development")
    confirmation = factorial_summary(rebuilt, "confirmation")
    cv_summary, predictions = cross_validated_errors(rebuilt, "confirmation")
    expected = {"development": development, "confirmation": confirmation,
        "confirmation_cross_task_prediction": cv_summary, "episodes": len(rebuilt),
        "actor_forward_contexts": manifest["live_api_calls"],
        "mean_remote_latency_ms": float(np.mean([row["remote_latency_ms"] for row in rebuilt]))}
    require(expected == read(directory / "summary.json"), "summary differs")
    require(predictions == rows(directory / "predictions.jsonl"), "cross-validation rows differ")
    report = {"verified": True, "tasks": len(tasks), "episodes": len(rebuilt),
        "confirmation_tasks": protocol["confirmation_tasks"],
        "token_matched_tasks": len(tasks), "weak_labels_disclosed": True,
        "source_data_hashes_match": True, "live_remote_requests": manifest["live_api_calls"],
        "credentials_persisted": False, "episodes_sha256": digest(directory / "episodes.jsonl"),
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
