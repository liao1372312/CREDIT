"""Replay audit for multi-tool randomized ToolSandbox-M calibration."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import dill

from . import toolsandbox_local_v3 as adapter
from .toolsandbox_m_factorial_training import aggregate, candidates, coefficients, placebo
from .toolsandbox_m_followup import initial_job_v2
from .user_data import digest


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def verify(root: Path) -> dict:
    protocol = read(root / "frozen_protocol.json")
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "source" / name) == expected, "frozen source changed: " + name)
        require(digest(Path(adapter.__file__).with_name(name)) == expected,
                "current source changed: " + name)
    upstream = Path(protocol["upstream_directory"])
    for name, expected in protocol["upstream_source_sha256"].items():
        require(digest(upstream / name) == expected, "upstream source changed: " + name)
    scenarios = dill.loads((root / "scenarios.pkl").read_bytes())
    pool = read(root / "memory_pool.json")
    u = adapter.upstream_imports(upstream); adapter.install_local_clock(protocol["clock"])
    combined = rows(root / "sessions.jsonl")
    replayed = requests = 0
    with tempfile.TemporaryDirectory(prefix="credit-factorial-audit-") as temporary:
        target = Path(temporary)
        for seed in protocol["actor_seeds"]:
            directory = root / f"actor_seed_{seed}"
            manifest = read(directory / "agent_manifest.json")
            tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
                manifest["tokenizer_path"], local_files_only=True)
            generations = rows(directory / "generations.jsonl")
            receipts = {row["generation_uid"]: row for row in generations}
            require(len(receipts) == len(generations) == manifest["live_api_calls"],
                    "generation count differs")
            actor_rows = rows(directory / "sessions.jsonl")
            require(adapter.fingerprint(actor_rows) == adapter.fingerprint(
                [row for row in combined if row["actor_seed"] == seed]),
                "root and actor rows differ")
            for saved in actor_rows:
                selected = candidates(pool, saved["task"], scenarios[saved["task"]])
                nulls = [placebo(note, slot, tokenizer)
                         for slot, note in enumerate(selected)]
                bank = [selected[index] if saved["assignment"][index] == 1 else nulls[index]
                        for index in range(len(selected))]
                require(saved["candidate_uids"] == [note["uid"] for note in selected],
                        "candidate selection differs")
                job = initial_job_v2(saved["task"], scenarios[saved["task"]], bank, u,
                    tokenizer, saved["stage"], saved["method"])
                for turn in saved["generations"]:
                    rendered = tokenizer.apply_chat_template(job["chat"], tokenize=False,
                                                              add_generation_prompt=True)
                    generation = receipts[turn["generation_uid"]]
                    require(sha256(rendered.encode()).hexdigest() == generation["prompt_sha256"],
                            "prompt differs")
                    body = json.dumps({"model": manifest["model"], "messages": job["chat"],
                        "temperature": manifest["temperature"], "seed": seed,
                        "max_tokens": manifest["max_new_tokens"]}, ensure_ascii=False,
                        separators=(",", ":")).encode()
                    require(sha256(body).hexdigest() == generation["request_sha256"],
                            "request differs")
                    adapter.consume(job, generation, u, protocol["maximum_tool_calls_per_session"])
                if not job["done"]:
                    job["termination"] = "turn_budget"
                rebuilt = adapter.finish(job, target, u)
                for field in ("calls", "generations", "ending_state_sha256", "upstream_evaluation",
                              "success", "verification_coverage", "termination"):
                    require(adapter.fingerprint(rebuilt[field]) == adapter.fingerprint(saved[field]),
                            f"replay differs: {seed}/{saved['task']}/{field}")
                replayed += 1
            requests += len(generations)
    expected = coefficients(combined)
    expected["aggregate"] = aggregate(expected["context_coefficients"], pool)
    require(adapter.fingerprint(expected) == adapter.fingerprint(read(root / "effects.json")),
            "effect estimates differ")
    report = {"verified": True, "sessions": len(combined), "replayed_episodes": replayed,
        "live_remote_requests": requests, "credentials_persisted": False,
        "sessions_sha256": digest(root / "sessions.jsonl"),
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
