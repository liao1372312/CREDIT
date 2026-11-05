"""Replay audit for the development-memory streaming optimization."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import dill

from . import toolsandbox_local_v3 as adapter
from .toolsandbox_m_followup import initial_job_v2
from .toolsandbox_m_streaming_optimization import (
    CONFIRM_METHODS, DEV_VARIANTS, harmful_pairs, leave_family_out,
    paired_statistics, select_bank, summarize,
)
from .user_data import digest


def read(path: Path): return json.loads(path.read_text())
def rows(path: Path): return [json.loads(line) for line in path.read_text().splitlines() if line]


def require(value: bool, message: str) -> None:
    if not value: raise ValueError(message)


def verify_actor(directory: Path, combined: list[dict], trajectories: list[dict],
                 methods: list[str], variants: dict[str, str], phase: str, pool: list[dict],
                 vetoes: set[tuple[str, str]], scenarios, u, protocol: dict, target: Path):
    manifest = read(directory / "agent_manifest.json")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        manifest["tokenizer_path"], local_files_only=True)
    receipts_list = rows(directory / "generations.jsonl")
    receipts = {row["generation_uid"]: row for row in receipts_list}
    require(len(receipts) == len(receipts_list) == manifest["live_api_calls"],
            "generation count differs")
    saved_rows = rows(directory / "sessions.jsonl")
    seed = read(directory / "summary.json")["actor_seed"]
    require(adapter.fingerprint(saved_rows) == adapter.fingerprint(
        [row for row in combined if row["actor_seed"] == seed]), "combined rows differ")
    indexed = {(row["method"], row["task"]): row for row in saved_rows}
    replayed = 0
    for trajectory in trajectories:
        available = leave_family_out(pool, trajectory["trajectory"]) \
            if phase == "stream_development" else pool
        for task in trajectory["sessions"]:
            for method in methods:
                saved = indexed[(method, task)]
                selector = "CREDIT" if method.startswith("CREDIT-") else method
                variant = variants.get(method, "diverse2")
                bank = select_bank(selector, available, task, scenarios[task], vetoes, variant)
                require(saved["selected_bank"] == saved["bank"] ==
                        [note["uid"] for note in bank], "bank differs")
                job = initial_job_v2(task, scenarios[task], bank, u, tokenizer,
                                     phase, method)
                for turn in saved["generations"]:
                    rendered = tokenizer.apply_chat_template(job["chat"], tokenize=False,
                                                              add_generation_prompt=True)
                    receipt = receipts[turn["generation_uid"]]
                    require(sha256(rendered.encode()).hexdigest() == receipt["prompt_sha256"],
                            "prompt differs")
                    body = json.dumps({"model": manifest["model"], "messages": job["chat"],
                        "temperature": manifest["temperature"], "seed": seed,
                        "max_tokens": manifest["max_new_tokens"]}, ensure_ascii=False,
                        separators=(",", ":")).encode()
                    require(sha256(body).hexdigest() == receipt["request_sha256"],
                            "request differs")
                    adapter.consume(job, receipt, u, protocol["maximum_tool_calls_per_session"])
                if not job["done"]: job["termination"] = "turn_budget"
                rebuilt = adapter.finish(job, target, u)
                for field in ("calls", "generations", "ending_state_sha256",
                              "upstream_evaluation", "success", "verification_coverage",
                              "termination"):
                    require(adapter.fingerprint(rebuilt[field]) ==
                            adapter.fingerprint(saved[field]), "replay differs: " + field)
                replayed += 1
    return replayed, len(receipts_list)


def verify(root: Path) -> dict:
    protocol = read(root / "frozen_protocol.json")
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "source" / name) == expected, "frozen source changed: " + name)
        require(digest(Path(adapter.__file__).with_name(name)) == expected,
                "current source changed: " + name)
    upstream = Path(protocol["upstream_directory"])
    for name, expected in protocol["upstream_source_sha256"].items():
        require(digest(upstream / name) == expected, "upstream source changed: " + name)
    require(digest(Path(protocol["source_results"]) / "development_sessions.jsonl") ==
            protocol["source_development_sha256"], "source development changed")
    require(digest(Path(protocol["factorial_results"]) / "effects.json") ==
            protocol["factorial_effects_sha256"], "factorial effects changed")
    scenarios = dill.loads((root / "scenarios.pkl").read_bytes())
    pool = read(root / "memory_pool.json")
    effects = read(Path(protocol["factorial_results"]) / "effects.json")
    vetoes = harmful_pairs(effects)
    u = adapter.upstream_imports(upstream); adapter.install_local_clock(protocol["clock"])
    development, confirmation = rows(root / "development_sessions.jsonl"), rows(root / "confirmation_sessions.jsonl")
    chosen = read(root / "frozen_selector.json")["chosen_variant"]
    replayed = requests = 0
    with tempfile.TemporaryDirectory(prefix="credit-stream-audit-") as temporary:
        target = Path(temporary)
        dev_methods = ["CREDIT-" + variant for variant in DEV_VARIANTS]
        for seed in protocol["development_seeds"]:
            a, b = verify_actor(root / f"stream_development_seed_{seed}", development,
                protocol["development"], dev_methods,
                {method: method.removeprefix("CREDIT-") for method in dev_methods},
                "stream_development", pool, vetoes, scenarios, u, protocol, target)
            replayed += a; requests += b
        for seed in protocol["confirmation_seeds"]:
            a, b = verify_actor(root / f"stream_confirmation_seed_{seed}", confirmation,
                protocol["confirmation"], list(CONFIRM_METHODS), {"CREDIT": chosen},
                "stream_confirmation", pool, vetoes, scenarios, u, protocol, target)
            replayed += a; requests += b
    summary = read(root / "summary.json")
    expected = summarize(confirmation, list(CONFIRM_METHODS),
                         summary["credit_training_latency_ms"])
    require(adapter.fingerprint(expected) == adapter.fingerprint(
        summary["confirmation_table"]), "summary differs")
    require(adapter.fingerprint(paired_statistics(confirmation)) ==
            adapter.fingerprint(summary["paired_statistics"]), "statistics differ")
    report = {"verified": True, "development_sessions": len(development),
        "confirmation_sessions": len(confirmation), "replayed_episodes": replayed,
        "live_remote_requests": requests, "credentials_persisted": False,
        "development_sha256": digest(root / "development_sessions.jsonl"),
        "confirmation_sha256": digest(root / "confirmation_sessions.jsonl"),
        "verifier_source_sha256": digest(Path(__file__))}
    (root / "execution_audit.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__": main()
