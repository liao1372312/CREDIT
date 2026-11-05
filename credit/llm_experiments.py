"""Run common mechanism experiments with pinned offline Qwen action choices."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .llm_policy import install_policy
from .shared_experiments import run_seed
from .stateful_experiments import _json, aggregate, plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("results/llm_policy"))
    parser.add_argument("--config", type=Path, default=Path("configs/llm_shared_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/llm_shared_v1"))
    args = parser.parse_args()
    scores_path = args.policy / "decision_scores.json"
    backend = json.loads((args.policy / "model_manifest.json").read_text())
    if sha256(scores_path.read_bytes()).hexdigest() != backend["decision_scores_sha256"]:
        raise ValueError("frozen model score checksum mismatch")
    if sha256(Path(__file__).with_name("llm_policy.py").read_bytes()).hexdigest() != backend["adapter_source_sha256"]:
        raise ValueError("policy adapter differs from frozen source")
    install_policy(scores_path)
    payload = args.config.read_bytes()
    config = json.loads(payload)
    args.output.mkdir(parents=True, exist_ok=True)
    names = ("llm_policy.py", "llm_experiments.py", "shared_mechanisms.py", "shared_experiments.py",
             "stateful.py", "mechanisms.py", "prospective.py", "stateful_experiments.py")
    _json(args.output / "frozen_protocol.json", {
        "config": config, "executed_seeds": config["seeds"], "config_sha256": sha256(payload).hexdigest(),
        "backend": backend, "policy_directory": str(args.policy.resolve()),
        "source_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names},
    })
    rows = []
    for seed in config["seeds"]:
        episodes, report = run_seed(config, seed, args.output / f"seed_{seed}")
        report["scope"] = "offline Qwen finite-context constrained choices; synthetic API snapshots"
        report["backend_policy_sha256"] = backend["decision_scores_sha256"]
        report["llm_forward_cost"] = "one shared 336-context cache; sampled repetitions are not extra LLM calls"
        _json(args.output / f"seed_{seed}" / "report.json", report)
        rows.extend(episodes)
        print(json.dumps({"seed": seed, "complete": True, "seconds": report["elapsed_seconds"]}), flush=True)
    table = aggregate(rows, args.output)
    # The common plotting function labels its controlled environment accurately.
    plot(table, args.output)
    _json(args.output / "summary.json", {
        "primary": "shared sparse joint CREDIT-empirical under Qwen decisions", "main": table,
        "scope": "constrained cached Qwen policy; finite synthetic APIs, no live provider or external benchmark",
        "llm_forward_evaluations_total": backend["llm_context_forward_evaluations"],
        "llm_input_tokens_total": backend["input_tokens"],
        "sampler": "independent draws from pinned constrained token probabilities; no teacher scores used for fitting",
    })


if __name__ == "__main__":
    main()
