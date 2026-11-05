"""Cross-suite audit for the v4 realism experiments."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from .llm_remote_policy_v2 import load_config
from .real_network_latency import summarize as latency_summary
from .semantic_text_interactions import cross_validated_errors, factorial_summary


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def equivalent(left, right) -> bool:
    """Allow machine-precision variation in reconstructed floating metrics."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return (left.keys() == right.keys() and
                all(equivalent(left[key], right[key]) for key in left))
    if isinstance(left, list) and isinstance(right, list):
        return (len(left) == len(right) and
                all(equivalent(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def source_matches(protocol: dict, module: Path) -> bool:
    return protocol["source_sha256"] == sha256(module.read_bytes()).hexdigest()


def verify(root: Path, env: Path) -> dict:
    web = root / "webapi_evolve_observed_v1"
    network = root / "real_network_latency_v1"
    semantic = root / "semantic_text_interactions_confirm_v1"
    sandbox = root / "toolsandbox_m_confirm_v1"

    web_protocol = read(web / "frozen_protocol.json")
    web_audit = read(web / "audit.json")
    web_rows = rows(web / "responses.jsonl")
    require(source_matches(web_protocol, Path("credit/webapi_evolve_observed.py")),
            "WebAPI source changed")
    require(web_audit["checks"]["audit_passed"] and len(web_rows) == 68,
            "WebAPI replay audit failed")
    require(all(row["status"] is not None and len(row["body_sha256"]) == 64 for row in web_rows),
            "WebAPI response ledger incomplete")

    network_protocol = read(network / "frozen_protocol.json")
    network_audit = read(network / "audit.json")
    network_rows = rows(network / "trials.jsonl")
    require(source_matches(network_protocol, Path("credit/real_network_latency.py")),
            "network source changed")
    require(network_audit["passed"] and network_audit["trials_sha256"]
            == sha256((network / "trials.jsonl").read_bytes()).hexdigest(),
            "network audit failed")
    require(latency_summary(network_rows) == read(network / "summary.json"),
            "network summary does not reconstruct")

    semantic_protocol = read(semantic / "frozen_protocol.json")
    semantic_audit = read(semantic / "audit.json")
    semantic_rows = rows(semantic / "episodes.jsonl")
    require(source_matches(semantic_protocol, Path("credit/semantic_text_interactions.py")),
            "semantic source changed")
    require(semantic_audit["passed"] and len(semantic_rows) == 256,
            "semantic audit failed")
    saved_semantic = read(semantic / "summary.json")
    require(equivalent(factorial_summary(semantic_rows, "development"),
                       saved_semantic["development"]),
            "semantic development summary differs")
    require(equivalent(factorial_summary(semantic_rows, "confirmation"),
                       saved_semantic["confirmation"]),
            "semantic confirmation summary differs")
    cv, predictions = cross_validated_errors(semantic_rows, "confirmation")
    require(equivalent(cv, saved_semantic["confirmation_cross_task_prediction"]),
            "semantic cross-validation differs")
    require(len(predictions) == 24 * 8 * 3, "semantic prediction count differs")

    sandbox_audit = read(sandbox / "execution_audit.json")
    sandbox_summary = read(sandbox / "summary.json")
    require(sandbox_audit["verified"] and sandbox_audit["replayed_episodes"] == 224,
            "ToolSandbox-M replay audit failed")
    require(sandbox_summary["test_sessions"] == 200
            and all(row["verification_coverage"] == 1 for row in sandbox_summary["table"]),
            "ToolSandbox-M result coverage differs")

    config = load_config(env)
    secret = config["api_key"].encode()
    credential_absent = not secret or not any(
        secret in path.read_bytes() for directory in (web, network, semantic, sandbox)
        for path in directory.rglob("*") if path.is_file())
    require(credential_absent, "LLM credential leaked into result artifacts")
    report = {"verified": True,
        "webapi": {"response_rows": len(web_rows), "applications": 2, "versions": 6,
                   "documented_changes": 6},
        "network": {"trials": len(network_rows), "external_https_calls": 40,
                    "workflow_successes": sum(row["workflow_success"] for row in network_rows)},
        "semantic": {"episodes": len(semantic_rows), "confirmation_tasks": 24,
                     "prediction_rows_reconstructed": len(predictions)},
        "toolsandbox_m": {"replayed_episodes": sandbox_audit["replayed_episodes"],
                          "test_sessions": sandbox_summary["test_sessions"],
                          "trajectories": sandbox_summary["trajectories"]},
        "credentials_absent": credential_absent,
        "verifier_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/realism_bundle_audit_v1.json"))
    args = parser.parse_args()
    report = verify(args.results, args.env)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
