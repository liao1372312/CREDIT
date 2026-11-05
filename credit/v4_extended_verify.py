"""Cross-artifact audit for the v4 extended realism experiments."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .llm_remote_policy_v2 import load_config
from .user_data import digest


def read(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path,
        default=Path("results/credit_v4_extended_realism_audit_v2.json"))
    args = parser.parse_args()
    tool = Path("results/toolsandbox_m_multi_actor_v3")
    text = Path("results/dataset_text_memory_interactions_v1")
    drift = Path("results/provider_drift_monitor_v1")
    followup = Path("results/toolsandbox_m_followup_v1")
    safe = Path("results/toolsandbox_m_safe_projection_v1")
    tool_audit = read(tool / "execution_audit.json")
    text_audit = read(text / "execution_audit.json")
    drift_audit = read(drift / "execution_audit.json")
    followup_audit = read(followup / "execution_audit.json")
    tool_summary = read(tool / "summary.json")
    text_summary = read(text / "summary.json")
    drift_summary = read(drift / "longitudinal_summary.json")
    followup_summary = read(followup / "summary.json")
    safe_summary = read(safe / "safe_projection.json")
    paper = Path("CREDIT_WWW2027_v4.tex").read_text()
    report = Path("CREDIT_V4_EXTENDED_REALISM_RESULTS.md").read_text()
    pdf_path = Path("build_v4/CREDIT_WWW2027_v4.pdf")
    build_log = Path("build_v4/CREDIT_WWW2027_v4.log").read_text(errors="ignore")
    config = load_config(args.env)
    forbidden = [value for value in (config["api_key"],) if value]
    # The URL itself is also private configuration; only its digest may persist.
    fields = {}
    for line in args.env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip().strip('"').strip("'")
    if fields.get("CREDIT_LLM_BASE_URL"):
        forbidden.append(fields["CREDIT_LLM_BASE_URL"].rstrip("/"))
    targets = [Path("CREDIT_WWW2027_v4.tex"), Path("CREDIT_V4_EXTENDED_REALISM_RESULTS.md"),
               Path("CREDIT_V4_OPTIMIZATION_FOLLOWUP_RESULTS.md")]
    for root in (tool, text, drift, followup, safe):
        targets.extend(path for path in root.rglob("*") if path.is_file()
                       and path.suffix in {".json", ".jsonl", ".md", ".csv", ".py"})
    credentials_absent = True
    for path in targets:
        content = path.read_text(errors="ignore")
        if any(secret in content for secret in forbidden):
            credentials_absent = False
            break
    ranked = {row["method"]: row for row in tool_summary["latency_ranking"]}
    followup_ranked = {row["method"]: row for row in followup_summary["latency_ranking"]}
    checks = {
        "toolsandbox_execution_replay": tool_audit["verified"] and tool_audit["replayed_episodes"] == 678,
        "toolsandbox_sessions": tool_audit["test_sessions"] == 600,
        "toolsandbox_remote_requests": tool_audit["live_remote_requests"] == 2738,
        "latency_rank_recomputed": ranked["Semantic"]["rank"] == 1 and ranked["CREDIT"]["rank"] == 5,
        "followup_execution_replay": followup_audit["verified"] and
            followup_audit["replayed_test_episodes"] == 696,
        "followup_sessions": followup_audit["confirmation_sessions"] == 600,
        "followup_requests": followup_audit["test_remote_requests"] == 3132,
        "followup_latency_rank": followup_ranked["Recency"]["rank"] == 1 and
            followup_ranked["CREDIT"]["rank"] == 5,
        "safe_projection_disclosed": safe_summary["status"] ==
            "exploratory_projection_not_new_online_confirmation",
        "dataset_execution_recomputed": text_audit["verified"] and text_audit["episodes"] == 256,
        "dataset_third_order_zero": text_summary["confirmation"]["mean_third_order_harsanyi"] == 0.0,
        "provider_waves_recomputed": drift_audit["verified"] and drift_audit["wave_rows"] == 96,
        "provider_long_term_not_claimed": not drift_summary["long_term_evidence_available"],
        "paper_has_no_pending_cells": "\\crpending" not in paper and "not yet measured" not in paper.lower(),
        "paper_reports_latency_rank": "ranked fifth" in paper and "five-actor" in paper,
        "paper_reports_short_span_limit": "predeclared 30-day threshold remains unmet" in paper,
        "report_matches_primary_values": "CREDIT 11.67%" in report and
            "Recency/FIFO/NoMem reached 8.33%" in report,
        "paper_pdf_built": pdf_path.exists() and pdf_path.stat().st_size > 100_000,
        "paper_build_has_no_undefined_references": not any(marker in build_log.lower() for marker in
            ("undefined references", "undefined citations", "fatal error", "emergency stop")),
        "credentials_absent": credentials_absent,
    }
    result = {"verified": all(checks.values()), "checks": checks,
        "toolsandbox": {key: tool_audit[key] for key in
                        ("actors", "replayed_episodes", "test_sessions", "live_remote_requests")},
        "toolsandbox_followup": {key: followup_audit[key] for key in
            ("development_sessions", "confirmation_sessions", "replayed_test_episodes",
             "test_remote_requests")},
        "dataset_text": {key: text_audit[key] for key in
                         ("tasks", "episodes", "confirmation_tasks", "live_remote_requests")},
        "provider_monitor": {"waves": drift_audit["waves"],
            "wave_rows": drift_audit["wave_rows"],
            "observed_span_hours": drift_summary["observed_span_hours"],
            "stable_temporal_deviations": drift_summary["stable_temporal_deviations"],
            "within_window_unstable_sentinels": drift_summary["within_window_unstable_sentinels"]},
        "paper_sha256": sha256(Path("CREDIT_WWW2027_v4.tex").read_bytes()).hexdigest(),
        "paper_pdf_sha256": digest(pdf_path),
        "report_sha256": digest(Path("CREDIT_V4_EXTENDED_REALISM_RESULTS.md")),
        "verifier_source_sha256": digest(Path(__file__))}
    args.output.write_text(json.dumps(result, indent=2))
    if not result["verified"]:
        raise ValueError("extended realism cross-audit failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
