"""Render the multi-actor, data-text, and provider-monitor results."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path("results")


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def main() -> None:
    tool = ROOT / "toolsandbox_m_multi_actor_v3"
    text = ROOT / "dataset_text_memory_interactions_v1"
    drift = ROOT / "provider_drift_monitor_v1"
    ts = read(tool / "summary.json")
    tx = read(text / "summary.json")
    ds = read(drift / "longitudinal_summary.json")
    generations = [row for actor in tool.glob("actor_seed_*")
                   for row in rows(actor / "generations.jsonl")]
    attempts = Counter(row["network_attempts"] for row in generations)
    lines = ["# CREDIT v4 Extended Realism Results", "",
        "## Multi-actor ToolSandbox-M and common latency budget", "",
        "Three remote actor seeds each ran all 20 two-session trajectories. Every actor turn made a live HTTPS model request; no response or latency was reused across methods. Method order was rotated across actors. The table has 120 scored sessions per method. The 10 s column counts a verified success only when its attributed latency fits the pre-specified budget. Common history is excluded, while CREDIT-only calibration latency is amortized across its 40 test sessions per actor.", "",
        "| Rank at 10 s | Method | VSR (%) | VSR at 10 s (%) | Completion at 10 s (%) | Mean attributed latency (s) | p95 (s) |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    rank = {row["method"]: row["rank"] for row in ts["latency_ranking"]}
    for row in sorted(ts["table"], key=lambda item: rank[item["method"]]):
        lines.append(f"| {rank[row['method']]} | {row['method']} | {pct(row['strict_VSR'])} | "
            f"{pct(row['VSR_at_10000ms'])} | {pct(row['completion_at_10000ms'])} | "
            f"{row['mean_attributed_remote_latency_ms']/1000:.2f} | "
            f"{row['p95_attributed_remote_latency_ms']/1000:.2f} |")
    comparisons = {row["baseline"]: row for row in ts["paired_statistics"]["comparisons"]}
    no_mem = comparisons["NoMem"]
    lines += ["", f"CREDIT's uncensored VSR difference from NoMem was {100*no_mem['CREDIT_delta_VSR']:.2f} points "
        f"(cluster bootstrap CI [{100*no_mem['cluster_bootstrap_CI95'][0]:.2f}, "
        f"{100*no_mem['cluster_bootstrap_CI95'][1]:.2f}], exact sign p={no_mem['one_sided_exact_sign_p']:.3f}). "
        "CREDIT tied Semantic on uncensored VSR and did not lead the latency ranking. Semantic ranked first at 10 s.", "",
        f"The run made {len(generations):,} live model requests ({attempts[1]:,} first-attempt successes and "
        f"{sum(count for attempt, count in attempts.items() if attempt > 1):,} requests requiring a retry). "
        f"The maximum complete request latency was {max(row['remote_latency_ms'] for row in generations)/1000:.2f} s. "
        "The provider returned one dated model snapshot but two system fingerprints. These observed tails are retained in the ranking.", "",
        "## Supplied-dataset text memories", "",
        "The source payloads are service profiles, endpoint interface documentation, and observed endpoint-test feedback from the user-provided dataset. Temporary handles, prompts, ablations, and placebos are constructed. Gold endpoints are weak supervision from `selection_tasks.jsonl`; the suite does not claim that the records are production agent-memory logs.", "",
        "| Coalition | Confirmation success (%) |", "|---|---:|"]
    for coalition, rate in tx["confirmation"]["coalition_success_rates"].items():
        lines.append(f"| {coalition} | {pct(rate)} |")
    ci = tx["confirmation"]["third_order_ci95"]
    lines += ["", f"Across 24 frozen confirmation tasks, BC and ABC both reached 100%. The mean third-order Harsanyi effect was "
        f"{tx['confirmation']['mean_third_order_harsanyi']:.3f} (95% bootstrap CI [{ci[0]:.3f}, {ci[1]:.3f}]). "
        "Thus the data-derived records did not reproduce the necessary third-order complement seen in the planted routing benchmark.", "",
        "## Closed-provider temporal sentinel", "",
        f"Two current replay waves used the same 48 prompts and request settings as a historical cached artifact. The artifact-to-current span is only {ds['observed_span_hours']:.1f} h. "
        f"Two sentinels ({pct(ds['stable_temporal_deviation_rate'])}%) changed from the historical answer and agreed across both current waves. "
        f"Four ({pct(ds['within_window_instability_rate'])}%) disagreed between the two current waves and are classified as within-window instability rather than temporal drift.", "",
        "The monitoring harness is complete, but the predeclared 30-day threshold has not been reached. These observations are short-span provider variability and stable output deviations; they are not evidence of long-term drift.", "",
        "## Reproduction", "", "```sh",
        "POLARS_MAX_THREADS=4 /tmp/credit-toolsandbox-env/bin/python -m credit.toolsandbox_m_multi_actor --output results/toolsandbox_m_multi_actor_v3 --actor-seeds 101 203 307 --workers 8",
        "PYTHONPATH=vendor/ToolSandbox POLARS_MAX_THREADS=4 /tmp/credit-toolsandbox-env/bin/python -m credit.toolsandbox_m_multi_actor_verify --results results/toolsandbox_m_multi_actor_v3",
        "python -m credit.dataset_text_memory_interactions --output results/dataset_text_memory_interactions_v1 --workers 8",
        "python -m credit.dataset_text_memory_interactions_verify --results results/dataset_text_memory_interactions_v1",
        "python -m credit.provider_drift_sentinel --output results/provider_drift_monitor_v1 --sentinels 48 --workers 8",
        "python -m credit.provider_drift_analyze --results results/provider_drift_monitor_v1", "```", ""]
    Path("CREDIT_V4_EXTENDED_REALISM_RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
