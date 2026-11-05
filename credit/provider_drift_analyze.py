"""Audit and separate stable temporal deviations from within-window instability."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from .provider_drift_sentinel import wilson
from .user_data import digest


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/provider_drift_monitor_v1"))
    args = parser.parse_args()
    protocol = json.loads((args.results / "monitor_protocol.json").read_text())
    baseline = {row["index"]: row for row in rows(Path(protocol["historical_baseline"]) / "responses.jsonl")}
    wave_paths = sorted((args.results / "waves").glob("*.jsonl"))
    if len(wave_paths) < 2:
        raise ValueError("at least two current replay waves are required")
    waves = [{row["index"]: row for row in rows(path)} for path in wave_paths]
    indices = protocol["sentinel_indices"]
    for wave in waves:
        if sorted(wave) != indices:
            raise ValueError("wave sentinel coverage differs")
        for index in indices:
            if wave[index]["prompt_sha256"] != baseline[index]["prompt_sha256"]:
                raise ValueError("wave prompt differs from historical baseline")
    unstable, stable_changed, stable_same = [], [], []
    for index in indices:
        values = {wave[index]["letter"] for wave in waves}
        if len(values) > 1:
            unstable.append(index)
        elif next(iter(values)) != baseline[index]["letter"]:
            stable_changed.append(index)
        else:
            stable_same.append(index)
    baseline_time = datetime.fromisoformat(protocol["historical_artifact_mtime_utc"])
    summaries = [json.loads(path.with_suffix(".summary.json").read_text()) for path in wave_paths]
    latest_time = max(datetime.fromisoformat(summary["collection_date_utc"]) for summary in summaries)
    span_hours = (latest_time - baseline_time).total_seconds() / 3600
    total = len(indices)
    summary = {"historical_artifact_mtime_utc": protocol["historical_artifact_mtime_utc"],
        "current_replay_waves": len(waves), "current_remote_requests": len(waves) * total,
        "sentinels_per_wave": total, "observed_span_hours": span_hours,
        "stable_temporal_deviations": len(stable_changed),
        "stable_temporal_deviation_rate": len(stable_changed) / total,
        "stable_temporal_deviation_rate_wilson_CI95": wilson(len(stable_changed), total),
        "within_window_unstable_sentinels": len(unstable),
        "within_window_instability_rate": len(unstable) / total,
        "within_window_instability_rate_wilson_CI95": wilson(len(unstable), total),
        "stable_unchanged_sentinels": len(stable_same),
        "stable_temporal_deviation_indices": stable_changed,
        "within_window_unstable_indices": unstable,
        "provider_models": sorted({row["provider_model"] for wave in waves for row in wave.values()}),
        "system_fingerprints": sorted({row["system_fingerprint_sha256"] for wave in waves
                                       for row in wave.values() if row["system_fingerprint_sha256"]}),
        "long_term_evidence_available": span_hours >= 24 * 30,
        "interpretation": "Stable deviations are repeated current outputs that differ from the historical artifact. Unstable sentinels are not attributed to temporal drift.",
        "long_term_limit": "The 30-day threshold is not met; future identical waves must be appended before claiming long-term provider drift.",
        "wave_sha256": {path.name: digest(path) for path in wave_paths}}
    (args.results / "longitudinal_summary.json").write_text(json.dumps(summary, indent=2))
    audit = {"verified": True, "historical_prompt_hashes_matched": True,
        "waves": len(waves), "wave_rows": len(waves) * total,
        "partition_complete": len(stable_changed) + len(unstable) + len(stable_same) == total,
        "credentials_persisted": False, "analysis_source_sha256": digest(Path(__file__))}
    (args.results / "execution_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
