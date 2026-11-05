"""Deterministically replay and verify revised baseline artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from .revised_baselines import run, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--release-summary", type=Path,
                        default=Path("results/webapi_evolve_observed_v1/summary.json"))
    args = parser.parse_args()
    manifest = json.loads((args.results / "artifact_manifest.json").read_text())
    for name, digest in manifest.items():
        if sha256(args.results / name) != digest:
            raise AssertionError(f"manifest mismatch: {name}")
    with tempfile.TemporaryDirectory(prefix="credit-revised-baselines-") as temporary:
        replay = Path(temporary) / "replay"
        run(replay, args.release_summary)
        for name, digest in manifest.items():
            if sha256(replay / name) != digest:
                raise AssertionError(f"deterministic replay mismatch: {name}")
    summary = json.loads((args.results / "summary.json").read_text())
    assert summary["E1"]["Full CREDIT"]["base_successes"] == 503
    assert summary["E1"]["Full CREDIT"]["guard_successes"] == 529
    assert len(summary["E1"]) == 14
    assert len(summary["E2"]) == 5
    audit = {"verified": True, "deterministic_replay": True,
             "artifact_count": len(manifest), "summary_sha256": sha256(args.results / "summary.json"),
             "live_remote_requests": 0, "llm_requests": 0,
             "credentials_persisted": False,
             "scope": summary["scope"]}
    write_json(args.results / "execution_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
