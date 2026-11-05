"""Replay and verify the revised CREDIT-only E1--E6 pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from .revised_credit_only import protocol, run, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.results / "frozen_protocol.json").read_text())
    release_summary = Path(frozen["e3"]["observed_release_summary"])
    if frozen != protocol(release_summary):
        raise ValueError("frozen protocol or source hash changed")
    manifest = json.loads((args.results / "artifact_manifest.json").read_text())
    for name, expected in manifest.items():
        if sha256(args.results / name) != expected:
            raise ValueError(f"stored artifact changed: {name}")
    with tempfile.TemporaryDirectory(prefix="credit-revised-replay-") as directory:
        replay = Path(directory) / "replay"
        run(replay, release_summary)
        replay_manifest = json.loads((replay / "artifact_manifest.json").read_text())
        if replay_manifest != manifest:
            differing = sorted(set(replay_manifest) | set(manifest))
            differing = [name for name in differing
                          if replay_manifest.get(name) != manifest.get(name)]
            raise ValueError("deterministic replay differs: " + ", ".join(differing))
    audit = {
        "verified": True,
        "deterministic_replay": True,
        "artifact_count": len(manifest),
        "summary_sha256": manifest["summary.json"],
        "live_remote_requests": 0,
        "llm_requests": 0,
        "credentials_persisted": False,
        "scope": frozen["claim_boundary"],
    }
    write_json(args.results / "execution_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
