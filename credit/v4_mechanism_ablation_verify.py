"""Deterministically replay the CREDIT v4 mechanism-aligned ablations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .v4_mechanism_ablation import protocol, run_seed, sha256, summarize, write_json


def equivalent(left, right) -> bool:
    """Compare deterministic replays while allowing BLAS round-off noise."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.results / "frozen_protocol.json").read_text())
    seeds = frozen["confirmation_seeds"]
    expected_protocol = protocol(seeds)
    if frozen != expected_protocol:
        raise ValueError("frozen protocol or source hash changed")
    stored = [json.loads(line) for line in
              (args.results / "seed_results.jsonl").read_text().splitlines()]
    replayed = [run_seed(seed) for seed in seeds]
    if not equivalent(stored, replayed):
        raise ValueError("seed-level deterministic replay differs")
    expected_summary = summarize(replayed)
    actual_summary = json.loads((args.results / "summary.json").read_text())
    if not equivalent(actual_summary, expected_summary):
        raise ValueError("summary differs")
    output = {
        "verified": True,
        "replayed_seeds": len(replayed),
        "seed_results_sha256": sha256(args.results / "seed_results.jsonl"),
        "credentials_persisted": False,
        "live_remote_requests": 0,
        "scope": frozen["scope"],
    }
    write_json(args.results / "execution_audit.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
