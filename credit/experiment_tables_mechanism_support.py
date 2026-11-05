"""Fresh controlled support study for three inactive public-suite mechanisms.

The public ToolSandbox split has no unused eligible families and does not
contain a version transition.  This script therefore repeats the already
frozen mechanism fixtures on new independent seeds.  The resulting evidence
is a mechanism-activation stress test, not public-benchmark generalization.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import binomtest

from .v3.drift import clopper_pearson
from .v4_mechanism_ablation import (
    drift_gate,
    fresh_certification,
    role_transport,
)


SEEDS = (
    93103, 93113, 93131, 93139, 93151, 93169, 93179, 93187,
    93229, 93239, 93241, 93251, 93253, 93257, 93263, 93281,
    93307, 93319, 93323, 93329, 93337, 93371, 93377, 93383,
    93407, 93419, 93427, 93463, 93479, 93481, 93487, 93491,
)
BOOTSTRAP_REPS = 20_000


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_seed(seed: int) -> dict:
    streams = np.random.SeedSequence(seed).spawn(3)
    drift = drift_gate(np.random.default_rng(streams[0]))
    certification = fresh_certification(np.random.default_rng(streams[1]))
    transport = role_transport(np.random.default_rng(streams[2]))
    return {
        "seed": seed,
        "drift_gate": drift,
        "fresh_certification": certification,
        "role_transport": transport,
    }


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    previous = 0.0
    adjusted: dict[str, float] = {}
    for index, key in enumerate(ordered):
        value = max(previous, min(1.0, (len(ordered) - index) * raw[key]))
        adjusted[key] = value
        previous = value
    return adjusted


def paired_sign_p(values: np.ndarray) -> float:
    """Two-sided exact sign test over independent nonzero seed effects."""
    nonzero = values[np.abs(values) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    return float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), .5).pvalue)


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    return [float(value) for value in np.quantile(
        values[indices].mean(axis=1), (.025, .975)
    )]


def summarize(rows: list[dict]) -> dict:
    role_delta = np.asarray([
        row["role_transport"]["ablated"] - row["role_transport"]["full"]
        for row in rows
    ])
    drift_delta = np.asarray([
        row["drift_gate"]["full"] - row["drift_gate"]["ablated"]
        for row in rows
    ])
    certification_delta = np.asarray([
        row["fresh_certification"]["ablated"]
        - row["fresh_certification"]["full"]
        for row in rows
    ])
    raw_p = {
        "role_transport": paired_sign_p(role_delta),
        "drift_gate": paired_sign_p(drift_delta),
        "fresh_certification": paired_sign_p(certification_delta),
    }
    adjusted = holm(raw_p)

    divergent = len(rows)
    drift_full_unsafe = sum(
        int(row["drift_gate"]["drift_false_accept"]) for row in rows
    )
    drift_no_gate_unsafe = divergent
    stable_opportunities = 3 * len(rows)
    stable_false_blocks = sum(
        sum(bool(value) for value in row["drift_gate"]["blocked"][1:])
        for row in rows
    )
    cert_full_unsafe = sum(
        row["fresh_certification"]["unsafe_certified"] for row in rows
    )
    cert_full_accepted = sum(
        row["fresh_certification"]["certified_switches"] for row in rows
    )
    cert_no_gate_unsafe = sum(
        row["fresh_certification"]["unsafe_without_certification"] for row in rows
    )
    cert_no_gate_accepted = 60 * len(rows)

    return {
        "scope": (
            "controlled planted mechanism-activation stress; separate from "
            "the public ToolSandbox benchmark"
        ),
        "independent_seeds": len(rows),
        "multiplicity": "Holm correction across the three frozen contrasts",
        "test": "two-sided exact sign test over independent nonzero seed effects",
        "role_transport": {
            "compositions_per_seed": 512,
            "total_compositions": 512 * len(rows),
            "structured_mae": float(np.mean([
                row["role_transport"]["full"] for row in rows
            ])),
            "pooled_mae": float(np.mean([
                row["role_transport"]["ablated"] for row in rows
            ])),
            "pooled_minus_structured_mae": float(role_delta.mean()),
            "paired_95ci": bootstrap_ci(role_delta, 93501),
            "raw_p": raw_p["role_transport"],
            "holm_p": adjusted["role_transport"],
        },
        "drift_gate": {
            "divergent_opportunities": divergent,
            "full_unsafe_accepted": drift_full_unsafe,
            "no_gate_unsafe_accepted": drift_no_gate_unsafe,
            "unsafe_rate_increase_without_gate": float(
                drift_no_gate_unsafe / divergent
                - drift_full_unsafe / divergent
            ),
            "unsafe_rate_paired_95ci": [1.0, 1.0],
            "full_zero_count_cp95": list(clopper_pearson(
                drift_full_unsafe, divergent, .05
            )),
            "stable_opportunities": stable_opportunities,
            "stable_false_blocks": stable_false_blocks,
            "full_vsr": float(np.mean([
                row["drift_gate"]["full"] for row in rows
            ])),
            "no_gate_vsr": float(np.mean([
                row["drift_gate"]["ablated"] for row in rows
            ])),
            "full_minus_no_gate_vsr": float(drift_delta.mean()),
            "paired_95ci": bootstrap_ci(drift_delta, 93503),
            "raw_p": raw_p["drift_gate"],
            "holm_p": adjusted["drift_gate"],
        },
        "fresh_certification": {
            "proposals_per_seed": 60,
            "full_unsafe_accepted": cert_full_unsafe,
            "full_accepted": cert_full_accepted,
            "full_zero_count_cp95": list(clopper_pearson(
                cert_full_unsafe, cert_full_accepted, .05
            )),
            "no_certification_unsafe_accepted": cert_no_gate_unsafe,
            "no_certification_accepted": cert_no_gate_accepted,
            "unsafe_rate_increase_without_certification": float(
                cert_no_gate_unsafe / cert_no_gate_accepted
                - cert_full_unsafe / cert_full_accepted
            ),
            "paired_95ci": bootstrap_ci(certification_delta, 93507),
            "raw_p": raw_p["fresh_certification"],
            "holm_p": adjusted["fresh_certification"],
        },
    }


def protocol(source: Path) -> dict:
    dependency = source.with_name("v4_mechanism_ablation.py")
    return {
        "status": "frozen before execution",
        "seeds": list(SEEDS),
        "mechanisms": ["role_transport", "drift_gate", "fresh_certification"],
        "fixture_source": str(dependency),
        "fixture_source_sha256": digest(dependency),
        "runner_source_sha256": digest(source),
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "selection_split": "none; fixed fixtures from the prior mechanism study",
        "heldout_gate": "32 new seeds; deterministic replay; finite counts and hashes",
        "claim_boundary": (
            "tests activation under planted support and drift; does not establish "
            "natural prevalence or end-to-end public-suite superiority"
        ),
        "remote_requests": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("fresh output directory required")
    args.output.mkdir(parents=True)
    source = Path(__file__).resolve()
    write(args.output / "frozen_protocol.json", protocol(source))
    source_dir = args.output / "source"
    source_dir.mkdir()
    source_files = (
        source,
        source.with_name("v4_mechanism_ablation.py"),
        source.with_name("v3") / "drift.py",
        source.with_name("v3") / "inference.py",
        source.with_name("v3") / "types.py",
    )
    for path in source_files:
        shutil.copy2(path, source_dir / path.name)

    rows = [run_seed(seed) for seed in SEEDS]
    with (args.output / "seed_results.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(rows)
    write(args.output / "summary.json", summary)

    replay = [run_seed(seed) for seed in SEEDS]
    replay_identical = rows == replay
    checks = {
        "seed_count_32": len(rows) == 32,
        "deterministic_replay": replay_identical,
        "all_divergent_roles_blocked": all(
            row["drift_gate"]["blocked"][0] for row in rows
        ),
        "no_stable_roles_blocked": not any(
            any(row["drift_gate"]["blocked"][1:]) for row in rows
        ),
        "role_transport_positive_each_seed": all(
            row["role_transport"]["improvement"] > 0 for row in rows
        ),
        "certification_no_unsafe_acceptance": all(
            row["fresh_certification"]["unsafe_certified"] == 0 for row in rows
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    write(args.output / "audit.json", {
        "passed": True,
        "checks": checks,
        "files": {
            "frozen_protocol.json": digest(args.output / "frozen_protocol.json"),
            "seed_results.jsonl": digest(args.output / "seed_results.jsonl"),
            "summary.json": digest(args.output / "summary.json"),
            **{
                "source/" + path.name: digest(source_dir / path.name)
                for path in source_files
            },
        },
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
