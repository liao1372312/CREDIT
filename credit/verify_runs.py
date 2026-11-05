"""Verify shared/LLM experiment source, checkpoints and saved action steps."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from .stateful import StateTask, run_episode


def verify(root: Path):
    frozen = json.loads((root / "frozen_protocol.json").read_text())
    for name, digest in frozen["source_sha256"].items():
        if sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"source differs from frozen run: {name}")
    if "backend" in frozen:
        from .llm_policy import install_policy
        scores = Path(frozen["policy_directory"]) / "decision_scores.json"
        if sha256(scores.read_bytes()).hexdigest() != frozen["backend"]["decision_scores_sha256"]:
            raise ValueError("pinned model scores differ")
        install_policy(scores)
    episodes, step_count, prior_traces = 0, 0, 0
    for seed in frozen["executed_seeds"]:
        directory = root / f"seed_{seed}"
        manifest = json.loads((directory / "split_manifest.json").read_text())
        tasks = {row["uid"]: StateTask(**{**row, "roles": tuple(row["roles"])}) for row in manifest["heldout"]}
        saved_steps = defaultdict(list)
        with (directory / "executed_steps.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                saved_steps[(row["method"], row["task"])].append(row)
        with (directory / "raw_episodes.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                trace = run_episode(tasks[row["task"]], tuple(filter(None, row["exposure"].split("|"))))
                assert trace.state_hash == row["state_hash"] and trace.checkpoint_hash == row["checkpoint_hash"]
                assert trace.outcome == int(row["success"])
                recorded = saved_steps.pop((row["method"], row["task"]), [])
                assert len(recorded) == len(trace.steps)
                for index, (actual, saved) in enumerate(zip(trace.steps, recorded)):
                    assert int(saved["step"]) == index
                    for key, value in asdict(actual).items():
                        assert str(value) == saved[key], (seed, row["task"], key)
                episodes += 1
                step_count += len(trace.steps)
        assert not saved_steps
        for split in ("training", "calibration"):
            for row in json.loads((directory / f"{split}_traces.json").read_text()):
                task = StateTask(**{**row["task"], "roles": tuple(row["task"]["roles"])})
                trace = run_episode(task, row["exposure"])
                assert trace.state_hash == row["state_hash"] and trace.checkpoint_hash == row["checkpoint_hash"]
                assert trace.outcome == row["outcome"]
                assert [asdict(step) for step in trace.steps] == row["steps"]
                prior_traces += 1
    report = {"all_source_hashes_match": True, "all_execution_hashes_match": True,
              "all_saved_steps_match": True, "heldout_episodes_verified": episodes,
              "heldout_steps_verified": step_count, "training_and_calibration_traces_verified": prior_traces,
              "verifier_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / "execution_integrity.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))


if __name__ == "__main__":
    main()
