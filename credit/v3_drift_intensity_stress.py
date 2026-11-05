"""Fresh-seed controlled drift-intensity stress test.

This is a finite CausalCompose adapter, not natural provider drift.  A bank is
selected and frozen at the nominal target setting before any stress outcome is
evaluated.  The same target tasks and random numbers are then replayed over a
continuous severity grid with and without version-scoped retrieval gating.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path

from .experiments import _validation_traces, collect_pairs
from .model import StructuralModel
from .paired_statistics import summarize
from .policy import bank_cost, optimize_bank, retrieve
from .sandbox import make_tasks, run_task
from .user_data import digest


DEV_SEEDS = (73001,)
CONFIRM_SEEDS = (74001, 74009, 74019, 74039, 74051, 74061, 74079, 74099)
INTENSITIES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
TASKS_PER_INTENSITY = 512
BANK_BUDGET = 512
DEPTH = 4


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def seeds_for(split: str) -> tuple[int, ...]:
    return DEV_SEEDS if split == "development" else CONFIRM_SEEDS


def protocol(split: str) -> dict:
    sources = (
        "credit/v3_drift_intensity_stress.py",
        "credit/sandbox.py",
        "credit/model.py",
        "credit/policy.py",
        "credit/experiments.py",
        "credit/paired_statistics.py",
    )
    return {
        "experiment": "credit_v3_controlled_drift_intensity_v1",
        "split": split,
        "seeds": list(seeds_for(split)),
        "development_seeds": list(DEV_SEEDS),
        "confirmation_seeds_frozen_in_source": list(CONFIRM_SEEDS),
        "drift_intensity_grid": list(INTENSITIES),
        "tasks_per_intensity_per_method_per_seed": TASKS_PER_INTENSITY,
        "methods": ["CREDIT-Gated", "NoDriftGate", "NoMem"],
        "bank_budget_tokens": BANK_BUDGET,
        "retrieval_depth": DEPTH,
        "selection_intensity": 1.0,
        "selection_isolation": (
            "bank and fitted model freeze before the first intensity-grid outcome"
        ),
        "pairing": (
            "methods and intensities share task seeds; method comparisons are paired "
            "within each frozen intensity"
        ),
        "scope": (
            "finite planted CausalCompose policy/API kernels; continuous injected "
            "drift, no natural provider evolution or network latency"
        ),
        "source_sha256": {name: digest(Path(name)) for name in sources},
    }


def fit_and_freeze(seed: int) -> tuple[dict, tuple[str, ...]]:
    historical = make_tasks(
        128, split="drift-history", version="v1",
        templates=("short", "chain", "double_lookup"), seed=seed + 1,
    )
    probes = make_tasks(
        80, split="drift-probes", version="v2",
        templates=("short", "chain"), seed=seed + 2, drift_intensity=1.0,
    )
    validation_v1 = make_tasks(
        40, split="drift-validation-v1", version="v1",
        templates=("short", "chain"), seed=seed + 3,
    )
    validation_v2 = make_tasks(
        40, split="drift-validation-v2", version="v2",
        templates=("short", "chain"), seed=seed + 4, drift_intensity=1.0,
    )
    history_pairs, history_traces = collect_pairs(
        historical, 384, seed=seed + 6,
    )
    probe_pairs, probe_traces = collect_pairs(
        probes, 120, seed=seed + 7,
    )
    validation = _validation_traces(
        validation_v1 + validation_v2, seed=seed + 8,
    )
    model = StructuralModel(max_order=2).fit(history_traces + probe_traces)
    model.calibrate(validation)
    bank, objective = optimize_bank(
        model, validation_v1 + validation_v2,
        budget=BANK_BUDGET, depth=DEPTH,
    )
    plan = {
        "seed": seed,
        "bank": list(bank),
        "bank_tokens": bank_cost(bank),
        "objective": objective,
        "calibration_error": model.calibration_error,
        "history_pairs": len(history_pairs),
        "probe_pairs": len(probe_pairs),
        "history_selection_calls": sum(pair.extra_calls for pair in history_pairs),
        "target_probe_calls": sum(pair.extra_calls for pair in probe_pairs),
        "validation_calls": sum(trace.api_calls for trace in validation),
        "selection_uses_stress_outcomes": False,
    }
    return plan, bank


def evaluate(seed: int, bank: tuple[str, ...]) -> list[dict]:
    rows = []
    for intensity in INTENSITIES:
        tasks = make_tasks(
            TASKS_PER_INTENSITY,
            split=f"drift-confirmation-{intensity:g}", version="v2",
            templates=("new_order", "long"), seed=seed + 1000,
            drift_intensity=intensity,
        )
        for method, selected, safeguard in (
            ("CREDIT-Gated", bank, True),
            ("NoDriftGate", bank, False),
            ("NoMem", (), True),
        ):
            for task in tasks:
                exposure = retrieve(
                    task, selected, depth=DEPTH, drift_safeguard=safeguard,
                )
                trace = run_task(task, exposure)
                rows.append({
                    "seed": seed,
                    "intensity": intensity,
                    "method": method,
                    "task": task.uid,
                    "family": task.family,
                    "success": int(trace.outcome or 0),
                    "api_calls": trace.api_calls,
                    "exposure": list(exposure),
                    "state_hash": trace.state_hash,
                })
    return rows


def statistics(rows: list[dict]) -> dict:
    results = {}
    for index, intensity in enumerate(INTENSITIES):
        selected = [row for row in rows if row["intensity"] == intensity]
        main, paired = summarize(
            selected, "CREDIT-Gated", repeats=4000,
            random_seed=28001 + index,
        )
        results[f"{intensity:g}"] = {"main": main, "paired": paired}
    return results


def compute(split: str) -> tuple[list[dict], list[dict], dict]:
    plans, rows = [], []
    for seed in seeds_for(split):
        plan, bank = fit_and_freeze(seed)
        plans.append(plan)
        rows.extend(evaluate(seed, bank))
        print("completed drift-intensity seed", split, seed, flush=True)
    return plans, rows, statistics(rows)


def dump_rows(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", compresslevel=6) as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def run(output: Path, split: str) -> None:
    if output.exists():
        raise FileExistsError("fresh drift-intensity output required")
    output.mkdir(parents=True)
    frozen = protocol(split)
    write(output / "frozen_protocol.json", frozen)
    source_dir = output / "source"
    source_dir.mkdir()
    for name in frozen["source_sha256"]:
        (source_dir / Path(name).name).write_bytes(Path(name).read_bytes())
    (source_dir / "paper_at_freeze.tex").write_bytes(
        Path("CREDIT_WWW2027_v4.tex").read_bytes()
    )
    plans, rows, stats = compute(split)
    write(output / "frozen_plans.json", plans)
    dump_rows(output / "raw_episodes.jsonl.gz", rows)
    write(output / "statistics.json", stats)
    write(output / "summary.json", {
        "split": split,
        "seeds": list(seeds_for(split)),
        "intensities": list(INTENSITIES),
        "methods": ["CREDIT-Gated", "NoDriftGate", "NoMem"],
        "episodes": len(rows),
        "evaluation_api_calls": sum(row["api_calls"] for row in rows),
        "selection_api_calls": sum(
            plan["history_selection_calls"] + plan["target_probe_calls"]
            + plan["validation_calls"] for plan in plans
        ),
        "natural_provider_drift": False,
    })
    write(output / "artifact_manifest.json", {
        str(path.relative_to(output)): digest(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    })


def audit(output: Path, split: str) -> None:
    frozen = json.loads((output / "frozen_protocol.json").read_text())
    if frozen != protocol(split):
        raise ValueError("frozen protocol or current source differs")
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for name, expected in manifest.items():
        if digest(output / name) != expected:
            raise ValueError("artifact changed: " + name)
    plans, rows, stats = compute(split)
    stored_rows = load_rows(output / "raw_episodes.jsonl.gz")
    if plans != json.loads((output / "frozen_plans.json").read_text()):
        raise ValueError("selection reconstruction differs")
    if rows != stored_rows:
        raise ValueError("episode reconstruction differs")
    if stats != json.loads((output / "statistics.json").read_text()):
        raise ValueError("statistics reconstruction differs")
    report = {
        "verified": True,
        "split": split,
        "seeds": list(seeds_for(split)),
        "episodes_reexecuted": len(rows),
        "evaluation_api_calls_reexecuted": sum(row["api_calls"] for row in rows),
        "selection_plans_reconstructed": len(plans),
        "artifact_manifest_sha256": digest(output / "artifact_manifest.json"),
        "auditor_sha256": digest(Path(__file__)),
        "checks": [
            "source and artifact binding", "pre-evaluation bank freeze",
            "selection reconstruction", "paired task coverage",
            "complete episode replay", "statistics reconstruction",
        ],
        "limits": frozen["scope"],
    }
    write(output / "execution_audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("development", "confirmation"), required=True,
    )
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        audit(args.output, args.split)
    else:
        run(args.output, args.split)
