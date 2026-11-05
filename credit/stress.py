"""Sensitivity grid, drift severity, replay validity, and online blocks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from .experiments import _validation_traces, collect_pairs, replay_efficiency
from .model import StructuralModel
from .policy import bank_cost, optimize_bank, retrieve
from .sandbox import Task, make_tasks, run_task
from .voi import rank_revalidation


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _vsr(tasks: list[Task], bank: tuple[str, ...], depth: int) -> float:
    return mean(int(run_task(t, retrieve(t, bank, depth=depth)).outcome or 0)
                for t in tasks)


def run_stress(seed: int, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    historical = make_tasks(128, split="stress-train", version="v1",
                            templates=("short", "chain", "double_lookup"), seed=seed + 1)
    future_probes = make_tasks(80, split="stress-probes", version="v2",
                              templates=("short", "chain"), seed=seed + 2)
    validation_v1 = make_tasks(40, split="stress-validation", version="v1",
                               templates=("short", "chain"), seed=seed + 3)
    validation_v2 = make_tasks(24, split="stress-validation-v2", version="v2",
                               templates=("short", "chain"), seed=seed + 4)
    future = make_tasks(80, split="stress-future", version="v2",
                        templates=("new_order", "long"), seed=seed + 5)
    _, history_traces = collect_pairs(historical, 384, seed=seed + 6)
    probe_pairs, probe_traces = collect_pairs(future_probes, 120, seed=seed + 7)
    validation = _validation_traces(validation_v1 + validation_v2, seed=seed + 8)
    fit_cache = {}
    for call_budget in (16, 64, 256):
        used_pairs = []
        calls = 0
        for pair in probe_pairs:
            if calls + pair.extra_calls > call_budget:
                break
            used_pairs.append(pair)
            calls += pair.extra_calls
        training = history_traces + [
            trace for pair in used_pairs if pair.valid
            for trace in (pair.control, pair.treatment)
        ]
        for order in (1, 2, 3):
            fit_cache[(call_budget, order)] = (
                StructuralModel(max_order=order).fit(training).calibrate(validation),
                calls,
            )
    grid = []
    for bank_budget in (512, 2048, 8192):
        for depth in (4, 8, 16):
            for call_budget in (16, 64, 256):
                for order in (1, 2, 3):
                    model, actual_calls = fit_cache[(call_budget, order)]
                    bank, objective = optimize_bank(
                        model, validation_v1 + validation_v2,
                        budget=bank_budget, depth=depth,
                    )
                    grid.append({
                        "bank_budget_tokens": bank_budget,
                        "requested_exposure_depth": depth,
                        "actual_exposure_cap": min(depth, 8),
                        "probe_call_budget": call_budget,
                        "actual_probe_calls": actual_calls,
                        "max_interaction_order": order,
                        "bank": "|".join(bank),
                        "bank_tokens": bank_cost(bank),
                        "candidate_pool_tokens": 1024,
                        "objective": objective,
                        "post_drift_vsr": _vsr(future, bank, depth),
                    })
    full_model, _ = fit_cache[(256, 2)]
    bank, _ = optimize_bank(full_model, validation_v1 + validation_v2,
                            budget=512, depth=4)
    drift = []
    for intensity in (0.0, 0.5, 1.0, 1.5, 2.0):
        tasks = make_tasks(80, split=f"severity-{intensity}", version="v2",
                           templates=("new_order", "long"), seed=seed + 20,
                           drift_intensity=intensity)
        outcomes = []
        errors = []
        for task in tasks:
            exposure = retrieve(task, bank, depth=4)
            outcome = int(run_task(task, exposure).outcome or 0)
            pred = full_model.predict(task, exposure)
            outcomes.append(outcome)
            errors.append(abs(outcome - pred.mean))
        drift.append({
            "drift_intensity": intensity,
            "vsr": mean(outcomes),
            "mean_episode_absolute_error": mean(errors),
            "n": len(tasks),
        })
    replay = []
    for invalid_rate in (0.0, 0.25, 0.5, 0.75):
        tasks = make_tasks(100, split=f"replay-{invalid_rate}", version="v1",
                           templates=("short", "chain", "long"), seed=seed + 30,
                           invalid_rate=invalid_rate,
                           unrestorable_rate=invalid_rate / 2)
        summary = replay_efficiency(tasks, seed=seed + 31)
        replay.append({"unreplayable_rate": invalid_rate, **summary})
    _write(output / "sensitivity_grid.csv", grid)
    _write(output / "drift_severity.csv", drift)
    _write(output / "replay_validity.csv", replay)
    report = {
        "status": "measured synthetic sensitivity only",
        "seed": seed, "grid_points": len(grid),
        "selected_bank": list(bank),
        "saturation_note": (
            "Only eight candidate memories exist (1024 normalized tokens). "
            "Budgets 2048/8192 and exposure depth 16 therefore saturate; "
            "these rows do not test genuine large-bank scaling."
        ),
        "call_budget_note": "R is counted as additional simulated service calls.",
    }
    (output / "stress_report.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")
    return report


def run_online(seed: int, output: Path) -> list[dict]:
    """Score each block before using its outcomes for the next bank update."""
    history = make_tasks(96, split="online-history", version="v1",
                         templates=("short", "chain"), seed=seed + 1)
    _, initial_traces = collect_pairs(history, 256, seed=seed + 2)
    initial_model = StructuralModel(max_order=2).fit(initial_traces)
    initial_bank, _ = optimize_bank(initial_model, history[:32], budget=512, depth=4)
    states = {
        "CREDIT-VOI": {"traces": list(initial_traces), "bank": initial_bank},
        "CREDIT-random-probes": {"traces": list(initial_traces), "bank": initial_bank},
    }
    rows = []
    for block in range(6):
        version = "v1" if block < 3 else "v2"
        tasks = make_tasks(48, split=f"online-block-{block}", version=version,
                           templates=("new_order", "long"), seed=seed + 100 + block)
        for method, current_bank in (
            *((name, state["bank"]) for name, state in states.items()),
            ("FIFO", ("lookup_key", "transform_rule", "submit_field", "stale_hint")),
            ("NoMem", ()),
        ):
            outcomes = [int(run_task(task, retrieve(task, current_bank, depth=4)).outcome or 0)
                        for task in tasks]
            rows.append({
                "block": block, "version": version, "method": method,
                "vsr": mean(outcomes), "n": len(tasks),
                "bank": "|".join(current_bank),
                "bank_tokens": bank_cost(current_bank),
                "revalidation_calls_after_scoring": 0,
                "revalidation_queue": "",
                "best_voi": "",
            })
        for name, state in states.items():
            scored = [run_task(task, retrieve(task, state["bank"], depth=4))
                      for task in tasks]
            state["traces"].extend(scored)
            revalidation_calls = 0
            queue: tuple[str, ...] = ()
            best_voi: float | str = ""
            if version == "v2":
                if name == "CREDIT-VOI":
                    ranking = rank_revalidation(
                        state["traces"], history[:16] + tasks,
                        tasks[0], state["bank"], seed=seed + 300 + block,
                    )
                    best_voi = ranking[0]["voi"] if ranking else ""
                    queue = tuple(row["memory"] for row in ranking
                                  if row["voi"] > 0)[:3]
                pairs, new_traces = collect_pairs(
                    tasks, 32, seed=seed + 200 + block, prioritize=queue
                )
                state["traces"].extend(new_traces)
                revalidation_calls = sum(pair.extra_calls for pair in pairs)
            model = StructuralModel(max_order=2).fit(state["traces"])
            model.calibrate(scored)
            state["bank"], _ = optimize_bank(model, history[:16] + tasks,
                                              budget=512, depth=4)
            matched = next(row for row in rows
                           if row["block"] == block and row["method"] == name)
            matched["revalidation_calls_after_scoring"] = revalidation_calls
            matched["revalidation_queue"] = "|".join(queue)
            matched["best_voi"] = best_voi
    _write(output / "online_blocks.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/stress"))
    args = parser.parse_args()
    report = run_stress(args.seed, args.output)
    online = run_online(args.seed, args.output)
    print(json.dumps({
        "output": str(args.output), "grid_points": report["grid_points"],
        "online_blocks": len({r["block"] for r in online}),
    }, indent=2))


if __name__ == "__main__":
    main()
