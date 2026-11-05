"""Budgeted uncertainty-per-call replay versus randomized replay."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from random import Random
from statistics import mean

import numpy as np

from .experiments import _shapley
from .model import StructuralModel
from .sandbox import (MEMORIES, MEMORY_ROLES, Task, Trace, make_tasks,
                      replay_pair, run_task)


def _proposals(tasks: list[Task], rng: Random, count: int = 16):
    result = []
    for _ in range(count):
        task = rng.choice(tasks)
        change = (tuple(rng.sample(MEMORIES[:7], 2)) if rng.random() < 0.25
                  else (rng.choice(MEMORIES),))
        background = frozenset(
            m for m in MEMORIES if m not in change and rng.random() < 0.5
        )
        first = next((i for i, role in enumerate(task.roles)
                      if any(role in MEMORY_ROLES[m] for m in change)),
                     len(task.roles))
        predicted_calls = 1 + len(task.roles) + max(0, len(task.roles) - first)
        result.append((task, background, frozenset(change), predicted_calls))
    return result


def _ensemble(traces: list[Trace], rng: Random) -> list[StructuralModel]:
    if not traces:
        return []
    return [
        StructuralModel(max_order=2, min_support=1).fit(
            [rng.choice(traces) for _ in traces]
        )
        for _ in range(4)
    ]


def _collect(tasks: list[Task], budget: int, *, seed: int, active: bool):
    rng = Random(seed)
    traces: list[Trace] = []
    calls = 0
    records = []
    models: list[StructuralModel] = []
    while budget - calls >= 9:
        slate = [item for item in _proposals(tasks, rng)
                 if item[3] <= budget - calls]
        if not slate:
            break
        if active and len(traces) >= 12 and len(records) % 8 == 0:
            models = _ensemble(traces, rng)
        if active and models:
            scores = []
            for task, background, change, expected_cost in slate:
                contrasts = [
                    (model.predict(task, background | change,
                                   require_support=False).mean
                     - model.predict(task, background,
                                     require_support=False).mean)
                    for model in models
                ]
                scores.append(float(np.var(contrasts, ddof=1)) / expected_cost)
            preferred = int(np.argmax(scores))
            exploration = rng.random() < 0.2
            chosen = rng.randrange(len(slate)) if exploration else preferred
            propensity = 0.2 / len(slate) + (0.8 if chosen == preferred else 0.0)
        else:
            chosen = rng.randrange(len(slate))
            propensity = 1 / len(slate)
        task, background, change, _ = slate[chosen]
        pair = replay_pair(task, background, background | change,
                           propensity=propensity)
        if pair.extra_calls > budget - calls:
            break
        calls += pair.extra_calls
        records.append(pair)
        if pair.valid:
            traces.extend((pair.control, pair.treatment))
    return records, traces, calls


def _attribution_mae(model: StructuralModel, reference: list[Task]) -> float:
    coalitions = [frozenset(group) for size in range(len(MEMORIES) + 1)
                  for group in combinations(MEMORIES, size)]
    true = {coalition: mean(int(run_task(task, coalition).outcome or 0)
                            for task in reference) for coalition in coalitions}
    estimate = {coalition: model.predict(reference[0], coalition,
                                        require_support=False).mean
                for coalition in coalitions}
    actual, predicted = _shapley(true), _shapley(estimate)
    return mean(abs(predicted[m] - actual[m]) for m in MEMORIES)


def run_active(seed: int, output: Path) -> list[dict]:
    tasks = make_tasks(80, split="active-training", version="v1",
                       templates=("short", "chain", "double_lookup"),
                       seed=seed + 1)
    reference = make_tasks(64, split="active-reference", version="v1",
                           templates=("new_order",), seed=seed + 2)
    rows = []
    for strategy in ("random", "uncertainty_per_call"):
        for budget in (16, 64, 256):
            pairs, traces, calls = _collect(
                tasks, budget, seed=seed + 3,
                active=strategy == "uncertainty_per_call"
            )
            model = StructuralModel(max_order=2, min_support=1).fit(traces)
            rows.append({
                "strategy": strategy, "call_budget": budget,
                "actual_calls": calls, "pairs": len(pairs),
                "minimum_propensity": min((p.propensity for p in pairs), default=0),
                "attr_mae": _attribution_mae(model, reference),
            })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "active_replay.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.7), layout="constrained")
    for strategy in ("random", "uncertainty_per_call"):
        selected = [row for row in rows if row["strategy"] == strategy]
        ax.plot([r["actual_calls"] for r in selected],
                [r["attr_mae"] for r in selected], marker="o", label=strategy)
    ax.set_xlabel("Additional simulated service calls")
    ax.set_ylabel("Shapley attribution MAE")
    ax.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"active_replay.{suffix}", dpi=300)
    plt.close(fig)
    (output / "active_report.json").write_text(json.dumps({
        "status": "measured synthetic active replay only",
        "seed": seed, "candidate_slate": 16, "exploration_probability": 0.2,
        "reference_tasks": len(reference),
        "limitation": ("Propensities are conditional on the sampled candidate slate; "
                       "this prototype does not estimate inclusion probabilities "
                       "over all possible task/coalition pairs."),
    }, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/active"))
    args = parser.parse_args()
    print(json.dumps(run_active(args.seed, args.output), indent=2))


if __name__ == "__main__":
    main()
