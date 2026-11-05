"""Fair baseline completion for the revised CREDIT E1--E5 pilot tables.

The implementations follow the contracts already printed in
``CREDIT_WWW2027_Revised_Expanded.tex``.  They share the CREDIT-only pilot's
families, actor seeds, executors, memory pool, capacity, evaluation draws, and
physical-call accounting.  This is a finite synthetic comparison, not a claim
about live LLM or remote-service performance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.stats import norm

from . import revised_credit_only as co


BANK_CAPACITY = 4
BOOTSTRAP_REPS = 20_000
PAIR_THRESHOLD = co.E2_PAIR_THRESHOLD
PRIMARY_METHODS = ("Semantic", "Behavior-filter", "Direct-Sandbox")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def empty_blocks(families: list[co.Family]) -> dict[str, set[int]]:
    return {family.uid: set() for family in families}


def exact_version_blocks(families: list[co.Family]) -> dict[str, set[int]]:
    return {family.uid: set(family.stale) for family in families}


def training_logs(families: list[co.Family], draws: int, seed: int) -> tuple[list[dict], int]:
    """Reconstruct the same randomized physical evidence used by CREDIT."""
    rng = np.random.default_rng(seed)
    rows, calls = [], 0
    order = 0
    for family in families:
        for template in range(co.TEMPLATES):
            for _ in range(draws):
                exposure = tuple(int(x) for x in np.flatnonzero(rng.random(co.MEMORIES) < .5))
                outcome = bool(rng.random() < co.probability(family, exposure))
                rows.append({"order": order, "family": family.uid, "template": template,
                             "exposure": exposure, "outcome": outcome})
                calls += family.depth
                order += 1
    return rows, calls


def cosine_role(memory: int, family: co.Family) -> float:
    return (1 / math.sqrt(len(family.roles))
            if int(co.MEMORY_ROLE[memory]) in family.roles else 0.0)


def semantic_bank(families: list[co.Family]) -> tuple[int, ...]:
    """Greedy mean maximum cosine coverage with one-hot operation-role embeddings."""
    bank: list[int] = []
    for _ in range(BANK_CAPACITY):
        candidates = []
        for memory in range(co.MEMORIES):
            if memory in bank:
                continue
            trial = [*bank, memory]
            score = np.mean([max(cosine_role(m, family) for m in trial)
                             for family in families])
            candidates.append((float(score), -memory, memory))
        bank.append(max(candidates)[-1])
    return tuple(bank)


def coverage_mmr_bank(families: list[co.Family], zeta: float = .15) -> tuple[int, ...]:
    bank: list[int] = []
    for _ in range(BANK_CAPACITY):
        candidates = []
        for memory in range(co.MEMORIES):
            if memory in bank:
                continue
            trial_roles = {int(co.MEMORY_ROLE[m]) for m in [*bank, memory]}
            coverage = np.mean([sum(role in trial_roles for role in family.roles)
                                / len(family.roles) for family in families])
            redundancy = max([float(co.MEMORY_ROLE[memory] == co.MEMORY_ROLE[m])
                              for m in bank] or [0.0])
            candidates.append((float(coverage - zeta * redundancy), -memory, memory))
        bank.append(max(candidates)[-1])
    return tuple(bank)


def outcome_weighted_bank(families: list[co.Family], logs: list[dict]) -> tuple[int, ...]:
    exposed = np.zeros(co.MEMORIES)
    successes = np.zeros(co.MEMORIES)
    for row in logs:
        for memory in row["exposure"]:
            exposed[memory] += 1
            successes[memory] += row["outcome"]
    mean_similarity = np.asarray([
        np.mean([cosine_role(memory, family) for family in families])
        for memory in range(co.MEMORIES)])
    scores = mean_similarity * (successes + 1) / (exposed + 2)
    return tuple(sorted(range(co.MEMORIES), key=lambda m: (-scores[m], m))[:BANK_CAPACITY])


def recency_bank(logs: list[dict]) -> tuple[int, ...]:
    latest = np.full(co.MEMORIES, -1, dtype=int)
    for row in logs:
        if row["outcome"]:
            for memory in row["exposure"]:
                latest[memory] = row["order"]
    return tuple(sorted(range(co.MEMORIES), key=lambda m: (-latest[m], m))[:BANK_CAPACITY])


def aggregate_effects(logs: list[dict]) -> tuple[float, np.ndarray]:
    y = np.asarray([row["outcome"] for row in logs], dtype=float)
    z = np.asarray([[float(m in row["exposure"]) for m in range(co.MEMORIES)]
                    for row in logs])
    effects = np.asarray([y[z[:, m] == 1].mean() - y[z[:, m] == 0].mean()
                          for m in range(co.MEMORIES)])
    intercept = float(y.mean() - .5 * effects.sum())
    return intercept, effects


def effect_predictor(intercept: float, effects: np.ndarray) -> Callable[[co.Family, tuple[int, ...]], float]:
    def predict(family: co.Family, exposure: tuple[int, ...]) -> float:
        value = intercept + sum(effects[m] for m in exposure if co.applicable(family, m))
        return float(np.clip(value, .02, .98))
    return predict


def cmp_bank(logs: list[dict]) -> tuple[tuple[int, ...], Callable]:
    intercept, effects = aggregate_effects(logs)
    positives = sorted(range(co.MEMORIES), key=lambda m: (-effects[m], m))
    return tuple(positives[:BANK_CAPACITY]), effect_predictor(intercept, effects)


def physical_marginals(families: list[co.Family], call_budget: int, seed: int,
                       *, shapley: bool) -> tuple[np.ndarray, int, list[int]]:
    rng = np.random.default_rng(seed)
    sums = np.zeros(co.MEMORIES)
    counts = np.zeros(co.MEMORIES, dtype=int)
    calls = 0
    family_index = 0
    while True:
        family = families[family_index % len(families)]
        family_index += 1
        order = list(rng.permutation(co.MEMORIES)) if shapley else list(range(co.MEMORIES))
        background: list[int] = []
        progressed = False
        for memory in order:
            before = tuple(background if shapley else [])
            after = tuple([*before, memory])
            cost = 2 * family.depth
            if calls + cost > call_budget:
                return sums / np.maximum(1, counts), calls, counts.tolist()
            u = rng.random()
            delta = float(u < co.probability(family, after)) - float(u < co.probability(family, before))
            sums[memory] += delta
            counts[memory] += 1
            calls += cost
            progressed = True
            if shapley:
                background.append(memory)
        if not progressed:
            break
    return sums / np.maximum(1, counts), calls, counts.tolist()


def select_effect_bank(effects: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(range(co.MEMORIES), key=lambda m: (-effects[m], m))[:BANK_CAPACITY])


def flat_features(family: co.Family, exposure: tuple[int, ...]) -> np.ndarray:
    selected = set(exposure)
    main = [float(m in selected and co.applicable(family, m)) for m in range(co.MEMORIES)]
    pairs = [float(i in selected and j in selected and co.applicable(family, i)
                   and co.applicable(family, j)) for i, j in co.MODELED_PAIRS]
    return np.asarray([1.0, *main, *pairs])


def fit_flat(train: list[co.Family], draws: int, seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    rows, outcomes, calls = [], [], 0
    for family in train:
        for _template in range(co.TEMPLATES):
            for _ in range(draws):
                exposure = tuple(int(x) for x in np.flatnonzero(rng.random(co.MEMORIES) < .5))
                rows.append(flat_features(family, exposure))
                outcomes.append(float(rng.random() < co.probability(family, exposure)))
                calls += family.depth
    x, y = np.vstack(rows), np.asarray(outcomes)
    penalty = np.eye(x.shape[1]) * 1e-5
    penalty[0, 0] = 0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y), calls


def flat_predictor(model: np.ndarray) -> Callable[[co.Family, tuple[int, ...]], float]:
    return lambda family, exposure: float(np.clip(flat_features(family, exposure) @ model, .02, .98))


def select_by_predictor(predict: Callable, families: list[co.Family],
                        blocked: dict[str, set[int]]) -> tuple[int, ...]:
    scored = []
    for bank in co.bank_candidates():
        values = np.asarray([predict(family, co.retrieve(bank, family, blocked))
                             for family in families for _ in range(co.TEMPLATES)])
        score = .7 * values.mean() + .3 * np.quantile(values, .10) - .0005 * len(bank)
        scored.append((float(score), -len(bank), tuple(-x for x in bank), bank))
    return max(scored)[-1]


def generic_ppi_certificate(predict: Callable, candidate: tuple[int, ...],
                            families: list[co.Family], blocked: dict[str, set[int]],
                            n: int, seed: int) -> tuple[bool, int]:
    rng = np.random.default_rng(seed)
    estimates, calls = [], 0
    for stratum in range(4):
        diffs, forecasts = [], []
        pool = [family for i, family in enumerate(families) if i % 4 == stratum]
        for index in range(n):
            family = pool[index % len(pool)]
            exposure = co.retrieve(candidate, family, blocked)
            p0, p1 = co.probability(family, ()), co.probability(family, exposure)
            u = rng.random()
            diffs.append(float(u < p1) - float(u < p0))
            forecasts.append(predict(family, exposure) - predict(family, ()))
            calls += 2 * family.depth
        y, f = np.asarray(diffs), np.asarray(forecasts)
        unlabeled = np.asarray([predict(family, co.retrieve(candidate, family, blocked))
                                - predict(family, ()) for family in pool])
        estimate = float(unlabeled.mean() + np.mean(y - f))
        se = float(np.std(y - f, ddof=1) / math.sqrt(n))
        estimates.append((estimate, se))
    mean = np.mean([x[0] for x in estimates])
    se = math.sqrt(sum((.25 * x[1]) ** 2 for x in estimates))
    aggregate = mean - norm.ppf(.975) * se
    guard_z = norm.ppf(1 - .025 / 4)
    passed = aggregate > 0 and min(m - guard_z * s for m, s in estimates) > -.05
    return bool(passed), calls


def direct_certificate(candidate: tuple[int, ...], families: list[co.Family],
                       n: int, seed: int) -> tuple[bool, int]:
    rng = np.random.default_rng(seed)
    estimates, calls = [], 0
    blocks = empty_blocks(families)
    for stratum in range(4):
        diffs = []
        pool = [family for i, family in enumerate(families) if i % 4 == stratum]
        for index in range(n):
            family = pool[index % len(pool)]
            exposure = co.retrieve(candidate, family, blocks)
            u = rng.random()
            diffs.append(float(u < co.probability(family, exposure))
                         - float(u < co.probability(family, ())))
            calls += 2 * family.depth
        y = np.asarray(diffs)
        estimates.append((float(y.mean()), float(y.std(ddof=1) / math.sqrt(n))))
    mean = np.mean([x[0] for x in estimates])
    se = math.sqrt(sum((.25 * x[1]) ** 2 for x in estimates))
    aggregate = mean - norm.ppf(.975) * se
    guard_z = norm.ppf(1 - .025 / 4)
    return bool(aggregate > 0 and min(m - guard_z * s for m, s in estimates) > -.05), calls


def direct_select(train: list[co.Family], call_budget: int, seed: int) -> tuple[tuple[int, ...], int, dict]:
    """Successive halving over the same capacity-four candidate set."""
    rng = np.random.default_rng(seed)
    active = list(co.bank_candidates())
    sums = defaultdict(float)
    counts = defaultdict(int)
    calls = 0
    stages = max(1, math.ceil(math.log2(len(active))))
    stage_budget = call_budget / stages
    history = []
    for stage in range(stages):
        if len(active) == 1:
            break
        stage_start = calls
        cursor = 0
        while calls - stage_start < stage_budget:
            bank = active[cursor % len(active)]
            family = train[int(rng.integers(len(train)))]
            cost = 2 * family.depth
            if calls + cost > call_budget:
                break
            exposure = co.retrieve(bank, family, empty_blocks([family]))
            u = rng.random()
            delta = float(u < co.probability(family, exposure)) - float(u < co.probability(family, ()))
            sums[bank] += delta
            counts[bank] += 1
            calls += cost
            cursor += 1
        active.sort(key=lambda bank: (-(sums[bank] / max(1, counts[bank])), len(bank), bank))
        keep = max(1, math.ceil(len(active) / 2))
        history.append({"stage": stage, "evaluated": len(active), "kept": keep,
                        "cumulative_calls": calls})
        active = active[:keep]
        if calls >= call_budget:
            break
    active.sort(key=lambda bank: (-(sums[bank] / max(1, counts[bank])), len(bank), bank))
    return active[0], calls, {"stages": history,
                              "observations": int(sum(counts.values()))}


def policy(bank: tuple[int, ...], blocked: dict[str, set[int]]) -> Callable[[co.Family], tuple[int, ...]]:
    return lambda family: co.retrieve(bank, family, blocked)


def score_policy(name: str, select: Callable[[co.Family], tuple[int, ...]],
                 families: list[co.Family], predictor: Callable | None,
                 adaptation_calls: int, seed: int = 9601) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    rows, latencies, forecast_errors = [], [], []
    stale = exposures = eval_calls = 0
    for family in families:
        exposed = tuple(select(family))
        if predictor is not None:
            forecast_errors.append(abs(predictor(family, exposed)
                                       - co.probability(family, exposed)))
        for template in range(co.TEMPLATES):
            for actor in co.ACTOR_SEEDS:
                u = rng.random()
                base_p = co.probability(family, exposed)
                guard_p = co.probability(family, exposed, guard=True)
                latency = float(family.depth * rng.lognormal(math.log(34), .18)
                                + 1.5 * len(exposed) + .25)
                row = {"method": name, "family": family.uid, "template": template,
                       "actor_seed": actor, "exposure": list(exposed),
                       "base_success": bool(u < base_p),
                       "guard_success": bool(u < guard_p),
                       "base_probability": base_p, "guard_probability": guard_p,
                       "accounted_latency_ms": latency}
                rows.append(row)
                latencies.append(latency)
                exposures += len(exposed)
                stale += sum(m in family.stale for m in exposed)
                eval_calls += 2 * family.depth
    return {
        "method": name,
        "episodes_per_executor": len(rows),
        "base_successes": sum(row["base_success"] for row in rows),
        "guard_successes": sum(row["guard_success"] for row in rows),
        "base_vsr": np.mean([row["base_success"] for row in rows]),
        "guard_vsr": np.mean([row["guard_success"] for row in rows]),
        "stale_exposures": stale,
        "all_exposures": exposures,
        "stale_exposure_rate": stale / max(1, exposures),
        "forecast_mae": (float(np.mean(forecast_errors)) if forecast_errors else None),
        "adaptation_calls": adaptation_calls,
        "evaluation_calls_two_executors": eval_calls,
        "total_cost": adaptation_calls + eval_calls,
        "accounted_latency_p95_ms": float(np.quantile(latencies, .95)),
    }, rows


def family_differences(rows_a: list[dict], rows_b: list[dict]) -> np.ndarray:
    by_a, by_b = defaultdict(list), defaultdict(list)
    for row in rows_a:
        by_a[row["family"]].append(float(row["base_success"]))
    for row in rows_b:
        by_b[row["family"]].append(float(row["base_success"]))
    keys = sorted(by_a)
    return np.asarray([np.mean(by_a[key]) - np.mean(by_b[key]) for key in keys])


def bootstrap_ci(differences: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(differences), size=(BOOTSTRAP_REPS, len(differences)))
    values = differences[indices].mean(axis=1)
    return [float(x) for x in np.quantile(values, [.025, .975])]


def exact_sign_flip_p(differences: np.ndarray) -> float:
    units = np.rint(differences * 32).astype(int)
    units = units[units != 0]
    counts = Counter({0: 1})
    for value in np.abs(units):
        nxt = Counter()
        for total, count in counts.items():
            nxt[total + int(value)] += count
            nxt[total - int(value)] += count
        counts = nxt
    observed = abs(int(units.sum()))
    return sum(count for total, count in counts.items() if abs(total) >= observed) / (2 ** len(units))


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted, running = {}, 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * raw[key]))
        adjusted[key] = running
    return adjusted


@dataclass
class MethodRun:
    summary: dict
    rows: list[dict]
    bank: tuple[int, ...]
    predictor: Callable | None
    blocked: dict[str, set[int]]


def e1_methods(train: list[co.Family], heldout: list[co.Family]) -> tuple[dict, list[dict], dict]:
    logs, evidence_calls = training_logs(train, co.BUDGET_DRAWS[co.PRIMARY_BUDGET], 9401)
    no_blocks = empty_blocks(heldout)
    version_blocks = exact_version_blocks(heldout)
    behavior_blocks, behavior_gate = co.gate_decisions(heldout, 9401)
    semantic = semantic_bank(train)
    coverage = coverage_mmr_bank(train)
    outcome = outcome_weighted_bank(train, logs)
    fifo = tuple(range(BANK_CAPACITY))
    recency = recency_bank(logs)
    cmp, cmp_predict = cmp_bank(logs)
    mc_effects, mc_calls, mc_counts = physical_marginals(train, evidence_calls, 10101, shapley=True)
    mc = select_effect_bank(mc_effects)
    mc_predict = effect_predictor(.48, mc_effects)
    car_effects, car_calls, car_counts = physical_marginals(train, evidence_calls, 10201, shapley=False)
    car = select_effect_bank(car_effects)
    car_predict = effect_predictor(.48, car_effects)
    flat_model, flat_calls = fit_flat(train, co.BUDGET_DRAWS[co.PRIMARY_BUDGET], 9401)
    flat_predict = flat_predictor(flat_model)
    flat_candidate = select_by_predictor(flat_predict, heldout, behavior_blocks)
    flat_passed, flat_cert_calls = generic_ppi_certificate(
        flat_predict, flat_candidate, heldout, behavior_blocks,
        co.CERT_N[co.PRIMARY_BUDGET], 9601)
    flat = flat_candidate if flat_passed else ()
    direct_candidate, direct_calls, direct_trace = direct_select(train, evidence_calls, 10301)
    direct_passed, direct_cert_calls = direct_certificate(
        direct_candidate, heldout, co.CERT_N[co.PRIMARY_BUDGET], 10401)
    direct = direct_candidate if direct_passed else ()

    specifications = [
        ("NoMem", (), no_blocks, None, 0),
        ("FIFO", fifo, no_blocks, None, 0),
        ("Recency", recency, no_blocks, None, 0),
        ("Semantic", semantic, no_blocks, None, 0),
        ("Outcome-Weighted", outcome, no_blocks, cmp_predict, evidence_calls),
        ("Coverage-MMR", coverage, no_blocks, None, 0),
        ("Version-filter", semantic, version_blocks, None, 0),
        ("Behavior-filter", semantic, behavior_blocks, None, behavior_gate["probe_calls"]),
        ("MC-Shapley", mc, no_blocks, mc_predict, mc_calls),
        ("CMP-adapted", cmp, no_blocks, cmp_predict, evidence_calls),
        ("CAR-adapted", car, no_blocks, car_predict, car_calls),
        ("Flat-surrogate", flat, behavior_blocks, flat_predict,
         flat_calls + behavior_gate["probe_calls"] + flat_cert_calls),
        ("Direct-Sandbox", direct, no_blocks, None, direct_calls + direct_cert_calls),
    ]
    methods: dict[str, MethodRun] = {}
    all_rows: list[dict] = []
    for name, bank, blocked, predictor, calls in specifications:
        summary, rows = score_policy(name, policy(tuple(bank), blocked), heldout,
                                     predictor, int(calls))
        summary["bank"] = list(bank)
        methods[name] = MethodRun(summary, rows, tuple(bank), predictor, blocked)
        all_rows.extend(rows)

    credit_result, credit_rows = co.full_credit_run(train, heldout, co.PRIMARY_BUDGET)
    credit_predict_model = co.fit_model(*co.collect_training(
        train, co.BUDGET_DRAWS[co.PRIMARY_BUDGET], 9401)[:2])
    def credit_predict(family: co.Family, exposure: tuple[int, ...]) -> float:
        return co.predicted(credit_predict_model, family, exposure)
    credit_summary = {"method": "Full CREDIT", **credit_result}
    methods["Full CREDIT"] = MethodRun(credit_summary, credit_rows,
                                        tuple(credit_result["deployed_bank"]),
                                        credit_predict, behavior_blocks)
    all_rows.extend({"method": "Full CREDIT", **row} for row in credit_rows)

    semantic_rows = methods["Semantic"].rows
    comparisons = {}
    for index, (name, run) in enumerate(methods.items()):
        if name == "Semantic":
            comparisons[name] = {"change_vs_semantic": 0.0, "ci": [0.0, 0.0],
                                 "sign_flip_p": 1.0}
            continue
        diff = family_differences(run.rows, semantic_rows)
        comparisons[name] = {"change_vs_semantic": float(diff.mean()),
                             "ci": bootstrap_ci(diff, 11001 + index),
                             "sign_flip_p": exact_sign_flip_p(diff),
                             "family_differences": diff.tolist()}
    raw_primary = {name: exact_sign_flip_p(family_differences(methods["Full CREDIT"].rows,
                                                               methods[name].rows))
                   for name in PRIMARY_METHODS}
    primary = {"raw": raw_primary, "holm": holm(raw_primary)}
    diagnostics = {
        "shared_evidence_calls": evidence_calls,
        "semantic_bank": list(semantic), "coverage_mmr_bank": list(coverage),
        "outcome_weighted_bank": list(outcome), "recency_bank": list(recency),
        "mc_counts": mc_counts, "car_counts": car_counts,
        "flat_candidate": list(flat_candidate), "flat_passed": flat_passed,
        "direct_candidate": list(direct_candidate), "direct_passed": direct_passed,
        "direct_trace": direct_trace, "primary_credit_tests": primary,
    }
    serial = {name: {**run.summary, **comparisons[name]} for name, run in methods.items()}
    return serial, all_rows, diagnostics


def model_matrix(z: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    k = z.shape[1]
    pairs = list(itertools.combinations(range(k), 2))
    return np.column_stack([np.ones(len(z)), z,
                            *[z[:, i] * z[:, j] for i, j in pairs]]), pairs


def ols_estimate(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, _ = model_matrix(z)
    inverse = np.linalg.pinv(x.T @ x)
    coef = inverse @ x.T @ y
    residual = y - x @ coef
    variance = float(residual @ residual / max(1, len(y) - x.shape[1]))
    return coef, np.sqrt(np.maximum(0, variance * np.diag(inverse)))


def fraction_rows(k: int, calls: int, rng: np.random.Generator) -> np.ndarray:
    full = co.all_coalitions(k)
    block_size = min(64, len(full))
    if len(full) == block_size:
        base = full.copy()
    else:
        while True:
            base = full[rng.choice(len(full), block_size, replace=False)]
            if np.linalg.matrix_rank(model_matrix(base)[0]) == model_matrix(base)[0].shape[1]:
                break
    rows = []
    while len(rows) < calls:
        block = base.copy()
        block *= rng.choice([-1., 1.], size=k)
        block = block[rng.permutation(len(block))]
        rows.extend(block.tolist())
    return np.asarray(rows[:calls])


def oat_rows(k: int, calls: int, rng: np.random.Generator) -> np.ndarray:
    pairs = list(itertools.combinations(range(k), 2))
    rows = []
    while len(rows) < calls:
        for memory in range(k):
            background = rng.choice([-1., 1.], size=k)
            for sign in (-1., 1.):
                row = background.copy()
                row[memory] = sign
                rows.append(row)
        for i, j in pairs:
            background = rng.choice([-1., 1.], size=k)
            for si, sj in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)):
                row = background.copy()
                row[i] = si
                row[j] = sj
                rows.append(row)
        if len(rows) >= calls:
            break
    return np.asarray(rows[:calls])


def mc_shapley_estimate(k: int, truth_fn: Callable[[np.ndarray], np.ndarray], calls: int,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float, int]:
    marginals = [[] for _ in range(k)]
    empty_draws = []
    used = 0
    while used + k + 1 <= calls:
        permutation = list(rng.permutation(k))
        state = -np.ones(k)
        values = [float(truth_fn(state[None, :])[0])]
        used += 1
        for memory in permutation:
            state = state.copy()
            state[memory] = 1
            values.append(float(truth_fn(state[None, :])[0]))
            used += 1
        draws = (rng.random(k + 1) < np.asarray(values)).astype(float)
        empty_draws.append(float(draws[0]))
        for position, memory in enumerate(permutation):
            marginals[memory].append(float(draws[position + 1] - draws[position]))
    estimates = np.asarray([np.mean(x) for x in marginals])
    ses = np.asarray([np.std(x, ddof=1) / math.sqrt(len(x)) for x in marginals])
    return estimates, ses, float(np.mean(empty_draws)), used


def e2_all_methods(seed: int = 12001) -> tuple[dict, list[dict]]:
    master = np.random.default_rng(seed)
    methods = ("Randomized fraction", "Uniform + OLS", "One-at-a-time",
               "MC-Shapley", "CMP-adapted")
    accum = {method: {"b2": [], "s2": [], "triple": [], "cover": [],
                      "tp": 0, "fp": 0, "fn": 0, "calls": 0}
             for method in methods}
    records = []
    family_index = 0
    for k in (4, 6, 8):
        coalitions = co.all_coalitions(k)
        x_all, pair_list = model_matrix(coalitions)
        for condition in ("additive", "redundant", "pair-complement", "pair-conflict", "triple"):
            for replicate in range(12):
                main = master.choice([-1, 1], k) * master.uniform(.012, .026, k)
                pairs = np.zeros(len(pair_list))
                triple = None
                triple_beta = 0.0
                if condition == "redundant":
                    pairs[pair_list.index((0, 1))] = -.035
                elif condition == "pair-complement":
                    pairs[pair_list.index((0, 1))] = .045
                elif condition == "pair-conflict":
                    pairs[pair_list.index((0, 1))] = -.045
                elif condition == "triple":
                    triple = (0, 1, 2)
                    triple_beta = .050

                def truth_fn(z: np.ndarray) -> np.ndarray:
                    value = .5 + z @ main
                    value += sum(beta * z[:, i] * z[:, j]
                                 for beta, (i, j) in zip(pairs, pair_list))
                    if triple is not None:
                        value += triple_beta * np.prod(z[:, triple], axis=1)
                    return value

                truth = truth_fn(coalitions)
                reference_rng = np.random.default_rng(13001 + family_index)
                reference = reference_rng.binomial(co.E2_REFERENCE_REPS, truth) / co.E2_REFERENCE_REPS
                target_b = co.banzhaf(reference, coalitions, k)
                target_s = co.shapley(reference, coalitions, k)
                target_b_true = co.banzhaf(truth, coalitions, k)
                target_s_true = co.shapley(truth, coalitions, k)
                actual_pairs = np.abs(pairs) >= .03

                for method_index, method in enumerate(methods):
                    rng = np.random.default_rng(14001 + family_index * 10 + method_index)
                    if method == "Randomized fraction":
                        z = fraction_rows(k, co.E2_ESTIMATION_CALLS, rng)
                        y = (rng.random(len(z)) < truth_fn(z)).astype(float)
                        coef, se = ols_estimate(z, y)
                        predicted = x_all @ coef
                        pair_estimates = coef[1 + k:]
                        estimate_b = co.banzhaf(predicted, coalitions, k)
                        estimate_s = co.shapley(predicted, coalitions, k)
                        covered = np.abs(estimate_b - target_b_true) <= 1.96 * (2 * se[1:1 + k])
                        calls = len(z)
                    elif method == "Uniform + OLS":
                        z = rng.choice([-1., 1.], size=(co.E2_ESTIMATION_CALLS, k))
                        y = (rng.random(len(z)) < truth_fn(z)).astype(float)
                        coef, se = ols_estimate(z, y)
                        predicted = x_all @ coef
                        pair_estimates = coef[1 + k:]
                        estimate_b = co.banzhaf(predicted, coalitions, k)
                        estimate_s = co.shapley(predicted, coalitions, k)
                        covered = np.abs(estimate_b - target_b_true) <= 1.96 * (2 * se[1:1 + k])
                        calls = len(z)
                    elif method == "One-at-a-time":
                        z = oat_rows(k, co.E2_ESTIMATION_CALLS, rng)
                        y = (rng.random(len(z)) < truth_fn(z)).astype(float)
                        coef, se = ols_estimate(z, y)
                        predicted = x_all @ coef
                        pair_estimates = coef[1 + k:]
                        estimate_b = co.banzhaf(predicted, coalitions, k)
                        estimate_s = co.shapley(predicted, coalitions, k)
                        covered = np.abs(estimate_b - target_b_true) <= 1.96 * (2 * se[1:1 + k])
                        calls = len(z)
                    elif method == "MC-Shapley":
                        estimate_s, se_s, empty, calls = mc_shapley_estimate(
                            k, truth_fn, co.E2_ESTIMATION_CALLS, rng)
                        predicted = np.clip(empty + ((coalitions + 1) / 2) @ estimate_s, .02, .98)
                        estimate_b = co.banzhaf(predicted, coalitions, k)
                        estimate_s = co.shapley(predicted, coalitions, k)
                        pair_estimates = np.zeros(len(pair_list))
                        covered = np.abs(estimate_s - target_s_true) <= 1.96 * se_s
                    else:
                        z = rng.choice([-1., 1.], size=(co.E2_ESTIMATION_CALLS, k))
                        y = (rng.random(len(z)) < truth_fn(z)).astype(float)
                        present = z > 0
                        effects = np.asarray([y[present[:, m]].mean() - y[~present[:, m]].mean()
                                              for m in range(k)])
                        ses = np.asarray([math.sqrt(y[present[:, m]].var(ddof=1) / present[:, m].sum()
                                                   + y[~present[:, m]].var(ddof=1) / (~present[:, m]).sum())
                                          for m in range(k)])
                        intercept = float(y.mean() - .5 * effects.sum())
                        predicted = np.clip(intercept + ((coalitions + 1) / 2) @ effects, .02, .98)
                        estimate_b = co.banzhaf(predicted, coalitions, k)
                        estimate_s = co.shapley(predicted, coalitions, k)
                        pair_estimates = np.zeros(len(pair_list))
                        covered = np.abs(estimate_b - target_b_true) <= 1.96 * ses
                        calls = len(z)

                    detected = np.abs(pair_estimates) >= PAIR_THRESHOLD
                    stats = accum[method]
                    stats["b2"].extend((estimate_b - target_b) ** 2)
                    stats["s2"].extend((estimate_s - target_s) ** 2)
                    if condition == "triple":
                        stats["triple"].append(float(np.mean(np.abs(predicted - reference))))
                    stats["cover"].extend(covered.tolist())
                    stats["tp"] += int(np.sum(actual_pairs & detected))
                    stats["fp"] += int(np.sum(~actual_pairs & detected))
                    stats["fn"] += int(np.sum(actual_pairs & ~detected))
                    stats["calls"] += calls
                    records.append({"family": family_index, "k": k, "condition": condition,
                                    "method": method,
                                    "banzhaf_rmse": float(np.sqrt(np.mean((estimate_b-target_b)**2))),
                                    "shapley_rmse": float(np.sqrt(np.mean((estimate_s-target_s)**2))),
                                    "coverage": float(np.mean(covered)), "calls": calls})
                family_index += 1

    summary = {}
    for method, stats in accum.items():
        precision = stats["tp"] / max(1, stats["tp"] + stats["fp"])
        recall = stats["tp"] / max(1, stats["tp"] + stats["fn"])
        summary[method] = {
            "families": family_index,
            "banzhaf_rmse": float(np.sqrt(np.mean(stats["b2"]))),
            "shapley_rmse": float(np.sqrt(np.mean(stats["s2"]))),
            "pair_f1": float(2 * precision * recall / max(1e-12, precision + recall)),
            "pair_precision": precision, "pair_recall": recall,
            "triple_forecast_mae": float(np.mean(stats["triple"])),
            "coverage": float(np.mean(stats["cover"])),
            "estimation_calls": int(stats["calls"]),
            "reference_calls": 2_580_480,
        }
    return summary, records


def observed_gate_metrics(mode: str) -> dict:
    opportunities, changed, stable = 24, 9, 15
    if mode == "accept-all":
        accepted_stable, false, abstained = stable, changed, 0
    elif mode == "block-version":
        accepted_stable, false, abstained = 0, 0, opportunities
    elif mode == "differential":
        accepted_stable, false, abstained = stable, 0, changed
    else:
        raise ValueError(mode)
    return {"observed_change_proxy_yield": accepted_stable / opportunities,
            "false_transfer_rate": false / changed,
            "abstention_rate": abstained / opportunities,
            "observed_opportunities": opportunities}


def e3_from_e1(e1: dict, train: list[co.Family], heldout: list[co.Family]) -> dict:
    modes = {"Semantic": "accept-all", "Version-filter": "block-version",
             "Behavior-filter": "differential", "Flat-surrogate": "differential",
             "CREDIT without gate": "accept-all", "Full CREDIT": "differential"}
    mapping = {"Semantic": "Semantic", "Version-filter": "Version-filter",
               "Behavior-filter": "Behavior-filter", "Flat-surrogate": "Flat-surrogate",
               "Full CREDIT": "Full CREDIT"}
    credit_result, _ = co.full_credit_run(train, heldout, co.PRIMARY_BUDGET)
    x, y, _calls = co.collect_training(train, co.BUDGET_DRAWS[co.PRIMARY_BUDGET], 9401)
    credit_model = co.fit_model(x, y)

    def predict(family: co.Family, exposure: tuple[int, ...]) -> float:
        return co.predicted(credit_model, family, exposure)

    no_gate_summary, _ = score_policy(
        "CREDIT without gate",
        policy(tuple(credit_result["candidate_bank"]), empty_blocks(heldout)),
        heldout, predict, credit_result["training_calls"])
    output = {}
    for name, mode in modes.items():
        source = no_gate_summary if name == "CREDIT without gate" else e1[mapping[name]]
        gate = observed_gate_metrics(mode)
        probe_calls = 48 if mode == "differential" else 0
        synthetic_gate_calls = 264 if name in ("Behavior-filter", "Flat-surrogate", "Full CREDIT") else 0
        output[name] = {
            "new_dag_vsr": source["base_vsr"],
            **gate,
            "forecast_mae": (source.get("forecast_mae")
                             if name in ("Flat-surrogate", "CREDIT without gate", "Full CREDIT") else None),
            "calls": source["evaluation_calls_two_executors"] // 2
                     + probe_calls + synthetic_gate_calls,
            "scope": "two-application gate-decision proxy; not agent VSR",
        }
    return output


def direct_normal_decision(rng: np.random.Generator, gains: np.ndarray,
                           reps: int, n: int, alpha: float, margin: float) -> np.ndarray:
    means, variances = co.draw_paired_mean(rng, gains, reps, n)
    weights = np.asarray([.20, .16, .14, .12, .10, .10, .10, .08])
    ses = np.sqrt(variances / n)
    aggregate = means @ weights
    aggregate_se = np.sqrt((ses ** 2) @ (weights ** 2))
    lower = aggregate - norm.ppf(1 - alpha / 2) * aggregate_se
    guard = means - norm.ppf(1 - alpha / 16) * ses
    return (lower > 0) & np.all(guard > -margin, axis=1)


def direct_stream(gains: np.ndarray, *, cycles: int, seed: int,
                  reps: int = 2000, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    switched = np.zeros(reps, dtype=bool)
    for cycle in range(1, cycles + 1):
        alpha = .05 if cycles == 1 else .05 / (cycle * (cycle + 1))
        switched |= direct_normal_decision(rng, gains, reps, n, alpha, .05)
    return switched


def e5_all_rules() -> dict:
    existing, _ = co.e5_certification()
    harmful = np.full(8, .03)
    harmful[-1] = -.10
    direct_null = direct_stream(np.zeros(8), cycles=1, seed=15001)
    direct_harm = direct_stream(harmful, cycles=1, seed=15011)
    direct_repeat = direct_stream(np.zeros(8), cycles=50, seed=15021)
    direct_power = direct_stream(np.full(8, .02), cycles=1, seed=15031)
    direct = {"false_improvement": float(direct_null.mean()),
              "guard_violation": float(direct_harm.mean()),
              "any_cycle_error_t50": float(direct_repeat.mean()),
              "power_two_point_gain": float(direct_power.mean()),
              "keep_rate_null": float(1 - direct_null.mean()),
              "certification_calls_per_cycle": 3200, "replications": 2000,
              "margin": .05}
    no_cert = {"false_improvement": 1.0, "guard_violation": 1.0,
               "any_cycle_error_t50": 1.0, "power_two_point_gain": 1.0,
               "keep_rate_null": 0.0, "certification_calls_per_cycle": 0,
               "replications": 2000, "margin": .05}
    return {"No certification": no_cert, "Paired direct normal": direct,
            "Paired PPI normal": existing["ppi"],
            "Finite-sample bounds": existing["finite"]}


def e4_methods(train: list[co.Family], heldout: list[co.Family], e1: dict) -> dict:
    semantic = semantic_bank(train)
    no_blocks = empty_blocks(heldout)
    behavior_blocks, gate = co.gate_decisions(heldout, 9401)
    output = {}
    for budget in co.BUDGET_DRAWS:
        if budget == co.PRIMARY_BUDGET:
            output[budget] = {
                name: {"vsr": e1[name]["base_vsr"],
                       "total_calls": e1[name]["total_cost"],
                       **({"bank": e1[name]["bank"]} if name == "Direct-Sandbox" else {})}
                for name in ("Semantic", "Behavior-filter", "Direct-Sandbox", "Full CREDIT")
            }
            continue
        semantic_summary, _ = score_policy("Semantic", policy(semantic, no_blocks),
                                           heldout, None, 0)
        behavior_summary, _ = score_policy("Behavior-filter", policy(semantic, behavior_blocks),
                                           heldout, None, gate["probe_calls"])
        selection_budget = co.full_credit_run(train, heldout, budget)[0]["training_calls"]
        candidate, select_calls, _trace = direct_select(
            train, selection_budget, 16001 + int(float(budget) * 100))
        passed, cert_calls = direct_certificate(candidate, heldout, co.CERT_N[budget],
                                                16101 + int(float(budget) * 100))
        deployed = candidate if passed else ()
        direct_summary, _ = score_policy("Direct-Sandbox", policy(deployed, no_blocks),
                                         heldout, None, select_calls + cert_calls)
        full, _ = co.full_credit_run(train, heldout, budget)
        output[budget] = {
            "Semantic": {"vsr": semantic_summary["base_vsr"],
                         "total_calls": semantic_summary["total_cost"]},
            "Behavior-filter": {"vsr": behavior_summary["base_vsr"],
                                "total_calls": behavior_summary["total_cost"]},
            "Direct-Sandbox": {"vsr": direct_summary["base_vsr"],
                               "total_calls": direct_summary["total_cost"],
                               "bank": list(deployed)},
            "Full CREDIT": {"vsr": full["base_vsr"], "total_calls": full["total_cost"]},
        }
    return output


def protocol(release_summary: Path) -> dict:
    return {
        "experiment": "Revised manuscript matched baseline completion pilot",
        "status": "finite synthetic matched comparison; post-protocol execution, not live-agent evidence",
        "source_sha256": sha256(Path(__file__)),
        "credit_source_sha256": sha256(Path(co.__file__)),
        "release_summary_sha256": sha256(release_summary),
        "fairness": {"same_families": True, "same_actor_seeds": True,
                     "same_base_and_guard_executors": True, "same_bank_capacity": BANK_CAPACITY,
                     "same_evaluation_random_stream": 9601,
                     "all_method_specific_physical_calls_charged": True},
        "methods": ["NoMem", "FIFO", "Recency", "Semantic", "Outcome-Weighted",
                    "Coverage-MMR", "Version-filter", "Behavior-filter", "MC-Shapley",
                    "CMP-adapted", "CAR-adapted", "Flat-surrogate", "Direct-Sandbox",
                    "Full CREDIT"],
        "e2_calls_per_family": co.E2_ESTIMATION_CALLS,
        "e2_coverage_definition": "nominal 95% effect-interval coverage; Banzhaf except MC-Shapley target",
        "bootstrap_replications": BOOTSTRAP_REPS,
        "primary_credit_contrasts": list(PRIMARY_METHODS),
        "claim_boundary": "results apply to planted finite fixtures; baseline performance is not intentionally degraded",
    }


def run(output: Path, release_summary: Path) -> None:
    if output.exists():
        raise FileExistsError("fresh output directory required")
    output.mkdir(parents=True)
    write_json(output / "frozen_protocol.json", protocol(release_summary))
    train = co.make_families(9001, co.TRAIN_FAMILIES, "train", allow_stale=False)
    heldout = co.make_families(9007, co.HELDOUT_FAMILIES, "heldout", allow_stale=True)

    e1, rows, diagnostics = e1_methods(train, heldout)
    write_json(output / "e1_methods.json", e1)
    write_jsonl(output / "e1_episodes.jsonl", rows)
    write_json(output / "e1_diagnostics.json", diagnostics)

    e2, e2_rows = e2_all_methods()
    write_json(output / "e2_methods.json", e2)
    write_jsonl(output / "e2_family_methods.jsonl", e2_rows)

    e3 = e3_from_e1(e1, train, heldout)
    write_json(output / "e3_methods.json", e3)
    e4 = e4_methods(train, heldout, e1)
    write_json(output / "e4_methods.json", e4)
    e5 = e5_all_rules()
    write_json(output / "e5_methods.json", e5)

    summary = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5,
               "scope": protocol(release_summary)["claim_boundary"]}
    write_json(output / "summary.json", summary)
    manifest = {path.name: sha256(path) for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_manifest.json"}
    write_json(output / "artifact_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-summary", type=Path,
                        default=Path("results/webapi_evolve_observed_v1/summary.json"))
    args = parser.parse_args()
    run(args.output, args.release_summary)


if __name__ == "__main__":
    main()
