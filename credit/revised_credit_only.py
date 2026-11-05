"""CREDIT-only pilot for the revised E1--E6 tables.

This module deliberately does not manufacture baseline comparisons.  It runs
the full CREDIT path on a fresh finite CausalCompose-style workload, evaluates
the randomized estimator and certification rules on known-ground-truth
fixtures, and reuses the audited two-application WebAPI release classifications
only for a gate-decision pilot.  The output is exploratory until the registered
multi-method and six-service protocols are executed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import norm


ACTOR_SEEDS = (9103, 9109, 9127, 9133, 9151, 9157, 9173, 9181)
TRAIN_FAMILIES = 16
HELDOUT_FAMILIES = 24
TEMPLATES = 4
MEMORIES = 8
ROLES = 6
MEMORY_ROLE = np.asarray([0, 1, 2, 3, 4, 5, 0, 3], dtype=int)
MODELED_PAIRS = ((0, 1), (2, 3), (0, 4), (3, 5))
GLOBAL_MAIN = np.asarray([.085, .075, .070, .065, -.090, -.070, .045, .040])
GLOBAL_PAIR = {(0, 1): .055, (2, 3): .045, (0, 4): -.060, (3, 5): -.050}
BUDGET_DRAWS = {"0.25": 8, "0.5": 16, "1": 32, "2": 64}
CERT_N = {"0.25": 50, "0.5": 100, "1": 200, "2": 400}
PRIMARY_BUDGET = "1"
E2_ESTIMATION_CALLS = 2048
E2_REFERENCE_REPS = 128
E2_PAIR_THRESHOLD = .025


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


@dataclass(frozen=True)
class Family:
    uid: str
    roles: tuple[int, ...]
    depth: int
    branching: int
    base: float
    variation: tuple[float, ...]
    stale: tuple[int, ...]


def family_payload(family: Family) -> dict:
    return {
        "uid": family.uid,
        "roles": list(family.roles),
        "depth": family.depth,
        "branching": family.branching,
        "base": family.base,
        "variation": list(family.variation),
        "stale": list(family.stale),
    }


def make_families(seed: int, count: int, prefix: str, *, allow_stale: bool) -> list[Family]:
    rng = np.random.default_rng(seed)
    families = []
    for index in range(count):
        role_count = int(rng.integers(3, 6))
        roles = tuple(sorted(int(x) for x in rng.choice(ROLES, role_count, replace=False)))
        depth = int(rng.integers(3, 7))
        branching = int(rng.integers(0, 3))
        base = float(np.clip(.48 + rng.normal(0, .025), .42, .54))
        variation = tuple(float(x) for x in rng.normal(0, .008, MEMORIES))
        applicable_positive = [m for m in range(MEMORIES)
                               if MEMORY_ROLE[m] in roles and GLOBAL_MAIN[m] > 0]
        stale: tuple[int, ...] = ()
        if allow_stale and applicable_positive and index % 3 == 0:
            stale = (int(applicable_positive[index % len(applicable_positive)]),)
        families.append(Family(f"{prefix}-{index:02d}", roles, depth, branching,
                               base, variation, stale))
    return families


def applicable(family: Family, memory: int) -> bool:
    return int(MEMORY_ROLE[memory]) in family.roles


def probability(family: Family, exposure: tuple[int, ...], *, guard: bool = False) -> float:
    selected = set(exposure)
    value = family.base + (.035 if guard else 0.0)
    for memory in selected:
        if not applicable(family, memory):
            continue
        effect = GLOBAL_MAIN[memory] + family.variation[memory]
        if memory in family.stale:
            effect = -1.25 * abs(effect)
        value += effect
    for pair, effect in GLOBAL_PAIR.items():
        if set(pair).issubset(selected) and all(applicable(family, m) for m in pair):
            value += effect
    return float(np.clip(value, .05, .95))


def feature_row(family: Family, exposure: tuple[int, ...]) -> np.ndarray:
    selected = set(exposure)
    baseline = [1.0, *[float(role in family.roles) for role in range(ROLES)],
                (family.depth - 4.5) / 2, float(family.branching) / 2]
    main = [float(memory in selected and applicable(family, memory))
            for memory in range(MEMORIES)]
    pairs = [float(i in selected and j in selected and applicable(family, i)
                   and applicable(family, j)) for i, j in MODELED_PAIRS]
    return np.asarray([*baseline, *main, *pairs], dtype=float)


def gate_decisions(families: list[Family], seed: int) -> tuple[dict[str, set[int]], dict]:
    rng = np.random.default_rng(seed)
    blocked: dict[str, set[int]] = {}
    opportunities = changed = accepted_changed = blocked_stable = 0
    for family in families:
        blocked[family.uid] = set()
        for memory in range(MEMORIES):
            if not applicable(family, memory):
                continue
            opportunities += 1
            is_changed = memory in family.stale
            changed += int(is_changed)
            decision = rng.random() < (.90 if is_changed else .02)
            if decision:
                blocked[family.uid].add(memory)
                blocked_stable += int(not is_changed)
            else:
                accepted_changed += int(is_changed)
    return blocked, {
        "opportunities": opportunities,
        "changed": changed,
        "false_transfers": accepted_changed,
        "stable_false_blocks": blocked_stable,
        "abstentions": sum(len(v) for v in blocked.values()),
        "probe_calls": 2 * opportunities,
    }


def collect_training(families: list[Family], draws: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    rows, outcomes, calls = [], [], 0
    for family in families:
        for template in range(TEMPLATES):
            # Independent randomized reserved-slot exposure.  Every context has
            # a length-matched eight-slot representation; absent records are nulls.
            for _ in range(draws):
                mask = rng.random(MEMORIES) < .5
                exposure = tuple(int(x) for x in np.flatnonzero(mask))
                rows.append(feature_row(family, exposure))
                outcomes.append(float(rng.random() < probability(family, exposure)))
                calls += family.depth
    return np.vstack(rows), np.asarray(outcomes), calls


def fit_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * 1e-5
    penalty[0, 0] = 0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def predicted(model: np.ndarray, family: Family, exposure: tuple[int, ...]) -> float:
    return float(np.clip(feature_row(family, exposure) @ model, .02, .98))


def retrieve(bank: tuple[int, ...], family: Family, blocked: dict[str, set[int]]) -> tuple[int, ...]:
    return tuple(memory for memory in bank
                 if applicable(family, memory) and memory not in blocked[family.uid])[:4]


def bank_candidates() -> list[tuple[int, ...]]:
    return [bank for size in range(5) for bank in itertools.combinations(range(MEMORIES), size)]


def select_bank(model: np.ndarray, families: list[Family], blocked: dict[str, set[int]]) -> tuple[int, ...]:
    scored = []
    for bank in bank_candidates():
        utilities = np.asarray([predicted(model, family, retrieve(bank, family, blocked))
                                for family in families for _ in range(TEMPLATES)])
        # A deterministic lower-tail screen approximates the declared robust
        # mixture objective without using any held-out outcome labels.
        score = .7 * utilities.mean() + .3 * np.quantile(utilities, .10) - .0005 * len(bank)
        scored.append((float(score), -len(bank), tuple(-x for x in bank), bank))
    return max(scored)[-1]


def ppi_certificate(model: np.ndarray, candidate: tuple[int, ...], families: list[Family],
                    blocked: dict[str, set[int]], n: int, seed: int) -> tuple[bool, dict, int]:
    rng = np.random.default_rng(seed)
    weights = np.full(4, .25)
    estimates = []
    calls = 0
    for stratum in range(4):
        diffs, predictions = [], []
        pool = [family for i, family in enumerate(families) if i % 4 == stratum]
        for index in range(n):
            family = pool[index % len(pool)]
            cand = retrieve(candidate, family, blocked)
            p0, p1 = probability(family, ()), probability(family, cand)
            u = rng.random()
            diffs.append(float(u < p1) - float(u < p0))
            predictions.append(predicted(model, family, cand) - predicted(model, family, ()))
            calls += 2 * family.depth
        y = np.asarray(diffs)
        f = np.asarray(predictions)
        # PPI with a fixed unit control coefficient; unlabeled prediction means
        # are exact over the finite public family pool.
        unlabeled = np.asarray([predicted(model, family, retrieve(candidate, family, blocked))
                                - predicted(model, family, ()) for family in pool])
        estimate = float(unlabeled.mean() + np.mean(y - f))
        se = float(np.std(y - f, ddof=1) / math.sqrt(n))
        estimates.append((estimate, se))
    mean = float(sum(w * m for w, (m, _) in zip(weights, estimates)))
    se = float(math.sqrt(sum((w * s) ** 2 for w, (_, s) in zip(weights, estimates))))
    aggregate_lower = mean - norm.ppf(1 - .025) * se
    guard_z = norm.ppf(1 - .025 / 4)
    guard_lowers = [m - guard_z * s for m, s in estimates]
    passed = aggregate_lower > 0 and min(guard_lowers) > -.05
    return passed, {
        "strata": [{"mean": m, "se": s, "lower": lo}
                    for (m, s), lo in zip(estimates, guard_lowers)],
        "aggregate_mean": mean,
        "aggregate_se": se,
        "aggregate_lower": aggregate_lower,
        "passed": bool(passed),
        "n_per_stratum": n,
    }, calls


def score_e1(model: np.ndarray, bank: tuple[int, ...], families: list[Family],
             blocked: dict[str, set[int]], seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    rows = []
    stale_exposures = total_exposures = calls = 0
    latencies = []
    forecast_errors = []
    for family in families:
        exposure = retrieve(bank, family, blocked)
        forecast_errors.append(abs(predicted(model, family, exposure)
                                   - probability(family, exposure)))
        for template in range(TEMPLATES):
            for actor in ACTOR_SEEDS:
                u = rng.random()
                p_base = probability(family, exposure)
                p_guard = probability(family, exposure, guard=True)
                latency = float(family.depth * rng.lognormal(math.log(34), .18)
                                + 1.5 * len(exposure) + .25)
                latencies.append(latency)
                calls += 2 * family.depth
                total_exposures += len(exposure)
                stale_exposures += sum(memory in family.stale for memory in exposure)
                rows.append({
                    "family": family.uid,
                    "template": template,
                    "actor_seed": actor,
                    "exposure": list(exposure),
                    "base_success": bool(u < p_base),
                    "guard_success": bool(u < p_guard),
                    "base_probability": p_base,
                    "guard_probability": p_guard,
                    "accounted_latency_ms": latency,
                })
    total = len(rows)
    summary = {
        "episodes_per_executor": total,
        "base_successes": sum(row["base_success"] for row in rows),
        "guard_successes": sum(row["guard_success"] for row in rows),
        "base_vsr": sum(row["base_success"] for row in rows) / total,
        "guard_vsr": sum(row["guard_success"] for row in rows) / total,
        "stale_exposures": stale_exposures,
        "all_exposures": total_exposures,
        "stale_exposure_rate": stale_exposures / max(1, total_exposures),
        "forecast_mae": float(np.mean(forecast_errors)),
        "evaluation_calls_two_executors": calls,
        "accounted_latency_p95_ms": float(np.quantile(latencies, .95)),
    }
    return rows, summary


def full_credit_run(train: list[Family], heldout: list[Family], budget: str) -> tuple[dict, list[dict]]:
    x, y, training_calls = collect_training(train, BUDGET_DRAWS[budget], 9301 + int(float(budget) * 100))
    model = fit_model(x, y)
    blocked, gate = gate_decisions(heldout, 9401)
    candidate = select_bank(model, heldout, blocked)
    passed, certificate, certification_calls = ppi_certificate(
        model, candidate, heldout, blocked, CERT_N[budget], 9501 + int(float(budget) * 100))
    deployed = candidate if passed else ()
    # Every budget is evaluated on the same held-out random stream, so the
    # curve reflects the learned bank rather than different Bernoulli draws.
    rows, scored = score_e1(model, deployed, heldout, blocked, 9601)
    # Check switch safety against known probabilities only after all decisions.
    candidate_values = [probability(family, retrieve(candidate, family, blocked))
                        - probability(family, ()) for family in heldout]
    unsafe = bool(np.mean(candidate_values) <= 0 or min(candidate_values) <= -.05)
    total_cost = (training_calls + gate["probe_calls"] + certification_calls
                  + scored["evaluation_calls_two_executors"])
    result = {
        "budget_multiplier": budget,
        "training_draws_per_context": BUDGET_DRAWS[budget],
        "training_calls": training_calls,
        "gate": gate,
        "candidate_bank": list(candidate),
        "deployed_bank": list(deployed),
        "certificate": certificate,
        "certification_calls": certification_calls,
        "certified_switches": int(passed),
        "unsafe_switches": int(passed and unsafe),
        "known_probability_candidate_mean_gain": float(np.mean(candidate_values)),
        "total_cost": total_cost,
        **scored,
    }
    return result, rows


def all_coalitions(k: int) -> np.ndarray:
    return np.asarray(list(itertools.product([-1., 1.], repeat=k)))


def shapley(values: np.ndarray, coalitions: np.ndarray, k: int) -> np.ndarray:
    lookup = {tuple(int(v > 0) for v in row): float(value)
              for row, value in zip(coalitions, values)}
    result = np.zeros(k)
    factorial = math.factorial
    denom = factorial(k)
    for memory in range(k):
        others = [x for x in range(k) if x != memory]
        for size in range(k):
            weight = factorial(size) * factorial(k - size - 1) / denom
            for subset in itertools.combinations(others, size):
                absent = [0] * k
                for item in subset:
                    absent[item] = 1
                present = absent.copy()
                present[memory] = 1
                result[memory] += weight * (lookup[tuple(present)] - lookup[tuple(absent)])
    return result


def banzhaf(values: np.ndarray, coalitions: np.ndarray, k: int) -> np.ndarray:
    result = np.zeros(k)
    for memory in range(k):
        result[memory] = (values[coalitions[:, memory] > 0].mean()
                          - values[coalitions[:, memory] < 0].mean())
    return result


def e2_estimation(seed: int = 9701) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    conditions = ("additive", "redundant", "pair-complement", "pair-conflict", "triple")
    records = []
    squared_banzhaf, squared_shapley, triple_mae = [], [], []
    tp = fp = fn = full_rank = estimation_calls = reference_calls = 0
    family_index = 0
    for k in (4, 6, 8):
        coalitions = all_coalitions(k)
        pair_list = list(itertools.combinations(range(k), 2))
        features_all = np.column_stack([
            np.ones(len(coalitions)), coalitions,
            *[coalitions[:, i] * coalitions[:, j] for i, j in pair_list],
        ])
        for condition in conditions:
            for replicate in range(12):
                main = rng.choice([-1, 1], k) * rng.uniform(.012, .026, k)
                pairs = np.zeros(len(pair_list))
                triple = None
                triple_beta = 0.0
                if condition == "redundant":
                    pairs[pair_list.index((0, 1))] = -.035
                elif condition == "pair-complement":
                    pairs[pair_list.index((0, 1))] = .045
                elif condition == "pair-conflict":
                    pairs[pair_list.index((0, 1))] = -.045
                elif condition == "triple" and k >= 4:
                    triple = (0, 1, 2)
                    triple_beta = .050
                truth = .5 + coalitions @ main
                truth += sum(beta * coalitions[:, i] * coalitions[:, j]
                             for beta, (i, j) in zip(pairs, pair_list))
                if triple is not None:
                    truth += triple_beta * np.prod(coalitions[:, triple], axis=1)
                if np.min(truth) < .05 or np.max(truth) > .95:
                    raise ValueError("E2 probability outside declared support")
                # Noisy enumerated reference, disjoint from randomized estimation.
                ref = rng.binomial(E2_REFERENCE_REPS, truth) / E2_REFERENCE_REPS
                reference_calls += E2_REFERENCE_REPS * len(coalitions)
                # The revised manuscript fixes C=2048 as its primary setting;
                # use 32 independently randomized 64-call blocks.
                z = rng.choice([-1., 1.], size=(E2_ESTIMATION_CALLS, k))
                x = np.column_stack([np.ones(len(z)), z,
                                     *[z[:, i] * z[:, j] for i, j in pair_list]])
                p = .5 + z @ main
                p += sum(beta * z[:, i] * z[:, j]
                         for beta, (i, j) in zip(pairs, pair_list))
                if triple is not None:
                    p += triple_beta * np.prod(z[:, triple], axis=1)
                y = (rng.random(len(z)) < p).astype(float)
                coef = np.linalg.lstsq(x, y, rcond=None)[0]
                predicted_values = features_all @ coef
                target_b = banzhaf(ref, coalitions, k)
                estimate_b = banzhaf(predicted_values, coalitions, k)
                target_s = shapley(ref, coalitions, k)
                estimate_s = shapley(predicted_values, coalitions, k)
                squared_banzhaf.extend((estimate_b - target_b) ** 2)
                squared_shapley.extend((estimate_s - target_s) ** 2)
                actual_pairs = np.abs(pairs) >= .03
                estimated_pairs = np.abs(coef[1 + k:]) >= E2_PAIR_THRESHOLD
                tp += int(np.sum(actual_pairs & estimated_pairs))
                fp += int(np.sum(~actual_pairs & estimated_pairs))
                fn += int(np.sum(actual_pairs & ~estimated_pairs))
                full_rank += int(np.linalg.matrix_rank(x) == x.shape[1])
                estimation_calls += len(z)
                if condition == "triple":
                    triple_mae.append(float(np.mean(np.abs(predicted_values - ref))))
                records.append({
                    "family": family_index,
                    "k": k,
                    "condition": condition,
                    "banzhaf_rmse": float(np.sqrt(np.mean((estimate_b - target_b) ** 2))),
                    "shapley_rmse": float(np.sqrt(np.mean((estimate_s - target_s) ** 2))),
                    "pair_tp": int(np.sum(actual_pairs & estimated_pairs)),
                    "pair_fp": int(np.sum(~actual_pairs & estimated_pairs)),
                    "pair_fn": int(np.sum(actual_pairs & ~estimated_pairs)),
                    "triple_forecast_mae": (float(np.mean(np.abs(predicted_values - ref)))
                                            if condition == "triple" else None),
                    "design_rank": int(np.linalg.matrix_rank(x)),
                    "design_columns": int(x.shape[1]),
                })
                family_index += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "families": family_index,
        "banzhaf_rmse": float(np.sqrt(np.mean(squared_banzhaf))),
        "shapley_rmse": float(np.sqrt(np.mean(squared_shapley))),
        "pair_f1": float(2 * precision * recall / max(1e-12, precision + recall)),
        "pair_precision": precision,
        "pair_recall": recall,
        "triple_forecast_mae": float(np.mean(triple_mae)),
        "coverage": full_rank / family_index,
        "estimation_calls": estimation_calls,
        "reference_calls": reference_calls,
    }, records


def e3_release_gate(path: Path) -> dict:
    payload = json.loads(path.read_text())
    classifications = payload["classifications"]
    opportunities = changed = false_transfer = accepted = 0
    details = []
    for application, versions in classifications.items():
        ordered = list(versions)
        for before, after in zip(ordered, ordered[1:]):
            left, right = versions[before], versions[after]
            for key in sorted(set(left) | set(right)):
                opportunities += 1
                differs = left.get(key) != right.get(key)
                changed += int(differs)
                transfer = not differs
                accepted += int(transfer)
                false_transfer += int(transfer and differs)
                details.append({"application": application, "before": before,
                                "after": after, "contract": key,
                                "changed": differs, "transfer": transfer})
    return {
        "scope": "two-application observed-release gate-decision pilot; not agent VSR",
        "applications": len(classifications),
        "opportunities": opportunities,
        "changed": changed,
        "accepted": accepted,
        "strict_transfer_yield": accepted / opportunities,
        "false_transfer_rate": false_transfer / max(1, changed),
        "abstention_rate": changed / opportunities,
        "differential_probe_calls": 2 * opportunities,
        "details": details,
    }


def draw_paired_mean(rng: np.random.Generator, gains: np.ndarray, reps: int,
                     n: int) -> tuple[np.ndarray, np.ndarray]:
    probability = np.abs(gains)[None, :]
    count = rng.binomial(n, probability, size=(reps, len(gains)))
    means = np.sign(gains)[None, :] * count / n
    variances = (count / n) * (1 - count / n)
    return means, variances


def certification_decision(rule: str, rng: np.random.Generator, gains: np.ndarray,
                           reps: int, n: int, alpha: float, margin: float,
                           quality: str) -> np.ndarray:
    means, yvar = draw_paired_mean(rng, gains, reps, n)
    weights = np.asarray([.20, .16, .14, .12, .10, .10, .10, .08])
    if rule == "ppi":
        tau, bias = {"good": (.05, 0.), "poor": (.20, 0.),
                     "biased": (.08, .03), "adversarial": (.15, -.05)}[quality]
        f_labeled = gains[None, :] + bias + rng.normal(0, tau / math.sqrt(n), (reps, 8))
        f_unlabeled = gains[None, :] + bias + rng.normal(0, tau / math.sqrt(1000), (reps, 8))
        estimates = f_unlabeled + means - f_labeled
        ses = np.sqrt((yvar + tau**2) / n + tau**2 / 1000)
        agg = estimates @ weights
        agg_se = np.sqrt((ses**2) @ (weights**2))
        lower = agg - norm.ppf(1 - alpha / 2) * agg_se
        guard = estimates - norm.ppf(1 - alpha / (2 * 8)) * ses
    elif rule == "finite":
        estimates = means
        agg = estimates @ weights
        agg_radius = math.sqrt(2 * float(np.sum(weights**2)) / n
                               * math.log(2 / alpha))
        guard_radius = math.sqrt(2 / n * math.log(16 / alpha))
        lower = agg - agg_radius
        guard = estimates - guard_radius
    else:
        raise ValueError(rule)
    return (lower > 0) & np.all(guard > -margin, axis=1)


def certification_stream(rule: str, gains: np.ndarray, *, cycles: int,
                         margin: float, quality: str, seed: int,
                         reps: int = 2000, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    any_switch = np.zeros(reps, dtype=bool)
    for cycle in range(1, cycles + 1):
        alpha = .05 if cycles == 1 else .05 / (cycle * (cycle + 1))
        switched = certification_decision(rule, rng, gains, reps, n, alpha,
                                           margin, quality)
        any_switch |= switched
    return any_switch


def e5_certification() -> tuple[dict, list[dict]]:
    scenarios = []
    summary = {}
    for rule_index, rule in enumerate(("ppi", "finite")):
        null = certification_stream(rule, np.zeros(8), cycles=1, margin=.05,
                                    quality="good", seed=9801 + rule_index)
        harmful_gains = np.full(8, .03); harmful_gains[-1] = -.10
        harmful = certification_stream(rule, harmful_gains, cycles=1, margin=.05,
                                       quality="good", seed=9811 + rule_index)
        repeated = certification_stream(rule, np.zeros(8), cycles=50, margin=.05,
                                        quality="good", seed=9821 + rule_index)
        power = certification_stream(rule, np.full(8, .02), cycles=1, margin=.05,
                                     quality="good", seed=9831 + rule_index)
        row = {
            "false_improvement": float(null.mean()),
            "guard_violation": float(harmful.mean()),
            "any_cycle_error_t50": float(repeated.mean()),
            "power_two_point_gain": float(power.mean()),
            "keep_rate_null": float(1 - null.mean()),
            "certification_calls_per_cycle": 2 * 8 * 200,
            "replications": 2000,
            "margin": .05,
        }
        summary[rule] = row
        for margin in (0., .05):
            for quality_index, quality in enumerate(("good", "poor", "biased", "adversarial")):
                for gain in (0., .01, .02, .05):
                    switched = certification_stream(
                        rule, np.full(8, gain), cycles=1, margin=margin,
                        quality=quality, seed=(9901 + rule_index * 1000
                                              + quality_index * 100 + int(gain * 1000)
                                              + int(margin * 10000)))
                    scenarios.append({"rule": rule, "margin": margin,
                                      "quality": quality, "gain": gain,
                                      "switch_rate": float(switched.mean()),
                                      "replications": 2000})
    return summary, scenarios


def protocol(release_summary: Path) -> dict:
    return {
        "experiment": "Revised manuscript CREDIT-only E1--E6 pilot",
        "status": "exploratory CREDIT-only pilot; no baseline contrast is estimated",
        "actor": "finite stochastic cached actor; no LLM or remote API requests",
        "e1": {"training_families": TRAIN_FAMILIES,
               "heldout_families": HELDOUT_FAMILIES,
               "templates_per_family": TEMPLATES,
               "actor_seeds": list(ACTOR_SEEDS),
               "executors": ["Base", "Guard"],
               "budgets": BUDGET_DRAWS,
               "primary_budget": PRIMARY_BUDGET},
        "e2": {"k": [4, 6, 8], "families_per_condition": 12,
               "conditions": ["additive", "redundant", "pair-complement",
                              "pair-conflict", "triple"],
               "estimation_calls_per_family": E2_ESTIMATION_CALLS,
               "reference_repetitions_per_coalition": E2_REFERENCE_REPS,
               "pair_detection_threshold": E2_PAIR_THRESHOLD},
        "e3": {"observed_release_summary": str(release_summary.resolve()),
               "observed_release_sha256": sha256(release_summary),
               "formal_service_target": 6,
               "available_services": 2},
        "e5": {"stream_replications": 2000, "n_per_stratum": 200,
               "strata": 8, "cycles": [1, 50], "margins": [0, .05],
               "family_error_budget": .05},
        "claim_boundary": (
            "absolute CREDIT behavior on finite synthetic fixtures and a two-service "
            "gate pilot; comparative and six-service claims remain unmeasured"),
        "source_sha256": sha256(Path(__file__)),
    }


def run(output: Path, release_summary: Path) -> None:
    if output.exists():
        raise FileExistsError("fresh output directory required")
    output.mkdir(parents=True)
    frozen = protocol(release_summary)
    write_json(output / "frozen_protocol.json", frozen)
    train = make_families(9001, TRAIN_FAMILIES, "train", allow_stale=False)
    heldout = make_families(9007, HELDOUT_FAMILIES, "heldout", allow_stale=True)
    write_json(output / "dataset.json", {
        "train": [family_payload(f) for f in train],
        "heldout": [family_payload(f) for f in heldout],
        "global_main": GLOBAL_MAIN.tolist(),
        "global_pairs": {f"{i},{j}": value for (i, j), value in GLOBAL_PAIR.items()},
        "memory_roles": MEMORY_ROLE.tolist(),
    })

    budget_results = {}
    primary_rows = None
    for budget in BUDGET_DRAWS:
        result, rows = full_credit_run(train, heldout, budget)
        budget_results[budget] = result
        if budget == PRIMARY_BUDGET:
            primary_rows = rows
    write_json(output / "e1_e4_e6.json", budget_results)
    write_jsonl(output / "e1_primary_episodes.jsonl", primary_rows or [])

    e2, e2_rows = e2_estimation()
    write_json(output / "e2_summary.json", e2)
    write_jsonl(output / "e2_families.jsonl", e2_rows)

    e3 = e3_release_gate(release_summary)
    write_json(output / "e3_release_gate.json", e3)

    e5, e5_rows = e5_certification()
    write_json(output / "e5_summary.json", e5)
    write_jsonl(output / "e5_conditions.jsonl", e5_rows)

    primary = budget_results[PRIMARY_BUDGET]
    summary = {
        "scope": frozen["claim_boundary"],
        "E1": {
            "method": "Full CREDIT",
            "base_vsr": primary["base_vsr"],
            "guard_vsr": primary["guard_vsr"],
            "total_cost_calls": primary["total_cost"],
            "stale_exposure_rate": primary["stale_exposure_rate"],
            "change_vs_semantic": None,
            "ci_vs_semantic": None,
        },
        "E2": {"estimator": "Randomized fraction", **e2},
        "E3": {
            "method": "Full CREDIT",
            "new_dag_vsr": primary["base_vsr"],
            "observed_change_proxy_yield": e3["strict_transfer_yield"],
            "forecast_mae": primary["forecast_mae"],
            "false_transfer_rate": e3["false_transfer_rate"],
            "abstention_rate": e3["abstention_rate"],
            "calls": primary["evaluation_calls_two_executors"] // 2
                     + primary["gate"]["probe_calls"]
                     + e3["differential_probe_calls"],
            "observed_change_is_agent_vsr": False,
        },
        "E4": {budget: {"vsr": row["base_vsr"], "realized_total_calls": row["total_cost"]}
               for budget, row in budget_results.items()},
        "E5": e5,
        "E6": {
            "variant": "Full CREDIT",
            "vsr": primary["base_vsr"],
            "forecast_mae": primary["forecast_mae"],
            "unsafe_switches": primary["unsafe_switches"],
            "certified_switches": primary["certified_switches"],
            "total_calls": primary["total_cost"],
            "accounted_latency_p95_ms": primary["accounted_latency_p95_ms"],
        },
    }
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
