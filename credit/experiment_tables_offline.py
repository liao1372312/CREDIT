"""Run the preregistered offline experiments for CREDIT_Experiment_Tables.tex.

The module produces two fresh experiments:

* 180 fixed attribution families at 512, 1024, and 2048 physical outcome
  draws per family; and
* 2,000 independent sequential-certification streams per truth condition.

All random seeds, estimands, thresholds, and error allocations are recorded in
``frozen_protocol.json`` before outcomes are generated.  Reference attribution
targets are exact enumerations of the noiseless fixture, not estimator calls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import norm


METHODS = ("Factorial exposure", "Uniform OLS", "OAT", "MC-Shapley", "CMP-adapted")
BUDGETS = (512, 1024, 2048)
K_VALUES = (4, 6, 8)
KERNELS = ("additive", "redundant", "pair-complement", "pair-conflict", "triple")
FAMILY_REPLICATES = 12
PAIR_THRESHOLD = 0.02
PAIR_ALPHA = 0.05
ATTRIBUTION_SEED = 271_001
STREAMS = 2_000
STRATA = 8
PAIRS_PER_CYCLE = 200
UNLABELED_PER_CYCLE = 2_000
MAX_CYCLES = 50
ALPHA_STAR = 0.05
MARGIN = 0.05
CERTIFICATION_SEED = 281_001
RULES = ("Normal, cyclewise alpha", "Normal, spent alpha",
         "PPI normal, spent alpha", "Finite bounds, spent alpha")


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def all_coalitions(k: int) -> np.ndarray:
    """Return the 2^K exposure vectors in {-1,+1} coding."""
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=k)), dtype=float)


def terms(k: int) -> tuple[list[int], list[tuple[int, int]], list[tuple[int, int, int]]]:
    return (list(range(k)), list(itertools.combinations(range(k), 2)),
            list(itertools.combinations(range(k), 3)))


def design(z: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int, int]]]:
    k = z.shape[1]
    _, pairs, triples = terms(k)
    columns = [np.ones(len(z)), *[z[:, i] for i in range(k)],
               *[z[:, i] * z[:, j] for i, j in pairs],
               *[z[:, i] * z[:, j] * z[:, q] for i, j, q in triples]]
    return np.column_stack(columns), pairs, triples


def banzhaf_weights(z: np.ndarray) -> np.ndarray:
    """Linear map from enumerated coalition utilities to Banzhaf effects."""
    k = z.shape[1]
    weights = np.zeros((k, len(z)))
    for memory in range(k):
        weights[memory, z[:, memory] > 0] = 1 / (2 ** (k - 1))
        weights[memory, z[:, memory] < 0] = -1 / (2 ** (k - 1))
    return weights


def shapley_weights(z: np.ndarray) -> np.ndarray:
    """Linear map from enumerated utilities to subset-form Shapley values."""
    k = z.shape[1]
    present = z > 0
    index = {tuple(row): i for i, row in enumerate(present.tolist())}
    weights = np.zeros((k, len(z)))
    for memory in range(k):
        others = [x for x in range(k) if x != memory]
        for size in range(k):
            coefficient = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            for subset in itertools.combinations(others, size):
                absent = [False] * k
                for item in subset:
                    absent[item] = True
                included = absent.copy(); included[memory] = True
                weights[memory, index[tuple(included)]] += coefficient
                weights[memory, index[tuple(absent)]] -= coefficient
    return weights


@dataclass(frozen=True)
class Fixture:
    uid: str
    k: int
    kernel: str
    intercept: float
    main: tuple[float, ...]
    pair: tuple[float, ...]
    triple: tuple[float, ...]

    def probability(self, z: np.ndarray) -> np.ndarray:
        x, _, _ = design(z)
        beta = np.asarray((self.intercept, *self.main, *self.pair, *self.triple))
        value = x @ beta
        if np.min(value) < 0.02 or np.max(value) > 0.98:
            raise ValueError(f"fixture {self.uid} leaves probability support")
        return value


def make_fixtures() -> list[Fixture]:
    rng = np.random.default_rng(ATTRIBUTION_SEED)
    fixtures = []
    for k in K_VALUES:
        _, pair_names, triple_names = terms(k)
        for kernel in KERNELS:
            for replicate in range(FAMILY_REPLICATES):
                main = rng.choice((-1.0, 1.0), k) * rng.uniform(0.010, 0.020, k)
                pair = np.zeros(len(pair_names))
                triple = np.zeros(len(triple_names))
                if kernel == "redundant":
                    pair[pair_names.index((0, 1))] = -rng.uniform(0.035, 0.050)
                elif kernel == "pair-complement":
                    pair[pair_names.index((0, 1))] = rng.uniform(0.035, 0.050)
                elif kernel == "pair-conflict":
                    pair[pair_names.index((0, 1))] = -rng.uniform(0.050, 0.065)
                    if len(pair_names) > 1:
                        pair[1] = rng.uniform(0.030, 0.045)
                elif kernel == "triple":
                    triple[triple_names.index((0, 1, 2))] = rng.choice((-1.0, 1.0)) * rng.uniform(0.040, 0.060)
                fixtures.append(Fixture(
                    uid=f"k{k}-{kernel}-r{replicate:02d}", k=k, kernel=kernel,
                    intercept=0.5, main=tuple(main.tolist()), pair=tuple(pair.tolist()),
                    triple=tuple(triple.tolist())))
    if len(fixtures) != 180:
        raise AssertionError("fixture count changed")
    return fixtures


def fit_ols(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, _, _ = design(z)
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    sigma2 = float(residual @ residual / max(1, len(y) - np.linalg.matrix_rank(x)))
    return beta, sigma2 * inverse


def balanced_factorial(k: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    base = all_coalitions(k)
    rows = []
    while len(rows) < budget:
        block = base[rng.permutation(len(base))]
        rows.extend(block.tolist())
    return np.asarray(rows[:budget])


def uniform_rows(k: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    # Reject only singular samples; the event is vanishingly rare at N>=512.
    while True:
        z = rng.choice((-1.0, 1.0), size=(budget, k))
        x, _, _ = design(z)
        if np.linalg.matrix_rank(x) == x.shape[1]:
            return z


def predicted_attributions(beta: np.ndarray, covariance: np.ndarray, z_all: np.ndarray):
    x_all, _, _ = design(z_all)
    utility = x_all @ beta
    wb, ws = banzhaf_weights(z_all), shapley_weights(z_all)
    bmap, smap = wb @ x_all, ws @ x_all
    b = bmap @ beta; s = smap @ beta
    bse = np.sqrt(np.maximum(0, np.diag(bmap @ covariance @ bmap.T)))
    sse = np.sqrt(np.maximum(0, np.diag(smap @ covariance @ smap.T)))
    return utility, b, s, bse, sse


def oat_estimate(fixture: Fixture, budget: int, rng: np.random.Generator):
    """Classical empty-versus-singleton one-at-a-time contrasts."""
    k = fixture.k
    arms = [tuple([-1.0] * k)]
    for memory in range(k):
        row = [-1.0] * k; row[memory] = 1.0; arms.append(tuple(row))
    counts = np.full(k + 1, budget // (k + 1), dtype=int)
    counts[:budget % (k + 1)] += 1
    means, variances = [], []
    for row, n in zip(arms, counts):
        p = float(fixture.probability(np.asarray([row]))[0])
        values = rng.binomial(1, p, size=n).astype(float)
        means.append(float(values.mean()))
        variances.append(float(values.var(ddof=1) / n))
    estimate = np.asarray(means[1:]) - means[0]
    se = np.sqrt(np.asarray(variances[1:]) + variances[0])
    return estimate, se, int(counts.sum())


def cmp_estimate(fixture: Fixture, budget: int, rng: np.random.Generator):
    z = rng.choice((-1.0, 1.0), size=(budget, fixture.k))
    y = rng.binomial(1, fixture.probability(z)).astype(float)
    effects, ses = [], []
    for memory in range(fixture.k):
        on, off = y[z[:, memory] > 0], y[z[:, memory] < 0]
        effects.append(float(on.mean() - off.mean()))
        ses.append(math.sqrt(float(on.var(ddof=1) / len(on) + off.var(ddof=1) / len(off))))
    return np.asarray(effects), np.asarray(ses), len(y)


def mc_shapley(fixture: Fixture, budget: int, rng: np.random.Generator):
    k = fixture.k
    samples = [[] for _ in range(k)]
    used = 0
    while used + k + 1 <= budget:
        order = list(rng.permutation(k))
        state = -np.ones(k)
        p = [float(fixture.probability(state[None, :])[0])]
        for memory in order:
            state = state.copy(); state[memory] = 1.0
            p.append(float(fixture.probability(state[None, :])[0]))
        # Common random number reduces variance and preserves each Bernoulli marginal.
        u = rng.random()
        outcomes = (u < np.asarray(p)).astype(float)
        for position, memory in enumerate(order):
            samples[memory].append(float(outcomes[position + 1] - outcomes[position]))
        used += k + 1
    estimate = np.asarray([np.mean(value) for value in samples])
    se = np.asarray([np.std(value, ddof=1) / math.sqrt(len(value)) for value in samples])
    return estimate, se, used


def attribution_experiment(output: Path, fixtures: list[Fixture]) -> dict:
    records = []
    for budget in BUDGETS:
        for family_index, fixture in enumerate(fixtures):
            z_all = all_coalitions(fixture.k)
            truth = fixture.probability(z_all)
            target_b = banzhaf_weights(z_all) @ truth
            target_s = shapley_weights(z_all) @ truth
            true_pair = np.asarray(fixture.pair)
            true_triple = np.asarray(fixture.triple)
            for method_index, method in enumerate(METHODS):
                rng = np.random.default_rng(ATTRIBUTION_SEED + 10_000 * budget +
                                            10 * family_index + method_index)
                item = {"budget": budget, "family": fixture.uid, "k": fixture.k,
                        "kernel": fixture.kernel, "method": method,
                        "banzhaf_squared_errors": None, "shapley_squared_errors": None,
                        "pair_tp": None, "pair_fp": None, "pair_fn": None,
                        "triple_absolute_errors": None, "covered": None}
                if method in {"Factorial exposure", "Uniform OLS"}:
                    z = (balanced_factorial(fixture.k, budget, rng) if method == "Factorial exposure"
                         else uniform_rows(fixture.k, budget, rng))
                    y = rng.binomial(1, fixture.probability(z)).astype(float)
                    beta, covariance = fit_ols(z, y)
                    _, b, s, bse, _ = predicted_attributions(beta, covariance, z_all)
                    pair_start = 1 + fixture.k
                    pair_est = beta[pair_start:pair_start + len(true_pair)]
                    pair_se = np.sqrt(np.maximum(0, np.diag(covariance)))[
                        pair_start:pair_start + len(true_pair)]
                    triple_est = beta[pair_start + len(true_pair):]
                    detected = (np.abs(pair_est) >= PAIR_THRESHOLD) & (
                        np.abs(pair_est) > norm.ppf(1 - PAIR_ALPHA / 2) * pair_se)
                    actual = np.abs(true_pair) >= PAIR_THRESHOLD
                    item.update(
                        banzhaf_squared_errors=((b - target_b) ** 2).tolist(),
                        shapley_squared_errors=((s - target_s) ** 2).tolist(),
                        pair_tp=int(np.sum(detected & actual)),
                        pair_fp=int(np.sum(detected & ~actual)),
                        pair_fn=int(np.sum(~detected & actual)),
                        triple_absolute_errors=np.abs(triple_est - true_triple).tolist(),
                        covered=(np.abs(b - target_b) <= norm.ppf(.975) * bse).tolist(),
                        actual_calls=len(y), interval_target="Banzhaf")
                elif method == "OAT":
                    b, bse, used = oat_estimate(fixture, budget, rng)
                    item.update(banzhaf_squared_errors=((b - target_b) ** 2).tolist(),
                                covered=(np.abs(b - target_b) <= norm.ppf(.975) * bse).tolist(),
                                actual_calls=used, interval_target="Banzhaf")
                elif method == "MC-Shapley":
                    s, sse, used = mc_shapley(fixture, budget, rng)
                    item.update(shapley_squared_errors=((s - target_s) ** 2).tolist(),
                                covered=(np.abs(s - target_s) <= norm.ppf(.975) * sse).tolist(),
                                actual_calls=used, interval_target="Shapley")
                else:
                    b, bse, used = cmp_estimate(fixture, budget, rng)
                    item.update(banzhaf_squared_errors=((b - target_b) ** 2).tolist(),
                                covered=(np.abs(b - target_b) <= norm.ppf(.975) * bse).tolist(),
                                actual_calls=used, interval_target="Banzhaf")
                records.append(item)

    summaries = []
    for budget in BUDGETS:
        for method in METHODS:
            subset = [row for row in records if row["budget"] == budget and row["method"] == method]
            def flatten(field: str):
                return [value for row in subset if row[field] is not None for value in row[field]]
            b2, s2 = flatten("banzhaf_squared_errors"), flatten("shapley_squared_errors")
            triple, covered = flatten("triple_absolute_errors"), flatten("covered")
            valid_pairs = [row for row in subset if row["pair_tp"] is not None]
            if valid_pairs:
                tp = sum(row["pair_tp"] for row in valid_pairs)
                fp = sum(row["pair_fp"] for row in valid_pairs)
                fn = sum(row["pair_fn"] for row in valid_pairs)
                pair_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            else:
                pair_f1 = None
            summaries.append({
                "budget": budget, "method": method, "families": len(subset),
                "banzhaf_rmse": math.sqrt(float(np.mean(b2))) if b2 else None,
                "shapley_rmse": math.sqrt(float(np.mean(s2))) if s2 else None,
                "pair_f1": pair_f1,
                "triple_mae": float(np.mean(triple)) if triple else None,
                "coverage": float(np.mean(covered)) if covered else None,
                "actual_calls_per_family_mean": float(np.mean([r["actual_calls"] for r in subset])),
                "actual_calls_total": int(sum(r["actual_calls"] for r in subset)),
                "reference_calls": 0,
            })
    dump(output / "attribution_records.json", records)
    dump(output / "attribution_summary.json", summaries)
    return {"records": records, "summary": summaries}


def _alpha(rule: str, cycle: int) -> tuple[float, float]:
    if rule == "Normal, cyclewise alpha":
        amount = ALPHA_STAR
    else:
        amount = ALPHA_STAR / (cycle * (cycle + 1))
    return amount / 2, amount / (2 * STRATA)


def _normal_bounds(mean: np.ndarray, se: np.ndarray, alpha_aggregate: float,
                   alpha_guard: float) -> tuple[np.ndarray, np.ndarray]:
    aggregate_mean = mean.mean(axis=1)
    aggregate_se = np.sqrt(np.sum(se ** 2, axis=1)) / STRATA
    aggregate_lower = aggregate_mean - norm.ppf(1 - alpha_aggregate) * aggregate_se
    guard_lower = mean - norm.ppf(1 - alpha_guard) * se
    return aggregate_lower, guard_lower


def _finite_bounds(mean: np.ndarray, n: int, alpha_aggregate: float,
                   alpha_guard: float) -> tuple[np.ndarray, np.ndarray]:
    # Paired differences lie in [-1,1]. The aggregate is also in [-1,1].
    aggregate_radius = math.sqrt(2 * math.log(1 / alpha_aggregate) / (n * STRATA))
    guard_radius = math.sqrt(2 * math.log(1 / alpha_guard) / n)
    return mean.mean(axis=1) - aggregate_radius, mean - guard_radius


def simulate_condition(name: str, gains: np.ndarray, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    control_p = np.full(STRATA, 0.45)
    treatment_p = control_p + gains
    if np.min(treatment_p) <= 0 or np.max(treatment_p) >= 1:
        raise ValueError("invalid certification probability")
    cumulative_control = np.zeros((STREAMS, STRATA), dtype=float)
    cumulative_treatment = np.zeros((STREAMS, STRATA), dtype=float)
    # PPI sufficient statistics use a predictor explaining 64% of variance.
    q = 0.64
    ppi_pred_sum = np.zeros((STREAMS, STRATA))
    ppi_resid_sum = np.zeros((STREAMS, STRATA))
    accepted = {rule: np.zeros(STREAMS, dtype=bool) for rule in RULES}
    accepted_cycle = {rule: np.zeros(STREAMS, dtype=int) for rule in RULES}
    aggregate_passed = {rule: np.zeros(STREAMS, dtype=bool) for rule in RULES}
    guard_passed = {rule: np.zeros(STREAMS, dtype=bool) for rule in RULES}
    final_decision = {rule: np.zeros(STREAMS, dtype=bool) for rule in RULES}

    for cycle in range(1, MAX_CYCLES + 1):
        cumulative_control += rng.binomial(PAIRS_PER_CYCLE, control_p,
                                           size=(STREAMS, STRATA))
        cumulative_treatment += rng.binomial(PAIRS_PER_CYCLE, treatment_p,
                                             size=(STREAMS, STRATA))
        n = cycle * PAIRS_PER_CYCLE
        mean = cumulative_treatment / n - cumulative_control / n
        pc = cumulative_control / n; pt = cumulative_treatment / n
        se = np.sqrt(np.maximum(1e-12, pt * (1 - pt) / n + pc * (1 - pc) / n))

        label_variance = treatment_p * (1 - treatment_p) + control_p * (1 - control_p)
        pred_sd = np.sqrt(q * label_variance / UNLABELED_PER_CYCLE)
        resid_sd = np.sqrt((1 - q) * label_variance / PAIRS_PER_CYCLE)
        ppi_pred_sum += rng.normal(gains, pred_sd, size=(STREAMS, STRATA)) * UNLABELED_PER_CYCLE
        ppi_resid_sum += rng.normal(0, resid_sd, size=(STREAMS, STRATA)) * PAIRS_PER_CYCLE
        ppi_mean = (ppi_pred_sum / (cycle * UNLABELED_PER_CYCLE) +
                    ppi_resid_sum / (cycle * PAIRS_PER_CYCLE))
        ppi_se = np.sqrt(q * label_variance / (cycle * UNLABELED_PER_CYCLE) +
                         (1 - q) * label_variance / (cycle * PAIRS_PER_CYCLE))
        ppi_se = np.broadcast_to(ppi_se, (STREAMS, STRATA))

        for rule in RULES:
            alpha_aggregate, alpha_guard = _alpha(rule, cycle)
            if rule == "Finite bounds, spent alpha":
                aggregate_lower, guard_lower = _finite_bounds(
                    mean, n, alpha_aggregate, alpha_guard)
            elif rule == "PPI normal, spent alpha":
                aggregate_lower, guard_lower = _normal_bounds(
                    ppi_mean, ppi_se, alpha_aggregate, alpha_guard)
            else:
                aggregate_lower, guard_lower = _normal_bounds(
                    mean, se, alpha_aggregate, alpha_guard)
            aggregate_ok = aggregate_lower > 0
            guard_ok = np.all(guard_lower > -MARGIN, axis=1)
            decision = aggregate_ok & guard_ok
            aggregate_passed[rule] |= aggregate_ok
            guard_passed[rule] |= guard_ok
            newly = decision & ~accepted[rule]
            accepted_cycle[rule][newly] = cycle
            accepted[rule] |= decision
            if cycle == MAX_CYCLES:
                final_decision[rule] = decision

    rows = []
    for rule in RULES:
        cycles_used = np.where(accepted_cycle[rule] > 0, accepted_cycle[rule], MAX_CYCLES)
        rows.append({
            "condition": name, "rule": rule, "streams": STREAMS,
            "ever_aggregate_pass": float(aggregate_passed[rule].mean()),
            "ever_guard_pass": float(guard_passed[rule].mean()),
            "ever_accept": float(accepted[rule].mean()),
            "final_cycle_accept": float(final_decision[rule].mean()),
            "mean_cycles_used": float(cycles_used.mean()),
            "cert_calls_per_stream": float((2 * STRATA * PAIRS_PER_CYCLE * cycles_used).mean()),
            "accept_count": int(accepted[rule].sum()),
        })
    return rows


def wilson_interval(successes: int, n: int) -> list[float]:
    z = norm.ppf(.975); p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def certification_experiment(output: Path) -> dict:
    conditions = {
        "aggregate_null": np.zeros(STRATA),
        "violating_stratum": np.asarray([0.05] * 7 + [-0.08]),
        "gain_2pp": np.full(STRATA, 0.02),
        "gain_5pp": np.full(STRATA, 0.05),
        "gain_10pp": np.full(STRATA, 0.10),
    }
    records = []
    for index, (name, gains) in enumerate(conditions.items()):
        records.extend(simulate_condition(name, gains,
                                          CERTIFICATION_SEED + index * 10_000))
    for row in records:
        row["ever_accept_95ci"] = wilson_interval(row["accept_count"], row["streams"])

    summary = []
    by = {(row["condition"], row["rule"]): row for row in records}
    for rule in RULES:
        null = by[("aggregate_null", rule)]
        harm = by[("violating_stratum", rule)]
        powers = [by[(f"gain_{gain}pp", rule)] for gain in (2, 5, 10)]
        summary.append({
            "rule": rule,
            "panel_a": {"false_improvement": null["ever_aggregate_pass"],
                        "guard_error": None, "any_cycle_error": null["ever_accept"],
                        "cert_calls_per_stream": null["cert_calls_per_stream"],
                        "any_cycle_error_95ci": null["ever_accept_95ci"]},
            "panel_b": {"false_improvement": None,
                        "guard_error": harm["ever_guard_pass"],
                        "any_cycle_error": harm["ever_accept"],
                        "cert_calls_per_stream": harm["cert_calls_per_stream"],
                        "any_cycle_error_95ci": harm["ever_accept_95ci"]},
            "panel_c": {"power": [row["ever_accept"] for row in powers],
                        "power_95ci": [row["ever_accept_95ci"] for row in powers],
                        "cert_calls_per_stream": [row["cert_calls_per_stream"] for row in powers]},
        })
    dump(output / "certification_records.json", records)
    dump(output / "certification_summary.json", summary)
    return {"records": records, "summary": summary}


def verify(output: Path, fixtures: list[Fixture], attribution: dict, certification: dict) -> dict:
    checks = {
        "fixture_count_180": len(fixtures) == 180,
        "attribution_record_count": len(attribution["records"]) == 180 * 3 * 5,
        "attribution_summary_count": len(attribution["summary"]) == 3 * 5,
        "all_attribution_budgets_matched": all(
            row["actual_calls"] <= row["budget"] and
            (row["method"] == "MC-Shapley" or row["actual_calls"] == row["budget"])
            for row in attribution["records"]),
        "certification_record_count": len(certification["records"]) == 5 * 4,
        "stream_count_2000": all(row["streams"] == 2000 for row in certification["records"]),
        "rates_in_unit_interval": all(0 <= row["ever_accept"] <= 1
                                      for row in certification["records"]),
        "finite_call_costs": all(math.isfinite(row["cert_calls_per_stream"])
                                 for row in certification["records"]),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    manifest = {"checks": checks, "files": {name: digest(output / name) for name in (
        "frozen_protocol.json", "fixtures.json", "attribution_records.json",
        "attribution_summary.json", "certification_records.json",
        "certification_summary.json")}}
    dump(output / "audit.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results/credit_experiment_tables_v1/offline"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    protocol = {
        "status": "frozen before outcome generation",
        "attribution": {"seed": ATTRIBUTION_SEED, "budgets": BUDGETS,
            "k": K_VALUES, "kernels": KERNELS, "replicates_per_cell": FAMILY_REPLICATES,
            "families": 180, "pair_detection_absolute_threshold": PAIR_THRESHOLD,
            "pair_detection_two_sided_alpha": PAIR_ALPHA,
            "reference": "exact noiseless enumeration; zero physical estimator calls",
            "outcome": "Bernoulli draw from fixture probability"},
        "certification": {"seed": CERTIFICATION_SEED, "streams_per_condition": STREAMS,
            "strata": STRATA, "labeled_pairs_per_stratum_per_cycle": PAIRS_PER_CYCLE,
            "unlabeled_predictions_per_stratum_per_cycle": UNLABELED_PER_CYCLE,
            "maximum_cycles": MAX_CYCLES, "alpha_star": ALPHA_STAR, "margin": MARGIN,
            "alpha_allocation": "half aggregate, half equally over 8 guards; spent alpha_t=alpha*/[t(t+1)]",
            "predictor_explained_variance": 0.64,
            "physical_call_unit": "one arm of one labeled paired episode"},
    }
    dump(args.output / "frozen_protocol.json", protocol)
    fixtures = make_fixtures()
    dump(args.output / "fixtures.json", [asdict(fixture) for fixture in fixtures])
    attribution = attribution_experiment(args.output, fixtures)
    certification = certification_experiment(args.output)
    audit = verify(args.output, fixtures, attribution, certification)
    print(json.dumps({"output": str(args.output), "audit": audit["checks"]}, indent=2))


if __name__ == "__main__":
    main()
