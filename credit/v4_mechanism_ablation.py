"""Mechanism-aligned controlled ablations for CREDIT.

This is a post-hoc activation stress test.  Each fixture is derived from the
assumption handled by one component and has a component-specific endpoint.
It is not a natural-task or end-to-end superiority experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .v3.inference import certify, ppi, worst_mixture


DEFAULT_SEEDS = [
    82003, 82007, 82009, 82013, 82021, 82031, 82037, 82039,
    82051, 82067, 82073, 82079, 82099, 82109, 82129, 82139,
]
DEVELOPMENT_SEED = 81001
COMPONENTS = (
    "randomized_exposure",
    "pair_interactions",
    "drift_gate",
    "fresh_certification",
    "robust_radius",
    "role_transport",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def features(x: np.ndarray, pairs: list[tuple[int, int]], interactions: bool = True) -> np.ndarray:
    columns = [np.ones(len(x)), *[x[:, i] for i in range(x.shape[1])]]
    if interactions:
        columns.extend(x[:, i] * x[:, j] for i, j in pairs)
    return np.column_stack(columns)


def linear_probability(difficulty: np.ndarray, x: np.ndarray,
                       main: np.ndarray, pair_beta: np.ndarray,
                       pairs: list[tuple[int, int]]) -> np.ndarray:
    value = 0.50 - 0.12 * difficulty + x @ main
    value += sum(beta * x[:, i] * x[:, j]
                 for beta, (i, j) in zip(pair_beta, pairs))
    if np.min(value) < 0.02 or np.max(value) > 0.98:
        raise ValueError("linear probability left its declared support")
    return value


def choose_bank(coef: np.ndarray, pairs: list[tuple[int, int]],
                interactions: bool, k: int = 6, capacity: int = 2) -> tuple[int, ...]:
    candidates = [bank for size in range(capacity + 1)
                  for bank in itertools.combinations(range(k), size)]
    rows = np.zeros((len(candidates), k))
    for r, bank in enumerate(candidates):
        rows[r, list(bank)] = 1
    prediction = features(rows, pairs, interactions) @ coef
    return candidates[int(np.argmax(prediction))]


def bank_value(bank: tuple[int, ...], main: np.ndarray, pair_beta: np.ndarray,
               pairs: list[tuple[int, int]]) -> float:
    x = np.zeros((1, len(main)))
    x[0, list(bank)] = 1
    return float(linear_probability(np.zeros(1), x, main, pair_beta, pairs)[0])


def randomized_and_interactions(rng: np.random.Generator) -> tuple[dict, dict]:
    k = 6
    pairs = list(itertools.combinations(range(k), 2))
    # Randomization fixture: a hidden task-difficulty variable drives both
    # deterministic retrieval propensity and outcome.  The randomized cube
    # breaks that path; unadjusted observed associations do not.
    random_main = np.asarray([0.03, -0.02, 0.04, 0.00, 0.02, -0.03])
    zero_pairs = np.zeros(len(pairs))
    n_random = 4096
    cube = ((np.arange(2**k)[:, None] >> np.arange(k)) & 1).astype(float)
    x_random = np.tile(cube, (n_random // len(cube), 1))
    rng.shuffle(x_random, axis=0)
    d_random = rng.uniform(-1, 1, n_random)
    p_random = linear_probability(
        d_random, x_random, random_main, zero_pairs, pairs)
    y_random = (rng.random(n_random) < p_random).astype(float)
    coef_random = np.linalg.lstsq(
        features(x_random, pairs, False), y_random, rcond=None)[0][1:]

    d_observed = rng.uniform(-1, 1, n_random)
    slopes = np.asarray([-3.0, -2.5, -2.0, 2.0, 2.5, 3.0])
    logits = d_observed[:, None] * slopes + np.linspace(-0.2, 0.2, k)
    propensity = 1 / (1 + np.exp(-logits))
    x_observed = (rng.random((n_random, k)) < propensity).astype(float)
    p_observed = linear_probability(
        d_observed, x_observed, random_main, zero_pairs, pairs)
    y_observed = (rng.random(n_random) < p_observed).astype(float)
    associations = np.asarray([
        y_observed[x_observed[:, j] == 1].mean()
        - y_observed[x_observed[:, j] == 0].mean()
        for j in range(k)
    ])
    random_rmse = float(np.sqrt(np.mean((coef_random - random_main) ** 2)))
    observed_rmse = float(np.sqrt(np.mean((associations - random_main) ** 2)))

    # Interaction fixture: pair (0,1) is the optimal capacity-two bank, but
    # its marginal coefficients are depressed by conflicts with 4 and 5.
    # Main-only selection instead favors 2 and 3 through interactions that
    # require memories outside the selected bank.
    main = np.zeros(k)
    pair_map = {
        (0, 1): 0.16,
        (0, 4): -0.06, (0, 5): -0.06,
        (1, 4): -0.06, (1, 5): -0.06,
        (2, 4): 0.05, (2, 5): 0.05,
        (3, 4): 0.05, (3, 5): 0.05,
    }
    pair_beta = np.asarray([pair_map.get(pair, 0.0) for pair in pairs])
    n_interaction = 8192
    x_interaction = np.tile(cube, (n_interaction // len(cube), 1))
    rng.shuffle(x_interaction, axis=0)
    d_interaction = rng.uniform(-1, 1, n_interaction)
    p_interaction = linear_probability(
        d_interaction, x_interaction, main, pair_beta, pairs)
    y_interaction = (rng.random(n_interaction) < p_interaction).astype(float)
    coef_full = np.linalg.lstsq(
        features(x_interaction, pairs, True), y_interaction, rcond=None)[0]
    coef_main = np.linalg.lstsq(features(x_interaction, pairs, False), y_interaction,
                                rcond=None)[0]
    full_bank = choose_bank(coef_full, pairs, True)
    main_bank = choose_bank(coef_main, pairs, False)
    candidates = [bank for size in range(3)
                  for bank in itertools.combinations(range(k), size)]
    optimum = max(bank_value(bank, main, pair_beta, pairs) for bank in candidates)
    full_regret = optimum - bank_value(full_bank, main, pair_beta, pairs)
    main_regret = optimum - bank_value(main_bank, main, pair_beta, pairs)

    threshold = 0.03
    actual = np.abs(pair_beta) >= threshold
    predicted = np.abs(coef_full[1 + k:]) >= threshold
    tp = int(np.sum(actual & predicted))
    f1 = 2 * tp / max(1, int(actual.sum() + predicted.sum()))
    return (
        {
            "full": random_rmse,
            "ablated": observed_rmse,
            "improvement": observed_rmse - random_rmse,
            "randomized_rmse": random_rmse,
            "observed_rmse": observed_rmse,
        },
        {
            "full": full_regret,
            "ablated": main_regret,
            "improvement": main_regret - full_regret,
            "full_bank": list(full_bank),
            "main_only_bank": list(main_bank),
            "bank_changed": full_bank != main_bank,
            "pair_recovery_f1": f1,
        },
    )


def paired_effect(rng: np.random.Generator, base: float, effect: float,
                  n: int) -> np.ndarray:
    heterogeneity = rng.uniform(-0.08, 0.08, n)
    probability = np.clip(base + heterogeneity, 0.05, 0.95)
    u = rng.random(n)
    return (u < np.clip(probability + effect, 0, 1)).astype(float) - (u < probability).astype(float)


def drift_gate(rng: np.random.Generator) -> dict:
    weights = np.asarray([0.35, 0.25, 0.25, 0.15])
    base = np.asarray([0.62, 0.58, 0.65, 0.60])
    source_effect = np.full(4, 0.06)
    target_effect = np.asarray([-0.12, 0.06, 0.06, 0.06])
    blocked = []
    for k in range(4):
        source = paired_effect(rng, base[k], source_effect[k], 128)
        target = paired_effect(rng, base[k], target_effect[k], 128)
        delta = target.mean() - source.mean()
        se = np.sqrt(source.var(ddof=1) / len(source) + target.var(ddof=1) / len(target))
        blocked.append(bool(delta + 1.645 * se < -0.05))

    full_success = no_gate_success = full_n = no_gate_n = 0
    counts = np.asarray(np.round(weights * 2000), dtype=int)
    counts[0] += 2000 - int(counts.sum())
    for k, n in enumerate(counts):
        heterogeneity = rng.uniform(-0.08, 0.08, n)
        probability = np.clip(base[k] + heterogeneity, 0.05, 0.95)
        u = rng.random(n)
        no_gate = u < np.clip(probability + target_effect[k], 0, 1)
        full_effect = 0.0 if blocked[k] else target_effect[k]
        full = u < np.clip(probability + full_effect, 0, 1)
        full_success += int(full.sum()); no_gate_success += int(no_gate.sum())
        full_n += n; no_gate_n += n
    full_vsr = full_success / full_n
    no_gate_vsr = no_gate_success / no_gate_n
    return {
        "full": full_vsr,
        "ablated": no_gate_vsr,
        "improvement": full_vsr - no_gate_vsr,
        "blocked": blocked,
        "drift_false_accept": not blocked[0],
        "stable_false_block_rate": float(np.mean(blocked[1:])),
    }


def fresh_certification(rng: np.random.Generator) -> dict:
    weights = np.asarray([.20, .16, .14, .12, .10, .10, .10, .08])
    mixture = {str(k): float(weight) for k, weight in enumerate(weights)}
    unsafe_certified = unsafe_no_cert = certified_switches = beneficial_switches = 0
    proposals = 60
    for proposal in range(proposals):
        kind = proposal % 3
        if kind == 0:
            gains = np.full(8, 0.03)
        elif kind == 1:
            gains = np.full(8, 0.03); gains[-1] = -0.10
        else:
            gains = np.full(8, -0.02)
        unsafe = bool(weights @ gains <= 0 or gains.min() <= -0.05)
        unsafe_no_cert += unsafe
        estimates = {}
        for k, mean in enumerate(np.linspace(.48, .72, 8)):
            n = 200
            baseline = rng.beta(mean * 25, (1 - mean) * 25, n)
            u = rng.random(n)
            incumbent = u < baseline
            candidate = u < np.clip(baseline + gains[k], 0, 1)
            outcome = candidate.astype(float) - incumbent.astype(float)
            estimates[str(k)] = ppi(outcome, np.zeros(n), np.zeros(1000))
        decision = certify(estimates, mixture, minimum_n=200)
        if decision["passed"]:
            certified_switches += 1
            unsafe_certified += unsafe
            beneficial_switches += kind == 0
    full_rate = unsafe_certified / proposals
    no_cert_rate = unsafe_no_cert / proposals
    return {
        "full": full_rate,
        "ablated": no_cert_rate,
        "improvement": no_cert_rate - full_rate,
        "certified_switches": certified_switches,
        "beneficial_switches": beneficial_switches,
        "unsafe_certified": unsafe_certified,
        "unsafe_without_certification": unsafe_no_cert,
    }


def robust_radius(rng: np.random.Generator) -> dict:
    weights = np.asarray([0.35, 0.30, 0.25, 0.10])
    truth = np.asarray([
        [0.76, 0.75, 0.73, 0.35],
        [0.68, 0.68, 0.68, 0.67],
        [0.63, 0.68, 0.67, 0.61],
        [0.67, 0.64, 0.63, 0.59],
    ])
    truth = np.clip(truth + rng.normal(0, 0.004, truth.shape), 0.05, 0.95)
    validation = np.asarray([
        [np.mean(rng.random(2048) < truth[b, s]) for s in range(4)]
        for b in range(len(truth))
    ])
    robust_choice = int(np.argmax([
        worst_mixture(row, weights, 0.2) for row in validation
    ]))
    nominal_choice = int(np.argmax(validation @ weights))
    robust_value = worst_mixture(truth[robust_choice], weights, 0.2)
    nominal_value = worst_mixture(truth[nominal_choice], weights, 0.2)
    return {
        "full": robust_value,
        "ablated": nominal_value,
        "improvement": robust_value - nominal_value,
        "robust_bank": robust_choice,
        "nominal_bank": nominal_choice,
        "bank_changed": robust_choice != nominal_choice,
        "truth": truth.tolist(),
    }


def role_transport(rng: np.random.Generator) -> dict:
    role_effect = np.asarray([0.08, -0.04, 0.06, -0.02])
    estimates = []
    for k, effect in enumerate(role_effect):
        differences = paired_effect(rng, 0.58 + 0.02 * k, effect, 256)
        estimates.append(differences.mean())
    estimates = np.asarray(estimates)
    pooled = float(estimates.mean())
    templates = (
        (0, 0, 0, 1), (2, 2, 3), (1, 3, 3, 3),
        (0, 2, 2, 2), (0, 1, 2, 3), (1, 1, 2),
    )
    truth = []; structured = []; unstructured = []
    for _ in range(512):
        roles = templates[int(rng.integers(len(templates)))]
        truth.append(float(np.mean(role_effect[list(roles)])))
        structured.append(float(np.mean(estimates[list(roles)])))
        unstructured.append(pooled)
    full_mae = float(np.mean(np.abs(np.asarray(structured) - truth)))
    pooled_mae = float(np.mean(np.abs(np.asarray(unstructured) - truth)))
    return {
        "full": full_mae,
        "ablated": pooled_mae,
        "improvement": pooled_mae - full_mae,
        "estimated_role_effects": estimates.tolist(),
    }


def run_seed(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    randomization, interactions = randomized_and_interactions(rng)
    return {
        "seed": seed,
        "randomized_exposure": randomization,
        "pair_interactions": interactions,
        "drift_gate": drift_gate(rng),
        "fresh_certification": fresh_certification(rng),
        "robust_radius": robust_radius(rng),
        "role_transport": role_transport(rng),
    }


def signflip_p(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    n = len(values)
    count = 0
    for mask in range(1 << n):
        signs = np.asarray([1 if mask & (1 << i) else -1 for i in range(n)])
        count += abs(float(np.mean(values * signs))) >= observed - 1e-15
    return count / (1 << n)


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[samples].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def summarize(rows: list[dict]) -> dict:
    comparisons = []
    for index, component in enumerate(COMPONENTS):
        full = np.asarray([row[component]["full"] for row in rows], dtype=float)
        ablated = np.asarray([row[component]["ablated"] for row in rows], dtype=float)
        improvement = np.asarray([row[component]["improvement"] for row in rows], dtype=float)
        comparisons.append({
            "component": component,
            "full_mean": float(full.mean()),
            "ablated_mean": float(ablated.mean()),
            "improvement_mean": float(improvement.mean()),
            "improvement_cluster_bootstrap_CI95": bootstrap_ci(improvement, 92001 + index),
            "two_sided_exact_signflip_p": signflip_p(improvement),
            "positive_seeds": int(np.sum(improvement > 0)),
            "negative_seeds": int(np.sum(improvement < 0)),
            "zero_seeds": int(np.sum(improvement == 0)),
        })
    ordered = sorted(comparisons, key=lambda x: x["two_sided_exact_signflip_p"])
    running = 0.0
    for index, row in enumerate(ordered):
        running = max(running, (len(ordered) - index) * row["two_sided_exact_signflip_p"])
        row["holm_p"] = min(1.0, running)
    return {
        "seeds": len(rows),
        "components": comparisons,
        "multiplicity": "Holm correction across six frozen component-specific contrasts",
        "bootstrap_replicates": 10000,
        "resampling_unit": "independent confirmation seed",
        "scope": "post-hoc controlled mechanism-activation stress; not natural-task generalization",
    }


def protocol(seeds: list[int]) -> dict:
    source = Path(__file__).resolve()
    return {
        "experiment": "CREDIT v4 mechanism-aligned RQ5b",
        "design_status": "parameters chosen on one disjoint development seed before confirmation",
        "development_seed": DEVELOPMENT_SEED,
        "confirmation_seeds": seeds,
        "components": list(COMPONENTS),
        "primary_contrasts": {
            "randomized_exposure": "coefficient RMSE: randomized factorial minus confounded observed exposure",
            "pair_interactions": "true selected-bank regret: pair model minus main-only model",
            "drift_gate": "target VSR: differential gate minus no gate",
            "fresh_certification": "unsafe-switch rate: certified minus automatic adoption",
            "robust_radius": "true worst-mixture utility: robust minus nominal selection",
            "role_transport": "new-composition MAE: role-structured minus pooled prediction",
        },
        "multiplicity": "Holm correction across six primary contrasts",
        "minimum_seed_resolution": "two-sided exact sign-flip minimum 2/2^16",
        "source_sha256": sha256(source),
        "scope": "controlled synthetic/finite Bernoulli fixtures with known structure; no LLM or live API calls",
        "claim_boundary": "tests whether mechanisms activate under their stated assumptions; does not prove real-world prevalence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("fresh output directory required")
    args.output.mkdir(parents=True)
    write_json(args.output / "frozen_protocol.json", protocol(args.seeds))
    rows = []
    with (args.output / "seed_results.jsonl").open("w") as stream:
        for seed in args.seeds:
            row = run_seed(seed)
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"seed": seed, "status": "complete"}), flush=True)
    write_json(args.output / "summary.json", summarize(rows))
    write_json(args.output / "audit.json", {
        "passed": len(rows) == len(args.seeds) and all(
            np.isfinite(row[component]["improvement"])
            for row in rows for component in COMPONENTS),
        "seeds": len(rows),
        "seed_results_sha256": sha256(args.output / "seed_results.jsonl"),
    })


if __name__ == "__main__":
    main()
