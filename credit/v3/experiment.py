"""One revision-3 update cycle on the bounded local integration adapter."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from ..stateful import FAMILIES
from .drift import Level, TransportGate, posterior_success, role_voi
from .factorial import admitted_pairs, eligible_slots, estimate, regular_fraction, context_probabilities
from .local import (POOL, StatefulExecutor, context, differential_probes, factory,
                    memories, ranked, retriever, stable_seed)
from .outcome import bootstrap_ensemble
from .selection import BankSelector, Ledger, screen_banks
from .types import reserved_slots


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def collect_designs(contexts, pool, executor, ledger, *, phase, seed, draws=2, k1=4, k2=2,
                    audit=False):
    rng = np.random.default_rng(seed)
    episodes, reports = [], []
    for base in contexts:
        slots = eligible_slots(base, pool, ranked(base, tuple(m.uid for m in pool)), k1, k2, rng)
        fraction = regular_fraction(len(slots), admitted_pairs(base, slots), seed=stable_seed(base.uid))
        assignments, outcomes = [], []
        for draw in range(draws):
            checkpoint = replace(base, uid=f"{base.uid}:draw-{draw}", seed=stable_seed(base.seed, draw))
            rows, ys, reference = fraction.draw(rng), [], None
            for row in rows:
                frame = reserved_slots(slots, row)
                if reference is None:
                    episode = ledger.execute(phase, executor, checkpoint, frame.exposure, slots=frame)
                    reference = episode
                else:
                    if ledger.remaining(phase) < 1:
                        raise ValueError("factorial budget exhausted before replay")
                    episode = executor.replay(checkpoint, frame.exposure, reference, slots=frame)
                    ledger.charge(phase, episode)
                if not episode.replay_valid:
                    # The physical attempt remains charged; reject this context rather than
                    # assigning a fabricated failure to an unexecuted potential outcome.
                    raise ValueError("unsafe factorial replay; context excluded from estimation")
                episode = replace(episode, randomized=True, assignment_probability=2.0 ** -len(slots))
                ledger.records[-1]["episode"] = episode
                episodes.append(episode)
                ys.append(episode.outcome)
            assignments.append(rows)
            outcomes.append(ys)
            if audit:
                full = ledger.execute("replay_audit", executor, checkpoint, episode.exposure)
                if (full.state_hash != episode.state_hash or full.verified != episode.verified
                        or full.events != episode.events or full.observations != episode.observations):
                    raise ValueError("incremental replay differs from full replay")
        effects = estimate(fraction, assignments, outcomes)
        reports.append({"context": asdict(base), "memory_ids": [m.uid for m in slots],
                        "slot_token_lengths": [m.cost for m in slots], "runs_per_draw": fraction.runs,
                        "generator_columns": fraction.columns, "minimum_fraction_proven": fraction.minimum_proven,
                        "effects": effects["effects"], "beta": effects["beta"].tolist(),
                        "banzhaf": effects["banzhaf"].tolist(),
                        "draw_standard_error": [float(v) if np.isfinite(v) else None
                                                for v in effects["draw_standard_error"]],
                        "assignment_probability": effects["assignment_probability"], "draws": draws,
                        "reserved_slots_length_matched": True})
    return episodes, reports


def run(args):
    output = args.output
    if output.exists():
        raise FileExistsError("use a fresh output directory to preserve frozen runs")
    if min(args.training_contexts, args.calibration_contexts, args.draws, args.members, args.epochs) < 1:
        raise ValueError("positive design/model counts required")
    if args.certification_n < 200 or args.pilot_n < 2 or args.unlabeled_n < 2:
        raise ValueError("normal certification requires at least 200 paired episodes per stratum")
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    (output / "source").mkdir()
    source = {str(p): digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    source["credit/stateful.py"] = digest("credit/stateful.py")
    for p in source:
        (output / "source" / Path(p).name).write_bytes(Path(p).read_bytes())
    protocol = {"experiment_family": "legacy bounded local simulation",
                "paper_7944_protocol": False, "source_sha256": source,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "primary_endpoint": "all required node postconditions AND submitted state",
                "unknown_is_failure": True, "strata": [f"{f}/{e}" for f in FAMILIES for e in ("source", "target")],
                "scope": "bounded stochastic state API, injected drift, lexical token ids; no real-provider/LLM claims",
                "certification": "normal, asymptotic alpha1+alpha2=0.10 per cycle, not a 0.05 finite-sample promise",
                "split": "source train/calibration; new graph screening/halving/pilot/certification/test ids and noise",
                "retriever": "fixed metadata; hidden_fix never ordinarily retrieved but eligible for randomized K2",
                "model": "shared content/role encoders, c/r heads, symmetric pairs, L1 type gates, isotonic calibration",
                "baselines": "legacy baselines remain historical; obsolete baselines are not reinstated"}
    write(output / "frozen_protocol.json", protocol)
    pool, executor = memories(), StatefulExecutor(repair_budget=1)
    write(output / "memories.json", [asdict(m) | {"scope": sorted(m.scope) if m.scope is not None else None}
                                    for m in pool])
    # A full cube is a conservative upper cap for any admissible six-slot fraction.
    max_runs = 64
    bank_slots = args.finalists + 1
    strata = len(protocol["strata"])
    ledger = Ledger({"factorial": args.training_contexts * args.draws * max_runs,
                     "calibration": args.calibration_contexts * max_runs,
                     "replay_audit": args.training_contexts * args.draws,
                     "halving": args.gamma, "pilot": args.pilot_n * strata * bank_slots,
                     "certification": args.certification_n * strata * bank_slots,
                     "test": args.test_n * strata * 2})
    train_contexts = [context(FAMILIES[i % 4], "source", "training", i, args.seed,
                              template="train" if (i // 4) % 2 == 0 else "alternate")
                      for i in range(args.training_contexts)]
    calibration_contexts = [context(FAMILIES[i % 4], "source", "calibration", i, args.seed,
                                    template="train" if (i // 4) % 2 else "alternate")
                            for i in range(args.calibration_contexts)]
    train, designs = collect_designs(train_contexts, pool, executor, ledger,
                                     phase="factorial", seed=args.seed, draws=args.draws, audit=True)
    calibration, calibration_designs = collect_designs(calibration_contexts, pool, executor, ledger,
                                                       phase="calibration", seed=args.seed + 1, draws=1)
    write(output / "factorial_estimates.json", designs)
    write(output / "calibration_designs.json", calibration_designs)
    models = bootstrap_ensemble(pool, train, calibration, members=args.members,
                                seed=args.seed, epochs=args.epochs)
    for i, model in enumerate(models):
        model.save(output / f"outcome_model_{i}.pt")
    write(output / "training_reports.json", [m.fit_report for m in models])
    support = Counter()
    for episode in train:
        nodes = {n.uid: n for n in episode.context.nodes}
        for o in episode.observations:
            if o.label in {"ok", "agent"}:
                n = nodes[o.node]
                support[(n.role.key, n.version)] += 1
    evidence, fixtures = differential_probes(seed=args.seed, support=support, fixtures=args.fixtures)
    gate = TransportGate(evidence)
    write(output / "transfer_levels.json", [asdict(p) for p in evidence])
    write(output / "differential_fixtures.json", fixtures)
    # Next-context priorities are computed from a bootstrap ensemble; the first
    # cycle's contexts were scheduled uniformly before any outcome/model existed.
    consistency = []
    max_effects = max(len(r["effects"]) for r in designs)
    coefficients = np.zeros((len(models), len(train_contexts), max_effects))
    lookup = {m.uid: m for m in pool}
    for index, (report, c) in enumerate(zip(designs, train_contexts)):
        measured = np.asarray(report["beta"])
        member_coefficients = [m.coefficients(c, tuple(lookup[i] for i in report["memory_ids"]), report["effects"])
                               for m in models]
        for member, values in enumerate(member_coefficients):
            coefficients[member, index, :len(values)] = values
        fitted = np.mean(member_coefficients, axis=0)
        consistency.append({"context": c.uid, "measured_beta": measured.tolist(),
                            "model_beta": fitted.tolist(), "coefficient_rmse": float(np.sqrt(np.mean((fitted - measured)**2)))})
    write(output / "design_consistency.json", consistency)
    priorities = context_probabilities(coefficients, [len(c.nodes) * 2 for c in train_contexts])
    write(output / "next_context_priorities.json", {c.uid: float(p) for c, p in zip(train_contexts, priorities)})
    mixture = {k: 1 / strata for k in protocol["strata"]}
    selector = BankSelector(pool, models, gate, retriever(args.exposure_depth), factory(args.seed),
                            executor, ledger, mixture=mixture, capacity=args.capacity,
                            cost_penalty=args.cost_penalty,
                            transported_strata={k for k in mixture if gate.transported(
                                context(*k.split("/"), "transport-manifest", 0, args.seed, template="future"))})
    incumbent = ()
    screening_contexts = selector.contexts("screening-unlabeled", args.screen_n)
    bundles = {tuple(sorted((a.uid, b.uid))) for c in train_contexts
               for i, j in admitted_pairs(c, pool) for a, b in [(pool[i], pool[j])]}
    screening = screen_banks(pool, incumbent, lambda b: selector.model_score(screening_contexts, b),
                             capacity=args.capacity, admitted_bundles=bundles, keep=args.candidates)
    write(output / "screening.json", screening)
    halving = selector.halve(screening["banks"], gamma=args.gamma, finalists=args.finalists)
    write(output / "halving.json", halving)
    switch = selector.switch(incumbent, halving["finalists"], pilot_n=args.pilot_n,
                             certification_n=args.certification_n, unlabeled_n=args.unlabeled_n,
                             freeze=lambda value: write(output / "certification_freeze.json", value))
    write(output / "switch_decision.json", switch)
    # VOI values are scheduling scores only. They cannot trigger an uncertified switch.
    queue, baseline = [], selector.model_score(screening_contexts, switch["bank"])
    for p in evidence:
        if p.level == Level.V:
            continue
        hypothetical = TransportGate([replace(q, level=Level.V) if q.role == p.role and q.version == p.version
                                       else q for q in evidence])
        probability = posterior_success(0, int(p.level == Level.D))
        value = role_voi(switch["bank"], pool, p.role,
                         lambda b: selector.model_score(screening_contexts, b, hypothetical),
                         capacity=args.capacity, probability=probability, probe_calls=args.fixtures,
                         probe_penalty=args.probe_penalty, incumbent_score=baseline)
        queue.append({"role": p.role, "version": p.version, "level": p.level.value,
                      "beta_posterior_success": probability, "voi": value})
    write(output / "probe_queue.json", sorted(queue, key=lambda q: -q["voi"]))
    status = []
    for m in pool:
        blocked = [p.version for p in evidence if p.level != Level.V and (m.scope is None or p.role in m.scope)]
        status.append({"memory": m.uid, "retained": m.uid in switch["bank"],
                       "status": "quarantined" if blocked else "active", "blocked_versions": blocked,
                       "archived": False, "archive_reason": "no paired upper joint-marginal evidence; retain reversible record"})
    write(output / "memory_status.json", status)
    evaluation_freeze = {"selected_bank": switch["bank"], "model_sha256": {
        p.name: digest(p) for p in output.glob("outcome_model_*.pt")},
        "decision_sha256": digest(output / "switch_decision.json")}
    write(output / "test_freeze.json", evaluation_freeze)
    test_contexts = selector.contexts("heldout-test", args.test_n)
    banks = list(dict.fromkeys([incumbent, tuple(switch["bank"])]))
    test_values = selector._executions("test", test_contexts, banks)
    write(output / "heldout_test.json", {"|".join(b): {k: v.tolist() for k, v in rows.items()}
                                         for b, rows in test_values.items()})
    write(output / "next_cycle_evidence.json", {"randomized_episode_ids": sorted({e.context.uid for e in train}),
                                                "logged_only_after_scoring": [c.uid for cs in test_contexts.values() for c in cs],
                                                "rule": "ordinary test traces can pretrain a only; exposure heads require new randomized designs"})
    with (output / "raw_episodes.jsonl").open("w") as stream:
        for record in ledger.records:
            stream.write(json.dumps({"phase": record["phase"], "episode": asdict(record["episode"])},
                                    sort_keys=True, allow_nan=False) + "\n")
    counts = ledger.report()
    counts.update(differential_fixture_calls=sum(r["api_calls"] for r in fixtures),
                  total_cycle_units=counts["total_executed_episodes"] + sum(r["api_calls"] for r in fixtures))
    write(output / "costs.json", counts)
    summary = {"experiment_family": "legacy bounded local simulation",
               "paper_7944_protocol": False,
               "switched": switch["switched"], "bank": switch["bank"],
               "test_macro_vsr": {"|".join(b): float(np.mean([v.mean() for v in rows.values()]))
                                  for b, rows in test_values.items()},
               "transfer_levels": dict(Counter(p.level.value for p in evidence)), "costs": counts,
               "scope": protocol["scope"], "paper_agent_experiments_complete": False,
               "certification_guarantee": "asymptotic per cycle; no lifetime alpha spending implemented"}
    write(output / "summary.json", summary)
    write(output / "artifact_manifest.json", {str(p.relative_to(output)): digest(p)
                                               for p in sorted(output.rglob("*")) if p.is_file()})
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/credit_v3_local"))
    for name, value in [("seed", 503), ("training-contexts", 16), ("calibration-contexts", 32),
                        ("draws", 2), ("members", 3), ("epochs", 200), ("fixtures", 96),
                        ("capacity", 80), ("exposure-depth", 4), ("candidates", 8), ("finalists", 2),
                        ("gamma", 512), ("pilot-n", 100), ("certification-n", 200),
                        ("unlabeled-n", 1000), ("screen-n", 16), ("test-n", 32)]:
        p.add_argument("--" + name, type=int, default=value)
    p.add_argument("--cost-penalty", type=float, default=0.0001)
    p.add_argument("--probe-penalty", type=float, default=0.0001)
    return p


def main():
    run(parser().parse_args())
