"""Retrieval-aware screening, execution halving and fresh certified switching.

The executor and context factory are injected: statistical code never imports
hidden simulator kernels. All paired phases share tasks and seeds between
banks, while different phases have disjoint context ids and noise seeds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
import math

import numpy as np

from .inference import certify, ppi, tune_lambda, worst_mixture


@dataclass
class Ledger:
    limits: dict[str, int]
    episodes: dict[str, int] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=dict)
    unknown: dict[str, int] = field(default_factory=dict)
    records: list = field(default_factory=list)

    def remaining(self, phase):
        return self.limits.get(phase, 0) - self.episodes.get(phase, 0)

    def charge(self, phase, episode):
        if self.remaining(phase) < 1:
            raise ValueError(f"{phase} episode budget exhausted")
        self.episodes[phase] = self.episodes.get(phase, 0) + 1
        self.calls[phase] = self.calls.get(phase, 0) + episode.calls
        self.excluded[phase] = self.excluded.get(phase, 0) + int(not episode.replay_valid)
        self.unknown[phase] = self.unknown.get(phase, 0) + int(episode.verified is None)
        self.records.append({"phase": phase, "episode": episode})

    def execute(self, phase, executor, context, exposure, *, slots=None):
        if self.remaining(phase) < 1:
            raise ValueError(f"{phase} episode budget exhausted before execution")
        episode = executor.run(context, exposure, slots=slots)
        self.charge(phase, episode)
        return episode

    def report(self):
        return {"episode_limits": self.limits, "episodes": self.episodes, "api_calls": self.calls,
                "excluded_unsafe_episodes": self.excluded, "unknown_verification": self.unknown,
                "total_executed_episodes": sum(self.episodes.values()),
                "total_api_calls": sum(self.calls.values())}


def bank_cost(bank, memories):
    lookup = {m.uid: m.cost for m in memories}
    if len(set(bank)) != len(bank) or not set(bank) <= lookup.keys():
        raise ValueError("invalid bank ids")
    return sum(lookup[m] for m in bank)


def screen_banks(memories, incumbent, score, *, capacity, admitted_bundles=(),
                 keep=8, iterations=4, beam=3):
    if keep < 1 or capacity < 0 or iterations < 1:
        raise ValueError("invalid screening configuration")
    pool = {m.uid for m in memories}
    cache = {}

    def evaluate(bank):
        bank = tuple(sorted(set(bank)))
        if set(bank) <= pool and bank_cost(bank, memories) <= capacity and bank not in cache:
            cache[bank] = float(score(bank))

    if bank_cost(incumbent, memories) > capacity:
        raise ValueError("incumbent exceeds the storage budget")
    bundles = tuple(tuple(sorted(set(b))) for b in admitted_bundles)
    for start in ((), tuple(incumbent), *((m.uid,) for m in memories), *bundles):
        evaluate(start)
    converged = False
    for iteration in range(iterations):
        before = max(cache.values())
        leaders = sorted(cache, key=lambda b: (-cache[b], bank_cost(b, memories), b))[:beam]
        for bank in leaders:
            selected = set(bank)
            for m in pool:
                evaluate(selected ^ {m})
            for old in bank:
                for new in pool - selected:
                    evaluate((selected - {old}) | {new})
            for bundle in bundles:
                evaluate(selected | set(bundle))
                evaluate(selected - set(bundle))
                for remove in combinations(bank, min(2, len(bank))):
                    evaluate((selected - set(remove)) | set(bundle))
        if max(cache.values()) <= before + 1e-10:
            converged = True
            break
    ordered = sorted(cache, key=lambda b: (-cache[b], bank_cost(b, memories), b))[:keep]
    return {"banks": ordered, "scores": {"|".join(b): cache[b] for b in ordered},
            "evaluated": len(cache), "passes": iteration + 1, "local_stopping_met": converged}


class BankSelector:
    def __init__(self, memories, models, gate, retrieve, factory, executor, ledger, *,
                 mixture, capacity=512, radius=0.2, cost_penalty=0.0001,
                 transported_strata=None):
        self.memories, self.models, self.gate = tuple(memories), tuple(models), gate
        self.retrieve, self.factory, self.executor, self.ledger = retrieve, factory, executor, ledger
        self.mixture = dict(mixture)
        self.capacity, self.radius, self.cost_penalty = capacity, radius, cost_penalty
        self.transported_strata = frozenset(transported_strata or ())
        if not self.transported_strata <= self.mixture.keys():
            raise ValueError("transport manifest names an unknown stratum")
        self.used_contexts = set()
        self.used_seeds = set()

    def contexts(self, phase, count):
        values = {k: tuple(self.factory(k, phase, i) for i in range(count)) for k in self.mixture}
        ids = [c.uid for group in values.values() for c in group]
        if any(not self.gate.transported(c) for k in self.transported_strata for c in values[k]):
            raise ValueError("task violates the independently frozen transport domain")
        if len(set(ids)) != len(ids) or self.used_contexts & set(ids):
            raise ValueError("context factory reused tasks across independent phases")
        self.used_contexts.update(ids)
        seeds = [c.seed for group in values.values() for c in group]
        if len(set(seeds)) != len(seeds) or self.used_seeds & set(seeds):
            raise ValueError("independent tasks/strata reused execution noise")
        self.used_seeds.update(seeds)
        return values

    def exposures(self, context, bank, gate=None):
        gate = self.gate if gate is None else gate
        candidates = gate.candidates(context, [m for m in self.memories if m.uid in bank])
        return self.retrieve(context, tuple(m.uid for m in candidates))

    def predictions(self, contexts, bank, gate=None):
        gate = self.gate if gate is None else gate
        items = [(c, self.exposures(c, bank, gate)) for c in contexts]
        if not self.models:
            return np.zeros(len(contexts))
        values = np.mean([m.predict_many(items) for m in self.models], axis=0)
        return np.asarray([v if gate.transported(c) else 0.0 for c, v in zip(contexts, values)])

    def model_score(self, contexts, bank, gate=None):
        keys = list(self.mixture)
        means = [float(self.predictions(contexts[k], bank, gate).mean()) for k in keys]
        return (worst_mixture(means, [self.mixture[k] for k in keys], self.radius)
                - self.cost_penalty * bank_cost(bank, self.memories))

    def _executions(self, phase, contexts, banks):
        values = {bank: {} for bank in banks}
        for k, tasks in contexts.items():
            for bank in banks:
                episodes = [self.ledger.execute(phase, self.executor, c, self.exposures(c, bank)) for c in tasks]
                if any(not e.replay_valid for e in episodes):
                    raise ValueError("unsafe executions cannot enter bank-selection estimates")
                values[bank][k] = np.asarray([e.outcome for e in episodes], dtype=float)
        return values

    def halve(self, banks, *, gamma=512, finalists=2):
        survivors = list(dict.fromkeys(tuple(b) for b in banks))
        if not survivors or finalists < 1 or gamma < 0:
            raise ValueError("invalid halving candidates/budget")
        rounds = math.ceil(math.log2(max(1, len(survivors) / finalists)))
        results, spent = [], 0
        for round_index in range(rounds):
            remaining_rounds = rounds - round_index
            budget = min((gamma - spent) // remaining_rounds, self.ledger.remaining("halving"))
            per_stratum = budget // (len(survivors) * len(self.mixture))
            if per_stratum < 2:
                return {"finalists": [], "rounds": results, "reason": "insufficient halving budget"}
            contexts = self.contexts(f"halving-{round_index}", per_stratum)
            real = self._executions("halving", contexts, survivors)
            # lambda=0 is PPI's direct-execution fallback, independent of surrogate bias.
            ranks = {b: worst_mixture([float(real[b][k].mean()) for k in self.mixture],
                                     list(self.mixture.values()), self.radius)
                        - self.cost_penalty * bank_cost(b, self.memories) for b in survivors}
            spent += per_stratum * len(survivors) * len(self.mixture)
            survivors = sorted(survivors, key=lambda b: (-ranks[b], bank_cost(b, self.memories), b))[
                :max(finalists, math.ceil(len(survivors) / 2))]
            results.append({"round": round_index, "episodes_per_stratum_bank": per_stratum,
                            "scores": {"|".join(b): v for b, v in ranks.items()}, "survivors": survivors})
        return {"finalists": survivors[:finalists], "rounds": results}

    def switch(self, incumbent, finalists, *, pilot_n=100, certification_n=200,
               unlabeled_n=1000, alpha1=0.05, alpha2=0.05, margin=0.05,
               method="normal", minimum_n=200, freeze=None):
        incumbent = tuple(sorted(incumbent))
        finalists = list(dict.fromkeys(tuple(sorted(b)) for b in finalists if tuple(sorted(b)) != incumbent))
        if not finalists:
            return {"bank": incumbent, "switched": False, "reason": "no challenger", "certificates": []}
        banks = [incumbent] + finalists
        required_pilot = pilot_n * len(banks) * len(self.mixture)
        required_cert = certification_n * len(banks) * len(self.mixture)
        if pilot_n < 2 or certification_n < minimum_n or unlabeled_n < 2:
            return {"bank": incumbent, "switched": False, "reason": "insufficient certification sizes", "certificates": []}
        if (required_pilot > self.ledger.remaining("pilot")
                or required_cert > self.ledger.remaining("certification")):
            return {"bank": incumbent, "switched": False, "reason": "insufficient certification budget", "certificates": []}
        pilot_contexts = self.contexts("pilot", pilot_n)
        pilot = self._executions("pilot", pilot_contexts, banks)
        labeled = self.contexts("certification", certification_n)
        unlabeled = self.contexts("certification-unlabeled", unlabeled_n)
        lambdas, predictions = {}, {}
        for b in banks:
            predictions[b] = {k: {"pilot": self.predictions(pilot_contexts[k], b),
                                  "labeled": self.predictions(labeled[k], b),
                                  "unlabeled": self.predictions(unlabeled[k], b)} for k in self.mixture}
        for b in finalists:
            lambdas[b] = {k: tune_lambda(pilot[b][k] - pilot[incumbent][k],
                                        predictions[b][k]["pilot"] - predictions[incumbent][k]["pilot"],
                                        labeled_n=certification_n, unlabeled_n=unlabeled_n,
                                        transported=k in self.transported_strata)
                          for k in self.mixture}
        # All finalists, model predictions and pilot weights freeze before any certification label.
        if freeze is not None:
            freeze({"incumbent": incumbent, "finalists": finalists, "mixture": self.mixture,
                    "pilot_weights": {"|".join(b): w for b, w in lambdas.items()},
                    "certification_contexts": {k: [asdict(c) for c in cs] for k, cs in labeled.items()},
                    "prediction_hash_input": {"|".join(b): {k: {p: v.tolist() for p, v in d.items()}
                                                            for k, d in predictions[b].items()} for b in banks},
                    "model_report": [m.fit_report for m in self.models]})
        real = self._executions("certification", labeled, banks)
        certificates = []
        for b in finalists:
            estimates = {k: ppi(real[b][k] - real[incumbent][k],
                                predictions[b][k]["labeled"] - predictions[incumbent][k]["labeled"],
                                predictions[b][k]["unlabeled"] - predictions[incumbent][k]["unlabeled"],
                                lambdas[b][k]) for k in self.mixture}
            certificate = certify(estimates, self.mixture,
                                  cost_difference=bank_cost(b, self.memories) - bank_cost(incumbent, self.memories),
                                  cost_penalty=self.cost_penalty, finalists=len(finalists), alpha1=alpha1,
                                  alpha2=alpha2, margin=margin, method=method, minimum_n=minimum_n)
            certificate.update(bank=b, estimates={k: asdict(v) for k, v in estimates.items()},
                               paired_correlation={k: float(np.corrcoef(real[b][k], real[incumbent][k])[0, 1])
                                                   if np.std(real[b][k]) * np.std(real[incumbent][k]) > 0 else None
                                                   for k in self.mixture})
            certificates.append(certificate)
        passed = [c for c in certificates if c["passed"]]
        best = max(passed, key=lambda c: c["aggregate_lower"] - c["cost_threshold"]) if passed else None
        return {"bank": best["bank"] if best else incumbent, "switched": bool(best),
                "certificates": certificates, "finalist_count": len(finalists),
                "fresh_certification": True, "labeled_episodes_per_stratum_bank": certification_n}

    def switch_registered(self, incumbent, finalist, *, workload, attempt_index,
                          freeze, pilot_n=100):
        """Execute paper 7944's single-finalist, spent-alpha switch attempt.

        ``freeze`` persists the registered candidate, target, predictions, and
        sampling plan before any certification outcomes are obtained. The
        caller must advance ``attempt_index`` for every initiated attempt,
        including a budget-abandoned attempt.
        """
        from credit.paper_7944_protocol import CatalogFactory, PAPER_7944, attempt_alpha

        if not callable(freeze):
            raise ValueError("a persistent pre-certification freeze callback is required")
        attempt_alpha(attempt_index)
        spec = next((w for w in PAPER_7944.workloads if w.name == workload), None)
        if spec is None:
            raise ValueError("unknown registered workload")
        if len(self.mixture) != spec.protected_guards or any(
                not np.isclose(weight, 1 / spec.protected_guards)
                for weight in self.mixture.values()):
            raise ValueError("registered guards require uniform target weights")
        if (not isinstance(self.factory, CatalogFactory) or
                self.factory.attempt_index != attempt_index or
                set(self.factory.catalogs) != set(self.mixture)):
            raise ValueError("registered certification requires matching frozen guard catalogs")
        if (len(self.models) != 1 or
                not getattr(self.models[0], "paper_7944_ensemble", False)):
            raise ValueError("registered certification requires the calibrated five-member ensemble")
        if (self.capacity != PAPER_7944.capacity_tokens or
                not np.isclose(self.radius, PAPER_7944.mixture_l1_radius) or
                not np.isclose(self.cost_penalty, PAPER_7944.storage_penalty_per_token)):
            raise ValueError("selector settings differ from the registered paper protocol")
        finalist = tuple(sorted(finalist))
        if not finalist or finalist == tuple(sorted(incumbent)):
            raise ValueError("freeze exactly one distinct finalist")
        if bank_cost(finalist, self.memories) > self.capacity:
            raise ValueError("finalist exceeds registered storage capacity")
        alpha_t = attempt_alpha(attempt_index)

        def registered_freeze(record):
            freeze({**record, "workload": workload, "attempt_index": attempt_index,
                    "attempt_alpha": alpha_t, "paper_protocol": PAPER_7944.manifest()})

        result = self.switch(incumbent, [finalist], pilot_n=pilot_n,
                             certification_n=PAPER_7944.fresh_pairs_per_guard,
                             unlabeled_n=spec.prediction_draws_per_guard,
                             alpha1=alpha_t / 2, alpha2=alpha_t / 2,
                             margin=PAPER_7944.noninferiority_margin,
                             method="normal", minimum_n=PAPER_7944.fresh_pairs_per_guard,
                             freeze=registered_freeze)
        return {**result, "workload": workload, "attempt_index": attempt_index,
                "attempt_alpha": alpha_t, "registered_single_finalist": True}
