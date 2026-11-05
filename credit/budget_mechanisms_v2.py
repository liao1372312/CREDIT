"""Interaction-order-controlled finite decision kernels and batched graph rollout.

Learned probabilities use recorded counts only. Public response alphabets and
dependency metadata are finite, not unrestricted semantic action encoders.
"""

from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .shared_mechanisms import SharedMechanisms
from .stateful import ACTIONS, NEIGHBORS, ROLES, STATUSES, decision_key, retrieve
from .policy import _worst_mixture


def design(keys, role, order):
    if order not in (1, 2, 3):
        raise ValueError('interaction order must be 1, 2 or 3')
    neighbors = NEIGHBORS[role]
    groups = [group for size in range(1, order + 1) for group in combinations(range(len(neighbors)), size)]
    # History coefficients are separate from exposure interaction order.
    matrix = [[key[1], key[2], *[int(all(key[3] & (1 << i) for i in group)) for group in groups]] for key in keys]
    return np.asarray(matrix, dtype=float), groups


def fit_order(policy, api, order):
    model = SharedMechanisms(policy, api)
    fits = {}
    for role in ROLES:
        records = [r for r in policy if r['key'][0] == role]
        keys = [tuple(r['key']) for r in records]
        x, groups = design(keys, role, order)
        features, labels, weights = [], [], []
        for row, feature in zip(records, x):
            for action, count in enumerate(row['counts']):
                if count:
                    features.append(feature); labels.append(action); weights.append(count)
        estimator = LogisticRegression(C=1., max_iter=2000, tol=1e-6).fit(features, labels, sample_weight=weights)
        probabilities = np.zeros((len(keys), len(ACTIONS)))
        probabilities[:, estimator.classes_] = estimator.predict_proba(x)
        for key, probability, row in zip(keys, probabilities, records):
            model.policy[key] = probability * sum(row['counts'])
        fits[role] = {'groups': groups, 'classes': estimator.classes_.tolist(),
            'coef': estimator.coef_.tolist(), 'intercept': estimator.intercept_.tolist(),
            'iterations': estimator.n_iter_.tolist(), 'feature_columns': int(x.shape[1]),
            'converged': bool(np.max(estimator.n_iter_) < estimator.max_iter)}
        if not fits[role]['converged']:
            raise RuntimeError('conditional optimizer did not converge; future scoring forbidden')
    model.certificate_valid = False
    return model, fits


class BatchForecast:
    """Exact finite state summation for many exposure sets in one call."""
    def __init__(self, model):
        self.model = model
        self.actions = {}; self.responses = {}; self.coverage = {}
        for role in ROLES:
            table = np.zeros((1 << len(NEIGHBORS[role]), 8, 4, 4))
            for mask in range(len(table)):
                exposure = [m for i, m in enumerate(NEIGHBORS[role]) if mask & (1 << i)]
                for state in range(8):
                    for previous_index, previous in enumerate(STATUSES):
                        table[mask, state, previous_index] = model.policy_distribution(decision_key(role, state, previous, exposure))
            self.actions[role] = table
        for epoch, family, role, state, action in model.api:
            key = epoch, family, role
            if key not in self.responses:
                table = np.zeros((8, 4, 33))
                covered = True
                for s in range(8):
                    for index, a in enumerate(ACTIONS):
                        counts = model.api.get((*key, s, a), {})
                        covered &= sum(counts.values()) >= model.min_support
                        for (status, after), probability in model.api_distribution((*key, s, a)).items():
                            destination = 32 if after & 8 else after * 4 + STATUSES.index(status)
                            table[s, index, destination] = probability
                self.responses[key] = table; self.coverage[key] = bool(covered)

    def predict(self, task, exposures):
        count = len(exposures)
        if any(role not in ROLES or (task.epoch, task.family, role) not in self.responses for role in task.roles):
            return np.zeros(count), np.zeros(count, dtype=bool)
        distribution = np.zeros((count, 33))
        distribution[:, 32 if task.initial_state & 8 else task.initial_state * 4] = 1
        supported = np.ones(count, dtype=bool)
        for role in task.roles:
            masks = np.array([sum(1 << i for i, m in enumerate(NEIGHBORS[role]) if m in exposure) for exposure in exposures])
            decisions = self.actions[role][masks]
            joint = np.einsum('bsh,bsha->bsa', distribution[:, :32].reshape(-1, 8, 4), decisions)
            following = np.einsum('bsa,sad->bd', joint, self.responses[(task.epoch, task.family, role)])
            following[:, 32] += distribution[:, 32]
            distribution = following
            # Conservative support uses complete public API input coverage. This
            # experiment starts with historical coverage, not scarce new APIs.
            supported &= self.coverage[(task.epoch, task.family, role)]
        return distribution[:, 32], supported

    def calibrate(self, traces):
        grouped = {}
        contexts = {}
        for trace in traces:
            key = trace.task.family, trace.task.epoch, trace.task.roles, trace.task.initial_state
            contexts.setdefault(key, []).append(trace)
        for rows in contexts.values():
            probabilities, _ = self.predict(rows[0].task, [r.exposure for r in rows])
            for trace, probability in zip(rows, probabilities):
                grouped.setdefault((trace.task.epoch, trace.task.template), []).append((trace.outcome, probability))
        self.model.calibration_error = max((abs(np.mean([y for y, _ in rows]) - np.mean([p for _, p in rows]))
            for rows in grouped.values() if len(rows) >= 16), default=0.)
        return float(self.model.calibration_error)


class BankSearch:
    def __init__(self, forecast, contexts, tokens, capacity, depth, *, radius=.2, token_cost=.000005):
        self.forecast, self.contexts = forecast, contexts
        self.tokens, self.capacity, self.depth = tokens, capacity, depth
        self.radius, self.token_cost = radius, token_cost
        self.cache = {}; self.pool = tuple(tokens)

    def feasible(self, bank):
        return sum(self.tokens[m] for m in bank) <= self.capacity

    def evaluate(self, banks):
        pending = sorted({tuple(sorted(b)) for b in banks if self.feasible(b)} - set(self.cache))
        if not pending:
            return
        values, supported = [], []
        for task in self.contexts:
            probability, support = self.forecast.predict(task, [retrieve(task, bank, self.depth) for bank in pending])
            values.append(probability); supported.append(support)
        values = np.asarray(values).T; supported = np.asarray(supported).T
        for bank, predictions, support in zip(pending, values, supported):
            cost = self.token_cost * sum(self.tokens[m] for m in bank)
            lower = np.maximum(0, predictions - self.forecast.model.calibration_error) * support
            self.cache[bank] = {'mean': float(predictions.mean()), 'coverage': float(support.mean()),
                'objective': _worst_mixture(lower.tolist(), [1.] * len(lower), self.radius) - cost,
                'mean_objective': _worst_mixture(predictions.tolist(), [1.] * len(lower), self.radius) - cost}

    def joint(self, field='objective', passes=4):
        groups = {tuple(sorted(g)) for neighbors in NEIGHBORS.values() for size in (2, 3)
            for g in combinations(neighbors, size)}
        self.evaluate([(), *((m,) for m in self.pool), *groups])
        used_passes = 0; converged = False
        for used_passes in range(1, passes + 1):
            beam = sorted(self.cache, key=lambda b: (-self.cache[b][field], len(b), b))[:3]
            before = max(r[field] for r in self.cache.values()); proposals = set()
            for bank in beam:
                selected = set(bank)
                for m in self.pool:
                    proposals.add(tuple(sorted(selected ^ {m})))
                for old in bank:
                    for new in set(self.pool) - selected:
                        proposals.add(tuple(sorted((selected - {old}) | {new})))
                for group in groups:
                    proposals.add(tuple(sorted(selected | set(group))))
                    proposals.add(tuple(sorted(selected - set(group))))
                    for remove in combinations(bank, min(2, len(bank))):
                        proposals.add(tuple(sorted((selected - set(remove)) | set(group))))
            self.evaluate(proposals)
            if max(r[field] for r in self.cache.values()) <= before + 1e-10:
                converged = True; break
        best = min(self.cache, key=lambda b: (-self.cache[b][field], len(b), b))
        return best, {'evaluated_banks': len(self.cache), 'passes': used_passes, 'converged': converged}

    def pack(self, ordering, *, positive_scores=None):
        bank = []
        for m in ordering:
            if positive_scores is not None and positive_scores[m] <= 0:
                continue
            if self.feasible(bank + [m]):
                bank.append(m)
        return tuple(sorted(bank))

    def scalar(self, seed):
        rng = np.random.default_rng(seed); full = set(self.pool)
        coalitions = [(), tuple(full), *[tuple(full - {m}) for m in self.pool]]
        paths = []
        for _ in range(32):
            order = rng.permutation(self.pool).tolist(); mask = set(); path = []
            for m in order:
                before = tuple(sorted(mask)); mask.add(m); after = tuple(sorted(mask))
                coalitions.extend([before, after]); path.append((m, before, after))
            paths.append(path)
        # Scalar attribution evaluates archived candidate interventions up to the
        # largest archive allowance, then packs an actual deployable bank.
        archive = BankSearch(self.forecast, self.contexts, self.tokens, sum(self.tokens.values()), self.depth,
            radius=self.radius, token_cost=0)
        archive.evaluate(coalitions)
        full_value = archive.cache[tuple(sorted(full))]['mean']
        loo = {m: full_value - archive.cache[tuple(sorted(full - {m}))]['mean'] for m in self.pool}
        phi = dict.fromkeys(self.pool, 0.)
        for path in paths:
            for m, before, after in path:
                phi[m] += (archive.cache[after]['mean'] - archive.cache[before]['mean']) / len(paths)
        ordered = lambda scores: sorted(self.pool, key=lambda m: (-scores[m], m))
        return self.pack(ordered(loo)), self.pack(ordered(phi)), {'LOO': loo, 'MC_Shapley': phi}
