"""One shared robust bank objective for selection and outcome-particle VOI."""

from collections import defaultdict
from itertools import combinations
from math import factorial

import numpy as np

from .emp_binding_pilot import bank_name
from .emp_binding_tools_v3 import retrieve_compatible
from .emp_structural import fingerprint, public_contract
from .policy import _worst_mixture


def coalitions(pool, capacity=None):
    limit = len(pool) if capacity is None else min(capacity, len(pool))
    return [list(c) for size in range(limit + 1) for c in combinations(pool, size)]


def exposure_cell(edge, notes):
    return fingerprint({'public_contract': public_contract(edge), 'consumer_query': edge['consumer_query'],
                        'notes': sorted(n['uid'] for n in notes)})


def robust_value(values, edges, bank, radius=.2, storage_cost=.01):
    groups = defaultdict(list)
    for value, edge in zip(values, edges):
        groups[edge['workflow_group']].append(float(value))
    means = [float(np.mean(v)) for _, v in sorted(groups.items())]
    if not means:
        raise ValueError('future context portfolio is empty')
    return _worst_mixture(means, [1 / len(means)] * len(means), radius) - storage_cost * len(bank)


def choose(model, future, banks, call_cost=.02, radius=.2, storage_cost=.01):
    scores = []
    for bank in banks:
        predictions = [model.predict(e, retrieve_compatible(e, bank)) for e in future]
        # Score the same complete portfolio for every bank. Unknowns have zero
        # supported success mass and the full remaining call budget cost.
        values = [p['known_success_mass'] - call_cost * (p['expected_consumer_calls'] if p['supported'] else 2)
                  for p in predictions]
        scores.append({'bank_uid': bank_name(bank), 'value': robust_value(values, future, bank, radius, storage_cost),
            'supported_contexts': sum(p['supported'] for p in predictions), 'contexts': len(future),
            'mean_unknown_mass': float(np.mean([p['unknown_mass'] for p in predictions])),
            'criterion': 'empirical supported success mass minus worst unknown call cost; not a confidence bound'})
    best = min(scores, key=lambda s: (-s['value'], len(next(b for b in banks if bank_name(b) == s['bank_uid'])), s['bank_uid']))
    return next(b for b in banks if bank_name(b) == best['bank_uid']), scores


def exact_attribution(rows, pool, all_banks):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['bank_uid']][row['workflow_group']].append(float(row['final']['binding_contract_success']) - .02 * row['consumer_calls'])
    utility = {key: float(np.mean([np.mean(v) for v in groups.values()])) for key, groups in grouped.items()}
    require_names = {bank_name(b) for b in all_banks}
    if set(utility) != require_names:
        raise ValueError('full-coalition attribution lacks training interventions')
    full = bank_name(pool); n = len(pool); loo, shapley = {}, {}
    for memory in pool:
        remaining = [m for m in pool if m['uid'] != memory['uid']]
        loo[memory['uid']] = utility[full] - utility[bank_name(remaining)]
        total = 0.
        for background in coalitions(remaining):
            k = len(background)
            total += factorial(k) * factorial(n - k - 1) / factorial(n) * (
                utility[bank_name(background + [memory])] - utility[bank_name(background)])
        shapley[memory['uid']] = total
    return {'full_bank_LOO': loo, 'exact_bank_Shapley': shapley, 'utility': utility,
        'efficiency_residual': sum(shapley.values()) - (utility[full] - utility['empty']),
        'estimand': 'training workflow mean bank utility with actual depth-two retrieval; not a published system port'}


def rank_revalidation(model, future, banks, observed, *, seed=42, particles=128,
                      probe_cost=.04, call_cost=.02, radius=.2, storage_cost=.01):
    """Beta outcome-particle approximation to prospective VOI.

    The policy/API learner is separate. Particles express an explicit utility
    prior for unobserved exposure cells, not a mechanism error certificate.
    A probe updates outcomes on one cell and reoptimizes the same fixed banks.
    """
    rng = np.random.default_rng(seed); cells, matrix = {}, []
    for bank in banks:
        indices = []
        for edge in future:
            exposure = retrieve_compatible(edge, bank); key = exposure_cell(edge, exposure)
            cells.setdefault(key, {'edge': edge, 'bank': bank, 'exposure': exposure})
            indices.append(key)
        matrix.append(indices)
    ordered = sorted(cells); position = {key: i for i, key in enumerate(ordered)}
    indices = np.asarray([[position[k] for k in row] for row in matrix])
    probabilities, costs = [], []
    for key in ordered:
        spec = cells[key]; prediction = model.predict(spec['edge'], spec['exposure'])
        if prediction['supported']:
            probabilities.append(np.full(particles, prediction['final_success']))
            costs.append(prediction['expected_consumer_calls'])
        else:
            probabilities.append(rng.beta(1, 1, particles)); costs.append(2.)
    particles_matrix = np.asarray(probabilities).T
    costs = np.asarray(costs)
    def optimum(weights):
        utilities = weights @ particles_matrix - call_cost * costs
        return max(robust_value(utilities[row], future, bank, radius, storage_cost)
                   for row, bank in zip(indices, banks))
    prior = np.full(particles, 1 / particles); before = optimum(prior)
    rows = []
    for i, key in enumerate(ordered):
        spec = cells[key]
        if key in observed:
            continue
        p = particles_matrix[:, i]; py = float(prior @ p)
        gains = []
        for success, probability in [(True, py), (False, 1 - py)]:
            likelihood = p if success else 1 - p
            posterior = prior * likelihood
            if posterior.sum() <= 0:
                continue
            posterior /= posterior.sum()
            gains.append(probability * (optimum(posterior) - before))
        gain = max(0., sum(gains))
        rows.append({'cell_uid': key, 'edge_uid': spec['edge']['uid'], 'bank_uid': bank_name(spec['bank']),
            'exposure': [m['uid'] for m in spec['exposure']], 'expected_gain': gain,
            'probe_cost': probe_cost, 'voi': gain - probe_cost, 'particles': particles,
            'prior': 'Beta(1,1) on unobserved final-outcome probabilities',
            'approximation': 'outcome-particle posterior; fixed call-cost forecasts; no confidence certificate'})
    return sorted(rows, key=lambda row: (-row['voi'], row['cell_uid']))
