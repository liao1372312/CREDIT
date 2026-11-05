"""Revalidation uses the same supported-value objective as bank selection.

For this frozen deterministic adapter, a complete probe resolves one exact
exposure cell. A Bernoulli prior over its unknown final outcome approximates
the remaining mechanism uncertainty; the two-call cost reservation is fixed.
This does not supply an open-world provider confidence bound.
"""

import numpy as np

from .emp_binding_pilot import bank_name
from .emp_binding_tools_v3 import retrieve_compatible
from .emp_prospective import choose, coalitions, exact_attribution, exposure_cell, robust_value


def rank_revalidation(model, future, banks, observed, *, seed=42, particles=128,
                      probe_cost=.04, call_cost=.02, radius=.2, storage_cost=.01):
    cells, indices, values = {}, [], []
    for bank in banks:
        keys, utility = [], []
        for edge in future:
            notes = retrieve_compatible(edge, bank); key = exposure_cell(edge, notes)
            p = model.predict(edge, notes)
            cells.setdefault(key, {'edge': edge, 'bank': bank, 'notes': notes,
                'supported': p['supported']})
            keys.append(key)
            utility.append(p['known_success_mass'] - call_cost * (p['expected_consumer_calls'] if p['supported'] else 2.))
        indices.append(keys); values.append(utility)
    values = np.asarray(values)
    def maximum(table):
        return max(robust_value(row, future, bank, radius, storage_cost) for row, bank in zip(table, banks))
    before = maximum(values); rows = []
    for key, spec in sorted(cells.items()):
        if key in observed or spec['supported']:
            continue
        gain = 0.
        for outcome in (0., 1.):
            updated = values.copy()
            for b, keys in enumerate(indices):
                for i, other in enumerate(keys):
                    if other == key:
                        updated[b, i] = outcome - 2 * call_cost
            gain += .5 * (maximum(updated) - before)
        rows.append({'cell_uid': key, 'edge_uid': spec['edge']['uid'], 'bank_uid': bank_name(spec['bank']),
            'exposure': [n['uid'] for n in spec['notes']], 'expected_gain': gain,
            'probe_cost': probe_cost, 'voi': gain - probe_cost, 'objective_before': before,
            'prior': 'Bernoulli(0.5) over a deterministic unknown final outcome',
            'integration': 'exact two-outcome expectation; no Monte Carlo',
            'approximation': 'probe resolves this exact cell; worst two-call cost; no sharing of information to other cells',
            'certificate': False})
    return sorted(rows, key=lambda r: (-r['voi'], r['cell_uid']))
