from dataclasses import replace

import numpy as np
import pytest

from credit.budget_grid_v2 import augmented, collect_probes, feasible_banks, prefix
from credit.budget_mechanisms_v2 import BatchForecast, design, fit_order
from credit.shared_mechanisms import SharedMechanisms
from credit.stateful import (MEMORIES, StateTask, api_probes, decision_probes)


def test_batch_graph_rollout_matches_scalar_and_rejects_unknown_domains():
    model = SharedMechanisms(decision_probes(1, 32), api_probes(2, 32, 'source') + api_probes(3, 32, 'target'))
    batch = BatchForecast(model)
    exposures = [(), ('lookup_key',), ('auth_pair_a', 'auth_pair_b', 'legacy_contract')]
    task = StateTask('new-topology', 'books', ('lookup', 'authorize', 'transform', 'authorize', 'submit'),
        'target', 1, 'new')
    means, coverage = batch.predict(task, exposures)
    np.testing.assert_allclose(means, [model.predict(task, e).mean for e in exposures], atol=1e-12)
    assert coverage.all()
    values, supported = batch.predict(replace(task, family='unseen-provider'), exposures)
    assert not supported.any() and not values.any()


def test_interaction_order_changes_the_exposure_function_class():
    from credit.stateful import NEIGHBORS
    a = NEIGHBORS['authorize'].index('auth_pair_a')
    b = NEIGHBORS['authorize'].index('auth_pair_b')
    keys = [('authorize', 0, 0, mask) for mask in (0, 1 << a, 1 << b, (1 << a) | (1 << b))]
    unary, _ = design(keys, 'authorize', 1)
    pairwise, _ = design(keys, 'authorize', 2)
    # An additive model has zero mixed second difference; a pair column does not.
    np.testing.assert_allclose(unary[3] - unary[2] - unary[1] + unary[0], 0)
    assert np.any(pairwise[3] - pairwise[2] - pairwise[1] + pairwise[0])
    with pytest.raises(ValueError):
        design(keys, 'authorize', 4)


def test_call_budget_includes_replay_validation_and_reused_data_counted_once():
    tokens = dict.fromkeys(MEMORIES, 1)
    admissible = feasible_banks(tokens, 2)
    probes = collect_probes(101, tokens, 2, 1, admissible, 64)
    for cap in (16, 64):
        selected, used = prefix(probes, cap)
        assert used <= cap
        assert used == sum(r['learning_calls'] + r['probe_audit_full_replay_calls'] for r in selected)
    base_p = decision_probes(1, 16)
    base_k = api_probes(2, 16, 'source') + api_probes(3, 16, 'target')
    policy, api = augmented(base_p, base_k, probes)
    expected = sum(r['learning_calls'] for r in probes)
    assert sum(r['samples'] for r in policy) - sum(r['samples'] for r in base_p) == expected
    assert sum(r['samples'] for r in api) - sum(r['samples'] for r in base_k) == expected


def test_order_fitter_uses_no_private_environment_probabilities(monkeypatch):
    policy = decision_probes(1, 32)
    api = api_probes(2, 32, 'source')
    import credit.stateful as environment
    def forbidden(*args, **kwargs):
        raise AssertionError('private kernel accessed')
    monkeypatch.setattr(environment, '_api', forbidden)
    monkeypatch.setattr(environment, '_policy', forbidden)
    model, fits = fit_order(policy, api, 2)
    task = StateTask('blind', 'books', ('submit',), 'source', 1, 'ready', initial_state=7)
    value, support = BatchForecast(model).predict(task, [('legacy_contract',)])
    assert support[0] and 0 <= value[0] <= 1
    assert fits['authorize']['feature_columns'] > len(environment.NEIGHBORS['authorize'])
