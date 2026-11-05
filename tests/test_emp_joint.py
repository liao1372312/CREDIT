from copy import deepcopy
import json

import numpy as np
import pytest

from credit.emp_binding import BindingEnvironment
from credit.emp_binding_tools_v3 import assemble
from credit.emp_joint_v2 import fit_joint, policy_features
from credit.emp_structural import replay_observation
from test_emp_structural import edge, trial


def test_action_alphabet_merges_only_equivalent_target_aliases_and_json_order():
    from credit.emp_joint_v2 import canonical_action
    env = BindingEnvironment(edge()); env.call(env.producer_request())
    bare = {'operation': 'bind_consumer', 'api_id': 'c',
        'bindings': {'itemCode': {'pointer': '/data/0/itemCode', 'operation': 'producer'}}}
    full = {'bindings': {'/itemCode': {'operation': 'producer', 'pointer': '/data/0/itemCode'}},
        'api_id': 'c', 'operation': 'bind_consumer'}
    assert canonical_action(env, json.dumps(bare)) == canonical_action(env, json.dumps(full))
    full['bindings']['/itemCode']['pointer'] = '/status'
    assert canonical_action(env, json.dumps(bare)) != canonical_action(env, json.dumps(full))


def example():
    case = edge(); observed = trial(case, repair=True)
    env = BindingEnvironment(case); env.call(env.producer_request())
    action = {'operation': 'bind_consumer', 'api_id': 'c',
        'bindings': {'itemCode': {'operation': 'producer', 'pointer': '/status'}}}
    request, errors = assemble(env, action)
    replay_observation(env, request, json.dumps(action), sdk_errors=errors)
    api = {'stage': 'api_fixture_train', 'edge_uid': 'e', 'bank_uid': 'probe',
           'exposure': [], 'forced_event_indices': [1], 'events': env.history, 'final': env.verify()}
    return case, observed, api


def test_exact_two_attempt_rollout_and_outcome_gradients_reach_both_heads():
    import torch
    case, row, api = example()
    model, _, _ = fit_joint([row], [api], {'e': case}, {}, epochs=1)
    matrix = model.matrices(case, [])
    pw = torch.zeros(23, dtype=torch.float64, requires_grad=True)
    kw = torch.zeros(52, dtype=torch.float64, requires_grad=True)
    _, _, first, final = model.forward(matrix, pw, kw)
    assert float(first.detach()) == pytest.approx(.25)
    assert float(final.detach()) == pytest.approx(.4375)
    (-final.log()).backward()
    assert float(pw.grad.abs().sum()) > 0 and float(kw.grad.abs().sum()) > 0


def test_policy_public_features_and_interactions_are_order_invariant_and_gold_free():
    case, row, _ = example(); env = BindingEnvironment(case); env.call(env.producer_request())
    raw = row['events'][-1]['agent_output']
    notes = [{'uid': 'a', 'verified_mapping': {'/itemCode': {'pointer': '/data/0/itemCode'}}},
             {'uid': 'b', 'verified_mapping': {'/itemCode': {'pointer': '/other/itemCode'}}}]
    vector = policy_features(env, notes, raw)
    assert np.allclose(vector, policy_features(env, notes[::-1], raw))
    env.edge['bindings'][0]['source'] = '/private/gold'
    assert np.allclose(vector, policy_features(env, notes, raw))


def test_shared_model_abstains_on_unknown_api_contract_without_using_validator(monkeypatch):
    case, row, api = example()
    model, _, _ = fit_joint([row], [api], {'e': case}, {}, epochs=2)
    def forbidden(*args, **kwargs):
        raise AssertionError('forecast called validator')
    monkeypatch.setattr(BindingEnvironment, 'call', forbidden)
    assert model.predict(case, [])['supported']
    unknown = deepcopy(case); unknown['consumer_api'] = 'unseen'
    result = model.predict(unknown, [])
    assert not result['supported'] and result['final_success'] is None
