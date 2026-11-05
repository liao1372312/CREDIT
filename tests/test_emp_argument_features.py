from copy import deepcopy
import json

import numpy as np

from credit.emp_binding import BindingEnvironment
from credit.emp_joint_v2 import request_features as old_features
from credit.emp_joint_v3 import argument_consistency, fit_joint, request_features
from test_emp_structural import edge, trial


def test_public_value_and_provenance_checks_distinguish_old_feature_collisions():
    case = edge(); env = BindingEnvironment(case); env.call(env.producer_request())
    good = env.reference_request()
    value = deepcopy(good); value['arguments']['itemCode'] += '__wrong'
    operation = deepcopy(good); operation['bindings']['/itemCode']['operation'] = 'other'
    fixed = deepcopy(good); fixed['arguments']['fixed'] = 99
    assert argument_consistency(env, good) == [1., 1., 1.]
    assert argument_consistency(env, value) == [0., 1., 1.]
    assert argument_consistency(env, operation) == [1., 0., 1.]
    assert argument_consistency(env, fixed) == [1., 1., 0.]
    for bad in (value, operation, fixed):
        assert np.array_equal(old_features(env, good), old_features(env, bad))
        assert not np.array_equal(request_features(env, good), request_features(env, bad))
        assert not env.call(bad)['accepted']


def test_argument_checks_use_declared_source_and_do_not_reveal_private_gold(monkeypatch):
    case = edge(); env = BindingEnvironment(case); env.call(env.producer_request())
    request = env.reference_request()
    def forbidden(*args, **kwargs):
        raise AssertionError('feature calculation looked up private correct arguments')
    monkeypatch.setattr(BindingEnvironment, 'expected_arguments', forbidden)
    env.edge['bindings'][0]['source'] = '/private/gold'
    assert argument_consistency(env, request) == [1., 1., 1.]
    request['arguments']['itemCode'] = 1
    assert argument_consistency(env, request)[0] == 0.


def test_corrected_features_connect_both_joint_heads_with_90_parameters():
    case = edge(); row = trial(case, repair=True)
    model, fit, _ = fit_joint([row], [], {'e': case}, {}, epochs=2)
    assert fit['parameters'] == 90
    assert model.policy_weights.shape == (26,) and model.api_weights.shape == (64,)
    prediction = model.predict(case, [])
    assert prediction['supported'] and 0 <= prediction['first_success'] <= prediction['final_success'] <= 1
