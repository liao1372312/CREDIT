from copy import deepcopy

import pytest

from credit.emp_binding import BindingEnvironment
from credit.emp_joint_v4 import argument_consistency, fit_joint, public_acceptance_possible
from test_emp_structural import edge, trial


def test_one_wrong_mandatory_value_among_thirteen_forbids_acceptance():
    case = edge()
    case['bindings'] = [{'target': '/f' + str(i), 'source': '/data/0/f' + str(i)} for i in range(13)]
    case['producer_catalog']['response_example'] = {'data': [{'f' + str(i): str(i) for i in range(13)}]}
    case['consumer_catalog']['request_example'] = {'f' + str(i): '' for i in range(13)}
    env = BindingEnvironment(case); env.call(env.producer_request()); good = env.reference_request()
    assert public_acceptance_possible(env, good)
    bad = deepcopy(good); bad['arguments']['f0'] = 'wrong'
    assert argument_consistency(env, bad)[0] == 0
    assert not public_acceptance_possible(env, bad)
    bad = deepcopy(good); bad['bindings']['/f0']['operation'] = 'other'
    assert argument_consistency(env, bad)[1] == 0
    assert not public_acceptance_possible(env, bad)


def test_public_mask_forbids_only_impossible_acceptance_without_correct_source_oracle():
    case = edge(); observed = trial(case, repair=True)
    env = BindingEnvironment(case); env.call(env.producer_request())
    request = env.reference_request(); request['arguments']['itemCode'] += '__wrong'
    env.call(request)
    api_row = {'stage': 'api_fixture_train', 'edge_uid': 'e', 'bank_uid': 'controlled', 'exposure': [],
        'forced_event_indices': [1], 'events': env.history, 'final': env.verify()}
    model, _, _ = fit_joint([observed], [api_row], {'e': case}, {}, epochs=2)
    env = BindingEnvironment(case); env.call(env.producer_request())
    distribution = model.api_distribution(env, request)
    import json
    assert distribution and sum(p for output, p in distribution.items() if json.loads(output)['accepted']) == 0
    good = env.reference_request()
    env.edge['bindings'][0]['source'] = '/private/wrong/gold'
    assert public_acceptance_possible(env, good)
    # A different public source with a consistent copied value can pass these
    # necessary checks. The guard does not prove semantic source correctness.
    good['bindings']['/itemCode']['pointer'] = '/status'
    good['arguments']['itemCode'] = 1
    assert public_acceptance_possible(env, good)
