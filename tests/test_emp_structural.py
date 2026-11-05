from copy import deepcopy
import json

import pytest

from credit.emp_binding import BindingEnvironment
from credit.emp_binding_tools_v3 import assemble
from credit.emp_prospective import coalitions, exact_attribution, rank_revalidation, robust_value
from credit.emp_structural import StructuredMechanisms, compare_api_kernels, environment, replay_observation


def edge(uid='e', version='document_v1'):
    return {'uid': uid, 'eligible': True, 'workflow_group': uid, 'fixture_version': version,
        'bindings': [{'target': '/itemCode', 'source': '/data/0/itemCode'}],
        'producer_api': 'p', 'consumer_api': 'c', 'producer_name': 'producer', 'consumer_name': 'consumer',
        'consumer_query': 'bind item', 'consumer_method': 'POST', 'consumer_endpoint': '/c',
        'producer_catalog': {'request_example': {}, 'response_example': {'status': 1, 'data': [{'itemCode': 'new'}]}},
        'consumer_catalog': {'request_example': {'itemCode': 'old', 'fixed': 3}, 'response_example': {'accepted': True}}}


def trial(case, repair=False, stage='intervention_train'):
    env = environment(case); env.call(env.producer_request())
    if repair:
        bad = {'operation': 'bind_consumer', 'api_id': 'c',
               'bindings': {'itemCode': {'operation': 'producer', 'pointer': '/fields_to_bind/0'}}}
        request, errors = assemble(env, bad)
        replay_observation(env, request, json.dumps(bad), sdk_errors=errors)
    good = {'operation': 'bind_consumer', 'api_id': 'c',
            'bindings': {'itemCode': {'operation': 'producer', 'pointer': '/data/0/itemCode'}}}
    request, errors = assemble(env, good)
    replay_observation(env, request, json.dumps(good), sdk_errors=errors)
    return {'stage': stage, 'edge_uid': case['uid'], 'workflow_group': case['workflow_group'],
        'bank_uid': 'empty', 'exposure': [], 'events': env.history,
        'consumer_calls': len(env.history) - 1, 'final': env.verify()}


def test_literal_rollout_preserves_response_order_without_using_validator(monkeypatch):
    case = edge(); row = trial(case, repair=True)
    model = StructuredMechanisms().update([row], {case['uid']: case}, {})
    model = StructuredMechanisms.from_dict(json.loads(json.dumps(model.to_dict(), sort_keys=True)))
    def forbidden(*args, **kwargs):
        raise AssertionError('forecast consulted reference validator')
    monkeypatch.setattr(BindingEnvironment, 'call', forbidden)
    monkeypatch.setattr(BindingEnvironment, 'expected_arguments', forbidden)
    result = model.predict(case, [])
    assert result['supported'] and result['first_success'] == 0 and result['final_success'] == 1
    assert result['expected_consumer_calls'] == 2 and not result['certificate']
    poisoned = deepcopy(case); poisoned['bindings'][0]['source'] = '/private/wrong/gold'
    assert model.predict(poisoned, []) == result


def test_missing_policy_or_api_cells_abstain_and_conserve_mass():
    case = edge(); row = trial(case)
    model = StructuredMechanisms().update([row], {case['uid']: case}, {})
    model.policy.clear()
    result = model.predict(case, [])
    assert result['unknown_mass'] == 1 and result['mass_interval'] == [0, 1]
    assert result['final_success'] is None


def test_model_rejects_evaluation_fit_and_deduplicates_cached_observations():
    case = edge(); row = trial(case)
    model = StructuredMechanisms().update([row, row], {case['uid']: case}, {})
    assert all(sum(v.values()) == 1 for v in model.policy.values())
    assert all(sum(v.values()) == 1 for v in model.api.values())
    row['stage'] = 'heldout_evaluation'
    with pytest.raises(ValueError, match='evaluation outcomes'):
        model.update([row], {case['uid']: case}, {})


def test_drift_uses_executed_changed_kernel_and_common_version_aligned_inputs():
    old, new = edge(), edge(version='injected_reject_v2')
    source = StructuredMechanisms().update([trial(old)], {'e': old}, {})
    target = StructuredMechanisms().update([trial(new)], {'e': new}, {})
    result = compare_api_kernels(source, target)
    assert result['common_cells'] == 2 and result['decision'] == 'empirical_change'
    assert max(r['empirical_tv'] for r in result['cells']) == 1
    assert not result['certificate']


def test_exact_bank_shapley_efficiency_and_workflow_duplicate_invariance():
    pool = [{'uid': 'a'}, {'uid': 'b'}]; banks = coalitions(pool)
    rows = []
    from credit.emp_binding_pilot import bank_name
    for bank in banks:
        rows.append({'bank_uid': bank_name(bank), 'workflow_group': 'w', 'consumer_calls': 1,
                     'final': {'binding_contract_success': len(bank) == 2}})
    result = exact_attribution(rows, pool, banks)
    assert result['exact_bank_Shapley'] == {'a': .5, 'b': .5}
    assert abs(result['efficiency_residual']) < 1e-12
    a, b = edge('a'), edge('b')
    assert robust_value([.1, .9], [a, b], []) == pytest.approx(robust_value([.1, .1, .9], [a, a, b], []))


def test_voi_reoptimizes_common_banks_and_observed_cells_are_excluded():
    case = edge()
    memory = {'uid': 'memory', 'verified_mapping': {'/itemCode': {'operation': 'producer', 'pointer': '/data/0/itemCode'}}}
    model = StructuredMechanisms()
    ranks = rank_revalidation(model, [case], [[], [memory]], set(), seed=7, particles=256, probe_cost=0)
    assert len(ranks) == 2 and ranks[0]['voi'] > 0
    seen = {r['cell_uid'] for r in ranks}
    assert rank_revalidation(model, [case], [[], [memory]], seen) == []


def test_workflow_compiler_preserves_unknown_when_any_dependency_contract_missing():
    from credit.emp_workflows import compile_workflows
    a, b = edge('a'), edge('b')
    for case, child in [(a, 2), (b, 3)]:
        case.update(task_uid='task', producer_step=1, consumer_step=child)
    task = {'id': 'task', 'metadata': {'composition_relation': 'data_dependency'},
        'TaskList': [{'id': 1, 'depends_on': []}, {'id': 2, 'depends_on': [1]}, {'id': 3, 'depends_on': [1]}]}
    graphs, excluded = compile_workflows([task], [a, b])
    assert len(graphs) == 1 and not excluded
    b['eligible'] = False
    graphs, excluded = compile_workflows([task], [a, b])
    assert not graphs and excluded[0]['binding_workflow_success'] is None


def test_whole_workflow_reuses_one_producer_and_verifies_every_branch(monkeypatch):
    from credit.emp_workflows import execute
    a, b = edge('a'), edge('b')
    a.update(task_uid='task', producer_step=1, consumer_step=2)
    b.update(task_uid='task', producer_step=1, consumer_step=3)
    rows = [trial(a), trial(b)]
    model = StructuredMechanisms().update(rows, {'a': a, 'b': b}, {})
    raw = rows[0]['events'][1]['agent_output']
    class Agent:
        def infer(self, chat):
            return {'agent_output': raw, 'prompt_sha256': 'unit-test-agent'}
    count, original = [], BindingEnvironment.call
    def measured(self, request):
        count.append(request['operation'])
        return original(self, request)
    monkeypatch.setattr(BindingEnvironment, 'call', measured)
    result = execute({'task_uid': 'task', 'edge_uids': ['a', 'b']}, {'a': a, 'b': b}, [], Agent(), model)
    assert count == ['producer', 'consumer', 'consumer']
    assert result['local_calls'] == 3 and result['saved_producer_calls'] == 1
    assert result['binding_workflow_success'] and result['business_goal_success'] is None


def test_revalidation_voi_uses_the_actual_supported_bank_objective():
    from credit.emp_prospective_v2 import choose, rank_revalidation
    case = edge()
    memory = {'uid': 'm', 'verified_mapping': {'/itemCode': {'operation': 'producer', 'pointer': '/data/0/itemCode'}}}
    model = StructuredMechanisms(); banks = [[], [memory]]
    selected, scores = choose(model, [case], banks)
    expected_before = max(row['value'] for row in scores)
    ranked = rank_revalidation(model, [case], banks, set(), probe_cost=0)
    assert selected == [] and all(r['objective_before'] == expected_before for r in ranked)
    empty = next(r for r in ranked if r['bank_uid'] == 'empty')
    assert empty['expected_gain'] == pytest.approx(.5)
    assert empty['integration'] == 'exact two-outcome expectation; no Monte Carlo'
