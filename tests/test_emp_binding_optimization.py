from copy import deepcopy
import json

import numpy as np
import pytest

from credit.emp_binding import BindingEnvironment
from credit.emp_binding_expansion import recover_example, spelling_bindings
from credit.emp_binding_shared import SharedBindingMechanism, distinct_trials, encode
from credit.emp_binding_tools_v2 import assemble, retrieve_compatible


def edge():
    return {'eligible': True, 'bindings': [{'target': '/records/0/ITEM_CODE', 'source': '/data/0/itemCode'}],
            'producer_api': 'source', 'consumer_api': 'target',
            'producer_catalog': {'request_example': {}, 'response_example': {'data': [{'itemCode': 'fresh'}]}},
            'consumer_catalog': {'request_example': {'records': [{'ITEM_CODE': 'old'}], 'year': 2024}, 'response_example': {}}}


def note(uid, source='/data/0/itemCode'):
    return {'uid': uid, 'verified_mapping': {'/itemCode': {'operation': 'producer', 'pointer': source}}}


def test_shared_features_ignore_private_gold_paths_and_note_order():
    case = edge()
    notes = [note('a'), note('b', '/absent/itemCode')]
    first, profile = encode(case, notes)
    assert np.allclose(first, encode(case, notes[::-1])[0])
    poisoned = deepcopy(case); poisoned['bindings'][0]['source'] = '/secret/reference/changed'
    assert np.allclose(first, encode(poisoned, notes)[0])
    assert profile == encode(poisoned, notes)[1]


def test_source_schema_gate_rejects_same_field_under_new_parent():
    case = edge(); _, profile = encode(case, [])
    model = SharedBindingMechanism(np.zeros((20, 5)), np.zeros(20), np.ones(20), [],
        [{'profile': profile, 'exposure_size': 0, 'workflow_group': 'training'}])
    assert model.predict(case, [])['supported']
    changed = deepcopy(case)
    changed['producer_catalog']['response_example'] = {'other': [{'itemCode': 'fresh'}]}
    result = model.predict(changed, [])
    assert not result['supported'] and result['final_success'] is None


def test_training_weights_balance_workflows_and_conflicting_replays_rejected():
    def row(workflow, uid, exposure):
        return {'workflow_group': workflow, 'edge_uid': uid, 'exposure': exposure,
                'first_category': 'accepted', 'final': {'binding_contract_success': True}, 'consumer_calls': 1}
    rows = [row('a', 'a1', []), row('a', 'a1', ['m']), row('a', 'a2', []), row('b', 'b1', [])]
    unique, weights = distinct_trials(rows + [rows[0]])
    assert sum(w for r, w in zip(unique, weights) if r['workflow_group'] == 'a') == pytest.approx(.5)
    assert sum(w for r, w in zip(unique, weights) if r['workflow_group'] == 'b') == pytest.approx(.5)
    altered = deepcopy(rows[0]); altered['final']['binding_contract_success'] = False
    with pytest.raises(ValueError):
        distinct_trials(rows + [altered])


def test_parser_recovers_only_structural_commas_preserving_string_values():
    raw = '{"name":"甲，乙\\u00a0丙"\n"values":[1,2,],}'
    parsed, repairs = recover_example(raw)
    assert parsed == {'name': '甲，乙\u00a0丙', 'values': [1, 2]}
    assert any(x.startswith('newline_property_commas') for x in repairs)
    for invalid in ['{"itemId":项目id}', '{"x":1', '{"x":1\n"x":2}', '{} {}', '{"x": [1 2]}']:
        assert recover_example(invalid)[0] is None


def test_identifier_aliases_do_not_merge_ambiguous_sources():
    request = {'ITEM_CODE': 'old'}
    assert spelling_bindings(request, {'data': {'itemCode': 'fresh'}})[0]['source'] == '/data/itemCode'
    assert not spelling_bindings(request, {'data': {'itemCode': 'a', 'item_code': 'b'}})


def test_nested_target_alias_preserves_source_and_configuration():
    env = BindingEnvironment(edge()); env.call(env.producer_request())
    action = {'operation': 'bind_consumer', 'api_id': 'target',
        'bindings': {'itemCode': {'operation': 'producer', 'pointer': '/data/0/itemCode'}}}
    request, errors = assemble(env, action)
    assert not errors and env.call(request)['accepted']
    assert request['arguments'] == {'records': [{'ITEM_CODE': 'fresh'}], 'year': 2024}
    action['bindings']['/records/0/ITEM_CODE'] = action['bindings']['itemCode']
    assert 'duplicate_target_alias:/records/0/ITEM_CODE' in assemble(env, action)[1]
    another = edge(); another['bindings'].append({'target': '/other/ITEM_CODE', 'source': '/data/0/itemCode'})
    second = BindingEnvironment(another); second.call(second.producer_request())
    action['bindings'].pop('/records/0/ITEM_CODE')
    assert 'ambiguous_target_alias:itemCode' in assemble(second, action)[1]


def test_retrieval_excludes_stale_paths_from_public_response():
    case = edge()
    assert [n['uid'] for n in retrieve_compatible(case, [note('bad', '/previous/itemCode'), note('good')])] == ['good']
    case['bindings'][0]['source'] = '/secret/gold/reference'
    assert [n['uid'] for n in retrieve_compatible(case, [note('good')])] == ['good']


def test_repair_feedback_exposes_all_public_sources_without_correcting_choice():
    import json
    from credit.emp_binding_tools_v3 import messages
    case = edge()
    case.update(consumer_query='bind item', producer_name='source', consumer_name='target',
                consumer_method='POST', consumer_endpoint='/target')
    case['producer_catalog']['response_example']['unrelated'] = 17
    env = BindingEnvironment(case); env.call(env.producer_request())
    action = {'operation': 'bind_consumer', 'api_id': 'target',
        'bindings': {'ITEM_CODE': {'operation': 'producer', 'pointer': '/fields_to_bind/0'}}}
    request, errors = assemble(env, action)
    assert errors and request['operation'] == 'invalid_sdk_action'
    env.call(request)
    env.history[-1]['response']['sdk_errors'] = errors
    env.history[-1]['agent_output'] = json.dumps(action)
    feedback = json.loads(messages(env, [])[-1]['content'])
    assert set(feedback['allowed_producer_response_pointers']) == {'/data/0/itemCode', '/unrelated'}
    assert 'fields_to_bind' in messages(env, [])[0]['content']
