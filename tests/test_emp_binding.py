from copy import deepcopy

import pytest

from credit.emp_binding import BindingEnvironment, parse_example, proposed_bindings


def fixture():
    return {'eligible': True, 'producer_api': 'list', 'consumer_api': 'save',
            'consumer_query': 'save the first record', 'producer_name': 'List records',
            'consumer_name': 'Save record', 'consumer_method': 'POST', 'consumer_endpoint': '/save',
            'producer_catalog': {'request_example': {'page': 1}, 'response_example': {'data': [{'itemId': 'fresh'}]}},
            'consumer_catalog': {'request_example': {'itemId': 'stale', 'year': 2024}, 'response_example': {'status': '000000'}},
            'bindings': [{'target': '/itemId', 'source': '/data/0/itemId'}]}


def test_forged_provenance_and_stale_arguments_rejected():
    env = BindingEnvironment(fixture())
    env.call(env.producer_request())
    request = env.reference_request()
    request['bindings']['/itemId']['pointer'] = '/not_the_source'
    assert not env.call(request)['accepted']
    request = env.reference_request()
    request['arguments']['itemId'] = 'stale'
    assert not env.call(request)['accepted']
    assert env.verify()['binding_contract_success'] is False
    assert env.verify()['business_goal_success'] is None
    assert env.call(env.reference_request())['accepted']


def test_restorable_intervention_and_state_hash():
    env = BindingEnvironment(fixture())
    env.call(env.producer_request())
    checkpoint, before = env.checkpoint(), env.state_hash()
    event = env.call(env.reference_request())
    assert env.state_hash() == event['state_after']
    env.restore(checkpoint)
    assert env.state_hash() == before
    replay = env.call(env.reference_request())
    assert replay == event


def test_consumer_cannot_bypass_producer_or_alter_configuration():
    env = BindingEnvironment(fixture())
    assert 'producer_not_executed' in env.call({'operation': 'consumer', 'api_id': 'save'})['errors']
    env.call(env.producer_request())
    req = env.reference_request()
    req['arguments']['year'] = 2025
    assert not env.call(req)['accepted']
    req = env.producer_request()
    req['arguments']['page'] = True
    assert not env.call(req)['accepted']


def test_ambiguous_field_and_unparseable_examples_are_unknown():
    assert not proposed_bindings({'itemId': 'a'}, {'data': {'first': {'itemId': 'b'}, 'second': {'itemId': 'c'}}})
    value, reason = parse_example('{"itemId":项目id}')
    assert value is None and 'invalid_json' in reason
    assert parse_example('"{""itemId"":""a"",}"')[0] == {'itemId': 'a'}
    edge = fixture(); edge['eligible'] = False
    with pytest.raises(ValueError):
        BindingEnvironment(edge)


def test_sdk_uses_observed_fields_without_reference_mapping_and_keeps_constants():
    import json
    from credit.emp_binding_sdk import assemble, messages
    env = BindingEnvironment(fixture()); env.call(env.producer_request())
    task = json.loads(messages(env, [])[1]['content'])
    assert task['observable_response_fields']['/data/0/itemId'] == 'fresh'
    assert task['consumer_parameter_template']['itemId'] is None
    assert 'source' not in str(task['fields_to_bind'])
    action = {'operation': 'bind_consumer', 'api_id': 'save',
              'bindings': {'/itemId': {'operation': 'producer', 'pointer': '/data/0/itemId'}}}
    request, errors = assemble(env, action)
    assert not errors
    assert request['arguments'] == {'itemId': 'fresh', 'year': 2024}
    assert env.call(request)['accepted']
    action['bindings']['/year'] = {'operation': 'producer', 'pointer': '/data/0/itemId'}
    _, errors = assemble(env, action)
    assert 'undeclared_target:/year' in errors


def test_sdk_root_alias_does_not_correct_wrong_sources_or_allow_conflicts():
    from credit.emp_binding_sdk import assemble
    env = BindingEnvironment(fixture()); env.call(env.producer_request())
    source = {'operation': 'producer', 'pointer': '/data/0/itemId'}
    action = {'operation': 'bind_consumer', 'api_id': 'save', 'bindings': {'itemId': source}}
    request, errors = assemble(env, action)
    assert not errors and env.call(request)['accepted']
    assert request['sdk_target_aliases'] == {'itemId': '/itemId'}
    action['bindings']['/itemId'] = source
    assert 'duplicate_target_alias:/itemId' in assemble(env, action)[1]
    action['bindings'] = {'itemId': {'operation': 'producer', 'pointer': '/wrongSource'}}
    assert assemble(env, action)[1]
