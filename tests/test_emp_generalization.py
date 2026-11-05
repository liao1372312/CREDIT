from copy import deepcopy

from credit.emp_binding import BindingEnvironment
from credit.emp_generalization_audit_v2 import contract_blocks, evaluate, observed_inputs
from test_emp_structural import edge, trial


def test_folds_join_workflow_and_contract_duplicates_transitively():
    a, b, c, d = [edge(uid) for uid in 'abcd']
    b['consumer_catalog']['request_example']['fixed'] = 4
    b['workflow_group'] = a['workflow_group']
    c['consumer_catalog'] = deepcopy(b['consumer_catalog'])
    d['consumer_catalog']['request_example']['fixed'] = 5
    assert contract_blocks({e['uid']: e for e in (a, b, c, d)}) == [['a', 'b', 'c'], ['d']]


def test_api_examples_deduplicate_aliases_and_preserve_positive_negative_classes():
    case = edge(); row = trial(case, repair=True)
    alias = deepcopy(row)
    alias['events'][-1]['request']['sdk_target_aliases'] = {'itemCode': '/itemCode'}
    examples = observed_inputs([row, alias], {'e': case})
    assert len(examples) == 2
    assert {example['output']['accepted'] for example in examples} == {False, True}


def test_fold_provenance_excludes_heldout_notes_and_keeps_unknown_unscored(monkeypatch):
    a, b = edge('a'), edge('b')
    b['consumer_catalog']['request_example']['fixed'] = 4
    rows = [trial(a, repair=True), trial(b, repair=True)]
    rows[1]['exposure'] = ['history_a']
    memories = {'history_a': {'source_edge': 'a'}}

    called_edges = []
    original_call = BindingEnvironment.call

    def track_training_replay(env, request):
        called_edges.append(env.edge['uid'])
        return original_call(env, request)

    monkeypatch.setattr(BindingEnvironment, 'call', track_training_replay)
    result = evaluate(rows, {'a': a, 'b': b}, memories)
    first_fold = result['folds'][0]
    assert first_fold['excluded_heldout_origin_notes'] == ['history_a']
    assert first_fold['excluded_exposure_rows'] == 1
    assert first_fold['training_edge_uids'] == []
    balanced = result['summary']['table'][1]
    assert balanced['inputs'] == 4 and balanced['accepted'] == 2
    assert balanced['always_reject_accuracy'] == .5
    assert balanced['covered_accuracy'] is None
    assert all(p['acceptance_probability'] is None for p in result['predictions'])
    # The first fold has no fitting rows; the second may replay only a. Heldout
    # b is never passed to the validator by observation extraction or forecast.
    assert called_edges == ['a'] * len(rows[0]['events'])
