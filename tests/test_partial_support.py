import numpy as np

from credit.budget_mechanisms_v2 import BatchForecast
from credit.partial_support import PartialSupportForecast
from credit.shared_mechanisms import SharedMechanisms
from credit.stateful import StateTask,api_probes,decision_probes


def test_missing_unreachable_inputs_do_not_block_reachable_workflow():
    policy=decision_probes(1,32);api=api_probes(2,32,'source')
    task=StateTask('graph','books',('lookup','transform','authorize','submit'),'source',1,'chain')
    exposure=[('lookup_key','transform_rule','auth_pair_a','auth_pair_b','legacy_contract')]
    complete=PartialSupportForecast(SharedMechanisms(policy,api)).predict(task,exposure)
    reduced=[r for r in api if r['key'][3] not in (2,4,5,6)]
    model=SharedMechanisms(policy,reduced)
    _,coarse=BatchForecast(model).predict(task,exposure)
    partial=PartialSupportForecast(model).predict(task,exposure)
    assert not coarse[0]
    assert partial['complete_path_coverage'][0]
    np.testing.assert_allclose(partial['known_success'],complete['known_success'],atol=1e-12)


def test_missing_action_mass_is_unknown_and_not_imputed_as_rejection():
    policy=decision_probes(1,32);api=api_probes(2,32,'source')
    task=StateTask('submit','books',('submit',),'source',1,'ready',initial_state=7)
    exposure=[('legacy_contract',)]
    full=SharedMechanisms(policy,api)
    reduced=[r for r in api if r['key']!=['source','books','submit',7,'standard']]
    value=PartialSupportForecast(SharedMechanisms(policy,reduced)).predict(task,exposure)
    from credit.stateful import decision_key
    p=full.policy_distribution(decision_key('submit',7,'start',exposure[0]))[0]
    assert value['unknown_mass'][0]>0
    np.testing.assert_allclose(value['unknown_mass'],[p],atol=1e-12)
    original=PartialSupportForecast(full).predict(task,exposure)['known_success'][0]
    assert value['known_success'][0]<=original<=value['conditional_upper'][0]


def test_missing_policy_input_is_unknown_even_with_complete_api_evidence():
    from credit.stateful import decision_key
    policy=decision_probes(1,32);api=api_probes(2,32,'source')
    exposure=('legacy_contract',)
    key=decision_key('submit',7,'start',exposure)
    policy=[r for r in policy if tuple(r['key'])!=key]
    task=StateTask('submit','books',('submit',),'source',1,'ready',initial_state=7)
    prediction=PartialSupportForecast(SharedMechanisms(policy,api)).predict(task,[exposure])
    assert prediction['unknown_mass'][0]==1
    assert prediction['known_success'][0]==0


def test_partial_support_is_used_by_retention_not_only_by_diagnostics():
    from credit.budget_mechanisms_v2 import BankSearch
    from credit.partial_retention import PartialBankSearch
    from credit.stateful import MEMORIES
    policy=decision_probes(1,32)
    api=[r for r in api_probes(2,32,'source') if r['key'][3] not in (2,4,5,6)]
    model=SharedMechanisms(policy,api)
    task=StateTask('submit','books',('submit',),'source',1,'ready',initial_state=7)
    tokens=dict.fromkeys(MEMORIES,1)
    coarse=BankSearch(BatchForecast(model),[task],tokens,1,1)
    partial=PartialBankSearch(PartialSupportForecast(model),[task],tokens,1,1)
    bank_a,_=coarse.joint();bank_b,_=partial.joint()
    assert bank_a==()
    assert bank_b
    assert partial.cache[bank_b]['objective']>partial.cache[()]['objective']
