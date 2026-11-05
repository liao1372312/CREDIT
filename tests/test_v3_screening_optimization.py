from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from credit.v3.local import StatefulExecutor, context, memories, retriever
from credit.v3.drift import TransportGate
from credit.v3_benchmark import CallLedger
from credit.v3_screening_optimization import ReuseSelector, distinct_banks, public_profiles, mixed_means


class OpenGate:
    def candidates(self, c, pool):
        return tuple(pool)

    def transported(self, c):
        return c.stratum.endswith('/source')


def selector(gate=None):
    return ReuseSelector(memories(), (), gate or OpenGate(), retriever(4), None,
        StatefulExecutor(), CallLedger({'direct-screen':100}),
        mixture={'books/source':.5,'books/target':.5},capacity=80,
        transported_strata={'books/source'})


def test_equivalent_retrieval_banks_have_one_cheapest_representative():
    sel = selector()
    profiles = public_profiles(sel.mixture)
    banks, feasible = distinct_banks(sel, profiles)
    assert len(banks) < feasible
    assert all('hidden_fix' not in bank for bank in banks)
    signatures = [tuple(sel.exposures(c,b) for c in profiles) for b in banks]
    assert len(set(signatures)) == len(banks)
    # Public deployment templates also cover other seeds and phase IDs.
    for b in banks:
        for c in profiles:
            assert sel.exposures(c,b) == sel.exposures(replace(c,uid='fresh',seed=778),b)


def test_execution_reuse_requires_identical_task_noise_and_ordered_exposure():
    sel = selector()
    c = context('books','source','screen',0,3,template='train')
    first = sel._executions('direct-screen',{'books/source':[c]},[('lookup_key',)])
    second = sel._executions('direct-screen',{'books/source':[c]},[('hidden_fix','lookup_key')])
    assert np.array_equal(first[('lookup_key',)]['books/source'],
                          second[('hidden_fix','lookup_key')]['books/source'])
    assert len(sel.ledger.records) == 1
    assert sel.reuse_records[0]['physical_record'] == 0
    sel._executions('direct-screen',{'books/source':[replace(c,seed=c.seed+1)]},[('lookup_key',)])
    assert len(sel.ledger.records) == 2
    sel._executions('direct-screen',{'books/source':[replace(c,initial_state=7)]},[('lookup_key',)])
    assert len(sel.ledger.records) == 3


def test_unsupported_screening_uses_real_values_without_target_prediction():
    source = context('books','source','unlabeled',0,3)
    target = context('books','target','screen',0,4)
    predicted, executed = [], []
    def predictions(tasks, bank):
        predicted.extend(c.stratum for c in tasks)
        assert all(c.stratum.endswith('/source') for c in tasks)
        return np.array([.6]*len(tasks))
    def execute(phase,tasks,banks):
        executed.extend(tasks)
        return {banks[0]:{k:np.array([1.,0.]) for k in tasks}}
    fake = SimpleNamespace(mixture={'books/source':.5,'books/target':.5},
        transported_strata={'books/source'},predictions=predictions,_executions=execute)
    means = mixed_means(fake,(),{'books/source':[source]}, {'books/target':[target]},'mixed')
    assert means == [.6,.5]
    assert executed == ['books/target'] and predicted == ['books/source']
    assert mixed_means(fake,(),{'books/source':[source]}, {}, 'model') == [.6,0.]


def test_reuse_does_not_cross_actual_gate_exposures_or_call_ceiling():
    sel = selector()
    sel.ledger.call_cap = 1
    c = context('books','source','budget',0,3,template='train')
    with pytest.raises(ValueError,match='allowance'):
        sel._executions('direct-screen',{'books/source':[c]},[()])
    assert not sel.execution_cache and not sel.ledger.records


def test_large_pool_cannot_silently_trigger_exponential_enumeration():
    sel = selector()
    sel.memories = tuple(replace(memories()[0],uid=f'm{i}') for i in range(17))
    with pytest.raises(ValueError,match='sixteen'):
        distinct_banks(sel, public_profiles(sel.mixture))
