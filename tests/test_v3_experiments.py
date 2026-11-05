from dataclasses import replace
import numpy as np
import pytest
from credit.v3.local import memories,context
from credit.v3_benchmark import CachedModel,CallLedger,pack
from credit.v3.selection import bank_cost
from credit.v3.types import Episode
from credit.v3_statistical import factorial

def test_static_retention_and_budget_are_actual_token_costs():
    pool=memories()
    bank=pack([m.uid for m in pool],pool,40)
    assert bank_cost(bank,pool)<=40
    assert len(bank)<len(pool)

def test_prediction_cache_ignores_only_execution_noise():
    class Model:
        fit_report={}
        calls=0
        def predict_many(self,items):
            self.calls+=len(items)
            return np.asarray([len(c.nodes)/10 for c,a in items])
    model=Model();cached=CachedModel(model)
    c=context('books','source','unit',0,1,template='train')
    d=replace(c,uid='other',seed=99)
    assert cached.predict_many([(c,()),(d,())]).tolist()==[.4,.4]
    assert model.calls==1
    x=context('books','source','unit',0,1,template='future_ready_submit')
    assert cached.predict_many([(x,())]).tolist()==[.1]
    assert model.calls==2

def test_call_budget_prevents_physical_execution():
    class Forbidden:
        def run(self,*args,**kwargs):raise AssertionError('must not execute')
    c=context('books','source','budget',0,1,template='train')
    ledger=CallLedger({'selection':100},call_cap=5)
    with pytest.raises(ValueError,match='allowance exhausted'):
        ledger.execute('selection',Forbidden(),c,())
    assert not ledger.records

def test_shared_calls_are_charged_to_each_strategy():
    c=context('books','source','budget',0,1,template='train')
    e=Episode(c,(),(),True,3,'s','p','p',True)
    ledger=CallLedger({'selection':1},call_cap=10,common_calls=4)
    ledger.charge('selection',e)
    assert ledger.spent==7
    assert ledger.report()['unspent_call_allowance']==3

def test_vectorized_simulation_records_each_equal_budget_estimate():
    raw,truth,table=factorial(4,256,False,918)
    assert raw.shape==(3,4,23)
    assert truth.shape==(23,)
    assert all(r['budget']==256 for r in table)
    assert np.isfinite(raw[:2]).all()
