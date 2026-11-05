from collections import Counter
from dataclasses import replace

import numpy as np
import torch

from credit.v3.local import context,ROLES
from credit.v3.drift import transfer_level,Level
from credit.v3_benchmark import CallLedger
from credit.v3_revalidation import candidate_memories,RevalidationExecutor,EvidenceGate
from credit.v3_online import balanced_designs,Store,OnlineModel,queue,PolicySelector
from credit.v3_online_experiment import archive


def test_balanced_optional_factor_coverage_is_guaranteed_before_outcomes():
    pool=candidate_memories();ledger=CallLedger({'target-factorial':8192,'replay_audit':128},100000)
    tasks=[context('travel','target','balanced-unit',i,31,
        template=('future','future_auth_repeat','future_ready_submit')[i%3]) for i in range(96)]
    episodes,reports=balanced_designs(tasks,pool,ledger,17)
    counts=Counter(uid for r in reports for uid in r['memory_ids'])
    assert counts['lookup_key']>=24 and counts['new_contract']>=56
    assert len(reports)==96 and len({r['context']['seed'] for r in reports})==96
    assert all(e.randomized and e.replay_valid for e in episodes)
    assert all(r['reserved_slots_length_matched'] for r in reports)


def test_voi_uses_feasible_moves_and_beta_and_does_not_mutate_real_gate():
    evidence=[]
    for family in ('books','travel','inventory','payments'):
        for epoch in ('source','target'):
            for role in ROLES.values():
                p=transfer_level(role,role,f'stateful:{family}:{next(k for k,v in ROLES.items() if v==role)}:{epoch}',
                                 changed=0,fixtures=1024,support=100)
                if family=='books' and epoch=='target' and role==ROLES['lookup']:p=replace(p,level=Level.U)
                evidence.append(p)
    class Model:
        def predict_many(self,items):return np.asarray([float('lookup_key' in a) for c,a in items])
    q=queue((Model(),),evidence,{},('lookup_key',),{'differential':(2,1)})
    row=q[0]
    assert row['probability']==3/5 and row['gain']>0
    assert np.isclose(row['voi'],.6*row['gain']-row['probe_penalty']*2048)
    assert row['candidate_moves']>=1
    real=EvidenceGate(evidence,{}, {'new_contract'})
    c=context('books','target','real-policy',0,1)
    assert not real.transported(c) and not any(m.uid=='lookup_key' for m in real.candidates(c,candidate_memories()))


def test_cold_storage_is_reversible_and_absence_alone_cannot_archive():
    s=Store(quarantine_capacity=0)
    c=context('books','target','cold-unit',0,1)
    s.refresh([c],EvidenceGate(),0)
    assert s.entries['books/target','lookup_key']['state']=='cold'
    s.fetch(ROLES['lookup'].key,'stateful:books:lookup:target',1)
    assert s.entries['books/target','lookup_key']['state']=='quarantined'
    ledger=CallLedger({'certification':1024},10000)
    state={'store':s,'bank':()}
    assert archive(state,EvidenceGate(),ledger,10)==[]
    assert not any(e['state']=='archived' for e in s.entries.values())


def test_archive_requires_every_eligible_stratum_and_two_inactive_cycles():
    profiles=[context(f,e,'profile-unit',0,8) for f in ('books','travel','inventory','payments') for e in ('source','target')]
    s=Store();s.refresh(profiles,EvidenceGate(),0)
    state={'store':s,'bank':()};ledger=CallLedger({'certification':1024},10000)
    for f in ('books','travel','inventory','payments'):
        for e in ('source','target'):
            for i in range(12):ledger.execute('certification',RevalidationExecutor(),context(f,e,'archive-unit',i,8),())
    assert archive(state,EvidenceGate(),ledger,0)==[]
    result=archive(state,EvidenceGate(),ledger,1)
    assert [r['memory'] for r in result]==['hidden_fix']
    assert len(result[0]['strata'])==8
    assert all(v['upper_joint_marginal']<0 for v in result[0]['strata'].values())


def test_version_features_and_logged_gradient_block(monkeypatch):
    from credit.v3_optimized_model import StateAwareOutcomeModel
    contexts=[context('books','source','version-unit',0,9),context('books','target','version-unit',0,9)]
    model=OnlineModel(candidate_memories(),contexts,seed=1)
    items=[(c,c.nodes[0],()) for c in contexts]
    assert model._pack(items)[0][:,-2:].tolist()==[[1.,0.],[0.,1.]]
    before={k:v.clone() for k,v in model.heads.state_dict().items() if not k.startswith('baseline.')}
    monkeypatch.setattr(StateAwareOutcomeModel,'fit',lambda self,*a,**k:self)
    logged=[RevalidationExecutor().run(contexts[0],('lookup_key',))]
    model.fit_online([],[],[],logged=logged,baseline_steps=2)
    assert all(torch.equal(v,model.heads.state_dict()[k]) for k,v in before.items())
    assert model.fit_report['logged_exposure_gradients_blocked']


def test_failed_same_bank_gate_change_retains_previous_complete_policy():
    from credit.v3_revalidation import domain_key
    role=ROLES['submit'];c=context('books','target','policy-prototype',0,29,template='future_ready_submit')
    p=transfer_level(role,role,c.nodes[0].version,changed=96,fixtures=96,support=100)
    prior=EvidenceGate([p],{domain_key(c):['new_contract']},{'new_contract'})
    proposed=EvidenceGate([p],novel_ids={'new_contract'})
    make=lambda k,phase,i:context(*k.split('/'),phase,i,29,template='future_ready_submit')
    ledger=CallLedger({'pilot':16,'certification':512},10000)
    selector=PolicySelector(candidate_memories(),(),proposed,
        __import__('credit.v3.local',fromlist=['retriever']).retriever(4),make,RevalidationExecutor(),ledger,
        mixture={'books/target':1.},capacity=80,radius=.2,cost_penalty=.0001,transported_strata=())
    frozen=[]
    decision,deployed=selector.switch_policy(('new_contract',),prior,[('new_contract',)],
        pilot_n=2,certification_n=200,unlabeled_n=2,freeze=frozen.append)
    assert frozen[0]['same_bank_gate_changes_are_challengers']
    assert not decision['switched'] and decision['policy']=='prior'
    assert deployed is prior and 'new_contract' in {m.uid for m in deployed.candidates(c,candidate_memories())}
    assert len(decision['certificates'])==1 and decision['certificates'][0]['estimates']['books/target']['mean']<0
