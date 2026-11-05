from dataclasses import asdict,replace

from credit.v3.local import context,memories,StatefulExecutor,ROLES
from credit.v3.drift import TransportGate,Level,transfer_level
from credit.v3_revalidation import (EvidenceGate,admit_memories,domain_key,reprobe,
                                    candidate_memories,RevalidationExecutor)
import pytest


def test_direct_lease_is_pinned_and_does_not_upgrade_transport():
    c = context('books','target','lease',0,1,template='future_ready_submit')
    role = ROLES['submit']
    p = transfer_level(role,role,c.nodes[0].version,changed=96,fixtures=96,support=100)
    g = EvidenceGate([p],{domain_key(c):['legacy_contract']})
    pool = [m for m in memories() if m.uid=='legacy_contract']
    assert g.level(c.nodes[0])==Level.D and not g.transported(c)
    assert g.candidates(c,pool)==tuple(pool)
    assert g.candidates(replace(c,uid='new-test',seed=99),pool)==tuple(pool)
    assert not g.candidates(replace(c,initial_state=3),pool)
    node = replace(c.nodes[0],version='stateful:books:submit:other-target')
    assert not g.candidates(replace(c,nodes=(node,)),pool)
    with pytest.raises(ValueError,match='cannot bypass'):
        g.candidates(c,pool,directly_validated=frozenset({'conflict'}))


def evidence_rows(n=24,positive=True):
    reports,episodes = [],[]
    for i in range(n):
        c = context('books','target','unit-revalidation',i,10,template='future_ready_submit')
        reports.append({'context':asdict(c),'memory_ids':['legacy_contract'],
                        'banzhaf':[.4 if positive else -.4]})
        episodes.append(StatefulExecutor().run(c,('legacy_contract',)))
    return reports,episodes


def test_admission_requires_positive_clustered_effect_and_real_node_support():
    r,e = evidence_rows()
    leases,decisions = admit_memories(r,e,memories(),TransportGate())
    assert all('legacy_contract' in ids for ids in leases.values())
    d = next(d for d in decisions if d['memory']=='legacy_contract')
    assert d['contexts']==24 and d['alpha_per_test']==.05/8
    r,e = evidence_rows(positive=False)
    leases,_ = admit_memories(r,e,memories(),TransportGate())
    assert all(not ids for ids in leases.values())
    r,e = evidence_rows(n=11)
    leases,_ = admit_memories(r,e,memories(),TransportGate())
    assert all(not ids for ids in leases.values())
    r,_ = evidence_rows()
    leases,_ = admit_memories(r,[],memories(),TransportGate())
    assert all(not ids for ids in leases.values())


def test_reprobe_uses_fresh_batch_and_keeps_divergent_roles_quarantined(monkeypatch):
    role = ROLES['submit']
    unresolved = transfer_level(role,role,'stateful:books:submit:target',changed=8,fixtures=96,support=100)
    divergent = transfer_level(role,role,'stateful:inventory:submit:target',changed=96,fixtures=96,support=100)
    assert unresolved.level==Level.U
    monkeypatch.setattr('credit.v3_revalidation.request_probe',lambda *a,**k:{'counts':{'ok:8':1}})
    result,records,decisions = reprobe([unresolved,divergent],12,fixtures=64)
    updated = next(p for p in result if 'books' in p.version)
    assert updated.changed==0 and updated.fixtures==64 and updated.level==Level.V
    assert divergent in result
    assert len(records)==64 and sum(r['api_calls'] for r in records)==128
    assert decisions[0]['uses_fresh_fixtures_only']


def test_novel_memory_requires_direct_evidence_even_on_verified_source():
    c = context('books','source','novel',0,1,template='future_ready_submit')
    role = ROLES['submit']
    p = transfer_level(role,role,c.nodes[0].version,changed=0,fixtures=1024,support=100)
    pool = [m for m in candidate_memories() if m.uid=='new_contract']
    g = EvidenceGate([p],novel_ids={'new_contract'})
    assert g.level(c.nodes[0])==Level.V
    assert not g.candidates(c,pool)
    target = context('books','target','novel',0,1,template='future_ready_submit')
    g = EvidenceGate([p],{domain_key(target):['new_contract']},{'new_contract'})
    assert g.candidates(target,pool)==tuple(pool)
    assert not g.candidates(c,pool)
    longer = context('books','target','novel',0,1,template='future')
    g = EvidenceGate([p],{domain_key(longer):['new_contract']},{'new_contract'})
    assert g.candidates(longer,pool)==tuple(pool)
    assert not g.candidates(replace(longer,edges=()),pool)


def test_new_submit_memory_incremental_replay_matches_full_execution():
    executor = RevalidationExecutor()
    c = context('books','target','replay-new',0,17,template='future')
    old = ('lookup_key','transform_rule','auth_pair_a','auth_pair_b')
    reference = executor.run(c,old)
    exposure = old+('new_contract',)
    partial = executor.replay(c,exposure,reference)
    full = executor.run(c,exposure)
    assert partial.events==full.events and partial.observations==full.observations
    assert partial.state_hash==full.state_hash and partial.outcome==full.outcome
    assert partial.calls+partial.reused_calls==full.calls
    assert partial.reused_calls>0 and partial.calls<full.calls
    with pytest.raises(ValueError,match='unknown memory scope'):
        executor.replay(c,old+('unknown',),reference)
