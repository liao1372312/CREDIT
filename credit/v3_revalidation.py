"""Current-version evidence leases, without transporting divergent source heads.

Leases admit memories to direct screening on a finite, pinned domain. They do
not certify a singleton's deployment value: an independent whole-bank switch
certificate is still required. Factorial admission intervals are approximate;
final switch validity does not depend on their correctness.
"""
from dataclasses import asdict, replace
import json

import numpy as np
from scipy.stats import t

from .v3.drift import Level, TransportGate, behavior_class, transfer_level
from .v3.local import ROLES, ROLE_NAMES, stable_seed, memories, StatefulExecutor
from .v3.types import Memory
from .stateful import request_probe


def domain_key(c):
    return json.dumps((c.stratum, tuple((n.role.key,n.version,n.features) for n in c.nodes),
                       c.edges,c.initial_state),sort_keys=True,separators=(',',':'))


def candidate_memories():
    text = 'Use the current patched submit binding with the derived identifier and authorization token.'
    return memories()+(Memory('new_contract',text,tuple(stable_seed(w)%2**31 for w in text.split()),
        frozenset({ROLES['submit'].key}),'new candidate from the public current submit contract; no numeric effects',8),)


class RevalidationExecutor(StatefulExecutor):
    def replay(self,context,exposure,reference,*,slots=None):
        changed = set(exposure)^set(reference.exposure)
        known = {m.uid:m for m in candidate_memories()}
        if not changed<=known.keys():
            raise ValueError('unknown memory scope; full replay required')
        first = next((i for i,n in enumerate(context.nodes)
                      if any(known[m].eligible(n.role) for m in changed)),len(context.nodes))
        return self.run(context,exposure,slots=slots,prefix=reference,start=first)


class EvidenceGate(TransportGate):
    def __init__(self, evidence=(), leases=None,novel_ids=()):
        super().__init__(evidence)
        self.leases = leases or {}
        self.novel_ids = frozenset(novel_ids)

    def allowed(self,c):
        return frozenset(self.leases.get(domain_key(c),()))

    def candidates(self,c,memories,*,directly_validated=frozenset()):
        approved = self.allowed(c)
        if not directly_validated <= approved:
            raise ValueError('caller cannot bypass the pinned evidence lease')
        eligible = [m for m in memories if m.uid not in self.novel_ids or m.uid in approved]
        return super().candidates(c,eligible,directly_validated=approved)


def reprobe(evidence,seed,*,fixtures=1024):
    """Fixed fresh batch for U roles; original 96 fixtures are not reused in CI."""
    updated,records,decisions = [],[],[]
    for p in sorted(evidence,key=lambda p:(p.version,p.role)):
        if p.level != Level.U or not p.version.endswith(':target'):
            updated.append(p)
            continue
        _,family,name,_ = p.version.split(':')
        role = ROLES[name]
        changes = 0
        for i in range(fixtures):
            outputs,classes = [],[]
            state = {'lookup':0,'transform':1,'authorize':3,'submit':7}[name]
            for epoch in ('source','target'):
                result = request_probe(stable_seed(seed,'fresh-role-revalidation',family,name,i),1,
                                       (epoch,family,name,state,'standard'))
                status,next_state = next(iter(result['counts'])).split(':')
                cls = behavior_class(200 if status=='ok' else 400 if status=='reject' else 500,True,
                    {field:'declared-id' if status=='ok' else 'missing' for field in role.consumed_outputs},
                    {'state':int(next_state)!=state},role)
                outputs.append(result);classes.append(cls)
            changed = classes[0]!=classes[1]
            changes += int(changed)
            records.append({'family':family,'role':role.key,'fixture':i,'outputs':outputs,
                            'changed':changed,'api_calls':2})
        # A single fresh decision per possible target role, with a 16-role error allocation.
        fresh = transfer_level(role,role,p.version,changed=changes,fixtures=fixtures,
                               support=p.support,alpha=.05/16)
        updated.append(fresh)
        decisions.append({'before':asdict(p),'after':asdict(fresh),
            'uses_fresh_fixtures_only':True,'alpha':.05/16,
            'prior_success_probability':.5,'posterior_success_probability':2/3 if fresh.level==Level.V else 1/3})
    return tuple(updated),records,decisions


def admit_memories(reports,episodes,pool,gate,*,minimum_contexts=12,minimum_node_labels=12,alpha=.05):
    """Positive clustered Banzhaf evidence admits a candidate on observed domains.

All draws of one base context are a single cluster. Selection effects refer
to randomized reserved-slot backgrounds, not arbitrary deployed retrieval.
"""
    samples,domains,support = {},{},{}
    for r in reports:
        from .v3.audit import load_context
        c = load_context(r['context'])
        domains.setdefault(c.stratum,set()).add(domain_key(c))
        for uid,value in zip(r['memory_ids'],r['banzhaf']):
            samples.setdefault((c.stratum,uid),[]).append(float(value))
    for e in episodes:
        nodes = {n.uid:n for n in e.context.nodes}
        for uid in e.exposure:
            memory = next(m for m in pool if m.uid==uid)
            for o in e.observations:
                n = nodes[o.node]
                if memory.eligible(n.role) and o.label in ('ok','agent'):
                    key = e.context.stratum,uid,n.role.key,n.version
                    support[key] = support.get(key,0)+1
    decisions,approved = [],{}
    # Familywise adjustment for every memory/family, including untested hypotheses.
    tests = max(1,len(pool)*len(domains))
    for stratum in sorted(domains):
        profiles = [r for r in reports if r['context']['stratum']==stratum]
        for m in sorted(pool,key=lambda m:m.uid):
            blocked = set()
            for r in profiles:
                from .v3.audit import load_context
                c = load_context(r['context'])
                blocked.update((n.role.key,n.version) for n in c.nodes
                               if m.eligible(n.role) and gate.level(n)!=Level.V)
            if not blocked:
                continue
            values = np.asarray(samples.get((stratum,m.uid),[]),dtype=float)
            n = len(values)
            mean = float(values.mean()) if n else None
            se = float(values.std(ddof=1)/np.sqrt(n)) if n>=2 else None
            lower = float(mean-t.ppf(1-alpha/tests,n-1)*se) if n>=2 else None
            node_counts = {role+'@'+version:support.get((stratum,m.uid,role,version),0)
                           for role,version in sorted(blocked)}
            passed = (n>=minimum_contexts and all(v>=minimum_node_labels for v in node_counts.values())
                      and lower is not None and lower>0)
            decisions.append({'stratum':stratum,'memory':m.uid,'contexts':n,'banzhaf_mean':mean,
                'standard_error':se,'lower':lower,'alpha_per_test':alpha/tests,'node_support':node_counts,
                'admitted':bool(passed),'interval':'clustered Student-t approximation; admission only'})
            if passed:
                approved.setdefault(stratum,[]).append(m.uid)
    leases = {key:sorted(approved.get(stratum,[])) for stratum,keys in domains.items() for key in keys}
    return leases,decisions
