"""Executed role knowledge gradients, balanced designs and reversible storage.

Planning may query a hypothetical restored role. Such values never grant
transport or deployment permission. Only collected evidence and fresh bank
certification can change an actual exposure policy.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict,replace
from itertools import combinations

import numpy as np
import torch

from .v3.audit import SavedCalibration
from .v3.drift import Level,TransportGate,posterior_success
from .v3.experiment import write
from .v3.factorial import admitted_pairs,regular_fraction,estimate
from .v3.inference import worst_mixture,ppi,tune_lambda,certify
from .v3.local import ranked,retriever,stable_seed
from .v3.outcome import Heads
from .v3.selection import bank_cost
from .v3.types import reserved_slots
from .v3_benchmark import CachedModel
from .v3_optimized_model import StateAwareOutcomeModel
from .v3_revalidation import EvidenceGate,candidate_memories,domain_key,RevalidationExecutor,admit_memories
from .v3_screening_optimization import public_profiles,ReuseSelector


class OnlineModel(StateAwareOutcomeModel):
    """Explicit snapshot features prevent source/target observations aliasing."""
    def __init__(self,pool,contexts,*,seed=0):
        super().__init__(pool,contexts,seed=seed)
        torch.manual_seed(seed)
        self.heads = Heads(len(self.roles),feature_size=34,memory_size=32,hidden=32)

    def _pack(self,items):
        packed = list(super()._pack(items))
        versions = [[float(n.version.endswith(':source')),float(n.version.endswith(':target'))]
                    for c,n,a in items]
        packed[0] = torch.cat((packed[0],torch.tensor(versions)),dim=1)
        return tuple(packed)

    def fit_online(self,train,validation,calibration,*,logged=(),baseline_steps=20):
        random_ids = {e.context.uid for e in train+validation+calibration}
        if random_ids & {e.context.uid for e in logged}:raise ValueError('logged/randomized split overlap')
        before = {k:v.detach().clone() for k,v in self.heads.state_dict().items() if not k.startswith('baseline.')}
        if logged:
            packed,labels,recovery = self.pack_observations(logged)
            packed,counts = self.compress(packed,labels,recovery)
            opt = torch.optim.Adam(self.heads.baseline.parameters(),lr=.005)
            for _ in range(baseline_steps):
                opt.zero_grad();loss = self.compressed_loss(self.heads(*packed,baseline_only=True),counts)
                loss.backward();opt.step()
        if any(not torch.equal(v,self.heads.state_dict()[k]) for k,v in before.items()):
            raise ValueError('observational baseline pretraining changed exposure parameters')
        super().fit(train,calibration,validation=validation,epochs=600)
        self.fit_report.update(logged_baseline_episodes=len(logged),logged_baseline_steps=baseline_steps if logged else 0,
                               logged_exposure_gradients_blocked=True,version_features='source/target snapshot indicators')
        return self


def fit_ensemble(train,validation,calibration,seed,directory,*,logged=(),members=2):
    directory.mkdir()
    grouped = defaultdict(list)
    for e in train:grouped[e.context.uid].append(e)
    ids = sorted(grouped);rng = np.random.default_rng(seed);models = []
    for i in range(members):
        sample = [e for uid in rng.choice(ids,len(ids),replace=True) for e in grouped[uid]]
        model = OnlineModel(candidate_memories(),[e.context for e in train],seed=seed+i)
        model.fit_online(sample,validation,calibration,logged=logged)
        model.save(directory/f'outcome_model_{i}.pt');models.append(CachedModel(model))
        print(seed,directory.name,i,'epochs',model.fit_report['iterations'],flush=True)
    write(directory/'training_reports.json',[m.fit_report for m in models])
    return tuple(models)


def load_ensemble(directory,contexts):
    models = []
    for path in sorted(directory.glob('outcome_model_*.pt')):
        data = torch.load(path,map_location='cpu',weights_only=False)
        model = OnlineModel(candidate_memories(),contexts)
        if model.roles!=data['role_indices'] or [m.uid for m in model.memories]!=data['memory_ids']:
            raise ValueError('online model registry differs')
        model.heads.load_state_dict(data['state_dict'])
        model.calibrator = SavedCalibration(data['calibration_x'],data['calibration_y'])
        model.fit_report = data['fit_report'];models.append(CachedModel(model))
    if len(models)!=2:raise ValueError('two-member ensemble required')
    return tuple(models)


def balanced_designs(contexts,pool,ledger,seed,*,phase='target-factorial',archived=()):
    """K2 is uniform marginally, without replacement across public-template blocks.

    Complete cycles over all optional combinations guarantee coverage without
    selecting assignments using outcomes. Each context has independent noise
    and an independent random sign vector. Partial final cycles remain random.
    """
    executor = RevalidationExecutor();reports,episodes = [],[]
    queues,cursors = {},defaultdict(int)
    rng = np.random.default_rng(seed)
    for c in contexts:
        eligible = [m for m in pool if m.uid not in archived and any(m.eligible(n.role) for n in c.nodes)]
        top = ranked(c,tuple(m.uid for m in eligible))[:min(4,max(len(eligible)-1,0))]
        remaining = sorted(m.uid for m in eligible if m.uid not in top)
        k2 = min(2,len(remaining));choices = list(combinations(remaining,k2))
        key = tuple(n.role.key for n in c.nodes),tuple(top),tuple(remaining)
        if key not in queues or cursors[key]==len(choices):
            queues[key] = [choices[i] for i in rng.permutation(len(choices))];cursors[key]=0
        optional = queues[key][cursors[key]];cursors[key]+=1
        ids = top+optional;slots = tuple(next(m for m in pool if m.uid==uid) for uid in ids)
        fraction = regular_fraction(len(slots),admitted_pairs(c,slots),seed=stable_seed(c.uid))
        checkpoint = replace(c,uid=c.uid+':draw-0',seed=stable_seed(c.seed,0))
        rows = fraction.draw(rng);ys,reference = [],None
        for row in rows:
            frame = reserved_slots(slots,row)
            if reference is None:
                e = ledger.execute(phase,executor,checkpoint,frame.exposure,slots=frame);reference=e
            else:
                if ledger.remaining(phase)<1:raise ValueError('balanced design episode budget exhausted')
                if ledger.spent+2*len(checkpoint.nodes)>ledger.call_cap:
                    raise ValueError('reserve suffix-call bound before replay')
                e = executor.replay(checkpoint,frame.exposure,reference,slots=frame);ledger.charge(phase,e)
            if not e.replay_valid:raise ValueError('unsafe balanced replay')
            e = replace(e,randomized=True,assignment_probability=2.**-len(slots))
            ledger.records[-1]['episode']=e;episodes.append(e);ys.append(e.outcome)
        full = ledger.execute('replay_audit',executor,checkpoint,e.exposure)
        if (full.events,full.observations,full.state_hash)!=(e.events,e.observations,e.state_hash):
            raise ValueError('balanced suffix differs from full execution')
        value = estimate(fraction,[rows],[ys])
        reports.append({'context':asdict(c),'memory_ids':list(ids),'banzhaf':value['banzhaf'].tolist(),
            'beta':value['beta'].tolist(),'effects':value['effects'],'generator_columns':fraction.columns,
            'runs_per_draw':fraction.runs,'minimum_fraction_proven':fraction.minimum_proven,
            'assignment_probability':value['assignment_probability'],'draws':1,
            'reserved_slots_length_matched':True,'optional_uniform_probability':k2/max(1,len(remaining)),
            'sampling':'random permutation of all optional combinations per template; independent signs/noise'})
    return episodes,reports


def admit(reports,episodes,pool,evidence,minimum_contexts=24):
    families = len({r['context']['stratum'] for r in reports})
    # Keep alpha .05/36 even when an adaptive policy has not sampled every family.
    return admit_memories(reports,episodes,pool,TransportGate(evidence),minimum_contexts=minimum_contexts,
                          alpha=.05*families/4)


class Store:
    def __init__(self,pool=None,*,quarantine_capacity=40,archive_after=2):
        self.pool = tuple(pool or candidate_memories());self.quarantine_capacity=quarantine_capacity
        self.archive_after=archive_after;self.entries={};self.events=[]

    def refresh(self,profiles,gate,cycle,*,role_values=None):
        for c in profiles:
            eligible = {m.uid for m in gate.candidates(c,self.pool)}
            for m in self.pool:
                if not any(m.eligible(n.role) for n in c.nodes):continue
                key = c.stratum,m.uid
                old = self.entries.get(key,{'last_exposed':-1,'state':'quarantined','versions':[]})
                versions = sorted({n.version for n in c.nodes if m.eligible(n.role)})
                if old['state']=='archived' and old['versions']==versions:continue
                state = 'active' if m.uid in eligible else 'quarantined'
                self.entries[key] = old|{'state':state,'versions':versions}
        for stratum in sorted({k[0] for k in self.entries}):
            keys = [k for k,e in self.entries.items() if k[0]==stratum and e['state']=='quarantined']
            cost = sum(next(m.cost for m in self.pool if m.uid==k[1]) for k in keys)
            for key in sorted(keys,key=lambda k:((role_values or {}).get(k,0.),k[1])):
                if cost<=self.quarantine_capacity:break
                m = next(m for m in self.pool if m.uid==key[1]);cost-=m.cost
                self.entries[key]['state']='cold'
                self.events.append({'cycle':cycle,'stratum':stratum,'memory':m.uid,'action':'cold',
                                    'reason':'quarantine capacity binds; reversible lowest VOI storage'})

    def expose(self,episodes,cycle):
        for e in episodes:
            for uid in e.exposure:
                key=e.context.stratum,uid
                if key in self.entries:self.entries[key]['last_exposed']=cycle

    def fetch(self,role,version,cycle):
        family=version.split(':')[1]
        for key,e in self.entries.items():
            m=next(m for m in self.pool if m.uid==key[1])
            if key[0]==family+'/target' and role in (m.scope or ()) and e['state']=='cold':
                e['state']='quarantined';self.events.append({'cycle':cycle,'stratum':key[0],
                    'memory':m.uid,'action':'fetch','reason':'role revalidation fetched reversible cold storage'})

    def available(self,c,uid):
        return self.entries.get((c.stratum,uid),{}).get('state') not in ('cold','archived')

    def snapshot(self):
        return deepcopy({'entries':[{'stratum':k[0],'memory':k[1]}|v for k,v in sorted(self.entries.items())],
                'events':self.events,'quarantine_capacity':self.quarantine_capacity,'archive_after':self.archive_after}
                )

    @classmethod
    def restore(cls,data):
        s=cls(quarantine_capacity=data['quarantine_capacity'],archive_after=data['archive_after'])
        s.entries={(r['stratum'],r['memory']):{k:v for k,v in r.items() if k not in ('stratum','memory')} for r in data['entries']}
        s.events=list(data['events']);return s


class StoredGate(EvidenceGate):
    def __init__(self,evidence,leases,store):
        super().__init__(evidence,leases,{'new_contract'});self.store=store

    def candidates(self,c,memories,*,directly_validated=frozenset()):
        result=super().candidates(c,memories,directly_validated=directly_validated)
        return tuple(m for m in result if self.store.available(c,m.uid))


def moves(incumbent,unblocked,pool,capacity=80):
    result={tuple(sorted(incumbent))}
    for uid in unblocked:
        result.add(tuple(sorted(set(incumbent)|{uid})))
        for old in incumbent:result.add(tuple(sorted((set(incumbent)-{old})|{uid})))
    return sorted(b for b in result if bank_cost(b,pool)<=capacity)


def queue(models,evidence,leases,incumbent,probabilities,*,probe_penalty=1e-6):
    """Evaluate Eq.VOI on feasible single add/swap moves and execute later.

    Restoration predictions are planning hypotheses, not allowed predictions
    for screening/certification. All joint-bank scores use one shared objective.
    """
    pool=candidate_memories();mixture={f'{f}/{e}':.125 for f in ('books','travel','inventory','payments') for e in ('source','target')}
    profiles=public_profiles(mixture);base=EvidenceGate(evidence,leases,{'new_contract'})
    def score(bank,gate):
        values=np.mean([m.predict_many([(c,retriever(4)(c,tuple(x.uid for x in gate.candidates(c,[x for x in pool if x.uid in bank]))))
                                        for c in profiles]) for m in models],axis=0)
        values=np.asarray([v if gate.transported(c) else 0. for c,v in zip(profiles,values)])
        means=[float(np.mean([v for c,v in zip(profiles,values) if c.stratum==k])) for k in mixture]
        return worst_mixture(means,list(mixture.values()),.2)-.0001*bank_cost(bank,pool)
    baseline=score(incumbent,base);rows=[]
    for p in sorted(evidence,key=lambda p:(p.version,p.role)):
        if not p.version.endswith(':target') or p.level==Level.V:continue
        family=p.version.split(':')[1]
        affected=[c for c in profiles if c.stratum==family+'/target']
        unblocked=[]
        for m in pool:
            if p.role not in (m.scope or ()):continue
            if p.level==Level.U and m.uid=='new_contract':continue
            if all(m.uid not in base.allowed(c) for c in affected):unblocked.append(m.uid)
        if not unblocked:continue
        hypothetical=[replace(e,level=Level.V) if (e.role,e.version)==(p.role,p.version) else e for e in evidence]
        hypothetical_leases=dict(leases)
        for c in affected:hypothetical_leases[domain_key(c)]=sorted(base.allowed(c)|set(unblocked))
        shadow=EvidenceGate(hypothetical,hypothetical_leases,{'new_contract'})
        proposed=moves(incumbent,unblocked,pool)
        evaluated=[(score(b,shadow),b) for b in proposed]
        potential,best=max(evaluated,key=lambda x:(x[0],tuple(reversed(x[1]))))
        kind='differential' if p.level==Level.U else 'factorial'
        successes,failures=probabilities.get(kind,(0,0));prob=posterior_success(successes,failures)
        # Fixtures use two calls each. Factorial cost is a conservative public
        # suffix-call bound, not a probability from the hidden simulator kernel.
        calls=2048 if kind=='differential' else 96*(32*2*6+2*6)
        gain=max(0.,potential-baseline)
        rows.append({'role':p.role,'version':p.version,'kind':kind,'level':p.level.value,
            'unblocked':unblocked,'candidate_moves':len(proposed),'best_move':best,'baseline_score':baseline,
            'potential_score':potential,'gain':gain,'beta_successes':successes,'beta_failures':failures,
            'probability':prob,'probe_call_bound':calls,'probe_penalty':probe_penalty,
            'voi':prob*gain-probe_penalty*calls,
            'prediction_scope':'counterfactual planning only; no transport/deployment permission',
            'success_event':'role reaches V' if kind=='differential' else 'a new role-scoped direct lease is admitted',
            'voi_scope':'Eq.VOI for U; optimistic lease-admission proxy for D/X,not a claim of source equivalence'})
    return sorted(rows,key=lambda r:(-r['voi'],r['version'],r['role']))


class PolicySelector(ReuseSelector):
    """Certify the whole bank/gate/store policy against an immutable prior."""
    def _policy_values(self,phase,contexts,policies):
        result={}
        for label,(bank,gate) in policies.items():
            result[label]={};self.ledger.bank={'policy':label,'bank':bank}
            for k,tasks in contexts.items():
                values=[]
                for c in tasks:
                    a=self.exposures(c,bank,gate);key=c,tuple(a)
                    if key in self.execution_cache:
                        index,e=self.execution_cache[key]
                        if not e.replay_valid or e.context!=c or e.exposure!=tuple(a):raise ValueError('policy reuse mismatch')
                        self.reuse_records.append({'phase':phase,'bank':list(bank),'policy':label,
                            'context':c.uid,'exposure':list(a),'physical_record':index})
                    else:
                        e=self.ledger.execute(phase,self.executor,c,a)
                        if not e.replay_valid:raise ValueError('unsafe policy evaluation')
                        self.execution_cache[key]=len(self.ledger.records)-1,e
                    values.append(e.outcome)
                result[label][k]=np.asarray(values,dtype=float)
        self.ledger.bank=None;return result

    def _policy_predictions(self,contexts,bank,gate):
        return self.predictions(contexts,bank,gate=gate)

    def switch_policy(self,incumbent,prior_gate,finalists,*,pilot_n=16,certification_n=200,unlabeled_n=1000,freeze=None):
        profiles=public_profiles(self.mixture);incumbent=tuple(sorted(incumbent))
        proposals=list(dict.fromkeys(tuple(sorted(b)) for b in [*finalists,incumbent]))
        proposals=[b for b in proposals if bank_cost(b,self.memories)!=bank_cost(incumbent,self.memories)
                   or any(self.exposures(c,b)!=self.exposures(c,incumbent,prior_gate) for c in profiles)]
        if not proposals:return {'bank':incumbent,'switched':False,'certificates':[],'policy':'prior','reason':'no policy challenger'},prior_gate
        if len(proposals)>3:raise ValueError('at most three policy challengers')
        policies={'prior':(incumbent,prior_gate)}|{f'challenger-{i}':(b,self.gate) for i,b in enumerate(proposals)}
        if (certification_n<200 or pilot_n<2 or unlabeled_n<2
            or certification_n*len(policies)*len(self.mixture)>self.ledger.remaining('certification')):
            return {'bank':incumbent,'switched':False,'certificates':[],'policy':'prior','reason':'insufficient policy certificate budget'},prior_gate
        pilot_contexts=self.contexts('pilot',pilot_n)
        pilot=self._policy_values('pilot',pilot_contexts,policies)
        labeled=self.contexts('certification',certification_n);unlabeled=self.contexts('certification-unlabeled',unlabeled_n)
        predictions={label:{k:{phase:self._policy_predictions(tasks[k],bank,gate)
                        for phase,tasks in [('pilot',pilot_contexts),('labeled',labeled),('unlabeled',unlabeled)]}
                    for k in self.mixture} for label,(bank,gate) in policies.items()}
        weights={label:{k:tune_lambda(pilot[label][k]-pilot['prior'][k],
                    predictions[label][k]['pilot']-predictions['prior'][k]['pilot'],
                    labeled_n=certification_n,unlabeled_n=unlabeled_n,
                    transported=k in self.transported_strata and all(prior_gate.transported(c) for c in pilot_contexts[k]))
                for k in self.mixture} for label in policies if label!='prior'}
        if freeze:freeze({'incumbent':incumbent,'finalists':proposals,'policy_labels':{k:list(v[0]) for k,v in policies.items()},
            'mixture':self.mixture,'pilot_weights':weights,'model_report':[m.fit_report for m in self.models],
            'certification_contexts':{k:[asdict(c) for c in cs] for k,cs in labeled.items()},
            'prediction_hash_input':{label:{k:{p:v.tolist() for p,v in values.items()} for k,values in strata.items()}
                                     for label,strata in predictions.items()},
            'prior_policy_frozen':True,'same_bank_gate_changes_are_challengers':True})
        real=self._policy_values('certification',labeled,policies);certificates=[]
        for label,(bank,gate) in policies.items():
            if label=='prior':continue
            estimates={k:ppi(real[label][k]-real['prior'][k],
                predictions[label][k]['labeled']-predictions['prior'][k]['labeled'],
                predictions[label][k]['unlabeled']-predictions['prior'][k]['unlabeled'],weights[label][k]) for k in self.mixture}
            cert=certify(estimates,self.mixture,cost_difference=bank_cost(bank,self.memories)-bank_cost(incumbent,self.memories),
                         cost_penalty=.0001,finalists=len(proposals),alpha1=.05,alpha2=.05,margin=.05,minimum_n=200)
            cert.update(bank=bank,policy=label,estimates={k:asdict(v) for k,v in estimates.items()});certificates.append(cert)
        passed=[c for c in certificates if c['passed']]
        best=max(passed,key=lambda c:c['aggregate_lower']-c['cost_threshold']) if passed else None
        decision={'bank':best['bank'] if best else incumbent,'policy':best['policy'] if best else 'prior',
                  'switched':bool(best),'certificates':certificates,'prior_policy_frozen':True,
                  'fresh_certification':True,'same_bank_gate_changes_are_challengers':True}
        return decision,self.gate if best else prior_gate
