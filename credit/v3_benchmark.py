"""Preregistered multiseed synthetic comparisons, ablations and budget sweeps.

All strategies have the same API-call allowance (actual expenditures are
reported), fixed actor, retriever and strict verifier. No oracle is imported
until every method's bank is frozen and every held-out episode scored.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
from itertools import combinations
import json
from pathlib import Path
import time
import numpy as np
import torch

from .stateful import FAMILIES, RECENCY
from .v3_data import Corpus, SEEDS
from .v3.drift import Level, TransportGate
from .v3.experiment import collect_designs, digest, write
from .v3.factorial import admitted_pairs
from .v3.local import StatefulExecutor, memories, retriever, stable_seed
from .v3.local import differential_probes, ranked
from .v3.outcome import OutcomeModel, text_encoding
from .v3.selection import BankSelector, Ledger, bank_cost, screen_banks

MAIN = ('NoMem', 'FIFO', 'Recency', 'Version-filter', 'Factorial-Banzhaf',
        'Model-MCShapley', 'Direct-Sandbox', 'CREDIT')
VARIANTS = ('Without-interactions', 'Without-drift-gating', 'Without-robust-ranking',
            'Without-certification', 'Without-role-sharing', 'Heldout-payments', 'Label-error-20pct',
            'C40-K4', 'C120-K4', 'C80-K2', 'C80-K6')


class CachedModel:
    def __init__(self, model): self.model, self.cache, self.fit_report = model, {}, model.fit_report
    @staticmethod
    def key(c, exposure):
        return (tuple((n.role.key, n.version, n.features) for n in c.nodes), c.edges, c.initial_state, tuple(exposure))
    def predict_many(self, items):
        absent = {self.key(c, a): (c, a) for c, a in items if self.key(c, a) not in self.cache}
        if absent:
            values = self.model.predict_many(list(absent.values()))
            self.cache.update(zip(absent, map(float, values)))
        return np.asarray([self.cache[self.key(c, a)] for c, a in items])


class NoPairs(OutcomeModel):
    def _pack(self, items):
        packed = list(super()._pack(items)); packed[4] = torch.zeros_like(packed[4]); return tuple(packed)


class GraphConditioned(OutcomeModel):
    def _pack(self, items):
        packed = list(super()._pack(items))
        # Graph-specific operation encodings replace the shared role signature.
        for i, (c, node, _) in enumerate(items):
            packed[0][i, :8] = torch.tensor(text_encoding(str((node.role.key, node.uid,
                                                           tuple(n.role.key for n in c.nodes)))))
        return tuple(packed)


def fit_models(pool, episodes, calibration, seed, model_class=OutcomeModel):
    rng = np.random.default_rng(seed)
    grouped = {}
    for e in episodes: grouped.setdefault(e.context.uid, []).append(e)
    ids = sorted(grouped)
    models = []
    for i in range(3):
        sampled = [e for uid in rng.choice(ids, len(ids), replace=True) for e in grouped[uid]]
        model = model_class(pool, [e.context for e in episodes], seed=seed+i)
        model.fit(sampled, calibration, epochs=200)
        models.append(model)
    return models


class CallLedger(Ledger):
    def __init__(self, limits, call_cap=150000, common_calls=0):
        super().__init__(limits)
        self.call_cap, self.common_calls, self.spent = call_cap, common_calls, common_calls
        self.tag = 'unassigned'
        self.bank = None
    def execute(self, phase, executor, context, exposure, *, slots=None):
        if self.spent + 2*len(context.nodes) > self.call_cap:
            raise ValueError('physical API-call allowance exhausted before execution')
        return super().execute(phase, executor, context, exposure, slots=slots)
    def charge(self, phase, episode):
        if self.spent + episode.calls > self.call_cap: raise ValueError('API allowance exceeded')
        super().charge(phase, episode)
        self.spent += episode.calls
        self.records[-1].update(method=self.tag, bank=self.bank)
    def report(self):
        return super().report() | {'call_allowance': self.call_cap, 'common_history_api_calls': self.common_calls,
                                  'selection_api_calls_including_common': self.spent,
                                  'unspent_call_allowance': self.call_cap-self.spent}


class TrackedSelector(BankSelector):
    def _executions(self, phase, contexts, banks):
        result = {}
        for bank in banks:
            self.ledger.bank = bank
            result.update(super()._executions(phase, contexts, [bank]))
        self.ledger.bank = None
        return result


def pack(order, pool, capacity):
    selected = []
    for uid in order:
        proposal = tuple(selected+[uid])
        if bank_cost(proposal, pool) <= capacity: selected.append(uid)
    return tuple(sorted(selected))


def make_selector(pool, models, gate, corpus, executor, ledger, *, capacity=80, depth=4, radius=.2):
    mixture = {f'{f}/{e}': 1/8 for f in FAMILIES for e in ('source','target')}
    transported = {k for k in mixture if all(gate.transported(corpus.make(k, 'screening-unlabeled', i)) for i in range(32))}
    return TrackedSelector(pool, models, gate, retriever(depth), corpus.make, executor, ledger,
                           mixture=mixture, capacity=capacity, radius=radius, cost_penalty=.0001,
                           transported_strata=transported)


def selection_limits():
    return {'direct-screen': 4096, 'halving': 4096, 'pilot': 2400, 'certification': 4800}


def stream_records(path, records):
    with gzip.open(path, 'wt', encoding='utf8', compresslevel=6) as f:
        for r in records:
            f.write(json.dumps(r | {'episode': asdict(r['episode'])}, separators=(',',':'))+'\n')


def learned_bank(method, selector, directory, *, direct=False, certify_bank=True):
    contexts = selector.contexts('screening-unlabeled', 32)
    pool = selector.memories
    bundles = {tuple(sorted((pool[i].uid, pool[j].uid))) for c in contexts['books/source']
               for i,j in admitted_pairs(c,pool)}
    if direct:
        tasks = selector.contexts('halving-5', 4)  # Dedicated, disjoint reserved screening block.
        counter = [0]
        # Reserve worst-case physical calls for halving, pilot and certification.
        reserve = 3 * sum(2*len(selector.factory(k, phase, i).nodes)
                          for k in selector.mixture for phase,n in [('pilot',100),('certification',200)]
                          for i in range(n))
        for phase,n,banks in [('halving-0',32,8),('halving-1',64,4)]:
            reserve += banks * sum(2*len(selector.factory(k,phase,i).nodes)
                                   for k in selector.mixture for i in range(n))
        def score(bank):
            if counter[0] >= 128: return -1 - .0001*bank_cost(bank,pool)
            upper=sum(2*len(c.nodes) for group in tasks.values() for c in group)
            if selector.ledger.spent + upper + reserve > selector.ledger.call_cap:
                return -1 - .0001*bank_cost(bank,pool)
            counter[0] += 1
            values = selector._executions('direct-screen', tasks, [bank])[bank]
            from .v3.inference import worst_mixture
            return worst_mixture([v.mean() for v in values.values()], [.125]*8,.2)-.0001*bank_cost(bank,pool)
    else:
        score = lambda b: selector.model_score(contexts,b)
    screen = screen_banks(pool,(),score,capacity=selector.capacity, admitted_bundles=bundles,keep=8)
    if direct:
        screen.update(real_candidates_executed=counter[0],
                      unexecuted_candidates_excluded=screen['evaluated']-counter[0])
        if any(v <= -1 for v in screen['scores'].values()):
            raise ValueError('direct finalist was not evaluated by real executions')
    halving = selector.halve(screen['banks'],gamma=4096 if direct else 1024,finalists=2)
    write(directory/'screening.json',screen);write(directory/'halving.json',halving)
    if certify_bank:
        decision = selector.switch((),halving['finalists'],pilot_n=100,certification_n=200,unlabeled_n=1000,
                                   freeze=lambda d:write(directory/'certification_freeze.json',d))
    else:
        bank = tuple(halving['finalists'][0]) if halving['finalists'] else ()
        decision = {'bank':bank,'switched':bool(bank),'certified':False,'reason':'certification removed in ablation'}
    write(directory/'switch_decision.json',decision)
    return tuple(decision['bank'])


def prepare(seed, corpus, output, *, call_cap, ablations=True):
    directory=output/f'seed_{seed}';directory.mkdir()
    pool, engine = memories(), StatefulExecutor()
    history = CallLedger({'factorial':4096,'calibration':2048,'replay_audit':64},call_cap)
    history.tag='shared-history'
    train_contexts = [c for c in corpus.phase('training') if int(c.uid.rsplit(':',1)[1]) < 8]
    val_contexts = [c for c in corpus.phase('calibration') if int(c.uid.rsplit(':',1)[1]) < 8]
    train, designs = collect_designs(train_contexts,pool,engine,history,phase='factorial',seed=seed,draws=2,audit=True)
    calibration,_ = collect_designs(val_contexts,pool,engine,history,phase='calibration',seed=seed+1,draws=1)
    write(directory/'factorial_estimates.json',designs)
    models=fit_models(pool,train,calibration,seed)
    for i,m in enumerate(models):m.save(directory/f'outcome_model_{i}.pt')
    write(directory/'training_reports.json',[m.fit_report for m in models])
    support=Counter()
    for e in train:
        lookup={n.uid:n for n in e.context.nodes}
        for o in e.observations:
            if o.label in ('ok','agent'):
                n=lookup[o.node];support[n.role.key,n.version]+=1
    evidence, fixtures=differential_probes(seed=seed,support=support,fixtures=96)
    write(directory/'transfer_levels.json',[asdict(p) for p in evidence]);write(directory/'fixtures.json',fixtures)
    gate=TransportGate(evidence)
    common=history.spent+sum(r['api_calls'] for r in fixtures)
    stream_records(directory/'history.jsonl.gz',history.records)
    write(directory/'history_costs.json',history.report() | {'fixtures':sum(r['api_calls'] for r in fixtures)})
    policies={};costs={}
    banks={}
    fifo=pack([m.uid for m in sorted(pool,key=lambda m:m.arrival)],pool,80)
    recency=pack(sorted((m.uid for m in pool),key=lambda m:-RECENCY[m]),pool,80)
    banks.update(NoMem=(),FIFO=fifo,Recency=recency,**{'Version-filter':recency})
    scores={m.uid:[] for m in pool}
    for d in designs:
        for uid,v in zip(d['memory_ids'],d['banzhaf']):scores[uid].append(v)
    banzhaf={m:float(np.mean(v)) if v else 0 for m,v in scores.items()}
    banks['Factorial-Banzhaf']=pack(sorted((m for m,v in banzhaf.items() if v>0),key=lambda m:(-banzhaf[m],m)),pool,80)
    write(directory/'banzhaf_scores.json',banzhaf)
    for name in ('NoMem','FIFO','Recency','Version-filter'):
        costs[name]={'call_allowance':call_cap,'selection_api_calls_including_common':0,'common_history_api_calls':0}
    costs['Factorial-Banzhaf']={'call_allowance':call_cap,'selection_api_calls_including_common':common,'common_history_api_calls':common}
    cached=tuple(CachedModel(m) for m in models)
    mc_ledger=CallLedger({},call_cap,common)
    mc=make_selector(pool,cached,gate,corpus,engine,mc_ledger)
    contexts=mc.contexts('screening-unlabeled',32)
    rng=np.random.default_rng(seed+400)
    shapley={m.uid:0. for m in pool};cache={}
    def value(bank):
        key=tuple(sorted(bank))
        if key not in cache:cache[key]=mc.model_score(contexts,key)
        return cache[key]
    for _ in range(64):
        bank=()
        for uid in rng.permutation([m.uid for m in pool]):
            following=bank+(uid,);shapley[uid]+=(value(following)-value(bank))/64;bank=following
    banks['Model-MCShapley']=pack(sorted((m for m,v in shapley.items() if v>0),key=lambda m:(-shapley[m],m)),pool,80)
    costs['Model-MCShapley']=mc_ledger.report();write(directory/'model_shapley_scores.json',shapley)
    learned=['CREDIT','Direct-Sandbox']+(list(VARIANTS) if ablations else [])
    for method in learned:
        print(seed,method,'selection',flush=True)
        md=directory/method;md.mkdir()
        active_models=cached;active_gate=gate;capacity=80;depth=4;radius=.2
        if method=='Without-interactions':
            fitted=fit_models(pool,train,calibration,seed,NoPairs)
            for i,m in enumerate(fitted):m.save(md/f'outcome_model_{i}.pt')
            write(md/'training_reports.json',[m.fit_report for m in fitted]);active_models=tuple(CachedModel(m) for m in fitted)
        if method=='Without-role-sharing':
            fitted=fit_models(pool,train,calibration,seed,GraphConditioned)
            for i,m in enumerate(fitted):m.save(md/f'outcome_model_{i}.pt')
            write(md/'training_reports.json',[m.fit_report for m in fitted]);active_models=tuple(CachedModel(m) for m in fitted)
        if method=='Heldout-payments':
            retained=[e for e in train if not e.context.stratum.startswith('payments/')]
            val=[e for e in calibration if not e.context.stratum.startswith('payments/')]
            fitted=fit_models(pool,retained,val,seed)
            for i,m in enumerate(fitted):m.save(md/f'outcome_model_{i}.pt')
            write(md/'training_reports.json',[m.fit_report for m in fitted]);active_models=tuple(CachedModel(m) for m in fitted)
            active_gate=TransportGate([replace(p,level=Level.X,support=0) if ':payments:' in p.version else p for p in evidence])
        if method=='Label-error-20pct':
            from .v3.types import NodeObservation
            corrupted=[];changes=0;eligible=0
            for e in train:
                observations=[]
                for o in e.observations:
                    eligible+=int(o.label in ('agent','service'))
                    if o.label in ('agent','service') and stable_seed(seed,'label-error',e.context.uid,e.exposure,o.node)%100 < 20:
                        changes+=1;label='service' if o.label=='agent' else 'agent'
                        observations.append(NodeObservation(o.node,label,False))
                    else:observations.append(o)
                corrupted.append(replace(e,observations=tuple(observations)))
            fitted=fit_models(pool,corrupted,calibration,seed)
            for i,m in enumerate(fitted):m.save(md/f'outcome_model_{i}.pt')
            write(md/'label_error_protocol.json',{'eligible_failure_labels':eligible,'flipped':changes,
                                                'rule':'fixed hash swaps agent/service, recovery cleared; no test labels changed'})
            write(md/'training_reports.json',[m.fit_report for m in fitted]);active_models=tuple(CachedModel(m) for m in fitted)
        if method=='Without-drift-gating':active_gate=TransportGate([replace(p,level=Level.V) for p in evidence])
        if method=='Without-robust-ranking':radius=0
        if method=='C40-K4':capacity=40
        if method=='C120-K4':capacity=120
        if method=='C80-K2':depth=2
        if method=='C80-K6':depth=6
        ledger=CallLedger(selection_limits(),call_cap,common);ledger.tag=method
        selector=make_selector(pool,active_models,active_gate,corpus,engine,ledger,capacity=capacity,depth=depth,radius=radius)
        banks[method]=learned_bank(method,selector,md,direct=method=='Direct-Sandbox',certify_bank=method!='Without-certification')
        costs[method]=ledger.report();write(md/'costs.json',costs[method]);stream_records(md/'selection.jsonl.gz',ledger.records)
        policies[method]=(active_gate,depth)
    write(directory/'frozen_banks.json',{k:list(v) for k,v in banks.items()})
    write(directory/'method_costs.json',costs)
    policies.update({m:(gate,4) for m in ('Factorial-Banzhaf','Model-MCShapley')})
    return {'seed':seed,'corpus':corpus,'banks':banks,'policies':policies,'models':models,'evidence':evidence,'directory':directory}


def score(prepared, output):
    pool,engine=memories(),StatefulExecutor();rows=[]
    with gzip.open(output/'test_traces.jsonl.gz','wt',compresslevel=6) as stream:
        for p in prepared:
            seed,corpus=p['seed'],p['corpus']
            for method,bank in p['banks'].items():
                for c in corpus.phase('heldout-test'):
                    if method in p['policies']:
                        gate,depth=p['policies'][method]
                        eligible=tuple(m.uid for m in gate.candidates(c,[m for m in pool if m.uid in bank]))
                    else:
                        depth=4;eligible=bank
                        if method=='Version-filter' and c.stratum.endswith('/target'):eligible=()
                    a=retriever(depth)(c,eligible)
                    started=time.perf_counter();e=engine.run(c,a);elapsed=time.perf_counter()-started
                    family,epoch=c.stratum.split('/')
                    row={'seed':seed,'method':method,'task':c.uid,'family':family,'epoch':epoch,
                         'nodes':len(c.nodes),'success':e.outcome,'api_calls':e.calls,'bank_tokens':bank_cost(bank,pool),
                         'exposure':list(a),'exposure_tokens':sum(m.cost for m in pool if m.uid in a),
                         'stale_exposures':sum(1 for m in pool if m.uid in a and any(p_.level==Level.D and
                                                   p_.version in {n.version for n in c.nodes} and p_.role in m.scope for p_ in p['evidence'])),
                         'verification_known':e.verified is not None,'latency_seconds':elapsed,'state_hash':e.state_hash}
                    rows.append(row)
                    stream.write(json.dumps(row | {'events':e.events},separators=(',',':'))+'\n')
                print(seed,method,'test scored',flush=True)
    write(output/'raw_test_results.json',rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('datasets/credit_v3_synthetic_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    p.add_argument('--call-cap',type=int,default=150000)
    p.add_argument('--no-ablations',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError('use a new directory')
    args.output.mkdir(parents=True);(args.output/'source').mkdir()
    sources=[Path(__file__),Path('credit/v3_data.py'),*Path('credit/v3').glob('*.py'),Path('credit/stateful.py')]
    write(args.output/'frozen_protocol.json',{'seeds':args.seeds,'main_methods':MAIN,'variants':[] if args.no_ablations else VARIANTS,
        'dataset':str(args.dataset.resolve()),'dataset_manifest_sha256':digest(args.dataset/'manifest.json'),
        'source_sha256':{str(f):digest(f) for f in sources},'call_allowance_per_strategy':args.call_cap,
        'budgets':'same physical API-call allowance; actual spending can differ; common history charged to users',
        'training':32,'calibration':32,'members':3,'epochs':200,'test_n_per_stratum':256,
        'primary_comparison':'CREDIT vs Direct-Sandbox','scope':'new frozen synthetic workflow corpus, not real-provider/LLM evaluation',
        'ablation_limits':'role-sharing ablation uses graph-conditioned encodings; revalidation/OAT/observational learning not measured here'})
    for f in sources:(args.output/'source'/f.name).write_bytes(f.read_bytes())
    prepared=[prepare(seed,Corpus(args.dataset,seed),args.output,call_cap=args.call_cap,ablations=not args.no_ablations) for seed in args.seeds]
    write(args.output/'evaluation_freeze.json',{str(p.relative_to(args.output)):digest(p)
        for p in args.output.rglob('*') if p.is_file() and p.name in ('frozen_banks.json','switch_decision.json','certification_freeze.json')})
    rows=score(prepared,args.output)
    from .paired_statistics import summarize
    main=[r for r in rows if r['method'] in MAIN]
    table,paired=summarize(main,'CREDIT',repeats=4000,random_seed=2991)
    main_table=table
    write(args.output/'main_statistics.json',{'main':table,'paired':paired})
    if not args.no_ablations:
        table,paired=summarize([r for r in rows if r['method']=='CREDIT' or r['method'] in VARIANTS],'CREDIT',repeats=4000,random_seed=2991)
        write(args.output/'ablation_statistics.json',{'main':table,'paired':paired})
    write(args.output/'summary.json',{'test_episodes':len(rows),'test_api_calls':sum(r['api_calls'] for r in rows),
        'main_statistics':main_table,'seeds':args.seeds,'dataset_sha256':digest(args.dataset/'manifest.json'),
        'paper_real_service_experiments_complete':False})
    write(args.output/'artifact_manifest.json',{str(p.relative_to(args.output)):digest(p) for p in args.output.rglob('*') if p.is_file()})
    print('Completed multiseed benchmark',flush=True)

if __name__=='__main__':main()
