"""Evaluation-only exhaustive coalition reference and real-execution controls.

The private simulator kernel is imported only after the main benchmark has
finished. It never enters a training record, score, bank or certificate.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from itertools import combinations
import json
import math
from pathlib import Path
import numpy as np
import torch
from .stateful import FAMILIES
from .v3.audit import SavedCalibration, load_context
from .v3.experiment import digest, write
from .v3.factorial import admitted_pairs, characters, regular_fraction
from .v3.local import ROLES, ROLE_NAMES, StatefulExecutor, context, memories, stable_seed, retriever
from .v3.drift import Level, ProbeEvidence, TransportGate
from .v3.inference import worst_mixture
from .v3.selection import bank_cost
from .v3.outcome import OutcomeModel
from .v3.types import Node


def coalitions(k):
    bits = ((np.arange(2**k)[:, None] >> np.arange(k)) & 1)
    return bits, 2*bits-1


def coefficients(values, z):
    effects = ((),) + tuple(s for size in range(1, z.shape[1]+1)
                            for s in combinations(range(z.shape[1]), size))
    return effects, np.asarray(values) @ characters(z, effects) / len(z)


def credits(values, z):
    effects, beta = coefficients(values, z)
    k = z.shape[1]
    banzhaf = 2*beta[1:1+k]
    shapley = np.zeros(k)
    for s, b in zip(effects[1:], beta[1:]):
        if len(s) % 2:
            for i in s: shapley[i] += 2*b/len(s)
    # An independent classical subset formula checks the Walsh identity.
    classic = np.zeros(k)
    for i in range(k):
        for mask in range(2**k):
            if mask & (1<<i): continue
            size = int(mask).bit_count()
            weight = math.factorial(size)*math.factorial(k-size-1)/math.factorial(k)
            classic[i] += weight*(values[mask | (1<<i)]-values[mask])
    if not np.allclose(classic, shapley): raise ValueError('Shapley identity')
    return effects, beta, banzhaf, shapley


def kernel_probability(c, exposure):
    # Evaluation oracle. No caller is allowed to invoke this before the
    # benchmark's held-out results and all decision hashes are finalized.
    from .stateful import ACTIONS, _api, _policy, local_context
    family, epoch = c.stratum.split('/')
    states = {c.initial_state: 1.}
    for node in c.nodes:
        role = ROLE_NAMES[node.role.key]
        after = {}
        for state, mass in states.items():
            if state & 8: continue  # Remaining required nodes cannot complete.
            policy = _policy(role, *local_context(role, state, 'start'), exposure)
            success = {}; invalid = 0.
            for action, p in zip(ACTIONS, policy):
                for (status, target), q in _api(role, family, epoch, state, action).items():
                    if status == 'ok': success[target] = success.get(target, 0.)+p*q
                    elif status == 'reject': invalid += p*q
            # Invalid first requests leave state unchanged; only these are
            # retried once. Service transients never enter the recovery term.
            for target, p in success.items():
                after[target] = after.get(target, 0.)+mass*p*(1+invalid)
        states = after
    return sum(p for state, p in states.items() if state & 8)


def freeze(directory, reps=32):
    if directory.exists(): raise FileExistsError(directory)
    directory.mkdir(parents=True)
    cases = []
    for family in FAMILIES:
        for epoch in ('source', 'target'):
            for depth, template in ((1, 'future_ready_submit'), (4, 'train'), (6, 'future'), (9, 'future')):
                c = context(family, epoch, 'enumeration', depth, 977, template=template)
                if depth == 9:
                    names = ('lookup','transform','lookup','transform','lookup','transform','authorize','authorize','submit')
                    nodes = tuple(Node(f'node-{i}', ROLES[r], f'stateful:{family}:{r}:{epoch}',
                                       (FAMILIES.index(family)/3, i/8, 0., 0.)) for i, r in enumerate(names))
                    c = replace(c, nodes=nodes, edges=tuple((nodes[i].uid,nodes[i+1].uid) for i in range(8)))
                cases.append(asdict(c))
    write(directory/'contexts.json', cases)
    write(directory/'protocol.json', {'seed':977, 'cases':len(cases), 'depths':[1,4,6,9],
          'replications':reps, 'coalitions':256, 'pair_threshold':.015,
          'design_episode_budget':128, 'fractional_draws_per_replication':2,
          'oat_episode_budget':128, 'design_pair_scope':'all 28 pairs, including nonadmitted diagnostic pairs',
          'design_scope':'contrasts reuse the physically executed cube; paired noise within each case/replication',
          'oracle_use':'evaluation only, after all main banks and held-out scores are frozen',
          'source_sha256':digest(__file__), 'contexts_sha256':digest(directory/'contexts.json')})


def load_models(directory, cases):
    models = []
    for path in sorted(directory.glob('outcome_model_*.pt')):
        data = torch.load(path, map_location='cpu', weights_only=False)
        m = OutcomeModel(memories(), cases)
        if m.roles != data['role_indices']: raise ValueError('role map')
        m.heads.load_state_dict(data['state_dict'])
        m.calibrator = SavedCalibration(data['calibration_x'], data['calibration_y'])
        m.fit_report = data['fit_report']; models.append(m)
    if len(models) != 3: raise ValueError('ensemble incomplete')
    return models


def run(dataset, benchmark, output):
    if output.exists(): raise FileExistsError(output)
    # Strict causal separation: no oracle is touched before main completion.
    if not (benchmark/'artifact_manifest.json').exists(): raise ValueError('main benchmark must finish first')
    manifest = json.loads((benchmark/'artifact_manifest.json').read_text())
    for name in ('evaluation_freeze.json', 'raw_test_results.json', 'summary.json'):
        if digest(benchmark/name) != manifest[name]: raise ValueError('main freeze changed')
    output.mkdir(parents=True)
    protocol = json.loads((dataset/'protocol.json').read_text())
    if digest(dataset/'contexts.json') != protocol['contexts_sha256'] or digest(__file__) != protocol['source_sha256']:
        raise ValueError('reference protocol changed')
    cases = [load_context(c) for c in json.loads((dataset/'contexts.json').read_text())]
    pool = memories(); ids = [m.uid for m in pool]; bits, z = coalitions(8)
    exposures = [tuple(m for m,b in zip(ids, row) if b) for row in bits]
    truth = np.asarray([[kernel_probability(c,a) for a in exposures] for c in cases])
    reps = protocol['replications']; engine = StatefulExecutor()
    observed = np.zeros((len(cases), reps, len(exposures)), dtype=np.uint8)
    calls = np.zeros_like(observed); hashes = np.empty(observed.shape, dtype='S64')
    labels = {key:0 for key in ('ok','agent','service','amb')}; checked = 0; controls = 0; control_calls = 0
    from .stateful import _api
    for i, c in enumerate(cases):
        family, epoch = c.stratum.split('/')
        for rep in range(reps):
            task = replace(c, seed=stable_seed(977,'reference-noise',c.uid,rep))
            for j, a in enumerate(exposures):
                e = engine.run(task,a)
                if e.restored_hash != e.checkpoint_hash or not e.replay_valid: raise ValueError('unsafe reference')
                observed[i,rep,j], calls[i,rep,j], hashes[i,rep,j] = e.outcome,e.calls,e.state_hash.encode()
                first = {v['node']:v for v in e.events if v['attempt']==0}
                for o in e.observations:
                    labels[o.label]+=1
                    if o.node not in first:
                        if o.label!='amb': raise ValueError('missing event')
                        continue
                    event = first[o.node]
                    support = _api(event['role'], family,epoch,event['state_before'],event['request_mode'])
                    valid = any(status=='ok' for status,_ in support)
                    planted = ('ok' if event['response_status']=='ok' else 'service') if valid else 'agent'
                    if planted != o.label: raise ValueError('contract checker confusion')
                    checked += 1
            # Actual extra A/A and placebo executions on the same fixed task.
            a = exposures[(i*31+rep*7)%256]
            aa = engine.run(task,a); again = engine.run(task,a); noop = engine.run(task,a+('noop',))
            if aa.state_hash!=again.state_hash or aa.state_hash!=noop.state_hash or aa.calls!=noop.calls:
                raise ValueError('negative control')
            controls += 3
            control_calls += aa.calls+again.calls+noop.calls
        print('reference case',i,c.stratum,'depth',len(c.nodes),flush=True)
    np.savez_compressed(output/'coalitions.npz', truth=truth, outcomes=observed, calls=calls, state_hashes=hashes, z=z)
    effects = None; oracle_beta=[]; oracle_b=[]; oracle_s=[]
    for row in truth:
        effects,b,bz,sp = credits(row,z); oracle_beta.append(b);oracle_b.append(bz);oracle_s.append(sp)
    oracle_beta=np.asarray(oracle_beta)
    # Equal 128 episode contrasts, reconstructed from actual replay outcomes.
    pairs=tuple(combinations(range(8),2)); d=regular_fraction(8,pairs,seed=977)
    if d.runs*2!=128: raise ValueError('reference design budget')
    rng=np.random.default_rng(1009); errors={'Randomized-fraction':[],'OAT':[]}
    cols = [effects.index(s) for s in d.effects]
    for i in range(len(cases)):
        for rep in range(reps):
            y=observed[i,rep]; contrasts=[]
            for _ in range(2):
                rows=d.draw(rng); masks=((rows+1)//2) @ (1<<np.arange(8))
                contrasts.append(y[masks] @ characters(rows,d.effects)/d.runs)
            fraction=np.mean(contrasts,axis=0)
            oat=[np.nan]
            for s in d.effects[1:]:
                mask=int(rng.integers(256)); subtotal=0.
                for signs in range(2**len(s)):
                    modified=mask; contrast=1.
                    for j,bit in enumerate(s):
                        value=(signs>>j)&1; modified=(modified | (1<<bit)) if value else (modified & ~(1<<bit))
                        contrast*=2*value-1
                    subtotal+=y[modified]*contrast/(2**len(s))
                oat.append(subtotal)
            for name, values in [('Randomized-fraction',fraction),('OAT',np.asarray(oat))]:
                errors[name].append(values-oracle_beta[i,cols])
    design_rows=[]
    for name, error in errors.items():
        error=np.asarray(error); np.savez_compressed(output/f'design_{name}.npz', errors=error)
        design_rows.append({'method':name,'logical_episodes_per_replication':128,'main_rmse':float(np.sqrt(np.mean(error[:,1:9]**2))),
                            'pair_rmse':float(np.sqrt(np.mean(error[:,9:]**2)))})
    model_rows=[]; all_predictions=[]; regret_rows=[]; torch.set_num_threads(1)
    for sd in sorted(benchmark.glob('seed_*')):
        models=load_models(sd,cases)
        predictions=np.mean([m.predict_many([(c,a) for c in cases for a in exposures]) for m in models],axis=0).reshape(truth.shape)
        all_predictions.append(predictions)
        for i,c in enumerate(cases):
            _,beta,bz,sp=credits(predictions[i],z)
            pair_cols=[effects.index(p) for p in admitted_pairs(c,pool)]
            actual=np.abs(oracle_beta[i,pair_cols])>=protocol['pair_threshold']; predicted=np.abs(beta[pair_cols])>=protocol['pair_threshold']
            tp=int(np.sum(actual & predicted)); denom=int(actual.sum()+predicted.sum())
            model_rows.append({'seed':int(sd.name.split('_')[1]),'stratum':c.stratum,'depth':len(c.nodes),
                               'utility_mae':float(np.mean(np.abs(predictions[i]-truth[i]))),
                               'banzhaf_mae':float(np.mean(np.abs(bz-oracle_b[i]))),
                               'shapley_mae':float(np.mean(np.abs(sp-oracle_s[i]))),
                               'pair_f1':None if denom==0 else 2*tp/denom,'pair_true_positive':tp,
                               'pair_truth_positive':int(actual.sum()),'pair_predicted_positive':int(predicted.sum())})
        evidence=[ProbeEvidence(**(r | {'level':Level(r['level'])})) for r in json.loads((sd/'transfer_levels.json').read_text())]
        gate=TransportGate(evidence)
        frozen=json.loads((sd/'frozen_banks.json').read_text())
        def bank_value(bank):
            per_stratum={}
            for i,c in enumerate(cases):
                eligible=tuple(m.uid for m in gate.candidates(c,[m for m in pool if m.uid in bank]))
                a=retriever(4)(c,eligible)
                mask=sum(1<<ids.index(m) for m in a)
                per_stratum.setdefault(c.stratum,[]).append(truth[i,mask])
            means=[np.mean(per_stratum[k]) for k in sorted(per_stratum)]
            return worst_mixture(means,[.125]*8,.2)-.0001*bank_cost(bank,pool)
        feasible=[a for a in exposures if bank_cost(a,pool)<=80]
        utilities=[bank_value(b) for b in feasible]
        optimum=feasible[int(np.argmax(utilities))]
        for method in ('CREDIT','Direct-Sandbox','Factorial-Banzhaf','Model-MCShapley'):
            actual=bank_value(frozen[method])
            regret_rows.append({'seed':int(sd.name.split('_')[1]),'method':method,'oracle_optimal_bank':optimum,
                                'oracle_robust_score':max(utilities),'chosen_robust_score':actual,
                                'reference_domain_regret':max(utilities)-actual})
    np.savez_compressed(output/'model_predictions.npz', predictions=np.asarray(all_predictions))
    write(output/'model_errors.json',model_rows);write(output/'design_statistics.json',design_rows)
    write(output/'reference_bank_regret.json',regret_rows)
    write(output/'summary.json',{'reference_cases':len(cases),'coalitions_per_case':256,
          'actual_reference_episodes':int(observed.size),'reference_api_calls':int(calls.sum()),
          'negative_control_episodes':controls,'negative_controls_passed':True,
          'negative_control_api_calls':control_calls,
          'contract_labels':labels,'nonambiguous_labels_checked_against_private_support':checked,
          'label_mismatches':0,'mc_utility_mae':float(np.mean(np.abs(observed.mean(axis=1)-truth))),
          'scope':'exact declared-kernel oracle and finite actual local-sandbox execution; not real-provider truth',
          'oracle_excluded_from_main_selection':True,'benchmark_manifest_sha256':digest(benchmark/'artifact_manifest.json')})
    write(output/'artifact_manifest.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('datasets/credit_v3_reference_v1'))
    p.add_argument('--benchmark',type=Path,default=Path('results/credit_v3_benchmark_v1'))
    p.add_argument('--output',type=Path,default=Path('results/credit_v3_reference_v1'))
    p.add_argument('--freeze-only',action='store_true')
    args=p.parse_args()
    if args.freeze_only: freeze(args.dataset)
    else: run(args.dataset,args.benchmark,args.output)
