"""Development-only exhaustive bank ceiling for the frozen finite LLM actor."""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from .llm_policy import install_policy
from .v3.experiment import digest,write
from .v3.inference import worst_mixture
from .v3.local import factory,retriever
from .v3.selection import bank_cost
from .v3_revalidation import RevalidationExecutor,candidate_memories
from .v3_remote_llm_experiment import policy_guard

DEV_SEEDS=(2909,2917)
CONFIRM_SEEDS=(3109,3119,3121,3137,3163,3167,3181,3191)
STRATA=tuple(f'{family}/{epoch}' for family in ('books','travel','inventory','payments')
             for epoch in ('source','target'))
TASKS_PER_STRATUM=64
CAPACITY=80


def run(policy,output):
    if output.exists():raise FileExistsError('fresh development directory required')
    manifest,_=policy_guard(policy)
    pool=candidate_memories()
    ids=tuple(m.uid for m in pool)
    banks=[bank for n in range(len(ids)+1) for bank in combinations(ids,n)
           if bank_cost(bank,pool)<=CAPACITY]
    output.mkdir(parents=True)
    write(output/'frozen_protocol.json',{
        'scope':'development oracle upper bound for finite actor; no gate, fitting, or certification',
        'development_seeds':DEV_SEEDS,'unused_confirmation_seeds':CONFIRM_SEEDS,
        'strata':STRATA,'tasks_per_stratum':TASKS_PER_STRATUM,
        'task_phase':'remote-opt-development','bank_capacity':CAPACITY,
        'retrieval_depth':4,'candidate_ids':ids,'candidate_banks':len(banks),
        'actor_scores_sha256':manifest['decision_scores_sha256'],
        'source_sha256':{name:digest(name) for name in (
            'credit/v3_remote_opt_diagnostic.py','credit/llm_policy.py',
            'credit/v3/local.py','credit/v3_revalidation.py','credit/stateful.py')},
        'planned_comparison':'best possible fixed bank versus new_contract-only Direct on the development tasks',
        'interpretation':'oracle uses development outcomes and is not a deployable CREDIT selector'})
    install_policy(policy/'decision_scores.json')
    make={seed:factory(seed) for seed in DEV_SEEDS}
    tasks=[(seed,stratum,make[seed](stratum,'remote-opt-development',i))
           for seed in DEV_SEEDS for stratum in STRATA for i in range(TASKS_PER_STRATUM)]
    execute=RevalidationExecutor();retrieve=retriever(4)
    successes=defaultdict(lambda:defaultdict(int))
    calls=defaultdict(int)
    unique_executions=0
    for seed,stratum,task in tasks:
        cache={}
        for bank in banks:
            exposure=retrieve(task,bank)
            if exposure not in cache:
                episode=execute.run(task,exposure)
                cache[exposure]=(int(episode.verified),episode.calls)
                unique_executions+=1
            success,api_calls=cache[exposure]
            successes[bank][stratum]+=success
            calls[bank]+=api_calls
    rows=[]
    for bank in banks:
        means=[successes[bank][s]/(len(DEV_SEEDS)*TASKS_PER_STRATUM) for s in STRATA]
        vsr=sum(means)/len(means)
        robust=worst_mixture(means,[1/len(STRATA)]*len(STRATA),.2)
        rows.append({'bank':bank,'cost':bank_cost(bank,pool),'vsr':vsr,
                     'robust_value':robust-.0001*bank_cost(bank,pool),
                     'stratum_successes':dict(successes[bank]),'physical_calls_if_uncached':calls[bank]})
    rows.sort(key=lambda r:(-r['vsr'],r['cost'],r['bank']))
    direct=next(r for r in rows if r['bank']==('new_contract',))
    summary={'tested_banks':len(rows),'tasks':len(tasks),'unique_executions':unique_executions,
             'best_by_vsr':rows[0],'best_by_robust_value':max(rows,key=lambda r:r['robust_value']),
             'direct_new_contract':direct,
             'number_strictly_better_than_direct':sum(r['vsr']>direct['vsr'] for r in rows),
             'number_equal_to_direct':sum(r['vsr']==direct['vsr'] for r in rows),
             'claim_limit':'development oracle, same finite actor and synthetic API; confirmation seeds untouched'}
    write(output/'bank_scores.json',rows)
    write(output/'summary.json',summary)
    write(output/'artifact_manifest.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    print({'tested_banks':len(rows),'tasks':len(tasks),'unique_executions':unique_executions,
           'best_vsr':rows[0]['vsr'],'direct_vsr':direct['vsr']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.policy,args.output)
