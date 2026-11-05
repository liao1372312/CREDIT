"""Pre-test compact-data control for the state-conditioned CREDIT revision.

Uses the first 16 source training and first 8 validation/calibration contexts
per family from the frozen corpus. No held-out result determines this subset.
"""
import argparse
from pathlib import Path
import json

import torch

from .paired_statistics import summarize
from .v3.experiment import digest,write
from .v3_benchmark import score
from .v3_data import Corpus
from .v3_optimized_data import SEEDS
from .v3_optimized_model import StateAwareOutcomeModel,StateAwareNoPairs
from .v3_optimization import history,ensemble,select


class LimitedCorpus:
    def __init__(self,corpus):
        self.corpus = corpus

    def phase(self,name):
        limit = {'training':16,'validation':8,'calibration':8}.get(name)
        rows = self.corpus.phase(name)
        return rows if limit is None else [c for c in rows if int(c.uid.rsplit(':',1)[1]) < limit]

    def make(self,*args):
        return self.corpus.make(*args)


def prepare(seed,corpus,output):
    sd = output/f'seed_{seed}'
    sd.mkdir()
    h = history(LimitedCorpus(corpus),seed,sd/'compact_history')
    models = ensemble(StateAwareOutcomeModel,*h[:3],seed,sd/'models_state')
    no_pairs = ensemble(StateAwareNoPairs,*h[:3],seed,sd/'models_no_pairs')
    gate,evidence,common = h[3:]
    hc = json.loads((sd/'compact_history'/'costs.json').read_text())
    direct_common = hc['api_calls']['factorial']+hc['api_calls']['replay_audit']+hc['fixtures']
    banks,costs,policies = {},{},{}
    specs = [
        ('CREDIT-Compact-K4',models,common,4,False),
        ('CREDIT-Compact-K6',models,common,6,False),
        ('CREDIT-Compact-NoPairs-K6',no_pairs,common,6,False),
        ('Direct-Compact-K4',(),direct_common,4,True),
        ('Direct-Compact-K6',(),direct_common,6,True)]
    for name,active,history_calls,depth,direct in specs:
        print(seed,name,'selection',flush=True)
        banks[name],costs[name] = select(name,active,gate,corpus,history_calls,sd/name,
                                        depth=depth,direct=direct,cap=150000)
        policies[name] = gate,depth
    write(sd/'frozen_banks.json',{k:list(v) for k,v in banks.items()})
    write(sd/'method_costs.json',costs)
    return {'seed':seed,'corpus':corpus,'banks':banks,'policies':policies,
            'evidence':evidence,'directory':sd}


def run(dataset,output,seeds=SEEDS,main=Path('results/credit_v3_optimization_v1')):
    if output.exists():
        raise FileExistsError('use a fresh output directory')
    # An explicit record distinguishes pre-test supplemental design from a
    # later analysis of measured test results. No main raw outcomes are read.
    before_test = not (main/'raw_test_results.json').exists() and not (main/'evaluation_freeze.json').exists()
    if not before_test:
        raise ValueError('this supplemental configuration must freeze before main test scoring begins')
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    (output/'source').mkdir()
    sources = [Path(__file__),Path('credit/v3_optimization.py'),Path('credit/v3_optimized_model.py'),
               Path('credit/v3_optimized_data.py'),Path('credit/v3_benchmark.py'),Path('credit/v3_data.py'),
               Path('credit/paired_statistics.py'),*Path('credit/v3').glob('*.py'),Path('credit/stateful.py')]
    write(output/'frozen_protocol.json',{
        'seeds':list(seeds),'dataset':str(dataset.resolve()),
        'dataset_manifest_sha256':digest(dataset/'manifest.json'),
        'source_sha256':{str(p):digest(p) for p in sources},
        'paper_sha256':digest('CREDIT_WWW2027_v3.tex'),'agents_sha256':digest('AGENTS.md'),
        'specified_before_main_test_scoring':before_test,'main_run':str(main.resolve()),
        'basis':'observed training/selection expense only; no main held-out outcomes inspected',
        'fitting':'64 source train contexts (1 draw),32 optimizer-validation and32 calibration contexts',
        'subsetting':'first 16 training and first 8 validation/calibration indices per family, outcome-independent',
        'comparisons':'same-K Direct; compact State-K6 vs compact NoPairs-K6; full/compact comparison is descriptive',
        'call_allowance':150000,'capacity':80,'halving_gamma':1024,'certification_n':200,
        'direct':'zero model, pilot_n=2, lambda=0; no unused model validation/calibration charge',
        'scope':'fixed synthetic actor/API; all 8 seeds/banks frozen before supplemental held-out scoring'})
    for p in sources:
        (output/'source'/p.name).write_bytes(p.read_bytes())
    prepared = [prepare(s,Corpus(dataset,s),output) for s in seeds]
    write(output/'evaluation_freeze.json',{str(p.relative_to(output)):digest(p)
          for p in output.rglob('*') if p.is_file() and p.name in
          ('frozen_banks.json','switch_decision.json','certification_freeze.json')})
    rows = score(prepared,output)
    table,paired = summarize(rows,'CREDIT-Compact-K4',repeats=4000,random_seed=4099)
    write(output/'statistics.json',{'main':table,'paired':paired})
    table,paired = summarize([r for r in rows if r['method'] in
                             ('CREDIT-Compact-K6','CREDIT-Compact-NoPairs-K6','Direct-Compact-K6')],
                             'CREDIT-Compact-K6',repeats=4000,random_seed=4099)
    write(output/'k6_statistics.json',{'main':table,'paired':paired})
    write(output/'summary.json',{'test_episodes':len(rows),'test_api_calls':sum(r['api_calls'] for r in rows),
          'independent_seeds':len(seeds),'seed_overlap_with_main':True,'no_new_independent_seeds_claimed':True})
    write(output/'artifact_manifest.json',{str(p.relative_to(output)):digest(p)
          for p in output.rglob('*') if p.is_file()})
    print('Completed compact state-model control',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    run(a.dataset,a.output)
