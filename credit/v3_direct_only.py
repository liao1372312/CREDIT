"""Corrective, genuinely model-free Direct-Sandbox comparison.

The initial benchmark's direct screen still used surrogate PPI at certification.
Keep that frozen run and perform an actual zero-model comparison separately.
The correction is specified before inspecting any held-out outcome.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import time
import torch
from .v3.experiment import digest,write
from .v3.local import StatefulExecutor,memories
from .v3.drift import Level,ProbeEvidence,TransportGate
from .v3_data import Corpus,SEEDS
from .v3_benchmark import CallLedger,make_selector,learned_bank,selection_limits,stream_records,score


def run(original,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(1)
    protocol=json.loads((original/'frozen_protocol.json').read_text());dataset=Path(protocol['dataset'])
    sources=[Path(__file__),Path('credit/v3_benchmark.py'),Path('credit/v3/selection.py'),Path('credit/v3/inference.py')]
    write(output/'frozen_protocol.json',{'seeds':list(SEEDS),'method':'Direct-Sandbox-Only','models':0,
          'surrogate_weights':0,'correction':'initial Direct-Sandbox used PPI proxies at certification; original preserved',
          'specified_before_heldout_inspection':True,'original':str(original.resolve()),
          'dataset_manifest_sha256':digest(dataset/'manifest.json'),'call_allowance_per_strategy':150000,
          'source_sha256':{str(p):digest(p) for p in sources},'scope':'fresh actual selection executions; same frozen paired held-out tasks'})
    prepared=[]
    for seed in SEEDS:
        sd=original/f'seed_{seed}'
        while not (sd/'frozen_banks.json').exists():time.sleep(5)
        corpus=Corpus(dataset,seed)
        evidence=[ProbeEvidence(**(r | {'level':Level(r['level'])})) for r in json.loads((sd/'transfer_levels.json').read_text())]
        common=json.loads((sd/'history_costs.json').read_text())
        common_calls=common['total_api_calls']+common['fixtures']
        gate=TransportGate(evidence);pool=memories();ledger=CallLedger(selection_limits(),150000,common_calls)
        method='Direct-Sandbox-Only';ledger.tag=method
        directory=output/f'seed_{seed}';directory.mkdir();md=directory/method;md.mkdir()
        selector=make_selector(pool,(),gate,corpus,StatefulExecutor(),ledger)
        bank=learned_bank(method,selector,md,direct=True)
        costs=ledger.report();write(md/'costs.json',costs);stream_records(md/'selection.jsonl.gz',ledger.records)
        write(directory/'method_costs.json',{method:costs});write(directory/'frozen_banks.json',{method:bank})
        write(directory/'transfer_levels.json',[asdict(e) for e in evidence])
        prepared.append({'seed':seed,'corpus':corpus,'banks':{method:bank},'policies':{method:(gate,4)},'evidence':evidence})
        print(seed,'zero-model Direct bank frozen',bank,flush=True)
    write(output/'evaluation_freeze.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p.name in ('frozen_banks.json','switch_decision.json','certification_freeze.json')})
    rows=score(prepared,output)
    write(output/'summary.json',{'test_episodes':len(rows),'test_api_calls':sum(r['api_calls'] for r in rows),'models':0,
          'original_direct_banks_equal':all(json.loads((original/f'seed_{p["seed"]}'/'frozen_banks.json').read_text())['Direct-Sandbox']==list(p['banks']['Direct-Sandbox-Only']) for p in prepared)})
    write(output/'artifact_manifest.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,default=Path('results/credit_v3_benchmark_v1'))
    p.add_argument('--output',type=Path,default=Path('results/credit_v3_direct_only_v1'))
    a=p.parse_args();run(a.original,a.output)
