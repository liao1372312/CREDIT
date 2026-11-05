"""Interpret the gate-free development bank enumeration by public stratum."""
import argparse
from pathlib import Path

from .v3.experiment import digest,write
from .v3_remote_llm_experiment import read


def analyze(source,output):
    if output.exists():raise FileExistsError('fresh analysis directory required')
    for name,h in read(source/'artifact_manifest.json').items():
        if digest(source/name)!=h:raise ValueError('development oracle artifact changed')
    rows=read(source/'bank_scores.json')
    protocol=read(source/'frozen_protocol.json')
    bybank={tuple(r['bank']):r for r in rows}
    denom=len(protocol['development_seeds'])*protocol['tasks_per_stratum']*len(protocol['strata'])
    cells=[]
    for stratum in protocol['strata']:
        best=max(rows,key=lambda r:(r['stratum_successes'][stratum],-r['cost']))
        direct=bybank[('new_contract',)] if stratum in ('books/target','inventory/target') else bybank[()]
        cells.append({'stratum':stratum,'oracle_successes':best['stratum_successes'][stratum],
                      'oracle_bank':best['bank'],'contract_gate_successes':direct['stratum_successes'][stratum],
                      'contract_gate_bank':direct['bank']})
    output.mkdir()
    write(output/'conditional_oracle.json',{
        'development_only':True,'gate_free_fixed_bank_ceiling_is_not_the_baseline':True,
        'tested_banks':len(rows),'tasks':denom,'cells':cells,
        'conditional_oracle_vsr':sum(x['oracle_successes'] for x in cells)/denom,
        'contract_gate_vsr':sum(x['contract_gate_successes'] for x in cells)/denom,
        'interpretation':'one bank per public stratum is an optimistic oracle; the contract gate matches it on these development tasks',
        'source_manifest_sha256':digest(source/'artifact_manifest.json')})
    write(output/'artifact_manifest.json',{'conditional_oracle.json':digest(output/'conditional_oracle.json'),
                                           'generator_sha256':digest(__file__)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.source,a.output)
