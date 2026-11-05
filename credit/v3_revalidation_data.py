"""Append balanced, independent target intervention contexts to a fresh corpus."""
import argparse
from dataclasses import asdict,replace
import gzip
import json
from pathlib import Path

from .stateful import FAMILIES
from .v3.local import context
from .v3.experiment import digest,write
from .v3_optimized_data import generate as base_generate
from .v3_revalidation import candidate_memories

SEEDS = (1103,1109,1117,1123,1129,1151,1153,1163)


def generate(output,seeds=SEEDS):
    base_generate(output,seeds)
    templates = ('future','future_auth_repeat','future_ready_submit')
    for seed in seeds:
        with gzip.open(output/f'seed_{seed}.jsonl.gz','at',compresslevel=6) as stream:
            for family in FAMILIES:
                for i in range(24):
                    c = context(family,'target','target-revalidation',i,seed,template=templates[i%3])
                    c = replace(c,uid=f'revalidation-{seed}:{c.uid}')
                    stream.write(json.dumps({'phase':'target-revalidation','context':asdict(c)},separators=(',',':'))+'\n')
    p = json.loads((output/'generation_protocol.json').read_text())
    p.update(target_revalidation={'per_family':24,'per_public_template':8,'versions':'target-only',
        'phase':'target-revalidation','independent_ids_and_noise':True,'outcomes_preassigned':False},
        topology_limit='all three deployment templates appear in target adaptation; no unseen-topology claim')
    p['source_sha256'][str(Path(__file__).relative_to(Path.cwd()))] = digest(__file__)
    p['source_sha256']['credit/v3_revalidation.py'] = digest('credit/v3_revalidation.py')
    write(output/'generation_protocol.json',p)
    write(output/'candidate_memories.json',[asdict(m)|{'scope':sorted(m.scope)} for m in candidate_memories()])
    s = json.loads((output/'summary.json').read_text())
    s['contexts'] += len(seeds)*4*24
    s['by_phase']['target-revalidation'] = len(seeds)*4*24
    write(output/'summary.json',s)
    write(output/'manifest.json',{p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})
    print('Target adaptation appended:',s['contexts'],'total contexts',flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args();generate(a.output)
