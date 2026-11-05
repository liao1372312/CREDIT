"""Frozen independent chronological blocks, with balanced target design frames."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip,json
from pathlib import Path

from .v3.experiment import digest,write
from .v3.local import context,factory
from .v3_optimized_data import generate as base_generate
from .v3_revalidation import candidate_memories

SEEDS=(1201,1213,1217,1223,1229,1231,1237,1249)
CYCLES=3
TEST_N=128


def generate(output):
    base_generate(output,SEEDS)
    phases={'screening-unlabeled':32,'direct-screening':16,'halving-0':512,'halving-1':512,
            'pilot':16,'certification':200,'certification-unlabeled':1000,'heldout-test':TEST_N,
            'archive':12}
    added=0;added_phases=Counter()
    for seed in SEEDS:
        make=factory(seed)
        with gzip.open(output/f'seed_{seed}.jsonl.gz','at',compresslevel=6) as stream:
            for cycle in range(CYCLES):
                for family in ('books','travel','inventory','payments'):
                    for epoch in ('source','target'):
                        for phase,n in phases.items():
                            name=f'cycle-{cycle}/{phase}'
                            for i in range(n):
                                c=make(family+'/'+epoch,name,i)
                                stream.write(json.dumps({'phase':name,'context':asdict(c)},separators=(',',':'))+'\n');added+=1;added_phases[name]+=1
                    for i in range(96):
                        name=f'cycle-{cycle}/target-adaptation'
                        c=context(family,'target',name,i,seed,
                            template=('future','future_auth_repeat','future_ready_submit')[i%3])
                        stream.write(json.dumps({'phase':name,'context':asdict(c)},separators=(',',':'))+'\n');added+=1;added_phases[name]+=1
    p=json.loads((output/'generation_protocol.json').read_text())
    p.update(chronological_cycles=CYCLES,test_per_stratum_cycle=TEST_N,
             target_contexts_per_family_cycle=96,all_three_templates_seen=True,
             split='source-only initial fitting; randomized target designs join later fits after block scoring; source validation/calibration',
             topology_limit='all three deployment templates appear in target adaptation before scoring; no unseen-topology claim',
             topology_claim='all deployment templates are seen in target adaptation; initial source layouts do not imply an online unseen-topology evaluation',
             source_sha256=p['source_sha256']|{str(Path(__file__).resolve()):digest(__file__)})
    write(output/'generation_protocol.json',p)
    write(output/'candidate_memories.json',[asdict(m)|{'scope':sorted(m.scope)} for m in candidate_memories()])
    s=json.loads((output/'summary.json').read_text());s['contexts']+=added;s['chronological_contexts']=added
    s['by_phase'].update(added_phases)
    write(output/'summary.json',s)
    write(output/'manifest.json',{p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})
    print('Saved',s['contexts'],'unlabeled contexts',flush=True)


class CycleCorpus:
    def __init__(self,corpus,cycle):self.corpus,self.cycle=corpus,cycle
    def make(self,stratum,phase,index):return self.corpus.make(stratum,f'cycle-{self.cycle}/{phase}',index)
    def phase(self,name):return self.corpus.phase(f'cycle-{self.cycle}/{name}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();generate(a.output)
