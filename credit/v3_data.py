"""Generate and save frozen synthetic CREDIT v3 tasks (no fabricated real labels)."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from .stateful import FAMILIES
from .v3.local import ROLES, TEMPLATES, context, factory, memories
from .v3.experiment import digest, write
from .v3.audit import load_context

SEEDS = (701, 709, 719, 727, 739)
PHASE_COUNTS = {'training': 32, 'calibration': 32, 'screening-unlabeled': 32,
                **{f'halving-{i}': 512 for i in range(6)},
                'pilot': 100, 'certification': 200, 'certification-unlabeled': 1000,
                'heldout-test': 256, 'reference': 16}


def generate(output, seeds=SEEDS):
    if output.exists(): raise FileExistsError('use a new dataset directory')
    output.mkdir(parents=True)
    registry = {'roles': {k: asdict(v) for k, v in ROLES.items()}, 'templates': TEMPLATES}
    write(output/'registry.json', registry)
    write(output/'memories.json', [asdict(m) | {'scope': sorted(m.scope)} for m in memories()])
    sources = [Path(__file__), Path('credit/v3/local.py'), Path('credit/stateful.py')]
    write(output/'generation_protocol.json', {'seeds': list(seeds), 'phase_counts': PHASE_COUNTS,
        'strata': [f'{f}/{e}' for f in FAMILIES for e in ('source', 'target')],
        'synthetic': True, 'natural_drift': False, 'outcomes_preassigned': False,
        'mechanism': 'unchanged fixed stateful actor/API simulator, all-required-node verifier, one agent repair',
        'generation': 'topologies and execution seeds sampled independently of any model/outcome',
        'training_graphs': ['train', 'alternate'], 'test_graphs': ['future', 'future_auth_repeat', 'future_ready_submit'],
        'source_sha256': {str(p): digest(p) for p in sources},
        'dataset_gap': 'Provided EMP/HSC/API files lack restorable state, full intervention traces and business verification'})
    counts = Counter()
    for seed in seeds:
        make = factory(seed)
        with gzip.open(output/f'seed_{seed}.jsonl.gz', 'wt', encoding='utf8', compresslevel=6) as stream:
            for phase, n in PHASE_COUNTS.items():
                strata = [f'{f}/source' for f in FAMILIES] if phase in ('training', 'calibration') else [f'{f}/{e}' for f in FAMILIES for e in ('source', 'target')]
                for k in strata:
                    for i in range(n):
                        if phase in ('training', 'calibration'):
                            family, epoch = k.split('/')
                            template = 'train' if i % 2 == int(phase == 'training') else 'alternate'
                            c = context(family, epoch, phase, i, seed, template=template)
                        else:
                            c = make(k, phase, i)
                        # Save full executable context, rather than only a generator seed.
                        d = asdict(c)
                        d['uid'] = f'synthetic-{seed}:{c.uid}'
                        stream.write(json.dumps({'phase': phase, 'context': d}, separators=(',', ':'))+'\n')
                        counts[phase] += 1
    write(output/'summary.json', {'contexts': sum(counts.values()), 'by_phase': dict(counts),
                                 'seeds': list(seeds), 'synthetic': True, 'labels': 'collected by execution in results, not generated as winners'})
    write(output/'manifest.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'directory': str(output), 'contexts': sum(counts.values()), 'seeds': list(seeds)}))


class Corpus:
    def __init__(self, directory, seed):
        self.directory, self.seed = Path(directory), seed
        manifest = json.loads((self.directory/'manifest.json').read_text())
        filename = f'seed_{seed}.jsonl.gz'
        if digest(self.directory/filename) != manifest[filename]: raise ValueError('dataset hash differs')
        self.records = {}
        with gzip.open(self.directory/filename, 'rt', encoding='utf8') as stream:
            counters = Counter()
            for line in stream:
                d = json.loads(line); c = load_context(d['context'])
                key = d['phase'], c.stratum
                self.records[key + (counters[key],)] = c
                counters[key] += 1
        ids = [c.uid for c in self.records.values()]
        noises = [c.seed for c in self.records.values()]
        if len(set(ids)) != len(ids) or len(set(noises)) != len(noises):
            raise ValueError('duplicate context or execution seed across splits')

    def make(self, stratum, phase, index):
        try: return self.records[phase, stratum, index]
        except KeyError as e: raise ValueError(f'unfrozen context requested: {phase}/{stratum}/{index}') from e

    def phase(self, name):
        return [c for (phase, k, i), c in sorted(self.records.items()) if phase == name]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('datasets/credit_v3_synthetic_v1'))
    args = p.parse_args(); generate(args.output)

if __name__ == '__main__': main()
