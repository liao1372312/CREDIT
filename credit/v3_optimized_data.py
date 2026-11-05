"""Fresh frozen tasks for the state-feature optimization, with no outcome labels."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path

from .stateful import FAMILIES
from .v3.experiment import digest, write
from .v3.local import context, factory, memories, ROLES, TEMPLATES

SEEDS = (811, 821, 823, 827, 829, 839, 853, 857)
PHASES = {
    'training': 32, 'validation': 16, 'calibration': 16,
    'legacy-training': 8, 'legacy-calibration': 8,
    'screening-unlabeled': 32, 'direct-screening': 16,
    **{f'halving-{i}': 512 for i in range(3)},
    'pilot': 100, 'certification': 200, 'certification-unlabeled': 1000,
    'heldout-test': 256,
}
# Initial-state coverage is declared without consulting execution outcomes.
# Only the six-step layout is genuinely absent from all fitting splits.
TRAIN_LAYOUTS = (
    ('train', 0), ('alternate', 0), ('future_ready_submit', 7), ('train', 1),
    ('alternate', 3), ('train', 7), ('alternate', 1), ('future_ready_submit', 7),
)


def generate(output, seeds=SEEDS):
    if output.exists():
        raise FileExistsError('use a fresh dataset directory')
    output.mkdir(parents=True)
    sources = [Path(__file__), Path('credit/v3/local.py'), Path('credit/stateful.py')]
    write(output/'generation_protocol.json', {
        'seeds': list(seeds), 'phases': PHASES, 'train_layouts': TRAIN_LAYOUTS,
        'synthetic': True, 'outcomes_preassigned': False,
        'actor_and_api': 'unchanged credit.stateful and credit.v3.local.StatefulExecutor',
        'split': 'fresh phase UIDs and independent execution seeds; source-only fitting',
        'test_distribution': 'same three layout families as historical main evaluation, new seeds/tasks',
        'topology_claim': 'six-step test layout unseen in training, optimizer validation and calibration',
        'source_sha256': {str(p): digest(p) for p in sources}})
    write(output/'registry.json', {'roles': {k: asdict(r) for k, r in ROLES.items()},
                                  'templates': TEMPLATES})
    write(output/'memories.json', [asdict(m) | {'scope': sorted(m.scope)} for m in memories()])
    counts = Counter()
    for seed in seeds:
        make = factory(seed)
        with gzip.open(output/f'seed_{seed}.jsonl.gz', 'wt', compresslevel=6) as stream:
            for phase, count in PHASES.items():
                fitting = phase in ('training', 'validation', 'calibration',
                                    'legacy-training', 'legacy-calibration')
                strata = [f'{f}/source' for f in FAMILIES] if fitting else [
                    f'{f}/{epoch}' for f in FAMILIES for epoch in ('source', 'target')]
                for stratum in strata:
                    family, epoch = stratum.split('/')
                    for i in range(count):
                        if phase.startswith('legacy-'):
                            c = context(family, epoch, phase, i, seed,
                                        template='train' if i % 2 else 'alternate')
                        elif fitting:
                            template, initial_state = TRAIN_LAYOUTS[i % len(TRAIN_LAYOUTS)]
                            c = replace(context(family, epoch, phase, i, seed, template=template),
                                        initial_state=initial_state)
                        else:
                            c = make(stratum, phase, i)
                        c = replace(c, uid=f'optimized-{seed}:{c.uid}')
                        stream.write(json.dumps({'phase': phase, 'context': asdict(c)},
                                                separators=(',', ':'))+'\n')
                        counts[phase] += 1
    write(output/'summary.json', {'contexts': sum(counts.values()), 'by_phase': dict(counts)})
    write(output/'manifest.json', {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(f'Frozen {sum(counts.values())} contexts for {len(seeds)} seeds', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    a = p.parse_args()
    generate(a.output, a.seeds)
