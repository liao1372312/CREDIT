"""Prospective paired-admission cost study for the frozen finite remote actor.

This is a specialized one-candidate CREDIT variant, not the general v3 method.
It uses only randomized target adaptation episodes to decide which public
family/version domains may expose the novel contract memory. Test episodes are
never used in admission or design selection.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from scipy.stats import binomtest

from .llm_policy import install_policy
from .v3.experiment import digest, write
from .v3.local import context, factory, retriever
from .v3_revalidation import RevalidationExecutor, candidate_memories
from .v3_remote_llm_experiment import policy_guard
from .v3_remote_opt_diagnostic import DEV_SEEDS, CONFIRM_SEEDS, STRATA

FAMILIES = ('books', 'travel', 'inventory', 'payments')
CYCLES = 3
TEST_N = 128
ALPHA = 0.05 / len(FAMILIES)
DESIGNS = (24, 96)


def adaptation(seed, cycle, n, executor):
    """Bonferroni paired sign test on disjoint adaptation contexts."""
    decisions = {}
    calls = 0
    for family in FAMILIES:
        gain = loss = ties = 0
        for i in range(n):
            task = context(family, 'target', f'focused-{cycle}/adaptation', i, seed,
                           template=('future', 'future_auth_repeat', 'future_ready_submit')[i % 3])
            empty = executor.run(task, ())
            exposed = executor.run(task, retriever(4)(task, ('new_contract',)))
            calls += empty.calls + exposed.calls
            gain += int(bool(exposed.verified) and not bool(empty.verified))
            loss += int(bool(empty.verified) and not bool(exposed.verified))
            ties += int(bool(empty.verified) == bool(exposed.verified))
        p = binomtest(gain, gain + loss, 0.5, alternative='greater').pvalue if gain + loss else 1.0
        decisions[family] = {'gain': gain, 'loss': loss, 'ties': ties, 'p_one_sided': p,
                             'admit': bool(p < ALPHA and gain > loss), 'contexts': n}
    return decisions, calls


def score(seed, cycle, decisions, executor):
    make = factory(seed)
    retrieve = retriever(4)
    rows = []
    for stratum in STRATA:
        family, epoch = stratum.split('/')
        focused = epoch == 'target' and decisions[family]['admit']
        public_gate = epoch == 'target' and family in ('books', 'inventory')
        for i in range(TEST_N):
            task = make(stratum, f'focused-{cycle}/heldout-test', i)
            exposure = retrieve(task, ('new_contract',))
            empty_episode = executor.run(task, ())
            exposed_episode = executor.run(task, exposure) if focused or public_gate else None
            for method, active in (('NoMem', False), ('CREDIT-Focused', focused),
                                   ('Public-Contract-Gate', public_gate)):
                episode = exposed_episode if active else empty_episode
                rows.append({'seed': seed, 'cycle': cycle, 'stratum': stratum,
                             'task': task.uid, 'method': method, 'success': bool(episode.verified),
                             'api_calls': episode.calls, 'exposure': list(episode.exposure),
                             'state_hash': episode.state_hash})
    return rows


def aggregate(rows, adaptation_calls):
    result = {}
    for method in ('NoMem', 'CREDIT-Focused', 'Public-Contract-Gate'):
        selected = [r for r in rows if r['method'] == method]
        success = sum(r['success'] for r in selected)
        result[method] = {'tasks': len(selected), 'successes': success,
                          'vsr': success / len(selected),
                          'test_api_calls': sum(r['api_calls'] for r in selected),
                          'extra_adaptation_calls': adaptation_calls if method == 'CREDIT-Focused' else 0}
    paired = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        paired[row['seed'], row['cycle']][row['stratum'], row['task']][row['method']] = row
    differences = 0
    for block in paired.values():
        for task in block.values():
            if task['CREDIT-Focused']['success'] != task['Public-Contract-Gate']['success']:
                differences += 1
    result['focused_vs_public_gate_different_tasks'] = differences
    return result


def run(policy, output, split, design=None):
    if output.exists():
        raise FileExistsError('use a fresh focused experiment directory')
    manifest, _ = policy_guard(policy)
    if split == 'development':
        seeds, ns = DEV_SEEDS, DESIGNS
    else:
        if design is None:
            raise ValueError('confirmation requires a frozen development design')
        frozen = json.loads(design.read_text())
        if frozen['chosen_contexts'] not in DESIGNS:
            raise ValueError('invalid frozen design')
        if digest(frozen['development_summary_path']) != frozen['development_summary_sha256']:
            raise ValueError('development summary changed')
        ns, seeds = (frozen['chosen_contexts'],), CONFIRM_SEEDS
    output.mkdir(parents=True)
    protocol = {'scope': 'finite actor paired-admission cost optimization; not general CREDIT validation',
                'split': split, 'seeds': seeds, 'all_unused_confirmation_seeds': CONFIRM_SEEDS,
                'cycles': CYCLES, 'test_n_per_stratum': TEST_N,
                'adaptation_contexts': ns, 'familywise_alpha': 0.05,
                'family_one_sided_alpha': ALPHA,
                'candidate': 'new_contract', 'candidate_pool_sha256': sha256(json.dumps(
                    [(m.uid, m.content, sorted(m.scope or ())) for m in candidate_memories()],
                    sort_keys=True).encode()).hexdigest(),
                'actor_scores_sha256': manifest['decision_scores_sha256'],
                'design_sha256': digest(design) if design else None,
                'generator_sha256': digest(__file__),
                'task_policy': 'adaptation and test have distinct deterministic phase names; test after admission',
                'comparison': 'NoMem and optimistic public-contract gate; neither incurs focused adaptation cost'}
    write(output / 'frozen_protocol.json', protocol)
    install_policy(policy / 'decision_scores.json')
    executor = RevalidationExecutor()
    all_summaries = {}
    for n in ns:
        rows, schedules, total_calls = [], [], 0
        for seed in seeds:
            for cycle in range(CYCLES):
                decisions, calls = adaptation(seed, cycle, n, executor)
                total_calls += calls
                schedules.append({'seed': seed, 'cycle': cycle, 'decisions': decisions,
                                  'adaptation_api_calls': calls})
                rows.extend(score(seed, cycle, decisions, executor))
                print(json.dumps({'split': split, 'contexts': n, 'seed': seed,
                                  'cycle': cycle, 'adaptation_calls': calls}), flush=True)
        write(output / f'admission_{n}.json', schedules)
        write(output / f'test_{n}.json', rows)
        all_summaries[str(n)] = aggregate(rows, total_calls)
    write(output / 'summary.json', all_summaries)
    write(output / 'artifact_manifest.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})


def choose(development, output):
    if output.exists():
        raise FileExistsError('fresh design path required')
    for name, expected in json.loads((development / 'artifact_manifest.json').read_text()).items():
        if digest(development / name) != expected:
            raise ValueError('development artifact changed')
    summary = json.loads((development / 'summary.json').read_text())
    schedules = {n: json.loads((development / f'admission_{n}.json').read_text()) for n in DESIGNS}
    def sufficient(n):
        rows = schedules[n]
        needed = all(all(row['decisions'][family]['admit'] for family in ('books', 'inventory'))
                     for row in rows)
        exact = summary[str(n)]['focused_vs_public_gate_different_tasks'] == 0
        return needed and exact
    candidates = [n for n in DESIGNS if sufficient(n)]
    selected = min(candidates) if candidates else max(DESIGNS)
    write(output, {'chosen_contexts': selected, 'selection_rule': 'minimum tested contexts with books/inventory admitted in every development block and zero public-gate test disagreements; else 96',
                   'development_summary_path': str((development / 'summary.json').resolve()),
                   'development_summary_sha256': digest(development / 'summary.json'),
                   'development_manifest_sha256': digest(development / 'artifact_manifest.json'),
                   'confirmation_seeds': CONFIRM_SEEDS, 'generator_sha256': digest(__file__)})
    print(json.dumps({'chosen_contexts': selected, 'development_sufficient': bool(candidates)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('development', 'confirmation', 'choose'), required=True)
    p.add_argument('--design', type=Path)
    p.add_argument('--development', type=Path)
    a = p.parse_args()
    if a.split == 'choose':
        choose(a.development, a.output)
    else:
        run(a.policy, a.output, a.split, a.design)
