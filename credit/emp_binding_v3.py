"""EMP v3 pilot: compatible retrieval, smaller shared mechanism and nested aliases."""

import argparse
from copy import deepcopy
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from .emp_binding import BindingEnvironment
from .emp_binding_agent import QwenJSONAgent, parse_request
from .emp_binding_pilot import MODEL, all_banks, attribution_banks, bank_name, memories_from_history, summarize
from .emp_binding_shared import score_predictions, tune_shared
from .emp_binding_tools_v3 import SYSTEM, assemble, messages, retrieve_compatible
from .user_data import digest, jsonl


class QwenToolAgent(QwenJSONAgent):
    def close(self):
        super().close()
        self.manifest.update({'binding_interface': 'sdk3', 'system_prompt': SYSTEM,
            'torch_version': self.torch.__version__, 'tool_version': 'response_relative_pointers_and_explicit_repair_v3'})
        (self.output / 'agent_manifest.json').write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2))


def run(agent, jobs, output, stage):
    active, finished = [], []
    for edge, method, notes in jobs:
        env = BindingEnvironment(edge); env.call(env.producer_request())
        active.append({'env': env, 'edge': edge, 'method': method, 'notes': notes,
            'checkpoint': env.checkpoint(), 'prefix_sha256': env.state_hash(), 'generations': []})
    for attempt in range(2):
        if not active:
            break
        outputs = agent.infer([messages(j['env'], j['notes']) for j in active])
        remaining = []
        for job, generation in zip(active, outputs):
            action, parse_error = parse_request(generation['agent_output'])
            request, sdk_errors = assemble(job['env'], action)
            event = job['env'].call(request)
            stored = job['env'].history[-1]
            if sdk_errors:
                stored['response']['sdk_errors'] = sdk_errors
            if parse_error:
                stored['response']['parse_error'] = parse_error
            stored['agent_output'] = generation['agent_output']; stored['state_after'] = job['env'].state_hash()
            job['generations'].append({'prompt_sha256': generation['prompt_sha256'], 'parse_error': parse_error,
                'sdk_errors': sdk_errors, 'hit_generation_limit': generation['hit_generation_limit']})
            if event['accepted'] or attempt == 1:
                finished.append(job)
            else:
                remaining.append(job)
        active = remaining
    result = []
    for job in finished:
        env, edge = job['env'], job['edge']; first = env.history[1]
        row = {'stage': stage, 'edge_uid': edge['uid'], 'task_uid': edge['task_uid'],
            'workflow_group': edge['workflow_group'], 'binding_group': edge['binding_group'],
            'bank_uid': job['method'], 'binding_interface': 'sdk3', 'exposure': [n['uid'] for n in job['notes']],
            'checkpoint': job['checkpoint'], 'prefix_sha256': job['prefix_sha256'], 'events': env.history,
            'generations': job['generations'], 'first_category': 'accepted' if first['accepted'] else 'malformed' if job['generations'][0]['parse_error'] else 'contract_error',
            'first_try_success': first['accepted'], 'final': env.verify(), 'consumer_calls': len(env.history) - 1,
            'binding_error_calls': sum(any('binding_' in e or 'missing_binding' in e for e in event['errors']) for event in env.history[1:]),
            'business_goal_success': None}
        with (output / 'trials.jsonl').open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        result.append(row)
    return result


def select(model, future, banks):
    common = [edge for edge in future if model.predict(edge, [])['supported']]
    scores = []
    for bank in banks:
        cells, missing = defaultdict(list), 0
        for edge in common:
            pred = model.predict(edge, retrieve_compatible(edge, bank))
            if pred['final_success'] is None:
                missing += 1; continue
            family = tuple(sorted(b['target'].split('/')[-1].casefold() for b in edge['bindings']))
            cells[family].append(pred['final_success'] - .02 * pred['expected_consumer_calls'])
        values = [mean(v) for v in cells.values()]
        scores.append({'bank_uid': bank_name(bank),
            'predicted_objective': .8 * mean(values) + .2 * min(values) - .01 * len(bank) if values and not missing else None,
            'retained_count': len(bank), 'supported_contexts': len(common) - missing,
            'unsupported_contexts': len(future) - len(common) + missing,
            'common_scored_edge_uids': [e['uid'] for e in common] if not missing else [],
            'selection_eligible': bool(values) and not missing})
    usable = [s for s in scores if s['selection_eligible']]
    if not usable:
        return [], scores
    chosen = max(usable, key=lambda s: (s['predicted_objective'], -s['retained_count'], s['bank_uid']))
    return next(bank for bank in banks if bank_name(bank) == chosen['bank_uid']), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('results/emp_binding_pilot_v5'))
    parser.add_argument('--expanded', type=Path, default=Path('results/emp_binding_expansion_v2'))
    parser.add_argument('--output', type=Path, default=Path('results/emp_binding_v3_run1'))
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new output directory')
    baseline = json.loads((args.baseline / 'frozen_protocol.json').read_text())
    expansion = json.loads((args.expanded / 'manifest.json').read_text())
    for name, expected in expansion['canonical_sha256'].items():
        if digest(args.expanded / name) != expected:
            raise ValueError('expanded input changed')
    if digest(Path(__file__).with_name('emp_binding_expansion.py')) != expansion['adapter_sha256']:
        raise ValueError('expansion adapter changed')
    edges = {e['uid']: e for e in jsonl(args.expanded / 'edges.jsonl')}
    splits = baseline['splits']
    partitions = {stage: [edges[uid] for uid, spec in splits.items() if spec['partition'] == stage]
                  for stage in ['history_train', 'validation', 'development_evaluation']}
    additional = list(jsonl(args.expanded / 'additional_edges.jsonl'))
    require_development = partitions['history_train'] + partitions['validation'] + partitions['development_evaluation'] + additional
    if any(e['partition'] != 'development' or not e['eligible'] for e in require_development):
        raise ValueError('reserved or unknown input cannot execute')
    args.output.mkdir(parents=True)
    names = ['emp_binding_v3.py', 'emp_binding_tools_v3.py', 'emp_binding_tools_v2.py', 'emp_binding_shared.py', 'emp_binding_expansion.py',
             'emp_binding.py', 'emp_binding_agent.py', 'emp_binding_pilot.py', 'emp_binding_sdk.py', 'user_data.py']
    protocol = {'scope': 'development SDK v3; document-derived first-record contracts; no business-goal or provider VSR',
        'baseline_directory': str(args.baseline.resolve()), 'expanded_directory': str(args.expanded.resolve()),
        'baseline_protocol_sha256': digest(args.baseline / 'frozen_protocol.json'),
        'expansion_manifest_sha256': digest(args.expanded / 'manifest.json'), 'splits': splits,
        'additional_edge_uids': [e['uid'] for e in additional],
        'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in names},
        'storage_max_notes': 2, 'note_max_tokens': 512, 'retrieval_max_notes': 2,
        'retrieval': 'canonical target name plus observable source path/key compatibility; exclude stale routes',
        'tool_version': 'sdk3', 'agent': {'do_sample': False, 'max_consumer_calls': 2, 'max_new_tokens': 3072},
        'tuning': 'historical-workflow CV only; also exclude heldout-origin notes and their exposures',
        'selection': 'public original validation and new binding contexts before their outcomes; equal coverage; mean/worst and cost',
        'reserved_workflows_executed': 0, 'certificate': False}
    (args.output / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2))
    (args.output / 'source').mkdir()
    for name in names:
        (args.output / 'source' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    began = perf_counter(); agent = QwenToolAgent(args.model, args.output, args.batch_size)
    agent.binding_interface = 'sdk3'
    history = run(agent, [(e, 'empty', []) for e in partitions['history_train']], args.output, 'history')
    demonstrations = []
    for edge in partitions['history_train']:
        env = BindingEnvironment(edge); env.call(env.producer_request()); checkpoint, prefix = env.checkpoint(), env.state_hash()
        env.call(env.reference_request())
        demonstrations.append({'stage': 'assisted_history', 'edge_uid': edge['uid'], 'bank_uid': 'document_reference',
            'events': env.history, 'checkpoint': checkpoint, 'prefix_sha256': prefix, 'final': env.verify(),
            'agent_source': 'document-contract reference, not an autonomous Qwen success'})
    with (args.output / 'assisted_history.jsonl').open('w') as handle:
        for row in demonstrations:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    pool = memories_from_history(history + demonstrations, edges, agent.tokenizer)
    if not pool:
        raise ValueError('no verified training memory')
    (args.output / 'memories.json').write_text(json.dumps(pool, ensure_ascii=False, indent=2))
    banks = all_banks(pool)
    training = run(agent, [(e, bank_name(bank), retrieve_compatible(e, bank)) for e in partitions['history_train'] for bank in banks], args.output, 'intervention_train')
    model, fit_info, cv = tune_shared(training, edges, pool)
    (args.output / 'model.json').write_text(json.dumps({'weights': model.weights.tolist(), 'mean': model.mean.tolist(),
        'scale': model.scale.tolist(), 'training_profiles': model.training_profiles, 'fit': fit_info}, indent=2))
    (args.output / 'grouped_cv.json').write_text(json.dumps(cv, indent=2))
    joint, scores = select(model, partitions['validation'] + additional, banks)
    loo, marginal, attribution = attribution_banks(training, pool, banks)
    chosen = {'NoMem': [], 'FIFO': pool[:2], 'Recency': pool[-2:], 'Restricted-LOO': loo,
              'Restricted-marginal': marginal, 'CREDIT-binding-v3': joint}
    (args.output / 'frozen_banks.json').write_text(json.dumps({'banks': {k: [m['uid'] for m in v] for k, v in chosen.items()},
        'scores': scores, 'training_attribution': attribution}, indent=2))
    prediction_rows = []
    for stage, subset in [('validation', partitions['validation']), ('development_evaluation', partitions['development_evaluation']), ('additional_development', additional)]:
        methods = {bank_name(bank): bank for bank in banks} if stage == 'validation' else chosen
        for edge in subset:
            for method, bank in methods.items():
                exposure = retrieve_compatible(edge, bank)
                prediction_rows.append({'stage': stage, 'edge_uid': edge['uid'], 'bank_uid': method,
                    'exposure': [m['uid'] for m in exposure], **model.predict(edge, exposure)})
    (args.output / 'frozen_predictions.json').write_text(json.dumps(prediction_rows, indent=2))
    validation = run(agent, [(e, bank_name(b), retrieve_compatible(e, b)) for e in partitions['validation'] for b in banks], args.output, 'validation')
    evaluation = run(agent, [(e, method, retrieve_compatible(e, bank)) for e in partitions['development_evaluation'] for method, bank in chosen.items()], args.output, 'development_evaluation')
    extra = run(agent, [(e, method, retrieve_compatible(e, bank)) for e in additional for method, bank in chosen.items()], args.output, 'additional_development')
    agent.close()
    prediction_map = {(r['stage'], r['edge_uid'], r['bank_uid']): r for r in prediction_rows}
    calibration = []
    for stage, subset in [('validation', validation), ('development_evaluation', evaluation), ('additional_development', extra)]:
        calibration.append({'stage': stage, **score_predictions(subset, lambda r: prediction_map[r['stage'], r['edge_uid'], r['bank_uid']])})
    summary = {'history_success': sum(r['final']['binding_contract_success'] for r in history), 'history_trials': len(history),
        'training_trials': len(training), 'validation_trials': len(validation), 'evaluation_trials': len(evaluation),
        'additional_trials': len(extra), 'main': summarize(evaluation), 'additional': summarize(extra) if extra else [],
        'fit': fit_info, 'prediction_coverage': calibration, 'elapsed_seconds': perf_counter() - began,
        'reserved_workflows_executed': 0, 'business_goal_success': None, 'scope': protocol['scope']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
