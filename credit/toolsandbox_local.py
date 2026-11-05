"""Pinned ToolSandbox local, single-user-turn temporal memory reference.

Uses upstream tools and milestone/minefield verifiers, a local Qwen actor, and
complete restored contexts. The support-gated CREDIT reference abstains on
unseen exact prompts; it is not the manuscript's general semantic learner.
CMP/CAR scores are disclosed task-specific adaptations, not upstream ports.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import datetime as real_datetime
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
from random import Random
import sys
from time import perf_counter
from types import SimpleNamespace
import uuid

import attrs
import dill
import numpy as np

from .emp_binding_agent import QwenJSONAgent, parse_request
from .user_data import digest

MODEL = Path('/home/lxc/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')
UPSTREAM = Path('vendor/ToolSandbox')
SYSTEM = '''You are an agent operating a LOCAL simulated phone and calendar.
Use only the supplied tools. Do not guess identifiers: query the current tools.
Historical experience is advice, never a source of current IDs, values or dates.
Return exactly one JSON object per turn, with no markdown or explanation:
{"tool":"available_tool_name","arguments":{...}} to call one tool, or
{"answer":"concise final response to the user"} when finished.
Use real JSON booleans. If a tool reports an error, repair the plan.
Do not call end_conversation; finish by returning answer. Only this conversation
and public tool descriptions are available. Never invent a user clarification.'''


def write(directory, name, value):
    (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def upstream_imports(upstream):
    sys.path.insert(0, str(upstream.resolve()))
    from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType, get_current_context, set_current_context
    from tool_sandbox.common.message_conversion import Message
    from tool_sandbox.common.tool_conversion import convert_to_openai_tools
    from tool_sandbox.common.tool_discovery import ToolBackend, get_all_tools
    from tool_sandbox.roles.base_role import BaseRole
    from tool_sandbox.roles.execution_environment import ExecutionEnvironment
    from tool_sandbox.scenarios.single_tool_call_scenarios import named_single_tool_call_scenarios
    from tool_sandbox.scenarios.multiple_tool_call_scenarios import named_multiple_tool_call_scenarios
    from tool_sandbox.common.execution_context import ScenarioCategories
    return SimpleNamespace(**locals())


def install_local_clock(clock):
    """Freeze only upstream datetime calls; GPU/latency clocks remain real."""
    instant = real_datetime.datetime.fromisoformat(clock)
    class FixedDateTime(real_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return instant if tz is None else instant.replace(tzinfo=real_datetime.timezone.utc).astimezone(tz)
        @classmethod
        def utcnow(cls):
            return instant
    proxy = SimpleNamespace(**{k: getattr(real_datetime, k) for k in dir(real_datetime)})
    proxy.datetime = FixedDateTime
    for name, module in list(sys.modules.items()):
        if name.startswith('tool_sandbox') and getattr(module, 'datetime', None) is real_datetime:
            module.datetime = proxy


def set_local_uuid_noise(task, tool_index):
    identifier = uuid.UUID(bytes=sha256(f'{task}:{tool_index}:local_uuid'.encode()).digest()[:16], version=4)
    for name, module in list(sys.modules.items()):
        if name.startswith('tool_sandbox.tools.') and hasattr(module, 'uuid4'):
            module.uuid4 = lambda: identifier


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def family(name):
    return name.removesuffix('_alt').removesuffix('_implicit')


def local_scenarios(u):
    scenarios = {**u.named_single_tool_call_scenarios(u.ToolBackend.DEFAULT),
        **u.named_multiple_tool_call_scenarios(u.ToolBackend.DEFAULT)}
    tools = u.get_all_tools(u.ToolBackend.DEFAULT)
    eligible, excluded = {}, []
    for name, scenario in sorted(scenarios.items()):
        names = scenario.starting_context.tool_allow_list or []
        if any('rapid_api' in tools[t].__module__ for t in names):
            reason = 'external_search_requires_provider_access'
        elif u.ScenarioCategories.SINGLE_USER_TURN not in scenario.categories:
            reason = 'requires_interactive_user_simulator'
        else:
            eligible[name] = scenario; continue
        excluded.append({'task': name, 'reason': reason})
    return eligible, excluded


def initial_job(name, scenario, bank, u, tokenizer, stage, method):
    context = deepcopy(scenario.starting_context); u.set_current_context(context)
    env = u.ExecutionEnvironment()
    messages = u.BaseRole.get_messages()
    for index, message in enumerate(messages):
        if message.sender == u.RoleType.SYSTEM and message.recipient == u.RoleType.EXECUTION_ENVIRONMENT:
            env.respond(ending_index=index)
    context = u.get_current_context()
    available = {name: tool for name, tool in context.get_available_tools(scrambling_allowed=True).items()
        if u.RoleType.AGENT in getattr(tool, 'visible_to', (u.RoleType.AGENT,))}
    schemas = u.convert_to_openai_tools(available)
    exposure = [note for note in bank if set(note['tools']) & set(available)]
    exposure = sorted(exposure, key=lambda n: (-len(set(n['tools']) & set(available)), n['uid']))[:2]
    if sum(note['tokens'] for note in exposure) > 512:
        raise ValueError('exposure token budget exceeded')
    public = [message for message in u.BaseRole.get_messages()
        if u.RoleType.AGENT in message.visible_to]
    system = SYSTEM + '\n' + '\n'.join(m.content for m in public if m.sender == u.RoleType.SYSTEM)
    system += '\nAvailable tools: ' + json.dumps(schemas, ensure_ascii=False)
    system += '\nHistorical experience: ' + json.dumps([n['text'] for n in exposure], ensure_ascii=False)
    chat = [{'role': 'system', 'content': system}]
    chat += [{'role': 'user', 'content': m.content} for m in public if m.sender == u.RoleType.USER]
    return {'task': name, 'family': family(name), 'scenario': scenario, 'context': context,
        'stage': stage, 'method': method, 'bank': [n['uid'] for n in bank], 'exposure': [n['uid'] for n in exposure],
        'tools': available, 'chat': chat, 'calls': [], 'generations': [],
        'initial_state_sha256': fingerprint(context.to_dict(serialize_console=False)),
        'done': False, 'termination': None, 'began': perf_counter()}


def physical_world(context):
    value = context.to_dict(serialize_console=False)
    value['_dbs'].pop('sandbox', None)
    return value


def consume(job, generation, u, max_calls):
    u.set_current_context(job['context'])
    raw = generation['agent_output']; action, parse_error = parse_request(raw)
    job['generations'].append({'prompt_sha256': generation['prompt_sha256'],
        'action': action, 'parse_error': parse_error, 'hit_generation_limit': generation['hit_generation_limit']})
    job['chat'].append({'role': 'assistant', 'content': raw})
    if 'answer' in action and isinstance(action['answer'], str):
        u.BaseRole.add_messages([u.Message(sender=u.RoleType.AGENT, recipient=u.RoleType.USER,
            content=action['answer'], conversation_active=False)])
        job['done'] = True; job['termination'] = 'final_answer'
    elif action.get('tool') in job['tools'] and isinstance(action.get('arguments'), dict):
        if len(job['calls']) >= max_calls:
            job['done'] = True; job['termination'] = 'tool_call_budget'; return
        name = action['tool']; arguments = action['arguments']; index = len(job['calls'])
        before = physical_world(u.get_current_context())
        set_local_uuid_noise(job['task'], index)
        # Only a validated tool identifier and JSON data enter Python execution.
        code = 'print(repr(' + name + '(**json.loads(' + repr(json.dumps(arguments)) + '))))'
        call_id = 'call_' + str(index)
        u.BaseRole.add_messages([u.Message(sender=u.RoleType.AGENT, recipient=u.RoleType.EXECUTION_ENVIRONMENT,
            content=code, openai_tool_call_id=call_id, openai_function_name=name)])
        u.ExecutionEnvironment().respond()
        replies = [m for m in u.BaseRole.get_messages() if m.sender == u.RoleType.EXECUTION_ENVIRONMENT
            and m.openai_tool_call_id == call_id]
        reply = replies[-1]
        job['calls'].append({'tool': name, 'arguments': arguments, 'response': reply.content,
            'exception': reply.tool_call_exception, 'world_before': before,
            'world_after': physical_world(u.get_current_context())})
        job['chat'].append({'role': 'user', 'content': json.dumps({'tool_result': reply.content,
            'tool_exception': reply.tool_call_exception}, ensure_ascii=False)})
    else:
        job['chat'].append({'role': 'user', 'content': json.dumps({'error': parse_error or 'invalid tool/action',
            'instruction': 'Return exactly one tool/arguments or answer JSON object.'})})
    job['context'] = u.get_current_context()


def finish(job, directory, u):
    context = job['context']; scenario = job['scenario']
    try:
        result = scenario.evaluation.evaluate(context, max_turn_count=32)
        evaluation = attrs.asdict(result)
        verified = bool(result.similarity >= 1 - 1e-8 and result.minefield_similarity == 0)
        verification_error = None
    except Exception as error:
        evaluation = None; verified = False; verification_error = type(error).__name__ + ': ' + str(error)
    value = context.to_dict(serialize_console=False)
    row = {key: job[key] for key in ('task', 'family', 'stage', 'method', 'bank', 'exposure', 'calls', 'generations',
        'initial_state_sha256', 'termination')}
    row.update(success=int(verified), upstream_evaluation=evaluation, verification_coverage=verification_error is None,
        verification_error=verification_error, tool_calls=len(job['calls']),
        latency_seconds=perf_counter() - job['began'], ending_context=value,
        ending_state_sha256=fingerprint(value), provider_drift=False)
    with (directory / 'episodes.jsonl').open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return row


def run_batch(jobs, agent, directory, u, *, max_calls=8):
    rows, active = [], jobs
    for turn in range(16):
        if not active:
            break
        records = agent.infer([job['chat'] for job in active])
        remaining = []
        for job, record in zip(active, records):
            consume(job, record, u, max_calls)
            if job['done']:
                rows.append(finish(job, directory, u))
            else:
                remaining.append(job)
        active = remaining
    for job in active:
        job['termination'] = 'turn_budget'; rows.append(finish(job, directory, u))
    return rows


def memory_from(row, tokenizer):
    if not row['success']:
        return None
    tools = sorted({call['tool'] for call in row['calls'] if not call['exception']})
    if not tools:
        return None
    text = 'Verified prior local task: ' + row['task'] + '. Tool sequence and argument names: ' + json.dumps(
        [{'tool': c['tool'], 'argument_fields': sorted(c['arguments'])} for c in row['calls']], ensure_ascii=False)
    text += '. Query current identifiers and values; never copy prior IDs or dates.'
    tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if tokens > 170:
        return None
    return {'uid': fingerprint([row['task'], text])[:16], 'text': text, 'tools': tools,
        'tokens': tokens, 'source_task': row['task'], 'source_ending_sha256': row['ending_state_sha256'],
        'source_verifier_success': True, 'arrival': row['task']}


def all_banks(pool, capacity=3):
    return [list(bank) for size in range(len(pool) + 1) for bank in combinations(pool, size)
        if sum(n['tokens'] for n in bank) <= 512 and size <= capacity]


def bank_key(bank):
    return tuple(sorted(n['uid'] for n in bank))


def scoring_banks(pool, interventions, future, agent, scenarios, u):
    banks = all_banks(pool, capacity=2); utilities = defaultdict(list)
    for row in interventions:
        utilities[tuple(sorted(row['bank']))].append(row['success'])
    utility = {key: float(np.mean(values)) for key, values in utilities.items()}
    full = tuple(sorted(n['uid'] for n in pool)); rng = Random(421)
    loo = {n['uid']: utility.get(full, 0.) - utility.get(tuple(u for u in full if u != n['uid']), 0.) for n in pool}
    shapley = dict.fromkeys(full, 0.)
    for _ in range(32):
        order = list(full); rng.shuffle(order); chosen = ()
        for uid in order:
            following = tuple(sorted(chosen + (uid,)))
            shapley[uid] += (utility.get(following, 0.) - utility.get(chosen, 0.)) / 32
            chosen = following
    cmp_scores, car_scores = {}, {}
    for uid in full:
        treated = [r['success'] for r in interventions if uid in r['exposure']]
        control = [r['success'] for r in interventions if uid not in r['exposure']]
        cmp_scores[uid] = float(np.mean(treated) - np.mean(control)) if treated and control else None
        by_task = {(r['task'], tuple(sorted(r['bank']))): r for r in interventions}
        differences = []
        for r in interventions:
            if uid not in r['bank']:
                continue
            baseline = by_task.get((r['task'], tuple(sorted(x for x in r['bank'] if x != uid))))
            if baseline is not None:
                differences.append(r['success'] - baseline['success'])
        car_scores[uid] = float(np.mean(differences)) if differences else None
    def ranked(scores):
        chosen = []
        for n in sorted(pool, key=lambda n: (-(scores.get(n['uid']) or 0.), n['uid'])):
            if sum(x['tokens'] for x in chosen) + n['tokens'] <= 512:
                chosen.append(n)
        return chosen
    configs = {'NoMem': [], 'FIFO': pool[:2], 'Recency': list(reversed(pool))[:2],
        'Version-aware heuristic': list(reversed(pool))[:2], 'LOO': ranked(loo)[:2], 'MC-Shapley': ranked(shapley)[:2],
        'CMP-adapted': ranked(cmp_scores)[:2], 'CAR-adapted': ranked(car_scores)[:2]}
    # Exact full prompt support is checked before reading any test outcomes.
    policy_keys = {g['prompt_sha256'] for row in interventions for g in row['generations']}
    forecasts = []
    for name in future:
        for bank in banks:
            job = initial_job(name, scenarios[name], bank, u, agent.tokenizer, 'forecast', 'candidate')
            rendered = agent.tokenizer.apply_chat_template(job['chat'], tokenize=False, add_generation_prompt=True)
            forecasts.append({'task': name, 'bank': list(bank_key(bank)),
                'initial_policy_supported': sha256(rendered.encode()).hexdigest() in policy_keys})
    if any(f['initial_policy_supported'] for f in forecasts):
        raise ValueError('unexpected test/train prompt overlap; inspect split before deployment')
    configs['CREDIT-support-gated'] = []
    return configs, {'LOO': loo, 'MC-Shapley': shapley, 'CMP-adapted': cmp_scores,
        'CAR-adapted': car_scores, 'known_bank_utilities': {','.join(k): v for k, v in utility.items()},
        'support_forecasts': forecasts, 'CREDIT_selection': 'unknown success mass 0 minus positive storage cost selects empty bank'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=UPSTREAM)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--output', type=Path, default=Path('results/toolsandbox_local_v1'))
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    u = upstream_imports(args.upstream)
    clock = '2026-10-12T12:00:00'; install_local_clock(clock)
    scenarios, exclusions = local_scenarios(u)
    # Single-tool tasks establish history; all multi-tool templates remain future.
    history = sorted(name for name, s in scenarios.items() if u.ScenarioCategories.SINGLE_TOOL_CALL in s.categories)
    future = sorted(set(scenarios) - set(history))
    if args.pilot:
        history = ['cellular_off', 'search_phone_number_with_name']; future = ['turn_on_wifi_low_battery_mode']
    args.output.mkdir(parents=True); (args.output / 'source').mkdir()
    names = ['toolsandbox_local.py', 'emp_binding_agent.py', 'emp_binding.py', 'user_data.py']
    upstream_sources = {str(p.relative_to(args.upstream)): digest(p) for p in args.upstream.rglob('*.py')}
    protocol = {'suite': 'ToolSandbox local single-user-turn temporal reference',
        'upstream_directory': str(args.upstream.resolve()), 'upstream_commit': 'c8571d7854316d2e1c5f288e59fe1e34e53f6dd1',
        'upstream_url': 'https://github.com/apple-aiml-research/ToolSandbox', 'upstream_source_sha256': upstream_sources,
        'source_sha256': {n: digest(Path(__file__).with_name(n)) for n in names},
        'history': history, 'future': future, 'clock': clock, 'storage_tokens': 512, 'exposure_tokens': 512,
        'retrieval_depth': 2, 'maximum_memory_pool': 3, 'extra_tool_call_budget': 256,
        'deployment_bank_memory_capacity': 2,
        'maximum_tool_calls_per_episode': 8, 'maximum_actor_turns': 16,
        'CREDIT_mode': 'exact recorded-policy support gate; no general semantic transport',
        'baselines': 'LOO/MC/CMP/CAR use the same frozen retrospective training intervention table; adapted scores, not upstream ports',
        'clock_and_uuid_noise': 'fixed local benchmark clock; UUID noise paired by task and call index; no verifier changes',
        'external_network_tool_calls': 0, 'provider_drift': False,
        'unknown_verification_counts_as_non_success': True, 'greedy_actor_independent_seeds': 1,
        'pilot': args.pilot, 'exclusions': exclusions}
    protocol['latency_scope'] = 'observed batched/cache-assisted run only; no equal-latency ranking claim'
    write(args.output, 'frozen_protocol.json', protocol)
    for n in names:
        (args.output / 'source' / n).write_bytes(Path(__file__).with_name(n).read_bytes())
    (args.output / 'scenarios.pkl').write_bytes(dill.dumps(scenarios))
    write(args.output, 'scenario_freeze.json', {'sha256': digest(args.output / 'scenarios.pkl')})
    agent = QwenJSONAgent(args.model, args.output, batch_size=8, max_new_tokens=384)
    agent.manifest['system_prompt'] = SYSTEM
    agent.manifest['benchmark'] = protocol['suite']
    history_rows = []
    for offset in range(0, len(history), 8):
        history_rows += run_batch([initial_job(name, scenarios[name], [], u, agent.tokenizer, 'history', 'NoMem')
            for name in history[offset:offset+8]], agent, args.output, u)
    pool = []
    for row in sorted(history_rows, key=lambda r: r['task']):
        note = memory_from(row, agent.tokenizer)
        if note and not any(set(n['tools']) == set(note['tools']) for n in pool):
            pool.append(note)
        if len(pool) == 3:
            break
    write(args.output, 'memories.json', pool)
    schedule = [(name, bank) for name in history for bank in all_banks(pool)]
    Random(420).shuffle(schedule)
    write(args.output, 'intervention_assignment.json', [{'task': name, 'bank': list(bank_key(bank)),
        'conditional_order_propensity': 1 / (len(schedule) - index)} for index, (name, bank) in enumerate(schedule)])
    interventions, used = [], 0
    for name, bank in schedule:
        if 256 - used < 8:
            break
        row = run_batch([initial_job(name, scenarios[name], bank, u, agent.tokenizer,
            'intervention_train', 'bank_probe')], agent, args.output, u)[0]
        interventions.append(row); used += row['tool_calls']
    configs, scores = scoring_banks(pool, interventions, future, agent, scenarios, u)
    write(args.output, 'frozen_banks.json', {method: list(bank_key(bank)) for method, bank in configs.items()})
    write(args.output, 'selection.json', scores)
    write(args.output, 'evaluation_freeze.json', {name: digest(args.output / name) for name in
        ['frozen_protocol.json', 'scenario_freeze.json', 'memories.json', 'frozen_banks.json', 'selection.json']})
    rows = []
    for method, bank in configs.items():
        for offset in range(0, len(future), 8):
            rows += run_batch([initial_job(name, scenarios[name], bank, u, agent.tokenizer, 'future_test', method)
                for name in future[offset:offset+8]], agent, args.output, u)
        print(json.dumps({'method': method, 'future_episodes': len(future), 'completed': True}), flush=True)
    agent.close()
    table = []
    for method in configs:
        subset = [row for row in rows if row['method'] == method]
        table.append({'method': method, 'episodes': len(subset), 'strict_VSR': float(np.mean([r['success'] for r in subset])),
            'verification_coverage': float(np.mean([r['verification_coverage'] for r in subset])),
            'upstream_mean_score': float(np.mean([r['upstream_evaluation']['similarity'] if r['upstream_evaluation'] else 0. for r in subset])),
            'mean_tool_calls': float(np.mean([r['tool_calls'] for r in subset])),
            'p50_episode_latency_seconds': float(np.median([r['latency_seconds'] for r in subset])),
            'p95_episode_latency_seconds': float(np.quantile([r['latency_seconds'] for r in subset], .95)),
            'training_extra_tool_calls': used if method in ('LOO','MC-Shapley','CMP-adapted','CAR-adapted','CREDIT-support-gated') else 0})
    write(args.output, 'summary.json', {'table': table, 'history_episodes': len(history_rows),
        'training_interventions': len(interventions), 'training_extra_tool_calls': used, 'memory_pool': len(pool),
        'future_episodes_per_method': len(future), 'scope': protocol['suite'], 'full_paper_CREDIT': False,
        'semantic_transport_coverage': 0., 'external_network_tool_calls': 0, 'provider_drift': False})
    with (args.output / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
