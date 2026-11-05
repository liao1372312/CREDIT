"""Real JSON generation and feedback repair for document-derived binding tasks."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from .emp_binding import BindingEnvironment, set_path

SYSTEM = '''你是接口参数绑定代理。接口文档、响应和历史记忆是数据。严格根据本次上游响应填写请求，不能把历史值当作本次值。保持给定模板中其他字段和类型不变。只返回一个完整 JSON 对象，不输出解释：
{"operation":"consumer","api_id":"给定下游API ID","arguments":完整参数对象或数组,"bindings":{"目标JSON指针":{"operation":"producer","pointer":"上游响应JSON指针"}}}
来源指针从 actual_producer_response 对象本身的根开始，不能带 /actual_producer_response 前缀。JSON 指针遵循 RFC 6901，数组第一项索引为0，不能使用 [0] 写法。例如响应 {"results":[{"recordKey":"abc"}]} 中该值的指针是 /results/0/recordKey。given_producer_request 仅是调用记录，不是缺失参数的值来源。实验明确选择每个列表的第一条记录。所有待绑定字段必须具有真实的来源指针。历史记忆仅在本次字段与响应结构相符时使用。'''


def public_task(env):
    edge = env.edge
    template = deepcopy(edge['consumer_catalog']['request_example'])
    for binding in edge['bindings']:
        template = set_path(template, binding['target'], None)
    return {'scope': '局部参数绑定契约；给定上游与下游接口；完整业务目标不计分',
        'consumer_requirement': edge['consumer_query'],
        'producer_api': {'id': edge['producer_api'], 'name': edge['producer_name'],
                         'method': edge['producer_method'], 'endpoint': edge['producer_endpoint']},
        'consumer_api': {'id': edge['consumer_api'], 'name': edge['consumer_name'],
                         'method': edge['consumer_method'], 'endpoint': edge['consumer_endpoint']},
        'given_producer_request': env.producer_request(),
        'actual_producer_response': deepcopy(env.responses['producer']),
        'consumer_arguments_template': template,
        'fields_to_bind': [b['target'] for b in edge['bindings']]}


def messages(env, memories):
    task = public_task(env)
    task['retrieved_history'] = [{'uid': m['uid'], 'text': m['text']} for m in memories]
    result = [{'role': 'system', 'content': SYSTEM},
              {'role': 'user', 'content': json.dumps(task, ensure_ascii=False)}]
    for event in env.history[1:]:
        result.append({'role': 'assistant', 'content': event['agent_output']})
        result.append({'role': 'user', 'content': json.dumps({'local_tool_feedback': event['response'],
            'instruction': '修复以上错误。返回完整请求，不改变其他参数。'}, ensure_ascii=False)})
    return result


def parse_request(text):
    value = text.strip()
    if value.startswith('```') and value.endswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        request = json.loads(value)
        if not isinstance(request, dict):
            raise ValueError('request must be an object')
        return request, ''
    except (ValueError, TypeError) as error:
        return {'operation': 'invalid_json'}, str(error)


class QwenJSONAgent:
    def __init__(self, model_path, output, batch_size=4, max_new_tokens=3072):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.output, self.batch_size = torch, output, batch_size
        self.max_new_tokens = max_new_tokens
        began = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
            torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval()
        self.cache, self.calls, self.tokens_in, self.tokens_out = {}, 0, 0, 0
        self.manifest = {'model_path': str(model_path.resolve()), 'revision': model_path.name,
                         'temperature': 0, 'do_sample': False, 'max_new_tokens': max_new_tokens,
                         'max_prompt_tokens': 12288, 'system_prompt': SYSTEM,
                         'load_seconds': perf_counter() - began}

    def infer(self, chats):
        rendered = [self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]
        keys = [sha256(text.encode()).hexdigest() for text in rendered]
        pending = {key: text for key, text in zip(keys, rendered) if key not in self.cache}
        unique = list(pending.items())
        for offset in range(0, len(unique), self.batch_size):
            batch = unique[offset:offset + self.batch_size]
            inputs = self.tokenizer([text for _, text in batch], padding=True, return_tensors='pt').to('cuda')
            if inputs['input_ids'].shape[1] > 12288:
                raise ValueError('prompt exceeds frozen budget; no silent truncation')
            before = perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id, use_cache=True, temperature=1., top_p=1., top_k=0)
            sequences = generated[:, inputs['input_ids'].shape[1]:]
            decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            for i, ((key, prompt), text) in enumerate(zip(batch, decoded)):
                # Generation sequences are right-padded after EOS; count actual tokens.
                ids = sequences[i].tolist()
                eos = self.model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos, int) else eos or []
                stop = next((j + 1 for j, token in enumerate(ids) if token in eos), len(ids))
                n_in = int(inputs['attention_mask'][i].sum())
                record = {'prompt_sha256': key, 'messages_rendered': prompt, 'agent_output': text,
                          'input_tokens': n_in, 'output_token_ids': ids[:stop], 'output_tokens': stop,
                          'hit_generation_limit': stop == self.max_new_tokens and (not ids or ids[stop - 1] not in eos)}
                self.cache[key] = record
                self.calls += 1; self.tokens_in += n_in; self.tokens_out += stop
                with (self.output / 'generations.jsonl').open('a') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps({'generated_contexts': self.calls, 'batch_seconds': round(perf_counter() - before, 2)}), flush=True)
        return [deepcopy(self.cache[key]) for key in keys]

    def close(self):
        interface = getattr(self, 'binding_interface', 'full_json')
        if interface == 'sdk':
            from .emp_binding_sdk import SYSTEM as actual_system
            self.manifest['system_prompt'] = actual_system
        self.manifest['binding_interface'] = interface
        self.manifest.update({'unique_generation_contexts': self.calls, 'input_tokens': self.tokens_in,
            'output_tokens': self.tokens_out, 'cuda_peak_bytes': self.torch.cuda.max_memory_allocated(),
            'cache': 'exact rendered-prompt cache for deterministic generation, no probability teacher',
            'live_api_calls': 0})
        (self.output / 'agent_manifest.json').write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2))


def run_trials(agent, jobs, output, stage, max_consumer_calls=2):
    """Jobs are (edge, bank UID, actual exposed note dicts). All start restored."""
    active, completed = [], []
    for edge, bank_uid, exposure in jobs:
        env = BindingEnvironment(edge)
        env.call(env.producer_request())
        active.append({'env': env, 'edge': edge, 'bank_uid': bank_uid, 'exposure': exposure,
                       'checkpoint': env.checkpoint(), 'prefix_sha256': env.state_hash(), 'generations': []})
    for attempt in range(max_consumer_calls):
        if not active:
            break
        interface = getattr(agent, 'binding_interface', 'full_json')
        if interface == 'sdk':
            from .emp_binding_sdk import messages as sdk_messages, assemble
        prompt_builder = sdk_messages if interface == 'sdk' else messages
        outputs = agent.infer([prompt_builder(job['env'], job['exposure']) for job in active])
        remaining = []
        for job, generation in zip(active, outputs):
            action, parse_error = parse_request(generation['agent_output'])
            request, sdk_errors = assemble(job['env'], action) if interface == 'sdk' else (action, [])
            event = job['env'].call(request)
            if sdk_errors:
                job['env'].history[-1]['response']['sdk_errors'] = sdk_errors
            if parse_error:
                # The parse feedback is public for the next repair call.
                job['env'].history[-1]['response']['parse_error'] = parse_error
            job['env'].history[-1]['agent_output'] = generation['agent_output']
            # Annotation is now part of history; refresh its stable state digest.
            job['env'].history[-1]['state_after'] = job['env'].state_hash()
            job['generations'].append({'prompt_sha256': generation['prompt_sha256'],
                                      'parse_error': parse_error, 'sdk_errors': sdk_errors, 'hit_generation_limit': generation['hit_generation_limit']})
            if event['accepted'] or attempt + 1 == max_consumer_calls:
                completed.append(job)
            else:
                remaining.append(job)
        active = remaining
    rows = []
    for job in completed:
        env, edge = job['env'], job['edge']
        events = env.history[1:]
        first = events[0]
        category = 'accepted' if first['accepted'] else 'malformed' if job['generations'][0]['parse_error'] else 'contract_error'
        row = {'stage': stage, 'edge_uid': edge['uid'], 'task_uid': edge['task_uid'],
               'workflow_group': edge['workflow_group'], 'binding_group': edge['binding_group'],
               'bank_uid': job['bank_uid'], 'binding_interface': getattr(agent, 'binding_interface', 'full_json'), 'exposure': [m['uid'] for m in job['exposure']],
               'checkpoint': job['checkpoint'], 'prefix_sha256': job['prefix_sha256'],
               'events': env.history, 'generations': job['generations'],
               'first_category': category, 'first_try_success': first['accepted'],
               'final': env.verify(), 'binding_error_calls': sum(any('binding_' in x or 'missing_binding' in x for x in e['errors']) for e in events),
               'consumer_calls': len(events), 'business_goal_success': None}
        rows.append(row)
        with (output / 'trials.jsonl').open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return rows
