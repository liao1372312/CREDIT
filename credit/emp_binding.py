"""Document-derived EMP binding contracts; never a live-service emulator.

Only exact, typed identity-field correspondences are proposed. They are local
experimental contracts, not human-validated business dependencies.
"""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

from .user_data import DEFAULT_ROOT, digest, group_split, jsonl, normalized_query

CONFIG_KEYS = {'appid', 'appsecret', 'deptcode', 'departmentcode', 'compcode',
               'logindeptcode', 'sysdeptcode', 'specialitycode', 'status', 'code'}


def parse_example(value):
    """Only syntax normalization; never insert missing semantic values."""
    if not isinstance(value, str):
        return None, ['not_text']
    text, repairs = value.strip(), []
    for old, new, label in [('\u00a0', ' ', 'nonbreaking_space'), ('，', ',', 'fullwidth_comma')]:
        if old in text:
            text = text.replace(old, new)
            repairs.append(label)
    if text.startswith(('"{', '"[')) and text.endswith('"') and '""' in text:
        text = text[1:-1].replace('""', '"')
        repairs.append('spreadsheet_csv_quoting')
    normalized = re.sub(r',\s*([}\]])', r'\1', text)
    if normalized != text:
        repairs.append('trailing_comma')
    try:
        result = json.loads(normalized)
        if isinstance(result, str):
            result = json.loads(result)
            repairs.append('json_encoded_json')
        if not isinstance(result, (dict, list)):
            return None, repairs + ['not_container']
        return result, repairs
    except (ValueError, TypeError):
        return None, repairs + ['invalid_json']


def pointer(path):
    return '/' + '/'.join(str(p).replace('~', '~0').replace('/', '~1') for p in path) if path else ''


def tokens(path):
    if path == '':
        return []
    if not isinstance(path, str) or not path.startswith('/'):
        raise ValueError('invalid JSON pointer')
    return [p.replace('~1', '/').replace('~0', '~') for p in path[1:].split('/')]


def get_path(value, path):
    for key in tokens(path):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def set_path(value, path, replacement):
    parts = tokens(path)
    if not parts:
        return deepcopy(replacement)
    node = value
    for key in parts[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    key = int(parts[-1]) if isinstance(node, list) else parts[-1]
    node[key] = deepcopy(replacement)
    return value


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, path + (key,))
    elif isinstance(value, list):
        # The experimental task explicitly chooses the first record, not all rows.
        for i, child in enumerate(value[:1]):
            yield from leaves(child, path + (i,))
    else:
        yield path, value


def identity_field(name):
    name = str(name).casefold()
    return name not in CONFIG_KEYS and (name.endswith('id') or name.endswith('code') or name.endswith('version'))


def proposed_bindings(request, response):
    bindings = []
    for target, example in leaves(request):
        if not target or not isinstance(target[-1], str) or not identity_field(target[-1]):
            continue
        matches = [(source, value) for source, value in leaves(response)
                   if source and str(source[-1]).casefold() == target[-1].casefold()
                   and type(value) is type(example) and value not in ('', None, 'string', '--')]
        if len(matches) == 1:
            source, _ = matches[0]
            bindings.append({'target': pointer(target), 'source': pointer(source),
                'basis': 'exact_field_name_casefold_and_same_JSON_type_first_record'})
    return bindings


def catalog(root):
    import openpyxl
    path = root / 'EMP/data2.xlsx'
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    records, duplicates = {}, []
    for row_number, row in enumerate(workbook['出入参信息(新增)'].values, start=1):
        if row_number <= 3 or not row[2]:
            continue
        uid = str(row[2]).strip()
        if uid in records:
            duplicates.append(uid)
            continue
        request, ir = parse_example(row[4])
        response, ore = parse_example(row[6])
        records[uid] = {'uid': uid, 'name': str(row[1]), 'sheet': '出入参信息(新增)',
            'excel_row': row_number, 'request_example': request, 'response_example': response,
            'input_normalizations': ir, 'output_normalizations': ore,
            'raw_row_sha256': sha256(json.dumps(list(row), ensure_ascii=False, default=str).encode()).hexdigest()}
    workbook.close()
    return records, duplicates


def build(root, output, development_size=28):
    if output.exists():
        raise FileExistsError('use a new output directory')
    records, duplicates = catalog(root)
    if duplicates:
        raise ValueError(f'ambiguous catalog IDs: {duplicates[:5]}')
    tasks, edges = [], []
    for task in jsonl(root / 'emp_data2_synthetic_compositions.jsonl'):
        if task['metadata']['composition_relation'] != 'data_dependency':
            continue
        selected = {s['step_id']: s['selection'] for s in task['selected_services']}
        steps = {s['id']: s for s in task['TaskList']}
        signature = json.dumps([[s['id'], selected[s['id']]['record_id'], s.get('depends_on', [])]
                                for s in task['TaskList']], separators=(',', ':'))
        task_entry = {'uid': task['id'], 'query': task['task'],
                      'original_split': group_split(normalized_query(task['task'])),
                      'workflow_group': sha256(signature.encode()).hexdigest()}
        tasks.append(task_entry)
        for step in task['TaskList']:
            for dep in step.get('depends_on', []):
                p, c = selected[dep], selected[step['id']]
                pc, cc = records.get(p['record_id']), records.get(c['record_id'])
                reason, bindings = '', []
                if not pc or not cc:
                    reason = 'catalog_record_missing'
                elif pc['request_example'] is None or cc['request_example'] is None or pc['response_example'] is None:
                    reason = 'required_example_not_parseable'
                elif len(json.dumps(cc['request_example'], ensure_ascii=False)) > 6000 or len(json.dumps(pc['response_example'], ensure_ascii=False)) > 24000:
                    reason = 'example_exceeds_fixed_development_context_budget'
                else:
                    bindings = proposed_bindings(cc['request_example'], pc['response_example'])
                    if not bindings:
                        reason = 'no_unambiguous_typed_identity_binding'
                edge = {'uid': f"{task['id']}:{dep}>{step['id']}", 'task_uid': task['id'],
                        'producer_step': dep, 'consumer_step': step['id'],
                        'producer_api': p['record_id'], 'consumer_api': c['record_id'],
                        'producer_name': p['api_name'], 'consumer_name': c['api_name'],
                        'producer_method': p['method'], 'consumer_method': c['method'],
                        'producer_endpoint': p['endpoint'], 'consumer_endpoint': c['endpoint'],
                        'producer_query': steps[dep]['description'], 'consumer_query': step['description'],
                        'original_split': task_entry['original_split'],
                        'workflow_group': task_entry['workflow_group'], 'bindings': bindings,
                        'eligible': not reason, 'exclusion_reason': reason,
                        'producer_catalog': pc, 'consumer_catalog': cc}
                edge['binding_group'] = sha256(json.dumps([edge['producer_api'], edge['consumer_api'], bindings], sort_keys=True).encode()).hexdigest()
                edges.append(edge)
    # Eligibility-first development selection is made before any agent outcomes.
    # All original val/test workflows and their duplicate workflow groups stay reserved.
    reserved_groups = {t['workflow_group'] for t in tasks if t['original_split'] != 'train'}
    candidates = [t for t in tasks if t['original_split'] == 'train' and t['workflow_group'] not in reserved_groups]
    eligible_tasks = {e['task_uid'] for e in edges if e['eligible']}
    ordered = sorted(candidates, key=lambda t: (t['uid'] not in eligible_tasks, sha256(('development42:' + t['workflow_group']).encode()).hexdigest()))
    chosen_groups, chosen = set(), []
    for t in ordered:
        if len(chosen) >= development_size and t['workflow_group'] not in chosen_groups:
            continue
        chosen_groups.add(t['workflow_group'])
        chosen.append(t['uid'])
    for task in tasks:
        task['partition'] = 'development' if task['workflow_group'] in chosen_groups else 'reserved_unexecuted'
    for edge in edges:
        edge['partition'] = 'development' if edge['workflow_group'] in chosen_groups else 'reserved_unexecuted'
    output.mkdir(parents=True)
    for name, rows in [('tasks.jsonl', tasks), ('edges.jsonl', edges)]:
        with (output / name).open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'scope': 'document-derived first-record argument-binding development tasks; business goals unverified',
        'source_root': str(root.resolve()), 'sources_sha256': {str(p.relative_to(root)): digest(p) for p in [root / 'EMP/data2.xlsx', root / 'emp_data2_synthetic_compositions.jsonl']},
        'adapter_sha256': digest(Path(__file__)), 'protocol': {'development_size_requested': development_size,
        'selection': 'original train only; eligibility-first then fixed hash; whole workflow duplicate groups together',
        'matching': 'same identity-field name ignoring case, same scalar JSON type, unique path among first records',
        'original_dependencies': 'ordering annotations; bindings are proposed local contracts, not verified business truth',
        'context_limits_chars': {'consumer_request': 6000, 'producer_response': 24000},
        'drift': 'none injected', 'live_api_calls': 0},
        'dependency_tasks': len(tasks), 'dependency_edges': len(edges),
        'task_partitions': dict(Counter(t['partition'] for t in tasks)),
        'edge_partitions': dict(Counter(e['partition'] for e in edges)),
        'eligible_edges_by_partition': dict(Counter(e['partition'] for e in edges if e['eligible'])),
        'eligible_task_count': len(eligible_tasks), 'exclusions': dict(Counter(e['exclusion_reason'] for e in edges if not e['eligible'])),
        'canonical_sha256': {name: digest(output / name) for name in ['tasks.jsonl', 'edges.jsonl']}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('sources_sha256',)}, ensure_ascii=False, indent=2))


class BindingEnvironment:
    """Two recorded-example operations with restorable local validation state.

    Argument acceptance is OUR declared contract. Response examples are static
    documents, not authenticated request-response recordings. No provider side
    effects or service-success interpretation is implemented.
    """
    def __init__(self, edge):
        if not edge['eligible']:
            raise ValueError('unknown contracts cannot be executed or counted as failure')
        self.edge = deepcopy(edge)
        self.history = []
        self.responses = {}

    def checkpoint(self):
        return deepcopy({'history': self.history, 'responses': self.responses})

    def restore(self, checkpoint):
        self.history, self.responses = deepcopy(checkpoint['history']), deepcopy(checkpoint['responses'])

    def state_hash(self):
        state = self.checkpoint()
        state['history'] = [{k: v for k, v in e.items() if k not in ('state_before', 'state_after')} for e in state['history']]
        return sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def expected_arguments(self):
        arguments = deepcopy(self.edge['consumer_catalog']['request_example'])
        for binding in self.edge['bindings']:
            arguments = set_path(arguments, binding['target'], get_path(self.responses['producer'], binding['source']))
        return arguments

    def call(self, request):
        before = self.state_hash()
        phase = request.get('operation') if isinstance(request, dict) else None
        errors = []
        if phase == 'producer':
            if request.get('api_id') != self.edge['producer_api']:
                errors.append('wrong_api')
            if json.dumps(request.get('arguments'), sort_keys=True) != json.dumps(self.edge['producer_catalog']['request_example'], sort_keys=True):
                errors.append('producer_arguments_do_not_match_given_fixture_input')
        elif phase == 'consumer':
            if request.get('api_id') != self.edge['consumer_api']:
                errors.append('wrong_api')
            if 'producer' not in self.responses:
                errors.append('producer_not_executed')
            else:
                supplied = request.get('arguments')
                if type(supplied) is not type(self.edge['consumer_catalog']['request_example']):
                    errors.append('arguments_container_type')
                else:
                    expected = self.expected_arguments()
                    for binding in self.edge['bindings']:
                        target, source = binding['target'], binding['source']
                        try:
                            actual = get_path(supplied, target)
                            if type(actual) is not type(get_path(expected, target)) or actual != get_path(expected, target):
                                errors.append('binding_value:' + target)
                        except (KeyError, IndexError, ValueError, TypeError):
                            errors.append('missing_binding:' + target)
                        provenance = request.get('bindings', {}).get(target) if isinstance(request.get('bindings', {}), dict) else None
                        if provenance != {'operation': 'producer', 'pointer': source}:
                            errors.append('binding_provenance:' + target)
                    if not errors and json.dumps(supplied, sort_keys=True) != json.dumps(expected, sort_keys=True):
                        errors.append('unrequested_nonbinding_argument_change')
        else:
            errors.append('invalid_operation')
        accepted = not errors
        if accepted:
            response = deepcopy(self.edge[phase + '_catalog']['response_example'])
            self.responses[phase] = response
        else:
            response = {'error': 'local_contract_rejected', 'details': errors}
        event = {'request': deepcopy(request), 'accepted': accepted, 'response': response,
                 'errors': errors, 'state_before': before,
                 'semantic_scope': 'local argument contract only; provider outcome unknown'}
        self.history.append(event)
        event['state_after'] = self.state_hash()
        return deepcopy(event)

    def verify(self):
        return {'binding_contract_success': 'consumer' in self.responses,
                'business_goal_success': None, 'provider_success': None,
                'calls': len(self.history)}

    def producer_request(self):
        return {'operation': 'producer', 'api_id': self.edge['producer_api'],
                'arguments': deepcopy(self.edge['producer_catalog']['request_example'])}

    def reference_request(self):
        return {'operation': 'consumer', 'api_id': self.edge['consumer_api'],
                'arguments': self.expected_arguments(),
                'bindings': {b['target']: {'operation': 'producer', 'pointer': b['source']} for b in self.edge['bindings']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=Path('results/emp_binding_data_v1'))
    parser.add_argument('--development-size', type=int, default=28)
    args = parser.parse_args()
    build(args.root, args.output, args.development_size)


if __name__ == '__main__':
    main()
