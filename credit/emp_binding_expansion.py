"""Conservative syntax recovery and identifier spelling normalization for EMP.

Expands only the already-declared development partition. Never synthesizes
values, semantic synonyms, missing delimiters, authentication, or API history.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

from .emp_binding import CONFIG_KEYS, identity_field, leaves, parse_example, pointer
from .emp_binding_shared import field_name
from .user_data import digest, jsonl


def recover_example(raw):
    if not isinstance(raw, str):
        return None, ['not_text']
    text, normalizations = raw.strip(), []
    if text.startswith(('"{', '"[')) and text.endswith('"') and '""' in text:
        text = text[1:-1].replace('""', '"')
        normalizations.append('spreadsheet_csv_quoting')
    else:
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                text = decoded
                normalizations.append('json_encoded_json')
        except ValueError:
            pass
    # Normalize punctuation only outside strings: preserve actual field values.
    normalized, in_string, escaped = [], False, False
    for char in text:
        if in_string:
            normalized.append(char)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            if char in ('\u00a0', '，'):
                replacement = ' ' if char == '\u00a0' else ','
                label = 'outside_nonbreaking_space' if char == '\u00a0' else 'outside_fullwidth_comma'
                if label not in normalizations:
                    normalizations.append(label)
                char = replacement
            normalized.append(char)
    text = ''.join(normalized)
    # Strict tokenization: no bare words/placeholders, comments, or invented strings.
    decoder, tokens, position = json.JSONDecoder(), [], 0
    while position < len(text):
        if text[position].isspace():
            position += 1; continue
        begin = position
        if text[position] in '{}[]:,':
            token = text[position]; position += 1
        elif text[position] == '"':
            try:
                _, position = decoder.raw_decode(text, position)
            except ValueError:
                return None, normalizations
            token = text[begin:position]
        else:
            match = re.match(r'(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)', text[position:])
            if not match:
                return None, normalizations
            token = match.group(0); position += len(token)
        tokens.append((token, begin, position))
    index, recovered = 0, []
    def parse():
        nonlocal index
        if index >= len(tokens):
            raise ValueError('truncated')
        token = tokens[index][0]; index += 1
        if token == '{':
            obj = {}
            if index < len(tokens) and tokens[index][0] == '}':
                index += 1; return obj
            while True:
                if index >= len(tokens) or not tokens[index][0].startswith('"'):
                    raise ValueError('expected quoted property')
                key = json.loads(tokens[index][0]); index += 1
                if tokens[index][0] != ':':
                    raise ValueError('expected colon')
                index += 1; child = parse()
                if key in obj:
                    raise ValueError('duplicate property')
                obj[key] = child
                if index >= len(tokens):
                    raise ValueError('truncated object')
                if tokens[index][0] == '}':
                    index += 1; return obj
                if tokens[index][0] == ',':
                    index += 1
                    if index < len(tokens) and tokens[index][0] == '}':
                        normalizations.append('trailing_comma')
                        index += 1; return obj
                    continue
                if tokens[index][0].startswith('"') and index + 1 < len(tokens) and tokens[index + 1][0] == ':':
                    gap = text[tokens[index - 1][2]:tokens[index][1]]
                    if '\n' not in gap:
                        raise ValueError('missing separator outside narrow newline rule')
                    recovered.append(tokens[index][1]); continue
                raise ValueError('invalid object separator')
        if token == '[':
            result = []
            if index < len(tokens) and tokens[index][0] == ']':
                index += 1; return result
            while True:
                result.append(parse())
                if index >= len(tokens):
                    raise ValueError('truncated array')
                if tokens[index][0] == ']':
                    index += 1; return result
                if tokens[index][0] != ',':
                    raise ValueError('missing array separator')
                index += 1
                if index < len(tokens) and tokens[index][0] == ']':
                    normalizations.append('trailing_comma')
                    index += 1; return result
        if token in '{}[]:,':
            raise ValueError('unexpected punctuation')
        return json.loads(token)
    try:
        result = parse()
        if index != len(tokens) or not isinstance(result, (dict, list)):
            raise ValueError('not a complete single example')
        return result, normalizations + (['newline_property_commas:' + ','.join(map(str, recovered))] if recovered else [])
    except (ValueError, IndexError, TypeError):
        return None, normalizations + ['invalid_json']


def spelling_bindings(request, response):
    bindings = []
    for target, example in leaves(request):
        if not target or not isinstance(target[-1], str):
            continue
        name = field_name(target[-1])
        if name in CONFIG_KEYS or not identity_field(name):
            continue
        matches = [(source, value) for source, value in leaves(response)
                   if source and isinstance(source[-1], str) and field_name(source[-1]) == name
                   and type(value) is type(example) and value not in ('', None, 'string', '--')]
        if len(matches) == 1:
            source, _ = matches[0]
            bindings.append({'target': pointer(target), 'source': pointer(source),
                'basis': 'identity_spelling_case_separator_normalization_same_type_first_record'})
    return bindings


def expand(data, output):
    import openpyxl
    if output.exists():
        raise FileExistsError('use a fresh output directory')
    old_manifest = json.loads((data / 'manifest.json').read_text())
    root = Path(old_manifest['source_root'])
    for name, expected in old_manifest['sources_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError('source changed since previous adaptation')
    for name, expected in old_manifest['canonical_sha256'].items():
        if digest(data / name) != expected:
            raise ValueError('canonical baseline changed')
    workbook = openpyxl.load_workbook(root / 'EMP/data2.xlsx', read_only=True, data_only=True)
    source_rows = {str(r[2]).strip(): r for i, r in enumerate(workbook['出入参信息(新增)'].values, 1) if i > 3 and r[2]}
    workbook.close()
    edges, changes = [], []
    for original in jsonl(data / 'edges.jsonl'):
        edge = deepcopy(original)
        if edge['partition'] != 'development':
            edges.append(edge); continue
        for role in ['producer', 'consumer']:
            record = edge[role + '_catalog']
            if not record:
                continue
            raw = source_rows[edge[role + '_api']]
            for column, key, normalize_key in [(4, 'request_example', 'input_normalizations'), (6, 'response_example', 'output_normalizations')]:
                parsed, repairs = recover_example(raw[column])
                record[key], record[normalize_key] = parsed, repairs
        producer, consumer = edge['producer_catalog'], edge['consumer_catalog']
        reason, bindings = '', []
        if not producer or not consumer:
            reason = 'catalog_record_missing'
        elif producer['request_example'] is None or producer['response_example'] is None or consumer['request_example'] is None:
            reason = 'required_example_not_parseable'
        elif len(json.dumps(consumer['request_example'], ensure_ascii=False)) > 6000 or len(json.dumps(producer['response_example'], ensure_ascii=False)) > 24000:
            reason = 'example_exceeds_fixed_development_context_budget'
        else:
            bindings = spelling_bindings(consumer['request_example'], producer['response_example'])
            if not bindings:
                reason = 'no_unambiguous_typed_identity_binding'
        edge['eligible'], edge['exclusion_reason'], edge['bindings'] = not reason, reason, bindings
        if not original['eligible'] and edge['eligible']:
            changes.append({'edge_uid': edge['uid'], 'task_uid': edge['task_uid'], 'old_reason': original['exclusion_reason'],
                            'new_bindings': bindings, 'original_split': edge['original_split']})
        elif original['eligible']:
            old_pairs = {(b['target'], b['source']) for b in original['bindings']}
            new_pairs = {(b['target'], b['source']) for b in bindings}
            if not old_pairs.issubset(new_pairs):
                raise ValueError('expansion removed a previous binding contract')
        edges.append(edge)
    output.mkdir(parents=True)
    with (output / 'edges.jsonl').open('w') as handle:
        for edge in edges:
            handle.write(json.dumps(edge, ensure_ascii=False) + '\n')
    added = [e for e in edges if any(c['edge_uid'] == e['uid'] for c in changes)]
    with (output / 'additional_edges.jsonl').open('w') as handle:
        for edge in added:
            handle.write(json.dumps(edge, ensure_ascii=False) + '\n')
    report = {'source_data_directory': str(data.resolve()), 'source_edges_sha256': digest(data / 'edges.jsonl'),
        'source_root': str(root), 'sources_sha256': old_manifest['sources_sha256'],
        'adapter_sha256': digest(Path(__file__)), 'field_normalization_sha256': digest(Path(__file__).with_name('emp_binding_shared.py')),
        'scope': 'extra document-derived contracts in already-seen development workflows only',
        'old_development_eligible': sum(e['eligible'] and e['partition'] == 'development' for e in jsonl(data / 'edges.jsonl')),
        'new_development_eligible': sum(e['eligible'] and e['partition'] == 'development' for e in edges),
        'additional_edges': len(added), 'changes': changes, 'reserved_edges_unchanged': True,
        'new_business_goals_verified': 0, 'canonical_sha256': {name: digest(output / name) for name in ['edges.jsonl', 'additional_edges.jsonl']}}
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('sources_sha256',)}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/emp_binding_data_v2'))
    parser.add_argument('--output', type=Path, default=Path('results/emp_binding_expansion_v1'))
    args = parser.parse_args()
    expand(args.data, args.output)


if __name__ == '__main__':
    main()
