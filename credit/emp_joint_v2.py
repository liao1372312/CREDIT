"""Shared four-term learner over finite, complete JSON action/output alphabets.

The alphabets come exclusively from training observations. Linear shared
features replace the manuscript's general semantic encoders; unsupported API
contracts abstain. This is an explicit bounded implementation, not a general
Web-service response generator or a statistical transfer certificate.
"""

from collections import defaultdict
from copy import deepcopy
from itertools import combinations
import json

import numpy as np

from .emp_binding import BindingEnvironment, get_path, leaves, pointer, tokens
from .emp_binding_agent import parse_request
from .emp_binding_shared import field_name
from .emp_binding_tools_v3 import assemble
from .emp_structural import StructuredMechanisms, api_observation, apply_learned, fingerprint, public_contract, replay_observation


def canonical_action(env, raw):
    action, _ = parse_request(raw)
    if isinstance(action.get('bindings'), dict):
        targets = {b['target'] for b in env.edge['bindings']}
        aliases = defaultdict(list)
        for target in targets:
            aliases[field_name(tokens(target)[-1])].append(target)
        rewritten = {}
        for key, value in action['bindings'].items():
            target = key
            if isinstance(key, str) and not key.startswith('/') and len(aliases[field_name(key)]) == 1:
                target = aliases[field_name(key)][0]
            if target in rewritten:
                # Preserve the invalid duplicate-alias semantics.
                return json.dumps({'operation': 'invalid_duplicate_alias'}, sort_keys=True)
            rewritten[target] = value
        action = {**action, 'bindings': rewritten}
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def frame_key(edge):
    return fingerprint(public_contract(edge))


def request_features(env, request):
    targets = [b['target'] for b in env.edge['bindings']]
    fields = {pointer(path): value for path, value in leaves(env.responses['producer'])} if 'producer' in env.responses else {}
    matches, present, typed = [], [], []
    mapping = request.get('bindings', {}) if isinstance(request, dict) else {}
    for target in targets:
        source = mapping.get(target, {}).get('pointer', '')
        exists = source in fields
        matches.append(float(exists and bool(tokens(source)) and field_name(tokens(source)[-1]) == field_name(tokens(target)[-1])))
        present.append(float(exists))
        try:
            typed.append(float(exists and type(fields[source]) is type(get_path(env.edge['consumer_catalog']['request_example'], target))))
        except (KeyError, TypeError, ValueError, IndexError):
            typed.append(0.)
    phase = request.get('operation') if isinstance(request, dict) else None
    return np.asarray([1., float(phase == 'producer'), float(phase == 'consumer'),
        float(phase not in ('producer', 'consumer')), float('producer' in env.responses),
        float('consumer' in env.responses), np.mean(matches), np.mean(present), np.mean(typed),
        min(len(targets) / 16, 1.), max(t.count('/') for t in targets) / 6,
        min(len(fields) / 100, 1.), float(request.get('api_id') in (env.edge['producer_api'], env.edge['consumer_api']))])


def policy_features(env, notes, raw):
    action, _ = parse_request(raw); request, errors = assemble(env, action)
    features = request_features(env, request)
    unary, pair = np.zeros(3), np.zeros(3)
    targets = {field_name(tokens(b['target'])[-1]) for b in env.edge['bindings']}
    chosen = {(field_name(tokens(t)[-1]), s.get('pointer', '')) for t, s in request.get('bindings', {}).items()}
    routes = []
    for note in notes:
        relevant = {(field_name(tokens(t)[-1]), s.get('pointer', '')) for t, s in note['verified_mapping'].items()
                    if field_name(tokens(t)[-1]) in targets}
        routes.append(relevant)
        unary += [1., len(chosen & relevant) / max(1, len(targets)), len(relevant) / max(1, len(targets))]
    for a, b in combinations(routes, 2):
        pair += [1., len(a & b) / max(1, len(targets)), float(a != b)]
    feedback = env.history[-1] if len(env.history) > 1 else {}
    return np.concatenate([features, [features[10] * (1 - features[7]),
        min(max(len(env.history) - 1, 0), 2) / 2 * features[6],
        float(bool((feedback.get('response') or {}).get('sdk_errors'))) * features[6],
        float(bool(errors))], unary, pair])


def output_features(env, request, output):
    x = request_features(env, request)
    errors = output['errors']; n = max(1, len(env.edge['bindings']))
    label = np.asarray([float(output['accepted']),
        sum(e.startswith('binding_value:') or e.startswith('missing_binding:') for e in errors) / n,
        sum(e.startswith('binding_provenance:') for e in errors) / n,
        float('invalid_operation' in errors)])
    return np.outer(x, label).ravel()


class JointStructuredMechanisms:
    def __init__(self, literal, action_catalog, policy_weights=None, api_weights=None):
        import torch
        self.torch = torch; self.literal = literal; self.catalog = action_catalog
        self.policy_weights = np.zeros(23) if policy_weights is None else np.asarray(policy_weights)
        self.api_weights = np.zeros(52) if api_weights is None else np.asarray(api_weights)
        self.outputs, self.output_phases = defaultdict(dict), defaultdict(lambda: defaultdict(set))
        for key, counts in literal.api.items():
            frame = fingerprint(literal.api_inputs[key]['contract'])
            phase = literal.api_inputs[key]['request'].get('operation')
            for encoded in counts:
                self.outputs[frame][encoded] = json.loads(encoded)
                self.output_phases[frame][phase].add(encoded)
        self.cache = {}

    def matrices(self, edge, notes):
        frame = frame_key(edge)
        cache_key = frame, tuple(sorted(n['uid'] for n in notes))
        if cache_key in self.cache:
            return self.cache[cache_key]
        if frame not in self.catalog:
            return None
        initial = BindingEnvironment(edge)
        # Actual producer transition is learned from data, not obtained by call().
        from .emp_structural import api_key
        known = self.literal.distribution(self.literal.api, api_key(initial, initial.producer_request()))
        if len(known) != 1 or not json.loads(known[0][0])['accepted']:
            return None
        apply_learned(initial, initial.producer_request(), json.loads(known[0][0]))
        actions = self.catalog[frame]; encoded_outputs = sorted(self.outputs[frame])
        outputs = [self.outputs[frame][x] for x in encoded_outputs]
        policy, api, allowed, repair = [], [], [], []
        for raw in actions:
            action, error = parse_request(raw); request, sdk_errors = assemble(initial, action)
            policy.append(policy_features(initial, notes, raw))
            api.append([output_features(initial, request, output) for output in outputs])
            allowed.append([encoded in self.output_phases[frame][request.get('operation')] for encoded in encoded_outputs])
            contexts = []
            for output in outputs:
                state = deepcopy(initial)
                # Accepted branches are absorbing and never evaluate a repair.
                if not output['accepted']:
                    apply_learned(state, request, output, raw, error, sdk_errors)
                contexts.append([policy_features(state, notes, other) for other in actions])
            repair.append(contexts)
        result = tuple(np.asarray(x) for x in [policy, api, allowed, repair,
            [float(o['accepted']) for o in outputs]])
        if not result[2].any(axis=1).all():
            return None
        self.cache[cache_key] = result
        return result

    def forward(self, matrices, pw, kw):
        torch = self.torch
        p, k, mask, repair, accepted = matrices
        p, k, repair, accepted = [torch.as_tensor(x, dtype=torch.float64) for x in [p, k, repair, accepted]]
        mask = torch.as_tensor(mask, dtype=torch.bool)
        policy = (p @ pw).softmax(-1)
        kernel = (k @ kw).masked_fill(~mask, -1e9).softmax(-1)
        second_policy = (repair @ pw).softmax(-1)
        first = kernel @ accepted
        after_rejection = (second_policy @ first)
        final = policy @ (first + (kernel * (1 - accepted) * after_rejection).sum(-1))
        return policy, kernel, policy @ first, final

    def predict(self, edge, notes):
        matrices = self.matrices(edge, notes)
        if matrices is None:
            return {'final_success': None, 'first_success': None, 'supported': False,
                'known_success_mass': 0., 'unknown_mass': 1., 'expected_consumer_calls': None,
                'mass_interval': [0., 1.], 'certificate': False}
        with self.torch.no_grad():
            _, _, first, final = self.forward(matrices, self.torch.tensor(self.policy_weights), self.torch.tensor(self.api_weights))
        return {'final_success': float(final), 'first_success': float(first), 'supported': True,
            'known_success_mass': float(final), 'unknown_mass': 0., 'expected_consumer_calls': float(2 - first),
            'mass_interval': [float(final), float(final)], 'certificate': False}

    def api_distribution(self, env, request):
        frame = frame_key(env.edge)
        if frame not in self.outputs:
            return {}
        labels = sorted(self.output_phases[frame].get(request.get('operation'), []))
        if not labels:
            return {}
        features = np.asarray([output_features(env, request, self.outputs[frame][x]) for x in labels])
        z = features @ self.api_weights; p = np.exp(z - z.max()); p /= p.sum()
        return dict(zip(labels, map(float, p)))

    def to_dict(self):
        return {'policy_weights': self.policy_weights.tolist(), 'api_weights': self.api_weights.tolist(),
            'action_catalog': self.catalog, 'representation': 'shared linear public request/action/response features; canonical complete finite JSON alphabets',
            'limits': 'API-contract support only; finite observed action/output alphabet; empirical, no certificate'}


def fit_joint(rows, api_rows, edges, memories, *, outcome_weight=.2, group_weight=.002, epochs=250):
    import torch
    torch.set_num_threads(2); torch.manual_seed(42)
    literal = StructuredMechanisms().update(rows + api_rows, edges, memories)
    catalogs = defaultdict(set); policy_examples, api_examples, outcomes = [], [], {}
    api_seen, policy_seen = set(), set()
    from .emp_structural import api_key, environment, policy_key
    for row in rows + api_rows:
        edge = edges[row['edge_uid']]; env = environment(edge)
        notes = [memories[u] for u in row.get('exposure', [])]
        for i, event in enumerate(row['events']):
            key = api_key(env, event['request'])
            if key not in api_seen:
                api_examples.append((deepcopy(env), event['request'], json.dumps(api_observation(event), ensure_ascii=False)))
                api_seen.add(key)
            if i > 0 and i not in row.get('forced_event_indices', []) and 'agent_output' in event:
                raw = canonical_action(env, event['agent_output'])
                catalogs[frame_key(edge)].add(raw)
                pkey = policy_key(env, notes)
                if pkey not in policy_seen:
                    policy_examples.append((deepcopy(env), notes, raw)); policy_seen.add(pkey)
            response = event['response'] or {}
            replay_observation(env, event['request'], event.get('agent_output'),
                response.get('parse_error', '') if isinstance(response, dict) else '',
                response.get('sdk_errors', []) if isinstance(response, dict) else [])
        if row['stage'] == 'intervention_train':
            outcomes[row['edge_uid'], tuple(sorted(row['exposure']))] = row
    model = JointStructuredMechanisms(literal, {k: sorted(v) for k, v in catalogs.items()})
    p_data, k_data, y_data = [], [], []
    for env, notes, raw in policy_examples:
        catalog = model.catalog[frame_key(env.edge)]
        p_data.append((torch.tensor(np.asarray([policy_features(env, notes, x) for x in catalog])), catalog.index(raw)))
    for env, request, encoded in api_examples:
        frame = frame_key(env.edge); labels = sorted(model.output_phases[frame][request.get('operation')])
        k_data.append((torch.tensor(np.asarray([output_features(env, request, model.outputs[frame][x]) for x in labels])), labels.index(encoded)))
    group_counts = defaultdict(int)
    for row in outcomes.values():
        group_counts[row['workflow_group']] += 1
    for row in outcomes.values():
        matrix = model.matrices(edges[row['edge_uid']], [memories[u] for u in row['exposure']])
        if matrix is None:
            raise ValueError('training outcome lacks complete action/API support')
        y_data.append((matrix, float(row['final']['binding_contract_success']),
            1 / (len(group_counts) * group_counts[row['workflow_group']])))
    pw = torch.nn.Parameter(torch.zeros(23, dtype=torch.float64))
    kw = torch.nn.Parameter(torch.zeros(52, dtype=torch.float64))
    optimizer = torch.optim.Adam([pw, kw], lr=.04); curve = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        decision = torch.stack([-torch.log_softmax(x @ pw, -1)[label] for x, label in p_data]).mean()
        api = torch.stack([-torch.log_softmax(x @ kw, -1)[label] for x, label in k_data]).mean()
        outcome = sum(weight * torch.nn.functional.binary_cross_entropy(model.forward(matrix, pw, kw)[3].clamp(1e-8, 1-1e-8), torch.tensor(y, dtype=torch.float64)) for matrix, y, weight in y_data)
        group = torch.linalg.vector_norm(pw[-3:])
        loss = decision + api + outcome_weight * outcome + group_weight * group + .0001 * (pw.square().sum() + kw.square().sum())
        loss.backward(); optimizer.step()
        if epoch % 25 == 0 or epoch + 1 == epochs:
            curve.append({'epoch': epoch + 1, 'decision_nll': float(decision.detach()), 'API_nll': float(api.detach()),
                'outcome_bce': float(outcome.detach()), 'group_norm': float(group.detach()), 'loss': float(loss.detach())})
    model.policy_weights = pw.detach().numpy(); model.api_weights = kw.detach().numpy()
    return model, {'parameters': 75, 'policy_contexts': len(p_data), 'API_inputs': len(k_data),
        'distinct_interventional_exposures': len(y_data), 'outcome_weight': outcome_weight,
        'group_weight': group_weight, 'epochs': epochs, 'curve': curve,
        'sampling': 'deduplicated deterministic prompt/API inputs; no independent-repeat claim',
        'outcome_weighting': 'uniform training workflows then distinct edge/exposure cells'}, literal
