"""Learn literal action/response/state kernels for a bounded EMP adapter.

No validator is invoked when forecasting. Missing conditional cells contribute
unknown probability mass, rather than an invented successful transition.
The empirical mass interval is not a statistical transfer certificate.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json

from .emp_binding import BindingEnvironment
from .emp_binding_agent import parse_request
from .emp_binding_tools_v3 import assemble, messages


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def fingerprint(value):
    return sha256(canonical(value).encode()).hexdigest()


class InjectedRejectEnvironment(BindingEnvironment):
    """Explicit test-only behavioral mutation; never a real provider snapshot."""
    def call(self, request):
        event = super().call(request)
        if isinstance(request, dict) and request.get('operation') == 'consumer' and event['accepted']:
            self.responses.pop('consumer', None)
            stored = self.history[-1]
            stored.update(accepted=False, response={'error': 'local_contract_rejected',
                'details': ['injected_contract_rejection']}, errors=['injected_contract_rejection'])
            stored['state_after'] = self.state_hash()
            return deepcopy(stored)
        return event


def environment(edge):
    return InjectedRejectEnvironment(edge) if edge.get('fixture_version') == 'injected_reject_v2' else BindingEnvironment(edge)


def public_contract(edge):
    # Deliberately exclude private source correspondences and dependency labels.
    return {'producer_api': edge['producer_api'], 'consumer_api': edge['consumer_api'],
        'producer_request': edge['producer_catalog']['request_example'],
        'producer_response': edge['producer_catalog']['response_example'],
        'consumer_request': edge['consumer_catalog']['request_example'],
        'targets': sorted(b['target'] for b in edge['bindings']),
        'version': edge.get('fixture_version', 'document_v1')}


def api_key(env, request):
    return fingerprint(api_input(env, request))


def api_input(env, request):
    return {'contract': public_contract(env.edge), 'state': deepcopy(env.responses),
            'request': deepcopy(request)}


def policy_key(env, notes):
    # Includes exact previous actions, public feedback, task text and exposure.
    return fingerprint(messages(env, notes))


def replay_observation(env, request, raw_output=None, parse_error='', sdk_errors=()):
    """Collection/replay only; forecasting never calls this validator."""
    event = env.call(request)
    stored = env.history[-1]
    if sdk_errors:
        stored['response']['sdk_errors'] = list(sdk_errors)
    if parse_error:
        stored['response']['parse_error'] = parse_error
    if raw_output is not None:
        stored['agent_output'] = raw_output
    stored['state_after'] = env.state_hash()
    return event


def api_observation(event):
    response = deepcopy(event['response'])
    if isinstance(response, dict):
        # These are SDK/parser annotations, not API-generated responses.
        response.pop('sdk_errors', None); response.pop('parse_error', None)
    return {'accepted': event['accepted'], 'response': response, 'errors': event['errors'],
            'semantic_scope': event['semantic_scope']}


def apply_learned(env, request, output, raw_output=None, parse_error='', sdk_errors=()):
    """Apply a learned transition without consulting the reference contract."""
    event = {'request': deepcopy(request), **deepcopy(output), 'state_before': env.state_hash()}
    if event['accepted']:
        phase = request.get('operation')
        if phase not in ('producer', 'consumer'):
            raise ValueError('learned transition accepts a structurally invalid operation')
        env.responses[phase] = deepcopy(event['response'])
    env.history.append(event)
    if sdk_errors:
        event['response']['sdk_errors'] = list(sdk_errors)
    if parse_error:
        event['response']['parse_error'] = parse_error
    if raw_output is not None:
        event['agent_output'] = raw_output
    event['state_after'] = env.state_hash()


class StructuredMechanisms:
    """Finite categorical MLE with complete JSON outputs and observable state.

    Exact message support is conservative: no semantic transport is inferred.
    Repeated rendered-prompt cache hits are counted once for policy fitting.
    Repeated identical API inputs are counted once for the deterministic fixture.
    """
    def __init__(self):
        self.policy = defaultdict(Counter); self.api = defaultdict(Counter)
        self.policy_seen, self.api_seen = {}, {}
        self.api_inputs = {}
        self.origins = []

    def update(self, rows, edges, memories):
        for row in rows:
            if row['stage'] not in ('intervention_train', 'repair_context_train', 'api_fixture_train', 'revalidation'):
                raise ValueError('evaluation outcomes cannot fit structural kernels')
            edge = edges[row['edge_uid']]; notes = [memories[u] for u in row.get('exposure', [])]
            env = environment(edge)
            forced = set(row.get('forced_event_indices', []))
            for index, event in enumerate(row['events']):
                # Preserve response property order: the frozen LLM prompt emits
                # fields in that order, even though API input keys are canonical.
                key = api_key(env, event['request']); output = json.dumps(api_observation(event), ensure_ascii=False)
                self.api_inputs[key] = api_input(env, event['request'])
                if key in self.api_seen and self.api_seen[key] != output:
                    raise ValueError('conflicting deterministic fixture outputs')
                if key not in self.api_seen:
                    self.api[key][output] += 1; self.api_seen[key] = output
                if index > 0 and index not in forced and 'agent_output' in event:
                    pkey, raw = policy_key(env, notes), event['agent_output']
                    if pkey in self.policy_seen and self.policy_seen[pkey] != raw:
                        raise ValueError('conflicting deterministic policy outputs')
                    if pkey not in self.policy_seen:
                        self.policy[pkey][raw] += 1; self.policy_seen[pkey] = raw
                response = event['response'] or {}
                replay_observation(env, event['request'], event.get('agent_output'),
                    response.get('parse_error', '') if isinstance(response, dict) else '',
                    response.get('sdk_errors', []) if isinstance(response, dict) else [])
                if env.history[-1] != event:
                    raise ValueError('training transition replay differs')
            self.origins.append({'stage': row['stage'], 'edge_uid': edge['uid'],
                                 'bank_uid': row['bank_uid'], 'trial_sha256': fingerprint(row)})
        return self

    @staticmethod
    def distribution(table, key):
        counts = table.get(key, {})
        n = sum(counts.values())
        return [(label, count / n) for label, count in sorted(counts.items())] if n else []

    def predict(self, edge, notes, max_consumer_calls=2):
        env = BindingEnvironment(edge); request = env.producer_request()
        initial = self.distribution(self.api, api_key(env, request))
        success, first, unknown, expected_calls, terminal = 0., 0., 0., 0., []
        if not initial:
            return {'final_success': None, 'first_success': None, 'supported': False,
                'known_success_mass': 0., 'unknown_mass': 1., 'mass_interval': [0., 1.],
                'expected_consumer_calls': None, 'certificate': False, 'terminal': []}
        active = []
        for encoded, mass in initial:
            output = json.loads(encoded); clone = deepcopy(env)
            apply_learned(clone, request, output)
            if output['accepted']:
                active.append((clone, mass))
            else:
                terminal.append({'mass': mass, 'success': False, 'consumer_calls': 0})
        for attempt in range(max_consumer_calls):
            remaining = []
            for state, mass in active:
                actions = self.distribution(self.policy, policy_key(state, notes))
                if not actions:
                    unknown += mass; continue
                expected_calls += mass
                for raw, action_probability in actions:
                    action, error = parse_request(raw); payload, sdk_errors = assemble(state, action)
                    outputs = self.distribution(self.api, api_key(state, payload))
                    branch = mass * action_probability
                    if not outputs:
                        unknown += branch; continue
                    for encoded, output_probability in outputs:
                        probability = branch * output_probability; output = json.loads(encoded)
                        clone = deepcopy(state)
                        apply_learned(clone, payload, output, raw, error, sdk_errors)
                        if output['accepted']:
                            success += probability; first += probability if attempt == 0 else 0.
                            terminal.append({'mass': probability, 'success': True, 'consumer_calls': attempt + 1})
                        elif attempt + 1 == max_consumer_calls:
                            terminal.append({'mass': probability, 'success': False, 'consumer_calls': attempt + 1})
                        else:
                            remaining.append((clone, probability))
            active = remaining
        if abs(success + sum(t['mass'] for t in terminal if not t['success']) + unknown - 1.) > 1e-9:
            raise ValueError('rollout probability mass lost')
        return {'final_success': success if unknown < 1e-12 else None,
            'first_success': first if unknown < 1e-12 else None, 'supported': unknown < 1e-12,
            'known_success_mass': success, 'unknown_mass': unknown,
            'mass_interval': [success, min(1., success + unknown)],
            'expected_consumer_calls': expected_calls if unknown < 1e-12 else None,
            'certificate': False, 'terminal': terminal}

    def to_dict(self):
        return {'policy': {k: dict(v) for k, v in self.policy.items()},
            'api': {k: dict(v) for k, v in self.api.items()}, 'api_inputs': self.api_inputs, 'origins': self.origins,
            'representation': 'literal structured action and response/state categorical kernels',
            'support': 'exact public messages and API state/request cells', 'certificate': False}

    @classmethod
    def from_dict(cls, value):
        model = cls()
        for attribute in ('policy', 'api'):
            setattr(model, attribute, defaultdict(Counter, {k: Counter(v) for k, v in value[attribute].items()}))
        model.policy_seen = {k: next(iter(v)) for k, v in model.policy.items()}
        model.api_seen = {k: next(iter(v)) for k, v in model.api.items()}
        model.origins = deepcopy(value['origins'])
        model.api_inputs = deepcopy(value['api_inputs'])
        return model


def compare_api_kernels(source, target):
    """Common supported cells only; empirical TV, never a confidence certificate."""
    def align(model):
        result = {}
        for key, payload in model.api_inputs.items():
            aligned = deepcopy(payload); aligned['contract'].pop('version', None)
            result[fingerprint(aligned)] = key
        return result
    source_inputs, target_inputs = align(source), align(target)
    common = sorted(set(source_inputs) & set(target_inputs)); rows = []
    for key in common:
        a = dict(source.distribution(source.api, source_inputs[key]))
        b = dict(target.distribution(target.api, target_inputs[key]))
        tv = .5 * sum(abs(a.get(label, 0.) - b.get(label, 0.)) for label in set(a) | set(b))
        rows.append({'input_cell': key, 'empirical_tv': tv, 'certificate': False})
    return {'common_cells': len(common), 'cells': rows,
            'decision': 'inconclusive' if not common else 'empirical_change' if any(r['empirical_tv'] > 0 for r in rows) else 'no_change_on_observed_cells',
            'unobserved_inputs': 'unknown', 'certificate': False}
