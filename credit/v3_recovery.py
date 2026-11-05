"""Versioned memory-guided recovery executor for the finite stateful API.

The original adapter retries the same operation after an attributable agent
error. That cannot recover a submit whose upstream authorization is missing.
This executor lets the declared ``repair_rule`` schedule the public upstream
roles needed to restore submit prerequisites, then retries submit. Service
failures are still not retried and every physical call is counted.
"""
from dataclasses import asdict
from hashlib import sha256
import json

from .stateful import StateTask, run_episode
from .v3.local import ROLE_NAMES, stable_seed, valid_request
from .v3.types import Episode, NodeObservation, safe_replay


class RecoveryExecutor:
    def __init__(self, repair_budget=1):
        if repair_budget != 1:
            raise ValueError('the frozen recovery protocol uses one recovery plan')
        self.repair_budget = repair_budget

    @staticmethod
    def _invoke(context, index, attempt, ordinal, role, state, exposure):
        family, epoch = context.stratum.split('/')
        task = StateTask(f'{context.uid}:node-{index}:attempt-{attempt}:recovery-{ordinal}',
                         family, (role,), epoch,
                         stable_seed(context.seed, index, attempt, ordinal, role),
                         'v3-memory-recovery', state)
        return run_episode(task, exposure).steps[0]

    def run(self, context, exposure, *, slots=None, prefix=None, start=0):
        if slots is not None and set(slots.exposure) != set(exposure):
            raise ValueError('reserved slots differ from actual intervention')
        if prefix is not None or start:
            raise ValueError('recovery executor uses full paired re-execution')
        family, epoch = context.stratum.split('/')
        checkpoint = sha256(json.dumps({'state': context.initial_state,
                                         'versions': [n.version for n in context.nodes]},
                                        sort_keys=True).encode()).hexdigest()
        state, observations, events, succeeded = context.initial_state, [], [], []
        failed_by_role = {}
        calls = 0
        for index, node in enumerate(context.nodes):
            role = ROLE_NAMES[node.role.key]
            if state & 8:
                observations.append(NodeObservation(node.uid, 'amb'))
                succeeded.append(False)
                continue
            first = self._invoke(context, index, 0, 0, role, state, exposure)
            calls += 1
            attributable = valid_request(role, state, first.request_mode, family, epoch)
            first_label = ('ok' if first.response_status == 'ok' else 'service') if attributable else 'agent'
            events.append({'node': node.uid, 'attempt': 0, 'recovery_step': 0,
                           **asdict(first), 'version': node.version})
            state = first.state_after
            node_ok = first.response_status == 'ok'
            recovered = False
            # Only an attributable agent error enters the paper's recovery head.
            # The repair memory can schedule missing public producer roles.
            if (not node_ok and first_label == 'agent' and 'repair_rule' in exposure
                    and role == 'submit'):
                recovery_roles = []
                if not state & 1:
                    recovery_roles.append('lookup')
                if state & 3 != 3:
                    recovery_roles.append('transform')
                if state & 7 != 7:
                    recovery_roles.append('authorize')
                recovery_roles.append('submit')
                for ordinal, recovery_role in enumerate(recovery_roles, 1):
                    step = self._invoke(context, index, 1, ordinal, recovery_role, state, exposure)
                    calls += 1
                    events.append({'node': node.uid, 'attempt': 1, 'recovery_step': ordinal,
                                   **asdict(step), 'version': node.version})
                    state = step.state_after
                    if step.response_status != 'ok':
                        break
                    # A successful compensating execution replaces the latest
                    # failed required occurrence of that public role. Its
                    # original first-attempt label remains in observations.
                    prior = failed_by_role.get(recovery_role)
                    if prior is not None:
                        succeeded[prior] = True
                        failed_by_role.pop(recovery_role, None)
                node_ok = bool(state & 8)
                recovered = node_ok
            observations.append(NodeObservation(node.uid, first_label, recovered))
            succeeded.append(node_ok)
            if not node_ok:
                failed_by_role[role] = len(succeeded) - 1
        terminal = sha256(json.dumps({'state': state, 'events': events}, sort_keys=True).encode()).hexdigest()
        restored_state = context.initial_state
        restored = sha256(json.dumps({'state': restored_state,
                                       'versions': [n.version for n in context.nodes]},
                                      sort_keys=True).encode()).hexdigest()
        replay_valid = all(safe_replay(n.role, mode='sandbox', version_pinned=True,
                                       snapshot_restored=True, checkpoint_hash=checkpoint,
                                       restored_hash=restored) for n in context.nodes)
        return Episode(context, tuple(exposure), tuple(observations),
                       bool(state & 8) and all(succeeded), calls, terminal,
                       checkpoint, restored, replay_valid, events=tuple(events), reused_calls=0)

    def replay(self, context, exposure, reference, *, slots=None):
        return self.run(context, exposure, slots=slots)
