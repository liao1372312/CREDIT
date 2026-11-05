"""State/graph features and independent validation for the v3 node heads.

The paper's masked c/r likelihood and calibrated product are unchanged.
Prediction features contain only the frozen skeleton, initial state and public
successful-operation postconditions. Training uses observed pre-invocation
states. Execution seeds, future events and simulator probabilities are absent.
This module is separate so historical source hashes remain reproducible.
"""
from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.isotonic import IsotonicRegression
import torch

from .v3.outcome import Heads, OutcomeModel, text_encoding


class ValidatedOutcomeModel(OutcomeModel):
    """Choose weights on separate context clusters; calibrate only afterwards."""

    def pack_observations(self, rows):
        items, labels, recovery = [], [], []
        for e in rows:
            lookup = {n.uid: n for n in e.context.nodes}
            for o in e.observations:
                items.append((e.context, lookup[o.node], e.exposure))
                labels.append(('ok', 'agent', 'service', 'amb').index(o.label))
                recovery.append(float(o.recovered))
        if not items:
            raise ValueError('node observations required')
        return self._pack(items), torch.tensor(labels), torch.tensor(recovery)

    @staticmethod
    def compress(packed, labels, recovery):
        """Aggregate identical predictor inputs without changing the likelihood."""
        signature = np.concatenate([packed[i].numpy() for i in (0, 2, 4, 5)], axis=1)
        _, selected, inverse = np.unique(signature, axis=0, return_index=True, return_inverse=True)
        grouped = list(packed)
        for i in (0, 2, 4, 5):
            grouped[i] = packed[i][selected]
        count = len(selected)
        targets = labels.numpy()
        ok = np.bincount(inverse, weights=targets == 0, minlength=count)
        agent = np.bincount(inverse, weights=targets == 1, minlength=count)
        repaired = np.bincount(inverse, weights=(targets == 1)*recovery.numpy(), minlength=count)
        return tuple(grouped), torch.tensor(np.column_stack((ok, agent, repaired)), dtype=torch.float32)

    @staticmethod
    def compressed_loss(logits, counts, penalty=0):
        from torch.nn.functional import softplus
        ok, agent, repaired = counts.T
        numerator = (softplus(-logits[:, 0])*ok + softplus(logits[:, 0])*agent
                     + softplus(-logits[:, 1])*repaired
                     + softplus(logits[:, 1])*(agent-repaired)).sum()
        return numerator / (ok+agent).sum().clamp_min(1) + penalty

    def fit(self, episodes, calibration, *, validation, epochs=1200,
            learning_rate=.01, gate_penalty=.001, patience=12, interval=10):
        if min(epochs, patience, interval) < 1 or learning_rate <= 0 or gate_penalty < 0:
            raise ValueError('invalid optimizer configuration')
        ids = [{e.context.uid for e in rows} for rows in (episodes, validation, calibration)]
        if any(not rows for rows in (episodes, validation, calibration)):
            raise ValueError('training, validation and calibration are required')
        if any(ids[i] & ids[j] for i, j in combinations(range(3), 2)):
            raise ValueError('training/validation/calibration context overlap')
        seeds = [{e.context.seed for e in rows} for rows in (episodes, validation, calibration)]
        if any(seeds[i] & seeds[j] for i, j in combinations(range(3), 2)):
            raise ValueError('training/validation/calibration noise overlap')
        if any(not e.randomized or not e.replay_valid or e.assignment_probability is None
               or not 0 < e.assignment_probability <= 1
               for rows in (episodes, validation, calibration) for e in rows):
            raise ValueError('replay-valid randomized evidence required')

        train, labels, recovery = self.pack_observations(episodes)
        val, val_labels, val_recovery = self.pack_observations(validation)
        if not ((labels == 0) | (labels == 1)).any():
            raise ValueError('no attributable observations')
        train, train_counts = self.compress(train, labels, recovery)
        val, val_counts = self.compress(val, val_labels, val_recovery)
        optimizer = torch.optim.Adam(self.heads.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=.5, patience=3, min_lr=1e-5)
        best, best_state, best_iteration, stale = float('inf'), None, 0, 0
        trajectory = []
        with torch.no_grad():
            initial_validation = float(self.compressed_loss(self.heads(*val), val_counts))
        stopped = False
        for iteration in range(1, epochs + 1):
            optimizer.zero_grad()
            loss = self.compressed_loss(self.heads(*train), train_counts,
                                        gate_penalty * self.heads.gates.abs().sum())
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite model loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.heads.parameters(), 5.)
            optimizer.step()
            if iteration % interval == 0 or iteration == epochs:
                with torch.no_grad():
                    value = float(self.compressed_loss(self.heads(*val), val_counts))
                if not np.isfinite(value):
                    raise FloatingPointError('nonfinite validation loss')
                trajectory.append({'epoch': iteration, 'training_loss': float(loss.detach()),
                                   'validation_nll': value, 'learning_rate': optimizer.param_groups[0]['lr']})
                if value < best - 1e-4:
                    best = value
                    best_state = {k: v.detach().clone() for k, v in self.heads.state_dict().items()}
                    best_iteration, stale = iteration, 0
                else:
                    stale += 1
                scheduler.step(value)
                if stale >= patience:
                    stopped = True
                    break
        self.heads.load_state_dict(best_state)
        self.training_ids = ids[0]
        support = Counter()
        for e in episodes:
            lookup = {n.uid: n for n in e.context.nodes}
            for o in e.observations:
                if o.label in ('ok', 'agent'):
                    n = lookup[o.node]
                    support[n.role.key, n.version] += 1
        self.support = dict(support)
        raw = self.predict_many([(e.context, e.exposure) for e in calibration], calibrated=False)
        targets = np.asarray([e.outcome for e in calibration])
        self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(raw, targets)
        calibrated = self.calibrator.predict(raw)
        self.fit_report = {
            'iterations': iteration, 'selected_epoch': best_iteration,
            'loss': trajectory[-1]['training_loss'], 'selected_validation_nll': best,
            'initial_validation_nll': initial_validation,
            'stopping_criterion': '12 consecutive validation checks without a 1e-4 NLL improvement',
            'stopping_criterion_met': stopped,
            'optimizer_stationarity_claimed': False,
            'randomized_episodes': len(episodes), 'validation_episodes': len(validation),
            'calibration_episodes': len(calibration),
            'calibration_context_clusters': len(ids[2]),
            'calibration_successes': int(targets.sum()),
            'masked_service_nodes': int((labels == 2).sum()),
            'masked_ambiguous_nodes': int((labels == 3).sum()),
            'compressed_training_rows': len(train_counts),
            'original_training_node_rows': len(labels),
            'calibration_brier': float(np.mean((calibrated-targets)**2)),
            'calibration_mae': float(np.mean(np.abs(calibrated-targets))),
            'trajectory': trajectory,
        }
        return self

    def save(self, path):
        if self.calibrator is None:
            x = y = None
        elif hasattr(self.calibrator, 'X_thresholds_'):
            x, y = self.calibrator.X_thresholds_, self.calibrator.y_thresholds_
        else:
            x, y = self.calibrator.x, self.calibrator.y
        torch.save({'state_dict':self.heads.state_dict(), 'role_indices':self.roles,
                    'memory_ids':[m.uid for m in self.memories],
                    'encoder':'signed-hash-bag-v1', 'memory_dimensions':self.encoding.shape[1],
                    'feature_dimensions':self.heads.baseline[0].in_features,
                    'calibration_x':x, 'calibration_y':y, 'fit_report':self.fit_report,
                    'model_class':type(self).__name__, 'model_revision':'conditional-state-graph-validation-v2'}, path)


class StateAwareOutcomeModel(ValidatedOutcomeModel):
    """Shared heads conditioned on public state and graph-neighborhood features.

    The product queries the node state on a successful prefix. This is the
    joint-success chain rule when public postconditions determine that state;
    uncertain service/agent probabilities still require held-out calibration.
    No hidden response or actor probability is encoded.
    """

    def __init__(self, memories, training_contexts, *, seed=0):
        super().__init__(memories, training_contexts, seed=seed)
        # More dimensions reduce collisions between the eight memory texts.
        self.encoding = torch.tensor(np.asarray([text_encoding(m.content, 32) for m in memories]))
        self.pair_encoding = (torch.stack([
            torch.cat((self.encoding[i]+self.encoding[j],
                       torch.abs(self.encoding[i]-self.encoding[j]), self.encoding[i]*self.encoding[j]))
            for i, j in self.pairs]) if self.pairs else torch.empty((0, 96)))
        torch.manual_seed(seed)
        self.heads = Heads(len(self.roles), feature_size=32, memory_size=32, hidden=32)

    @staticmethod
    def successful_prefix_state(c, node):
        # Public state postconditions of the local adapter. A new record clears
        # derived fields/authorization; transformation clears authorization.
        from .v3.local import ROLE_NAMES
        state = c.initial_state
        for before in c.nodes:
            if before.uid == node.uid:
                return state
            role = ROLE_NAMES[before.role.key]
            if role == 'lookup':
                state = 1
            elif role == 'transform':
                state = (state | 3) & ~4
            elif role == 'authorize':
                state |= 7
            elif role == 'submit':
                state |= 8
        raise ValueError('node not in skeleton')

    def _pack(self, items):
        packed = list(super()._pack(items))
        # The local adapter publicly defines bit fields for record/derived/auth/receipt.
        # No realized intermediate state is used during prospective prediction.
        extra = []
        for c, node, _ in items:
            position = next(i for i, n in enumerate(c.nodes) if n.uid == node.uid)
            prefix = Counter(n.role.key for n in c.nodes[:position])
            suffix = Counter(n.role.key for n in c.nodes[position+1:])
            role_keys = sorted(self.roles)
            if len(role_keys) != 4:
                raise ValueError('stateful adapter expects four registered operation roles')
            extra.append(
                [float(bool(self.successful_prefix_state(c,node) & (1 << i))) for i in range(4)]
                + [float(bool(c.initial_state & (1 << i))) for i in range(4)]
                + [prefix[k]/8 for k in role_keys] + [suffix[k]/8 for k in role_keys]
                + [len(c.nodes)/8, position/max(len(c.nodes)-1, 1),
                   sum(b == node.uid for a, b in c.edges)/4,
                   sum(a == node.uid for a, b in c.edges)/4])
        packed[0] = torch.cat((packed[0], torch.tensor(np.asarray(extra), dtype=torch.float32)), dim=1)
        return tuple(packed)

    def pack_observations(self, rows):
        packed, labels, recovery = super().pack_observations(rows)
        offset = 0
        for e in rows:
            first = {event['node']:event for event in e.events if event['attempt'] == 0}
            for o in e.observations:
                if o.node in first:
                    state = first[o.node]['state_before']
                    packed[0][offset,12:16] = torch.tensor([
                        float(bool(state & (1 << i))) for i in range(4)])
                elif o.label != 'amb':
                    raise ValueError('observed pre-invocation state required for attributable labels')
                offset += 1
        return packed, labels, recovery


class StateAwareNoPairs(StateAwareOutcomeModel):
    def _pack(self, items):
        packed = list(super()._pack(items))
        packed[4] = torch.zeros_like(packed[4])
        return tuple(packed)


def load_model(path, memories, contexts):
    """Restore the versioned feature encoder and calibration with frozen weights."""
    from .v3.audit import SavedCalibration
    data = torch.load(path, map_location='cpu', weights_only=False)
    classes = {c.__name__: c for c in (OutcomeModel, ValidatedOutcomeModel,
                                      StateAwareOutcomeModel, StateAwareNoPairs)}
    cls = classes[data.get('model_class', 'OutcomeModel')]
    model = cls(memories, contexts)
    if model.roles != data['role_indices'] or [m.uid for m in memories] != data['memory_ids']:
        raise ValueError('model role/memory registry changed')
    model.heads.load_state_dict(data['state_dict'])
    model.calibrator = SavedCalibration(data['calibration_x'], data['calibration_y'])
    model.fit_report = data['fit_report']
    return model
