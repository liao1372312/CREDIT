"""Eq. heads/trainloss/compose: shared correctness and recovery heads.

Only valid randomized episodes update exposure parameters. Logged traces
may initialize the baseline separately. Service/ambiguous first attempts
never supply correctness or recovery targets. Calibration is held out.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations, product
import re

import numpy as np
from sklearn.isotonic import IsotonicRegression
import torch
from torch import nn
from torch.nn import functional as functional

from .factorial import characters


def text_encoding(text, dimensions=8):
    """Shared deterministic content encoding; adapters can replace this encoder."""
    vector = np.zeros(dimensions, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        digest = sha256(word.encode()).digest()
        vector[int.from_bytes(digest[:2], "big") % dimensions] += 1 if digest[2] & 1 else -1
    return vector / max(float(np.linalg.norm(vector)), 1.0)


class Heads(nn.Module):
    def __init__(self, role_types, feature_size=12, memory_size=8, hidden=16):
        super().__init__()
        self.baseline = nn.Sequential(nn.Linear(feature_size, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.unary = nn.Sequential(nn.Linear(feature_size + memory_size, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.pair = nn.Sequential(nn.Linear(feature_size + 3 * memory_size, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.gates = nn.Parameter(torch.ones(max(1, role_types) * 2, 2))

    def forward(self, x, memories, unary_mask, pair_features, pair_mask, gate_index,
                *, baseline_only=False):
        eta = self.baseline(x)
        if baseline_only:
            return eta
        n, m = len(x), len(memories)
        features = torch.cat((x[:, None, :].expand(n, m, -1), memories[None].expand(n, m, -1)), dim=-1)
        eta = eta + (self.unary(features) * unary_mask[:, :, None]).sum(dim=1)
        if len(pair_features):
            p = len(pair_features)
            features = torch.cat((x[:, None, :].expand(n, p, -1),
                                  pair_features[None].expand(n, p, -1)), dim=-1)
            eta = eta + (self.pair(features) * self.gates[gate_index]
                         * pair_mask[:, :, None]).sum(dim=1)
        return eta


class OutcomeModel:
    def __init__(self, memories, training_contexts, *, seed=0):
        self.memories = tuple(memories)
        if len({m.uid for m in memories}) != len(memories):
            raise ValueError("memory ids must be unique")
        self.roles = {key: i for i, key in enumerate(sorted({n.role.key for c in training_contexts for n in c.nodes}))}
        self.pairs = tuple(combinations(range(len(memories)), 2))
        self.encoding = torch.tensor(np.asarray([text_encoding(m.content) for m in memories]))
        self.pair_encoding = (torch.stack([torch.cat((self.encoding[i] + self.encoding[j],
                                                     torch.abs(self.encoding[i] - self.encoding[j]),
                                                     self.encoding[i] * self.encoding[j])) for i, j in self.pairs])
                              if self.pairs else torch.empty((0, 24)))
        torch.manual_seed(seed)
        self.heads = Heads(len(self.roles))
        self.calibrator = None
        self.fit_report = {}
        self.support = {}
        self.training_ids = set()

    def _pack(self, items):
        features, unaries, pairs, indices = [], [], [], []
        for context, node, exposure in items:
            if len(node.features) > 4:
                raise ValueError("default node encoder accepts at most four declared features")
            role_text = repr((node.role.inputs, node.role.consumed_outputs,
                              node.role.side_effect.value, node.role.authorization))
            features.append(np.concatenate((text_encoding(role_text),
                                            np.pad(node.features, (0, 4 - len(node.features))))))
            selected = set(exposure)
            neighbors = {i for i, m in enumerate(self.memories) if m.eligible(node.role)}
            adjacent = {b if a == node.uid else a for a, b in context.edges if node.uid in (a, b)}
            others = {i for n in context.nodes if n.uid in adjacent
                      for i, m in enumerate(self.memories) if m.eligible(n.role)}
            unaries.append([float(m.uid in selected and i in neighbors) for i, m in enumerate(self.memories)])
            pair_masks, gate_indices = [], []
            for i, j in self.pairs:
                within = i in neighbors and j in neighbors
                cross = (i in neighbors and j in others) or (j in neighbors and i in others)
                pair_masks.append(float((within or cross) and self.memories[i].uid in selected
                                        and self.memories[j].uid in selected))
                gate_indices.append(2 * self.roles.get(node.role.key, 0) + (0 if within else 1))
            pairs.append(pair_masks)
            indices.append(gate_indices)
        return (torch.tensor(np.asarray(features), dtype=torch.float32), self.encoding,
                torch.tensor(unaries, dtype=torch.float32), self.pair_encoding,
                torch.tensor(pairs, dtype=torch.float32), torch.tensor(indices, dtype=torch.long))

    @staticmethod
    def _loss(logits, labels, recovery, penalty):
        correct_mask = (labels == 0) | (labels == 1)
        recovery_mask = labels == 1
        loss = logits.sum() * 0
        # Sum both masked likelihood terms, normalized by eligible first attempts.
        if correct_mask.any():
            loss = functional.binary_cross_entropy_with_logits(
                logits[correct_mask, 0], (labels[correct_mask] == 0).float(), reduction="sum")
        if recovery_mask.any():
            loss = loss + functional.binary_cross_entropy_with_logits(
                logits[recovery_mask, 1], recovery[recovery_mask], reduction="sum")
        return loss / max(int(correct_mask.sum()), 1) + penalty

    def fit(self, episodes, calibration, *, logged=(), epochs=250, baseline_epochs=50,
            learning_rate=0.02, gate_penalty=0.001, tolerance=1e-6):
        if epochs < 1 or baseline_epochs < 0 or learning_rate <= 0 or gate_penalty < 0:
            raise ValueError("invalid outcome-model optimizer configuration")
        if not episodes or not calibration:
            raise ValueError("randomized training and independent calibration are required")
        training_ids = {e.context.uid for e in episodes}
        if training_ids & {e.context.uid for e in calibration}:
            raise ValueError("calibration contexts overlap training")
        if any(not e.randomized or not e.replay_valid or e.assignment_probability is None
               or not 0 < e.assignment_probability <= 1 for e in episodes):
            raise ValueError("exposure heads require replay-valid randomized evidence")
        if any(not e.replay_valid for e in calibration):
            raise ValueError("calibration requires replay-valid held-out executions")
        if any(not e.randomized for e in calibration):
            raise ValueError("calibration must use randomized exposures")
        if any(e.context.uid in training_ids or e.context.uid in {c.context.uid for c in calibration}
               for e in logged):
            raise ValueError("logged pretraining must be separate from randomized/calibration contexts")

        def training_arrays(data):
            items, labels, recovered = [], [], []
            for episode in data:
                nodes = {n.uid: n for n in episode.context.nodes}
                for observation in episode.observations:
                    items.append((episode.context, nodes[observation.node], episode.exposure))
                    labels.append(("ok", "agent", "service", "amb").index(observation.label))
                    recovered.append(float(observation.recovered))
            if not items:
                raise ValueError("contract-labelled node observations required")
            return self._pack(items), torch.tensor(labels), torch.tensor(recovered)

        if logged:
            packed, labels, recovered = training_arrays(logged)
            optimizer = torch.optim.Adam(self.heads.baseline.parameters(), lr=learning_rate)
            for _ in range(baseline_epochs):
                optimizer.zero_grad()
                loss = self._loss(self.heads(*packed, baseline_only=True), labels, recovered, 0)
                loss.backward()
                optimizer.step()  # No unary/pair/gate parameter belongs to this optimizer.
        packed, labels, recovered = training_arrays(episodes)
        if not ((labels == 0) | (labels == 1)).any():
            raise ValueError("no attributable first-attempt observations")
        optimizer = torch.optim.Adam(self.heads.parameters(), lr=learning_rate)
        previous, stable, converged = float("inf"), 0, False
        for iteration in range(epochs):
            optimizer.zero_grad()
            loss = self._loss(self.heads(*packed), labels, recovered,
                              gate_penalty * self.heads.gates.abs().sum())
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite outcome-model loss")
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            stable = stable + 1 if abs(previous - value) < tolerance else 0
            previous = value
            if stable >= 8:
                converged = True
                break
        self.training_ids = training_ids
        for episode in episodes:
            nodes = {n.uid: n for n in episode.context.nodes}
            for observation in episode.observations:
                node = nodes[observation.node]
                if observation.label in {"ok", "agent"}:
                    key = (node.role.key, node.version)
                    self.support[key] = self.support.get(key, 0) + 1
        predictions = self.predict_many([(e.context, e.exposure) for e in calibration], calibrated=False)
        self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            predictions, [e.outcome for e in calibration])
        self.fit_report = {"iterations": iteration + 1, "loss": previous,
                           "stopping_criterion_met": converged, "randomized_episodes": len(episodes),
                           "calibration_episodes": len(calibration),
                           "calibration_successes": sum(e.outcome for e in calibration),
                           "calibration_context_clusters": len({e.context.uid for e in calibration}),
                           "masked_service_nodes": int((labels == 2).sum()),
                           "masked_ambiguous_nodes": int((labels == 3).sum()),
                           "baseline_pretraining_episodes": len(logged),
                           "calibration_mae": float(np.mean(np.abs(self.calibrator.predict(predictions)
                                                                   - [e.outcome for e in calibration])))}
        return self

    def predict_many(self, items, *, calibrated=True):
        if not items:
            return np.asarray([])
        packed = self._pack([(c, n, a) for c, a in items for n in c.nodes])
        with torch.no_grad():
            probabilities = torch.sigmoid(self.heads(*packed)).numpy()
        success = probabilities[:, 0] + (1 - probabilities[:, 0]) * probabilities[:, 1]
        result, offset = [], 0
        for context, _ in items:
            result.append(float(np.prod(success[offset:offset + len(context.nodes)])))
            offset += len(context.nodes)
        result = np.asarray(result)
        return self.calibrator.predict(result) if calibrated and self.calibrator is not None else result

    def predict(self, context, exposure):
        return float(self.predict_many([(context, exposure)])[0])

    def coefficients(self, context, memories, effects):
        if len(memories) > 16:
            raise ValueError("exact consistency diagnostics limited to 16 slots")
        rows = np.asarray(list(product([-1, 1], repeat=len(memories))))
        values = self.predict_many([(context, tuple(m.uid for m, s in zip(memories, row) if s == 1))
                                    for row in rows])
        return values @ characters(rows, effects) / len(rows)

    def save(self, path):
        """Save weights plus encoder, role and calibration metadata without pickled models."""
        torch.save({"state_dict": self.heads.state_dict(), "role_indices": self.roles,
                    "memory_ids": [m.uid for m in self.memories], "encoder": "signed-hash-bag-v1",
                    "calibration_x": None if self.calibrator is None else self.calibrator.X_thresholds_,
                    "calibration_y": None if self.calibrator is None else self.calibrator.y_thresholds_,
                    "fit_report": self.fit_report}, path)


def bootstrap_ensemble(memories, episodes, calibration, *, members=3, seed=0, **fit_options):
    if members < 1:
        raise ValueError("ensemble must contain at least one model")
    rng = np.random.default_rng(seed)
    # Resample whole contexts/draw clusters; rows sharing a checkpoint are dependent.
    clusters = sorted({e.context.uid for e in episodes})
    grouped = {uid: [e for e in episodes if e.context.uid == uid] for uid in clusters}
    models = []
    for member in range(members):
        sampled = [e for uid in rng.choice(clusters, len(clusters), replace=True) for e in grouped[uid]]
        model = OutcomeModel(memories, [e.context for e in episodes], seed=seed + member)
        models.append(model.fit(sampled, calibration, **fit_options))
    return models
