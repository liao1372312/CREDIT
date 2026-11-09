"""Paper 7944 role ensemble and ensemble-level calibration.

Requires the optional ``neural`` dependencies. The caller supplies independently
split, verified episodes and a source-group key; this module never creates
experimental labels or claims measured effectiveness.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn

from .paper_7944_protocol import PAPER_7944
from .v3.outcome import Heads, OutcomeModel


def _two_layer_head(inputs: int) -> nn.Sequential:
    width, dropout = PAPER_7944.hidden_units, PAPER_7944.dropout
    return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(width, 2))


class PaperHeads(Heads):
    def __init__(self, role_types: int, feature_size=12, memory_size=8):
        nn.Module.__init__(self)
        self.baseline = _two_layer_head(feature_size)
        self.unary = _two_layer_head(feature_size + memory_size)
        self.pair = _two_layer_head(feature_size + 3 * memory_size)
        self.gates = nn.Parameter(torch.ones(max(1, role_types) * 2, 2))


class PaperOutcomeModel(OutcomeModel):
    """Use a frozen paper feature encoder, never the local signed-hash fallback."""

    REQUIRED_PREEXECUTION_FIELDS = frozenset((
        "task_metadata", "visible_state", "initial_user_text", "initial_embedding",
        "intent", "slots", "roles", "retrieval_scores"))

    def __init__(self, memories, training_contexts, *, feature_encoder, seed):
        if (not callable(getattr(feature_encoder, "pack", None)) or
                not isinstance(getattr(feature_encoder, "fingerprint", None), str) or
                not feature_encoder.fingerprint or
                not isinstance(getattr(feature_encoder, "feature_size", None), int) or
                not isinstance(getattr(feature_encoder, "memory_size", None), int) or
                feature_encoder.feature_size < 1 or feature_encoder.memory_size < 1 or
                getattr(feature_encoder, "frozen", False) is not True or
                set(getattr(feature_encoder, "used_fields", ())) !=
                self.REQUIRED_PREEXECUTION_FIELDS):
            raise ValueError("registered model needs a frozen feature encoder with provenance")
        super().__init__(memories, training_contexts, seed=seed)
        self.feature_encoder = feature_encoder
        self.heads = PaperHeads(len(self.roles), feature_encoder.feature_size,
                                feature_encoder.memory_size)

    def _pack(self, items):
        packed = self.feature_encoder.pack(items, self.memories, self.roles, self.pairs)
        if len(packed) != 6:
            raise ValueError("paper feature encoder must supply six masked tensors")
        _, _, unaries, _, pairs, gate_indices = packed
        n, m, p = len(items), len(self.memories), len(self.pairs)
        f, d = self.feature_encoder.feature_size, self.feature_encoder.memory_size
        expected = ((n, f), (m, d), (n, m), (p, 3 * d), (n, p), (n, p))
        if any(tuple(tensor.shape) != shape for tensor, shape in zip(packed, expected)):
            raise ValueError("paper feature tensor shapes do not match frozen model dimensions")
        if not all(torch.isfinite(tensor).all() for tensor in packed):
            raise ValueError("paper features contain nonfinite values")
        if torch.any((unaries < 0) | (unaries > 1)) or torch.any((pairs < 0) | (pairs > 1)):
            raise ValueError("exposure masks must lie in [0,1]")
        if torch.any(gate_indices < 0) or torch.any(gate_indices >= max(1, len(self.roles)) * 2):
            raise ValueError("pair gate index is outside the registered role table")
        return packed


def _observations(model: OutcomeModel, episodes):
    items, labels, recovered = [], [], []
    for episode in episodes:
        nodes = {node.uid: node for node in episode.context.nodes}
        for observation in episode.observations:
            items.append((episode.context, nodes[observation.node], episode.exposure))
            labels.append(("ok", "agent", "service", "amb").index(observation.label))
            recovered.append(float(observation.recovered))
    if not items or not any(label in (0, 1) for label in labels):
        raise ValueError("verified correctness/recovery observations are required")
    return model._pack(items), torch.tensor(labels), torch.tensor(recovered)


def _subset(packed, indices):
    features, memories, unaries, pair_features, pairs, gate_indices = packed
    return (features[indices], memories, unaries[indices], pair_features,
            pairs[indices], gate_indices[indices])


class PaperEnsemble:
    paper_7944_ensemble = True

    def __init__(self, members, calibrator, reports, feature_fingerprint):
        self.members = tuple(members)
        self.calibrator = calibrator
        self.fit_report = {"members": reports,
                           "feature_fingerprint": feature_fingerprint,
                           "ensemble_calibration": "isotonic on mean raw workflow prediction"}

    def predict_many(self, items):
        if not items:
            return np.asarray([], dtype=float)
        raw = np.mean([member.predict_many(items, calibrated=False)
                       for member in self.members], axis=0)
        return self.calibrator.predict(raw)


def fit_paper_ensemble(memories, training, validation, calibration, *, source_group,
                       feature_encoder):
    """Fit five group-bootstrap members with independent ensemble calibration."""
    protocol = PAPER_7944
    if (len(training) != protocol.training_episodes_per_workload or
            len(validation) != protocol.validation_episodes_per_workload or
            len(calibration) != protocol.calibration_episodes_per_workload):
        raise ValueError("paper training/validation/calibration episode counts differ")
    partitions = [{source_group(e) for e in rows}
                  for rows in (training, validation, calibration)]
    if any(partitions[i] & partitions[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("source groups overlap independent model partitions")
    context_ids = [{e.context.uid for e in rows}
                   for rows in (training, validation, calibration)]
    if any(context_ids[i] & context_ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("contexts overlap independent model partitions")
    if any(not e.randomized or not e.replay_valid or
           e.assignment_probability is None or e.assignment_probability <= 0
           for e in training):
        raise ValueError("training requires replay-valid randomized exposures")
    if any(not e.randomized or not e.replay_valid for e in (*validation, *calibration)):
        raise ValueError("validation/calibration require randomized verified replay")

    grouped = {group: [e for e in training if source_group(e) == group]
               for group in partitions[0]}
    groups = sorted(grouped)
    members, reports = [], []
    for seed in protocol.ensemble_seeds:
        rng = np.random.default_rng(seed)
        sampled = [e for group in rng.choice(groups, size=len(groups), replace=True)
                   for e in grouped[group]]
        model = PaperOutcomeModel(memories, [e.context for e in training],
                                  feature_encoder=feature_encoder, seed=seed)
        train_pack, train_labels, train_recovery = _observations(model, sampled)
        valid_pack, valid_labels, valid_recovery = _observations(model, validation)
        optimizer = torch.optim.AdamW(model.heads.parameters(), lr=protocol.learning_rate,
                                      weight_decay=protocol.weight_decay)
        best_loss, best_state, stale, epochs_run = float("inf"), None, 0, 0
        for epoch in range(protocol.max_epochs):
            model.heads.train()
            order = rng.permutation(len(train_labels))
            for start in range(0, len(order), protocol.batch_size):
                batch = order[start:start + protocol.batch_size]
                optimizer.zero_grad()
                logits = model.heads(*_subset(train_pack, batch))
                loss = model._loss(logits, train_labels[batch], train_recovery[batch],
                                   protocol.interaction_l1 * model.heads.gates.abs().sum())
                loss.backward()
                optimizer.step()
            model.heads.eval()
            with torch.no_grad():
                validation_loss = float(model._loss(model.heads(*valid_pack), valid_labels,
                                                    valid_recovery, 0))
            epochs_run = epoch + 1
            if validation_loss < best_loss - 1e-7:
                best_loss, best_state, stale = validation_loss, deepcopy(model.heads.state_dict()), 0
            else:
                stale += 1
                if stale >= protocol.early_stop_patience:
                    break
        model.heads.load_state_dict(best_state)
        model.heads.eval()
        members.append(model)
        reports.append({"seed": seed, "epochs": epochs_run, "validation_loss": best_loss,
                        "bootstrapped_source_groups": len(groups)})
    calibration_items = [(e.context, e.exposure) for e in calibration]
    raw = np.mean([model.predict_many(calibration_items, calibrated=False)
                   for model in members], axis=0)
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
        raw, [e.outcome for e in calibration])
    return PaperEnsemble(members, calibrator, reports, feature_encoder.fingerprint)
