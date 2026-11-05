"""Shared encoders, dependency interactions and differentiable outcome training.

Finite structured action/state reference for the paper's four-term objective.
No simulator probabilities or held-out outcome labels are accessed here.
Neural fits have empirical calibration only, never categorical MLE certificates.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mechanisms import FiniteMechanisms, StatePrediction
from .stateful import ACTIONS, FAMILIES, MEMORIES, NEIGHBORS, ROLES, STATUSES, decision_key

RESPONSE_LABELS = tuple(f"{status}:{state}" for state in range(16)
                        for status in ("ok", "reject", "transient"))


class SharedMechanisms(FiniteMechanisms):
    """Vectorized finite evaluation; shared neural parameters lack MLE bounds."""
    def policy_distribution(self, key):
        if not hasattr(self, "_decision_cache"):
            self._decision_cache = {}
        if key not in self._decision_cache:
            self._decision_cache[key] = super().policy_distribution(key)
        return self._decision_cache[key]

    def _matrix(self, role, task, exposure):
        key = role, task.family, task.epoch, exposure
        if key in self._matrices:
            return self._matrices[key]
        decision = np.asarray([[self.policy_distribution(decision_key(role, state, previous, exposure))
                                for previous in STATUSES] for state in range(8)])
        if not hasattr(self, "_api_blocks"):
            self._api_blocks = {}
        api_key = role, task.family, task.epoch
        if api_key not in self._api_blocks:
            api = np.zeros((8, 4, 33))
            for state in range(8):
                for action_index, action in enumerate(ACTIONS):
                    for (status, after), probability in self.api_distribution(
                            (task.epoch, task.family, role, state, action)).items():
                        api[state, action_index, 32 if after & 8 else after * 4 + STATUSES.index(status)] += probability
            self._api_blocks[api_key] = api
        api = self._api_blocks[api_key]
        matrix = np.zeros((33, 33))
        matrix[:32] = np.einsum("sha,sad->shd", decision, api).reshape(32, 33)
        matrix[32, 32] = 1
        if len(self._matrices) >= 4096:
            self._matrices.clear()
        self._matrices[key] = matrix
        return matrix

    def predict(self, task, exposure):
        exposure = tuple(sorted(exposure))
        key = task.family, task.roles, task.epoch, task.initial_state, exposure
        if key in self._predictions:
            return self._predictions[key]
        if any(role not in ROLES for role in task.roles):
            return StatePrediction(0, 0, 0, False, 1)
        distribution = np.zeros(33)
        distribution[32 if task.initial_state & 8 else task.initial_state * 4] = 1
        supported = True
        if not hasattr(self, "_api_support"):
            self._api_support = {key: sum(counts.values()) >= self.min_support - 1e-5
                                 for key, counts in self.api.items()}
        for role in task.roles:
            for index in np.flatnonzero(distribution[:32] > 0):
                state, previous = divmod(int(index), 4)
                pkey = decision_key(role, state, STATUSES[previous], exposure)
                supported &= self.policy.get(pkey, np.zeros(4)).sum() >= self.min_support - 1e-5
                for action, probability in zip(ACTIONS, self.policy_distribution(pkey)):
                    if probability:
                        supported &= self._api_support.get((task.epoch, task.family, role, state, action), False)
            distribution = distribution @ self._matrix(role, task, exposure)
        probability = float(distribution[32])
        result = StatePrediction(probability, max(0, probability - self.calibration_error) if supported else 0,
                                 0, bool(supported), 1)
        self._predictions[key] = result
        return result


class SharedKernels(nn.Module):
    def __init__(self, policy_keys, api_keys, *, width=8, hidden=32, interactions=True):
        super().__init__()
        self.policy_keys, self.api_keys = tuple(policy_keys), tuple(api_keys)
        self.memory = nn.Embedding(len(MEMORIES), width)
        self.role = nn.Embedding(len(ROLES), width)
        self.family = nn.Embedding(len(FAMILIES), width)
        self.epoch = nn.Embedding(2, width)
        self.action = nn.Embedding(len(ACTIONS), width)
        self.base = nn.Sequential(nn.Linear(width + 2, hidden), nn.Tanh(), nn.Linear(hidden, 4))
        self.unary = nn.Sequential(nn.Linear(2 * width + 2, hidden), nn.Tanh(), nn.Linear(hidden, 4))
        self.interaction = nn.Sequential(nn.Linear(2 * width + 3, hidden), nn.Tanh(), nn.Linear(hidden, 4))
        self.response = nn.Sequential(nn.Linear(4 * width + 3, 64), nn.Tanh(), nn.Linear(64, 48))
        self.groups = tuple((role, tuple(group)) for role in ROLES
                            for size in (2, 3) for group in combinations(NEIGHBORS[role], size)) if interactions else ()
        self.group_weights = nn.Parameter(torch.full((len(self.groups), 4), 0.05))
        masks = [[int(memory in {m for i, m in enumerate(NEIGHBORS[key[0]]) if key[3] & (1 << i)})
                  for memory in MEMORIES] for key in policy_keys]
        self.register_buffer("masks", torch.tensor(masks, dtype=torch.float32))
        self.register_buffer("policy_roles", torch.tensor([ROLES.index(key[0]) for key in policy_keys]))
        self.register_buffer("history", torch.tensor([[key[1], key[2]] for key in policy_keys], dtype=torch.float32))
        membership = [[int(memory in group) for memory in MEMORIES] for _, group in self.groups]
        self.register_buffer("membership", torch.tensor(membership, dtype=torch.float32).reshape(-1, len(MEMORIES)))
        present = [[int(key[0] == role and all(mask[MEMORIES.index(m)] for m in group))
                    for role, group in self.groups] for key, mask in zip(policy_keys, masks)]
        self.register_buffer("group_present", torch.tensor(present, dtype=torch.float32).reshape(len(policy_keys), -1))
        self.register_buffer("group_size", torch.tensor([len(group) for _, group in self.groups], dtype=torch.float32))
        self.register_buffer("api_features", torch.tensor([
            [ROLES.index(role), FAMILIES.index(family), ("source", "target").index(epoch),
             ACTIONS.index(action), *[int(state & (1 << bit) != 0) for bit in range(3)]]
            for epoch, family, role, state, action in api_keys]))

    def decision_logits(self):
        context = torch.cat((self.role(self.policy_roles), self.history), dim=1)
        count = len(self.policy_keys)
        memory = self.memory.weight.unsqueeze(0).expand(count, -1, -1)
        unary_context = context.unsqueeze(1).expand(-1, len(MEMORIES), -1)
        logits = self.base(context) + (self.unary(torch.cat((memory, unary_context), dim=2))
                                      * self.masks.unsqueeze(2)).sum(dim=1)
        if self.groups:
            pooled = (self.membership @ self.memory.weight).unsqueeze(0).expand(count, -1, -1)
            group_context = context.unsqueeze(1).expand(-1, len(self.groups), -1)
            sizes = self.group_size.view(1, -1, 1).expand(count, -1, -1)
            contribution = self.interaction(torch.cat((pooled, group_context, sizes), dim=2))
            logits = logits + (contribution * self.group_weights.unsqueeze(0)
                               * self.group_present.unsqueeze(2)).sum(dim=1)
        return logits

    def api_logits(self):
        data = self.api_features
        features = torch.cat((self.role(data[:, 0]), self.family(data[:, 1]), self.epoch(data[:, 2]),
                              self.action(data[:, 3]), data[:, 4:].float()), dim=1)
        return self.response(features)

    @torch.no_grad()
    def shrink_groups(self, threshold):
        if self.groups:
            norm = self.group_weights.norm(dim=1, keepdim=True)
            self.group_weights.mul_((1 - threshold / norm.clamp_min(1e-12)).clamp_min(0))


class OutcomeBatch:
    """Precomputed public context indices; exact differentiable state summation."""
    def __init__(self, traces, policy_keys, api_keys):
        policy_index = {key: index for index, key in enumerate(policy_keys)}
        api_index = {key: index for index, key in enumerate(api_keys)}
        self.groups = []
        roles_groups = {}
        for trace in traces:
            roles_groups.setdefault(trace.task.roles, []).append(trace)
        self.total = len(traces)
        for roles, rows in roles_groups.items():
            decisions, responses = [], []
            for role in roles:
                decisions.append(torch.tensor([[[policy_index[decision_key(role, state, previous, trace.exposure)]
                                                  for previous in STATUSES] for state in range(8)] for trace in rows]))
                responses.append(torch.tensor([[[api_index[(trace.task.epoch, trace.task.family, role, state, action)]
                                                  for action in ACTIONS] for state in range(8)] for trace in rows]))
            initial = torch.zeros((len(rows), 33))
            for index, trace in enumerate(rows):
                initial[index, 32 if trace.task.initial_state & 8 else trace.task.initial_state * 4] = 1
            self.groups.append((initial, torch.tensor([trace.outcome for trace in rows], dtype=torch.float32),
                                decisions, responses))
        self.destinations = torch.tensor([32 if state & 8 else state * 4 + STATUSES.index(status)
                                         for state in range(16) for status in ("ok", "reject", "transient")])

    def predictions(self, decision, api):
        outputs = []
        for initial, labels, decisions, responses in self.groups:
            distribution = initial.to(dtype=decision.dtype)
            for pindex, kindex in zip(decisions, responses):
                joint_actions = (distribution[:, :32].reshape(-1, 8, 4).unsqueeze(3)
                                 * decision[pindex]).sum(dim=2)
                outputs_by_response = (joint_actions.unsqueeze(3) * api[kindex]).sum(dim=(1, 2))
                following = torch.zeros_like(distribution).scatter_add(
                    1, self.destinations.unsqueeze(0).expand(len(labels), -1), outputs_by_response)
                terminal = F.one_hot(torch.tensor(32), 33).float().unsqueeze(0) * distribution[:, 32:33]
                distribution = following + terminal
            outputs.append((distribution[:, 32], labels))
        return outputs

    def loss(self, decision, api):
        return sum(F.binary_cross_entropy(prediction.clamp(1e-7, 1 - 1e-7), labels.to(prediction.dtype), reduction="sum")
                   for prediction, labels in self.predictions(decision, api)) / max(1, self.total)


def fit_shared(policy_records, api_records, intervention_traces, *, seed=7, epochs=80,
               learning_rate=0.01, outcome_weight=0.2, sparse_weight=0.002,
               interactions=True, threads=1):
    """Four-term training with proximal group penalty; fixed epoch budget.

    Conditional NLLs are weighted by observed cell counts. Outcome BCE uses
    only randomized replay-valid traces. Proximal shrinkage optimizes the group
    norm term. Nonconvex Adam/proximal training has no global convergence claim.
    """
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    policy_keys = [tuple(row["key"]) for row in policy_records]
    api_keys = [tuple(row["key"]) for row in api_records]
    network = SharedKernels(policy_keys, api_keys, interactions=interactions)
    pcounts = torch.tensor([row["counts"] for row in policy_records], dtype=torch.float32)
    kcounts = torch.tensor([[row["counts"].get(label, 0) for label in RESPONSE_LABELS]
                           for row in api_records], dtype=torch.float32)
    batch = OutcomeBatch(intervention_traces, policy_keys, api_keys)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        plogits, klogits = network.decision_logits(), network.api_logits()
        p_loss = -(pcounts * F.log_softmax(plogits, dim=1)).sum() / pcounts.sum()
        k_loss = -(kcounts * F.log_softmax(klogits, dim=1)).sum() / kcounts.sum()
        outcome_loss = batch.loss(plogits.softmax(1), klogits.softmax(1)) if outcome_weight and batch.total else p_loss * 0
        smooth_loss = p_loss + k_loss + outcome_weight * outcome_loss
        smooth_loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), 10))
        optimizer.step()
        network.shrink_groups(learning_rate * sparse_weight)
        group_norm = float(network.group_weights.norm(dim=1).sum().detach())
        history.append({"epoch": epoch + 1, "decision_nll": float(p_loss.detach()),
                        "api_nll": float(k_loss.detach()), "outcome_bce": float(outcome_loss.detach()),
                        "group_norm": group_norm, "penalized_objective": float(smooth_loss.detach()) + sparse_weight * group_norm,
                        "gradient_norm_before_clip": gradient_norm})
    with torch.no_grad():
        decision = network.decision_logits().softmax(1).numpy().astype(float)
        response = network.api_logits().softmax(1).numpy().astype(float)
    model = SharedMechanisms([], [])
    model.policy = {key: probabilities * max(0, counts.sum())
                    for key, probabilities, counts in zip(policy_keys, decision, pcounts.numpy())}
    model.api = {key: {label: probability * counts.sum() for label, probability in zip(RESPONSE_LABELS, probabilities)}
                 for key, probabilities, counts in zip(api_keys, response, kcounts.numpy())}
    model.total_cells = len(model.policy) + len(model.api)
    model.certificate_valid = False
    # Only counts control support; smooth neural probabilities are not evidence.
    model.training_history = history
    model.network = network
    return model


def augment_counts(policy_records, api_records, traces):
    """Add observed trace decisions/responses; expose the identical data to fits."""
    policy = {tuple(row["key"]): {**row, "counts": list(row["counts"])} for row in policy_records}
    api = {tuple(row["key"]): {**row, "counts": dict(row["counts"])} for row in api_records}
    for trace in traces:
        for step in trace.steps:
            pkey = decision_key(step.role, step.state_before, step.previous_status, trace.exposure)
            kkey = (trace.task.epoch, trace.task.family, step.role, step.state_before, step.request_mode)
            policy[pkey]["counts"][ACTIONS.index(step.request_mode)] += 1
            policy[pkey]["samples"] += 1
            label = f"{step.response_status}:{step.state_after}"
            api[kkey]["counts"][label] = api[kkey]["counts"].get(label, 0) + 1
            api[kkey]["samples"] += 1
    return list(policy.values()), list(api.values())
