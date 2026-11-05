"""Corrected kernels for the controlled sandbox, kept separate from frozen v1.

Public mutation postconditions constrain responses, without disclosing request
acceptance guards, service drift or success probabilities. Bounded interaction
outputs prevent arbitrary inverse scaling of the penalized group weights.
"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .shared_mechanisms import OutcomeBatch, RESPONSE_LABELS, SharedKernels, SharedMechanisms
from .stateful import ACTIONS, decision_key


def response_contract(role, state):
    """Public postconditions, conditional on success; acceptance remains learned.

    These are the stateful-v1 wrapper's mutation contract, shared by every
    family/version/action. Failure is atomic and preserves the prior state.
    This is NOT the set of labels merely seen in a finite sample.
    """
    after = {"lookup": 1, "transform": (state | 3) & ~4,
             "authorize": state | 7, "submit": state | 8}[role]
    return {f"ok:{after}", f"reject:{state}", f"transient:{state}"}


class SupportedKernels(SharedKernels):
    def __init__(self, policy_keys, api_keys, **kwargs):
        super().__init__(policy_keys, api_keys, **kwargs)
        self.interaction.add_module("bounded_output", nn.Tanh())
        self.register_buffer("response_allowed", torch.tensor([
            [label in response_contract(key[2], key[3]) for label in RESPONSE_LABELS]
            for key in api_keys], dtype=torch.bool))

    def api_logits(self):
        return super().api_logits().masked_fill(~self.response_allowed, -torch.inf)


def _conditional_loss(logits, counts):
    # Avoid 0 * -inf for structurally inadmissible labels.
    logs = F.log_softmax(logits, dim=1).masked_fill(counts == 0, 0)
    return -(counts * logs).sum() / counts.sum().clamp_min(1)


def fit_supported(policy_records, api_records, intervention_traces, *, seed=7, epochs=400,
                  learning_rate=0.01, outcome_weight=0.2, sparse_weight=0.002,
                  interactions=True, threads=1, validation_traces=(), patience=60):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    pkeys = [tuple(row["key"]) for row in policy_records]
    kkeys = [tuple(row["key"]) for row in api_records]
    network = SupportedKernels(pkeys, kkeys, interactions=interactions)
    pcounts = torch.tensor([row["counts"] for row in policy_records], dtype=torch.float32)
    kcounts = torch.tensor([[row["counts"].get(label, 0) for label in RESPONSE_LABELS]
                           for row in api_records], dtype=torch.float32)
    if ((kcounts > 0) & ~network.response_allowed).any():
        raise ValueError("observations violate declared public mutation contract")
    vp, vk = torch.zeros_like(pcounts), torch.zeros_like(kcounts)
    pi, ki = {key: i for i, key in enumerate(pkeys)}, {key: i for i, key in enumerate(kkeys)}
    for trace in validation_traces:
        for step in trace.steps:
            vp[pi[decision_key(step.role, step.state_before, step.previous_status, trace.exposure)],
               ACTIONS.index(step.request_mode)] += 1
            vk[ki[(trace.task.epoch, trace.task.family, step.role, step.state_before, step.request_mode)],
               RESPONSE_LABELS.index(f"{step.response_status}:{step.state_after}")] += 1
    if ((vk > 0) & ~network.response_allowed).any():
        raise ValueError("validation violates declared mutation contract")
    batch = OutcomeBatch(intervention_traces, pkeys, kkeys)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    history, best, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        plogits, klogits = network.decision_logits(), network.api_logits()
        p_loss, k_loss = _conditional_loss(plogits, pcounts), _conditional_loss(klogits, kcounts)
        outcome = batch.loss(plogits.softmax(1), klogits.softmax(1)) if outcome_weight and batch.total else p_loss * 0
        smooth = p_loss + k_loss + outcome_weight * outcome
        smooth.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(network.parameters(), 10))
        optimizer.step()
        network.shrink_groups(learning_rate * sparse_weight)
        with torch.no_grad():
            post_p, post_k = network.decision_logits(), network.api_logits()
            score = float(_conditional_loss(post_p, vp) + _conditional_loss(post_k, vk)) if vp.sum() else float(smooth)
            group_norm = float(network.group_weights.norm(dim=1).sum())
            post_smooth = _conditional_loss(post_p, pcounts) + _conditional_loss(post_k, kcounts)
            if outcome_weight and batch.total:
                post_smooth += outcome_weight * batch.loss(post_p.softmax(1), post_k.softmax(1))
            if score < best - 1e-6:
                best, best_epoch, best_state = score, epoch, deepcopy(network.state_dict())
        history.append({"epoch": epoch, "decision_nll": float(p_loss.detach()), "api_nll": float(k_loss.detach()),
            "outcome_bce": float(outcome.detach()), "group_norm": group_norm,
            "penalized_objective": float(post_smooth) + sparse_weight * group_norm,
            "gradient_norm_before_clip": gradient, "validation_conditional_nll": score})
        if validation_traces and epoch - best_epoch >= patience:
            break
    network.load_state_dict(best_state)
    with torch.no_grad():
        decision = network.decision_logits().softmax(1).numpy().astype(float)
        response = network.api_logits().softmax(1).numpy().astype(float)
    model = SharedMechanisms([], [])
    model.policy = {key: probabilities * counts.sum()
                    for key, probabilities, counts in zip(pkeys, decision, pcounts.numpy())}
    model.api = {key: {label: probability * counts.sum() for label, probability in zip(RESPONSE_LABELS, probabilities)
                      if probability > 0} for key, probabilities, counts in zip(kkeys, response, kcounts.numpy())}
    model.total_cells, model.certificate_valid = len(pkeys) + len(kkeys), False
    model.training_history, model.network = history, network
    model.selected_epoch, model.best_validation_nll = best_epoch, best
    return model
