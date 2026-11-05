"""Restricted local policy mechanism for the EMP development pilot.

Actions are coarsened to accepted / malformed / contract error, followed by
one repair. The API kernel is the known local validator. This is not the
paper's full variable-argument decision and learned provider-response model.
"""

from itertools import combinations
import json
from math import log1p
from pathlib import Path

import numpy as np

from .emp_binding_agent import public_task
from .emp_binding import BindingEnvironment, leaves


def context(edge):
    env = BindingEnvironment(edge); env.call(env.producer_request())
    task = public_task(env)
    names = [b['target'].split('/')[-1].casefold() for b in edge['bindings']]
    return np.array([1., log1p(len(json.dumps(task['actual_producer_response']))) / 10,
        log1p(len(json.dumps(task['consumer_arguments_template']))) / 10,
        len(names) / 15, max(b['target'].count('/') for b in edge['bindings']) / 6,
        max(len(path) for path, _ in leaves(task['actual_producer_response'])) / 6,
        float(any('itemid' == name for name in names)), float(any('itemcode' == name for name in names)),
        float(any('modelcode' == name for name in names)), float(any('indexcode' == name for name in names)),
        float(any('tmplid' == name for name in names)), float(any(any(isinstance(part, int) for part in path) for path, _ in leaves(task['actual_producer_response'])))], dtype=np.float32)


def features(edge, exposure, pool):
    c = context(edge)
    ids = {m['uid'] if isinstance(m, dict) else m for m in exposure}
    active = [m['uid'] in ids for m in pool]
    blocks = [c] + [c * value for value in active]
    blocks += [c * (active[i] and active[j]) for i, j in combinations(range(len(pool)), 2)]
    return np.concatenate(blocks)


class LocalMechanism:
    def __init__(self, pool, weights):
        self.pool, self.weights = pool, np.asarray(weights, dtype=np.float32)

    def predict(self, edge, exposure):
        z = features(edge, exposure, self.pool) @ self.weights
        p = np.exp(z[:3] - z[:3].max()); p /= p.sum()
        repair = 1 / (1 + np.exp(-np.clip(z[3:], -40, 40)))
        return {'first_success': float(p[0]), 'final_success': float(p[0] + p[1] * repair[0] + p[2] * repair[1]),
                'expected_consumer_calls': float(2 - p[0])}


def fit(rows, edges, pool, output, epochs=300):
    import torch
    torch.manual_seed(42)
    torch.set_num_threads(2)
    labels = {'accepted': 0, 'malformed': 1, 'contract_error': 2}
    x = torch.tensor(np.array([features(edges[row['edge_uid']], row['exposure'], pool) for row in rows]))
    y = torch.tensor([labels[row['first_category']] for row in rows])
    final = torch.tensor([float(row['final']['binding_contract_success']) for row in rows])
    w = torch.nn.Parameter(torch.zeros(x.shape[1], 5))
    optimizer = torch.optim.Adam([w], lr=.03)
    curve = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = x @ w
        p = logits[:, :3].softmax(-1)
        r = logits[:, 3:].sigmoid()
        prediction = (p[:, 0] + p[:, 1] * r[:, 0] + p[:, 2] * r[:, 1]).clamp(1e-6, 1 - 1e-6)
        decision_loss = torch.nn.functional.cross_entropy(logits[:, :3], y)
        failed = y > 0
        repair_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[failed, 3 + y[failed] - 1], final[failed]) if failed.any() else w.sum() * 0
        outcome_loss = torch.nn.functional.binary_cross_entropy(prediction, final)
        # Direct coefficient groups: no free head can undo the group penalty.
        groups = w[12:].reshape(-1, 12, 5)
        penalty = torch.linalg.vector_norm(groups, dim=(1, 2)).sum() if groups.numel() else w.sum() * 0
        loss = decision_loss + repair_loss + .2 * outcome_loss + .005 * penalty + .0001 * w.square().sum()
        loss.backward(); optimizer.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            curve.append({'epoch': epoch, 'loss': float(loss.detach()), 'decision_nll': float(decision_loss.detach()),
                          'repair_nll': float(repair_loss.detach()), 'outcome_nll': float(outcome_loss.detach())})
    model = LocalMechanism(pool, w.detach().numpy())
    (output / 'local_mechanism.json').write_text(json.dumps({'weights': model.weights.tolist(), 'training_curve': curve,
        'epochs': epochs, 'group_penalty': .005, 'outcome_weight': .2,
        'action_categories': labels, 'api_kernel': 'public deterministic local-contract validator',
        'limits': 'coarsened policy, fixed memory pool, development support only; no learned live provider kernel or formal certificate'}, indent=2))
    return model
