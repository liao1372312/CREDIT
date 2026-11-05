"""Small shared, permutation-invariant binding and repair policy.

Fits the saved development interventions only. No provider kernel or formal
transport guarantee is inferred. Unseen exposure configurations abstain.
"""

from collections import defaultdict
from itertools import combinations
import json
from math import log1p
import re

import numpy as np

from .emp_binding import BindingEnvironment, leaves, pointer


def field_name(value):
    return re.sub(r'[_\-\s]', '', str(value)).casefold()


def public_structure(edge):
    response = edge['producer_catalog']['response_example']
    fields = [(pointer(path), path, value) for path, value in leaves(response)]
    targets = [b['target'] for b in edge['bindings']]
    matches = [[(p, parts, value) for p, parts, value in fields
                if parts and field_name(parts[-1]) == field_name(target.split('/')[-1])]
               for target in targets]
    depths = [len(parts) for group in matches for _, parts, _ in group]
    # Derived from all observable matching fields, not private reference sources.
    skeleton = tuple(sorted((target.count('/'), tuple(sorted((len(parts), pointer(parts[:-1]), type(value).__name__) for _, parts, value in group)))
                            for target, group in zip(targets, matches)))
    profile = {'skeleton': skeleton, 'target_count': len(targets),
               'request_container': type(edge['consumer_catalog']['request_example']).__name__}
    base = np.asarray([1., log1p(len(json.dumps(response))) / 10,
        log1p(len(json.dumps(edge['consumer_catalog']['request_example']))) / 10,
        log1p(len(targets)), max(t.count('/') for t in targets) / 6,
        float(np.mean(depths)) / 6 if depths else 0.,
        float(np.mean([len(group) != 1 for group in matches])),
        float(any(any(isinstance(part, int) for part in parts) for group in matches for _, parts, _ in group)),
        float(any('/data/data/' in p for p, _, _ in fields)),
        float(any('/resultList/' in p or '/resultValue/' in p for p, _, _ in fields))], dtype=np.float64)
    return fields, targets, matches, profile, base


def encode(edge, exposure):
    fields, targets, matches, profile, base = public_structure(edge)
    paths = {p for p, _, _ in fields}
    target_names = {field_name(t.split('/')[-1]) for t in targets}
    vectors, routes = [], []
    for memory in exposure:
        mappings = [(target, source.get('pointer', '')) for target, source in memory['verified_mapping'].items()
                    if field_name(target.split('/')[-1]) in target_names]
        route_set = {(field_name(target.split('/')[-1]), source) for target, source in mappings}
        routes.append(route_set)
        count = max(1, len(mappings))
        vectors.append(np.asarray([1., len(mappings) / max(1, len(targets)),
            sum(source in paths for _, source in mappings) / count,
            sum(target in targets for target, _ in mappings) / count,
            sum('/data/data/' in source for _, source in mappings) / count,
            sum('/0/' in source for _, source in mappings) / count], dtype=np.float64))
    unary = np.sum(vectors, axis=0) if vectors else np.zeros(6)
    pair = np.zeros(4)
    for i, j in combinations(range(len(routes)), 2):
        names_i, names_j = {n for n, _ in routes[i]}, {n for n, _ in routes[j]}
        pair += np.asarray([1., len(names_i & names_j), len(routes[i] & routes[j]),
            float(bool(names_i & names_j) and routes[i] != routes[j])])
    return np.concatenate([base, unary, pair]), profile


def distinct_trials(rows):
    """Uniform workflow, then uniform distinct exposure; not uniform bank."""
    cells = {}
    per_edge = defaultdict(list)
    for row in rows:
        key = row['edge_uid'], tuple(sorted(row['exposure']))
        signature = (row['first_category'], row['final']['binding_contract_success'], row['consumer_calls'])
        if key in cells:
            old = cells[key]
            if (old['first_category'], old['final']['binding_contract_success'], old['consumer_calls']) != signature:
                raise ValueError('identical deterministic exposures have conflicting outcomes')
        else:
            cells[key] = row
            per_edge[row['edge_uid']].append(key)
    ordered = [cells[key] for key in sorted(cells)]
    workflow_edges = defaultdict(set)
    for row in ordered:
        workflow_edges[row['workflow_group']].add(row['edge_uid'])
    weights = np.asarray([1 / (len(workflow_edges) * len(workflow_edges[row['workflow_group']]) * len(per_edge[row['edge_uid']])) for row in ordered])
    return ordered, weights


class SharedBindingMechanism:
    def __init__(self, weights, mean, scale, pool, training_profiles):
        self.weights, self.mean, self.scale = [np.asarray(v) for v in (weights, mean, scale)]
        self.pool = {m['uid']: m for m in pool}
        self.training_profiles = training_profiles

    def predict(self, edge, exposure, *, abstain=True):
        notes = [self.pool[x] if isinstance(x, str) else x for x in exposure]
        vector, profile = encode(edge, notes)
        z = ((vector - self.mean) / self.scale) @ self.weights
        p = np.exp(z[:3] - z[:3].max()); p /= p.sum()
        repair = 1 / (1 + np.exp(-np.clip(z[3:], -40, 40)))
        # Structural compatibility is a necessary empirical gate, not a certificate.
        witnesses = [x for x in self.training_profiles if x['profile'] == profile and x['exposure_size'] == len(notes)]
        groups = sorted({x['workflow_group'] for x in witnesses})
        supported = bool(groups)
        values = {'first_success': float(p[0]), 'final_success': float(p[0] + p[1] * repair[0] + p[2] * repair[1]),
                  'expected_consumer_calls': float(2 - p[0])}
        return {**{k: v if supported or not abstain else None for k, v in values.items()},
                'supported': supported, 'support_workflow_groups': groups,
                'support_reason': 'observed request/response binding skeleton and exposure size' if supported else 'unseen binding skeleton or exposure size',
                'certificate': False}


def fit_shared(rows, edges, pool, *, ridge=1., outcome_weight=.2, epochs=220):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(42)
    unique, weights = distinct_trials(rows)
    if any(row['stage'] != 'intervention_train' for row in unique):
        raise ValueError('only historical intervention training rows may fit the mechanism')
    notes = {m['uid']: m for m in pool}
    data, profiles = [], []
    for row in unique:
        vector, profile = encode(edges[row['edge_uid']], [notes[uid] for uid in row['exposure']])
        data.append(vector)
        profiles.append({'profile': profile, 'exposure_size': len(row['exposure']), 'workflow_group': row['workflow_group']})
    data = np.asarray(data); sample_weights = weights / weights.sum()
    avg = (data * sample_weights[:, None]).sum(axis=0)
    scale = np.sqrt(((data - avg) ** 2 * sample_weights[:, None]).sum(axis=0))
    scale[scale < .05] = 1.
    avg[0], scale[0] = 0., 1.
    # Do not center exposure blocks: empty exposure must have zero memory effect.
    avg[10:] = 0.
    x = torch.tensor((data - avg) / scale, dtype=torch.float64)
    sw = torch.tensor(sample_weights, dtype=torch.float64)
    labels = {'accepted': 0, 'malformed': 1, 'contract_error': 2}
    y = torch.tensor([labels[r['first_category']] for r in unique])
    final = torch.tensor([float(r['final']['binding_contract_success']) for r in unique], dtype=torch.float64)
    coeff = torch.nn.Parameter(torch.zeros(20, 5, dtype=torch.float64))
    optimizer = torch.optim.Adam([coeff], lr=.04)
    for _ in range(epochs):
        optimizer.zero_grad(); z = x @ coeff
        p, repair = z[:, :3].softmax(-1), z[:, 3:].sigmoid()
        pred = (p[:, 0] + p[:, 1] * repair[:, 0] + p[:, 2] * repair[:, 1]).clamp(1e-8, 1 - 1e-8)
        decision = (sw * torch.nn.functional.cross_entropy(z[:, :3], y, reduction='none')).sum()
        failed = y > 0
        repair_nll = (sw[failed] * torch.nn.functional.binary_cross_entropy_with_logits(z[failed, 3 + y[failed] - 1], final[failed], reduction='none')).sum() if failed.any() else coeff.sum() * 0
        outcome = (sw * torch.nn.functional.binary_cross_entropy(pred, final, reduction='none')).sum()
        # No independent multiplier/head can cancel these direct penalties.
        penalty = ridge * .01 * coeff[1:].square().sum() + .01 * torch.linalg.vector_norm(coeff[16:])
        loss = decision + repair_nll + outcome_weight * outcome + penalty
        loss.backward(); optimizer.step()
    model = SharedBindingMechanism(coeff.detach().numpy(), avg, scale, pool, profiles)
    info = {'raw_rows': len(rows), 'distinct_exposure_rows': len(unique), 'parameters': 100,
            'ridge': ridge, 'outcome_weight': outcome_weight, 'epochs': epochs, 'loss': float(loss.detach()),
            'estimand': 'uniform training workflows then uniform distinct visible exposures',
            'limits': 'coarsened binding/repair actions; deterministic local API kernel; empirical support, no certificate'}
    return model, info


def score_predictions(rows, predict):
    values = []
    for row in rows:
        pred = predict(row)
        if pred['final_success'] is None:
            continue
        target = float(row['final']['binding_contract_success'])
        first = float(row['first_try_success'])
        p = np.clip(pred['final_success'], 1e-6, 1 - 1e-6)
        values.append({'final_mae': abs(p - target), 'first_mae': abs(pred['first_success'] - first),
            'brier': (p - target) ** 2, 'final_nll': -target * np.log(p) - (1 - target) * np.log(1 - p)})
    return {'rows': len(rows), 'covered_rows': len(values), 'coverage': len(values) / len(rows) if rows else 0.,
        **{k: float(np.mean([v[k] for v in values])) if values else None
           for k in ['final_mae', 'first_mae', 'brier', 'final_nll']}}


def tune_shared(rows, edges, pool, grid=((.1, 0.), (.1, .2), (1., 0.), (1., .2), (10., 0.), (10., .2))):
    groups = sorted({r['workflow_group'] for r in rows})
    if len(groups) < 3:
        raise ValueError('grouped hyperparameter selection requires three training workflows')
    tuning = []
    for ridge, outcome in grid:
        folds = []
        for heldout in groups:
            blocked_notes = {m['uid'] for m in pool if edges[m['source_edge']]['workflow_group'] == heldout}
            eligible = [r for r in rows if not blocked_notes.intersection(r['exposure'])]
            training = [r for r in eligible if r['workflow_group'] != heldout]
            test, _ = distinct_trials([r for r in eligible if r['workflow_group'] == heldout])
            fold_pool = [m for m in pool if m['uid'] not in blocked_notes]
            model, _ = fit_shared(training, edges, fold_pool, ridge=ridge, outcome_weight=outcome)
            # Tune on every held-out row, including unsupported extrapolations.
            # This prevents selecting a model by only its easiest covered cases.
            metrics = score_predictions(test, lambda r: model.predict(edges[r['edge_uid']], r['exposure'], abstain=False))
            folds.append({'heldout_group': heldout, 'excluded_origin_notes': sorted(blocked_notes), **metrics})
        tuning.append({'ridge': ridge, 'outcome_weight': outcome, 'folds': folds,
            'group_mean_nll': float(np.mean([f['final_nll'] for f in folds])),
            'group_mean_brier': float(np.mean([f['brier'] for f in folds]))})
    chosen = min(tuning, key=lambda x: (x['group_mean_nll'], x['group_mean_brier'], -x['ridge']))
    model, info = fit_shared(rows, edges, pool, ridge=chosen['ridge'], outcome_weight=chosen['outcome_weight'])
    return model, info, tuning
