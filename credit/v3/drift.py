"""Differential behavior probes, exact binomial intervals and role statuses."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
from scipy.stats import beta

from .types import Role


class Level(str, Enum):
    V = "V"
    U = "U"
    D = "D"
    X = "X"


def clopper_pearson(changed, fixtures, alpha=0.05):
    if not isinstance(changed, int) or not isinstance(fixtures, int) or not 0 <= changed <= fixtures:
        raise ValueError("integer change counts in [0,n] required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    if fixtures == 0:
        return 0.0, 1.0
    lower = 0.0 if changed == 0 else float(beta.ppf(alpha / 2, changed, fixtures - changed + 1))
    upper = 1.0 if changed == fixtures else float(beta.ppf(1 - alpha / 2, changed + 1, fixtures - changed))
    return lower, upper


@dataclass(frozen=True)
class ProbeEvidence:
    role: str
    version: str
    changed: int
    fixtures: int
    support: int
    lower: float
    upper: float
    level: Level


def transfer_level(source: Role | None, target: Role, version: str, *, changed: int,
                   fixtures: int, support: int, delta0=0.1, n_min=50, n_support=12,
                   alpha=0.05, identical_pinned_version=False):
    if not 0 <= delta0 <= 1 or n_min < 1 or n_support < 1 or support < 0:
        raise ValueError("invalid transfer thresholds")
    lower, upper = clopper_pearson(changed, fixtures, alpha)
    if source != target or support < n_support:
        level = Level.X
    elif identical_pinned_version:
        level = Level.V  # Source identity, not transport to a new version.
    elif lower > delta0:
        level = Level.D
    elif fixtures >= n_min and upper <= delta0:
        level = Level.V
    else:
        level = Level.U
    return ProbeEvidence(target.key, version, changed, fixtures, support, lower, upper, level)


def behavior_class(status: int, schema_valid: bool, consumed_formats: Mapping[str, str],
                   postconditions: Mapping[str, bool], role: Role):
    """Ignore response fields not consumed downstream; compare checked semantics."""
    return (int(status) // 100, bool(schema_valid),
            tuple((field, consumed_formats.get(field, "missing")) for field in role.consumed_outputs),
            tuple(sorted(postconditions.items())))


class TransportGate:
    def __init__(self, evidence=()):
        self.evidence = {(p.role, p.version): p for p in evidence}

    def level(self, node):
        p = self.evidence.get((node.role.key, node.version))
        return p.level if p is not None else Level.X

    def transported(self, context):
        return all(self.level(node) == Level.V for node in context.nodes)

    def candidates(self, context, memories, *, directly_validated=frozenset()):
        # U quarantine follows the revalidation section; X also requires direct evidence.
        blocked = {n.role.key for n in context.nodes if self.level(n) != Level.V}
        return tuple(m for m in memories if m.uid in directly_validated or
                     not blocked or (m.scope is not None and not (m.scope & blocked)))


def role_voi(incumbent, memories, role_key, score, *, capacity, probability,
             probe_calls, probe_penalty, incumbent_score=None):
    """Eq. voi: only feasible single-add/swap moves using unblocked memories."""
    if not 0 <= probability <= 1 or probe_calls < 0 or probe_penalty < 0:
        raise ValueError("invalid probe probability/cost")
    pool = {m.uid: m for m in memories}
    baseline = score(tuple(incumbent)) if incumbent_score is None else incumbent_score
    proposals = {tuple(sorted(incumbent))}
    for memory in memories:
        if memory.scope is not None and role_key not in memory.scope:
            continue
        proposals.add(tuple(sorted(set(incumbent) | {memory.uid})))
        for old in incumbent:
            proposals.add(tuple(sorted((set(incumbent) - {old}) | {memory.uid})))
    feasible = [b for b in proposals if sum(pool[m].cost for m in b) <= capacity]
    gain = max([score(b) - baseline for b in feasible] + [0.0])
    return probability * max(0.0, gain) - probe_penalty * probe_calls


def posterior_success(successes, failures, prior=(1.0, 1.0)):
    if min(successes, failures) < 0 or min(prior) <= 0:
        raise ValueError("nonnegative observations and positive Beta prior required")
    return (successes + prior[0]) / (successes + failures + sum(prior))


def archive_eligible(inactive_cycles, required_cycles, upper_joint_marginals):
    return (inactive_cycles >= required_cycles and bool(upper_joint_marginals)
            and all(np.isfinite(v) and v <= 0 for v in upper_joint_marginals))
