"""Instrumented stateful API sandbox; planted probabilities remain private.

Actions bind concrete request modes, replies mutate observable service state,
and rejection/transient replies permit subsequent repair actions. Source and
target snapshots expose the same schema and version label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from random import Random

import numpy as np

MEMORIES = (
    "lookup_key", "transform_rule", "submit_field", "legacy_contract",
    "new_contract", "auth_pair_a", "auth_pair_b", "repair_rule", "conflict",
    "noop", "cache_hint", "format_hint", "retry_hint", "quota_hint",
    "obsolete_lookup", "hidden_fix",
)
ROLES = ("lookup", "transform", "authorize", "submit")
ACTIONS = ("standard", "patched", "repair", "invalid")
STATUSES = ("start", "ok", "reject", "transient")
FAMILIES = ("books", "travel", "inventory", "payments")
DEPENDENCIES = {
    "lookup_key": {"lookup"}, "transform_rule": {"transform"},
    "submit_field": {"submit"}, "legacy_contract": {"submit"},
    "new_contract": {"submit"}, "auth_pair_a": {"authorize"},
    "auth_pair_b": {"authorize"}, "repair_rule": {"transform", "authorize"},
    "conflict": {"transform", "submit"}, "noop": set(),
    "cache_hint": {"lookup"}, "format_hint": {"transform"},
    "retry_hint": {"authorize"}, "quota_hint": {"authorize"},
    "obsolete_lookup": {"lookup"}, "hidden_fix": {"submit"},
}
NEIGHBORS = {role: tuple(m for m in MEMORIES if role in DEPENDENCIES[m]) for role in ROLES}
_arrival = list(MEMORIES)
Random(904).shuffle(_arrival)
ARRIVAL = tuple(_arrival)
_recent = list(MEMORIES)
Random(2026).shuffle(_recent)
RECENCY = {memory: index for index, memory in enumerate(_recent)}
WORKFLOWS = {
    "chain": ("lookup", "transform", "authorize", "submit"),
    "repeat": ("lookup", "transform", "lookup", "authorize", "submit"),
    "reordered": ("transform", "lookup", "authorize", "submit"),
    "repair_path": ("lookup", "authorize", "transform", "authorize", "submit"),
}
SCHEMA_FINGERPRINT = sha256(b"state-api-v1:role,mode,idempotency_key -> status,state").hexdigest()


@dataclass(frozen=True)
class StateTask:
    uid: str
    family: str
    roles: tuple[str, ...]
    epoch: str
    seed: int
    template: str
    initial_state: int = 0


@dataclass(frozen=True)
class StateStep:
    role: str
    state_before: int
    previous_status: str
    request_mode: str
    response_status: str
    state_after: int


@dataclass(frozen=True)
class StateTrace:
    task: StateTask
    exposure: tuple[str, ...]
    steps: tuple[StateStep, ...]
    outcome: int
    checkpoint_hash: str
    state_hash: str
    calls: int
    reused: int = 0


def local_context(role: str, state: int, previous_status: str) -> tuple[int, int]:
    required = {"lookup": 0, "transform": 1, "authorize": 3, "submit": 7}[role]
    return int(state & required != required), int(previous_status == "transient")


def local_mask(role: str, exposure) -> int:
    return sum(1 << index for index, memory in enumerate(NEIGHBORS[role]) if memory in exposure)


def decision_key(role: str, state: int, previous_status: str, exposure) -> tuple:
    return (role, *local_context(role, state, previous_status), local_mask(role, exposure))


def retrieve(task: StateTask, bank, depth: int = 4) -> tuple[str, ...]:
    """Frozen metadata retriever with no knowledge of hidden contract changes."""
    candidates = [(10 * len(DEPENDENCIES[m] & set(task.roles)) + RECENCY[m], m)
                  for m in bank if DEPENDENCIES[m] & set(task.roles)]
    return tuple(sorted(m for _, m in sorted(candidates, reverse=True)[:depth]))


def _policy(role: str, missing: int, transient: int, exposure) -> np.ndarray:
    a = set(exposure)
    standard, patched, repair = -0.2, -2.0, -3.0
    if role == "lookup":
        standard += 1.5 * ("lookup_key" in a) + 0.3 * ("cache_hint" in a)
        standard -= 0.9 * ("obsolete_lookup" in a)
    elif role == "transform":
        standard += 1.5 * ("transform_rule" in a) + 0.4 * ("format_hint" in a)
        standard -= 1.0 * ("conflict" in a)
        repair += 2.0 * ("repair_rule" in a) + 1.2 * missing
    elif role == "authorize":
        standard -= 0.3
        standard += 0.4 * ("auth_pair_a" in a) + 0.4 * ("auth_pair_b" in a)
        standard += 1.5 * ({"auth_pair_a", "auth_pair_b"} <= a)
        standard += 0.3 * ("quota_hint" in a)
        repair += 1.6 * ("repair_rule" in a) + 1.2 * missing + 0.5 * ("retry_hint" in a)
    else:
        standard += 0.6 * ("submit_field" in a) + 1.5 * ("legacy_contract" in a)
        patched += 0.6 * ("submit_field" in a) + 1.8 * ("new_contract" in a)
        standard += 0.3 * ("hidden_fix" in a)
        patched += 0.3 * ("hidden_fix" in a)
        standard -= 0.8 * ("conflict" in a)
    repair += 0.4 * transient
    logits = np.asarray([standard, patched, repair, 0.0])
    probabilities = np.exp(logits - logits.max())
    return probabilities / probabilities.sum()


def _api(role: str, family: str, epoch: str, state: int, action: str) -> dict[tuple[str, int], float]:
    """Hidden snapshot behavior; the public schema is identical across epochs."""
    required_contract = "patched" if epoch == "target" and family in ("books", "inventory") else "standard"
    if role == "lookup":
        valid = action in ("standard", "repair")
        next_state = 1  # A fresh record invalidates derived fields and credentials.
    elif role == "transform":
        valid = (action == "standard" and bool(state & 1)) or action == "repair"
        next_state = (state | 3) & ~4
    elif role == "authorize":
        valid = ((action == "standard" and state & 3 == 3)
                 or (action == "repair" and bool(state & 1)))
        next_state = state | 7
    else:
        valid = state & 7 == 7 and action == required_contract
        next_state = state | 8
    if not valid:
        return {("reject", state): 1.0}
    success = (0.96, 0.89, 0.93, 0.91)[FAMILIES.index(family)]
    success -= {"lookup": 0.0, "transform": 0.02, "authorize": 0.04, "submit": 0.01}[role]
    if epoch == "target" and family == "travel" and role == "lookup":
        success -= 0.18
    return {("ok", next_state): success, ("transient", state): 1 - success}


def _draw(probabilities, uniform: float):
    cumulative = 0.0
    for value, probability in probabilities:
        cumulative += probability
        if uniform < cumulative:
            return value
    return value


def _uniform(seed: int, index: int, channel: str) -> float:
    digest = sha256(f"stateful:{seed}:{index}:{channel}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _checkpoint(task: StateTask) -> str:
    return sha256(json.dumps(asdict(task), sort_keys=True).encode()).hexdigest()


def run_episode(task: StateTask, exposure, *, prefix: StateTrace | None = None,
                start: int = 0) -> StateTrace:
    exposure = tuple(sorted(exposure))
    if not set(exposure) <= set(MEMORIES):
        raise ValueError("unknown exposure")
    steps = []
    state, previous = task.initial_state, "start"
    if prefix is not None:
        if prefix.checkpoint_hash != _checkpoint(task) or start > len(prefix.steps):
            raise ValueError("checkpoint snapshot mismatch")
        steps = list(prefix.steps[:start])
        if steps:
            state, previous = steps[-1].state_after, steps[-1].response_status
    elif start:
        raise ValueError("prefix required")
    for index in range(start, len(task.roles)):
        if state & 8:
            break
        role = task.roles[index]
        missing, transient = local_context(role, state, previous)
        probabilities = _policy(role, missing, transient, exposure)
        action = _draw(zip(ACTIONS, probabilities), _uniform(task.seed, index, "decision"))
        status, after = _draw(_api(role, task.family, task.epoch, state, action).items(),
                              _uniform(task.seed, index, "response"))
        steps.append(StateStep(role, state, previous, action, status, after))
        state, previous = after, status
    body = json.dumps([asdict(step) for step in steps], sort_keys=True)
    return StateTrace(task, exposure, tuple(steps), int(bool(state & 8)), _checkpoint(task),
                      sha256(body.encode()).hexdigest(), len(steps) - start, start)


def replay(task: StateTask, control, treatment, incremental: bool = True) -> tuple[StateTrace, StateTrace]:
    base = run_episode(task, control)
    changed = set(control) ^ set(treatment)
    first = next((index for index, role in enumerate(task.roles)
                  if any(role in DEPENDENCIES[m] for m in changed)), len(task.roles))
    start = min(first, len(base.steps)) if incremental else 0
    target = run_episode(task, treatment, prefix=base if start else None, start=start)
    return base, target


def make_tasks(seed: int, split: str, count: int, *, templates=None) -> list[StateTask]:
    templates = tuple(WORKFLOWS) if templates is None else tuple(templates)
    rng = Random(seed)
    # The factorial assignment covers actual family mechanisms, epochs, and graphs.
    cells = [(family, epoch, template) for family in FAMILIES
             for epoch in ("source", "target") for template in templates]
    tasks = []
    for index in range(count):
        family, epoch, template = cells[index % len(cells)]
        tasks.append(StateTask(f"{split}-{index:05d}", family, WORKFLOWS[template], epoch,
                               rng.randrange(2**31), template))
    return tasks


def decision_probes(seed: int, samples: int) -> list[dict]:
    """Randomized finite-context action observations, using independent trials."""
    rng = np.random.default_rng(seed)
    records = []
    for role in ROLES:
        neighborhood = NEIGHBORS[role]
        for mask in range(1 << len(neighborhood)):
            exposure = {memory for index, memory in enumerate(neighborhood) if mask & (1 << index)}
            for missing in ((0,) if role == "lookup" else (0, 1)):
                for transient in (0, 1):
                    probabilities = _policy(role, missing, transient, exposure)
                    counts = np.bincount(np.searchsorted(np.cumsum(probabilities), rng.random(samples)),
                                         minlength=len(ACTIONS))
                    records.append({"key": [role, missing, transient, mask],
                                    "counts": counts.tolist(), "samples": samples,
                                    "assignment_probability": 1 / ((1 << len(neighborhood)) * (2 if role == "lookup" else 4)),
                                    "seed": seed})
    return records


def api_probes(seed: int, samples: int, epoch: str) -> list[dict]:
    """Safe fixture requests cover every finite state/action API input."""
    rng = np.random.default_rng(seed)
    records = []
    for family in FAMILIES:
        for role in ROLES:
            for state in range(8):
                for action in ACTIONS:
                    kernel = _api(role, family, epoch, state, action)
                    labels, probabilities = list(kernel), list(kernel.values())
                    draws = np.bincount(np.searchsorted(np.cumsum(probabilities), rng.random(samples)),
                                        minlength=len(labels))
                    records.append({"key": [epoch, family, role, state, action],
                                    "counts": {f"{status}:{after}": int(draws[index])
                                               for index, (status, after) in enumerate(labels) if draws[index]},
                                    "samples": samples, "seed": seed,
                                    "schema_fingerprint": SCHEMA_FINGERPRINT})
    return records


def request_probe(seed: int, samples: int, key: tuple) -> dict:
    """Execute a selected safe revalidation batch, without exposing true probabilities."""
    epoch, family, role, state, action = key
    kernel = _api(role, family, epoch, state, action)
    labels, probabilities = list(kernel), list(kernel.values())
    draws = np.random.default_rng(seed).multinomial(samples, probabilities)
    return {"key": list(key), "counts": {f"{status}:{after}": int(draws[index])
                                           for index, (status, after) in enumerate(labels) if draws[index]},
            "samples": samples, "seed": seed, "schema_fingerprint": SCHEMA_FINGERPRINT}


def oracle_probability(task: StateTask, exposure) -> float:
    """Evaluation-only exact forward calculation; the learner never calls it."""
    distribution = {(task.initial_state, "start"): 1.0}
    for role in task.roles:
        following = {}
        for (state, previous), weight in distribution.items():
            if state & 8:
                following[(state, previous)] = following.get((state, previous), 0.0) + weight
                continue
            missing, transient = local_context(role, state, previous)
            for action, p_action in zip(ACTIONS, _policy(role, missing, transient, exposure)):
                for (status, after), p_api in _api(role, task.family, task.epoch, state, action).items():
                    key = after, status
                    following[key] = following.get(key, 0.0) + weight * p_action * p_api
        distribution = following
    return sum(probability for (state, _), probability in distribution.items() if state & 8)
