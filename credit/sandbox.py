"""Executable planted-causal workflows; ground-truth rules stay in this module."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import exp
from random import Random
from typing import Iterable


MEMORIES = (
    "lookup_key",
    "transform_rule",
    "submit_field",
    "stale_hint",
    "version_patch",
    "hidden_fix",
    "conflict",
    "noop",
)

MEMORY_ROLES = {
    "lookup_key": frozenset({"lookup", "submit"}),
    # The lookup/transform/submit bundle also changes the submit decision.
    # Replay dependencies must include every interaction-induced effect.
    "transform_rule": frozenset({"transform", "submit"}),
    "submit_field": frozenset({"submit"}),
    "stale_hint": frozenset({"submit"}),
    "version_patch": frozenset({"submit"}),
    "hidden_fix": frozenset({"submit"}),
    "conflict": frozenset({"submit"}),
    "noop": frozenset(),
}

TEMPLATES = {
    "short": ("lookup", "submit"),
    "chain": ("lookup", "transform", "submit"),
    "double_lookup": ("lookup", "lookup", "submit"),
    "new_order": ("transform", "lookup", "submit"),
    "long": ("lookup", "transform", "lookup", "submit"),
    "unknown_role": ("lookup", "authorize", "submit"),
}


@dataclass(frozen=True)
class Task:
    uid: str
    roles: tuple[str, ...]
    version: str
    template: str
    family: str
    seed: int
    incremental_safe: bool = True
    restorable: bool = True
    verifiable: bool = True
    drift_intensity: float = 1.0


@dataclass(frozen=True)
class Step:
    role: str
    correct: int
    api_ok: int


@dataclass(frozen=True)
class Trace:
    task_id: str
    version: str
    roles: tuple[str, ...]
    exposure: tuple[str, ...]
    steps: tuple[Step, ...]
    outcome: int | None
    checkpoint_hash: str
    state_hash: str
    api_calls: int
    reused_steps: int = 0


@dataclass(frozen=True)
class ReplayPair:
    control: Trace
    treatment: Trace
    propensity: float
    valid: bool
    mode: str
    extra_calls: int


def _uniform(seed: int, index: int, channel: str) -> float:
    digest = sha256(f"{seed}:{index}:{channel}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def _decision_probability(role: str, version: str, exposure: frozenset[str],
                          drift_intensity: float) -> float:
    """Frozen synthetic agent. This function is never imported by the learner."""
    if role == "lookup":
        logit = -0.2 + 1.55 * ("lookup_key" in exposure)
    elif role == "transform":
        logit = -0.25 + 1.55 * ("transform_rule" in exposure)
    elif role == "submit":
        logit = -0.55
        logit += 1.2 * ("submit_field" in exposure)
        logit += 0.85 * ("hidden_fix" in exposure)
        logit += 1.3 * ({"lookup_key", "submit_field"} <= exposure)
        logit += 0.55 * ({"lookup_key", "transform_rule", "submit_field"} <= exposure)
        logit -= 1.05 * ("conflict" in exposure)
        logit += (0.75 if version == "v1" else
                  0.75 - 2.0 * drift_intensity) * ("stale_hint" in exposure)
        logit += (0.0 if version == "v1" else
                  1.4 * drift_intensity) * ("version_patch" in exposure)
    else:
        logit = -0.65
    return _sigmoid(logit)


def _service_probability(role: str, version: str, correct: int,
                         drift_intensity: float) -> float:
    """Versioned API response kernel, separate from the agent decision."""
    if not correct:
        return 0.04
    if role == "submit" and version == "v2":
        return max(0.3, 0.96 - 0.14 * drift_intensity)
    if role == "authorize":
        return 0.86
    return 0.96


def _state_hash(steps: Iterable[Step]) -> str:
    body = "|".join(f"{s.role}:{s.correct}:{s.api_ok}" for s in steps)
    return sha256(body.encode()).hexdigest()


def _checkpoint_hash(task: Task) -> str:
    return sha256(f"{task.uid}|{task.version}|{task.seed}|empty".encode()).hexdigest()


def run_task(
    task: Task,
    exposure: Iterable[str],
    *,
    prefix: Trace | None = None,
    start: int = 0,
) -> Trace:
    """Run from a clean checkpoint, optionally reusing a validated prefix."""
    exposed = frozenset(exposure)
    if not exposed <= set(MEMORIES):
        raise ValueError("unknown memory in exposure")
    if prefix is not None:
        if prefix.task_id != task.uid or prefix.version != task.version:
            raise ValueError("checkpoint/task version mismatch")
        if prefix.checkpoint_hash != _checkpoint_hash(task):
            raise ValueError("checkpoint state mismatch")
        if start > len(prefix.steps) or start > len(task.roles):
            raise ValueError("incomplete checkpoint prefix")
        steps = list(prefix.steps[:start])
        if tuple(s.role for s in steps) != task.roles[:start]:
            raise ValueError("checkpoint dependency mismatch")
        if steps and not steps[-1].api_ok:
            raise ValueError("cannot resume after terminal failure")
    else:
        if start:
            raise ValueError("start requires a checkpoint prefix")
        steps = []
    for index in range(start, len(task.roles)):
        role = task.roles[index]
        correct = int(_uniform(task.seed, index, "decision") <
                      _decision_probability(role, task.version, exposed,
                                            task.drift_intensity))
        api_ok = int(_uniform(task.seed, index, "api") <
                     _service_probability(role, task.version, correct,
                                          task.drift_intensity))
        steps.append(Step(role, correct, api_ok))
        if not api_ok:
            break
    complete = len(steps) == len(task.roles) and all(s.api_ok and s.correct for s in steps)
    outcome = int(complete) if task.verifiable else None
    return Trace(task.uid, task.version, task.roles, tuple(sorted(exposed)), tuple(steps),
                 outcome, _checkpoint_hash(task), _state_hash(steps),
                 len(steps) - start, start)


def replay_pair(
    task: Task,
    control: Iterable[str],
    treatment: Iterable[str],
    *,
    propensity: float,
    incremental: bool = True,
) -> ReplayPair:
    """Use coupled randomness and a common version-pinned checkpoint."""
    a, b = frozenset(control), frozenset(treatment)
    if not task.restorable:
        empty = run_task(task, a)
        return ReplayPair(empty, empty, propensity, False, "excluded",
                          empty.api_calls + 1)
    base = run_task(task, a)
    changed = a ^ b
    affected = next((i for i, role in enumerate(task.roles)
                     if any(role in MEMORY_ROLES[m] for m in changed)), len(task.roles))
    can_reuse = (incremental and task.incremental_safe and affected > 0
                 and affected <= len(base.steps)
                 and all(step.api_ok for step in base.steps[:affected]))
    if incremental and task.incremental_safe and affected >= len(base.steps):
        target = replace(base, exposure=tuple(sorted(b)), api_calls=0,
                         reused_steps=len(base.steps))
        mode = "identical" if a == b else "unaffected"
    elif can_reuse and affected < len(task.roles):
        target = run_task(task, b, prefix=base, start=affected)
        mode = "incremental"
    else:
        target = run_task(task, b)
        mode = "full"
    return ReplayPair(base, target, propensity, True, mode,
                      base.api_calls + target.api_calls + 1)


def make_tasks(
    count: int,
    *,
    split: str,
    version: str,
    templates: tuple[str, ...],
    seed: int,
    invalid_rate: float = 0.0,
    unrestorable_rate: float = 0.0,
    drift_intensity: float = 1.0,
) -> list[Task]:
    rng = Random(seed)
    tasks = []
    for index in range(count):
        template = templates[index % len(templates)]
        tasks.append(Task(
            uid=f"{split}-{version}-{index:04d}",
            roles=TEMPLATES[template],
            version=version,
            template=template,
            family=f"family-{index % 8}",
            seed=rng.randrange(2**31),
            incremental_safe=rng.random() >= invalid_rate,
            restorable=rng.random() >= unrestorable_rate,
            drift_intensity=drift_intensity,
        ))
    return tasks
