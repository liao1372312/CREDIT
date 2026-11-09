"""Immutable inputs and verified observations for CREDIT revision 3.

Role signatures deliberately omit endpoint names and unused output fields.
Adapters, rather than statistical models, must establish replay safety.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Protocol


class SideEffect(str, Enum):
    RO = "RO"
    ID = "ID"
    RV = "RV"
    IR = "IR"


@dataclass(frozen=True)
class Role:
    inputs: tuple[str, ...]
    consumed_outputs: tuple[str, ...]
    side_effect: SideEffect
    authorization: str

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(sorted(set(self.inputs))))
        object.__setattr__(self, "consumed_outputs", tuple(sorted(set(self.consumed_outputs))))

    @property
    def key(self) -> str:
        body = (self.inputs, self.consumed_outputs, self.side_effect.value, self.authorization)
        return sha256(json.dumps(body).encode()).hexdigest()[:24]

    def partially_matches(self, other: "Role") -> bool:
        return (self.side_effect == other.side_effect
                and self.authorization == other.authorization
                and bool(set(self.inputs) & set(other.inputs)
                         or set(self.consumed_outputs) & set(other.consumed_outputs)))


@dataclass(frozen=True)
class Memory:
    uid: str
    content: str
    tokens: tuple[int, ...]
    scope: frozenset[str] | None  # None means unscoped (*), empty means no scope.
    provenance: str
    arrival: int = 0

    @property
    def cost(self) -> int:
        return len(self.tokens)

    def eligible(self, role: Role) -> bool:
        return self.scope is None or role.key in self.scope


@dataclass(frozen=True)
class Node:
    uid: str
    role: Role
    version: str
    features: tuple[float, ...] = ()


@dataclass(frozen=True)
class Context:
    uid: str
    stratum: str
    nodes: tuple[Node, ...]
    edges: tuple[tuple[str, str], ...]
    seed: int
    initial_state: int = 0
    initial_user_text: str | None = None
    initial_embedding: tuple[float, ...] | None = None
    task_id: str | None = None
    source_group: str | None = None

    def __post_init__(self):
        positions = {node.uid: i for i, node in enumerate(self.nodes)}
        if not self.nodes or len(positions) != len(self.nodes):
            raise ValueError("a nonempty skeleton with unique nodes is required")
        if any(a not in positions or b not in positions or positions[a] >= positions[b]
               for a, b in self.edges):
            raise ValueError("nodes must be topologically ordered and edges acyclic")


@dataclass(frozen=True)
class Slots:
    memories: tuple[str, ...]
    signs: tuple[int, ...]
    token_ids: tuple[tuple[int, ...], ...]

    @property
    def exposure(self) -> tuple[str, ...]:
        return tuple(m for m, sign in zip(self.memories, self.signs) if sign == 1)


def reserved_slots(memories: tuple[Memory, ...], signs, null_token: int = 0) -> Slots:
    signs = tuple(int(v) for v in signs)
    if len(signs) != len(memories) or any(v not in (-1, 1) for v in signs):
        raise ValueError("one +/-1 assignment per reserved slot is required")
    return Slots(tuple(m.uid for m in memories), signs,
                 tuple(m.tokens if s == 1 else (null_token,) * m.cost
                       for m, s in zip(memories, signs)))


@dataclass(frozen=True)
class NodeObservation:
    node: str
    label: str  # ok / agent / service / amb on first invocation
    recovered: bool = False

    def __post_init__(self):
        if self.label not in {"ok", "agent", "service", "amb"}:
            raise ValueError("invalid first-invocation contract label")
        if self.recovered and self.label != "agent":
            raise ValueError("recovery labels are defined only after agent failures")


@dataclass(frozen=True)
class Episode:
    context: Context
    exposure: tuple[str, ...]
    observations: tuple[NodeObservation, ...]
    verified: bool | None
    calls: int
    state_hash: str
    checkpoint_hash: str
    restored_hash: str
    replay_valid: bool
    randomized: bool = False
    assignment_probability: float | None = None
    events: tuple[dict, ...] = ()
    reused_calls: int = 0
    request_ids: tuple[str, ...] = ()

    @property
    def outcome(self) -> int:
        return int(self.verified is True)  # Unknown always contributes zero.


class Executor(Protocol):
    """Execute the current version, reset state, and emit checked node labels."""
    def run(self, context: Context, exposure: tuple[str, ...], *,
            slots: Slots | None = None) -> Episode: ...


def safe_replay(role: Role, *, mode: str, version_pinned: bool,
                snapshot_restored: bool, checkpoint_hash: str, restored_hash: str) -> bool:
    if mode not in {"sandbox", "shadow", "passive"}:
        raise ValueError("unknown operating mode")
    if mode == "passive":
        return False  # Live exposure needs a separate randomized-traffic adapter.
    if mode == "shadow":
        return role.side_effect in {SideEffect.RO, SideEffect.ID}
    if not version_pinned:
        return False
    if role.side_effect == SideEffect.IR and not snapshot_restored:
        return False
    if role.side_effect in {SideEffect.RV, SideEffect.IR}:
        return bool(checkpoint_hash and restored_hash == checkpoint_hash)
    return True
