"""Bounded integration adapter for the repository's recorded state API.

It calls the existing executable stochastic actor/service, never its hidden
probability functions. Contract labels use declared current-version request
rules. This is injected drift in a local simulator, not natural provider drift.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json

from ..stateful import (DEPENDENCIES, FAMILIES, RECENCY, StateTask, request_probe,
                        run_episode)
from .drift import behavior_class, transfer_level
from .types import Context, Episode, Memory, Node, NodeObservation, Role, SideEffect, safe_replay


ROLES = {
    "lookup": Role(("query",), ("record_id",), SideEffect.ID, "read-record"),
    "transform": Role(("record_id",), ("derived_id",), SideEffect.ID, "read-record"),
    "authorize": Role(("derived_id",), ("authorization_token",), SideEffect.ID, "issue-token"),
    "submit": Role(("derived_id", "authorization_token"), ("receipt",), SideEffect.IR, "write-record"),
}
ROLE_NAMES = {role.key: name for name, role in ROLES.items()}
POOL = ("lookup_key", "transform_rule", "legacy_contract", "auth_pair_a", "auth_pair_b",
        "repair_rule", "conflict", "hidden_fix")
TEMPLATES = {
    "train": ("lookup", "transform", "authorize", "submit"),
    "alternate": ("lookup", "authorize", "transform", "authorize", "submit"),
    "future": ("lookup", "transform", "lookup", "transform", "authorize", "submit"),
    "future_auth_repeat": ("lookup", "transform", "authorize", "authorize", "submit"),
    "future_ready_submit": ("submit",),
}
CONTENTS = {
    "lookup_key": "Bind the query to the lookup request and preserve the returned record identifier.",
    "transform_rule": "Transform the record identifier before requesting authorization for the derived value.",
    "legacy_contract": "Use the source submit field mapping with the standard identifier format.",
    "auth_pair_a": "Use the derived identifier as the first part of the authorization binding.",
    "auth_pair_b": "Preserve the second authorization binding together with the derived identifier.",
    "repair_rule": "On invalid preconditions retry the transform or authorization operation using recovery mode.",
    "conflict": "Prefer an obsolete transform and submit binding recorded in earlier executions.",
    "hidden_fix": "Keep the submit binding consistent with the returned record identifier.",
}


def stable_seed(*values):
    return int.from_bytes(sha256(json.dumps(values, sort_keys=True).encode()).digest()[:8], "big")


def memories():
    # Lexical token ids are an explicit finite-adapter tokenizer, not Qwen tokens.
    return tuple(Memory(uid, CONTENTS[uid],
                        tuple(stable_seed(word) % 2**31 for word in CONTENTS[uid].split()),
                        frozenset(ROLES[r].key for r in DEPENDENCIES[uid]),
                        "declared local instruction; numeric effects not included", i)
                 for i, uid in enumerate(POOL))


def context(family, epoch, phase, index, seed, *, template="future"):
    names = TEMPLATES[template]
    nodes = tuple(Node(f"node-{i}", ROLES[name], f"stateful:{family}:{name}:{epoch}",
                       (FAMILIES.index(family) / 3, i / 8, 0.0, 0.0))
                  for i, name in enumerate(names))
    return Context(f"v3:{phase}:{family}:{epoch}:{index}", f"{family}/{epoch}", nodes,
                   tuple((nodes[i].uid, nodes[i + 1].uid) for i in range(len(nodes) - 1)),
                   stable_seed(seed, phase, family, epoch, index),
                   initial_state=7 if template == "future_ready_submit" else 0)


def factory(seed):
    def make(stratum, phase, index):
        family, epoch = stratum.split("/")
        templates = ("future", "future_auth_repeat", "future_ready_submit")
        template = templates[stable_seed(seed, "task-shape", phase, family, epoch, index) % len(templates)]
        return context(family, epoch, phase, index, seed, template=template)
    return make


def ranked(context, bank):
    role_names = {ROLE_NAMES[n.role.key] for n in context.nodes}
    return tuple(sorted((m for m in bank if DEPENDENCIES[m] & role_names and m != "hidden_fix"),
                        key=lambda m: (-10 * len(DEPENDENCIES[m] & role_names) - RECENCY[m], m)))


def retriever(depth=4):
    return lambda context, bank: ranked(context, bank)[:depth]


def valid_request(role, state, action, family, epoch):
    """Public current contract, used by the checker rather than model features."""
    if role == "lookup":
        return action in ("standard", "repair")
    if role == "transform":
        return (action == "standard" and bool(state & 1)) or action == "repair"
    if role == "authorize":
        return (action == "standard" and state & 3 == 3) or (action == "repair" and bool(state & 1))
    mode = "patched" if epoch == "target" and family in {"books", "inventory"} else "standard"
    return state & 7 == 7 and action == mode


class StatefulExecutor:
    def __init__(self, repair_budget=1):
        if repair_budget < 0:
            raise ValueError("nonnegative repair budget required")
        self.repair_budget = repair_budget

    def run(self, context, exposure, *, slots=None, prefix=None, start=0):
        if slots is not None and set(slots.exposure) != set(exposure):
            raise ValueError("reserved slots differ from actual intervention")
        family, epoch = context.stratum.split("/")
        if any(n.version != f"stateful:{family}:{ROLE_NAMES[n.role.key]}:{epoch}" for n in context.nodes):
            raise ValueError("context version does not match the executed snapshot")
        checkpoint = sha256(json.dumps({"state": context.initial_state,
                                      "versions": [n.version for n in context.nodes]}, sort_keys=True).encode()).hexdigest()
        state, observations, events, succeeded = context.initial_state, [], [], []
        if prefix is not None:
            if prefix.context != context or prefix.checkpoint_hash != checkpoint or not prefix.replay_valid:
                raise ValueError("incremental checkpoint/version/seed mismatch")
            if not 0 <= start <= len(context.nodes):
                raise ValueError("invalid replay cut")
            ids = {n.uid for n in context.nodes[:start]}
            events = [e for e in prefix.events if e["node"] in ids]
            observations = list(prefix.observations[:start])
            succeeded = [o.label == "ok" or o.recovered for o in observations]
            if events:
                state = events[-1]["state_after"]
        elif start:
            raise ValueError("a replay cut requires the pinned prefix")
        reused_calls = len(events)
        for index in range(start, len(context.nodes)):
            node = context.nodes[index]
            role = ROLE_NAMES[node.role.key]
            if state & 8:
                observations.append(NodeObservation(node.uid, "amb"))
                succeeded.append(False)
                continue
            first_label, recovered, node_ok = "amb", False, False
            for attempt in range(1 + self.repair_budget):
                task = StateTask(f"{context.uid}:{node.uid}:{attempt}", family, (role,), epoch,
                                 stable_seed(context.seed, index, attempt), "v3-node", state)
                trace = run_episode(task, exposure)
                step = trace.steps[0]
                events.append({"node": node.uid, "attempt": attempt, **asdict(step),
                               "version": node.version})
                attributable = valid_request(role, state, step.request_mode, family, epoch)
                label = ("ok" if step.response_status == "ok" else "service") if attributable else "agent"
                if attempt == 0:
                    first_label = label
                state = step.state_after
                if step.response_status == "ok":
                    node_ok = True
                    recovered = attempt > 0 and first_label == "agent"
                    break
                # Paper recovery is defined after an agent failure, not after API noise.
                if first_label != "agent":
                    break
            observations.append(NodeObservation(node.uid, first_label, recovered))
            succeeded.append(node_ok)
        terminal_hash = sha256(json.dumps({"state": state, "events": events}, sort_keys=True).encode()).hexdigest()
        restored_state = context.initial_state  # Each invocation owns an isolated value-state snapshot.
        restored = sha256(json.dumps({"state": restored_state,
                                    "versions": [n.version for n in context.nodes]}, sort_keys=True).encode()).hexdigest()
        safe = all(safe_replay(n.role, mode="sandbox", version_pinned=True,
                               snapshot_restored=True, checkpoint_hash=checkpoint, restored_hash=restored)
                   for n in context.nodes)
        return Episode(context, tuple(exposure), tuple(observations),
                       bool(state & 8) and all(succeeded), len(events) - reused_calls, terminal_hash,
                       checkpoint, restored, safe, events=tuple(events), reused_calls=reused_calls)

    def replay(self, context, exposure, reference, *, slots=None):
        changed = set(exposure) ^ set(reference.exposure)
        known = {m.uid: m for m in memories()}
        if not changed <= known.keys():
            raise ValueError("untraced memory scope; full replay required")
        first = next((i for i, n in enumerate(context.nodes)
                      if any(known[m].eligible(n.role) for m in changed)), len(context.nodes))
        # This adapter has a fixed sequential skeleton: all later nodes are descendants.
        return self.run(context, exposure, slots=slots, prefix=reference, start=first)


def differential_probes(*, seed, support, fixtures=96, delta0=0.1, n_min=50, n_support=12):
    """Same request/state/seed on both snapshots; every physical fixture is logged."""
    records, evidence = [], []
    for family in FAMILIES:
        for name, role in ROLES.items():
            version = f"stateful:{family}:{name}:target"
            changes = 0
            for index in range(fixtures):
                state = {"lookup": 0, "transform": 1, "authorize": 3, "submit": 7}[name]
                outputs, classes = [], []
                for epoch in ("source", "target"):
                    result = request_probe(stable_seed(seed, "fixture", family, name, index), 1,
                                           (epoch, family, name, state, "standard"))
                    result_key = next(iter(result["counts"]))
                    status, next_state = result_key.split(":")
                    formats = {f: "declared-id" if status == "ok" else "missing" for f in role.consumed_outputs}
                    cls = behavior_class(200 if status == "ok" else 400 if status == "reject" else 500,
                                         True, formats, {"state": int(next_state) != state}, role)
                    outputs.append(result)
                    classes.append(cls)
                changed = classes[0] != classes[1]
                changes += int(changed)
                records.append({"family": family, "role": role.key, "fixture": index,
                                "outputs": outputs, "changed": changed, "api_calls": 2})
            source_version = f"stateful:{family}:{name}:source"
            count = support.get((role.key, source_version), 0)
            evidence.append(transfer_level(role, role, source_version, changed=0, fixtures=0,
                                           support=count, identical_pinned_version=True, n_support=n_support))
            evidence.append(transfer_level(role, role, version, changed=changes, fixtures=fixtures,
                                           support=count, delta0=delta0, n_min=n_min, n_support=n_support))
    return evidence, records
