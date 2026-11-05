from __future__ import annotations

import json
from itertools import combinations
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from credit.experiments import collect_pairs
from credit.model import StructuralModel
from credit.policy import RECENCY, bank_cost, fifo_bank, recency_bank, retrieve, version_aware_bank
from credit.rest_evolve import local_service
from credit.sandbox import MEMORIES, Task, make_tasks, replay_pair, run_task
from credit.voi import one_step_voi
import numpy as np


def test_incremental_replay_matches_full_counterfactual() -> None:
    tasks = [
        task for version in ("v1", "v2")
        for task in make_tasks(12, split="replay-test", version=version,
                               templates=("short", "double_lookup", "chain", "long"),
                               seed=123)
    ]
    for task in tasks:
        for memory in MEMORIES:
            other = tuple(m for m in MEMORIES if m != memory)
            for size in range(len(other) + 1):
                for coalition in combinations(other, size):
                    background = set(coalition)
                    inc = replay_pair(task, background, background | {memory},
                                      propensity=1 / 1024)
                    full = replay_pair(task, background, background | {memory},
                                       propensity=1 / 1024, incremental=False)
                    assert inc.valid and full.valid
                    assert inc.control.outcome == full.control.outcome
                    assert inc.treatment.outcome == full.treatment.outcome
                    assert inc.treatment.state_hash == full.treatment.state_hash


def test_version_pinning_and_invalid_replay() -> None:
    source = Task("fixed", ("lookup", "submit"), "v1", "short", "test", 9)
    target = Task("fixed", ("lookup", "submit"), "v2", "short", "test", 9)
    old = run_task(source, ())
    try:
        run_task(target, (), prefix=old, start=1)
    except ValueError as error:
        assert "version mismatch" in str(error)
    else:
        raise AssertionError("cross-version checkpoint was accepted")
    unsafe = Task("unsafe", ("lookup", "submit"), "v1", "short", "test", 9,
                  restorable=False)
    assert not replay_pair(unsafe, (), ("lookup_key",), propensity=0.5).valid
    placebo = replay_pair(source, ("lookup_key",), ("lookup_key",), propensity=0.5)
    assert placebo.valid and placebo.mode == "identical"
    assert placebo.control.outcome == placebo.treatment.outcome
    assert placebo.control.state_hash == placebo.treatment.state_hash
    assert placebo.treatment.api_calls == 0


def test_randomized_support_for_hidden_memory_and_abstention() -> None:
    tasks = make_tasks(96, split="train-test", version="v1",
                       templates=("short", "chain"), seed=8)
    pairs, traces = collect_pairs(tasks, 192, seed=9)
    assert all(pair.propensity > 0 for pair in pairs)
    assert any("hidden_fix" in pair.treatment.exposure for pair in pairs)
    assert "hidden_fix" not in retrieve(tasks[0], MEMORIES)
    model = StructuralModel().fit(traces)
    unknown = Task("unsupported", ("authorize",), "v1", "unknown_role", "test", 12)
    prediction = model.predict(unknown, ())
    assert not prediction.supported
    assert prediction.lower == 0


def test_local_http_same_schema_changes_contract() -> None:
    def post(base: str, path: str, body: dict) -> int:
        request = Request(f"{base}/{path}", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=5) as response:
                return response.status
        except HTTPError as error:
            return error.code

    with local_service() as (state, base):
        for version, expected in (("v1", 200), ("v2", 400)):
            task = Task("case", ("lookup", "submit"), version, "short", "test", 1)
            state.reset(task)
            assert post(base, "lookup", {"key": "case"}) == 200
            assert post(base, "submit", {"token": "case", "contract": "v1"}) == expected


def test_one_step_voi_matches_perfect_binary_information() -> None:
    utilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    success = np.array([1.0, 0.0])
    assert abs(one_step_voi(utilities, success, probe_cost=0.1) - 0.4) < 1e-12


def test_version_aware_bank_uses_scope_and_arrival_order() -> None:
    v1 = Task("v1", ("lookup", "transform", "submit"), "v1", "chain", "test", 1)
    v2 = Task("v2", ("lookup", "transform", "submit"), "v2", "chain", "test", 2)
    assert version_aware_bank([v1, v2]) == (
        "lookup_key", "submit_field", "transform_rule", "version_patch"
    )
    assert version_aware_bank([v1], budget=256) == (
        "lookup_key", "transform_rule"
    )
    assert version_aware_bank([v1, v2], budget=0) == ()


def test_retention_baselines_respect_timestamps_and_budget() -> None:
    assert set(recency_bank()) == {"version_patch", "stale_hint", "submit_field", "lookup_key"}
    assert min(RECENCY[memory] for memory in recency_bank()) > max(
        RECENCY[memory] for memory in set(MEMORIES) - set(recency_bank())
    )
    for budget in (0, 127, 128, 256, 512, 1024):
        for choose in (fifo_bank, recency_bank):
            assert bank_cost(choose(budget)) <= budget


def test_local_kernels_do_not_learn_effects_for_unrelated_memories() -> None:
    tasks = make_tasks(64, split="local-kernel-test", version="v1",
                       templates=("short", "chain"), seed=15)
    _, traces = collect_pairs(tasks, 192, seed=16)
    model = StructuralModel().fit(traces)
    for task in tasks:
        assert model.predict(task, ("lookup_key", "submit_field")).mean == model.predict(
            task, ("lookup_key", "submit_field", "noop")
        ).mean
    lookup = Task("lookup-only", ("lookup",), "v1", "lookup-only", "test", 1)
    assert model.predict(lookup, ("lookup_key",)).mean == model.predict(
        lookup, ("lookup_key", "submit_field", "conflict", "noop")
    ).mean
