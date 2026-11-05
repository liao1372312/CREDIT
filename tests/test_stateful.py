from dataclasses import replace
from random import Random

import pytest

from credit.mechanisms import FiniteMechanisms, StatePrediction, mechanism_tests, tv_radius
from credit.model import Prediction
from credit.policy import bank_objective
from credit.prospective import score_values, select_bank
from credit.sandbox import Task
from credit.stateful import (
    MEMORIES, SCHEMA_FINGERPRINT, StateTask, api_probes, decision_probes,
    make_tasks, oracle_probability, replay, run_episode,
)


def test_stateful_replay_recomputes_state_descendants() -> None:
    rng = Random(23)
    tasks = make_tasks(24, "replay-regression", 32)
    for task in tasks:
        for memory in MEMORIES:
            for _ in range(3):
                background = {m for m in MEMORIES if m != memory and rng.random() < 0.5}
                control, incremental = replay(task, background, background | {memory})
                _, full = replay(task, background, background | {memory}, incremental=False)
                assert incremental.state_hash == full.state_hash
                assert incremental.outcome == full.outcome
                assert control.checkpoint_hash == incremental.checkpoint_hash
    task = tasks[0]
    before = run_episode(task, ())
    with pytest.raises(ValueError, match="snapshot mismatch"):
        run_episode(replace(task, epoch="target"), (), prefix=before, start=1)


def test_workflow_order_changes_real_state_utility() -> None:
    chain = StateTask("chain", "books", ("lookup", "transform", "authorize", "submit"),
                      "source", 9, "chain")
    reordered = replace(chain, uid="reordered", roles=("transform", "lookup", "authorize", "submit"))
    exposure = ("lookup_key", "transform_rule", "auth_pair_a", "auth_pair_b", "legacy_contract")
    assert oracle_probability(chain, exposure) > oracle_probability(reordered, exposure) + 0.01


def test_local_model_runs_without_planted_kernel_queries(monkeypatch) -> None:
    import credit.stateful as environment
    model = FiniteMechanisms(decision_probes(1, 256), api_probes(2, 256, "source"))
    task = StateTask("known", "books", ("submit",), "source", 1, "probe", initial_state=7)
    def forbidden(*args, **kwargs):
        raise AssertionError("learner accessed planted API kernel")
    monkeypatch.setattr(environment, "_api", forbidden)
    prediction = model.predict(task, ("legacy_contract",))
    assert prediction.supported and prediction.mean > 0
    assert model.predict(task, ("legacy_contract", "noop")).mean == prediction.mean
    unknown = model.predict(replace(task, epoch="unknown"), ("legacy_contract",))
    assert not unknown.supported and unknown.certified_lower == 0


def test_finite_certificate_requires_evidence_and_shrinks_with_samples() -> None:
    assert tv_radius(0, 4, 100) == 1
    assert tv_radius(4096, 4, 100) < tv_radius(128, 4, 100)
    model = FiniteMechanisms(decision_probes(5, 4096), api_probes(6, 4096, "source"))
    task = StateTask("cert", "books", ("submit",), "source", 1, "probe", initial_state=7)
    exposure = ("legacy_contract", "submit_field")
    prediction = model.predict(task, exposure)
    assert 0 < prediction.certified_lower <= prediction.mean
    assert prediction.certified_lower <= oracle_probability(task, exposure)


def test_schema_stability_does_not_mask_contract_drift() -> None:
    source = api_probes(1, 4096, "source")
    target = api_probes(2, 4096, "target")
    assert {row["schema_fingerprint"] for row in source + target} == {SCHEMA_FINGERPRINT}
    tests = mechanism_tests(source, target)
    changed = [row for row in tests if row["family"] == "books" and row["role"] == "submit"
               and row["state"] == 7 and row["action"] == "standard"]
    assert changed[0]["status"] == "changed"
    assert any(row["status"] == "equivalent" for row in tests)


def test_explicit_future_weights_do_not_follow_probe_counts() -> None:
    class Fixed:
        def predict(self, task, exposure):
            probability = {"v1": 0.8, "v2": 0.2}[task.version]
            return Prediction(probability, probability, True, "test")
    source = Task("source", ("lookup",), "v1", "short", "family", 1)
    target = replace(source, uid="target", version="v2")
    options = dict(version_weights={"v1": 0.5, "v2": 0.5}, mixture_radius=0)
    assert bank_objective(Fixed(), [source, target], (), **options) == pytest.approx(0.5)
    assert bank_objective(Fixed(), [source] * 40 + [target], (), **options) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        bank_objective(Fixed(), [source, target], (), version_weights={"v1": 1})
    tasks = [StateTask("a", "books", ("submit",), "source", 1, "same"),
             StateTask("b", "books", ("submit",), "target", 2, "same")]
    weights = {"source": 0.5, "target": 0.5}
    first = score_values([0.8, 0.2], tasks, (), epoch_weights=weights, radius=0)
    second = score_values([0.8] * 30 + [0.2], [tasks[0]] * 30 + [tasks[1]], (),
                          epoch_weights=weights, radius=0)
    assert first == pytest.approx(second) == pytest.approx(0.5)


def test_bundle_search_recovers_joint_gain_missed_by_singleton_ranking() -> None:
    class Complement:
        def predict(self, task, exposure):
            probability = 0.8 if {"auth_pair_a", "auth_pair_b"} <= set(exposure) else 0.1
            return StatePrediction(probability, probability, 0, True, 1)
    task = StateTask("joint", "books", ("authorize",), "source", 1, "joint")
    options = dict(epoch_weights={"source": 1}, capacity=2, depth=2, radius=0,
                   pool=("auth_pair_a", "auth_pair_b", "quota_hint"))
    scalar = select_bank(Complement(), [task], joint=False, **options)
    joint = select_bank(Complement(), [task], joint=True, **options)
    assert scalar["bank"] == ()
    assert set(joint["bank"]) == {"auth_pair_a", "auth_pair_b"}
    assert joint["objective"] > scalar["objective"]


def test_identical_low_sample_kernels_are_not_certified_equivalent() -> None:
    source = [{"key": ["source", "books", "submit", 7, "standard"],
               "counts": {"ok:15": 128}}]
    target = [{"key": ["target", "books", "submit", 7, "standard"],
               "counts": {"ok:15": 128}}]
    result = mechanism_tests(source, target)[0]
    assert result["tv_estimate"] == 0
    assert result["status"] == "inconclusive"
