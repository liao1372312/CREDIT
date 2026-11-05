from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.optimize import linprog

from credit.v3.drift import (Level, TransportGate, archive_eligible, behavior_class,
                             role_voi, transfer_level)
from credit.v3.factorial import (characters, context_probabilities, eligible_slots,
                                estimate, regular_fraction)
from credit.v3.inference import certify, ppi, tune_lambda, worst_mixture
from credit.v3.local import StatefulExecutor, context, memories, stable_seed
from credit.v3.selection import BankSelector, Ledger
from credit.v3.types import (Context, Episode, Memory, Node, NodeObservation, Role,
                            SideEffect, reserved_slots, safe_replay)


def test_fraction_aliasing_is_unbiased_after_all_sign_flips():
    design = regular_fraction(4, [(0, 1)])
    draws, ys = [], []
    for signs in product([-1, 1], repeat=4):
        rows = design.rows * signs
        draws.append(rows)
        ys.append(0.5 + 0.04 * rows[:, 0] + 0.1 * rows[:, 0] * rows[:, 2] * rows[:, 3])
    result = estimate(design, draws, ys)
    assert result["beta"][1] == pytest.approx(0.04)
    assert result["banzhaf"][0] == pytest.approx(0.08)
    assert np.allclose(result["beta"][2:], 0)
    assert result["assignment_probability"] == 1 / 16
    assert result["draw_standard_error"].shape == (6,)


def test_admitted_columns_orthogonal_and_shared_additive_shock_cancels():
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11),
             (0, 2), (1, 3), (4, 6), (5, 7)]
    design = regular_fraction(12, pairs)
    assert design.runs == 32 and design.minimum_proven
    x = characters(design.rows, design.effects)
    assert np.allclose(x.T @ x, 32 * np.eye(23))
    assert np.allclose(np.full(32, 0.31) @ x[:, 1:], 0)


def test_reserved_slots_and_nonretrieved_memory_positivity():
    pool = memories()
    c = context("books", "source", "unit", 0, 31, template="train")
    rng = np.random.default_rng(19)
    observed = set()
    for _ in range(100):
        slots = eligible_slots(c, pool, ["lookup_key", "transform_rule"], 2, 1, rng)
        observed.add(slots[-1].uid)
        frame = reserved_slots(slots, (1, -1, 1))
        assert [len(t) for t in frame.token_ids] == [m.cost for m in slots]
        assert frame.token_ids[1] == (0,) * slots[1].cost
    assert "hidden_fix" in observed


def test_ensemble_probe_priorities_keep_uniform_exploration():
    p = context_probabilities(np.array([[[0., 0.], [0., 0.]], [[2., 1.], [0., 0.]]]),
                              [1., 2.], epsilon=.2)
    assert p.tolist() == pytest.approx([.9, .1])


def test_role_match_uses_consumed_fields_and_effect_authorization():
    a = Role(("id", "query"), ("id",), SideEffect.RO, "read")
    b = Role(("query", "id"), ("id",), SideEffect.RO, "read")
    assert a == b and a.key == b.key
    assert behavior_class(200, True, {"id": "uuid", "unused": "int"}, {}, a) == \
        behavior_class(200, True, {"id": "uuid", "unused": "string"}, {}, b)
    c = replace(b, consumed_outputs=("new_id",))
    assert a != c and a.partially_matches(c)


def test_exact_drift_bounds_and_unsupported_roles():
    role = Role(("id",), ("id",), SideEffect.ID, "read")
    assert transfer_level(role, role, "new", changed=0, fixtures=96, support=100).level == Level.V
    assert transfer_level(role, role, "new", changed=96, fixtures=96, support=100).level == Level.D
    assert transfer_level(role, role, "new", changed=0, fixtures=2, support=100).level == Level.U
    assert transfer_level(role, role, "new", changed=0, fixtures=96, support=0).level == Level.X
    assert transfer_level(role, replace(role, authorization="other"), "new",
                          changed=0, fixtures=96, support=100).level == Level.X


def test_irreversible_and_reversible_restore_guards():
    role = Role(("id",), ("receipt",), SideEffect.IR, "write")
    assert not safe_replay(role, mode="shadow", version_pinned=True, snapshot_restored=True,
                           checkpoint_hash="same", restored_hash="same")
    assert not safe_replay(role, mode="sandbox", version_pinned=True, snapshot_restored=False,
                           checkpoint_hash="same", restored_hash="same")
    assert not safe_replay(replace(role, side_effect=SideEffect.RV), mode="sandbox", version_pinned=True,
                           snapshot_restored=True, checkpoint_hash="before", restored_hash="after")


def test_ppi_rectifies_surrogate_bias_and_accounts_unlabeled_variance():
    y = np.array([0, 1, 1, 0] * 100)
    f = y * .5 + .4
    u = np.array([.4, .9] * 500)
    e = ppi(y, f, u, .7)
    assert e.mean == pytest.approx(.5)
    expected = np.var(y - .7 * f, ddof=1) / len(y) + .7**2 * np.var(u, ddof=1) / len(u)
    assert e.standard_error**2 == pytest.approx(expected)
    assert tune_lambda(y, f, labeled_n=200, unlabeled_n=1000) > 0
    assert tune_lambda(y, f, labeled_n=200, unlabeled_n=1000, transported=False) == 0
    assert tune_lambda(y, -f, labeled_n=200, unlabeled_n=1000) == 0


def test_aggregate_guard_prevents_one_stratum_regression():
    means = {"good": ppi([.2] * 200, [0.] * 200, [0.] * 1000),
             "bad": ppi([-.1] * 200, [0.] * 200, [0.] * 1000)}
    result = certify(means, {"good": .9, "bad": .1})
    assert result["aggregate_lower"] > 0
    assert not result["passed"]
    assert result["familywise_error_budget"] == .1
    assert result["guarantee"] == "asymptotic"


def test_cost_and_finite_sample_guard_are_enforced():
    e = ppi([.02] * 200, [0.] * 200, [0.] * 1000)
    assert not certify({"a": e}, {"a": 1}, cost_difference=10, cost_penalty=.003)["passed"]
    assert certify({"a": e}, {"a": 1})["passed"]
    assert not certify({"a": e}, {"a": 1}, method="bernstein")["passed"]
    assert not certify({"a": e}, {"a": 1}, minimum_n=300)["passed"]


def test_robust_closed_form_matches_linear_program():
    rng = np.random.default_rng(7)
    for _ in range(12):
        v, w, r = rng.random(5), rng.dirichlet(np.ones(5)), float(rng.random() * 2)
        # Variables are weights q and absolute deviation t.
        constraints = np.vstack([np.hstack([np.eye(5), -np.eye(5)]),
                                 np.hstack([-np.eye(5), -np.eye(5)]),
                                 np.r_[np.zeros(5), np.ones(5)]])
        answer = linprog(np.r_[v, np.zeros(5)], A_ub=constraints, b_ub=np.r_[w, -w, r],
                         A_eq=[np.r_[np.ones(5), np.zeros(5)]], b_eq=[1], bounds=(0, None), method="highs")
        assert answer.success
        assert worst_mixture(v, w, r) == pytest.approx(answer.fun)


def test_role_voi_counts_existing_bank_unblocking_and_no_retrieval_archive():
    role = Role(("id",), ("id",), SideEffect.ID, "read")
    m = Memory("m", "hint", (1,), frozenset({role.key}), "unit")
    value = role_voi(("m",), (m,), role.key, lambda b: .2 if b else 0,
                     capacity=1, probability=.5, probe_calls=1, probe_penalty=.01,
                     incumbent_score=0)
    assert value == pytest.approx(.09)
    assert not archive_eligible(100, 3, [])
    assert not archive_eligible(100, 3, [-.1, .01])
    assert archive_eligible(100, 3, [-.1, 0])


def test_v3_incremental_replay_matches_full_with_recovery_and_new_versions():
    engine = StatefulExecutor(repair_budget=1)
    for family in ("books", "travel"):
        for epoch in ("source", "target"):
            for seed in range(12):
                c = context(family, epoch, "replay-test", seed, seed, template="alternate")
                base = engine.run(c, ("lookup_key", "transform_rule"))
                a = ("lookup_key", "transform_rule", "auth_pair_a", "legacy_contract")
                incremental, full = engine.replay(c, a, base), engine.run(c, a)
                assert incremental.state_hash == full.state_hash
                assert incremental.events == full.events
                assert incremental.observations == full.observations
                assert incremental.verified == full.verified
                assert incremental.calls + incremental.reused_calls == full.calls
                assert incremental.restored_hash == incremental.checkpoint_hash


def test_new_snapshot_cannot_reuse_old_prefix():
    engine = StatefulExecutor()
    c = context("books", "source", "replay", 0, 7)
    base = engine.run(c, ())
    with pytest.raises(ValueError, match="mismatch"):
        engine.replay(replace(c, seed=99), (), base)


def test_bad_model_can_screen_but_cannot_switch_to_harmful_bank():
    role = Role(("id",), ("id",), SideEffect.RO, "read")
    m = Memory("bad", "misleading", (1,), frozenset({role.key}), "unit")
    gate = TransportGate([transfer_level(role, role, "v", changed=0, fixtures=96, support=100)])

    def make(k, phase, index):
        return Context(f"{phase}:{index}", k, (Node("node", role, "v"),), (), stable_seed(phase, index))

    class BadModel:
        fit_report = {}
        def predict_many(self, items):
            return np.array([1. if a else 0. for c, a in items])

    class Engine:
        def run(self, c, a, *, slots=None):
            u = (c.seed % 100_000) / 100_000
            y = u < (.1 if a else .6)
            return Episode(c, a, (NodeObservation("node", "ok"),), y, 1, "end", "pin", "pin", True)

    ledger = Ledger({"pilot": 80, "certification": 800})
    selector = BankSelector((m,), (BadModel(),), gate, lambda c, b: b, make, Engine(), ledger,
                            mixture={"stratum": 1.}, capacity=1)
    freeze = []
    result = selector.switch((), [("bad",)], pilot_n=40, certification_n=400,
                             unlabeled_n=500, freeze=freeze.append)
    assert not result["switched"] and result["bank"] == ()
    assert freeze and result["certificates"][0]["aggregate_lower"] < 0
    assert ledger.episodes == {"pilot": 80, "certification": 800}
    assert {r["episode"].context.uid for r in ledger.records if r["phase"] == "pilot"}.isdisjoint(
        {r["episode"].context.uid for r in ledger.records if r["phase"] == "certification"})


def test_insufficient_budget_keeps_incumbent_without_execution():
    class CannotRun:
        def run(self, *args, **kwargs):
            raise AssertionError("budget should prevent execution")
    role = Role(("id",), ("id",), SideEffect.RO, "read")
    m = Memory("x", "hint", (1,), frozenset({role.key}), "unit")
    selector = BankSelector((m,), (), TransportGate(), None, None, CannotRun(), Ledger({}),
                            mixture={"s": 1}, capacity=1)
    assert not selector.switch((), [("x",)])["switched"]


def test_service_and_ambiguous_labels_do_not_train_correctness_or_recovery():
    torch = pytest.importorskip("torch")
    from credit.v3.outcome import OutcomeModel
    logits = torch.zeros((4, 2), requires_grad=True)
    loss = OutcomeModel._loss(logits, torch.tensor([0, 1, 2, 3]), torch.tensor([0., 1., 0., 0.]), 0)
    loss.backward()
    assert torch.equal(logits.grad[2:], torch.zeros((2, 2)))
    assert logits.grad[0, 1] == 0 and logits.grad[1, 1] != 0


def test_exposure_training_rejects_logged_or_leaked_calibration():
    pytest.importorskip("torch")
    from credit.v3.outcome import OutcomeModel
    c = context("books", "source", "fit", 0, 1, template="train")
    engine = StatefulExecutor()
    train = engine.run(c, ())
    validation = engine.run(replace(c, uid="validation", seed=2), ())
    model = OutcomeModel(memories(), [c])
    with pytest.raises(ValueError, match="randomized evidence"):
        model.fit([train], [validation])
    train = replace(train, randomized=True, assignment_probability=1 / 64)
    with pytest.raises(ValueError, match="overlap"):
        model.fit([train], [train])
