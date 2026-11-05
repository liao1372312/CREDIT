import numpy as np
import pytest

torch = pytest.importorskip("torch")

from credit.mechanisms import FiniteMechanisms
from credit.shared_mechanisms import OutcomeBatch, RESPONSE_LABELS, fit_shared
from credit.stateful import api_probes, decision_probes, make_tasks, run_episode


def test_differentiable_rollout_matches_reference_and_connects_both_kernels():
    policy = decision_probes(301, 32)
    api = api_probes(302, 32, "source") + api_probes(303, 32, "target")
    model = FiniteMechanisms(policy, api)
    traces = [run_episode(task, ("legacy_contract", "repair_rule"))
              for task in make_tasks(304, "gradient-check", 16)]
    batch = OutcomeBatch(traces, list(model.policy), list(model.api))
    p = torch.tensor(np.asarray([model.policy_distribution(key) for key in model.policy]),
                     dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[model.api[key].get(label, 0) / sum(model.api[key].values())
                       for label in RESPONSE_LABELS] for key in model.api],
                     dtype=torch.float64, requires_grad=True)
    actual = sorted(float(value.detach()) for values, _ in batch.predictions(p, k) for value in values)
    expected = sorted(model.predict(trace.task, trace.exposure).mean for trace in traces)
    assert np.allclose(actual, expected, atol=1e-10)
    batch.loss(p, k).backward()
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert k.grad is not None and torch.isfinite(k.grad).all() and k.grad.abs().sum() > 0


def test_shared_fit_uses_observations_and_never_claims_mle_certificate(monkeypatch):
    import credit.stateful as environment
    policy = decision_probes(311, 32)
    api = api_probes(312, 32, "source") + api_probes(313, 32, "target")
    traces = [run_episode(task, ("repair_rule",)) for task in make_tasks(314, "fit-check", 16)]
    def forbidden(*args, **kwargs):
        raise AssertionError("training queried simulator probabilities")
    monkeypatch.setattr(environment, "_policy", forbidden)
    monkeypatch.setattr(environment, "_api", forbidden)
    model = fit_shared(policy, api, traces, seed=315, epochs=3)
    assert len(model.training_history) == 3
    assert model.training_history[0]["outcome_bce"] > 0
    assert model.training_history[0]["group_norm"] > 0
    prediction = model.predict(traces[0].task, ("repair_rule",))
    assert 0 <= prediction.mean <= 1
    assert prediction.certified_lower == 0 and not model.certificate_valid
    assert model.predict(traces[0].task, ("repair_rule", "noop")).mean == prediction.mean
    reference = FiniteMechanisms([], [])
    reference.policy, reference.api = model.policy, model.api
    reference.total_cells = model.total_cells
    reference.certificate_valid = False
    assert reference.predict(traces[0].task, ("repair_rule",)).mean == pytest.approx(prediction.mean, abs=1e-10)
