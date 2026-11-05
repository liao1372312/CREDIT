import numpy as np
import pytest

torch = pytest.importorskip("torch")

from credit.shared_mechanisms import RESPONSE_LABELS
from credit.stateful import api_probes, decision_probes, make_tasks, run_episode
from credit.supported_mechanisms import SupportedKernels, fit_supported, response_contract


def test_public_response_contract_covers_snapshots_without_inventing_failures():
    from credit.stateful import _api, ACTIONS, FAMILIES, ROLES
    for family in FAMILIES:
        for epoch in ("source", "target"):
            for role in ROLES:
                for state in range(8):
                    for action in ACTIONS:
                        for status, after in _api(role, family, epoch, state, action):
                            assert f"{status}:{after}" in response_contract(role, state)
                        assert f"reject:{state}" in response_contract(role, state)
                        if role != "submit":
                            assert not any(int(label.split(":")[1]) & 8 for label in response_contract(role, state))


def test_masked_training_stays_finite_and_cannot_submit_at_lookup(monkeypatch):
    import credit.stateful as environment
    policy = decision_probes(901, 32)
    api = api_probes(902, 32, "source") + api_probes(903, 32, "target")
    traces = [run_episode(t, ("repair_rule",)) for t in make_tasks(904, "fit", 16)]
    def forbidden(*args, **kwargs):
        raise AssertionError("private simulator queried by training")
    monkeypatch.setattr(environment, "_api", forbidden)
    monkeypatch.setattr(environment, "_policy", forbidden)
    model = fit_supported(policy, api, traces, epochs=4, validation_traces=traces[:4])
    assert all(np.isfinite(row["penalized_objective"]) for row in model.training_history)
    for key, counts in model.api.items():
        assert set(counts) <= response_contract(key[2], key[3])
    from dataclasses import replace
    assert model.predict(replace(traces[0].task, roles=("lookup", "lookup")), ()).mean == 0
    assert model.certificate_valid is False
    assert 1 <= model.selected_epoch <= 4


def test_interaction_outputs_are_bounded_after_shared_head_rescaling():
    policy = decision_probes(911, 8)
    api = api_probes(912, 8, "source")
    network = SupportedKernels([tuple(r["key"]) for r in policy], [tuple(r["key"]) for r in api])
    x = torch.ones(3, network.interaction[0].in_features)
    with torch.no_grad():
        network.interaction[2].weight.mul_(100)
        network.interaction[2].bias.mul_(100)
    assert network.interaction(x).abs().max() <= 1
