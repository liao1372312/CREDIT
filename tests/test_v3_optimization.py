from dataclasses import replace

import numpy as np
import pytest
import torch

from credit.v3.local import context, memories, StatefulExecutor
from credit.v3.outcome import OutcomeModel
from credit.v3_optimized_model import StateAwareOutcomeModel, ValidatedOutcomeModel, load_model
from credit.v3_optimization import deterministic_screen, designs
from credit.v3.selection import Ledger


def test_features_use_public_state_and_graph_but_ignore_noise_and_task_ids():
    c = context('books','source','features',0,1,template='train')
    model = StateAwareOutcomeModel(memories(),[c])
    x = model._pack([(c,c.nodes[0],())])[0]
    d = replace(c,uid='unrelated',seed=987)
    assert torch.equal(x, model._pack([(d,d.nodes[0],())])[0])
    state = replace(c,initial_state=7)
    assert not torch.equal(x, model._pack([(state,state.nodes[0],())])[0])
    future = context('books','source','features',1,1,template='future')
    assert not torch.equal(x,model._pack([(future,future.nodes[0],())])[0])


def test_public_success_postconditions_match_observed_prefix_states():
    c = context('books','source','prefix',0,1,template='future')
    model = StateAwareOutcomeModel(memories(),[c])
    checked = 0
    for seed in range(30):
        task = replace(c,seed=seed)
        e = StatefulExecutor().run(task,('lookup_key','transform_rule','auth_pair_a',
                                         'auth_pair_b','legacy_contract','repair_rule'))
        first = {event['node']:event for event in e.events if event['attempt'] == 0}
        for n,o in zip(task.nodes,e.observations):
            assert model.successful_prefix_state(task,n) == first[n.uid]['state_before']
            checked += 1
            if o.label != 'ok' and not o.recovered:
                break
    assert checked > 30


def test_training_uses_observed_preinvocation_state_after_a_failed_prefix():
    c = context('books','source','failed-prefix',0,1,template='train')
    model = StateAwareOutcomeModel(memories(),[c])
    e = next(StatefulExecutor().run(replace(c,seed=seed),()) for seed in range(200)
             if StatefulExecutor().run(replace(c,seed=seed),()).events[0]['response_status'] != 'ok'
             and any(event['node'] == 'node-1' and event['state_before'] == 0
                     for event in StatefulExecutor().run(replace(c,seed=seed),()).events))
    packed, _, _ = model.pack_observations([e])
    assert torch.equal(packed[0][1,12:16],torch.zeros(4))
    prediction = model._pack([(c,c.nodes[1],())])
    assert prediction[0][0,12] == 1


def test_compressed_masked_likelihood_and_gradients_equal_original():
    model = OutcomeModel(memories(),[context('books','source','unit',0,1,template='train')])
    c = context('books','source','unit',0,1,template='train')
    packed = model._pack([(c,c.nodes[0],())]*5)
    labels = torch.tensor([0,1,1,2,3])
    recovery = torch.tensor([0.,1.,0.,0.,0.])
    logits = torch.tensor([[.3,-.8]],requires_grad=True)
    original = OutcomeModel._loss(logits.expand(5,2),labels,recovery,0)
    original.backward()
    expected_gradient = logits.grad.clone()
    logits.grad = None
    compressed, counts = ValidatedOutcomeModel.compress(packed,labels,recovery)
    result = ValidatedOutcomeModel.compressed_loss(logits,counts)
    result.backward()
    assert len(compressed[0]) == 1
    assert torch.allclose(result,original)
    assert torch.allclose(logits.grad,expected_gradient)


def test_validation_is_separate_from_training_and_calibration():
    c = context('books','source','unit',0,1,template='train')
    e = replace(StatefulExecutor().run(c,()),randomized=True,assignment_probability=1/64)
    cal = replace(e,context=replace(c,uid='cal',seed=2))
    model = StateAwareOutcomeModel(memories(),[c])
    with pytest.raises(ValueError,match='overlap'):
        model.fit([e],[cal],validation=[e],epochs=1)
    val = replace(e,context=replace(c,uid='val'))
    with pytest.raises(ValueError,match='noise overlap'):
        model.fit([e],[cal],validation=[val],epochs=1)


def test_ready_submit_design_preserves_randomized_nonretrieved_memory(tmp_path):
    c = context('books','source','ready',0,1,template='future_ready_submit')
    ledger = Ledger({'factorial':128,'replay_audit':2})
    episodes, reports = designs([c],memories(),StatefulExecutor(),ledger,
                                phase='factorial',seed=5,draws=1)
    assert reports[0]['assignment_probability'] == 1/8
    assert 'hidden_fix' in reports[0]['memory_ids']
    assert len(episodes) == 8
    assert all(e.replay_valid and e.randomized for e in episodes)


def test_screening_order_is_independent_of_memory_and_bundle_input_order():
    pool = memories()
    visited = []
    def score(bank):
        visited.append(bank)
        return len(bank)/10
    first = deterministic_screen(pool,score,bundles=[('auth_pair_b','auth_pair_a'),('conflict','repair_rule')])
    order = visited.copy()
    visited.clear()
    second = deterministic_screen(tuple(reversed(pool)),score,
                                  bundles=[('repair_rule','conflict'),('auth_pair_a','auth_pair_b')])
    assert first == second
    assert visited == order


def test_versioned_model_save_load_preserves_state_sensitive_predictions(tmp_path):
    c = context('books','source','unit',0,1,template='train')
    model = StateAwareOutcomeModel(memories(),[c])
    from credit.v3.audit import SavedCalibration
    model.calibrator = SavedCalibration(np.asarray([0.,1.]),np.asarray([0.,1.]))
    path = tmp_path/'model.pt'
    model.save(path)
    restored = load_model(path,memories(),[c])
    queries = [(c,()),(replace(c,initial_state=7),('repair_rule',))]
    assert np.array_equal(model.predict_many(queries),restored.predict_many(queries))
