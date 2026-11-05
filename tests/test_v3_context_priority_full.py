import numpy as np

from credit.v3.local import factory
from credit.v3.factorial import context_probabilities
from credit.v3_context_priority_full import prototypes


def test_priority_includes_intercept_disagreement():
    class ConstantModel:
        def __init__(self, value):
            self.value = value

        def predict_many(self, items):
            return np.full(len(items), self.value)

    make = factory(1501)
    contexts = [make('books/source', 'priority-unit', i) for i in range(2)]
    _, coefficients, costs = prototypes([ConstantModel(.2), ConstantModel(.8)], contexts, 1501)
    assert coefficients.shape == (2, 2, 46)
    np.testing.assert_allclose(coefficients[0, :, 0], .2)
    np.testing.assert_allclose(coefficients[1, :, 0], .8)
    np.testing.assert_allclose(coefficients[:, :, 1:], 0, atol=1e-14)
    assert np.all(np.var(coefficients, axis=0).sum(axis=1) > 0)
    probability = context_probabilities(coefficients, costs, epsilon=.1)
    assert np.all(probability > 0)
    np.testing.assert_allclose(probability.sum(), 1)
