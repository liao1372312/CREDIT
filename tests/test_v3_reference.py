"""Algebraic reference checks with no private simulator oracle."""
import numpy as np
import pytest
from credit.v3_reference import coalitions, coefficients, credits, run


def test_coalition_credit_recovers_third_order_shapley():
    _, z = coalitions(3)
    values = .4+.05*z[:,0]-.03*z[:,1]+.08*z.prod(axis=1)
    _, _, banzhaf, shapley = credits(values,z)
    assert np.allclose(banzhaf,[.1,-.06,0])
    assert np.allclose(shapley,[.1+ .16/3,-.06+.16/3,.16/3])
    assert np.isclose(shapley.sum(),values[-1]-values[0])


def test_walsh_transform_reconstructs_an_arbitrary_cube():
    _,z=coalitions(4)
    values=np.random.default_rng(9).uniform(size=16)
    effects,beta=coefficients(values,z)
    from credit.v3.factorial import characters
    assert np.allclose(characters(z,effects) @ beta, values)


def test_reference_cannot_start_before_main_freeze(tmp_path):
    with pytest.raises(ValueError,match='main benchmark must finish'):
        run(tmp_path/'data',tmp_path/'unfinished',tmp_path/'reference')
    assert not (tmp_path/'reference').exists()
