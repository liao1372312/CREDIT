import numpy as np
import pytest

from credit.paper_attribution import attribution_weights, pair_f1, signed_pairs
from credit.toolsandbox_local_v3 import stable_exception_text
from credit.toolsandbox_verify import paired_report


def test_shapley_recovers_signed_synergy_and_efficiency():
    # Independent closed form: effects 2, -1, 0 plus a 6-point 0/1 synergy.
    values = np.array([2 * bool(m & 1) - bool(m & 2) + 6 * (m & 3 == 3) for m in range(8)])
    phi = attribution_weights(3) @ values
    np.testing.assert_allclose(phi, [5, 2, 0], atol=1e-12)
    assert phi.sum() == pytest.approx(values[-1] - values[0])
    pairs = signed_pairs(values, 3, .01)
    assert pairs[0][2:] == (6., 1)
    assert pair_f1(pairs, pairs) == 1
    assert pair_f1(signed_pairs(np.zeros(8), 3, .01), signed_pairs(np.zeros(8), 3, .01)) is None


def test_tool_exception_normalization_preserves_actual_hex_values():
    first = "NoDataError: predicate=<Expr ['person_id'] at 0x7F50A3CCA270> found; user id=0xA12"
    second = "NoDataError: predicate=<Expr ['person_id'] at 0xBBCC> found; user id=0xA12"
    assert stable_exception_text(first) == stable_exception_text(second)
    assert 'id=0xA12' in stable_exception_text(first)
    assert stable_exception_text(None) is None


def test_task_cluster_comparison_rejects_missing_method_coverage():
    def episode(task, method, success):
        return {'task': task, 'family': task, 'method': method, 'stage': 'future_test',
            'success': success, 'tool_calls': 1, 'upstream_evaluation': {'similarity': float(success)}}
    settings = {'bootstrap_seed': 422, 'paired_bootstrap_replicates': 100}
    data = [episode(t, m, int(m == 'Memory')) for t in ('a', 'b') for m in ('NoMem', 'Memory')]
    report = paired_report(data, settings)
    effect = next(r for r in report['table'] if r['method'] == 'Memory' and r['metric'] == 'strict_VSR')
    assert effect['delta_vs_NoMem'] == 1
    assert effect['paired_delta_CI95'] == [1., 1.]
    assert effect['relative_change_vs_NoMem'] is None
    with pytest.raises(ValueError, match='unpaired'):
        paired_report(data[:-1], settings)
