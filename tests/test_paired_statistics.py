import pytest
from credit.paired_statistics import summarize


def test_identical_methods_have_identical_intervals_and_zero_paired_effect():
    rows = [{"method": method, "seed": seed, "family": family, "task": f"{family}-{i}",
             "success": (i + seed) % 2} for method in ("corrected/CREDIT", "original/CREDIT")
            for seed in (1, 2) for family in ("a", "b") for i in range(3)]
    main, comparisons = summarize(rows, "corrected/CREDIT", repeats=100)
    assert main[0]["ci_low"] == main[1]["ci_low"]
    assert main[0]["ci_high"] == main[1]["ci_high"]
    assert comparisons[0]["delta_vsr"] == 0
    assert comparisons[0]["seed_sign_flip_p"] == 1
    with pytest.raises(ValueError, match="coverage"):
        summarize(rows[:-1], "corrected/CREDIT")
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + rows[:1], "corrected/CREDIT")
