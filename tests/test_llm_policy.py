import json

import numpy as np
import pytest

pytest.importorskip("transformers")

from credit.llm_policy import contexts, install_policy
from credit.stateful import decision_key


def test_cached_llm_policy_covers_contexts_and_preserves_local_dependencies(tmp_path, monkeypatch):
    import credit.stateful as environment
    rows = [{"key": list(key), "probabilities": [6, 2, 1, 1]} for key, prompt in contexts()]
    assert len(rows) == 336 and len({tuple(row["key"]) for row in rows}) == 336
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(rows))
    # Restore the actual environment policy after exercising adapter installation.
    monkeypatch.setattr(environment, "_policy", environment._policy)
    install_policy(path)
    assert np.allclose(environment._policy("submit", 0, 0, ()), [0.6, 0.2, 0.1, 0.1])
    assert np.allclose(environment._policy("submit", 0, 0, ("noop", "lookup_key")), [0.6, 0.2, 0.1, 0.1])
    for role in environment.ROLES:
        key = decision_key(role, 0, "transient", environment.MEMORIES)
        assert key in {tuple(row["key"]) for row in rows}
