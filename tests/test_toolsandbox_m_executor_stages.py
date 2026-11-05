import json

import pytest

from credit import toolsandbox_m_executor_stages as stages


def _generation(action):
    return {
        "prompt_sha256": "prompt",
        "agent_output": json.dumps(action),
        "hit_generation_limit": False,
    }


def test_repair_stage_executes_public_repair_without_local_completion(monkeypatch):
    job = {"tools": {"write": object()}, "calls": [], "generations": [],
           "done": False}
    proposed = {"tool": "write", "arguments": {"value": "old"}}
    repaired = {"tool": "write", "arguments": {"value": "new"}}
    seen = []

    monkeypatch.setattr(stages.adapter, "parse_request",
                        lambda raw: (proposed, None))
    monkeypatch.setattr(stages.v3, "fast_contract_violation",
                        lambda *args: "wrong public value")
    monkeypatch.setattr(stages.v3, "propose_fast_repair",
                        lambda *args: (repaired, "restore public value"))

    def execute(job, generation, action, u, max_calls):
        seen.append(action)
        turn = {"action": action}
        job["generations"].append(turn)
        job["calls"].append({"tool": action["tool"], "exception": None})
        return turn, True

    monkeypatch.setattr(stages.base, "_execute_action", execute)
    monkeypatch.setattr(stages.base, "_repair_metadata",
                        lambda turn, proposed, executed, reason, violation:
                        turn.update({"deterministic_repair": reason}))
    monkeypatch.setattr(stages.v3, "_maybe_local_complete",
                        lambda *args: pytest.fail("Repair must not complete locally"))

    stages.consume_repair(job, _generation(proposed), object(), 4, "request")
    assert seen == [repaired]
    assert job["done"] is False
    assert job["generations"][0]["deterministic_repair"] == \
        "restore public value"


def test_stage_dispatch_is_exact(monkeypatch):
    calls = []
    monkeypatch.setattr(stages.adapter, "consume",
                        lambda *args: (calls.append("Base"),
                        args[0]["generations"].append({})))
    monkeypatch.setattr(stages.v3, "consume_validate",
                        lambda *args: calls.append("Validate"))
    monkeypatch.setattr(stages, "consume_repair",
                        lambda *args: calls.append("Repair"))
    monkeypatch.setattr(stages.v3, "consume_fast",
                        lambda *args: calls.append("Full"))
    for stage in stages.STAGES:
        job = {"generations": []}
        stages.consume_stage(stage, job, {}, None, 1, "request")
    assert calls == list(stages.STAGES)


def test_unknown_stage_fails_closed():
    with pytest.raises(ValueError, match="unknown executor stage"):
        stages.consume_stage("Other", {}, {}, None, 1, "request")
