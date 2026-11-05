import json
from pathlib import Path

import pytest

from credit.user_data import Candidate, SelectionCase, five_domain, group_split, hsc, normalized_query, processed
from credit.user_data_baselines import llm_prompt


def test_fallback_labels_are_not_primary_targets_and_prompts_hide_annotations(tmp_path):
    candidate = {"rank": 1, "api_name": "Weather", "domain": "Weather", "description": "Forecast",
                 "endpoint": "/weather", "method": "GET", "required_parameters": [{"name": "city", "default": "DO_NOT_EXPOSE_DEFAULT"}]}
    row = {"id": "x", "task": "Forecast", "user_query": "Forecast", "metadata": {"paper_domain": "Tourism"},
        "TaskList": [{"id": 1, "description": "Forecast", "top_candidates": [candidate], "required_inputs": ["city"],
            "llm_gold_selection": {"selected_rank": 1, "selected_api_name": "Weather", "confidence": 0.9,
                "rationale": "Fallback after LLM error"}}]}
    (tmp_path / "five_domain_1400_gold.jsonl").write_text(json.dumps(row) + "\n")
    cases, _ = five_domain(tmp_path)
    assert not cases[0].eligible and cases[0].exclusion_reason == "fallback_annotation"
    payload = json.dumps(cases[0].public_input())
    assert all(s not in payload for s in ("gold", "confidence", "rationale", "selected_rank", "DO_NOT_EXPOSE_DEFAULT"))
    _, prompt = llm_prompt(cases[0])
    assert "Weather" in prompt and "DO_NOT_EXPOSE_DEFAULT" not in prompt


def test_duplicate_query_groups_share_split_even_under_unicode_variations():
    a, b = normalized_query("  ＡＰＩ\n Query "), normalized_query("api query")
    assert a == b and group_split(a) == group_split(b)


def test_hsc_parallel_branches_all_map_to_solution_indices(tmp_path):
    directory = tmp_path / "HSC-main"; directory.mkdir()
    files = {"normalized_model": [{"function": "tts"}, {"function": "vision"}, {"function": "asr"}],
             "requirements": ["Task"], "workflow": [{"flow": [["tts", "vision"], ["asr"]], "obj_func": {}, "cons": {}}],
             "best_solution": [{"req_number": 0, "best_solution": [0, 1, 2], "best_value": 1}], "raw_models": {}}
    for name, value in files.items():
        (directory / f"{name}.json").write_text(json.dumps(value))
    tasks, stats = hsc(tmp_path)
    assert len(tasks[0]["nodes"]) == 3
    assert tasks[0]["nodes"][0]["stage"] == tasks[0]["nodes"][1]["stage"] == 0
    assert tasks[0]["nodes"][2]["stage"] == 1
    assert stats["all_branch_index_checks_passed"]
    files["best_solution"][0]["best_solution"] = [1, 0, 2]
    (directory / "best_solution.json").write_text(json.dumps(files["best_solution"]))
    with pytest.raises(ValueError, match="function mismatch"):
        hsc(tmp_path)


def test_candidate_pool_cannot_cross_official_service_splits(tmp_path):
    directory = tmp_path / "processed"; (directory / "splits").mkdir(parents=True)
    values = {"services": [{"service_id": "a"}, {"service_id": "b"}],
        "endpoints": [{"endpoint_id": "a1", "service_id": "a", "split": "train"},
                      {"endpoint_id": "b1", "service_id": "b", "split": "test"}],
        "feedback_events": [], "selection_tasks": [{"candidate_endpoint_ids": ["a1", "b1"], "split": "train"}]}
    for name, rows in values.items():
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "splits/service_splits.json").write_text(json.dumps({"a": "train", "b": "test"}))
    with pytest.raises(ValueError, match="candidate crosses"):
        processed(tmp_path)
