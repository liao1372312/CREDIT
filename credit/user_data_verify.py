"""Verify supplied-data provenance, public prompts and baseline summaries."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .user_data import digest, jsonl, load_selection_case
from .user_data_baselines import evaluate, llm_prompt


def verify(data, results):
    manifest = json.loads((data / "manifest.json").read_text())
    frozen = json.loads((results / "frozen_protocol.json").read_text())
    if digest(data / "selection_cases.jsonl") != frozen["data_sha256"] or digest(data / "manifest.json") != frozen["data_manifest_sha256"]:
        raise ValueError("canonical data differs from frozen input")
    for name, value in frozen["source_sha256"].items():
        if digest(Path(__file__).with_name(name)) != value:
            raise ValueError(f"source changed: {name}")
    source = Path(manifest["source_root"])
    for name, value in manifest["files_sha256"].items():
        if digest(source / name) != value:
            raise ValueError(f"source data changed: {name}")
    raw_hashes = json.loads((data / "raw_api_hashes.json").read_text())
    for name, value in raw_hashes.items():
        if digest(source / name) != value:
            raise ValueError(f"raw API data changed: {name}")
    all_cases = [load_selection_case(row) for row in jsonl(data / "selection_cases.jsonl")]
    cases = [case for case in all_cases if case.split == "test" and case.eligible]
    by_id = {case.uid: case for case in cases}
    groups = defaultdict(set)
    for case in all_cases:
        groups[case.dataset, case.group].add(case.split)
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("split group leaked")
    predictions = defaultdict(dict)
    with (results / "raw_selection_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            case = by_id[row["case_uid"]]
            if case.uid in predictions[row["method"]] or row["gold_uid"] != case.gold_uid:
                raise ValueError("duplicate prediction or annotation mismatch")
            if row["prediction_uid"] not in {c.uid for c in case.candidates}:
                raise ValueError("prediction outside candidate pool")
            if int(row["label_agreement"]) != int(row["prediction_uid"] == case.gold_uid):
                raise ValueError("saved agreement differs")
            predictions[row["method"]][case.uid] = row["prediction_uid"]
    if any(set(choices) != set(by_id) for choices in predictions.values()):
        raise ValueError("test coverage differs by method")
    qwen_rows = 0
    if frozen["llm"]:
        from hashlib import sha256
        from transformers import AutoTokenizer
        metadata = json.loads((results / "qwen_manifest.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(metadata["model_path"], local_files_only=True)
        seen = set()
        for row in jsonl(results / "qwen_predictions.jsonl"):
            case = by_id[row["case_uid"]]
            if case.uid in seen:
                raise ValueError("duplicate Qwen context")
            seen.add(case.uid)
            ordered, prompt = llm_prompt(case)
            chat = tokenizer.apply_chat_template([{"role": "system", "content": metadata["system_prompt"]},
                {"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            if sha256(chat.encode()).hexdigest() != row["prompt_sha256"] or [c.uid for c in ordered] != row["candidate_order"]:
                raise ValueError("public prompt/candidate order differs")
            logits = np.asarray(row["choice_logits"])
            probabilities = np.exp(logits - logits.max()); probabilities /= probabilities.sum()
            if not np.allclose(probabilities, row["probabilities"], atol=1e-6) or ordered[int(logits.argmax())].uid != row["prediction_uid"]:
                raise ValueError("choice does not match frozen logits")
            if predictions["Qwen NoMem"][case.uid] != row["prediction_uid"]:
                raise ValueError("Qwen raw result differs from score cache")
            qwen_rows += 1
        if seen != set(by_id):
            raise ValueError("Qwen coverage incomplete")
    _, recomputed = evaluate(cases, predictions, all_cases)
    original = json.loads((results / "summary.json").read_text())["main"]
    key = lambda row: (row["dataset"], row["method"])
    if sorted(recomputed, key=key) != sorted(original, key=key):
        raise ValueError("summary differs from raw predictions")
    report = {"source_files_unchanged": True, "raw_api_files_verified": len(raw_hashes),
        "no_groups_cross_splits": True, "all_public_prompt_hashes_match": True,
        "qwen_contexts_verified": qwen_rows, "raw_selection_results_verified": sum(len(c) for c in predictions.values()),
        "summary_matches_raw_predictions": True, "execution_api_calls": 0,
        "verifier_source_sha256": digest(Path(__file__)),
        "limits": "Verifies saved offline choices and annotations; does not verify live functional execution."}
    (results / "integrity.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/user_data_v1"))
    parser.add_argument("--results", type=Path, default=Path("results/user_data_baselines_v1"))
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.results), indent=2))


if __name__ == "__main__":
    main()
