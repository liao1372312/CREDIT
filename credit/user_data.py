"""Read-only adapters and provenance audit for the user-provided datasets.

Selection annotations are evaluation labels, not verified execution outcomes.
Public model inputs exclude gold selections, scores, feedback and confidence.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import argparse
import json
from pathlib import Path
import re
import unicodedata

DEFAULT_ROOT = Path("/home/lxc/next_work/A/WWW/dataset")


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_query(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def group_split(group, seed=42):
    bucket = int(sha256(f"{seed}:{group}".encode()).hexdigest()[:12], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


@dataclass(frozen=True)
class Candidate:
    uid: str
    name: str
    description: str
    method: str
    endpoint: str
    required_parameters: tuple[str, ...]

    def text(self):
        return f"{self.name}. {self.description}. {self.method} {self.endpoint}. Required: {', '.join(self.required_parameters)}"


@dataclass(frozen=True)
class SelectionCase:
    uid: str
    task_uid: str
    dataset: str
    split: str
    domain: str
    query: str
    candidates: tuple[Candidate, ...]
    gold_uid: str
    label_source: str
    eligible: bool
    exclusion_reason: str
    group: str
    step_id: int
    depends_on: tuple[int, ...]
    required_inputs: tuple[str, ...]

    def public_input(self, seed=42):
        # Do not expose annotation rank, confidence, rationale or outcome fields.
        order = sorted(self.candidates, key=lambda c: sha256(f"{seed}:{self.uid}:{c.uid}".encode()).hexdigest())
        return {"uid": self.uid, "query": self.query, "required_inputs": list(self.required_inputs),
                "candidates": [asdict(c) for c in order]}


def _parameters(value):
    return tuple(str(p.get("name", "")) if isinstance(p, dict) else str(p) for p in (value or []))


def candidate_from_record(record, uid=None):
    identity = uid or record.get("record_id") or sha256(json.dumps([
        record.get("api_name"), record.get("method"), record.get("endpoint")], ensure_ascii=False).encode()).hexdigest()[:24]
    return Candidate(str(identity), str(record.get("api_name", "")), str(record.get("description", "")),
                     str(record.get("method", "")), str(record.get("endpoint", "")),
                     _parameters(record.get("required_parameters")))


def five_domain(root):
    cases, task_records = [], []
    for row in jsonl(root / "five_domain_1400_gold.jsonl"):
        group = normalized_query(row.get("task", row["user_query"]))
        split = group_split(group)
        task_records.append({"uid": row["id"], "group": group, "split": split,
                             "domain": row["metadata"]["paper_domain"], "steps": len(row["TaskList"])})
        for step in row["TaskList"]:
            gold = step["llm_gold_selection"]
            selected = [c for c in step["top_candidates"] if c["rank"] == gold.get("selected_rank")]
            reason = ("fallback_annotation" if "fallback" in str(gold.get("rationale", "")).casefold()
                      else "zero_confidence_annotation" if not gold.get("confidence", 0) > 0
                      else "invalid_annotation" if len(selected) != 1 or selected[0]["api_name"] != gold.get("selected_api_name")
                      else "")
            candidates = tuple(candidate_from_record(c) for c in step["top_candidates"])
            if len({c.uid for c in candidates}) != len(candidates):
                reason = "duplicate_candidate_identity"
            gold_uid = candidate_from_record(selected[0]).uid if len(selected) == 1 else ""
            cases.append(SelectionCase(f"five:{row['id']}:{step['id']}", f"five:{row['id']}", "five_domain",
                split, row["metadata"]["paper_domain"], step["description"], candidates, gold_uid,
                "gpt-4o-mini_annotation", not reason, reason, group, int(step["id"]), (),
                tuple(str(x) for x in step.get("required_inputs", []))))
    return cases, task_records


def emp(root):
    cases, task_records = [], []
    for row in jsonl(root / "emp_data2_synthetic_compositions.jsonl"):
        group, steps = normalized_query(row["task"]), {s["id"]: s for s in row["TaskList"]}
        split = group_split(group)
        task_records.append({"uid": row["id"], "group": group, "split": split,
            "composition_relation": row["metadata"].get("composition_relation"), "steps": len(steps)})
        for selected in row["selected_services"]:
            step = steps[selected["step_id"]]
            candidates = tuple(candidate_from_record(c) for c in selected["top_candidates"])
            gold_uid = str(selected["selection"]["record_id"])
            reason = "" if gold_uid in {c.uid for c in candidates} else "gold_not_in_candidates"
            if len({c.uid for c in candidates}) != len(candidates):
                reason = "duplicate_candidate_identity"
            if any(dep not in steps or dep >= step["id"] for dep in step.get("depends_on", [])):
                reason = "invalid_dependency"
            cases.append(SelectionCase(f"emp:{row['id']}:{step['id']}", f"emp:{row['id']}", "emp", split,
                str(step["domain"]), step["description"], candidates, gold_uid, "synthetic_composition_annotation",
                not reason, reason, group, int(step["id"]), tuple(step.get("depends_on", [])),
                tuple(str(x) for x in step.get("required_inputs", []))))
    return cases, task_records


def processed(root):
    directory = root / "processed"
    services = {s["service_id"]: s for s in jsonl(directory / "services.jsonl")}
    endpoints = {e["endpoint_id"]: e for e in jsonl(directory / "endpoints.jsonl")}
    service_splits = json.loads((directory / "splits/service_splits.json").read_text())
    if set(services) != set(service_splits):
        raise ValueError("service split inventory mismatch")
    for row in endpoints.values():
        if row["split"] != service_splits[row["service_id"]]:
            raise ValueError("endpoint crosses service split")
    events, counts = 0, Counter()
    for event in jsonl(directory / "feedback_events.jsonl"):
        endpoint = endpoints[event["endpoint_id"]]
        if event["split"] != endpoint["split"] or event["service_id"] != endpoint["service_id"]:
            raise ValueError("feedback crosses service split")
        events += 1
        counts[str(event.get("success"))] += 1
    cases = []
    for row in jsonl(directory / "selection_tasks.jsonl"):
        if any(endpoints[key]["split"] != row["split"] for key in row["candidate_endpoint_ids"]):
            raise ValueError("candidate crosses service split")
        candidates = tuple(Candidate(key, endpoints[key]["endpoint_name"],
            f"{endpoints[key]['service_name']}. {endpoints[key]['service_description']}. {endpoints[key]['endpoint_description']}",
            endpoints[key]["method"], endpoints[key]["url"], _parameters(endpoints[key]["required_parameters"]))
            for key in row["candidate_endpoint_ids"])
        if row["positive_endpoint_id"] not in {c.uid for c in candidates}:
            raise ValueError("positive endpoint missing")
        cases.append(SelectionCase(f"processed:{row['task_id']}", f"processed:{row['task_id']}", "processed",
            row["split"], row["positive_category"], row["query"], candidates, row["positive_endpoint_id"],
            "endpoint_description_weak_supervision", True, "", row["positive_service_id"], 1, (), ()))
    statistics = {"services": len(services), "endpoints": len(endpoints), "feedback_events": events,
        "feedback_success_values": dict(counts), "service_split_counts": dict(Counter(service_splits.values())),
        "all_service_split_checks_passed": True,
        "label_warning": "Queries are derived from positive endpoint descriptions; lexical matching may be trivial."}
    return cases, statistics


def hsc(root):
    directory = root / "HSC-main"
    models = json.loads((directory / "normalized_model.json").read_text())
    requirements = json.loads((directory / "requirements.json").read_text())
    workflows = json.loads((directory / "workflow.json").read_text())
    solutions = json.loads((directory / "best_solution.json").read_text())
    raw = json.loads((directory / "raw_models.json").read_text())
    solution_map = {s["req_number"]: s for s in solutions}
    if len(requirements) != len(workflows) or set(solution_map) != set(range(len(workflows))):
        raise ValueError("HSC requirement/workflow/solution alignment mismatch")
    tasks = []
    for index, workflow in enumerate(workflows):
        # Every category in an inner stage is a mandatory branch, not a choice.
        nodes = [(stage, branch, role) for stage, functions in enumerate(workflow["flow"])
                 for branch, role in enumerate(functions)]
        indices = solution_map[index]["best_solution"]
        if len(nodes) != len(indices):
            raise ValueError("HSC flattened branch count mismatch")
        for (_, _, role), model_index in zip(nodes, indices):
            if not isinstance(model_index, int) or not 0 <= model_index < len(models) or models[model_index]["function"] != role:
                raise ValueError("HSC solution branch/function mismatch")
        signature = json.dumps(workflow["flow"], separators=(",", ":"))
        tasks.append({"uid": f"hsc:{index}", "requirement": requirements[index], "stages": workflow["flow"],
            "nodes": [{"stage": stage, "branch": branch, "role": role} for stage, branch, role in nodes],
            "objective": workflow["obj_func"], "constraints": workflow["cons"],
            "label_model_indices": indices, "label_best_value": solution_map[index]["best_value"],
            "split": group_split(signature), "group": sha256(signature.encode()).hexdigest(),
            "label_source": "offline_optimizer_reference"})
    return tasks, {"raw_models": len(raw), "normalized_models": len(models), "workflows": len(workflows),
        "all_branch_index_checks_passed": True, "outer_stages_are_serial_inner_functions_are_parallel_required_branches": True,
        "readme_normalized_count": 17535, "actual_normalized_count": len(models),
        "readme_semantics_warning": "README describes inner functions as alternatives; actual labels match all flattened mandatory functions.",
        "objective_warning": "QoS aggregation/constraint and best_value semantics require confirmation; labels are not execution outcomes."}


def load_selection_case(row):
    return SelectionCase(**{**row, "candidates": tuple(Candidate(**{**c, "required_parameters": tuple(c["required_parameters"])})
        for c in row["candidates"]), "depends_on": tuple(row["depends_on"]), "required_inputs": tuple(row["required_inputs"])})


def adapt(root, output):
    if output.exists():
        raise FileExistsError("use a new output directory; source data stays read-only")
    cases_five, tasks_five = five_domain(root)
    cases_emp, tasks_emp = emp(root)
    cases_processed, processed_stats = processed(root)
    tasks_hsc, hsc_stats = hsc(root)
    cases = cases_five + cases_emp + cases_processed
    if len({case.uid for case in cases}) != len(cases):
        raise ValueError("duplicate canonical case ID")
    output.mkdir(parents=True)
    with (output / "selection_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    with (output / "hsc_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks_hsc:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    statistics = {}
    for source in ("five_domain", "emp", "processed"):
        subset = [case for case in cases if case.dataset == source]
        statistics[source] = {"tasks": len({c.task_uid for c in subset}), "steps": len(subset),
            "eligible_steps": sum(c.eligible for c in subset), "exclusions": dict(Counter(c.exclusion_reason for c in subset if not c.eligible)),
            "split_step_counts": dict(Counter(c.split for c in subset)),
            "eligible_split_step_counts": dict(Counter(c.split for c in subset if c.eligible)),
            "label_source": sorted({c.label_source for c in subset}),
            "groups": len({c.group for c in subset})}
    statistics["processed"].update(processed_stats)
    statistics["emp"]["composition_relations"] = dict(Counter(t["composition_relation"] for t in tasks_emp))
    statistics["hsc"] = {**hsc_stats, "workflow_split_counts": dict(Counter(t["split"] for t in tasks_hsc)),
                         "workflow_structure_groups": len({t["group"] for t in tasks_hsc})}
    files = list(root.glob("*.jsonl")) + list((root / "processed").glob("*.jsonl")) + list((root / "HSC-main").glob("*.json"))
    files += [root / "processed/splits/service_splits.json", root / "EMP/data2.xlsx"]
    # Hash the complete raw API inventory without copying or disclosing responses.
    raw_files = sorted((root / "api").rglob("*.json"))
    raw_digests = {str(path.relative_to(root)): digest(path) for path in raw_files}
    manifest = {"source_root": str(root.resolve()), "files_sha256": {str(p.relative_to(root)): digest(p) for p in files},
        "raw_api_files": len(raw_files), "raw_api_inventory_sha256": sha256(json.dumps(raw_digests, sort_keys=True).encode()).hexdigest(),
        "adapter_source_sha256": digest(Path(__file__)), "split_seed": 42,
        "split_policy": "processed: official service split; five/EMP: normalized whole-query groups; HSC: full nested workflow groups",
        "limits": ["No live endpoints called", "No execution success invented from selection labels", "Fallback/zero-confidence labels excluded from primary five-domain selection evaluation"]}
    (output / "raw_api_hashes.json").write_text(json.dumps(raw_digests, indent=2))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "audit.json").write_text(json.dumps(statistics, indent=2, ensure_ascii=False))
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("results/user_data_v1"))
    args = parser.parse_args()
    print(json.dumps(adapt(args.root, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
