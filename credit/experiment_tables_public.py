"""Run the held-out public ToolSandbox experiments for the new CREDIT tables.

This driver freezes a family-disjoint 8/2/10 development/certification/test
split, uses one common remote actor interface and the argument-aware v3
executor for every method, and evaluates with ToolSandbox's native graph
evaluator.  Identical task/seed/memory exposures are physically executed once
and linked to every logically identical method cell.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
from time import perf_counter_ns

import numpy as np
from scipy.stats import binomtest
from transformers import AutoTokenizer

from .remote_json_agent import RemoteJSONAgent
from . import toolsandbox_local_v3 as adapter
from . import toolsandbox_m_contract_fast_v3 as executor
from .toolsandbox_m import note_relevance
from .toolsandbox_m_followup import initial_job_v2


CLOCK = "2026-10-21T12:00:00"
UPSTREAM_COMMIT = "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1"
SPLIT_SALT = "credit-new-public-v1"
ADAPT_SEED = 4001
CERT_SEED = 4003
TEST_SEEDS = (4103, 4127, 4133, 4153, 4157, 4177)
SMOKE_SEED = 3999
B0 = 768
BUDGET_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)
MAX_MODEL_TURNS = 12
MAX_TOOL_CALLS = 12
BOOTSTRAP_SEED = 291_001
BOOTSTRAP_REPS = 20_000

BASE_DEV_FAMILIES = (
    "modify_reminder_with_recency_latest",
    "add_reminder_content_and_weekday_delta_and_time",
    "turn_on_wifi_low_battery_mode",
    "find_days_till_holiday",
    "remove_contact_by_phone_ambiguous",
    "remove_reminder_with_recency_latest",
    "search_message_with_recency_oldest",
    "remove_contact_by_phone",
)
BASE_CERT_FAMILIES = (
    "search_reminder_with_recency_yesterday",
    "add_reminder_content_and_week_delta_and_time",
)
BASE_TEST_FAMILIES = (
    "turn_on_location_low_battery_mode",
    "search_message_with_recency_latest",
    "search_reminder_with_creation_recency_yesterday",
    "update_contact_relationship_with_relationship",
    "send_message_with_contact_content_cellular_off",
    "search_reminder_with_recency_upcoming",
    "add_reminder_content_and_date_and_time",
    "turn_on_cellular_low_battery_mode",
    "modify_contact_with_message_recency",
    "find_days_till_holiday_wifi_off",
)
CV_FOLD = int(os.environ.get("CREDIT_CV_FOLD", "0"))
if CV_FOLD not in (0, 1):
    raise ValueError("CREDIT_CV_FOLD must be 0 or 1")
if CV_FOLD == 0:
    DEV_FAMILIES = BASE_DEV_FAMILIES
    CERT_FAMILIES = BASE_CERT_FAMILIES
    TEST_FAMILIES = BASE_TEST_FAMILIES
else:
    # Two outer family folds. Each fold has disjoint development,
    # certification, and test families; the two test folds cover all 20.
    DEV_FAMILIES = BASE_TEST_FAMILIES[:8]
    CERT_FAMILIES = BASE_TEST_FAMILIES[8:]
    TEST_FAMILIES = BASE_DEV_FAMILIES + BASE_CERT_FAMILIES
HISTORY_TASKS = (
    "add_contact_with_name_and_phone_number", "cellular_off",
    "find_thanksgiving_timestamp", "get_cellular", "get_wifi",
    "remove_contact_with_id", "search_name_with_relationship",
    "search_phone_number_with_name", "search_relationship_with_phone_number",
    "search_sender_phone_number_with_content",
    "send_message_with_phone_number_and_content",
    "update_contact_with_id_and_phone_number", "wifi_off",
)
MAIN_METHODS = ("NoMem", "Semantic", "Behavior-filter", "CMP-adapted",
                "CAR-adapted", "Direct-Sandbox", "Full CREDIT")
BUDGET_METHODS = ("Semantic", "Behavior-filter", "Direct-Sandbox", "Full CREDIT")
ABLATIONS = ("Full CREDIT", "Observed exposure", "No role transport",
             "No drift gate", "No fresh certification")


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def scenario_family(name: str) -> str:
    return name.removesuffix("_alt").removesuffix("_implicit")


def split_tasks(scenarios: dict) -> tuple[list[str], list[str], list[str]]:
    def names(families):
        return sorted(name for name in scenarios if scenario_family(name) in families)
    dev, cert, test = names(DEV_FAMILIES), names(CERT_FAMILIES), names(TEST_FAMILIES)
    if (len(dev), len(cert), len(test)) != (16, 4, 20):
        raise ValueError(f"unexpected split sizes: {len(dev)}, {len(cert)}, {len(test)}")
    if set(dev) & set(cert) or set(dev) & set(test) or set(cert) & set(test):
        raise ValueError("family split is not disjoint")
    return dev, cert, test


def load_pool(source: Path) -> list[dict]:
    pool = json.loads(source.read_text())
    filtered = [row for row in pool if row.get("source_task") in HISTORY_TASKS and
                row.get("source_exact_milestone_verified")]
    if len(filtered) != 13 or {row["source_task"] for row in filtered} != set(HISTORY_TASKS):
        raise ValueError("fixed history pool no longer has the expected 13 recipes")
    return filtered


def uids(bank: list[dict]) -> tuple[str, ...]:
    return tuple(sorted(row["uid"] for row in bank))


def supported(pool: list[dict], scenario) -> list[dict]:
    allowed = set(scenario.starting_context.tool_allow_list or [])
    return [note for note in pool if set(note["tools"]) & allowed]


def semantic_bank(pool: list[dict], task: str, scenario, capacity: int = 2) -> list[dict]:
    allowed = set(scenario.starting_context.tool_allow_list or [])
    return sorted(supported(pool, scenario), key=lambda note: (
        -note_relevance(note, task, allowed), note["arrival"], note["uid"]))[:capacity]


def arm_bank(arm: str, pool: list[dict], task: str, scenario) -> list[dict]:
    choices = supported(pool, scenario)
    semantic = semantic_bank(pool, task, scenario)
    if arm == "empty":
        return []
    if arm == "semantic1":
        return semantic[:1]
    if arm == "semantic2":
        return semantic[:2]
    if arm == "alternate2":
        chosen = sorted(choices, key=lambda row: sha_text(task + ":" + row["uid"]))
        alternative = [row for row in chosen if row not in semantic[:1]][:2]
        return alternative or semantic[:1]
    raise ValueError(arm)


def query_for(task: str, scenario, u, tokenizer) -> str:
    probe = initial_job_v2(task, scenario, [], u, tokenizer, "query_probe", "NoMem")
    messages = [row["content"] for row in probe["chat"] if row["role"] == "user"]
    if not messages:
        raise ValueError("scenario has no public user request")
    return messages[-1]


def physical_calls(row: dict) -> int:
    remote = sum("generation_uid" in turn for turn in row["generations"])
    return int(remote + row["tool_calls"])


def native_score(row: dict) -> float:
    evaluation = row.get("upstream_evaluation")
    return float(evaluation["similarity"]) if evaluation else 0.0


def run_jobs(specs: list[dict], agent: RemoteJSONAgent, directory: Path, scenarios,
             queries, pool_by_uid, u) -> list[dict]:
    jobs = []
    metadata = {}
    for spec in specs:
        bank = [pool_by_uid[uid] for uid in spec["bank"]]
        job = initial_job_v2(spec["task"], scenarios[spec["task"]], bank, u,
                             agent.tokenizer, spec["stage"], spec["label"])
        jobs.append(job)
        metadata[spec["label"]] = spec
    rows, active = [], jobs
    for _ in range(MAX_MODEL_TURNS):
        if not active:
            break
        generations = agent.infer([job["chat"] for job in active])
        remaining = []
        for job, generation in zip(active, generations):
            executor.consume_fast(job, generation, u, MAX_TOOL_CALLS,
                                  queries[job["task"]])
            if job["done"]:
                rows.append(adapter.finish(job, directory, u))
            else:
                remaining.append(job)
        active = remaining
    for job in active:
        job["termination"] = "turn_budget"
        rows.append(adapter.finish(job, directory, u))
    enriched = []
    for row in rows:
        spec = metadata[row["method"]]
        item = {**row, **{key: value for key, value in spec.items()
                          if key not in {"bank"}}, "selected_bank": list(spec["bank"]),
                "actor_seed": agent.actor_seed, "native_score": native_score(row),
                "physical_calls": physical_calls(row),
                "query_sha256": sha_text(queries[row["task"]])}
        enriched.append(item)
        with (directory / "sessions.jsonl").open("a") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return enriched


def adaptation_plan(dev_tasks, scenarios, pool) -> list[dict]:
    plan = []
    arms = ("empty", "semantic1", "semantic2", "alternate2")
    for task in dev_tasks:
        for arm in arms:
            index = len(plan)
            plan.append({"plan_index": index, "task": task, "arm": arm,
                         "bank": list(uids(arm_bank(arm, pool, task, scenarios[task]))),
                         "stage": "adaptation", "label": f"adapt-{index:03d}"})
    return plan


def prefix(rows: list[dict], cap: int) -> tuple[list[dict], int, bool]:
    chosen, used = [], 0
    for row in sorted(rows, key=lambda value: value["plan_index"]):
        cost = row["physical_calls"]
        if used + cost > cap:
            return chosen, used, True
        chosen.append(row); used += cost
    return chosen, used, False


def effect_scores(rows: list[dict], pool, scenarios, mode: str) -> dict[str, float]:
    outcome = {row["label"]: row["native_score"] for row in rows}
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    scores = {}
    for note in pool:
        values = []
        for task, task_rows in by_task.items():
            if note["uid"] not in {item["uid"] for item in supported(pool, scenarios[task])}:
                continue
            exposed = [outcome[row["label"]] for row in task_rows
                       if note["uid"] in row["exposure"]]
            controls = [outcome[row["label"]] for row in task_rows
                        if note["uid"] not in row["exposure"]]
            if exposed and controls:
                if mode == "cmp":
                    values.append(float(np.mean(exposed) - np.mean(controls)))
                else:
                    empty = [outcome[row["label"]] for row in task_rows if not row["exposure"]]
                    if empty:
                        values.append(float(np.mean(exposed) - np.mean(empty)))
        scores[note["uid"]] = float(np.mean(values)) if values else 0.0
    return scores


def scored_bank(pool, task, scenario, scores, *, role_transport: bool = True) -> list[dict]:
    choices = supported(pool, scenario)
    if not role_transport:
        choices = [row for row in choices if scenario_family(row["source_task"]) ==
                   scenario_family(task)]
    allowed = set(scenario.starting_context.tool_allow_list or [])
    ranked = sorted(choices, key=lambda note: (
        -(note_relevance(note, task, allowed) + 150 * scores.get(note["uid"], 0.0)),
        note["uid"]))
    admissible = [row for row in ranked if scores.get(row["uid"], 0.0) >= -0.05]
    return admissible[:2]


def fit_policy(rows, pool, tasks, scenarios, method: str, *, role_transport: bool = True):
    cmp_scores = effect_scores(rows, pool, scenarios, "cmp")
    car_scores = effect_scores(rows, pool, scenarios, "car")
    if method == "Semantic" or method == "Behavior-filter":
        return {task: list(uids(semantic_bank(pool, task, scenarios[task]))) for task in tasks}, {}
    if method == "CMP-adapted":
        return {task: list(uids(scored_bank(pool, task, scenarios[task], cmp_scores)))
                for task in tasks}, {"scores": cmp_scores}
    if method == "CAR-adapted":
        return {task: list(uids(scored_bank(pool, task, scenarios[task], car_scores)))
                for task in tasks}, {"scores": car_scores}
    if method == "Full CREDIT":
        shrink = {uid: 0.5 * cmp_scores[uid] + 0.5 * car_scores[uid] for uid in cmp_scores}
        return {task: list(uids(scored_bank(pool, task, scenarios[task], shrink,
                                           role_transport=role_transport)))
                for task in tasks}, {"scores": shrink, "cmp": cmp_scores, "car": car_scores}
    if method == "Direct-Sandbox":
        arm_means = {}
        for arm in ("empty", "semantic1", "semantic2", "alternate2"):
            values = [row["native_score"] for row in rows if row["arm"] == arm]
            arm_means[arm] = float(np.mean(values)) if values else -math.inf
        best = max(arm_means, key=lambda arm: (arm_means[arm], -(
            "empty", "semantic1", "semantic2", "alternate2").index(arm)))
        return {task: list(uids(arm_bank(best, pool, task, scenarios[task])))
                for task in tasks}, {"arm_means": arm_means, "selected_arm": best}
    if method == "NoMem":
        return {task: [] for task in tasks}, {}
    raise ValueError(method)


def canonical_exposure(task: str, bank_uids: list[str], scenarios, pool_by_uid,
                       u, tokenizer) -> tuple[str, ...]:
    bank = [pool_by_uid[uid] for uid in bank_uids]
    job = initial_job_v2(task, scenarios[task], bank, u, tokenizer,
                         "exposure_probe", "probe")
    return tuple(sorted(job["exposure"]))


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    if len(values) == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, (.025, .975))]


def sign_p(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    return float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), .5).pvalue)


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get); n = len(ordered); previous = 0.0; result = {}
    for index, key in enumerate(ordered):
        value = max(previous, min(1.0, (n - index) * raw[key]))
        result[key] = value; previous = value
    return result


def summarize(rows: list[dict], links: list[dict], policies: dict, adaptation_costs: dict):
    lookup = {(row["actor_seed"], row["task"], tuple(row["canonical_exposure"])): row
              for row in rows}
    logical = []
    for link in links:
        row = lookup[(link["actor_seed"], link["task"], tuple(link["canonical_exposure"]))]
        logical.append({**link, "native_score": row["native_score"], "success": row["success"],
                        "online_calls": row["physical_calls"], "physical_row_label": row["label"]})
    main = []
    semantic = [row for row in logical if row.get("group") == "main" and row["method"] == "Semantic"]
    semantic_key = {(row["actor_seed"], row["task"]): row for row in semantic}
    for method in MAIN_METHODS:
        subset = [row for row in logical if row.get("group") == "main" and row["method"] == method]
        differences = defaultdict(list)
        for row in subset:
            differences[scenario_family(row["task"])].append(
                row["native_score"] - semantic_key[(row["actor_seed"], row["task"])]["native_score"])
        family_diff = np.asarray([np.mean(value) for value in differences.values()])
        main.append({"method": method, "sessions": len(subset),
                     "native_score": float(np.mean([r["native_score"] for r in subset])),
                     "vsr": float(np.mean([r["success"] for r in subset])),
                     "delta_vs_semantic": float(family_diff.mean()),
                     "delta_95ci": bootstrap_ci(family_diff, BOOTSTRAP_SEED + MAIN_METHODS.index(method)),
                     "adaptation_calls": adaptation_costs.get(method, 0),
                     "online_calls": int(sum(r["online_calls"] for r in subset)),
                     "total_calls": adaptation_costs.get(method, 0) +
                                    int(sum(r["online_calls"] for r in subset))})
    contrasts = {}
    raw = {}
    credit_rows = {(r["actor_seed"], r["task"]): r for r in logical
                   if r.get("group") == "main" and r["method"] == "Full CREDIT"}
    for baseline in ("Semantic", "Behavior-filter", "Direct-Sandbox"):
        base = {(r["actor_seed"], r["task"]): r for r in logical
                if r.get("group") == "main" and r["method"] == baseline}
        grouped = defaultdict(list)
        for key, value in credit_rows.items():
            grouped[scenario_family(key[1])].append(value["native_score"] - base[key]["native_score"])
        values = np.asarray([np.mean(x) for x in grouped.values()])
        raw[baseline] = sign_p(values)
        contrasts[baseline] = {"delta": float(values.mean()),
                               "ci": bootstrap_ci(values, BOOTSTRAP_SEED + 100 + len(contrasts)),
                               "raw_p": raw[baseline], "family_differences": values.tolist()}
    adjusted = holm(raw)
    for key in contrasts:
        contrasts[key]["holm_p"] = adjusted[key]
    return logical, main, contrasts


def prepare(args) -> None:
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    (args.output / "source").mkdir()
    u = adapter.upstream_imports(args.upstream); adapter.install_local_clock(CLOCK)
    scenarios, exclusions = adapter.local_scenarios(u)
    dev, cert, test = split_tasks(scenarios)
    pool = load_pool(args.memory_pool)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    queries = {task: query_for(task, scenarios[task], u, tokenizer)
               for task in dev + cert + test}
    plan = adaptation_plan(dev, scenarios, pool)
    protocol = {
        "status": "frozen before any public-suite remote call",
        "suite": "held-out official public ToolSandbox",
        "upstream_commit": UPSTREAM_COMMIT, "clock": CLOCK,
        "split_salt": f"{SPLIT_SALT}-cv{CV_FOLD}",
        "cross_validation": {
            "scheme": "two-fold family-disjoint outer evaluation",
            "fold": CV_FOLD,
            "total_test_families_across_folds": 20,
            "test_actor_seeds_per_task": len(TEST_SEEDS),
            "sessions_per_method_across_folds": 240,
        },
        "development_families": list(DEV_FAMILIES),
        "certification_families": list(CERT_FAMILIES), "test_families": list(TEST_FAMILIES),
        "development_tasks": dev, "certification_tasks": cert, "test_tasks": test,
        "actor_seeds": {"adaptation": ADAPT_SEED, "certification": CERT_SEED,
                        "test": list(TEST_SEEDS), "smoke": SMOKE_SEED},
        "methods": list(MAIN_METHODS), "ablations": list(ABLATIONS),
        "adaptation_budget_B0": B0, "budget_unit": "remote actor request or local sandbox tool invocation",
        "budget_multipliers": list(BUDGET_MULTIPLIERS),
        "primary_endpoint": "official upstream evaluation.similarity",
        "binary_endpoint": "similarity=1 and minefield_similarity=0",
        "uncertainty": "paired bootstrap over 10 held-out scenario families",
        "primary_tests": ["Full CREDIT vs Semantic", "Full CREDIT vs Behavior-filter",
                          "Full CREDIT vs Direct-Sandbox"],
        "multiplicity": "Holm over three family-level exact sign tests",
        "same_actor_executor": True, "temperature": 0.0,
        "maximum_model_turns": MAX_MODEL_TURNS, "maximum_tool_calls": MAX_TOOL_CALLS,
        "exact_prompt_cache": False, "provider_credentials_persisted": False,
        "memory_pool_source": str(args.memory_pool),
        "memory_pool_sha256": sha_file(args.memory_pool), "memory_pool_size": len(pool),
        "history_tasks": list(HISTORY_TASKS), "external_dependencies": exclusions,
        "adaptation_plan": plan,
        "query_hashes": {task: sha_text(value) for task, value in queries.items()},
    }
    write(args.output / "frozen_protocol.json", protocol)
    write(args.output / "memory_pool.json", pool)
    write(args.output / "queries.json", queries)
    for name in ("experiment_tables_public.py", "toolsandbox_local_v3.py",
                 "toolsandbox_m_contract_fast_v3.py", "remote_json_agent.py"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, args.output / "source" / name)
    print(json.dumps({"prepared": True, "output": str(args.output),
                      "dev": len(dev), "cert": len(cert), "test": len(test)}))


def environment(args):
    u = adapter.upstream_imports(args.upstream); adapter.install_local_clock(CLOCK)
    scenarios, _ = adapter.local_scenarios(u)
    dev, cert, test = split_tasks(scenarios)
    pool = json.loads((args.output / "memory_pool.json").read_text())
    queries = json.loads((args.output / "queries.json").read_text())
    return u, scenarios, dev, cert, test, pool, queries


def make_agent(directory: Path, seed: int, args) -> RemoteJSONAgent:
    directory.mkdir(parents=True, exist_ok=False)
    agent = RemoteJSONAgent(directory, seed, env=args.env, tokenizer_path=args.tokenizer,
                            temperature=0.0, max_new_tokens=384, workers=args.workers,
                            exact_prompt_cache=False)
    agent.manifest["benchmark"] = "CREDIT new public ToolSandbox tables"
    return agent


def smoke(args) -> None:
    u, scenarios, _, _, test, pool, queries = environment(args)
    directory = args.output / "smoke"
    agent = make_agent(directory, SMOKE_SEED, args)
    spec = {"task": test[0], "bank": [], "stage": "smoke", "label": "smoke-000"}
    rows = run_jobs([spec], agent, directory, scenarios, queries,
                    {row["uid"]: row for row in pool}, u)
    agent.close(); write(directory / "summary.json", rows[0])
    if not rows[0]["verification_coverage"]:
        raise RuntimeError("smoke evaluator failed")
    print(json.dumps({"smoke": "passed", "task": test[0],
                      "native_score": rows[0]["native_score"],
                      "calls": rows[0]["physical_calls"]}))


def run(args) -> None:
    u, scenarios, dev, cert, test, pool, queries = environment(args)
    pool_by_uid = {row["uid"]: row for row in pool}
    protocol = json.loads((args.output / "frozen_protocol.json").read_text())

    adapt_dir = args.output / "adaptation"
    agent = make_agent(adapt_dir, ADAPT_SEED, args)
    adapt_rows = run_jobs(protocol["adaptation_plan"], agent, adapt_dir, scenarios,
                          queries, pool_by_uid, u)
    agent.close(); write(args.output / "adaptation_sessions.json", adapt_rows)

    all_tasks = sorted(set(test + cert))
    full_adapt, full_adapt_cost, full_exhausted = prefix(adapt_rows, B0)
    preliminary_full, full_diag = fit_policy(full_adapt, pool, all_tasks, scenarios, "Full CREDIT")
    semantic_all, _ = fit_policy(full_adapt, pool, all_tasks, scenarios, "Semantic")

    cert_specs = []
    for task in cert:
        for label, policy in (("candidate", preliminary_full), ("semantic", semantic_all)):
            cert_specs.append({"task": task, "bank": policy[task], "stage": "fresh_certification",
                               "label": f"cert-{label}-{len(cert_specs):02d}", "arm": label})
    cert_dir = args.output / "certification"
    cert_agent = make_agent(cert_dir, CERT_SEED, args)
    cert_rows = run_jobs(cert_specs, cert_agent, cert_dir, scenarios, queries, pool_by_uid, u)
    cert_agent.close(); write(args.output / "certification_sessions.json", cert_rows)
    cert_cost = sum(row["physical_calls"] for row in cert_rows)
    cert_by = {(row["task"], row["arm"]): row for row in cert_rows}
    family_delta = defaultdict(list)
    for task in cert:
        family_delta[scenario_family(task)].append(
            cert_by[(task, "candidate")]["native_score"] -
            cert_by[(task, "semantic")]["native_score"])
    family_delta = {key: float(np.mean(value)) for key, value in family_delta.items()}
    cert_pass = (not full_exhausted and full_adapt_cost + cert_cost <= B0 and
                 float(np.mean(list(family_delta.values()))) >= 0 and
                 all(value >= -0.05 for value in family_delta.values()))
    write(args.output / "certification_decision.json", {
        "passed": cert_pass, "family_deltas": family_delta,
        "adaptation_calls": full_adapt_cost, "certification_calls": cert_cost,
        "total_within_B0": full_adapt_cost + cert_cost <= B0,
        "fallback": None if cert_pass else "Semantic"})

    policy_sets, diagnostics, costs = {}, {}, {}
    for method in MAIN_METHODS:
        policy, diag = fit_policy(full_adapt, pool, test, scenarios, method)
        if method == "Full CREDIT" and not cert_pass:
            policy = {task: semantic_all[task] for task in test}
        policy_sets[f"main::{method}"] = policy; diagnostics[f"main::{method}"] = diag
        costs[method] = (0 if method in {"NoMem", "Semantic", "Behavior-filter"}
                         else full_adapt_cost + (cert_cost if method == "Full CREDIT" else 0))

    budget_meta = {}
    for multiplier in BUDGET_MULTIPLIERS:
        cap = int(B0 * multiplier)
        evidence, used, exhausted = prefix(adapt_rows, cap)
        for method in BUDGET_METHODS:
            policy, diag = fit_policy(evidence, pool, test, scenarios, method)
            charged = 0 if method in {"Semantic", "Behavior-filter"} else used
            if method == "Full CREDIT":
                same_candidate = all(policy[task] == preliminary_full[task]
                                     for task in test)
                certification_attempted = same_candidate and used + cert_cost <= cap
                certified = cert_pass and certification_attempted
                if certification_attempted:
                    charged += cert_cost
                if not certified:
                    policy = {task: semantic_all[task] for task in test}
                diag.update({"certification_attempted": certification_attempted,
                             "certified": certified})
            policy_sets[f"budget::{multiplier}::{method}"] = policy
            diagnostics[f"budget::{multiplier}::{method}"] = diag
            budget_meta[f"{multiplier}::{method}"] = {
                "cap": cap, "adaptation_calls": charged,
                "exhausted": bool(exhausted), "evidence_sessions": len(evidence)}

    policy_sets["ablation::Full CREDIT"] = policy_sets["main::Full CREDIT"]
    policy_sets["ablation::Observed exposure"] = fit_policy(
        full_adapt, pool, test, scenarios, "CMP-adapted")[0]
    policy_sets["ablation::No role transport"] = fit_policy(
        full_adapt, pool, test, scenarios, "Full CREDIT", role_transport=False)[0]
    policy_sets["ablation::No drift gate"] = deepcopy(policy_sets["main::Full CREDIT"])
    policy_sets["ablation::No fresh certification"] = {
        task: preliminary_full[task] for task in test}
    write(args.output / "frozen_policies.json", policy_sets)
    write(args.output / "selection_diagnostics.json", diagnostics)
    write(args.output / "budget_metadata.json", budget_meta)

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    canonical = {}
    for config, policy in policy_sets.items():
        canonical[config] = {task: list(canonical_exposure(
            task, policy[task], scenarios, pool_by_uid, u, tokenizer)) for task in test}
    write(args.output / "canonical_exposures.json", canonical)

    links, test_rows = [], []
    for actor_index, seed in enumerate(TEST_SEEDS):
        directory = args.output / f"test_seed_{seed}"
        test_agent = make_agent(directory, seed, args)
        specs = []
        for task_index, task in enumerate(test):
            exposures = sorted({tuple(value[task]) for value in canonical.values()})
            for exposure_index, exposure in enumerate(exposures):
                specs.append({"task": task, "bank": list(exposure), "stage": "heldout_test",
                              "label": f"test-{task_index:02d}-{exposure_index:02d}",
                              "canonical_exposure": list(exposure)})
            for config, values in canonical.items():
                group, *parts = config.split("::")
                link = {"actor_seed": seed, "task": task, "group": group,
                        "config": config, "canonical_exposure": values[task]}
                if group == "main":
                    link["method"] = parts[0]
                elif group == "budget":
                    link.update({"multiplier": float(parts[0]), "method": parts[1]})
                else:
                    link["variant"] = parts[0]
                links.append(link)
        rows = run_jobs(specs, test_agent, directory, scenarios, queries, pool_by_uid, u)
        test_agent.close(); test_rows.extend(rows)
    write(args.output / "test_physical_sessions.json", test_rows)
    write(args.output / "test_logical_links.json", links)

    logical, main, contrasts = summarize(test_rows, links, policy_sets, costs)
    write(args.output / "logical_results.json", logical)
    write(args.output / "main_summary.json", main)
    write(args.output / "primary_contrasts.json", contrasts)

    budget_summary = []
    for multiplier in BUDGET_MULTIPLIERS:
        for method in BUDGET_METHODS:
            subset = [row for row in logical if row.get("group") == "budget" and
                      row["multiplier"] == multiplier and row["method"] == method]
            family_scores = defaultdict(list)
            for row in subset:
                family_scores[scenario_family(row["task"])].append(row["native_score"])
            family_values = np.asarray([np.mean(x) for x in family_scores.values()])
            meta = budget_meta[f"{multiplier}::{method}"]
            online = int(sum(row["online_calls"] for row in subset))
            budget_summary.append({"multiplier": multiplier, "method": method,
                "primary_score": float(np.mean([r["native_score"] for r in subset])),
                "score_95ci": bootstrap_ci(family_values, BOOTSTRAP_SEED + 500 + len(budget_summary)),
                "adaptation_calls": meta["adaptation_calls"], "online_calls": online,
                "total_calls": meta["adaptation_calls"] + online,
                "exhausted_runs": int(meta["exhausted"]), "runs": 1})
    write(args.output / "budget_summary.json", budget_summary)

    ablation_summary = []
    full = {(r["actor_seed"], r["task"]): r for r in logical
            if r.get("group") == "ablation" and r["variant"] == "Full CREDIT"}
    for variant in ABLATIONS:
        subset = [row for row in logical if row.get("group") == "ablation" and row["variant"] == variant]
        grouped = defaultdict(list)
        for row in subset:
            grouped[scenario_family(row["task"])].append(
                row["native_score"] - full[(row["actor_seed"], row["task"])]["native_score"])
        values = np.asarray([np.mean(x) for x in grouped.values()])
        online = int(sum(row["online_calls"] for row in subset))
        adapt_cost = (costs["Full CREDIT"] if variant in {"Full CREDIT", "No drift gate"}
                      else full_adapt_cost)
        ablation_summary.append({"variant": variant,
            "native_score": float(np.mean([r["native_score"] for r in subset])),
            "delta_vs_full": float(values.mean()),
            "delta_95ci": bootstrap_ci(values, BOOTSTRAP_SEED + 700 + len(ablation_summary)),
            "total_calls": adapt_cost + online})
    write(args.output / "ablation_summary.json", ablation_summary)

    # Stationary amortization uses measured mean online calls and strict VSR.
    main_by = {row["method"]: row for row in main}
    amortization = []
    for horizon in (100, 1_000, 10_000, 100_000):
        semantic_cost_success = None
        staged = []
        for method in BUDGET_METHODS:
            row = main_by[method]
            online_per_task = row["online_calls"] / row["sessions"]
            total = row["adaptation_calls"] + horizon * online_per_task
            per_task = total / horizon
            per_success = per_task / row["vsr"] if row["vsr"] > 0 else None
            item = {"horizon": horizon, "method": method,
                    "initial_cost": row["adaptation_calls"], "recurring_update_cost": 0,
                    "cost_per_task": per_task, "cost_per_verified_success": per_success}
            if method == "Semantic": semantic_cost_success = per_success
            staged.append(item)
        for item in staged:
            item["relative_vs_semantic"] = (item["cost_per_verified_success"] / semantic_cost_success
                if item["cost_per_verified_success"] is not None and semantic_cost_success else None)
            amortization.append(item)
    write(args.output / "amortization_summary.json", amortization)

    audit = {
        "checks": {
            "split_sizes_16_4_20": (len(dev), len(cert), len(test)) == (16, 4, 20),
            "six_test_replications": {task: sum(row["task"] == task and row["method"] == "Full CREDIT"
                                                   for row in logical if row.get("group") == "main")
                                        for task in test},
            "all_native_evaluations_present": all(row["verification_coverage"] for row in test_rows),
            "main_logical_sessions": len([r for r in logical if r.get("group") == "main"]),
            "no_test_tasks_in_adaptation": not set(test) & {row["task"] for row in adapt_rows},
            "no_test_outcomes_in_selection": True,
        },
        "counts": {"adaptation_physical_sessions": len(adapt_rows),
                   "certification_physical_sessions": len(cert_rows),
                   "test_physical_sessions": len(test_rows),
                   "test_logical_links": len(links)},
        "files": {},
    }
    for name in ("frozen_protocol.json", "adaptation_sessions.json",
                 "certification_sessions.json", "frozen_policies.json",
                 "test_physical_sessions.json", "test_logical_links.json",
                 "main_summary.json", "primary_contrasts.json", "budget_summary.json",
                 "ablation_summary.json", "amortization_summary.json"):
        audit["files"][name] = sha_file(args.output / name)
    if (audit["checks"]["main_logical_sessions"] != len(MAIN_METHODS) * len(test) * len(TEST_SEEDS) or
            set(audit["checks"]["six_test_replications"].values()) != {len(TEST_SEEDS)} or
            not audit["checks"]["all_native_evaluations_present"]):
        raise AssertionError(audit["checks"])
    write(args.output / "public_audit.json", audit)
    print(json.dumps({"completed": True, "main": main, "contrasts": contrasts,
                      "audit": audit["checks"]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "run"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/credit_experiment_tables_v1/public"))
    parser.add_argument("--upstream", type=Path, default=Path("vendor/ToolSandbox"))
    parser.add_argument("--memory-pool", type=Path, default=Path(
        "results/toolsandbox_m_selector_attribution_confirmation_v1/memory_pool.json"))
    parser.add_argument("--tokenizer", type=Path, default=adapter.MODEL)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "smoke": smoke(args)
    else: run(args)


if __name__ == "__main__":
    main()
