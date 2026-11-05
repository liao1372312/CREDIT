"""Frozen mechanism-blind ToolSandbox-M selector confirmation.

The tasks combine requested mutations in six graphs absent from the preceding
selector-attribution suite.  Every method uses the same v3 executor and wide
tool catalogue.  Selectors may inspect only the public request, public schemas,
and the frozen historical memory pool.
"""

from __future__ import annotations

import argparse
import datetime
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter_ns

import numpy as np
from scipy.stats import binomtest

from . import toolsandbox_local_v3 as adapter
from .remote_json_agent import RemoteJSONAgent
from . import toolsandbox_m_contract_fast as v1
from . import toolsandbox_m_contract_fast_v3 as v3
from . import toolsandbox_m_contract_fast_v3_confirmation as scenario_support
from . import toolsandbox_m_selector_attribution as attribution
from .toolsandbox_m_compose_confirmation import _imports
from .toolsandbox_m_contract_confirmation import novelty_audit
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_intent_optimization import balanced_order, select as legacy_select
from .toolsandbox_m_scoped_compose_selector import (
    scoped_compose_select, scoped_contract_select, scoped_full_select,
)
from .toolsandbox_m_selector_dependency_audit import closure
from .user_data import digest


CLOCK = scenario_support.CLOCK
ACTOR_SEEDS = (3203, 3217, 3221, 3229, 3251, 3253, 3257, 3259)
METHODS = (
    "NoMem/v3", "FIFO/v3", "Semantic/v3", "ScopedCompose/v3",
    "ContractOnly/v3", "FullScoped/v3",
)
PRIMARY_CONTRASTS = (
    ("ScopedCompose/v3", "NoMem/v3", "scoped history versus empty memory"),
    ("ScopedCompose/v3", "FIFO/v3", "scoped history versus FIFO history"),
    ("ScopedCompose/v3", "Semantic/v3", "scoped versus semantic history"),
    ("FullScoped/v3", "ContractOnly/v3", "history beyond request contract"),
    ("FullScoped/v3", "ScopedCompose/v3", "request contract beyond history"),
)
LATENCY_BUDGETS_MS = (10000, 15000, 20000, 30000)
PRIMARY_BUDGET_MS = 15000
MAX_MODEL_TURNS = 12
BOOTSTRAP_SEED = 20261024
BOOTSTRAP_REPLICATES = 20000
PLAN = Path("CREDIT_V4_MECHANISM_BLIND_PLAN.md")


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def build_scenarios(u):
    """Construct six new requested-mutation graphs with two prompts each."""
    x = _imports()
    pl, M, C = x["pl"], x["Milestone"], x["SnapshotConstraint"]
    DB, R, Cat = x["DatabaseNamespace"], x["RoleType"], x["ScenarioCategories"]
    add, snap = x["addition_similarity"], x["snapshot_similarity"]
    Extension = x["ScenarioExtension"]
    base = x["named_base_scenarios"](u.ToolBackend.DEFAULT)["base"]
    user_instruction = x["USER_INSTRUCTION"]
    wide_tools = list(scenario_support.WIDE_TOOLS)

    times = {
        "audit": datetime.datetime(2026, 10, 21, 9, 15).timestamp(),
        "route": datetime.datetime(2026, 10, 21, 11, 5).timestamp(),
        "filing": datetime.datetime(2026, 10, 21, 13, 25).timestamp(),
        "offline": datetime.datetime(2026, 10, 21, 15, 40).timestamp(),
    }
    specs = (
        ("contact_reminder", (
            "Create an auditor contact named 'Archive Node' at +14155550401 and add "
            "an 'Audit archive' reminder for October 21, 2026 at 09:15.",
            "Save +14155550401 under 'Archive Node' as an auditor contact; schedule "
            "a reminder titled 'Audit archive' at 9:15 AM on October 21, 2026."),
         {"contact": {"name": "Archive Node", "phone_number": "+14155550401",
                       "relationship": "auditor"},
          "reminder": {"content": "Audit archive", "timestamp": times["audit"]}}),
        ("contact_network_pair", (
            "Create a technician contact named 'Signal Desk' at +14155550402, then "
            "disable Wi-Fi and mobile data.",
            "Record +14155550402 under 'Signal Desk' as a technician contact; switch "
            "both cellular service and Wi-Fi off."),
         {"contact": {"name": "Signal Desk", "phone_number": "+14155550402",
                       "relationship": "technician"},
          "settings": {"wifi": False, "cellular": False}}),
        ("message_reminder_location", (
            "Text +14155550403 exactly 'Route confirmed', add a 'Check route' reminder "
            "for October 21, 2026 at 11:05, and disable location services.",
            "Send the literal message 'Route confirmed' to +14155550403; schedule a "
            "reminder titled 'Check route' at 11:05 AM on October 21, 2026; switch "
            "location service off."),
         {"message": {"phone_number": "+14155550403", "content": "Route confirmed"},
          "reminder": {"content": "Check route", "timestamp": times["route"]},
          "settings": {"location_service": False}}),
        ("contact_message_reminder_wifi", (
            "Create a coordinator contact named 'Release Clerk' at +14155550404, text "
            "it exactly 'Filed', add a 'Verify filing' reminder for October 21, 2026 "
            "at 13:25, and disable Wi-Fi.",
            "Save a coordinator contact named 'Release Clerk' at +14155550404; send that number "
            "the literal 'Filed', schedule a reminder titled 'Verify filing' at 1:25 PM "
            "on October 21, 2026, and switch Wi-Fi off."),
         {"contact": {"name": "Release Clerk", "phone_number": "+14155550404",
                       "relationship": "coordinator"},
          "message": {"phone_number": "+14155550404", "content": "Filed"},
          "reminder": {"content": "Verify filing", "timestamp": times["filing"]},
          "settings": {"wifi": False}}),
        ("reminder_network_triple", (
            "Add an 'Offline review' reminder for October 21, 2026 at 15:40, then "
            "disable Wi-Fi, mobile data, and location services.",
            "Schedule a reminder titled 'Offline review' at 3:40 PM on October 21, "
            "2026; switch Wi-Fi, cellular service, and location service off."),
         {"reminder": {"content": "Offline review", "timestamp": times["offline"]},
          "settings": {"wifi": False, "cellular": False,
                       "location_service": False}}),
        ("contact_message_battery", (
            "Create a custodian contact named 'Battery Steward' at +14155550406, text "
            "it exactly 'Power plan saved', and after sending enable low battery mode.",
            "Save +14155550406 under 'Battery Steward' as a custodian contact; send that "
            "number the literal 'Power plan saved'; then turn battery saver on."),
         {"contact": {"name": "Battery Steward", "phone_number": "+14155550406",
                       "relationship": "custodian"},
          "message": {"phone_number": "+14155550406", "content": "Power plan saved"},
          "settings": {"low_battery_mode": True}}),
    )

    def messages(private: str, public: str) -> list[dict]:
        return [
            {"sender": R.SYSTEM, "recipient": R.USER,
             "content": user_instruction + private +
                        " You do not have any more information."},
            {"sender": R.USER, "recipient": R.AGENT, "content": public},
        ]

    scenarios, metadata = {}, {}
    for family, prompts, expected in specs:
        milestones = []
        if "contact" in expected:
            milestones.append(M(snapshot_constraints=[C(
                database_namespace=DB.CONTACT, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({**expected["contact"],
                                               "is_self": False}))]))
        if "message" in expected:
            milestones.append(M(snapshot_constraints=[C(
                database_namespace=DB.MESSAGING, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({
                    "recipient_phone_number": expected["message"]["phone_number"],
                    "content": expected["message"]["content"]}))]))
        if "reminder" in expected:
            milestones.append(M(snapshot_constraints=[C(
                database_namespace=DB.REMINDER, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({
                    "content": expected["reminder"]["content"],
                    "reminder_timestamp": expected["reminder"]["timestamp"]}))]))
        if "settings" in expected:
            milestones.append(M(snapshot_constraints=[C(
                database_namespace=DB.SETTING, snapshot_constraint=snap,
                target_dataframe=pl.DataFrame(expected["settings"]))]))
        edges = [(index, index + 1) for index in range(len(milestones) - 1)]
        private = prompts[0]
        for index, public in enumerate(prompts):
            name = "mechanism_" + family + ("" if index == 0 else "_alt")
            scenario = Extension(
                name=name, base_scenario=base, messages=messages(private, public),
                tool_allow_list=wide_tools, milestones=milestones,
                milestone_edge_list=edges,
                categories=[Cat.SINGLE_USER_TURN, Cat.CANONICALIZATION],
            ).get_extended_scenario()[name]
            scenarios[name] = scenario
            required = sorted(v1.request_required_mutations(public, set(wide_tools)))
            metadata[name] = {"family": family, "public_request": public,
                "expected": expected, "required_mutations": required,
                "mechanism_signature": _sha(json.dumps(required))}
    trajectories = [{"trajectory": family, "sessions": sorted(
        name for name, item in metadata.items() if item["family"] == family)}
        for family, _, _ in specs]
    return scenarios, metadata, trajectories


def _latest_rows(context: dict, namespace: str) -> list[dict]:
    rows = context["_dbs"][namespace]
    indices = [row["sandbox_message_index"] for row in rows
               if row["sandbox_message_index"] is not None]
    latest = max(indices)
    return [row for row in rows if row["sandbox_message_index"] == latest
            and any(value is not None for key, value in row.items()
                    if key != "sandbox_message_index")]


def requested_state_checks(row: dict, spec: dict) -> list[float]:
    context, expected = row["ending_context"], spec["expected"]
    checks: list[float] = []
    if "contact" in expected:
        contacts = _latest_rows(context, "CONTACT")
        target = expected["contact"]
        checks.append(float(any(
            item.get("name") == target["name"] and
            item.get("phone_number") == target["phone_number"] and
            item.get("relationship") == target["relationship"] and
            item.get("is_self") is False for item in contacts)))
    if "message" in expected:
        messages = _latest_rows(context, "MESSAGING")
        target = expected["message"]
        checks.append(float(any(
            item.get("recipient_phone_number") == target["phone_number"] and
            item.get("content") == target["content"] for item in messages)))
    if "reminder" in expected:
        reminders = _latest_rows(context, "REMINDER")
        target = expected["reminder"]
        checks.append(float(any(
            item.get("content") == target["content"] and
            abs(float(item.get("reminder_timestamp", -1)) -
                float(target["timestamp"])) < 1e-6 for item in reminders)))
    if "settings" in expected:
        setting = _latest_rows(context, "SETTING")[0]
        checks.extend(float(setting.get(key) is value)
                      for key, value in expected["settings"].items())
    return checks


def annotate(row: dict, metadata: dict) -> dict:
    spec = metadata[row["task"]]
    checks = requested_state_checks(row, spec)
    successful = {call["tool"] for call in row["calls"]
                  if not call.get("exception")}
    required = set(spec["required_mutations"])
    row.update({"family": spec["family"],
        "requested_state_score": float(np.mean(checks)),
        "requested_state_success": int(all(value >= 1 - 1e-8 for value in checks)),
        "process_state_score": len(required & successful) / len(required),
        "process_state_success": int(required <= successful),
        "mechanism_signature": spec["mechanism_signature"],
        "exact_state_predicates_registered": True})
    return row


def golden_self_check(u, scenarios: dict, metadata: dict) -> dict:
    checks = []
    with tempfile.TemporaryDirectory(prefix="mechanism-golden-") as temporary:
        for name, scenario in scenarios.items():
            expected = metadata[name]["expected"]
            actions = []
            if "contact" in expected:
                actions.append({"tool": "add_contact", "arguments": expected["contact"]})
            if "message" in expected:
                actions.append({"tool": "send_message_with_phone_number",
                    "arguments": {"phone_number": expected["message"]["phone_number"],
                                  "content": expected["message"]["content"]}})
            if "reminder" in expected:
                actions.append({"tool": "add_reminder", "arguments": {
                    "content": expected["reminder"]["content"],
                    "reminder_timestamp": expected["reminder"]["timestamp"]}})
            setting_tools = {"wifi": "set_wifi_status",
                "cellular": "set_cellular_service_status",
                "location_service": "set_location_service_status",
                "low_battery_mode": "set_low_battery_mode_status"}
            for field, value in expected.get("settings", {}).items():
                actions.append({"tool": setting_tools[field],
                                "arguments": {"on": value}})
            job = initial_job_v2(name, scenario, [], u, None,
                                 "mechanism_blind_golden", "gold")
            for index, action in enumerate(actions):
                adapter.consume(job, {"agent_output": json.dumps(action),
                    "prompt_sha256": f"gold-{name}-{index}",
                    "hit_generation_limit": False}, u, 12)
            adapter.consume(job, {"agent_output": json.dumps({"answer": "Completed."}),
                "prompt_sha256": f"gold-{name}-final",
                "hit_generation_limit": False}, u, 12)
            row = scenario_support.prior.finish_direct(job, Path(temporary), u)
            row = annotate(row, metadata)
            checks.append({"task": name,
                "requested_state_success": row["requested_state_success"],
                "process_state_success": row["process_state_success"]})
    passed = all(item["requested_state_success"] ==
                 item["process_state_success"] == 1 for item in checks)
    if not passed:
        raise RuntimeError("golden evaluator failed: " + json.dumps(checks))
    return {"passed": True, "tasks": len(checks), "checks": checks}


def argument_extraction_self_check(metadata: dict) -> dict:
    checks = []
    for name, spec in metadata.items():
        expected = spec["expected"].get("contact")
        actual = v3.explicit_add_contact_arguments(spec["public_request"])
        if expected is not None:
            checks.append({"task": name, "expected": expected, "actual": actual,
                           "passed": actual == expected})
    if not all(item["passed"] for item in checks):
        raise RuntimeError("argument extraction failed: " + json.dumps(checks))
    return {"passed": True, "tasks": len(checks), "checks": checks}


def mechanism_novelty_audit(u, metadata: dict) -> dict:
    old_scenarios, old_metadata, _ = attribution.build_confirmation_scenarios(u)
    del old_scenarios
    old = {tuple(sorted(v1.request_required_mutations(
        item["public_request"], set(scenario_support.WIDE_TOOLS))))
        for item in old_metadata.values()}
    new_by_family = {}
    for item in metadata.values():
        signature = tuple(item["required_mutations"])
        previous = new_by_family.setdefault(item["family"], signature)
        if previous != signature:
            raise ValueError("prompt variants changed the mechanism signature")
    overlaps = sorted([{"family": family, "required_mutations": list(signature)}
                       for family, signature in new_by_family.items()
                       if signature in old], key=lambda row: row["family"])
    unique = len(set(new_by_family.values())) == len(new_by_family)
    passed = not overlaps and unique
    if not passed:
        raise RuntimeError("mechanism novelty failed: " + json.dumps(overlaps))
    return {"passed": True, "reference_suite":
        "ToolSandbox-M selector-by-executor attribution",
        "comparison": "exact requested-write tool set",
        "reference_signatures": [list(row) for row in sorted(old)],
        "new_signatures": {key: list(value)
                           for key, value in sorted(new_by_family.items())},
        "exact_signature_overlap": overlaps,
        "new_signatures_unique": unique}


def _supported(pool: list[dict], scenario) -> list[dict]:
    allowed = set(scenario.starting_context.tool_allow_list or [])
    return [note for note in pool if set(note["tools"]) & allowed]


def select(method: str, pool: list[dict], task: str, query: str, scenario,
           pair_vetoes: set[tuple[str, str]], tokenizer) -> list[dict]:
    if method == "NoMem/v3":
        return []
    if method == "FIFO/v3":
        return _supported(pool, scenario)[:2]
    if method == "Semantic/v3":
        return legacy_select("Semantic", pool, task, query, scenario, pair_vetoes)
    if method == "ScopedCompose/v3":
        return scoped_compose_select(pool, query, scenario)
    if method == "ContractOnly/v3":
        return scoped_contract_select(query, scenario, tokenizer)
    if method == "FullScoped/v3":
        return scoped_full_select(pool, query, scenario, tokenizer)
    raise ValueError("unknown method: " + method)


def run_batch(jobs: list[dict], query: str, agent, directory: Path, u,
              max_calls: int = 12) -> list[dict]:
    rows, active = [], jobs
    for _ in range(MAX_MODEL_TURNS):
        if not active:
            break
        records = agent.infer([job["chat"] for job in active])
        remaining = []
        for job, record in zip(active, records):
            began = perf_counter_ns()
            v3.consume_fast(job, record, u, max_calls, query)
            job["generations"][-1]["local_consume_latency_ms"] = (
                perf_counter_ns() - began) / 1e6
            if job["done"]:
                rows.append(scenario_support.prior.finish_direct(job, directory, u))
            else:
                remaining.append(job)
        active = remaining
    for job in active:
        job["termination"] = "turn_budget"
        rows.append(scenario_support.prior.finish_direct(job, directory, u))
    return rows


def _latency(row: dict) -> tuple[float, float, float]:
    remote = session_latency(row)
    local = float(row.get("selection_latency_ms", 0.0)) + sum(
        float(turn.get("local_consume_latency_ms", 0.0))
        for turn in row["generations"])
    return remote, local, remote + local


def run_actor(seed: int, actor_index: int, trajectories: list[dict], root: Path,
              args, scenarios, metadata, pool, pair_vetoes, u):
    directory = root / f"mechanism_blind_seed_{seed}"
    directory.mkdir()
    agent = RemoteJSONAgent(directory, seed, env=args.env,
        tokenizer_path=args.tokenizer, temperature=0.0, max_new_tokens=384,
        workers=args.workers, exact_prompt_cache=False)
    agent.manifest["benchmark"] = "ToolSandbox-M mechanism-blind selector confirmation"
    sessions, positions = [], defaultdict(lambda: [0] * len(METHODS))
    for trajectory_index, trajectory in enumerate(trajectories):
        for session_index, task in enumerate(trajectory["sessions"]):
            scenario, query = scenarios[task], metadata[task]["public_request"]
            order = balanced_order(list(METHODS), actor_index,
                                   trajectory_index, session_index)
            jobs, info = [], {}
            for position, method in enumerate(order):
                positions[method][position] += 1
                began = perf_counter_ns()
                bank = select(method, pool, task, query, scenario,
                              pair_vetoes, agent.tokenizer)
                selection_ms = (perf_counter_ns() - began) / 1e6
                job = initial_job_v2(task, scenario, bank, u, agent.tokenizer,
                                     "mechanism_blind_confirmation", method)
                job["selection_latency_ms"] = selection_ms
                jobs.append(job)
                info[method] = (bank, position)
            for row in run_batch(jobs, query, agent, directory, u):
                bank, position = info[row["method"]]
                remote, local, accounted = _latency(row)
                item = annotate({**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"],
                    "session_index": session_index,
                    "selected_bank": [note["uid"] for note in bank],
                    "selected_memory_schemas": [note.get("memory_schema") for note in bank],
                    "remote_latency_ms": remote, "local_runtime_ms": local,
                    "accounted_latency_ms": accounted, "batch_position": position,
                    "public_request_sha256": _sha(query)}, metadata)
                sessions.append(item)
                with (directory / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    agent.close()
    ledger = [json.loads(line) for line in
              (directory / "generations.jsonl").read_text().splitlines() if line]
    summary = {"actor_seed": seed, "sessions": len(sessions),
        "logical_remote_generations": len(ledger),
        "physical_http_attempts": sum(row.get("network_attempts", 1) for row in ledger),
        "method_batch_position_counts": dict(positions)}
    adapter.write(directory, "summary.json", summary)
    return sessions, summary


def summarize(rows: list[dict]) -> list[dict]:
    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        latency = np.asarray([row["accounted_latency_ms"] for row in subset])
        turns = [turn for row in subset for turn in row["generations"]]
        item = {"method": method, "sessions": len(subset),
            "requested_state_VSR": float(np.mean([
                row["requested_state_success"] for row in subset])),
            "process_state_VSR": float(np.mean([
                row["process_state_success"] for row in subset])),
            "mean_requested_state_score": float(np.mean([
                row["requested_state_score"] for row in subset])),
            "mean_selected_records": float(np.mean([
                len(row["selected_bank"]) for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "mean_remote_requests": float(np.mean([sum(
                "remote_latency_ms" in turn for turn in row["generations"])
                for row in subset])),
            "mean_accounted_latency_ms": float(np.mean(latency)),
            "p95_accounted_latency_ms": float(np.quantile(latency, .95)),
            "tool_call_exceptions": sum(bool(call.get("exception"))
                for row in subset for call in row["calls"]),
            "validator_rejections": sum("contract_violation" in turn for turn in turns),
            "deterministic_repairs": sum("deterministic_repair" in turn for turn in turns),
            "local_completions": sum(row["termination"] == "local_contract_complete"
                                     for row in subset)}
        for budget in LATENCY_BUDGETS_MS:
            item[f"requested_state_VSR_at_{budget}ms"] = float(np.mean([
                row["requested_state_success"] and value <= budget
                for row, value in zip(subset, latency)]))
        table.append(item)
    return table


def _sign(values: np.ndarray) -> dict:
    wins, losses = int(np.sum(values > 0)), int(np.sum(values < 0))
    p = float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue) \
        if wins + losses else 1.0
    return {"wins": wins, "losses": losses,
            "ties": int(len(values) - wins - losses), "one_sided_exact_p": p}


def _crossed_bootstrap(matrix: np.ndarray, rng) -> list[float]:
    actors, families = matrix.shape
    output = np.empty(BOOTSTRAP_REPLICATES)
    for index in range(BOOTSTRAP_REPLICATES):
        actor_draw = rng.integers(0, actors, actors)
        family_draw = rng.integers(0, families, families)
        output[index] = float(np.mean(matrix[np.ix_(actor_draw, family_draw)]))
    return np.quantile(output, [.025, .975]).tolist()


def paired_statistics(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for row in rows:
        by[row["method"]][(row["actor_seed"], row["trajectory"],
                           row["session_index"])] = row
    actors = sorted({row["actor_seed"] for row in rows})
    families = sorted({row["trajectory"] for row in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = []
    for treatment, baseline, estimand in PRIMARY_CONTRASTS:
        matrix = np.empty((len(actors), len(families)))
        budget_matrix = np.empty_like(matrix)
        for ai, actor in enumerate(actors):
            for fi, family in enumerate(families):
                keys = [(actor, family, session) for session in (0, 1)]
                matrix[ai, fi] = np.mean([
                    by[treatment][key]["requested_state_success"] -
                    by[baseline][key]["requested_state_success"] for key in keys])
                budget_matrix[ai, fi] = np.mean([
                    (by[treatment][key]["requested_state_success"] and
                     by[treatment][key]["accounted_latency_ms"] <= PRIMARY_BUDGET_MS) -
                    (by[baseline][key]["requested_state_success"] and
                     by[baseline][key]["accounted_latency_ms"] <= PRIMARY_BUDGET_MS)
                    for key in keys])
        actor_effects, family_effects = np.mean(matrix, axis=1), np.mean(matrix, axis=0)
        comparisons.append({"treatment": treatment, "baseline": baseline,
            "estimand": estimand, "delta_requested_state_VSR": float(np.mean(matrix)),
            "crossed_actor_family_bootstrap_CI95": _crossed_bootstrap(matrix, rng),
            "actor_level_sign_test": _sign(actor_effects),
            "family_level_sign_sensitivity": _sign(family_effects),
            "actor_effects": actor_effects.tolist(),
            "family_effects": dict(zip(families, family_effects.tolist())),
            "delta_state_VSR_at_15000ms": float(np.mean(budget_matrix)),
            "budget_crossed_bootstrap_CI95": _crossed_bootstrap(budget_matrix, rng)})
    order = sorted(range(len(comparisons)), key=lambda index:
                   comparisons[index]["actor_level_sign_test"]["one_sided_exact_p"])
    running = 0.0
    for rank, index in enumerate(order):
        raw = comparisons[index]["actor_level_sign_test"]["one_sided_exact_p"]
        running = max(running, min(1.0, (len(order) - rank) * raw))
        comparisons[index]["holm_actor_p"] = running
    return {"comparisons": comparisons, "actors": len(actors),
        "families": len(families), "prompts_per_actor_family": 2,
        "primary_resampling": "crossed actor-by-family bootstrap",
        "primary_test_unit": "actor-level paired mean",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED}


def task_breakdown(rows: list[dict]) -> list[dict]:
    output = []
    for family in sorted({row["trajectory"] for row in rows}):
        for method in METHODS:
            subset = [row for row in rows if row["trajectory"] == family
                      and row["method"] == method]
            output.append({"family": family, "method": method,
                "sessions": len(subset),
                "requested_state_VSR": float(np.mean([
                    row["requested_state_success"] for row in subset])),
                "requested_state_VSR_at_15000ms": float(np.mean([
                    row["requested_state_success"] and
                    row["accounted_latency_ms"] <= PRIMARY_BUDGET_MS
                    for row in subset]))})
    return output


def freeze(args, metadata: dict, trajectories: list[dict], pair_vetoes,
           preflight: dict, upstream_protocol: dict) -> dict:
    modules = closure({"toolsandbox_m_mechanism_blind",
                       "toolsandbox_m_mechanism_blind_verify"})
    source_dir = args.output / "source"
    source_dir.mkdir()
    source_sha = {}
    for module in sorted(modules):
        name = module + ".py"
        source = Path(__file__).with_name(name)
        shutil.copy2(source, source_dir / name)
        source_sha[name] = digest(source)
    protocol = {"suite": "ToolSandbox-M mechanism-blind selector confirmation",
        "frozen_before_remote_calls": True, "clock": CLOCK,
        "actor_seeds": list(args.seeds), "methods": list(METHODS),
        "primary_endpoint": "exact requested-world-state VSR",
        "secondary_latency_endpoint_ms": PRIMARY_BUDGET_MS,
        "primary_contrasts": [list(row) for row in PRIMARY_CONTRASTS],
        "multiplicity": "Holm over five registered actor-level sign tests",
        "uncertainty": "crossed actor-by-family bootstrap",
        "latency_definition": "remote request plus selection plus local consume; excludes concurrent batch wait",
        "latency_endpoints_ms": list(LATENCY_BUDGETS_MS),
        "maximum_model_turns_per_session": MAX_MODEL_TURNS,
        "maximum_tool_calls_per_session": 12, "exact_prompt_cache": False,
        "provider_credentials_persisted": False,
        "wide_identical_tool_catalogue": list(scenario_support.WIDE_TOOLS),
        "tasks": [{"name": name, "family": metadata[name]["family"],
                   "public_request_sha256": _sha(metadata[name]["public_request"]),
                   "required_mutations": metadata[name]["required_mutations"],
                   "mechanism_signature": metadata[name]["mechanism_signature"]}
                  for name in sorted(metadata)],
        "trajectories": trajectories, "pair_vetoes": [list(x) for x in pair_vetoes],
        "memory_pool_source": str(args.protocol_source),
        "memory_pool_source_sha256": digest(args.protocol_source / "memory_pool.json"),
        "novelty_sources": [str(path) for path in args.novelty_sources],
        "source_closure_complete": True, "source_sha256": source_sha,
        "plan_sha256": digest(PLAN),
        "upstream_directory": upstream_protocol["upstream_directory"],
        "upstream_source_sha256": upstream_protocol["upstream_source_sha256"],
        "preflight": preflight}
    adapter.write(args.output, "frozen_protocol.json", protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ACTOR_SEEDS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--protocol-source", type=Path,
        default=Path("results/toolsandbox_m_selector_attribution_confirmation_v1"))
    parser.add_argument("--novelty-sources", type=Path, nargs="+", default=[
        Path("results/toolsandbox_m_contract_fast_v3_confirmation_v1"),
        Path("results/toolsandbox_m_selector_attribution_confirmation_v1")])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    previous = json.loads((args.protocol_source / "frozen_protocol.json").read_text())
    u = adapter.upstream_imports(Path(previous["upstream_directory"]))
    adapter.install_local_clock(CLOCK)
    scenarios, metadata, trajectories = build_scenarios(u)
    pool = json.loads((args.protocol_source / "memory_pool.json").read_text())
    shutil.copy2(args.protocol_source / "memory_pool.json", args.output / "memory_pool.json")
    pair_vetoes = {tuple(x) for x in previous["pair_vetoes"]}
    preflight = {"golden_evaluator": golden_self_check(u, scenarios, metadata),
        "argument_extraction": argument_extraction_self_check(metadata),
        "exact_prompt_novelty": novelty_audit(
            u, scenarios, metadata, args.novelty_sources),
        "mechanism_novelty": mechanism_novelty_audit(u, metadata)}
    adapter.write(args.output, "preflight.json", preflight)
    freeze(args, metadata, trajectories, pair_vetoes, preflight, previous)
    first_manifest = next(args.protocol_source.glob("selector_attribution_seed_*/agent_manifest.json"))
    manifest = json.loads(first_manifest.read_text())
    args.tokenizer = args.tokenizer or Path(manifest["tokenizer_path"])
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    selection = {}
    for task, scenario in scenarios.items():
        selection[task] = {}
        for method in METHODS:
            bank = select(method, pool, task, metadata[task]["public_request"],
                          scenario, pair_vetoes, tokenizer)
            selection[task][method] = {
                "selected_uids": [note["uid"] for note in bank],
                "memory_schemas": [note.get("memory_schema") for note in bank],
                "tokens": sum(note["tokens"] for note in bank)}
    adapter.write(args.output, "frozen_selection.json", selection)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "tasks": len(metadata),
            "methods": len(METHODS), "source_modules": len(
                json.loads((args.output / "frozen_protocol.json").read_text())["source_sha256"]),
            "preflight": preflight}, indent=2))
        return

    rows, actors = [], []
    for index, seed in enumerate(args.seeds):
        found, actor = run_actor(seed, index, trajectories, args.output, args,
                                 scenarios, metadata, pool, pair_vetoes, u)
        rows.extend(found)
        actors.append(actor)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table, statistics = summarize(rows), paired_statistics(rows)
    ranking = sorted((dict(row) for row in table), key=lambda row: (
        -row["requested_state_VSR"],
        -row[f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms"],
        row["mean_accounted_latency_ms"], row["method"]))
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    breakdown = task_breakdown(rows)
    summary = {"confirmation_table": table, "ranking": ranking,
        "paired_statistics": statistics, "task_breakdown": breakdown,
        "actor_summaries": actors, "frozen_selection": selection}
    adapter.write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    with (args.output / "task_family_breakdown.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
        writer.writeheader(); writer.writerows(breakdown)
    adapter.write(args.output, "audit.json", {"sessions": len(rows),
        "expected_sessions": len(args.seeds) * len(metadata) * len(METHODS),
        "logical_remote_generations": sum(
            actor["logical_remote_generations"] for actor in actors),
        "physical_http_attempts": sum(actor["physical_http_attempts"] for actor in actors),
        "credentials_persisted": False, "all_tasks_share_identical_tools": True,
        "all_registered_sessions_present": len(rows) ==
            len(args.seeds) * len(metadata) * len(METHODS)})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
