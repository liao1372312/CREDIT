"""Held-out confirmation for literal-safe adaptive contract execution."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
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
from . import toolsandbox_m_contract_fast_v2 as v2
from . import toolsandbox_m_contract_fast_confirmation as prior
from .toolsandbox_m_compose_confirmation import _imports
from .toolsandbox_m_compose_selector import compose_select
from .toolsandbox_m_contract_confirmation import novelty_audit
from .toolsandbox_m_followup import initial_job_v2, session_latency
from .toolsandbox_m_intent_optimization import balanced_order, select as prior_select
from .user_data import digest


CLOCK = prior.CLOCK
ACTOR_SEEDS = (2603, 2611, 2619, 2627, 2633)
METHODS = ("Contract-Adaptive-v1", "Contract-Adaptive-v2", "Contract-Text",
           "Semantic", "CREDIT-Compose")
LATENCY_BUDGETS_MS = (10000, 15000, 20000, 30000)
PRIMARY_BUDGET_MS = 15000
MAX_MODEL_TURNS = 12
BOOTSTRAP_SEED = 20261021
WIDE_TOOLS = prior.WIDE_TOOLS + ("calculate_lat_lon_distance",)


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _retarget(scenario, public: str, u):
    """Clone a scenario, replace its public turn, and expose the v2 catalogue."""
    clone = deepcopy(scenario)
    frame = clone.starting_context.get_database(u.DatabaseNamespace.SANDBOX)
    frame = frame.with_columns(
        __import__("polars").lit(public).alias("content"))
    clone.starting_context.update_database(u.DatabaseNamespace.SANDBOX, frame)
    clone.starting_context.tool_allow_list = list(WIDE_TOOLS)
    return clone


def build_confirmation_scenarios(u):
    """Build three literal-decoy and three real-risk families."""
    x = _imports()
    pl, M, C = x["pl"], x["Milestone"], x["SnapshotConstraint"]
    DB, R, Cat = x["DatabaseNamespace"], x["RoleType"], x["ScenarioCategories"]
    add, snap = x["addition_similarity"], x["snapshot_similarity"]
    Extension = x["ScenarioExtension"]
    bases = x["named_base_scenarios"](u.ToolBackend.DEFAULT)
    user_instruction = x["USER_INSTRUCTION"]
    old_scenarios, old_metadata, _ = prior.build_confirmation_scenarios(u)

    scenarios, metadata = {}, {}

    def messages(private, public):
        return [
            {"sender": R.SYSTEM, "recipient": R.USER,
             "content": user_instruction + private +
                        " You do not have any more information."},
            {"sender": R.USER, "recipient": R.AGENT, "content": public},
        ]

    def literal_extension(name, family, public, private, expected):
        field = expected["setting_field"]
        milestones = [
            M(snapshot_constraints=[C(
                database_namespace=DB.CONTACT, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({
                    "name": expected["name"], "phone_number": expected["phone"],
                    "relationship": expected["relationship"], "is_self": False}))]),
            M(snapshot_constraints=[C(
                database_namespace=DB.MESSAGING, snapshot_constraint=add,
                target_dataframe=pl.DataFrame({
                    "recipient_phone_number": expected["phone"],
                    "content": expected["content"]}))]),
            M(snapshot_constraints=[C(
                database_namespace=DB.SETTING, snapshot_constraint=snap,
                target_dataframe=pl.DataFrame({field: False}))]),
        ]
        scenario = Extension(
            name=name, base_scenario=bases["base"], messages=messages(private, public),
            tool_allow_list=list(WIDE_TOOLS), milestones=milestones,
            milestone_edge_list=[(0, 1), (1, 2)],
            categories=[Cat.SINGLE_USER_TURN, Cat.CANONICALIZATION],
        ).get_extended_scenario()[name]
        scenarios[name] = scenario
        metadata[name] = {"family": family, "public_request": public,
            "operation_graph": list(WIDE_TOOLS), "expected": expected}

    literal_specs = (
        ("literal_wifi", "Replace Queue", "+14155550201", "reviewer",
         "swap out latest draft", "wifi", (
             "Add a contact named 'Replace Queue' at +14155550201 as reviewer, "
             "text it exactly 'swap out latest draft', and turn Wi-Fi off.",
             "Save +14155550201 under 'Replace Queue' with relationship reviewer; "
             "send the exact text 'swap out latest draft' and disable wifi.")),
        ("literal_cellular", "Recreate Latest", "+14155550202", "auditor",
         "rebuild next reminder", "cellular", (
             "Create an auditor contact called \"Recreate Latest\" at +14155550202, "
             "send exactly \"rebuild next reminder\", and disable cellular.",
             "Record +14155550202 as auditor under \"Recreate Latest\"; text that "
             "contact exactly \"rebuild next reminder\" and turn mobile data off.")),
        ("literal_location", "Substitute Reminder", "+14155550203", "coordinator",
         "replace newest message", "location_service", (
             "Add a contact named 'Substitute Reminder' at +14155550203 as "
             "coordinator, text it "
             "exactly 'replace newest message', and turn location services off.",
             "Save a coordinator contact named 'Substitute Reminder' with number "
             "+14155550203; send the literal 'replace newest message' and disable "
             "location service.")),
    )
    for family, name, phone, relationship, content, setting_field, prompts in literal_specs:
        expected = {"name": name, "phone": phone, "relationship": relationship,
                    "content": content, "setting_field": setting_field}
        private = (f"Add {name} at {phone} as {relationship}, send the exact literal "
                   f"{content}, and disable {setting_field}.")
        for index, public in enumerate(prompts):
            suffix = "" if index == 0 else "_alt"
            literal_extension("v2_" + family + suffix, family, public, private,
                              expected)

    retargets = {
        "contact_replace_message": (
            "Swap out John Petrucci for JP Archive at his retained phone as "
            "collaborator, send 'Profile moved' there, and remind me tomorrow at "
            "14:10 to Review JP record.",
            "Substitute the John Petrucci entry with JP Archive at the same number "
            "as collaborator; text 'Profile moved' to it and schedule a reminder "
            "named 'Review JP record' at 2:10 PM tomorrow."),
        "reminder_replace_boss": (
            "Substitute the upcoming reminder with 'Confirm venue' at the unchanged "
            "time, mark my boss as director, and text that contact 'Reminder updated'.",
            "Swap out the next reminder for 'Confirm venue' without moving its time; "
            "set the boss relationship to director and send 'Reminder updated'."),
        "latest_branch_contact": (
            "At 7:30 tomorrow, use my most recent text unchanged as a reminder; mark "
            "my boss as manager, save Ops Desk (+14155550182) as a coordinator "
            "contact, and message the boss with 'Digest: ' plus that text.",
            "Make a 07:30 tomorrow reminder from the raw latest message, change my boss "
            "relationship to manager, add a coordinator contact named Ops Desk at "
            "+14155550182, then send the boss Digest: followed by the raw message."),
    }
    for family, prompts in retargets.items():
        old_names = sorted(name for name, item in old_metadata.items()
                           if item["family"] == family)
        for index, (old_name, public) in enumerate(zip(old_names, prompts)):
            new_name = "v2_true_" + family + ("" if index == 0 else "_alt")
            scenarios[new_name] = _retarget(old_scenarios[old_name], public, u)
            item = deepcopy(old_metadata[old_name])
            item.update({"family": family, "public_request": public,
                         "operation_graph": list(WIDE_TOOLS)})
            metadata[new_name] = item

    order = ("literal_wifi", "literal_cellular", "literal_location",
             "contact_replace_message", "reminder_replace_boss",
             "latest_branch_contact")
    trajectories = [{"trajectory": family, "sessions": sorted(
        name for name, item in metadata.items() if item["family"] == family)}
        for family in order]
    return scenarios, metadata, trajectories


def _latest_rows(context, namespace):
    return prior._latest_rows(context, namespace)


def requested_state_checks(row, spec):
    family = spec["family"]
    if not family.startswith("literal_"):
        return prior.requested_state_checks(row, spec)
    context, expected = row["ending_context"], spec["expected"]
    contacts = _latest_rows(context, "CONTACT")
    messages = _latest_rows(context, "MESSAGING")
    settings = _latest_rows(context, "SETTING")
    contact = any(item.get("name") == expected["name"] and
                  item.get("phone_number") == expected["phone"] and
                  item.get("relationship") == expected["relationship"] and
                  item.get("is_self") is False for item in contacts)
    message = any(item.get("recipient_phone_number") == expected["phone"] and
                  item.get("content") == expected["content"] for item in messages)
    setting = next(item for item in settings if item.get("device_id") is not None)
    return [float(contact), float(message),
            float(setting.get(expected["setting_field"]) is False)]


def annotate_direct(row, metadata):
    spec = metadata[row["task"]]
    checks = requested_state_checks(row, spec)
    required = v2.request_required_mutations(
        spec["public_request"], set(row["allowed_tools"]))
    successful = {call["tool"] for call in row["calls"] if not call.get("exception")}
    row.update({"requested_state_score": float(np.mean(checks)),
        "requested_state_success": int(all(value >= 1 - 1e-8 for value in checks)),
        "process_state_score": len(required & successful) / len(required),
        "process_state_success": int(required <= successful),
        "family": spec["family"], "exact_state_predicates_registered": True})
    return row


def golden_self_check(u, scenarios, metadata):
    actual = {name: scenario for name, scenario in scenarios.items()
              if not metadata[name]["family"].startswith("literal_")}
    actual_meta = {name: metadata[name] for name in actual}
    checks = list(prior.golden_self_check(u, actual, actual_meta)["checks"])
    with tempfile.TemporaryDirectory(prefix="fast-v2-golden-") as temporary:
        for name, scenario in scenarios.items():
            spec = metadata[name]
            if not spec["family"].startswith("literal_"):
                continue
            expected = spec["expected"]
            setting_tool = {
                "wifi": "set_wifi_status", "cellular": "set_cellular_service_status",
                "location_service": "set_location_service_status",
            }[expected["setting_field"]]
            actions = [
                {"tool": "add_contact", "arguments": {
                    "name": expected["name"], "phone_number": expected["phone"],
                    "relationship": expected["relationship"]}},
                {"tool": "send_message_with_phone_number", "arguments": {
                    "phone_number": expected["phone"], "content": expected["content"]}},
                {"tool": setting_tool, "arguments": {"on": False}},
            ]
            job = initial_job_v2(name, scenario, [], u, None,
                                 "golden_v2_preflight", "gold")
            for index, action in enumerate(actions):
                adapter.consume(job, {"agent_output": json.dumps(action),
                    "prompt_sha256": f"gold-v2-{index}",
                    "hit_generation_limit": False}, u, 12)
            adapter.consume(job, {"agent_output": json.dumps({"answer": "Completed."}),
                "prompt_sha256": "gold-v2-final", "hit_generation_limit": False}, u, 12)
            row = prior.finish_direct(job, Path(temporary), u)
            row = annotate_direct(row, metadata)
            checks.append({"task": name,
                "requested_state_success": row["requested_state_success"],
                "process_state_success": row["process_state_success"]})
    if not all(item["requested_state_success"] ==
               item["process_state_success"] == 1 for item in checks):
        raise RuntimeError("v2 golden evaluator failed: " + json.dumps(checks))
    return {"passed": True, "tasks": len(checks), "checks": checks}


def select(method, pool, task, query, scenario, pair_vetoes, tokenizer):
    if method in {"Contract-Adaptive-v1", "Contract-Adaptive-v2"}:
        return v1.fast_select(pool, query, scenario, tokenizer, adaptive=True)
    if method == "Contract-Text":
        return v1.fast_contract_select(pool, query, scenario, tokenizer)
    if method == "CREDIT-Compose":
        return compose_select(pool, query, scenario)
    return prior_select(method, pool, task, query, scenario, pair_vetoes)


def _consume(job, record, u, max_calls, query):
    began = perf_counter_ns()
    if job["method"] == "Contract-Adaptive-v1":
        v1.consume_fast(job, record, u, max_calls, query)
    elif job["method"] == "Contract-Adaptive-v2":
        v2.consume_fast(job, record, u, max_calls, query)
    else:
        adapter.consume(job, record, u, max_calls)
        job["generations"][-1]["local_consume_latency_ms"] = (
            perf_counter_ns() - began) / 1e6


def run_batch(jobs, queries, agent, directory, u, *, max_calls=12):
    rows, active = [], jobs
    for _ in range(MAX_MODEL_TURNS):
        if not active:
            break
        records = agent.infer([job["chat"] for job in active])
        remaining = []
        for job, record in zip(active, records):
            _consume(job, record, u, max_calls, queries[job["task"]])
            if job["done"]:
                rows.append(prior.finish_direct(job, directory, u))
            else:
                remaining.append(job)
        active = remaining
    for job in active:
        job["termination"] = "turn_budget"
        rows.append(prior.finish_direct(job, directory, u))
    return rows


def _latency(row):
    remote = session_latency(row)
    local = float(row.get("selection_latency_ms", 0.0)) + sum(
        float(turn.get("local_consume_latency_ms", 0.0))
        for turn in row["generations"])
    return remote, local, remote + local


def run_actor(seed, actor_index, trajectories, root, args, scenarios, metadata,
              pool, pair_vetoes, u):
    directory = root / f"fast_v2_confirmation_seed_{seed}"
    directory.mkdir()
    agent = RemoteJSONAgent(directory, seed, env=args.env,
        tokenizer_path=args.tokenizer, temperature=0.0, max_new_tokens=384,
        workers=args.workers, exact_prompt_cache=False)
    agent.manifest["benchmark"] = "ToolSandbox-M literal-safe contract confirmation"
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
                                     "fast_v2_contract_confirmation", method)
                job["selection_latency_ms"] = selection_ms
                jobs.append(job); info[method] = (bank, position)
            for row in run_batch(jobs, {task: query}, agent, directory, u):
                bank, position = info[row["method"]]
                remote, local, accounted = _latency(row)
                item = annotate_direct({**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"],
                    "session_index": session_index,
                    "selected_bank": [note["uid"] for note in bank],
                    "remote_latency_ms": remote, "local_runtime_ms": local,
                    "accounted_latency_ms": accounted, "batch_position": position,
                    "public_request_sha256": _sha(query)}, metadata)
                sessions.append(item)
                with (directory / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    agent.close()
    summary = {"actor_seed": seed, "sessions": len(sessions),
        "remote_requests": agent.calls,
        "method_batch_position_counts": dict(positions)}
    adapter.write(directory, "summary.json", summary)
    return sessions, summary


def summarize(rows):
    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        latency = [row["accounted_latency_ms"] for row in subset]
        turns = [turn for row in subset for turn in row["generations"]]
        item = {"method": method, "sessions": len(subset),
            "requested_state_VSR": float(np.mean([
                row["requested_state_success"] for row in subset])),
            "process_state_VSR": float(np.mean([
                row["process_state_success"] for row in subset])),
            "mean_requested_state_score": float(np.mean([
                row["requested_state_score"] for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "mean_remote_requests": float(np.mean([
                sum("remote_latency_ms" in turn for turn in row["generations"])
                for row in subset])),
            "mean_accounted_latency_ms": float(np.mean(latency)),
            "p95_accounted_latency_ms": float(np.quantile(latency, .95)),
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


def paired_statistics(rows):
    by = defaultdict(dict)
    for row in rows:
        by[row["method"]][(row["actor_seed"], row["trajectory"],
                           row["session_index"])] = row
    clusters = sorted({(row["actor_seed"], row["trajectory"]) for row in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    output = []
    for baseline in ("Contract-Adaptive-v1", "Semantic"):
        state, budget = [], []
        for seed, trajectory in clusters:
            keys = [(seed, trajectory, index) for index in (0, 1)]
            state.append(float(np.mean([
                by["Contract-Adaptive-v2"][key]["requested_state_success"] -
                by[baseline][key]["requested_state_success"] for key in keys])))
            budget.append(float(np.mean([
                (by["Contract-Adaptive-v2"][key]["requested_state_success"] and
                 by["Contract-Adaptive-v2"][key]["accounted_latency_ms"] <= PRIMARY_BUDGET_MS) -
                (by[baseline][key]["requested_state_success"] and
                 by[baseline][key]["accounted_latency_ms"] <= PRIMARY_BUDGET_MS)
                for key in keys])))
        state_array, budget_array = np.asarray(state), np.asarray(budget)
        state_boot = np.mean(state_array[draws], axis=1)
        budget_boot = np.mean(budget_array[draws], axis=1)
        wins, losses = sum(x > 0 for x in budget), sum(x < 0 for x in budget)
        p = float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue) \
            if wins + losses else 1.0
        output.append({"baseline": baseline,
            "delta_requested_state_VSR": float(np.mean(state)),
            "state_cluster_bootstrap_CI95": np.quantile(
                state_boot, [.025, .975]).tolist(),
            "delta_state_VSR_at_15000ms": float(np.mean(budget)),
            "budget_cluster_bootstrap_CI95": np.quantile(
                budget_boot, [.025, .975]).tolist(),
            "budget_cluster_wins": wins, "budget_cluster_losses": losses,
            "budget_one_sided_exact_sign_p": p})
    return {"comparisons": output, "clusters": len(clusters),
        "resampling_unit": "actor seed by two-prompt task family",
        "bootstrap_replicates": 10000, "bootstrap_seed": BOOTSTRAP_SEED}


def holm_primary(report):
    ordered = sorted((dict(row) for row in report["comparisons"]),
                     key=lambda row: row["budget_one_sided_exact_sign_p"])
    running = 0.0
    for index, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) *
                      row["budget_one_sided_exact_sign_p"]))
        row["holm_adjusted_p"] = running
    return sorted(ordered, key=lambda row: row["baseline"])


def task_breakdown(rows):
    output = []
    for family in sorted({row["trajectory"] for row in rows}):
        for method in METHODS:
            subset = [row for row in rows
                      if row["trajectory"] == family and row["method"] == method]
            output.append({"family": family, "method": method,
                "sessions": len(subset),
                "requested_state_VSR": float(np.mean([
                    row["requested_state_success"] for row in subset])),
                "requested_state_VSR_at_15000ms": float(np.mean([
                    row["requested_state_success"] and
                    row["accounted_latency_ms"] <= PRIMARY_BUDGET_MS
                    for row in subset])),
                "mean_remote_requests": float(np.mean([
                    sum("remote_latency_ms" in turn for turn in row["generations"])
                    for row in subset])),
                "mean_accounted_latency_ms": float(np.mean([
                    row["accounted_latency_ms"] for row in subset]))})
    return output


def freeze(args, scenarios, metadata, trajectories, pool, pair_vetoes, preflight):
    names = ["toolsandbox_m_contract_fast_v2_confirmation.py",
             "toolsandbox_m_contract_fast_v2_confirmation_verify.py",
             "toolsandbox_m_contract_fast_v2.py",
             "toolsandbox_m_contract_fast.py",
             "toolsandbox_m_contract_fast_confirmation.py",
             "toolsandbox_m_compose_selector.py", "toolsandbox_m_intent_optimization.py",
             "toolsandbox_m_followup.py", "toolsandbox_local_v3.py",
             "remote_json_agent.py"]
    source_dir = args.output / "source"; source_dir.mkdir()
    source_sha = {}
    for name in names:
        source = Path(__file__).with_name(name)
        shutil.copy2(source, source_dir / name)
        source_sha[name] = digest(source)
    development_sha = {}
    for source in args.development_sources:
        path = source / "confirmation_sessions.jsonl"
        if not path.exists():
            path = source / "sessions.jsonl"
        if not path.exists():
            candidates = sorted(source.glob("*/sessions.jsonl"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"expected one development sessions file under {source}")
            path = candidates[0]
        development_sha[str(source)] = {"filename": str(path.relative_to(source)),
                                         "sha256": digest(path)}
    previous = json.loads((args.protocol_source / "frozen_protocol.json").read_text())
    protocol = {"suite": "literal-safe fast-contract held-out confirmation",
        "frozen_before_remote_calls": True, "clock": CLOCK,
        "actor_seeds": list(args.seeds), "methods": list(METHODS),
        "primary_endpoint": "exact requested-world-state VSR",
        "primary_latency_endpoint": PRIMARY_BUDGET_MS,
        "primary_comparisons": ["Contract-Adaptive-v2 vs Contract-Adaptive-v1",
                                "Contract-Adaptive-v2 vs Semantic"],
        "multiplicity": "Holm over two registered 15-second comparisons",
        "latency_endpoints_ms": list(LATENCY_BUDGETS_MS),
        "maximum_model_turns_per_session": MAX_MODEL_TURNS,
        "maximum_tool_calls_per_session": 12,
        "exact_prompt_cache": False, "provider_credentials_persisted": False,
        "wide_identical_tool_catalogue": list(WIDE_TOOLS),
        "tasks": [{"name": name, "family": metadata[name]["family"],
                   "public_request_sha256": _sha(metadata[name]["public_request"])}
                  for name in sorted(metadata)],
        "trajectories": trajectories, "pair_vetoes": [list(x) for x in pair_vetoes],
        "memory_pool_source": str(args.protocol_source),
        "memory_pool_source_sha256": digest(args.protocol_source / "memory_pool.json"),
        "development_sources": [str(x) for x in args.development_sources],
        "development_source_sha256": development_sha,
        "source_sha256": source_sha,
        "upstream_directory": previous["upstream_directory"],
        "upstream_source_sha256": previous["upstream_source_sha256"],
        "preflight": preflight}
    adapter.write(args.output, "frozen_protocol.json", protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ACTOR_SEEDS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--protocol-source", type=Path,
        default=Path("results/toolsandbox_m_contract_fast_confirmation_v1"))
    parser.add_argument("--development-sources", type=Path, nargs="+", default=[
        Path("results/toolsandbox_m_contract_fast_confirmation_v1"),
        Path("results/toolsandbox_m_contract_fast_v2_development_v1")])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prior_protocol = json.loads((args.protocol_source / "frozen_protocol.json").read_text())
    u = adapter.upstream_imports(Path(prior_protocol["upstream_directory"]))
    adapter.install_local_clock(CLOCK)
    scenarios, metadata, trajectories = build_confirmation_scenarios(u)
    pool = json.loads((args.protocol_source / "memory_pool.json").read_text())
    shutil.copy2(args.protocol_source / "memory_pool.json", args.output / "memory_pool.json")
    pair_vetoes = {tuple(x) for x in prior_protocol["pair_vetoes"]}
    preflight = {"golden_evaluator": golden_self_check(u, scenarios, metadata),
        "novelty": novelty_audit(u, scenarios, metadata, args.development_sources)}
    adapter.write(args.output, "preflight.json", preflight)
    freeze(args, scenarios, metadata, trajectories, pool, pair_vetoes, preflight)
    manifest = json.loads((args.protocol_source /
        "fast_confirmation_seed_1501/agent_manifest.json").read_text())
    args.tokenizer = args.tokenizer or Path(manifest["tokenizer_path"])
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    selection = {}
    for task, scenario in scenarios.items():
        bank = v1.fast_contract_select(
            pool, metadata[task]["public_request"], scenario, tokenizer)
        selection[task] = {"selected_uids": [note["uid"] for note in bank],
            "contract_text": bank[0]["text"], "contract_tools": bank[0]["tools"],
            "exposed_tools": list(WIDE_TOOLS)}
    adapter.write(args.output, "frozen_selection.json", selection)
    if args.prepare_only:
        print(json.dumps({"prepared": True, "tasks": len(metadata),
                          "preflight": preflight}, indent=2))
        return

    rows, actors = [], []
    for index, seed in enumerate(args.seeds):
        found, actor = run_actor(seed, index, trajectories, args.output, args,
                                 scenarios, metadata, pool, pair_vetoes, u)
        rows.extend(found); actors.append(actor)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table, statistics = summarize(rows), paired_statistics(rows)
    ranking = sorted((dict(row) for row in table), key=lambda row: (
        -row[f"requested_state_VSR_at_{PRIMARY_BUDGET_MS}ms"],
        -row["requested_state_VSR"], row["mean_accounted_latency_ms"], row["method"]))
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    summary = {"confirmation_table": table, "ranking": ranking,
        "paired_statistics": statistics, "holm_primary": holm_primary(statistics),
        "task_breakdown": task_breakdown(rows), "actor_summaries": actors,
        "frozen_selection": selection}
    adapter.write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    breakdown = summary["task_breakdown"]
    with (args.output / "task_family_breakdown.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
        writer.writeheader(); writer.writerows(breakdown)
    adapter.write(args.output, "audit.json", {"sessions": len(rows),
        "expected_sessions": len(args.seeds) * len(metadata) * len(METHODS),
        "live_remote_requests": sum(actor["remote_requests"] for actor in actors),
        "credentials_persisted": False, "all_tasks_share_identical_tools": True})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
