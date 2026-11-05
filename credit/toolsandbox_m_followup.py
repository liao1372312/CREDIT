"""Leakage-aware follow-up for multi-actor ToolSandbox-M.

The previous confirmation is treated as immutable diagnostic evidence.  This
follow-up trains one shared causal policy, tunes only on a hash-defined
development split, and evaluates the frozen policy with new actor seeds on a
disjoint trajectory split.  Methods for the same task are sent in the same
remote batch so provider load is paired rather than confounded with method
order.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from statistics import mean, stdev

import dill
import numpy as np
from scipy.stats import binomtest

from .remote_json_agent import RemoteJSONAgent
from .toolsandbox_local_v3 import (
    MODEL, UPSTREAM, family, fingerprint, initial_job, install_local_clock,
    local_scenarios, run_batch, upstream_imports, write,
)
from .toolsandbox_m import continuous_score, make_trajectories, note_relevance
from .user_data import digest


CONFIRM_METHODS = ("NoMem", "FIFO", "Recency", "Semantic", "CREDIT")
DEV_VARIANTS = ("compact1", "coverage2", "quality2")
LATENCY_BUDGETS_MS = (5000, 10000, 20000)
SPLIT_SALT = "credit-followup-v1"
FINAL_ANSWER_GUIDANCE = (
    " Before the final answer, re-read the request and report the requested "
    "value or state precisely, preserving units and relevant tool-result values."
)


def _parse_response(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def _scalar_paths(value, prefix: str = "result") -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_scalar_paths(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for item in value[:8]:
            rows.extend(_scalar_paths(item, prefix + "[*]"))
    elif isinstance(value, (str, int, float, bool)) and value not in ("", None):
        rows.append((prefix, value))
    return rows


def _dataflow_rules(calls: list[dict]) -> list[str]:
    rules: list[str] = []
    for left_index, left in enumerate(calls):
        if left.get("exception"):
            continue
        paths = _scalar_paths(_parse_response(left.get("response", "")))
        for right in calls[left_index + 1:]:
            if right.get("exception"):
                continue
            for argument, value in right.get("arguments", {}).items():
                matches = [path for path, old in paths if old == value and
                           not isinstance(value, bool)]
                if matches:
                    rule = f"{right['tool']}.{argument} <- {left['tool']}.{matches[0]}"
                    if rule not in rules:
                        rules.append(rule)
    return rules[:4]


def memory_from_v2(row: dict, tokenizer) -> dict | None:
    """Create a concise recipe from verified milestones and successful calls.

    Failed calls are not placed in the main recipe.  When a later retry of the
    same tool succeeds, the error type and intervening repair tools are retained
    as an explicit recovery rule.
    """
    evaluation = row.get("upstream_evaluation")
    if not evaluation or evaluation["minefield_similarity"] != 0 or not any(
            value[1] >= 1 - 1e-8 for value in evaluation["milestone_mapping"].values()):
        return None
    calls = row.get("calls", [])
    successful = [call for call in calls if not call.get("exception")]
    tools = sorted({call["tool"] for call in successful})
    if not tools:
        return None
    steps = [f"{call['tool']}({','.join(sorted(call.get('arguments', {})))})"
             for call in successful]
    pieces = ["Verified related-task recipe: " + " -> ".join(steps) + "."]
    flows = _dataflow_rules(calls)
    if flows:
        pieces.append("Bindings: " + "; ".join(flows) + ".")
    repairs = []
    for index, failed in enumerate(calls):
        if not failed.get("exception"):
            continue
        later = next((j for j in range(index + 1, len(calls))
                      if calls[j]["tool"] == failed["tool"] and not calls[j].get("exception")), None)
        if later is not None:
            bridge = [call["tool"] for call in calls[index + 1:later]
                      if not call.get("exception")]
            kind = str(failed["exception"]).split(":", 1)[0]
            repairs.append(f"on {kind} from {failed['tool']}, run " +
                           (" -> ".join(bridge) if bridge else "a corrected retry") + " then retry")
    if repairs:
        pieces.append("Recovery: " + "; ".join(repairs[:2]) + ".")
    pieces.append("Query current identifiers, dates, and values; never copy prior literals.")
    text = " ".join(pieces)
    tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if tokens > 170:
        # Dataflow is more reusable than verbose recovery detail.
        pieces = pieces[:2] + [pieces[-1]]
        text = " ".join(pieces)
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if tokens > 170:
        return None
    return {
        "uid": fingerprint(["v2", row["task"], text])[:16],
        "text": text,
        "tools": tools,
        "tokens": tokens,
        "source_task": row["task"],
        "source_ending_sha256": row["ending_state_sha256"],
        "source_verifier_success": bool(row["success"]),
        "source_exact_milestone_verified": True,
        "source_upstream_score": continuous_score(row),
        "arrival": row["task"],
        "memory_schema": "verified_recipe_v2",
    }


def initial_job_v2(name, scenario, bank, u, tokenizer, stage, method):
    job = initial_job(name, scenario, bank, u, tokenizer, stage, method)
    job["chat"][0]["content"] += FINAL_ANSWER_GUIDANCE
    return job


def _ranked(pool: list[dict], task: str, scenario, causal_model: dict,
            *, quality: bool) -> list[dict]:
    allowed = set(scenario.starting_context.tool_allow_list or [])
    supported = [note for note in pool if set(note["tools"]) & allowed]

    def score(note: dict) -> tuple[float, float, float, str]:
        causal = causal_model.get(note["source_task"], {})
        status = causal.get("status", "unresolved")
        causal_term = 25.0 if status == "beneficial" else (-40.0 if status == "harmful" else 0.0)
        quality_term = 12.0 * float(note["source_upstream_score"]) if quality else 0.0
        value = note_relevance(note, task, allowed) + causal_term + quality_term
        return (-value, -float(note["source_upstream_score"]),
                -float(note.get("source_verifier_success", False)), note["uid"])

    return sorted(supported, key=score)


def select_bank_v2(method: str, pool: list[dict], task: str, scenario,
                   causal_model: dict, *, variant: str = "coverage2") -> list[dict]:
    allowed = set(scenario.starting_context.tool_allow_list or [])
    supported = [note for note in pool if set(note["tools"]) & allowed]
    if method == "NoMem":
        return []
    if method == "FIFO":
        return supported[:2]
    if method == "Recency":
        return list(reversed(supported))[:2]
    if method == "Semantic":
        return sorted(supported, key=lambda note: (-note_relevance(note, task, allowed),
                                                    note["arrival"], note["uid"]))[:2]
    if method != "CREDIT" and not method.startswith("CREDIT-"):
        raise ValueError("unknown method: " + method)
    if variant not in DEV_VARIANTS:
        raise ValueError("unknown CREDIT variant: " + variant)
    ranked = _ranked(pool, task, scenario, causal_model, quality=(variant == "quality2"))
    admissible = [note for note in ranked
                  if causal_model.get(note["source_task"], {}).get("status") != "harmful"]
    # Certification-style fallback: unresolved causal evidence does not by
    # itself reorder the semantic incumbent.  Stable harmful evidence can veto
    # exposure; stable benefit can promote it through _ranked.
    if not any(causal_model.get(note["source_task"], {}).get("status") in
               {"beneficial", "harmful"} for note in supported):
        admissible = select_bank_v2("Semantic", pool, task, scenario, causal_model)
    if not admissible:
        return []
    if variant == "compact1":
        return admissible[:1]
    if variant == "quality2":
        return admissible[:2]
    chosen = [admissible[0]]
    covered = set(admissible[0]["tools"]) & allowed
    for note in admissible[1:]:
        adds_tool = bool((set(note["tools"]) & allowed) - covered)
        exact_family = family(note["source_task"]) == family(task)
        if adds_tool or exact_family:
            chosen.append(note)
            break
    return chosen


def session_latency(row: dict) -> float:
    return float(sum(turn.get("remote_latency_ms", 0.0) for turn in row["generations"]))


def split_trajectories(trajectories: list[dict], development: int = 8):
    ranked = sorted(trajectories, key=lambda row: sha256(
        (SPLIT_SALT + ":" + row["trajectory"]).encode()).hexdigest())
    return ranked[:development], ranked[development:]


def _agent(directory: Path, seed: int, args) -> RemoteJSONAgent:
    directory.mkdir(parents=True)
    agent = RemoteJSONAgent(directory, seed, env=args.env, tokenizer_path=args.tokenizer,
                            temperature=0.0, max_new_tokens=384,
                            workers=args.workers, exact_prompt_cache=False)
    agent.binding_interface = "tool_json"
    agent.manifest["benchmark"] = "ToolSandbox-M leakage-aware optimization follow-up"
    return agent


def train_shared_policy(root: Path, args, scenarios, history, u) -> tuple[list[dict], dict, dict]:
    base_seed = args.training_seeds[0]
    base_dir = root / f"training_pool_seed_{base_seed}"
    base_agent = _agent(base_dir, base_seed, args)
    controls = []
    for offset in range(0, len(history), args.workers):
        controls += run_batch([initial_job_v2(name, scenarios[name], [], u, base_agent.tokenizer,
            "shared_history", "NoMem") for name in history[offset:offset + args.workers]],
            base_agent, base_dir, u)
    pool = [note for row in sorted(controls, key=lambda row: history.index(row["task"]))
            if (note := memory_from_v2(row, base_agent.tokenizer)) is not None]
    base_agent.close()
    write(root, "memory_pool.json", pool)

    pairs = []
    total_latency = 0.0
    for seed in args.training_seeds:
        directory = root / f"calibration_seed_{seed}"
        agent = _agent(directory, seed, args)
        for offset in range(0, len(pool), max(1, args.workers // 2)):
            jobs = []
            for note in pool[offset:offset + max(1, args.workers // 2)]:
                task = note["source_task"]
                jobs.extend([
                    initial_job_v2(task, scenarios[task], [], u, agent.tokenizer,
                                   "causal_calibration", "control"),
                    initial_job_v2(task, scenarios[task], [note], u, agent.tokenizer,
                                   "causal_calibration", "single_memory_treatment"),
                ])
            rows = run_batch(jobs, agent, directory, u)
            by_task_method = {(row["task"], row["method"]): row for row in rows}
            for note in pool[offset:offset + max(1, args.workers // 2)]:
                task = note["source_task"]
                control = by_task_method[(task, "control")]
                treatment = by_task_method[(task, "single_memory_treatment")]
                pairs.append({
                    "actor_seed": seed, "uid": note["uid"], "source_task": task,
                    "control_score": continuous_score(control),
                    "treatment_score": continuous_score(treatment),
                    "paired_effect": continuous_score(treatment) - continuous_score(control),
                    "control_success": control["success"],
                    "treatment_success": treatment["success"],
                })
                total_latency += session_latency(control) + session_latency(treatment)
        agent.close()
    write(root, "causal_calibration.json", pairs)
    causal_model = {}
    for task in sorted({row["source_task"] for row in pairs}):
        effects = [row["paired_effect"] for row in pairs if row["source_task"] == task]
        epsilon = 0.005
        if min(effects) > epsilon:
            status = "beneficial"
        elif max(effects) < -epsilon:
            status = "harmful"
        else:
            status = "unresolved"
        causal_model[task] = {
            "n": len(effects), "mean_effect": float(mean(effects)),
            "sd_effect": float(stdev(effects)) if len(effects) > 1 else 0.0,
            "min_effect": float(min(effects)), "max_effect": float(max(effects)),
            "status": status,
            "decision_rule": "beneficial iff every effect > .005; harmful iff every effect < -.005",
        }
    write(root, "causal_model.json", causal_model)
    training = {
        "pool_seed": base_seed, "training_seeds": args.training_seeds,
        "history_sessions": len(controls), "memory_records": len(pool),
        "calibration_pairs": len(pairs), "calibration_sessions": 2 * len(pairs),
        "calibration_remote_latency_ms": total_latency,
    }
    write(root, "training_summary.json", training)
    return pool, causal_model, training


def _balanced_order(methods: list[str], actor_index: int, trajectory_index: int,
                    session_index: int) -> list[str]:
    shift = (actor_index + trajectory_index + session_index) % len(methods)
    return methods[shift:] + methods[:shift]


def run_actor(seed: int, actor_index: int, phase: str, trajectories: list[dict],
              methods: list[str], root: Path, args, scenarios, initial_pool: list[dict],
              causal_model: dict, variants: dict[str, str], u) -> tuple[list[dict], dict]:
    directory = root / f"{phase}_seed_{seed}"
    agent = _agent(directory, seed, args)
    pools = {method: {trajectory["trajectory"]: [dict(note) for note in initial_pool]
                      for trajectory in trajectories} for method in methods}
    memories = {note["uid"]: note for note in initial_pool}
    sessions = []
    position_counts = defaultdict(lambda: [0] * len(methods))
    for trajectory_index, trajectory in enumerate(trajectories):
        for session_index, task in enumerate(trajectory["sessions"]):
            order = _balanced_order(methods, actor_index, trajectory_index, session_index)
            jobs, metadata = [], {}
            for position, method in enumerate(order):
                position_counts[method][position] += 1
                pool = pools[method][trajectory["trajectory"]]
                variant = variants.get(method, "coverage2")
                selector_method = "CREDIT" if method.startswith("CREDIT-") else method
                bank = select_bank_v2(selector_method, pool, task, scenarios[task],
                                      causal_model, variant=variant)
                jobs.append(initial_job_v2(task, scenarios[task], bank, u, agent.tokenizer,
                                           phase, method))
                metadata[method] = (pool, bank, [note["uid"] for note in pool], position)
            for row in run_batch(jobs, agent, directory, u):
                pool, bank, before, position = metadata[row["method"]]
                note = memory_from_v2(row, agent.tokenizer)
                inserted = None
                if note is not None and not any(old["uid"] == note["uid"] for old in pool):
                    pool.append(note)
                    memories[note["uid"]] = note
                    inserted = note["uid"]
                session = {**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"], "session_index": session_index,
                    "available_memory_before": before,
                    "selected_bank": [note["uid"] for note in bank],
                    "inserted_after_scoring": inserted,
                    "cross_session_exposure": session_index == 1 and any(
                        note["source_task"] == trajectory["sessions"][0] for note in bank),
                    "remote_latency_ms": session_latency(row), "batch_position": position,
                    "credit_variant": variants.get(row["method"]),
                }
                sessions.append(session)
                with (directory / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(session, ensure_ascii=False) + "\n")
    write(directory, "all_memories.json", list(memories.values()))
    agent.close()
    summary = {
        "actor_seed": seed, "phase": phase, "test_sessions": len(sessions),
        "remote_requests": agent.calls,
        "method_batch_position_counts": dict(position_counts),
    }
    write(directory, "summary.json", summary)
    return sessions, summary


def summarize(rows: list[dict], methods: list[str], *, calibration_ms: float = 0.0,
              credit_method: str = "CREDIT") -> list[dict]:
    table = []
    credit_count = sum(row["method"] == credit_method for row in rows)
    amortized = calibration_ms / credit_count if credit_count else 0.0
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        attributed = [row["remote_latency_ms"] + (amortized if method == credit_method else 0.0)
                      for row in subset]
        entry = {
            "method": method, "sessions": len(subset),
            "strict_VSR": float(np.mean([row["success"] for row in subset])),
            "upstream_mean_score": float(np.mean([continuous_score(row) for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "mean_remote_requests": float(np.mean([len(row["generations"]) for row in subset])),
            "mean_online_remote_latency_ms": float(np.mean([row["remote_latency_ms"] for row in subset])),
            "mean_attributed_remote_latency_ms": float(np.mean(attributed)),
            "p95_attributed_remote_latency_ms": float(np.quantile(attributed, .95)),
        }
        for budget in LATENCY_BUDGETS_MS:
            entry[f"VSR_at_{budget}ms"] = float(np.mean([
                row["success"] and latency <= budget for row, latency in zip(subset, attributed)]))
            entry[f"completion_at_{budget}ms"] = float(np.mean([
                latency <= budget for latency in attributed]))
        table.append(entry)
    return table


def choose_variant(rows: list[dict]) -> tuple[str, list[dict]]:
    methods = ["CREDIT-" + variant for variant in DEV_VARIANTS]
    table = summarize(rows, methods)
    best = max(table, key=lambda row: (row["VSR_at_10000ms"], row["strict_VSR"],
        row["upstream_mean_score"], -row["mean_online_remote_latency_ms"], row["method"]))
    return best["method"].removeprefix("CREDIT-"), table


def paired_statistics(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for row in rows:
        by[row["method"]][(row["actor_seed"], row["trajectory"], row["session_index"])] = row
    clusters = sorted({(row["actor_seed"], row["trajectory"]) for row in rows})
    rng = np.random.default_rng(20261014)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    report = []
    for method in CONFIRM_METHODS:
        effects = []
        for seed, trajectory in clusters:
            keys = [(seed, trajectory, index) for index in (0, 1)]
            effects.append(float(np.mean([by["CREDIT"][key]["success"] -
                                          by[method][key]["success"] for key in keys])))
        boot = np.mean(np.asarray(effects)[draws], axis=1)
        wins, losses = sum(x > 0 for x in effects), sum(x < 0 for x in effects)
        p = float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue) \
            if wins + losses else 1.0
        report.append({"baseline": method, "CREDIT_delta_VSR": float(np.mean(effects)),
            "cluster_bootstrap_CI95": np.quantile(boot, [.025, .975]).tolist(),
            "cluster_wins": wins, "cluster_losses": losses,
            "one_sided_exact_sign_p": p})
    return {"comparisons": report, "clusters": len(clusters),
            "resampling_unit": "actor-seed by two-session trajectory",
            "bootstrap_replicates": 10000, "bootstrap_seed": 20261014}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--tokenizer", type=Path, default=MODEL)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/toolsandbox_m_followup_v1"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[701, 709, 719])
    parser.add_argument("--dev-seeds", type=int, nargs="+", default=[733, 739])
    parser.add_argument("--confirm-seeds", type=int, nargs="+",
                        default=[751, 757, 761, 769, 773])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    (args.output / "source").mkdir()
    u = upstream_imports(args.upstream)
    clock = "2026-10-14T12:00:00"
    install_local_clock(clock)
    scenarios, exclusions = local_scenarios(u)
    history = sorted(name for name, scenario in scenarios.items()
                     if u.ScenarioCategories.SINGLE_TOOL_CALL in scenario.categories)
    future = sorted(set(scenarios) - set(history))
    trajectories = make_trajectories(future)
    development, confirmation = split_trajectories(trajectories)
    if args.smoke:
        args.training_seeds = args.training_seeds[:1]
        args.dev_seeds = args.dev_seeds[:1]
        args.confirm_seeds = args.confirm_seeds[:1]
        history = history[:2]
        development = development[:1]
        confirmation = confirmation[:1]
    source_names = ["toolsandbox_m_followup.py", "toolsandbox_m_followup_verify.py",
                    "remote_json_agent.py", "toolsandbox_local_v3.py", "toolsandbox_m.py"]
    source_hashes = {name: digest(Path(__file__).with_name(name)) for name in source_names}
    for name in source_names:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    upstream_hashes = {str(path.relative_to(args.upstream)): digest(path)
                       for path in args.upstream.rglob("*.py")}
    protocol = {
        "suite": "ToolSandbox-M leakage-aware optimization follow-up",
        "upstream_directory": str(args.upstream.resolve()),
        "upstream_commit": "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1",
        "upstream_source_sha256": upstream_hashes, "source_sha256": source_hashes,
        "clock": clock, "history": history, "development": development,
        "confirmation": confirmation, "split_salt": SPLIT_SALT,
        "training_seeds": args.training_seeds, "development_seeds": args.dev_seeds,
        "confirmation_seeds": args.confirm_seeds,
        "development_variants": list(DEV_VARIANTS),
        "confirmation_methods": list(CONFIRM_METHODS),
        "temperature": 0.0, "request_parallelism": args.workers,
        "paired_scheduling": "all methods for a task share one concurrent remote batch",
        "memory_schema": "verified_recipe_v2; failed calls excluded from primary recipe",
        "selection_rule": "hash split, development selection by 10-s VSR then raw VSR, grader, latency",
        "causal_training": "one shared three-seed paired calibration; frozen before confirmation",
        "calibration_attribution": "all paired control and treatment latency amortized over CREDIT confirmation sessions",
        "primary_latency_budget_ms": 10000,
        "latency_budgets_ms": list(LATENCY_BUDGETS_MS),
        "maximum_tool_calls_per_session": 8, "maximum_actor_turns": 16,
        "old_confirmation_used_for": "diagnosis only; no row enters training or model selection",
        "test_frozen_before_generation": True, "provider_credentials_persisted": False,
        "known_scope_limit": "new actor seeds and disjoint hash split, but task families come from the previously observed public suite",
        "exclusions": exclusions, "smoke": args.smoke,
    }
    write(args.output, "frozen_protocol.json", protocol)
    (args.output / "scenarios.pkl").write_bytes(dill.dumps(scenarios))
    write(args.output, "scenario_freeze.json", {"sha256": digest(args.output / "scenarios.pkl")})

    pool, causal_model, training = train_shared_policy(
        args.output, args, scenarios, history, u)
    dev_methods = ["CREDIT-" + variant for variant in DEV_VARIANTS]
    dev_rows, dev_actor_summaries = [], []
    variants = {method: method.removeprefix("CREDIT-") for method in dev_methods}
    for index, seed in enumerate(args.dev_seeds):
        rows, actor_summary = run_actor(seed, index, "development", development,
            dev_methods, args.output, args, scenarios, pool, causal_model, variants, u)
        dev_rows.extend(rows); dev_actor_summaries.append(actor_summary)
    with (args.output / "development_sessions.jsonl").open("w") as stream:
        for row in dev_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    chosen, dev_table = choose_variant(dev_rows)
    selection = {"chosen_variant": chosen, "table": dev_table,
                 "selection_keys": ["VSR_at_10000ms", "strict_VSR",
                                    "upstream_mean_score", "negative_mean_latency"]}
    write(args.output, "development_selection.json", selection)

    confirm_rows, confirm_actor_summaries = [], []
    for index, seed in enumerate(args.confirm_seeds):
        rows, actor_summary = run_actor(seed, index, "confirmation", confirmation,
            list(CONFIRM_METHODS), args.output, args, scenarios, pool, causal_model,
            {"CREDIT": chosen}, u)
        confirm_rows.extend(rows); confirm_actor_summaries.append(actor_summary)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in confirm_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = summarize(confirm_rows, list(CONFIRM_METHODS),
                      calibration_ms=training["calibration_remote_latency_ms"])
    ranking = sorted((dict(row) for row in table), key=lambda row: (-row["VSR_at_10000ms"],
        -row["strict_VSR"], row["mean_attributed_remote_latency_ms"], row["method"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    summary = {
        "chosen_credit_variant": chosen, "development_table": dev_table,
        "confirmation_table": table, "latency_ranking": ranking,
        "training": training, "development_actor_summaries": dev_actor_summaries,
        "confirmation_actor_summaries": confirm_actor_summaries,
        "paired_statistics": paired_statistics(confirm_rows),
        "primary_latency_budget_ms": 10000,
        "latency_definition": "complete measured HTTPS latency including retries and backoff",
        "budget_rule": "completed-response censoring with shared causal-training latency amortized over CREDIT confirmation sessions",
    }
    write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    audit = {
        "passed": len(confirm_rows) == len(args.confirm_seeds) * len(confirmation) * 2 * len(CONFIRM_METHODS),
        "development_sessions": len(dev_rows), "confirmation_sessions": len(confirm_rows),
        "unique_confirmation_keys": len({(row["actor_seed"], row["method"], row["task"])
                                         for row in confirm_rows}),
        "all_transport_latencies_positive": all(row["remote_latency_ms"] > 0 for row in confirm_rows),
        "all_verification_covered": all(row["verification_coverage"] for row in confirm_rows),
        "confirmation_sha256": digest(args.output / "confirmation_sessions.jsonl"),
        "protocol_sha256": digest(args.output / "frozen_protocol.json"),
    }
    audit["passed"] = audit["passed"] and audit["unique_confirmation_keys"] == len(confirm_rows) \
        and audit["all_transport_latencies_positive"] and audit["all_verification_covered"]
    write(args.output, "audit.json", audit)
    print(json.dumps({"selection": selection, "summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
