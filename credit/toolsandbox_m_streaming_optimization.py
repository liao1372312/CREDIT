"""Optimize CREDIT with development-derived natural multi-tool memories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil

import dill
import numpy as np
from scipy.stats import binomtest

from .remote_json_agent import RemoteJSONAgent
from .toolsandbox_local_v3 import install_local_clock, run_batch, upstream_imports, write
from .toolsandbox_m import continuous_score, note_relevance
from .toolsandbox_m_followup import initial_job_v2, memory_from_v2, session_latency
from .user_data import digest


DEV_VARIANTS = ("compact1", "quality2", "diverse2")
CONFIRM_METHODS = ("NoMem", "FIFO", "Recency", "Semantic", "CREDIT")
LATENCY_BUDGETS_MS = (5000, 10000, 20000)


def build_pool(source: Path, tokenizer) -> tuple[list[dict], list[dict]]:
    """Append high-quality, deduplicated memories made only from dev episodes."""
    initial = json.loads((source / "memory_pool.json").read_text())
    rows = [json.loads(line) for line in
            (source / "development_sessions.jsonl").read_text().splitlines() if line]
    best: dict[str, tuple[tuple, dict]] = {}
    for row in rows:
        note = memory_from_v2(row, tokenizer)
        if note is None or note["source_upstream_score"] < 0.75:
            continue
        note["origin_phase"] = "development"
        rank = (note["source_upstream_score"], note["source_verifier_success"],
                -note["tokens"], -len(row["calls"]), note["source_task"])
        if note["text"] not in best or rank > best[note["text"]][0]:
            best[note["text"]] = (rank, note)
    added = sorted((value[1] for value in best.values()),
                   key=lambda note: (note["source_task"], note["uid"]))
    return [*initial, *added], added


def leave_family_out(pool: list[dict], trajectory: str) -> list[dict]:
    return [note for note in pool if not (
        note.get("origin_phase") == "development" and
        (note["source_task"] == trajectory or
         note["source_task"].startswith(trajectory + "_")))]


def harmful_pairs(effects: dict) -> set[tuple[str, str]]:
    return {tuple(key.split("|")) for key, value in
            effects["aggregate"]["pairs"].items() if value["ci95"][1] < -0.005}


def _supported(pool: list[dict], scenario) -> tuple[list[dict], set[str]]:
    allowed = set(scenario.starting_context.tool_allow_list or [])
    return [note for note in pool if set(note["tools"]) & allowed], allowed


def select_bank(method: str, pool: list[dict], task: str, scenario,
                pair_vetoes: set[tuple[str, str]], variant: str = "diverse2") -> list[dict]:
    supported, allowed = _supported(pool, scenario)
    if method == "NoMem":
        return []
    if method == "FIFO":
        return supported[:2]
    if method == "Recency":
        return list(reversed(supported))[:2]
    semantic = sorted(supported, key=lambda note: (
        -note_relevance(note, task, allowed), note["arrival"], note["uid"]))
    if method == "Semantic":
        return semantic[:2]
    if method != "CREDIT":
        raise ValueError("unknown method: " + method)
    ranked = sorted(supported, key=lambda note: (
        -(note_relevance(note, task, allowed) +
          12.0 * float(note["source_upstream_score"])),
        -float(note["source_upstream_score"]), note["uid"]))
    if not ranked:
        return []
    if variant == "compact1":
        chosen = ranked[:1]
    elif variant == "quality2":
        chosen = ranked[:2]
    elif variant == "diverse2":
        chosen = [ranked[0]]
        covered = set(ranked[0]["tools"]) & allowed
        second = next((note for note in ranked[1:]
                       if (set(note["tools"]) & allowed) - covered), None)
        if second is None and len(ranked) > 1:
            second = ranked[1]
        if second is not None:
            chosen.append(second)
    else:
        raise ValueError("unknown CREDIT variant: " + variant)
    if len(chosen) == 2 and tuple(sorted(note["uid"] for note in chosen)) in pair_vetoes:
        chosen = chosen[:1]
    return chosen


def balanced_order(methods: list[str], actor_index: int, trajectory_index: int,
                   session_index: int) -> list[str]:
    shift = (actor_index + trajectory_index + session_index) % len(methods)
    return methods[shift:] + methods[:shift]


def run_actor(seed: int, actor_index: int, phase: str, trajectories: list[dict],
              methods: list[str], root: Path, args, scenarios, pool: list[dict],
              pair_vetoes: set[tuple[str, str]], variants: dict[str, str], u):
    directory = root / f"{phase}_seed_{seed}"; directory.mkdir()
    agent = RemoteJSONAgent(directory, seed, env=args.env, tokenizer_path=args.tokenizer,
        temperature=0.0, max_new_tokens=384, workers=args.workers,
        exact_prompt_cache=False)
    agent.binding_interface = "tool_json"; agent.manifest["benchmark"] = phase
    sessions, positions = [], defaultdict(lambda: [0] * len(methods))
    for trajectory_index, trajectory in enumerate(trajectories):
        available = leave_family_out(pool, trajectory["trajectory"]) \
            if phase == "stream_development" else pool
        for session_index, task in enumerate(trajectory["sessions"]):
            order = balanced_order(methods, actor_index, trajectory_index, session_index)
            jobs, metadata = [], {}
            for position, label in enumerate(order):
                positions[label][position] += 1
                selector = "CREDIT" if label.startswith("CREDIT-") else label
                variant = variants.get(label, "diverse2")
                bank = select_bank(selector, available, task, scenarios[task],
                                   pair_vetoes, variant)
                jobs.append(initial_job_v2(task, scenarios[task], bank, u,
                    agent.tokenizer, phase, label))
                metadata[label] = (bank, position, variant)
            for row in run_batch(jobs, agent, directory, u):
                bank, position, variant = metadata[row["method"]]
                item = {**row, "actor_seed": seed,
                    "trajectory": trajectory["trajectory"],
                    "session_index": session_index,
                    "selected_bank": [note["uid"] for note in bank],
                    "available_memory_count": len(available),
                    "remote_latency_ms": session_latency(row),
                    "batch_position": position,
                    "credit_variant": variant if row["method"].startswith("CREDIT") else None}
                sessions.append(item)
                with (directory / "sessions.jsonl").open("a") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    agent.close()
    summary = {"actor_seed": seed, "phase": phase, "sessions": len(sessions),
        "remote_requests": agent.calls, "method_batch_position_counts": dict(positions)}
    write(directory, "summary.json", summary)
    return sessions, summary


def summarize(rows: list[dict], methods: list[str], calibration_ms: float = 0.0) -> list[dict]:
    credit_n = sum(row["method"] == "CREDIT" for row in rows)
    amortized = calibration_ms / credit_n if credit_n else 0.0
    table = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        attributed = [row["remote_latency_ms"] +
                      (amortized if method == "CREDIT" else 0.0) for row in subset]
        entry = {"method": method, "sessions": len(subset),
            "strict_VSR": float(np.mean([row["success"] for row in subset])),
            "upstream_mean_score": float(np.mean([continuous_score(row) for row in subset])),
            "mean_tool_calls": float(np.mean([row["tool_calls"] for row in subset])),
            "mean_remote_requests": float(np.mean([len(row["generations"]) for row in subset])),
            "mean_online_remote_latency_ms": float(np.mean([row["remote_latency_ms"] for row in subset])),
            "mean_attributed_remote_latency_ms": float(np.mean(attributed)),
            "p95_attributed_remote_latency_ms": float(np.quantile(attributed, .95))}
        for budget in LATENCY_BUDGETS_MS:
            entry[f"VSR_at_{budget}ms"] = float(np.mean([
                row["success"] and latency <= budget for row, latency in zip(subset, attributed)]))
            entry[f"completion_at_{budget}ms"] = float(np.mean([
                latency <= budget for latency in attributed]))
        table.append(entry)
    return table


def choose_variant(rows: list[dict]):
    methods = ["CREDIT-" + variant for variant in DEV_VARIANTS]
    table = summarize(rows, methods)
    winner = max(table, key=lambda row: (row["VSR_at_10000ms"], row["strict_VSR"],
        row["upstream_mean_score"], -row["mean_online_remote_latency_ms"], row["method"]))
    return winner["method"].removeprefix("CREDIT-"), table


def paired_statistics(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for row in rows:
        by[row["method"]][(row["actor_seed"], row["trajectory"],
                           row["session_index"])] = row
    clusters = sorted({(row["actor_seed"], row["trajectory"]) for row in rows})
    rng = np.random.default_rng(20261014)
    draws = rng.integers(0, len(clusters), (10000, len(clusters)))
    report = []
    for method in CONFIRM_METHODS:
        effects, score_effects = [], []
        for seed, trajectory in clusters:
            keys = [(seed, trajectory, index) for index in (0, 1)]
            effects.append(float(np.mean([by["CREDIT"][key]["success"] -
                by[method][key]["success"] for key in keys])))
            score_effects.append(float(np.mean([continuous_score(by["CREDIT"][key]) -
                continuous_score(by[method][key]) for key in keys])))
        boot = np.mean(np.asarray(effects)[draws], axis=1)
        score_boot = np.mean(np.asarray(score_effects)[draws], axis=1)
        wins, losses = sum(x > 0 for x in effects), sum(x < 0 for x in effects)
        p = float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue) \
            if wins + losses else 1.0
        report.append({"baseline": method,
            "CREDIT_delta_VSR": float(np.mean(effects)),
            "VSR_cluster_bootstrap_CI95": np.quantile(boot, [.025, .975]).tolist(),
            "CREDIT_delta_upstream_score": float(np.mean(score_effects)),
            "score_cluster_bootstrap_CI95": np.quantile(score_boot, [.025, .975]).tolist(),
            "cluster_wins": wins, "cluster_losses": losses,
            "one_sided_exact_sign_p": p})
    return {"comparisons": report, "clusters": len(clusters),
        "resampling_unit": "actor seed by two-session trajectory",
        "bootstrap_replicates": 10000, "bootstrap_seed": 20261014}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/toolsandbox_m_followup_v1"))
    parser.add_argument("--factorial", type=Path,
                        default=Path("results/toolsandbox_m_factorial_training_v2"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/toolsandbox_m_streaming_optimization_v1"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--dev-seeds", type=int, nargs="+", default=[877, 881])
    parser.add_argument("--confirm-seeds", type=int, nargs="+", default=[883, 887, 907, 911, 919])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True); (args.output / "source").mkdir()
    old_protocol = json.loads((args.source / "frozen_protocol.json").read_text())
    args.tokenizer = args.tokenizer or Path(json.loads(
        (args.source / "training_pool_seed_701" / "agent_manifest.json").read_text())["tokenizer_path"])
    scenarios = dill.loads((args.source / "scenarios.pkl").read_bytes())
    development, confirmation = old_protocol["development"], old_protocol["confirmation"]
    if args.smoke:
        args.dev_seeds = args.dev_seeds[:1]; args.confirm_seeds = args.confirm_seeds[:1]
        development = development[:1]; confirmation = confirmation[:1]
    upstream = Path(old_protocol["upstream_directory"])
    u = upstream_imports(upstream); install_local_clock(old_protocol["clock"])
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    pool, added = build_pool(args.source, tokenizer)
    effects = json.loads((args.factorial / "effects.json").read_text())
    pair_vetoes = harmful_pairs(effects)
    names = ["toolsandbox_m_streaming_optimization.py",
             "toolsandbox_m_streaming_optimization_verify.py", "toolsandbox_m_followup.py",
             "toolsandbox_local_v3.py", "toolsandbox_m.py", "remote_json_agent.py"]
    hashes = {name: digest(Path(__file__).with_name(name)) for name in names}
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), args.output / "source" / name)
    protocol = {"suite": "ToolSandbox-M development-memory streaming optimization",
        "source_results": str(args.source.resolve()),
        "source_development_sha256": digest(args.source / "development_sessions.jsonl"),
        "old_confirmation_outcomes_used": False,
        "factorial_results": str(args.factorial.resolve()),
        "factorial_effects_sha256": digest(args.factorial / "effects.json"),
        "source_sha256": hashes, "upstream_directory": str(upstream),
        "upstream_source_sha256": old_protocol["upstream_source_sha256"],
        "clock": old_protocol["clock"], "development": development,
        "confirmation": confirmation, "development_seeds": args.dev_seeds,
        "confirmation_seeds": args.confirm_seeds,
        "development_variants": list(DEV_VARIANTS),
        "confirmation_methods": list(CONFIRM_METHODS),
        "memory_quality_floor": 0.75,
        "memory_deduplication": "exact recipe text; retain highest dev score",
        "development_memory_rule": "leave target trajectory family out",
        "pair_veto_rule": "drop second note when factorial interaction upper CI95 < -.005",
        "harmful_pair_uids": [list(pair) for pair in sorted(pair_vetoes)],
        "selection_rule": "10-s VSR, raw VSR, upstream score, latency",
        "paired_scheduling": "all methods for a task share one concurrent remote batch",
        "temperature": 0.0, "maximum_tool_calls_per_session": 8,
        "provider_credentials_persisted": False, "smoke": args.smoke}
    write(args.output, "frozen_protocol.json", protocol)
    (args.output / "scenarios.pkl").write_bytes((args.source / "scenarios.pkl").read_bytes())
    write(args.output, "memory_pool.json", pool); write(args.output, "new_memories.json", added)

    dev_methods = ["CREDIT-" + variant for variant in DEV_VARIANTS]
    variants = {method: method.removeprefix("CREDIT-") for method in dev_methods}
    dev_rows, dev_summaries = [], []
    for index, seed in enumerate(args.dev_seeds):
        found, summary = run_actor(seed, index, "stream_development", development,
            dev_methods, args.output, args, scenarios, pool, pair_vetoes, variants, u)
        dev_rows.extend(found); dev_summaries.append(summary)
    with (args.output / "development_sessions.jsonl").open("w") as stream:
        for row in dev_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    chosen, dev_table = choose_variant(dev_rows)
    write(args.output, "frozen_selector.json", {"chosen_variant": chosen,
        "development_table": dev_table, "harmful_pair_uids": protocol["harmful_pair_uids"],
        "confirmation_outcomes_used": False})

    confirm_rows, confirm_summaries = [], []
    for index, seed in enumerate(args.confirm_seeds):
        found, summary = run_actor(seed, index, "stream_confirmation", confirmation,
            list(CONFIRM_METHODS), args.output, args, scenarios, pool, pair_vetoes,
            {"CREDIT": chosen}, u)
        confirm_rows.extend(found); confirm_summaries.append(summary)
    with (args.output / "confirmation_sessions.jsonl").open("w") as stream:
        for row in confirm_rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    factor_ms = json.loads((args.factorial / "summary.json").read_text())[
        "calibration_remote_latency_ms"]
    dev_ms = sum(row["remote_latency_ms"] for row in dev_rows)
    training_ms = factor_ms + dev_ms
    table = summarize(confirm_rows, list(CONFIRM_METHODS), training_ms)
    ranking = sorted((dict(row) for row in table), key=lambda row: (
        -row["VSR_at_10000ms"], -row["strict_VSR"],
        row["mean_attributed_remote_latency_ms"], row["method"]))
    for index, row in enumerate(ranking, 1): row["rank"] = index
    summary = {"chosen_variant": chosen, "initial_memories": len(pool) - len(added),
        "new_development_memories": len(added), "expanded_memories": len(pool),
        "development_table": dev_table, "confirmation_table": table,
        "latency_ranking": ranking, "paired_statistics": paired_statistics(confirm_rows),
        "factorial_calibration_latency_ms": factor_ms,
        "variant_selection_latency_ms": dev_ms,
        "credit_training_latency_ms": training_ms,
        "development_actor_summaries": dev_summaries,
        "confirmation_actor_summaries": confirm_summaries}
    write(args.output, "summary.json", summary)
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    expected_dev = len(args.dev_seeds) * len(development) * 2 * len(dev_methods)
    expected_confirm = len(args.confirm_seeds) * len(confirmation) * 2 * len(CONFIRM_METHODS)
    audit = {"passed": len(dev_rows) == expected_dev and len(confirm_rows) == expected_confirm
        and all(row["verification_coverage"] for row in dev_rows + confirm_rows),
        "development_sessions": len(dev_rows), "confirmation_sessions": len(confirm_rows),
        "development_sha256": digest(args.output / "development_sessions.jsonl"),
        "confirmation_sha256": digest(args.output / "confirmation_sessions.jsonl")}
    write(args.output, "audit.json", audit)
    print(json.dumps({"selection": chosen, "summary": summary, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
