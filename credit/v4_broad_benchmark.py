"""Complete the controlled chronological benchmark table on one frozen test set.

The existing baseline confirmation contains seven baseline families and CREDIT
on eight seeds and three cycles.  This module adds Direct-Sandbox on those same
frozen tasks, then derives the five prespecified table metrics.  Direct uses
only physical sandbox outcomes for bank search.  Its drift evidence is restored
from the shared, pre-test adaptation snapshot; the CREDIT bank and held-out
labels are never inputs to Direct selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from .llm_policy import install_policy
from .v3.drift import Level, ProbeEvidence
from .v3.experiment import digest, write
from .v3.inference import worst_mixture
from .v3.local import ROLE_NAMES, factory, retriever, valid_request
from .v3.selection import bank_cost
from .v3_data import Corpus
from .v3_online import Store, StoredGate
from .v3_online_data import CycleCorpus
from .v3_optimization import select as direct_select
from .v3_recovery import RecoveryExecutor
from .v3_remote_recovery_full import RECOVERY_POOL, install_runtime, registry_payload


SEEDS = (64007, 64013, 64019, 64033, 64037, 64063, 64067, 64081)
CYCLES = 3
DEPTH = 4
BASE = Path("results/credit_v3_remote_recovery_baselines_confirm_v1")
DATASET = Path("datasets/credit_v3_remote_recovery_baselines_confirm_v1")
POLICY = Path("results/credit_v3_remote_llm_policy_v2")
METHOD_MAP = {
    "NoMem": "NoMem",
    "FIFO": "FIFO / Recency",
    "Physical-MCShapley": "MC-Shapley",
    "CMP-IPW-adapted": "CMP-adapted",
    "CAR-Memory-adapted": "CAR-adapted",
    "Probe-Filter": "Version-filter",
    "Direct-Sandbox": "Direct-Sandbox",
    "CREDIT-Covered": "CREDIT",
}


def load_gate(base: Path, seed: int, cycle: int) -> StoredGate:
    root = base / f"seed_{seed}/cycle_{cycle}/CREDIT-Covered/adaptation"
    evidence = [ProbeEvidence(**(row | {"level": Level(row["level"])}))
                for row in json.loads((root / "transfer_levels.json").read_text())]
    leases = json.loads((root / "leases.json").read_text())
    store = Store.restore(json.loads((root / "store_before_test.json").read_text()))
    return StoredGate(evidence, leases, store)


def exposure(context, bank, gate):
    pool = [memory for memory in RECOVERY_POOL if memory.uid in bank]
    admitted = gate.candidates(context, pool)
    return retriever(DEPTH)(context, tuple(memory.uid for memory in admitted))


def direct_common_cost(base: Path, seed: int, cycle: int) -> int:
    adaptation = json.loads((base / f"seed_{seed}/cycle_{cycle}/CREDIT-Covered/adaptation/costs.json").read_text())
    cost = int(adaptation["selection_api_calls_including_common"])
    if cycle == 0:
        source = json.loads((base / f"seed_{seed}/source_history/costs.json").read_text())
        # Direct needs randomized source labels, replay checks, and fixtures for
        # the shared drift gate, but does not train/calibrate the outcome model.
        cost += int(source["api_calls"]["factorial"])
        cost += int(source["api_calls"]["replay_audit"])
        cost += int(source["fixtures"])
    return cost


def protocol(base: Path, dataset: Path, policy: Path) -> dict:
    source = Path(__file__)
    return {
        "experiment": "credit_v4_broad_chronological_benchmark_v2",
        "seeds": list(SEEDS), "cycles": CYCLES,
        "direct_selection_cap_per_seed_cycle": "CREDIT attributed calls minus identical shared evidence calls in that seed-cycle",
        "dataset": str(dataset.resolve()),
        "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "baseline_confirmation": str(base.resolve()),
        "baseline_artifact_manifest_sha256": digest(base / "artifact_manifest.json"),
        "remote_policy": str(policy.resolve()),
        "remote_policy_scores_sha256": digest(policy / "decision_scores.json"),
        "registry": registry_payload(), "capacity": 80, "retrieval_depth": DEPTH,
        "direct_selector": "physical sandbox outcomes only; no outcome model; frozen finalists consume the residual matched allowance on complete CRN groups",
        "matched_validation_stream": "6,000 deterministic contexts per stratum and seed-cycle under a dedicated non-heldout phase; executed only until the matched call allowance cannot fit another complete group",
        "shared_gate": "pre-test transfer levels, leases, and store_before_test from the common adaptation stream",
        "cost_rule": "shared drift evidence plus physical Direct selection; held-out evaluation calls excluded for every method",
        "new_workflow_metric": "source-version held-out unseen templates/topologies",
        "drift_metric": "changed target-version held-out templates/topologies",
        "source_sha256": {str(source.relative_to(Path.cwd())): digest(source)},
        "heldout_labels_used_for_selection": False,
        "scope": "finite controlled API with cached remote actor; not natural provider drift",
    }


def credit_costs(base: Path) -> dict[tuple[int, int], int]:
    rows = json.loads((base / "costs.json").read_text())
    return {(row["seed"], row["cycle"]): int(row["attributed_api_calls"])
            for row in rows if row["method"] == "CREDIT-Covered"}


def frozen_candidates(directory: Path, selected: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    values = [tuple(selected), ()]
    for filename, key in (("halving.json", "finalists"), ("screening.json", "banks")):
        path = directory / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for bank in data.get(key, ()):
            candidate = tuple(bank)
            if bank_cost(candidate, RECOVERY_POOL) <= 80:
                values.append(candidate)
        if filename == "halving.json" and data.get(key):
            break
    return tuple(dict.fromkeys(values))[:3]


def complete_matched_budget(seed: int, cycle: int, gate, candidates,
                            initial: int, allowance: int, directory: Path):
    """Spend residual calls only on frozen finalists and complete CRN groups."""
    executor = RecoveryExecutor()
    strata = tuple(f"{family}/{epoch}" for family in
                   ("books", "travel", "inventory", "payments")
                   for epoch in ("source", "target"))
    outcomes = {bank: defaultdict(list) for bank in candidates}
    records, spent, groups = [], int(initial), 0
    make = factory(seed)
    phase = f"v4-direct-matched-validation-cycle-{cycle}"

    def upper(index):
        return sum(2 * len(make(stratum, phase, index).nodes)
                   for stratum in strata for _ in candidates)

    for index in range(6000):
        bound = upper(index)
        if spent + bound > allowance:
            break
        group = []
        for stratum in strata:
            context = make(stratum, phase, index)
            for bank in candidates:
                exposed = exposure(context, bank, gate)
                episode = executor.run(context, exposed)
                group.append({"phase": "matched-budget-validation", "bank": list(bank),
                              "episode": asdict(episode)})
        physical = sum(row["episode"]["calls"] for row in group)
        if spent + physical > allowance:
            raise ValueError("matched validation reservation failed")
        records.extend(group); spent += physical; groups += 1
        for row in group:
            episode = row["episode"]
            outcomes[tuple(row["bank"])][episode["context"]["stratum"]].append(
                int(episode["verified"] is True))
    scores = {}
    for bank in candidates:
        if not all(outcomes[bank][stratum] for stratum in strata):
            scores[bank] = -1.0 - .0001 * bank_cost(bank, RECOVERY_POOL)
        else:
            means = [float(np.mean(outcomes[bank][stratum])) for stratum in strata]
            scores[bank] = worst_mixture(means, [1 / 8] * 8, .2) - .0001 * bank_cost(bank, RECOVERY_POOL)
    chosen = max(candidates, key=lambda bank: (scores[bank], -bank_cost(bank, RECOVERY_POOL), bank))
    next_bound = upper(groups)
    with gzip.open(directory / "matched_validation.jsonl.gz", "wt", compresslevel=6) as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    write(directory / "matched_validation.json", {
        "candidate_banks_frozen_before_validation": [list(bank) for bank in candidates],
        "scores": {"|".join(bank): value for bank, value in scores.items()},
        "groups": groups, "initial_selection_calls": initial,
        "validation_calls": spent - initial, "actual_selection_calls": spent,
        "allowance": allowance, "unspent": allowance - spent,
        "next_complete_group_upper_bound": next_bound,
        "selected_bank": list(chosen), "heldout_labels_used": False,
    })
    return chosen, spent, next_bound


def run_direct(base: Path, dataset: Path, policy: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("use a fresh output directory")
    install_runtime(SEEDS)
    install_policy(policy / "decision_scores.json")
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    (output / "source").mkdir()
    frozen = protocol(base, dataset, policy)
    write(output / "frozen_protocol.json", frozen)
    source = Path(__file__)
    (output / "source" / source.name).write_bytes(source.read_bytes())
    rows, costs = [], []
    matched = credit_costs(base)
    for cycle in range(CYCLES):
        cycle_rows = []
        for seed in SEEDS:
            corpus = CycleCorpus(Corpus(dataset, seed), cycle)
            gate = load_gate(base, seed, cycle)
            directory = output / f"seed_{seed}/cycle_{cycle}/Direct-Sandbox"
            directory.parent.mkdir(parents=True, exist_ok=True)
            common_calls = direct_common_cost(base, seed, cycle)
            selection_allowance = matched[seed, cycle] - common_calls
            bank, report = direct_select("Direct-Sandbox", (), gate, corpus, 0,
                                         directory, depth=DEPTH,
                                         cap=selection_allowance, direct=True)
            initial_calls = int(report["selection_api_calls_including_common"])
            candidates = frozen_candidates(directory, bank)
            bank, selection_calls, next_bound = complete_matched_budget(
                seed, cycle, gate, candidates, initial_calls, selection_allowance, directory)
            write(directory / "final_selection.json", {
                "bank": list(bank), "bank_tokens": bank_cost(bank, RECOVERY_POOL),
                "selection_allowance": selection_allowance,
                "actual_selection_calls": selection_calls,
                "unspent": selection_allowance - selection_calls,
                "next_complete_group_upper_bound": next_bound,
                "heldout_labels_used": False,
            })
            costs.append({"seed": seed, "cycle": cycle, "method": "Direct-Sandbox",
                          "bank": list(bank), "bank_tokens": bank_cost(bank, RECOVERY_POOL),
                          "physical_selection_calls": selection_calls,
                          "shared_evidence_calls": common_calls,
                          "attributed_api_calls": selection_calls + common_calls,
                          "matched_credit_calls": matched[seed, cycle],
                          "selection_allowance": selection_allowance,
                          "unspent": selection_allowance - selection_calls,
                          "next_complete_group_upper_bound": next_bound})
            executor = RecoveryExecutor()
            for context in corpus.phase("heldout-test"):
                exposed = exposure(context, bank, gate)
                episode = executor.run(context, exposed)
                nodes = {node.uid: node for node in context.nodes}
                row = {"seed": seed, "cycle": cycle, "method": "Direct-Sandbox",
                       "task": context.uid, "family": context.stratum.split("/")[0],
                       "epoch": context.stratum.split("/")[1],
                       "success": episode.outcome, "api_calls": episode.calls,
                       "bank_tokens": bank_cost(bank, RECOVERY_POOL),
                       "exposure": list(exposed), "state_hash": episode.state_hash,
                       "verification_known": episode.verified is not None,
                       "contract_invalid_calls": sum(
                           not valid_request(ROLE_NAMES[nodes[event["node"]].role.key],
                                             event["state_before"], event["request_mode"],
                                             *context.stratum.split("/"))
                           for event in episode.events)}
                rows.append(row); cycle_rows.append(row)
        write(output / f"raw_cycle_{cycle}.json", cycle_rows)
        print(json.dumps({"cycle": cycle, "direct_heldout_rows": len(cycle_rows)}), flush=True)
    write(output / "direct_test_results.json", rows)
    write(output / "direct_costs.json", costs)
    write(output / "direct_summary.json", {
        "seeds": list(SEEDS), "cycles": CYCLES, "test_episodes": len(rows),
        "test_api_calls": sum(row["api_calls"] for row in rows),
        "selection_api_calls": sum(row["physical_selection_calls"] for row in costs),
        "shared_evidence_calls": sum(row["shared_evidence_calls"] for row in costs),
        "attributed_extra_calls": sum(row["attributed_api_calls"] for row in costs),
        "vsr": float(np.mean([row["success"] for row in rows])),
    })
    write(output / "artifact_manifest.json", {
        str(path.relative_to(output)): digest(path) for path in output.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "execution_audit.json",
                                                "table.json", "table.csv", "analysis_protocol.json"}
    })


def stale(memory: str, family: str, epoch: str) -> bool:
    """Independent public-contract invalidation used for the table metric."""
    if memory == "conflict":
        return True
    patched_submit = epoch == "target" and family in {"books", "inventory"}
    if memory == "legacy_contract":
        return patched_submit
    if memory == "new_contract":
        return not patched_submit
    return False


def clustered_interval(rows: list[dict], field, seed: int) -> tuple[float, float]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["seed"], row["family"]].append(float(field(row)))
    means = np.asarray([np.mean(value) for value in groups.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(10_000, len(means)))
    values = means[draws].mean(axis=1)
    return tuple(float(x) for x in np.quantile(values, [.025, .975]))


def analyse(base: Path, output: Path) -> None:
    baseline_rows = json.loads((base / "raw_test_results.json").read_text())
    direct_rows = json.loads((output / "direct_test_results.json").read_text())
    rows = baseline_rows + direct_rows
    baseline_costs = json.loads((base / "costs.json").read_text())
    direct_costs = json.loads((output / "direct_costs.json").read_text())
    calls = defaultdict(int)
    for row in baseline_costs + direct_costs:
        calls[row["method"]] += int(row["attributed_api_calls"])
    table = []
    selected = list(METHOD_MAP)
    for index, raw_method in enumerate(selected):
        subset = [row for row in rows if row["method"] == raw_method]
        if not subset:
            raise ValueError("missing method rows: " + raw_method)
        source = [row for row in subset if row["epoch"] == "source"]
        target = [row for row in subset if row["epoch"] == "target"]
        exposures = [(memory, row["family"], row["epoch"])
                     for row in subset for memory in row["exposure"]]
        stale_count = sum(stale(*item) for item in exposures)
        all_ci = clustered_interval(subset, lambda row: row["success"], 4100 + index)
        source_ci = clustered_interval(source, lambda row: row["success"], 5100 + index)
        target_ci = clustered_interval(target, lambda row: row["success"], 6100 + index)
        table.append({
            "method": METHOD_MAP[raw_method], "raw_method": raw_method,
            "episodes": len(subset), "all_vsr": float(np.mean([r["success"] for r in subset])),
            "all_vsr_ci": list(all_ci),
            "new_workflow_vsr": float(np.mean([r["success"] for r in source])),
            "new_workflow_vsr_ci": list(source_ci),
            "drift_vsr": float(np.mean([r["success"] for r in target])),
            "drift_vsr_ci": list(target_ci),
            "stale_exposures": stale_count, "total_exposures": len(exposures),
            "stale_exposure_rate": stale_count / len(exposures) if exposures else 0.0,
            "extra_calls": calls[raw_method],
            "extra_calls_per_test_episode": calls[raw_method] / len(subset),
        })
    if not all(row["verification_known"] for row in rows):
        raise ValueError("unknown verifier outcome in broad benchmark")
    fifo = next(row for row in baseline_rows if row["method"] == "FIFO")
    # Full row equality is checked by keyed outcomes below; this small lookup
    # keeps the method merge explicit in the artifact.
    by_method = defaultdict(dict)
    for row in baseline_rows:
        by_method[row["method"]][row["seed"], row["cycle"], row["task"]] = row
    for key, row in by_method["FIFO"].items():
        other = by_method["Recency"][key]
        if row["success"] != other["success"] or row["exposure"] != other["exposure"]:
            raise ValueError("FIFO and Recency cannot be merged")
    analysis_protocol = {
        "rows": len(rows), "methods": [row["method"] for row in table],
        "metric_definitions": {
            "all_vsr": "all frozen held-out chronological episodes",
            "new_workflow_vsr": "source-version unseen templates/topologies",
            "drift_vsr": "changed target-version unseen templates/topologies",
            "stale_exposure_rate": "invalidated memory records divided by exposed records; NoMem is zero by empty-exposure convention",
            "extra_calls": "all attributed pre-test evidence and selection calls divided by held-out test episodes",
        },
        "stale_rule": {"conflict": "always obsolete by construction",
                       "legacy_contract": "stale on patched books/inventory target submit",
                       "new_contract": "stale outside patched books/inventory target submit"},
        "interval": "10,000-replicate seed-by-family cluster bootstrap",
        "fifo_recency_merge": "exactly identical exposures and outcomes on all frozen held-out rows",
        "unused_probe": fifo["task"],
    }
    write(output / "analysis_protocol.json", analysis_protocol)
    write(output / "table.json", table)
    with (output / "table.csv").open("w") as stream:
        stream.write("method,all_vsr,new_workflow_vsr,drift_vsr,stale_exposure_rate,extra_calls_per_test_episode\n")
        for row in table:
            stream.write(f'{row["method"]},{row["all_vsr"]:.8f},{row["new_workflow_vsr"]:.8f},'
                         f'{row["drift_vsr"]:.8f},{row["stale_exposure_rate"]:.8f},'
                         f'{row["extra_calls_per_test_episode"]:.8f}\n')


def audit(base: Path, dataset: Path, policy: Path, output: Path) -> None:
    expected = protocol(base, dataset, policy)
    if json.loads((output / "frozen_protocol.json").read_text()) != expected:
        raise ValueError("frozen protocol differs")
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for name, value in manifest.items():
        if digest(output / name) != value:
            raise ValueError("artifact differs: " + name)
    if digest(dataset / "manifest.json") != expected["dataset_manifest_sha256"]:
        raise ValueError("dataset differs")
    install_runtime(SEEDS)
    install_policy(policy / "decision_scores.json")
    indexed = {(row["seed"], row["cycle"], row["task"]): row
               for row in json.loads((output / "direct_test_results.json").read_text())}
    costs = {(row["seed"], row["cycle"]): row
             for row in json.loads((output / "direct_costs.json").read_text())}
    replayed = calls = 0
    for cycle in range(CYCLES):
        for seed in SEEDS:
            directory = output / f"seed_{seed}/cycle_{cycle}/Direct-Sandbox"
            decision = json.loads((directory / "final_selection.json").read_text())
            cost = json.loads((directory / "costs.json").read_text())
            bank = tuple(decision["bank"])
            validation = json.loads((directory / "matched_validation.json").read_text())
            if (bank_cost(bank, RECOVERY_POOL) > 80
                    or validation["heldout_labels_used"]
                    or decision["heldout_labels_used"]
                    or validation["initial_selection_calls"] != cost["selection_api_calls_including_common"]
                    or validation["actual_selection_calls"] != decision["actual_selection_calls"]
                    or costs[seed, cycle]["physical_selection_calls"]
                    != decision["actual_selection_calls"]
                    or costs[seed, cycle]["attributed_api_calls"]
                    > costs[seed, cycle]["matched_credit_calls"]
                    or not 0 <= costs[seed, cycle]["unspent"]
                    < costs[seed, cycle]["next_complete_group_upper_bound"]):
                raise ValueError("selection budget/capacity differs")
            with gzip.open(directory / "selection.jsonl.gz", "rt") as stream:
                selection = [json.loads(line) for line in stream]
            with gzip.open(directory / "matched_validation.jsonl.gz", "rt") as stream:
                validation_rows = [json.loads(line) for line in stream]
            if (sum(row["episode"]["calls"] for row in validation_rows)
                    != validation["validation_calls"]):
                raise ValueError("matched validation ledger differs")
            if any("heldout-test" in row["episode"]["context"]["uid"]
                   for row in selection + validation_rows):
                raise ValueError("held-out label used in Direct selection")
            gate = load_gate(base, seed, cycle)
            corpus = CycleCorpus(Corpus(dataset, seed), cycle)
            executor = RecoveryExecutor()
            for context in corpus.phase("heldout-test"):
                actual = indexed[seed, cycle, context.uid]
                exposed = exposure(context, bank, gate)
                episode = executor.run(context, exposed)
                if (actual["success"] != episode.outcome
                        or actual["api_calls"] != episode.calls
                        or actual["exposure"] != list(exposed)
                        or actual["state_hash"] != episode.state_hash):
                    raise ValueError("Direct held-out replay differs")
                replayed += 1; calls += episode.calls
    analyse(base, output)
    report = {"verified": True, "direct_heldout_episodes_replayed": replayed,
              "direct_heldout_api_calls_replayed": calls,
              "methods_in_table": len(METHOD_MAP), "table_cells": len(METHOD_MAP) * 5,
              "heldout_labels_used_for_selection": False,
              "maximum_total_call_gap": max(row["unspent"] for row in costs.values()),
              "auditor_source_sha256": digest(__file__),
              "limits": expected["scope"]}
    write(output / "execution_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASE)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--output", type=Path,
                        default=Path("results/credit_v4_broad_benchmark_v2"))
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        audit(args.baseline, args.dataset, args.policy, args.output)
    else:
        run_direct(args.baseline, args.dataset, args.policy, args.output)
        analyse(args.baseline, args.output)
