"""Equal-actual-selection-call curve for CREDIT and direct bank search.

Common source and target evidence is collected once per seed-cycle and reported
separately.  Each arm receives the same physical selection-call allowance.  Any
remaining allowance evaluates its frozen finalists on complete eight-stratum
CRN groups; the final bank is chosen only from those pre-heldout evaluations.
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
from .paired_statistics import summarize
from .v3.experiment import digest, write
from .v3.inference import worst_mixture
from .v3.local import ROLE_NAMES, factory, retriever, stable_seed, valid_request
from .v3.selection import bank_cost
from .v3_data import Corpus
from .v3_online_data import CycleCorpus
from .v3_optimization import select as optimizer_select
from .v3_recovery import RecoveryExecutor
from .v3_remote_llm_experiment import policy_guard
from .v3_remote_recovery_full import (RECOVERY_POOL, install_runtime,
                                      registry_payload)


DEV_SEEDS = (65003,)
CONFIRM_SEEDS = (66007, 66017, 66029, 66037, 66041, 66047, 66067, 66071)
CYCLES = 3
BUDGETS = (50_000, 75_000, 100_000, 125_000)
PRIMARY_BUDGET = 75_000
CAPACITY = 80
DEPTH = 4
BUDGET_CONTEXTS_PER_STRATUM = 2_500
STRATA = tuple(f"{family}/{epoch}"
               for family in ("books", "travel", "inventory", "payments")
               for epoch in ("source", "target"))


class PassGate:
    def candidates(self, context, memories, *, directly_validated=frozenset()):
        if directly_validated:
            raise ValueError("PassGate has no direct override")
        return tuple(memories)


def seeds_for(split):
    return DEV_SEEDS if split == "development" else CONFIRM_SEEDS


def generate_dataset(dataset, split):
    seeds = seeds_for(split)
    _, _, data = install_runtime(seeds)
    data.SEEDS = tuple(seeds)
    data.CYCLES = CYCLES
    data.generate(dataset)
    added = 0
    phase_additions = defaultdict(int)
    for seed in seeds:
        make = factory(seed)
        with gzip.open(dataset / f"seed_{seed}.jsonl.gz", "at", compresslevel=6) as stream:
            for cycle in range(CYCLES):
                phase = f"cycle-{cycle}/budget-evaluation"
                for stratum in STRATA:
                    for index in range(BUDGET_CONTEXTS_PER_STRATUM):
                        context = make(stratum, phase, index)
                        stream.write(json.dumps({"phase": phase, "context": asdict(context)},
                                                separators=(",", ":")) + "\n")
                        added += 1
                        phase_additions[phase] += 1
                # The chronological dataset has pilot indices 0..15.  The
                # common optimizer uses 100 for model-guided selection.
                pilot_phase = f"cycle-{cycle}/pilot"
                for stratum in STRATA:
                    for index in range(16, 100):
                        context = make(stratum, pilot_phase, index)
                        stream.write(json.dumps({"phase": pilot_phase, "context": asdict(context)},
                                                separators=(",", ":")) + "\n")
                        added += 1
                        phase_additions[pilot_phase] += 1
    summary = json.loads((dataset / "summary.json").read_text())
    summary["contexts"] += added
    summary["equal_call_contexts"] = added
    for phase, count in phase_additions.items():
        summary["by_phase"][phase] = summary["by_phase"].get(phase, 0) + count
    write(dataset / "summary.json", summary)
    write(dataset / "equal_call_dataset_protocol.json", {
        "split": split, "seeds": list(seeds), "cycles": CYCLES,
        "budgets": list(BUDGETS), "primary_budget": PRIMARY_BUDGET,
        "budget_contexts_per_stratum_cycle": BUDGET_CONTEXTS_PER_STRATUM,
        "pilot_contexts_per_stratum_cycle": 100,
        "outcomes_preassigned": False, "synthetic": True,
        "registry": registry_payload(),
        "generator_sha256": digest(__file__),
        "recovery_executor_sha256": digest("credit/v3_recovery.py"),
    })
    write(dataset / "manifest.json", {path.name: digest(path) for path in dataset.iterdir()
                                      if path.is_file() and path.name != "manifest.json"})


def frozen_protocol(policy, dataset, split):
    actor, _ = policy_guard(policy)
    sources = (
        "credit/v3_remote_recovery_equal_call.py",
        "credit/v3_remote_recovery_full.py",
        "credit/v3_recovery.py",
        "credit/v3_online_experiment.py",
        "credit/v3_online.py",
        "credit/v3_online_fast.py",
        "credit/v3_optimization.py",
        "credit/v3_revalidation.py",
        "credit/llm_policy.py",
        "credit/stateful.py",
        "credit/paired_statistics.py",
    )
    return {
        "experiment": "credit_v3_remote_recovery_equal_actual_selection_calls_v1",
        "split": split, "seeds": list(seeds_for(split)), "cycles": CYCLES,
        "budgets": list(BUDGETS), "primary_budget": PRIMARY_BUDGET,
        "budget_scope": "physical bank-selection calls after common evidence acquisition",
        "common_evidence": "one source history and one covered target-adaptation stream shared by all arms; costs reported but excluded identically",
        "initial_incumbent": "empty bank independently in each seed-cycle-budget arm",
        "candidate_search": "model-ranked CREDIT versus physical direct search; both use the same optimizer and recovery executor",
        "budget_completion": "remaining calls evaluate frozen finalists in complete eight-stratum CRN groups; final empirical robust choice uses no heldout labels",
        "capacity": CAPACITY, "retrieval_depth": DEPTH,
        "dataset": str(dataset.resolve()),
        "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "remote_policy": str(policy.resolve()),
        "actor_scores_sha256": actor["decision_scores_sha256"],
        "registry": registry_payload(),
        "source_sha256": {name: digest(name) for name in sources},
        "paper_sha256": digest("CREDIT_WWW2027_v3.tex"),
        "agents_sha256": digest("AGENTS.md"),
        "development_seed": list(DEV_SEEDS),
        "confirmation_seeds_frozen_before_development_results": list(CONFIRM_SEEDS),
        "primary": "CREDIT versus Direct at 75000 actual selection calls",
        "secondary": "descriptive quality-call curve at 50000,100000,125000",
        "scope": "finite synthetic API and cached remote actor; shared evidence cost is outside the equal-call x-axis",
    }


def bank_exposure(context, bank, gate):
    candidates = gate.candidates(context, [memory for memory in RECOVERY_POOL
                                            if memory.uid in bank])
    return retriever(DEPTH)(context, tuple(memory.uid for memory in candidates))


def frozen_candidates(directory, selected):
    values = [tuple(selected), ()]
    for filename, key in (("halving.json", "finalists"),
                          ("screening.json", "banks")):
        path = directory / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for bank in data.get(key, ()):
            bank = tuple(bank)
            if bank_cost(bank, RECOVERY_POOL) <= CAPACITY:
                values.append(bank)
        if filename == "halving.json" and data.get(key):
            break
    return tuple(dict.fromkeys(values))[:3]


def complete_budget(corpus, gate, candidates, initial_spent, budget, directory):
    """Use only whole eight-stratum, all-candidate validation groups."""
    executor = RecoveryExecutor()
    outcomes = {bank: defaultdict(list) for bank in candidates}
    records, spent, groups = [], int(initial_spent), 0
    # Recovery can add at most four calls per public node execution.  The
    # doubled bound mirrors the frozen replay reservation and is conservative.
    def upper(index):
        return sum(2 * (len(corpus.make(stratum, "budget-evaluation", index).nodes) + 4)
                   for stratum in STRATA for _ in candidates)
    for index in range(BUDGET_CONTEXTS_PER_STRATUM):
        bound = upper(index)
        if spent + bound > budget:
            break
        group = []
        for stratum in STRATA:
            context = corpus.make(stratum, "budget-evaluation", index)
            for bank in candidates:
                exposure = bank_exposure(context, bank, gate)
                episode = executor.run(context, exposure)
                group.append({"phase": "budget-evaluation", "bank": list(bank),
                              "episode": asdict(episode)})
        calls = sum(row["episode"]["calls"] for row in group)
        if spent + calls > budget:
            raise ValueError("conservative group reservation failed")
        records.extend(group)
        spent += calls
        groups += 1
        for row in group:
            episode = row["episode"]
            outcomes[tuple(row["bank"])][episode["context"]["stratum"]].append(
                int(episode["verified"] is True))
    scores = {}
    for bank in candidates:
        if not all(outcomes[bank][stratum] for stratum in STRATA):
            scores[bank] = -1.0 - .0001 * bank_cost(bank, RECOVERY_POOL)
            continue
        means = [float(np.mean(outcomes[bank][stratum])) for stratum in STRATA]
        scores[bank] = worst_mixture(means, [1 / len(STRATA)] * len(STRATA), .2) - .0001 * bank_cost(bank, RECOVERY_POOL)
    chosen = max(candidates, key=lambda bank: (scores[bank], -bank_cost(bank, RECOVERY_POOL), bank))
    remaining_bound = upper(groups) if groups < BUDGET_CONTEXTS_PER_STRATUM else None
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / "budget_validation.jsonl.gz", "wt", compresslevel=6) as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    write(directory / "budget_validation.json", {
        "candidate_banks": [list(bank) for bank in candidates],
        "groups": groups, "strata_per_group": len(STRATA),
        "scores": {"|".join(bank): value for bank, value in scores.items()},
        "selected_bank": list(chosen), "initial_selection_calls": initial_spent,
        "validation_calls": spent - initial_spent, "actual_selection_calls": spent,
        "allowance": budget, "unspent": budget - spent,
        "next_complete_group_upper_bound": remaining_bound,
        "heldout_labels_used": False,
    })
    return chosen, spent, remaining_bound or 0, len(records)


def budget_select(models, gate, corpus, method, budget, directory):
    from . import v3_optimization as optimization
    optimization.memories = lambda: RECOVERY_POOL
    optimization.StatefulExecutor = RecoveryExecutor
    direct = method == "Direct"
    selected, costs = optimizer_select(
        f"{method}-B{budget}", () if direct else models, gate, corpus, 0,
        directory, depth=DEPTH, cap=budget, direct=direct)
    initial = int(costs["selection_api_calls_including_common"])
    candidates = frozen_candidates(directory, selected)
    bank, spent, group_bound, validation_episodes = complete_budget(
        corpus, gate, candidates, initial, budget, directory)
    write(directory / "final_selection.json", {
        "method": method, "budget": budget, "bank": list(bank),
        "bank_tokens": bank_cost(bank, RECOVERY_POOL),
        "candidate_banks_frozen_before_budget_validation": [list(value) for value in candidates],
        "actual_selection_calls": spent, "unspent": budget - spent,
        "next_complete_group_upper_bound": group_bound,
        "validation_episodes": validation_episodes,
        "selection_uses_heldout_labels": False,
    })
    return bank, spent, group_bound


def score(prepared, output, cycle):
    rows = []
    executor = RecoveryExecutor()
    with gzip.open(output / f"test_cycle_{cycle}.jsonl.gz", "wt", compresslevel=6) as stream:
        for seed, corpus, policies in prepared:
            for method, budget, bank, gate in policies:
                label = f"{method}-B{budget}"
                for context in corpus.phase("heldout-test"):
                    exposure = bank_exposure(context, bank, gate)
                    episode = executor.run(context, exposure)
                    nodes = {node.uid: node for node in context.nodes}
                    row = {
                        "seed": seed, "cycle": cycle, "method": label,
                        "base_method": method, "budget": budget,
                        "task": context.uid, "family": context.stratum.split("/")[0],
                        "epoch": context.stratum.split("/")[1],
                        "success": episode.outcome, "api_calls": episode.calls,
                        "bank_tokens": bank_cost(bank, RECOVERY_POOL),
                        "exposure": list(exposure), "state_hash": episode.state_hash,
                        "contract_invalid_calls": sum(
                            not valid_request(ROLE_NAMES[nodes[event["node"]].role.key],
                                              event["state_before"], event["request_mode"],
                                              *context.stratum.split("/"))
                            for event in episode.events),
                    }
                    rows.append(row)
                    stream.write(json.dumps(row | {"episode": asdict(episode)},
                                            separators=(",", ":")) + "\n")
            print("scored equal-call block", cycle, seed, flush=True)
    return rows


def run(policy, dataset, output, split):
    if output.exists():
        raise FileExistsError("fresh equal-call output required")
    seeds = seeds_for(split)
    experiment, _, _ = install_runtime(seeds)
    experiment.SPECS = (("CREDIT-Covered", "covered"),)
    install_policy(policy / "decision_scores.json")
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    (output / "source").mkdir()
    protocol = frozen_protocol(policy, dataset, split)
    write(output / "frozen_protocol.json", protocol)
    for name in protocol["source_sha256"]:
        path = Path(name)
        (output / "source" / path.name).write_bytes(path.read_bytes())

    inputs = {}
    for seed in seeds:
        corpus = Corpus(dataset, seed)
        history, states = experiment.initial(seed, corpus, output / f"seed_{seed}")
        inputs[seed] = corpus, history, states["CREDIT-Covered"]

    all_rows, all_costs = [], []
    for cycle in range(CYCLES):
        prepared = []
        for seed in seeds:
            base, history, state = inputs[seed]
            corpus = CycleCorpus(base, cycle)
            common = output / f"seed_{seed}/cycle_{cycle}/common"
            common.mkdir(parents=True)
            if cycle:
                state["models"] = experiment.fit_ensemble(
                    history[0] + state["episodes"], history[1], history[2],
                    stable_seed(seed, cycle, "equal-call-refit") % 2**31,
                    common / "models", logged=())
            gate, adaptation_calls = experiment.adaptation(
                state, corpus, seed, cycle, common / "adaptation")
            source_calls = state["source_fee"] if cycle == 0 else 0
            write(common / "costs.json", {
                "adaptation_calls": adaptation_calls,
                "source_calls": source_calls,
                "common_evidence_calls_excluded_identically": adaptation_calls + source_calls,
            })
            policies = []
            for budget in BUDGETS:
                for method in ("CREDIT", "Direct"):
                    directory = output / f"seed_{seed}/cycle_{cycle}/{method}-B{budget}"
                    bank, spent, bound = budget_select(
                        state["models"], gate, corpus, method, budget, directory)
                    policies.append((method, budget, bank, gate))
                    all_costs.append({"seed": seed, "cycle": cycle,
                                      "method": method, "budget": budget,
                                      "actual_selection_calls": spent,
                                      "unspent": budget - spent,
                                      "next_complete_group_upper_bound": bound,
                                      "bank": list(bank),
                                      "bank_tokens": bank_cost(bank, RECOVERY_POOL),
                                      "common_evidence_calls_excluded": adaptation_calls + source_calls})
            prepared.append((seed, corpus, policies))
        frozen = {str(path.relative_to(output)): digest(path) for path in output.rglob("*")
                  if path.is_file() and f"cycle_{cycle}/" in str(path)
                  and path.name in ("final_selection.json", "certification_freeze.json",
                                    "budget_validation.json")}
        write(output / f"evaluation_freeze_{cycle}.json", frozen)
        rows = score(prepared, output, cycle)
        all_rows.extend(rows)
        write(output / f"raw_cycle_{cycle}.json", rows)
        write(output / f"costs_cycle_{cycle}.json",
              [row for row in all_costs if row["cycle"] == cycle])

    write(output / "raw_test_results.json", all_rows)
    write(output / "costs.json", all_costs)
    statistics = {}
    for budget in BUDGETS:
        subset = [row for row in all_rows if row["budget"] == budget]
        main, paired = summarize(subset, f"CREDIT-B{budget}", repeats=4000,
                                 random_seed=11173 + budget)
        statistics[str(budget)] = {"main": main, "paired": paired}
    write(output / "statistics.json", statistics)
    write(output / "summary.json", {
        "split": split, "seeds": list(seeds), "cycles": CYCLES,
        "budgets": list(BUDGETS), "primary_budget": PRIMARY_BUDGET,
        "test_episodes": len(all_rows),
        "test_api_calls": sum(row["api_calls"] for row in all_rows),
        "equal_actual_selection_call_curve_complete": True,
        "paper_experiments_complete": False,
    })
    write(output / "artifact_manifest.json", {
        str(path.relative_to(output)): digest(path) for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"})
    print("completed equal-call curve", split, flush=True)


def audit(policy, dataset, output, split):
    seeds = seeds_for(split)
    install_runtime(seeds)
    install_policy(policy / "decision_scores.json")
    protocol = frozen_protocol(policy, dataset, split)
    if json.loads((output / "frozen_protocol.json").read_text()) != protocol:
        raise ValueError("frozen protocol differs")
    for name, expected in json.loads((output / "artifact_manifest.json").read_text()).items():
        if digest(output / name) != expected:
            raise ValueError("artifact changed: " + name)
    for name, expected in protocol["source_sha256"].items():
        if digest(name) != expected or digest(output / "source" / Path(name).name) != expected:
            raise ValueError("source differs: " + name)
    if digest(dataset / "manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("dataset differs")

    policies, max_pair_gap = {}, 0
    costs = json.loads((output / "costs.json").read_text())
    cost_index = {(row["seed"], row["cycle"], row["method"], row["budget"]): row
                  for row in costs}
    for seed in seeds:
        for cycle in range(CYCLES):
            for budget in BUDGETS:
                pair = []
                for method in ("CREDIT", "Direct"):
                    directory = output / f"seed_{seed}/cycle_{cycle}/{method}-B{budget}"
                    final = json.loads((directory / "final_selection.json").read_text())
                    validation = json.loads((directory / "budget_validation.json").read_text())
                    row = cost_index[seed, cycle, method, budget]
                    bank = tuple(final["bank"])
                    with gzip.open(directory / "selection.jsonl.gz", "rt") as stream:
                        selection_records = [json.loads(line) for line in stream]
                    with gzip.open(directory / "budget_validation.jsonl.gz", "rt") as stream:
                        validation_records = [json.loads(line) for line in stream]
                    initial_calls = sum(record["episode"]["calls"]
                                        for record in selection_records)
                    validation_calls = sum(record["episode"]["calls"]
                                           for record in validation_records)
                    if (final["selection_uses_heldout_labels"]
                            or validation["heldout_labels_used"]
                            or bank_cost(bank, RECOVERY_POOL) > CAPACITY
                            or final["actual_selection_calls"] != row["actual_selection_calls"]
                            or initial_calls != validation["initial_selection_calls"]
                            or validation_calls != validation["validation_calls"]
                            or len(validation_records) != final["validation_episodes"]
                            or initial_calls + validation_calls != row["actual_selection_calls"]
                            or any("heldout-test" in record["episode"]["context"]["uid"]
                                   for record in selection_records + validation_records)
                            or not 0 <= row["unspent"] < max(1, row["next_complete_group_upper_bound"])):
                        raise ValueError("budget/capacity/freeze invariant differs")
                    if "heldout-test" in (directory / "final_selection.json").read_text():
                        raise ValueError("heldout identifier in selection")
                    pair.append(row["actual_selection_calls"])
                    policies[seed, cycle, method, budget] = bank
                max_pair_gap = max(max_pair_gap, abs(pair[0] - pair[1]))
                bound = max(cost_index[seed, cycle, method, budget]["next_complete_group_upper_bound"]
                            for method in ("CREDIT", "Direct"))
                if abs(pair[0] - pair[1]) >= max(1, bound):
                    raise ValueError("actual call difference exceeds one complete group")

    executor = RecoveryExecutor()
    all_rows, checked, calls = [], 0, 0
    for cycle in range(CYCLES):
        indexed = {(row["seed"], row["method"], row["task"]): row
                   for row in json.loads((output / f"raw_cycle_{cycle}.json").read_text())}
        with gzip.open(output / f"test_cycle_{cycle}.jsonl.gz", "rt") as stream:
            traces = [json.loads(line) for line in stream]
        trace_index = {(row["seed"], row["method"], row["task"]): row for row in traces}
        expected = set()
        for seed in seeds:
            corpus = CycleCorpus(Corpus(dataset, seed), cycle)
            # Finalists were evaluated under the common covered gate. Rebuild it
            # from its frozen pre-test adaptation artifact.
            from .v3.drift import Level, ProbeEvidence
            from .v3_online import Store, StoredGate
            common = output / f"seed_{seed}/cycle_{cycle}/common/adaptation"
            evidence = [ProbeEvidence(**(row | {"level": Level(row["level"])}))
                        for row in json.loads((common / "transfer_levels.json").read_text())]
            leases = json.loads((common / "leases.json").read_text())
            store = Store.restore(json.loads((common / "store_before_test.json").read_text()))
            gate = StoredGate(evidence, leases, store)
            for budget in BUDGETS:
                for method in ("CREDIT", "Direct"):
                    label = f"{method}-B{budget}"
                    bank = policies[seed, cycle, method, budget]
                    for context in corpus.phase("heldout-test"):
                        key = seed, label, context.uid
                        expected.add(key)
                        row, trace = indexed[key], trace_index[key]
                        exposure = bank_exposure(context, bank, gate)
                        episode = executor.run(context, exposure)
                        if json.loads(json.dumps(asdict(episode))) != trace["episode"]:
                            raise ValueError("heldout episode differs")
                        if ({k: v for k, v in trace.items() if k != "episode"} != row
                                or row["success"] != episode.outcome
                                or row["api_calls"] != episode.calls
                                or row["exposure"] != list(exposure)):
                            raise ValueError("heldout row differs")
                        checked += 1
                        calls += episode.calls
        if set(indexed) != expected or set(trace_index) != expected:
            raise ValueError("heldout coverage differs")
        all_rows.extend(indexed.values())
    if all_rows != json.loads((output / "raw_test_results.json").read_text()):
        raise ValueError("combined result differs")
    statistics = {}
    for budget in BUDGETS:
        subset = [row for row in all_rows if row["budget"] == budget]
        main, paired = summarize(subset, f"CREDIT-B{budget}", repeats=4000,
                                 random_seed=11173 + budget)
        statistics[str(budget)] = {"main": main, "paired": paired}
    if statistics != json.loads((output / "statistics.json").read_text()):
        raise ValueError("statistics differ")
    report = {
        "verified": True, "split": split, "seeds": list(seeds),
        "heldout_episodes_reexecuted": checked,
        "heldout_api_calls_reexecuted": calls,
        "maximum_actual_call_gap": max_pair_gap,
        "artifact_manifest_sha256": digest(output / "artifact_manifest.json"),
        "auditor_sha256": digest(__file__),
        "checks": ["source/data/policy binding", "capacity", "heldout isolation",
                   "physical selection and validation call-ledger sums", "budget bound",
                   "actual call gap below one complete group",
                   "complete heldout reconstruction", "paired statistics"],
        "limits": protocol["scope"],
    }
    write(output / "execution_audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path,
                        default=Path("results/credit_v3_remote_llm_policy_v2"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--generate-dataset", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.generate_dataset:
        generate_dataset(args.dataset, args.split)
    elif args.audit:
        audit(args.policy, args.dataset, args.output, args.split)
    else:
        run(args.policy, args.dataset, args.output, args.split)
