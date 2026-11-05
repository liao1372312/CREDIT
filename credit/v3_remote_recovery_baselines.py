"""Fresh same-protocol baselines for the versioned recovery experiment.

The suite keeps the previously reported recovery run immutable.  Every
selection method freezes its bank before the chronological held-out block and
records physical calls separately from calls attributed to shared evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from .llm_policy import install_policy
from .paired_statistics import summarize
from .stateful import RECENCY
from .v3.audit import load_context
from .v3.drift import Level
from .v3.experiment import digest, write
from .v3.local import ROLE_NAMES, ranked, retriever, stable_seed, valid_request
from .v3.selection import bank_cost
from .v3_benchmark import CallLedger, stream_records
from .v3_data import Corpus
from .v3_online import balanced_designs
from .v3_online_data import CycleCorpus
from .v3_recovery import RecoveryExecutor
from .v3_remote_llm_experiment import policy_guard
from .v3_remote_recovery_full import (RECOVERY_POOL, install_runtime,
                                      registry_payload)


DEV_SEEDS = (63001,)
CONFIRM_SEEDS = (64007, 64013, 64019, 64033, 64037, 64063, 64067, 64081)
CYCLES = 3
CAPACITY = 80
DEPTH = 4
CMP_CONTEXTS_PER_FAMILY = 24
CAR_CONTEXTS_PER_FAMILY = 24
SHAPLEY_CONTEXTS_PER_STRATUM = 16

BASELINES = (
    "FIFO",
    "Recency",
    "Probe-Filter",
    "Physical-MCShapley",
    "CMP-IPW-adapted",
    "CAR-Memory-adapted",
)
METHODS = ("CREDIT-Covered",) + BASELINES + ("NoMem",)


class PassGate:
    """Expose the selected bank without borrowing CREDIT transport evidence."""

    def candidates(self, context, memories, *, directly_validated=frozenset()):
        if directly_validated:
            raise ValueError("PassGate has no direct-evidence override")
        return tuple(memories)

    def transported(self, context):
        return True


class ProbeFilterGate(PassGate):
    """Suppress source memories scoped to roles with measured target drift."""

    def __init__(self, evidence):
        self.evidence = {(p.role, p.version): p for p in evidence}

    def candidates(self, context, memories, *, directly_validated=frozenset()):
        if directly_validated:
            raise ValueError("ProbeFilterGate has no direct-evidence override")
        blocked = {node.role.key for node in context.nodes
                   if self.evidence.get((node.role.key, node.version)) is None
                   or self.evidence[node.role.key, node.version].level != Level.V}
        # A target-native candidate is not stale source memory.
        return tuple(memory for memory in memories
                     if memory.uid == "new_contract"
                     or not (set(memory.scope or ()) & blocked))


def seeds_for(split):
    return DEV_SEEDS if split == "development" else CONFIRM_SEEDS


def normalize(value):
    return json.loads(json.dumps(value, sort_keys=True))


def pack_order(memories):
    bank = []
    for memory in memories:
        if bank_cost((*bank, memory.uid), RECOVERY_POOL) <= CAPACITY:
            bank.append(memory.uid)
    return tuple(sorted(bank))


def additive_bank(scores):
    """Exact capacity-constrained additive choice over the nine-item registry."""
    ids = tuple(sorted(memory.uid for memory in RECOVERY_POOL))
    candidates = []
    for size in range(len(ids) + 1):
        for bank in combinations(ids, size):
            cost = bank_cost(bank, RECOVERY_POOL)
            if cost <= CAPACITY:
                value = sum(max(0.0, float(scores.get(uid, 0.0))) for uid in bank)
                candidates.append((value, -cost, tuple(reversed(bank)), bank))
    return max(candidates)[-1]


def execute_cached(ledger, phase, executor, context, exposure, cache, reuse):
    key = context.uid, tuple(exposure)
    if key in cache:
        reuse.append({"phase": phase, "context": context.uid,
                      "exposure": list(exposure), "physical_record": cache[key][0]})
        return cache[key][1]
    episode = ledger.execute(phase, executor, context, tuple(exposure))
    cache[key] = len(ledger.records) - 1, episode
    return episode


def selection_result(directory, method, bank, scores, ledger=None, *,
                     details=None, attributed_calls=None, reuse=()):
    directory.mkdir(parents=True, exist_ok=True)
    physical = 0 if ledger is None else sum(ledger.calls.values())
    attributed = physical if attributed_calls is None else attributed_calls
    payload = {
        "method": method,
        "bank": list(bank),
        "bank_tokens": bank_cost(bank, RECOVERY_POOL),
        "scores": {uid: float(scores.get(uid, 0.0))
                   for uid in sorted(memory.uid for memory in RECOVERY_POOL)},
        "capacity": CAPACITY,
        "retrieval_depth": DEPTH,
        "selection_uses_heldout_labels": False,
        "details": details or {},
    }
    write(directory / "bank_selection.json", payload)
    costs = {
        "physical_api_calls": physical,
        "attributed_api_calls": int(attributed),
        "shared_calls_reused": int(attributed - physical),
        "physical_episodes": 0 if ledger is None else sum(ledger.episodes.values()),
        "phase_api_calls": {} if ledger is None else ledger.calls,
    }
    write(directory / "costs.json", costs)
    write(directory / "execution_reuse.json", list(reuse))
    if ledger is not None:
        stream_records(directory / "executions.jsonl.gz", ledger.records)
    return bank, PassGate(), costs


def heuristic_selections(directory, evidence, probe_fee):
    fifo = pack_order(sorted(RECOVERY_POOL, key=lambda m: (m.arrival, m.uid)))
    recent = pack_order(sorted(RECOVERY_POOL,
                               key=lambda m: (-RECENCY[m.uid], m.uid)))
    outputs = {}
    for name, bank in (("FIFO", fifo), ("Recency", recent)):
        outputs[name] = selection_result(
            directory / name, name, bank, {},
            details={"ordering": "arrival ascending" if name == "FIFO"
                     else "frozen RECENCY descending", "deterministic": True})
    bank, _, costs = selection_result(
        directory / "Probe-Filter", "Probe-Filter", recent, {},
        details={"base": "Recency", "rule": "suppress source memories whose role scope intersects a non-V target role",
                 "non_v_roles": sorted({p.role + "@" + p.version for p in evidence
                                        if p.version.endswith(":target") and p.level != Level.V}),
                 "probe_evidence_shared_from_source_history": True},
        attributed_calls=probe_fee)
    outputs["Probe-Filter"] = bank, ProbeFilterGate(evidence), costs
    return outputs


def physical_shapley(corpus, seed, cycle, directory):
    ledger = CallLedger({"physical-shapley": 20000}, 150000)
    ledger.tag = "Physical-MCShapley"
    executor = RecoveryExecutor()
    cache, reuse = {}, []
    samples = defaultdict(list)
    rng = np.random.default_rng(stable_seed(seed, cycle, "physical-mc-shapley"))
    strata = [f"{family}/{epoch}" for family in ("books", "travel", "inventory", "payments")
              for epoch in ("source", "target")]
    assignments = []
    for stratum in strata:
        for index in range(SHAPLEY_CONTEXTS_PER_STRATUM):
            context = corpus.make(stratum, "direct-screening", index)
            permutation = tuple(rng.permutation([m.uid for m in RECOVERY_POOL]).tolist())
            assignments.append({"context": context.uid, "permutation": list(permutation)})
            prefix = []
            exposure = retriever(DEPTH)(context, tuple(prefix))
            previous = execute_cached(ledger, "physical-shapley", executor, context,
                                      exposure, cache, reuse).outcome
            for uid in permutation:
                prefix.append(uid)
                exposure = retriever(DEPTH)(context, tuple(prefix))
                current = execute_cached(ledger, "physical-shapley", executor, context,
                                         exposure, cache, reuse).outcome
                samples[uid].append(current - previous)
                previous = current
    scores = {memory.uid: float(np.mean(samples[memory.uid]))
              if samples[memory.uid] else 0.0 for memory in RECOVERY_POOL}
    bank = additive_bank(scores)
    return selection_result(
        directory, "Physical-MCShapley", bank, scores, ledger,
        details={"estimator": "real CRN outcome differences along uniformly randomized coalition prefixes",
                 "permutations": len(assignments), "assignments": assignments,
                 "samples_per_memory": {uid: len(v) for uid, v in samples.items()},
                 "structural_exact_reuse": len(reuse)}, reuse=reuse)


def cmp_ipw(corpus, seed, cycle, directory):
    ledger = CallLedger({"target-factorial": 40000, "replay_audit": 2000}, 150000)
    ledger.tag = "CMP-IPW-adapted"
    reports, episodes = [], []
    for family in ("books", "travel", "inventory", "payments"):
        contexts = [c for c in corpus.phase("target-adaptation")
                    if c.stratum == family + "/target"][:CMP_CONTEXTS_PER_FAMILY]
        observed, estimates = balanced_designs(
            contexts, RECOVERY_POOL, ledger,
            stable_seed(seed, cycle, family, "cmp-balanced-ipw"))
        episodes.extend(observed)
        reports.extend(estimates)
    # Equal-probability balanced rows make the self-normalized treated/control
    # contrast equal to the recorded context-level Banzhaf contrast.
    samples, positivity = defaultdict(list), defaultdict(lambda: {"contexts": 0,
                                                                  "min_probability": 1.0,
                                                                  "max_probability": 0.0})
    for report in reports:
        probability = float(report["assignment_probability"])
        for uid, value in zip(report["memory_ids"], report["banzhaf"]):
            samples[uid].append(float(value))
            row = positivity[uid]
            row["contexts"] += 1
            row["min_probability"] = min(row["min_probability"], probability)
            row["max_probability"] = max(row["max_probability"], probability)
    scores = {memory.uid: float(np.mean(samples[memory.uid]))
              if samples[memory.uid] else 0.0 for memory in RECOVERY_POOL}
    bank = additive_bank(scores)
    directory.mkdir(parents=True)
    write(directory / "factorial_estimates.json", reports)
    return selection_result(
        directory, "CMP-IPW-adapted", bank, scores, ledger,
        details={"adapter": "task-level randomized exposure adapter, not an upstream CMP port",
                 "estimator": "self-normalized treated/control contrast under equal-probability reserved-slot assignments",
                 "contexts": len(reports), "randomized_episodes": len(episodes),
                 "positivity": dict(positivity), "aggregation": "flat mean over observed target contexts"})


def forced_exposures(context, uid):
    all_ids = tuple(memory.uid for memory in RECOVERY_POOL)
    base = list(retriever(DEPTH)(context, all_ids))
    memory = next(memory for memory in RECOVERY_POOL if memory.uid == uid)
    if not any(memory.eligible(node.role) for node in context.nodes):
        return tuple(base), tuple(base)
    if uid in base:
        treated = tuple(base)
        control = retriever(DEPTH)(context, tuple(x for x in all_ids if x != uid))
    else:
        kept = base[:max(0, DEPTH - 1)] + [uid]
        treated = ranked(context, tuple(kept))[:DEPTH]
        control = tuple(base)
    return tuple(treated), tuple(control)


def car_memory(corpus, seed, cycle, directory):
    ledger = CallLedger({"car-forward-replay": 20000}, 150000)
    ledger.tag = "CAR-Memory-adapted"
    executor = RecoveryExecutor()
    cache, reuse = {}, []
    samples, active = defaultdict(list), defaultdict(int)
    interventions = []
    for family in ("books", "travel", "inventory", "payments"):
        contexts = [c for c in corpus.phase("target-adaptation")
                    if c.stratum == family + "/target"][:CAR_CONTEXTS_PER_FAMILY]
        for context in contexts:
            for memory in RECOVERY_POOL:
                treated, control = forced_exposures(context, memory.uid)
                if treated == control:
                    samples[memory.uid].append(0.0)
                    continue
                y1 = execute_cached(ledger, "car-forward-replay", executor, context,
                                    treated, cache, reuse).outcome
                y0 = execute_cached(ledger, "car-forward-replay", executor, context,
                                    control, cache, reuse).outcome
                samples[memory.uid].append(float(y1 - y0))
                active[memory.uid] += 1
                interventions.append({"context": context.uid, "memory": memory.uid,
                                      "treated": list(treated), "control": list(control)})
    scores = {memory.uid: float(np.mean(samples[memory.uid]))
              if samples[memory.uid] else 0.0 for memory in RECOVERY_POOL}
    bank = additive_bank(scores)
    return selection_result(
        directory, "CAR-Memory-adapted", bank, scores, ledger,
        details={"adapter": "task-level time-zero memory intervention, not an upstream CAR port",
                 "estimator": "paired full-trajectory forward re-execution with common task noise",
                 "contexts": 4 * CAR_CONTEXTS_PER_FAMILY,
                 "active_interventions": dict(active),
                 "interventions": interventions,
                 "structural_exact_reuse": len(reuse)}, reuse=reuse)


def dataset_protocol(dataset, split, seeds):
    return {
        "split": split, "seeds": list(seeds), "cycles": CYCLES,
        "registry": registry_payload(), "synthetic": True,
        "outcomes_preassigned": False,
        "purpose": "fresh chronological same-protocol baseline evaluation",
        "generator_sha256": digest("credit/v3_remote_recovery_baselines.py"),
        "recovery_executor_sha256": digest("credit/v3_recovery.py"),
    }


def generate_dataset(dataset, split):
    seeds = seeds_for(split)
    _, _, data = install_runtime(seeds)
    data.SEEDS = tuple(seeds)
    data.CYCLES = CYCLES
    data.generate(dataset)
    write(dataset / "baseline_dataset_protocol.json",
          dataset_protocol(dataset, split, seeds))
    write(dataset / "manifest.json", {path.name: digest(path) for path in dataset.iterdir()
                                      if path.is_file() and path.name != "manifest.json"})


def frozen_protocol(policy, dataset, split):
    actor, _ = policy_guard(policy)
    seeds = seeds_for(split)
    sources = (
        "credit/v3_remote_recovery_baselines.py",
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
        "experiment": "credit_v3_remote_recovery_baselines_v1",
        "split": split, "seeds": list(seeds), "cycles": CYCLES,
        "methods": list(METHODS), "capacity": CAPACITY,
        "retrieval_depth": DEPTH,
        "dataset": str(dataset.resolve()),
        "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "remote_policy": str(policy.resolve()),
        "actor_scores_sha256": actor["decision_scores_sha256"],
        "source_sha256": {name: digest(name) for name in sources},
        "paper_sha256": digest("CREDIT_WWW2027_v3.tex"),
        "agents_sha256": digest("AGENTS.md"),
        "registry": registry_payload(),
        "development_seed": list(DEV_SEEDS),
        "confirmation_seeds_frozen_before_development_results": list(CONFIRM_SEEDS),
        "cmp_contexts_per_family": CMP_CONTEXTS_PER_FAMILY,
        "car_contexts_per_family": CAR_CONTEXTS_PER_FAMILY,
        "shapley_contexts_per_stratum": SHAPLEY_CONTEXTS_PER_STRATUM,
        "primary": "CREDIT-Covered versus each frozen baseline",
        "inference": "paired seed and family bootstrap; exact seed sign-flip; Holm across frozen baseline family",
        "scope": "finite synthetic API, cached remote actor, three chronological cycles; adapted CMP/CAR are not upstream ports",
    }


def probe_fee(source_directory):
    costs = json.loads((source_directory / "costs.json").read_text())
    return (costs["api_calls"]["factorial"]
            + costs["api_calls"]["replay_audit"] + costs["fixtures"])


def score(prepared, output, cycle):
    rows, credit_episodes = [], {}
    executor = RecoveryExecutor()
    with gzip.open(output / f"test_cycle_{cycle}.jsonl.gz", "wt", compresslevel=6) as stream:
        for seed, corpus, credit_state, policies, _, _ in prepared:
            for method in METHODS:
                bank, gate = policies[method][:2]
                collected = []
                for context in corpus.phase("heldout-test"):
                    candidates = gate.candidates(context, [memory for memory in RECOVERY_POOL
                                                            if memory.uid in bank])
                    exposure = retriever(DEPTH)(context, tuple(memory.uid for memory in candidates))
                    episode = executor.run(context, exposure)
                    nodes = {node.uid: node for node in context.nodes}
                    row = {
                        "seed": seed, "cycle": cycle, "method": method,
                        "task": context.uid, "family": context.stratum.split("/")[0],
                        "epoch": context.stratum.split("/")[1],
                        "success": episode.outcome, "api_calls": episode.calls,
                        "bank_tokens": bank_cost(bank, RECOVERY_POOL),
                        "exposure": list(exposure), "state_hash": episode.state_hash,
                        "verification_known": episode.verified is not None,
                        "contract_invalid_calls": sum(
                            not valid_request(ROLE_NAMES[nodes[event["node"]].role.key],
                                              event["state_before"], event["request_mode"],
                                              *context.stratum.split("/"))
                            for event in episode.events),
                    }
                    rows.append(row)
                    stream.write(json.dumps(row | {"episode": asdict(episode)},
                                            separators=(",", ":")) + "\n")
                    if method == "CREDIT-Covered":
                        collected.append(episode)
                if method == "CREDIT-Covered":
                    credit_episodes[seed] = collected
            print("scored baseline block", cycle, seed, flush=True)
    return rows, credit_episodes


def run(policy, dataset, output, split):
    if output.exists():
        raise FileExistsError("fresh baseline output required")
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
            base, history, credit_state = inputs[seed]
            corpus = CycleCorpus(base, cycle)
            credit_dir = output / f"seed_{seed}" / f"cycle_{cycle}" / "CREDIT-Covered"
            credit_dir.mkdir(parents=True)
            if cycle:
                credit_state["models"] = experiment.fit_ensemble(
                    history[0] + credit_state["episodes"], history[1], history[2],
                    stable_seed(seed, cycle, "online-refit") % 2**31,
                    credit_dir / "models", logged=credit_state["logged"])
            write(credit_dir / "pre_adaptation.json", {
                "bank": credit_state["bank"], "posterior": credit_state["posterior"],
                "store": credit_state["store"].snapshot(),
                "logged_ids": [episode.context.uid for episode in credit_state["logged"]],
            })
            gate, adaptation_calls = experiment.adaptation(
                credit_state, corpus, seed, cycle, credit_dir / "adaptation")
            ledger, gate = experiment.selection(
                credit_state, gate, corpus,
                adaptation_calls + (credit_state["source_fee"] if cycle == 0 else 0),
                credit_dir / "selection")
            write(credit_dir / "frozen_policy.json", credit_state["deployment_policy"])
            adaptation_costs = json.loads((credit_dir / "adaptation/costs.json").read_text())
            source_costs = json.loads((output / f"seed_{seed}/source_history/costs.json").read_text())
            physical_episodes = (sum(ledger.episodes.values())
                                 + adaptation_costs["total_executed_episodes"]
                                 + (source_costs["total_executed_episodes"] if cycle == 0 else 0))
            credit_costs = {"physical_api_calls": ledger.spent,
                            "attributed_api_calls": ledger.spent,
                            "shared_calls_reused": 0,
                            "physical_episodes": physical_episodes,
                            "phase_api_calls": ledger.calls}
            write(credit_dir / "costs.json", credit_costs)

            methods = heuristic_selections(
                output / f"seed_{seed}" / f"cycle_{cycle}", history[4],
                probe_fee(output / f"seed_{seed}" / "source_history") if cycle == 0 else 0)
            methods["Physical-MCShapley"] = physical_shapley(
                corpus, seed, cycle,
                output / f"seed_{seed}" / f"cycle_{cycle}" / "Physical-MCShapley")
            methods["CMP-IPW-adapted"] = cmp_ipw(
                corpus, seed, cycle,
                output / f"seed_{seed}" / f"cycle_{cycle}" / "CMP-IPW-adapted")
            methods["CAR-Memory-adapted"] = car_memory(
                corpus, seed, cycle,
                output / f"seed_{seed}" / f"cycle_{cycle}" / "CAR-Memory-adapted")
            methods["CREDIT-Covered"] = credit_state["bank"], gate, credit_costs
            methods["NoMem"] = (), PassGate(), {
                "physical_api_calls": 0, "attributed_api_calls": 0,
                "shared_calls_reused": 0, "physical_episodes": 0,
                "phase_api_calls": {}}
            for method in METHODS:
                bank, _, costs = methods[method]
                if bank_cost(bank, RECOVERY_POOL) > CAPACITY:
                    raise ValueError(method + " exceeded storage capacity")
                all_costs.append({"seed": seed, "cycle": cycle, "method": method,
                                  "bank": list(bank), "bank_tokens": bank_cost(bank, RECOVERY_POOL),
                                  **costs})
            prepared.append((seed, corpus, credit_state, methods, ledger, gate))

        frozen = {str(path.relative_to(output)): digest(path) for path in output.rglob("*")
                  if path.is_file() and f"cycle_{cycle}/" in str(path)
                  and path.name in ("bank_selection.json", "frozen_policy.json",
                                    "certification_freeze.json", "pre_adaptation.json")}
        write(output / f"evaluation_freeze_{cycle}.json", frozen)
        rows, credit_episodes = score(prepared, output, cycle)
        all_rows.extend(rows)
        write(output / f"raw_cycle_{cycle}.json", rows)

        for seed, corpus, state, methods, ledger, deployed_gate in prepared:
            episodes = credit_episodes[seed]
            state["logged"].extend(episodes)
            state["store"].expose(episodes, cycle)
            credit_dir = output / f"seed_{seed}" / f"cycle_{cycle}" / "CREDIT-Covered"
            bounds = experiment.archive(state, deployed_gate, ledger, cycle)
            write(credit_dir / "archive_bounds.json", bounds)
            write(credit_dir / "store_after_test.json", state["store"].snapshot())
            write(credit_dir / "evidence_ingestion.json", {
                      "scored_before_ingestion": True,
                      "block_ids": [episode.context.uid for episode in episodes],
                      "cumulative_logged_episodes": len(state["logged"]),
                      "randomized_training_episodes": len(state["episodes"]),
                  })
        write(output / f"costs_cycle_{cycle}.json",
              [row for row in all_costs if row["cycle"] == cycle])

    write(output / "raw_test_results.json", all_rows)
    main, paired = summarize(all_rows, "CREDIT-Covered", repeats=4000,
                             random_seed=9173)
    write(output / "statistics.json", {"main": main, "paired": paired})
    write(output / "costs.json", all_costs)
    write(output / "summary.json", {
        "split": split, "seeds": list(seeds), "cycles": CYCLES,
        "methods": list(METHODS), "test_episodes": len(all_rows),
        "test_api_calls": sum(row["api_calls"] for row in all_rows),
        "chronological_complete": True, "same_protocol_baselines_complete": True,
        "paper_experiments_complete": False,
    })
    write(output / "artifact_manifest.json", {
        str(path.relative_to(output)): digest(path) for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"})
    print("completed recovery baseline suite", split, flush=True)


def audit(policy, dataset, output, split):
    seeds = seeds_for(split)
    install_runtime(seeds)
    install_policy(policy / "decision_scores.json")
    expected_protocol = frozen_protocol(policy, dataset, split)
    if json.loads((output / "frozen_protocol.json").read_text()) != expected_protocol:
        raise ValueError("frozen protocol differs")
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for name, expected in manifest.items():
        if digest(output / name) != expected:
            raise ValueError("artifact changed: " + name)
    for name, expected in expected_protocol["source_sha256"].items():
        if digest(name) != expected or digest(output / "source" / Path(name).name) != expected:
            raise ValueError("source changed: " + name)
    if digest(dataset / "manifest.json") != expected_protocol["dataset_manifest_sha256"]:
        raise ValueError("dataset changed")
    if digest("CREDIT_WWW2027_v3.tex") != expected_protocol["paper_sha256"]:
        raise ValueError("paper changed")
    if digest("AGENTS.md") != expected_protocol["agents_sha256"]:
        raise ValueError("AGENTS changed")

    policies = {}
    for seed in seeds:
        for cycle in range(CYCLES):
            root = output / f"seed_{seed}" / f"cycle_{cycle}"
            credit = json.loads((root / "CREDIT-Covered/frozen_policy.json").read_text())
            policies[seed, cycle, "CREDIT-Covered"] = tuple(credit["bank"]), None
            for method in BASELINES:
                selected = json.loads((root / method / "bank_selection.json").read_text())
                bank = tuple(selected["bank"])
                if selected["selection_uses_heldout_labels"] or bank_cost(bank, RECOVERY_POOL) > CAPACITY:
                    raise ValueError("invalid selection: " + method)
                if "heldout-test" in (root / method / "bank_selection.json").read_text():
                    raise ValueError("heldout identifier in selection: " + method)
                costs = json.loads((root / method / "costs.json").read_text())
                if costs["attributed_api_calls"] < costs["physical_api_calls"]:
                    raise ValueError("attributed calls below physical calls: " + method)
                execution_path = root / method / "executions.jsonl.gz"
                records = []
                if execution_path.exists():
                    with gzip.open(execution_path, "rt") as stream:
                        records = [json.loads(line) for line in stream]
                    calls = sum(record["episode"]["calls"] for record in records)
                    if (calls != costs["physical_api_calls"]
                            or len(records) != costs["physical_episodes"]):
                        raise ValueError("physical cost ledger differs: " + method)
                elif costs["physical_api_calls"] or costs["physical_episodes"]:
                    raise ValueError("missing physical execution ledger: " + method)
                for reuse in json.loads((root / method / "execution_reuse.json").read_text()):
                    if not 0 <= reuse["physical_record"] < len(records):
                        raise ValueError("reuse reference out of range: " + method)
                    physical = records[reuse["physical_record"]]
                    if (physical["phase"] != reuse["phase"]
                            or physical["episode"]["context"]["uid"] != reuse["context"]
                            or physical["episode"]["exposure"] != reuse["exposure"]):
                        raise ValueError("reuse reference differs: " + method)
                if method == "CMP-IPW-adapted":
                    randomized = [record for record in records
                                  if record["phase"] == "target-factorial"]
                    if not randomized or any(not record["episode"]["randomized"]
                                             or not record["episode"]["assignment_probability"]
                                             for record in randomized):
                        raise ValueError("CMP randomized exposure provenance differs")
                gate = None
                policies[seed, cycle, method] = bank, gate
            policies[seed, cycle, "NoMem"] = (), PassGate()

    executor = RecoveryExecutor()
    all_rows, checked, calls = [], 0, 0
    for cycle in range(CYCLES):
        indexed = {(row["seed"], row["method"], row["task"]): row
                   for row in json.loads((output / f"raw_cycle_{cycle}.json").read_text())}
        expected_keys = set()
        with gzip.open(output / f"test_cycle_{cycle}.jsonl.gz", "rt") as stream:
            traces = [json.loads(line) for line in stream]
        trace_index = {(row["seed"], row["method"], row["task"]): row for row in traces}
        for seed in seeds:
            corpus = Corpus(dataset, seed)
            cycle_corpus = CycleCorpus(corpus, cycle)
            source_evidence = json.loads((output / f"seed_{seed}/source_history/transfer_levels.json").read_text())
            evidence = [__import__("credit.v3.drift", fromlist=["ProbeEvidence"]).ProbeEvidence(
                **(row | {"level": Level(row["level"])})) for row in source_evidence]
            credit_data = json.loads((output / f"seed_{seed}/cycle_{cycle}/CREDIT-Covered/frozen_policy.json").read_text())
            from .v3_online_experiment import restore_policy
            credit_gate = restore_policy(credit_data)
            for method in METHODS:
                bank, gate = policies[seed, cycle, method]
                if method == "CREDIT-Covered":
                    gate = credit_gate
                elif method == "Probe-Filter":
                    gate = ProbeFilterGate(evidence)
                else:
                    gate = gate or PassGate()
                for context in cycle_corpus.phase("heldout-test"):
                    key = seed, method, context.uid
                    expected_keys.add(key)
                    row, trace = indexed[key], trace_index[key]
                    candidates = gate.candidates(context, [memory for memory in RECOVERY_POOL
                                                            if memory.uid in bank])
                    exposure = retriever(DEPTH)(context, tuple(memory.uid for memory in candidates))
                    episode = executor.run(context, exposure)
                    if normalize(asdict(episode)) != trace["episode"]:
                        raise ValueError("heldout episode reconstruction differs")
                    if ({k: v for k, v in trace.items() if k != "episode"} != row
                            or episode.outcome != row["success"]
                            or episode.calls != row["api_calls"]
                            or list(exposure) != row["exposure"]
                            or bank_cost(bank, RECOVERY_POOL) != row["bank_tokens"]):
                        raise ValueError("heldout summary differs")
                    checked += 1
                    calls += episode.calls
        if set(indexed) != expected_keys or set(trace_index) != expected_keys:
            raise ValueError("heldout coverage differs")
        all_rows.extend(indexed.values())

    if all_rows != json.loads((output / "raw_test_results.json").read_text()):
        raise ValueError("combined rows differ")
    main, paired = summarize(all_rows, "CREDIT-Covered", repeats=4000,
                             random_seed=9173)
    if {"main": main, "paired": paired} != json.loads((output / "statistics.json").read_text()):
        raise ValueError("statistics differ")
    report = {
        "verified": True, "split": split, "seeds": list(seeds),
        "heldout_episodes_reexecuted": checked,
        "heldout_api_calls_reexecuted": calls,
        "artifact_manifest_sha256": digest(output / "artifact_manifest.json"),
        "auditor_sha256": digest(__file__),
        "checks": ["source/data/policy binding", "bank capacity", "heldout isolation marker",
                   "selection physical-call accounting", "exact execution reuse references",
                   "CMP randomized propensity provenance", "complete heldout reconstruction",
                   "paired statistics"],
        "limits": expected_protocol["scope"],
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
