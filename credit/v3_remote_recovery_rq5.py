"""Fresh-seed RQ5 mechanism ablations and stress tests for recovery CREDIT.

This experiment is deliberately scoped to the finite recovery adapter.  It
reuses one chronological evidence stream per seed/cycle, but every learned or
physical selection is completed before held-out scoring.  The no-interaction
arm is retrained with all pair features zeroed.  OAT and observational arms are
explicit design replacements rather than names for the full CREDIT method.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
from itertools import chain
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from .llm_policy import install_policy
from .paired_statistics import summarize
from .v3.drift import Level, ProbeEvidence
from .v3.experiment import digest, write
from .v3.inference import worst_mixture
from .v3.local import (FAMILIES, ROLE_NAMES, ROLES, Context, Node, factory,
                       retriever, stable_seed, valid_request)
from .v3.selection import bank_cost
from .v3_benchmark import CachedModel
from .v3_data import Corpus
from .v3_online import StoredGate
from .v3_online_data import CycleCorpus
from .v3_online_fast import FastOnlineModel
from .v3_recovery import RecoveryExecutor
from .v3_revalidation import EvidenceGate
from .v3_remote_llm_experiment import policy_guard
from .v3_remote_recovery_full import (RECOVERY_POOL, install_runtime,
                                      registry_payload)
from . import v3_remote_recovery_equal_call as equal_v1
from . import v3_remote_recovery_equal_call_v2 as equal_v2


DEV_SEEDS = (69001,)
CONFIRM_SEEDS = (70001, 70009, 70019, 70039, 70051, 70061, 70079, 70099)
CYCLES = 3
BUDGET = 75_000
CAPACITY = 80
DEPTH = 4
STRATA = tuple(f"{family}/{epoch}" for family in FAMILIES
               for epoch in ("source", "target"))
CAPACITIES = (24, 40, 60, 80, 103)
DEPTHS = (2, 4, 6)
PROBE_CONTEXTS = (0, 24, 48, 96)
UNREPLAYABLE_SHARES = (0, 25, 50, 75)
SKEW_WEIGHTS = {
    "books/source": .05, "books/target": .25,
    "travel/source": .10, "travel/target": .10,
    "inventory/source": .05, "inventory/target": .25,
    "payments/source": .10, "payments/target": .10,
}


class NoInteractionFastModel(FastOnlineModel):
    """The current online architecture with pair features held at zero."""

    def _pack(self, items):
        packed = list(super()._pack(items))
        packed[4] = torch.zeros_like(packed[4])
        return tuple(packed)


class PassGate:
    def candidates(self, context, memories, *, directly_validated=frozenset()):
        if directly_validated:
            raise ValueError("PassGate does not accept direct overrides")
        return tuple(memories)


class UnreplayableGate:
    def __init__(self, base, blocked):
        self.base = base
        self.blocked = frozenset(blocked)

    def candidates(self, context, memories, *, directly_validated=frozenset()):
        if context.stratum in self.blocked:
            return ()
        return self.base.candidates(context, memories,
                                    directly_validated=directly_validated)


def seeds_for(split):
    return DEV_SEEDS if split == "development" else CONFIRM_SEEDS


def generate_dataset(dataset, split):
    equal_v1.DEV_SEEDS = DEV_SEEDS
    equal_v1.CONFIRM_SEEDS = CONFIRM_SEEDS
    equal_v1.generate_dataset(dataset, split)
    write(dataset / "rq5_dataset_protocol.json", {
        "split": split, "seeds": list(seeds_for(split)), "cycles": CYCLES,
        "selection_budget": BUDGET, "capacity_grid": list(CAPACITIES),
        "depth_grid": list(DEPTHS), "probe_context_grid": list(PROBE_CONTEXTS),
        "unreplayable_percent_grid": list(UNREPLAYABLE_SHARES),
        "generator_sha256": digest(__file__),
        "base_generator_sha256": digest("credit/v3_remote_recovery_equal_call.py"),
    })
    write(dataset / "manifest.json", {
        path.name: digest(path) for path in dataset.iterdir()
        if path.is_file() and path.name != "manifest.json"})


def fit_no_interaction(train, validation, calibration, seed, directory, *, members=2):
    directory.mkdir(parents=True)
    grouped = defaultdict(list)
    for episode in train:
        grouped[episode.context.uid].append(episode)
    ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    models = []
    for member in range(members):
        sample = [episode for uid in rng.choice(ids, len(ids), replace=True)
                  for episode in grouped[uid]]
        model = NoInteractionFastModel(
            RECOVERY_POOL, [episode.context for episode in train],
            seed=seed + member)
        model.fit_online(sample, validation, calibration)
        model.fit_report.update({
            "ablation": "all pair exposure features fixed to zero in training and prediction",
            "neural_computation": "shared fixed feature vectors",
        })
        model.save(directory / f"outcome_model_{member}.pt")
        models.append(CachedModel(model))
    write(directory / "training_reports.json", [model.fit_report for model in models])
    return tuple(models)


def protocol(policy, dataset, split):
    actor, _ = policy_guard(policy)
    sources = (
        "credit/v3_remote_recovery_rq5.py",
        "credit/v3_remote_recovery_equal_call.py",
        "credit/v3_remote_recovery_equal_call_v2.py",
        "credit/v3_remote_recovery_full.py",
        "credit/v3_recovery.py",
        "credit/v3_online_experiment.py",
        "credit/v3_online.py",
        "credit/v3_online_fast.py",
        "credit/v3_revalidation.py",
        "credit/paired_statistics.py",
        "credit/stateful.py",
    )
    return {
        "experiment": "credit_v3_remote_recovery_rq5_v1",
        "split": split, "seeds": list(seeds_for(split)), "cycles": CYCLES,
        "dataset": str(dataset.resolve()),
        "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "remote_policy": str(policy.resolve()),
        "actor_scores_sha256": actor["decision_scores_sha256"],
        "registry": registry_payload(),
        "selection_call_allowance": BUDGET,
        "main_ablation_family": [
            "CREDIT", "NoInteractions", "NoDriftGate", "NoFreshCertification",
            "OAT-Physical", "Nonrandom-Observed", "Robust-Skew", "NoRobust-Skew",
        ],
        "stress_grids": {
            "capacity_tokens": list(CAPACITIES), "retrieval_depth": list(DEPTHS),
            "differential_probe_contexts_per_role": list(PROBE_CONTEXTS),
            "unreplayable_strata_percent": list(UNREPLAYABLE_SHARES),
        },
        "skew_mixture": SKEW_WEIGHTS,
        "selection_isolation": "all banks, gates, and candidate families freeze before held-out scoring",
        "oat_scope": "physical singleton-versus-empty contrasts followed by additive packing; no joint certificate",
        "observational_scope": "deterministic depth-4 exposure under the covered gate; unidentified memories are excluded",
        "capacity_scope": "physically validated union of current finalists, cap-specific model leaders, empty, and singleton banks",
        "unseen_topology_scope": "two new supported-role sequences; not a general DAG or natural-provider claim",
        "source_sha256": {name: digest(name) for name in sources},
        "development_seed": list(DEV_SEEDS),
        "confirmation_seeds_frozen_before_development_results": list(CONFIRM_SEEDS),
        "scope": "finite synthetic API and cached remote actor; local wall-clock latency; no natural provider drift",
    }


def normalize(value):
    return json.loads(json.dumps(value))


def dump_records(path, records):
    with gzip.open(path, "wt", compresslevel=6) as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def load_records(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def validation_outcomes(directory):
    values = defaultdict(lambda: defaultdict(list))
    for row in load_records(directory / "budget_validation.jsonl.gz"):
        bank = tuple(row["bank"])
        episode = row["episode"]
        values[bank][episode["context"]["stratum"]].append(
            int(episode["verified"] is True))
    return values


def bank_score(bank, outcomes, weights, radius):
    if not all(outcomes[bank][stratum] for stratum in STRATA):
        return -float("inf")
    means = [float(np.mean(outcomes[bank][stratum])) for stratum in STRATA]
    mixture = [weights[stratum] for stratum in STRATA]
    return worst_mixture(means, mixture, radius) - .0001 * bank_cost(bank, RECOVERY_POOL)


def choose_validated(candidates, outcomes, weights, radius, capacity):
    feasible = [tuple(bank) for bank in candidates
                if bank_cost(tuple(bank), RECOVERY_POOL) <= capacity]
    if not feasible:
        return ()
    return max(feasible, key=lambda bank: (
        bank_score(bank, outcomes, weights, radius),
        -bank_cost(bank, RECOVERY_POOL), bank))


def candidate_family(full_directory, full_bank):
    screening = json.loads((full_directory / "screening.json").read_text())
    halving = json.loads((full_directory / "halving.json").read_text())
    rows = screening["all_candidate_scores"]
    candidates = [(), tuple(full_bank)]
    candidates.extend(tuple(bank) for bank in screening["banks"])
    candidates.extend(tuple(bank) for bank in halving["finalists"])
    candidates.extend((memory.uid,) for memory in RECOVERY_POOL)
    for capacity in CAPACITIES:
        feasible = [row for row in rows
                    if bank_cost(tuple(row["bank"]), RECOVERY_POOL) <= capacity]
        if feasible:
            best = max(feasible, key=lambda row: (
                row["score"], -bank_cost(tuple(row["bank"]), RECOVERY_POOL),
                tuple(row["bank"])))
            candidates.append(tuple(best["bank"]))
    return tuple(dict.fromkeys(tuple(sorted(bank)) for bank in candidates))


def oat_select(corpus, gate, directory):
    candidates = ((),) + tuple((memory.uid,) for memory in RECOVERY_POOL)
    _, spent, next_bound, _ = equal_v1.complete_budget(
        corpus, gate, candidates, 0, BUDGET, directory)
    outcomes = validation_outcomes(directory)
    empty = np.mean([np.mean(outcomes[()][stratum]) for stratum in STRATA])
    effects = {}
    for memory in RECOVERY_POOL:
        bank = (memory.uid,)
        effects[memory.uid] = float(
            np.mean([np.mean(outcomes[bank][stratum]) for stratum in STRATA]) - empty)
    selected = []
    for uid in sorted(effects, key=lambda value: (
            -effects[value] / next(memory.cost for memory in RECOVERY_POOL if memory.uid == value),
            value)):
        if effects[uid] <= 0:
            continue
        proposal = tuple(sorted(selected + [uid]))
        if bank_cost(proposal, RECOVERY_POOL) <= CAPACITY:
            selected.append(uid)
    bank = tuple(sorted(selected))
    write(directory / "oat_selection.json", {
        "bank": list(bank), "effects": effects, "empty_mean": float(empty),
        "actual_selection_calls": spent, "unspent": BUDGET - spent,
        "next_complete_group_upper_bound": next_bound,
        "selection_uses_heldout_labels": False,
        "scope": "singleton-versus-empty physical contrasts; additive packing",
    })
    return bank, spent


def observational_select(corpus, gate, directory):
    executor = RecoveryExecutor()
    full_archive = tuple(memory.uid for memory in RECOVERY_POOL)
    rows, spent, groups = [], 0, 0
    for index in range(equal_v1.BUDGET_CONTEXTS_PER_STRATUM):
        contexts = [corpus.make(stratum, "budget-evaluation", index)
                    for stratum in STRATA]
        bound = sum(len(context.nodes) + 4 for context in contexts)
        if spent + bound > BUDGET:
            break
        group = []
        for context in contexts:
            exposure = equal_v1.bank_exposure(context, full_archive, gate)
            episode = executor.run(context, exposure)
            group.append({"phase": "observational-selection",
                          "episode": normalize(asdict(episode))})
        calls = sum(row["episode"]["calls"] for row in group)
        if spent + calls > BUDGET:
            raise ValueError("observational reservation failed")
        rows.extend(group)
        spent += calls
        groups += 1
    associations, positivity = {}, {}
    for memory in RECOVERY_POOL:
        exposed = [int(row["episode"]["verified"] is True) for row in rows
                   if memory.uid in row["episode"]["exposure"]]
        absent = [int(row["episode"]["verified"] is True) for row in rows
                  if memory.uid not in row["episode"]["exposure"]]
        identified = bool(exposed and absent)
        associations[memory.uid] = (float(np.mean(exposed) - np.mean(absent))
                                    if identified else None)
        positivity[memory.uid] = {"exposed": len(exposed), "absent": len(absent),
                                  "identified": identified}
    selected = []
    ranking = sorted((uid for uid, value in associations.items()
                      if value is not None and value > 0),
                     key=lambda uid: (-associations[uid], uid))
    for uid in ranking:
        proposal = tuple(sorted(selected + [uid]))
        if bank_cost(proposal, RECOVERY_POOL) <= CAPACITY:
            selected.append(uid)
    bank = tuple(sorted(selected))
    directory.mkdir(parents=True, exist_ok=True)
    dump_records(directory / "observational_selection.jsonl.gz", rows)
    write(directory / "observational_selection.json", {
        "bank": list(bank), "associations": associations,
        "positivity": positivity, "groups": groups,
        "actual_selection_calls": spent, "unspent": BUDGET - spent,
        "selection_uses_heldout_labels": False,
        "scope": "unadjusted association under deterministic retrieval; unidentified memories excluded",
    })
    return bank, spent


def probe_evidence(seed, cycle, initial_evidence, fixtures):
    from .v3.local import differential_probes
    support = {(row.role, row.version): row.support for row in initial_evidence
               if row.version.endswith(":source")}
    return differential_probes(
        seed=stable_seed(seed, "rq5-nested-probes", cycle),
        support=support, fixtures=fixtures)


def unseen_context(seed, cycle, stratum, topology, index):
    family, epoch = stratum.split("/")
    sequences = {
        "skip-authorize": ("lookup", "transform", "submit"),
        "late-transform": ("lookup", "authorize", "transform", "submit"),
    }
    names = sequences[topology]
    nodes = tuple(Node(
        f"node-{position}", ROLES[name], f"stateful:{family}:{name}:{epoch}",
        (FAMILIES.index(family) / 3, position / 8, 1.0, 0.0))
        for position, name in enumerate(names))
    return Context(
        f"v3:cycle-{cycle}/unseen:{topology}:{family}:{epoch}:{index}",
        stratum, nodes,
        tuple((nodes[i].uid, nodes[i + 1].uid) for i in range(len(nodes) - 1)),
        stable_seed(seed, cycle, "unseen", topology, family, epoch, index),
        initial_state=0)


def policy_record(method, bank, depth=DEPTH, gate_kind="common", **extra):
    return {"method": method, "bank": list(bank), "depth": depth,
            "gate_kind": gate_kind, **extra}


def build_policies(seed, cycle, corpus, state, history, gate, no_models, directory):
    full_dir = directory / "CREDIT"
    full_bank, full_calls, _ = equal_v2.current_budget_select(
        state["models"], gate, corpus, "CREDIT", BUDGET, full_dir)
    noint_dir = directory / "NoInteractions"
    noint_bank, noint_calls, _ = equal_v2.current_budget_select(
        no_models, gate, corpus, "NoInteractions", BUDGET, noint_dir)

    halving = json.loads((full_dir / "halving.json").read_text())
    nocert_bank = tuple(halving["finalists"][0]) if halving["finalists"] else ()

    candidates = candidate_family(full_dir, full_bank)
    family_dir = directory / "SharedCandidateValidation"
    _, family_calls, family_bound, _ = equal_v1.complete_budget(
        corpus, gate, candidates, 0, BUDGET, family_dir)
    family_outcomes = validation_outcomes(family_dir)
    uniform = {stratum: 1 / len(STRATA) for stratum in STRATA}
    capacity_banks = {
        capacity: choose_validated(candidates, family_outcomes, uniform, .2, capacity)
        for capacity in CAPACITIES
    }
    robust_skew = choose_validated(
        candidates, family_outcomes, SKEW_WEIGHTS, .2, CAPACITY)
    nonrobust_skew = choose_validated(
        candidates, family_outcomes, SKEW_WEIGHTS, 0., CAPACITY)
    write(family_dir / "rq5_choices.json", {
        "candidate_banks": [list(bank) for bank in candidates],
        "capacity_banks": {str(k): list(v) for k, v in capacity_banks.items()},
        "robust_skew_bank": list(robust_skew),
        "nonrobust_skew_bank": list(nonrobust_skew),
        "actual_selection_calls": family_calls, "unspent": BUDGET - family_calls,
        "next_complete_group_upper_bound": family_bound,
        "selection_uses_heldout_labels": False,
    })

    oat_bank, oat_calls = oat_select(corpus, gate, directory / "OAT-Physical")
    observed_bank, observed_calls = observational_select(
        corpus, gate, directory / "Nonrandom-Observed")

    records = [
        policy_record("CREDIT", full_bank),
        policy_record("NoInteractions", noint_bank),
        policy_record("NoDriftGate", full_bank, gate_kind="pass"),
        policy_record("NoFreshCertification", nocert_bank),
        policy_record("OAT-Physical", oat_bank),
        policy_record("Nonrandom-Observed", observed_bank),
        policy_record("Robust-Skew", robust_skew),
        policy_record("NoRobust-Skew", nonrobust_skew),
    ]
    records.extend(policy_record(f"Cap{capacity}", capacity_banks[capacity])
                   for capacity in CAPACITIES)
    records.extend(policy_record(f"Depth{depth}", full_bank, depth=depth)
                   for depth in DEPTHS)

    probe_payload = {}
    for fixtures in PROBE_CONTEXTS:
        evidence, rows = probe_evidence(seed, cycle, history[4], fixtures)
        probe_payload[str(fixtures)] = {
            "evidence": [normalize(asdict(value)) for value in evidence],
            "physical_probe_calls": sum(row["api_calls"] for row in rows),
        }
        records.append(policy_record(
            f"Probe{fixtures}", full_bank, gate_kind="probe",
            evidence=probe_payload[str(fixtures)]["evidence"], fixtures=fixtures))

    for share in UNREPLAYABLE_SHARES:
        count = round(len(STRATA) * share / 100)
        blocked = tuple(STRATA[:count])
        records.append(policy_record(
            f"Unreplayable{share}", full_bank, gate_kind="unreplayable",
            blocked_strata=list(blocked)))
    records.append(policy_record("NoMem", ()))

    write(directory / "policies.json", records)
    write(directory / "probe_grid.json", probe_payload)
    write(directory / "selection_costs.json", {
        "CREDIT": full_calls, "NoInteractions": noint_calls,
        "SharedCandidateValidation": family_calls,
        "OAT-Physical": oat_calls, "Nonrandom-Observed": observed_calls,
        "derived_policy_calls_are_shared_not_double_counted": True,
    })
    return records


def gate_for(record, common_gate):
    kind = record["gate_kind"]
    if kind == "common":
        return common_gate
    if kind == "pass":
        return PassGate()
    if kind == "probe":
        evidence = [ProbeEvidence(**(row | {"level": Level(row["level"])}))
                    for row in record["evidence"]]
        return StoredGate(evidence, common_gate.leases, common_gate.store)
    if kind == "unreplayable":
        return UnreplayableGate(common_gate, record["blocked_strata"])
    raise ValueError("unknown gate kind " + kind)


def score_block(seed, cycle, corpus, records, common_gate, output):
    executor = RecoveryExecutor()
    rows = []
    trace_path = output / f"test_cycle_{cycle}.jsonl.gz"
    with gzip.open(trace_path, "at", compresslevel=6) as stream:
        for record in records:
            method = record["method"]
            bank = tuple(record["bank"])
            gate = gate_for(record, common_gate)
            retrieve = retriever(record["depth"])
            for context in corpus.phase("heldout-test"):
                candidates = gate.candidates(
                    context, [memory for memory in RECOVERY_POOL if memory.uid in bank])
                exposure = retrieve(context, tuple(memory.uid for memory in candidates))
                started = perf_counter_ns()
                episode = executor.run(context, exposure)
                latency = (perf_counter_ns() - started) / 1e6
                nodes = {node.uid: node for node in context.nodes}
                row = {
                    "seed": seed, "cycle": cycle, "method": method,
                    "task": context.uid, "family": context.stratum.split("/")[0],
                    "epoch": context.stratum.split("/")[1],
                    "success": episode.outcome, "api_calls": episode.calls,
                    "latency_ms": latency,
                    "bank_tokens": bank_cost(bank, RECOVERY_POOL),
                    "exposure": list(exposure), "state_hash": episode.state_hash,
                    "gate_kind": record["gate_kind"], "depth": record["depth"],
                    "contract_invalid_calls": sum(
                        not valid_request(
                            ROLE_NAMES[nodes[event["node"]].role.key],
                            event["state_before"], event["request_mode"],
                            *context.stratum.split("/"))
                        for event in episode.events),
                }
                rows.append(row)
                stream.write(json.dumps(row | {"episode": normalize(asdict(episode))},
                                        separators=(",", ":")) + "\n")
    return rows


def score_unseen(seed, cycle, records, common_gate, output):
    keep = {"CREDIT", "NoInteractions", "NoDriftGate", "NoMem"}
    executor = RecoveryExecutor()
    rows = []
    with gzip.open(output / "unseen_topology.jsonl.gz", "at", compresslevel=6) as stream:
        for record in records:
            if record["method"] not in keep:
                continue
            bank = tuple(record["bank"])
            gate = gate_for(record, common_gate)
            retrieve = retriever(record["depth"])
            for topology in ("skip-authorize", "late-transform"):
                for stratum in STRATA:
                    for index in range(32):
                        context = unseen_context(seed, cycle, stratum, topology, index)
                        candidates = gate.candidates(
                            context, [memory for memory in RECOVERY_POOL if memory.uid in bank])
                        exposure = retrieve(context, tuple(memory.uid for memory in candidates))
                        episode = executor.run(context, exposure)
                        row = {
                            "seed": seed, "cycle": cycle, "method": record["method"],
                            "topology": topology, "task": context.uid,
                            "family": context.stratum.split("/")[0],
                            "epoch": context.stratum.split("/")[1],
                            "success": episode.outcome, "api_calls": episode.calls,
                            "bank_tokens": bank_cost(bank, RECOVERY_POOL),
                            "exposure": list(exposure), "state_hash": episode.state_hash,
                        }
                        rows.append(row)
                        stream.write(json.dumps(row | {"episode": normalize(asdict(episode))},
                                                separators=(",", ":")) + "\n")
    return rows


def statistics(rows, unseen):
    main, paired = summarize(rows, "CREDIT", repeats=4000, random_seed=17031)
    latency = {}
    for method in sorted({row["method"] for row in rows}):
        values = np.asarray([row["latency_ms"] for row in rows if row["method"] == method])
        latency[method] = {
            "p50_ms": float(np.quantile(values, .5)),
            "p95_ms": float(np.quantile(values, .95)),
            "mean_ms": float(np.mean(values)),
        }
    skew = {}
    for method in ("Robust-Skew", "NoRobust-Skew"):
        means = {stratum: float(np.mean([
            row["success"] for row in rows
            if row["method"] == method and f"{row['family']}/{row['epoch']}" == stratum]))
            for stratum in STRATA}
        skew[method] = {
            "stratum_means": means,
            "weighted_vsr": sum(SKEW_WEIGHTS[key] * means[key] for key in STRATA),
        }
    unseen_main, unseen_paired = summarize(
        unseen, "CREDIT", repeats=4000, random_seed=17033)
    return {"main": main, "paired": paired, "latency": latency,
            "skew_weighted": skew,
            "unseen_topology": {"main": unseen_main, "paired": unseen_paired}}


def run(policy, dataset, output, split):
    if output.exists():
        raise FileExistsError("fresh RQ5 output required")
    seeds = seeds_for(split)
    experiment, _, _ = install_runtime(seeds)
    install_policy(policy / "decision_scores.json")
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    (output / "source").mkdir()
    frozen = protocol(policy, dataset, split)
    write(output / "frozen_protocol.json", frozen)
    for name in frozen["source_sha256"]:
        path = Path(name)
        (output / "source" / path.name).write_bytes(path.read_bytes())
    (output / "source" / "paper_at_freeze.tex").write_bytes(
        Path("CREDIT_WWW2027_v4.tex").read_bytes())
    (output / "source" / "agents_at_freeze.md").write_bytes(Path("AGENTS.md").read_bytes())
    write(output / "snapshot_hashes.json", {
        "paper_at_freeze.tex": digest(output / "source/paper_at_freeze.tex"),
        "agents_at_freeze.md": digest(output / "source/agents_at_freeze.md"),
    })

    inputs = {}
    for seed in seeds:
        corpus = Corpus(dataset, seed)
        history, states = experiment.initial(seed, corpus, output / f"seed_{seed}")
        inputs[seed] = corpus, history, states["CREDIT-Covered"]

    all_rows, unseen_rows = [], []
    for cycle in range(CYCLES):
        cycle_rows = []
        for seed in seeds:
            base, history, state = inputs[seed]
            corpus = CycleCorpus(base, cycle)
            common = output / f"seed_{seed}/cycle_{cycle}/common"
            common.mkdir(parents=True)
            training = history[0] + state["episodes"]
            refit_seed = stable_seed(seed, cycle, "rq5-refit") % 2**31
            if cycle:
                state["models"] = experiment.fit_ensemble(
                    training, history[1], history[2], refit_seed,
                    common / "models")
            no_models = fit_no_interaction(
                training, history[1], history[2], refit_seed,
                common / "models_no_interactions")
            gate, adaptation_calls = experiment.adaptation(
                state, corpus, seed, cycle, common / "adaptation")
            write(common / "evidence_costs.json", {
                "adaptation_calls": adaptation_calls,
                "source_calls": state["source_fee"] if cycle == 0 else 0,
            })
            selection_dir = output / f"seed_{seed}/cycle_{cycle}/rq5"
            selection_dir.mkdir(parents=True)
            records = build_policies(
                seed, cycle, corpus, state, history, gate, no_models, selection_dir)
            rows = score_block(seed, cycle, corpus, records, gate, output)
            cycle_rows.extend(rows)
            unseen_rows.extend(score_unseen(seed, cycle, records, gate, output))
            print("completed RQ5 block", split, cycle, seed, flush=True)
        all_rows.extend(cycle_rows)
        write(output / f"raw_cycle_{cycle}.json", cycle_rows)
        freeze = {str(path.relative_to(output)): digest(path)
                  for path in output.rglob("*") if path.is_file()
                  and f"cycle_{cycle}/" in str(path)
                  and path.name in ("policies.json", "final_selection.json",
                                    "rq5_choices.json", "oat_selection.json",
                                    "observational_selection.json")}
        write(output / f"evaluation_freeze_{cycle}.json", freeze)

    write(output / "raw_test_results.json", all_rows)
    write(output / "unseen_topology_results.json", unseen_rows)
    stats = statistics(all_rows, unseen_rows)
    write(output / "statistics.json", stats)
    write(output / "summary.json", {
        "split": split, "seeds": list(seeds), "cycles": CYCLES,
        "methods": sorted({row["method"] for row in all_rows}),
        "heldout_episodes": len(all_rows),
        "heldout_api_calls": sum(row["api_calls"] for row in all_rows),
        "unseen_topology_episodes": len(unseen_rows),
        "rq5_controlled_grid_complete": True,
        "natural_provider_experiments_complete": False,
    })
    write(output / "artifact_manifest.json", {
        str(path.relative_to(output)): digest(path) for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"})
    print("completed RQ5", split, flush=True)


def restore_common_gate(output, seed, cycle):
    directory = output / f"seed_{seed}/cycle_{cycle}/common/adaptation"
    evidence = [ProbeEvidence(**(row | {"level": Level(row["level"])}))
                for row in json.loads((directory / "transfer_levels.json").read_text())]
    leases = json.loads((directory / "leases.json").read_text())
    from .v3_online import Store
    store = Store.restore(json.loads((directory / "store_before_test.json").read_text()))
    return StoredGate(evidence, leases, store)


def audit_selection(output, seed, cycle):
    directory = output / f"seed_{seed}/cycle_{cycle}/rq5"
    costs = json.loads((directory / "selection_costs.json").read_text())
    for method in ("CREDIT", "NoInteractions"):
        selected = directory / method
        final = json.loads((selected / "final_selection.json").read_text())
        selection = load_records(selected / "selection.jsonl.gz")
        validation = load_records(selected / "budget_validation.jsonl.gz")
        calls = sum(row["episode"]["calls"] for row in chain(selection, validation))
        if (calls != final["actual_selection_calls"] or calls != costs[method]
                or final["selection_uses_heldout_labels"]
                or any("heldout-test" in row["episode"]["context"]["uid"]
                       for row in chain(selection, validation))):
            raise ValueError("current selector isolation/calls differ")
    for name, summary_name, records_name in (
            ("SharedCandidateValidation", "rq5_choices.json", "budget_validation.jsonl.gz"),
            ("OAT-Physical", "oat_selection.json", "budget_validation.jsonl.gz"),
            ("Nonrandom-Observed", "observational_selection.json",
             "observational_selection.jsonl.gz")):
        selected = directory / name
        summary = json.loads((selected / summary_name).read_text())
        records = load_records(selected / records_name)
        calls = sum(row["episode"]["calls"] for row in records)
        if (calls != summary["actual_selection_calls"] or calls != costs[name]
                or summary["selection_uses_heldout_labels"]
                or any("heldout-test" in row["episode"]["context"]["uid"]
                       for row in records)):
            raise ValueError(name + " isolation/calls differ")


def audit(policy, dataset, output, split):
    seeds = seeds_for(split)
    install_runtime(seeds)
    install_policy(policy / "decision_scores.json")
    expected = protocol(policy, dataset, split)
    if json.loads((output / "frozen_protocol.json").read_text()) != expected:
        raise ValueError("frozen protocol differs")
    for name, expected_hash in json.loads((output / "artifact_manifest.json").read_text()).items():
        if digest(output / name) != expected_hash:
            raise ValueError("artifact changed: " + name)
    for name, expected_hash in expected["source_sha256"].items():
        if (digest(name) != expected_hash
                or digest(output / "source" / Path(name).name) != expected_hash):
            raise ValueError("source differs: " + name)
    if digest(dataset / "manifest.json") != expected["dataset_manifest_sha256"]:
        raise ValueError("dataset differs")

    all_expected, unseen_expected = [], []
    executor = RecoveryExecutor()
    checked, calls = 0, 0
    for cycle in range(CYCLES):
        cycle_rows = json.loads((output / f"raw_cycle_{cycle}.json").read_text())
        indexed = {(row["seed"], row["method"], row["task"]): row
                   for row in cycle_rows}
        traces = load_records(output / f"test_cycle_{cycle}.jsonl.gz")
        traces = [row for row in traces if row["cycle"] == cycle]
        trace_index = {(row["seed"], row["method"], row["task"]): row for row in traces}
        expected_keys = set()
        for seed in seeds:
            audit_selection(output, seed, cycle)
            corpus = CycleCorpus(Corpus(dataset, seed), cycle)
            common_gate = restore_common_gate(output, seed, cycle)
            records = json.loads((output / f"seed_{seed}/cycle_{cycle}/rq5/policies.json").read_text())
            for record in records:
                bank = tuple(record["bank"])
                gate = gate_for(record, common_gate)
                retrieve = retriever(record["depth"])
                for context in corpus.phase("heldout-test"):
                    key = seed, record["method"], context.uid
                    expected_keys.add(key)
                    row, trace = indexed[key], trace_index[key]
                    candidates = gate.candidates(
                        context, [memory for memory in RECOVERY_POOL if memory.uid in bank])
                    exposure = retrieve(context, tuple(memory.uid for memory in candidates))
                    episode = executor.run(context, exposure)
                    if normalize(asdict(episode)) != trace["episode"]:
                        raise ValueError("heldout episode reconstruction differs")
                    comparable = {k: v for k, v in trace.items()
                                  if k not in ("episode", "latency_ms")}
                    stored = {k: v for k, v in row.items() if k != "latency_ms"}
                    if (comparable != stored or row["latency_ms"] < 0
                            or row["success"] != episode.outcome
                            or row["api_calls"] != episode.calls
                            or row["exposure"] != list(exposure)):
                        raise ValueError("heldout summary reconstruction differs")
                    checked += 1
                    calls += episode.calls
        if set(indexed) != expected_keys or set(trace_index) != expected_keys:
            raise ValueError("heldout coverage differs")
        all_expected.extend(cycle_rows)

    stored_unseen = json.loads((output / "unseen_topology_results.json").read_text())
    traces = load_records(output / "unseen_topology.jsonl.gz")
    if len(stored_unseen) != len(traces):
        raise ValueError("unseen topology trace count differs")
    for row, trace in zip(stored_unseen, traces):
        if {k: v for k, v in trace.items() if k != "episode"} != row:
            raise ValueError("unseen topology row differs")
        unseen_expected.append(row)
    if all_expected != json.loads((output / "raw_test_results.json").read_text()):
        raise ValueError("combined heldout rows differ")
    recomputed = statistics(all_expected, unseen_expected)
    if recomputed != json.loads((output / "statistics.json").read_text()):
        raise ValueError("statistics differ")
    report = {
        "verified": True, "split": split, "seeds": list(seeds),
        "heldout_episodes_reexecuted": checked,
        "heldout_api_calls_reexecuted": calls,
        "unseen_topology_rows_checked": len(unseen_expected),
        "artifact_manifest_sha256": digest(output / "artifact_manifest.json"),
        "auditor_sha256": digest(__file__),
        "checks": [
            "source/data/policy binding", "pre-heldout policy freeze",
            "selection physical-call sums", "capacity", "heldout isolation",
            "complete heldout replay", "latency nonnegativity", "paired statistics",
        ],
        "limits": expected["scope"],
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
