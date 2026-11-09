"""Orchestrate one registered CREDIT workload using supplied real benchmark data.

This entry point refuses incomplete inputs. It does not invent task IDs,
source groups, actor responses, development pairs, or paper measurements.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .paper_7944_preflight import inspect as inspect_preflight
from .paper_7944_protocol import (
    PAPER_7944,
    AttemptBook,
    CatalogFactory,
    registered_halving,
    validate_final_test,
    validate_source_groups,
)
from .paper_7944_runtime import (
    DevelopmentPair,
    FrozenAttemptStore,
    PaperRetriever,
    PhysicalRequestLedger,
)
from .v3.drift import TransportGate
from .v3.selection import BankSelector, Ledger, screen_banks
from .v3.types import Context, Episode, Memory


@dataclass
class RegisteredInputs:
    workload: str
    splits: dict[str, dict[str, list]]
    groups_by_task: dict[str, dict[str, str]]
    memories: tuple[Memory, ...]
    memory_embeddings: dict[str, tuple[float, ...]]
    training: tuple[Episode, ...]
    validation: tuple[Episode, ...]
    calibration: tuple[Episode, ...]
    source_group: Callable[[Episode], str]
    feature_encoder: object
    gate: object
    executor: object
    development_factory: Callable[[str, str, int], Context]
    screening_contexts: dict[str, tuple[Context, ...]]
    acquire_pairs: Callable[[tuple[str, ...], int, int], list]
    rank: Callable[[tuple[str, ...], list], float]
    catalogs: dict[str, tuple[Context, ...]]
    residual_weights: dict[str, float]
    test_contexts: tuple[Context, ...]
    tool_sandbox_task_kinds: dict[str, str] | None
    request_ledger: PhysicalRequestLedger
    output_directory: Path
    seed: int
    attempt_index: int
    incumbent: tuple[str, ...] = ()


def validate_inputs(inputs: RegisteredInputs) -> None:
    validate_source_groups(inputs.splits, inputs.groups_by_task)
    spec = next((row for row in PAPER_7944.workloads if row.name == inputs.workload), None)
    if spec is None:
        raise ValueError("unknown registered workload")
    if len(inputs.memories) != PAPER_7944.source_memories:
        raise ValueError("paper run requires 64 source memories per workload")
    if len({memory.uid for memory in inputs.memories}) != len(inputs.memories):
        raise ValueError("source memory IDs must be unique")
    if len(inputs.catalogs) != spec.protected_guards or set(inputs.catalogs) != set(
            inputs.screening_contexts) or set(inputs.catalogs) != set(inputs.residual_weights):
        raise ValueError("screening, catalog and residual-weight guards differ")
    if any(not contexts for contexts in inputs.screening_contexts.values()):
        raise ValueError("development screening needs actual frozen contexts")
    if not isinstance(inputs.request_ledger, PhysicalRequestLedger):
        raise TypeError("paper execution needs a physical request ledger")
    if inputs.request_ledger.journal_path is None:
        raise ValueError("paper execution needs a durable append-only request journal")
    if not isinstance(inputs.gate, TransportGate):
        raise TypeError("paper execution needs the role transport gate")
    expected_commit = (PAPER_7944.toolsandbox_commit if inputs.workload == "ToolSandbox"
                       else PAPER_7944.tau_bench_commit)
    if (not getattr(inputs.executor, "paper_7944_adapter", False) or
            getattr(inputs.executor, "workload", None) != inputs.workload or
            not str(getattr(inputs.executor, "benchmark_commit", "")).startswith(expected_commit) or
            getattr(inputs.executor, "actor_model", None) != PAPER_7944.actor_model or
            getattr(inputs.executor, "simulator_model", None) != PAPER_7944.simulator_model):
        raise ValueError("real benchmark adapter and frozen provider configuration are required")
    if inputs.attempt_index < 1 or not callable(inputs.acquire_pairs) or not callable(inputs.rank):
        raise ValueError("registered attempt index and development callbacks are required")
    validate_final_test(inputs.workload, inputs.test_contexts,
                        inputs.splits, inputs.groups_by_task,
                        inputs.tool_sandbox_task_kinds)
    certification_ids = set(inputs.splits[inputs.workload]["certification"])
    for guard, rows in inputs.catalogs.items():
        if len(rows) != PAPER_7944.catalog_contexts_per_guard:
            raise ValueError(f"{guard} has an incomplete frozen catalog")
        if any(row.task_id not in certification_ids or
               row.source_group != inputs.groups_by_task[inputs.workload][row.task_id]
               for row in rows):
            raise ValueError(f"{guard} catalog uses tasks outside certification split")
    development_ids = set(inputs.splits[inputs.workload]["development"])
    for rows in inputs.screening_contexts.values():
        if any(row.task_id not in development_ids or
               row.source_group != inputs.groups_by_task[inputs.workload][row.task_id]
               for row in rows):
            raise ValueError("screening context uses tasks outside development split")


def run_registered(inputs: RegisteredInputs) -> dict:
    """Fit, screen 32 banks, halve 1,536 pairs, certify, and test 2,000 episodes."""
    validate_inputs(inputs)
    preflight = inspect_preflight(Path(__file__).resolve().parents[1],
                                  {"splits": inputs.splits,
                                   "groups_by_task": inputs.groups_by_task})
    if not preflight["ready"]:
        raise RuntimeError("paper run preflight failed: " + "; ".join(preflight["issues"]))
    from .paper_7944_model import fit_paper_ensemble

    spec = next(row for row in PAPER_7944.workloads if row.name == inputs.workload)
    ensemble = fit_paper_ensemble(inputs.memories, inputs.training, inputs.validation,
                                  inputs.calibration, source_group=inputs.source_group,
                                  feature_encoder=inputs.feature_encoder)
    retriever = PaperRetriever(inputs.memories, inputs.memory_embeddings)
    factory = CatalogFactory(inputs.catalogs, seed=inputs.seed,
                             attempt_index=inputs.attempt_index,
                             development_factory=inputs.development_factory)
    limits = {"certification": 2 * PAPER_7944.fresh_pairs_per_guard *
              spec.protected_guards, "test": spec.test_episodes}
    ledger = Ledger(limits, request_ledger=inputs.request_ledger,
                    require_request_ids=True)
    mixture = {guard: 1 / spec.protected_guards for guard in inputs.catalogs}
    selector = BankSelector(inputs.memories, (ensemble,), inputs.gate, retriever,
                            factory, inputs.executor, ledger, mixture=mixture,
                            capacity=PAPER_7944.capacity_tokens,
                            radius=PAPER_7944.mixture_l1_radius,
                            cost_penalty=PAPER_7944.storage_penalty_per_token,
                            transported_strata={guard for guard, weight in
                                                 inputs.residual_weights.items() if weight > 0})
    screening = screen_banks(inputs.memories, inputs.incumbent,
                             lambda bank: selector.model_score(inputs.screening_contexts, bank),
                             capacity=PAPER_7944.capacity_tokens, keep=32)
    if len(screening["banks"]) != 32:
        raise ValueError("development screening did not produce 32 distinct banks")
    episode_keys = set()
    used_branch_requests = set()

    def acquire_verified(bank, count, stage):
        rows = list(inputs.acquire_pairs(bank, count, stage))
        for row in rows:
            if (not isinstance(row, DevelopmentPair) or
                    row.candidate_bank != bank or
                    row.incumbent_bank != inputs.incumbent or
                    row.guard not in inputs.catalogs):
                raise ValueError("development pair does not match the registered bank and guard")
            keys = {row.candidate_episode_key, row.incumbent_episode_key}
            if episode_keys & keys:
                raise ValueError("development execution reused an episode key")
            episode_keys.update(keys)
            request_ids = row.candidate_request_ids + row.incumbent_request_ids
            if (len(request_ids) != len(set(request_ids)) or
                    used_branch_requests & set(request_ids)):
                raise ValueError("development branches reused a physical request ID")
            for branch, branch_ids in (("candidate", row.candidate_request_ids),
                                       ("incumbent", row.incumbent_request_ids)):
                for request_id in branch_ids:
                    if request_id not in inputs.request_ledger:
                        raise ValueError("development execution lacks a physical request record")
                    request = inputs.request_ledger[request_id]
                    if (not request.origin_stage.startswith("halving") or
                            request.bank_pair != row.bank_pair_id or
                            request.execution_branch != branch):
                        raise ValueError("development request provenance differs from paired execution")
            used_branch_requests.update(request_ids)
        return rows

    halving = registered_halving(screening["banks"], acquire_verified, inputs.rank)
    book = AttemptBook(initiated=inputs.attempt_index - 1)
    switch = book.attempt(selector, inputs.incumbent, halving["finalist"],
                          workload=inputs.workload,
                          residual_weights=inputs.residual_weights,
                          freeze=FrozenAttemptStore(inputs.output_directory / "frozen"))
    bank = switch["bank"]
    test = []
    for context in inputs.test_contexts:
        exposure = selector.exposures(context, bank)
        episode = ledger.execute("test", inputs.executor, context, exposure)
        if not episode.replay_valid:
            raise ValueError("unsafe final-test execution cannot be reported")
        test.append(episode.outcome)
    inputs.output_directory.mkdir(parents=True, exist_ok=True)
    inputs.request_ledger.save(inputs.output_directory / "physical_requests.jsonl")
    return {"workload": inputs.workload, "final_bank": bank,
            "screened_banks": len(screening["banks"]),
            "development_pairs": halving["development_pair_count"],
            "switch": switch, "final_test_episodes": len(test),
            "final_test_successes": sum(test),
            "physical_requests": inputs.request_ledger.report(),
            "episode_ledger": ledger.report()}
