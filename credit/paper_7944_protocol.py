"""Registered settings from paper 7944; this module does not generate results.

Keep these settings separate from the earlier local and CV20 experiments. A
paper result requires matching execution records and a physical-request ledger.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random

from .v3.types import Context


@dataclass(frozen=True)
class Workload:
    name: str
    test_task_ids: int
    test_episodes: int
    protected_guards: int
    prediction_draws_per_guard: int


@dataclass(frozen=True)
class PaperProtocol:
    capacity_tokens: int = 800
    source_memories: int = 64
    max_exposed_records: int = 2
    storage_penalty_per_token: float = 1.25e-5
    mixture_l1_radius: float = 0.25
    candidate_counts: tuple[int, ...] = (32, 16, 8, 4, 2, 1)
    new_pairs_per_bank: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    catalog_contexts_per_guard: int = 256
    fresh_pairs_per_guard: int = 200
    noninferiority_margin: float = 0.05
    lifetime_error: float = 0.05
    probe_minimum: int = 80
    role_source_support_minimum: int = 30
    probe_difference_upper_limit: float = 0.05
    ensemble_members: int = 5
    ensemble_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    training_episodes_per_workload: int = 1200
    validation_episodes_per_workload: int = 300
    calibration_episodes_per_workload: int = 300
    hidden_units: int = 128
    hidden_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 50
    early_stop_patience: int = 8
    interaction_l1: float = 1e-4
    actor_model: str = "gpt-4.1-mini-2025-04-14"
    secondary_actor_model: str = "gpt-4.1-2025-04-14"
    simulator_model: str = "gpt-4o-2024-08-06"
    embedding_model: str = "text-embedding-3-large"
    actor_temperature: float = 0.0
    simulator_temperature: float = 0.7
    toolsandbox_commit: str = "165848b"
    tau_bench_commit: str = "59a200c6d575d595120f1cb70fea53cef0632f6b"
    workloads: tuple[Workload, ...] = (
        Workload("ToolSandbox", 80, 2000, 6, 2000),
        Workload("Retail", 70, 2000, 10, 4000),
        Workload("Airline", 25, 2000, 7, 3000),
    )

    def validate(self) -> None:
        if len(self.candidate_counts) != len(self.new_pairs_per_bank):
            raise ValueError("search schedule lengths differ")
        if self.candidate_counts[-1] != 1:
            raise ValueError("exactly one finalist must enter certification")
        if any(n * pairs != 256 for n, pairs in zip(
                self.candidate_counts, self.new_pairs_per_bank)):
            raise ValueError("each search stage must acquire 256 new pairs")
        if self.fresh_pairs_per_guard != 200 or self.catalog_contexts_per_guard != 256:
            raise ValueError("registered certification sample sizes changed")
        if any(w.test_episodes != 2000 for w in self.workloads):
            raise ValueError("registered final-test horizon changed")
        if len(self.ensemble_seeds) != self.ensemble_members:
            raise ValueError("one frozen seed per ensemble member is required")

    def manifest(self) -> dict:
        self.validate()
        return {**asdict(self), "new_search_pairs_total": sum(
            n * pairs for n, pairs in zip(self.candidate_counts, self.new_pairs_per_bank)),
            "certification_rule": "one frozen finalist; alpha_t=0.05/[t(t+1)]; "
                                  "half to aggregate, half split over guards",
            "evidence_status": "protocol only; no paper outcome records generated"}


PAPER_7944 = PaperProtocol()


def attempt_alpha(attempt_index: int) -> float:
    if not isinstance(attempt_index, int) or attempt_index < 1:
        raise ValueError("attempt_index must be a positive integer")
    return PAPER_7944.lifetime_error / (attempt_index * (attempt_index + 1))


class CatalogFactory:
    """Sample fixed guard catalogs with replacement for paired certification.

    The same returned Context is used for both banks in a labeled pair. Unique
    episode IDs and seeds keep repeated catalog indices as fresh executions.
    Labeled and prediction-only phases have disjoint random streams.
    """

    def __init__(self, catalogs: dict[str, tuple[Context, ...]], *, seed: int,
                 attempt_index: int, development_factory=None):
        if (not catalogs or not isinstance(seed, int) or
                not isinstance(attempt_index, int) or attempt_index < 1):
            raise ValueError("registered catalogs, seed, and attempt index are required")
        for guard, rows in catalogs.items():
            if len(rows) != PAPER_7944.catalog_contexts_per_guard:
                raise ValueError(f"{guard} must have exactly 256 frozen contexts")
            if len({row.uid for row in rows}) != len(rows) or any(
                    row.stratum != guard for row in rows):
                raise ValueError("catalog context IDs and guard labels are invalid")
        self.catalogs = {guard: tuple(rows) for guard, rows in catalogs.items()}
        self.seed = seed
        self.attempt_index = attempt_index
        self.development_factory = development_factory

    def __call__(self, guard: str, phase: str, index: int) -> Context:
        if guard not in self.catalogs or not isinstance(index, int) or index < 0:
            raise ValueError("unknown guard or negative draw index")
        if phase not in {"certification", "certification-unlabeled"}:
            if self.development_factory is None:
                raise ValueError("a separate development factory is required")
            return self.development_factory(guard, phase, index)
        key = f"{self.seed}|{self.attempt_index}|{guard}|{phase}|{index}".encode("utf-8")
        draw_seed = int.from_bytes(sha256(key).digest()[:8], "big")
        chosen = Random(draw_seed).randrange(PAPER_7944.catalog_contexts_per_guard)
        source = self.catalogs[guard][chosen]
        execution_seed = int.from_bytes(sha256(key + b"|execution").digest()[:8], "big")
        return replace(source, uid=f"attempt-{self.attempt_index}:{phase}:{guard}:{index}:{source.uid}",
                       seed=execution_seed)


class AttemptBook:
    """Advance the lifetime test index before every initiated bank update."""

    def __init__(self, *, initiated: int = 0):
        if not isinstance(initiated, int) or initiated < 0:
            raise ValueError("initiated attempt count must be nonnegative")
        self.initiated = initiated

    def attempt(self, selector, incumbent, finalist, *, workload, freeze, pilot_n=100):
        self.initiated += 1
        return selector.switch_registered(incumbent, finalist, workload=workload,
                                          attempt_index=self.initiated, freeze=freeze,
                                          pilot_n=pilot_n)


def registered_halving(candidates, acquire_pairs, rank):
    """Run the paper's six-stage, pooled-evidence development search.

    ``acquire_pairs(bank, n, stage)`` must physically execute and return ``n``
    new candidate/incumbent pair records. ``rank(bank, history)`` evaluates the
    retrieval-aware development objective on all records accumulated so far.
    Neither function may use certification or final-test observations.
    """
    protocol = PAPER_7944
    survivors = list(dict.fromkeys(tuple(sorted(bank)) for bank in candidates))
    if len(survivors) != protocol.candidate_counts[0]:
        raise ValueError("screening must supply exactly 32 distinct banks")
    histories = {bank: [] for bank in survivors}
    stage_reports = []
    for stage, (expected, new_pairs) in enumerate(zip(
            protocol.candidate_counts, protocol.new_pairs_per_bank)):
        if len(survivors) != expected:
            raise ValueError("registered survivor schedule changed")
        for bank in survivors:
            fresh = list(acquire_pairs(bank, new_pairs, stage))
            if len(fresh) != new_pairs:
                raise ValueError("incomplete paired execution stage")
            histories[bank].extend(fresh)
        scores = {bank: float(rank(bank, histories[bank])) for bank in survivors}
        if any(not math.isfinite(value) for value in scores.values()):
            raise ValueError("development ranking produced a nonfinite score")
        scored = sorted(survivors, key=lambda bank: (-scores[bank], bank))
        next_count = protocol.candidate_counts[stage + 1] if stage + 1 < len(
            protocol.candidate_counts) else 1
        survivors = scored[:next_count]
        stage_reports.append({"stage": stage, "new_pairs_per_bank": new_pairs,
                              "pair_count": expected * new_pairs,
                              "survivors": survivors})
    return {"finalist": survivors[0], "stages": stage_reports,
            "development_pair_count": sum(row["pair_count"] for row in stage_reports),
            "finalist_history": histories[survivors[0]]}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the frozen protocol JSON")
    args = parser.parse_args()
    data = json.dumps(PAPER_7944.manifest(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
