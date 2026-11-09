"""Strict, data-independent interfaces for the registered paper run.

Embeddings and initial messages must be frozen before a bank is selected. This
module performs local cosine ranking and records physical requests; it does not
call a provider or synthesize benchmark tasks.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .paper_7944_protocol import PAPER_7944
from .v3.types import Context, Memory


def _unit(vector, label):
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be a finite nonempty embedding")
    norm = float(np.linalg.norm(values))
    if not norm:
        raise ValueError(f"{label} must have positive norm")
    return values / norm


def paper_token_ids(content: str) -> tuple[int, ...]:
    """Count stored memory with the paper's frozen o200k_base tokenizer."""
    if version("tiktoken") != "0.9.0":
        raise RuntimeError("paper 7944 requires tiktoken 0.9.0")
    import tiktoken
    return tuple(tiktoken.get_encoding("o200k_base").encode(content))


class PaperRetriever:
    """Frozen dense retrieval with role-scope filtering and at most two records."""

    def __init__(self, memories: tuple[Memory, ...], embeddings: Mapping[str, tuple[float, ...]],
                 *, model: str = PAPER_7944.embedding_model, verify_tokens=True):
        self.memories = {memory.uid: memory for memory in memories}
        if len(self.memories) != len(memories) or set(embeddings) != set(self.memories):
            raise ValueError("every unique memory needs exactly one frozen embedding")
        if model != PAPER_7944.embedding_model:
            raise ValueError("embedding model differs from paper 7944")
        if any(not memory.tokens for memory in memories):
            raise ValueError("memory token IDs must be frozen before ranking")
        if verify_tokens and any(memory.tokens != paper_token_ids(memory.content)
                                 for memory in memories):
            raise ValueError("memory token IDs differ from paper o200k_base counts")
        self.embeddings = {uid: _unit(vector, uid) for uid, vector in embeddings.items()}
        dimensions = {len(value) for value in self.embeddings.values()}
        if len(dimensions) != 1:
            raise ValueError("memory embeddings have inconsistent dimensions")
        self.dimension = dimensions.pop()
        self.model = model
        self.verify_tokens = verify_tokens

    def scores(self, context: Context, bank: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
        if context.initial_user_text is None or context.initial_embedding is None:
            raise ValueError("paper retrieval needs cached initial text and embedding")
        query = _unit(context.initial_embedding, "initial user message")
        if len(query) != self.dimension:
            raise ValueError("query and memory embedding dimensions differ")
        if len(set(bank)) != len(bank) or not set(bank) <= self.memories.keys():
            raise ValueError("bank has duplicate or unknown memory IDs")
        eligible = (self.memories[uid] for uid in bank if any(
            self.memories[uid].eligible(node.role) for node in context.nodes))
        return tuple(sorted(((memory.uid, float(query @ self.embeddings[memory.uid]))
                             for memory in eligible), key=lambda row: (-row[1], row[0])))

    def __call__(self, context: Context, bank: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(uid for uid, _ in self.scores(context, bank)[
            :PAPER_7944.max_exposed_records])


@dataclass(frozen=True)
class PhysicalRequest:
    request_id: str
    origin_stage: str
    bank_pair: str
    execution_branch: str
    provider: str
    status: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None

    def __post_init__(self):
        if (not all((self.request_id, self.origin_stage, self.bank_pair,
                     self.execution_branch, self.provider, self.status)) or
                any(value is not None and (not isinstance(value, int) or value < 0)
                    for value in (self.input_tokens, self.output_tokens))):
            raise ValueError("physical request identity, origin, status and token counts are required")
        expected = {"actor": (PAPER_7944.actor_model, PAPER_7944.actor_temperature),
                    "secondary_actor": (PAPER_7944.secondary_actor_model,
                                        PAPER_7944.actor_temperature),
                    "simulator": (PAPER_7944.simulator_model,
                                  PAPER_7944.simulator_temperature)}
        if self.provider in expected and (self.model != expected[self.provider][0] or
                                          self.temperature != expected[self.provider][1]):
            raise ValueError("actor or simulator request differs from paper model settings")
        if self.provider == "simulator" and self.top_p != 1.0:
            raise ValueError("paper simulator requires top_p=1")
        if self.provider == "embedding" and self.model != PAPER_7944.embedding_model:
            raise ValueError("embedding request differs from paper model settings")


class PhysicalRequestLedger:
    """Count each actual provider request once, including failures and retries."""

    def __init__(self, journal: str | Path | None = None):
        self._requests: dict[str, PhysicalRequest] = {}
        self.journal_path = Path(journal) if journal is not None else None
        if self.journal_path is not None:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("xb"):
                pass

    def charge(self, request: PhysicalRequest) -> None:
        if request.request_id in self._requests:
            raise ValueError("physical request ID was charged twice")
        if self.journal_path is not None:
            payload = (json.dumps(asdict(request), ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("utf-8")
            with self.journal_path.open("ab") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        self._requests[request.request_id] = request

    def charge_many(self, requests) -> None:
        rows = tuple(requests)
        ids = [row.request_id for row in rows]
        if len(ids) != len(set(ids)) or any(uid in self._requests for uid in ids):
            raise ValueError("physical request batch contains a repeated ID")
        for row in rows:
            self.charge(row)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._requests

    def __getitem__(self, request_id: str) -> PhysicalRequest:
        return self._requests[request_id]

    def report(self) -> dict:
        rows = tuple(self._requests.values())
        stages = sorted({row.origin_stage for row in rows})
        return {"physical_requests": len(rows),
                "by_origin_stage": {stage: sum(row.origin_stage == stage for row in rows)
                                    for stage in stages},
                "status_counts": {status: sum(row.status == status for row in rows)
                                  for status in sorted({row.status for row in rows})},
                "input_tokens_recorded": sum(row.input_tokens or 0 for row in rows),
                "output_tokens_recorded": sum(row.output_tokens or 0 for row in rows),
                "requests_with_unknown_tokens": sum(
                    row.input_tokens is None or row.output_tokens is None for row in rows),
                "request_ids_sha256": sha256(json.dumps(sorted(self._requests)).encode()).hexdigest()}

    def records(self) -> tuple[dict, ...]:
        return tuple(asdict(row) for row in self._requests.values())

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":")) + "\n").encode("utf-8")
                           for row in self.records())
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return destination


@dataclass(frozen=True)
class DevelopmentPair:
    """One physically executed candidate/incumbent comparison in development."""

    candidate_bank: tuple[str, ...]
    incumbent_bank: tuple[str, ...]
    bank_pair_id: str
    guard: str
    context_uid: str
    candidate_episode_key: str
    incumbent_episode_key: str
    candidate_outcome: int
    incumbent_outcome: int
    candidate_request_ids: tuple[str, ...]
    incumbent_request_ids: tuple[str, ...]

    def __post_init__(self):
        if (not all((self.bank_pair_id, self.guard, self.context_uid,
                     self.candidate_episode_key, self.incumbent_episode_key)) or
                self.candidate_episode_key == self.incumbent_episode_key or
                self.candidate_outcome not in (0, 1) or
                self.incumbent_outcome not in (0, 1) or
                not self.candidate_request_ids or not self.incumbent_request_ids):
            raise ValueError("development pair requires matched, verified branch executions")


class FrozenAttemptStore:
    """Persist a pre-certification snapshot before any labels are requested."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def __call__(self, snapshot: dict) -> Path:
        workload, index = snapshot["workload"], snapshot["attempt_index"]
        if workload not in {spec.name for spec in PAPER_7944.workloads}:
            raise ValueError("unknown registered workload")
        if not isinstance(index, int) or index < 1:
            raise ValueError("invalid attempt index")
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{workload}-attempt-{index}.json"
        payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return destination
