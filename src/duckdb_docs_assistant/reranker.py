from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .retrieval import searchable_text

FloatVector = NDArray[np.float32]


class PairScorer(Protocol):
    device: str

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> FloatVector: ...


class CrossEncoderScorer:
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str,
        cache_folder: Path,
        max_length: int,
    ) -> None:
        from sentence_transformers import CrossEncoder

        resolved_device = None if device == "auto" else device
        self.model = CrossEncoder(
            model_name,
            revision=revision,
            device=resolved_device,
            cache_folder=str(cache_folder),
            max_length=max_length,
        )
        self.device = str(self.model.device)

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> FloatVector:
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


@dataclass(frozen=True)
class RerankCandidate:
    chunk_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class RerankResult:
    chunk_id: str
    score: float
    rank: int
    retrieval_rank: int
    retrieval_score: float


def rerank(
    query: str,
    candidates: list[RerankCandidate],
    chunks_by_id: dict[str, dict[str, Any]],
    scorer: PairScorer,
    *,
    batch_size: int,
    top_k: int,
) -> list[RerankResult]:
    if top_k <= 0 or not candidates:
        return []
    if len({candidate.chunk_id for candidate in candidates}) != len(candidates):
        raise ValueError("Rerank candidates must have unique chunk IDs")

    try:
        passages = [searchable_text(chunks_by_id[item.chunk_id]) for item in candidates]
    except KeyError as error:
        raise ValueError(f"Unknown candidate chunk ID: {error.args[0]}") from error
    scores = scorer.predict(
        [(query, passage) for passage in passages], batch_size=batch_size
    )
    if scores.shape != (len(candidates),):
        raise ValueError(
            f"Unexpected reranker score shape {scores.shape}; expected {(len(candidates),)}"
        )

    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].rank, item[0].chunk_id),
    )[:top_k]
    return [
        RerankResult(
            chunk_id=candidate.chunk_id,
            score=round(float(score), 8),
            rank=rank,
            retrieval_rank=candidate.rank,
            retrieval_score=candidate.score,
        )
        for rank, (candidate, score) in enumerate(ranked, start=1)
    ]
