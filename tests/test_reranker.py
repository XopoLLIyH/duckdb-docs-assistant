from __future__ import annotations

import numpy as np
import pytest

from duckdb_docs_assistant.reranker import RerankCandidate, rerank


class FakeScorer:
    device = "cpu"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> np.ndarray:
        self.pairs = pairs
        return np.asarray(self.scores, dtype=np.float32)


def _chunks() -> dict[str, dict]:
    return {
        "a": {"title": "A", "heading_path": ["Alpha"], "text": "first"},
        "b": {"title": "B", "heading_path": ["Beta"], "text": "second"},
    }


def test_rerank_orders_by_cross_encoder_score() -> None:
    candidates = [
        RerankCandidate("a", rank=1, score=0.9),
        RerankCandidate("b", rank=2, score=0.8),
    ]
    scorer = FakeScorer([0.1, 0.9])

    results = rerank(
        "question", candidates, _chunks(), scorer, batch_size=2, top_k=2
    )

    assert [result.chunk_id for result in results] == ["b", "a"]
    assert results[0].retrieval_rank == 2
    assert scorer.pairs[0] == ("question", "A\nAlpha\nfirst")


def test_rerank_uses_retrieval_rank_as_tie_breaker() -> None:
    candidates = [
        RerankCandidate("a", rank=1, score=0.9),
        RerankCandidate("b", rank=2, score=0.8),
    ]

    results = rerank(
        "question", candidates, _chunks(), FakeScorer([0.5, 0.5]), batch_size=2, top_k=2
    )

    assert [result.chunk_id for result in results] == ["a", "b"]


def test_rerank_rejects_unknown_or_duplicate_candidates() -> None:
    duplicate = [
        RerankCandidate("a", rank=1, score=0.9),
        RerankCandidate("a", rank=2, score=0.8),
    ]
    with pytest.raises(ValueError, match="unique"):
        rerank("question", duplicate, _chunks(), FakeScorer([]), batch_size=2, top_k=2)

    unknown = [RerankCandidate("missing", rank=1, score=0.9)]
    with pytest.raises(ValueError, match="Unknown"):
        rerank("question", unknown, _chunks(), FakeScorer([]), batch_size=1, top_k=1)
