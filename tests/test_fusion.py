from __future__ import annotations

from dataclasses import dataclass

import pytest

from duckdb_docs_assistant.fusion import detect_query_language, reciprocal_rank_fusion


@dataclass(frozen=True)
class Item:
    chunk_id: str
    rank: int


def test_rrf_rewards_candidates_present_in_both_lists() -> None:
    rankings = {
        "bm25": [Item("lexical", 1), Item("shared", 2)],
        "dense": [Item("semantic", 1), Item("shared", 2)],
    }

    results = reciprocal_rank_fusion(rankings, rrf_k=60, top_k=3)

    assert [result.chunk_id for result in results] == ["shared", "lexical", "semantic"]
    assert results[0].source_ranks == {"bm25": 2, "dense": 2}


def test_rrf_supports_source_weights_and_deterministic_ties() -> None:
    rankings = {
        "bm25": [Item("b", 1)],
        "dense": [Item("a", 1)],
    }

    tied = reciprocal_rank_fusion(rankings, rrf_k=0, top_k=2)
    weighted = reciprocal_rank_fusion(
        rankings, rrf_k=0, weights={"bm25": 2.0, "dense": 1.0}, top_k=2
    )

    assert [result.chunk_id for result in tied] == ["a", "b"]
    assert [result.chunk_id for result in weighted] == ["b", "a"]


def test_rrf_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion({}, rrf_k=-1)
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion(
            {"bm25": [Item("a", 1)]}, weights={"bm25": -1.0}
        )


@pytest.mark.parametrize(
    ("query", "language"),
    [
        ("How do I use read_csv?", "en"),
        ("Как использовать read_csv?", "ru"),
        ("memory_limit", "en"),
    ],
)
def test_detect_query_language(query: str, language: str) -> None:
    assert detect_query_language(query) == language
