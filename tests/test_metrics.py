from __future__ import annotations

import pytest

from duckdb_docs_assistant.metrics import aggregate_metrics, ranking_metrics


def test_ranking_metrics_use_binary_qrels() -> None:
    metrics = ranking_metrics(["wrong", "rel-a", "rel-b"], {"rel-a", "rel-b"}, [1, 3])

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@3"] == 0.5
    assert metrics["ndcg@3"] == pytest.approx(
        (1 / 1.5849625007 + 1 / 2) / (1 + 1 / 1.5849625007)
    )


def test_aggregate_metrics_separates_languages() -> None:
    rows = [
        {
            "language": "en",
            "retrieved_ids": ["a"],
            "metrics": {"recall@1": 1.0, "mrr@1": 1.0, "ndcg@1": 1.0},
        },
        {
            "language": "ru",
            "retrieved_ids": [],
            "metrics": {"recall@1": 0.0, "mrr@1": 0.0, "ndcg@1": 0.0},
        },
    ]

    result = aggregate_metrics(rows, [1])

    assert result["overall"]["recall@1"] == 0.5
    assert result["en"]["recall@1"] == 1.0
    assert result["ru"]["recall@1"] == 0.0
