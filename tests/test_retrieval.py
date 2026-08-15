from __future__ import annotations

from duckdb_docs_assistant.retrieval import BM25Index, tokenize


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "title": "DuckDB",
        "heading_path": [],
        "text": text,
    }


def test_tokenizer_preserves_technical_identifiers() -> None:
    assert tokenize("Use read_csv and duckdb.sql with Python 3.12") == [
        "use",
        "read_csv",
        "and",
        "duckdb.sql",
        "with",
        "python",
        "3.12",
    ]


def test_bm25_ranks_exact_technical_match_first() -> None:
    index = BM25Index(
        [
            _chunk("csv", "Read a CSV file with the read_csv function."),
            _chunk("json", "Read JSON documents with read_json."),
        ]
    )

    results = index.search("How do I use read_csv?", top_k=2)

    assert results[0].chunk_id == "csv"
    assert len(results) == 1


def test_bm25_returns_no_results_without_lexical_overlap() -> None:
    index = BM25Index([_chunk("csv", "Read a CSV file")])

    assert index.search("несвязанный русский запрос") == []
