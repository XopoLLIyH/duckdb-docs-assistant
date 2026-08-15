from __future__ import annotations

from duckdb_docs_assistant.analysis import analyze_corpus


def test_analyze_corpus_reports_quality_and_categories() -> None:
    records = [
        {
            "chunk_id": "one",
            "text": "DuckDB reads CSV files.",
            "code_blocks": [],
            "source_path": "docs/current/data/csv/overview.md",
            "source_url": "https://example.test/csv",
            "source_commit": "abc",
            "version": "current",
        },
        {
            "chunk_id": "two",
            "text": "SELECT * FROM read_csv('file.csv');",
            "code_blocks": ["SELECT * FROM read_csv('file.csv');"],
            "source_path": "docs/current/data/csv/overview.md",
            "source_url": "https://example.test/csv",
            "source_commit": "abc",
            "version": "current",
        },
    ]

    result = analyze_corpus(records)

    assert result["chunks"] == 2
    assert result["documents"] == 1
    assert result["code_chunks"] == 1
    assert result["chunks_by_category"] == {"data": 2}
    assert result["duplicate_chunk_ids"] == 0
