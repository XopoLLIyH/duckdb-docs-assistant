from __future__ import annotations

import json

import pytest

from duckdb_docs_assistant.generation import (
    GenerationValidationError,
    GroundedAnswer,
    build_context,
    build_user_prompt,
    estimate_tokens,
    render_answer,
    validate_answer,
)


def _chunks() -> dict[str, dict]:
    return {
        "a": {
            "title": "Reading Parquet",
            "section": "Examples",
            "source_url": "https://duckdb.org/a",
            "text": "Use this SQL:\n\n```sql\nSELECT * FROM read_parquet('*.parquet');\n```",
        },
        "b": {
            "title": "Reading Parquet copy",
            "section": "Examples",
            "source_url": "https://duckdb.org/b",
            "text": "Use this SQL:\n\n```sql\nSELECT * FROM read_parquet('*.parquet');\n```",
        },
        "c": {
            "title": "Long section",
            "section": "Reference",
            "source_url": "https://duckdb.org/c",
            "text": "```sql\n" + "SELECT 1;\n" * 200 + "```",
        },
        "d": {
            "title": "Broken Markdown",
            "section": "Example",
            "source_url": "https://duckdb.org/d",
            "text": "```sql\nSELECT 42;",
        },
    }


def test_build_context_deduplicates_and_labels_sources() -> None:
    bundle = build_context(
        "How?", ["a", "b", "c"], _chunks(), token_budget=220, max_sources=3
    )

    assert [source.source_id for source in bundle.sources] == ["S1", "S2"]
    assert [source.chunk_id for source in bundle.sources] == ["a", "c"]
    assert bundle.sources[1].truncated is True
    assert bundle.sources[1].text.count("```") % 2 == 0
    assert bundle.estimated_tokens <= 220
    assert "[S1]" in build_user_prompt(bundle)


def test_build_context_rejects_unknown_chunk() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        build_context("How?", ["missing"], _chunks(), token_budget=100, max_sources=1)


def test_build_context_closes_unbalanced_source_fence() -> None:
    source = build_context(
        "How?", ["d"], _chunks(), token_budget=100, max_sources=1
    ).sources[0]

    assert source.text.endswith("```")
    assert source.text.count("```") == 2


def test_token_estimate_is_positive_for_multilingual_text() -> None:
    assert estimate_tokens("DuckDB") > 0
    assert estimate_tokens("Как использовать DuckDB?") > estimate_tokens("DuckDB")


def test_validate_and_render_grounded_answer() -> None:
    raw = json.dumps(
        {
            "status": "answered",
            "answer": "Use `read_parquet` [S1].",
            "citations": ["S1"],
        }
    )
    answer = validate_answer(raw, {"S1", "S2"})
    source = build_context(
        "How?", ["a"], _chunks(), token_budget=200, max_sources=1
    ).sources

    rendered = render_answer(answer, source)

    assert answer == GroundedAnswer("answered", "Use `read_parquet` [S1].", ["S1"])
    assert "https://duckdb.org/a" in rendered


def test_validate_refusal_without_citations() -> None:
    raw = json.dumps(
        {
            "status": "insufficient_context",
            "answer": "В источниках нет ответа.",
            "citations": [],
        }
    )

    answer = validate_answer(raw, {"S1"})

    assert render_answer(answer, []) == "В источниках нет ответа."


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "answered", "answer": "No citation.", "citations": []},
        {"status": "answered", "answer": "Claim [S2].", "citations": ["S2"]},
        {"status": "answered", "answer": "Claim [S1].", "citations": []},
        {
            "status": "insufficient_context",
            "answer": "No answer [S1].",
            "citations": ["S1"],
        },
    ],
)
def test_validate_rejects_ungrounded_outputs(payload: dict) -> None:
    with pytest.raises(GenerationValidationError):
        validate_answer(json.dumps(payload), {"S1"})
