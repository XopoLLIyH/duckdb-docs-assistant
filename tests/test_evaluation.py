from __future__ import annotations

from duckdb_docs_assistant.evaluation import build_qrels, validate_evaluation

CORPUS = [
    {
        "chunk_id": "chunk-one",
        "source_path": "docs/current/data/csv/overview.md",
        "section": "CSV Loading",
    },
    {
        "chunk_id": "chunk-two",
        "source_path": "docs/current/data/csv/overview.md",
        "section": "Parameters",
    },
]


def _question(query_id: str, language: str) -> dict:
    return {
        "query_id": query_id,
        "intent_id": "csv-loading",
        "language": language,
        "category": "data-import",
        "question": "How do I read a CSV file?",
        "answerable": True,
        "relevant_sources": [
            {
                "source_path": "docs/current/data/csv/overview.md",
                "sections": ["CSV Loading"],
            }
        ],
        "expected_facts": ["Use read_csv."],
    }


def test_valid_bilingual_questions_build_section_level_qrels() -> None:
    questions = [_question("csv-en", "en"), _question("csv-ru", "ru")]

    assert validate_evaluation(questions, CORPUS) == []
    assert build_qrels(questions, CORPUS) == [
        {"query_id": "csv-en", "relevant_chunk_ids": ["chunk-one"]},
        {"query_id": "csv-ru", "relevant_chunk_ids": ["chunk-one"]},
    ]


def test_validator_rejects_unknown_sections() -> None:
    questions = [_question("csv-en", "en"), _question("csv-ru", "ru")]
    questions[0]["relevant_sources"][0]["sections"] = ["Missing"]

    errors = validate_evaluation(questions, CORPUS)

    assert any("unknown section" in error for error in errors)
