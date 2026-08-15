from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def validate_evaluation(
    questions: Iterable[dict[str, Any]], corpus: Iterable[dict[str, Any]]
) -> list[str]:
    question_rows = list(questions)
    corpus_rows = list(corpus)
    errors: list[str] = []
    query_ids: set[str] = set()
    intent_languages: dict[str, set[str]] = defaultdict(set)
    sections_by_path: dict[str, set[str]] = defaultdict(set)
    for chunk in corpus_rows:
        sections_by_path[chunk["source_path"]].add(chunk["section"])

    for row_number, question in enumerate(question_rows, start=1):
        prefix = f"row {row_number}"
        query_id = question.get("query_id", "")
        if not query_id:
            errors.append(f"{prefix}: missing query_id")
        elif query_id in query_ids:
            errors.append(f"{prefix}: duplicate query_id {query_id}")
        query_ids.add(query_id)

        language = question.get("language")
        if language not in {"en", "ru"}:
            errors.append(f"{prefix}: language must be en or ru")
        intent_id = question.get("intent_id", "")
        if not intent_id:
            errors.append(f"{prefix}: missing intent_id")
        elif language:
            intent_languages[intent_id].add(language)

        if not question.get("question", "").strip():
            errors.append(f"{prefix}: empty question")
        if not question.get("expected_facts"):
            errors.append(f"{prefix}: expected_facts must not be empty")

        relevant_sources = question.get("relevant_sources", [])
        if question.get("answerable") is True and not relevant_sources:
            errors.append(f"{prefix}: answerable question has no relevant_sources")
        if question.get("answerable") is False and relevant_sources:
            errors.append(f"{prefix}: unanswerable question has relevant_sources")
        if not isinstance(question.get("answerable"), bool):
            errors.append(f"{prefix}: answerable must be boolean")

        for source in relevant_sources:
            path = source.get("source_path", "")
            if path not in sections_by_path:
                errors.append(f"{prefix}: unknown source_path {path}")
                continue
            for section in source.get("sections", []):
                if section not in sections_by_path[path]:
                    errors.append(f"{prefix}: unknown section {section!r} in {path}")

    for intent_id, languages in intent_languages.items():
        if languages != {"en", "ru"}:
            errors.append(f"intent {intent_id}: expected en and ru, got {sorted(languages)}")
    return errors


def build_qrels(
    questions: Iterable[dict[str, Any]], corpus: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    corpus_rows = list(corpus)
    result: list[dict[str, Any]] = []
    for question in questions:
        relevant_ids: set[str] = set()
        for source in question.get("relevant_sources", []):
            sections = set(source.get("sections", []))
            for chunk in corpus_rows:
                if chunk["source_path"] != source["source_path"]:
                    continue
                if sections and chunk["section"] not in sections:
                    continue
                relevant_ids.add(chunk["chunk_id"])
        result.append(
            {
                "query_id": question["query_id"],
                "relevant_chunk_ids": sorted(relevant_ids),
            }
        )
    return result
