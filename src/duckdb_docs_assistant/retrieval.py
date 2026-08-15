from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

TOKEN_RE = re.compile(r"(?u)\b\w+(?:[.+-]\w+)*\b")


def tokenize(text: str) -> list[str]:
    """Tokenize prose while preserving identifiers such as read_csv and duckdb.sql."""
    return TOKEN_RE.findall(text.lower())


def searchable_text(chunk: dict[str, Any]) -> str:
    heading_path = " ".join(chunk.get("heading_path", []))
    return "\n".join((chunk.get("title", ""), heading_path, chunk["text"]))


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    score: float
    rank: int


class BM25Index:
    """Small in-memory Okapi BM25 index with an inverted posting list."""

    def __init__(
        self,
        documents: Iterable[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_lengths: list[int] = []
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)

        for doc_index, document in enumerate(self.documents):
            tokens = tokenize(searchable_text(document))
            self.doc_lengths.append(len(tokens))
            for term, frequency in Counter(tokens).items():
                self.postings[term][doc_index] = frequency

        self.document_count = len(self.documents)
        self.average_doc_length = (
            sum(self.doc_lengths) / self.document_count if self.document_count else 0.0
        )
        self.idf = {
            term: math.log(
                1.0
                + (self.document_count - len(posting) + 0.5) / (len(posting) + 0.5)
            )
            for term, posting in self.postings.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if top_k <= 0 or not self.document_count:
            return []

        scores: dict[int, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))
        for term, query_frequency in query_terms.items():
            posting = self.postings.get(term)
            if not posting:
                continue
            query_weight = 1.0 + math.log(query_frequency)
            idf = self.idf[term]
            for doc_index, term_frequency in posting.items():
                length_ratio = self.doc_lengths[doc_index] / self.average_doc_length
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                scores[doc_index] += (
                    idf
                    * query_weight
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                )

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.documents[item[0]]["chunk_id"]),
        )[:top_k]
        return [
            SearchResult(
                chunk_id=self.documents[doc_index]["chunk_id"],
                score=round(score, 8),
                rank=rank,
            )
            for rank, (doc_index, score) in enumerate(ranked, start=1)
            if score > 0
        ]
