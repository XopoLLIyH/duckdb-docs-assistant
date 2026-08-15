from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


class RankedItem(Protocol):
    chunk_id: str
    rank: int


@dataclass(frozen=True)
class FusionResult:
    chunk_id: str
    score: float
    rank: int
    source_ranks: dict[str, int]


def detect_query_language(query: str) -> str:
    """Route Cyrillic queries to Russian weights; default to English otherwise."""
    return "ru" if CYRILLIC_RE.search(query) else "en"


def reciprocal_rank_fusion(
    rankings: Mapping[str, Iterable[RankedItem]],
    *,
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
    top_k: int = 10,
) -> list[FusionResult]:
    """Fuse ranked lists without requiring their raw scores to be comparable."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if top_k <= 0:
        return []

    resolved_weights = dict(weights or {})
    scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}

    for source, ranking in rankings.items():
        weight = resolved_weights.get(source, 1.0)
        if weight < 0:
            raise ValueError("RRF weights must be non-negative")
        seen: set[str] = set()
        for item in ranking:
            if item.rank <= 0:
                raise ValueError("Ranks must be positive")
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + weight / (
                rrf_k + item.rank
            )
            source_ranks.setdefault(item.chunk_id, {})[source] = item.rank

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [
        FusionResult(
            chunk_id=chunk_id,
            score=round(scores[chunk_id], 10),
            rank=rank,
            source_ranks=source_ranks[chunk_id],
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]
