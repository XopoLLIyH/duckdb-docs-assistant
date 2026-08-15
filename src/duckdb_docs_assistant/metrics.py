from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def ranking_metrics(
    retrieved_ids: list[str], relevant_ids: set[str], cutoffs: Iterable[int]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        retrieved = retrieved_ids[:cutoff]
        relevant_retrieved = sum(chunk_id in relevant_ids for chunk_id in retrieved)
        metrics[f"recall@{cutoff}"] = (
            relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
        )

        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break
        metrics[f"mrr@{cutoff}"] = reciprocal_rank

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, chunk_id in enumerate(retrieved, start=1)
            if chunk_id in relevant_ids
        )
        ideal_hits = min(cutoff, len(relevant_ids))
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        metrics[f"ndcg@{cutoff}"] = dcg / ideal_dcg if ideal_dcg else 0.0
    return metrics


def aggregate_metrics(
    per_query: Iterable[dict[str, Any]], cutoffs: Iterable[int]
) -> dict[str, dict[str, float | int]]:
    rows = list(per_query)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[row["language"]].append(row)

    metric_names = [
        f"{metric}@{cutoff}"
        for cutoff in cutoffs
        for metric in ("recall", "mrr", "ndcg")
    ]
    result: dict[str, dict[str, float | int]] = {}
    for group_name in ("overall", "en", "ru"):
        group_rows = groups.get(group_name, [])
        aggregated: dict[str, float | int] = {
            "queries": len(group_rows),
            "queries_with_results": sum(bool(row["retrieved_ids"]) for row in group_rows),
        }
        for metric_name in metric_names:
            values = [row["metrics"][metric_name] for row in group_rows]
            aggregated[metric_name] = round(statistics.fmean(values), 4) if values else 0.0
        result[group_name] = aggregated
    return result
