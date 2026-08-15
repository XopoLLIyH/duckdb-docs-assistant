from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from collections.abc import Iterable
from typing import Any

LENGTH_BUCKETS = (
    (0, 255, "0000-0255"),
    (256, 511, "0256-0511"),
    (512, 1023, "0512-1023"),
    (1024, 1535, "1024-1535"),
    (1536, 2047, "1536-2047"),
    (2048, 10**9, "2048+"),
)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _category(source_path: str) -> str:
    prefixes = (
        "docs/current/clients/python/",
        "docs/current/clients/cli/",
        "docs/current/core_extensions/",
        "docs/current/operations_manual/",
        "docs/current/configuration/",
        "docs/current/connect/",
        "docs/current/data/",
        "docs/current/guides/",
        "docs/current/sql/",
    )
    for prefix in prefixes:
        if source_path.startswith(prefix):
            return prefix.removeprefix("docs/current/").rstrip("/")
    return "other"


def analyze_corpus(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    lengths = [len(row["text"]) for row in rows]
    source_paths = {row["source_path"] for row in rows}
    code_chunks = sum(bool(row.get("code_blocks")) for row in rows)
    categories = Counter(_category(row["source_path"]) for row in rows)
    documents_by_category = Counter(_category(path) for path in source_paths)
    document_chunk_counts = Counter(row["source_path"] for row in rows)
    text_hashes = Counter(
        hashlib.sha256(" ".join(row["text"].split()).encode("utf-8")).hexdigest()
        for row in rows
    )

    buckets = {
        label: sum(lower <= length <= upper for length in lengths)
        for lower, upper, label in LENGTH_BUCKETS
    }
    duplicates = sum(count - 1 for count in text_hashes.values() if count > 1)

    return {
        "chunks": len(rows),
        "documents": len(source_paths),
        "characters": sum(lengths),
        "chunk_length": {
            "min": min(lengths, default=0),
            "mean": round(statistics.fmean(lengths), 1) if lengths else 0,
            "median": round(statistics.median(lengths), 1) if lengths else 0,
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths, default=0),
            "buckets": buckets,
        },
        "code_chunks": code_chunks,
        "code_chunk_rate": round(code_chunks / len(rows), 4) if rows else 0,
        "empty_text_chunks": sum(not row["text"].strip() for row in rows),
        "empty_source_urls": sum(not row.get("source_url", "").strip() for row in rows),
        "duplicate_chunk_ids": len(rows) - len({row["chunk_id"] for row in rows}),
        "duplicate_normalized_texts": duplicates,
        "chunks_by_category": dict(sorted(categories.items())),
        "documents_by_category": dict(sorted(documents_by_category.items())),
        "largest_documents": [
            {"source_path": path, "chunks": count}
            for path, count in document_chunk_counts.most_common(10)
        ],
        "source_commits": sorted({row["source_commit"] for row in rows}),
        "versions": sorted({row["version"] for row in rows}),
    }


def render_markdown_report(analysis: dict[str, Any]) -> str:
    length = analysis["chunk_length"]
    lines = [
        "# Corpus analysis",
        "",
        "## Summary",
        "",
        f"- Indexed documents: {analysis['documents']}",
        f"- Chunks: {analysis['chunks']}",
        f"- Characters: {analysis['characters']:,}",
        (
            f"- Chunks with code: {analysis['code_chunks']} "
            f"({analysis['code_chunk_rate']:.1%})"
        ),
        f"- Duplicate chunk IDs: {analysis['duplicate_chunk_ids']}",
        f"- Duplicate normalized texts: {analysis['duplicate_normalized_texts']}",
        "",
        "## Chunk length",
        "",
        "| Metric | Characters |",
        "|---|---:|",
        f"| Minimum | {length['min']} |",
        f"| Mean | {length['mean']} |",
        f"| Median | {length['median']} |",
        f"| P90 | {length['p90']} |",
        f"| P95 | {length['p95']} |",
        f"| Maximum | {length['max']} |",
        "",
        "## Corpus composition",
        "",
        "| Category | Documents | Chunks |",
        "|---|---:|---:|",
    ]
    categories = sorted(
        set(analysis["documents_by_category"]) | set(analysis["chunks_by_category"])
    )
    for category in categories:
        lines.append(
            f"| `{category}` | {analysis['documents_by_category'].get(category, 0)} | "
            f"{analysis['chunks_by_category'].get(category, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Largest documents",
            "",
            "| Source | Chunks |",
            "|---|---:|",
        ]
    )
    for item in analysis["largest_documents"]:
        lines.append(f"| `{item['source_path']}` | {item['chunks']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The corpus combines exact SQL, CLI and API identifiers with explanatory prose. "
                "This makes it suitable for comparing BM25, dense retrieval and their fusion. "
                "The bilingual evaluation set should be reported separately by language because "
                "English BM25 is not expected to retrieve English documentation reliably from "
                "Russian queries without translation."
            ),
            "",
        ]
    )
    return "\n".join(lines)
