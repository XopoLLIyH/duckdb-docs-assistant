from __future__ import annotations

from duckdb_docs_assistant.corpus import chunk_markdown, select_source_paths

CONFIG = {
    "product": "DuckDB",
    "version_label": "current",
    "source_repo": "https://github.com/duckdb/duckdb-web",
    "source_commit": "abc123",
    "license": "MIT",
    "include": ["docs/current/clients/python/*.md"],
    "exclude": ["**/reference/**"],
    "chunking": {"max_chars": 500, "overlap_chars": 50, "min_chars": 20},
}


def test_select_source_paths_applies_scope_and_exclusions() -> None:
    tree = [
        {"type": "blob", "path": "docs/current/clients/python/overview.md"},
        {"type": "blob", "path": "docs/current/clients/python/reference/api.md"},
        {"type": "blob", "path": "docs/current/clients/java.md"},
        {"type": "tree", "path": "docs/current/clients/python"},
    ]

    assert select_source_paths(tree, CONFIG) == [
        "docs/current/clients/python/overview.md"
    ]


def test_chunk_markdown_preserves_headings_code_and_provenance() -> None:
    markdown = """---
title: Python API
---
# Connecting

Open a connection to an in-memory database using the documented function.

```python
import duckdb
duckdb.connect()
```

## Read-only mode

Use the read-only flag when concurrent processes only need to query a database.
"""

    chunks = chunk_markdown(markdown, "docs/current/clients/python/overview.md", CONFIG)

    assert len(chunks) == 2
    assert chunks[0]["title"] == "Python API"
    assert chunks[0]["heading_path"] == ["Connecting"]
    assert "duckdb.connect()" in chunks[0]["code_blocks"][0]
    assert chunks[1]["heading_path"] == ["Connecting", "Read-only mode"]
    assert chunks[0]["source_commit"] == "abc123"
    assert chunks[0]["source_url"].startswith("https://duckdb.org/docs/stable/")


def test_repeated_headings_have_unique_ids() -> None:
    markdown = """# Example

First section with enough explanatory text to become an indexed chunk.

# Example

Second section with enough explanatory text to become an indexed chunk.
"""

    chunks = chunk_markdown(markdown, "docs/current/clients/python/overview.md", CONFIG)

    assert len(chunks) == 2
    assert len({chunk["chunk_id"] for chunk in chunks}) == 2


def test_overlap_does_not_start_in_the_middle_of_a_word() -> None:
    config = CONFIG | {
        "chunking": {"max_chars": 100, "overlap_chars": 20, "min_chars": 10}
    }
    markdown = "# Long\n\n" + "alpha bravo charlie delta echo foxtrot " * 8

    chunks = chunk_markdown(markdown, "docs/current/clients/python/overview.md", config)

    assert len(chunks) > 1
    assert all(chunk["text"].split()[0] in markdown.split() for chunk in chunks)
