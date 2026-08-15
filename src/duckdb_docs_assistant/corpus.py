from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
FENCED_CODE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
LIQUID_LINK_RE = re.compile(r"\{%-?\s*link\s+([^%]+?)\s*-?%\}")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class Section:
    heading_path: tuple[str, ...]
    body: str


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(fnmatch.fnmatchcase(normalized, pattern) for pattern in patterns)


def select_source_paths(tree: Iterable[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    include = config["include"]
    exclude = config.get("exclude", [])
    selected = {
        item["path"]
        for item in tree
        if item.get("type") == "blob"
        and item["path"].endswith(".md")
        and _matches_any(item["path"], include)
        and not _matches_any(item["path"], exclude)
    }
    return sorted(selected)


def _parse_front_matter(markdown: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER_RE.match(markdown.replace("\r\n", "\n"))
    if not match:
        return {}, markdown

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line[:1].isspace():
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        if value:
            metadata[key.strip()] = value
    return metadata, markdown[match.end() :]


def _clean_markdown(markdown: str) -> str:
    markdown = markdown.replace("\r\n", "\n")
    markdown = HTML_COMMENT_RE.sub("", markdown)
    markdown = LIQUID_LINK_RE.sub(lambda match: match.group(1).strip(), markdown)
    markdown = re.sub(r"\{%\s*(?:include|capture|endcapture|assign).*?%\}", "", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def _split_sections(markdown: str, fallback_title: str) -> list[Section]:
    sections: list[Section] = []
    headings: list[str] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append(Section(tuple(headings or [fallback_title]), text))
        body.clear()

    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING_RE.match(line)
        if not match:
            body.append(line)
            continue

        flush()
        level = len(match.group(1))
        heading = re.sub(r"\s*\{#.*?\}\s*$", "", match.group(2)).strip()
        headings[:] = headings[: level - 1]
        headings.append(heading)

    flush()
    return sections


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph.strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            overlap_start = max(0, len(current) - overlap_chars)
            if overlap_start:
                boundary = current.find(" ", overlap_start)
                overlap_start = boundary + 1 if boundary >= 0 else overlap_start
            overlap = current[overlap_start:].lstrip() if overlap_chars else ""
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            current = paragraph.strip()

        while len(current) > max_chars:
            split_at = current.rfind("\n", 0, max_chars)
            if split_at < max_chars // 2:
                split_at = current.rfind(" ", 0, max_chars)
            if split_at < max_chars // 2:
                split_at = max_chars
            chunks.append(current[:split_at].strip())
            start = max(0, split_at - overlap_chars)
            if start:
                boundary = current.find(" ", start, split_at)
                start = boundary + 1 if boundary >= 0 else start
            current = current[start:].strip()

    if current:
        chunks.append(current)
    return chunks


def _source_url(source_path: str) -> str:
    if source_path == "faq.md":
        return "https://duckdb.org/faq"
    if source_path.startswith("docs/current/"):
        relative = source_path.removeprefix("docs/current/").removesuffix(".md")
        if relative.endswith("/index"):
            relative = relative.removesuffix("index")
        return f"https://duckdb.org/docs/stable/{relative}".rstrip("/") + "/"
    return f"https://duckdb.org/{source_path.removesuffix('.md')}"


def chunk_markdown(
    markdown: str,
    source_path: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    metadata, body = _parse_front_matter(markdown)
    body = _clean_markdown(body)
    fallback_title = metadata.get("title") or Path(source_path).stem.replace("_", " ").title()
    chunking = config["chunking"]
    records: list[dict[str, Any]] = []
    content_sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    for section_index, section in enumerate(_split_sections(body, fallback_title)):
        for part_index, text in enumerate(
            _split_long_text(
                section.body,
                max_chars=chunking["max_chars"],
                overlap_chars=chunking["overlap_chars"],
            )
        ):
            if len(text) < chunking["min_chars"]:
                continue
            identity = (
                f"{source_path}#{section_index}#{'/'.join(section.heading_path)}#{part_index}"
            )
            records.append(
                {
                    "chunk_id": hashlib.sha1(identity.encode("utf-8")).hexdigest(),
                    "product": config["product"],
                    "version": config["version_label"],
                    "title": fallback_title,
                    "section": section.heading_path[-1],
                    "heading_path": list(section.heading_path),
                    "text": text,
                    "code_blocks": [code.strip() for code in FENCED_CODE_RE.findall(text)],
                    "source_path": source_path,
                    "source_url": _source_url(source_path),
                    "source_repo": config["source_repo"],
                    "source_commit": config["source_commit"],
                    "license": config["license"],
                    "content_sha256": content_sha256,
                    "part_index": part_index,
                }
            )
    return records


def _download_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "duckdb-docs-assistant/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "duckdb-docs-assistant/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _load_tree(config: dict[str, Any], cache_dir: Path, tree_index: Path | None) -> list[dict[str, Any]]:
    if tree_index:
        return json.loads(tree_index.read_text(encoding="utf-8"))["tree"]

    cache_path = cache_dir / f"tree-{config['source_commit']}.json"
    if not cache_path.exists():
        url = (
            "https://api.github.com/repos/duckdb/duckdb-web/git/trees/"
            f"{config['source_commit']}?recursive=1"
        )
        cache_path.write_text(json.dumps(_download_json(url)), encoding="utf-8")
    return json.loads(cache_path.read_text(encoding="utf-8"))["tree"]


def _download_pages(
    paths: list[str], config: dict[str, Any], raw_dir: Path, workers: int
) -> dict[str, str]:
    base_url = (
        "https://raw.githubusercontent.com/duckdb/duckdb-web/"
        f"{config['source_commit']}/"
    )
    pages: dict[str, str] = {}

    def fetch(path: str) -> tuple[str, str]:
        local_path = raw_dir / path
        if local_path.exists():
            return path, local_path.read_text(encoding="utf-8")
        text = _download_text(base_url + path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(text, encoding="utf-8")
        return path, text

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, path): path for path in paths}
        for future in as_completed(futures):
            path, text = future.result()
            pages[path] = text
    return pages


def build_corpus(
    config_path: Path,
    output_dir: Path,
    cache_dir: Path,
    tree_index: Path | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir.parent / "raw"

    tree = _load_tree(config, cache_dir, tree_index)
    paths = select_source_paths(tree, config)
    pages = _download_pages(paths, config, raw_dir, workers)

    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(chunk_markdown(pages[path], path, config))

    indexed_paths = sorted({record["source_path"] for record in records})
    skipped_paths = sorted(set(paths) - set(indexed_paths))

    corpus_path = output_dir / "duckdb_docs.jsonl"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "product": config["product"],
        "version": config["version_label"],
        "source_repo": config["source_repo"],
        "source_commit": config["source_commit"],
        "license": config["license"],
        "downloaded_documents": len(paths),
        "indexed_documents": len(indexed_paths),
        "skipped_documents": len(skipped_paths),
        "chunks": len(records),
        "corpus_file": corpus_path.name,
        "source_paths": paths,
        "skipped_paths": skipped_paths,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
