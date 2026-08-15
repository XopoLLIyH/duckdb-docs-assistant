# DuckDB Documentation Assistant

Интеллектуальный ассистент по официальной документации DuckDB. Целевая архитектура:
параллельный dense retrieval и BM25, объединение кандидатов, reranker и генерация ответа
LLM с обязательными ссылками на источники.

На текущем этапе реализован воспроизводимый ingestion-пайплайн. Он:

- использует только официальные Markdown-исходники `duckdb/duckdb-web`;
- фиксирует commit источника;
- выбирает практические разделы документации и исключает архивы и большие API reference;
- сохраняет исходные страницы локально;
- разбивает страницы по иерархии заголовков, не теряя блоки кода;
- создаёт JSONL с URL, версией, лицензией, SHA-256 документа и другими метаданными;
- формирует manifest с точным списком загруженных, проиндексированных и пропущенных страниц.

## Зафиксированный источник

- Репозиторий: <https://github.com/duckdb/duckdb-web>
- Commit: `f8863ffe1d9bc2f3c100d775f899305c1a9ec899`
- Лицензия: MIT
- Версия корпуса: `current` на указанном commit

Commit является главным идентификатором версии. Метка `current` сама по себе изменчива и не
используется для воспроизводимости.

## Состав MVP-корпуса

Включены Python API, CLI, SQL, CSV/JSON/Parquet, конфигурация, эксплуатация, основные
интеграции, performance и troubleshooting. Исключены архивные версии, блоги, документация
разработчиков, C/Java/R API и автоматически сгенерированный Python reference.

Правила отбора находятся в `config/corpus.json`.

## Сборка корпуса

Python-зависимости для ingestion не нужны:

```powershell
cd duckdb-docs-assistant
$env:PYTHONPATH="src"
python scripts/build_corpus.py
```

При повторном запуске уже загруженные страницы берутся из `data/raw`. Результаты:

```text
data/raw/                         исходные Markdown-страницы
data/processed/duckdb_docs.jsonl очищенные фрагменты
data/processed/manifest.json     происхождение и состав корпуса
```

Для офлайн-запуска можно заранее сохранить ответ GitHub Tree API и передать его через
`--tree-index`:

```powershell
python scripts/build_corpus.py --tree-index path/to/tree.json
```

## Формат записи

```json
{
  "chunk_id": "stable deterministic id",
  "product": "DuckDB",
  "version": "current",
  "title": "Python API",
  "section": "Connecting",
  "heading_path": ["Python API", "Connecting"],
  "text": "...",
  "code_blocks": ["import duckdb\nduckdb.connect()"],
  "source_path": "docs/current/clients/python/overview.md",
  "source_url": "https://duckdb.org/docs/stable/clients/python/overview/",
  "source_commit": "f8863ffe...",
  "license": "MIT",
  "content_sha256": "..."
}
```

## Проверка

```powershell
python -m pytest -q
python -m ruff check src tests scripts
```

Следующие этапы: исследование корпуса, ручная разметка evaluation-набора, BM25 baseline,
dense retrieval, RRF fusion, CrossEncoder-reranker и LLM-ответы с цитированием.
