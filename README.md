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

## Анализ корпуса и evaluation-набор

После сборки корпуса можно воспроизвести EDA и проверить разметку:

```powershell
$env:PYTHONPATH="src"
python scripts/analyze_corpus.py
python scripts/validate_eval.py
```

Текущий корпус содержит 265 документов и 2 351 чанк. Evaluation seed включает 16 интентов:
по одному эквивалентному вопросу на английском и русском языках. Для 15 интентов размечены
релевантные секции документации; ещё одна пара проверяет отказ от неподтверждённого ответа.

Результаты анализа сохраняются в `reports/corpus_analysis.md` и
`reports/corpus_analysis.json`. Вопросы и qrels находятся в `data/eval`.

Далее корпус и разметка используются для воспроизводимого сравнения retrieval-методов.

## BM25 baseline

Локальный Okapi BM25 индексирует `title`, `heading_path` и текст каждого чанка. Токенизатор
сохраняет технические идентификаторы вроде `read_csv`, `memory_limit` и `duckdb.sql`.

```powershell
$env:PYTHONPATH="src"
python scripts/evaluate_bm25.py
```

Результат на answerable-части evaluation seed:

| Язык | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| English | 0.4167 | 0.5246 | 0.3562 | 0.3832 |
| Russian | 0.1722 | 0.2222 | 0.1467 | 0.1566 |

Подробный отчёт сохраняется в `reports/bm25_metrics.md`, агрегированные и per-query метрики —
в `reports/bm25_metrics.json`, а первые десять результатов каждого запроса — в
`reports/bm25_run.jsonl`.

## Multilingual dense retrieval

Dense retriever использует `intfloat/multilingual-e5-base` с зафиксированной ревизией модели.
Запросы получают префикс `query:`, документы — `passage:`; нормализованные 768-мерные
эмбеддинги ранжируются по скалярному произведению. Матрица документов и manifest сохраняются
локально и повторно используются только при полном совпадении модели, ревизии и порядка чанков.

```powershell
python -m pip install -e ".[dense,dev]"
python scripts/evaluate_dense.py
```

Сравнение на тех же answerable-запросах и qrels:

| Retriever | Язык | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| BM25 | English | 0.4167 | 0.5246 | 0.3562 | 0.3832 |
| BM25 | Russian | 0.1722 | 0.2222 | 0.1467 | 0.1566 |
| Dense E5 | English | 0.3746 | 0.6713 | 0.4179 | 0.4301 |
| Dense E5 | Russian | 0.4079 | 0.5524 | 0.3674 | 0.3830 |

Подробные результаты находятся в `reports/dense_metrics.md` и
`reports/dense_metrics.json`, top-10 выдача — в `reports/dense_run.jsonl`, а краткое сравнение —
в `reports/retrieval_comparison.md`. Эмбеддинги и файлы модели намеренно не входят в Git.

Следующий этап — объединить кандидатов BM25 и dense retrieval через Reciprocal Rank Fusion,
а затем добавить reranker.
