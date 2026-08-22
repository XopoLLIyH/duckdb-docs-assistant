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

## Hybrid retrieval: Reciprocal Rank Fusion

Hybrid retriever независимо получает до 50 кандидатов из BM25 и dense E5, после чего объединяет
их по позициям с RRF (`k=60`). Простая проверка кириллицы выбирает веса по языку запроса:
английские запросы используют оба канала с равными весами, русские — только multilingual dense,
поскольку документация англоязычная и BM25 не даёт полезного лексического сигнала.

```powershell
python scripts/evaluate_hybrid.py
```

| Retriever | Язык | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| Dense E5 | English | 0.3746 | **0.6713** | 0.4179 | 0.4301 |
| Hybrid RRF | English | **0.4746** | 0.6514 | **0.4506** | **0.4740** |
| Dense E5 | Russian | 0.4079 | 0.5524 | 0.3674 | 0.3830 |
| Hybrid RRF | Russian | 0.4079 | 0.5524 | 0.3674 | 0.3830 |

RRF улучшил раннее ранжирование английской выдачи, сохранив качество dense retrieval для русской.
Веса выбраны на текущем development seed, поэтому до заявления о генерализации потребуется
отдельный held-out evaluation-набор. Отчёты сохраняются в `reports/hybrid_metrics.*`, а выдача со
вкладами отдельных каналов — в `reports/hybrid_run.jsonl`.

## Multilingual cross-encoder reranker

Финальный retrieval-этап использует
[`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1),
обученный на multilingual MS MARCO. Ревизия модели зафиксирована. В отличие от bi-encoder,
cross-encoder совместно обрабатывает вопрос и каждый документ, поэтому лучше оценивает их
релевантность, но работает только на коротком списке кандидатов.

```powershell
python scripts/evaluate_reranker.py
```

Reranker переставляет top-10 RRF-кандидатов. Эксперимент с top-20 на development seed снизил
MRR и Recall@10, поэтому для MVP выбран top-10; этот параметр нужно перепроверить на held-out
наборе. Результаты эксперимента сохранены в `reports/reranker_ablation.md`.

| Этап | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| Hybrid RRF | 0.6019 | 0.4090 | 0.4285 |
| Cross-encoder | 0.6019 | **0.5236** | **0.4679** |

На CPU reranking десяти кандидатов занимает около 1.3 секунды на запрос. Полный отчёт находится
в `reports/reranker_metrics.*`, а `reports/reranker_run.jsonl` хранит cross-encoder score и
исходную RRF-позицию каждого результата. Файлы модели не входят в Git.

## Grounded generation

Локальная генерация настроена на `qwen3:8b-q4_K_M` через Ollama. Retrieval и reranker отдают
десять кандидатов, после чего context builder удаляет точные дубли и упаковывает до пяти лучших
фрагментов в консервативный бюджет 2 700 токенов. Каждый фрагмент получает идентификатор `[S1]`,
заголовок, раздел и официальный URL.

Ollama получает нестриминговый `/api/chat` запрос с JSON Schema, `think=false`, контекстом 4096 и
низкой температурой. Модель должна вернуть один из двух статусов: `answered` или
`insufficient_context`. Валидатор не пропускает:

- содержательный ответ без цитат;
- ссылку на источник, которого не было в промпте;
- отказ, содержащий ссылки;
- невалидный JSON или неожиданные поля.

Список использованных источников извлекается непосредственно из маркеров `[S1]` в тексте. Это
устраняет второй, потенциально противоречивый список цитат в JSON.

```powershell
python scripts/ask.py "Как прочитать несколько Parquet-файлов?"
```

Для проверки собранного контекста без обращения к Ollama:

```powershell
python scripts/ask.py --preview-context "Как прочитать несколько Parquet-файлов?"
```

Конфигурация находится в `config/generation.json`. Клиент использует только локальный адрес
`127.0.0.1:11434`; сгенерированный SQL автоматически не выполняется.

Live smoke test Qwen сохранён в `reports/generation_smoke.md`.

Полная generation evaluation запускается на зафиксированном `reranker_run`, поэтому измерение
генерации не требует повторной загрузки dense retrieval и cross-encoder моделей:

```powershell
python scripts/evaluate_generation.py
```

Прогон возобновляемый: после каждого вопроса сохраняется partial JSONL, а завершённые ответы с
тем же fingerprint повторно не генерируются. При нарушении JSON/citation-контракта разрешена одна
корректирующая попытка. На 32 вопросах итоговый прогон получил 100% валидных ответов и 100%
expected-status accuracy; один запрос потребовал retry. Медианная задержка Ollama — 2.81 с,
скорость генерации — 45.52 токена/с.

Автоматические grounding-прокси заметно строже: qrel найден в контексте для 66.67% ответных
вопросов, micro precision цитат относительно qrels — 35.56%, citation coverage абзацев — 85.71%.
Эти числа не доказывают семантическую корректность: qrels могут быть неполными, а наличие цитаты
не означает, что источник действительно подтверждает утверждение. Полный отчёт находится в
`reports/generation_metrics.md`, сырые ответы и обе попытки retry — в
`reports/generation_run.jsonl`. Следующий этап — ручная оценка correctness/completeness и
citation entailment по заранее определённой рубрике.
