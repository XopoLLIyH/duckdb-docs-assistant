# BM25 baseline

Okapi BM25 indexes the chunk title, heading path and text. Metrics are macro-averaged over answerable queries only.

| Language | Queries | Results | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| overall | 30 | 29 | 0.2944 | 0.3734 | 0.2514 | 0.2699 |
| en | 15 | 15 | 0.4167 | 0.5246 | 0.3562 | 0.3832 |
| ru | 15 | 14 | 0.1722 | 0.2222 | 0.1467 | 0.1566 |

## Missed queries at K=10

- `q003_ru` (ru): Как DuckDB автоматически определяет столбцы и типы данных CSV-файла?
- `q004_ru` (ru): Как прочитать несколько Parquet-файлов одним запросом DuckDB?
- `q005_ru` (ru): Как сохранить результат SQL-запроса в Parquet-файл?
- `q006_en` (en): Which functions can DuckDB use to load JSON data?
- `q007_ru` (ru): Как открыть файл базы DuckDB только для чтения из Python?
- `q008_ru` (ru): Могут ли несколько процессов одновременно записывать в один файл базы DuckDB?
- `q009_en` (en): How can I limit the amount of memory used by DuckDB?
- `q009_ru` (ru): Как ограничить объём оперативной памяти, используемой DuckDB?
- `q011_en` (en): How do I connect DuckDB to an existing PostgreSQL database?
- `q012_en` (en): How should I configure AWS credentials for reading Parquet files from S3?
- `q012_ru` (ru): Как настроить AWS-учётные данные для чтения Parquet-файлов из S3?
- `q013_ru` (ru): Как экспортировать всю базу DuckDB в каталог?
- `q014_ru` (ru): Как безопасно передавать параметры в запрос DuckDB из Python?
- `q015_ru` (ru): Как импортировать CSV-файл с помощью dot-команд DuckDB CLI?

## Interpretation

The documentation is English. English BM25 measures the lexical baseline, while Russian BM25 is intentionally a cross-lingual stress test. A large language gap supports using multilingual dense retrieval or explicit query translation.
