# Corpus analysis

## Summary

- Indexed documents: 265
- Chunks: 2351
- Characters: 1,595,378
- Chunks with code: 940 (40.0%)
- Duplicate chunk IDs: 0
- Duplicate normalized texts: 15

## Chunk length

| Metric | Characters |
|---|---:|
| Minimum | 80 |
| Mean | 678.6 |
| Median | 414 |
| P90 | 1848 |
| P95 | 2202 |
| Maximum | 2395 |

## Corpus composition

| Category | Documents | Chunks |
|---|---:|---:|
| `clients/cli` | 10 | 77 |
| `clients/python` | 9 | 91 |
| `configuration` | 3 | 90 |
| `connect` | 2 | 9 |
| `core_extensions` | 11 | 121 |
| `data` | 26 | 156 |
| `guides` | 66 | 274 |
| `operations_manual` | 13 | 65 |
| `other` | 2 | 33 |
| `sql` | 123 | 1435 |

## Largest documents

| Source | Chunks |
|---|---:|
| `docs/current/sql/functions/text.md` | 111 |
| `docs/current/sql/functions/list.md` | 104 |
| `docs/current/sql/functions/aggregates.md` | 79 |
| `docs/current/sql/functions/numeric.md` | 64 |
| `docs/current/sql/functions/utility.md` | 59 |
| `docs/current/sql/functions/timestamptz.md` | 50 |
| `docs/current/configuration/pragmas.md` | 44 |
| `docs/current/sql/functions/timestamp.md` | 42 |
| `docs/current/configuration/overview.md` | 37 |
| `docs/current/sql/functions/window_functions.md` | 32 |

## Interpretation

The corpus combines exact SQL, CLI and API identifiers with explanatory prose. This makes it suitable for comparing BM25, dense retrieval and their fusion. The bilingual evaluation set should be reported separately by language because English BM25 is not expected to retrieve English documentation reliably from Russian queries without translation.
