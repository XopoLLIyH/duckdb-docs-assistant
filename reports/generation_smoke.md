# Grounded generation smoke test

Date: 2026-08-21

Model: `qwen3:8b-q4_K_M` (`500a1f067a9f`)

Runtime: Ollama 0.32.5, RTX 3060 12 GiB, 4096-token context, 100% GPU offload

No generated SQL or code was executed.

| Case | Expected | Result | Citations | Prompt / completion tokens | Pipeline latency |
|---|---|---|---|---:|---:|
| English answerable: multiple Parquet files | Grounded English answer | Passed | `S2` | 934 / 133 | 26.70 s |
| Russian answerable: read-only Python connection | Grounded Russian answer | Passed | `S2` | 1481 / 191 | 30.49 s |
| Russian unanswerable: users and row-level permissions | Refusal | Passed | None | 1559 / 78 | 23.86 s |

## Observations

- Ollama reported `100% GPU` with no CPU offload and context 4096.
- During the first generation sample, total GPU memory use was 7496 MiB, with 4615 MiB free;
  temperature was 52 °C.
- The English response used `read_parquet` with a list of glob patterns and cited the official
  multi-file documentation.
- The Russian answer used `duckdb.connect(database=..., read_only=True)` and cited the Python
  DB API documentation.
- The intentionally unsupported permissions question returned `insufficient_context` without
  citations.

## Issue found and fixed

The initial response schema required citations both as `[S#]` markers in the answer and as a
separate JSON array. Qwen cited `[S2]` in the answer but returned `[S2, S3]` in the array, and the
validator correctly rejected the response. The redundant array was removed; citation IDs are now
derived only from validated answer markers. The repeated English test then passed.

## Scope

This is a three-case smoke test, not a quality estimate. The next evaluation must cover the full
question seed and score answer correctness, groundedness, citation completeness, refusal quality,
latency and generation speed.
