# Manual generation review

Rubric: `data/eval/generation_review_rubric.md`. One non-independent reviewer; scores require a second review before use as a reliable human benchmark.

| Scope | Queries | Normalized score | Strict pass | Correct | Complete | Entailed | Language |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 32 | 0.9609 | 0.7188 | 0.9688 | 0.8438 | 0.9375 | 0.9375 |
| en | 16 | 0.9688 | 0.7500 | 1.0000 | 0.8125 | 0.9375 | 1.0000 |
| ru | 16 | 0.9531 | 0.6875 | 0.9375 | 0.8750 | 0.9375 | 0.8750 |

## Qrel audit

- Answerable questions without an annotated qrel in context: 10
- Of those, answers receiving full correctness: 9
- Confirmed retrieval mismatch: `q010_ru` selected the Docker UI subsection for a CLI question.

The other nine qrel misses cite directly relevant official sections, including CSV type detection, JSON readers, S3 credentials, Parquet export, and resource limits. This is evidence that the current qrels are incomplete; they should be independently adjudicated rather than expanded automatically from model-selected sources.

## Findings requiring action

| Query | C | K | E | L | Error types | Review note |
|---|---:|---:|---:|---:|---|---|
| q002_ru | 2 | 2 | 2 | 1 | presentation | Correct answer, but it reproduces an internal relative documentation link and adds an irrelevant append detail. |
| q003_en | 2 | 1 | 2 | 2 | incomplete | Explains dialect and type inference but omits how detection can be configured. |
| q008_en | 2 | 2 | 1 | 2 | citation_gap | The conclusion is correct, but the sole citation supports read-only sharing and not the Quack or DuckLake alternatives mentioned. |
| q010_ru | 1 | 2 | 2 | 2 | retrieval_mismatch, irrelevant_detail | The container opens the CLI, but retrieval selected the UI subsection; the answer unnecessarily enables host networking and tells the user to start the UI. |
| q011_en | 2 | 1 | 2 | 2 | incomplete | ATTACH and the connection string are correct, but the expected extension setup is omitted. |
| q011_ru | 2 | 1 | 2 | 2 | incomplete | ATTACH and the connection string are correct, but the expected extension setup is omitted. |
| q012_ru | 2 | 2 | 2 | 1 | mixed_language | Technically complete, but one Russian phrase contains Chinese characters. |
| q014_en | 2 | 1 | 2 | 2 | incomplete | Positional parameters are explained, but named parameters are omitted. |
| q014_ru | 2 | 1 | 1 | 2 | incomplete, citation_gap | Named parameters are shown, but positional parameters are omitted and the injection/sanitization claims are not supported by the cited chunk. |

## Error taxonomy

- `citation_gap`: 2
- `incomplete`: 5
- `irrelevant_detail`: 1
- `mixed_language`: 1
- `presentation`: 1
- `retrieval_mismatch`: 1

## Interpretation

The normalized score summarizes this rubric only; it is not an accuracy confidence interval. Protocol validity was 100%, while manual review found one misleading retrieval result, five incomplete answers, two citation gaps, and two presentation defects. The next defensible step is a blinded second review plus qrel adjudication.
