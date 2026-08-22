# Grounded generation evaluation

Model: `qwen3:8b-q4_K_M`. Fixed candidate run: `reports\reranker_run.jsonl`.

| Scope | Queries | Valid JSON + grounding | Expected-status accuracy |
|---|---:|---:|---:|
| overall | 32 | 1.0000 | 1.0000 |
| en | 16 | 1.0000 | 1.0000 |
| ru | 16 | 1.0000 | 1.0000 |

## Grounding proxies

- Answerable questions answered: 1.0000
- Unanswerable refusal accuracy: 1.0000
- Retrieved context contains a qrel: 0.6667
- Cited chunks that are qrels (micro precision): 0.3556
- Answers citing at least one qrel: 0.5333
- Prose paragraphs with a citation: 0.8571

## Runtime

- Ollama latency median: 2.81 s
- Ollama latency P95: 5.14 s
- Prompt tokens: 49857
- Completion tokens: 4108
- Generation throughput: 45.52 token/s
- Queries requiring validation retry: 1

## Interpretation

Qrel overlap and paragraph coverage are automated grounding proxies. They do not establish semantic correctness, completeness, or citation entailment. The raw answers in `generation_run.jsonl` still require manual review. Expected-status accuracy only checks whether answerable questions were answered and deliberately unanswerable questions were refused.
