# Hybrid retrieval with Reciprocal Rank Fusion

BM25 and multilingual E5 each retrieve an independent candidate pool. Reciprocal Rank Fusion combines ranks rather than incomparable raw scores.

- RRF k: `60`
- Candidate pool per retriever: `50`
- English weights: `{'bm25': 1.0, 'dense': 1.0}`
- Russian weights: `{'bm25': 0.0, 'dense': 1.0}`
- Weight selection status: development seed; held-out validation required

| Language | Queries | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| overall | 30 | 0.4413 | 0.6019 | 0.4090 | 0.4285 |
| en | 15 | 0.4746 | 0.6514 | 0.4506 | 0.4740 |
| ru | 15 | 0.4079 | 0.5524 | 0.3674 | 0.3830 |

## Runtime

- Device: `cpu`
- Document cache reused: `True`
- Query latency median: 58.400 ms
- Query latency P95: 75.456 ms
