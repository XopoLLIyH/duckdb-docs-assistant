# Multilingual cross-encoder reranker

Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` at revision `1427fd652930e4ba29e8149678df786c240d8825`.
The cross-encoder scores each query-passage pair jointly and reranks the top 10 Hybrid RRF candidates.

| Language | Queries | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| overall | 30 | 0.3964 | 0.6019 | 0.5236 | 0.4679 |
| en | 15 | 0.4056 | 0.6514 | 0.5472 | 0.5080 |
| ru | 15 | 0.3873 | 0.5524 | 0.5000 | 0.4277 |

## Runtime

- Device: `cpu`
- Model load: 11.260 s
- Query reranking median: 1285.684 ms
- Query reranking P95: 1764.006 ms
