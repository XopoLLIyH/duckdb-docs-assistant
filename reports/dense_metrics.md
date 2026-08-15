# Multilingual dense retrieval

Model: `intfloat/multilingual-e5-base` at revision `d128750597153bb5987e10b1c3493a34e5a4502a`.
Documents use the `passage:` prefix; questions use `query:`. Embeddings are L2 normalized and ranked by dot product.

| Language | Queries | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| overall | 30 | 0.3913 | 0.6118 | 0.3926 | 0.4065 |
| en | 15 | 0.3746 | 0.6713 | 0.4179 | 0.4301 |
| ru | 15 | 0.4079 | 0.5524 | 0.3674 | 0.3830 |

## Runtime

- Device: `cpu`
- Document cache reused: `True`
- Document encoding: 0.007 s
- Query encoding median: 87.777 ms
- Query encoding P95: 129.745 ms
