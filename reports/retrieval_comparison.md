# Retrieval comparison

| Retriever | Language | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| BM25 | en | 0.4167 | 0.5246 | 0.3562 | 0.3832 |
| BM25 | ru | 0.1722 | 0.2222 | 0.1467 | 0.1566 |
| Dense E5 | en | 0.3746 | 0.6713 | 0.4179 | 0.4301 |
| Dense E5 | ru | 0.4079 | 0.5524 | 0.3674 | 0.3830 |

Dense and BM25 use the same answerable queries, qrels and cutoffs.
