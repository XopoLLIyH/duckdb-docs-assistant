# Reranker candidate-pool ablation

Both variants use the same multilingual cross-encoder, corpus, answerable development queries
and qrels. This is parameter selection on a development seed, not a held-out estimate.

| Candidates reranked | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---:|---:|---:|---:|---:|
| 10 | 0.3964 | 0.6019 | 0.5236 | 0.4679 |
| 20 | 0.3798 | 0.4921 | 0.4940 | 0.4078 |

Reranking 20 candidates promoted semantically plausible passages that were not marked relevant,
reducing both ranking quality and recall at the measured cutoffs. The MVP therefore reranks 10
candidates; the choice must be revisited on a larger held-out evaluation set.
