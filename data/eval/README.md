# Evaluation dataset

`questions.jsonl` is a manually curated seed set for retrieval evaluation. Each intent has
equivalent English and Russian questions. Relevant sources are labeled at document and section
level, then `scripts/validate_eval.py` resolves them to deterministic chunk IDs in `qrels.jsonl`.

Report metrics separately for `en` and `ru`. English BM25 is a meaningful lexical baseline;
Russian questions against English documentation are primarily a multilingual dense-retrieval
stress test unless query translation is added explicitly.

Unanswerable questions have no qrels and are reserved for testing abstention at the answer
generation stage. They must not be included in Recall@K or MRR denominators.
