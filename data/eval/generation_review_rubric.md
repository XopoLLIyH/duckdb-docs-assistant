# Generation human-review rubric v1

Each generated answer receives four independent scores from 0 to 2. Reviewers must use only
the evaluation question, its expected facts, and the source chunks actually cited by the answer.

## Correctness

- `2`: the answer is factually correct and contains no material misleading claim.
- `1`: the core answer is correct, but a secondary claim is misleading, unnecessary, or too broad.
- `0`: the core answer is wrong or unsafe.

## Completeness

- `2`: all expected facts needed to answer the question are present.
- `1`: the answer is useful but misses at least one expected fact.
- `0`: it does not address the requested task.

## Citation entailment

- `2`: cited chunks directly support all material technical claims.
- `1`: the main claim is supported, but at least one material secondary claim is not.
- `0`: the main answer is not supported by its citations.

## Language and presentation

- `2`: clear, concise, in the requested language, with usable code and links.
- `1`: understandable, but contains a visible language, formatting, or relevance defect.
- `0`: difficult to use or in the wrong language.

For deliberately unanswerable questions, a concise refusal with no technical claims or citations
receives `2` in every dimension. The aggregate score is descriptive, not a statistical estimate;
this first review was performed by one non-independent reviewer and needs a second reviewer before
being presented as a reliable human benchmark.
