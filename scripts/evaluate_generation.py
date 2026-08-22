from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from duckdb_docs_assistant.evaluation import load_jsonl
from duckdb_docs_assistant.generation import (
    ANSWER_SCHEMA,
    CITATION_RE,
    SYSTEM_PROMPT,
    GenerationValidationError,
    build_context,
    build_retry_prompt,
    build_user_prompt,
    validate_answer,
)
from duckdb_docs_assistant.ollama_client import OllamaClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(config: dict[str, Any], paths: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    digest.update(SYSTEM_PROMPT.encode("utf-8"))
    digest.update(json.dumps(ANSWER_SCHEMA, sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _load_partial(path: Path, fingerprint: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("run_fingerprint") == fingerprint:
            if "task_success" in row and "expected_status_match" not in row:
                row["expected_status_match"] = row.pop("task_success")
            rows[row["query_id"]] = row
    return rows


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)] if ordered else 0.0


def _prose_paragraph_coverage(answer: str) -> tuple[int, int]:
    prose = CODE_FENCE_RE.sub("", answer)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", prose) if part.strip()]
    return sum(bool(CITATION_RE.search(part)) for part in paragraphs), len(paragraphs)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _summarize(rows: list[dict[str, Any]], language: str | None = None) -> dict[str, Any]:
    selected = [row for row in rows if language is None or row["language"] == language]
    valid = [row for row in selected if row["validation_valid"]]
    successes = [row for row in selected if row["expected_status_match"]]
    return {
        "queries": len(selected),
        "valid_response_rate": _safe_ratio(len(valid), len(selected)),
        "expected_status_accuracy": _safe_ratio(len(successes), len(selected)),
    }


def _render_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    runtime = report["runtime"]
    lines = [
        "# Grounded generation evaluation",
        "",
        f"Model: `{report['model']}`. Fixed candidate run: `{report['candidate_run']}`.",
        "",
        "| Scope | Queries | Valid JSON + grounding | Expected-status accuracy |",
        "|---|---:|---:|---:|",
    ]
    for scope in ("overall", "en", "ru"):
        values = metrics[scope]
        lines.append(
            f"| {scope} | {values['queries']} | {values['valid_response_rate']:.4f} | "
            f"{values['expected_status_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Grounding proxies",
            "",
            f"- Answerable questions answered: {metrics['answerable_answer_rate']:.4f}",
            f"- Unanswerable refusal accuracy: {metrics['unanswerable_refusal_accuracy']:.4f}",
            f"- Retrieved context contains a qrel: {metrics['context_qrel_coverage']:.4f}",
            f"- Cited chunks that are qrels (micro precision): {metrics['citation_qrel_precision']:.4f}",
            f"- Answers citing at least one qrel: {metrics['answers_citing_qrel_rate']:.4f}",
            f"- Prose paragraphs with a citation: {metrics['paragraph_citation_coverage']:.4f}",
            "",
            "## Runtime",
            "",
            f"- Ollama latency median: {runtime['latency_seconds_median']:.2f} s",
            f"- Ollama latency P95: {runtime['latency_seconds_p95']:.2f} s",
            f"- Prompt tokens: {runtime['prompt_tokens_total']}",
            f"- Completion tokens: {runtime['completion_tokens_total']}",
            f"- Generation throughput: {runtime['generation_tokens_per_second']:.2f} token/s",
            f"- Queries requiring validation retry: {runtime['queries_retried']}",
            "",
            "## Interpretation",
            "",
            (
            "Qrel overlap and paragraph coverage are automated grounding proxies. They do not "
            "establish semantic correctness, completeness, or citation entailment. The raw "
            "answers in `generation_run.jsonl` still require manual review. Expected-status "
            "accuracy only checks whether answerable questions were answered and deliberately "
            "unanswerable questions were refused."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _build_report(
    rows: list[dict[str, Any]], config: dict[str, Any], candidate_run: Path
) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    answered = [
        row
        for row in answerable
        if row["validation_valid"] and row["status"] == "answered"
    ]
    cited_count = sum(len(row["cited_chunk_ids"]) for row in answered)
    cited_relevant = sum(
        len(set(row["cited_chunk_ids"]) & set(row["relevant_chunk_ids"]))
        for row in answered
    )
    paragraph_cited = sum(row["cited_prose_paragraphs"] for row in answered)
    paragraph_total = sum(row["prose_paragraphs"] for row in answered)
    latencies = [row["total_duration_ns"] / 1e9 for row in rows if row["total_duration_ns"]]
    eval_duration_ns = sum(row["eval_duration_ns"] for row in rows)
    completion_tokens = sum(row["completion_tokens"] for row in rows)
    metrics = {
        "overall": _summarize(rows),
        "en": _summarize(rows, "en"),
        "ru": _summarize(rows, "ru"),
        "answerable_answer_rate": _safe_ratio(len(answered), len(answerable)),
        "unanswerable_refusal_accuracy": _safe_ratio(
            sum(
                row["validation_valid"] and row["status"] == "insufficient_context"
                for row in unanswerable
            ),
            len(unanswerable),
        ),
        "context_qrel_coverage": _safe_ratio(
            sum(bool(set(row["context_chunk_ids"]) & set(row["relevant_chunk_ids"])) for row in answerable),
            len(answerable),
        ),
        "citation_qrel_precision": _safe_ratio(cited_relevant, cited_count),
        "answers_citing_qrel_rate": _safe_ratio(
            sum(bool(set(row["cited_chunk_ids"]) & set(row["relevant_chunk_ids"])) for row in answered),
            len(answered),
        ),
        "paragraph_citation_coverage": _safe_ratio(paragraph_cited, paragraph_total),
    }
    return {
        "model": config["model"],
        "candidate_run": str(candidate_run.relative_to(PROJECT_ROOT)),
        "candidate_run_sha256": _sha256(candidate_run),
        "queries": len(rows),
        "metrics": metrics,
        "runtime": {
            "latency_seconds_mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "latency_seconds_median": round(statistics.median(latencies), 3) if latencies else 0.0,
            "latency_seconds_p95": round(_percentile(latencies, 0.95), 3),
            "prompt_tokens_total": sum(row["prompt_tokens"] for row in rows),
            "completion_tokens_total": completion_tokens,
            "generation_tokens_per_second": round(
                completion_tokens / (eval_duration_ns / 1e9), 3
            )
            if eval_duration_ns
            else 0.0,
            "queries_retried": sum(row["generation_attempts"] > 1 for row in rows),
            "attempts_total": sum(row["generation_attempts"] for row in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounded generation with fixed candidates.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "generation.json")
    parser.add_argument(
        "--corpus", type=Path, default=PROJECT_ROOT / "data" / "processed" / "duckdb_docs.jsonl"
    )
    parser.add_argument(
        "--questions", type=Path, default=PROJECT_ROOT / "data" / "eval" / "questions.jsonl"
    )
    parser.add_argument("--qrels", type=Path, default=PROJECT_ROOT / "data" / "eval" / "qrels.jsonl")
    parser.add_argument(
        "--candidate-run", type=Path, default=PROJECT_ROOT / "reports" / "reranker_run.jsonl"
    )
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus = load_jsonl(args.corpus)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in corpus}
    questions = load_jsonl(args.questions)
    qrels = {row["query_id"]: row["relevant_chunk_ids"] for row in load_jsonl(args.qrels)}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.candidate_run):
        candidates[row["query_id"]].append(row)
    for candidate_rows in candidates.values():
        candidate_rows.sort(key=lambda row: row["rank"])

    fingerprint = _fingerprint(config, [args.corpus, args.questions, args.qrels, args.candidate_run])
    args.report_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.report_dir / "generation_run.partial.jsonl"
    run_path = args.report_dir / "generation_run.jsonl"
    resume_path = partial_path if partial_path.exists() else run_path
    completed = _load_partial(resume_path, fingerprint)
    if completed:
        partial_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in completed.values()),
            encoding="utf-8",
        )
    client = OllamaClient(config)
    context_config = config["context"]
    token_budget = context_config["max_input_tokens"] - context_config["reserved_prompt_tokens"]

    with partial_path.open("a", encoding="utf-8", newline="\n") as partial:
        for index, question in enumerate(questions, start=1):
            query_id = question["query_id"]
            if query_id in completed:
                print(f"[{index}/{len(questions)}] {query_id}: resumed", flush=True)
                continue
            ranked_ids = [row["chunk_id"] for row in candidates.get(query_id, [])]
            if not ranked_ids:
                raise ValueError(f"No fixed candidates for {query_id}")
            bundle = build_context(
                question["question"],
                ranked_ids,
                chunks_by_id,
                token_budget=token_budget,
                max_sources=context_config["max_sources"],
            )
            user_prompt = build_user_prompt(bundle)
            validation_error = None
            status = None
            answer_text = ""
            citations: list[str] = []
            attempts: list[dict[str, Any]] = []
            response = None
            for attempt in range(config.get("validation_retries", 0) + 1):
                response = client.chat(SYSTEM_PROMPT, user_prompt, ANSWER_SCHEMA)
                try:
                    answer = validate_answer(
                        response.content, {source.source_id for source in bundle.sources}
                    )
                    status = answer.status
                    answer_text = answer.answer
                    citations = answer.citations
                    attempt_error = None
                except GenerationValidationError as error:
                    attempt_error = str(error)
                attempts.append(
                    {
                        "raw_content": response.content,
                        "validation_error": attempt_error,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_duration_ns": response.total_duration_ns,
                        "load_duration_ns": response.load_duration_ns,
                        "prompt_eval_duration_ns": response.prompt_eval_duration_ns,
                        "eval_duration_ns": response.eval_duration_ns,
                    }
                )
                if attempt_error is None:
                    break
                validation_error = attempt_error
                user_prompt = build_retry_prompt(user_prompt, attempt_error)
            if response is None:
                raise RuntimeError("Generation loop completed without an Ollama response")
            if status is not None:
                validation_error = None
            sources_by_id = {source.source_id: source for source in bundle.sources}
            cited_chunk_ids = [sources_by_id[source_id].chunk_id for source_id in citations]
            cited_paragraphs, paragraphs = _prose_paragraph_coverage(answer_text)
            expected_status = "answered" if question["answerable"] else "insufficient_context"
            row = {
                "run_fingerprint": fingerprint,
                "query_id": query_id,
                "intent_id": question["intent_id"],
                "language": question["language"],
                "category": question["category"],
                "question": question["question"],
                "answerable": question["answerable"],
                "expected_status": expected_status,
                "status": status,
                "expected_status_match": validation_error is None and status == expected_status,
                "validation_valid": validation_error is None,
                "validation_error": validation_error,
                "answer": answer_text,
                "raw_content": response.content,
                "generation_attempts": len(attempts),
                "attempts": attempts,
                "citations": citations,
                "context_chunk_ids": [source.chunk_id for source in bundle.sources],
                "cited_chunk_ids": cited_chunk_ids,
                "relevant_chunk_ids": qrels[query_id],
                "sources": [
                    {
                        "source_id": source.source_id,
                        "chunk_id": source.chunk_id,
                        "title": source.title,
                        "section": source.section,
                        "url": source.url,
                        "truncated": source.truncated,
                    }
                    for source in bundle.sources
                ],
                "estimated_context_tokens": bundle.estimated_tokens,
                "cited_prose_paragraphs": cited_paragraphs,
                "prose_paragraphs": paragraphs,
                "model": response.model,
                "prompt_tokens": sum(item["prompt_tokens"] for item in attempts),
                "completion_tokens": sum(item["completion_tokens"] for item in attempts),
                "total_duration_ns": sum(item["total_duration_ns"] for item in attempts),
                "load_duration_ns": sum(item["load_duration_ns"] for item in attempts),
                "prompt_eval_duration_ns": sum(
                    item["prompt_eval_duration_ns"] for item in attempts
                ),
                "eval_duration_ns": sum(item["eval_duration_ns"] for item in attempts),
            }
            partial.write(json.dumps(row, ensure_ascii=False) + "\n")
            partial.flush()
            completed[query_id] = row
            result = status if validation_error is None else f"invalid: {validation_error}"
            print(f"[{index}/{len(questions)}] {query_id}: {result}", flush=True)

    rows = [completed[question["query_id"]] for question in questions]
    run_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    report = _build_report(rows, config, args.candidate_run)
    (args.report_dir / "generation_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.report_dir / "generation_metrics.md").write_text(
        _render_report(report), encoding="utf-8"
    )
    partial_path.unlink()
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
