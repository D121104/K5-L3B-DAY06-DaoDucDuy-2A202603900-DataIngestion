"""Run the reproducible RFC 9111 pgvector retrieval experiment.

From ``starter/`` run ``python -m src.run``. Outputs are written under
``results/``: run_config.json, page_errors.json, chunk_stats.csv,
retrieval_traces.json, metrics.json, coherence_audit_template.csv,
metadata_audit_template.csv, and grounding_audit_template.csv.
"""
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STRATEGIES = ("A", "B", "C")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _validate_inputs(manifest: dict[str, Any], questions: list[dict[str, Any]], pages: list[Any]) -> list[int]:
    extraction = manifest.get("extraction", {})
    valid_pages = extraction.get("valid_pages")
    if not isinstance(valid_pages, list) or len(set(valid_pages)) != len(valid_pages):
        raise ValueError("manifest extraction.valid_pages must be a unique page-number list")
    if len(valid_pages) < 30:
        raise ValueError(f"manifest contains only {len(valid_pages)} valid pages; at least 30 are required")
    page_map = {page.page: page for page in pages}
    missing = sorted(set(valid_pages) - page_map.keys())
    if missing:
        raise ValueError(f"manifest references missing PDF pages: {missing}")
    useful = [number for number in valid_pages if page_map[number].text.strip()]
    minimum = int(extraction.get("useful_content_page_count", 30))
    if len(useful) < 30 or len(useful) < minimum:
        raise ValueError(f"only {len(useful)} manifest-valid pages contain text; at least 30 and the declared {minimum} are required")
    if len(questions) != 20:
        raise ValueError(f"benchmark must contain 20 questions, found {len(questions)}")
    splits = Counter(question.get("split") for question in questions)
    if splits != Counter({"dev": 15, "test": 5}):
        raise ValueError(f"benchmark must have exactly 15 dev and 5 test questions, found {dict(splits)}")
    filtered = [question for question in questions if question.get("filter") is not None]
    if len(filtered) < 3 or not any(question.get("split") == "test" for question in filtered):
        raise ValueError("benchmark requires at least three filtered questions including one test question")
    ids: set[str] = set()
    for question in questions:
        required = ("id", "question", "expected_answer", "gold_source_id", "gold_page", "evidence_quote")
        if any(not question.get(key) for key in required):
            raise ValueError(f"benchmark question is missing required gold fields: {question.get('id', '<unknown>')}")
        if question["id"] in ids:
            raise ValueError(f"duplicate benchmark question id: {question['id']}")
        ids.add(question["id"])
        if question["gold_page"] not in useful or question["gold_source_id"] != manifest["source_id"]:
            raise ValueError(f"gold source/page is outside this corpus: {question['id']}")
        filter_spec = question.get("filter")
        if filter_spec is not None:
            if not isinstance(filter_spec, dict) or set(filter_spec) != {"field", "op", "value"}:
                raise ValueError(f"malformed filter for {question['id']}")
            if filter_spec["field"] != "section" or filter_spec["op"] != "eq":
                raise ValueError(f"only the supported metadata filter section eq is accepted ({question['id']})")
    return useful


def _assign_sections(pages: list[Any], valid_pages: set[int]) -> dict[int, tuple[str | None, list[tuple[int, str]]]]:
    """Record page-start sections and transient heading offsets for eligible pages."""
    active: str | None = None
    section_pattern = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+?)\s*$")
    page_spans: dict[int, tuple[str | None, list[tuple[int, str]]]] = {}
    for page in pages:
        if page.page not in valid_pages:
            continue
        start_section = active
        spans: list[tuple[int, str]] = []
        offset = 0
        for line in page.text.splitlines(keepends=True):
            candidate = line.strip()
            match = section_pattern.fullmatch(candidate)
            if match:
                number, title = match.groups()
                if len(title) <= 110 and not title.endswith((".", ";", ",")):
                    active = f"{number} {title}"
                    spans.append((offset + len(line) - len(line.lstrip()), active))
            offset += len(line)
        page.metadata["section"] = start_section
        page_spans[page.page] = (start_section, spans)
    return page_spans


def _section_at(page_info: tuple[str | None, list[tuple[int, str]]], offset: int) -> str | None:
    section, spans = page_info
    for heading_offset, heading_section in spans:
        if heading_offset > offset:
            break
        section = heading_section
    return section


def _assign_chunk_sections(chunks: list[Any], page: Any,
                           page_info: tuple[str | None, list[tuple[int, str]]],
                           strategy: str, size: int = 0, overlap: int = 0) -> None:
    """Apply section metadata from each chunk's source start without persisting spans."""
    cursor = 0
    step = size - overlap
    for chunk in chunks:
        if strategy == "fixed_size":
            source_start = chunk.metadata["chunk_index"] * step
        else:
            source_start = page.text.find(chunk.text, cursor)
            if source_start < 0:
                chunk.metadata["section"] = None
                continue
            cursor = source_start + len(chunk.text)
        chunk.metadata["section"] = _section_at(page_info, source_start)


def _filter_for(question: dict[str, Any]) -> dict[str, Any] | None:
    specification = question.get("filter")
    if specification is None:
        return None
    field, operation, value = (specification.get("field"), specification.get("op"), specification.get("value"))
    if operation != "eq":
        raise ValueError(f"unsupported filter operation {operation!r} for {question['id']}")
    if field != "section":
        raise ValueError(f"unsupported filter field {field!r} for {question['id']}")
    return {field: value}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_started = time.perf_counter()
    config: dict[str, Any] = {
        "run_id": run_id, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m src.run (from starter/)", "corpus": "data/rfc9111.pdf",
        "manifest": "data/manifest.json", "benchmark": "benchmark/questions.json",
        "strategies": {
            "A": {"chunker": "fixed_size", "size_characters": 500, "overlap_characters": 50, "filtering": "none"},
            "B": {"chunker": "by_paragraph", "max_characters": 1000, "filtering": "none"},
            "C": {"chunker": "by_paragraph", "max_characters": 1000, "filtering": "question-supplied section equality only"},
        },
        "store": {"backend": "PostgreSQL + pgvector", "distance": "cosine", "top_k": 3, "persistent": True},
        "manual_assessment": {"performed": False, "labels": "blank templates; no human judgments inferred"},
    }
    _write_json(RESULTS / "run_config.json", config)
    try:
        env_path = ROOT / ".env"
        if env_path.is_file():
            try:
                from dotenv import load_dotenv
            except ImportError as exc:
                raise RuntimeError("starter/.env exists but python-dotenv is unavailable; install starter/requirements.txt to load it") from exc
            load_dotenv(env_path, override=False)

        manifest_path = ROOT / "data" / "manifest.json"
        benchmark_path = ROOT / "benchmark" / "questions.json"
        pdf_path = ROOT / "data" / "rfc9111.pdf"
        for path in (manifest_path, benchmark_path, pdf_path):
            if not path.is_file():
                raise FileNotFoundError(f"required input file does not exist: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        questions = json.loads(benchmark_path.read_text(encoding="utf-8"))
        try:
            from .ingestion import load_pdf
            from .chunking import by_paragraph, fixed_size
            from .embeddings import MyEmbedder
            from .store import PgVectorStore
            from .evaluation import evaluate_query, summarize
        except ImportError as exc:
            raise RuntimeError(f"pipeline dependency/module unavailable: {exc}") from exc

        source = (manifest.get("source_path_or_url") or {}).get("canonical_source")
        if not source:
            source = (manifest.get("source_path_or_url") or {}).get("bundled_file") or str(pdf_path.relative_to(ROOT))
        pages = load_pdf(pdf_path, manifest["source_id"], source)
        useful = _validate_inputs(manifest, questions, pages)
        page_map = {page.page: page for page in pages}
        section_spans = _assign_sections(pages, set(useful))
        eligible = [page_map[number] for number in useful]
        errors = [{"page": page.page, "status": page.metadata.get("extraction_status"),
                   "error": page.metadata.get("extraction_error")} for page in pages
                  if page.metadata.get("extraction_status") != "extracted"]
        _write_json(RESULTS / "page_errors.json", {"count": len(errors), "pages": errors})

        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is not configured; set it in starter/.env or the environment")
        model_name = os.environ.get("EMBEDDING_MODEL") or "global.cohere.embed-v4:0"
        embedder = MyEmbedder(model_name)
        config.update({
            "model": {"provider": "AWS Bedrock Runtime", "name": embedder.model_name, "dimensions": embedder.dimensions,
                      "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-2",
                      "document_input_type": "search_document", "query_input_type": "search_query"},
            "database": {"configured": True, "url": "redacted", "schema": "chunks (strategy-scoped)"},
            "environment": {"dotenv_loaded_without_override": True, "api_keys_recorded": False},
        })
        _write_json(RESULTS / "run_config.json", config)
        store = PgVectorStore(database_url, embedder.dimensions)
        try:
            store.create_schema()
        except Exception as exc:
            raise RuntimeError(f"could not initialize PostgreSQL/pgvector schema ({type(exc).__name__}); verify DATABASE_URL, server, and vector extension") from exc

        chunks_by_strategy: dict[str, list[Any]] = {"A": [], "B": []}
        for page in eligible:
            fixed_chunks = fixed_size(page, 500, 50)
            paragraph_chunks = by_paragraph(page, 1000)
            _assign_chunk_sections(fixed_chunks, page, section_spans[page.page], "fixed_size", 500, 50)
            _assign_chunk_sections(paragraph_chunks, page, section_spans[page.page], "by_paragraph")
            chunks_by_strategy["A"].extend(fixed_chunks)
            chunks_by_strategy["B"].extend(paragraph_chunks)
        for question in questions:
            filter_spec = question.get("filter")
            if filter_spec is None:
                continue
            expected_section = filter_spec["value"]
            gold_chunks = [chunk for chunk in chunks_by_strategy["B"]
                           if chunk.page == question["gold_page"] and chunk.metadata.get("section") == expected_section]
            if not gold_chunks:
                raise ValueError(f"filter section {expected_section!r} is absent from gold page/chunks for {question['id']}")
        chunks_by_strategy["C"] = chunks_by_strategy["B"]
        stats: list[dict[str, Any]] = []
        for strategy in STRATEGIES:
            chunks = chunks_by_strategy[strategy]
            embed_started = time.perf_counter()
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
            store_started = time.perf_counter()
            store.upsert(strategy, chunks, vectors)
            upsert_seconds = time.perf_counter() - store_started
            lengths = [len(chunk.text) for chunk in chunks]
            stats.append({"strategy": strategy, "chunk_count": len(chunks),
                          "total_characters": sum(lengths), "mean_chunk_characters": sum(lengths) / len(lengths) if lengths else 0,
                          "min_chunk_characters": min(lengths, default=0), "max_chunk_characters": max(lengths, default=0),
                          "embedding_seconds": store_started - embed_started,
                          "upsert_seconds": upsert_seconds,
                          "ingest_seconds": store_started - embed_started + upsert_seconds,
                          "estimated_cost_usd": None, "cost_note": "Provider usage/cost is not reported by this client; not estimated."})

        traces: list[dict[str, Any]] = []
        for strategy in STRATEGIES:
            for question in questions:
                applied_filter = _filter_for(question) if strategy == "C" else None
                try:
                    query_vector = embedder.embed_query(question["question"])
                    hits = store.search(strategy, query_vector, top_k=3, filters=applied_filter)
                    for hit in hits:
                        hit["filter_compliant"] = (
                            all(hit.get("source") == value if key == "source" else
                                hit.get("page") == value if key == "page" else
                                hit.get("metadata", {}).get(key) == value
                                for key, value in applied_filter.items()) if applied_filter else None
                        )
                    evaluation = evaluate_query(question, hits)
                    traces.append({"run_id": run_id, "strategy": strategy, "question_id": question["id"],
                                   "split": question["split"], "question": question["question"],
                                   "filter": applied_filter, "gold": {key: question.get(key) for key in
                                   ("gold_source_id", "gold_page", "evidence_quote", "expected_answer")},
                                   "hits": hits, "evaluation": evaluation})
                except Exception as exc:
                    raise RuntimeError(f"retrieval failed for {strategy}/{question['id']} ({type(exc).__name__}); see run_errors.json") from exc
        _write_json(RESULTS / "retrieval_traces.json", {"run_id": run_id, "trace_count": len(traces), "traces": traces})
        metrics = summarize(traces)
        _write_json(RESULTS / "metrics.json", {"run_id": run_id, "method": "Evidence is an exact normalized substring of a single retrieved chunk, on the gold source and page; no human-verified alternatives are counted.", "metrics": metrics})
        _write_csv(RESULTS / "chunk_stats.csv", list(stats[0]) if stats else [], stats)

        position_pages = useful[:10]
        coherence_rows = []
        for position, page_number in enumerate(position_pages, start=1):
            page = page_map[page_number]
            sample = page.text[:300]
            for strategy in STRATEGIES:
                matching = [chunk.chunk_id for chunk in chunks_by_strategy[strategy] if chunk.page == page_number]
                coherence_rows.append({"position": position, "source": page.source, "page": page_number,
                                       "source_excerpt_for_orientation_only": sample, "strategy": strategy,
                                       "candidate_chunk_ids": json.dumps(matching), "score_0_to_2": "", "reason": ""})
        _write_csv(RESULTS / "coherence_audit_template.csv", list(coherence_rows[0]) if coherence_rows else [], coherence_rows)
        metadata_rows, grounding_rows = [], []
        test_questions = [question for question in questions if question["split"] == "test"]
        trace_lookup = {(trace["strategy"], trace["question_id"]): trace for trace in traces}
        for question in test_questions:
            for strategy in STRATEGIES:
                trace = trace_lookup[(strategy, question["id"])]
                for rank, hit in enumerate(trace["hits"], start=1):
                    metadata_rows.append({"question_id": question["id"], "strategy": strategy, "rank": rank,
                                          "chunk_id": hit["chunk_id"], "source": hit["source"], "page": hit["page"],
                                          "metadata": json.dumps(hit["metadata"], ensure_ascii=False),
                                          "expected_filter": json.dumps(trace["filter"], ensure_ascii=False),
                                          "source_page_correct_0_or_1": "", "filter_compliant_0_or_1": "", "notes": ""})
                grounding_rows.append({"question_id": question["id"], "strategy": strategy,
                                       "question": question["question"], "expected_answer": question["expected_answer"],
                                       "gold_source_id": question["gold_source_id"], "gold_page": question["gold_page"],
                                       "evidence_quote": question["evidence_quote"],
                                       "top3_chunk_ids": json.dumps([hit["chunk_id"] for hit in trace["hits"]]),
                                       "score_0_to_2": "", "reason": ""})
        _write_csv(RESULTS / "metadata_audit_template.csv", list(metadata_rows[0]) if metadata_rows else [], metadata_rows)
        _write_csv(RESULTS / "grounding_audit_template.csv", list(grounding_rows[0]) if grounding_rows else [], grounding_rows)
        config.update({"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "total_wall_seconds": time.perf_counter() - run_started,
                       "page_count_physical": len(pages), "eligible_page_count": len(eligible),
                       "benchmark_count": len(questions), "trace_count": len(traces), "status": "completed"})
        _write_json(RESULTS / "run_config.json", config)
        (RESULTS / "run_errors.json").unlink(missing_ok=True)
        print(f"Completed run {run_id}; results written to {RESULTS}")
    except Exception as exc:
        config.update({"status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "error_type": type(exc).__name__, "error": str(exc),
                       "total_wall_seconds": time.perf_counter() - run_started})
        _write_json(RESULTS / "run_config.json", config)
        _write_json(RESULTS / "run_errors.json", {"run_id": run_id, "error_type": type(exc).__name__,
                                                    "error": str(exc)})
        print(f"Pipeline failed: {exc}. See {RESULTS / 'run_errors.json'}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
