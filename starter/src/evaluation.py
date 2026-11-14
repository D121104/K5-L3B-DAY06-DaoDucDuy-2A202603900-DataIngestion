"""Exact, reproducible retrieval metrics (not human grounding judgments)."""
from typing import Any


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check exact gold evidence on one chunk and expose transparent labels.

    Accept any documented alternative gold evidence. Do not infer human grounding
    or chunk coherence labels from exact string match.
    """
    evidence = question.get("evidence_quotes") or [question["evidence_quote"]]
    if isinstance(evidence, str):
        evidence = [evidence]
    gold_sources = question.get("gold_source_ids") or [question["gold_source_id"]]
    gold_pages = question.get("gold_pages") or [question["gold_page"]]
    normalized_quotes = [_normalize(quote) for quote in evidence if isinstance(quote, str) and quote.strip()]
    matching_ranks = []
    labels = []
    for rank, hit in enumerate(hits, start=1):
        source_match = hit.get("document_id") in gold_sources
        page_match = hit.get("page") in gold_pages
        evidence_match = any(quote in _normalize(hit.get("text", "")) for quote in normalized_quotes)
        exact_hit = source_match and page_match and evidence_match
        if exact_hit:
            matching_ranks.append(rank)
        labels.append({"rank": rank, "chunk_id": hit.get("chunk_id"), "gold_source_match": source_match,
                       "gold_page_match": page_match, "exact_evidence_in_single_chunk": evidence_match,
                       "exact_gold_hit": exact_hit, "label_type": "automatic_exact_string_proxy_not_human_assessed"})
    return {"hit_at_1": bool(matching_ranks and matching_ranks[0] == 1),
            "evidence_hit_at_3": bool(matching_ranks), "matching_ranks": matching_ranks,
            "hit_labels": labels, "evidence_algorithm": "casefold and collapse whitespace; one evidence quote must occur as a substring within one retrieved chunk, whose document_id and physical page must match a declared gold source/page; no across-chunk aggregation or human alternatives"}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate separately by strategy and predeclared split with denominators."""
    result: dict[str, Any] = {}
    for strategy in sorted({trace["strategy"] for trace in traces}):
        result[strategy] = {}
        for split in ("dev", "test"):
            selected = [trace for trace in traces if trace["strategy"] == strategy and trace["split"] == split]
            hit1 = sum(bool(trace["evaluation"]["hit_at_1"]) for trace in selected)
            hit3 = sum(bool(trace["evaluation"]["evidence_hit_at_3"]) for trace in selected)
            result[strategy][split] = {"denominator": len(selected), "hit_at_1_count": hit1,
                                       "hit_at_1_rate": hit1 / len(selected) if selected else None,
                                       "evidence_hit_at_3_count": hit3,
                                       "evidence_hit_at_3_rate": hit3 / len(selected) if selected else None}
    return result
