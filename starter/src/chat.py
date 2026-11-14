"""Streamlit chat grounded in the strategy-B RFC 9111 index."""
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "amazon.nova-lite-v1:0"
ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "rfc9111.pdf"


def _chunk_phrases(text: str) -> list[str]:
    """Return normalized full text, lines, and overlapping windows for search."""
    lines = [" ".join(line.split()) for line in text.splitlines() if line.split()]
    normalized = " ".join(text.split())
    candidates = [normalized, *lines]
    for line in lines:
        words = line.split()
        if len(words) >= 4:
            size, stride = 10, 6
            starts = list(range(0, max(1, len(words) - size + 1), stride))
            final_start = max(0, len(words) - size)
            if not starts or starts[-1] != final_start:
                starts.append(final_start)
            candidates.extend(" ".join(words[index:index + size]) for index in starts)
    return list(dict.fromkeys(value for value in candidates if value))


def _annotated_pdf(
    references: list[dict[str, Any]],
) -> tuple[bytes | None, dict[int, dict[str, int]], dict[int, bytes]]:
    """Annotate cited pages in memory and render each cited page as PNG."""
    if not PDF_PATH.is_file():
        return None, {}, {}
    try:
        import fitz
    except ImportError:
        return None, {}, {}

    document = fitz.open(PDF_PATH)
    matches: dict[int, dict[str, int]] = {}
    rectangles_by_page: dict[int, set[tuple[float, float, float, float]]] = {}
    for reference in references:
        page_number = reference.get("page")
        if not isinstance(page_number, int) or not 1 <= page_number <= len(document):
            if isinstance(page_number, int):
                matches[page_number] = {"matched": 0, "searched": 0}
            continue
        page = document[page_number - 1]
        phrases = _chunk_phrases(reference.get("text", ""))
        matched = 0
        rectangles_seen = rectangles_by_page.setdefault(page_number, set())
        for phrase in phrases:
            rectangles = page.search_for(phrase, quads=False)
            if rectangles:
                matched += 1
                for rectangle in rectangles:
                    key = tuple(round(float(value), 1) for value in rectangle)
                    if key not in rectangles_seen:
                        page.add_highlight_annot(rectangle)
                        rectangles_seen.add(key)
        status = matches.setdefault(page_number, {"matched": 0, "searched": 0})
        status["matched"] += matched
        status["searched"] += len(phrases)
    page_images = {
        page_number: document[page_number - 1].get_pixmap().tobytes("png")
        for page_number in matches
        if 1 <= page_number <= len(document)
    }
    return document.tobytes(garbage=4, deflate=True), matches, page_images


def _show_references(st: Any, references: list[dict[str, Any]]) -> None:
    for index, reference in enumerate(references, start=1):
        label = f"{reference['source']} · page {reference['page']}"
        if reference.get("score") is not None:
            label += f" · score {reference['score']:.3f}"
        with st.expander(label):
            st.caption(f"Retrieved chunk {index} · strategy B")
            st.text(reference.get("text") or "No chunk text was returned.")

    if not references:
        return
    pdf_bytes, page_matches, page_images = _annotated_pdf(references)
    if pdf_bytes is None:
        st.info("PDF highlighting is unavailable. Install pymupdf>=1.24,<2 to view highlighted source pages; retrieved chunks are shown above.")
        return
    st.caption("Page previews are rendered from an in-memory annotated copy; the bundled source PDF is unchanged.")
    st.download_button("Download annotated source PDF", pdf_bytes,
                       file_name="rfc9111-highlighted.pdf", mime="application/pdf",
                       key=f"pdf-{hash(pdf_bytes)}")
    shown_pages: set[tuple[str, int]] = set()
    for reference in references:
        page_number = reference.get("page")
        source = reference.get("source") or "RFC 9111"
        if not isinstance(page_number, int):
            continue
        page_key = (source, page_number)
        if page_key in shown_pages:
            continue
        shown_pages.add(page_key)
        label = f"{source} · page {page_number}"
        with st.expander(f"Cited page preview: {label}"):
            status = page_matches.get(page_number, {"matched": 0, "searched": 0})
            if status["matched"] == status["searched"] and status["matched"]:
                st.caption(f"All {status['matched']} searched phrases matched this page; matching text is highlighted.")
            elif status["matched"]:
                st.caption(f"Partial match: {status['matched']} of {status['searched']} searched phrases matched; only matching text is highlighted.")
            else:
                st.info("No matching text rectangles were found on this cited page. The source page is shown without an implied highlight.")
            image = page_images.get(page_number)
            if image is not None:
                st.image(image, caption=label, use_container_width=True)


def _answer(message: str, embedder: Any, store: Any, client: Any,
            model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Retrieve top-three strategy-B chunks and generate a cited answer."""
    vector = embedder.embed_query(message)
    hits = store.search("B", vector, top_k=3)
    references = [{"source": hit.get("source") or "RFC 9111", "page": hit.get("page"),
                   "score": hit.get("score"), "text": hit.get("text", "")} for hit in hits]
    if not hits:
        return {"answer": "I could not find relevant evidence in the indexed RFC 9111 corpus.", "references": []}
    evidence = "\n\n".join(f"[RFC 9111, p. {hit.get('page')}]\n{hit.get('text', '')}" for hit in hits)
    prompt = (
        "Answer using only the source excerpts below; do not rely on outside knowledge or invent details. "
        "Answer entirely in the same natural language as the user's question. "
        "If the question is in Vietnamese, answer in Vietnamese. "
        "For mixed-language or unclear questions, answer in the dominant language; prefer Vietnamese when Vietnamese is dominant. "
        "Preserve RFC technical terms and directive names, and do not unnecessarily translate identifiers or source quotations. "
        "Cite every factual claim as [RFC 9111, p. N], using the excerpt page numbers. "
        "If these excerpts do not establish an answer, say the corpus evidence is insufficient.\n\n"
        f"SOURCE EXCERPTS:\n{evidence}\n\nUSER QUESTION:\n{message}"
    )
    response = client.converse(modelId=model, messages=[{"role": "user", "content": [{"text": prompt}]}],
                               inferenceConfig={"maxTokens": 800, "temperature": 0.2})
    answer = "".join(part.get("text", "") for part in response["output"]["message"]["content"])
    if not answer.strip():
        raise RuntimeError("Bedrock returned an empty response")
    return {"answer": answer, "references": references}


def _load_services() -> tuple[Any, Any, Any, str]:
    """Load local configuration and initialize retrieval and Bedrock clients."""
    from dotenv import load_dotenv

    env_path = ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured; set it in starter/.env or the environment.")

    import boto3
    from src.embeddings import MyEmbedder
    from src.store import PgVectorStore

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-2"
    embedder = MyEmbedder(os.environ.get("EMBEDDING_MODEL") or "global.cohere.embed-v4:0")
    store = PgVectorStore(database_url, embedder.dimensions)
    client = boto3.client("bedrock-runtime", region_name=region)
    return embedder, store, client, os.environ.get("CHAT_MODEL") or DEFAULT_MODEL


def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        raise SystemExit("Streamlit is unavailable; install starter/requirements.txt") from None

    st.set_page_config(page_title="RFC 9111 Chat", page_icon="RFC", layout="centered")
    st.title("RFC 9111 Chat")
    st.caption("Ask about HTTP caching. Answers are grounded in locally indexed RFC 9111 chunks.")

    @st.cache_resource(show_spinner="Connecting to the RFC index and Bedrock...")
    def services() -> tuple[Any, Any, Any, str]:
        return _load_services()

    try:
        embedder, store, client, model = services()
    except Exception as exc:
        st.error("Chat services could not be initialized. Check starter/.env, database availability, "
                 f"and AWS credentials. ({type(exc).__name__})")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _show_references(st, message.get("references", []))

    question = st.chat_input("Ask a question about RFC 9111...", max_chars=4000)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the RFC and preparing an answer..."):
                try:
                    result = _answer(question, embedder, store, client, model)
                    st.markdown(result["answer"])
                    _show_references(st, result["references"])
                    st.session_state.messages.append({"role": "assistant", **result,
                                                      "content": result["answer"]})
                except Exception as exc:
                    error = ("Chat is temporarily unavailable. Check database connectivity, "
                             f"AWS credentials, and model access. ({type(exc).__name__})")
                    st.error(error)
                    st.session_state.messages.append({"role": "assistant", "content": error})


if __name__ == "__main__":
    main()
