"""Read and normalize source data."""
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page, recording pages without extractable text.

    Detect and record pages with no text. OCR, when needed, is a separate step.
    """
    try:
        reader = PdfReader(str(path), strict=True)
        if reader.is_encrypted:
            if reader.decrypt("") == 0:
                raise ValueError("PDF is encrypted and cannot be opened without a password")

        pages: list[DocumentPage] = []
        for page_number, pdf_page in enumerate(reader.pages, start=1):
            try:
                extracted = pdf_page.extract_text()
            except Exception as exc:
                pages.append(
                    DocumentPage(
                        document_id=document_id,
                        source=source,
                        page=page_number,
                        text="",
                        metadata={"extraction_status": "error", "extraction_error": str(exc)},
                    )
                )
                continue

            text = normalize_text(extracted or "")
            status = "extracted" if text else "no_text"
            pages.append(
                DocumentPage(
                    document_id=document_id,
                    source=source,
                    page=page_number,
                    text=text,
                    metadata={"extraction_status": status},
                )
            )
        return pages
    except Exception as exc:
        if isinstance(exc, (ValueError, FileNotFoundError, PermissionError)):
            raise
        raise ValueError(f"Unable to read PDF {path!s}: {exc}") from exc


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving line structure."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\f", "\n")

    lines: list[str] = []
    for line in text.split("\n"):
        # Remove control/format characters except tabs, which are whitespace.
        line = "".join(
            char for char in line
            if char == "\t" or unicodedata.category(char)[0] != "C"
        )
        line = line.expandtabs(4)
        indentation = line[: len(line) - len(line.lstrip(" "))]
        content = " ".join(line[len(indentation) :].split())
        line = indentation + content
        lines.append(line)

    # Trim outer blank lines and reduce excessive vertical whitespace without
    # joining paragraphs, bullets, or indented code lines together.
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    normalized: list[str] = []
    blank_run = 0
    for line in lines:
        if line:
            blank_run = 0
            normalized.append(line)
        else:
            blank_run += 1
            if blank_run == 1:
                normalized.append("")
    return "\n".join(normalized)
