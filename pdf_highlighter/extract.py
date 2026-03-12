"""Extract text from PDF with page boundaries using PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf


def extract_text_by_page(pdf_path: str | Path) -> list[tuple[int, str]]:
    """
    Open a PDF and return a list of (page_index, text) for each page.
    Page indices are 0-based.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    doc = pymupdf.open(path)
    try:
        result: list[tuple[int, str]] = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text")  # reading order, plain text
            result.append((i, text.strip()))
        return result
    finally:
        doc.close()


def open_document(pdf_path: str | Path):
    """Open a PDF and return the PyMuPDF document (caller must close or use as context)."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return pymupdf.open(path)
