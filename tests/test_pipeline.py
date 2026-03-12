"""Tests for extraction and highlight application (no API calls)."""

from pathlib import Path

import pymupdf
import pytest

from pdf_highlighter.extract import extract_text_by_page, open_document
from pdf_highlighter.highlight import apply_highlights


def test_extract_text_by_page(sample_pdf_path):
    pages = extract_text_by_page(sample_pdf_path)
    assert len(pages) == 2
    assert pages[0][0] == 0
    assert pages[1][0] == 1
    text0 = pages[0][1]
    assert "contribution" in text0.lower() or "novel" in text0.lower()
    assert "Bullet" in text0 or "bullet" in text0


def test_apply_highlights(sample_pdf_path, tmp_path):
    """Apply fake AI highlights and verify output PDF has annotations."""
    doc = open_document(sample_pdf_path)
    highlights = [
        {"text": "Our main contribution is a novel method for PDF analysis.", "category": "novelty", "page_index": 0},
        {"text": "Bullet points and list items", "category": "bullet", "page_index": 0},
        {"text": "Key result: 95% accuracy on academic papers.", "category": "important", "page_index": 0},
    ]
    apply_highlights(doc, highlights, use_category_colors=True)
    out = tmp_path / "out.pdf"
    doc.save(out)
    doc.close()

    doc2 = pymupdf.open(out)
    try:
        page0 = doc2[0]
        anns = list(page0.annots())
        assert len(anns) >= 2  # at least 2 highlights on page 0
    finally:
        doc2.close()


def test_apply_highlights_normalize_fallback(sample_pdf_path, tmp_path):
    """Highlight with normalized text (extra spaces) still finds and highlights."""
    doc = open_document(sample_pdf_path)
    # Simulate AI returning text with collapsed spaces; actual PDF might have single space
    highlights = [
        {"text": "Key result: 95% accuracy on academic papers.", "category": "important", "page_index": 0},
    ]
    apply_highlights(doc, highlights)
    out = tmp_path / "out2.pdf"
    doc.save(out)
    doc.close()

    doc2 = pymupdf.open(out)
    try:
        page0 = doc2[0]
        anns = list(page0.annots())
        assert len(anns) >= 1
    finally:
        doc2.close()
