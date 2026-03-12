"""Pytest fixtures: sample PDF with known text for highlighting."""

import tempfile
from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a 2-page PDF with bullet points, important details, and novelty phrasing."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    # Page 0
    page0 = doc.new_page()
    page0.insert_text((72, 72), "Our main contribution is a novel method for PDF analysis.", fontsize=12)
    page0.insert_text((72, 100), "We propose an AI-powered highlighter that identifies:", fontsize=12)
    page0.insert_text((90, 128), "- Bullet points and list items", fontsize=11)
    page0.insert_text((90, 148), "- Important technical details", fontsize=11)
    page0.insert_text((90, 168), "- Novelty and contribution claims", fontsize=11)
    page0.insert_text((72, 210), "Key result: 95% accuracy on academic papers.", fontsize=12)
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((72, 72), "To the best of our knowledge, we are the first to combine", fontsize=12)
    page1.insert_text((72, 100), "PyMuPDF with LLMs for automatic highlighting.", fontsize=12)
    page1.insert_text((72, 140), "Limitations:", fontsize=12)
    page1.insert_text((90, 168), "- Requires API key for OpenAI or Hugging Face", fontsize=11)
    page1.insert_text((90, 188), "- Best with text-based PDFs", fontsize=11)
    doc.save(path)
    doc.close()
    return path
