"""Apply highlight annotations to a PDF from AI-returned spans."""

from __future__ import annotations

import os
import re
from typing import Any

import pymupdf

from .ai_analyze import CATEGORY_BULLET, CATEGORY_IMPORTANT, CATEGORY_NOVELTY

# Env keys for highlight colors (RGB 0-1: "r,g,b" e.g. "1,1,0", or hex "#RRGGBB")
ENV_HIGHLIGHT_COLOR = "PDF_HIGHLIGHT_COLOR"
ENV_HIGHLIGHT_COLOR_BULLET = "PDF_HIGHLIGHT_COLOR_BULLET"
ENV_HIGHLIGHT_COLOR_IMPORTANT = "PDF_HIGHLIGHT_COLOR_IMPORTANT"
ENV_HIGHLIGHT_COLOR_NOVELTY = "PDF_HIGHLIGHT_COLOR_NOVELTY"

# Defaults (yellow; light gray bullets; light green novelty) when env not set
_DEFAULT_COLORS = {
    CATEGORY_BULLET: (0.85, 0.85, 0.75),
    CATEGORY_IMPORTANT: (1.0, 1.0, 0.0),
    CATEGORY_NOVELTY: (0.6, 1.0, 0.6),
}
_DEFAULT_SINGLE = (1.0, 1.0, 0.0)


def _parse_color(value: str | None) -> tuple[float, float, float] | None:
    """Parse color from env: 'r,g,b' (0-1) or '#RRGGBB'. Returns (r,g,b) in 0-1 or None."""
    if not value or not value.strip():
        return None
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
        if len(value) == 6 and all(c in "0123456789AaBbCcDdEeFf" for c in value):
            r = int(value[0:2], 16) / 255.0
            g = int(value[2:4], 16) / 255.0
            b = int(value[4:6], 16) / 255.0
            return (r, g, b)
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        return None
    try:
        r, g, b = (float(p) for p in parts)
        if not (0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1):
            return None
        return (r, g, b)
    except ValueError:
        return None


def get_highlight_colors(use_category_colors: bool) -> tuple[dict[str, tuple[float, float, float]], tuple[float, float, float]]:
    """
    Return (category_colors_map, default_color) from environment or built-in defaults.
    use_category_colors: when True, per-category env vars are used for bullet/important/novelty.
    """
    def get_default_color(env_key: str, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        parsed = _parse_color(os.environ.get(env_key))
        return parsed if parsed is not None else fallback

    default_single = get_default_color(ENV_HIGHLIGHT_COLOR, _DEFAULT_SINGLE)
    category_colors = {
        CATEGORY_BULLET: get_default_color(ENV_HIGHLIGHT_COLOR_BULLET, _DEFAULT_COLORS[CATEGORY_BULLET]),
        CATEGORY_IMPORTANT: get_default_color(ENV_HIGHLIGHT_COLOR_IMPORTANT, _DEFAULT_COLORS[CATEGORY_IMPORTANT]),
        CATEGORY_NOVELTY: get_default_color(ENV_HIGHLIGHT_COLOR_NOVELTY, _DEFAULT_COLORS[CATEGORY_NOVELTY]),
    }
    return category_colors, default_single


def _normalize_text(s: str) -> str:
    """Collapse whitespace and remove common hyphenation/line-break artifacts for fallback search."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # Optional: remove soft hyphens that might appear in PDF text as hyphen + newline
    s = re.sub(r"-\s+", "", s)
    return s


def _add_highlight(
    page: pymupdf.Page,
    text: str,
    color: tuple[float, float, float],
    quads: bool = False,
) -> bool:
    """
    Search for text on page and add highlight. Returns True if at least one match.
    """
    rects = page.search_for(text, quads=quads)
    if rects:
        annot = page.add_highlight_annot(rects)
        annot.set_colors(stroke=color)
        annot.update()
        return True
    return False


def apply_highlights(
    doc: pymupdf.Document,
    highlights: list[dict[str, Any]],
    use_category_colors: bool = False,
) -> None:
    """
    Apply highlight annotations to doc. highlights is a list of
    { "text", "category", "page_index" } from AI analysis.
    Colors are read from env (PDF_HIGHLIGHT_COLOR, PDF_HIGHLIGHT_COLOR_BULLET, etc.) or defaults.
    Modifies doc in place.
    """
    category_colors, default_color = get_highlight_colors(use_category_colors)
    for item in highlights:
        page_index = item.get("page_index", 0)
        text = (item.get("text") or "").strip()
        if not text or page_index < 0 or page_index >= len(doc):
            continue
        page = doc[page_index]
        category = item.get("category") or "important"
        color = category_colors.get(category, default_color) if use_category_colors else default_color

        if _add_highlight(page, text, color):
            continue
        # Fallback: normalize and retry
        normalized = _normalize_text(text)
        if normalized and normalized != text:
            if _add_highlight(page, normalized, color):
                continue
        # Try with quads in case of rotated text
        if _add_highlight(page, text, color, quads=True):
            continue
        if normalized and _add_highlight(page, normalized, color, quads=True):
            continue
        # Could add block-based fallback here (optional)
