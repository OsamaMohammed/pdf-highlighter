"""AI analysis: ask model to identify bullet points, important details, and novelty text."""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Optional imports for providers
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None  # type: ignore


# Maximum chars per chunk to stay within context (conservative for HF models)
CHUNK_CHAR_LIMIT = 6000

# Number of pages to send in one API call (fewer calls, more context)
PAGES_PER_CHUNK = 4

# Categories for highlights
CATEGORY_BULLET = "bullet"
CATEGORY_IMPORTANT = "important"
CATEGORY_NOVELTY = "novelty"


def _chunk_pages(pages: list[tuple[int, str]], size: int = PAGES_PER_CHUNK) -> list[list[tuple[int, str]]]:
    """Split pages into chunks of up to `size` (non-empty pages only)."""
    chunk: list[tuple[int, str]] = []
    chunks: list[list[tuple[int, str]]] = []
    for page_index, text in pages:
        if not text.strip():
            continue
        chunk.append((page_index, text))
        if len(chunk) >= size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks


def _system_prompt() -> str:
    return """You analyze excerpts from academic papers. For each excerpt you receive, identify:
1. All bullet points (list items).
2. The most important factual or technical details (key findings, numbers, definitions).
3. Any novelty/contribution claims (e.g. "we propose", "our contribution", "first to", "novel", "main contribution").

You must respond with a JSON array. Each item has:
- "text": exact verbatim substring from the excerpt (copy-paste, do not paraphrase).
- "category": one of "bullet", "important", "novelty".
- "page": integer 1, 2, 3, ... indicating which page in the excerpt (first block=1, second=2, etc.) the text appears on.

Return only the JSON array, no other text. If nothing to highlight, return []."""


def _user_prompt(page_index: int, text: str) -> str:
    return f"""Excerpt from page {page_index + 1} (0-based page index {page_index}):

---
{text}
---

List all bullet points, important details, and novelty/contribution phrases as a JSON array of objects with "text", "category", and "page" (use 1 for this single page). Use exact verbatim strings from the excerpt."""


def _user_prompt_multi(chunk: list[tuple[int, str]]) -> str:
    """Build prompt for multiple pages; each item in the response must include "page" (1-based index within this excerpt)."""
    blocks = []
    for i, (page_index, text) in enumerate(chunk):
        one_based = i + 1
        truncated = text[: CHUNK_CHAR_LIMIT] + "\n[... truncated ...]" if len(text) > CHUNK_CHAR_LIMIT else text
        blocks.append(f"## Page {one_based} (document page {page_index + 1})\n\n{truncated}")
    excerpt = "\n\n---\n\n".join(blocks)
    return f"""Excerpt with {len(chunk)} pages:

{excerpt}

---

List all bullet points, important details, and novelty/contribution phrases as a JSON array. Each object must have "text", "category" (one of "bullet", "important", "novelty"), and "page" (integer 1 to {len(chunk)} indicating which page above the text is on). Use exact verbatim strings from the excerpt."""


def _parse_json_from_response(raw: str) -> list[dict[str, Any]]:
    """Extract JSON array from model response, handling markdown code blocks and wrapper objects."""
    raw = raw.strip()
    # Remove markdown code block if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items") or data.get("highlights") or data.get("highlights_list")
        if not isinstance(items, list):
            return []
    else:
        return []
    return [x for x in items if isinstance(x, dict) and x.get("text")]


# Env key for OpenAI model (e.g. gpt-4o, gpt-4o-mini). Fallback: gpt-4o
ENV_OPENAI_MODEL = "OPENAI_MODEL"
DEFAULT_OPENAI_MODEL = "gpt-4o"

# Gemini configuration
ENV_GEMINI_MODEL = "GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


def get_highlights_openai(
    pages: list[tuple[int, str]],
    model_name: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Use OpenAI API with structured output. Returns list of { "text", "category", "page_index" }.
    Model: --model-name > OPENAI_MODEL in .env > gpt-4o.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI package not installed. pip install openai")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        key = key.strip().strip('"\'')
    if not key:
        raise ValueError("Set OPENAI_API_KEY in .env or pass api_key")
    model = (model_name or os.environ.get(ENV_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL

    client = OpenAI(api_key=key)
    results: list[dict[str, Any]] = []
    chunks = _chunk_pages(pages, PAGES_PER_CHUNK)

    for chunk in tqdm(chunks, desc="AI highlighting", unit="chunk"):
        if not chunk:
            continue
        first_page_index = chunk[0][0]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": _user_prompt_multi(chunk)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "highlights",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string", "description": "Exact verbatim substring"},
                                            "category": {
                                                "type": "string",
                                                "enum": ["bullet", "important", "novelty"],
                                                "description": "Category of highlight",
                                            },
                                            "page": {"type": "integer", "description": "1-based page within this excerpt (1=first page, 2=second, ...)"},
                                        },
                                        "required": ["text", "category", "page"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["items"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
            content = response.choices[0].message.content
            if not content:
                continue
            data = json.loads(content)
            items = data.get("items") if isinstance(data, dict) else []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("text"):
                    continue
                page_one = max(1, min(int(item.get("page") or 1), len(chunk)))
                page_index = first_page_index + page_one - 1
                results.append({
                    "text": item["text"].strip(),
                    "category": item.get("category") or "important",
                    "page_index": page_index,
                })
        except (json.JSONDecodeError, KeyError):
            try:
                parsed = _parse_json_from_response(content)
                for item in parsed:
                    if not isinstance(item, dict) or not item.get("text"):
                        continue
                    page_one = max(1, min(int(item.get("page") or 1), len(chunk)))
                    page_index = first_page_index + page_one - 1
                    results.append({
                        "text": (item.get("text") or "").strip(),
                        "category": item.get("category") or "important",
                        "page_index": page_index,
                    })
            except Exception:
                pass

    return results


def _get_content_from_hf_response(response: Any) -> str:
    """Extract text content from Hugging Face chat_completion response (object or dict)."""
    try:
        choices = response.choices
    except AttributeError:
        choices = response.get("choices") if isinstance(response, dict) else []
    if not choices:
        return ""
    first = choices[0]
    try:
        message = first.message
    except AttributeError:
        message = first.get("message") if isinstance(first, dict) else {}
    try:
        return (message.content or "") if message else ""
    except AttributeError:
        return (message.get("content") or "") if isinstance(message, dict) else ""


def _get_text_from_gemini_response(response: Any) -> str:
    """Extract text content from Gemini generate_content response."""
    if response is None:
        return ""
    # google-genai responses typically expose a .text convenience attribute
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    # Fallback to candidates structure if needed
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    first = candidates[0]
    content = getattr(first, "content", None)
    if not content:
        return ""
    parts = getattr(content, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        t = getattr(part, "text", None)
        if isinstance(t, str):
            texts.append(t)
    return "\n".join(texts).strip()


# Default model for Hugging Face: use HF's own serverless (hf-inference), not third-party providers like Together.
# Mixtral-8x7B is deprecated on Together (410 Gone). Qwen2.5-7B is available on hf-inference.
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def get_highlights_huggingface(
    pages: list[tuple[int, str]],
    model_name: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """
    Use Hugging Face Inference API. Returns list of { "text", "category", "page_index" }.
    Uses provider="hf-inference" so requests go to Hugging Face's serverless API, not deprecated third-party providers.
    """
    if InferenceClient is None:
        raise RuntimeError("huggingface_hub not installed. pip install huggingface_hub")
    tok = token or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        tok = tok.strip().strip('"\'')
    if not tok:
        raise ValueError("Set HUGGINGFACE_TOKEN in .env or pass token")

    # Use Hugging Face's own inference (avoids Together/410 Gone for deprecated models)
    client = InferenceClient(token=tok, provider="hf-inference")
    model = model_name or DEFAULT_HF_MODEL
    results: list[dict[str, Any]] = []
    chunks = _chunk_pages(pages, PAGES_PER_CHUNK)

    for chunk in tqdm(chunks, desc="AI highlighting", unit="chunk"):
        if not chunk:
            continue
        first_page_index = chunk[0][0]
        prompt = f"{_system_prompt()}\n\n{_user_prompt_multi(chunk)}"
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=2048,
            )
            content = _get_content_from_hf_response(response)
            if not content:
                continue
            parsed = _parse_json_from_response(content)
            for item in parsed:
                if not isinstance(item, dict) or not item.get("text"):
                    continue
                try:
                    page_one = max(1, min(int(item.get("page") or 1), len(chunk)))
                except (TypeError, ValueError):
                    page_one = 1
                page_index = first_page_index + page_one - 1
                results.append({
                    "text": (item.get("text") or "").strip(),
                    "category": item.get("category") or "important",
                    "page_index": page_index,
                })
        except Exception as e:
            import warnings
            warnings.warn(
                f"Hugging Face API error on chunk (pages {chunk[0][0] + 1}-{chunk[-1][0] + 1}): {e}. Check HUGGINGFACE_TOKEN and model name.",
                UserWarning,
                stacklevel=2,
            )

    return results


def get_highlights_gemini(
    pages: list[tuple[int, str]],
    model_name: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Use Google Gemini API. Returns list of { "text", "category", "page_index" }.
    Model: --model-name > GEMINI_MODEL in .env > gemini-1.5-flash.
    """
    if genai is None:
        raise RuntimeError("google-genai package not installed. pip install google-genai")
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if key:
        key = key.strip().strip('"\'')
    if not key:
        raise ValueError("Set GOOGLE_API_KEY in .env or pass api_key")
    model_name_or_default = (model_name or os.environ.get(ENV_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

    client = genai.Client(api_key=key)

    results: list[dict[str, Any]] = []
    chunks = _chunk_pages(pages, PAGES_PER_CHUNK)

    for chunk in tqdm(chunks, desc="AI highlighting", unit="chunk"):
        if not chunk:
            continue
        first_page_index = chunk[0][0]
        try:
            response = client.models.generate_content(
                model=model_name_or_default,
                contents=_user_prompt_multi(chunk),
                config=types.GenerateContentConfig(
                    system_instruction=_system_prompt(),
                ),
            )
            content = _get_text_from_gemini_response(response)
            if not content:
                continue
        except Exception as e:  # pragma: no cover - network/API issues
            import warnings

            warnings.warn(
                f"Gemini API error on chunk (pages {chunk[0][0] + 1}-{chunk[-1][0] + 1}): {e}. Check GOOGLE_API_KEY and model name.",
                UserWarning,
                stacklevel=2,
            )
            continue
        
        parsed = _parse_json_from_response(content)
        for item in parsed:
            if not isinstance(item, dict) or not item.get("text"):
                continue
            try:
                page_one = max(1, min(int(item.get("page") or 1), len(chunk)))
            except (TypeError, ValueError):
                page_one = 1
            page_index = first_page_index + page_one - 1
            results.append(
                {
                    "text": (item.get("text") or "").strip(),
                    "category": item.get("category") or "important",
                    "page_index": page_index,
                }
            )

    return results


def get_highlights(
    pages: list[tuple[int, str]],
    provider: str = "openai",
    model_name: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Get highlights from AI. provider in ("openai", "huggingface", "gemini").
    kwargs passed to the provider (e.g. api_key, token).
    """
    if provider == "openai":
        return get_highlights_openai(
            pages,
            model_name=model_name,
            api_key=kwargs.get("api_key"),
        )
    if provider == "huggingface":
        return get_highlights_huggingface(
            pages,
            model_name=model_name or DEFAULT_HF_MODEL,
            token=kwargs.get("token"),
        )
    if provider == "gemini":
        return get_highlights_gemini(
            pages,
            model_name=model_name,
            api_key=kwargs.get("api_key"),
        )
    raise ValueError(f"Unknown provider: {provider}")
