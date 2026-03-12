"""CLI for PDF AI Highlighter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .ai_analyze import get_highlights
from .extract import extract_text_by_page, open_document
from .highlight import apply_highlights


def _check_api_key(provider: str) -> None:
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Error: Set OPENAI_API_KEY in .env or environment (see .env.example).")
    if provider == "huggingface" and not os.environ.get("HUGGINGFACE_TOKEN"):
        raise SystemExit("Error: Set HUGGINGFACE_TOKEN in .env or environment (see .env.example).")


def _default_output_path(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}_highlighted.pdf"


def run(
    input_pdf: Path,
    output_pdf: Path | None = None,
    provider: str = "openai",
    model_name: str | None = None,
    category_colors: bool = False,
    verbose: bool = False,
) -> None:
    """Run the highlighter on one PDF."""
    load_dotenv()
    _check_api_key(provider)
    out = output_pdf or _default_output_path(input_pdf)
    out = Path(out)

    pages = extract_text_by_page(input_pdf)
    if not pages:
        raise SystemExit("No text extracted from PDF (possibly image-only).")

    if verbose:
        import warnings
        warnings.simplefilter("always", UserWarning)
    highlights = get_highlights(
        pages,
        provider=provider,
        model_name=model_name,
        api_key=os.environ.get("OPENAI_API_KEY"),
        token=os.environ.get("HUGGINGFACE_TOKEN"),
    )
    if not highlights:
        print("No highlights returned by AI; saving copy without new annotations.")
        if verbose:
            print("Tip: run with -v to see API warnings. Check API key, model name, and that the PDF has extractable text.")
    else:
        doc = open_document(input_pdf)
        try:
            apply_highlights(doc, highlights, use_category_colors=category_colors)
            doc.save(out, deflate=True)
        finally:
            doc.close()
        print(f"Saved highlighted PDF to {out}")
        return

    # No highlights: still save a copy so user gets an output file
    import pymupdf
    doc = pymupdf.open(input_pdf)
    try:
        doc.save(out, deflate=True)
    finally:
        doc.close()
    print(f"No highlights applied; saved copy to {out}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Highlight bullet points, important details, and novelty in academic PDFs using AI.",
    )
    parser.add_argument(
        "input_pdf",
        type=Path,
        nargs="+",
        help="Path(s) to input PDF file(s)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <input_stem>_highlighted.pdf). For multiple inputs, use one -o per file or output dir.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "huggingface"),
        default="openai",
        help="AI provider (default: openai)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (e.g. gpt-4o, mistralai/Mixtral-8x7B-Instruct-v0.1). Defaults per provider.",
    )
    parser.add_argument(
        "--category-colors",
        action="store_true",
        help="Use different highlight colors for bullet / important / novelty",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show warnings (e.g. API errors) when no highlights are returned",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input_pdf]
    output = args.output
    # If single input and -o is a file, use it for that input
    # If multiple inputs, -o could be a directory or we use default per file
    if len(inputs) == 1:
        run(
            inputs[0],
            output_pdf=output,
            provider=args.provider,
            model_name=args.model_name,
            category_colors=args.category_colors,
            verbose=args.verbose,
        )
    else:
        # Multiple inputs: -o as output directory, or default per file
        out_dir = None
        if output:
            p = Path(output)
            if p.suffix.lower() != ".pdf":
                out_dir = p
                out_dir.mkdir(parents=True, exist_ok=True)
        for i, inp in enumerate(inputs):
            if out_dir:
                out_path = out_dir / f"{inp.stem}_highlighted.pdf"
            elif output and i == 0:
                out_path = Path(output)
            else:
                out_path = None
            run(
                inp,
                output_pdf=out_path,
                provider=args.provider,
                model_name=args.model_name,
                category_colors=args.category_colors,
                verbose=args.verbose,
            )


if __name__ == "__main__":
    main()
