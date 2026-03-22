# PDF AI Highlighter

Highlights bullet points, important details, and novelty/contribution text in academic PDFs using AI (OpenAI, Hugging Face, or Google Gemini), then saves a new PDF with annotations.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set at least one API key:

   - **OpenAI**: Get an API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys). Set `OPENAI_API_KEY` in `.env`.
   - **Hugging Face**: Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with "Inference API" scope. Set `HUGGINGFACE_TOKEN` in `.env`.
   - **Google Gemini**: Create an API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Set `GOOGLE_API_KEY` in `.env`.

## Usage

```bash
# Using OpenAI (default model: gpt-4o)
python -m pdf_highlighter paper.pdf -o paper_highlighted.pdf --provider openai

# Using Hugging Face (uses HF serverless; optional --model-name)
python -m pdf_highlighter paper.pdf -o paper_highlighted.pdf --provider huggingface

# Using Google Gemini (default model: gemini-1.5-flash)
python -m pdf_highlighter paper.pdf -o paper_highlighted.pdf --provider gemini

# Different colors per category (bullet / important / novelty)
python -m pdf_highlighter paper.pdf --category-colors
```

## Options

- `input_pdf`: Path to the input PDF (or multiple paths).
- `-o`, `--output`: Output PDF path (default: `{input_stem}_highlighted.pdf`).
- `--provider`: `openai`, `huggingface`, or `gemini`.
- `--model-name`: Model name (overrides `.env`). For OpenAI: e.g. `gpt-4o`, `gpt-4o-mini`. For Hugging Face: e.g. `Qwen/Qwen2.5-7B-Instruct`. For Gemini: e.g. `gemini-1.5-flash`, `gemini-1.5-pro`.
- `--category-colors`: Use different highlight colors for bullets, important, and novelty.
- `-v`, `--verbose`: Show API warnings when no highlights are returned.

## Environment (.env)

**OpenAI model:** Set `OPENAI_MODEL` to choose the model when using `--provider openai` (default: `gpt-4o`). Examples: `OPENAI_MODEL=gpt-4o-mini`, `OPENAI_MODEL=gpt-4o`. Command-line `--model-name` overrides this.

**Gemini model:** Set `GEMINI_MODEL` to choose the model when using `--provider gemini` (default: `gemini-1.5-flash`). Command-line `--model-name` overrides this.

**Highlight colors** – You can override highlight colors in `.env`:

- **`PDF_HIGHLIGHT_COLOR`** – Default color for all highlights (when not using `--category-colors`). Format: `r,g,b` with values 0–1 (e.g. `1,1,0` for yellow) or hex `#RRGGBB` (e.g. `#FFFF00`).
- **`PDF_HIGHLIGHT_COLOR_BULLET`**, **`PDF_HIGHLIGHT_COLOR_IMPORTANT`**, **`PDF_HIGHLIGHT_COLOR_NOVELTY`** – Used when you pass `--category-colors`; same format as above.

Examples: `PDF_HIGHLIGHT_COLOR=#E8F4EA`, `PDF_HIGHLIGHT_COLOR=0.9,0.95,0.7`

## Output

For each input PDF, the tool writes a new PDF with the same content and highlight annotations. The original file is never modified.
