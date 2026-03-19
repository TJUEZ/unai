# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese AI-generated text (AIGC) detection web application built with Flask and BERT/RoBERTa model. Provides a reader-style interface for detecting whether text is AI-generated or human-written.

## Running the Application

```bash
# Install dependencies
pip install -r requirements_web.txt

# Start the Flask server
python app.py
```

The server runs on http://localhost:5000

## Key Files

- `app.py` - Flask backend with REST API endpoints for text detection
- `templates/index.html` - Frontend UI with theme system, chunk detection, and auto-save
- `requirements_web.txt` - Python dependencies
- Model files in root directory: `pytorch_model.bin`, `config.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/api/detect` | POST | Simple text detection |
| `/api/detect-full` | POST | Full detection with chunking |
| `/api/detect-chunk` | POST | Single chunk detection |
| `/api/upload` | POST | File upload (txt/docx/pdf) |

## Architecture

- **Backend**: Flask + PyTorch + Transformers (BERT model)
- **Frontend**: Vanilla HTML/CSS/JS with CSS custom properties for theming
- **Model**: Chinese RoBERTa fine-tuned for binary classification (AI vs Human text)

## Important Notes

- Model is loaded globally at startup and uses `MODEL_PATH = "."` (current directory)
- Detection returns probability that text is AI-generated (0 = human, 1 = AI)
- Frontend uses localStorage for auto-save and theme persistence
- Edit detection uses debounce (500ms) to prevent duplicate API calls

## Gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available skills: /office-hours, /plan-ceo-review, /plan-eng-review, /plan-design-review, /design-consultation, /review, /ship, /browse, /qa, /qa-only, /design-review, /setup-browser-cookies, /retro, /debug, /document-release
