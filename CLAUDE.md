# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Avantika University Student Knowledge Base Chatbot

A RAG-based conversational assistant that answers student queries using official Avantika University documents (Student Handbook + Academic Ordinances for B.Tech, B.Des, MBA, M.Des, M.Tech, MCA).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build knowledge base (required once, or after updating knowledge_base/*.txt)
python ingest.py

# Run the chatbot
streamlit run app.py
```

## Architecture

**Two-phase pipeline:**

1. **Ingestion** (`ingest.py`): Reads `knowledge_base/*.txt` → splits on `===` section headers then by paragraph → generates embeddings with `sentence-transformers/all-MiniLM-L6-v2` → stores in a FAISS `IndexFlatIP` (cosine similarity via normalized vectors). Outputs to `vector_store/index.faiss` + `vector_store/metadata.json`.

2. **Retrieval + Generation** (`app.py`): Embeds user query → `IndexFlatIP.search()` returns top-5 chunks (score > 0.2) → chunks formatted into system prompt → streamed response via Groq API (`llama-3.3-70b-versatile`).

**Key files:**
- `knowledge_base/student_handbook.txt` — structured content from the AU Student Handbook 2025-26
- `knowledge_base/ordinance_programs.txt` — Academic Ordinances 26–31 (all programs, grading, ATKT, award of degree)
- `vector_store/` — generated at runtime by `ingest.py`; never commit this directory

## Environment

Requires `GROQ_API_KEY` in a `.env` file. Copy `.env.example` to `.env` and fill in the key.

## Updating the Knowledge Base

To add or update content:
1. Edit or add `.txt` files in `knowledge_base/`
2. Re-run `python ingest.py` to rebuild the FAISS index
3. The app auto-detects a missing index and rebuilds on first launch

## Design Theme

Avantika University brand colors used throughout the Streamlit UI:
- Primary: `#003087` (deep navy)
- Accent: `#FF6600` (orange — MIT Group)
- Background: `#EEF3FF` (light blue-white)
