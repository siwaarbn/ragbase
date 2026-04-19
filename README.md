# RAGbase 🧠 — Chat with any PDF

> Upload a PDF, ask questions in plain English, get answers grounded in the actual pages of your document.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Anthropic](https://img.shields.io/badge/Claude-Haiku-orange.svg)](https://anthropic.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-green.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. Plain-English version:

Imagine you have a giant textbook and someone asks you a question. Instead of reading the whole thing, you:
1. **Search** the index for the most relevant pages
2. **Read** only those pages
3. **Write** an answer based on what you just read

That's exactly what RAGbase does — automated:
- **Retrieval**: semantic search finds the most relevant chunks of your PDF
- **Generation**: Claude reads those chunks and writes a grounded answer
- **Verification**: source page numbers are shown with every answer

---

## Architecture

```
PDF file
    │
    ▼
[ingestion.py] — PyMuPDF loads pages, splits into 512-token chunks (50-token overlap)
    │
    ▼
[embedder.py] — sentence-transformers all-MiniLM-L6-v2 turns each chunk into a vector
    │
    ▼
[store.py] — ChromaDB persists vectors; one collection per document
    │
    ▼ (at query time, top-5 nearest chunks retrieved)
    │
[rag.py] — builds context prompt → calls Claude Haiku → returns answer + source pages
    │
    ▼
[evaluate.py] — RAGAS scores the answer: faithfulness + answer relevancy
    │
    ▼
[app.py] — Streamlit UI: upload, chat, source pages, evaluation button
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- An [Anthropic API key](https://console.anthropic.com/)

### Run it

```bash
git clone https://github.com/siwaarbn/ragbase.git
cd ragbase
uv sync
cp .env.example .env   # add your ANTHROPIC_API_KEY
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How to use it

1. Upload a PDF using the sidebar
2. Wait a few seconds while it's chunked, embedded, and indexed
3. Type a question in the chat box
4. Get an answer with source page numbers
5. Click **"Evaluate last answer"** to get RAGAS faithfulness and relevancy scores

---

## Tech Stack

| Tool | Purpose | Why we chose it |
|---|---|---|
| PyMuPDF | PDF loading | Fast, handles complex layouts |
| sentence-transformers | Embeddings | Free, runs locally, surprisingly good |
| ChromaDB | Vector store | Simple API, persistent, no server needed |
| Anthropic Claude Haiku | LLM answering | Fast, cheap, excellent instruction following |
| RAGAS | Evaluation | LLM-based metrics, more meaningful than BLEU |
| Streamlit | UI | Zero boilerplate, fast to iterate |

---

## What I learned

Building RAGbase taught me things tutorials don't cover: chunking strategy matters more than the model — a 512-token chunk with 50-token overlap is a starting point, but the right size depends on your documents. ChromaDB returns distance (lower = better) but you want a score (higher = better), so `score = 1 - distance`. RAGAS evaluation is itself LLM-powered, which means evaluation costs tokens but produces meaningful scores. And Streamlit reruns the whole script on every interaction, so you have to be deliberate about what lives in `st.session_state`.

---

## Roadmap

- [ ] Support for multiple PDFs in one session
- [ ] Swap embedding models via UI
- [ ] Export chat history as PDF
- [ ] Deploy to Streamlit Cloud with one click

---

## License

MIT
