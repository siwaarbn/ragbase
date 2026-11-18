# RAGbase

> ask questions about a PDF and get answers that point back to the pages they came from

## Why I built this

I wanted to understand retrieval-augmented generation properly, not just call a framework that hides every step. So I wrote each part myself: loading the PDF, chunking it, embedding the chunks, storing them, retrieving the right ones and building the prompt.

The part I cared about most was being able to check the answers. Every answer lists the pages it used, and there's a button that scores the last answer with RAGAS, so I can see when the model makes something up instead of guessing.

## How it works

1. **Load and chunk.** PyMuPDF reads the text page by page. Each page is split into chunks of roughly 512 tokens with a 50-token overlap. Every chunk keeps its page number.
2. **Embed.** `all-MiniLM-L6-v2` from sentence-transformers turns each chunk into a vector. It runs locally, so indexing a PDF costs nothing.
3. **Store.** ChromaDB keeps the vectors on disk in `chroma_db/`, one collection per document, using cosine distance.
4. **Retrieve and answer.** The question is embedded the same way and the 5 closest chunks go into the prompt. The model (Llama 3.3 70B through the Groq API) is told to answer only from that context, and to say so when the answer isn't there.
5. **Evaluate.** "Evaluate last answer" runs RAGAS faithfulness and answer relevancy on the last reply. RAGAS needs an LLM as a judge; I use Claude Haiku for that.

```
app.py                  Streamlit UI: upload, chat, sources, evaluation
src/ragbase/
  ingestion.py          PDF loading and chunking
  embedder.py           sentence-transformers wrapper
  store.py              ChromaDB indexing and search
  rag.py                prompt building and the Groq call
  evaluate.py           RAGAS scoring
```

## Running it

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/siwaarbn/ragbase
cd ragbase
uv sync
cp .env.example .env
uv run streamlit run app.py
```

In `.env`, `GROQ_API_KEY` is required for answering. `ANTHROPIC_API_KEY` is only used by the evaluation button, so you can leave it out if you don't need scores.

Then open `http://localhost:8501`, upload a PDF in the sidebar and start asking.

## Things I learned

- Chunk size mattered more than which model answered. 512 tokens with 50 overlap is a starting point, not a rule; it depends a lot on the document.
- ChromaDB returns a distance (lower is closer), so I convert it with `score = 1 - distance` to get something that reads like a similarity.
- RAGAS is itself LLM-based, so every evaluation costs tokens. That's why it's a button and not automatic.
- Streamlit reruns the whole script on every click. Anything that should survive, like the chat history or the current document, has to live in `st.session_state`.
- The first version used Claude for answering. I later switched to Groq; because the LLM call lives in one function in `rag.py`, the switch was a small change.

## Known issues / stuff I want to fix

- token counts are estimated from word counts, not a real tokenizer
- scanned PDFs without a text layer give you nothing, there's no OCR
- the collection is named after the file name, so two different PDFs with the same name overwrite each other
- only one document at a time

## License

MIT
