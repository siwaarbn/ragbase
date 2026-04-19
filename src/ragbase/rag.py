import os
from dataclasses import dataclass
import anthropic
from dotenv import load_dotenv
from ragbase.store import query_store, SearchResult

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
TOP_K = 5


@dataclass
class RAGAnswer:
    answer: str
    source_pages: list[int]
    contexts: list[str]
    question: str


def _build_prompt(question: str, contexts: list[SearchResult]) -> str:
    context_block = "\n\n".join(f"[Page {r.page}]\n{r.text}" for r in contexts)
    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context_block}

Question: {question}

Answer:"""


def ask(doc_id: str, question: str) -> RAGAnswer:
    contexts = query_store(doc_id, question, top_k=TOP_K)
    prompt = _build_prompt(question, contexts)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    answer_text = message.content[0].text
    source_pages = sorted({r.page for r in contexts})
    return RAGAnswer(
        answer=answer_text,
        source_pages=source_pages,
        contexts=[r.text for r in contexts],
        question=question,
    )
