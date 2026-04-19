import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAGbase — Chat with your PDF",
    page_icon="📄",
    layout="centered",
)

st.markdown(
    """
    <style>
    body, .stApp { background-color: #0e1117; color: #e0e0e0; }
    .chat-bubble-user {
        background: #1e2a3a; border-radius: 12px;
        padding: 10px 14px; margin: 6px 0; text-align: right;
    }
    .chat-bubble-assistant {
        background: #1a1f2e; border-radius: 12px;
        padding: 10px 14px; margin: 6px 0;
    }
    .source-tag {
        font-size: 0.75rem; color: #7a8fa8;
        margin-top: 4px;
    }
    .score-box {
        background: #1e2a3a; border-radius: 8px;
        padding: 8px 14px; font-size: 0.85rem;
        display: inline-block; margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None

st.title("📄 RAGbase")
st.caption("Upload a PDF and ask questions about its contents.")

with st.sidebar:
    st.header("Upload PDF")
    uploaded = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

    if uploaded is not None:
        new_doc_id = uploaded.name

        if st.session_state.doc_id != new_doc_id:
            with st.spinner("Ingesting and indexing PDF…"):
                from ragbase.ingestion import ingest_pdf
                from ragbase.store import index_chunks

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                try:
                    chunks = ingest_pdf(tmp_path)
                    index_chunks(new_doc_id, chunks)
                    st.session_state.doc_id = new_doc_id
                    st.session_state.messages = []
                    st.session_state.last_eval = None
                    st.success(f"Indexed {len(chunks)} chunks from **{new_doc_id}**")
                finally:
                    os.unlink(tmp_path)
        else:
            st.success(f"**{new_doc_id}** is ready.")

    st.divider()
    st.caption("Powered by Claude Haiku · all-MiniLM-L6-v2 · ChromaDB")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if msg.get("pages"):
                pages_str = ", ".join(str(p) for p in msg["pages"])
                st.markdown(
                    f'<div class="source-tag">Sources: page(s) {pages_str}</div>',
                    unsafe_allow_html=True,
                )

if prompt := st.chat_input(
    "Ask a question about the PDF…",
    disabled=st.session_state.doc_id is None,
):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("Set ANTHROPIC_API_KEY in your .env file before asking questions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            from ragbase.rag import ask
            result = ask(st.session_state.doc_id, prompt)

        st.markdown(result.answer)
        if result.source_pages:
            pages_str = ", ".join(str(p) for p in result.source_pages)
            st.markdown(
                f'<div class="source-tag">Sources: page(s) {pages_str}</div>',
                unsafe_allow_html=True,
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "pages": result.source_pages,
            "contexts": result.contexts,
            "question": result.question,
        }
    )
    st.session_state.last_eval = None

last_assistant = next(
    (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
    None,
)

if last_assistant:
    st.divider()
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🔍 Evaluate last answer", use_container_width=True):
            with st.spinner("Scoring with RAGAS…"):
                from ragbase.evaluate import score_answer
                scores = score_answer(
                    question=last_assistant["question"],
                    answer=last_assistant["content"],
                    contexts=last_assistant.get("contexts", []),
                )
            st.session_state.last_eval = scores

    if st.session_state.last_eval is not None:
        scores = st.session_state.last_eval
        with col2:
            st.markdown(
                f'<span class="score-box">Faithfulness: <b>{scores.faithfulness:.2f}</b></span>'
                f'<span class="score-box">Answer Relevancy: <b>{scores.answer_relevancy:.2f}</b></span>',
                unsafe_allow_html=True,
            )
