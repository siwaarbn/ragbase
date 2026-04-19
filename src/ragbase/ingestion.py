from pathlib import Path
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Chunk:
    text: str
    page: int
    chunk_index: int


def load_pdf(path: str | Path) -> list[tuple[str, int]]:
    doc = fitz.open(str(path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append((text, page_num))
    doc.close()
    return pages


def _approximate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def split_into_chunks(
    pages: list[tuple[str, int]],
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for text, page_num in pages:
        words = text.split()
        words_per_chunk = int(chunk_size / 1.3)
        words_overlap = int(overlap / 1.3)

        start = 0
        while start < len(words):
            end = start + words_per_chunk
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(Chunk(text=chunk_text, page=page_num, chunk_index=chunk_index))
            chunk_index += 1
            if end >= len(words):
                break
            start = end - words_overlap

    return chunks


def ingest_pdf(path: str | Path) -> list[Chunk]:
    pages = load_pdf(path)
    return split_into_chunks(pages)
