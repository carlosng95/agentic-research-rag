from ..types import Chunk, Page

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )
    if overlap < 0:
        raise ValueError(
            'overlap cannot be negative'
        )
    if overlap >= chunk_size:
        raise ValueError(
            'overlap must be smaller than chunk_size'
        )
    
    text = text.strip()
    
    if not text:
        return []

    chunks: list[str] = []
    
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

def chunk_page(page: Page, start_chunk_id: int = 0, chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]:
    texts = chunk_text(text = page.text, chunk_size = chunk_size, overlap = overlap)
    
    chunks: list[Chunk] = []
    
    for idx, text in enumerate(texts):
        chunk = Chunk(
            chunk_id = start_chunk_id + idx,
            document_name = page.document_name,
            page_number = page.page_number,
            text = text
        )
        chunks.append(chunk)
    return chunks

def chunk_pages(pages: list[Page], chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    next_chunk_id = 0
    
    for page in pages:
        page_chunks = chunk_page(
            page = page,
            start_chunk_id = next_chunk_id,
            chunk_size = chunk_size,
            overlap = overlap
        )
        chunks.extend(page_chunks)
        next_chunk_id += len(page_chunks)
    
    return chunks