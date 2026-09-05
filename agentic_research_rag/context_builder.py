from .types import Chunk

def build_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return ""
    sections: list[str] = []
    
    for source_number, chunk in enumerate(chunks, start = 1):
        section = (
            f'[SOURCE {source_number}]\n'
            f'Document: {chunk.document_name}\n'
            f'Page: {chunk.page_number}\n'
            f'Chunk ID: {chunk.chunk_id}\n\n'
            f'{chunk.text}'
        )
        sections.append(section)
        
    return '\n\n---\n\n'.join(sections)