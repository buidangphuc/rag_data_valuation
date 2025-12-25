from typing import List
from ...dv.models.entities import Chunk
from ...utils.hashing import calculate_chunk_hash

def fixed_length_chunker(text: str, chunk_size: int = 100, overlap: int = 20) -> List[Chunk]:
    """Splits text into chunks of fixed length with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunk_id = calculate_chunk_hash(chunk_text)
        chunks.append(Chunk(id=chunk_id, text=chunk_text, metadata={"type": "fixed", "start": start, "end": end}))
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks
