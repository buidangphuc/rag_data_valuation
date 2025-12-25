from typing import List
from ...dv.models.entities import Chunk
from ...utils.hashing import calculate_chunk_hash

def recursive_chunker(text: str, chunk_size: int = 100, separators: List[str] = ["\n\n", "\n", " ", ""]) -> List[Chunk]:
    """Splits text recursively using a list of separators."""
    # This is a simplified recursive splitter
    def split_text(txt: str, seps: List[str]) -> List[str]:
        if len(txt) <= chunk_size or not seps:
            return [txt]
        
        sep = seps[0]
        parts = txt.split(sep)
        result = []
        current = ""
        
        for p in parts:
            if len(current) + len(p) + len(sep) <= chunk_size:
                current += (sep if current else "") + p
            else:
                if current:
                    result.append(current)
                current = p
        if current:
            result.append(current)
            
        # Refine parts that are still too large
        final_result = []
        for r in result:
            if len(r) > chunk_size:
                final_result.extend(split_text(r, seps[1:]))
            else:
                final_result.append(r)
        return final_result

    raw_chunks = split_text(text, separators)
    return [
        Chunk(id=calculate_chunk_hash(c), text=c, metadata={"type": "recursive"})
        for c in raw_chunks
    ]
