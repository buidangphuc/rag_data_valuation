from typing import List
from ...dv.models.entities import Chunk
from ...utils.hashing import calculate_chunk_hash

def semantic_chunker(text: str, threshold: float = 0.5) -> List[Chunk]:
    """Splits text into chunks based on semantic similarity of sentences."""
    sentences = text.split(". ")
    if not sentences:
        return []

    # Simplified semantic logic: sentences are grouped if they share keywords
    # Real implementation would use sentence embeddings (e.g., Sentence-Transformers)
    chunks = []
    current_chunk_sentences = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Mock similarity check
        s1 = set(sentences[i-1].lower().split())
        s2 = set(sentences[i].lower().split())
        similarity = len(s1 & s2) / max(len(s1 | s2), 1)
        
        if similarity >= threshold:
            current_chunk_sentences.append(sentences[i])
        else:
            chunk_text = ". ".join(current_chunk_sentences) + "."
            chunks.append(Chunk(id=calculate_chunk_hash(chunk_text), text=chunk_text, metadata={"type": "semantic"}))
            current_chunk_sentences = [sentences[i]]
            
    if current_chunk_sentences:
        chunk_text = ". ".join(current_chunk_sentences) + "."
        chunks.append(Chunk(id=calculate_chunk_hash(chunk_text), text=chunk_text, metadata={"type": "semantic"}))
        
    return chunks
