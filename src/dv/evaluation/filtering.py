from typing import List
from src.dv.models.entities import Chunk, ValuationResult

def filter_negative_chunks(chunks: List[Chunk], results: List[ValuationResult], threshold: float = 0.0) -> List[Chunk]:
    """Filters out chunks that have a valuation score below the threshold."""
    # Group results by chunk_id
    chunk_scores = {}
    for res in results:
        if res.chunk_id not in chunk_scores:
            chunk_scores[res.chunk_id] = []
        chunk_scores[res.chunk_id].append(res.score)
    
    # Calculate average score per chunk
    avg_scores = {cid: sum(scores)/len(scores) for cid, scores in chunk_scores.items()}
    
    # Filter
    return [c for c in chunks if avg_scores.get(c.id, 1.0) >= threshold]
