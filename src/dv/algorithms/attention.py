from typing import List
from src.dv.interfaces import Valuator, Signaler
from src.dv.models.entities import Chunk, ValuationResult, ValuationMethod

class AttentionValuator(Valuator):
    def __init__(self, signaler: Signaler):
        self.signaler = signaler

    def evaluate(self, query: str, chunks: List[Chunk], answer: str) -> List[ValuationResult]:
        _ = self.signaler.get_signals(query, " ".join([c.text for c in chunks]), answer)
        
        # Logic to map specific attention weights to specific chunks.
        # For simplicity in this implementation, we assume the signaler
        # provides some aggregated attention scores per chunk.
        # In a real implementation, we would use token-to-chunk mapping.
        
        # Mocking aggregated scores for now as token mapping is complex.
        # Each chunk gets a portion of the total attention.
        results = []
        for chunk in chunks:
            # Placeholder for real attention mapping logic
            score = 0.5 # Default middle value
            results.append(ValuationResult(
                chunk_id=chunk.id,
                method=ValuationMethod.ATTENTION,
                score=score
            ))
            
        return results
