from typing import List
from src.dv.interfaces import Valuator, Judge
from src.dv.models.entities import Chunk, ValuationResult, ValuationMethod

class LOOValuator(Valuator):
    def __init__(self, judge: Judge):
        self.judge = judge

    def evaluate(self, query: str, chunks: List[Chunk], answer: str) -> List[ValuationResult]:
        full_context = " ".join([c.text for c in chunks])
        full_score = self.judge.get_faithfulness(query, full_context, answer)
        
        results = []
        for i, chunk in enumerate(chunks):
            # Create context without current chunk
            partial_chunks = chunks[:i] + chunks[i+1:]
            partial_context = " ".join([c.text for c in partial_chunks])
            
            partial_score = self.judge.get_faithfulness(query, partial_context, answer)
            
            # Value is the marginal contribution
            score = full_score - partial_score
            
            results.append(ValuationResult(
                chunk_id=chunk.id,
                method=ValuationMethod.LOO,
                score=score
            ))
            
        return results
