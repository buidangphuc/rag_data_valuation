from typing import List, Dict
from src.dv.interfaces import Valuator
from src.dv.models.entities import Chunk, ValuationResult

class ValuationSuite:
    def __init__(self, valuators: Dict[str, Valuator]):
        self.valuators = valuators

    def evaluate_all(self, query: str, chunks: List[Chunk], answer: str) -> List[ValuationResult]:
        """Runs all configured valuation methods and aggregates results."""
        all_results = []
        for name, valuator in self.valuators.items():
            results = valuator.evaluate(query, chunks, answer)
            all_results.extend(results)
        return all_results
