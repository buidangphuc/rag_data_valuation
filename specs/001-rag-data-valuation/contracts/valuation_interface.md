# Internal API Contracts: RAG Data Valuation

## Valuator Interface
All valuation algorithms must implement this interface.

```python
class Valuator(ABC):
    @abstractmethod
    def evaluate(self, query: str, chunks: List[Chunk], answer: str) -> List[ValuationResult]:
        """
        Calculates value for each chunk.
        
        Args:
            query: The user query string.
            chunks: List of retrieved Chunk objects.
            answer: The LLM-generated answer.
            
        Returns:
            List of ValuationResult objects, one per chunk.
        """
        pass
```

## Judge Interface
Used by Valuators to get quality signals (e.g., faithfulness).

```python
class Judge(ABC):
    @abstractmethod
    def get_faithfulness(self, query: str, context: str, answer: str) -> float:
        """
        Returns a 0.0-1.0 faithfulness score for the given answer based ONLY on context.
        """
        pass
```

## Ragas Implementation Details
`RagasJudge` will wrap `ragas.evaluate` for the `Judge` interface.

### Expected Behavior
- **Batching**: While the `Judge` interface is per-call, the `ValuationSuite` or specific Valuators (like `ShapleyValuator`) should attempt to batch calls to the underlying Ragas engine to optimize LLM usage.
- **Fail-safe**: If Ragas fails (API timeout, etc.), the judge should return a neutral score (0.5) or raise a descriptive `EvaluationError`.
