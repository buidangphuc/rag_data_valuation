# Valuation API Contracts

## Core Interfaces

### `Valuator` Interface
```python
class Valuator(ABC):
    @abstractmethod
    def evaluate(self, query: str, chunks: List[Chunk], answer: str) -> List[ValuationResult]:
        """Calculates value for each chunk given the query and answer."""
        pass
```

### `Signaler` Interface (Internal)
```python
class Signaler(ABC):
    @abstractmethod
    def get_signals(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """Extracts model-internal signals (attention, gradients)."""
        pass
```

### `Judge` Interface
```python
class Judge(ABC):
    @abstractmethod
    def get_faithfulness(self, query: str, context: str, answer: str) -> float:
        """Returns a 0.0-1.0 faithfulness score."""
        pass
```

## CLI Interface

### `rag-dv evaluate`
- **Arguments**:
  - `--query`: The question asked.
  - `--chunks-file`: Path to JSON containing retrieved chunks.
  - `--answer`: The LLM generated response.
- **Options**:
  - `--methods`: comma-separated (loo, shapley, attention).
  - `--surrogate`: model name (e.g., meta-llama/Llama-3-8B).
- **Output**: JSON report to stdout/file.

### `rag-dv analyze-experiment`
- **Arguments**:
  - `--results-dir`: Path to stored experiment runs.
- **Output**: Summary table of DV vs. Faithfulness agreement (Kendall Tau).
