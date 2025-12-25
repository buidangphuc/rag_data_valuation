# Quickstart: RAG Data Valuation with Ragas

## Installation

```bash
pip install -e .
pip install ragas datasets langchain
```

## Basic Usage

### 1. Define your RAG Components
```python
from src.dv.evaluation.judges import RagasJudge
from src.dv.algorithms.loo import LOOValuator

# Initialize Ragas-based Judge
judge = RagasJudge(model_name="gpt-4o-mini")

# Initialize Valuator
valuator = LOOValuator(judge=judge)
```

### 2. Run Valuation
```python
query = "What is the capital of France?"
answer = "Paris is the capital of France."
chunks = [
    Chunk(id="c1", text="Paris is the capital of France."),
    Chunk(id="c2", text="France is a country in Europe."),
    Chunk(id="c3", text="The Eiffel Tower is in Paris.")
]

results = valuator.evaluate(query, chunks, answer)

for res in results:
    print(f"Chunk {res.chunk_id} Value: {res.score}")
```

### 3. Compare with Benchmarks
Open `tests/notebooks/dv_real_benchmark.ipynb` to run full benchmarks against real-world datasets and visualize Ragas metrics.
