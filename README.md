# RAG Data Valuation (rag_dv)

A framework for evaluating the contribution of retrieved data chunks to the quality and faithfulness of RAG (Retrieval-Augmented Generation) system answers.

## Overview

This project provides tools to measure "data value" in RAG pipelines using various algorithms like Leave-One-Out (LOO), Shapley Values, and Attention-based signals. The goal is to identify high-value chunks that improve answer quality and low-value chunks that might lead to hallucinations.

## Key Features

- **Multiple Valuation Algorithms**: LOO, Shapley, Attention, and Proxy-based valuation.
- **Interface Driven**: Consistent `Valuator`, `Signaler`, and `Judge` interfaces for easy extension.
- **Experimental Benchmarking**: Dedicated environments for testing new methods against real-world data.
- **Pythonic & Modern**: Built with Python 3.12, PyTorch, and Transformers, enforced by Ruff.

## Project Structure

```text
src/dv/
├── algorithms/      # Implementation of valuation methods (LOO, Shapley, etc.)
├── evaluation/      # Judges and metrics for faithfulness scoring
├── models/          # Data entities (Chunk, ValuationResult, etc.)
├── interfaces.py    # Core abstract base classes
└── core.py          # ValuationSuite management
tests/
├── notebooks/       # Experimental benchmarks (dv_real_benchmark.ipynb)
└── ...              # Unit and integration tests
```

## Getting Started

### Prerequisites

- Python 3.12+
- PyTorch (with MPS support for Mac)
- OpenAI API Key (optional, for cloud-based judges)

### Installation

```bash
pip install -r requirements.txt
```

## Development Guide

### Adding New Valuation Modules

To add a new data valuation algorithm:

1.  **Implement the Interface**: Create a new file in `src/dv/algorithms/` and implement the `Valuator` class from `src.dv.interfaces`.
2.  **Register the Method**: Add the new method to the `ValuationMethod` enum in `src/dv/models/entities.py`.
3.  **Integrate**: Use the `ValuationSuite` in `src/dv/core.py` to include your new valuator.

```python
from src.dv.interfaces import Valuator
from src.dv.models.entities import Chunk, ValuationResult

class MyNewValuator(Valuator):
    def evaluate(self, query: str, chunks: List[Chunk], answer: str) -> List[ValuationResult]:
        # Your valuation logic here
        pass
```

### Experimental Benchmarking

⚠️ **Important**: The `tests/notebooks/dv_real_benchmark.ipynb` and related `dv_real...` files are designated as **experimental environments**. 

These notebooks are used for:
- Prototyping new valuation algorithms.
- Benchmarking performance against real-world RAG datasets.
- Visualizing the distribution of data values.
- Validating "faithfulness" before promoting a method to a stable module.

Always verify your experiments in these notebooks before integrating them into the core `src/` directory.

## Quality Standards

This project adheres to the [RAG Data Valuation Constitution](.specify/memory/constitution.md). 
Key highlights:
- All code must be Python 3.12 and Ruff-compliant.
- Logic must be verified with `pytest`.
- Benchmarks must be reproducible in Jupyter.

## License

[Specify License if applicable]
