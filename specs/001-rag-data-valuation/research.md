# Research: RAG Data Valuation with Ragas

## Decision: Integrate Ragas as the Primary Evaluation Engine

### Rationale
Ragas (Retrieval-Augmented Generation Assessment) is the industry standard for evaluating RAG pipelines. Using `ragas.metrics.faithfulness` replaces the need for custom NLI or LLM-as-a-judge prompts, providing:
- Standardized scoring mechanism.
- Support for high-quality LLM-based judging (GPT-4) and local alternatives (via Langchain).
- Built-in handling of context-answer alignment.

### Ragas Integration Strategy
To maintain the existing `Valuator` architecture, we will implement `RagasJudge` which satisfies the `src.dv.interfaces.Judge` interface.

```python
class RagasJudge(Judge):
    def __init__(self, metrics=[faithfulness], llm=None):
        self.metrics = metrics
        self.llm = llm

    def get_faithfulness(self, query: str, context: str, answer: str) -> float:
        # Wrap input into Ragas-compatible format
        dataset = Dataset.from_dict({
            "question": [query],
            "contexts": [[context]],
            "answer": [answer]
        })
        result = evaluate(dataset, metrics=self.metrics, llm=self.llm)
        return result["faithfulness"]
```

### Performance Considerations (Shapley & LOO)
Shapley Value calculations require evaluating $2^k$ subsets. Calling `ragas.evaluate` for each subset individually is inefficient.
- **Optimization**: Batch all subsets into a single `Dataset` and call `ragas.evaluate` once per query valuation.
- **Local Surrogate**: For Attention-based signals, we will use a local surrogate model as planned (Llama-3), but for final metrics evaluation, `ragas` will be the source of truth.

### Alternatives Considered
1. **MNLI (Local)**: Fast and free, but less nuanced than LLM-based Ragas. Will be kept as a "Fast Evaluation" option.
2. **DeepEval**: Similar to Ragas, but Ragas has better community support for RAG-specific retrieval metrics.

## Dependencies to Add
- `ragas`
- `datasets`
- `langchain` (for LLM integration)
